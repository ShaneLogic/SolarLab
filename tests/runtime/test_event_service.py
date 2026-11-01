"""Infrastructure-only SQLite + loopback HTTP contracts; no scientific worker.

The clients consume actual Uvicorn socket streams. Thread events and consumed
frames establish ordering; no sleep is used to infer disconnect or commit.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sqlite3
import threading
import time

import httpx
import numpy as np
import pytest
import uvicorn

from solarlab.io.artifacts import array_artifact, bytes_artifact
from solarlab.io.run_store import RunStore
from solarlab_server import app as service_app
from solarlab_server.app import create_app


IDENTITY = {"H_input": "infrastructure:input", "H_physics": "infrastructure:unused",
            "H_execution": "infrastructure:execution", "schema": "fixture.v1"}


def _receipt(data):
    destination = os.environ.get("SOLARLAB_HTTP_RECEIPT")
    if destination:
        with open(destination, "a", encoding="utf-8") as stream:
            stream.write(json.dumps({"test": os.environ.get("PYTEST_CURRENT_TEST"), **data}) + "\n")


@dataclass
class _Subscription:
    started: threading.Event = field(default_factory=threading.Event)
    finished: threading.Event = field(default_factory=threading.Event)


class _ObservedApp:
    """Test-only ASGI lifecycle barriers; the actual app/store are unchanged."""

    def __init__(self, app):
        self.app = app
        self.subscribers = {}

    async def __call__(self, scope, receive, send):
        label = dict(scope.get("headers", [])).get(b"x-test-subscriber", b"").decode()
        observation = self.subscribers.get(label)

        async def observed_send(message):
            await send(message)
            if observation and message["type"] == "http.response.start":
                observation.started.set()

        try:
            await self.app(scope, receive, observed_send)
        finally:
            if observation:
                observation.finished.set()


class _ReadyServer(uvicorn.Server):
    def __init__(self, config):
        super().__init__(config)
        self.ready = threading.Event()

    async def startup(self, sockets=None):
        try:
            await super().startup(sockets=sockets)
        finally:
            self.ready.set()


class _HTTPServer:
    def __init__(self, root, **options):
        self.app = create_app(root, event_page_size=2, poll_interval=0.01,
                              keepalive_interval=0.04, **options)
        self.observed = _ObservedApp(self.app)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(("127.0.0.1", 0))
        self.port = self.socket.getsockname()[1]
        self.server = _ReadyServer(uvicorn.Config(
            self.observed, host="127.0.0.1", port=self.port, log_config=None,
            access_log=False, lifespan="on", loop="asyncio", http="h11",
            timeout_graceful_shutdown=0.2,
        ))
        self.errors = []
        self.before_threads = set(threading.enumerate())
        self.thread = threading.Thread(target=self._run, name="event-service-http")
        self.stopped = False

    def _run(self):
        try:
            asyncio.run(self.server.serve(sockets=[self.socket]))
        except BaseException as error:
            self.errors.append(error)
            self.server.ready.set()

    def __enter__(self):
        self.thread.start()
        try:
            assert self.server.ready.wait(3), "server startup was not observed"
            assert self.server.started and not self.errors
        except BaseException:
            self.stop()
            raise
        return self

    def observe(self, name):
        observation = _Subscription()
        self.observed.subscribers[name] = observation
        return observation

    def client(self):
        return httpx.Client(base_url=f"http://127.0.0.1:{self.port}", timeout=2, trust_env=False)

    def stop(self):
        if self.stopped:
            return
        self.server.should_exit = True
        self.thread.join(3)
        self.socket.close()
        assert not self.thread.is_alive(), "owned server thread did not exit"
        # asyncio/AnyIO owns worker threads used by the synchronous readers.
        # Observe those exits as well, instead of relying on daemon teardown.
        other_owned = set(threading.enumerate()) - self.before_threads - {self.thread}
        deadline = time.monotonic() + 1
        for thread in other_owned:
            thread.join(max(0, deadline - time.monotonic()))
        assert not any(thread.is_alive() for thread in other_owned)
        assert not self.errors
        assert self.socket.fileno() == -1
        assert not self.server.server_state.connections
        assert not self.server.server_state.tasks
        assert self.app.state.run_store is None
        assert all(item.finished.is_set() for item in self.observed.subscribers.values())
        self.stopped = True
        _receipt({"kind": "server_cleanup", "port": self.port,
                  "thread_joined": True, "socket_closed": True, "store_closed": True,
                  "remaining_connections": 0, "remaining_asgi_tasks": 0,
                  "subscriber_count": len(self.observed.subscribers),
                  "all_subscribers_finished": True, "worker_threads_joined": len(other_owned)})

    def __exit__(self, *error):
        self.stop()


@pytest.fixture
def store(tmp_path):
    with RunStore((tmp_path / "store").resolve()) as value:
        yield value


def _enqueue(store, run_id="fixture", **metadata):
    return store.enqueue(run_id, opaque_identity=IDENTITY, input_metadata=metadata)


def _claim(store, run_id="fixture"):
    queued = store.get_run(run_id)
    return store.claim(run_id, expected_attempt_id=queued["current_attempt_id"],
                       expected_generation=queued["generation"], owner_id="fixture-writer")


def _url(token):
    return f"/runs/{token.run_id}/attempts/{token.attempt_id}/events"


def _blocks(response):
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    block = {}
    for line in response.iter_lines():
        if not line:
            if block:
                yield block
            block = {}
        elif line.startswith(":"):
            block["comment"] = line
        else:
            key, value = line.split(":", 1)
            block[key] = value.lstrip(" ")
    assert not block, "truncated SSE frame"


def _next_event(blocks):
    for block in blocks:
        if "data" in block:
            record = json.loads(block["data"])
            assert int(block["id"]) == record["sequence"]
            assert block["event"] == record["kind"]
            return record
    raise AssertionError("stream ended before the expected committed event")


def _all_events(response):
    records = []
    for block in _blocks(response):
        if "data" in block:
            record = json.loads(block["data"])
            assert int(block["id"]) == record["sequence"]
            assert block["event"] == record["kind"]
            records.append(record)
    return records


def test_two_subscribers_keepalive_disconnect_reconnect_and_terminal(store):
    _enqueue(store, zero=0, negative_zero=-0.0, explicit_null=None)
    token = _claim(store)
    with _HTTPServer(store.root) as server, server.client() as first, server.client() as second:
        first_observed = server.observe("first")
        second_observed = server.observe("second")
        with first.stream("GET", _url(token), headers={"X-Test-Subscriber": "first"}) as one:
            with second.stream("GET", _url(token), headers={"X-Test-Subscriber": "second"}) as two:
                assert first_observed.started.wait(2) and second_observed.started.wait(2)
                a, b = _blocks(one), _blocks(two)
                baseline = store.events(run_id="fixture")
                assert [_next_event(a), _next_event(a)] == baseline
                assert [_next_event(b), _next_event(b)] == baseline
                keepalive = next(a)
                assert keepalive == {"comment": ": keepalive"}
                progress = store.append_event(token, event_id="first", payload={"zero": 0, "null": None})
                assert _next_event(a) == _next_event(b) == store.events(after=progress - 1)[0]
                one.close()
                assert first_observed.finished.wait(2), "disconnect did not end its subscriber"
                assert store.get_run("fixture")["state"] == "running"
                later = store.append_event(token, event_id="later", payload={"value": -0.0})
                assert _next_event(b)["sequence"] == later
                with server.client() as reconnect:
                    reconnected = server.observe("reconnected")
                    with reconnect.stream("GET", _url(token), headers={
                        "Last-Event-ID": str(progress), "X-Test-Subscriber": "reconnected",
                    }) as three:
                        c = _blocks(three)
                        assert _next_event(c)["sequence"] == later
                        result = {"value": -0.0, "zero": 0, "missing": None,
                                  "metadata": {"identity": IDENTITY, "qualification": None}}
                        terminal = store.complete(token, state="succeeded", result=result)
                        final_b, final_c = _next_event(b), _next_event(c)
                        assert final_b == final_c == store.events(after=terminal - 1)[0]
                        assert final_b["payload"]["result"] == result
                        assert math.copysign(1, final_b["payload"]["result"]["value"]) == -1
                        assert list(b) == [] and list(c) == []
                        assert reconnected.finished.wait(2)
                assert second_observed.finished.wait(2)
        assert [e["kind"] for e in store.events()].count("terminal") == 1


def test_bounded_pages_terminal_replay_and_server_restart(store):
    expected_run = _enqueue(store, explicit_null=None, negative_zero=-0.0)
    token = _claim(store)
    _enqueue(store, "other")  # Real global-sequence gap within this run's pages.
    progress = store.append_event(token, event_id="p", payload={"label": "infrastructure"})
    terminal = store.complete(token, state="succeeded", result={"zero": 0, "null": None})
    expected = store.events(run_id="fixture")
    with _HTTPServer(store.root) as server, server.client() as client:
        cursor, obtained = 0, []
        while True:
            response = client.get("/runs/fixture/events", params={"after": cursor, "limit": 2})
            assert response.status_code == 200
            page = response.json()
            assert page["store_id"] == response.headers["X-RunStore-ID"] == store.store_id
            assert len(page["events"]) <= 2
            if not page["events"]:
                assert page["next_cursor"] == cursor
                break
            obtained.extend(page["events"])
            cursor = page["next_cursor"]
        assert obtained == expected
        assert client.get("/runs/fixture/events?limit=3").status_code == 422
        assert client.get("/runs/fixture/events?after=-1").status_code == 422
        assert client.get("/runs/unknown/events").status_code == 404
    root = store.root
    store.close()
    with _HTTPServer(root) as restarted, restarted.client() as client:
        response = client.get("/runs/fixture")
        assert response.headers["X-RunStore-ID"] == store.store_id
        run = response.json()
        assert run["request"] == expected_run["request"]
        assert run["state"] == "succeeded"
        response = client.get(f"/runs/fixture/attempts/{token.attempt_id}")
        assert response.headers["X-RunStore-ID"] == store.store_id
        attempt = response.json()
        assert attempt["terminal"]["result"] == {"zero": 0, "null": None}
        with client.stream("GET", _url(token) + "?after=0", headers={"Last-Event-ID": str(progress)}) as response:
            assert _all_events(response) == expected[-1:]
        acknowledged = client.get(_url(token), headers={"Last-Event-ID": str(terminal)})
        assert acknowledged.status_code == 204 and acknowledged.content == b""
        assert acknowledged.headers["X-RunStore-ID"] == store.store_id
        with client.stream("GET", _url(token)) as response:
            assert _all_events(response) == expected


def test_retry_scope_and_foreign_malformed_or_future_cursors(store):
    _enqueue(store)
    old = _claim(store)
    old_terminal = store.cancel("fixture", expected_attempt_id=old.attempt_id,
                                expected_generation=old.generation, reason="fixture_cancel")
    store.retry("fixture", expected_attempt_id=old.attempt_id, expected_generation=old.generation)
    current = _claim(store)
    store.complete(current, state="failed", result={"reason": "fixture_timeout", "calls": None})
    _enqueue(store, "foreign")
    foreign_sequence = store.events(run_id="foreign")[0]["sequence"]
    expected = store.events(run_id="fixture")
    with _HTTPServer(store.root) as server, server.client() as client:
        for token in (old, current):
            with client.stream("GET", _url(token)) as response:
                assert _all_events(response) == [e for e in expected if e["attempt_id"] == token.attempt_id]
        for bad in (str(old_terminal), str(foreign_sequence), "99999", "-1", "1.0", str(2**63)):
            assert client.get(_url(current), headers={"Last-Event-ID": bad}).status_code == 400
        assert client.get("/runs/fixture/attempts/unknown/events").status_code == 404
        current_record = client.get(f"/runs/fixture/attempts/{current.attempt_id}").json()
        assert current_record["result"] == {"reason": "fixture_timeout", "calls": None}
        assert current_record["generation"] != old.generation


def test_artifacts_exact_array_words_publication_and_later_integrity_errors(store):
    _enqueue(store)
    token = _claim(store)
    words = np.array([0, 0x8000000000000000, 0x7FF8000000000001], dtype=np.uint64)
    payload = array_artifact(words.view(np.float64), unit="fixture-unit", metadata={"missing_reason": None})
    array = store.publish(token, "array", payload)
    raw = store.publish(token, "bytes", bytes_artifact(b"\x00\x80exact\x00", metadata={"null": None}))
    with _HTTPServer(store.root) as server, server.client() as client:
        for record in (array, raw):
            assert client.get(f"/artifacts/{record['artifact_id']}").status_code == 409
            assert client.get(f"/artifacts/{record['artifact_id']}/content").status_code == 409
        pending = client.get(f"/runs/fixture/attempts/{token.attempt_id}").json()
        assert pending["result"] is None and pending["terminal"] is None
        store.complete(token, state="succeeded", result={"label": "raw infrastructure words"},
                       artifacts=[array["artifact_id"], raw["artifact_id"]])
        meta = client.get(f"/artifacts/{array['artifact_id']}")
        body = client.get(f"/artifacts/{array['artifact_id']}/content")
        assert meta.status_code == body.status_code == 200
        assert meta.headers["X-RunStore-ID"] == body.headers["X-RunStore-ID"] == store.store_id
        assert meta.json() == store.artifact(array["artifact_id"])
        assert meta.json()["metadata"]["missing_reason"] is None
        assert body.content == payload.data
        assert body.headers["etag"] == '"' + hashlib.sha256(payload.data).hexdigest() + '"'
        assert client.get(f"/artifacts/{raw['artifact_id']}/content").content == b"\x00\x80exact\x00"
        # Only the fixture's committed objects are deliberately damaged.
        (store.root / array["relative_path"]).write_bytes(b"corrupt")
        (store.root / raw["relative_path"]).unlink()
        for record in (array, raw):
            for suffix in ("", "/content"):
                response = client.get(f"/artifacts/{record['artifact_id']}{suffix}")
                assert response.status_code == 409
                assert response.json()["detail"]["code"] == "artifact_integrity"
        assert client.get("/artifacts/unknown").status_code == 404
        assert client.get("/runs/fixture").json()["state"] == "succeeded"
        assert store.attempts("fixture")[0]["terminal"]["state"] == "succeeded"


def test_artifact_transport_cap_does_not_serve_partial_or_unverified_bytes(store):
    _enqueue(store)
    token = _claim(store)
    record = store.publish(token, "bytes", bytes_artifact(b"x" * 32, metadata={}))
    store.complete(token, state="succeeded", result={}, artifacts=[record["artifact_id"]])
    with _HTTPServer(store.root, max_artifact_bytes=8) as server, server.client() as client:
        for suffix in ("", "/content"):
            response = client.get(f"/artifacts/{record['artifact_id']}{suffix}")
            assert response.status_code == 413
            assert "byte limit" in response.json()["detail"]
        assert store.read_artifact(record["artifact_id"]) == b"x" * 32


def test_each_successful_read_captures_one_store_for_body_and_header(store, tmp_path, monkeypatch):
    _enqueue(store)
    token = _claim(store)
    artifact = store.publish(token, "bytes", bytes_artifact(b"fixture", metadata={}))
    terminal = store.complete(token, state="succeeded", result={}, artifacts=[artifact["artifact_id"]])
    expected_events = store.events(run_id="fixture")
    with RunStore((tmp_path / "other-store").resolve()) as other:
        assert other.store_id != store.store_id
        resolutions = []

        def changing_store(request):
            # A second lookup within this request would see another store.
            selected = store if not resolutions else other
            resolutions.append(selected.store_id)
            return selected

        monkeypatch.setattr(service_app, "_store", changing_store)
        cases = [
            ("/runs/fixture", {}, store.get_run("fixture")),
            (f"/runs/fixture/attempts/{token.attempt_id}", {}, store.attempts("fixture")[0]),
            ("/runs/fixture/events", {}, {"store_id": store.store_id, "events": expected_events[:2],
                                          "next_cursor": expected_events[1]["sequence"]}),
            (f"/artifacts/{artifact['artifact_id']}", {}, store.artifact(artifact["artifact_id"])),
            (f"/artifacts/{artifact['artifact_id']}/content", {}, b"fixture"),
            (_url(token), {"Last-Event-ID": str(terminal)}, b""),
        ]
        with _HTTPServer(store.root) as server, server.client() as client:
            for path, headers, expected in cases:
                resolutions.clear()
                response = client.get(path, headers=headers)
                assert response.status_code == (204 if expected == b"" else 200)
                assert response.headers["X-RunStore-ID"] == store.store_id
                assert resolutions == [store.store_id]
                assert (response.content if isinstance(expected, bytes) else response.json()) == expected


def test_rolled_back_terminal_is_not_published_then_real_failure_is_replayed(store, monkeypatch):
    _enqueue(store)
    token = _claim(store)
    artifact = store.publish(token, "ready", bytes_artifact(b"ready", metadata={}))
    original_transaction = store._transaction

    @contextmanager
    def fail_before_commit():
        store._db.execute("BEGIN IMMEDIATE")
        try:
            yield
            raise sqlite3.OperationalError("fixture failure before terminal commit")
        finally:
            store._db.execute("ROLLBACK")

    with _HTTPServer(store.root) as server, server.client() as client:
        with client.stream("GET", _url(token)) as response:
            blocks = _blocks(response)
            assert [_next_event(blocks)["kind"], _next_event(blocks)["kind"]] == ["queued", "running"]
            monkeypatch.setattr(store, "_transaction", fail_before_commit)
            with pytest.raises(sqlite3.OperationalError, match="before terminal commit"):
                store.complete(token, state="succeeded", result={"uncommitted": True},
                               artifacts=[artifact["artifact_id"]])
            monkeypatch.setattr(store, "_transaction", original_transaction)
            assert client.get("/runs/fixture").json()["state"] == "running"
            assert store.artifact(artifact["artifact_id"])["phase"] == "published"
            assert not any(e["kind"] == "terminal" for e in store.events())
            terminal = store.complete(token, state="failed", result={"reason": "fixture_original_failure", "calls": None})
            record = _next_event(blocks)
            assert record["sequence"] == terminal
            assert record["payload"]["state"] == "failed"
            assert record["payload"]["result"] == {"reason": "fixture_original_failure", "calls": None}
            assert list(blocks) == []
            _receipt({"kind": "terminal_commit_boundary", "uncommitted_terminal_rolled_back": True,
                      "terminal_sequence": terminal, "one_committed_terminal_observed": True})


def test_reader_restart_does_not_recover_queued_or_running_work(store):
    _enqueue(store, "queued")
    _enqueue(store, "running")
    _claim(store, "running")
    before = store.events()
    for _ in range(2):
        with _HTTPServer(store.root) as server, server.client() as client:
            assert client.get("/runs/queued").json()["state"] == "queued"
            assert client.get("/runs/running").json()["state"] == "running"
        assert store.events() == before


def test_server_shutdown_ends_open_subscriber_without_cancelling_run(store):
    _enqueue(store)
    token = _claim(store)
    with _HTTPServer(store.root) as server, server.client() as client:
        observed = server.observe("shutdown")
        with client.stream("GET", _url(token), headers={"X-Test-Subscriber": "shutdown"}) as response:
            assert _next_event(_blocks(response))["kind"] == "queued"
            server.stop()
            assert observed.finished.is_set()
        assert store.get_run("fixture")["state"] == "running"
        assert all(e["kind"] != "terminal" for e in store.events())


def test_factory_requires_explicit_existing_store_and_has_no_construction_io(tmp_path):
    absent = (tmp_path / "absent").resolve()
    app = create_app(absent)
    assert not absent.exists() and app.state.run_store is None

    async def start():
        async with app.router.lifespan_context(app):
            pytest.fail("a reader must not create an empty store")

    with pytest.raises(ValueError, match="existing RunStore"):
        asyncio.run(start())
    assert not absent.exists()
    with pytest.raises(ValueError, match="absolute"):
        create_app(Path("relative"))
    with pytest.raises(ValueError, match="event_page_size"):
        create_app(absent, event_page_size=1001)
    with pytest.raises(ValueError, match="positive and finite"):
        create_app(absent, keepalive_interval=float("nan"))
