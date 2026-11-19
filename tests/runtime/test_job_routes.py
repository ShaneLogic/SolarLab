"""Real loopback V2 solarlab submissions, registry refusals, durable streams, and historical readback."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import struct
import time

import pytest

from solarlab.experiments.execution import JobHandle, ProcessSupervisor
from solarlab.experiments.worker import WorkerBinding
from solarlab.io.artifacts import bytes_artifact
from solarlab.io.run_store import RunStore
from solarlab.materials.source import SourceDocument
from solarlab_server.configuration import ConfigurationPreviewContext
from solarlab_server.job_service import JobService, JobSubmission, canonical, legacy_job_id
from test_event_service import _HTTPServer as _BaseHTTPServer, _all_events, _blocks, _next_event
from job_route_fixtures import catalog, request as declaration
import job_workers


class _HTTPServer(_BaseHTTPServer):
    def client(self):
        client = super().client()
        client.timeout = 15.0
        return client


@pytest.fixture(scope="module")
def registered():
    return catalog()


def supervisor(store, work, registration, *, gated=None, bad_binding=False, max_pending=1):
    binding = registration.worker_binding()
    if gated is not None:
        binding = WorkerBinding.from_function(
            binding.id,
            gated,
            registry_sha256=binding.registry_sha256,
            schema_sha256=binding.schema_sha256,
        )
    if bad_binding:
        binding = replace(binding, source_sha256="0" * 64)
    return ProcessSupervisor(
        store,
        work,
        (binding,),
        max_active=1,
        max_pending=max_pending,
        shared_rss_bytes=4 * 1024**3,
        shared_output_bytes=512 * 1024**2,
        soft_grace_seconds=0.1,
        terminate_grace_seconds=0.1,
        poll_seconds=0.01,
    )


def headers(registration):
    return {
        "X-Solarlab-Registry": registration.document.sha256,
        "X-Solarlab-Job-Schema": registration.schema.sha256,
    }


def handle(store, response):
    return JobHandle(store.store_id, response["run_id"], response["attempt_id"])


def ready(store, run_id):
    until = time.monotonic() + 4
    while time.monotonic() < until:
        if any(e["payload"].get("ui_progress", {}).get("stage") == "ready" for e in store.events(run_id=run_id)):
            return
        time.sleep(0.005)
    raise AssertionError("child did not reach the observed ready barrier")


def test_real_registry_submit_full_stream_verified_result_and_read_only_restart(tmp_path, registered):
    registration, job = registered
    data = job.execution.inputs.prepared
    preview = ConfigurationPreviewContext(data.defaults, data.resources, data.sources)
    database, work = tmp_path / "store", tmp_path / "work"
    with RunStore(database) as store, supervisor(store, work, registration) as pool:
        services = JobService(registration, pool)
        with _HTTPServer(database, job_service=services, configuration_preview=preview) as server, server.client() as http:
            advertised = http.get("/experiments")
            assert advertised.status_code == 200
            assert advertised.headers["x-solarlab-registry"] == registration.document.sha256
            assert advertised.json()["registry"]["registered_executors"] == []
            assert advertised.json()["bindings"][0]["scientific_qualification"] is False
            assert advertised.json()["bindings"][0]["engine"] == "solarlab"
            assert "legacy_facade_sha256" not in advertised.json()
            assert http.get("/schema").json()["array"]["properties"]["shape"]["maxItems"] == 32
            assert http.get("/configs").json()["sources"]
            source = data.sources[0]
            assert http.get("/configs/" + source.id).content == source.content
            table = data.resources.tables[0]
            assert http.get("/materials").json()["resource_library_sha256"] == data.resources.content_sha256
            assert http.get("/materials/" + table.name).content == table.source.content
            response = http.post("/jobs", json=declaration(), headers=headers(registration))
            assert response.status_code == 202, response.text
            submitted = response.json()
            duplicate = http.post("/jobs", json=declaration(), headers=headers(registration))
            assert duplicate.status_code == 202 and duplicate.json()["attempt_id"] == submitted["attempt_id"]
            with http.stream("GET", submitted["events_url"]) as stream:
                events = _all_events(stream)
            assert events[-1]["kind"] == "terminal" and events[-1]["payload"]["state"] == "succeeded"
            assert sum(e["kind"] == "terminal" for e in events) == 1
            receipt = pool.wait(handle(store, submitted), 10).data
            assert receipt["process_joined"] and receipt["process_exit_observed"]
            result = http.get(submitted["result_url"]).json()
            assert result["result"]["worker_result"]["experiment_id"] == "steady_jv"
            assert result["result"]["worker_result"]["scientific_qualification"] is False
            payload = next(a for a in result["artifacts"] if a["name"] == "voltage_V")
            raw = http.get(payload["content_url"]).content
            assert hashlib.sha256(raw).hexdigest() == payload["sha256"]
            assert raw == store.read_artifact(payload["artifact_id"])
            manifest = next(a for a in result["artifacts"] if a["name"] == "execution_manifest")
            assert manifest["sha256"] == receipt["worker_result"]["manifest_sha256"]
            evidence_path = os.environ.get("SOLARLAB_HTTP_JOB_EVIDENCE")
            if evidence_path:
                Path(evidence_path).write_text(
                    json.dumps(
                        {
                            "routed_payload_sha256": payload["sha256"],
                            "routed": receipt,
                            "scientific_qualification": False,
                        },
                        indent=2,
                    )
                    + "\n"
                )
    with _HTTPServer(database) as server, server.client() as http:
        assert http.get(submitted["result_url"]).json() == result
        with http.stream("GET", submitted["events_url"]) as stream:
            assert _all_events(stream) == events
        assert http.post("/jobs", json=declaration(), headers=headers(registration)).status_code == 503


def test_headers_input_engine_and_capability_refuse_before_enqueue(tmp_path, registered):
    registration, _ = registered
    with RunStore(tmp_path / "store") as store, supervisor(store, tmp_path / "work", registration) as pool:
        with _HTTPServer(store.root, job_service=JobService(registration, pool)) as server, server.client() as http:
            assert http.post("/jobs", json=declaration()).status_code == 428
            for field in headers(registration):
                invalid = {**headers(registration), field: "0" * 64}
                assert http.post("/jobs", json=declaration(), headers=invalid).status_code == 409
            for change in ({"engine": "legacy"}, {"experiment_id": "impedance"}, {"version": 99}):
                assert http.post("/jobs", json={**declaration(), **change}, headers=headers(registration)).status_code == 422
            raw = declaration()
            raw["input"]["ignored_extra"] = True
            assert http.post("/jobs", json=raw, headers=headers(registration)).status_code == 422
            raw = declaration()
            raw["limits"]["wall_seconds"] = 31.0
            assert http.post("/jobs", json=raw, headers=headers(registration)).status_code == 422
            # Legacy POST routes are retired with HTTP 410 and zero fallback.
            assert http.post("/api/jobs", json={"kind": "jv", "device": None}).status_code == 410
            assert http.post("/api/jv", json={"device": None}).status_code == 410
            assert store.events() == []
    denied, _ = catalog(denied=True)
    with RunStore(tmp_path / "denied") as store, supervisor(store, tmp_path / "denied-work", denied) as pool:
        with _HTTPServer(store.root, job_service=JobService(denied, pool)) as server, server.client() as http:
            result = http.post("/jobs", json=declaration(), headers=headers(denied))
            assert result.status_code == 422 and "capability refused" in result.text
            assert store.events() == []


def test_two_live_subscribers_disconnect_queued_cancel_and_following_job(tmp_path, registered):
    registration, _ = registered
    with RunStore(tmp_path / "store") as store, supervisor(
        store, tmp_path / "work", registration, gated=job_workers.gated_steady
    ) as pool:
        with _HTTPServer(store.root, job_service=JobService(registration, pool)) as server:
            with server.client() as first, server.client() as second:
                item = first.post("/jobs", json=declaration(), headers=headers(registration)).json()
                ready(store, item["run_id"])
                one_observed, two_observed = server.observe("one"), server.observe("two")
                with first.stream("GET", item["events_url"], headers={"X-Test-Subscriber": "one"}) as one:
                    with second.stream("GET", item["events_url"], headers={"X-Test-Subscriber": "two"}) as two:
                        a, b = _blocks(one), _blocks(two)
                        assert _next_event(a) == _next_event(b)
                        one.close()
                        assert one_observed.finished.wait(2)
                        assert store.get_run(item["run_id"])["state"] == "running"
                        queued = first.post("/jobs", json=declaration("queued"), headers=headers(registration)).json()
                        full = first.post("/jobs", json=declaration("full"), headers=headers(registration))
                        assert full.status_code == 429
                        for selected in (queued, item):
                            row = store.get_run(selected["run_id"])
                            target = f"/jobs/{selected['run_id']}/attempts/{selected['attempt_id']}/cancel"
                            assert (
                                first.post(target, json={"expected_generation": row["generation"] + 10, "reason": "stale"}).status_code
                                == 409
                            )
                            response = first.post(
                                target, json={"expected_generation": row["generation"], "reason": "explicit HTTP cancellation"}
                            )
                            assert response.status_code == 200
                        assert not pool.wait(handle(store, queued), 3).data["process_started"]
                        later = [json.loads(frame["data"]) for frame in b if "data" in frame]
                        assert later[-1]["payload"]["state"] == "cancelled"
                        assert two_observed.finished.wait(2)
                        terminal = str(later[-1]["sequence"])
                        assert first.get(item["events_url"], headers={"Last-Event-ID": terminal}).status_code == 204
                next_job = first.post("/jobs", json=declaration("after"), headers=headers(registration)).json()
                ready(store, "after")
                (tmp_path / "work" / (store.store_id + "-" + next_job["attempt_id"]) / "release").touch()
                assert pool.wait(handle(store, next_job), 10).data["store_state"] == "succeeded"


def test_cancel_rechecks_generation_atomically_after_the_http_read(tmp_path, registered, monkeypatch):
    registration, _ = registered
    with RunStore(tmp_path / "store") as store, supervisor(
        store, tmp_path / "work", registration, gated=job_workers.gated_steady
    ) as pool:
        with _HTTPServer(store.root, job_service=JobService(registration, pool)) as server, server.client() as http:
            first = http.post("/jobs", json=declaration("hold"), headers=headers(registration)).json()
            ready(store, first["run_id"])
            queued = http.post("/jobs", json=declaration("race"), headers=headers(registration)).json()
            original = store.cancel
            changed = []

            def changed_generation(run_id, **kwargs):
                if run_id == "race" and not changed:
                    changed.append(
                        store.claim(
                            run_id,
                            expected_attempt_id=kwargs["expected_attempt_id"],
                            expected_generation=kwargs["expected_generation"],
                            owner_id="metadata-race-fixture",
                        )
                    )
                return original(run_id, **kwargs)

            monkeypatch.setattr(store, "cancel", changed_generation)
            response = http.post(
                f"/jobs/race/attempts/{queued['attempt_id']}/cancel",
                json={"expected_generation": 0, "reason": "stale after read"},
            )
            assert response.status_code == 409
            assert store.get_run("race")["state"] == "running"
            monkeypatch.setattr(store, "cancel", original)
            store.complete(changed[0], state="failed", result={"reason": "metadata-only fixture complete", "process_started": False})
            pool.cancel(handle(store, first))


@pytest.mark.parametrize("field", ["job_schema_sha256", "model_registry_sha256"])
def test_real_worker_rejects_internally_wrong_registry_or_schema(tmp_path, registered, field):
    registration, _ = registered
    spec, binding = registration.prepare(JobSubmission.model_validate(declaration()))
    claimed = json.loads(registration.document.content)
    if field == "model_registry_sha256":
        claimed["bindings"][0]["model_registry_sha256"] = "0" * 64
    claimed[field] = "0" * 64
    altered = SourceDocument("explicit-inconsistent-registry-fixture", canonical(claimed))
    worker = replace(registration.worker_binding(), registry_sha256=altered.sha256)
    with RunStore(tmp_path / "store") as store, ProcessSupervisor(
        store,
        tmp_path / "work",
        (worker,),
        max_active=1,
        max_pending=0,
        shared_rss_bytes=4 * 1024**3,
        shared_output_bytes=256 * 1024**2,
    ) as pool:
        result = pool.wait(
            pool.submit("bad-registry", worker.id, replace(spec.request, registry=altered), spec.limits, execution=binding),
            10,
        ).data
        assert result["store_state"] == "failed" and result["worker_result"]["reason"] == "worker_exception"
        assert "registry" in result["worker_result"]["detail"]
        assert not result["terminal_artifacts"] and result["process_joined"]


@pytest.mark.parametrize("failure", ["timeout", "worker_source", "driver_failure"])
def test_real_worker_failures_produce_business_error_then_done(tmp_path, registered, failure):
    registration, _ = registered
    gated = (
        job_workers.stuck_steady
        if failure == "timeout"
        else job_workers.failing_steady
        if failure == "driver_failure"
        else None
    )
    with RunStore(tmp_path / "store") as store, supervisor(
        store, tmp_path / "work", registration, gated=gated, bad_binding=failure == "worker_source"
    ) as pool:
        with _HTTPServer(store.root, job_service=JobService(registration, pool)) as server, server.client() as http:
            data = declaration()
            if failure == "timeout":
                data["limits"]["wall_seconds"] = 0.5
            response = http.post("/jobs", json=data, headers=headers(registration))
            assert response.status_code == 202, response.text
            item = response.json()
            with http.stream("GET", f"/api/jobs/{item['job_id']}/events") as response:
                frames = [f for f in _blocks(response) if "data" in f]
            assert [f["event"] for f in frames[-2:]] == ["error", "done"]
            assert frames[-2]["id"] != frames[-1]["id"]
            error = json.loads(frames[-2]["data"])
            assert error["message"] == ("wall_limit" if failure == "timeout" else "worker_exception")
            receipt = pool.wait(handle(store, item), 5).data
            assert receipt["store_state"] == "failed" and receipt["process_joined"]
            with http.stream(
                "GET", f"/api/jobs/{item['job_id']}/events", headers={"Last-Event-ID": frames[-2]["id"]}
            ) as replay:
                assert [f["event"] for f in _blocks(replay) if "data" in f] == ["done"]
            assert http.get(f"/api/jobs/{item['job_id']}/events", headers={"Last-Event-ID": frames[-1]["id"]}).status_code == 204


def test_historical_legacy_result_artifact_readback_and_phase_replay_without_legacy_engine(tmp_path):
    with RunStore(tmp_path / "store") as store:
        queued = store.enqueue(
            "historical-legacy",
            opaque_identity={"H_input": "hist:i", "H_physics": "hist:p", "H_execution": "hist:e"},
            input_metadata={"scope": "historical_readback_only"},
        )
        token = store.claim("historical-legacy", expected_attempt_id=queued["current_attempt_id"], expected_generation=0, owner_id="hist")
        store.append_event(token, event_id="step1", payload={"ui_progress": {"stage": "historical_step", "current": 1, "total": 2}})
        arr_bytes = struct.pack("<2d", 0.0, 0.95)
        envelope = canonical(
            {
                "schema": "solarlab.legacy-raw-result.v1",
                "driver": "scaps_transient",
                "result": {
                    "V_fwd": {"dtype": "<f8", "shape": [2], "raw_hex": arr_bytes.hex()},
                    "V_oc": {"binary64_be": struct.pack(">d", 0.95).hex()},
                    "nonfinite": {"binary64_be": struct.pack(">d", float("nan")).hex()},
                },
            }
        )
        saved = store.publish(token, "legacy_result", bytes_artifact(envelope, metadata={"scientific_qualification": False}))
        store.complete(
            token,
            state="succeeded",
            result={"worker_result": {"engine": "legacy", "scientific_qualification": False}},
            artifacts=[saved["artifact_id"]],
        )
        job_id = legacy_job_id(JobHandle(store.store_id, "historical-legacy", token.attempt_id))

    with _HTTPServer(tmp_path / "store") as server, server.client() as http:
        direct = http.get(f"/api/jobs/{job_id}/result")
        assert direct.status_code == 200
        body = direct.json()
        assert body["V_fwd"] == [0.0, 0.95]
        assert body["V_oc"] == 0.95
        assert body["nonfinite"] is None
        assert body["execution"]["worker_result"]["scientific_qualification"] is False

        with http.stream("GET", f"/api/jobs/{job_id}/events") as stream:
            frames = [f for f in _blocks(stream) if "data" in f]
        assert [f["event"] for f in frames] == ["progress", "progress", "progress", "result", "done"]
        with http.stream("GET", f"/api/jobs/{job_id}/events", headers={"Last-Event-ID": frames[-2]["id"]}) as tail:
            assert [f["event"] for f in _blocks(tail) if "data" in f] == ["done"]
        assert http.get(f"/api/jobs/{job_id}/events", headers={"Last-Event-ID": frames[-1]["id"]}).status_code == 204
        for invalid in ("999999.event", frames[-1]["id"].replace("done", "error"), "1", "-1.result"):
            assert http.get(f"/api/jobs/{job_id}/events", headers={"Last-Event-ID": invalid}).status_code == 400


def test_large_result_uses_verified_references_and_corruption_does_not_rewrite_history(tmp_path):
    with RunStore(tmp_path / "store") as store:
        queued = store.enqueue(
            "large",
            opaque_identity={"H_input": "fixture", "H_physics": "fixture", "H_execution": "fixture"},
            input_metadata={"scope": "serialization-only; no physical execution"},
        )
        token = store.claim("large", expected_attempt_id=queued["current_attempt_id"], expected_generation=0, owner_id="fixture")
        data = canonical({"dtype": "<f8", "shape": [20000], "raw_hex": "0000000000000000" * 20000})
        saved = store.publish(token, "legacy_result", bytes_artifact(data, metadata={"scope": "non-scientific fixture"}))
        store.complete(token, state="succeeded", result={"scientific_qualification": False}, artifacts=[saved["artifact_id"]])
        job_id = legacy_job_id(JobHandle(store.store_id, "large", token.attempt_id))
        with _HTTPServer(store.root) as server, server.client() as http:
            url = f"/results/large/attempts/{token.attempt_id}"
            result = http.get(url)
            assert result.status_code == 200 and result.json()["artifacts"][0]["sha256"] == saved["sha256"]
            with http.stream("GET", f"/api/jobs/{job_id}/events") as stream:
                frames = [f for f in _blocks(stream) if "data" in f]
            content = json.loads(frames[-2]["data"])
            assert content["payload_inline"] is False and len(frames[-2]["data"]) < 10000
            assert http.get(content["artifacts"][0]["content_url"]).content == data
            (store.root / saved["relative_path"]).write_bytes(b"explicit corrupt fixture")
            assert http.get(url).status_code == 409
            assert store.get_run("large")["state"] == "succeeded"
