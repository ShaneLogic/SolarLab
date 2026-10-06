"""Actual spawn/SQLite/file checks for the infrastructure preparation only."""

from dataclasses import FrozenInstanceError, asdict, replace
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import struct
import time

import pytest

import execution_workers as workers
import solarlab.experiments.execution as execution_module
from solarlab.experiments.execution import ProcessSupervisor, QueueFull
from solarlab.experiments.worker import ResourceLimits, WorkerBinding, WorkerRequest, _directory_bytes, _process_info
from solarlab.io.run_store import RunStore, StateConflict
from solarlab.materials.source import SourceDocument

REGISTRY = SourceDocument("infrastructure-registry.v1", b'{"scope":"infrastructure-only"}')
SCHEMA = SourceDocument("infrastructure-request.v1", b'{"type":"opaque infrastructure fixture"}')


def request(**values):
    source = SourceDocument("request", json.dumps(values, sort_keys=True).encode())
    execution = SourceDocument("execution-identity", b'{"scope":"non-scientific identity fixture"}')
    return WorkerRequest(source, REGISTRY, SCHEMA, execution, source.sha256, hashlib.sha256(b"no physical model").hexdigest())


def limits(**changes):
    return replace(ResourceLimits(3.0, 256 * 1024**2, 256 * 1024, 10000, 10000), **changes)


def bindings():
    return tuple(WorkerBinding.from_function(name, getattr(workers, name), registry_sha256=REGISTRY.sha256,
                    schema_sha256=SCHEMA.sha256) for name in
                 ("normal", "cooperative", "uncooperative", "delayed_success", "barrier_success", "exception", "crash",
                  "counter_limit", "missing_counters", "disk_limit", "nested_pool", "malformed_terminal", "finalize_after_release"))


def supervisor(store, root, **changes):
    values = dict(max_active=1, max_pending=2, shared_rss_bytes=768 * 1024**2,
                  shared_output_bytes=16 * 1024**2, soft_grace_seconds=0.1,
                  terminate_grace_seconds=0.1, poll_seconds=0.01)
    values.update(changes)
    return ProcessSupervisor(store, root, bindings(), **values)


def ready(store, handle):
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        for event in store.events(run_id=handle.run_id):
            if event["attempt_id"] == handle.attempt_id and event["payload"].get("ui_progress", {}).get("stage") == "ready":
                return event["payload"]["ui_progress"]["pid"]
        time.sleep(0.005)
    raise AssertionError("owned fixture child did not reach its ready boundary")


def terminal_events(store, run_id):
    return [event for event in store.events(run_id=run_id, limit=1000) if event["kind"] == "terminal"]


def release(tmp_path, store, handle):
    (tmp_path / "work" / (store.store_id + "-" + handle.attempt_id) / "release").touch()


def assert_reaped(receipt):
    value = receipt.data
    assert value["process_started"] and value["process_exit_observed"] and value["process_joined"]
    assert type(value["pid"]) is int and type(value["exitcode"]) is int
    with pytest.raises(ChildProcessError):
        os.waitpid(value["pid"], os.WNOHANG)
    return value


def test_actual_spawn_coalesces_ui_progress_and_commits_independent_artifact(tmp_path):
    with RunStore(tmp_path / "db") as store, supervisor(store, tmp_path / "work") as pool:
        handle = pool.submit("normal", "normal", request(progress_count=1500), limits())
        receipt = pool.wait(handle, 4)
        value = assert_reaped(receipt)
        assert value["store_state"] == "succeeded" and value["terminal_committed"]
        assert value["resources"]["accepted_steps"] == value["resources"]["calls"] == 1500
        assert value["resources"]["child_peak_rss_bytes"] > 0
        events = store.events(run_id="normal", limit=1000)
        progress = [e for e in events if "ui_progress" in e["payload"]]
        assert 0 < len(progress) < 1500
        assert len(terminal_events(store, "normal")) == 1
        assert events[-1]["kind"] == "terminal"
        artifact = store.artifact(value["terminal_artifacts"][0])
        assert store.read_artifact(artifact["artifact_id"]) == struct.pack("<2d", 0.0, -0.0)
        assert artifact["metadata"]["missing_reason"] is None
        data = store.attempts("normal")[0]["result"]["worker_result"]
        assert data["explicit_null"] is None and struct.pack("d", data["negative_zero"]) == struct.pack("d", -0.0)
        assert pool.accounting()["counter_totals"] == {"accepted_steps": 1500, "calls": 1500}
        family = pool.accounting()
        assert os.getpid() in family["sampled_family_pids"] and len(family["sampled_family_pids"]) >= 3
        assert family["rss_scope"] == "conservative_parent_process_family_and_located_prior_intents"
        detached = receipt.data
        detached["store_state"] = "forged"
        assert receipt.data["store_state"] == "succeeded"


def test_shared_bounded_queue_duplicates_and_both_cancellation_paths(tmp_path):
    with RunStore(tmp_path / "db") as store, supervisor(store, tmp_path / "work", max_pending=1) as pool:
        first = pool.submit("running", "cooperative", request(), limits())
        ready(store, first)
        queued = pool.submit("queued", "normal", request(), limits())
        assert pool.submit("queued", "normal", request(), limits()) == queued
        with pytest.raises(QueueFull):
            pool.submit("full", "normal", request(), limits())
        with pytest.raises(KeyError):
            store.get_run("full")
        assert pool.cancel(queued)
        assert not pool.wait(queued, 1).data["process_started"]
        assert store.get_run("queued")["state"] == "cancelled"
        assert pool.cancel(first)
        result = assert_reaped(pool.wait(first, 3))
        assert result["store_state"] == "cancelled" and result["signals"] == ["soft_stop"]
        assert not pool.cancel(first)
        follow = pool.submit("next", "normal", request(), limits())
        assert assert_reaped(pool.wait(follow, 3))["store_state"] == "succeeded"
        assert len(terminal_events(store, "running")) == len(terminal_events(store, "queued")) == 1


def test_uncooperative_worker_requires_kill_and_holds_slot_until_join(tmp_path):
    with RunStore(tmp_path / "db") as store, supervisor(store, tmp_path / "work") as pool:
        stuck = pool.submit("stuck", "uncooperative", request(), limits())
        ready(store, stuck)
        following = pool.submit("after-stuck", "normal", request(), limits())
        assert pool.cancel(stuck)
        assert pool.accounting()["active"] == 1
        assert store.get_run("after-stuck")["state"] == "queued"
        value = assert_reaped(pool.wait(stuck, 3))
        assert value["store_state"] == "cancelled" and value["exitcode"] == -signal.SIGKILL
        assert value["signals"] == ["soft_stop", "terminate", "kill"]
        assert assert_reaped(pool.wait(following, 3))["store_state"] == "succeeded"


def test_timeout_records_full_cleanup_cost_and_allows_next_job(tmp_path):
    with RunStore(tmp_path / "db") as store, supervisor(store, tmp_path / "work") as pool:
        handle = pool.submit("timeout", "uncooperative", request(), limits(wall_seconds=0.4))
        value = assert_reaped(pool.wait(handle, 3))
        assert value["store_state"] == "failed" and value["worker_result"]["reason"] == "wall_limit"
        assert value["resources"]["owned_wall_seconds"] >= 0.4
        following = pool.submit("after-timeout", "normal", request(), limits())
        assert pool.wait(following, 3).data["store_state"] == "succeeded"


@pytest.mark.parametrize("worker, reason", [("exception", "worker_exception"), ("crash", "child_crash"), ("nested_pool", "worker_exception")])
def test_child_exception_crash_and_nested_worker_rejection_are_terminal(tmp_path, worker, reason):
    with RunStore(tmp_path / "db") as store, supervisor(store, tmp_path / "work") as pool:
        handle = pool.submit("broken", worker, request(), limits())
        value = assert_reaped(pool.wait(handle, 3))
        assert value["store_state"] == "failed" and value["worker_result"]["reason"] == reason
        if worker == "crash":
            assert value["exitcode"] == 17
        if worker == "nested_pool":
            assert value["worker_result"]["exception_type"] == "AssertionError"
        assert len(terminal_events(store, "broken")) == 1
        assert pool.wait(pool.submit("after-broken", "normal", request(), limits()), 3).data["store_state"] == "succeeded"


def test_cumulative_counter_failure_and_missing_counter_are_not_zero_qualified(tmp_path):
    with RunStore(tmp_path / "db") as store, supervisor(store, tmp_path / "work") as pool:
        handle = pool.submit("counter", "counter_limit", request(), limits(accepted_steps=2))
        value = assert_reaped(pool.wait(handle, 3))
        assert value["store_state"] == "failed" and value["resources"]["accepted_steps"] == 3
        assert value["worker_result"]["reason"] == "accepted_steps_limit"
        missing = pool.wait(pool.submit("missing", "missing_counters", request(), limits(calls=0)), 3).data
        assert missing["store_state"] == "failed" and missing["resources"]["calls"] is None
        assert "unobserved" in missing["worker_result"].get("detail", "")
        assert pool.accounting()["counter_totals"]["calls"] is None


@pytest.mark.parametrize("worker, caps, reason", [
    ("normal", {"rss_bytes": 1}, "rss_limit"),
    ("disk_limit", {"output_bytes": 16384}, "output_limit"),
])
def test_actual_bounded_rss_and_disk_failures_preserve_observed_usage(tmp_path, worker, caps, reason):
    with RunStore(tmp_path / "db") as store, supervisor(store, tmp_path / "work") as pool:
        value = assert_reaped(pool.wait(pool.submit("cap", worker, request(), limits(**caps)), 3))
        assert value["store_state"] == "failed" and value["worker_result"]["reason"] == reason
        if reason == "rss_limit":
            assert value["resources"]["child_peak_rss_bytes"] > 1
        else:
            assert value["resources"]["output_bytes"] > 16384
            assert (tmp_path / "work" / (store.store_id + "-" + value["attempt_id"]) / "worker.log").stat().st_size == 16384


def test_external_cancel_and_new_generation_reject_late_success(tmp_path):
    with RunStore(tmp_path / "db") as store, supervisor(store, tmp_path / "work", soft_grace_seconds=0.4) as pool:
        payload = request()
        first = pool.submit("retry", "barrier_success", payload, limits())
        ready(store, first)
        old = store.get_run("retry")
        store.cancel("retry", expected_attempt_id=first.attempt_id, expected_generation=old["generation"], reason="external authority")
        queued = store.retry("retry", expected_attempt_id=first.attempt_id, expected_generation=old["generation"])
        release(tmp_path, store, first)
        following = pool.submit("retry", "barrier_success", payload, limits())
        assert following.attempt_id == queued["current_attempt_id"] != first.attempt_id
        value = assert_reaped(pool.wait(first, 3))
        assert value["store_state"] == "cancelled" and not value["terminal_committed"]
        assert value["worker_result"]["discarded_terminal_state"] == "succeeded"
        ready(store, following)
        release(tmp_path, store, following)
        assert assert_reaped(pool.wait(following, 3))["store_state"] == "succeeded"
        assert [a["state"] for a in store.attempts("retry")] == ["cancelled", "succeeded"]
        assert len(terminal_events(store, "retry")) == 2


def test_cancel_decision_wins_over_success_terminal_file(tmp_path):
    with RunStore(tmp_path / "db") as store, supervisor(store, tmp_path / "work", soft_grace_seconds=0.4) as pool:
        handle = pool.submit("late", "barrier_success", request(), limits())
        ready(store, handle)
        pool.cancel(handle)
        release(tmp_path, store, handle)
        value = assert_reaped(pool.wait(handle, 3))
        directory = tmp_path / "work" / (store.store_id + "-" + handle.attempt_id)
        assert json.loads((directory / "terminal.json").read_text())["state"] == "succeeded"
        assert value["store_state"] == "cancelled" and value["worker_result"]["reason"] == "cancel_requested"
        assert len(terminal_events(store, "late")) == 1


def test_reopen_reconciles_actual_join_after_db_failure_and_retains_queued_work(tmp_path, monkeypatch):
    database, work = tmp_path / "db", tmp_path / "work"
    store = RunStore(database)
    pool = supervisor(store, work)
    current = pool.submit("interrupted", "cooperative", request(), limits())
    ready(store, current)
    queued = pool.submit("queued-restart", "normal", request(), limits())
    def fail_commit(*args, **kwargs):
        raise sqlite3.OperationalError("injected after actual child join before terminal commit")
    monkeypatch.setattr(store, "complete", fail_commit)
    with pytest.raises(RuntimeError, match="supervisor failed"):
        pool.close()
    first = assert_reaped(pool.wait(current, 1))
    assert not first["terminal_committed"] and first["store_state"] == "running"
    assert store.get_run("queued-restart")["state"] == "queued"
    store.close()
    with RunStore(database) as reopened, supervisor(reopened, work) as recovered:
        assert recovered.accounting()["unresolved_owned_slots"] == 1
        recovery = recovered.reconcile_interrupted()
        assert recovery["released_owned_reservations"] == [current.attempt_id]
        assert recovery["invalidated_owned_attempts"] == [current.attempt_id]
        assert reopened.get_run("interrupted")["state"] == "failed"
        resumed = recovered.submit("queued-restart", "normal", request(), limits())
        assert resumed == queued
        assert assert_reaped(recovered.wait(resumed, 3))["store_state"] == "succeeded"
        assert len(reopened.attempts("interrupted")) == len(reopened.attempts("queued-restart")) == 1
    # Positive reconciliation remains deterministic on another reopen.
    with RunStore(database) as reopened, supervisor(reopened, work) as recovered:
        assert recovered.accounting()["unresolved_owned_slots"] == 0


def make_uncertain_intent(store, work, *, reused):
    """Manufactured metadata exercises PID safety; it does not claim a spawned child."""
    row = store.enqueue("old-owner", opaque_identity={"H_input": "i", "H_physics": "p", "H_execution": "e"},
                        input_metadata={"fixture": "metadata-only restored PID counterexample"})
    token = store.claim("old-owner", expected_attempt_id=row["current_attempt_id"], expected_generation=row["generation"], owner_id="old-owner")
    work.mkdir()
    directory = work / (store.store_id + "-" + token.attempt_id)
    directory.mkdir()
    identity = _process_info([os.getpid()])[os.getpid()]
    if reused:
        identity = {**identity, "start_lstart": "different former process start"}
    (directory / "launch.json").write_text(json.dumps({"token": asdict(token)}))
    (directory / "started.json").write_text(json.dumps({"token": asdict(token), "pid": os.getpid(), "os_identity": identity}))
    return token


@pytest.mark.parametrize("reused", [False, True])
def test_recovery_never_kills_a_restored_pid_and_only_reserves_affected_slot(tmp_path, monkeypatch, reused):
    with RunStore(tmp_path / "db") as store:
        token = make_uncertain_intent(store, tmp_path / "work", reused=reused)
        calls = []
        real_kill = os.kill
        restored_pid = os.getpid()
        def presence_only(pid, sig):
            if pid == restored_pid:
                calls.append((pid, sig))
                assert sig == 0, "an unowned restored PID must never be signalled"
            return real_kill(pid, sig)
        monkeypatch.setattr(os, "kill", presence_only)
        with supervisor(store, tmp_path / "work", max_active=2) as pool:
            recovery = pool.reconcile_interrupted()
            assert recovery["invalidated_owned_attempts"] == [token.attempt_id]
            assert pool.accounting()["unresolved_owned_slots"] == (0 if reused else 1)
            following = pool.submit("unrelated", "normal", request(), limits())
            assert pool.wait(following, 3).data["store_state"] == "succeeded"
            assert pool.accounting()["counter_totals"]["calls"] is None
        assert calls and all(sig == 0 for _, sig in calls)


def test_request_allowlist_source_schema_and_immutable_transport(tmp_path):
    original = request(value=-0.0, clear=None)
    assert WorkerRequest.from_bytes(original.to_bytes()) == original
    with pytest.raises(FrozenInstanceError):
        original.H_input = "forged"
    malformed = json.loads(original.to_bytes())
    malformed["module"] = "os"
    with pytest.raises(ValueError, match="fields"):
        WorkerRequest.from_bytes(json.dumps(malformed).encode())
    def closure(a, b):
        return (a, b)
    with pytest.raises(ValueError, match="closures"):
        WorkerBinding.from_function("nested", closure, registry_sha256=REGISTRY.sha256, schema_sha256=SCHEMA.sha256)
    with RunStore(tmp_path / "db") as store, supervisor(store, tmp_path / "work") as pool:
        with pytest.raises(ValueError, match="allowlist"):
            pool.submit("bad", "os.system", original, limits())
        with pytest.raises(ValueError, match="registry/schema"):
            pool.submit("bad", "normal", replace(original, schema=SourceDocument("same-name", b"changed")), limits())
        assert store.events() == []
        h = pool.submit("valid", "normal", original, limits())
        with pytest.raises(StateConflict):
            pool.submit("valid", "normal", request(value=2), limits())
        assert pool.wait(h, 3).data["store_state"] == "succeeded"


def test_changed_trusted_worker_source_digest_is_rejected_in_the_spawn_child(tmp_path):
    with RunStore(tmp_path / "db") as store:
        bad = replace(bindings()[0], source_sha256="0" * 64)
        with ProcessSupervisor(store, tmp_path / "work", (bad,), max_active=1, max_pending=0,
                               shared_rss_bytes=768 * 1024**2, shared_output_bytes=16 * 1024**2) as pool:
            value = assert_reaped(pool.wait(pool.submit("source", bad.id, request(), limits()), 3))
            assert value["store_state"] == "failed" and "source content changed" in value["worker_result"]["detail"]


@pytest.mark.parametrize("malformed", ["not_mapping", "missing_name"])
def test_malformed_terminal_is_isolated_and_independent_jobs_continue(tmp_path, malformed):
    with RunStore(tmp_path / "db") as store, supervisor(store, tmp_path / "work", max_active=2) as pool:
        independent = pool.submit("independent", "barrier_success", request(), limits())
        ready(store, independent)
        bad = pool.submit("bad-terminal", "malformed_terminal", request(malformed=malformed), limits())
        following = pool.submit("following", "normal", request(), limits())
        value = assert_reaped(pool.wait(bad, 4))
        assert value["store_state"] == "failed" and value["worker_result"]["reason"] == "invalid_terminal"
        release(tmp_path, store, independent)
        assert pool.wait(independent, 3).data["store_state"] == "succeeded"
        assert pool.wait(following, 3).data["store_state"] == "succeeded"
        assert len(terminal_events(store, "bad-terminal")) == 1


@pytest.mark.parametrize("kind, size", [("artifact", 98304), ("result", 65536)])
def test_shared_output_reserve_covers_publication_and_terminal_growth(tmp_path, kind, size):
    cap = 1024 * 1024
    with RunStore(tmp_path / "db") as store, supervisor(store, tmp_path / "work", shared_output_bytes=cap) as pool:
        handle = pool.submit("near-cap", "finalize_after_release", request(kind=kind, bytes=size), limits(output_bytes=512 * 1024))
        ready(store, handle)
        existing = pool.accounting()["retained_output_peak_bytes"]
        # Simulate concurrently retained output: the child result alone fits,
        # but its parent publication/terminal copies cannot fit the shared cap.
        retained = cap - 3 * size // 2 - existing
        assert retained > 0
        (tmp_path / "work" / "retained-output.bin").write_bytes(b"r" * retained)
        directory = tmp_path / "work" / (store.store_id + "-" + handle.attempt_id)
        (directory / "release").touch()
        value = assert_reaped(pool.wait(handle, 4))
        assert value["store_state"] == "failed"
        assert value["worker_result"]["reason"].startswith("shared_output_")
        assert not value["terminal_artifacts"]
        assert (directory / "terminal.json").exists() and (directory / "finalization.json").exists()
        finalization = value["finalization"]
        assert finalization["shared_output_after_finalization_record_bytes"] <= pool.accounting()["retained_output_peak_bytes"]
        assert finalization["finalization_overrun"] == (finalization["shared_output_after_finalization_record_bytes"] > cap)
        assert (tmp_path / "work" / "retained-output.bin").stat().st_size == retained


def test_post_spawn_setup_failure_reaps_only_affected_child(tmp_path, monkeypatch):
    real_write = execution_module._atomic_json
    def fail_one_spawn(path, value):
        if path.name == "spawned.json" and value["token"]["run_id"] == "setup-error":
            raise OSError("injected after actual process.start")
        return real_write(path, value)
    monkeypatch.setattr(execution_module, "_atomic_json", fail_one_spawn)
    with RunStore(tmp_path / "db") as store, supervisor(store, tmp_path / "work", max_active=2) as pool:
        independent = pool.submit("setup-independent", "barrier_success", request(), limits())
        ready(store, independent)
        bad = pool.submit("setup-error", "cooperative", request(), limits())
        value = assert_reaped(pool.wait(bad, 4))
        assert value["store_state"] == "failed" and value["worker_result"]["reason"] == "spawn_setup_failed"
        assert "injected" in value["worker_result"]["setup_error"]
        release(tmp_path, store, independent)
        assert pool.wait(independent, 3).data["store_state"] == "succeeded"


def test_directory_census_reobserves_an_atomic_publication(tmp_path, monkeypatch):
    partial, published = tmp_path / "started.json.part", tmp_path / "started.json"
    payload = b'{"state":"ready"}'
    partial.write_bytes(payload)
    (tmp_path / "stable.bin").write_bytes(b"stable")
    original = Path.lstat
    renamed = False
    def publish_before_stat(path, *args, **kwargs):
        nonlocal renamed
        if path == partial and not renamed:
            partial.replace(published)
            renamed = True
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "lstat", publish_before_stat)
    assert _directory_bytes(tmp_path) == len(payload) + len(b"stable")
    assert renamed and published.read_bytes() == payload and not partial.exists()


def test_directory_census_preserves_other_stat_errors(tmp_path, monkeypatch):
    target = tmp_path / "unreadable.bin"
    target.write_bytes(b"data")
    original = Path.lstat
    def denied(path, *args, **kwargs):
        if path == target:
            raise PermissionError("fixture access denied")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "lstat", denied)
    with pytest.raises(PermissionError, match="fixture access denied"):
        _directory_bytes(tmp_path)
