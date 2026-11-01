"""A missing startup identity cannot become the registered worker's failure."""
from dataclasses import asdict, replace
from functools import partial
import json
import os
from types import SimpleNamespace

import pytest

from solarlab.experiments import execution, worker
from solarlab.io.run_store import AttemptToken, RunStore, StateConflict
from solarlab.materials.source import SourceDocument
from solarlab_server.job_service import JobSubmission, canonical
from job_route_fixtures import catalog, request as declaration
from ps_timeout_workers import identity_timeout


# Caller patience only: the four CPU-hog pilot's slowest case took 12.84 s.
# Use 6x the measured maximum, with a 30 s floor, rounded up to 30 s.
# Job limits stay unchanged, so hung workers still fail the assertions.
_LIVE_WORKER_WAIT_SECONDS = 90


@pytest.fixture(scope="module")
def registered():
    return catalog()[0]


def assert_identity_readback(tmp_path, bound, handle, result, *, unavailable):
    directory = tmp_path / "work" / (handle.store_id + "-" + handle.attempt_id)
    terminal = json.loads((directory / "terminal.json").read_text())
    started = json.loads((directory / "started.json").read_text())
    disk_receipt = json.loads((directory / "receipt.json").read_text())
    expected = {"scope": "child_start_identity_only", "status": "unavailable", "missing_samples": 3}
    with RunStore(tmp_path / "store") as store, execution.ProcessSupervisor(store, tmp_path / "work", (bound,),
            max_active=1, max_pending=0, shared_rss_bytes=4 * 1024**3, shared_output_bytes=256 * 1024**2) as pool:
        recovered = pool.wait(handle, 1).data
        persisted = store.attempts(handle.run_id)[0]["result"]
        events = [event for event in store.events(run_id=handle.run_id) if event["kind"] == "terminal"]
        assert len(events) == 1
        for record in (result, disk_receipt, recovered, persisted, events[0]["payload"]["result"]):
            if unavailable:
                assert record["child_start_identity_observation"] == expected
            else:
                assert "child_start_identity_observation" not in record
            assert record["worker_result"] == terminal["result"]
            assert "child_start_identity_observation" not in record["worker_result"]
        process = persisted["process"]
        # Parent sampling stays real even when the child's startup query failed.
        assert process["os_identity"]["pid"] == started["pid"] == result["pid"]
        assert "child_start_identity_observation" not in process
        assert started["token"] == terminal["token"] == result["token"] == process["token"]
        assert process["process_joined"] and process["process_exit_observed"]
        assert result["terminal_committed"] and result["signals"] == process["signals"] == []
        with pytest.raises(ChildProcessError):
            os.waitpid(result["pid"], os.WNOHANG)
        token = AttemptToken(**result["token"])
        for stale in (replace(token, generation=token.generation + 1), replace(token, owner_id="other-owner")):
            with pytest.raises(StateConflict):
                store.complete(stale, state=result["store_state"], result=persisted,
                               artifacts=result["terminal_artifacts"])
        assert store.attempts(handle.run_id)[0]["result"] == persisted
        assert [event for event in store.events(run_id=handle.run_id) if event["kind"] == "terminal"] == events


@pytest.mark.parametrize("field", ["job_schema_sha256", "model_registry_sha256", "legacy_facade_sha256"])
@pytest.mark.parametrize("misses", [1, 3])
def test_startup_ps_timeout_preserves_real_registry_refusal(tmp_path, registered, monkeypatch, field, misses):
    spec, binding = registered.prepare(JobSubmission.model_validate(declaration()))
    claimed = json.loads(registered.document.content)
    claimed[field] = "0" * 64
    altered = SourceDocument("explicit-inconsistent-registry-fixture", canonical(claimed))
    bound = replace(registered.worker_binding(), registry_sha256=altered.sha256)
    monkeypatch.setattr(execution, "_child_main", partial(identity_timeout, misses=misses))
    with RunStore(tmp_path / "store") as store, execution.ProcessSupervisor(store, tmp_path / "work", (bound,),
            max_active=1, max_pending=0, shared_rss_bytes=4 * 1024**3, shared_output_bytes=256 * 1024**2) as pool:
        handle = pool.submit("bad-registry", bound.id, replace(spec.request, registry=altered), spec.limits,
                             execution=binding)
        result = pool.wait(handle, _LIVE_WORKER_WAIT_SECONDS).data
        directory = tmp_path / "work" / (store.store_id + "-" + handle.attempt_id)
        terminal = json.loads((directory / "terminal.json").read_text())
        assert result["store_state"] == "failed" and result["worker_result"]["reason"] == "worker_exception"
        assert "registry" in result["worker_result"]["detail"], result
        assert result["worker_result"]["exception_type"] == "ValueError"
        assert not result["terminal_artifacts"] and result["process_joined"]
        assert result["process_exit_observed"] and result["pid"] != os.getpid()
        assert result["worker_result"] == terminal["result"]
        events = [event for event in store.events(run_id="bad-registry") if event["kind"] == "terminal"]
        assert len(events) == 1
        assert events[0]["payload"]["result"]["worker_result"] == terminal["result"]
        started = json.loads((directory / "started.json").read_text())
        observation = started["os_identity_observation"]
        assert observation["missing_samples"] == misses
        assert observation["attempts"] == (2 if misses == 1 else 3)
        assert (started["os_identity"] is None) == (misses == 3)
        assert started["pid"] == result["pid"]
        probes = json.loads((directory / "identity-probe.json").read_text())["observations"]
        assert len(probes) == observation["attempts"]
        assert all(row["pid"] == result["pid"] and row["ppid"] == os.getpid() for row in probes)
        assert all(row["timeout"] == 0.5 and not row["started_record_exists"] for row in probes)
        assert [row["outcome"] for row in probes] == ["injected_timeout"] * misses + (["controlled_success"] if misses == 1 else [])
    assert_identity_readback(tmp_path, bound, handle, result, unavailable=misses == 3)


@pytest.mark.parametrize("misses", [0, 3])
def test_successful_worker_retains_only_degraded_startup_observation(tmp_path, registered, monkeypatch, misses):
    spec, binding = registered.prepare(JobSubmission.model_validate(declaration()))
    bound = registered.worker_binding()
    monkeypatch.setattr(execution, "_child_main", partial(identity_timeout, misses=misses))
    with RunStore(tmp_path / "store") as store, execution.ProcessSupervisor(store, tmp_path / "work", (bound,),
            max_active=1, max_pending=0, shared_rss_bytes=4 * 1024**3, shared_output_bytes=256 * 1024**2) as pool:
        handle = pool.submit("identity-success", bound.id, spec.request, spec.limits, execution=binding)
        result = pool.wait(handle, _LIVE_WORKER_WAIT_SECONDS).data
        assert result["store_state"] == result["proposed_state"] == "succeeded", result
        assert result["terminal_artifacts"] and result["process_joined"]
        directory = tmp_path / "work" / (store.store_id + "-" + handle.attempt_id)
        started = json.loads((directory / "started.json").read_text())
        assert started["os_identity_observation"]["missing_samples"] == misses
        assert (started["os_identity"] is None) == (misses == 3)
    assert_identity_readback(tmp_path, bound, handle, result, unavailable=misses == 3)


@pytest.mark.parametrize("case", ["matching", "token", "bool_generation", "pid", "scope", "observed", "bool_count",
                                  "negative_count", "too_many", "malformed", "missing"])
def test_startup_diagnostic_requires_own_fence_and_unavailable_observation(tmp_path, case):
    token = AttemptToken("store", "run", "attempt", "owner", 1)
    observation = {"scope": "child_start_identity_only", "status": "unavailable", "missing_samples": 3}
    started = {"token": asdict(token), "pid": 123, "os_identity": None, "os_identity_observation": observation}
    if case == "token":
        started["token"]["generation"] += 1
    elif case == "bool_generation":
        started["token"]["generation"] = True
    elif case == "pid":
        started["pid"] = 456
    elif case == "scope":
        observation["scope"] = "parent_rss"
    elif case == "observed":
        observation["status"] = "observed"
    elif case in {"bool_count", "negative_count", "too_many"}:
        observation["missing_samples"] = {"bool_count": True, "negative_count": -1, "too_many": 4}[case]
    if case != "missing":
        (tmp_path / "started.json").write_text("{" if case == "malformed" else json.dumps(started))
    pool = object.__new__(execution.ProcessSupervisor)
    job = SimpleNamespace(directory=tmp_path, token=token, pid=123)
    diagnostic = pool._child_start_identity_diagnostic(job)
    assert diagnostic == ({"child_start_identity_observation": observation} if case == "matching" else {})


def test_startup_identity_bounds_match_supervisor_rss_sampler():
    assert worker._STARTUP_IDENTITY_MAX_MISSING_SAMPLES == execution._RSS_MAX_MISSING_SAMPLES == 3
    assert worker._STARTUP_IDENTITY_MAX_GAP_SECONDS == execution._RSS_MAX_GAP_SECONDS == 2.0


@pytest.mark.parametrize("mode", ["missing", "late_success", "late_retry"])
def test_child_identity_observation_stops_at_existing_bounds(monkeypatch, mode):
    clock = [0.0]
    calls = []

    def sample(pids):
        calls.append(pids)
        if mode == "missing":
            return {}
        if mode == "late_retry" and len(calls) == 1:
            clock[0] = 0.5
            return {}
        clock[0] = 2.0
        return {123: {"pid": 123, "start_lstart": "too late", "rss_bytes": 1}}

    monkeypatch.setattr(worker, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(worker, "_process_info", sample)
    identity, observation = worker._observe_child_identity(123)
    assert identity is None
    assert len(calls) == {"missing": 3, "late_success": 1, "late_retry": 2}[mode]
    assert observation["status"] == "unavailable"
    assert observation["max_missing_samples"] == execution._RSS_MAX_MISSING_SAMPLES == 3
    assert observation["max_gap_seconds"] == execution._RSS_MAX_GAP_SECONDS == 2.0
