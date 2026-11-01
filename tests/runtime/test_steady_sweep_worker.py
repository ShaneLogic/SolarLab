"""Real cold process points, bounded lifecycle, and solver-free stored mapping."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


from solarlab.config.identity import ExecutionEngine
from solarlab.experiments.execution import JobHandle, ProcessSupervisor, ScanSpec
from solarlab.experiments.jv.worker import SteadyJVProtocol, steady_jv_worker_binding
from solarlab.experiments.worker import ResourceLimits, WorkerBinding
from solarlab.io.run_store import RunStore, StateConflict
from solarlab.sweeps.inputs import SweepInput
from solarlab.sweeps.preparation import prepare_sweep
from solarlab.sweeps.steady_worker import (
    SteadySweepJobs, prepare_steady_sweep_jobs, read_steady_sweep_jobs,
)

pytestmark = pytest.mark.requires_clean_core

ROOT = Path(__file__).resolve().parents[2]
LIMITS = ResourceLimits(60., 1536 * 1024**2, 96 * 1024**2, None, None)
PROTOCOL = SteadyJVProtocol((0., .05, .1), 12, 1., None)


@pytest.fixture(scope="module")
def implementation():
    from examples.research_1d.service import committed_implementation
    commit = os.environ.get("SOLARLAB_PROCESS_SWEEP_SOURCE_PIN") or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=5).strip()
    # The full suite has loaded non-DC modules too; bind their committed bytes.
    return committed_implementation(commit)


@pytest.fixture(scope="module")
def sweep():
    from examples.research_1d.jv_scan import make_sweep
    return make_sweep(ROOT / "examples/research_1d/device.json", study_id="process-worker-test",
                      parameter="tau_n", values=["1 us", "2 us"])


@pytest.fixture(scope="module")
def jobs(sweep, implementation):
    return prepare_steady_sweep_jobs(sweep, PROTOCOL, limits=LIMITS, implementation=implementation,
        current_repository_head=implementation.build_source_commit, engine=ExecutionEngine("solarlab"))


def pool(store, work, extra=()):
    return ProcessSupervisor(store, work, (steady_jv_worker_binding(), *extra), max_active=1, max_pending=1,
        shared_rss_bytes=3 * 1024**3, shared_output_bytes=384 * 1024**2, poll_seconds=.01)


def terminals(store, run_id):
    return [event for event in store.events(run_id=run_id, limit=1000) if event["kind"] == "terminal"]


@pytest.fixture(scope="module")
def completed(tmp_path_factory, jobs):
    root = tmp_path_factory.mktemp("steady-process-sweep")
    with RunStore(root / "store") as store, pool(store, root / "work") as supervisor:
        scan = jobs.submit(supervisor)
        status = scan.wait(150)
        assert status["state"] == "succeeded", status
        receipts = [supervisor.wait(JobHandle(store.store_id, point["run_id"], point["attempt_id"]), 5).data
                    for point in status["points"]]
        declaration = scan.declaration_path
    return root, declaration, status, receipts


def test_real_two_point_cold_scan_commits_distinct_manifests_and_eight_arrays(completed, jobs, sweep):
    root, declaration, status, receipts = completed
    assert not sweep.can_execute and not status["scientific_qualification_inferred"]
    assert status["max_in_flight"] == 1 and not status["coordinator_uses_process_slot"]
    assert ScanSpec.from_bytes(declaration.read_bytes()) == jobs.spec
    assert jobs.executions[0].inputs.prepared is not jobs.executions[1].inputs.prepared
    assert jobs.spec.points[0].spec.manifest_sha256 != jobs.spec.points[1].spec.manifest_sha256
    assert jobs.spec.points[0].spec.initial_state_sha256 != jobs.spec.points[1].spec.initial_state_sha256
    for binding in jobs.executions:
        assert json.loads(binding.execution.initial_state.content)["cross_job_reuse"] is False
        preparation = json.loads(binding.execution.preparation.content)
        assert preparation["cold_compile_per_job"] is True
        assert any(item["id"] == "solarlab/sweeps/steady_worker.py" for item in preparation["adapter_sources"])
    assert all(r["process_joined"] and r["process_exit_observed"] and r["exitcode"] == 0 for r in receipts)
    with RunStore(root / "store", read_only=True) as store:
        points = read_steady_sweep_jobs(store, jobs.spec)
        assert [p["point_id"] for p in points] == [p["id"] for p in sweep.to_mapping()["points"]]
        assert [p["coordinates"]["normalized_coordinates"]["tau_n"]["value"] for p in points] == [1e-6, 2e-6]
        for point in points:
            assert point["state"] == "succeeded" and len(point["attempts"]) == 1
            assert len(terminals(store, point["run_id"])) == 1
            assert len(point["attempts"][0]["terminal"]["artifacts"]) == 9
            result = point["result"]
            assert len(result["arrays"]) == 8 and result["arrays"]["potential_V"].shape == (3, 37)
            assert result["arrays"]["voltage_V"].tolist() == [0., .05, .1]
            assert all(item["accepted"] for item in result["document"]["solver_evidence"])
            assert result["document"]["metrics"]["open_circuit_voltage_V"] is None
            assert result["document"]["metrics"]["efficiency"] is None
            assert result["document"]["metrics"]["limitation"] == "open_circuit_not_bracketed"
        assert points[0]["result"]["document"]["compiled_identity"] != points[1]["result"]["document"]["compiled_identity"]
        assert points[0]["result"]["arrays"]["output_current_A_m2"].tobytes() != points[1]["result"]["arrays"]["output_current_A_m2"].tobytes()


def test_independent_reader_never_imports_solver_and_refuses_changed_point(completed, jobs, tmp_path):
    root, declaration, _, _ = completed
    code = '''
import importlib.abc, json, pathlib, sys
class NoSolver(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == p or fullname.startswith(p + '.') for p in (
            'solarlab.solvers', 'solarlab.equations', 'solarlab.device.compile_1d', 'perovskite_sim')):
            raise AssertionError('reader attempted numerical import: ' + fullname)
sys.meta_path.insert(0, NoSolver())
from solarlab.experiments.execution import ScanSpec
from solarlab.io.run_store import RunStore
from solarlab.sweeps.steady_worker import read_steady_sweep_jobs
with RunStore(pathlib.Path(sys.argv[1]), read_only=True) as store:
    points = read_steady_sweep_jobs(store, ScanSpec.from_bytes(pathlib.Path(sys.argv[2]).read_bytes()))
    assert all(p['state'] == 'succeeded' and len(p['result']['arrays']) == 8 for p in points)
    print(json.dumps([p['result']['worker_result']['array_payloads'] for p in points], sort_keys=True))
'''
    readback = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(root / "store"), str(declaration)],
        cwd=tmp_path, capture_output=True, text=True, timeout=20)
    assert readback.returncode == 0, readback.stderr
    with RunStore(root / "store", read_only=True) as store:
        points = read_steady_sweep_jobs(store, jobs.spec)
        assert json.loads(readback.stdout) == [p["result"]["worker_result"]["array_payloads"] for p in points]
        coordinates = json.loads(jobs.spec.points[0].coordinates_json)
        coordinates["normalized_coordinates"]["tau_n"]["value"] = 3e-6
        changed = replace(jobs.spec.points[0], coordinates_json=json.dumps(coordinates))
        with pytest.raises(ValueError, match="association differs"):
            read_steady_sweep_jobs(store, replace(jobs.spec, points=(changed, jobs.spec.points[1])))


def test_wall_failure_preserves_attempt_and_next_point_runs_cold(tmp_path, jobs):
    first = jobs.spec.points[0]
    timed = replace(first, spec=replace(first.spec, limits=replace(LIMITS, wall_seconds=.000001)))
    bounded = SteadySweepJobs(replace(jobs.spec, points=(timed, jobs.spec.points[1])), jobs.executions)
    with RunStore(tmp_path / "store") as store, pool(store, tmp_path / "work") as supervisor:
        status = bounded.submit(supervisor).wait(90)
        assert status["complete"] and status["state"] == "failed"
        points = read_steady_sweep_jobs(store, bounded.spec)
        assert [p["state"] for p in points] == ["failed", "succeeded"]
        assert points[0]["result"] is None and points[0]["reason"] == "wall_limit"
        assert points[0]["attempts"][0]["terminal"]["artifacts"] == []
        assert len(points[1]["result"]["arrays"]) == 8
        assert all(len(terminals(store, p["run_id"])) == 1 for p in points)


def _hold(request, context):
    """Non-numerical fixture occupying the shared owner's one process slot."""
    context.consume_manifest(request, engine="solarlab")
    context.progress({"stage": "shared_slot_ready"})
    while True:
        context.check_cancel()
        time.sleep(.005)


def test_shared_capacity_exact_cancel_and_explicit_retry_keep_prior_attempts(tmp_path, jobs):
    request, execution = jobs.spec.points[0].spec.request, jobs.executions[0]
    blocker = WorkerBinding.from_function("shared_slot", _hold,
        registry_sha256=request.registry.sha256, schema_sha256=request.schema.sha256)
    with RunStore(tmp_path / "store") as store, pool(store, tmp_path / "work", (blocker,)) as supervisor:
        held = supervisor.submit("other-owner", "shared_slot", request, LIMITS, execution=execution)
        deadline = time.monotonic() + 15
        while not any(e["payload"].get("ui_progress", {}).get("stage") == "shared_slot_ready"
                      for e in store.events(run_id=held.run_id, limit=1000)):
            if time.monotonic() >= deadline:
                pytest.fail("bounded shared-slot fixture never became ready")
            time.sleep(.01)
        scan = jobs.submit(supervisor)
        status = scan.advance()
        assert [p["state"] for p in status["points"]] == ["queued", "declared"]
        assert supervisor.accounting()["active"] == 1
        first = status["points"][0]
        handle = JobHandle(store.store_id, first["run_id"], first["attempt_id"])
        assert supervisor.cancel(handle, reason="point_cancel")
        assert supervisor.wait(handle, 5).data["process_started"] is False
        retry = scan.retry(first["point_id"], expected_attempt_id=handle.attempt_id)
        assert retry.attempt_id != handle.attempt_id
        assert not supervisor.cancel(handle)  # An old handle cannot cancel the new attempt.
        assert scan.cancel(reason="study_cancel")
        assert scan.wait(5)["state"] == "cancelled"
        points = read_steady_sweep_jobs(store, jobs.spec)
        assert all(p["state"] == "cancelled" and p["result"] is None for p in points)
        assert [a["state"] for a in points[0]["attempts"]] == ["cancelled", "cancelled"]
        assert points[0]["attempts"][0]["reason"] == "point_cancel"
        assert points[0]["attempts"][1]["reason"] == "study_cancel"
        assert len(terminals(store, first["run_id"])) == 2
        assert len(terminals(store, points[1]["run_id"])) == 1
        assert not scan.cancel(reason="duplicate")
        with pytest.raises(StateConflict, match="cancelled scan"):
            scan.retry(first["point_id"], expected_attempt_id=retry.attempt_id)
        assert store.get_run(held.run_id)["state"] == "running"
        assert supervisor.cancel(held) and supervisor.wait(held, 10).data["process_joined"]


@pytest.mark.parametrize("counter", ["accepted_steps", "calls"])
def test_unsupported_counter_caps_refuse_before_preparing_any_job(sweep, implementation, monkeypatch, counter):
    import solarlab.sweeps.steady_worker as adapter
    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported counters reached point preparation")
    monkeypatch.setattr(adapter, "prepare_steady_jv_job", forbidden)
    with pytest.raises(ValueError, match="counter caps are unsupported"):
        prepare_steady_sweep_jobs(sweep, PROTOCOL, limits=replace(LIMITS, **{counter: 1}),
            implementation=implementation, current_repository_head=implementation.build_source_commit,
            engine=ExecutionEngine("solarlab"))


def test_unsupported_ions_and_unresolved_points_refuse_without_native_work(sweep, implementation, monkeypatch):
    import solarlab.device.compile_1d as compiler
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid sweep reached native compilation")
    monkeypatch.setattr(compiler, "compile_device_1d", forbidden)
    raw = sweep.to_input().model_dump(mode="json", exclude_unset=True)
    raw["base"]["layers"][1]["parameters"]["D_ion"] = 1e-15
    unsupported = prepare_sweep(SweepInput.model_validate(raw), sweep.defaults, sweep.resources, sources=sweep.sources)
    with pytest.raises(ValueError, match="D_ion"):
        prepare_steady_sweep_jobs(unsupported, PROTOCOL, limits=LIMITS, implementation=implementation,
            current_repository_head=implementation.build_source_commit, engine=ExecutionEngine("solarlab"))
    raw = sweep.to_input().model_dump(mode="json", exclude_unset=True)
    raw["axes"][0]["coordinates"][0]["value"] = "-1 us"
    unresolved = prepare_sweep(SweepInput.model_validate(raw), sweep.defaults, sweep.resources, sources=sweep.sources)
    with pytest.raises(ValueError, match="unresolved sweep point"):
        prepare_steady_sweep_jobs(unresolved, PROTOCOL, limits=LIMITS, implementation=implementation,
            current_repository_head=implementation.build_source_commit, engine=ExecutionEngine("solarlab"))
