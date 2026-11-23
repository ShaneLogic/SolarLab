"""P06-05 closeout: real RunSpec sweep equivalence, interrupted resume, retry and full-identity reuse."""
from __future__ import annotations

from dataclasses import replace
import json
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import time

import pytest

from solarlab.config.identity import ExecutionEngine
from solarlab.experiments.execution import JobHandle, ProcessSupervisor, ScanSpec
from solarlab.experiments.jv.worker import SteadyJVProtocol, steady_jv_worker_binding
from solarlab.experiments.worker import ResourceLimits
from solarlab.io.run_store import RunStore, StateConflict
from solarlab.materials.source import SourceDocument
from solarlab.sweeps.inputs import SweepInput
from solarlab.sweeps.preparation import prepare_sweep
from solarlab.sweeps.steady_worker import (
    SteadySweepJobs, prepare_steady_sweep_jobs, read_steady_sweep_jobs,
)

pytestmark = pytest.mark.requires_clean_core

ROOT = Path(__file__).resolve().parents[2]
LIMITS = ResourceLimits(60.0, 1536 * 1024**2, 96 * 1024**2, None, None)
PROTOCOL = SteadyJVProtocol((0.0, 0.05, 0.1), 8, 1.0, None)
ARRAY_NAMES = (
    "nodes_m",
    "voltage_V",
    "current_density_A_m2",
    "contact_current_density_A_m2",
    "output_current_A_m2",
    "potential_V",
    "electron_density_m3",
    "hole_density_m3",
)


@pytest.fixture(scope="module")
def implementation():
    from examples.research_1d.service import committed_implementation

    commit = os.environ.get("SOLARLAB_PROCESS_SWEEP_SOURCE_PIN") or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=5
    ).strip()
    return committed_implementation(commit)


@pytest.fixture(scope="module")
def three_point_sweep():
    from examples.research_1d.jv_scan import make_sweep

    return make_sweep(
        ROOT / "examples/research_1d/device.json",
        study_id="p06-05-three-point",
        parameter="tau_n",
        values=["1 us", "2 us", "4 us"],
    )


@pytest.fixture(scope="module")
def three_point_jobs(three_point_sweep, implementation):
    return prepare_steady_sweep_jobs(
        three_point_sweep,
        PROTOCOL,
        limits=LIMITS,
        implementation=implementation,
        current_repository_head=implementation.build_source_commit,
        engine=ExecutionEngine("solarlab"),
    )


def _pool(store: RunStore, work: Path, *, max_active: int = 1, max_pending: int = 2) -> ProcessSupervisor:
    return ProcessSupervisor(
        store,
        work,
        (steady_jv_worker_binding(),),
        max_active=max_active,
        max_pending=max_pending,
        shared_rss_bytes=3 * 1024**3,
        shared_output_bytes=384 * 1024**2,
        soft_grace_seconds=0.1,
        terminate_grace_seconds=0.1,
        poll_seconds=0.01,
    )


def _terminals(store: RunStore, run_id: str) -> list[dict]:
    return [event for event in store.events(run_id=run_id, limit=1000) if event["kind"] == "terminal"]


def _daemon_nested_supervisor_probe(store_root: str, work_root: str, result_queue: multiprocessing.Queue) -> None:
    try:
        with RunStore(Path(store_root)) as child_store:
            ProcessSupervisor(
                child_store,
                Path(work_root),
                (steady_jv_worker_binding(),),
                max_active=1,
                max_pending=1,
                shared_rss_bytes=512 * 1024**2,
                shared_output_bytes=64 * 1024**2,
            )
    except ValueError as error:
        result_queue.put(("rejected", str(error)))
    else:
        result_queue.put(("allowed", ""))


@pytest.fixture(scope="module")
def serial_reference(tmp_path_factory, three_point_jobs):
    root = tmp_path_factory.mktemp("p06-05-serial-ref")
    with RunStore(root / "store") as store, _pool(store, root / "work", max_active=1, max_pending=2) as supervisor:
        scan = three_point_jobs.submit(supervisor, max_in_flight=1)
        status = scan.wait(180)
        assert status["complete"] and status["state"] == "succeeded", status
        assert status["max_in_flight"] == 1 and status["coordinator_uses_process_slot"] is False
        assert status["scientific_qualification_inferred"] is False
        declaration_bytes = scan.declaration_path.read_bytes()
    with RunStore(root / "store", read_only=True) as reader:
        points = read_steady_sweep_jobs(reader, ScanSpec.from_bytes(declaration_bytes))
    return points


def _assert_points_bitwise_equivalent(actual: tuple[dict, ...], reference: tuple[dict, ...]) -> None:
    assert [p["point_id"] for p in actual] == [p["point_id"] for p in reference]
    assert [p["coordinates"]["index"] for p in actual] == [0, 1, 2]
    assert [p["coordinates"]["normalized_coordinates"]["tau_n"]["value"] for p in actual] == [1e-6, 2e-6, 4e-6]
    for got, ref in zip(actual, reference, strict=True):
        assert got["state"] == ref["state"] == "succeeded"
        got_res, ref_res = got["result"], ref["result"]
        assert got_res is not None and ref_res is not None
        assert got_res["worker_result"]["scientific_qualification"] is False
        assert ref_res["worker_result"]["scientific_qualification"] is False
        assert got_res["document"]["compiled_identity"] == ref_res["document"]["compiled_identity"]
        assert got_res["document"]["metrics"] == ref_res["document"]["metrics"]
        assert got_res["document"]["solver_evidence"] == ref_res["document"]["solver_evidence"]
        assert set(got_res["arrays"]) == set(ARRAY_NAMES)
        for name in ARRAY_NAMES:
            assert got_res["arrays"][name].dtype == ref_res["arrays"][name].dtype
            assert got_res["arrays"][name].shape == ref_res["arrays"][name].shape
            assert got_res["arrays"][name].tobytes() == ref_res["arrays"][name].tobytes()


def test_real_sweep_serial_parallel_and_random_order_bitwise_equivalence_and_non_starvation(
    tmp_path, three_point_sweep, three_point_jobs, serial_reference
):
    assert not three_point_sweep.can_execute

    # 1. Bounded-parallel execution (max_active=2, max_in_flight=2).
    par_store = (tmp_path / "parallel-store").resolve()
    par_work = (tmp_path / "parallel-work").resolve()
    par_jobs = prepare_steady_sweep_jobs(
        three_point_sweep,
        PROTOCOL,
        limits=LIMITS,
        implementation=three_point_jobs.executions[0].execution.implementation,
        current_repository_head=three_point_jobs.executions[0].execution.current_repository_head,
        engine=ExecutionEngine("solarlab"),
        scan_id="p06-05-parallel",
    )
    with RunStore(par_store) as store, _pool(store, par_work, max_active=2, max_pending=2) as supervisor:
        par_scan = par_jobs.submit(supervisor, max_in_flight=2)
        par_status = par_scan.wait(180)
        assert par_status["complete"] and par_status["state"] == "succeeded", par_status
        assert par_status["max_in_flight"] == 2
        assert par_status["coordinator_uses_process_slot"] is False
        assert par_status["scientific_qualification_inferred"] is False
        par_decl = par_scan.declaration_path.read_bytes()
    with RunStore(par_store, read_only=True) as reader:
        par_points = read_steady_sweep_jobs(reader, ScanSpec.from_bytes(par_decl))
    _assert_points_bitwise_equivalent(par_points, serial_reference)

    # 2. Permuted / random point dispatch order [p2, p0, p1] with max_in_flight=2 > max_active=1:
    # proves coordinator never occupies or starves the single worker slot AND readback restores declared order.
    declared_ids = [p.point_id for p in three_point_jobs.spec.points]
    permuted_ids = [declared_ids[2], declared_ids[0], declared_ids[1]]
    reordered_jobs = three_point_jobs.reordered(permuted_ids, scan_id="p06-05-random-order")
    assert [p.point_id for p in reordered_jobs.spec.points] == permuted_ids
    assert [json.loads(p.coordinates_json)["index"] for p in reordered_jobs.spec.points] == [2, 0, 1]

    with pytest.raises(ValueError, match="exact permutation"):
        three_point_jobs.reordered([declared_ids[0], declared_ids[0], declared_ids[1]])

    rand_store = (tmp_path / "random-store").resolve()
    rand_work = (tmp_path / "random-work").resolve()
    with RunStore(rand_store) as store, _pool(store, rand_work, max_active=1, max_pending=1) as supervisor:
        rand_scan = reordered_jobs.submit(supervisor, max_in_flight=2)
        rand_status = rand_scan.wait(180)
        assert rand_status["complete"] and rand_status["state"] == "succeeded", rand_status
        assert rand_status["max_in_flight"] == 2
        assert rand_status["coordinator_uses_process_slot"] is False
        rand_decl = rand_scan.declaration_path
        restored = SteadySweepJobs.from_declaration(rand_decl, executions=three_point_jobs.executions)
        assert [p.point_id for p in restored.spec.points] == permuted_ids
    with RunStore(rand_store, read_only=True) as reader:
        rand_points = read_steady_sweep_jobs(reader, restored.spec)
    _assert_points_bitwise_equivalent(rand_points, serial_reference)

    # 3. Daemon worker processes cannot construct nested ProcessSupervisors (no nested worker-pool deadlock).
    ctx = multiprocessing.get_context("spawn")
    nested_store = (tmp_path / "nested-store").resolve()
    nested_work = (tmp_path / "nested-work").resolve()
    with RunStore(nested_store):
        pass
    queue: multiprocessing.Queue = ctx.Queue()
    proc = ctx.Process(
        target=_daemon_nested_supervisor_probe,
        args=(str(nested_store), str(nested_work), queue),
        daemon=True,
    )
    proc.start()
    outcome, detail = queue.get(timeout=15)
    proc.join(timeout=10)
    assert proc.exitcode == 0
    assert outcome == "rejected" and "nested supervisors or scans" in detail


def test_real_sweep_interrupted_resumption_and_explicit_point_retry_preserve_failed_attempt(
    tmp_path, three_point_jobs, serial_reference
):
    store_root = (tmp_path / "resume-store").resolve()
    work_root = (tmp_path / "resume-work").resolve()
    p0, p1, p2 = three_point_jobs.spec.points

    # Session 1: p0 completes, p1 is killed mid-flight, p2 remains unstarted ('declared').
    with RunStore(store_root) as store, _pool(store, work_root, max_active=1, max_pending=1) as supervisor:
        scan = three_point_jobs.submit(supervisor, max_in_flight=1)
        declaration_path = scan.declaration_path

        deadline = time.monotonic() + 90
        p1_attempt_id = None
        p1_pid = None
        while time.monotonic() < deadline:
            snap = scan.advance()
            s0, s1, s2 = snap["points"]
            if s0["state"] == "succeeded" and s1["attempt_id"] is not None:
                p1_attempt_id = s1["attempt_id"]
                spawned_file = work_root / f"{store.store_id}-{p1_attempt_id}" / "spawned.json"
                if spawned_file.exists():
                    p1_pid = json.loads(spawned_file.read_text())["pid"]
                    assert s2["state"] == "declared" and s2["attempt_id"] is None
                    break
            time.sleep(0.002)
        else:
            pytest.fail("timed out waiting for p0 completion and p1 spawn")

        assert isinstance(p1_pid, int) and p1_pid > 0
        os.kill(p1_pid, signal.SIGKILL)
        failed_receipt = supervisor.wait(JobHandle(store.store_id, p1.run_id, p1_attempt_id), 15).data
        assert failed_receipt["store_state"] == "failed"
        assert failed_receipt["process_exit_observed"] is True and failed_receipt["process_joined"] is True
        assert store.get_run(p0.run_id)["state"] == "succeeded"
        assert store.get_run(p1.run_id)["state"] == "failed"
        with pytest.raises(KeyError):
            store.get_run(p2.run_id)  # p2 was never enqueued before session interruption.

    # Session 2: reopen RunStore + ProcessSupervisor, restore persisted ScanSpec, reuse p0, run p2, retry p1.
    with RunStore(store_root) as store2, _pool(store2, work_root, max_active=1, max_pending=1) as supervisor2:
        resumed_jobs = SteadySweepJobs.from_declaration(declaration_path, executions=three_point_jobs.executions)
        resumed = resumed_jobs.submit(supervisor2, max_in_flight=1)
        mid = resumed.advance()
        assert mid["points"][0]["state"] == "succeeded" and mid["points"][0]["reused"] is True
        assert len(store2.attempts(p0.run_id)) == 1
        assert mid["points"][1]["state"] == "failed" and mid["points"][1]["attempt_id"] == p1_attempt_id
        assert mid["points"][2]["state"] in {"queued", "running"}

        with pytest.raises(StateConflict, match="exact current failed/cancelled attempt"):
            resumed.retry(p1.point_id, expected_attempt_id="stale-attempt-id")
        retry_handle = resumed.retry(p1.point_id, expected_attempt_id=p1_attempt_id)
        assert retry_handle.attempt_id != p1_attempt_id

        final = resumed.wait(120)
        assert final["complete"] and final["state"] == "succeeded", final
        assert final["points"][0]["reused"] is True
        assert len(store2.attempts(p0.run_id)) == 1
        assert len(store2.attempts(p2.run_id)) == 1
        p1_attempts = store2.attempts(p1.run_id)
        assert [a["state"] for a in p1_attempts] == ["failed", "succeeded"]
        assert p1_attempts[0]["attempt_id"] == p1_attempt_id
        assert p1_attempts[0]["terminal"]["artifacts"] == []
        assert len(p1_attempts[1]["terminal"]["artifacts"]) == 9
        assert len(_terminals(store2, p0.run_id)) == 1
        assert len(_terminals(store2, p1.run_id)) == 2
        assert len(_terminals(store2, p2.run_id)) == 1

    with RunStore(store_root, read_only=True) as reader:
        points = read_steady_sweep_jobs(reader, resumed_jobs.spec)
        assert len(points[1]["attempts"]) == 2
        assert points[1]["attempts"][0]["state"] == "failed" and points[1]["attempts"][0]["steady_result"] is None
        assert points[1]["attempts"][1]["state"] == "succeeded" and points[1]["attempts"][1]["steady_result"] is not None
    _assert_points_bitwise_equivalent(points, serial_reference)


def test_real_sweep_completed_point_reuse_bound_to_full_execution_identity_and_mismatch_refusal(
    tmp_path, implementation
):
    from examples.research_1d.jv_scan import make_sweep

    sweep_a = make_sweep(
        ROOT / "examples/research_1d/device.json",
        study_id="p06-05-exec-a",
        parameter="tau_n",
        values=["1 us", "2 us"],
    )
    sweep_b = make_sweep(
        ROOT / "examples/research_1d/device.json",
        study_id="p06-05-exec-b",
        parameter="tau_n",
        values=["2 us", "4 us"],
    )
    jobs_a = prepare_steady_sweep_jobs(
        sweep_a,
        PROTOCOL,
        limits=LIMITS,
        implementation=implementation,
        current_repository_head=implementation.build_source_commit,
        engine=ExecutionEngine("solarlab"),
        reuse_scope="execution_identity",
    )
    jobs_b = prepare_steady_sweep_jobs(
        sweep_b,
        PROTOCOL,
        limits=LIMITS,
        implementation=implementation,
        current_repository_head=implementation.build_source_commit,
        engine=ExecutionEngine("solarlab"),
        reuse_scope="execution_identity",
    )
    shared_run_id = jobs_a.spec.points[1].run_id
    assert shared_run_id.startswith("exec:")
    assert jobs_b.spec.points[0].run_id == shared_run_id
    assert jobs_a.spec.points[0].run_id != shared_run_id
    assert jobs_b.spec.points[1].run_id != shared_run_id

    store_root = (tmp_path / "exec-reuse-store").resolve()
    work_root = (tmp_path / "exec-reuse-work").resolve()
    with RunStore(store_root) as store, _pool(store, work_root, max_active=1, max_pending=2) as supervisor:
        status_a = jobs_a.submit(supervisor, max_in_flight=1).wait(120)
        assert status_a["complete"] and status_a["state"] == "succeeded"
        assert [p["reused"] for p in status_a["points"]] == [False, False]
        shared_events_before = store.events(run_id=shared_run_id, limit=1000)

        status_b = jobs_b.submit(supervisor, max_in_flight=1).wait(120)
        assert status_b["complete"] and status_b["state"] == "succeeded"
        assert status_b["points"][0]["reused"] is True
        assert status_b["points"][1]["reused"] is False
        # Zero new attempts or events were created on the reused '2 us' point.
        assert len(store.attempts(shared_run_id)) == 1
        assert store.events(run_id=shared_run_id, limit=1000) == shared_events_before

        # Negative checks: mutating H_input, H_execution/protocol, BuildSource, ResourceLimits, or schema
        # never maps to the same execution-identity run_id, and forcing the old run_id raises StateConflict.
        raw_input_changed = sweep_b.to_input().model_dump(mode="json", exclude_unset=True)
        raw_input_changed["base"]["layers"][1]["thickness"] = "420 nm"
        sweep_input_changed = prepare_sweep(
            SweepInput.model_validate(raw_input_changed),
            sweep_b.defaults,
            sweep_b.resources,
            sources=sweep_b.sources,
        )
        jobs_input_changed = prepare_steady_sweep_jobs(
            sweep_input_changed,
            PROTOCOL,
            limits=LIMITS,
            implementation=implementation,
            current_repository_head=implementation.build_source_commit,
            engine=ExecutionEngine("solarlab"),
            scan_id="p06-05-input-diff",
            reuse_scope="execution_identity",
        )
        jobs_protocol_changed = prepare_steady_sweep_jobs(
            sweep_b,
            SteadyJVProtocol((0.0, 0.04, 0.1), 8, 1.0, None),
            limits=LIMITS,
            implementation=implementation,
            current_repository_head=implementation.build_source_commit,
            engine=ExecutionEngine("solarlab"),
            scan_id="p06-05-proto-diff",
            reuse_scope="execution_identity",
        )
        jobs_limits_changed = prepare_steady_sweep_jobs(
            sweep_b,
            PROTOCOL,
            limits=replace(LIMITS, wall_seconds=45.0),
            implementation=implementation,
            current_repository_head=implementation.build_source_commit,
            engine=ExecutionEngine("solarlab"),
            scan_id="p06-05-limits-diff",
            reuse_scope="execution_identity",
        )
        jobs_source_changed = prepare_steady_sweep_jobs(
            sweep_b,
            PROTOCOL,
            limits=LIMITS,
            implementation=replace(implementation, build_source_commit="1" * 40),
            current_repository_head=implementation.build_source_commit,
            engine=ExecutionEngine("solarlab"),
            scan_id="p06-05-source-diff",
            reuse_scope="execution_identity",
        )

        for index, variant in enumerate(
            (jobs_input_changed, jobs_protocol_changed, jobs_limits_changed, jobs_source_changed)
        ):
            assert variant.spec.points[0].run_id != shared_run_id
            forced_point = replace(variant.spec.points[0], run_id=shared_run_id)
            forced_jobs = SteadySweepJobs(
                ScanSpec(f"forced-mismatch-{index}", (forced_point,)),
                (variant.executions[0],),
            )
            with pytest.raises(StateConflict, match="differs from the complete stored execution request"):
                forced_jobs.submit(supervisor)

        # Also verify schema-digest mismatch refusal on WorkerRequest.
        altered_schema = SourceDocument("solarlab.steady-jv.worker-request.v2", b'{"schema":"v2"}')
        bad_req = replace(jobs_b.spec.points[0].spec.request, schema=altered_schema)
        bad_spec = replace(jobs_b.spec.points[0].spec, request=bad_req)
        with pytest.raises(ValueError, match="schema"):
            SteadySweepJobs(
                ScanSpec("forced-schema-mismatch", (replace(jobs_b.spec.points[0], spec=bad_spec),)),
                (jobs_b.executions[0],),
            ).submit(supervisor)

        assert len(store.attempts(shared_run_id)) == 1

    with RunStore(store_root, read_only=True) as reader:
        read_a = read_steady_sweep_jobs(reader, jobs_a.spec)
        read_b = read_steady_sweep_jobs(reader, jobs_b.spec)
        assert read_a[1]["coordinates"]["index"] == 1
        assert read_b[0]["coordinates"]["index"] == 0
        for name in ARRAY_NAMES:
            assert read_a[1]["result"]["arrays"][name].tobytes() == read_b[0]["result"]["arrays"][name].tobytes()
