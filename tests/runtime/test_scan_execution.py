"""P06-05 real spawn, bounded shared capacity, resume and exact result reuse."""

from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import struct
import sys
import time
from types import SimpleNamespace

import pytest

from solarlab.config.identity import ExecutionEngine, execution_manifest
from solarlab.experiments.execution import SCAN_BYTES, JobHandle, ProcessSupervisor, QueueFull, ScanPoint, ScanSpec
from solarlab.experiments.worker import ManifestBinding, ResourceLimits, RunSpec, WorkerBinding, WorkerRequest, observe_environment
from solarlab.io.artifacts import ArtifactIntegrityError
from solarlab.io.run_store import RunStore, StateConflict
from solarlab.materials.source import SourceDocument
import scan_workers

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/refactor"))
from test_execution_identity_preparation import device as device, execution as execution, identity_inputs  # noqa: E402

REGISTRY = SourceDocument("scan-infrastructure-registry.v1", b'{"scope":"non-scientific runtime fixture"}')
SCHEMA = SourceDocument("scan-infrastructure-schema.v1", b'{"point":"integer","action":"fixture enum"}')
THREADS = (("OMP_NUM_THREADS", "1"), ("OPENBLAS_NUM_THREADS", "1"))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@pytest.fixture
def make_spec(device, execution):
    inputs = identity_inputs(device)
    runtime = replace(observe_environment(("pydantic",), tuple(k for k, _ in THREADS)),
                      settings=tuple((k, json.dumps(v)) for k, v in THREADS))

    def make(point=0, *, action="normal", **caps):
        payload = SourceDocument("runtime-fixture-protocol", canonical({"point": point, "action": action}))
        initial = SourceDocument("runtime-fixture-initial-state", canonical({"qualification": "not_a_DAE_seed",
            "words": (struct.pack("<2d", point + .25, -0.0) + bytes.fromhex("010000000000f87f")).hex()}))
        description = replace(execution, protocol=payload, initial_state=initial, environment=runtime)
        engine = ExecutionEngine("solarlab")
        manifest = execution_manifest(inputs, description, engine=engine)
        binding = ManifestBinding(manifest, inputs, description, engine, manifest.sha256, THREADS)
        identity = manifest.export()["document"]["execution_identity"]
        request = WorkerRequest(payload, REGISTRY, SCHEMA, SourceDocument("execution", canonical(identity)),
                                identity["payload"]["H_input"], identity["payload"]["H_physics"])
        limits = replace(ResourceLimits(5., 512 * 1024**2, 4 * 1024**2, 10, 10), **caps)
        return RunSpec.from_binding("point", request, limits, binding), binding
    return make


def bindings():
    return (WorkerBinding.from_function("point", scan_workers.point,
        registry_sha256=REGISTRY.sha256, schema_sha256=SCHEMA.sha256),)


def pool(store, root, **changes):
    values = dict(max_active=1, max_pending=1, shared_rss_bytes=2 * 1024**3,
        shared_output_bytes=64 * 1024**2, soft_grace_seconds=.1, terminate_grace_seconds=.1, poll_seconds=.01)
    values.update(changes)
    return ProcessSupervisor(store, root, bindings(), **values)


def declaration(name, records, *, order=None, run_prefix=None):
    order = list(range(len(records))) if order is None else order
    prefix = name if run_prefix is None else run_prefix
    points = tuple(ScanPoint(f"p{i}", f"{prefix}-{i}", canonical({"index": i, "zero": -0.0, "optional": None}).decode(), records[i][0]) for i in order)
    return ScanSpec(name, points), tuple(records[i][1] for i in order)


def ready(store, run_id):
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        for event in store.events(run_id=run_id):
            if event["payload"].get("ui_progress", {}).get("stage") == "ready":
                return event["attempt_id"]
        time.sleep(.005)
    raise AssertionError("bounded fixture never reached its real ready boundary")


def release(store, work, attempt):
    (work / (store.store_id + "-" + attempt) / "release").touch()


def artifact(store, run_id, name="words"):
    attempt = store.attempts(run_id)[-1]
    records = [store.artifact(item["artifact_id"]) for item in attempt["terminal"]["artifacts"]]
    item = next(item for item in records if item["name"] == name)
    return store.read_artifact(item["artifact_id"]), item


def terminals(store, run_id):
    return [e for e in store.events(run_id=run_id, limit=1000) if e["kind"] == "terminal"]


def test_runspec_roundtrip_binds_actual_sources_limits_and_requires_trusted_reconstruction(make_spec):
    spec, binding = make_spec()
    assert RunSpec.from_bytes(spec.to_bytes()) == spec
    spec.validate(binding)
    assert spec.initial_state_sha256 == binding.execution.initial_state.sha256
    assert spec.to_document()["request"]["H_execution"] == binding.manifest.execution_sha256
    with pytest.raises(FrozenInstanceError):
        spec.worker_id = "another"
    for changes in ({"manifest_sha256": "0" * 64}, {"initial_state_sha256": "0" * 64}, {"initial_state_id": "different-source"}):
        with pytest.raises(ValueError, match="actual binding"):
            replace(spec, **changes).validate(binding)
    with pytest.raises(ValueError, match="independently"):
        spec.validate(None)
    raw = spec.to_document()
    raw["module"] = "os"
    with pytest.raises(ValueError, match="fields"):
        RunSpec.from_bytes(canonical(raw))
    for bad in (float("inf"), -1, True):
        with pytest.raises(ValueError):
            replace(spec.limits, wall_seconds=bad)
    scan, _ = declaration("serialize", [(spec, binding)])
    assert ScanSpec.from_bytes(scan.to_bytes()) == scan
    with pytest.raises(ValueError, match="unique"):
        ScanSpec("duplicate", (scan.points[0], scan.points[0]))
    with pytest.raises(ValueError, match="canonical"):
        ScanSpec.from_bytes(scan.to_bytes() + b"\n")


def test_server_and_scan_share_one_bounded_pool_without_a_coordinator_slot(tmp_path, make_spec):
    blocker, execution = make_spec(9, action="release")
    records = [make_spec(i) for i in range(3)]
    spec, executions = declaration("shared", records)
    work = tmp_path / "work"
    with RunStore(tmp_path / "db") as store, pool(store, work, max_pending=0) as supervisor:
        server = supervisor.submit("server", blocker.worker_id, blocker.request, blocker.limits, execution=execution)
        ready(store, "server")
        scan = supervisor.scan(spec, executions=executions, max_in_flight=1)
        assert scan.advance()["points"][0]["state"] == "declared"
        assert supervisor.accounting()["active"] == 1 and supervisor.accounting()["pending"] == 0
        with pytest.raises(KeyError):
            store.get_run("shared-0")
        with pytest.raises(QueueFull):
            supervisor.submit("full", records[0][0].worker_id, records[0][0].request, records[0][0].limits, execution=records[0][1])
        with pytest.raises(TimeoutError, match="not execution cancellation"):
            scan.wait(.01)
        release(store, work, server.attempt_id)
        assert supervisor.wait(server, 6).data["store_state"] == "succeeded"
        result = scan.wait(12)
        assert result["state"] == "succeeded" and not result["coordinator_uses_process_slot"]
        assert len(store.sweep_points("shared")) == 3
        assert all(p["reason"] == "completed" and p["initial_state"]["sha256"] == records[i][0].initial_state_sha256
                   for i, p in enumerate(result["points"]))


def test_scan_transport_matches_the_metadata_limit_without_truncation(make_spec):
    spec, _ = make_spec(0)
    points = [ScanPoint(f"p{i}", f"r{i}", canonical({"pad": ""}).decode(), spec) for i in range(64)]

    def document():
        return canonical({"schema": "solarlab.scan-spec.v1", "scan_id": "size-limit",
                          "points": [p.to_document() for p in points]})

    limit = 1024**2
    remaining = limit - len(document())
    assert 0 < remaining <= len(points) * 15000
    for index, point in enumerate(points):
        size = min(15000, remaining)
        points[index] = replace(point, coordinates_json=canonical({"pad": "x" * size}).decode())
        remaining -= size
    exact = document()
    assert remaining == 0 and len(exact) == limit
    scan = ScanSpec("size-limit", tuple(points))
    assert scan.to_bytes() == exact and ScanSpec.from_bytes(exact) == scan
    points[-1] = replace(points[-1], coordinates_json=canonical({"pad": "x"}).decode())
    assert len(document()) == limit + 1
    with pytest.raises(ValueError, match="1 MiB|bounded"):
        ScanSpec("size-limit", tuple(points))
    with pytest.raises(ValueError, match="1 MiB|bounded"):
        ScanSpec.from_bytes(document())
    assert SCAN_BYTES == limit  # The published bound must match both actual codec entries.


def test_stuck_fixture_ignores_sigterm_before_admission_or_progress(monkeypatch):
    installed = []
    monkeypatch.setattr(scan_workers.signal, "signal", lambda *args: installed.append(args))

    def stop_at_admission(request, *, engine):
        assert installed == [(scan_workers.signal.SIGTERM, scan_workers.signal.SIG_IGN)]
        raise RuntimeError("admission boundary reached with SIGTERM ignored")

    request = SimpleNamespace(request=SimpleNamespace(content=b'{"action":"stuck"}'))
    with pytest.raises(RuntimeError, match="admission boundary reached"):
        scan_workers.point(request, SimpleNamespace(consume_manifest=stop_at_admission))


@pytest.mark.parametrize("action,reason,caps", [
    ("exception", "worker_exception", {}), ("crash", "child_crash", {}),
    # CI 40923590700 needed up to 2.53 s for a two-point failure/normal pair.
    # The old 0.8 s cap could kill Python/NumPy startup before point() armed
    # SIG_IGN. Allow >2x that entire-pair bound for this one child, while
    # staying below the fixture's 8 s escape deadline. Production defaults
    # and the three-signal ladder are unchanged.
    ("stuck", "wall_limit", {"wall_seconds": 6.}),
    ("counter", "accepted_steps_limit", {"accepted_steps": 2}),
    ("nested_scan", "worker_exception", {}),
])
def test_point_failure_budget_or_nested_scan_does_not_pollute_following_point(tmp_path, make_spec, action, reason, caps, request):
    records = [make_spec(0, action=action, **caps), make_spec(1)]
    spec, executions = declaration("isolation", records)
    work = tmp_path / "work"
    with RunStore(tmp_path / "db") as store, pool(store, work) as supervisor:
        result = supervisor.scan(spec, executions=executions, max_in_flight=1).wait(16 if action == "stuck" else 12)
        first, second = result["points"]
        assert result["state"] == "failed" and first["reason"] == reason
        assert second["state"] == "succeeded" and second["result"]["worker_result"]["fresh_process_state"]
        assert all(len(terminals(store, p["run_id"])) == 1 for p in result["points"])
        assert artifact(store, "isolation-1")[0] == bytes.fromhex(json.loads(executions[1].execution.initial_state.content)["words"])
        receipt = supervisor.wait(JobHandle(store.store_id, first["run_id"], first["attempt_id"]), 1).data
        assert receipt["process_exit_observed"] and receipt["process_joined"]
        if action == "stuck":
            progress = [event["payload"]["ui_progress"] for event in store.events(run_id=first["run_id"])
                        if event["payload"].get("ui_progress", {}).get("stage") == "ready"]
            assert progress and progress[-1]["signals_ignored"], "wall budget expired before the stuck fixture was ready"
            launch = json.loads((work / (store.store_id + "-" + first["attempt_id"]) / "launch.json").read_bytes())
            startup = progress[-1]["ready_monotonic"] - launch["started_monotonic"]
            assert 0 <= startup < caps["wall_seconds"]
            assert receipt["signals"] == ["soft_stop", "terminate", "kill"]
            request.node.user_properties.append(("stuck_startup", {"seconds": startup,
                "wall_cap_seconds": caps["wall_seconds"], "signals": receipt["signals"],
                "process_exit_observed": receipt["process_exit_observed"], "process_joined": receipt["process_joined"]}))
        if action == "nested_scan":
            assert "nested supervisors or scans" in first["result"]["worker_result"]["detail"]


def test_cancel_running_queued_and_unsubmitted_points_and_reject_late_success(tmp_path, make_spec):
    records = [make_spec(0, action="late"), make_spec(1), make_spec(2)]
    spec, executions = declaration("cancel", records)
    work = tmp_path / "work"
    with RunStore(tmp_path / "db") as store, pool(store, work, soft_grace_seconds=.6) as supervisor:
        scan = supervisor.scan(spec, executions=executions, max_in_flight=2)
        scan.advance()
        current = ready(store, "cancel-0")
        assert store.get_run("cancel-1")["state"] == "queued"
        with pytest.raises(KeyError):
            store.get_run("cancel-2")
        assert scan.cancel(reason="explicit scan cancellation")
        assert store.get_run("cancel-1")["state"] == store.get_run("cancel-2")["state"] == "cancelled"
        assert supervisor.accounting()["active"] == 1
        release(store, work, current)
        result = scan.wait(6)
        assert result["state"] == "cancelled" and all(p["state"] == "cancelled" for p in result["points"])
        assert all(p["reason"] == "explicit scan cancellation" for p in result["points"])
        receipt = supervisor.wait(JobHandle(store.store_id, "cancel-0", current), 1).data
        assert receipt["worker_result"]["discarded_terminal_state"] == "succeeded"
        assert not scan.cancel(reason="same completed scan")
        assert all(len(terminals(store, p["run_id"])) == 1 for p in result["points"])
        with pytest.raises(StateConflict, match="stays cancelled"):
            scan.retry("p0", expected_attempt_id=current)
        following, binding = make_spec(9)
        h = supervisor.submit("after-cancel", "point", following.request, following.limits, execution=binding)
        assert supervisor.wait(h, 6).data["store_state"] == "succeeded"
    with RunStore(tmp_path / "db") as store, pool(store, work) as supervisor:
        count = len(store.events(limit=1000))
        restored = supervisor.scan(ScanSpec.from_bytes(spec.to_bytes()), executions=executions, max_in_flight=1).wait(1)
        assert restored["state"] == "cancelled" and len(store.events(limit=1000)) == count


def test_serial_parallel_random_order_and_restart_preserve_exact_payload_and_evidence(tmp_path, make_spec):
    records = [make_spec(i) for i in range(3)]
    order = [0, 1, 2]
    random.Random(1847).shuffle(order)
    outputs = []
    with RunStore(tmp_path / "db") as store, pool(store, tmp_path / "work", max_active=2) as supervisor:
        for name, positions, width in (("serial", None, 1), ("parallel", None, 2), ("random", order, 2)):
            spec, executions = declaration(name, records, order=positions)
            result = supervisor.scan(spec, executions=executions, max_in_flight=width).wait(15)
            assert result["state"] == "succeeded" and not any(p["reused"] for p in result["points"])
            outputs.append({p["point_id"]: (artifact(store, p["run_id"])[0], p["result"]["worker_result"]) for p in result["points"]})
        assert outputs[0] == outputs[1] == outputs[2]
        events = store.events(limit=1000)
    with RunStore(tmp_path / "db") as store, pool(store, tmp_path / "work") as supervisor:
        spec, executions = declaration("serial", records)
        saved = next((tmp_path / "work").glob("scan-*-serial.json")).read_bytes()
        restored = supervisor.scan(ScanSpec.from_bytes(saved), executions=executions, max_in_flight=1).wait(2)
        assert restored["state"] == "succeeded" and all(p["reused"] for p in restored["points"])
        assert store.events(limit=1000) == events and supervisor.accounting()["active"] == 0
        assert not any(p["result"]["worker_result"]["scientific_qualification"] for p in restored["points"])


@pytest.mark.parametrize("change", ["initial", "protocol", "budget", "worker", "corrupt_artifact"])
def test_completed_reuse_refuses_incomplete_identity_or_corrupt_artifact_without_replay(tmp_path, make_spec, change):
    records = [make_spec(0)]
    spec, executions = declaration("original", records)
    with RunStore(tmp_path / "db") as store, pool(store, tmp_path / "work") as supervisor:
        assert supervisor.scan(spec, executions=executions, max_in_flight=1).wait(6)["state"] == "succeeded"
        events = store.events(limit=1000)
        original, binding = records[0]
        replacement = original
        if change == "initial":
            replacement = replace(original, initial_state_sha256="0" * 64)
        elif change == "protocol":
            replacement, binding = make_spec(1)
        elif change == "budget":
            replacement = replace(original, limits=replace(original.limits, wall_seconds=6.))
        elif change == "worker":
            replacement = replace(original, worker_id="not_allowlisted")
        else:
            _, item = artifact(store, "original-0")
            (store.root / item["relative_path"]).write_bytes(b"corrupt")
        changed, bindings_ = declaration("reuse", [(replacement, binding)], run_prefix="original")
        with pytest.raises((ValueError, StateConflict, ArtifactIntegrityError)):
            supervisor.scan(changed, executions=bindings_, max_in_flight=1).wait(3)
        assert store.events(limit=1000) == events
        assert store.get_run("original-0")["state"] == "succeeded"  # Historical success is never rewritten.


@pytest.mark.parametrize("damage", ["missing_audit", "wall_overrun"])
def test_completed_reuse_requires_positive_final_resource_audit(tmp_path, make_spec, damage):
    spec, executions = declaration("audit", [make_spec(0)])
    work = tmp_path / "work"
    with RunStore(tmp_path / "db") as store, pool(store, work) as supervisor:
        result = supervisor.scan(spec, executions=executions, max_in_flight=1).wait(6)
        path = work / (store.store_id + "-" + result["points"][0]["attempt_id"]) / "finalization.json"
        events = store.events(limit=1000)
    # Explicitly damaged local receipt fixture, never a claimed new observation.
    if damage == "missing_audit":
        path.unlink()
    else:
        audit = json.loads(path.read_bytes())
        audit["owned_wall_after_audit_seconds"] = 6.
        path.write_bytes(canonical(audit))
    with RunStore(tmp_path / "db") as store, pool(store, work) as supervisor:
        with pytest.raises(StateConflict, match="resource audit|in-budget"):
            supervisor.scan(spec, executions=executions, max_in_flight=1).wait(2)
        assert store.get_run("audit-0")["state"] == "succeeded"
        assert store.events(limit=1000) == events


@pytest.mark.parametrize("phase", ["normal", "before_record", "after_record", "wrong_token"])
def test_restart_reuse_checks_both_finalization_overrun_records(tmp_path, make_spec, phase):
    spec, executions = declaration("finalization", [make_spec(0)])
    work = tmp_path / "work"
    with RunStore(tmp_path / "db") as store, pool(store, work) as supervisor:
        result = supervisor.scan(spec, executions=executions, max_in_flight=1).wait(6)
        point = result["points"][0]
        handle = JobHandle(store.store_id, point["run_id"], point["attempt_id"])
        receipt = supervisor.wait(handle, 1).data
        assert result["state"] == "succeeded" and receipt["process_joined"] is True
        assert receipt["finalization"]["finalization_overrun"] is False
        directory = work / (store.store_id + "-" + handle.attempt_id)
        events, attempts = store.events(limit=1000), store.attempts(handle.run_id)
        words = artifact(store, handle.run_id)[0]

    # These are explicit persisted-receipt fixtures following a real child exit;
    # they do not claim that this short worker actually exhausted disk space.
    path = directory / "finalization.json"
    cap = 64 * 1024**2
    if phase == "before_record":
        audit = json.loads(path.read_bytes())
        audit.update(finalization_overrun=True, shared_output_peak_bytes=cap + 1)
        path.write_bytes(canonical(audit))
    elif phase in {"after_record", "wrong_token"}:
        path = directory / "finalization_overrun.json"
        token = dict(receipt["token"])
        if phase == "wrong_token":
            token["generation"] += 1
        path.write_bytes(canonical({"token": token, "observed_bytes": cap + 1, "limit_bytes": cap,
            "measurement_phase": "after_finalization_record_write", "store_terminal_is_historical": True}))
    persisted = path.read_bytes()
    with RunStore(tmp_path / "db") as store:
        if phase == "wrong_token":
            with pytest.raises(ValueError, match="finalization overrun belongs to a different attempt"):
                pool(store, work)
        else:
            with pool(store, work) as supervisor:
                loaded = supervisor.wait(handle, 1).data
                if phase == "after_record":
                    assert loaded["finalization"]["finalization_overrun"] is False
                    assert loaded["finalization_overrun"]["token"] == receipt["token"]
                    assert supervisor.accounting()["retained_output_peak_bytes"] >= cap + 1
                if phase == "normal":
                    restored = supervisor.scan(spec, executions=executions, max_in_flight=1).wait(2)
                    assert restored["state"] == "succeeded" and restored["points"][0]["reused"] is True
                else:
                    with pytest.raises(StateConflict, match="resource receipt|resource audit"):
                        supervisor.scan(spec, executions=executions, max_in_flight=1).wait(2)
                assert supervisor.accounting()["active"] == supervisor.accounting()["pending"] == 0
        assert store.get_run(handle.run_id)["state"] == "succeeded"
        assert store.events(limit=1000) == events and store.attempts(handle.run_id) == attempts
        assert artifact(store, handle.run_id)[0] == words and path.read_bytes() == persisted


def test_association_failure_never_starts_the_affected_child(tmp_path, make_spec, monkeypatch):
    spec, executions = declaration("association", [make_spec(0)])
    with RunStore(tmp_path / "db") as store, pool(store, tmp_path / "work") as supervisor:
        scan = supervisor.scan(spec, executions=executions, max_in_flight=1)
        original = store.bind_sweep_point
        def conflicting(*args, **kwargs):
            raise StateConflict("injected conflicting point association")
        monkeypatch.setattr(store, "bind_sweep_point", conflicting)
        with pytest.raises(StateConflict, match="conflicting"):
            scan.advance()
        row = store.get_run("association-0")
        receipt = supervisor.wait(JobHandle(store.store_id, row["run_id"], row["current_attempt_id"]), 1).data
        assert row["state"] == "cancelled" and receipt["process_started"] is False
        assert receipt["reason"] == "scan_association_failed"
        monkeypatch.setattr(store, "bind_sweep_point", original)
        following, bindings_ = declaration("independent", [make_spec(1)])
        assert supervisor.scan(following, executions=bindings_, max_in_flight=1).wait(6)["state"] == "succeeded"


def test_real_parent_crash_preserves_completed_and_queued_points_and_explicit_new_attempt(tmp_path, make_spec):
    records = [make_spec(0), make_spec(1, action="late"), make_spec(2)]
    spec, executions = declaration("restart", records)
    database, work, checkpoint = tmp_path / "db", tmp_path / "work", tmp_path / "checkpoint.json"
    process = multiprocessing.get_context("spawn").Process(target=scan_workers.crash_coordinator,
        args=(str(database), str(work), spec.to_bytes(), bindings(), executions, str(checkpoint)))
    process.start()
    process.join(10)
    assert process.exitcode == 72
    old = json.loads(checkpoint.read_bytes())
    old_ids = [p["attempt_id"] for p in old["points"]]
    with RunStore(database) as store, pool(store, work) as supervisor:
        scan = supervisor.scan(ScanSpec.from_bytes(spec.to_bytes()), executions=executions, max_in_flight=2)
        with pytest.raises(StateConflict, match="reconciliation"):
            scan.advance()
        recovery = supervisor.reconcile_interrupted()
        assert recovery["invalidated_owned_attempts"] == [old_ids[1]]
        assert recovery["unowned_processes_signalled"] is False
        release(store, work, old_ids[1])
        deadline = time.monotonic() + 6
        while supervisor.accounting()["unresolved_owned_slots"] and time.monotonic() < deadline:
            time.sleep(.025)
            supervisor.reconcile_interrupted()
        assert supervisor.accounting()["unresolved_owned_slots"] == 0
        result = scan.wait(8)
        assert [p["state"] for p in result["points"]] == ["succeeded", "failed", "succeeded"]
        assert result["points"][0]["reused"] and result["points"][1]["reason"] == "interrupted"
        assert [p["attempt_id"] for p in result["points"]] == old_ids
        old_terminal = terminals(store, "restart-1")
        next_attempt = scan.retry("p1", expected_attempt_id=old_ids[1])
        assert next_attempt.attempt_id != old_ids[1]
        scan.advance()
        ready(store, "restart-1")
        # Match the new ready attempt, not the old durable ready event.
        directory = work / (store.store_id + "-" + next_attempt.attempt_id)
        deadline = time.monotonic() + 4
        while not (directory / "started.json").exists() and time.monotonic() < deadline:
            time.sleep(.005)
        release(store, work, next_attempt.attempt_id)
        completed = scan.wait(8)
        assert completed["state"] == "succeeded"
        assert [a["state"] for a in store.attempts("restart-1")] == ["failed", "succeeded"]
        assert terminals(store, "restart-1")[:1] == old_terminal
        assert artifact(store, "restart-1")[0] == bytes.fromhex(json.loads(executions[1].execution.initial_state.content)["words"])


def test_real_legacy_facade_scan_matches_single_runs_serial_and_random_parallel(tmp_path):
    from legacy_fixtures import make_job
    from solarlab_server.legacy_engine import legacy_worker_binding

    jobs = [make_job(arguments={"V_max": voltage})[0] for voltage in (.02, .03)]
    limits = ResourceLimits(60., 1024**3, 64 * 1024**2, None, None)
    records = [(RunSpec.from_binding("legacy", job.request, limits, job.execution), job.execution) for job in jobs]
    evidence = []
    with RunStore(tmp_path / "db") as store, ProcessSupervisor(store, tmp_path / "work", (legacy_worker_binding(),),
        max_active=2, max_pending=1, shared_rss_bytes=4 * 1024**3, shared_output_bytes=512 * 1024**2) as supervisor:
        originals = []
        for i, job in enumerate(jobs):
            h = job.submit(supervisor, f"single-{i}", limits)
            returned = supervisor.wait(h, 65).data
            assert returned["store_state"] == "succeeded", returned
            originals.append(artifact(store, h.run_id, "legacy_result")[0])
        for name, order, width in (("legacy-serial", [0, 1], 1), ("legacy-random", [1, 0], 2)):
            spec, executions = declaration(name, records, order=order)
            result = supervisor.scan(spec, executions=executions, max_in_flight=width).wait(125)
            assert result["state"] == "succeeded", result
            for point in result["points"]:
                i = int(point["point_id"][1:])
                content, _ = artifact(store, point["run_id"], "legacy_result")
                assert content == originals[i]
                assert point["initial_state"]["sha256"] == jobs[i].execution.execution.initial_state.sha256
                assert point["result"]["worker_result"]["scientific_qualification"] is False
                receipt = supervisor.wait(JobHandle(store.store_id, point["run_id"], point["attempt_id"]), 1).data
                evidence.append({"scan": name, "point": point["point_id"], "bitwise_equal": True,
                    "payload_sha256": hashlib.sha256(content).hexdigest(), "H_execution": point["H_execution"],
                    "initial_state": point["initial_state"], "process": point["result"]["process"],
                    "resources": receipt["resources"], "scientific_qualification": False})
    path = os.environ.get("SOLARLAB_SCAN_EVIDENCE")
    if path:
        Path(path).write_bytes(canonical({"real_legacy_points": evidence}) + b"\n")
