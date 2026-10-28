"""Real new-core spawn, exact publication, and bounded failure/cancellation."""
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

import pytest

from solarlab.config.identity import BuildSource, ExecutionEngine
from solarlab.experiments.execution import ProcessSupervisor
from solarlab.experiments.jv.worker import (
    SteadyJVProtocol, prepare_steady_jv_job, read_steady_jv_job, steady_jv_worker_binding,
)
from solarlab.experiments.worker import ResourceLimits, WorkerBinding
from solarlab.io.run_store import RunStore
from solarlab.materials.source import SourceDocument

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def prepared():
    from examples.research_1d.steady import load_device
    return load_device()


@pytest.fixture(scope="module")
def implementation():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    content = subprocess.check_output(["git", "archive", commit, "--", "src/solarlab"], cwd=ROOT)
    sources = []
    with tarfile.open(fileobj=io.BytesIO(content)) as archive:
        for item in archive.getmembers():
            if item.isfile() and item.name.endswith(".py") and item.size:
                sources.append(SourceDocument(item.name.removeprefix("src/"), archive.extractfile(item).read()))
    return BuildSource(commit, tuple(sources))


@pytest.fixture(scope="module")
def job(prepared, implementation):
    return prepare_steady_jv_job(prepared, SteadyJVProtocol((0.0, 0.05, 0.1), 12, 1.0, None),
        implementation=implementation, current_repository_head=implementation.build_source_commit,
        engine=ExecutionEngine("solarlab"))


def limits():
    return ResourceLimits(45.0, 1536 * 1024**2, 96 * 1024**2, None, None)


def pool(store, work, bindings=None):
    return ProcessSupervisor(store, work, bindings or (steady_jv_worker_binding(),), max_active=1, max_pending=2,
        shared_rss_bytes=3 * 1024**3, shared_output_bytes=384 * 1024**2,
        soft_grace_seconds=0.2, terminate_grace_seconds=0.2, poll_seconds=0.02)


def terminal_events(store, run_id):
    return [event for event in store.events(run_id=run_id, limit=1000) if event["kind"] == "terminal"]


def _hold(request, context):
    """Non-numerical queue-occupancy fixture using the actual runtime stop."""
    context.consume_manifest(request, engine="solarlab")
    context.progress({"stage": "queue_fixture_ready"})
    while True:
        context.check_cancel()
        time.sleep(0.005)


def test_real_steady_jv_job_has_one_terminal_and_original_publication_bytes(tmp_path, job, prepared):
    root = Path(os.environ.get("SOLARLAB_STEADY_WORKER_STORE", str(tmp_path / "store")))
    work = root.with_name(root.name + "-work")
    root.parent.mkdir(parents=True, exist_ok=True)
    with RunStore(root) as store, pool(store, work) as supervisor:
        handle = job.submit(supervisor, "steady-case", limits())
        receipt = supervisor.wait(handle, 60).data
        assert receipt["process_started"] and receipt["process_joined"] and receipt["process_exit_observed"]
        assert receipt["store_state"] == "succeeded", receipt
        assert receipt["terminal_committed"] and receipt["exitcode"] == 0
        assert len(terminal_events(store, handle.run_id)) == 1
        assert len(receipt["terminal_artifacts"]) == 9  # Eight arrays plus the existing manifest artifact.
        assert receipt["resources"]["calls"] is None and receipt["resources"]["accepted_steps"] is None
        loaded = read_steady_jv_job(store, handle.run_id, attempt_id=handle.attempt_id)
        assert len(loaded["arrays"]) == 8
        assert loaded["arrays"]["potential_V"].shape == (3, 37)
        assert loaded["document"]["metrics"]["open_circuit_voltage_V"] is None
        assert loaded["document"]["metrics"]["limitation"] == "open_circuit_not_bracketed"
        assert all(point["accepted"] and "density_basin_stages" in point for point in loaded["document"]["solver_evidence"])
        assert not job.execution.manifest.can_execute and not loaded["worker_result"]["scientific_qualification"]
        assert loaded["worker_result"]["H_execution"] == job.execution.manifest.execution_sha256

        # A short direct reference verifies the public publisher retains its
        # existing output and exact numerical bytes across the transport change.
        from solarlab.device.compile_1d import compile_device_1d
        from solarlab.experiments.jv.steady import publish_steady_jv, read_steady_jv, run_steady_jv, steady_jv_payload
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            reference = run_steady_jv(compile_device_1d(prepared, intervals_per_layer=12), (0.0, 0.05, 0.1))
        row = store.enqueue("direct-reference", opaque_identity=store.get_run(handle.run_id)["request"]["identity"],
                            input_metadata={"scope": "real direct publisher parity reference"})
        token = store.claim("direct-reference", expected_attempt_id=row["current_attempt_id"],
                            expected_generation=row["generation"], owner_id="publisher-parity-test")
        published = publish_steady_jv(store, token, reference)
        store.complete(token, state="succeeded", result=published, artifacts=published["artifact_ids"])
        metadata, named = steady_jv_payload(reference)
        assert metadata == loaded["worker_result"]["steady_result"]
        assert {name: payload.sha256 for name, payload in named} == loaded["worker_result"]["array_payloads"]
        arrays = read_steady_jv(store, published)
        for name, value in loaded["arrays"].items():
            assert value.dtype == arrays[name].dtype and value.shape == arrays[name].shape
            assert value.tobytes() == arrays[name].tobytes()
        with pytest.raises(ValueError, match="attempt of this run"):
            read_steady_jv_job(store, handle.run_id, attempt_id=token.attempt_id)
        code = '''
import importlib.abc, pathlib, sys
class NoSolver(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == p or fullname.startswith(p + '.') for p in (
            'solarlab.solvers', 'solarlab.equations', 'solarlab.device.compile_1d', 'perovskite_sim')):
            raise AssertionError('readback imported numerical execution: ' + fullname)
sys.meta_path.insert(0, NoSolver())
from solarlab.io.run_store import RunStore
from solarlab.experiments.jv.worker import read_steady_jv_job
with RunStore(pathlib.Path(sys.argv[1]), read_only=True) as store:
    result = read_steady_jv_job(store, 'steady-case')
    assert len(result['arrays']) == 8 and result['arrays']['potential_V'].shape == (3, 37)
'''
        import sys
        readback = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(root)], cwd=tmp_path,
                                  capture_output=True, text=True, timeout=15)
        assert readback.returncode == 0, readback.stderr
        archive = os.environ.get("SOLARLAB_STEADY_WORKER_EVIDENCE")
        if archive:
            path = Path(archive)
            path.mkdir(parents=True, exist_ok=True)
            evidence = {"actual_store": str(root), "work": str(work), "receipt": receipt,
                "run_id": handle.run_id, "attempt_id": handle.attempt_id,
                "manifest_sha256": job.execution.manifest.sha256,
                "core_build_source_commit": job.execution.execution.implementation.build_source_commit,
                "transport_bytes": len(job.request.to_bytes()),
                "manifest_bytes": len(job.execution.manifest.canonical_json),
                "array_fingerprints": {name: {"shape": list(value.shape), "dtype": value.dtype.str,
                    "sha256": hashlib.sha256(value.tobytes()).hexdigest()} for name, value in loaded["arrays"].items()},
                "original_publisher_bitwise_match": True, "terminal_count": len(terminal_events(store, handle.run_id)),
                "fresh_solver_free_reader": {"exit_code": readback.returncode, "stderr": readback.stderr}}
            (path / "ActualWorker.json").write_text(json.dumps(evidence, indent=2) + "\n")


def test_invalid_request_is_one_failed_terminal_before_native_work(tmp_path, job):
    body = json.loads(job.request.request.content)
    body["engine"] = "legacy"
    request = replace(job.request, request=SourceDocument(job.request.request.id,
                      json.dumps(body, sort_keys=True, separators=(",", ":")).encode()))
    with RunStore(tmp_path / "store") as store, pool(store, tmp_path / "work") as supervisor:
        handle = supervisor.submit("invalid", "steady_jv", request, limits(), execution=job.execution)
        receipt = supervisor.wait(handle, 60).data
        assert receipt["store_state"] == "failed" and receipt["terminal_committed"]
        assert receipt["worker_result"]["exception_type"] == "ValueError"
        assert "request differs" in receipt["worker_result"]["detail"]
        assert not receipt["terminal_artifacts"] and len(terminal_events(store, "invalid")) == 1
        assert not any(e["payload"].get("ui_progress", {}).get("stage") == "steady_jv_native"
                       for e in store.events(run_id="invalid", limit=1000))


def test_unsupported_ions_and_bad_protocol_refuse_before_native_work(prepared, implementation, monkeypatch):
    from solarlab.config.resolve_device import resolve_device
    from solarlab.device.inputs import DeviceInput
    import solarlab.device.compile_1d as compiler
    import solarlab.solvers.nonlinear.steady_1d as solver

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid preparation reached a native operation")
    monkeypatch.setattr(compiler, "compile_device_1d", forbidden)
    monkeypatch.setattr(solver, "solve_steady_state", forbidden)
    value = prepared.to_input().editing_data()
    value["layers"][1]["parameters"]["D_ion"] = 1e-15
    unsupported = resolve_device(DeviceInput.model_validate(value), prepared.defaults, prepared.resources,
                                  sources=prepared.sources)
    with pytest.raises(ValueError, match="D_ion"):
        prepare_steady_jv_job(unsupported, SteadyJVProtocol((0., .05), 12, 1., None),
            implementation=implementation, current_repository_head=implementation.build_source_commit,
            engine=ExecutionEngine("solarlab"))
    for voltages in ((0.0,), (0.0, 0.0), (0.0, float("nan"))):
        with pytest.raises(ValueError, match="monotonic"):
            SteadyJVProtocol(voltages, 12, 1.0, None)


def test_queued_steady_cancellation_uses_existing_unique_terminal(tmp_path, job):
    blocking = WorkerBinding.from_function("queue_fixture", _hold,
        registry_sha256=job.request.registry.sha256, schema_sha256=job.request.schema.sha256)
    with RunStore(tmp_path / "store") as store, pool(store, tmp_path / "work",
            (steady_jv_worker_binding(), blocking)) as supervisor:
        first = supervisor.submit("queue-owner", "queue_fixture", job.request, limits(), execution=job.execution)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if any(e["payload"].get("ui_progress", {}).get("stage") == "queue_fixture_ready"
                   for e in store.events(run_id="queue-owner", limit=1000)):
                break
            time.sleep(.01)
        else:
            raise AssertionError("queue fixture did not become ready")
        queued = job.submit(supervisor, "cancelled-steady", limits())
        assert supervisor.cancel(queued)
        receipt = supervisor.wait(queued, 5).data
        assert not receipt["process_started"] and receipt["store_state"] == "cancelled"
        assert len(terminal_events(store, "cancelled-steady")) == 1
        assert not supervisor.cancel(queued)
        assert supervisor.cancel(first)
        assert supervisor.wait(first, 10).data["store_state"] == "cancelled"


def test_running_steady_cancellation_discards_late_output(tmp_path, prepared, implementation):
    job = prepare_steady_jv_job(prepared, SteadyJVProtocol(tuple(i * .025 for i in range(51)), 12, 1., None),
        implementation=implementation, current_repository_head=implementation.build_source_commit,
        engine=ExecutionEngine("solarlab"))
    with RunStore(tmp_path / "store") as store, pool(store, tmp_path / "work") as supervisor:
        handle = job.submit(supervisor, "cancel-native", limits())
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if any(e["payload"].get("ui_progress", {}).get("stage") == "steady_jv_native"
                   for e in store.events(run_id=handle.run_id, limit=1000)):
                break
            time.sleep(.005)
        else:
            raise AssertionError("steady child did not reach native work")
        assert supervisor.cancel(handle)
        receipt = supervisor.wait(handle, 10).data
        assert receipt["process_started"] and receipt["process_exit_observed"] and receipt["process_joined"]
        assert receipt["store_state"] == "cancelled" and receipt["terminal_committed"]
        assert not receipt["terminal_artifacts"] and len(terminal_events(store, handle.run_id)) == 1
        assert not supervisor.cancel(handle)
        archive = os.environ.get("SOLARLAB_STEADY_WORKER_EVIDENCE")
        if archive:
            path = Path(archive)
            path.mkdir(parents=True, exist_ok=True)
            (path / "NativeCancellation.json").write_text(json.dumps(receipt, indent=2) + "\n")
