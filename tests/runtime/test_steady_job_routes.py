"""Installed steady jobs through the existing local ASGI/owner/store paths."""
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from fastapi.testclient import TestClient
import numpy as np
import pytest

from examples.research_1d.service import ROOT, committed_implementation
from examples.research_1d.steady import load_device
from solarlab.experiments.execution import JobHandle, ProcessSupervisor
from solarlab.experiments.jv.worker import read_steady_jv_job
from solarlab.experiments.worker import ResourceLimits, WorkerBinding
from solarlab.io.run_store import RunStore
from solarlab.materials.source import SourceDocument
from solarlab_server.app import create_app
from solarlab_server.job_service import JobService, JobSubmission, canonical
from solarlab_server.steady_preparation import SteadyPreparationContext, steady_execution_catalog


@pytest.fixture(scope="module")
def context():
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    commit = os.environ.get("SOLARLAB_STEADY_SERVICE_COMMIT", head)
    return SteadyPreparationContext((("research_1d", load_device()),), committed_implementation(commit), head)


def limits():
    return ResourceLimits(60.0, 1536 * 1024**2, 96 * 1024**2, None, None)


@pytest.fixture(scope="module")
def catalog(context):
    return steady_execution_catalog(context, ceiling=limits())


def submission(run_id="http-steady"):
    return {"schema_version": "solarlab.job-submission.v1", "run_id": run_id,
        "experiment_id": "steady_jv", "version": 1, "engine": "solarlab",
        "input": {"device_id": "research_1d", "voltage_V": [0.0, 0.05, 0.1],
            "intervals_per_layer": 12, "generation_scale": 1.0, "incident_power_W_m2": None},
        "limits": {"wall_seconds": 60.0, "rss_bytes": 1536 * 1024**2, "output_bytes": 96 * 1024**2,
            "accepted_steps": None, "calls": None}}


def headers(catalog):
    return {"X-Solarlab-Registry": catalog.document.sha256, "X-Solarlab-Job-Schema": catalog.schema.sha256}


def pool(store, work, catalog, *, bindings=None):
    return ProcessSupervisor(store, work, bindings or (catalog.worker_binding("solarlab"),),
        max_active=1, max_pending=2, shared_rss_bytes=3 * 1024**3, shared_output_bytes=384 * 1024**2,
        soft_grace_seconds=0.1, terminate_grace_seconds=0.1, poll_seconds=0.01)


def terminal_events(store, run_id):
    return [event for event in store.events(run_id=run_id, limit=1000) if event["kind"] == "terminal"]


def test_real_steady_post_reads_eight_arrays_manifest_and_one_terminal(tmp_path, catalog, context):
    root = Path(os.environ.get("SOLARLAB_STEADY_SERVICE_STORE", str(tmp_path / "store")))
    root.parent.mkdir(parents=True, exist_ok=True)
    with RunStore(root) as store, pool(store, root.with_name(root.name + "-work"), catalog) as supervisor:
        with TestClient(create_app(root, job_service=JobService(catalog, supervisor))) as http:
            advertised = http.get("/experiments")
            assert advertised.status_code == 200 and advertised.headers["x-solarlab-registry"] == catalog.document.sha256
            entry = advertised.json()["bindings"][0]
            assert entry["engine"] == "solarlab" and entry["scientific_qualification"] is False
            assert entry["scope"] == "static_classical_ion_free_1d_dc"
            assert entry["preparation_context"] == context.export()
            assert entry["legacy_kinds"] == [] and entry["synchronous_paths"] == []
            assert advertised.json()["registry"]["registered_executors"] == []
            response = http.post("/jobs", json=submission(), headers=headers(catalog))
            assert response.status_code == 202, response.text
            accepted = response.json()
            duplicate = http.post("/jobs", json=submission(), headers=headers(catalog))
            assert duplicate.status_code == 202 and duplicate.json()["attempt_id"] == accepted["attempt_id"]
            handle = JobHandle(store.store_id, accepted["run_id"], accepted["attempt_id"])
            receipt = supervisor.wait(handle, 75).data
            assert receipt["store_state"] == "succeeded", receipt
            assert receipt["process_started"] and receipt["process_joined"] and receipt["process_exit_observed"]
            assert len(terminal_events(store, handle.run_id)) == 1
            result_response = http.get(accepted["result_url"])
            assert result_response.status_code == 200
            result = result_response.json()
            assert result["state"] == "succeeded" and len(result["artifacts"]) == 9
            loaded = read_steady_jv_job(store, handle.run_id, attempt_id=handle.attempt_id)
            assert len(loaded["arrays"]) == 8 and loaded["arrays"]["potential_V"].shape == (3, 37)
            assert loaded["arrays"]["voltage_V"].tolist() == [0.0, 0.05, 0.1]
            worker_result = result["result"]["worker_result"]
            assert worker_result["registered_device_id"] == "research_1d"
            assert worker_result["execution_catalog_sha256"] == catalog.document.sha256
            assert worker_result["scientific_qualification"] is False
            assert loaded["document"]["metrics"]["open_circuit_voltage_V"] is None
            assert loaded["document"]["metrics"]["limitation"] == "open_circuit_not_bracketed"
            assert all(point["accepted"] for point in loaded["document"]["solver_evidence"])
            fingerprints = {}
            for item in result["artifacts"]:
                response = http.get(item["content_url"])
                assert response.status_code == 200
                assert hashlib.sha256(response.content).hexdigest() == item["sha256"]
                assert response.content == store.read_artifact(item["artifact_id"])
                if item["name"] == "execution_manifest":
                    assert item["sha256"] == worker_result["manifest_sha256"]
                else:
                    array = np.load(io.BytesIO(response.content), allow_pickle=False)
                    assert array.tobytes() == loaded["arrays"][item["name"]].tobytes()
                    fingerprints[item["name"]] = {"shape": list(array.shape), "dtype": array.dtype.str,
                        "payload_sha256": item["sha256"], "word_sha256": hashlib.sha256(array.tobytes()).hexdigest()}
            stream = http.get(accepted["events_url"])
            events = [json.loads(line[6:]) for line in stream.text.splitlines() if line.startswith("data: ")]
            assert events[-1]["kind"] == "terminal" and events[-1]["payload"]["state"] == "succeeded"
            assert sum(event["kind"] == "terminal" for event in events) == 1
            assert http.get(accepted["events_url"], headers={"Last-Event-ID": str(events[-1]["sequence"])}).status_code == 204
            archive = os.environ.get("SOLARLAB_STEADY_SERVICE_EVIDENCE")
            if archive:
                path = Path(archive)
                path.mkdir(parents=True, exist_ok=True)
                (path / "ActualService.json").write_text(json.dumps({"accepted": accepted,
                    "http_status": 202, "catalog": advertised.json(), "store": str(root),
                    "source_pin": context.implementation.build_source_commit,
                    "current_repository_head": context.current_repository_head,
                    "receipt": receipt, "arrays": fingerprints, "terminal_count": 1,
                    "readback": "HTTP NPY bytes independently decoded; existing steady reader also verified",
                    "scientific_qualification": False}, indent=2) + "\n")
        with TestClient(create_app(root)) as reader:
            assert reader.get(accepted["result_url"]).json() == result
            assert reader.post("/jobs", json=submission(), headers=headers(catalog)).status_code == 503


def test_steady_refuses_headers_schema_engine_capability_and_budgets_before_enqueue(tmp_path, catalog, context, monkeypatch):
    import solarlab.device.compile_1d as compiler
    import solarlab.solvers.nonlinear.steady_1d as solver

    def forbidden(*args, **kwargs):
        raise AssertionError("refused input reached a compiler or solver")

    monkeypatch.setattr(compiler, "compile_device_1d", forbidden)
    monkeypatch.setattr(solver, "solve_steady_state", forbidden)
    with RunStore(tmp_path / "store") as store, pool(store, tmp_path / "work", catalog) as supervisor:
        with TestClient(create_app(store.root, job_service=JobService(catalog, supervisor))) as http:
            assert http.post("/jobs", json=submission()).status_code == 428
            for header in headers(catalog):
                assert http.post("/jobs", json=submission(), headers={**headers(catalog), header: "0" * 64}).status_code == 409
            for changes in ({"engine": "legacy"}, {"engine": "invented"}, {"experiment_id": "impedance"},
                            {"version": 2}, {"schema_version": "solarlab.job-submission.v2"},
                            {"source_commit": context.implementation.build_source_commit}):
                response = http.post("/jobs", json={**submission(), **changes}, headers=headers(catalog))
                assert response.status_code == 422, response.text
            for changes in ({"device_id": "/tmp/device.json"}, {"device_id": "unregistered"},
                            {"voltage_V": [0.0, 0.0]}, {"voltage_V": [0.0]},
                            {"voltage_V": [float(i) for i in range(102)]},
                            {"intervals_per_layer": 257}, {"intervals_per_layer": True},
                            {"generation_scale": -1.0}, {"incident_power_W_m2": 0.0},
                            {"source_bytes": "arbitrary"}, {"config_path": "/tmp/device.json"},
                            {"python_function": "os.system"}, {"dimension": 2}, {"mobile_ions": True}):
                request = submission()
                request["input"].update(changes)
                response = http.post("/jobs", json=request, headers=headers(catalog))
                assert response.status_code == 422, response.text
            request = submission()
            del request["input"]["generation_scale"]
            assert http.post("/jobs", json=request, headers=headers(catalog)).status_code == 422
            for key, value in (("wall_seconds", 61.0), ("rss_bytes", 2 * 1024**3),
                               ("output_bytes", 128 * 1024**2), ("output_bytes", 0),
                               ("accepted_steps", 1), ("calls", 1)):
                request = submission()
                request["limits"][key] = value
                assert http.post("/jobs", json=request, headers=headers(catalog)).status_code == 422
            assert http.post("/jobs", content=canonical(submission()).replace(b'"generation_scale":1.0',
                b'"generation_scale":NaN'), headers={**headers(catalog), "Content-Type": "application/json"}).status_code == 400
            assert store.events() == []

    from solarlab.experiments.registry import ExperimentRegistry
    definition = catalog.registry.get("steady_jv", 1)
    denied_registry = ExperimentRegistry((replace(definition, capability=replace(definition.capability, dimensions=(2,))),))
    denied = replace(catalog, registry=denied_registry, expected_registry_sha256=denied_registry.content_sha256)
    with RunStore(tmp_path / "denied") as store, pool(store, tmp_path / "denied-work", denied) as supervisor:
        with TestClient(create_app(store.root, job_service=JobService(denied, supervisor))) as http:
            response = http.post("/jobs", json=submission(), headers=headers(denied))
            assert response.status_code == 422 and "capability refused" in response.text
            assert store.events() == []


def test_unsupported_installed_physics_and_mutable_factory_are_not_admitted(context):
    from solarlab.config.resolve_device import resolve_device
    from solarlab.device.inputs import DeviceInput
    from solarlab_server.job_service import SteadyRegistration

    prepared = context.device("research_1d")
    value = prepared.to_input().editing_data()
    value["layers"][1]["parameters"]["D_ion"] = 1e-15
    ions = resolve_device(DeviceInput.model_validate(value), prepared.defaults, prepared.resources, sources=prepared.sources)
    with pytest.raises(ValueError, match="D_ion"):
        replace(context, devices=(("ions", ions),))
    with pytest.raises(ValueError, match="unique registered IDs"):
        replace(context, devices=(("research_1d", prepared), ("research_1d", prepared)))
    with pytest.raises(TypeError):
        SteadyRegistration("steady_jv", 1, context, prepare=lambda input: None)


@pytest.mark.parametrize("field", ["device", "implementation", "model", "worker"])
def test_spawned_route_rechecks_installed_binding_before_native_work(tmp_path, catalog, field):
    spec, execution = catalog.prepare(JobSubmission.model_validate(submission()))
    claimed = json.loads(catalog.document.content)
    entry = claimed["bindings"][0]
    if field == "device":
        entry["preparation_context"]["devices"][0]["defaults_sha256"] = "0" * 64
    elif field == "implementation":
        entry["preparation_context"]["implementation"]["source_manifest_sha256"] = "0" * 64
    else:
        entry["model_registry_sha256" if field == "model" else "worker_source_sha256"] = "0" * 64
    altered = SourceDocument("explicit-inconsistent-steady-catalog-fixture", canonical(claimed))
    # Rebind transport deliberately: the child must check the relation to its
    # trusted manifest and installed worker, not merely the outer byte digest.
    worker = replace(catalog.worker_binding("solarlab"), registry_sha256=altered.sha256)
    with RunStore(tmp_path / "store") as store, pool(store, tmp_path / "work", catalog, bindings=(worker,)) as supervisor:
        handle = supervisor.submit("bad-binding", worker.id, replace(spec.request, registry=altered),
                                   spec.limits, execution=execution)
        receipt = supervisor.wait(handle, 15).data
        assert receipt["store_state"] == "failed" and receipt["process_joined"]
        assert receipt["worker_result"]["exception_type"] == "ValueError"
        assert "steady" in receipt["worker_result"]["detail"]
        assert not receipt["terminal_artifacts"] and len(terminal_events(store, handle.run_id)) == 1
        assert not any(e["payload"].get("ui_progress", {}).get("stage") == "steady_jv_native"
                       for e in store.events(run_id=handle.run_id, limit=1000))


def _hold(request, context):
    """Source-bound, non-numerical queue occupancy using the existing stop."""
    context.consume_manifest(request, engine="solarlab")
    context.progress({"stage": "http_queue_ready"})
    while True:
        context.check_cancel()
        time.sleep(0.005)


def test_http_cancel_fences_attempt_and_stops_only_the_shared_owner(tmp_path, catalog):
    hold = WorkerBinding.from_function("http_queue_fixture", _hold,
        registry_sha256=catalog.document.sha256, schema_sha256=catalog.schema.sha256)
    spec, execution = catalog.prepare(JobSubmission.model_validate(submission()))
    with RunStore(tmp_path / "store") as store, pool(store, tmp_path / "work", catalog,
            bindings=(catalog.worker_binding("solarlab"), hold)) as supervisor:
        first = supervisor.submit("queue-owner", hold.id, spec.request, spec.limits, execution=execution)
        until = time.monotonic() + 10
        while time.monotonic() < until:
            if any(e["payload"].get("ui_progress", {}).get("stage") == "http_queue_ready"
                   for e in store.events(run_id=first.run_id, limit=1000)):
                break
            time.sleep(0.005)
        else:
            raise AssertionError("queue owner did not reach the observed barrier")
        with TestClient(create_app(store.root, job_service=JobService(catalog, supervisor))) as http:
            response = http.post("/jobs", json=submission("queued-steady"), headers=headers(catalog))
            assert response.status_code == 202, response.text
            accepted = response.json()
            queued = JobHandle(store.store_id, accepted["run_id"], accepted["attempt_id"])
            row = store.get_run(queued.run_id)
            url = f"/jobs/{queued.run_id}/attempts/{queued.attempt_id}/cancel"
            assert http.post(url, json={"expected_generation": row["generation"] + 1, "reason": "stale"}).status_code == 409
            assert http.post(url.replace(queued.attempt_id, "different-attempt"),
                json={"expected_generation": row["generation"], "reason": "wrong attempt"}).status_code == 409
            response = http.post(url, json={"expected_generation": row["generation"], "reason": "explicit HTTP stop"})
            assert response.status_code == 200 and response.json()["state"] == "cancelled"
            stopped = supervisor.wait(queued, 5).data
            assert stopped["store_state"] == "cancelled" and not stopped["process_started"]
            assert len(terminal_events(store, queued.run_id)) == 1
            assert store.attempts(queued.run_id)[0]["terminal"]["artifacts"] == []
            assert store.get_run(first.run_id)["state"] == "running"
            owner = store.get_run(first.run_id)
            response = http.post(f"/jobs/{first.run_id}/attempts/{first.attempt_id}/cancel",
                json={"expected_generation": owner["generation"], "reason": "stop the observed owner"})
            assert response.status_code == 200
            receipt = supervisor.wait(first, 10).data
            assert receipt["store_state"] == "cancelled" and receipt["process_joined"]
            assert receipt["process_exit_observed"] and len(terminal_events(store, first.run_id)) == 1


def test_mixed_catalog_keeps_legacy_and_steady_model_and_engine_bindings_separate(catalog):
    from job_route_fixtures import catalog as legacy_catalog, request as legacy_request
    from solarlab.experiments.registry import ExperimentRegistry

    legacy, _ = legacy_catalog()
    sources = {source.id: source for source in (*legacy.sources, *catalog.sources)}
    declarations = ExperimentRegistry((*legacy.registry.definitions, *catalog.registry.definitions))
    mixed = replace(legacy, registry=declarations, registrations=(*legacy.registrations, *catalog.registrations),
        sources=tuple(sources.values()), expected_registry_sha256=declarations.content_sha256, ceiling=limits())
    legacy_spec, old = mixed.prepare(JobSubmission.model_validate(legacy_request()))
    steady_spec, new = mixed.prepare(JobSubmission.model_validate(submission()))
    assert legacy_spec.worker_id == "registered_legacy" and old.engine.name == "legacy"
    assert steady_spec.worker_id == "registered_steady" and new.engine.name == "solarlab"
    assert old.inputs.registry.content_sha256 != new.inputs.registry.content_sha256
    assert not old.manifest.can_execute and not new.manifest.can_execute
    for engine in ("legacy", "solarlab"):
        assert mixed.worker_binding(engine).registry_sha256 == mixed.document.sha256
    for data in ({**submission(), "engine": "legacy"}, {**legacy_request(), "engine": "solarlab"}):
        with pytest.raises(ValueError, match="fallback is forbidden"):
            mixed.prepare(JobSubmission.model_validate(data))


def test_local_service_script_keeps_engine_imports_after_worker_isolation(tmp_path):
    code = '''
import importlib.abc, runpy, sys
class NoEarlyEngine(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"scipy", "perovskite_sim"}:
            raise AssertionError("engine imported during spawn main reconstruction: " + fullname)
sys.meta_path.insert(0, NoEarlyEngine())
runpy.run_path(sys.argv[1], run_name="__mp_main__")
assert not ({"scipy", "perovskite_sim"} & set(sys.modules))
'''
    process = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(ROOT / "examples/research_1d/service.py")],
                             cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert process.returncode == 0, process.stderr
