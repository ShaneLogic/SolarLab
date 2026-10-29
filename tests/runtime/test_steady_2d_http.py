"""One genuine HTTP success; original failed process data is reused exactly."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import time

from fastapi.testclient import TestClient
import httpx
import numpy as np
import pytest


from examples.research_1d.process_ionic_transient import source_build
from examples.research_2d.http_steady import example_submission, installed_catalog, resource_limits
from solarlab.experiments.execution import ProcessSupervisor
from solarlab.experiments.two_dimensional.steady_results import read_steady_2d_job
from solarlab.experiments.two_dimensional.steady_worker import _request_contract, steady_2d_worker_binding
from solarlab.experiments.worker import WorkerRequest
from solarlab.io.run_store import RunStore
from solarlab.materials.source import SourceDocument
from solarlab_server.app import create_app
from solarlab_server.job_service import ExecutionCatalog, JobService, JobSubmission, canonical
from solarlab_server.steady_2d_preparation import (
    Steady2DJobInput, execute_registered_steady_2d, prepare_steady_2d_input, read_registered_steady_2d_job,
)
from test_transient_job_service import port_closed, ready

pytestmark = pytest.mark.requires_clean_core

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_SHA256 = "98f6e3c174106153ec72564e117e316f6395a065353ecee89449faa5abedbaa2"
FAILED_RUN = "steady2d-failed"
FAILED_ATTEMPT = "f553077fc14c4d2ba163b33926dcf3e2"


def extract_historical(folder):
    archive = Path(__file__).parent / "fixtures/steady_2d_process_saved.tar.gz"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == FIXTURE_SHA256
    with tarfile.open(archive) as bundle:
        assert sum(item.size for item in bundle.getmembers()) < 16 * 1024**2
        bundle.extractall(folder, filter="data")
    proof = json.loads((folder / "Fixture.json").read_text())
    assert proof["original_full_store_snapshot"] and not proof["derived_sqlite_subset"]
    assert proof["new_numerical_jobs"] == 0
    for name, sha in proof["files"].items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == sha
    assert (folder / "Store/partial").is_dir() and (folder / "Store/.writer.lock").is_file()
    return folder / "Store", proof


@pytest.fixture(scope="module")
def implementation():
    commit = os.environ.get("SOLARLAB_STEADY2D_HTTP_COMMIT") or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=5).strip()
    archive = os.environ.get("SOLARLAB_STEADY2D_HTTP_ARCHIVE")
    return source_build(commit, Path(archive) if archive else None)


@pytest.fixture(scope="module")
def catalog(implementation):
    return installed_catalog(implementation, implementation.build_source_commit)


@pytest.fixture(scope="module")
def historical(tmp_path_factory):
    return extract_historical(tmp_path_factory.mktemp("steady2d-historical"))


def owner(store, work, catalog):
    return ProcessSupervisor(store, work, catalog.worker_bindings(), max_active=1, max_pending=2,
        shared_rss_bytes=3 * 1024**3, shared_output_bytes=384 * 1024**2)


def test_preflight_preserves_core_body_and_actual_dimension2(catalog):
    body = example_submission()
    spec, binding = catalog.prepare(JobSubmission.model_validate(body))
    original = prepare_steady_2d_input(Steady2DJobInput.model_validate(body["input"]), catalog.registrations[0].context)
    assert spec.worker_id == "registered_steady_2d" and spec.request.request == original.request.request
    assert spec.request.registry == catalog.document and spec.request.schema == catalog.schema
    assert binding.inputs.selection.context.dimension == 2
    assert binding.inputs.spatial_geometry is not None
    assert json.loads(binding.inputs.spatial_geometry.content)["Nx"] == 4
    assert _request_contract(spec.request, steady_2d_worker_binding().source_sha256)
    assert _request_contract(original.request, steady_2d_worker_binding().source_sha256)
    binding.validate_request(spec.request)
    entry = json.loads(catalog.document.content)["bindings"][0]
    assert entry["driver"] == "steady_2d" and entry["scientific_qualification"] is False
    assert entry["preparation_context"]["fixed_numerical_policy"]["numerical_coordinates"] == 195


def test_registered_dispatch_reaches_unchanged_request_without_numerical_call(catalog, monkeypatch):
    import solarlab_server.steady_2d_preparation as adapter
    spec, binding = catalog.prepare(JobSubmission.model_validate(example_submission()))
    seen = []

    class ReachedNumericalBoundary(Exception):
        pass

    def stop_before_solve(request, context):
        seen.append(request)
        raise ReachedNumericalBoundary()

    class Context:
        def check_cancel(self):
            pass
        def consume_manifest(self, request, *, engine):
            assert engine == "solarlab"
            binding.validate_request(request)
            return binding

    monkeypatch.setattr(adapter, "run_steady_2d_worker", stop_before_solve)
    with pytest.raises(ReachedNumericalBoundary):
        execute_registered_steady_2d(spec.request, Context())
    assert seen == [spec.request]
    broken = json.loads(spec.request.registry.content)
    broken["bindings"][0]["worker_source_sha256"] = "0" * 64
    changed = replace(spec.request, registry=SourceDocument(spec.request.registry.id, canonical(broken)))
    assert not _request_contract(changed, steady_2d_worker_binding().source_sha256)
    with pytest.raises(ValueError, match="source"):
        execute_registered_steady_2d(changed, Context())
    assert len(seen) == 1


@pytest.mark.parametrize("failure", [False, True])
def test_full_controls_accept_only_the_two_explicit_policies(failure):
    body = example_submission(insufficient_budget=failure)["input"]
    parsed = Steady2DJobInput.model_validate(body)
    assert parsed.options().electronic.max_iterations == (1 if failure else 100)
    assert parsed.controls["auxiliary_1d"] == example_submission()["input"]["controls"]["auxiliary_1d"]
    for changed in (True, 1.000001, float("nan")):
        broken = deepcopy(body)
        broken["controls"]["full_2d"]["electronic"]["potential_step_V"] = changed
        with pytest.raises(ValueError):
            Steady2DJobInput.model_validate(broken)


def test_invalid_requests_refused_before_enqueue_and_catalog_get(tmp_path, catalog):
    with RunStore(tmp_path / "Store") as store, owner(store, tmp_path / "Work", catalog) as supervisor:
        with TestClient(create_app(store.root, job_service=JobService(catalog, supervisor))) as client:
            advertised = client.get("/experiments")
            assert advertised.status_code == 200
            headers = {key: advertised.headers[key] for key in ("x-solarlab-registry", "x-solarlab-job-schema")}
            assert client.post("/jobs", json=example_submission()).status_code == 428
            for key in headers:
                assert client.post("/jobs", json=example_submission(), headers={**headers, key: "0" * 64}).status_code == 409
            for update in ({"engine": "legacy"}, {"experiment_id": "steady_jv"}, {"version": 2}, {"source_commit": "fake"}):
                assert client.post("/jobs", json={**example_submission(), **update}, headers=headers).status_code == 422
            for update in ({"device_id": "missing"}, {"voltage_V": .003}, {"generation_scales": [1., 0.]},
                {"generation_scales": [True]}, {"generation_scales": [2.]}, {"controls": {}}, {"initial": {}},
                {"config_path": "/tmp/a"}, {"python_function": "os.system"}, {"initial_state": [0.]}):
                body = example_submission()
                body["input"].update(update)
                assert client.post("/jobs", json=body, headers=headers).status_code == 422
            for key, value in (("Nx", True), ("Nx", 5), ("lateral_length_m", 2e-6), ("lateral_bc", "periodic"),
                               ("microstructure", {}), ("geometry_policy", "other")):
                body = example_submission()
                body["input"]["spatial"][key] = value
                assert client.post("/jobs", json=body, headers=headers).status_code == 422
            for key, value in (("wall_seconds", 31.), ("rss_bytes", 2 * 1024**3), ("calls", 1), ("accepted_steps", 1)):
                body = example_submission()
                body["limits"][key] = value
                assert client.post("/jobs", json=body, headers=headers).status_code == 422
            assert store.events() == []


def test_coinstalled_1d_catalog_keeps_its_own_worker_and_model(implementation, catalog, tmp_path):
    from examples.research_1d.steady import load_device
    from solarlab.experiments.registry import ExperimentRegistry
    from solarlab_server.steady_preparation import SteadyPreparationContext, steady_execution_catalog
    one = steady_execution_catalog(SteadyPreparationContext((("one", load_device()),), implementation,
        implementation.build_source_commit), ceiling=resource_limits())
    registry = ExperimentRegistry((*one.registry.definitions, *catalog.registry.definitions))
    sources = {s.id: s for s in (*one.sources, *catalog.sources)}
    mixed = ExecutionCatalog(registry, one.models, tuple(sources.values()), (*one.registrations, *catalog.registrations),
        registry.content_sha256, one.models.content_sha256, resource_limits())
    assert {b.id for b in mixed.worker_bindings()} == {"registered_steady", "registered_steady_2d"}
    with RunStore(tmp_path / "Store") as store, owner(store, tmp_path / "Work", mixed) as supervisor:
        with TestClient(create_app(store.root, job_service=JobService(mixed, supervisor))) as client:
            entries = client.get("/experiments").json()["bindings"]
            assert {e["driver"] for e in entries} == {"steady_jv", "steady_2d"}
            assert entries[0]["model_registry_sha256"] != entries[1]["model_registry_sha256"]
            assert store.events() == []


def observe_http_case(client, store_root, run_id, attempt_id, *, registered, work=None):
    events_url = f"/runs/{run_id}/attempts/{attempt_id}/events"
    streamed = client.get(events_url)
    assert streamed.status_code == 200, streamed.text
    events = [json.loads(line[6:]) for line in streamed.text.splitlines() if line.startswith("data: ")]
    response = client.get(f"/results/{run_id}/attempts/{attempt_id}")
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["steady_2d"]["origin"] == ("registered_http" if registered else "original_process")
    payloads, arrays = {}, {}
    assert len(result["artifacts"]) == 14
    for item in result["artifacts"]:
        content = client.get(item["content_url"])
        record = client.get("/artifacts/" + item["artifact_id"])
        assert content.status_code == record.status_code == 200
        assert hashlib.sha256(content.content).hexdigest() == item["sha256"]
        assert content.headers["etag"].strip('"') == item["sha256"]
        payloads[item["name"]] = {"sha256": item["sha256"], "bytes": len(content.content), "metadata": record.json()}
        if item["format"] == "npy":
            arrays[item["name"]] = np.load(io.BytesIO(content.content), allow_pickle=False)
    terminal = [e for e in events if e["kind"] == "terminal"]
    assert len(terminal) == 1
    replay = client.get(events_url, headers={"Last-Event-ID": str(events[0]["sequence"])})
    acknowledged = client.get(events_url, headers={"Last-Event-ID": str(events[-1]["sequence"])})
    assert replay.status_code == 200 and acknowledged.status_code == 204
    assert '"kind":"terminal"' in replay.text.replace(" ", "")
    with RunStore(store_root, read_only=True) as store:
        reader = read_registered_steady_2d_job if registered else read_steady_2d_job
        loaded = reader(store, run_id, attempt_id=attempt_id)
        assert len(arrays) == 12 and len(loaded["arrays"]) == 22
        assert result["state"] == loaded["run_state"]
        assert result["steady_2d"]["array_fingerprints"] == loaded["array_fingerprints"]
        for name, array in arrays.items():
            assert array.tobytes() == loaded["stored_arrays"][name].tobytes()
        case = {"run_id": run_id, "attempt_id": attempt_id, "registered_http": registered,
            "result": result, "events": events, "terminal_count": len(terminal), "payloads": payloads,
            "replay_status": replay.status_code, "terminal_ack_status": acknowledged.status_code,
            "logical_arrays": loaded["array_fingerprints"], "document": loaded["document"]}
        if work is not None:
            directory = Path(work) / (store.store_id + "-" + attempt_id)
            deadline = time.monotonic() + 5
            while not (directory / "receipt.json").exists():
                assert time.monotonic() < deadline
                time.sleep(.01)
            receipt = json.loads((directory / "receipt.json").read_text())
            assert receipt["process_joined"] and receipt["process_exit_observed"] and receipt["exitcode"] == 0
            assert receipt["signals"] == [] and receipt["resources"]["owned_wall_seconds"] <= 30
            child = json.loads((directory / "terminal.json").read_text())
            for item in child["artifacts"]:
                assert hashlib.sha256((directory / item["file"]).read_bytes()).hexdigest() == payloads[item["name"]]["sha256"]
            case["receipt"] = receipt
    return case


def test_historical_process_http_reads_keep_original_identity(historical):
    root, proof = historical
    with TestClient(create_app(root)) as client:
        for original in proof["cases"]:
            case = observe_http_case(client, root, original["run_id"], original["attempt_id"], registered=False)
            assert case["result"]["state"] == original["state"]
            assert case["document"]["accepted_count"] == (2 if original["state"] == "succeeded" else 0)
            if original["state"] == "failed":
                assert case["document"]["unattempted_count"] == 1 and case["document"]["reason"] == "budget_exhausted"
            assert case["logical_arrays"] == original["array_fingerprints"]
            assert case["result"]["result"]["worker_result"]["capture"]["build_source_commit"] == proof["source_commit"]


def run_http_witness(folder, implementation, historical_root, *, archive=None, evidence_path=None, review_pause=False):
    root, work = folder / "Store", folder / "Work"
    shutil.copytree(historical_root, root)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    assert port_closed(port)
    command = [sys.executable, "-B", "-m", "examples.research_2d.http_steady", "--store", str(root), "--work", str(work),
        "--build-source-commit", implementation.build_source_commit, "--current-repository-head", implementation.build_source_commit,
        "--port", str(port)]
    if archive:
        command += ["--source-archive", str(archive)]
    env = dict(os.environ)
    env.update({k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")})
    out_path, err_path = folder / "http.stdout.log", folder / "http.stderr.log"
    evidence = {"command": command, "store": str(root), "work": str(work), "port": port,
        "port_closed_before_start": True, "source_commit": implementation.build_source_commit, "cases": [],
        "new_numerical_submissions": 0}

    def save():
        if evidence_path is not None:
            Path(evidence_path).write_text(json.dumps(evidence, indent=2, allow_nan=False))

    with out_path.open("x") as out, err_path.open("x") as err:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err, start_new_session=True)
        evidence["service_pid"] = process.pid
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=45) as client:
                advertised = ready(client, process)
                headers = {k: advertised.headers[k] for k in ("x-solarlab-registry", "x-solarlab-job-schema")}
                evidence.update(catalog=advertised.json(), submission_headers=headers)
                evidence["cases"].append(observe_http_case(client, root, FAILED_RUN, FAILED_ATTEMPT, registered=False))
                body = example_submission()
                response = client.post("/jobs", json=body, headers=headers)
                evidence.update(request=body, post_status=response.status_code, post=response.json(), new_numerical_submissions=1)
                save()
                assert response.status_code == 202, response.text
                target = response.json()
                evidence["cases"].append(observe_http_case(client, root, target["run_id"], target["attempt_id"], registered=True, work=work))
                assert evidence["cases"][-1]["result"]["state"] == "succeeded"
                reopened = client.get("/runs/" + target["run_id"])
                assert reopened.status_code == 200 and reopened.json()["current_attempt_id"] == target["attempt_id"]
                evidence["reopened_attempt_id"] = reopened.json()["current_attempt_id"]
                save()
                if review_pause:
                    print(json.dumps({"review_ready": True, "port": port, "service_pid": process.pid,
                        "run_id": target["run_id"], "attempt_id": target["attempt_id"], "evidence": str(evidence_path)}), flush=True)
                    input("GET-only review window; press Enter to close owned service: ")
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
            code = process.wait(timeout=20)
            evidence.update(service_exit_code=code, port_closed=port_closed(port), stdout_log=str(out_path), stderr_log=str(err_path))
            messages = [json.loads(line) for line in out_path.read_text().splitlines() if line.startswith("{")]
            evidence["closure"] = messages[-1] if messages else None
            save()
    assert code == 0 and evidence["port_closed"], err_path.read_text()
    return evidence


@pytest.fixture(scope="module")
def actual(tmp_path_factory, implementation, historical):
    if saved := os.environ.get("SOLARLAB_STEADY2D_HTTP_SAVED"):
        return json.loads(Path(saved).read_text())
    return run_http_witness(tmp_path_factory.mktemp("steady2d-http"), implementation, historical[0],
        archive=os.environ.get("SOLARLAB_STEADY2D_HTTP_ARCHIVE"))


def test_actual_http_success_and_original_failed_prefix(actual):
    assert actual["new_numerical_submissions"] == 1
    assert actual["post_status"] == 202 and actual["service_exit_code"] == 0 and actual["port_closed"]
    assert actual["closure"] == {"service_closed": True, "read_handle_closed": True, "process_owner_closed": True}
    old, new = actual["cases"]
    assert old["result"]["state"] == "failed" and old["document"]["accepted_count"] == 0
    assert new["result"]["state"] == "succeeded" and new["document"]["accepted_count"] == 2
    assert new["document"]["computation_counts"]["auxiliary_1d_seed_calls"] == 2
    assert new["document"]["computation_counts"]["full_2d_newton_calls"] == 2
    assert new["document"]["shape_Ny_Nx"] == [13, 5] and new["document"]["photovoltaic_metrics"] is None
    assert old["terminal_count"] == new["terminal_count"] == 1
    assert len(old["logical_arrays"]) == len(new["logical_arrays"]) == 22


def test_saved_registered_reader_forbids_numerical_imports(actual):
    code = """import importlib.abc,json,sys
from pathlib import Path
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.startswith(('solarlab.solvers','solarlab.equations','solarlab.device.compile_','perovskite_sim')):
   raise AssertionError('forbidden numerical read import: '+fullname)
sys.meta_path.insert(0,Block())
from solarlab.io.run_store import RunStore
from solarlab_server.steady_2d_preparation import read_registered_steady_2d_job
with RunStore(Path(sys.argv[1]),read_only=True) as store:
 loaded=read_registered_steady_2d_job(store,sys.argv[2],attempt_id=sys.argv[3])
 print(json.dumps(loaded['array_fingerprints']))
"""
    case = actual["cases"][-1]
    result = subprocess.run([sys.executable, "-B", "-c", code, actual["store"], case["run_id"], case["attempt_id"]],
        cwd=ROOT, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == case["logical_arrays"]


@pytest.mark.parametrize("change", ["source", "geometry", "schema"])
def test_saved_http_catalog_tamper_rejected(actual, change):
    case = actual["cases"][-1]
    with RunStore(Path(actual["store"]), read_only=True) as store:
        class Changed:
            def __getattr__(self, name):
                return getattr(store, name)
            def get_run(self, run_id):
                row = deepcopy(store.get_run(run_id))
                request = WorkerRequest.from_bytes(canonical(row["request"]["input_metadata"]["transport"]))
                catalog = json.loads(request.registry.content)
                entry = catalog["bindings"][0]
                if change == "source":
                    entry["worker_source_sha256"] = "0" * 64
                elif change == "geometry":
                    entry["preparation_context"]["fixed_numerical_policy"]["spatial"]["Nx"] = 6
                else:
                    entry["input_schema"]["title"] = "changed"
                request = replace(request, registry=SourceDocument(request.registry.id, canonical(catalog)))
                row["request"]["input_metadata"]["transport"] = json.loads(request.to_bytes())
                return row
        with pytest.raises(ValueError):
            read_registered_steady_2d_job(Changed(), case["run_id"], attempt_id=case["attempt_id"])
