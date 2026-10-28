"""One real ionic HTTP pair; reuse its exact attempts for all saved reads."""
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from fastapi.testclient import TestClient
import httpx
import numpy as np
import pytest

from examples.research_1d.ionic_transient_service import ROOT, example_submission, installed_catalog
from examples.research_1d.process_ionic_transient import example_job, source_build
from solarlab.experiments.execution import JobHandle
from solarlab.experiments.worker import WorkerBinding
from solarlab.io.run_store import RunStore
from solarlab.materials.source import SourceDocument
from solarlab_server.app import create_app
from solarlab_server.ionic_transient_preparation import (
    IonicTransientPreparationContext, read_registered_ionic_transient_job,
)
from solarlab_server.job_service import JobService, JobSubmission, canonical
from test_transient_job_service import live_http, pool, port_closed, ready, terminal_events


def save(name, data):
    if value := os.environ.get("SOLARLAB_IONIC_HTTP_EVIDENCE"):
        target = Path(value)
        target.mkdir(parents=True, exist_ok=True)
        (target / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


@pytest.fixture(scope="module")
def implementation():
    commit = os.environ.get("SOLARLAB_IONIC_HTTP_COMMIT") or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=5).strip()
    archive = os.environ.get("SOLARLAB_IONIC_HTTP_ARCHIVE")
    return source_build(commit, Path(archive) if archive else None)


@pytest.fixture(scope="module")
def catalog(implementation):
    return installed_catalog(implementation, implementation.build_source_commit)


@pytest.fixture(scope="module")
def actual(tmp_path_factory, implementation):
    if existing := os.environ.get("SOLARLAB_IONIC_HTTP_SAVED"):
        return json.loads(Path(existing).read_text())
    folder = tmp_path_factory.mktemp("ionic-http")
    root = Path(os.environ.get("SOLARLAB_IONIC_HTTP_STORE", str(folder / "store"))).resolve()
    work = Path(os.environ.get("SOLARLAB_IONIC_HTTP_WORK", str(root.parent / "work"))).resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    assert port_closed(port)
    command = [sys.executable, "-B", "-m", "examples.research_1d.ionic_transient_service", "--store", str(root),
        "--work", str(work), "--build-source-commit", implementation.build_source_commit,
        "--current-repository-head", implementation.build_source_commit, "--port", str(port)]
    if archive := os.environ.get("SOLARLAB_IONIC_HTTP_ARCHIVE"):
        command += ["--source-archive", archive]
    env = dict(os.environ)
    names = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
    env.update({name: "1" for name in names})
    stdout, stderr = root.parent / "http.stdout.log", root.parent / "http.stderr.log"
    evidence = {"command": command, "cwd": str(ROOT), "store": str(root), "work": str(work), "port": port,
        "source_commit": implementation.build_source_commit, "port_closed_before_start": True,
        "thread_environment": {name: env[name] for name in names}, "cases": []}
    with stdout.open("x") as out, stderr.open("x") as err:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err, start_new_session=True)
        evidence["service_pid"] = process.pid
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=45) as client:
                advertised = ready(client, process)
                headers = {name: advertised.headers[name] for name in ("x-solarlab-registry", "x-solarlab-job-schema")}
                evidence.update(catalog=advertised.json(), submission_headers=headers)
                for run_id, failure, state in (("http-ionic-completed", False, "succeeded"),
                                              ("http-ionic-failed-prefix", True, "failed")):
                    body = example_submission(run_id, failure=failure)
                    response = client.post("/jobs", json=body, headers=headers)
                    case = {"request": body, "post_status": response.status_code, "post_body": response.json()}
                    evidence["cases"].append(case)
                    save("ActualHTTP.json", evidence)
                    assert response.status_code == 202, response.text
                    accepted = response.json()
                    streamed = client.get(accepted["events_url"])
                    events = [json.loads(line[6:]) for line in streamed.text.splitlines() if line.startswith("data: ")]
                    case.update(events=events, event_status=streamed.status_code)
                    result_response = client.get(accepted["result_url"])
                    result = result_response.json()
                    case.update(result_status=result_response.status_code, result=result)
                    save("ActualHTTP.json", evidence)
                    assert result_response.status_code == 200 and result["state"] == state, result
                    assert len(result["artifacts"]) == 16
                    arrays, payloads = {}, {}
                    for item in result["artifacts"]:
                        content = client.get(item["content_url"])
                        metadata = client.get("/artifacts/" + item["artifact_id"])
                        assert content.status_code == metadata.status_code == 200
                        assert hashlib.sha256(content.content).hexdigest() == item["sha256"]
                        assert content.headers["etag"].strip('"') == item["sha256"]
                        payloads[item["name"]] = {"sha256": item["sha256"], "http_status": content.status_code,
                            "etag": content.headers["etag"], "bytes": len(content.content), "metadata": metadata.json()}
                        if item["format"] == "npy":
                            arrays[item["name"]] = np.load(io.BytesIO(content.content), allow_pickle=False)
                    replay = client.get(accepted["events_url"] + "?after=" + str(events[-1]["sequence"]),
                        headers={"Last-Event-ID": str(events[0]["sequence"])})
                    acknowledged = client.get(accepted["events_url"], headers={"Last-Event-ID": str(events[-1]["sequence"])})
                    repeated = client.post("/jobs", json=body, headers=headers)
                    assert repeated.status_code == 202 and repeated.json()["attempt_id"] == accepted["attempt_id"]
                    assert replay.status_code == 200 and acknowledged.status_code == 204
                    assert '"kind":"terminal"' in replay.text.replace(" ", "")
                    with RunStore(root, read_only=True) as store:
                        attempt, = store.attempts(run_id)
                        directory = work / (store.store_id + "-" + attempt["attempt_id"])
                        deadline = time.monotonic() + 5
                        while not (directory / "receipt.json").exists():
                            assert time.monotonic() < deadline
                            time.sleep(.01)
                        receipt = json.loads((directory / "receipt.json").read_text())
                        loaded = read_registered_ionic_transient_job(store, run_id, attempt_id=accepted["attempt_id"])
                        assert len(arrays) == 13 and len(loaded["arrays"]) == 26
                        for name, array in arrays.items():
                            assert array.tobytes() == loaded["stored_arrays"][name].tobytes()
                        raw = json.loads((directory / "terminal.json").read_text())
                        for item in raw["artifacts"]:
                            assert hashlib.sha256((directory / item["file"]).read_bytes()).hexdigest() == payloads[item["name"]]["sha256"]
                        case.update(receipt=receipt, terminal_count=len(terminal_events(store, run_id)),
                            accepted_times_s=loaded["arrays"]["accepted_times_s"].tolist(),
                            first_failure=None if state == "succeeded" else {k: loaded["diagnostics"]["history"]["attempts"][-1][k]
                                for k in ("start_s", "end_s", "termination", "accepted")},
                            logical_arrays={name: {"sha256": hashlib.sha256(a.tobytes()).hexdigest(), "shape": list(a.shape),
                                "unit": loaded["document"]["logical_array_units"][name]} for name, a in loaded["arrays"].items()})
                    case.update(payloads=payloads, replay_status=replay.status_code, terminal_acknowledgement_status=acknowledged.status_code,
                                repeated_attempt_id=repeated.json()["attempt_id"])
                    assert receipt["process_joined"] and receipt["process_exit_observed"] and receipt["exitcode"] == 0
                    assert case["terminal_count"] == sum(e["kind"] == "terminal" for e in events) == 1
                    save("ActualHTTP.json", evidence)
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
            try:
                code = process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.terminate()
                code = process.wait(timeout=5)
            evidence.update(service_exit_code=code, port_closed=port_closed(port), stdout_log=str(stdout), stderr_log=str(stderr))
            messages = [json.loads(line) for line in stdout.read_text().splitlines() if line.startswith("{")]
            evidence["closure"] = messages[-1] if messages else None
            save("ActualHTTP.json", evidence)
    assert code == 0 and evidence["port_closed"], stderr.read_text()
    assert evidence["closure"] == {"service_closed": True, "read_handle_closed": True, "process_owner_closed": True}
    return evidence


def test_actual_http_pair_retains_terminal_truth_and_exact_packed_arrays(actual):
    assert len(actual["cases"]) == 2 and actual["service_exit_code"] == 0 and actual["port_closed"]
    for case, state, times in zip(actual["cases"], ("succeeded", "failed"), ([0., .25, .5, .75, 1.], [0., .25]), strict=True):
        assert case["result"]["state"] == state and case["accepted_times_s"] == times
        worker = case["result"]["result"]["worker_result"]
        assert worker["registered_device_id"] == "research_1d_shared"
        assert worker["execution_catalog_sha256"] == actual["submission_headers"]["x-solarlab-registry"]
        assert worker["capture"]["build_source_commit"] == actual["source_commit"]
        assert not worker["scientific_qualification"] and worker["ionic_result"]["physical_certificate_status"] == "not_evaluated"
        assert len(case["logical_arrays"]) == 26 and len(case["payloads"]) == 16 and case["terminal_count"] == 1
        assert case["logical_arrays"]["potential_V"]["shape"] == [len(times), 37]
        assert case["logical_arrays"]["positive_ion_density_m3"]["shape"] == [len(times), 13]
        assert case["logical_arrays"]["total_current_density_A_m2"]["shape"] == [len(times) - 1, 38]
        if state == "failed":
            assert case["first_failure"] == {"start_s": .25, "end_s": .5, "termination": "budget_exhausted", "accepted": False}


def test_saved_http_reader_and_get_routes_forbid_solvers(actual, tmp_path):
    code = '''
import hashlib,importlib.abc,json,pathlib,sys
sys.path.insert(0,sys.argv[1])
class NoSolver(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if any(fullname==n or fullname.startswith(n+'.') for n in ('solarlab.solvers','solarlab.equations',
            'solarlab.device.compile_1d','solarlab.device.compile_ionic_1d','solarlab.experiments.ionic_transient',
            'solarlab.experiments.transient','perovskite_sim')):
            raise AssertionError('saved reader imported solver: '+fullname)
sys.meta_path.insert(0,NoSolver())
from fastapi.testclient import TestClient
from solarlab.io.run_store import RunStore
from solarlab_server.app import create_app
from solarlab_server.ionic_transient_preparation import read_registered_ionic_transient_job
root=pathlib.Path(sys.argv[2]).resolve()
with RunStore(root,read_only=True) as store,TestClient(create_app(root)) as client:
    for name in ('http-ionic-completed','http-ionic-failed-prefix'):
        loaded=read_registered_ionic_transient_job(store,name)
        attempt,=store.attempts(name)
        response=client.get('/results/'+name+'/attempts/'+attempt['attempt_id'])
        assert response.status_code==200 and response.json()['state']==loaded['run_state']
        print(json.dumps({'run_id':name,'state':loaded['run_state'],'arrays':{
            key:hashlib.sha256(a.tobytes()).hexdigest() for key,a in loaded['arrays'].items()}}))
'''
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(ROOT / "src"), actual["store"]],
                            cwd=tmp_path, text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
    for row, case in zip(result.stdout.splitlines(), actual["cases"], strict=True):
        assert json.loads(row)["arrays"] == {name: value["sha256"] for name, value in case["logical_arrays"].items()}
    save("SolverFreeHTTPRead.json", {"argv": result.args, "exit_code": result.returncode, "stdout": result.stdout,
        "stderr": result.stderr, "new_numerical_jobs": 0})


def test_explicit_admission_refuses_headers_scope_code_and_budgets(tmp_path, catalog):
    with RunStore(tmp_path / "store") as store, pool(store, tmp_path / "work", catalog) as owner:
        with TestClient(create_app(store.root, job_service=JobService(catalog, owner))) as client:
            headers = {"X-Solarlab-Registry": catalog.document.sha256, "X-Solarlab-Job-Schema": catalog.schema.sha256}
            assert client.post("/jobs", json=example_submission()).status_code == 428
            for key in headers:
                assert client.post("/jobs", json=example_submission(), headers={**headers, key: "0" * 64}).status_code == 409
            for updates in ({"engine": "legacy"}, {"experiment_id": "transient_1d"}, {"version": 2}, {"source_commit": "claim"}):
                assert client.post("/jobs", json={**example_submission(), **updates}, headers=headers).status_code == 422
            for updates in ({"device_id": "unknown"}, {"times_s": [0., 0.]}, {"times_s": [0., 2.]},
                {"voltage_V": [.1] * 5}, {"generation_scale": [1.] * 5}, {"intervals_per_layer": [12, 12]},
                {"intervals_per_layer": [True, 12, 12]}, {"intervals_per_layer": [65, 12, 12]},
                {"ion_domain_layer": "other"}, {"D_ion": 0.}, {"initial_state": [0.]}, {"python_function": "os.system"},
                {"source_bytes": "x"}, {"config_path": "/tmp/config"}):
                body = example_submission()
                body["input"].update(updates)
                assert client.post("/jobs", json=body, headers=headers).status_code == 422
            body = example_submission()
            body["input"]["controls"]["transient"]["poisson_tolerance"] = 1.
            assert client.post("/jobs", json=body, headers=headers).status_code == 422
            for name, value in (("wall_seconds", 31.), ("rss_bytes", 2 * 1024**3), ("output_bytes", 128 * 1024**2),
                                ("calls", 1), ("accepted_steps", 1)):
                body = example_submission()
                body["limits"][name] = value
                assert client.post("/jobs", json=body, headers=headers).status_code == 422
            assert store.events() == []


def test_ionic_preparation_is_exact_and_existing_steady_route_is_preserved(catalog, implementation):
    for failure in (False, True):
        spec, _ = catalog.prepare(JobSubmission.model_validate(example_submission(failure=failure)))
        original = example_job(implementation, failure=failure, current_repository_head=implementation.build_source_commit)
        assert spec.request.request.content == original.request.request.content
    from examples.research_1d.transient_service import installed_catalog as existing_catalog
    from solarlab.experiments.registry import ExperimentRegistry
    other = existing_catalog(implementation, implementation.build_source_commit)
    sources = {s.id: s for s in (*catalog.sources, *other.sources)}
    registry = ExperimentRegistry((*catalog.registry.definitions, *other.registry.definitions))
    combined = replace(catalog, registry=registry, expected_registry_sha256=registry.content_sha256,
        sources=tuple(sources.values()), registrations=(*catalog.registrations, *other.registrations))
    assert {b.id for b in combined.worker_bindings()} == {"registered_ionic_transient", "registered_steady", "registered_transient"}
    with pytest.raises(ValueError, match="explicit experiment_id"):
        combined.worker_binding("solarlab")
    body = example_submission()
    body.update(experiment_id="steady_jv", input={"device_id": "research_1d", "voltage_V": [0., .05, .1],
        "intervals_per_layer": 12, "generation_scale": 1., "incident_power_W_m2": None})
    spec, _ = combined.prepare(JobSubmission.model_validate(body))
    assert spec.worker_id == "registered_steady"


@pytest.mark.parametrize("change", ["device", "server", "model", "policy"])
def test_ionic_child_rejects_changed_registration_before_native(tmp_path, catalog, change):
    spec, execution = catalog.prepare(JobSubmission.model_validate(example_submission()))
    data = json.loads(catalog.document.content)
    entry = data["bindings"][0]
    if change == "device":
        next(d for d in entry["preparation_context"]["devices"] if d["device_id"] == "research_1d_shared")["prepared_sha256"] = "0" * 64
    elif change == "server":
        entry["preparation_context"]["server_sources"][0]["sha256"] = "0" * 64
    elif change == "policy":
        entry["preparation_context"]["fixed_numerical_policy"]["transient"]["ion_logit_step"] = 2.
    else:
        entry["model_registry_sha256"] = "0" * 64
    document = SourceDocument("changed-ionic-catalog", canonical(data))
    binding = replace(catalog.worker_binding("solarlab"), registry_sha256=document.sha256)
    with RunStore(tmp_path / "store") as store, pool(store, tmp_path / "work", catalog, bindings=(binding,)) as owner:
        handle = owner.submit("invalid", binding.id, replace(spec.request, registry=document), spec.limits, execution=execution)
        receipt = owner.wait(handle, 15).data
        assert receipt["store_state"] == "failed" and not receipt["terminal_artifacts"]
        assert receipt["process_joined"] and receipt["process_exit_observed"]
        assert not any(e["payload"].get("ui_progress", {}).get("stage", "").startswith("ionic_") for e in store.events(run_id="invalid", limit=1000))


def _hold(request, context):
    """Queue ownership fixture only; no scientific computation."""
    context.consume_manifest(request, engine="solarlab")
    context.progress({"stage": "queue_fixture_ready"})
    while True:
        context.check_cancel()
        time.sleep(.005)


def test_http_exact_queued_cancel_has_no_ionic_history(tmp_path, catalog):
    spec, execution = catalog.prepare(JobSubmission.model_validate(example_submission()))
    hold = WorkerBinding.from_function("queue_fixture", _hold, registry_sha256=catalog.document.sha256, schema_sha256=catalog.schema.sha256)
    with RunStore(tmp_path / "store") as store, pool(store, tmp_path / "work", catalog,
            bindings=(*catalog.worker_bindings(), hold)) as owner:
        blocker = owner.submit("blocker", hold.id, spec.request, spec.limits, execution=execution)
        deadline = time.monotonic() + 10
        while not any(e["payload"].get("ui_progress", {}).get("stage") == "queue_fixture_ready" for e in store.events(run_id="blocker", limit=1000)):
            assert time.monotonic() < deadline
            time.sleep(.01)
        app = create_app(store.root, job_service=JobService(catalog, owner))
        with live_http(app) as (client, port):
            headers = {"X-Solarlab-Registry": catalog.document.sha256, "X-Solarlab-Job-Schema": catalog.schema.sha256}
            response = client.post("/jobs", json=example_submission("cancelled-ionic"), headers=headers)
            assert response.status_code == 202, response.text
            accepted = response.json()
            row = store.get_run("cancelled-ionic")
            url = f"/jobs/{row['run_id']}/attempts/{row['current_attempt_id']}/cancel"
            assert client.post(url, json={"expected_generation": row["generation"] + 1, "reason": "stale"}).status_code == 409
            assert client.post(url, json={"expected_generation": row["generation"], "reason": "cancel this queued attempt"}).status_code == 200
            receipt = owner.wait(JobHandle(store.store_id, row["run_id"], row["current_attempt_id"]), 5).data
            result = client.get(accepted["result_url"]).json()
            assert result["state"] == "cancelled" and result["artifacts"] == [] and not receipt["process_started"]
            assert len(terminal_events(store, row["run_id"])) == 1
            with pytest.raises(ValueError, match="committed"):
                read_registered_ionic_transient_job(store, row["run_id"])
            assert store.get_run("blocker")["state"] == "running"
            assert owner.cancel(blocker)
            closed = owner.wait(blocker, 5).data
            assert closed["process_exit_observed"] and closed["process_joined"]
        save("CancellationHTTP.json", {"store": str(store.root), "result": result, "receipt": receipt,
            "control_fixture_receipt": closed, "port_closed": port_closed(port), "reader_closed": app.state.run_store is None,
            "new_ionic_numerical_jobs": 0})


def test_registration_refuses_ambiguous_or_immobile_device(catalog):
    from examples.research_1d.ionic_multilayer import prepare_confined_dual_ion_device
    context = catalog.registrations[0].context
    name, prepared = context.devices[0]
    with pytest.raises(ValueError, match="unique"):
        replace(context, devices=((name, prepared), ("alias", prepared)))
    immobile = prepare_confined_dual_ion_device(positive_diffusivity_m2_s=0.)
    with pytest.raises(ValueError, match="D0"):
        IonicTransientPreparationContext(((name, immobile),), "absorber", context.implementation, context.current_repository_head)
