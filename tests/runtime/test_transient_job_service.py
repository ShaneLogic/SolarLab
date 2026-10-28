"""Real loopback transient HTTP jobs, bounded admission and exact owner cleanup."""
from contextlib import contextmanager
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
import threading
import time

from fastapi.testclient import TestClient
import httpx
import numpy as np
import pytest
import uvicorn

from examples.research_1d.transient_service import ROOT, example_submission, installed_catalog
from solarlab.experiments.execution import JobHandle, ProcessSupervisor
from solarlab.experiments.transient_worker import read_transient_job
from solarlab.experiments.worker import WorkerBinding
from solarlab.io.run_store import RunStore
from solarlab.materials.source import SourceDocument
from solarlab_server.app import create_app
from solarlab_server.job_service import JobService, JobSubmission, canonical


def terminal_events(store, run_id):
    return [e for e in store.events(run_id=run_id, limit=1000) if e["kind"] == "terminal"]


def save(name, data):
    if target := os.environ.get("SOLARLAB_TRANSIENT_SERVICE_EVIDENCE"):
        folder = Path(target)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def ready(client, process=None):
    until = time.monotonic() + 20
    while time.monotonic() < until:
        if process is not None and process.poll() is not None:
            raise AssertionError("service process exited before its HTTP catalog became available")
        try:
            response = client.get("/experiments")
            if response.status_code == 200:
                return response
        except httpx.HTTPError:
            pass
        time.sleep(.02)
    raise AssertionError("local service did not become ready")


def port_closed(port):
    with socket.socket() as probe:
        probe.settimeout(.5)
        return probe.connect_ex(("127.0.0.1", port)) != 0


@pytest.fixture(scope="module")
def commit():
    return os.environ.get("SOLARLAB_TRANSIENT_SERVICE_COMMIT") or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=5).strip()


@pytest.fixture(scope="module")
def catalog(commit):
    # Full committed core is the shared-suite default, including modules that
    # earlier AC/ionic test collection may have imported. An isolated local
    # witness can opt into the example's narrower immutable scope explicitly.
    if os.environ.get("SOLARLAB_TRANSIENT_SERVICE_ISOLATED_SOURCES") == "1":
        from examples.research_1d.process_transient import source_build
        implementation = source_build(commit)
    else:
        from examples.research_1d.service import committed_implementation
        implementation = committed_implementation(commit)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=5).strip()
    return installed_catalog(implementation, head)


def pool(store, work, catalog, *, bindings=None):
    return ProcessSupervisor(store, work, bindings or catalog.worker_bindings(), max_active=1, max_pending=2,
        shared_rss_bytes=3 * 1024**3, shared_output_bytes=384 * 1024**2,
        soft_grace_seconds=.1, terminate_grace_seconds=.1, poll_seconds=.01)


@contextmanager
def live_http(app):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="on", loop="asyncio", http="h11"))
    thread = threading.Thread(target=lambda: server.run(sockets=[listener]), daemon=True)
    thread.start()
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=20) as client:
            ready(client)
            yield client, port
    finally:
        server.should_exit = True
        thread.join(timeout=15)
        listener.close()
        assert not thread.is_alive() and app.state.run_store is None and port_closed(port)


def test_real_loopback_completed_failed_steady_reconnect_and_normal_shutdown(tmp_path, commit):
    root = Path(os.environ.get("SOLARLAB_TRANSIENT_SERVICE_STORE", str(tmp_path / "store"))).resolve()
    work = root.parent / "work"
    root.parent.mkdir(parents=True, exist_ok=True)
    stdout, stderr = root.parent / "server.stdout.log", root.parent / "server.stderr.log"
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    command = [sys.executable, "-B", str(ROOT / "examples/research_1d/transient_service.py"),
               "--store", str(root), "--work", str(work), "--build-source-commit", commit, "--port", str(port)]
    env = dict(os.environ)
    thread_names = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
    env.update({name: "1" for name in thread_names})
    evidence = {"command": command, "store": str(root), "work": str(work), "port": port,
                "source_commit": commit, "cases": [], "thread_environment": {n: env[n] for n in thread_names}}
    with stdout.open("x") as out, stderr.open("x") as err:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err, start_new_session=True)
        evidence["service_pid"] = process.pid
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=75) as client:
                advertised = ready(client, process)
                catalog = advertised.json()
                evidence["catalog"] = catalog
                headers = {name: advertised.headers[name] for name in ("x-solarlab-registry", "x-solarlab-job-schema")}
                assert {r["driver"] for r in catalog["bindings"]} == {"steady_jv", "transient_1d"}
                assert all(not r["scientific_qualification"] for r in catalog["bindings"])
                assert catalog["registry"]["registered_executors"] == []
                entries = {r["driver"]: r for r in catalog["bindings"]}
                assert entries["transient_1d"]["preparation_context"]["implementation"]["build_source_commit"] == commit
                assert entries["transient_1d"]["model_registry_sha256"] != entries["steady_jv"]["model_registry_sha256"]
                steady = {"schema_version": "solarlab.job-submission.v1", "run_id": "http-steady-regression",
                    "experiment_id": "steady_jv", "version": 1, "engine": "solarlab",
                    "input": {"device_id": "research_1d", "voltage_V": [0., .05, .1], "intervals_per_layer": 12,
                              "generation_scale": 1., "incident_power_W_m2": None},
                    "limits": example_submission()["limits"]}
                for body, state, count in [(example_submission("http-transient-completed"), "succeeded", 16),
                        (example_submission("http-transient-failed", failure=True), "failed", 16),
                        (steady, "succeeded", 9)]:
                    response = client.post("/jobs", json=body, headers=headers)
                    assert response.status_code == 202, response.text
                    accepted = response.json()
                    event_response = client.get(accepted["events_url"])
                    assert event_response.status_code == 200
                    events = [json.loads(line[6:]) for line in event_response.text.splitlines() if line.startswith("data: ")]
                    assert events[-1]["kind"] == "terminal" and events[-1]["payload"]["state"] == state
                    assert sum(e["kind"] == "terminal" for e in events) == 1
                    replay = client.get(accepted["events_url"] + "?after=" + str(events[-1]["sequence"]),
                                        headers={"Last-Event-ID": str(events[0]["sequence"])})
                    assert replay.status_code == 200 and '"kind":"terminal"' in replay.text.replace(" ", "")
                    assert client.get(accepted["events_url"], headers={"Last-Event-ID": str(events[-1]["sequence"])}).status_code == 204
                    result_response = client.get(accepted["result_url"])
                    assert result_response.status_code == 200, result_response.text
                    result = result_response.json()
                    assert result["state"] == state and len(result["artifacts"]) == count
                    worker = result["result"]["worker_result"]
                    assert worker["registered_device_id"] == "research_1d"
                    assert worker["execution_catalog_sha256"] == advertised.headers["x-solarlab-registry"]
                    assert not worker["scientific_qualification"]
                    arrays = {}
                    for item in result["artifacts"]:
                        content = client.get(item["content_url"])
                        assert content.status_code == 200 and hashlib.sha256(content.content).hexdigest() == item["sha256"]
                        assert content.headers["etag"].strip('"') == item["sha256"]
                        if item["format"] == "npy":
                            arrays[item["name"]] = np.load(io.BytesIO(content.content), allow_pickle=False)
                    assert client.post("/jobs", json=body, headers=headers).json()["attempt_id"] == accepted["attempt_id"]
                    with RunStore(root, read_only=True) as store:
                        attempt, = store.attempts(body["run_id"])
                        assert len(terminal_events(store, body["run_id"])) == 1
                        directory = work / (store.store_id + "-" + attempt["attempt_id"])
                        until = time.monotonic() + 5
                        while not (directory / "receipt.json").exists():
                            assert time.monotonic() < until
                            time.sleep(.01)
                        receipt = json.loads((directory / "receipt.json").read_text())
                        assert receipt["process_joined"] and receipt["process_exit_observed"] and receipt["exitcode"] == 0
                        if body["experiment_id"] == "transient_1d":
                            loaded = read_transient_job(store, body["run_id"], attempt_id=accepted["attempt_id"])
                            assert len(arrays) == 14
                            for name, value in loaded["arrays"].items():
                                assert arrays[name].tobytes() == value.tobytes()
                            assert loaded["document"]["physical_certificate_status"] == "not_evaluated"
                            assert arrays["potential_V"].shape == (3 if state == "succeeded" else 2, 37)
                            if state == "failed":
                                assert worker["transient_result"]["reason"] == "budget_exhausted"
                                assert arrays["accepted_times_s"].tolist() == [0., 1e-7]
                                assert len(arrays["requested_times_s"]) == 4
                                assert [a["accepted"] for a in loaded["diagnostics"]["attempts"]] == [True, False]
                        else:
                            assert len(arrays) == 8 and arrays["potential_V"].shape == (3, 37)
                            assert worker["steady_result"]["metrics"]["open_circuit_voltage_V"] is None
                    evidence["cases"].append({"request": body, "accepted": accepted, "result": result,
                        "events": events, "receipt": receipt, "terminal_reconnect_status": 204,
                        "array_fingerprints": {name: {"shape": list(value.shape), "dtype": value.dtype.str,
                            "sha256": hashlib.sha256(value.tobytes()).hexdigest()} for name, value in arrays.items()}})
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
            try:
                code = process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.terminate()
                code = process.wait(timeout=5)
            evidence.update(service_exit_code=code, port_closed=port_closed(port),
                            stdout_log=str(stdout), stderr_log=str(stderr))
            save("ActualHTTP.json", evidence)
        assert code == 0 and evidence["port_closed"], stderr.read_text()
    messages = [json.loads(line) for line in stdout.read_text().splitlines() if line.startswith("{")]
    assert messages[-1] == {"service_closed": True, "read_handle_closed": True, "process_owner_closed": True}
    evidence["closure"] = messages[-1]
    save("ActualHTTP.json", evidence)


def test_transient_header_schema_scope_and_budget_rejection_has_no_enqueue(tmp_path, catalog):
    with RunStore(tmp_path / "store") as store, pool(store, tmp_path / "work", catalog) as supervisor:
        with TestClient(create_app(store.root, job_service=JobService(catalog, supervisor))) as client:
            advertised = client.get("/experiments")
            headers = {name: advertised.headers[name] for name in ("x-solarlab-registry", "x-solarlab-job-schema")}
            assert client.post("/jobs", json=example_submission()).status_code == 428
            for key in headers:
                assert client.post("/jobs", json=example_submission(), headers={**headers, key: "0" * 64}).status_code == 409
            for changes in ({"engine": "legacy"}, {"engine": "unknown"}, {"experiment_id": "ionic_transient"},
                            {"version": 2}, {"schema_version": "unknown"}, {"source_commit": "user-claim"}):
                assert client.post("/jobs", json={**example_submission(), **changes}, headers=headers).status_code == 422
            for changes in ({"device_id": "/tmp/device.json"}, {"device_id": "unknown"}, {"dimension": 2},
                {"times_s": [0., 0., 1e-7]}, {"times_s": [0., 2.]}, {"times_s": [float(i)/100 for i in range(34)]},
                {"voltage_V": [0., 0.]}, {"generation_scale": [1., -1., 1.]}, {"intervals_per_layer": 257},
                {"intervals_per_layer": True}, {"mobile_ions": True}, {"python_function": "os.system"},
                {"source_bytes": "x"}, {"config_path": "/tmp/device.json"}, {"initial_state": [0.]},
                {"controls": {"max_steps": 32, "wall_s": 30., "tolerance": 1.}}):
                body = example_submission()
                body["input"].update(changes)
                assert client.post("/jobs", json=body, headers=headers).status_code == 422
            for control, value in (("max_steps", 1), ("max_steps", 33), ("wall_s", 31.)):
                body = example_submission()
                body["input"]["controls"][control] = value
                assert client.post("/jobs", json=body, headers=headers).status_code == 422
            for field, value in (("max_iterations", 51), ("max_residual_evaluations", 1001),
                                 ("max_jacobian_evaluations", 51), ("max_backtracks", 25)):
                body = example_submission()
                body["input"]["controls"]["newton_budget"][field] = value
                assert client.post("/jobs", json=body, headers=headers).status_code == 422
            for key, value in (("wall_seconds", 61.), ("rss_bytes", 2*1024**3),
                               ("output_bytes", 128*1024**2), ("calls", 1), ("accepted_steps", 1)):
                body = example_submission()
                body["limits"][key] = value
                assert client.post("/jobs", json=body, headers=headers).status_code == 422
            assert store.events() == []


def test_coinstalled_workers_require_unambiguous_selection(catalog):
    with pytest.raises(ValueError, match="explicit experiment_id"):
        catalog.worker_binding("solarlab")
    assert catalog.worker_binding("solarlab", experiment_id="steady_jv").id == "registered_steady"
    assert catalog.worker_binding("solarlab", experiment_id="transient_1d").id == "registered_transient"
    assert {b.id for b in catalog.worker_bindings()} == {"registered_steady", "registered_transient"}


@pytest.mark.parametrize("change", ["device", "server", "model", "worker", "policy"])
def test_transient_child_rejects_inconsistent_installed_bindings(tmp_path, catalog, change):
    spec, execution = catalog.prepare(JobSubmission.model_validate(example_submission()))
    data = json.loads(catalog.document.content)
    entry = next(item for item in data["bindings"] if item["driver"] == "transient_1d")
    if change == "device":
        entry["preparation_context"]["devices"][0]["prepared_sha256"] = "0" * 64
    elif change == "server":
        entry["preparation_context"]["server_sources"][0]["sha256"] = "0" * 64
    elif change == "policy":
        entry["preparation_context"]["fixed_numerical_policy"]["transient"]["poisson_tolerance"] = 1.
    else:
        entry["model_registry_sha256" if change == "model" else "worker_source_sha256"] = "0" * 64
    changed = SourceDocument("inconsistent-transient-test-catalog", canonical(data))
    binding = replace(catalog.worker_binding("solarlab", experiment_id="transient_1d"), registry_sha256=changed.sha256)
    with RunStore(tmp_path / "store") as store, pool(store, tmp_path / "work", catalog, bindings=(binding,)) as supervisor:
        handle = supervisor.submit("invalid", binding.id, replace(spec.request, registry=changed), spec.limits, execution=execution)
        receipt = supervisor.wait(handle, 20).data
        assert receipt["store_state"] == "failed" and not receipt["terminal_artifacts"]
        assert receipt["process_joined"] and len(terminal_events(store, handle.run_id)) == 1
        assert "transient" in receipt["worker_result"]["detail"]
        assert not any(e["payload"].get("ui_progress", {}).get("stage", "").startswith("transient_")
                       for e in store.events(run_id=handle.run_id, limit=1000))


def _hold(request, context):
    """Non-numerical occupancy fixture; no model execution is claimed."""
    context.consume_manifest(request, engine="solarlab")
    context.progress({"stage": "queue_fixture_ready"})
    while True:
        context.check_cancel()
        time.sleep(.005)


def test_loopback_exact_cancel_keeps_cancelled_history_empty(tmp_path, catalog):
    root = Path(os.environ.get("SOLARLAB_TRANSIENT_CANCEL_STORE", str(tmp_path / "store"))).resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    hold = WorkerBinding.from_function("queue_fixture", _hold, registry_sha256=catalog.document.sha256,
                                      schema_sha256=catalog.schema.sha256)
    spec, execution = catalog.prepare(JobSubmission.model_validate(example_submission()))
    with RunStore(root) as store, pool(store, root.parent / "cancel-work", catalog,
            bindings=(*catalog.worker_bindings(), hold)) as supervisor:
        owner = supervisor.submit("queue-owner", hold.id, spec.request, spec.limits, execution=execution)
        until = time.monotonic() + 15
        while not any(e["payload"].get("ui_progress", {}).get("stage") == "queue_fixture_ready"
                      for e in store.events(run_id=owner.run_id, limit=1000)):
            assert time.monotonic() < until
            time.sleep(.01)
        app = create_app(root, job_service=JobService(catalog, supervisor))
        with live_http(app) as (client, port):
            headers = {"X-Solarlab-Registry": catalog.document.sha256, "X-Solarlab-Job-Schema": catalog.schema.sha256}
            accepted = client.post("/jobs", json=example_submission("cancelled-transient"), headers=headers).json()
            handle = JobHandle(store.store_id, accepted["run_id"], accepted["attempt_id"])
            row = store.get_run(handle.run_id)
            cancel_url = f"/jobs/{handle.run_id}/attempts/{handle.attempt_id}/cancel"
            assert client.post(cancel_url, json={"expected_generation": row["generation"] + 1, "reason": "stale"}).status_code == 409
            assert client.post(cancel_url, json={"expected_generation": row["generation"], "reason": "explicit cancellation"}).status_code == 200
            receipt = supervisor.wait(handle, 10).data
            result = client.get(accepted["result_url"]).json()
            assert result["state"] == "cancelled" and result["artifacts"] == []
            assert not receipt["process_started"] and len(terminal_events(store, handle.run_id)) == 1
            with pytest.raises(ValueError, match="committed"):
                read_transient_job(store, handle.run_id)
            assert store.get_run(owner.run_id)["state"] == "running"
            row = store.get_run(owner.run_id)
            assert client.post(f"/jobs/{owner.run_id}/attempts/{owner.attempt_id}/cancel",
                json={"expected_generation": row["generation"], "reason": "close fixture owner"}).status_code == 200
            closed = supervisor.wait(owner, 10).data
            assert closed["process_joined"] and closed["process_exit_observed"]
        save("CancellationHTTP.json", {"store": str(root), "cancelled_result": result, "receipt": receipt,
             "fixture_owner_receipt": closed, "port": port, "port_closed": port_closed(port),
             "reader_closed": app.state.run_store is None, "scope": "real queued HTTP cancellation; non-numerical occupancy fixture"})


def test_transient_service_main_reconstruction_does_not_import_engine(tmp_path):
    code = '''
import importlib.abc,runpy,sys
class NoEngine(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ('scipy','perovskite_sim'):
            raise AssertionError('early engine import: '+fullname)
sys.meta_path.insert(0,NoEngine())
runpy.run_path(sys.argv[1],run_name='__mp_main__')
'''
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code,
        str(ROOT / "examples/research_1d/transient_service.py")], cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
