"""One HTTP solve; historical failures are exact saved data, never recomputed."""
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


from examples.research_1d.ionic_jv_service import example_submission, installed_catalog, resource_limits
from examples.research_1d.process_ionic_transient import source_build
from solarlab.experiments.execution import JobHandle
from solarlab.experiments.jv.ionic_worker import read_ionic_jv_job
from solarlab.experiments.worker import WorkerBinding, WorkerRequest
from solarlab.io.artifacts import ArtifactPayload, _json_document
from solarlab.io.run_store import RunStore
from solarlab.materials.source import SourceDocument
from solarlab_server.app import create_app
from solarlab_server.ionic_jv_preparation import (
    IonicJVJobInput, ionic_jv_execution_catalog, prepare_ionic_jv_input, read_registered_ionic_jv_job,
)
from solarlab_server.job_service import JobService, JobSubmission, canonical
from test_ionic_transient_job_service import _hold
from test_ionic_jv_worker import extract_saved_ionic_jv_fixture
from test_transient_job_service import pool, port_closed, ready, terminal_events

pytestmark = pytest.mark.requires_clean_core

ROOT = Path(__file__).resolve().parents[2]
FAILED_RUN = "ionic-jv-failed-prefix"
FAILED_ATTEMPT = "08b9019bf98a4978b72ed612279742bd"
FIXTURE_SHA256 = "aff127aec623125b52468e6bd9e96c15f04beef86acaa45099009a27015db738"


def save(name, value):
    if directory := os.environ.get("SOLARLAB_IONIC_JV_HTTP_EVIDENCE"):
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        (target / name).write_text(json.dumps(value, indent=2, allow_nan=False))


@pytest.fixture(scope="module")
def implementation():
    commit = os.environ.get("SOLARLAB_IONIC_JV_HTTP_COMMIT") or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=5).strip()
    archive = os.environ.get("SOLARLAB_IONIC_JV_HTTP_ARCHIVE")
    return source_build(commit, Path(archive) if archive else None)


@pytest.fixture(scope="module")
def catalog(implementation):
    return installed_catalog(implementation, implementation.build_source_commit)


@pytest.fixture(scope="module")
def historical(tmp_path_factory):
    archive = Path(__file__).parent / "fixtures/ionic_jv_failed_prefix.tar.gz"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == FIXTURE_SHA256
    folder = tmp_path_factory.mktemp("ionic-jv-historical")
    with tarfile.open(archive) as bundle:
        assert sum(m.size for m in bundle.getmembers()) < 8 * 1024**2
        bundle.extractall(folder, filter="data")
    proof = json.loads((folder / "Fixture.json").read_text())
    assert proof["derived_sqlite_subset"] and not proof["whole_original_store_byte_identity"]
    assert proof["run_id"] == FAILED_RUN and proof["attempt_id"] == FAILED_ATTEMPT
    for name, sha in proof["files"].items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == sha
    assert (folder / "Store/partial").is_dir() and (folder / "Store/.writer.lock").is_file()
    return folder / "Store", proof


def publication_failure(store_root, run_id, attempt_id, result, events, *, work):
    """Preserve the original failure even if an auxiliary diagnostic read fails."""
    diagnostic = {"run_id": run_id, "attempt_id": attempt_id, "result": result, "events": events}
    outcome = result.get("result") or {}
    diagnostic["terminal_reason"] = (outcome.get("worker_result") or {}).get("reason", outcome.get("reason"))
    try:
        with RunStore(store_root, read_only=True) as store:
            run = store.get_run(run_id)
            diagnostic["run_state"] = {key: run[key] for key in (
                "run_id", "state", "current_attempt_id", "generation")}
            diagnostic["attempt_state"] = next(a for a in store.attempts(run_id) if a["attempt_id"] == attempt_id)
            if work is not None:
                directory = Path(work) / (store.store_id + "-" + attempt_id)
                receipt = directory / "receipt.json"
                # Store commit precedes the final receipt fsync. This wait is
                # only for diagnostics on an already-failed artifact assertion.
                deadline = time.monotonic() + 5
                while not receipt.exists() and time.monotonic() < deadline:
                    time.sleep(.01)
                diagnostic["receipt_path"] = str(receipt)
                diagnostic["receipt"] = json.loads(receipt.read_text()) if receipt.exists() else None
    except Exception as error:
        diagnostic["read_error"] = f"{type(error).__name__}: {error}"
    target = Path(os.environ.get("SOLARLAB_IONIC_JV_HTTP_EVIDENCE") or Path(store_root).parent)
    path = target / f"PublicationFailure-{attempt_id}.json"
    diagnostic["evidence_path"] = str(path)
    try:
        target.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(diagnostic, indent=2, allow_nan=False))
    except Exception as error:
        diagnostic["write_error"] = f"{type(error).__name__}: {error}"
    return json.dumps(diagnostic, sort_keys=True, allow_nan=False)


def observe_http_case(client, store_root, run_id, attempt_id, *, registered, work=None):
    result_url = f"/results/{run_id}/attempts/{attempt_id}"
    events_url = f"/runs/{run_id}/attempts/{attempt_id}/events"
    streamed = client.get(events_url)
    assert streamed.status_code == 200
    events = [json.loads(line[6:]) for line in streamed.text.splitlines() if line.startswith("data: ")]
    response = client.get(result_url)
    assert response.status_code == 200, response.text
    result = response.json()
    payloads, arrays = {}, {}
    assert len(result["artifacts"]) == 15, publication_failure(
        store_root, run_id, attempt_id, result, events, work=work)
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
    replay = client.get(events_url + "?after=" + str(events[-1]["sequence"]), headers={"Last-Event-ID": str(events[0]["sequence"])})
    acknowledged = client.get(events_url, headers={"Last-Event-ID": str(events[-1]["sequence"])})
    assert replay.status_code == 200 and acknowledged.status_code == 204
    assert '"kind":"terminal"' in replay.text.replace(" ", "")
    with RunStore(store_root, read_only=True) as store:
        reader = read_registered_ionic_jv_job if registered else read_ionic_jv_job
        loaded = reader(store, run_id, attempt_id=attempt_id)
        assert len(arrays) == 12 and len(loaded["arrays"]) == 23
        assert result["state"] == loaded["run_state"]
        for name, array in arrays.items():
            assert array.tobytes() == loaded["stored_arrays"][name].tobytes()
        case = {"run_id": run_id, "attempt_id": attempt_id, "registered_http": registered,
            "result": result, "events": events, "terminal_count": len(terminal_events(store, run_id)),
            "payloads": payloads, "replay_status": replay.status_code, "terminal_ack_status": acknowledged.status_code,
            "accepted_voltage_V": loaded["arrays"]["voltage_V"].tolist(), "first_failure": loaded["document"]["first_failure"],
            "logical_arrays": {k: {"sha256": hashlib.sha256(a.tobytes()).hexdigest(), "shape": list(a.shape),
                "unit": loaded["document"]["logical_array_units"][k]} for k, a in loaded["arrays"].items()}}
        assert case["terminal_count"] == 1
        if work is not None:
            directory = Path(work) / (store.store_id + "-" + attempt_id)
            deadline = time.monotonic() + 5
            while not (directory / "receipt.json").exists():
                assert time.monotonic() < deadline
                time.sleep(.01)
            receipt = json.loads((directory / "receipt.json").read_text())
            assert receipt["process_joined"] and receipt["process_exit_observed"] and receipt["exitcode"] == 0
            assert receipt["signals"] == [] and receipt["resources"]["owned_wall_seconds"] <= resource_limits().wall_seconds
            child = json.loads((directory / "terminal.json").read_text())
            for item in child["artifacts"]:
                assert hashlib.sha256((directory / item["file"]).read_bytes()).hexdigest() == payloads[item["name"]]["sha256"]
            case["receipt"] = receipt
    return case


@pytest.fixture(scope="module")
def actual(tmp_path_factory, implementation, historical):
    if saved := os.environ.get("SOLARLAB_IONIC_JV_HTTP_SAVED"):
        return json.loads(Path(saved).read_text())
    folder = tmp_path_factory.mktemp("ionic-jv-http")
    root, work = folder / "store", folder / "work"
    shutil.copytree(historical[0], root)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    assert port_closed(port)
    command = [sys.executable, "-B", "-m", "examples.research_1d.ionic_jv_service", "--store", str(root), "--work", str(work),
        "--build-source-commit", implementation.build_source_commit, "--current-repository-head", implementation.build_source_commit,
        "--port", str(port)]
    if archive := os.environ.get("SOLARLAB_IONIC_JV_HTTP_ARCHIVE"):
        command += ["--source-archive", archive]
    env = dict(os.environ)
    env.update({k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")})
    out_path, err_path = folder / "http.stdout.log", folder / "http.stderr.log"
    evidence = {"command": command, "store": str(root), "work": str(work), "port": port,
        "port_closed_before_start": True, "source_commit": implementation.build_source_commit, "cases": []}
    with out_path.open("x") as out, err_path.open("x") as err:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err, start_new_session=True)
        evidence["service_pid"] = process.pid
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=resource_limits().wall_seconds + 15) as client:
                advertised = ready(client, process)
                headers = {k: advertised.headers[k] for k in ("x-solarlab-registry", "x-solarlab-job-schema")}
                evidence.update(catalog=advertised.json(), submission_headers=headers)
                evidence["cases"].append(observe_http_case(client, root, FAILED_RUN, FAILED_ATTEMPT, registered=False))
                body = example_submission()
                response = client.post("/jobs", json=body, headers=headers)
                evidence.update(request=body, post_status=response.status_code, post=response.json())
                save("ActualHTTP.json", evidence)
                assert response.status_code == 202, response.text
                submitted = response.json()
                case = observe_http_case(client, root, submitted["run_id"], submitted["attempt_id"], registered=True, work=work)
                evidence["cases"].append(case)
                reopened = client.get("/runs/" + submitted["run_id"])
                assert reopened.status_code == 200 and reopened.json()["current_attempt_id"] == submitted["attempt_id"]
                evidence["reopened_attempt_id"] = reopened.json()["current_attempt_id"]
                evidence["new_numerical_submissions"] = 1
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
            code = process.wait(timeout=20)
            evidence.update(service_exit_code=code, port_closed=port_closed(port), stdout_log=str(out_path), stderr_log=str(err_path))
            messages = [json.loads(line) for line in out_path.read_text().splitlines() if line.startswith("{")]
            evidence["closure"] = messages[-1] if messages else None
            save("ActualHTTP.json", evidence)
    assert code == 0 and evidence["port_closed"], err_path.read_text()
    return evidence


@pytest.mark.parametrize("reason", ["wall_limit", "wall_limit_during_publication", "supervisor_control_failure"])
def test_missing_publication_preserves_terminal_diagnosis_and_still_refuses(tmp_path, monkeypatch, reason):
    evidence = tmp_path / "evidence"
    monkeypatch.setenv("SOLARLAB_IONIC_JV_HTTP_EVIDENCE", str(evidence))
    with RunStore(tmp_path / "store") as store:
        run = store.enqueue("missing-publication", opaque_identity={
            "H_input": "diagnostic-input", "H_physics": "diagnostic-physics", "H_execution": "diagnostic-execution",
        }, input_metadata={})
        token = store.claim(run["run_id"], expected_attempt_id=run["current_attempt_id"],
            expected_generation=run["generation"], owner_id="test")
        result = {"worker_result": {"reason": reason, "control_reason": "injected"},
            "resources": {"owned_wall_seconds": 30.1}}
        store.complete(token, state="failed", result=result)
        work = tmp_path / "work"
        directory = work / (store.store_id + "-" + token.attempt_id)
        directory.mkdir(parents=True)
        receipt = {"store_state": "failed", "worker_result": result["worker_result"], "terminal_committed": True}
        (directory / "receipt.json").write_text(json.dumps(receipt))
        with TestClient(create_app(store.root)) as client:
            with pytest.raises(AssertionError, match=reason) as failure:
                observe_http_case(client, store.root, run["run_id"], token.attempt_id, registered=True, work=work)
        observed = json.loads((evidence / f"PublicationFailure-{token.attempt_id}.json").read_text())
        assert observed["result"]["state"] == "failed" and observed["result"]["artifacts"] == []
        assert observed["result"]["result"] == result
        assert observed["run_state"]["state"] == observed["attempt_state"]["state"] == "failed"
        assert observed["terminal_reason"] == reason and observed["receipt"] == receipt
        assert json.dumps(observed, sort_keys=True) in str(failure.value)
        terminal = [event for event in observed["events"] if event["kind"] == "terminal"]
        assert len(terminal) == 1 and terminal[0]["payload"]["state"] == "failed"


def test_prepared_request_routes_to_existing_worker_without_changing_body(catalog):
    body = example_submission()
    spec, execution = catalog.prepare(JobSubmission.model_validate(body))
    original = prepare_ionic_jv_input(IonicJVJobInput.model_validate(body["input"]), catalog.registrations[0].context)
    assert spec.worker_id == "registered_ionic_jv" and spec.request.request == original.request.request
    assert spec.request.registry == catalog.document and spec.request.schema == catalog.schema
    execution.validate_request(spec.request)
    schema = catalog.document
    entry = json.loads(schema.content)["bindings"][0]
    assert entry["input_schema"]["properties"]["controls"]["const"] == body["input"]["controls"]
    assert entry["scientific_qualification"] is False


def test_fixed_policy_accepts_equal_json_numbers_without_replacing_input(catalog):
    original = example_submission()["input"]

    def javascript_numbers(value):
        if isinstance(value, dict):
            return {key: javascript_numbers(item) for key, item in value.items()}
        if isinstance(value, list):
            return [javascript_numbers(item) for item in value]
        return int(value) if type(value) is float and value.is_integer() else value

    supplied = javascript_numbers(original)
    assert canonical(supplied["controls"]) != canonical(original["controls"])
    parsed = IonicJVJobInput.model_validate(supplied)
    assert canonical(parsed.controls) == canonical(supplied["controls"])
    assert type(parsed.controls["steady"]["ion_logit_step"]) is int
    context = catalog.registrations[0].context
    before = prepare_ionic_jv_input(IonicJVJobInput.model_validate(original), context)
    after = prepare_ionic_jv_input(parsed, context)
    assert before.request.request == after.request.request
    assert before.request.H_input == after.request.H_input and before.request.H_physics == after.request.H_physics


@pytest.mark.parametrize("change", ["boolean", "numeric_value", "nonfinite", "missing_key", "extra_key"])
def test_fixed_policy_rejects_changed_values_or_structure(change):
    body = example_submission()["input"]
    steady = body["controls"]["steady"]
    if change == "boolean":
        steady["ion_logit_step"] = True  # True == 1.0 in Python, but is not a JSON number.
    elif change == "numeric_value":
        steady["ion_logit_step"] = 1.0000000000000002
    elif change == "nonfinite":
        steady["ion_logit_step"] = float("nan")
    elif change == "missing_key":
        del steady["ion_logit_step"]
    else:
        steady["unexpected_control"] = 1
    with pytest.raises(ValueError):
        IonicJVJobInput.model_validate(body)


def test_request_rejections_happen_before_enqueue(tmp_path, catalog):
    with RunStore(tmp_path / "store") as store, pool(store, tmp_path / "work", catalog) as owner:
        with TestClient(create_app(store.root, job_service=JobService(catalog, owner))) as client:
            headers = {"X-Solarlab-Registry": catalog.document.sha256, "X-Solarlab-Job-Schema": catalog.schema.sha256}
            assert client.post("/jobs", json=example_submission()).status_code == 428
            for key in headers:
                assert client.post("/jobs", json=example_submission(), headers={**headers, key: "0" * 64}).status_code == 409
            for update in ({"engine": "legacy"}, {"experiment_id": "ionic_transient_1d"}, {"version": 2}, {"source_commit": "claim"}):
                assert client.post("/jobs", json={**example_submission(), **update}, headers=headers).status_code == 422
            for update in ({"device_id": "absent"}, {"voltage_V": [0., .025]}, {"generation_scale": 0.},
                {"generation_scale": True}, {"incident_power_W_m2": 1000.}, {"intervals_per_layer": [13, 12, 12]},
                {"intervals_per_layer": [True, 12, 12]}, {"python_function": "os.system"}, {"config_path": "/tmp/x"},
                {"initial": {"policy": "saved_accepted", "voltage_V": .002, "wall_s": 25.}}, {"initial_state": [0.]},
                {"controls": {}}, {"ion_domain_layer": "other"}):
                body = example_submission()
                body["input"].update(update)
                assert client.post("/jobs", json=body, headers=headers).status_code == 422
            body = example_submission()
            body["input"]["controls"]["steady"]["inventory_relative_tolerance"] = 1e-3
            assert client.post("/jobs", json=body, headers=headers).status_code == 422
            body = example_submission()
            body["input"]["voltage_V"][1] = .026
            assert client.post("/jobs", json=body, headers=headers).status_code == 422
            for key, value in (("wall_seconds", resource_limits().wall_seconds + 1), ("rss_bytes", 2 * 1024**3), ("calls", 1), ("accepted_steps", 1)):
                body = example_submission()
                body["limits"][key] = value
                assert client.post("/jobs", json=body, headers=headers).status_code == 422
            assert store.events() == []


def test_registered_wall_allowance_keeps_the_ordinary_catalog_ceiling(catalog):
    registration = catalog.registrations[0]
    context = registration.context
    assert registration.wall_profile_sha256 == context.registered_wall_profile()
    assert catalog.ceiling.wall_seconds == 30.
    document = json.loads(catalog.document.content)
    assert document["resource_ceiling"]["wall_seconds"] == 30.
    assert "default_wall_ceiling_seconds" not in document
    assert context.wall_ceiling_seconds("research_1d_shared") == 30.
    assert context.wall_ceiling_seconds("research_1d_shared",
        wall_profile_sha256=registration.wall_profile_sha256) == resource_limits().wall_seconds == 128.
    assert document["bindings"][0]["whole_job_wall_allowance"] == {
        "profile_sha256": registration.wall_profile_sha256, "wall_seconds": 128.}
    spec, _ = catalog.prepare(JobSubmission.model_validate(example_submission()))
    assert spec.limits == resource_limits()
    assert spec.limits.rss_bytes == 1536 * 1024**2 and spec.limits.output_bytes == 96 * 1024**2
    assert spec.limits.accepted_steps is spec.limits.calls is None

    with pytest.raises(ValueError, match="ordinary job ceiling"):
        ionic_jv_execution_catalog(context, ceiling=resource_limits(),
                                   wall_profile_sha256=registration.wall_profile_sha256)
    ordinary_catalog = ionic_jv_execution_catalog(context, ceiling=catalog.ceiling)
    assert ordinary_catalog.registrations[0].wall_profile_sha256 is None
    assert "whole_job_wall_allowance" not in json.loads(ordinary_catalog.document.content)["bindings"][0]
    # Even a wider global service ceiling cannot grant an unapproved request
    # this device-specific allowance.
    ordinary_catalog = replace(ordinary_catalog, ceiling=resource_limits())
    body = example_submission()
    body["limits"]["wall_seconds"] = 31.
    with pytest.raises(ValueError, match="installed request wall ceiling"):
        ordinary_catalog.prepare(JobSubmission.model_validate(body))
    body["limits"]["wall_seconds"] = 30.
    assert ordinary_catalog.prepare(JobSubmission.model_validate(body))[0].limits.wall_seconds == 30.


@pytest.mark.parametrize("profile", [True, False, 128, "0" * 64, "another-profile"])
def test_registered_wall_allowance_rejects_unapproved_profiles(catalog, profile):
    with pytest.raises(ValueError, match="wall profile"):
        replace(catalog.registrations[0], wall_profile_sha256=profile)


@pytest.mark.parametrize("change", ["name", "content", "both"])
def test_wall_profile_cannot_authorize_another_prepared_device(tmp_path, catalog, change):
    from examples.research_1d.shared_site_ionic_fixture import prepare_confined_shared_site_ion_device
    registration = catalog.registrations[0]
    device_id = "other_shared" if change in {"name", "both"} else "research_1d_shared"
    prepared = (prepare_confined_shared_site_ion_device(positive_diffusivity_m2_s=2e-16)
                if change in {"content", "both"} else registration.context.devices[0][1])
    context = replace(registration.context, devices=((device_id, prepared),))
    with pytest.raises(ValueError):
        replace(registration, context=context)
    with pytest.raises(ValueError):
        ionic_jv_execution_catalog(context, ceiling=catalog.ceiling,
                                   wall_profile_sha256=registration.wall_profile_sha256)
    ordinary = replace(ionic_jv_execution_catalog(context, ceiling=catalog.ceiling), ceiling=resource_limits())
    body = example_submission()
    body["input"]["device_id"] = device_id
    body["limits"]["wall_seconds"] = 30.
    assert ordinary.prepare(JobSubmission.model_validate(body))[0].limits.wall_seconds == 30.
    body["limits"]["wall_seconds"] = 31.
    with pytest.raises(ValueError, match="installed request wall ceiling"):
        ordinary.prepare(JobSubmission.model_validate(body))
    with RunStore(tmp_path / "store") as store, pool(store, tmp_path / "work", ordinary) as owner:
        with TestClient(create_app(store.root, job_service=JobService(ordinary, owner))) as client:
            headers = {"X-Solarlab-Registry": ordinary.document.sha256, "X-Solarlab-Job-Schema": ordinary.schema.sha256}
            for wall in (31., 128.):
                body["limits"]["wall_seconds"] = wall
                assert client.post("/jobs", json=body, headers=headers).status_code == 422
            assert store.events() == []


def test_other_device_in_the_opted_in_catalog_keeps_its_default(catalog):
    from examples.research_1d.shared_site_ionic_fixture import prepare_confined_shared_site_ion_device
    registration = catalog.registrations[0]
    context = replace(registration.context, devices=(*registration.context.devices,
        ("other_shared", prepare_confined_shared_site_ion_device(positive_diffusivity_m2_s=2e-16))))
    combined = ionic_jv_execution_catalog(context, ceiling=catalog.ceiling,
                                          wall_profile_sha256=registration.wall_profile_sha256)
    body = example_submission()
    assert combined.prepare(JobSubmission.model_validate(body))[0].limits.wall_seconds == 128.
    body["input"]["device_id"] = "other_shared"
    body["limits"]["wall_seconds"] = 31.
    with pytest.raises(ValueError, match="wall_seconds"):
        combined.prepare(JobSubmission.model_validate(body))
    body["limits"]["wall_seconds"] = 30.
    assert combined.prepare(JobSubmission.model_validate(body))[0].limits.wall_seconds == 30.


def test_wall_profile_normalizes_locations_without_changing_prepared_identity(tmp_path, monkeypatch, catalog):
    from examples.research_1d.shared_site_ionic_fixture import prepare_confined_shared_site_ion_device
    from examples.research_1d.steady import load_device
    registration = catalog.registrations[0]
    source = tmp_path / "relocated-device.json"
    source.write_bytes((ROOT / "examples/research_1d/device.json").read_bytes())
    monkeypatch.setattr("examples.research_1d.ionic_multilayer.load_device", lambda: load_device(source))
    prepared = prepare_confined_shared_site_ion_device()
    original_sha = prepared.content_sha256
    assert original_sha != registration.context.devices[0][1].content_sha256
    context = replace(registration.context, devices=(("research_1d_shared", prepared),))
    assert context.registered_wall_profile() == registration.wall_profile_sha256
    relocated = ionic_jv_execution_catalog(context, ceiling=catalog.ceiling,
                                           wall_profile_sha256=registration.wall_profile_sha256)
    assert relocated.prepare(JobSubmission.model_validate(example_submission()))[0].limits.wall_seconds == 128.
    assert prepared.content_sha256 == original_sha


@pytest.fixture(scope="module")
def wall_companions(implementation):
    from ionic_jv_wall_fixtures import companions
    return companions(implementation)


@pytest.mark.parametrize("kind", ["legacy", "ion_free", "ionic_transient", "two_dimensional"])
@pytest.mark.parametrize("method", ["ionic_base", "other_base", "constructor"])
@pytest.mark.parametrize("reverse", [False, True])
def test_catalog_compositions_match_the_base_wall_oracle(catalog, wall_companions, kind, method, reverse):
    from ionic_jv_wall_fixtures import admit, compose, submissions
    oracle = json.loads((ROOT / "tests/runtime/fixtures/ionic_jv_wall_composition_baseline.json").read_text())
    assert oracle["base_commit"] == "cbec4788a0f83ae5c6c4719b1fd95c0fc037d09d"
    combined = compose(catalog, wall_companions[kind], method, reverse)
    assert combined.ceiling.wall_seconds == oracle["ordinary_ceilings"][kind]
    for key, body in submissions(kind):
        expected = oracle["admitted"][key]
        if body["experiment_id"] == "ionic_steady_jv":
            expected = True  # The sole approved request; all sampled walls are <=128.
        assert admit(combined, body) is expected, (method, reverse, key)


def test_historical_failed_fixture_retains_original_store_identity_and_http_reads(historical):
    root, proof = historical
    with RunStore(root, read_only=True) as store, TestClient(create_app(root)) as client:
        assert store.store_id == proof["source_store_id"]
        loaded = read_ionic_jv_job(store, FAILED_RUN, attempt_id=FAILED_ATTEMPT)
        assert loaded["run_state"] == "failed" and loaded["document"]["accepted_count"] == 9
        assert loaded["document"]["first_failure"]["voltage_V"] == .225
        assert loaded["worker_result"].get("execution_catalog_sha256") is None
        response = client.get(f"/results/{FAILED_RUN}/attempts/{FAILED_ATTEMPT}")
        assert response.status_code == 200 and response.json()["state"] == "failed"
        assert len(response.json()["artifacts"]) == 15
        with pytest.raises(ValueError, match="installed route"):
            read_registered_ionic_jv_job(store, FAILED_RUN)


def test_coinstalled_steady_transient_and_ionic_routes_are_unchanged(catalog, implementation):
    from examples.research_1d.ionic_transient_service import example_submission as ionic_submission, installed_catalog as ionic_catalog
    from examples.research_1d.transient_service import installed_catalog as transient_catalog
    from solarlab.experiments.registry import ExperimentRegistry
    others = [ionic_catalog(implementation, implementation.build_source_commit), transient_catalog(implementation, implementation.build_source_commit)]
    all_catalogs = [catalog, *others]
    sources = {s.id: s for c in all_catalogs for s in c.sources}
    registry = ExperimentRegistry(tuple(d for c in all_catalogs for d in c.registry.definitions))
    combined = replace(catalog, registry=registry, expected_registry_sha256=registry.content_sha256,
        sources=tuple(sources.values()), registrations=tuple(r for c in all_catalogs for r in c.registrations))
    assert {b.id for b in combined.worker_bindings()} == {"registered_ionic_jv", "registered_ionic_transient", "registered_transient", "registered_steady"}
    with pytest.raises(ValueError, match="explicit experiment_id"):
        combined.worker_binding("solarlab")
    body = example_submission()
    body.update(experiment_id="steady_jv", input={"device_id": "research_1d", "voltage_V": [0., .05, .1],
        "intervals_per_layer": 12, "generation_scale": 1., "incident_power_W_m2": None})
    body["limits"]["wall_seconds"] = 30.
    spec, _ = combined.prepare(JobSubmission.model_validate(body))
    assert spec.worker_id == "registered_steady"
    body["limits"]["wall_seconds"] = 31.
    with pytest.raises(ValueError, match="service ceiling: wall_seconds"):
        combined.prepare(JobSubmission.model_validate(body))
    other = ionic_submission()
    other["limits"]["wall_seconds"] = 31.
    with pytest.raises(ValueError, match="service ceiling: wall_seconds"):
        combined.prepare(JobSubmission.model_validate(other))
    # The ion-free service has an existing 60 s policy; it is unchanged too.
    assert others[0].ceiling.wall_seconds == 30. and others[1].ceiling.wall_seconds == 60.


@pytest.mark.parametrize("change", ["server", "model", "policy", "wall", "wall_profile", "wall_type"])
def test_child_refuses_changed_registration_before_numerical_work(tmp_path, catalog, change):
    spec, execution = catalog.prepare(JobSubmission.model_validate(example_submission()))
    data = json.loads(catalog.document.content)
    entry = data["bindings"][0]
    if change == "server":
        entry["preparation_context"]["server_sources"][0]["sha256"] = "0" * 64
    elif change == "model":
        entry["model_registry_sha256"] = "0" * 64
    elif change == "policy":
        entry["preparation_context"]["fixed_numerical_policy"]["controls"]["wall_s"] = 1.
    elif change == "wall":
        entry["whole_job_wall_allowance"]["wall_seconds"] = resource_limits().wall_seconds + 1
    elif change == "wall_profile":
        entry["whole_job_wall_allowance"]["profile_sha256"] = "0" * 64
    else:
        entry["whole_job_wall_allowance"]["wall_seconds"] = True
    document = SourceDocument("changed-catalog", canonical(data))
    binding = replace(catalog.worker_binding("solarlab"), registry_sha256=document.sha256)
    with RunStore(tmp_path / "store") as store, pool(store, tmp_path / "work", catalog, bindings=(binding,)) as owner:
        handle = owner.submit("invalid", binding.id, replace(spec.request, registry=document), spec.limits, execution=execution)
        receipt = owner.wait(handle, 15).data
        assert receipt["store_state"] == "failed" and receipt["terminal_artifacts"] == []
        assert receipt["process_joined"] and receipt["process_exit_observed"]
        assert not any(e["payload"].get("ui_progress", {}).get("stage", "").startswith("ionic_jv") for e in store.events(run_id="invalid", limit=1000))


@pytest.mark.parametrize("change", ["name", "content", "absent_profile"])
def test_child_rechecks_profile_authorization_before_numerical_work(tmp_path, catalog, change):
    from examples.research_1d.shared_site_ionic_fixture import prepare_confined_shared_site_ion_device
    registration = catalog.registrations[0]
    device_id = "other_shared" if change == "name" else "research_1d_shared"
    prepared = (prepare_confined_shared_site_ion_device(positive_diffusivity_m2_s=2e-16)
                if change == "content" else registration.context.devices[0][1])
    context = replace(registration.context, devices=((device_id, prepared),))
    ordinary = ionic_jv_execution_catalog(context, ceiling=catalog.ceiling)
    body = example_submission()
    body["input"]["device_id"] = device_id
    body["limits"]["wall_seconds"] = 30.
    spec, execution = ordinary.prepare(JobSubmission.model_validate(body))
    data = json.loads(ordinary.document.content)
    if change != "absent_profile":
        data["bindings"][0]["whole_job_wall_allowance"] = {
            "profile_sha256": registration.wall_profile_sha256, "wall_seconds": 128.}
    document = SourceDocument("unapproved-wall-catalog", canonical(data))
    binding = replace(ordinary.worker_binding("solarlab"), registry_sha256=document.sha256)
    with RunStore(tmp_path / "store") as store, pool(store, tmp_path / "work", ordinary, bindings=(binding,)) as owner:
        handle = owner.submit("unapproved", binding.id, replace(spec.request, registry=document),
                              resource_limits(), execution=execution)
        receipt = owner.wait(handle, 15).data
        assert receipt["store_state"] == "failed" and receipt["terminal_artifacts"] == []
        assert receipt["process_joined"] and receipt["process_exit_observed"]
        assert not any(e["payload"].get("ui_progress", {}).get("stage", "").startswith("ionic_jv")
                       for e in store.events(run_id="unapproved", limit=1000))
        directory = tmp_path / "work" / (store.store_id + "-" + handle.attempt_id)
        result = json.loads((directory / "terminal.json").read_text())["result"]
        assert result["reason"] == "worker_exception" and result["exception_type"] == "ValueError"
        expected = "approved prepared device and recipe" if change == "content" else "wall<=30s"
        assert expected in result["detail"]
        save("ChildProfile-" + change + ".json", {"result": result, "receipt": receipt})


def test_registered_child_rejects_a_different_v1_recipe(tmp_path, catalog):
    from solarlab.config.identity import ExecutionEngine
    from solarlab.experiments.jv.ionic_worker import IonicJVProtocol, prepare_ionic_jv_job
    from solarlab_server.ionic_jv_preparation import _initial, _options
    registration = catalog.registrations[0]
    context = registration.context
    declared = IonicJVJobInput.model_validate(example_submission()["input"])
    job = prepare_ionic_jv_job(context.device(declared.device_id),
        IonicJVProtocol(tuple(declared.voltage_V), 2., declared.incident_power_W_m2),
        intervals_per_layer=tuple(declared.intervals_per_layer), ion_domain_layer=context.ion_domain_layer,
        ion_domain_endpoints=("blocking", "blocking"), options=_options(), initial=_initial(),
        implementation=context.implementation, current_repository_head=context.current_repository_head,
        engine=ExecutionEngine("solarlab"))
    request = replace(job.request, registry=catalog.document, schema=catalog.schema)
    with RunStore(tmp_path / "store") as store, pool(store, tmp_path / "work", catalog) as owner:
        handle = owner.submit("wrong-recipe", catalog.worker_binding("solarlab").id, request,
                              resource_limits(), execution=job.execution)
        receipt = owner.wait(handle, 15).data
        assert receipt["store_state"] == "failed" and receipt["terminal_artifacts"] == []
        assert receipt["process_joined"] and receipt["process_exit_observed"]
        directory = tmp_path / "work" / (store.store_id + "-" + handle.attempt_id)
        result = json.loads((directory / "terminal.json").read_text())["result"]
        assert result["reason"] == "worker_exception" and result["exception_type"] == "ValidationError"
        assert "generation_scale" in result["detail"]
        assert not any(e["payload"].get("ui_progress", {}).get("stage", "").startswith("ionic_jv")
                       for e in store.events(run_id="wrong-recipe", limit=1000))
        save("ChildRecipe.json", {"result": result, "receipt": receipt})


def test_direct_submit_and_raw_binding_keep_the_30_second_ceiling(tmp_path, catalog):
    from solarlab.experiments.jv.ionic_worker import ionic_jv_worker_binding
    job = prepare_ionic_jv_input(IonicJVJobInput.model_validate(example_submission()["input"]),
                                 catalog.registrations[0].context)
    for wall in (31., 128.):
        with pytest.raises(ValueError, match="wall<=30s"):
            job.submit(None, "direct-too-large", replace(resource_limits(), wall_seconds=wall))
    binding = ionic_jv_worker_binding()
    with RunStore(tmp_path / "store") as store, pool(store, tmp_path / "work", catalog, bindings=(binding,)) as owner:
        handle = owner.submit("raw-too-large", binding.id, job.request, resource_limits(), execution=job.execution)
        receipt = owner.wait(handle, 15).data
        assert receipt["store_state"] == "failed" and receipt["terminal_artifacts"] == []
        assert receipt["process_joined"] and receipt["process_exit_observed"]
        directory = tmp_path / "work" / (store.store_id + "-" + handle.attempt_id)
        result = json.loads((directory / "terminal.json").read_text())["result"]
        assert result["reason"] == "worker_exception" and result["exception_type"] == "ValueError"
        assert "wall<=30s" in result["detail"]
        assert not any(e["payload"].get("ui_progress", {}).get("stage", "").startswith("ionic_jv")
                       for e in store.events(run_id="raw-too-large", limit=1000))
        save("RawBinding.json", {"result": result, "receipt": receipt})


def test_existing_exact_queue_cancellation_has_no_ionic_curve(tmp_path, catalog):
    spec, execution = catalog.prepare(JobSubmission.model_validate(example_submission()))
    hold = WorkerBinding.from_function("queue_fixture", _hold, registry_sha256=catalog.document.sha256, schema_sha256=catalog.schema.sha256)
    with RunStore(tmp_path / "store") as store, pool(store, tmp_path / "work", catalog, bindings=(*catalog.worker_bindings(), hold)) as owner:
        blocker = owner.submit("blocker", hold.id, spec.request, spec.limits, execution=execution)
        deadline = time.monotonic() + 10
        while not any(e["payload"].get("ui_progress", {}).get("stage") == "queue_fixture_ready" for e in store.events(run_id="blocker", limit=1000)):
            assert time.monotonic() < deadline
            time.sleep(.01)
        with TestClient(create_app(store.root, job_service=JobService(catalog, owner))) as client:
            headers = {"X-Solarlab-Registry": catalog.document.sha256, "X-Solarlab-Job-Schema": catalog.schema.sha256}
            response = client.post("/jobs", json=example_submission("queued-jv"), headers=headers)
            assert response.status_code == 202
            selected = response.json()
            row = store.get_run("queued-jv")
            url = f"/jobs/queued-jv/attempts/{row['current_attempt_id']}/cancel"
            assert client.post(url, json={"expected_generation": row["generation"] + 1, "reason": "stale"}).status_code == 409
            assert client.post(url, json={"expected_generation": row["generation"], "reason": "cancel queued"}).status_code == 200
            receipt = owner.wait(JobHandle(store.store_id, "queued-jv", row["current_attempt_id"]), 5).data
            result = client.get(selected["result_url"]).json()
            assert result["state"] == "cancelled" and result["artifacts"] == [] and not receipt["process_started"]
            assert len(terminal_events(store, "queued-jv")) == 1
            assert owner.cancel(blocker)
            closed = owner.wait(blocker, 5).data
            assert closed["process_joined"] and closed["process_exit_observed"]
            save("QueuedCancellation.json", {"receipt": receipt, "hold_receipt": closed, "new_numerical_jobs": 0})


def test_actual_fixed51_http_outcome_and_historical_failure_truth(actual):
    """A fixed25s wave may preserve a budget prefix on a slower CI host."""
    assert actual["new_numerical_submissions"] == 1 and actual["service_exit_code"] == 0 and actual["port_closed"]
    assert actual["closure"] == {"service_closed": True, "read_handle_closed": True, "process_owner_closed": True}
    failed, current = actual["cases"]
    assert failed["run_id"] == FAILED_RUN and failed["attempt_id"] == FAILED_ATTEMPT
    assert failed["result"]["state"] == "failed" and len(failed["accepted_voltage_V"]) == 9
    assert failed["first_failure"]["voltage_V"] == .225 and not failed["registered_http"]
    assert failed["result"]["result"]["worker_result"].get("execution_catalog_sha256") is None
    worker = current["result"]["result"]["worker_result"]
    assert worker["execution_catalog_sha256"] == actual["submission_headers"]["x-solarlab-registry"]
    with RunStore(Path(actual["store"]), read_only=True) as store:
        loaded = read_registered_ionic_jv_job(store, current["run_id"], attempt_id=current["attempt_id"])
    summary, raw = loaded["document"], loaded["diagnostics"]
    requested = [tick / 40. for tick in range(51)]
    count = summary["accepted_count"]
    assert loaded["arrays"]["requested_voltage_V"].tolist() == requested
    assert loaded["arrays"]["voltage_V"].tolist() == requested[:count] == current["accepted_voltage_V"]
    assert raw["controls"]["wall_s"] == 25. and raw["controls"]["max_points"] == 51
    assert raw["initial_recipe"]["policy"] == "dark_direct_density_seed_v1"
    attempts = raw["curve"]["attempts"]
    assert all(point["index"] == i and point["voltage_V"] == requested[i]
               and point["accepted"] == (i < count) for i, point in enumerate(attempts))
    counts = summary["computation_counts"]
    assert counts["cold_steady"]["solver_calls"] == counts["illumination"]["coupled_solver_calls"] == 1
    assert counts["illumination"]["component_solver_calls"] == 0 and counts["illumination"]["density_preparation_calls"] == 1
    if current["result"]["state"] == "succeeded":
        assert count == len(attempts) == 51 and summary["outcome"] == "completed"
        assert summary["first_failure"] is None and counts["curve"]["solver_calls"] == 51
    else:
        # Only this diagnosed computation limit is an admissible partial result.
        # Numerical errors, initialization failures and unknown causes still fail.
        assert current["result"]["state"] == "failed" and summary["outcome"] == "partial"
        assert worker["reason"] == summary["reason"] == "wall_budget_exhausted", summary
        assert 0 < count < 51 and len(attempts) == count + 1
        first = summary["first_failure"]
        assert first["index"] == count and first["voltage_V"] == requested[count]
        assert first["termination"] == "wall_budget_exhausted" and attempts[-1]["termination"] == "wall_budget_exhausted"
        assert first["solver_called"] in (True, False)
        assert counts["curve"]["solver_calls"] == count + int(first["solver_called"])
        assert len(requested[len(attempts):]) == 51 - count - 1
    assert counts["total_solver_calls"] == counts["curve"]["solver_calls"] + 2
    assert worker["ionic_jv_result"]["physical_certificate_status"] == "not_evaluated"
    assert worker["ionic_jv_result"]["metrics"]["efficiency"] is None
    assert worker["scientific_qualification"] is False
    assert current["attempt_id"] == actual["reopened_attempt_id"]
    receipt = current["receipt"]
    assert receipt["proposed_state"] == receipt["store_state"] == current["result"]["state"]
    assert receipt["commit_error"] is None and receipt["publication_error"] is None
    assert receipt["resources"]["observed_limit_violations"] == [] and receipt["resources"]["owned_wall_seconds"] <= resource_limits().wall_seconds
    assert all(c["terminal_count"] == 1 and len(c["payloads"]) == 15 and len(c["logical_arrays"]) == 23 for c in actual["cases"])


def test_saved_full51_http_success_remains_readable(tmp_path_factory):
    """A genuine complete HTTP result stays covered independent of host speed."""
    folder, proof = extract_saved_ionic_jv_fixture(tmp_path_factory)
    original, = [case for case in proof["cases"] if case["registered_http"]]
    root = folder / "Store"
    with TestClient(create_app(root)) as client:
        observed = observe_http_case(client, root, original["run_id"], original["attempt_id"], registered=True)
    assert observed["result"]["state"] == original["state"] == "succeeded"
    assert len(observed["accepted_voltage_V"]) == original["accepted_count"] == 51 and observed["first_failure"] is None
    worker = observed["result"]["result"]["worker_result"]
    assert worker["capture"]["result_sha256"] == original["result_sha256"]
    assert worker["manifest_sha256"] == original["manifest_sha256"]
    assert worker["ionic_jv_result"]["computation_counts"]["total_solver_calls"] == 53
    assert worker["scientific_qualification"] is False and worker["ionic_jv_result"]["metrics"]["efficiency"] is None
    for name, actual_array in observed["logical_arrays"].items():
        assert all(actual_array[key] == original["logical_arrays"][name][key] for key in ("sha256", "shape", "unit"))


def test_registered_and_historical_http_readers_forbid_numerical_imports(actual, tmp_path):
    code = '''
import hashlib,importlib.abc,json,pathlib,sys
sys.path.insert(0,sys.argv[1])
class NoSolver(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if any(fullname==n or fullname.startswith(n+'.') for n in ('solarlab.solvers','solarlab.equations',
            'solarlab.device.compile_1d','solarlab.device.compile_ionic_1d','solarlab.experiments.ionic_transient',
            'solarlab.experiments.ionic_initialization','solarlab.experiments.jv.ionic_steady','perovskite_sim')):
            raise AssertionError('saved reader imported solver: '+fullname)
sys.meta_path.insert(0,NoSolver())
from fastapi.testclient import TestClient
from solarlab.io.run_store import RunStore
from solarlab.experiments.jv.ionic_worker import read_ionic_jv_job
from solarlab_server.ionic_jv_preparation import read_registered_ionic_jv_job
from solarlab_server.app import create_app
root=pathlib.Path(sys.argv[2])
with RunStore(root,read_only=True) as store,TestClient(create_app(root)) as client:
    for name,registered in json.loads(sys.argv[3]):
        reader=read_registered_ionic_jv_job if registered else read_ionic_jv_job
        loaded=reader(store,name)
        attempt=loaded['document']['attempt_id']
        assert client.get('/results/'+name+'/attempts/'+attempt).json()['state']==loaded['run_state']
        print(json.dumps({'run_id':name,'arrays':{k:hashlib.sha256(a.tobytes()).hexdigest() for k,a in loaded['arrays'].items()}}))
'''
    names = [[c["run_id"], c["registered_http"]] for c in actual["cases"]]
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(ROOT / "src"), actual["store"], json.dumps(names)],
        cwd=tmp_path, capture_output=True, text=True, timeout=25)
    assert result.returncode == 0, result.stderr
    for line, case in zip(result.stdout.splitlines(), actual["cases"], strict=True):
        assert json.loads(line)["arrays"] == {k: v["sha256"] for k, v in case["logical_arrays"].items()}
    save("SolverFreeRead.json", {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr, "new_numerical_jobs": 0})


@pytest.mark.parametrize("change", ["model", "implementation", "context_head"])
def test_coherent_catalog_claims_cannot_override_computation_manifest(actual, tmp_path, change):
    """Alter both catalog/wrapper claims; the original computation stays pinned."""
    success = actual["cases"][1]
    with RunStore(Path(actual["store"]), read_only=True) as original:
        run = original.get_run(success["run_id"])
        attempt = next(a for a in original.attempts(success["run_id"]) if a["attempt_id"] == success["attempt_id"])
        result = attempt["result"]
        objects = [(original.artifact(r["artifact_id"]), original.read_artifact(r["artifact_id"])) for r in attempt["terminal"]["artifacts"]]
    metadata = run["request"]["input_metadata"]
    transport = WorkerRequest.from_bytes(canonical(metadata["transport"]))
    catalog = json.loads(transport.registry.content)
    entry = next(e for e in catalog["bindings"] if e["driver"] == "ionic_steady_jv")
    if change == "model":
        entry["model_registry_sha256"] = "0" * 64
        result["worker_result"]["registered_model_registry_sha256"] = "0" * 64
    elif change == "implementation":
        entry["preparation_context"]["implementation"]["build_source_commit"] = "0" * 40
    else:
        entry["preparation_context"]["current_repository_head"] = "0" * 40
    document = SourceDocument(transport.registry.id, canonical(catalog))
    transport = replace(transport, registry=document)
    metadata["transport"] = transport.to_document()
    result["worker_result"]["execution_catalog_sha256"] = document.sha256
    with RunStore(tmp_path / "tampered") as store:
        row = store.enqueue("tampered", opaque_identity=run["request"]["identity"], input_metadata=metadata)
        token = store.claim("tampered", expected_attempt_id=row["current_attempt_id"], expected_generation=row["generation"], owner_id="read-fixture")
        ids = [store.publish(token, item["name"], ArtifactPayload(raw, item["format"], _json_document(item["metadata"])))["artifact_id"]
               for item, raw in objects]
        store.complete(token, state="succeeded", result=result, artifacts=ids)
        with pytest.raises(ValueError):
            read_registered_ionic_jv_job(store, "tampered")
