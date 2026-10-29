"""One real shared-site pair; reuse its committed data for read/lifecycle checks."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest


from examples.research_1d.process_ionic_transient import example_job, source_build
from solarlab.experiments.ionic_transient_worker import (
    ionic_transient_worker_binding, read_ionic_transient_job,
)
from solarlab.experiments.execution import ProcessSupervisor
from solarlab.experiments.transient_results import _decode, _encode
from solarlab.experiments.worker import ResourceLimits, WorkerBinding, WorkerRequest
from solarlab.io.artifacts import ArtifactIntegrityError, ArtifactPayload, _json_document
from solarlab.io.run_store import RunStore
from solarlab.materials.source import SourceDocument

pytestmark = pytest.mark.requires_clean_core

ROOT = Path(__file__).resolve().parents[2]
NAMES = ("ionic-transient-completed", "ionic-transient-failed-prefix")


def limits():
    return ResourceLimits(30., 1536 * 1024**2, 96 * 1024**2, None, None)


def pool(store, work, bindings=None):
    return ProcessSupervisor(store, work, bindings or (ionic_transient_worker_binding(),),
        max_active=1, max_pending=1, shared_rss_bytes=3 * 1024**3, shared_output_bytes=384 * 1024**2,
        poll_seconds=.01, soft_grace_seconds=.1, terminate_grace_seconds=.1)


def evidence(name, value):
    if target := os.environ.get("SOLARLAB_IONIC_PROCESS_EVIDENCE"):
        folder = Path(target)
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / name).open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)


@pytest.fixture(scope="module")
def implementation():
    commit = os.environ.get("SOLARLAB_IONIC_PROCESS_COMMIT") or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=5).strip()
    archive = os.environ.get("SOLARLAB_IONIC_PROCESS_ARCHIVE")
    return source_build(commit, Path(archive) if archive else None)


@pytest.fixture(scope="module")
def job(implementation):
    return example_job(implementation, current_repository_head=implementation.build_source_commit)


@pytest.fixture(scope="module")
def saved(tmp_path_factory, implementation):
    if path := os.environ.get("SOLARLAB_IONIC_PROCESS_STORE"):
        root = Path(path).resolve()
        work = Path(os.environ.get("SOLARLAB_IONIC_PROCESS_WORK", str(root.parent / "work")))
        return root, work
    root = tmp_path_factory.mktemp("ionic-transient")
    with RunStore(root / "store") as store, pool(store, root / "work") as supervisor:
        for name, failure in zip(NAMES, (False, True), strict=True):
            prepared = example_job(implementation, failure=failure,
                                   current_repository_head=implementation.build_source_commit)
            handle = prepared.submit(supervisor, name, limits())
            receipt = supervisor.wait(handle, 40).data
            assert receipt["store_state"] == ("failed" if failure else "succeeded"), receipt
            assert receipt["process_joined"] and receipt["process_exit_observed"]
    return root / "store", root / "work"


@pytest.mark.parametrize("name,count,state", [(NAMES[0], 5, "succeeded"), (NAMES[1], 2, "failed")])
def test_actual_cold_histories_source_payload_words_and_single_terminal(saved, name, count, state):
    root, work = saved
    with RunStore(root, read_only=True) as store:
        loaded = read_ionic_transient_job(store, name)
        attempt, = store.attempts(name)
        directory = work / (store.store_id + "-" + attempt["attempt_id"])
        receipt = json.loads((directory / "receipt.json").read_text())
        assert loaded["run_state"] == receipt["store_state"] == state and receipt["exitcode"] == 0
        assert receipt["process_joined"] and receipt["process_exit_observed"] and receipt["terminal_committed"]
        with pytest.raises(ChildProcessError):
            os.waitpid(receipt["pid"], os.WNOHANG)
        terminal = [event for event in store.events(run_id=name, limit=1000) if event["kind"] == "terminal"]
        assert len(terminal) == 1 and len(attempt["terminal"]["artifacts"]) == 16
        assert len(loaded["stored_arrays"]) == 13 and len(loaded["arrays"]) == 26
        assert loaded["arrays"]["potential_V"].shape == (count, 37)
        assert loaded["arrays"]["state_coordinates"].shape == (count, 137)
        assert loaded["arrays"]["positive_ion_density_m3"].shape == (count, 13)
        assert loaded["arrays"]["negative_ion_density_m3"].shape == (count, 13)
        assert loaded["arrays"]["total_current_density_A_m2"].shape == (count - 1, 38)
        assert loaded["arrays"]["terminal_total_current_density_A_m2"].shape == (count - 1, 2)
        assert loaded["arrays"]["accepted_times_s"].tolist() == [0., .25, .5, .75, 1.][:count]
        assert loaded["arrays"]["requested_times_s"].tolist() == [0., .25, .5, .75, 1.]
        assert loaded["document"]["ion_site_contract"] == "shared_finite_sites"
        assert loaded["document"]["physical_certificate_status"] == "not_evaluated"
        assert not loaded["worker_result"]["scientific_qualification"] and not loaded["manifest"]["can_execute"]
        assert receipt["resources"]["calls"] is None and receipt["resources"]["accepted_steps"] is None
        history = loaded["diagnostics"]["history"]
        if state == "failed":
            assert loaded["document"]["reason"] == "budget_exhausted"
            assert [a["accepted"] for a in history["attempts"]] == [True, False]
            assert history["attempts"][-1]["end_s"] == .5
        child = json.loads((directory / "terminal.json").read_text())
        by_name = {store.artifact(ref["artifact_id"])["name"]: store.artifact(ref["artifact_id"])
                   for ref in attempt["terminal"]["artifacts"]}
        for original in child["artifacts"]:
            item = by_name[original["name"]]
            raw = (directory / original["file"]).read_bytes()
            assert raw == store.read_artifact(item["artifact_id"])
            assert hashlib.sha256(raw).hexdigest() == original["sha256"] == item["sha256"]
        with pytest.raises(ValueError, match="committed"):
            read_ionic_transient_job(store, name, attempt_id="other-attempt")


def test_fresh_reader_forbids_all_solver_compiler_and_equation_imports(saved, tmp_path):
    code = '''
import hashlib, importlib.abc, json, pathlib, sys
sys.path.insert(0, sys.argv[1])
class NoSolver(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in (
            'solarlab.solvers', 'solarlab.equations', 'solarlab.device.compile_1d',
            'solarlab.device.compile_ionic_1d', 'solarlab.experiments.ionic_transient',
            'solarlab.experiments.transient', 'perovskite_sim')):
            raise AssertionError('solver-free reader imported: ' + fullname)
sys.meta_path.insert(0, NoSolver())
from solarlab.experiments.ionic_transient_worker import read_ionic_transient_job
from solarlab.io.run_store import RunStore
with RunStore(pathlib.Path(sys.argv[2]), read_only=True) as store:
    for name in ('ionic-transient-completed', 'ionic-transient-failed-prefix'):
        result = read_ionic_transient_job(store, name)
        assert len(result['arrays']) == 26
        print(json.dumps({'run_id': name, 'state': result['run_state'], 'arrays': {
            key: hashlib.sha256(a.tobytes()).hexdigest() for key, a in result['arrays'].items()}}))
'''
    check = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(ROOT / "src"), str(saved[0])],
                           cwd=tmp_path, capture_output=True, text=True, timeout=20)
    assert check.returncode == 0, check.stderr
    with RunStore(saved[0], read_only=True) as store:
        for line in check.stdout.splitlines():
            document = json.loads(line)
            expected = read_ionic_transient_job(store, document["run_id"])
            assert document["arrays"] == {name: hashlib.sha256(a.tobytes()).hexdigest() for name, a in expected["arrays"].items()}
    evidence("SolverFreeRead.json", {"argv": check.args, "exit_code": check.returncode, "stdout": check.stdout, "stderr": check.stderr})


def reseal(saved, target, change):
    with RunStore(saved[0], read_only=True) as original:
        run = original.get_run(NAMES[1])
        attempt, = original.attempts(NAMES[1])
        result = attempt["result"]
        items = [(original.artifact(ref["artifact_id"]), original.read_artifact(ref["artifact_id"]))
                 for ref in attempt["terminal"]["artifacts"]]
    with RunStore(target) as store:
        row = store.enqueue("fixture", opaque_identity=run["request"]["identity"], input_metadata=run["request"]["input_metadata"])
        token = store.claim("fixture", expected_attempt_id=row["current_attempt_id"], expected_generation=row["generation"], owner_id="reader-fixture")
        ids = []
        for item, raw in items:
            if change == "unit" and item["name"] == "potential_V":
                item["metadata"]["unit"] = "wrong"
            if change == "component-order" and item["name"] == "carrier_density_m3":
                item["metadata"]["components"].reverse()
            if change in {"nonfinite-accepted-accounting", "protocol"} and item["name"] == "ionic_transient_diagnostics":
                data = _decode(json.loads(raw))
                if change == "protocol":
                    data["history"]["protocol"]["voltage_V"] = data["history"]["protocol"]["voltage_V"].copy()
                    data["history"]["protocol"]["voltage_V"][1] += .001
                else:
                    data["history"]["attempts"][0]["accounting"]["maxwell_spread_A_m2"] = float("nan")
                raw = _json_document(_encode(data)).encode()
            payload = ArtifactPayload(raw, item["format"], _json_document(item["metadata"]))
            if item["name"] != "execution_manifest":
                result["worker_result"]["payloads"][item["name"]] = {
                    "sha256": payload.sha256, "format": payload.format, "metadata": item["metadata"]}
            record = store.publish(token, item["name"], payload)
            ids.append(record["artifact_id"])
        for key, label in (("initial_artifact_sha256", "initial-binding"), ("request_sha256", "request-binding"),
                           ("result_sha256", "result-binding")):
            if change == label:
                result["worker_result"]["capture"][key] = "0" * 64
        if change == "source-binding":
            result["worker_result"]["capture"]["source_sha256"]["solarlab/experiments/ionic_transient.py"] = "0" * 64
        store.complete(token, state="succeeded" if change == "false-success" else "failed", result=result, artifacts=ids)
        if change == "terminal-reference":
            terminal = store.attempts("fixture")[0]["terminal"]
            terminal["artifacts"][0]["manifest_sha256"] = "0" * 64
            store._db.execute("UPDATE attempts SET terminal_json=? WHERE attempt_id=?", (_json_document(terminal), token.attempt_id))
        if change == "corrupt-bytes":
            path = target / store.artifact(ids[0])["relative_path"]
            raw = bytearray(path.read_bytes())
            raw[-1] ^= 1
            path.write_bytes(raw)


@pytest.mark.parametrize("change", ["unit", "component-order", "initial-binding", "request-binding", "result-binding",
    "source-binding", "false-success", "terminal-reference", "corrupt-bytes", "protocol", "nonfinite-accepted-accounting"])
def test_resealed_or_corrupt_history_refuses_false_payloads(saved, tmp_path, change):
    reseal(saved, tmp_path / "store", change)
    with RunStore(tmp_path / "store", read_only=True) as store, pytest.raises((ValueError, ArtifactIntegrityError)):
        read_ionic_transient_job(store, "fixture")


def test_lossless_failed_diagnostic_words_do_not_become_finite_or_unknown():
    words = np.asarray([np.nan, np.inf, -np.inf, -0.0])
    decoded = _decode(json.loads(_json_document(_encode({"failed_candidate": words, "missing": None}))))
    assert decoded["failed_candidate"].tobytes() == words.tobytes() and decoded["missing"] is None


def _hold_slot(request, context):
    """Queue ownership fixture only; not a computation witness."""
    context.progress({"stage": "hold_ready"})
    while True:
        context.check_cancel()
        time.sleep(.005)


def test_exact_queued_cancel_creates_no_ionic_history(tmp_path, job):
    data = SourceDocument("control-fixture", b'{"hold":true}')
    request = WorkerRequest(data, data, data, data, data.sha256, data.sha256)
    held = WorkerBinding.from_function("ionic_queue_fixture", _hold_slot, registry_sha256=data.sha256, schema_sha256=data.sha256)
    with RunStore(tmp_path / "store") as store, pool(store, tmp_path / "work", (held, ionic_transient_worker_binding())) as owner:
        blocker = owner.submit("blocker", held.id, request, limits())
        deadline = time.monotonic() + 8
        while not any(event["payload"].get("ui_progress", {}).get("stage") == "hold_ready" for event in store.events(run_id="blocker", limit=1000)):
            assert time.monotonic() < deadline
            time.sleep(.01)
        queued = job.submit(owner, "queued-ionic", limits())
        assert store.get_run(queued.run_id)["state"] == "queued"
        assert owner.cancel(queued)
        receipt = owner.wait(queued, 5).data
        attempt, = store.attempts(queued.run_id)
        assert receipt["store_state"] == "cancelled" and not receipt["process_started"]
        assert attempt["terminal"]["artifacts"] == []
        assert len([e for e in store.events(run_id=queued.run_id, limit=1000) if e["kind"] == "terminal"]) == 1
        assert not owner.cancel(queued)
        with pytest.raises(ValueError, match="committed"):
            read_ionic_transient_job(store, queued.run_id)
        assert owner.cancel(blocker)
        hold_receipt = owner.wait(blocker, 5).data
        assert hold_receipt["process_joined"] and hold_receipt["process_exit_observed"]
        evidence("QueuedCancellation.json", {"queued": receipt, "control_fixture": hold_receipt,
            "new_ionic_numerical_processes": 0, "terminal_count": 1})


def test_bad_request_and_unobserved_caps_refuse_before_native(tmp_path, job):
    with RunStore(tmp_path / "store") as store, pool(store, tmp_path / "work") as owner:
        for name in ("calls", "accepted_steps"):
            with pytest.raises(ValueError, match="counter budgets"):
                job.submit(owner, name, replace(limits(), **{name: 1}))
        with pytest.raises(ValueError, match="wall ceiling"):
            job.submit(owner, "long", replace(limits(), wall_seconds=31))
        body = json.loads(job.request.request.content)
        body["protocol"]["voltage_V"][1] += .001
        changed = replace(job.request, request=SourceDocument(job.request.request.id, _json_document(body).encode()))
        handle = owner.submit("invalid", "ionic_transient_1d", changed, limits(), execution=job.execution)
        receipt = owner.wait(handle, 15).data
        assert receipt["store_state"] == "failed" and not receipt["terminal_artifacts"]
        assert "differ" in receipt["worker_result"]["detail"]
        assert not any(e["payload"].get("ui_progress", {}).get("stage", "").startswith("ionic_")
                       for e in store.events(run_id=handle.run_id, limit=1000))
