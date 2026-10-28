"""Real cold transient jobs; reuse their saved histories for reader regressions."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from examples.research_1d.process_transient import example_job, source_build
from solarlab.experiments.execution import ProcessSupervisor
from solarlab.experiments.transient_worker import read_transient_job, transient_worker_binding
from solarlab.experiments.worker import ResourceLimits, WorkerBinding, WorkerRequest, WorkerResult
from solarlab.io.artifacts import ArtifactIntegrityError, ArtifactPayload, _json_document, bytes_artifact
from solarlab.io.run_store import RunStore
from solarlab.materials.source import SourceDocument

ROOT = Path(__file__).resolve().parents[2]


def limits():
    return ResourceLimits(60., 1536 * 1024**2, 96 * 1024**2, None, None)


def pool(store, work, bindings=None, **changes):
    settings = {"max_active": 1, "max_pending": 1, "shared_rss_bytes": 3 * 1024**3,
        "shared_output_bytes": 384 * 1024**2, "soft_grace_seconds": .3,
        "terminate_grace_seconds": .1, "poll_seconds": .01}
    settings.update(changes)
    return ProcessSupervisor(store, work, bindings or (transient_worker_binding(),), **settings)


def terminals(store, run_id):
    return [e for e in store.events(run_id=run_id, limit=1000) if e["kind"] == "terminal"]


def evidence(name, value):
    if path := os.environ.get("SOLARLAB_TRANSIENT_PROCESS_EVIDENCE"):
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        (target / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


@pytest.fixture(scope="module")
def implementation():
    commit = os.environ.get("SOLARLAB_TRANSIENT_SOURCE_COMMIT")
    if commit is None:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=5).strip()
    archive = os.environ.get("SOLARLAB_TRANSIENT_SOURCE_ARCHIVE")
    return source_build(commit, Path(archive) if archive else None)


@pytest.fixture(scope="module")
def job(implementation):
    return example_job(implementation)


@pytest.fixture(scope="module")
def saved(tmp_path_factory, implementation):
    if root := os.environ.get("SOLARLAB_TRANSIENT_PROCESS_STORE"):
        # Explicit evidence-reuse mode, never represented as a new computation.
        store_path = Path(root).resolve()
        work = store_path.parent / "work"
        with RunStore(store_path, read_only=True) as store:
            for run_id in ("transient-completed", "transient-failed-prefix"):
                read_transient_job(store, run_id)
        return store_path, work
    root = tmp_path_factory.mktemp("transient-process")
    with RunStore(root / "store") as store, pool(store, root / "work") as supervisor:
        for name, failure in (("transient-completed", False), ("transient-failed-prefix", True)):
            handle = example_job(implementation, failure=failure).submit(supervisor, name, limits())
            receipt = supervisor.wait(handle, 90).data
            assert receipt["store_state"] == ("failed" if failure else "succeeded"), receipt
            assert receipt["process_joined"] and receipt["process_exit_observed"]
    return root / "store", root / "work"


@pytest.mark.parametrize("name,count,outcome", [("transient-completed", 3, "succeeded"), ("transient-failed-prefix", 2, "failed")])
def test_actual_histories_have_exact_artifacts_one_terminal_and_closed_process(saved, name, count, outcome):
    import numpy as np

    root, work = saved
    with RunStore(root, read_only=True) as store:
        loaded = read_transient_job(store, name)
        attempt, = store.attempts(name)
        receipt = json.loads((work / (store.store_id + "-" + attempt["attempt_id"]) / "receipt.json").read_text())
        assert receipt["store_state"] == outcome and receipt["exitcode"] == 0
        assert receipt["process_joined"] and receipt["process_exit_observed"] and receipt["terminal_committed"]
        with pytest.raises(ChildProcessError):
            os.waitpid(receipt["pid"], os.WNOHANG)
        assert len(terminals(store, name)) == 1 and len(attempt["terminal"]["artifacts"]) == 16
        assert receipt["resources"]["accepted_steps"] is None and receipt["resources"]["calls"] is None
        assert loaded["arrays"]["potential_V"].shape == (count, 37)
        assert loaded["arrays"]["total_current_density_A_m2"].shape == (count - 1, 36)
        assert loaded["arrays"]["terminal_total_current_density_A_m2"].shape == (count - 1, 2)
        assert len(loaded["arrays"]) == 14 and all(np.isfinite(a).all() for a in loaded["arrays"].values())
        assert np.array_equal(loaded["arrays"]["accepted_times_s"], [0., 1e-7, 2e-7][:count])
        assert loaded["document"]["physical_certificate_status"] == "not_evaluated"
        assert not loaded["manifest"]["can_execute"] and not loaded["worker_result"]["scientific_qualification"]
        assert loaded["worker_result"]["cold_start"]["inputs"] == loaded["worker_result"]["transient_capture"]["description"]["inputs"]
        if outcome == "failed":
            assert loaded["document"]["reason"] == "budget_exhausted"
            assert loaded["arrays"]["requested_times_s"].tolist() == [0., 1e-7, 2e-7, 3e-7]
            assert [a["accepted"] for a in loaded["diagnostics"]["attempts"]] == [True, False]
            assert loaded["diagnostics"]["attempts"][-1]["end_s"] == 2e-7
            assert loaded["diagnostics"]["attempts"][-1]["newton"] is not None
        child = json.loads((work / (store.store_id + "-" + attempt["attempt_id"]) / "terminal.json").read_text())
        for record in child["artifacts"]:
            saved_record = next(store.artifact(ref["artifact_id"]) for ref in attempt["terminal"]["artifacts"]
                                if store.artifact(ref["artifact_id"])["name"] == record["name"])
            original = (work / (store.store_id + "-" + attempt["attempt_id"]) / record["file"]).read_bytes()
            assert original == store.read_artifact(saved_record["artifact_id"])
            assert hashlib.sha256(original).hexdigest() == record["sha256"] == saved_record["sha256"]
        with pytest.raises(ValueError, match="committed"):
            read_transient_job(store, name, attempt_id="different-attempt")


def test_fresh_reader_process_imports_no_numerical_execution(saved, tmp_path):
    code = '''
import importlib.abc, pathlib, sys
class NoSolver(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == p or fullname.startswith(p + '.') for p in (
            'solarlab.solvers', 'solarlab.equations', 'solarlab.device.compile_1d',
            'solarlab.experiments.transient', 'perovskite_sim')):
            raise AssertionError('reader imported numerical execution: ' + fullname)
sys.meta_path.insert(0, NoSolver())
from solarlab.experiments.transient_worker import read_transient_job
from solarlab.io.run_store import RunStore
with RunStore(pathlib.Path(sys.argv[1]), read_only=True) as store:
    for name in ('transient-completed', 'transient-failed-prefix'):
        result = read_transient_job(store, name)
        assert len(result['arrays']) == 14
        print(name, result['run_state'], result['arrays']['accepted_times_s'].tolist())
'''
    check = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(saved[0])], cwd=tmp_path,
                           capture_output=True, text=True, timeout=20)
    assert check.returncode == 0, check.stderr
    evidence("SolverFreeRead.json", {"returncode": check.returncode, "stdout": check.stdout, "stderr": check.stderr})


def reseal(saved, target, change):
    with RunStore(saved[0], read_only=True) as original:
        run = original.get_run("transient-failed-prefix")
        attempt, = original.attempts(run["run_id"])
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
                result["worker_result"]["payloads"]["potential_V"]["metadata"]["unit"] = "wrong"
            record = store.publish(token, item["name"], ArtifactPayload(raw, item["format"], _json_document(item["metadata"])))
            ids.append(record["artifact_id"])
        if change == "cold-capture":
            result["worker_result"]["cold_start"]["compiled_identity"] = "0" * 64
        if change == "capture-digest":
            result["worker_result"]["transient_capture"]["result_sha256"] = "0" * 64
        store.complete(token, state="succeeded" if change == "false-success" else "failed", result=result, artifacts=ids)
        if change == "terminal-reference":
            terminal = store.attempts("fixture")[0]["terminal"]
            terminal["artifacts"][0]["manifest_sha256"] = "0" * 64
            store._db.execute("UPDATE attempts SET terminal_json=? WHERE attempt_id=?", (_json_document(terminal), token.attempt_id))
        if change == "corrupt-bytes":
            path = target / store.artifact(ids[0])["relative_path"]
            data = bytearray(path.read_bytes())
            data[-1] ^= 1
            path.write_bytes(data)


@pytest.mark.parametrize("change", ["unit", "cold-capture", "capture-digest", "false-success", "terminal-reference", "corrupt-bytes"])
def test_resealed_or_corrupt_transient_history_is_rejected(saved, tmp_path, change):
    target = tmp_path / "store"
    reseal(saved, target, change)
    with RunStore(target, read_only=True) as store, pytest.raises((ValueError, ArtifactIntegrityError)):
        read_transient_job(store, "fixture")


def test_invalid_request_and_unobserved_caps_fail_before_native_work(tmp_path, job):
    body = json.loads(job.request.request.content)
    body["protocol"]["generation_scale"][1] = 1.5
    request = replace(job.request, request=SourceDocument(job.request.request.id, _json_document(body).encode()))
    with RunStore(tmp_path / "store") as store, pool(store, tmp_path / "work") as supervisor:
        for cap in ("calls", "accepted_steps"):
            with pytest.raises(ValueError, match="counter budgets"):
                job.submit(supervisor, "unsupported-" + cap, replace(limits(), **{cap: 1}))
        handle = supervisor.submit("invalid", "transient_1d", request, limits(), execution=job.execution)
        receipt = supervisor.wait(handle, 20).data
        assert receipt["store_state"] == "failed" and receipt["terminal_committed"]
        assert "differ" in receipt["worker_result"]["detail"] and not receipt["terminal_artifacts"]
        assert len(terminals(store, handle.run_id)) == 1
        assert not any(e["payload"].get("ui_progress", {}).get("stage", "").startswith("transient_")
                       for e in store.events(run_id=handle.run_id, limit=1000))


def test_actual_running_transient_cancellation_closes_owner_without_publication(tmp_path, job):
    root = Path(os.environ.get("SOLARLAB_TRANSIENT_PROCESS_CANCEL_STORE", str(tmp_path / "store")))
    with RunStore(root) as store, pool(store, root.parent / "cancel-work", soft_grace_seconds=.05) as supervisor:
        handle = job.submit(supervisor, "transient-cancelled", limits())
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if any(e["payload"].get("ui_progress", {}).get("stage") == "transient_initial_dc"
                   for e in store.events(run_id=handle.run_id, limit=1000)):
                break
            time.sleep(.005)
        else:
            raise AssertionError("real child did not reach cold initial DC")
        assert supervisor.cancel(handle)
        receipt = supervisor.wait(handle, 10).data
        assert receipt["store_state"] == "cancelled" and receipt["terminal_committed"]
        assert receipt["process_joined"] and receipt["process_exit_observed"]
        assert not receipt["terminal_artifacts"] and len(terminals(store, handle.run_id)) == 1
        assert not supervisor.cancel(handle)
        with pytest.raises(ValueError, match="committed"):
            read_transient_job(store, handle.run_id)
        evidence("Cancellation.json", {"store": str(root.resolve()), "receipt": receipt,
                                       "terminal_count": len(terminals(store, handle.run_id))})


def _returned_failure(request, context):
    """Infrastructure fixture, not a numerical computation claim."""
    body = json.loads(request.request.content)
    context.progress({"stage": "ready"})
    if body.get("wait"):
        while not Path("release").exists():
            time.sleep(.005)  # Deliberately return late after the parent cancellation.
    return WorkerResult.from_dict({"reason": "fixture_numerical_failure"},
        artifacts=(("prefix", bytes_artifact(b"retained fixture prefix", metadata={"fixture": True})),), state="failed")


def fixture_request(**body):
    source = SourceDocument("fixture", _json_document(body).encode())
    binding = WorkerBinding.from_function("fixture_failure", _returned_failure, registry_sha256=source.sha256,
                                          schema_sha256=source.sha256)
    return WorkerRequest(source, source, source, source, source.sha256, source.sha256), binding


def test_worker_result_defaults_and_refuses_cancelled_or_unknown_state():
    assert WorkerResult.from_dict({}).state == "succeeded"
    for state in ("cancelled", "partial", True):
        with pytest.raises(ValueError, match="succeed or fail"):
            WorkerResult.from_dict({}, state=state)


@pytest.mark.parametrize("override", ["cancel", "publication-error", "publication-wall", "missing-counter"])
def test_failed_payloads_cannot_bypass_parent_cancellation_or_resource_guards(tmp_path, monkeypatch, override):
    request, binding = fixture_request(wait=override == "cancel")
    cap = replace(limits(), wall_seconds=2., calls=1 if override == "missing-counter" else None)
    with RunStore(tmp_path / "store") as store, pool(store, tmp_path / "work", (binding,), soft_grace_seconds=.8) as supervisor:
        if override == "publication-error":
            def broken(*args, **kwargs):
                raise OSError("injected publication failure")
            monkeypatch.setattr(store, "publish", broken)
        if override == "publication-wall":
            original = store.publish
            def slow(*args, **kwargs):
                record = original(*args, **kwargs)
                time.sleep(cap.wall_seconds)
                return record
            monkeypatch.setattr(store, "publish", slow)
        handle = supervisor.submit("guarded", binding.id, request, cap)
        if override == "cancel":
            deadline = time.monotonic() + 2
            while not any(e["payload"].get("ui_progress", {}).get("stage") == "ready"
                          for e in store.events(run_id=handle.run_id, limit=1000)):
                assert time.monotonic() < deadline
                time.sleep(.005)
            assert supervisor.cancel(handle)
            (tmp_path / "work" / (store.store_id + "-" + handle.attempt_id) / "release").touch()
        receipt = supervisor.wait(handle, 8).data
        assert receipt["store_state"] == ("cancelled" if override == "cancel" else "failed")
        assert not receipt["terminal_artifacts"] and len(terminals(store, handle.run_id)) == 1
        assert receipt["process_joined"] and receipt["process_exit_observed"]
        reason = receipt["worker_result"]["reason"]
        assert reason == {"cancel": "cancel_requested", "publication-error": "artifact_publication",
                          "publication-wall": "wall_limit_during_publication", "missing-counter": "worker_exception"}[override]
