"""One real illuminated pair; reuse its saved bytes for all read/tamper tests."""
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

from examples.research_1d.process_ionic_jv import example_job, source_build
from solarlab.experiments.execution import ProcessSupervisor
from solarlab.experiments.jv.ionic_results import INITIAL, RAW, _diagnostic_payload, _parse_diagnostics, result_digest
from solarlab.experiments.jv.ionic_worker import IonicJVInitial, IonicJVProtocol, ionic_jv_worker_binding, read_ionic_jv_job
from solarlab.experiments.transient_results import _decode, _encode
from solarlab.experiments.worker import ResourceLimits, WorkerBinding, WorkerRequest
from solarlab.io.artifacts import ArtifactIntegrityError, ArtifactPayload, _json_document
from solarlab.io.run_store import RunStore
from solarlab.materials.source import SourceDocument

ROOT = Path(__file__).resolve().parents[2]
NAMES = ("ionic-jv-completed-v2", "ionic-jv-failed-prefix")


def limits():
    return ResourceLimits(30., 1536 * 1024**2, 96 * 1024**2, None, None)


def pool(store, work, bindings=None):
    return ProcessSupervisor(store, work, bindings or (ionic_jv_worker_binding(),), max_active=1, max_pending=1,
        shared_rss_bytes=3 * 1024**3, shared_output_bytes=384 * 1024**2,
        poll_seconds=.01, soft_grace_seconds=.1, terminate_grace_seconds=.1)


@pytest.fixture(scope="module")
def implementation():
    commit = os.environ.get("SOLARLAB_IONIC_JV_COMMIT") or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=5).strip()
    archive = os.environ.get("SOLARLAB_IONIC_JV_ARCHIVE")
    return source_build(commit, Path(archive) if archive else None)


@pytest.fixture(scope="module")
def job(implementation):
    return example_job(implementation)


@pytest.fixture(scope="module")
def saved(tmp_path_factory, implementation):
    if path := os.environ.get("SOLARLAB_IONIC_JV_STORE"):
        root = Path(path).resolve()
        return root, Path(os.environ.get("SOLARLAB_IONIC_JV_WORK", str(root.parent / "work")))
    root = tmp_path_factory.mktemp("ionic-jv")
    with RunStore(root / "store") as store, pool(store, root / "work") as supervisor:
        initial_bytes = initial_sha = None
        for name, failure in zip(NAMES, (False, True), strict=True):
            job = example_job(implementation, failure=failure, initial_bytes=initial_bytes, initial_sha256=initial_sha)
            handle = job.submit(supervisor, name, limits())
            receipt = supervisor.wait(handle, 40).data
            assert receipt["store_state"] == ("failed" if failure else "succeeded"), receipt
            assert receipt["process_joined"] and receipt["process_exit_observed"]
            if not failure:
                initial_bytes = read_ionic_jv_job(store, name)["initial_bytes"]
                initial_sha = hashlib.sha256(initial_bytes).hexdigest()
    return root / "store", root / "work"


@pytest.mark.parametrize("name,state", [(NAMES[0], "succeeded"), (NAMES[1], "failed")])
def test_real_curve_exact_arrays_metrics_initial_calls_and_single_terminal(saved, name, state):
    root, work = saved
    with RunStore(root, read_only=True) as store:
        loaded = read_ionic_jv_job(store, name)
        attempt, = store.attempts(name)
        directory = work / (store.store_id + "-" + attempt["attempt_id"])
        receipt = json.loads((directory / "receipt.json").read_text())
        assert loaded["run_state"] == receipt["store_state"] == state and receipt["exitcode"] == 0
        assert receipt["process_joined"] and receipt["process_exit_observed"] and receipt["terminal_committed"]
        assert receipt["resources"]["owned_wall_seconds"] <= 30 and receipt["signals"] == []
        with pytest.raises(ChildProcessError):
            os.waitpid(receipt["pid"], os.WNOHANG)
        assert len([e for e in store.events(run_id=name, limit=1000) if e["kind"] == "terminal"]) == 1
        assert len(attempt["terminal"]["artifacts"]) == 15
        assert len(loaded["stored_arrays"]) == 12 and len(loaded["arrays"]) == 23
        curve, summary, arrays = loaded["diagnostics"]["curve"], loaded["document"], loaded["arrays"]
        count = summary["accepted_count"]
        assert arrays["requested_voltage_V"].tolist() == [tick / 40. for tick in range(51)]
        assert arrays["potential_V"].shape == (count, 37) and arrays["state_coordinates"].shape == (count, 137)
        for species in ("positive", "negative"):
            assert arrays[species + "_ion_density_m3"].shape == (count, 13)
            assert arrays[species + "_ion_inventory_m2"].shape == (count,)
        assert arrays["total_current_density_A_m2"].shape == (count, 38)
        assert arrays["terminal_total_current_density_A_m2"].shape == (count, 2)
        assert np.array_equal(arrays["output_current_A_m2"], -curve["voltage_polarity"] * arrays["total_current_density_A_m2"][:, 0])
        assert summary["ion_site_contract"] == "shared_finite_sites"
        assert summary["physical_certificate_status"] == summary["voltage_grid_qualification"] == "not_evaluated"
        assert summary["incident_power_W_m2"] is None and summary["metrics"]["efficiency"] is None
        assert loaded["worker_result"]["scientific_qualification"] is False and not loaded["manifest"]["can_execute"]
        counts = summary["computation_counts"]
        assert counts["illumination"]["component_solver_calls"] == 0
        if state == "succeeded":
            assert count == 51 and summary["first_failure"] is None
            assert counts["cold_steady"]["solver_calls"] == counts["illumination"]["coupled_solver_calls"] == 1
            assert counts["illumination"]["density_preparation_calls"] == 1
            assert counts["total_solver_calls"] == 53
            assert summary["metrics"]["open_circuit_voltage_V"] is not None
        else:
            assert 0 < count < 51 and len(curve["attempts"]) == count + 1
            assert curve["attempts"][-1]["accepted"] is False
            assert summary["first_failure"]["termination"] == "wall_budget_exhausted"
            assert summary["initial_policy"] == "saved_accepted"
            assert counts["cold_steady"]["solver_calls"] == counts["illumination"]["coupled_solver_calls"] == 0
            assert counts["total_solver_calls"] == curve["solver_calls"]
        child = json.loads((directory / "terminal.json").read_text())
        by_name = {store.artifact(ref["artifact_id"])["name"]: store.artifact(ref["artifact_id"]) for ref in attempt["terminal"]["artifacts"]}
        for payload in child["artifacts"]:
            item = by_name[payload["name"]]
            original = (directory / payload["file"]).read_bytes()
            assert original == store.read_artifact(item["artifact_id"])
            assert hashlib.sha256(original).hexdigest() == payload["sha256"] == item["sha256"]
        if state == "succeeded":
            # Regression: full raw51-point history is an artifact, not1MiB metadata.
            assert len(store.read_artifact(by_name[RAW]["artifact_id"])) > 2**20
        with pytest.raises(ValueError, match="committed"):
            read_ionic_jv_job(store, name, attempt_id="absent-attempt")


def test_saved_initial_is_exact_and_not_relabelled(saved):
    with RunStore(saved[0], read_only=True) as store:
        success, failed = (read_ionic_jv_job(store, name) for name in NAMES)
    assert success["initial_bytes"] == failed["initial_bytes"]
    assert success["document"]["compiled_identity"] == failed["document"]["compiled_identity"]
    assert success["diagnostics"]["controls"]["steady"] == failed["diagnostics"]["controls"]["steady"]
    assert success["diagnostics"]["controls"]["wall_s"] == 25.
    assert failed["diagnostics"]["controls"]["wall_s"] == 1.


def test_fresh_solver_blocked_reader_preserves_every_array_word(saved, tmp_path):
    code = '''
import hashlib, importlib.abc, json, pathlib, sys
sys.path.insert(0,sys.argv[1])
class NoSolver(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if any(fullname==n or fullname.startswith(n+'.') for n in (
            'solarlab.solvers','solarlab.equations','solarlab.device.compile_1d','solarlab.device.compile_ionic_1d',
            'solarlab.experiments.ionic_transient','solarlab.experiments.ionic_initialization',
            'solarlab.experiments.jv.ionic_steady','perovskite_sim')):
            raise AssertionError('solver import: '+fullname)
sys.meta_path.insert(0,NoSolver())
from solarlab.experiments.jv.ionic_worker import read_ionic_jv_job
from solarlab.io.run_store import RunStore
with RunStore(pathlib.Path(sys.argv[2]),read_only=True) as store:
    for name in sys.argv[3:]:
        loaded=read_ionic_jv_job(store,name)
        print(json.dumps({'run_id':name,'state':loaded['run_state'],'arrays':{
            k:hashlib.sha256(a.tobytes()).hexdigest() for k,a in loaded['arrays'].items()}}))
'''
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(ROOT / "src"), str(saved[0]), *NAMES],
                            cwd=tmp_path, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    with RunStore(saved[0], read_only=True) as store:
        for line in result.stdout.splitlines():
            read = json.loads(line)
            loaded = read_ionic_jv_job(store, read["run_id"])
            assert read["arrays"] == {k: hashlib.sha256(a.tobytes()).hexdigest() for k, a in loaded["arrays"].items()}


def reseal(saved, target, change):
    with RunStore(saved[0], read_only=True) as original:
        run = original.get_run(NAMES[1])
        attempt, = original.attempts(NAMES[1])
        result = attempt["result"]
        items = [(original.artifact(ref["artifact_id"]), original.read_artifact(ref["artifact_id"])) for ref in attempt["terminal"]["artifacts"]]
    with RunStore(target) as store:
        row = store.enqueue("fixture", opaque_identity=run["request"]["identity"], input_metadata=run["request"]["input_metadata"])
        token = store.claim("fixture", expected_attempt_id=row["current_attempt_id"], expected_generation=row["generation"], owner_id="reader-fixture")
        ids, payloads = [], {}
        for item, data in items:
            if change == "unit" and item["name"] == "potential_V":
                item["metadata"]["unit"] = "wrong"
            if change == "component-order" and item["name"] == "current_density_A_m2":
                item["metadata"]["components"].reverse()
            if item["name"] == RAW and change in {"counts", "nonfinite-accepted", "metrics", "protocol", "initial-controls", "polarity"}:
                raw = _decode(json.loads(data))
                if change == "counts":
                    raw["curve"]["solver_calls"] += 1
                elif change == "nonfinite-accepted":
                    raw["curve"]["attempts"][0]["solution"]["diagnostics"]["current_spread_A_m2"] = float("nan")
                elif change == "metrics":
                    raw["curve"]["metrics"]["efficiency"] = .9
                elif change == "protocol":
                    raw["protocol"]["generation_scale"] = 2.
                elif change == "initial-controls":
                    raw["controls"]["steady"]["budget"]["max_iterations"] = 1
                elif change == "polarity":
                    raw["curve"]["voltage_polarity"] *= -1
                data = json.dumps(_encode(raw), allow_nan=False).encode()
            payload = ArtifactPayload(data, item["format"], _json_document(item["metadata"]))
            if item["name"] != "execution_manifest":
                payloads[item["name"]] = payload
                result["worker_result"]["payloads"][item["name"]] = {"sha256": payload.sha256, "format": payload.format, "metadata": item["metadata"]}
            ids.append(store.publish(token, item["name"], payload)["artifact_id"])
        worker = result["worker_result"]
        worker["capture"]["result_sha256"] = result_digest(worker["ionic_jv_result"], payloads)
        if change == "initial-binding":
            worker["capture"]["initial_artifact_sha256"] = "0" * 64
        if change == "source-binding":
            worker["capture"]["source_sha256"]["solarlab/experiments/jv/ionic_steady.py"] = "0" * 64
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


@pytest.mark.parametrize("change", ["unit", "component-order", "counts", "nonfinite-accepted", "metrics", "protocol",
    "initial-controls", "initial-binding", "source-binding", "false-success", "terminal-reference", "corrupt-bytes", "polarity"])
def test_resealed_or_corrupt_curve_is_rejected(saved, tmp_path, change):
    reseal(saved, tmp_path / "store", change)
    with RunStore(tmp_path / "store", read_only=True) as store, pytest.raises((ValueError, ArtifactIntegrityError)):
        read_ionic_jv_job(store, "fixture")


def test_lossless_failed_words_remain_distinct_from_missing():
    words = np.asarray([np.nan, np.inf, -np.inf, -0.])
    data = _decode(json.loads(json.dumps(_encode({"failed_trial": words, "missing": None}), allow_nan=False)))
    assert data["failed_trial"].tobytes() == words.tobytes() and data["missing"] is None


def test_raw_artifact_above_metadata_ceiling_roundtrips_without_computation(tmp_path):
    raw = {"codec_fixture": [np.arange(20000, dtype=np.float64) for _ in range(10)],
           "failed_words": np.asarray([np.nan, np.inf, -0.]), "missing": None}
    payload = _diagnostic_payload(raw, {"schema": "codec_fixture_only"})
    assert len(payload.data) > 2**20
    with pytest.raises(ValueError, match="metadata"):
        _json_document({"large": "x" * 2**20})
    with RunStore(tmp_path / "store") as store:
        identity = {"H_input": "1" * 64, "H_physics": "2" * 64, "H_execution": "3" * 64}
        row = store.enqueue("codec-fixture", opaque_identity=identity, input_metadata={"fixture_only": True})
        token = store.claim("codec-fixture", expected_attempt_id=row["current_attempt_id"], expected_generation=row["generation"], owner_id="codec-fixture")
        artifact = store.publish(token, "raw-fixture", payload)
        store.complete(token, state="succeeded", result={"fixture_only": True, "solver_calls": 0}, artifacts=[artifact["artifact_id"]])
        data = store.read_artifact(artifact["artifact_id"])
        assert data == payload.data and hashlib.sha256(data).hexdigest() == payload.sha256
    decoded = _parse_diagnostics(data)
    assert all(a.tobytes() == b.tobytes() for a, b in zip(decoded["codec_fixture"], raw["codec_fixture"]))
    assert decoded["failed_words"].tobytes() == raw["failed_words"].tobytes() and decoded["missing"] is None


@pytest.mark.parametrize("data", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1e999}', b'{"x":"\\ud800"}'])
def test_raw_json_duplicate_or_nonfinite_words_are_refused(data):
    with pytest.raises(ValueError):
        _parse_diagnostics(data)


def _hold_slot(request, context):
    """Ownership fixture only; never a substitute for a native computation."""
    context.progress({"stage": "hold_ready"})
    while True:
        context.check_cancel()
        time.sleep(.005)


def test_queued_cancel_has_one_terminal_and_no_curve(tmp_path, job):
    data = SourceDocument("hold", b'{"hold":true}')
    request = WorkerRequest(data, data, data, data, data.sha256, data.sha256)
    held = WorkerBinding.from_function("ionic_jv_hold", _hold_slot, registry_sha256=data.sha256, schema_sha256=data.sha256)
    with RunStore(tmp_path / "store") as store, pool(store, tmp_path / "work", (held, ionic_jv_worker_binding())) as owner:
        blocker = owner.submit("blocker", held.id, request, limits())
        deadline = time.monotonic() + 8
        while not any(e["payload"].get("ui_progress", {}).get("stage") == "hold_ready" for e in store.events(run_id="blocker", limit=1000)):
            assert time.monotonic() < deadline
            time.sleep(.01)
        queued = job.submit(owner, "queued", limits())
        assert store.get_run(queued.run_id)["state"] == "queued" and owner.cancel(queued)
        receipt = owner.wait(queued, 5).data
        assert receipt["store_state"] == "cancelled" and receipt["process_started"] is False
        attempt, = store.attempts(queued.run_id)
        assert attempt["terminal"]["artifacts"] == []
        assert len([e for e in store.events(run_id=queued.run_id, limit=1000) if e["kind"] == "terminal"]) == 1
        assert not owner.cancel(queued)
        with pytest.raises(ValueError, match="committed"):
            read_ionic_jv_job(store, queued.run_id)
        assert owner.cancel(blocker)
        closed = owner.wait(blocker, 5).data
        assert closed["process_joined"] and closed["process_exit_observed"]


def test_explicit_protocol_initial_hash_and_whole_job_limits(job):
    job.execution.validate_request(job.request)
    assert len(job.execution.manifest.canonical_json) > 4 * 1024**2
    for values in ((0., 0.), (0., float("nan")), ()):
        with pytest.raises(ValueError):
            IonicJVProtocol(values, 1., None)
    with pytest.raises(ValueError):
        IonicJVProtocol((0., .025), 0., None)
    with pytest.raises(ValueError, match="SHA-256"):
        IonicJVInitial("saved_accepted", .002, 25., b'{}', "0" * 64)
    for bound in (replace(limits(), calls=1), replace(limits(), accepted_steps=1), replace(limits(), wall_seconds=31.)):
        with pytest.raises(ValueError, match="wall<=30s"):
            job.submit(None, "invalid", bound)


def test_initial_artifact_is_not_a_synthesized_complete_curve(saved):
    with RunStore(saved[0], read_only=True) as store:
        loaded = read_ionic_jv_job(store, NAMES[1])
        refs = store.attempts(NAMES[1])[0]["terminal"]["artifacts"]
        initial = next(store.artifact(r["artifact_id"]) for r in refs if store.artifact(r["artifact_id"])["name"] == INITIAL)
        assert store.read_artifact(initial["artifact_id"]) == loaded["initial_bytes"]
    value = json.loads(loaded["initial_bytes"])
    assert value["summary"]["generation_scale"] == 1. and value["summary"]["voltage_V"] == .002
    assert loaded["document"]["outcome"] == "partial" and loaded["document"]["first_failure"] is not None
