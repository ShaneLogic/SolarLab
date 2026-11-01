"""One real study, independent solver-free readback, and explicit failure paths."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pytest

from examples.research_1d import jv_scan, read_results
from solarlab.experiments.jv.steady import read_steady_jv, steady_jv_metrics
from solarlab.io.artifacts import ArtifactIntegrityError
from solarlab.io.run_store import RunStore

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples/research_1d"
GUARD = r'''
import importlib.abc, json, pathlib, runpy, sys
prefixes = tuple(json.loads(sys.argv[1]))
class NoExecutionImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == prefix or fullname.startswith(prefix + '.') for prefix in prefixes):
            raise AssertionError('forbidden execution import: ' + fullname)
sys.meta_path.insert(0, NoExecutionImports())
script = pathlib.Path(sys.argv[2])
sys.path.insert(0, str(script.parent))
sys.argv = [str(script), *sys.argv[3:]]
runpy.run_path(str(script), run_name='__main__')
'''


def command(script, arguments, *, reader=False, timeout=300):
    forbidden = ["perovskite_sim", "backend", "solarlab_server"]
    if reader:
        forbidden += ["solarlab.solvers", "solarlab.device.compile_1d"]
    argv = [sys.executable, "-I", "-B", "-c", GUARD, json.dumps(forbidden),
            str(EXAMPLES / script), *arguments]
    env = dict(os.environ)
    env.update({key: "1" for key in jv_scan.THREAD_NAMES})
    start = time.monotonic()
    result = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
    return result, {"argv": argv, "elapsed_seconds": time.monotonic() - start,
                    "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    root = Path(os.environ.get("SOLARLAB_RESEARCH_STUDY_PATH", str(tmp_path_factory.mktemp("research") / "store")))
    result, record = command("jv_scan.py", ["--store", str(root), "--values", "1 us", "2 us"])
    archive = os.environ.get("SOLARLAB_RESEARCH_EVIDENCE")
    if archive:
        path = Path(archive)
        path.mkdir(parents=True, exist_ok=True)
        (path / "ExampleCommand.json").write_text(json.dumps(record, indent=2) + "\n")
        if result.stdout:
            (path / "ExampleOutput.json").write_text(result.stdout)
    assert result.returncode == 0, result.stdout + result.stderr
    return root, json.loads(result.stdout)


def test_genuine_lifetime_sweep_has_actual_identities_and_bracketed_metrics(study):
    root, summary = study
    assert summary["outcome"] == "completed_with_estimated_metrics"
    assert not summary["scientific_qualification"]
    assert summary["runtime"]["python"] == sys.version
    for name in ("numpy", "scipy", "pydantic", "threadpoolctl"):
        assert summary["runtime"]["packages"][name] == importlib.metadata.version(name)
    assert all(pool["num_threads"] == 1 for pool in summary["runtime"]["threadpools"])
    points = summary["points"]
    assert len(points) == 2 and all(p["state"] == "succeeded" for p in points)
    for name in ("H_input", "H_physics", "H_execution"):
        assert len({point["identity"][name] for point in points}) == 2
    assert len({point["compiled_identity"] for point in points}) == 2
    with RunStore(root, read_only=True) as store:
        for point in points:
            attempt = store.attempts(point["run_id"])[0]
            arrays = read_steady_jv(store, attempt["result"])
            np.testing.assert_array_equal(arrays["voltage_V"], np.linspace(0., 1.25, 51))
            assert arrays["nodes_m"].shape == (37,)
            current, voltage = arrays["output_current_A_m2"], arrays["voltage_V"]
            assert current[0] > 0 and current[-1] < 0
            crossing = np.flatnonzero(current <= 0)[0]
            assert voltage[crossing - 1] <= point["metrics"]["open_circuit_voltage_V"] <= voltage[crossing]
            assert point["metrics_status"] == "estimated"
            assert point["metrics"]["efficiency"] is None  # No fabricated irradiance/PCE.
            assert all(evidence["accepted"] for evidence in attempt["result"]["solver_evidence"])
            assert store.get_run(point["run_id"])["request"]["input_metadata"]["attempt_phase"] == summary["attempt_phase"]


def test_independent_roundtrip_and_fresh_process_reader_never_load_a_solver(study):
    root, summary = study
    result, record = command("read_results.py", ["--store", str(root)], reader=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == summary
    with RunStore(root, read_only=True) as store:
        for point in summary["points"]:
            document = store.attempts(point["run_id"])[0]["result"]
            arrays = read_steady_jv(store, document)
            for name, identifier in document["arrays"].items():
                original = document["research_array_fingerprints"][name]
                record_on_disk = store.artifact(identifier)
                independent = np.load(root / record_on_disk["relative_path"], allow_pickle=False)
                assert independent.tobytes() == arrays[name].tobytes()
                assert hashlib.sha256(independent.tobytes()).hexdigest() == original["sha256"]
                assert independent.dtype.str == original["dtype"] and list(independent.shape) == original["shape"]
    archive = os.environ.get("SOLARLAB_RESEARCH_EVIDENCE")
    if archive:
        (Path(archive) / "ReadCommand.json").write_text(json.dumps(record, indent=2) + "\n")


def test_a_real_unbracketed_prefix_has_missing_metrics_not_zero_or_pass(study):
    root, _ = study
    loaded = read_results.read_study(root, "lifetime")
    for point in loaded["points"]:
        arrays = point["current"]["arrays"]
        prefix = arrays["voltage_V"] <= .5
        metrics = steady_jv_metrics(arrays["voltage_V"][prefix], arrays["output_current_A_m2"][prefix])
        assert metrics.open_circuit_voltage_V is None and metrics.fill_factor is None
        point["current"]["result"]["metrics"] = asdict(metrics)
    # A read-only derived prefix check; no saved result or identity is changed.
    summary = read_results.summarize(loaded)
    assert summary["outcome"] == "completed_with_unavailable_metrics"
    assert all(p["metrics_status"] == "unavailable" and p["metrics"]["open_circuit_voltage_V"] is None
               for p in summary["points"])


def test_unsupported_physics_and_failed_attempts_are_retained_without_solver(tmp_path, monkeypatch):
    import solarlab.solvers.nonlinear.steady_1d as solver

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported point reached a solver")

    monkeypatch.setattr(solver, "solve_steady_state", forbidden)
    document = json.loads((EXAMPLES / "device.json").read_text())
    document["device"]["layers"][1]["parameters"]["D_ion"] = 1e-15
    config = tmp_path / "unsupported.json"
    config.write_text(json.dumps(document))
    root = tmp_path / "failed-store"
    summary = jv_scan.run_study(root, config=config, values=["-1 us", "1 us"], voltage_points=2)
    assert summary["outcome"] == "completed_with_failed_points"
    assert all(p["state"] == "failed" and p["compiled_identity"] is None for p in summary["points"])
    assert any("D_ion" in p["reason"] for p in summary["points"])
    with RunStore(root) as store:
        run_id = summary["points"][0]["run_id"]
        previous = store.attempts(run_id)
        row = store.get_run(run_id)
        queued = store.retry(run_id, expected_attempt_id=row["current_attempt_id"], expected_generation=row["generation"])
        token = store.claim(run_id, expected_attempt_id=queued["current_attempt_id"],
                            expected_generation=queued["generation"], owner_id="explicit_saved_failure_test")
        store.complete(token, state="failed", result={"reason": "explicit failed-attempt readback fixture; no solve"})
        assert store.attempts(run_id)[0] == previous[0]
    loaded = read_results.read_study(root, "lifetime")
    assert len(loaded["points"][0]["attempts"]) == 2
    assert all(a["state"] == "failed" for a in loaded["points"][0]["attempts"])
    with pytest.raises(FileExistsError):
        jv_scan.run_study(root, config=config)


def test_identity_and_artifact_tampering_refuse_readback(study, tmp_path):
    root, summary = study
    with RunStore(root, read_only=True) as store:
        request = deepcopy(store.get_run(summary["points"][0]["run_id"])["request"])
    request["input_metadata"]["identity_payloads"]["H_execution"]["protocol"]["voltage_V"][0] = .01
    with pytest.raises(ValueError, match="identity digest"):
        read_results.verify_identity(request)
    copied = tmp_path / "corruption-copy"
    shutil.copytree(root, copied)
    with RunStore(copied, read_only=True) as store:
        result = store.attempts(summary["points"][0]["run_id"])[0]["result"]
        record = store.artifact(result["arrays"]["voltage_V"])
    (copied / record["relative_path"]).write_bytes(b"explicit corrupted artifact fixture")
    with pytest.raises(ArtifactIntegrityError):
        read_results.read_study(copied, "lifetime")


def test_additional_recombination_is_a_runnable_unqualified_example():
    result, record = command("steady.py", ["--additional-recombination"], timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    output = json.loads(result.stdout)
    assert "not full scientific qualification" in output["qualification"]
    assert all(gate["passed"] for gate in output["gates"])
    assert len(output["compiled_identity"]) == 64
    archive = os.environ.get("SOLARLAB_RESEARCH_EVIDENCE")
    if archive:
        (Path(archive) / "ExtensionCommand.json").write_text(json.dumps(record, indent=2) + "\n")

