"""The sole completed/insufficient-budget pair, then exact read/tamper checks."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from examples.research_1d.steady import load_device
from examples.research_2d.dc_assembly import spatial_input
from solarlab.device.compile_2d import compile_device_2d
from solarlab.experiments.two_dimensional.steady_results import (
    STORED_UNITS, _arrays, _load_raw, _summary, _validate, read_steady_2d_job, steady_2d_payload,
)
from solarlab.experiments.two_dimensional.steady_worker import (
    INITIAL_POLICY, Steady2DProtocol, _body, _lateral_initial, _validate as validate_input,
)
from solarlab.io.run_store import RunStore
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions
from solarlab.solvers.nonlinear.steady_2d import Steady2DOptions

pytestmark = pytest.mark.requires_clean_core

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def geometry_only():
    prepared, spatial = load_device(), spatial_input()
    return prepared, spatial, compile_device_2d(prepared, spatial)


def test_original_lateral_initial_formula_keeps_exact_195_words(geometry_only):
    from examples.research_2d.steady import lateral_initial
    prepared, spatial, device = geometry_only
    column = np.r_[device.column.reference_potential_V, np.zeros(2 * device.geometry.ny)]
    actual = _lateral_initial(device, column, voltage_V=.002)
    expected = lateral_initial(device, column, voltage_V=.002)
    assert actual.u.shape == (195,) and actual.potential_V.shape == (13, 5)
    assert actual.u.tobytes() == expected.u.tobytes()
    assert INITIAL_POLICY["auxiliary_seed_is_2d_completion"] is False
    assert INITIAL_POLICY["cross_target_seed_reuse"] is False


@pytest.mark.parametrize("scales", [(True,), (2.,), (1., 0.), (), (0., 0.)])
def test_nonexplicit_targets_refused(scales):
    with pytest.raises(ValueError, match="dark/G1"):
        Steady2DProtocol(.002, scales)


def test_failure_budget_changes_only_full_2d_counts(geometry_only):
    prepared, spatial, _ = geometry_only
    base = SteadyOptions()
    failure = replace(base, max_iterations=1, max_residual_evaluations=1, max_jacobian_evaluations=1)
    validate_input(prepared, spatial, Steady2DOptions(failure))
    body = _body(prepared, spatial, Steady2DProtocol(.002, (0., 1.)), Steady2DOptions(failure))
    assert body["controls"]["auxiliary_1d"]["max_residual_evaluations"] == base.max_residual_evaluations
    assert body["controls"]["full_2d"]["electronic"]["max_residual_evaluations"] == 1
    assert body["controls"]["full_2d"]["electronic"]["poisson_tolerance"] == base.poisson_tolerance
    with pytest.raises(ValueError, match="physical gates"):
        validate_input(prepared, spatial, Steady2DOptions(replace(base, poisson_tolerance=1e-4)))
    with pytest.raises(ValueError, match="positive integer"):
        validate_input(prepared, spatial, Steady2DOptions(replace(base, max_iterations=True)))


def test_empty_accepted_prefix_is_unavailable_fields_not_completion(geometry_only):
    prepared, spatial, device = geometry_only
    body = _body(prepared, spatial, Steady2DProtocol(.002, (0., 1.)), Steady2DOptions())
    # Serialization-only fixture. No seed or Newton call occurred.
    attempt = {"target": {"voltage_V": .002, "generation_scale": 0.}, "accepted": False,
        "termination": "synthetic_preparation_failure", "error": "payload validation fixture only",
        "auxiliary_1d_seed_called": False, "full_2d_adapter_called": False,
        "auxiliary_1d_seed": None, "failed_auxiliary_seed_newton": None, "result_2d": None,
        "auxiliary_seed_is_2d_completion": False}
    summary, payloads = steady_2d_payload(device, prepared, body, [attempt], .0)
    assert not summary["completed"] and summary["accepted_count"] == 0 and summary["unattempted_count"] == 1
    assert len(payloads) == 13 and len(STORED_UNITS) == 12
    raw = _load_raw(payloads["steady_2d_diagnostics"].data)
    arrays = _arrays(raw)
    assert len(arrays) == 22 and arrays["potential_V"].shape == (0, 13, 5)
    assert arrays["face_current_density_A_m2"].shape == (0, 112)
    assert arrays["terminal_mean_current_density_A_m2"].shape == (0, 2)
    broken = dict(summary, completed=True)
    with pytest.raises(ValueError, match="outcome"):
        _validate(raw, broken, arrays)
    broken_raw = deepcopy(raw)
    broken_raw["geometry"]["face_pairs"][0] = [0, 5]
    with pytest.raises(ValueError, match="connectivity"):
        _validate(broken_raw, _summary(broken_raw), _arrays(broken_raw))


@pytest.fixture(scope="module")
def actual_cases(tmp_path_factory):
    if saved := os.environ.get("SOLARLAB_STEADY2D_WORKER_SAVED"):
        return json.loads(Path(saved).read_text())
    # CI is self-contained. Local reruns use the exact saved pair above.
    root = tmp_path_factory.mktemp("steady-2d-worker")
    commit = os.environ.get("SOLARLAB_STEADY2D_COMMIT") or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    archive = Path(os.environ.get("SOLARLAB_STEADY2D_ARCHIVE", root / "source.tar"))
    if not archive.exists():
        subprocess.run(["git", "archive", "--format=tar", "--output", str(archive), commit, "--", "src/solarlab"],
                       cwd=ROOT, check=True)
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
        env[key] = "1"
    cases = []
    for failure in (False, True):
        name = "steady2d-failed" if failure else "steady2d-completed"
        receipt = root / (name + ".json")
        argv = [sys.executable, "-B", "-m", "examples.research_2d.process_steady", "run",
            "--store", str(root / "Store"), "--work", str(root / "Work"), "--run-id", name,
            "--build-source-commit", commit, "--source-archive", str(archive), "--current-repository-head", commit,
            "--receipt-json", str(receipt)]
        if failure:
            argv.append("--insufficient-budget")
        result = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True, timeout=45)
        (root / (name + ".stdout")).write_text(result.stdout)
        (root / (name + ".stderr")).write_text(result.stderr)
        assert result.returncode == (2 if failure else 0), result.stderr + result.stdout[-2000:]
        cases.append(json.loads(receipt.read_text()))
    evidence = {"store": str(root / "Store"), "work": str(root / "Work"), "source_commit": commit, "cases": cases}
    (root / "ActualCases.json").write_text(json.dumps(evidence, indent=2))
    return evidence


def test_actual_pair_preserves_2d_states_counts_sources_and_one_terminal(actual_cases):
    with RunStore(Path(actual_cases["store"]), read_only=True) as store:
        for case in actual_cases["cases"]:
            loaded = read_steady_2d_job(store, case["run_id"], attempt_id=case["attempt_id"])
            summary, arrays = loaded["document"], loaded["arrays"]
            complete = case["state"] == "succeeded"
            assert loaded["run_state"] == case["state"] and summary["dimension"] == 2
            assert summary["accepted_count"] == (2 if complete else 0)
            assert summary["computation_counts"]["auxiliary_1d_seed_calls"] == (2 if complete else 1)
            assert summary["computation_counts"]["full_2d_newton_calls"] == (2 if complete else 1)
            assert arrays["potential_V"].shape == (2 if complete else 0, 13, 5)
            assert summary["physical_certificate_status"] == "not_evaluated"
            attempt = next(a for a in store.attempts(case["run_id"]) if a["attempt_id"] == case["attempt_id"])
            records = [store.artifact(ref["artifact_id"]) for ref in attempt["terminal"]["artifacts"]]
            assert len(records) == 14
            terminals = [e for e in store.events(run_id=case["run_id"], limit=1000) if e["kind"] == "terminal"]
            assert len(terminals) == 1
            receipt = case["receipt"]
            assert receipt["process_started"] and receipt["process_joined"] and receipt["process_exit_observed"]
            assert receipt["exitcode"] == 0 and not receipt["signals"]
            assert receipt["resources"]["owned_wall_seconds"] <= 30
            for attempt in loaded["diagnostics"]["attempts"]:
                raw = attempt["result_2d"]
                assert attempt["auxiliary_1d_seed"]["newton"]["accepted"]
                assert raw["solver_controls"]["full_2d_newton_calls"] == 1
                assert np.any(raw["initial_diagnostics"]["lateral_current_density_A_m2"] != 0)
                if not complete:
                    assert not raw["newton"]["accepted"] and raw["termination"] == "budget_exhausted"
                    assert raw["candidate_coordinates"].shape == (195,)
            directory = Path(actual_cases["work"]) / (store.store_id + "-" + case["attempt_id"])
            terminal = json.loads((directory / "terminal.json").read_text())
            by_name = {r["name"]: r for r in records}
            for item in terminal["artifacts"]:
                assert hashlib.sha256((directory / item["file"]).read_bytes()).hexdigest() == by_name[item["name"]]["sha256"]


def test_actual_pair_solver_imports_forbidden_and_all_44_arrays_exact(actual_cases):
    code = """
import importlib.abc,json,sys
from pathlib import Path
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.startswith(('solarlab.solvers','solarlab.equations','solarlab.device.compile_','perovskite_sim')):
   raise AssertionError('forbidden numerical read import: '+fullname)
sys.meta_path.insert(0,Block())
from solarlab.experiments.two_dimensional.steady_results import read_steady_2d_job
from solarlab.io.run_store import RunStore
with RunStore(Path(sys.argv[1]),read_only=True) as store:
 loaded=read_steady_2d_job(store,sys.argv[2],attempt_id=sys.argv[3])
 print(json.dumps(dict(state=loaded['run_state'],arrays=loaded['array_fingerprints'])))
"""
    for case in actual_cases["cases"]:
        result = subprocess.run([sys.executable, "-B", "-c", code, actual_cases["store"], case["run_id"], case["attempt_id"]],
            cwd=ROOT, capture_output=True, text=True, timeout=20)
        assert result.returncode == 0, result.stderr
        returned = json.loads(result.stdout)
        assert returned["state"] == case["state"]
        assert returned["arrays"] == case["readback"]["array_fingerprints"]
        assert len(returned["arrays"]) == 22


@pytest.mark.parametrize("change", ["unit", "ownership", "generation"])
def test_saved_artifact_metadata_tamper_rejected(actual_cases, change):
    case = actual_cases["cases"][0]
    with RunStore(Path(actual_cases["store"]), read_only=True) as store:
        class ChangedRecord:
            def __getattr__(self, name):
                return getattr(store, name)
            def artifact(self, artifact_id):
                data = deepcopy(store.artifact(artifact_id))
                if data["name"] == "potential_V":
                    if change == "unit":
                        data["metadata"]["unit"] = "mV"
                    elif change == "ownership":
                        data["attempt_id"] = "wrong_attempt"
                    else:
                        data["generation"] += 1
                return data
        with pytest.raises(ValueError):
            read_steady_2d_job(ChangedRecord(), case["run_id"], attempt_id=case["attempt_id"])


def test_saved_raw_bytes_tamper_rejected_without_touching_original(actual_cases, tmp_path):
    case = actual_cases["cases"][0]
    clone = tmp_path / "Store"
    shutil.copytree(actual_cases["store"], clone)
    with RunStore(clone, read_only=True) as store:
        attempt = next(a for a in store.attempts(case["run_id"]) if a["attempt_id"] == case["attempt_id"])
        record = next(store.artifact(ref["artifact_id"]) for ref in attempt["terminal"]["artifacts"]
                      if store.artifact(ref["artifact_id"])["name"] == "steady_2d_diagnostics")
        path = clone / record["relative_path"]
        raw = bytearray(path.read_bytes())
        raw[len(raw) // 2] ^= 1
        path.write_bytes(raw)
        with pytest.raises(ValueError):
            read_steady_2d_job(store, case["run_id"], attempt_id=case["attempt_id"])
