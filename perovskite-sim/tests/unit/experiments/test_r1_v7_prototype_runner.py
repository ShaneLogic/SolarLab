"""V7 prototype evidence must distinguish a real complete trace from a prefix."""
from copy import deepcopy
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


path = Path(__file__).resolve().parents[3] / "scripts/run_r1_v7_prototype.py"
spec = spec_from_file_location("r1_v7_prototype_runner", path)
runner = module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def contract(tmp_path, monkeypatch):
    # A synthetic clock exercises the contract machinery without a trajectory.
    times = [0.0] + [index * 2.154434690031884 / 113 for index in range(1, 114)]
    historical = tmp_path / "Historical.json"
    historical.write_text(json.dumps({"times_s": times + [100.0]}))
    monkeypatch.setattr(runner, "HISTORICAL_REQUEST_SHA256", runner.sha(historical))
    monkeypatch.setattr(runner, "TIMES_SHA256", hashlib.sha256(runner.canonical(times).encode()).hexdigest())
    return runner.make_contract(historical)


def save_contract(tmp_path, value):
    path = tmp_path / "ExternalRequest.json"
    runner.write(path, value)
    return path, runner.sha(path)


def complete_rows(case):
    rows = []
    for level in case["time_substeps"]:
        rows.append({"substeps": level, "time_s": 0.0, "phase": "zero_plus", "state": {}})
        for left, right in zip(case["times_s"][:-1], case["times_s"][1:]):
            for step in range(1, level + 1):
                time = right if step == level else left + (right - left) * step / level
                rows.append({"substeps": level, "time_s": time, "phase": "accepted", "state": {}})
    return rows


def result_for(rows):
    return {"accepted_steps": rows, "certificate": {
        "certified": True, "limits": deepcopy(runner.LIMITS),
        "metrics": {name: 0.0 for name in runner.LIMITS}}}


def test_request_rejects_changed_bytes_even_when_physics_would_be_same(tmp_path, contract):
    path, digest = save_contract(tmp_path, contract)
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="external SHA256"):
        runner.load_contract(path, digest, tmp_path / "run")


@pytest.mark.parametrize("key,value", [
    ("control", "A"), ("intervals", 256), ("nonlinear_factor", 0.01),
    ("time_substeps", [4, 8, 16]), ("amplitude_V", 0.001), ("fault", "zero_ion"),
])
def test_resealed_different_case_cannot_replace_original(tmp_path, contract, key, value):
    contract["case"][key] = value
    path, digest = save_contract(tmp_path, contract)
    with pytest.raises(ValueError, match="case identity"):
        runner.load_contract(path, digest, tmp_path / "run")


def test_short_clock_and_changed_gate_do_not_become_valid_by_resealing(tmp_path, contract):
    bad = deepcopy(contract)
    bad["case"]["times_s"] = [0, 1e-9, 1e-8, 1e-6, 1e-4]
    path, digest = save_contract(tmp_path, bad)
    with pytest.raises(ValueError, match="114-point"):
        runner.load_contract(path, digest, tmp_path / "run")
    bad = deepcopy(contract)
    bad["certificate_limits"]["eliminated_operator_error"] = 2.0
    path, digest = save_contract(tmp_path, bad)
    with pytest.raises(ValueError, match="limits changed"):
        runner.load_contract(path, digest, tmp_path / "run")


def test_request_cannot_be_inside_result_and_cannot_claim_qualification(tmp_path, contract):
    path, digest = save_contract(tmp_path, contract)
    with pytest.raises(ValueError, match="outside"):
        runner.load_contract(path, digest, tmp_path)
    contract["scientifically_accepted"] = True
    path, digest = save_contract(tmp_path, contract)
    with pytest.raises(ValueError, match="grant scientific"):
        runner.load_contract(path, digest, tmp_path / "run")


def test_exact_794_rows_are_counted_by_each_level_and_prefix_is_not_complete(contract):
    rows = complete_rows(contract["case"])
    report = runner.extent(rows, contract["case"])
    assert report["complete"]
    assert [item["accepted_rows"] for item in report["levels"]] == [114, 227, 453]
    assert report["accepted_rows"] == 794
    prefix = runner.extent(rows[:-1], contract["case"])
    assert not prefix["complete"]
    assert prefix["levels"][-1]["accepted_rows"] == 452


def test_duplicate_or_foreign_level_cannot_make_prefix_count_complete(contract):
    rows = complete_rows(contract["case"])
    rows[-2] = deepcopy(rows[-3])
    assert not runner.extent(rows, contract["case"])["complete"]
    rows = complete_rows(contract["case"])
    rows[-1]["substeps"] = 8
    report = runner.extent(rows, contract["case"])
    assert not report["complete"] and report["unassigned_rows"] == 1


def test_a_forged_certified_flag_does_not_hide_changed_or_nonfinite_limits():
    result = result_for([])
    result["certificate"]["metrics"]["eliminated_operator_error"] = 2.0
    assert not runner.certificate_check(result)["passed"]
    result["certificate"]["limits"]["eliminated_operator_error"] = 2.0
    assert not runner.certificate_check(result)["passed"]
    result = result_for([])
    result["certificate"]["metrics"]["nonlinear_residual"] = float("nan")
    assert not runner.certificate_check(result)["passed"]


def test_recording_zero_low_bits_does_not_prove_consumption():
    state = {"precision_" + field + "_" + side: [1.0] if side == "hi" else [0.0]
             for field in ("phi_V", "n_m3", "p_m3", "positive_m3", "dqfn_V", "dqfp_V")
             for side in ("hi", "lo")}
    report = runner.precision_record_check([{"state": state}], "compensated")
    assert report["record_fields_present"]
    assert report["proves_low_bit_consumption"] is False
    state["precision_phi_V_lo"] = [float("inf")]
    assert not runner.precision_record_check([{"state": state}], "compensated")["record_fields_present"]


def fake_api(rows, *, fail_after=None, replay_certified=True, result_rows=None):
    prepared = SimpleNamespace(to_dict=lambda: {"fixture": "unit-test"})

    def run(_, observer):
        observed = rows if fail_after is None else rows[:fail_after]
        for row in observed:
            observer(row)
        result = result_for(observed if result_rows is None else result_rows)
        if fail_after is not None:
            error = RuntimeError("recorded Newton failure")
            error.result = result
            raise error
        return result

    return {"json_data": lambda value: value, "prepare": lambda: prepared, "run": run,
            "replay": lambda prepared, result, incomplete: {"certified": replay_certified,
                                                           "incomplete": incomplete}}


def test_runner_retains_exception_and_prefix_without_promoting_replay(tmp_path, contract):
    rows = complete_rows(contract["case"])
    report = runner.run_trajectory(tmp_path, contract["case"], "baseline",
                                  fake_api(rows, fail_after=120), lambda: None)
    assert report["execution_status"] == "failed"
    assert report["extent"]["accepted_rows"] == 120
    assert not report["four_predicates_passed"]
    assert report["four_predicates"]["replay_certified"]
    assert not report["four_predicates"]["run_completed"]
    assert (tmp_path / "FailureV1.json").exists()
    assert len((tmp_path / "AcceptedStepsV1.jsonl").read_text().splitlines()) == 120
    assert json.loads((tmp_path / "PhysicsReplayV1.json").read_text())["incomplete"]


def test_successful_execution_with_missing_low_fields_is_explicit(tmp_path, contract):
    report = runner.run_trajectory(tmp_path, contract["case"], "compensated",
                                  fake_api(complete_rows(contract["case"])), lambda: None)
    assert report["four_predicates_passed"]
    assert report["precision_records"]["required"]
    assert not report["precision_records"]["record_fields_present"]
    assert not report["precision_records"]["proves_low_bit_consumption"]


def test_fine_preparation_failure_retains_its_seed_and_actual_state(tmp_path, contract):
    api = fake_api([])
    payload = {"schema": "R1FailedPreparationPairV1", "certified": False,
               "raw_preparation": {"seed_preparation": {"sha256": "seed"},
                                   "state": {"precision_phi_V_lo": [1e-30]}}}

    def prepare():
        error = RuntimeError("fine equilibrium failed original gate")
        error.result = payload
        raise error

    api["prepare"] = prepare
    report = runner.run_trajectory(tmp_path, contract["case"], "compensated", api, lambda: None)
    assert report["execution_status"] == "failed"
    assert report["extent"]["accepted_rows"] == 0
    assert not report["four_predicates_passed"]
    assert json.loads((tmp_path / "PreparationFailureV1.json").read_text()) == payload


def test_observer_mismatch_rejects_otherwise_certified_result(tmp_path, contract):
    rows = complete_rows(contract["case"])
    report = runner.run_trajectory(tmp_path, contract["case"], "baseline",
                                  fake_api(rows, result_rows=rows[:-1]), lambda: None)
    assert report["extent"]["complete"]
    assert not report["four_predicates"]["observer_matches_result"]
    assert not report["four_predicates_passed"]


def test_source_change_after_integration_keeps_real_trace_but_fails(tmp_path, contract):
    calls = []

    def guard():
        calls.append(True)
        if len(calls) > 1:
            raise ValueError("source changed")

    report = runner.run_trajectory(tmp_path, contract["case"], "baseline",
                                  fake_api(complete_rows(contract["case"])), guard)
    assert report["extent"]["complete"]
    assert report["execution_status"] == "failed"
    assert not report["four_predicates_passed"]
    assert (tmp_path / "FailureV1.json").exists()


def test_cost_can_compare_completed_failed_baseline_but_never_prefix():
    baseline = {"schema": "R1V7PrototypeRunV1", "mode": "baseline", "request_sha256": "a" * 64,
                "source_unchanged": True, "extent": {"complete": True},
                "runtime_identity": {"python": "same"}, "four_predicates_passed": False,
                "cost": {"elapsed_s": 10., "peak_rss_bytes": 100., "bytes_per_row": 50.}}
    current = {"runtime_identity": baseline["runtime_identity"], "extent": {"complete": True},
               "cost": {"elapsed_s": 50., "peak_rss_bytes": 200., "bytes_per_row": 150.}}
    report = runner.cost_comparison(current, baseline, "a" * 64)
    assert report["qualified"]
    assert report["ratios"] == runner.ENGINEERING_LIMITS
    current["cost"]["elapsed_s"] = 50.1
    assert not runner.cost_comparison(current, baseline, "a" * 64)["qualified"]
    current["cost"]["elapsed_s"] = 10.
    baseline["extent"]["complete"] = False
    assert not runner.cost_comparison(current, baseline, "a" * 64)["qualified"]


def test_source_snapshot_rejects_dirty_tree_and_wrong_full_commit(tmp_path, monkeypatch):
    project = tmp_path / "perovskite-sim"
    project.mkdir()
    (project / "solver.py").write_text("value = 1\n")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-qm", "fixture"], cwd=tmp_path, check=True)
    monkeypatch.setattr(runner, "PROJECT", project)
    commit = runner.git("rev-parse", "HEAD")
    assert runner.source_snapshot(commit)["source_commit"] == commit
    with pytest.raises(ValueError, match="full commit"):
        runner.source_snapshot("f" * 40)
    (project / "solver.py").write_text("value = 2\n")
    with pytest.raises(ValueError, match="clean source"):
        runner.source_snapshot(commit)
    subprocess.run(["git", "update-index", "--assume-unchanged", "perovskite-sim/solver.py"],
                   cwd=tmp_path, check=True)
    assert not runner.git("status", "--porcelain")
    with pytest.raises(ValueError, match="source bytes differ"):
        runner.source_snapshot(commit)
