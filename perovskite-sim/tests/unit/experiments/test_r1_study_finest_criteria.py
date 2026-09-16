"""Independent counterexamples for finest-pair and failed-tail study gates."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[3] / "scripts/run_one_dimensional_mechanism_r1_physics_study.py"
    spec = importlib.util.spec_from_file_location("r1_finest_criteria_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def study(runner, tmp_path):
    item = runner.Study.__new__(runner.Study)
    item.args = SimpleNamespace(case_filter="", max_cases=None, retry_failed=False)
    item.output, item.rows, item.attempted = tmp_path, [], 0
    item.started = "unit-regression"
    item.grids, item.controls, item.window = (16, 32, 64), ("D",), "full"
    item.source_unchanged = lambda: None
    return item


def test_coarse_difference_does_not_veto_converged_finest_pairs(runner, study):
    def current(item, key):
        value = {16: 2., 32: 1.004, 64: 1.}[item.intervals]
        return runner.R1Response(np.full((2, 2), value), {"time_s": np.array([0., 1.])},
                                 ("left", "right"))
    study.current_response = current
    _, numerical_error, audit = study.amplitude_error(.005)
    assert [x["passed"] for x in audit["axes"]["intervals"]["comparisons"]] == [False, True]
    assert audit["axes_passed"]
    np.testing.assert_allclose(numerical_error.values, .004, rtol=0, atol=1e-16)


def _failed_tail(runner, study, partial):
    prepared = SimpleNamespace(sha256="prepared", to_dict=lambda: {"state": {}})
    study.prepared = lambda n: prepared
    study.stack, study.binding = None, {"sha256": "reference"}
    study.record_dependency = lambda key: None
    path = study.output / "Long/N16/D/AttemptV1"
    path.mkdir(parents=True)
    failure = {"type": "R1RunError", "message": "recorded failure", "partial_result": partial}
    runner.write_json(path / "RequestV1.json", {"intervals": 16})
    runner.write_json(path / "FailureV1.json", failure)
    runner.write_json(path / "CompletionV1.json", {
        "case": "Long/N16/D", "status": "failed", "scientific_checks_passed": False,
        "failure": {k: value for k, value in failure.items() if k != "partial_result"}})
    runner.seal(path)
    return path


def _tail_payload():
    return {"prepared_sha256": "prepared", "reference_sha256": "reference", "control_label": "D",
            "amplitude_V": .005, "times_s": [0., 100.], "output_states": {},
            "regular_currents": [{"report_contact_current_A_m2": [1., 1.]}],
            "accepted_steps": [{"time_s": 100., "substeps": 4, "dt_s": 1.,
                                "physical_checks_passed": True, "state": {}}]}


def test_failed_tail_without_equation_replay_cannot_reach_dc(runner, study, monkeypatch):
    _failed_tail(runner, study, _tail_payload())
    monkeypatch.setattr(runner, "solve_controlled_dc", lambda *a, **k: pytest.fail("unverified prefix reached DC"))
    report = study.tail_record()
    assert report["comparison_available"] is False


def test_failed_tail_physics_rejection_propagates_before_dc(runner, study, monkeypatch):
    partial = _tail_payload()
    partial["physics_reconstruction"] = {"schema": "present"}
    _failed_tail(runner, study, partial)
    seen = []
    def reject(*args, **kwargs):
        seen.append(kwargs)
        raise ValueError("physical prefix mismatch")
    monkeypatch.setattr(runner, "verify_r1_step_physics", reject)
    monkeypatch.setattr(runner, "solve_controlled_dc", lambda *a, **k: pytest.fail("rejected prefix reached DC"))
    with pytest.raises(ValueError, match="physical prefix mismatch"):
        study.tail_record()
    assert seen == [{"allow_incomplete": True}]


def _record_case(runner, study, key, conditions):
    path = study.output / key / "AttemptV1"
    path.mkdir(parents=True)
    runner.write_json(path / "RequestV1.json", conditions)
    runner.write_json(path / "ResultV1.json", {})
    runner.write_json(path / "CompletionV1.json", {
        "case": key, "status": "completed", "scientific_checks_passed": True,
        "run_class": "formal", "scope": conditions.get("scope", "test")})
    runner.seal(path)
    study.expected_cases[key] = conditions


def test_verification_failure_vetoes_saved_study_success(runner, study):
    study.run_class = "formal"
    study.expected_cases = {}
    for case in runner.base_convergence_cases():
        _record_case(runner, study, study.matrix_key(case), asdict(case))
    for axis in ("intervals", "time_substeps", "nonlinear_factor"):
        _record_case(runner, study, "MatrixCompare/" + axis + "/finest", {
            "axis": axis, "control": "D", "required_finest_pair": True})
    _record_case(runner, study, "Linearity/selected", {"kind": "amplitude_linearity"})
    _record_case(runner, study, "DoubleDomain/selected", {"kind": "double_domain"})
    study.rows.append({"case": "invocation", "status": "failed", "reason": "physical verification rejected content"})
    assert study.finish() == 1
    summary = json.loads((study.output / "StudySummaryV1.json").read_text())
    assert summary["study_exit_passed"] is False
