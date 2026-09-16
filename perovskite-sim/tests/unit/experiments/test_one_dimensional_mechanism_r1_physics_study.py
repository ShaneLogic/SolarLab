"""Durable scientific-case semantics, independent of expensive device solves."""

import importlib.util
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[3]/"scripts/run_one_dimensional_mechanism_r1_physics_study.py"
    spec = importlib.util.spec_from_file_location("r1_physics_study_test_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def study(runner, tmp_path):
    value = runner.Study.__new__(runner.Study)
    value.args = SimpleNamespace(case_filter="", max_cases=None, retry_failed=False)
    value.output, value.rows, value.attempted = tmp_path, [], 0
    value.preparations, value.started = {}, "test"
    value.source_unchanged = lambda: None
    return value


def test_completed_comparison_failure_is_not_a_scientific_pass(runner, study):
    request = {"scope": "test-comparison"}
    result = study.case("Comparison", request, lambda directory: {"within_compared_budgets": False})
    assert result == {"within_compared_budgets": False}
    completion = runner.checked_read(study.latest("Comparison"), "CompletionV1.json")
    assert completion["status"] == "completed"
    assert completion["scientific_checks_passed"] is False
    assert study.finish() == 1
    index = json.loads((study.output/"FailureIndexV1.json").read_text())
    assert index["cases"][0]["conditions"] == request


def test_unknown_linearity_stays_unknown_and_not_study_acceptance(runner, study):
    study.case("Linearity", {"scope": "unknown-error"}, lambda directory: {"linearity_certified": False,
                                                                             "absolute_error": None})
    completion = runner.checked_read(study.latest("Linearity"), "CompletionV1.json")
    assert completion["scientific_checks_passed"] is None
    assert not completion["study_exit_passed"]


def test_failed_case_retains_raw_prefix_and_next_case_runs(runner, study):
    def fail(directory):
        (directory/"AcceptedStepsV1.jsonl").write_text('{"time_s":1}\n')
        error = RuntimeError("failed conserved current")
        error.result = {"accepted_steps": [{"time_s": 1.}], "certificate": {"certified": False}}
        raise error
    assert study.case("Failed", {"scope": "window"}, fail) is None
    failure = runner.checked_read(study.latest("Failed"), "FailureV1.json")
    assert failure["partial_result"]["accepted_steps"] == [{"time_s": 1.}]
    assert study.case("Next", {"scope": "window"}, lambda directory: {"ok": True}) == {"ok": True}
    assert study.rows[-1]["status"] == "completed"


def test_resume_skips_verified_result_and_rejects_modified_content(runner, study):
    request = {"scope": "resume"}
    operation = lambda directory: {"certificate": {"certified": True}}
    study.case("Case", request, operation)
    def must_not_run(directory):
        raise AssertionError("existing successful case was rerun")
    study.case("Case", request, must_not_run)
    assert study.rows[-1]["resumed"]
    (study.latest("Case")/"ResultV1.json").write_text('{}')
    with pytest.raises(ValueError, match="manifest"):
        study.case("Case", request, operation)


def test_failed_retry_preserves_earlier_attempt(runner, study):
    request = {"scope": "retry"}
    def fail(directory):
        raise ValueError("initial failure")
    study.case("Case", request, fail)
    original = (study.latest("Case")/"FailureV1.json").read_bytes()
    study.args.retry_failed = True
    study.case("Case", request, lambda directory: {"certificate": {"certified": True}})
    assert study.latest("Case").name == "AttemptV2"
    assert (study.output/"Case/AttemptV1/FailureV1.json").read_bytes() == original


def test_interrupted_attempt_is_preserved_and_resumes_into_new_attempt(runner, study):
    request = {"scope": "interrupted"}
    previous = study.output/"Case/AttemptV1"
    previous.mkdir(parents=True)
    runner.write_json(previous/"RequestV1.json", request)
    (previous/"AcceptedStepsV1.jsonl").write_text('{"partial":true}\n')
    study.case("Case", request, lambda directory: {"ok": True})
    assert study.latest("Case").name == "AttemptV2"
    assert (previous/"AcceptedStepsV1.jsonl").read_text() == '{"partial":true}\n'


def test_case_budget_does_not_start_extra_operation(study):
    study.args.max_cases = 1
    study.case("First", {"scope": "bounded"}, lambda directory: {})
    assert study.case("Second", {"scope": "bounded"}, lambda directory: pytest.fail("over budget")) is None
    assert study.rows[-1]["status"] == "not_started_case_budget"
    assert not (study.output/"Second").exists()


def test_serializer_preserves_complex_and_explains_nonfinite(runner):
    value = MappingProxyType({"ac": np.array([1+2j]), "unknown": float("inf")})
    result = json.loads(runner.raw_json(value))
    assert result["ac"] == [{"real": 1., "imag": 2.}]
    assert result["unknown"]["value"] is None
    assert result["unknown"]["reason"] == "nonfinite_numeric_evidence"


def test_negative_ac_point_is_a_scoped_failure_even_if_execution_completes(runner):
    assert runner.scientific_checks({"numerically_eligible_frequency_points": [True, False]}) is False
    assert runner.scientific_checks({"target_bias": {"certified": True},
                                     "conductance": {"finest_pair_agrees": False}}) is False


def test_dc_ac_section_alias_expands_without_starting_other_sections(runner, monkeypatch, tmp_path):
    seen = []
    class FakeStudy:
        def __init__(self, args):
            pass
        def run(self, sections):
            seen.extend(sections)
        def finish(self):
            return 0
    monkeypatch.setattr(runner, "Study", FakeStudy)
    assert runner.main(["--output-dir", str(tmp_path), "--section", "dc-ac"]) == 0
    assert seen == ["dc", "ac"]


def electrical_record(baseline):
    return {"times_s": [0., 1.], "policy": {"refinement_substeps": [1, 2, 4]},
            "regular_currents": [{"report_contact_current_A_m2": [baseline+2., baseline+2.]}]*2,
            "initial_event": {"impulse_charge_C_m2": .5},
            "accepted_steps": [{"substeps": 4, "time_s": 0., "regular_integrated_charge_C_m2": 0.},
                               {"substeps": 4, "time_s": 1., "regular_integrated_charge_C_m2": baseline+2.}]}


def test_electrical_comparison_removes_each_baseline_and_retains_impulse(runner):
    result = runner.compare_step_current_charge(electrical_record(.25), electrical_record(.5), [.25, .25], [.5, .5])
    assert result["within_compared_budgets"]
    changed = electrical_record(.5)
    changed["initial_event"]["impulse_charge_C_m2"] = 1.
    result = runner.compare_step_current_charge(electrical_record(.25), changed, [.25, .25], [.5, .5])
    assert result["regular_current"]["passed"]
    assert not result["integrated_charge"]["passed"]


def test_electrical_comparison_rejects_missing_or_duplicate_finest_times(runner):
    changed = electrical_record(0.)
    changed["accepted_steps"].append(changed["accepted_steps"][-1].copy())
    with pytest.raises(ValueError, match="exactly one"):
        runner.compare_step_current_charge(electrical_record(0.), changed, [0., 0.], [0., 0.])


def test_unavailable_dependencies_are_not_invented_numerical_failures(runner, study):
    result = study.case("Comparison", {"scope": "missing-inputs"},
                        lambda directory: study.compare_available(["MissingLeft", "MissingRight"], lambda: pytest.fail("no inputs")))
    assert not result["comparison_available"]
    completion = runner.checked_read(study.latest("Comparison"), "CompletionV1.json")
    assert completion["status"] == "unavailable"
    assert completion["scientific_checks_passed"] is None
    assert study.finish() == 2
    assert json.loads((study.output/"FailureIndexV1.json").read_text())["cases"] == []
    assert len(json.loads((study.output/"UnavailableComparisonIndexV1.json").read_text())["cases"]) == 1


def test_ac_mesh_comparison_checks_small_real_component_independently(runner, study):
    a = {"reference_sha256": "same", "control": "D", "voltage_V": 0., "frequency_Hz": [1.],
         "admittance_S_m2": [{"real": .001, "imag": 1e6}], "numerically_eligible_frequency_points": [True]}
    b = {**a, "admittance_S_m2": [{"real": .002, "imag": 1e6}]}
    study.saved = lambda key: a if "N16" in key else b
    result = study.ac_mesh_comparison(16, 32)
    assert not result["within_compared_budgets"]
    assert result["component_comparison"]["failure_count"] == 1


def test_resumed_passing_subset_does_not_hide_existing_failure(runner, study):
    def fail(directory):
        raise ValueError("real failure")
    study.case("Bad", {"scope": "same-study"}, fail)
    study.case("Good", {"scope": "same-study"}, lambda directory: {"certificate": {"certified": True}})
    study.rows.clear()
    study.case("Good", {"scope": "same-study"}, lambda directory: pytest.fail("must not execute"))
    assert study.finish() == 1
    assert json.loads((study.output/"StudySummaryV1.json").read_text())["active_case_count"] == 2


def test_resume_recomputes_comparison_instead_of_trusting_resealed_flag(runner, study):
    request = {"scope": "comparison"}
    operation = lambda directory: {"within_compared_budgets": False}
    study.case("Compare/Probe", request, operation)
    path = study.latest("Compare/Probe")
    result = runner.checked_read(path, "ResultV1.json")
    result["within_compared_budgets"] = True
    completion = runner.checked_read(path, "CompletionV1.json")
    completion["scientific_checks_passed"] = True
    runner.write_json(path/"ResultV1.json", result)
    runner.write_json(path/"CompletionV1.json", completion)
    runner.seal(path)
    with pytest.raises(ValueError, match="independent replay"):
        study.case("Compare/Probe", request, operation)


def test_checked_read_rejects_unsealed_extra_file_and_symlink(runner, study):
    study.case("Example", {"scope": "test"}, lambda directory: {"ok": True})
    path = study.latest("Example")
    (path/"extra.json").write_text('{}')
    with pytest.raises(ValueError, match="coverage"):
        runner.checked_read(path, "ResultV1.json")
    (path/"extra.json").unlink()
    (path/"alias").symlink_to(path/"ResultV1.json")
    runner.seal(path)
    with pytest.raises(ValueError, match="manifest"):
        runner.checked_read(path, "ResultV1.json")


def test_physics_recomputation_failure_propagates_out_of_step(runner, study, monkeypatch, tmp_path):
    prepared = SimpleNamespace(sha256="test-parent")
    study.prepared = lambda n: prepared
    study.stack, study.binding = None, None
    record = {"certificate": {"certified": True}}
    monkeypatch.setattr(runner, "run_r1_step", lambda *a, **kw: record)
    monkeypatch.setattr(runner, "verify_r1_step_physics", lambda *a, **kw: {
        "certified": False, "content_matches_recomputed": True, "violations": ["current"]})
    assert study.case("Probe", {"scope": "physics"}, lambda directory: study.step(
        directory, 16, "D", (1, 2, 4), .1, (0., 1e-9))) is None
    completion = runner.checked_read(study.latest("Probe"), "CompletionV1.json")
    assert completion["status"] == "failed"
    assert not completion["scientific_checks_passed"]
    assert study.finish() == 1
