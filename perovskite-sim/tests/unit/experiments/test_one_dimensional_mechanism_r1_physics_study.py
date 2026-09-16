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


def test_dc_amplitude_section_is_bounded_to_requested_grids_and_controls(runner, study):
    seen = []
    study.grids, study.controls = (16, 32), ("D", "B")
    study.case = lambda key, request, operation: seen.append((key, request))
    study.run(["amplitude-dc"])
    assert [key for key, _ in seen] == ["AmplitudeDC/N16/D", "AmplitudeDC/N16/B",
                                       "AmplitudeDC/N32/D", "AmplitudeDC/N32/B"]
    for _, request in seen:
        assert request["operating_voltage_V"] == 0.
        assert request["amplitudes_V"] == list(runner.AMPLITUDES_V)
        assert request["kind"] == "dc_amplitude_endpoints"


def test_dc_amplitude_scientific_pass_is_only_the_dc_diagnostic(runner):
    result = {"schema": "R1DCEndpointAmplitudeStudyV1", "dc_states_certified": True,
              "linearity_certified": False, "full_transient_linearity_certified": False}
    assert runner.scientific_checks(result)
    assert not runner.scientific_checks({**result, "linearity_certified": True})
    assert not runner.scientific_checks({**result, "dc_states_certified": False})


def test_dc_amplitude_audit_routes_to_independent_response_replay(runner, study, monkeypatch):
    import perovskite_sim.experiments.one_dimensional_mechanism_r1_response as response
    seen = []
    def verify(record, **kwargs):
        seen.append((record, kwargs))
        return {"certified": True, "content_matches_recomputed": True}
    monkeypatch.setattr(response, "verify_response_content", verify)
    study.stack, study.binding = "stack", "binding"
    study.prepared = lambda n: ("prepared", n)
    request = {"intervals": 16, "control": "D", "operating_voltage_V": 0.,
               "amplitudes_V": list(runner.AMPLITUDES_V)}
    record = {"schema": "R1DCEndpointAmplitudeStudyV1"}
    assert study.audit_result("AmplitudeDC/N16/D", request, record)["content_matches_recomputed"]
    assert seen[0][1]["request"] is request
    assert seen[0][1]["prepared"] == ("prepared", 16)


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


@pytest.mark.parametrize("physical_pass", [True, False], ids=["success", "physical-gate-failure"])
def test_full_window_numpy_times_seal_success_and_physical_failure(runner, study, monkeypatch, physical_pass):
    times = runner.observation_times()
    assert isinstance(times, np.ndarray) and len(times) == 134
    assert times[0] == 0. and times[1] == 1e-9 and times[-1] == 100.
    request = {"intervals": 16, "control": "D", "time_substeps": (1, 2, 4),
               "nonlinear_factor": .1, "times_s": times, "scope": "full_window_independent_axis_case"}
    prepared = SimpleNamespace(sha256="test-parent")
    study.prepared = lambda n: prepared
    study.stack, study.binding = None, None
    endpoint = {"time_s": times[-1], "substeps": 4}
    record = {"times_s": times, "certificate": {"certified": physical_pass},
              "accepted_steps": [endpoint]}
    audit = {"certified": physical_pass, "content_matches_recomputed": True,
             "violations": [] if physical_pass else ["eliminated_operator_error"]}

    def simulate(*args, **kwargs):
        np.testing.assert_array_equal(kwargs["times_s"], times)
        kwargs["accepted_step_observer"](endpoint)
        return record

    monkeypatch.setattr(runner, "run_r1_step", simulate)
    monkeypatch.setattr(runner, "verify_r1_step_physics", lambda *a, **kw: audit)
    key = "Matrix/N16/T1/F0p1"
    result = study.case(key, request, lambda directory: study.step(directory, 16, "D", (1, 2, 4), .1, times))
    path = study.latest(key)
    saved_request = runner.checked_read(path, "RequestV1.json")
    completion = runner.checked_read(path, "CompletionV1.json")
    assert saved_request["times_s"] == times.tolist()
    assert completion["status"] == ("completed" if physical_pass else "failed")
    assert completion["scientific_checks_passed"] is physical_pass
    assert completion["study_exit_passed"] is False
    historical = completion["historical_observation"]
    assert historical["matched_conditions"]["times_s"] == times.tolist()
    assert all(type(value) is float for value in historical["matched_conditions"]["times_s"])
    assert historical["waives_checks"] is False
    assert historical["changes_acceptance_thresholds"] is False
    assert runner.checked_read(path, "PhysicsRecomputationV1.json") == audit
    assert json.loads((path/"AcceptedStepsV1.jsonl").read_text()) == runner.ready(endpoint)
    assert study.produced_cases[key] == runner.sha(path/"ManifestV1.json")
    assert request["times_s"] is times
    assert request["time_substeps"] == (1, 2, 4)
    if physical_pass:
        assert result is record
        assert runner.checked_read(path, "ResultV1.json") == runner.ready(record)
    else:
        assert result is None
        failure = runner.checked_read(path, "FailureV1.json")
        assert failure["message"] == "completed trajectory fails reconstructed physical gates"
        assert failure["partial_result"] == runner.ready(record)
        assert completion["failure"] == {k: v for k, v in failure.items() if k != "partial_result"}


@pytest.mark.parametrize("times", [
    np.array([0., np.nan]), np.array([0., np.inf]), np.array([0., 0.]),
    np.array([0., -1.]), np.array([1., 2.]), np.array([0.]),
    np.array([False, True]), np.array(["0", "1"]), np.array([[0., 1.], [2., 3.]]),
], ids=["nan", "infinite", "duplicate", "negative", "missing-zero", "too-short", "boolean", "text", "matrix"])
def test_case_numpy_conversion_does_not_accept_invalid_observation_times(runner, study, times):
    study.prepared = lambda n: None
    request = {"intervals": 16, "control": "D", "time_substeps": (1, 2, 4),
               "nonlinear_factor": .1, "times_s": times, "scope": "invalid-time-probe"}
    with pytest.raises(ValueError, match="invalid failure registry observation times"):
        study.case("InvalidTimes", request, lambda directory: {"certificate": {"certified": True}})
    assert not (study.latest("InvalidTimes")/"CompletionV1.json").exists()
    assert not (study.latest("InvalidTimes")/"ManifestV1.json").exists()
