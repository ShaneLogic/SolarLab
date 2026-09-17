"""Independent analytic cases for scoped response qualification, not device evidence."""

import copy

import numpy as np
import pytest

from perovskite_sim.experiments.one_dimensional_mechanism_r1_convergence import step_current_charge_responses
from perovskite_sim.experiments.one_dimensional_mechanism_r1_qualification import (
    DOUBLE_DOMAIN_PREREQUISITES, assess_current_error_budget, assess_double_domain_prerequisites,
    assess_transient_linearity, evidence_digest, qualification_scope, select_response_amplitude,
)
from perovskite_sim.experiments.one_dimensional_mechanism_r1_response import descriptor_frequency_response
from perovskite_sim.experiments.one_dimensional_mechanism_r1_study_response import frequency_window_report
from tests.unit.experiments.test_one_dimensional_mechanism_r1_response import synthetic_ac_record


def identity():
    return {"source": {"sha256": "analytic-source"}, "prepared_sha256": "analytic-prepared",
            "reference_sha256": "analytic-reference", "intervals": 16, "control_label": "D"}


def scope():
    return qualification_scope(identity(), state_sha256="same-initial-state", domain="analytic_fixture")


def step(amplitude=.005, response=(2., 2., 2.)):
    times = [0., .5, 1.]
    return {**identity(), "schema": "R1ControlledStepV1", "amplitude_V": amplitude,
            "times_s": times, "voltage_V": [amplitude]*3, "certificate": {"certified": True},
            "policy": {"refinement_substeps": [1, 2, 4]},
            "regular_currents": [{"report_contact_current_A_m2": [amplitude*g]*2} for g in response],
            "accepted_steps": [{"substeps": 4, "time_s": t, "regular_integrated_charge_C_m2": amplitude*2*t} for t in times],
            "initial_event": {"impulse_charge_C_m2": 0., "zero_minus": {"voltage_V": 0.},
                              "zero_plus": {"voltage_V": amplitude}, "voltage_jump_V": amplitude}}


def current_budget(record, name):
    response = step_current_charge_responses(record, [0., 0.])[0]
    return {"schema": "R1CurrentErrorBudgetV1", "evidence_id": name, "scope": scope(),
            "classification": "bounded", "units": "A/m2", "amplitude_V": record["amplitude_V"],
            "method": "independent exact analytic response", "propagation_method": "sum_nonnegative_absolute_bounds",
            "source_evidence": ["analytic-response-formula-and-roundoff-proof"],
            "coordinates": {"time_s": record["times_s"]}, "components": ["left_contact", "right_contact"],
            "response_sha256": evidence_digest(response), "required_terms": ["analytic_oracle", "evaluation"],
            "terms_A_m2": {"analytic_oracle": np.zeros((3, 2)).tolist(), "evaluation": np.full((3, 2), 1e-12).tolist()},
            "review_status": "approved", "review_id": "independent-analytic-oracle-test"}


def linearity_inputs(coarse=None, fine=None):
    coarse = step() if coarse is None else coarse
    fine = step(.0025) if fine is None else fine
    a, b = current_budget(coarse, "coarse-budget"), current_budget(fine, "fine-budget")
    trust = {a["evidence_id"]: evidence_digest(a), b["evidence_id"]: evidence_digest(b),
             "verified-coarse": evidence_digest(coarse), "verified-fine": evidence_digest(fine)}
    kwargs = dict(coarse_baseline_A_m2=[0., 0.], fine_baseline_A_m2=[0., 0.], expected_scope=scope(),
                  expected_times_s=[0., .5, 1.], coarse_budget=a, fine_budget=b, trusted_evidence=trust)
    return coarse, fine, kwargs


def test_missing_budget_is_unknown_and_never_replaced_with_zero():
    record = step()
    response = step_current_charge_responses(record, [0., 0.])[0]
    result = assess_current_error_budget(None, response, expected_scope=scope(), amplitude_V=.005)
    assert result["status"] == "unknown" and result["bound_A_m2"] is None
    assert not result["qualified"]
    budget = current_budget(record, "budget")
    result = assess_current_error_budget(budget, response, expected_scope=scope(), amplitude_V=.005)
    assert result["status"] == "unapproved" and not result["qualified"]


def test_runtime_integer_refinement_keys_match_saved_json_without_collisions():
    assert evidence_digest({"charge": {1: 2., 2: 3.}}) == evidence_digest({"charge": {"1": 2., "2": 3.}})
    with pytest.raises(ValueError, match="collide"):
        evidence_digest({1: 2., "1": 3.})
    with pytest.raises(ValueError, match="keys must"):
        evidence_digest({True: 1.})


@pytest.mark.parametrize("change,reason", [
    (lambda b: b.update(units="S/m2"), "units"),
    (lambda b: b.update(amplitude_V=.0025), "amplitude"),
    (lambda b: b["scope"].update(state_sha256="other-initial-state"), "scope"),
    (lambda b: b["coordinates"].update(time_s=[0., .75, 1.]), "coordinates"),
    (lambda b: b["terms_A_m2"].pop("evaluation"), "error term"),
    (lambda b: b["terms_A_m2"].update(evaluation=[[-1e-12]*2]*3), "nonnegative"),
    (lambda b: b["terms_A_m2"].update(evaluation=[0., 0.]), "shape"),
    (lambda b: b.update(response_sha256="borrowed-response"), "different response"),
])
def test_budget_domain_units_terms_and_exact_samples_are_checked_even_with_a_digest(change, reason):
    record = step()
    response = step_current_charge_responses(record, [0., 0.])[0]
    budget = current_budget(record, "budget")
    change(budget)
    result = assess_current_error_budget(budget, response, expected_scope=scope(), amplitude_V=.005,
                                         trusted_evidence={"budget": evidence_digest(budget)})
    assert result["status"] == "invalid" and not result["qualified"]
    assert reason in result["reasons"][0]


@pytest.mark.parametrize("classification", ["estimate_only", "unknown"])
def test_review_does_not_turn_an_estimate_or_unknown_into_a_bound(classification):
    record = step()
    response = step_current_charge_responses(record, [0., 0.])[0]
    budget = current_budget(record, "budget")
    budget["classification"] = classification
    result = assess_current_error_budget(budget, response, expected_scope=scope(), amplitude_V=.005,
                                         trusted_evidence={"budget": evidence_digest(budget)})
    assert not result["qualified"]
    assert result["classification"] == classification
    if classification == "unknown":
        assert result["bound_A_m2"] is None


def test_linear_analytic_history_qualifies_only_its_fixture_domain():
    a, b, kwargs = linearity_inputs()
    result = assess_transient_linearity(a, b, **kwargs)
    assert result["qualified"] and result["full_transient_linearity_certified"]
    assert result["comparison"]["scalar_comparison_count"] == 6
    assert not result["device_qualified"]
    selection = select_response_amplitude([result], expected_scope=scope(),
                                          verified_report_digests=[evidence_digest(result)])
    assert selection["qualified_amplitude_V"] == .0025
    assert selection["diagnostic_amplitude_V"] == .005 and not selection["device_qualified"]
    assert select_response_amplitude([result], expected_scope=scope())["qualified_amplitude_V"] is None
    device = {**scope(), "domain": "device"}
    assert select_response_amplitude([result], expected_scope=device,
                                     verified_report_digests=[evidence_digest(result)])["qualified_amplitude_V"] is None


def test_equal_endpoints_do_not_hide_nonlinearity_at_an_interior_time():
    a, b, kwargs = linearity_inputs(coarse=step(response=(2., 3., 2.)))
    result = assess_transient_linearity(a, b, **kwargs)
    assert result["status"] == "response_not_linear" and not result["qualified"]
    assert result["diagnostic"]["absolute_difference_S_m2"][0] == [0., 0.]
    assert result["diagnostic"]["absolute_difference_S_m2"][-1] == [0., 0.]
    assert {f["coordinates"]["time_s"] for f in result["comparison"]["failures"]} == {.5}
    assert {f["component"] for f in result["comparison"]["failures"]} == {"left_contact", "right_contact"}


def test_zero_signal_and_insufficient_error_resolution_cannot_certify_linearity():
    a, b, kwargs = linearity_inputs(coarse=step(response=(0., 0., 0.)), fine=step(.0025, response=(0., 0., 0.)))
    result = assess_transient_linearity(a, b, **kwargs)
    assert result["status"] == "linearity_undetermined" and not result["qualified"]
    a, b, kwargs = linearity_inputs()
    for name in ("coarse_budget", "fine_budget"):
        kwargs[name]["terms_A_m2"]["evaluation"] = np.full((3, 2), .01).tolist()
        kwargs["trusted_evidence"][kwargs[name]["evidence_id"]] = evidence_digest(kwargs[name])
    assert assess_transient_linearity(a, b, **kwargs)["status"] == "linearity_undetermined"


def test_complete_time_axis_verified_inputs_and_budgets_are_independent_requirements():
    a, b, kwargs = linearity_inputs()
    kwargs["coarse_budget"] = None
    result = assess_transient_linearity(a, b, **kwargs)
    assert not result["qualified"] and result["comparison"] is None
    a, b, kwargs = linearity_inputs()
    kwargs["trusted_evidence"].pop("verified-coarse")
    assert assess_transient_linearity(a, b, **kwargs)["status"] == "unqualified_inputs"
    a, b, kwargs = linearity_inputs()
    a["times_s"] = [0., 1.]
    assert assess_transient_linearity(a, b, **kwargs)["status"] == "invalid"
    a, b, kwargs = linearity_inputs()
    a["certificate"]["certified"] = False
    kwargs["trusted_evidence"]["verified-coarse"] = evidence_digest(a)
    assert assess_transient_linearity(a, b, **kwargs)["status"] == "unqualified_inputs"


def pole_record(frequency=None):
    frequency = np.r_[0., np.logspace(-2, 2, 17)] if frequency is None else np.asarray(frequency)
    tau = 1/(2*np.pi)  # Analytically known turnover: 1 Hz.
    pole = descriptor_frequency_response(frequency, storage=[[tau]], rate=[[-1.]], forcing=[1.],
        storage_voltage=[0.], conduction=[[0.]], conduction_voltage=[.125],
        displacement=[[.03125]], displacement_voltage=[.0625])
    expected = .125+2j*np.pi*frequency*.0625+2j*np.pi*frequency*.03125/(1+2j*np.pi*frequency*tau)
    np.testing.assert_allclose(pole["admittance_S_m2"][:, 0], expected, rtol=1e-14, atol=1e-14)
    ac = synthetic_ac_record()
    ac.update({**identity(), "control": "D", "frequency_Hz": frequency})
    ac.pop("control_label")
    for level in ac["derivative_levels"]:
        for key, value in tuple(level.items()):
            if isinstance(value, np.ndarray) and value.shape[0] == 3:
                level[key] = np.repeat(value[:1], len(frequency), axis=0)
        level["frequency_Hz"] = frequency.copy()
        for key in ("conduction_S_m2", "displacement_F_m2", "admittance_S_m2"):
            level[key] = np.repeat(pole[key], 2, axis=1)
        level["electron_admittance_S_m2"] = level["conduction_S_m2"]/2
        level["hole_admittance_S_m2"] = level["conduction_S_m2"]/2
    ac["admittance_S_m2"] = expected
    # Save the actual phasor arithmetic used by the numeric-content assessor.
    ac["admittance_S_m2"] = ac["derivative_levels"][-1]["admittance_S_m2"][:, 0].copy()
    evidence = {"schema": "R1TurnoverEvidenceV1", "evidence_id": "analytic-one-pole", "scope": scope(),
                "units": "Hz", "turnover_frequency_Hz": [1.], "coverage": "all_applicable_modes",
                "unresolved_modes": [], "review_status": "approved", "review_id": "analytic-pole-identity",
                "ac_sha256": evidence_digest(ac)}
    return ac, evidence


def frequency_report(ac, evidence):
    return frequency_window_report(ac, turnover_evidence=evidence, expected_scope=scope(),
                                   trusted_evidence={evidence["evidence_id"]: evidence_digest(evidence)})


def test_analytic_single_pole_has_a_true_evidence_driven_coverage_path():
    ac, evidence = pole_record()
    report = frequency_report(ac, evidence)
    assert report["frequency_window_complete"]
    assert report["qualification"]["status"] == "certified"
    assert not report["device_frequency_window_certified"]
    assert not report["uncovered_reasons"]
    assert not frequency_window_report(ac)["frequency_window_complete"]


@pytest.mark.parametrize("frequency", [np.r_[0., np.logspace(-.5, 2, 11)],
                                       np.r_[0., np.logspace(-2, .5, 11)],
                                       np.r_[0., np.logspace(-2, 2, 9)]])
def test_turnover_truncation_missing_margin_or_sparse_sampling_prevents_coverage(frequency):
    ac, evidence = pole_record(frequency)
    report = frequency_report(ac, evidence)
    assert report["qualification"]["status"] == "incomplete"
    assert not report["frequency_window_complete"]


def test_approved_turnover_from_another_state_and_a_bad_middle_frequency_do_not_pass():
    ac, evidence = pole_record()
    evidence["scope"]["state_sha256"] = "different-state"
    assert frequency_report(ac, evidence)["qualification"]["status"] == "invalid"
    ac, evidence = pole_record()
    ac["derivative_levels"][0]["linear_backward_error"][8] = 2e-10
    evidence["ac_sha256"] = evidence_digest(ac)
    report = frequency_report(ac, evidence)
    assert not report["frequency_window_complete"]
    assert len(report["valid_contiguous_bands"]) == 2
    assert "one_or_more_frequency_points_failed_numeric_checks" in report["uncovered_reasons"]
    ac, evidence = pole_record()
    ac["voltage_V"] = 1e-300
    assert frequency_report(ac, evidence)["qualification"]["status"] == "invalid"


def double_evidence():
    frequency = [0., 1., 10.]
    application = {"step_sha256": "step", "ac_sha256": "ac", "conductance_sha256": "dc",
                   "step_amplitude_V": .0025, "times_s": [0., .5, 1.], "reconstruction_request_sha256": "integration-request"}
    evidence = {kind: {"schema": "R1QualificationEvidenceV1", "evidence_id": kind, "kind": kind,
                       "scope": scope(), "application": copy.deepcopy(application), "frequency_Hz": frequency,
                       "eligible_frequency_points": [True]*3, "qualified": True, "classification": "bounded",
                       "review_status": "approved", "review_id": "analytic-prerequisite-proof"}
                for kind in DOUBLE_DOMAIN_PREREQUISITES}
    kwargs = dict(expected_scope=scope(), frequency_Hz=frequency, expected_application=application,
                  trusted_evidence={kind: evidence_digest(item) for kind, item in evidence.items()})
    return evidence, kwargs


@pytest.mark.parametrize("missing", DOUBLE_DOMAIN_PREREQUISITES)
def test_every_double_domain_prerequisite_is_required(missing):
    evidence, kwargs = double_evidence()
    good = assess_double_domain_prerequisites(evidence, **kwargs)
    assert good["qualified"] and not good["device_qualified"]
    evidence.pop(missing)
    result = assess_double_domain_prerequisites(evidence, **kwargs)
    assert not result["qualified"] and missing in result["missing_prerequisites"]
    assert not any(result["eligible_frequency_points"])


def test_double_domain_evidence_cannot_be_borrowed_for_another_amplitude_or_frequency():
    evidence, kwargs = double_evidence()
    evidence["finite_amplitude_linearity"]["application"]["step_amplitude_V"] = .005
    kwargs["trusted_evidence"]["finite_amplitude_linearity"] = evidence_digest(evidence["finite_amplitude_linearity"])
    result = assess_double_domain_prerequisites(evidence, **kwargs)
    assert "finite_amplitude_linearity" in result["invalid_prerequisites"]
    evidence, kwargs = double_evidence()
    evidence["tail_omission"]["classification"] = "estimate_only"
    kwargs["trusted_evidence"]["tail_omission"] = evidence_digest(evidence["tail_omission"])
    assert not assess_double_domain_prerequisites(evidence, **kwargs)["qualified"]
    evidence, kwargs = double_evidence()
    kwargs["trusted_evidence"] = {}
    assert not assess_double_domain_prerequisites(evidence, **kwargs)["qualified"]
