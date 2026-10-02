"""Scoped V4 decision tests; synthetic device-domain cases are not device evidence."""

import copy
from dataclasses import asdict

import numpy as np
import pytest

from perovskite_sim.experiments.one_dimensional_mechanism_r1_convergence import (
    R1Response, observation_times, step_current_charge_responses,
)
from perovskite_sim.experiments.one_dimensional_mechanism_r1_qualification import (
    DOUBLE_DOMAIN_PREREQUISITES, assess_current_error_budget, assess_double_domain_prerequisites,
    assess_transient_linearity, device_response_gate, evidence_digest, qualification_scope,
    select_response_amplitude,
)
from perovskite_sim.experiments.one_dimensional_mechanism_r1_study_response import reconstruct_study_response
from perovskite_sim.experiments.one_dimensional_mechanism_r1_window import (
    build_window_spec, validate_window_relation, validate_window_spec, window_spec_digest,
)
from tests.unit.experiments.test_r1_response_qualification import step
from tests.unit.experiments.test_one_dimensional_mechanism_r1_study_response import inputs as reconstruction_inputs


# Independent protocol expectation. Do not derive this test inventory from
# the production tuple whose deletion this test must detect.
REQUIRED = (
    "finite_amplitude_linearity", "single_axis_convergence", "window_extension", "earlier_start",
    "stricter_integration", "tail_dc_agreement", "frequency_window_coverage", "input_trajectory_certified",
    "ac_content_verified", "current_error_budget", "early_omission", "tail_omission", "interpolation",
    "impulse_charge", "baseline_current_error", "dc_conductance_error",
)


@pytest.mark.parametrize("first,last", [(1e-9, 100.), (1e-9, 1000.), (1e-10, 100.), (1e-12, 1e5)])
def test_device_window_preserves_legal_default_and_extended_grids(first, last):
    times = observation_times(first_time_s=first, last_time_s=last)
    window = build_window_spec(times)
    assert validate_window_spec(window) == window
    assert window_spec_digest(window) == evidence_digest(window)
    assert window["first_time_s"] == first and window["last_time_s"] == last
    assert window["sampling"]["intervals_per_decade"] == 12
    detached = validate_window_spec(window)
    detached["times_s"][2] += 1.
    np.testing.assert_array_equal(window["times_s"], times)


@pytest.mark.parametrize("fault", ["missing", "extra", "wrong_tick", "sampling", "zero", "endpoint", "domain"])
def test_window_cannot_redefine_coverage_using_its_own_metadata(fault):
    window = build_window_spec(observation_times())
    if fault == "missing":
        window["times_s"].pop(40)
    elif fault == "extra":
        window["unreviewed_rule"] = True
    elif fault == "wrong_tick":
        window["times_s"][40] *= 1.00001
    elif fault == "sampling":
        window["sampling"]["intervals_per_decade"] = 4
    elif fault == "zero":
        window["zero_time_definition"] = "zero_minus"
    elif fault == "endpoint":
        window["last_time_s"] = 1000.
    else:
        window["domain"] = "analytic_fixture"
    with pytest.raises(ValueError):
        validate_window_spec(window)


@pytest.mark.parametrize("times", [[0., 1e-9, 1e-4], [0., True, 2.], [0., 1., np.nan], [0., 1., 1.], [1., 2., 3.]])
def test_short_or_invalid_axes_cannot_be_device_full_windows(times):
    with pytest.raises(ValueError):
        build_window_spec(times)
    if times == [0., 1e-9, 1e-4]:
        assert build_window_spec(times, "analytic_fixture")["domain"] == "analytic_fixture"


def test_window_relations_preserve_exact_shared_ticks_and_declared_direction():
    base = build_window_spec(observation_times())
    extended = build_window_spec(observation_times(last_time_s=1000.))
    earlier = build_window_spec(observation_times(first_time_s=1e-10))
    assert validate_window_relation(base, extended, relation="tenfold_extension")["verified"]
    assert validate_window_relation(base, earlier, relation="earlier_start")["verified"]
    with pytest.raises(ValueError):
        validate_window_relation(base, earlier, relation="tenfold_extension")
    with pytest.raises(ValueError):
        validate_window_relation(extended, base, relation="tenfold_extension")


def linearity_case(*, first=1e-9, last=100., classification="validated_estimate"):
    times = observation_times(first_time_s=first, last_time_s=last)
    records = []
    for amplitude in (.005, .0025):
        record = step(amplitude)
        record.update(times_s=times.tolist(), voltage_V=[amplitude]*len(times),
            regular_currents=[{"report_contact_current_A_m2": [2*amplitude]*2} for _ in times],
            accepted_steps=[{"substeps": 4, "time_s": float(t), "regular_integrated_charge_C_m2": 2*amplitude*float(t)}
                            for t in times])
        records.append(record)
    window = build_window_spec(times)
    scope = qualification_scope(records[0], state_sha256="checked-synthetic-initial-state", window_spec=window)
    responses = [step_current_charge_responses(record, [0., 0.])[0] for record in records]
    budgets = [{"schema": "R1CurrentErrorBudgetV1", "evidence_id": "budget-"+str(i), "scope": scope,
                "classification": classification, "units": "A/m2", "amplitude_V": records[i]["amplitude_V"],
                "method": "synthetic independently checked error model", "source_evidence": ["synthetic-known-solution"],
                "propagation_method": "sum_nonnegative_absolute_uncertainties",
                "coordinates": {"time_s": times.tolist()}, "components": list(response.components),
                "response_sha256": evidence_digest(response), "required_terms": ["evaluation"],
                "terms_A_m2": {"evaluation": np.full(response.values.shape, 1e-12).tolist()},
                "review_status": "approved", "review_id": "unit-test-review"}
               for i, response in enumerate(responses)]
    errors = [R1Response(np.zeros(response.values.shape), response.coordinates, response.components) for response in responses]
    kwargs = dict(coarse_baseline_A_m2=[0., 0.], fine_baseline_A_m2=[0., 0.], expected_scope=scope,
                  expected_times_s=times, window_spec=window, declared_amplitudes_V=[.01, .005, .0025],
                  coarse_budget=budgets[0], fine_budget=budgets[1],
                  trusted_evidence={budget["evidence_id"]: evidence_digest(budget) for budget in budgets},
                  verified_record_digests={name: evidence_digest(record) for name, record in zip(("coarse_step", "fine_step"), records)},
                  measured_evidence={"coarse_current_error": errors[0], "fine_current_error": errors[1],
                                     "coarse_axes_passed": True, "fine_axes_passed": True})
    return records, responses, kwargs


@pytest.mark.parametrize("first,last", [(1e-9, 100.), (1e-9, 1000.), (1e-10, 100.), (1e-12, 1e5)])
def test_linearity_accepts_approved_numerical_estimates_on_the_exact_external_window(first, last):
    records, _, kwargs = linearity_case(first=first, last=last)
    result = assess_transient_linearity(*records, **kwargs)
    assert result["qualified"] and result["device_qualified"] and result["complete_time_scope"]
    assert result["coarse_budget"]["classification"] == "validated_estimate"
    assert result["coarse_budget"]["bound_A_m2"] is None
    assert not result["coarse_budget"]["rigorous_bound"]
    selection = select_response_amplitude([result], expected_scope=kwargs["expected_scope"],
        window_spec=kwargs["window_spec"], declared_amplitudes_V=kwargs["declared_amplitudes_V"],
        verified_report_digests=[evidence_digest(result)])
    assert selection["device_qualified"] and selection["qualified_amplitude_V"] == .0025


@pytest.mark.parametrize("fault", ["window_missing", "other_window", "record_missing", "wrong_role", "larger_pair",
                                   "ladder_missing", "measured_missing", "estimate_only", "unknown"])
def test_device_qualification_requires_each_external_and_measured_condition(fault):
    records, _, kwargs = linearity_case()
    if fault == "window_missing":
        kwargs["window_spec"] = None
    elif fault == "other_window":
        kwargs["window_spec"] = build_window_spec(observation_times(last_time_s=1000.))
    elif fault == "record_missing":
        kwargs["verified_record_digests"] = None
        kwargs["trusted_evidence"].update({"unrelated-a": evidence_digest(records[0]), "unrelated-b": evidence_digest(records[1])})
    elif fault == "wrong_role":
        kwargs["verified_record_digests"] = {"arbitrary": evidence_digest(records[0]), "other": evidence_digest(records[1])}
    elif fault == "larger_pair":
        kwargs["declared_amplitudes_V"] = [.01, .005, .0025, .00125]
    elif fault == "ladder_missing":
        kwargs["declared_amplitudes_V"] = None
    elif fault == "measured_missing":
        kwargs["measured_evidence"] = None
    else:
        kwargs["coarse_budget"]["classification"] = fault
        kwargs["trusted_evidence"]["budget-0"] = evidence_digest(kwargs["coarse_budget"])
    result = assess_transient_linearity(*records, **kwargs)
    assert not result["qualified"] and not result["device_qualified"]


def test_smaller_reviewed_error_cannot_silently_override_measured_resolution_failure():
    records, responses, kwargs = linearity_case()
    kwargs["measured_evidence"]["coarse_current_error"] = R1Response(
        np.full(responses[0].values.shape, .0005), responses[0].coordinates, responses[0].components)
    result = assess_transient_linearity(*records, **kwargs)
    assert result["comparison"]["passed"]  # Tiny declared uncertainty alone would pass.
    assert not result["qualified"]
    consistency = result["measured_consistency"]
    assert consistency["comparison"]["status"] == "linearity_undetermined"
    evidence = {"schema": "R1NumericalEvidenceReconciliationV1", "evidence_id": "reconciliation",
                "scope": kwargs["expected_scope"], "measured_evidence_sha256": consistency["measured_evidence_sha256"],
                "budget_evidence_sha256": [evidence_digest(kwargs[key]) for key in ("coarse_budget", "fine_budget")],
                "response_sha256": [evidence_digest(response) for response in responses],
                "resolved_concerns": consistency["concerns"], "decision": "supersedes_empirical_estimate",
                "method": "independent exact synthetic solution", "explanation": "coarse comparison overestimates the finest error",
                "source_evidence": ["exact-synthetic-solution-and-roundoff"], "review_status": "approved", "review_id": "independent-review"}
    kwargs["reconciliation_evidence"] = evidence
    assert not assess_transient_linearity(*records, **kwargs)["qualified"]
    kwargs["trusted_evidence"]["reconciliation"] = evidence_digest(evidence)
    explained = assess_transient_linearity(*records, **kwargs)
    assert explained["qualified"]
    assert explained["coarse_budget"]["classification"] == "validated_estimate"
    evidence["response_sha256"][0] = "borrowed-response"
    kwargs["trusted_evidence"]["reconciliation"] = evidence_digest(evidence)
    assert not assess_transient_linearity(*records, **kwargs)["qualified"]


def test_selector_cannot_relabel_a_short_fixture_or_select_the_largest_ladder_level():
    records, _, kwargs = linearity_case()
    report = assess_transient_linearity(*records, **kwargs)
    for change in ({"times_s": [0., .5, 1.]}, {"fine_amplitude_V": .01}, {"complete_time_scope": False}):
        altered = {**report, **change}
        result = select_response_amplitude([altered], expected_scope=kwargs["expected_scope"],
            window_spec=kwargs["window_spec"], declared_amplitudes_V=kwargs["declared_amplitudes_V"],
            verified_report_digests=[evidence_digest(altered)])
        assert not result["qualified"]


def test_reviewed_uncertainty_cannot_waive_an_actual_failed_numerical_axis():
    records, responses, kwargs = linearity_case()
    kwargs["measured_evidence"]["coarse_axes_passed"] = False
    result = assess_transient_linearity(*records, **kwargs)
    consistency = result["measured_consistency"]
    assert result["comparison"]["passed"]
    assert result["status"] == "measured_convergence_failed"
    evidence = {"schema": "R1NumericalEvidenceReconciliationV1", "evidence_id": "cannot-waive-axis",
        "scope": kwargs["expected_scope"], "measured_evidence_sha256": consistency["measured_evidence_sha256"],
        "budget_evidence_sha256": [evidence_digest(kwargs[key]) for key in ("coarse_budget", "fine_budget")],
        "response_sha256": [evidence_digest(response) for response in responses],
        "resolved_concerns": consistency["concerns"], "decision": "supersedes_empirical_estimate",
        "method": "synthetic bound", "explanation": "attempt to waive the numerical axis",
        "source_evidence": ["synthetic"], "review_status": "approved", "review_id": "test-review"}
    kwargs["reconciliation_evidence"] = evidence
    kwargs["trusted_evidence"][evidence["evidence_id"]] = evidence_digest(evidence)
    assert not assess_transient_linearity(*records, **kwargs)["qualified"]


def reconstruction_case(*, last=100., first=1e-9):
    step_record, common, dc, ac, conditions, errors = reconstruction_inputs(
        times=observation_times(first_time_s=first, last_time_s=last))
    common["state"] = {"synthetic_physical_state": [0., 1., 0.]}
    window = build_window_spec(step_record["times_s"])
    scope = qualification_scope(step_record, state_sha256=evidence_digest(common["state"]), window_spec=window)
    application = {"step_sha256": evidence_digest(step_record), "ac_sha256": evidence_digest(ac),
                   "conductance_sha256": evidence_digest(dc), "step_amplitude_V": step_record["amplitude_V"],
                   "times_s": step_record["times_s"].tolist(), "window_spec_sha256": window_spec_digest(window),
                   "reconstruction_request_sha256": evidence_digest({"errors": errors,
                       "quadrature_absolute_tolerance_F_m2": 1e-12, "quadrature_relative_tolerance": 1e-10})}
    evidence = {kind: {"schema": "R1QualificationEvidenceV1", "kind": kind, "evidence_id": kind,
                       "scope": scope, "application": application, "frequency_Hz": ac["frequency_Hz"].tolist(),
                       "eligible_frequency_points": [True]*len(ac["frequency_Hz"]), "qualified": True,
                       "classification": "validated_estimate", "method": "exact synthetic response with checked numerical estimates",
                       "source_evidence": ["synthetic-reference"], "review_status": "approved", "review_id": "test-review"}
                for kind in REQUIRED}
    kwargs = dict(errors=errors, prerequisites=conditions, expected_scope=scope, window_spec=window,
                  qualification_evidence=evidence, trusted_evidence={key: evidence_digest(value) for key, value in evidence.items()})
    return (step_record, common, dc, ac), kwargs


def test_required_evidence_inventory_is_independent_of_the_production_constant():
    assert len(DOUBLE_DOMAIN_PREREQUISITES) == 16
    assert set(DOUBLE_DOMAIN_PREREQUISITES) == set(REQUIRED)


@pytest.mark.parametrize("missing", REQUIRED)
def test_each_independently_named_prerequisite_remains_required(missing):
    args, kwargs = reconstruction_case()
    evidence = kwargs["qualification_evidence"]
    application = evidence[REQUIRED[0]]["application"]
    evidence.pop(missing)
    result = assess_double_domain_prerequisites(evidence, expected_scope=kwargs["expected_scope"],
        expected_application=application, frequency_Hz=args[3]["frequency_Hz"], window_spec=kwargs["window_spec"],
        trusted_evidence=kwargs["trusted_evidence"])
    assert not result["qualified"] and not any(result["device_eligible_frequency_points"])


@pytest.mark.parametrize("first,last", [(1e-9, 100.), (1e-9, 1000.), (1e-10, 100.)])
def test_reconstruction_and_stage_gate_accept_only_the_frozen_device_window(first, last):
    args, kwargs = reconstruction_case(first=first, last=last)
    result = reconstruct_study_response(*args, **kwargs)
    assert result["device_double_domain_consistent"]
    assert not result["reconstruction"]["unknown_error_sources"]
    gate = device_response_gate(result, expected_scope=kwargs["expected_scope"], window_spec=kwargs["window_spec"],
                                required_frequency_Hz=args[3]["frequency_Hz"])
    assert gate["device_qualified"]
    bad = {**kwargs["expected_scope"], "state_sha256": "self-consistently-wrong-state"}
    with pytest.raises(ValueError, match="initial-state digest"):
        reconstruct_study_response(*args, **{**kwargs, "expected_scope": bad})


def test_partial_frequency_intersection_does_not_become_a_full_spectrum_claim():
    args, kwargs = reconstruction_case()
    item = kwargs["qualification_evidence"]["tail_omission"]
    item["eligible_frequency_points"][1] = False
    kwargs["trusted_evidence"]["tail_omission"] = evidence_digest(item)
    result = reconstruct_study_response(*args, **kwargs)
    assert not result["device_double_domain_consistent"]
    gate_args = dict(expected_scope=kwargs["expected_scope"], window_spec=kwargs["window_spec"])
    assert device_response_gate(result, required_frequency_Hz=args[3]["frequency_Hz"][[0, -1]], **gate_args)["qualified"]
    assert not device_response_gate(result, required_frequency_Hz=args[3]["frequency_Hz"], **gate_args)["qualified"]
    altered = copy.deepcopy(result)
    altered["device_double_domain_consistent_frequency_points"][1] = True
    assert not device_response_gate(altered, required_frequency_Hz=args[3]["frequency_Hz"], **gate_args)["qualified"]
    fixture_scope = {**kwargs["expected_scope"], "domain": "analytic_fixture"}
    assert not device_response_gate(result, required_frequency_Hz=args[3]["frequency_Hz"],
                                    expected_scope=fixture_scope, window_spec=kwargs["window_spec"])["qualified"]
