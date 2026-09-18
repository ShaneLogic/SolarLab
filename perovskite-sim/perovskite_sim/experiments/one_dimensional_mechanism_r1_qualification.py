"""Scoped response qualification from independently held evidence digests.

This module does not approve a budget, discover missing physical modes or
authenticate a producer. The caller supplies an externally fixed scope and
digest mapping after its own input verification and independent review.
Evidence carrying an ``approved`` label without that mapping is unqualified.
Analytic fixtures and device evidence have different, non-interchangeable
domains. No function in this module solves a device trajectory.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import hashlib
import json
import math

import numpy as np

from perovskite_sim.experiments.one_dimensional_mechanism_r1_convergence import (
    AMPLITUDES_V, R1Response, compare_amplitude_halving,
    step_current_charge_responses,
)
from perovskite_sim.experiments.one_dimensional_mechanism_r1_window import (
    validate_window_spec, window_spec_digest,
)


SCOPE_FIELDS = ("source_sha256", "prepared_sha256", "state_sha256", "reference_sha256",
                "control", "intervals", "operating_voltage_V", "domain")
DOUBLE_DOMAIN_PREREQUISITES = (
    "finite_amplitude_linearity", "single_axis_convergence", "window_extension",
    "earlier_start", "stricter_integration", "tail_dc_agreement", "frequency_window_coverage",
    "input_trajectory_certified", "ac_content_verified", "current_error_budget",
    "early_omission", "tail_omission", "interpolation", "impulse_charge",
    "baseline_current_error", "dc_conductance_error",
)
_ERROR_PREREQUISITES = frozenset(DOUBLE_DOMAIN_PREREQUISITES[9:])
_NUMERICAL_EVIDENCE_CLASSES = frozenset(("validated_estimate", "conditional_bound", "bounded"))


def _plain(value):
    if isinstance(value, R1Response):
        return {"values": _plain(value.values), "coordinates": _plain(value.coordinates),
                "components": list(value.components)}
    if is_dataclass(value):
        return {field.name: _plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        # Runtime charge integrals use integer refinement levels; their JSON
        # representation uses strings. Preserve one canonical identity while
        # rejecting ambiguous containers such as {1: x, "1": y}.
        if any(not isinstance(key, str) and type(key) is not int for key in value):
            raise ValueError("evidence keys must be strings or integer refinement levels")
        keys = [str(key) for key in value]
        if len(set(keys)) != len(keys):
            raise ValueError("evidence keys collide after JSON normalization")
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, complex):
        return {"real": _plain(float(value.real)), "imag": _plain(float(value.imag))}
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("qualification evidence must be finite")
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError("unsupported qualification evidence value")


def evidence_digest(value):
    """Digest JSON scientific content; this computes identity, not approval."""
    payload = json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _scope(value):
    if not isinstance(value, Mapping) or set(value) not in (set(SCOPE_FIELDS), set(SCOPE_FIELDS) | {"window_spec_sha256"}):
        raise ValueError("qualification scope must contain exactly the declared identity fields")
    result = _plain(value)
    for key in ("source_sha256", "prepared_sha256", "state_sha256", "reference_sha256"):
        if not isinstance(result[key], str) or not result[key]:
            raise ValueError("qualification scope has no " + key)
    if result["control"] not in ("A", "B", "C", "D"):
        raise ValueError("qualification scope requires an A-D control")
    if type(result["intervals"]) is not int or result["intervals"] < 1:
        raise ValueError("qualification scope requires a positive grid size")
    if type(result["operating_voltage_V"]) not in (int, float):
        raise ValueError("qualification scope requires a finite operating voltage")
    if result["domain"] not in ("device", "analytic_fixture"):
        raise ValueError("qualification domain must be device or analytic_fixture")
    if "window_spec_sha256" in result:
        digest = result["window_spec_sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("qualification scope has an invalid window specification digest")
    return result


def qualification_scope(record, *, state_sha256, domain="device", window_spec=None):
    """Bind the shared initial state, never an amplitude-dependent end state.

    ``state_sha256`` is supplied by the caller from the independently checked
    initial prepared state. The same value must be used for both amplitudes.
    Source/preparation/reference/control/grid/bias come from the checked
    record. The caller, not an artifact label, chooses the evidence domain.
    """
    if not isinstance(record, Mapping) or not isinstance(record.get("source"), Mapping):
        raise ValueError("qualification scope requires a source-bound record")
    initial_event = record.get("initial_event", {})
    if not isinstance(initial_event, Mapping) or not isinstance(initial_event.get("zero_minus", {}), Mapping):
        raise ValueError("qualification scope has an invalid initial-state event")
    event = initial_event.get("zero_minus", {})
    prepared = record.get("prepared_sha256")
    if prepared is None and record.get("kind") == "equilibrium_D":
        prepared = record.get("sha256")
    scope = {
        "source_sha256": record.get("source", {}).get("sha256"),
        "prepared_sha256": prepared, "state_sha256": state_sha256,
        "reference_sha256": record.get("reference_sha256"),
        "control": record.get("control", record.get("control_label", "D")),
        "intervals": record.get("intervals"),
        "operating_voltage_V": event.get("voltage_V", record.get("operating_voltage_V", record.get("voltage_V", 0.))),
        "domain": domain,
    }
    if window_spec is not None:
        window = validate_window_spec(window_spec)
        if window["domain"] != domain:
            raise ValueError("qualification and window domains differ")
        scope["window_spec_sha256"] = window_spec_digest(window)
    return _scope(scope)


def _bound_window(scope, window_spec, *, times_s=None):
    if window_spec is None:
        if scope["domain"] == "device" or "window_spec_sha256" in scope:
            raise ValueError("device qualification requires an externally bound window specification")
        return None
    window = validate_window_spec(window_spec)
    if window["domain"] != scope["domain"] or scope.get("window_spec_sha256") != window_spec_digest(window):
        raise ValueError("window specification differs from the requested qualification scope")
    if times_s is not None and not np.array_equal(times_s, window["times_s"]):
        raise ValueError("response does not contain the complete externally requested window")
    return window


def _amplitude_pair(declared, *, coarse, fine, device):
    if declared is None:
        if device:
            raise ValueError("device qualification requires an externally declared amplitude ladder")
        declared = (coarse, fine)
    array = np.asarray(declared)
    if (array.dtype.kind not in "fiu" or array.ndim != 1 or len(array) < 2
            or not np.all(np.isfinite(array)) or any(value not in AMPLITUDES_V for value in array)
            or any(array[i] != 2*array[i+1] for i in range(len(array)-1))):
        raise ValueError("declared amplitudes must be a contiguous descending part of the prescribed ladder")
    if (coarse, fine) != tuple(array[-2:]):
        raise ValueError("linearity must compare the smallest two declared amplitudes")
    return array.astype(float).tolist()


def _trusted(evidence, trusted_evidence):
    identifier = evidence.get("evidence_id")
    return (isinstance(identifier, str) and bool(identifier) and isinstance(trusted_evidence, Mapping)
            and trusted_evidence.get(identifier) == evidence_digest(evidence))


def _checked_record(record, trusted_evidence, *, role, verified_record_digests, domain):
    if isinstance(verified_record_digests, Mapping):
        return verified_record_digests.get(role) == evidence_digest(record)
    # Historical fixture calls retain their old API. Device callers must use
    # named digests obtained from their data-verification receipts, separately
    # from the independently reviewed scientific-evidence map.
    return (domain == "analytic_fixture" and isinstance(trusted_evidence, Mapping)
            and evidence_digest(record) in trusted_evidence.values())


def _base_report(schema, expected_scope):
    return {"schema": schema, "scope": _scope(expected_scope), "status": "unknown", "qualified": False,
            "device_qualified": False, "reasons": [], "evidence_digests": []}


def assess_current_error_budget(budget, response, *, expected_scope, amplitude_V, trusted_evidence=None):
    """Check an approved, dimensioned budget at every requested sample.

    The budget declares ``required_terms`` and a same-shaped ``terms_A_m2``
    mapping. Independent review approves that coverage; this function checks
    that no declared term is missing and sums their nonnegative bounds. The
    digest mapping must be held outside the evidence being assessed.
    """
    report = _base_report("R1CurrentErrorBudgetAssessmentV1", expected_scope)
    report.update(classification="unknown", bound_A_m2=None, uncertainty_A_m2=None,
                  rigorous_bound=False, units="A/m2")
    if budget is None:
        report["reasons"].append("independent_current_error_budget_missing")
        return report
    try:
        if not isinstance(response, R1Response) or not isinstance(budget, Mapping):
            raise ValueError("budget assessment requires an explicit response and evidence record")
        if budget.get("schema") != "R1CurrentErrorBudgetV1" or _scope(budget.get("scope")) != report["scope"]:
            raise ValueError("current budget scope differs from the requested initial state")
        classification = budget.get("classification")
        if classification not in _NUMERICAL_EVIDENCE_CLASSES | {"estimate_only", "unknown"}:
            raise ValueError("unknown current budget classification")
        report["classification"] = classification
        if budget.get("units") != "A/m2" or budget.get("amplitude_V") != amplitude_V:
            raise ValueError("current budget units or amplitude differ")
        if (not isinstance(budget.get("method"), str) or not budget["method"]
                or budget.get("propagation_method") not in ("sum_nonnegative_absolute_bounds", "sum_nonnegative_absolute_uncertainties")
                or not isinstance(budget.get("source_evidence"), list) or not budget["source_evidence"]
                or any(not isinstance(value, str) or not value for value in budget["source_evidence"])):
            raise ValueError("current budget lacks source evidence or propagation method")
        if (budget.get("coordinates") != _plain(response.coordinates)
                or budget.get("components") != list(response.components)):
            raise ValueError("current budget coordinates or components differ")
        if budget.get("response_sha256") != evidence_digest(response):
            raise ValueError("current budget is bound to a different response")
        required, terms = budget.get("required_terms"), budget.get("terms_A_m2")
        if (not isinstance(required, list) or not required or any(not isinstance(k, str) or not k for k in required)
                or len(set(required)) != len(required) or not isinstance(terms, Mapping) or set(terms) != set(required)):
            raise ValueError("current budget omits or adds a declared error term")
        if classification == "unknown" or any(value is None for value in terms.values()):
            report["reasons"].append("one_or_more_current_error_terms_unknown")
            return report
        total = np.zeros(response.values.shape)
        for name, value in terms.items():
            raw = np.asarray(value)
            if raw.dtype.kind not in "fiu" or raw.shape != response.values.shape:
                raise ValueError("current error term shape or type differs: " + name)
            array = np.asarray(raw, dtype=float)
            if not np.all(np.isfinite(array)) or np.any(array < 0):
                raise ValueError("current error term is not a finite nonnegative bound: " + name)
            with np.errstate(over="ignore", invalid="ignore"):
                total += array
        if not np.all(np.isfinite(total)):
            raise ValueError("current budget sum is nonfinite")
        report["uncertainty_A_m2"] = total.tolist()
        report["bound_A_m2"] = total.tolist() if classification in ("conditional_bound", "bounded") else None
        report["evidence_digests"] = [evidence_digest(budget)]
        if classification not in _NUMERICAL_EVIDENCE_CLASSES:
            report.update(status="estimate_only", reasons=["estimate_is_not_an_approved_absolute_bound"])
        elif (budget.get("review_status") != "approved" or not isinstance(budget.get("review_id"), str)
              or not budget["review_id"] or not _trusted(budget, trusted_evidence)):
            report.update(status="unapproved", reasons=["independent_budget_approval_not_externally_bound"])
        else:
            report.update(status="qualified", qualified=True, device_qualified=report["scope"]["domain"] == "device",
                          rigorous_bound=classification == "bounded")
    except (TypeError, ValueError, KeyError) as exc:
        report.update(status="invalid", qualified=False, device_qualified=False, reasons=[str(exc)],
                      bound_A_m2=None, uncertainty_A_m2=None, rigorous_bound=False)
    return report


def _halving_diagnostic(coarse, fine, coarse_amplitude, fine_amplitude):
    with np.errstate(over="ignore", invalid="ignore"):
        a, b = coarse.values/coarse_amplitude, fine.values/fine_amplitude
        difference, scale = np.abs(a-b), .01*np.maximum(np.abs(a), np.abs(b))
    if not np.all(np.isfinite(difference)) or not np.all(np.isfinite(scale)):
        raise ValueError("linearity diagnostic arithmetic is nonfinite")
    return {"absolute_difference_S_m2": difference.tolist(), "one_percent_response_scale_S_m2": scale.tolist(),
            "exceeds_one_percent_response_scale": (difference > scale).tolist(),
            "absolute_numerical_error_budget_S_m2": None,
            "scope": "response_difference_diagnostic_without_error_budget"}


def _measured_error(value, response):
    if isinstance(value, Mapping) and set(value) == {"values", "coordinates", "components"}:
        value = R1Response(np.asarray(value["values"]), value["coordinates"], tuple(value["components"]))
    if (not isinstance(value, R1Response) or value.values.dtype.kind not in "fiu" or value.values.shape != response.values.shape
            or value.components != response.components or set(value.coordinates) != set(response.coordinates)
            or any(not np.array_equal(value.coordinates[key], response.coordinates[key]) for key in response.coordinates)
            or not np.all(np.isfinite(value.values)) or np.any(value.values < 0)):
        raise ValueError("measured current uncertainty must align with every current coordinate and component")
    return value


def _measurement_consistency(coarse, fine, coarse_budget, fine_budget, budget_reports, *,
                             coarse_amplitude, fine_amplitude, expected_scope, measured_evidence,
                             reconciliation_evidence, trusted_evidence):
    """Preserve an empirical failure until its particular conflict is reviewed.

    A refinement difference is not a known true error of the finest solution.
    A smaller independently justified estimate/bound may therefore be used,
    but only with an explicit review binding both measurements and budgets.
    This function does not turn the refinement difference into a strict bound.
    """
    result = {"qualified": False, "status": "unknown", "concerns": [], "comparison": None,
              "measured_evidence_sha256": None, "reconciliation_evidence_sha256": None}
    if measured_evidence is None:
        if expected_scope["domain"] == "analytic_fixture":
            result.update(qualified=True, status="fixture_without_refinement_evidence")
        else:
            result["status"] = "measured_numerical_evidence_missing"
            result["concerns"] = ["measured_numerical_evidence_missing"]
        return result
    required = {"coarse_current_error", "fine_current_error", "coarse_axes_passed", "fine_axes_passed"}
    if (not isinstance(measured_evidence, Mapping) or not required <= set(measured_evidence)
            or set(measured_evidence) - required - {"comparison"}
            or any(type(measured_evidence[key]) is not bool for key in ("coarse_axes_passed", "fine_axes_passed"))):
        raise ValueError("measured numerical evidence requires two aligned errors and explicit axis verdicts")
    errors = [_measured_error(measured_evidence[key], response) for key, response in
              (("coarse_current_error", coarse), ("fine_current_error", fine))]
    comparison = compare_amplitude_halving(coarse, fine, coarse_amplitude_V=coarse_amplitude,
        fine_amplitude_V=fine_amplitude, coarse_current_error=errors[0], fine_current_error=errors[1])
    if measured_evidence.get("comparison") is not None and _plain(measured_evidence["comparison"]) != _plain(comparison):
        raise ValueError("published empirical linearity differs from the measured errors and actual trajectories")
    measurement = {**{key: measured_evidence[key] for key in required}, "comparison": comparison}
    result.update(comparison=comparison, measured_evidence_sha256=evidence_digest(measurement))
    concerns = []
    for label, report, error in zip(("coarse", "fine"), budget_reports, errors):
        if not measured_evidence[label+"_axes_passed"]:
            concerns.append(label+"_measured_convergence_failed")
        if np.any(np.asarray(report["uncertainty_A_m2"]) < error.values):
            concerns.append(label+"_declared_uncertainty_smaller_than_refinement_estimate")
    if comparison["passed"] is not True:
        concerns.append("measured_linearity_"+comparison["status"])
    result["concerns"] = concerns
    if not measured_evidence["coarse_axes_passed"] or not measured_evidence["fine_axes_passed"]:
        # Section 10.2 axis acceptance is a fixed numerical requirement.
        # Reviewing a stronger uncertainty estimate does not waive it.
        result["status"] = "measured_convergence_failed"
        return result
    if not concerns:
        result.update(qualified=True, status="consistent_with_measured_evidence")
        return result
    evidence = reconciliation_evidence
    if not isinstance(evidence, Mapping):
        result["status"] = "unresolved_numerical_evidence_conflict"
        return result
    expected = {"schema": "R1NumericalEvidenceReconciliationV1", "scope": expected_scope,
                "measured_evidence_sha256": result["measured_evidence_sha256"],
                "budget_evidence_sha256": [evidence_digest(coarse_budget), evidence_digest(fine_budget)],
                "response_sha256": [evidence_digest(coarse), evidence_digest(fine)],
                "decision": "supersedes_empirical_estimate"}
    if (any(_plain(evidence.get(key)) != _plain(value) for key, value in expected.items())
            or sorted(evidence.get("resolved_concerns", [])) != sorted(concerns)
            or any(not isinstance(evidence.get(key), str) or not evidence[key]
                   for key in ("method", "explanation", "review_id"))
            or not isinstance(evidence.get("source_evidence"), list) or not evidence["source_evidence"]
            or any(not isinstance(item, str) or not item for item in evidence["source_evidence"])
            or evidence.get("review_status") != "approved" or not _trusted(evidence, trusted_evidence)):
        result["status"] = "unresolved_numerical_evidence_conflict"
        return result
    result.update(qualified=True, status="independently_explained_estimate_conflict",
                  reconciliation_evidence_sha256=evidence_digest(evidence))
    return result


def assess_transient_linearity(coarse_step, fine_step, *, coarse_baseline_A_m2, fine_baseline_A_m2,
                               expected_scope, expected_times_s=None, coarse_budget=None, fine_budget=None,
                               trusted_evidence=None, window_spec=None, declared_amplitudes_V=None,
                               verified_record_digests=None, measured_evidence=None, reconciliation_evidence=None):
    """Assess the full requested regular-current history, including both contacts.

    Device qualification requires the externally frozen section-6 window.
    An analytic fixture can exercise a smaller declared grid but cannot be
    promoted to device qualification. Impulse/tail/transform uncertainty is
    separately required by the double-domain qualification layer.
    """
    report = _base_report("R1TransientLinearityQualificationV1", expected_scope)
    report.update(full_transient_linearity_certified=False, comparison=None,
                  coarse_budget=None, fine_budget=None, diagnostic=None,
                  quantity="regular_current_response", complete_time_scope=False,
                  window_spec=None, measured_consistency=None)
    try:
        scope = report["scope"]
        window = validate_window_spec(window_spec) if window_spec is not None else None
        if expected_times_s is None and window is not None:
            expected_times_s = window["times_s"]
        expected = np.asarray(expected_times_s, dtype=float)
        if (expected.ndim != 1 or len(expected) < 3 or expected[0] != 0
                or not np.all(np.isfinite(expected)) or np.any(np.diff(expected) <= 0)):
            raise ValueError("linearity requires a complete increasing requested time axis")
        for step in (coarse_step, fine_step):
            if step.get("schema") != "R1ControlledStepV1":
                raise ValueError("linearity requires controlled step records")
            actual = qualification_scope(step, state_sha256=scope["state_sha256"], domain=scope["domain"],
                                         window_spec=window)
            if actual != scope:
                raise ValueError("linearity inputs do not share the requested initial-state scope")
        a, b = coarse_step["amplitude_V"], fine_step["amplitude_V"]
        if a not in AMPLITUDES_V or b not in AMPLITUDES_V or a != 2*b:
            raise ValueError("linearity requires adjacent declared amplitude levels")
        declared = _amplitude_pair(declared_amplitudes_V, coarse=a, fine=b, device=scope["domain"] == "device")
        for step in (coarse_step, fine_step):
            event = step["initial_event"]
            if (not np.array_equal(step.get("voltage_V"), np.full(expected.shape, scope["operating_voltage_V"]+step["amplitude_V"]))
                    or event.get("voltage_jump_V") != step["amplitude_V"]
                    or event.get("zero_plus", {}).get("voltage_V") != scope["operating_voltage_V"]+step["amplitude_V"]):
                raise ValueError("linearity step voltage does not match its declared amplitude and operating state")
        coarse = step_current_charge_responses(coarse_step, coarse_baseline_A_m2, expected_times_s=expected)[0]
        fine = step_current_charge_responses(fine_step, fine_baseline_A_m2, expected_times_s=expected)[0]
        if not np.all(np.isfinite(coarse.values)) or not np.all(np.isfinite(fine.values)):
            raise ValueError("linearity response contains nonfinite current")
        report.update(coarse_amplitude_V=a, fine_amplitude_V=b, times_s=expected.tolist(),
                      components=list(coarse.components), diagnostic=_halving_diagnostic(coarse, fine, a, b),
                      declared_amplitudes_V=declared)
        try:
            report["window_spec"] = _bound_window(scope, window, times_s=expected)
            complete = True
        except ValueError as exc:
            complete = False
            report["reasons"].append(str(exc))
        report["complete_time_scope"] = complete
        inputs_checked = all(_checked_record(step, trusted_evidence, role=role,
                              verified_record_digests=verified_record_digests, domain=scope["domain"])
                             and step.get("certificate", {}).get("certified") is True and "failure" not in step
                             for role, step in (("coarse_step", coarse_step), ("fine_step", fine_step)))
        budgets = [assess_current_error_budget(value, response, expected_scope=scope, amplitude_V=amplitude,
                                               trusted_evidence=trusted_evidence)
                   for value, response, amplitude in ((coarse_budget, coarse, a), (fine_budget, fine, b))]
        report["coarse_budget"], report["fine_budget"] = budgets
        report["evidence_digests"] = [evidence_digest(step) for step in (coarse_step, fine_step)]
        if not complete or not inputs_checked:
            report.update(status="unqualified_inputs", reasons=["full_time_scope_or_verified_physical_inputs_missing"])
        elif not all(item["qualified"] for item in budgets):
            report.update(status="budget_unqualified", reasons=["approved_independent_current_budgets_required"])
        else:
            consistency = _measurement_consistency(coarse, fine, coarse_budget, fine_budget, budgets,
                coarse_amplitude=a, fine_amplitude=b, expected_scope=scope, measured_evidence=measured_evidence,
                reconciliation_evidence=reconciliation_evidence, trusted_evidence=trusted_evidence)
            report["measured_consistency"] = consistency
            error_a, error_b = [R1Response(np.asarray(item["uncertainty_A_m2"]), coarse.coordinates, coarse.components)
                                for item in budgets]
            comparison = compare_amplitude_halving(coarse, fine, coarse_amplitude_V=a, fine_amplitude_V=b,
                                                   coarse_current_error=error_a, fine_current_error=error_b)
            qualified = comparison["passed"] is True and consistency["qualified"] is True
            status = comparison["status"] if consistency["qualified"] else consistency["status"]
            report.update(comparison=comparison, status=status, qualified=qualified,
                          full_transient_linearity_certified=qualified,
                          device_qualified=qualified and scope["domain"] == "device",
                          reasons=[] if qualified else [status])
    except (KeyError, TypeError, ValueError) as exc:
        report.update(status="invalid", reasons=[str(exc)])
    return _plain(report)


def select_response_amplitude(reports, *, expected_scope, verified_report_digests=(), diagnostic_amplitude_V=.005,
                              window_spec=None, declared_amplitudes_V=None):
    """Select only independently replayed qualifications, never a default pass."""
    scope = _scope(expected_scope)
    if diagnostic_amplitude_V not in AMPLITUDES_V:
        raise ValueError("diagnostic amplitude must belong to the declared ladder")
    candidates, reasons = [], []
    for report in reports:
        if (not isinstance(report, Mapping) or report.get("schema") != "R1TransientLinearityQualificationV1" or report.get("scope") != scope
                or evidence_digest(report) not in verified_report_digests):
            continue
        try:
            window = _bound_window(scope, window_spec, times_s=report.get("times_s"))
            if window is not None and report.get("window_spec") != window:
                raise ValueError("linearity report does not bind the externally requested window")
            declared = _amplitude_pair(declared_amplitudes_V, coarse=report.get("coarse_amplitude_V"),
                                       fine=report.get("fine_amplitude_V"), device=scope["domain"] == "device")
            if scope["domain"] == "device" and (report.get("declared_amplitudes_V") != declared
                    or report.get("device_qualified") is not True
                    or not isinstance(report.get("measured_consistency"), Mapping)
                    or report["measured_consistency"].get("qualified") is not True):
                raise ValueError("device amplitude lacks complete declared and measured evidence")
        except (TypeError, ValueError) as exc:
            reasons.append(str(exc))
            continue
        if (report.get("qualified") is True and report.get("full_transient_linearity_certified") is True
                and report.get("complete_time_scope") is True
                and isinstance(report.get("comparison"), Mapping) and report["comparison"].get("passed") is True
                and isinstance(report.get("coarse_budget"), Mapping) and report["coarse_budget"].get("qualified") is True
                and isinstance(report.get("fine_budget"), Mapping) and report["fine_budget"].get("qualified") is True):
            amplitude = report.get("fine_amplitude_V")
            if amplitude in AMPLITUDES_V:
                candidates.append((amplitude, evidence_digest(report)))
    chosen = min(candidates) if candidates else (None, None)
    return {"schema": "R1ResponseAmplitudeSelectionV1", "scope": scope,
            "qualified_amplitude_V": chosen[0], "qualification_report_sha256": chosen[1],
            "diagnostic_amplitude_V": diagnostic_amplitude_V, "qualified": chosen[0] is not None,
            "device_qualified": chosen[0] is not None and scope["domain"] == "device",
            "status": "qualified" if candidates else "diagnostic_only", "reasons": reasons}


def assess_frequency_coverage(frequency_Hz, numerically_eligible, *, turnover_evidence=None,
                              expected_scope=None, trusted_evidence=None):
    """Evaluate the frozen one-decade margins and four intervals per decade."""
    report = {"schema": "R1FrequencyCoverageQualificationV1", "scope": None, "status": "unknown",
              "qualified": False, "device_qualified": False, "frequency_window_complete": False,
              "reasons": [], "evidence_digests": []}
    try:
        frequency = np.asarray(frequency_Hz)
        numeric = np.asarray(numerically_eligible)
        if (frequency.dtype.kind not in "fiu" or frequency.ndim != 1 or not len(frequency)
                or not np.all(np.isfinite(frequency)) or np.any(frequency < 0) or np.any(np.diff(frequency) <= 0)
                or numeric.dtype.kind != "b" or numeric.shape != frequency.shape):
            raise ValueError("invalid frequency or numerical eligibility axis")
        report["frequency_Hz"] = frequency.tolist()
        if expected_scope is None or turnover_evidence is None:
            report["reasons"] = ["same_state_turnover_evidence_missing"]
            return report
        scope = _scope(expected_scope)
        report["scope"] = scope
        evidence = turnover_evidence
        if not isinstance(evidence, Mapping):
            raise ValueError("turnover qualification requires evidence, not a pass flag")
        if evidence.get("schema") != "R1TurnoverEvidenceV1" or _scope(evidence.get("scope")) != scope:
            raise ValueError("turnover evidence belongs to a different source/state/control")
        if (evidence.get("coverage") != "all_applicable_modes" or evidence.get("unresolved_modes") != []
                or evidence.get("review_status") != "approved" or not isinstance(evidence.get("review_id"), str)
                or not evidence.get("review_id")
                or not _trusted(evidence, trusted_evidence)):
            report["reasons"] = ["complete_turnover_coverage_not_independently_approved"]
            return report
        scales = np.asarray(evidence.get("turnover_frequency_Hz"))
        if (scales.dtype.kind not in "fiu" or scales.ndim != 1 or not len(scales)
                or not np.all(np.isfinite(scales)) or np.any(scales <= 0)):
            raise ValueError("turnover frequencies must be finite positive values in Hz")
        if evidence.get("units") != "Hz":
            raise ValueError("turnover frequency units must be Hz")
        positive = frequency[frequency > 0]
        reasons = []
        if not len(positive) or positive[0] > min(scales)/10 or positive[-1] < max(scales)*10:
            reasons.append("required_turnovers_and_one_decade_margins_not_covered")
        if (len(positive) < 2 or np.any(np.diff(np.log10(positive)) > .25+32*np.finfo(float).eps)):
            reasons.append("four_frequency_intervals_per_decade_not_covered")
        if np.any(positive < 1e-6) or np.any(positive > 1e10):
            reasons.append("frequency_samples_outside_protocol_limits")
        if not np.all(numeric):
            reasons.append("one_or_more_frequency_points_failed_numeric_checks")
        qualified = not reasons
        report.update(status="certified" if qualified else "incomplete", qualified=qualified,
                      device_qualified=qualified and scope["domain"] == "device",
                      frequency_window_complete=qualified, reasons=reasons,
                      turnover_frequency_Hz=scales.tolist(), evidence_digests=[evidence_digest(evidence)])
    except (KeyError, TypeError, ValueError) as exc:
        report.update(status="invalid", reasons=[str(exc)])
    return report


def assess_double_domain_prerequisites(evidence, *, expected_scope, frequency_Hz, trusted_evidence=None,
                                       expected_application=None, window_spec=None):
    """Check every independently qualified prerequisite on the exact AC axis.

    Each item is R1QualificationEvidenceV1 with ``kind``, scope, frequency_Hz,
    boolean eligible_frequency_points, qualified/review status and an external
    digest. ``application`` binds the exact step/AC/DC records, amplitude and
    full time axis; sharing an initial-state scope alone is insufficient.
    Error items additionally require classification='bounded'.
    """
    report = _base_report("R1DoubleDomainPrerequisitesV1", expected_scope)
    frequency = np.asarray(frequency_Hz)
    if (frequency.dtype.kind not in "fiu" or frequency.ndim != 1 or not len(frequency)
            or not np.all(np.isfinite(frequency)) or np.any(frequency < 0) or np.any(np.diff(frequency) <= 0)):
        raise ValueError("double-domain frequency axis is invalid")
    evidence = evidence or {}
    if not isinstance(evidence, Mapping):
        raise ValueError("double-domain prerequisites must be named evidence records")
    application_keys = {"step_sha256", "ac_sha256", "conductance_sha256", "step_amplitude_V",
                        "times_s", "reconstruction_request_sha256"}
    application_valid = (isinstance(expected_application, Mapping)
                         and set(expected_application) in (application_keys, application_keys | {"window_spec_sha256"}))
    if application_valid:
        application_times = np.asarray(expected_application["times_s"])
        application_valid = (all(isinstance(expected_application[key], str) and expected_application[key]
                                 for key in ("step_sha256", "ac_sha256", "conductance_sha256", "reconstruction_request_sha256"))
                             and expected_application["step_amplitude_V"] in AMPLITUDES_V
                             and application_times.dtype.kind in "fiu" and application_times.ndim == 1
                             and len(application_times) >= 3 and np.all(np.isfinite(application_times))
                             and application_times[0] == 0 and np.all(np.diff(application_times) > 0))
        if application_valid:
            try:
                window = _bound_window(report["scope"], window_spec, times_s=application_times)
                if window is not None:
                    application_valid = expected_application.get("window_spec_sha256") == window_spec_digest(window)
                elif "window_spec_sha256" in expected_application:
                    application_valid = False
            except ValueError:
                application_valid = False
    masks, missing, invalid = {}, [], {}
    for kind in DOUBLE_DOMAIN_PREREQUISITES:
        item = evidence.get(kind)
        mask = np.zeros(len(frequency), dtype=bool)
        if item is None:
            missing.append(kind)
        else:
            try:
                if not isinstance(item, Mapping):
                    raise ValueError("prerequisite must be evidence, not a bare pass flag")
                if (item.get("schema") != "R1QualificationEvidenceV1" or item.get("kind") != kind
                        or _scope(item.get("scope")) != report["scope"]):
                    raise ValueError("prerequisite identity or kind differs")
                if not application_valid or item.get("application") != _plain(expected_application):
                    raise ValueError("prerequisite does not bind the requested trajectory/amplitude/AC/DC application")
                if not np.array_equal(item.get("frequency_Hz"), frequency):
                    raise ValueError("prerequisite frequency coverage differs")
                values = np.asarray(item.get("eligible_frequency_points"))
                if values.dtype.kind != "b" or values.shape != frequency.shape:
                    raise ValueError("prerequisite requires an explicit boolean frequency mask")
                if (item.get("qualified") is not True or item.get("review_status") != "approved"
                        or not isinstance(item.get("review_id"), str) or not item.get("review_id")
                        or not _trusted(item, trusted_evidence)):
                    raise ValueError("prerequisite has no externally bound qualification")
                if kind in _ERROR_PREREQUISITES:
                    if item.get("classification") not in _NUMERICAL_EVIDENCE_CLASSES:
                        raise ValueError("prerequisite is not independently reviewed numerical error evidence")
                    if report["scope"]["domain"] == "device" and (
                        not isinstance(item.get("method"), str) or not item["method"]
                        or not isinstance(item.get("source_evidence"), list) or not item["source_evidence"]
                        or any(not isinstance(value, str) or not value for value in item["source_evidence"])):
                        raise ValueError("device error evidence lacks method or underlying sources")
                mask = values.copy()
                report["evidence_digests"].append(evidence_digest(item))
            except (KeyError, TypeError, ValueError) as exc:
                invalid[kind] = str(exc)
        masks[kind] = mask.tolist()
    eligible = np.logical_and.reduce([np.asarray(mask) for mask in masks.values()])
    qualified = bool(np.all(eligible))
    report.update(status="qualified" if qualified else "unqualified", qualified=qualified,
                  device_qualified=qualified and report["scope"]["domain"] == "device",
                  prerequisites=masks, eligible_frequency_points=eligible.tolist(),
                  device_eligible_frequency_points=(eligible & (report["scope"]["domain"] == "device")).tolist(),
                  missing_prerequisites=missing, invalid_prerequisites=invalid,
                  reasons=missing+list(invalid), frequency_Hz=frequency.tolist(),
                  application=_plain(expected_application) if application_valid else None)
    return report


def device_response_gate(report, *, expected_scope, window_spec, required_frequency_Hz):
    """Assess a caller-declared device intersection, not all possible spectra.

    The caller must first verify/replay the supplied reconstruction. This
    stage adapter rechecks scope, external window, application and exact
    frequency masks; it does not authenticate a report or approve its method.
    A nonempty required frequency set is frozen by the caller, never inferred
    by trimming failed points from the result.
    """
    result = {"schema": "R1DeviceResponseGateV1", "qualified": False, "device_qualified": False,
              "status": "unqualified", "reasons": [], "required_frequency_Hz": None}
    try:
        scope = _scope(expected_scope)
        if scope["domain"] != "device":
            raise ValueError("a device stage cannot consume analytic-fixture qualification")
        window = _bound_window(scope, window_spec)
        if (not isinstance(report, Mapping) or report.get("schema") != "R1StudyReconstructedResponseV1"
                or report.get("full_time_window_covered") is not True):
            raise ValueError("device stage requires a complete reconstructed response")
        qualification = report["qualification"]
        if not isinstance(qualification, Mapping):
            raise ValueError("device stage requires a qualification evidence report")
        if qualification.get("scope") != scope or report.get("window_spec") != window:
            raise ValueError("device reconstruction scope or window differs")
        application = qualification.get("application", {})
        if not isinstance(application, Mapping):
            raise ValueError("device stage requires a bound reconstruction application")
        if (application.get("window_spec_sha256") != window_spec_digest(window)
                or not np.array_equal(application.get("times_s"), window["times_s"])):
            raise ValueError("device application does not bind the requested window")
        identity = report["published_identity"]
        for key in ("prepared_sha256", "reference_sha256", "intervals", "control", "source_sha256", "operating_voltage_V"):
            if identity.get(key) != scope[key]:
                raise ValueError("device published identity differs: "+key)
        frequency = np.asarray(report["reconstruction"]["frequency_Hz"])
        required = np.asarray(required_frequency_Hz)
        if (required.dtype.kind not in "fiu" or required.ndim != 1 or not len(required)
                or not np.all(np.isfinite(required)) or np.any(required < 0) or np.any(np.diff(required) <= 0)
                or frequency.dtype.kind not in "fiu" or frequency.ndim != 1 or not len(frequency)
                or not np.all(np.isfinite(frequency)) or np.any(frequency < 0) or np.any(np.diff(frequency) <= 0)):
            raise ValueError("device stage requires explicit increasing finite frequency axes")
        if not np.all(np.isin(required, frequency)) or not np.array_equal(qualification.get("frequency_Hz"), frequency):
            raise ValueError("required device frequencies are missing or differ from the qualified axis")
        numeric = np.asarray(report["comparison"]["double_domain_consistent_frequency_points"])
        evidence = np.asarray(qualification.get("device_eligible_frequency_points"))
        published = np.asarray(report.get("device_double_domain_consistent_frequency_points"))
        if any(mask.dtype.kind != "b" or mask.shape != frequency.shape for mask in (numeric, evidence, published)):
            raise ValueError("device stage requires explicit aligned boolean qualification masks")
        mask = numeric & evidence
        if not np.array_equal(mask, published):
            raise ValueError("published device mask differs from numeric and evidence intersection")
        accepted = bool(np.all(mask[np.isin(frequency, required)]))
        result.update(qualified=accepted, device_qualified=accepted,
                      status="qualified" if accepted else "required_frequency_points_unqualified",
                      required_frequency_Hz=required.tolist(), scope=scope,
                      window_spec_sha256=window_spec_digest(window), eligible_frequency_points=mask.tolist(),
                      reasons=[] if accepted else ["one_or_more_required_frequency_points_unqualified"])
    except (KeyError, TypeError, ValueError) as exc:
        result.update(status="invalid", reasons=[str(exc)])
    return result


__all__ = ["SCOPE_FIELDS", "DOUBLE_DOMAIN_PREREQUISITES", "evidence_digest", "qualification_scope",
           "assess_current_error_budget", "assess_transient_linearity", "select_response_amplitude",
           "assess_frequency_coverage", "assess_double_domain_prerequisites", "device_response_gate"]
