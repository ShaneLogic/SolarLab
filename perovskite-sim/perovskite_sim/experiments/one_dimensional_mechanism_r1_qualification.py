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
    AMPLITUDES_V, R1Response, compare_amplitude_halving, observation_times,
    step_current_charge_responses,
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
    if not isinstance(value, Mapping) or set(value) != set(SCOPE_FIELDS):
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
    return result


def qualification_scope(record, *, state_sha256, domain="device"):
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
    return _scope({
        "source_sha256": record.get("source", {}).get("sha256"),
        "prepared_sha256": prepared, "state_sha256": state_sha256,
        "reference_sha256": record.get("reference_sha256"),
        "control": record.get("control", record.get("control_label", "D")),
        "intervals": record.get("intervals"),
        "operating_voltage_V": event.get("voltage_V", record.get("operating_voltage_V", record.get("voltage_V", 0.))),
        "domain": domain,
    })


def _trusted(evidence, trusted_evidence):
    identifier = evidence.get("evidence_id")
    return (isinstance(identifier, str) and bool(identifier) and isinstance(trusted_evidence, Mapping)
            and trusted_evidence.get(identifier) == evidence_digest(evidence))


def _checked_record(record, trusted_evidence):
    return isinstance(trusted_evidence, Mapping) and evidence_digest(record) in trusted_evidence.values()


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
    report.update(classification="unknown", bound_A_m2=None, units="A/m2")
    if budget is None:
        report["reasons"].append("independent_current_error_budget_missing")
        return report
    try:
        if not isinstance(response, R1Response) or not isinstance(budget, Mapping):
            raise ValueError("budget assessment requires an explicit response and evidence record")
        if budget.get("schema") != "R1CurrentErrorBudgetV1" or _scope(budget.get("scope")) != report["scope"]:
            raise ValueError("current budget scope differs from the requested initial state")
        classification = budget.get("classification")
        if classification not in ("bounded", "estimate_only", "unknown"):
            raise ValueError("unknown current budget classification")
        report["classification"] = classification
        if budget.get("units") != "A/m2" or budget.get("amplitude_V") != amplitude_V:
            raise ValueError("current budget units or amplitude differ")
        if (not isinstance(budget.get("method"), str) or not budget["method"]
                or budget.get("propagation_method") != "sum_nonnegative_absolute_bounds"
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
        report["bound_A_m2"] = total.tolist()
        report["evidence_digests"] = [evidence_digest(budget)]
        if classification != "bounded":
            report.update(status="estimate_only", reasons=["estimate_is_not_an_approved_absolute_bound"])
        elif (budget.get("review_status") != "approved" or not isinstance(budget.get("review_id"), str)
              or not budget["review_id"] or not _trusted(budget, trusted_evidence)):
            report.update(status="unapproved", reasons=["independent_budget_approval_not_externally_bound"])
        else:
            report.update(status="qualified", qualified=True, device_qualified=report["scope"]["domain"] == "device")
    except (TypeError, ValueError, KeyError) as exc:
        report.update(status="invalid", qualified=False, device_qualified=False, reasons=[str(exc)], bound_A_m2=None)
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


def assess_transient_linearity(coarse_step, fine_step, *, coarse_baseline_A_m2, fine_baseline_A_m2,
                               expected_scope, expected_times_s, coarse_budget=None, fine_budget=None,
                               trusted_evidence=None):
    """Assess the full requested regular-current history, including both contacts.

    Device qualification requires the frozen full 0--100 s observation grid.
    An analytic fixture can exercise a smaller declared grid but cannot be
    promoted to device qualification. Impulse/tail/transform uncertainty is
    separately required by the double-domain qualification layer.
    """
    report = _base_report("R1TransientLinearityQualificationV1", expected_scope)
    report.update(full_transient_linearity_certified=False, comparison=None,
                  coarse_budget=None, fine_budget=None, diagnostic=None,
                  quantity="regular_current_response", complete_time_scope=False)
    try:
        expected = np.asarray(expected_times_s, dtype=float)
        if (expected.ndim != 1 or len(expected) < 3 or expected[0] != 0
                or not np.all(np.isfinite(expected)) or np.any(np.diff(expected) <= 0)):
            raise ValueError("linearity requires a complete increasing requested time axis")
        scope = report["scope"]
        for step in (coarse_step, fine_step):
            if step.get("schema") != "R1ControlledStepV1":
                raise ValueError("linearity requires controlled step records")
            actual = qualification_scope(step, state_sha256=scope["state_sha256"], domain=scope["domain"])
            if actual != scope:
                raise ValueError("linearity inputs do not share the requested initial-state scope")
        a, b = coarse_step["amplitude_V"], fine_step["amplitude_V"]
        if a not in AMPLITUDES_V or b not in AMPLITUDES_V or a != 2*b:
            raise ValueError("linearity requires adjacent declared amplitude levels")
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
                      components=list(coarse.components), diagnostic=_halving_diagnostic(coarse, fine, a, b))
        complete = scope["domain"] == "analytic_fixture" or np.array_equal(expected, observation_times())
        report["complete_time_scope"] = bool(complete)
        inputs_checked = all(_checked_record(step, trusted_evidence)
                             and step.get("certificate", {}).get("certified") is True and "failure" not in step
                             for step in (coarse_step, fine_step))
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
            error_a, error_b = [R1Response(np.asarray(item["bound_A_m2"]), coarse.coordinates, coarse.components)
                                for item in budgets]
            comparison = compare_amplitude_halving(coarse, fine, coarse_amplitude_V=a, fine_amplitude_V=b,
                                                   coarse_current_error=error_a, fine_current_error=error_b)
            qualified = comparison["passed"] is True
            report.update(comparison=comparison, status=comparison["status"], qualified=qualified,
                          full_transient_linearity_certified=qualified,
                          device_qualified=qualified and scope["domain"] == "device",
                          reasons=[] if qualified else [comparison["status"]])
    except (KeyError, TypeError, ValueError) as exc:
        report.update(status="invalid", reasons=[str(exc)])
    return _plain(report)


def select_response_amplitude(reports, *, expected_scope, verified_report_digests=(), diagnostic_amplitude_V=.005):
    """Select only independently replayed qualifications, never a default pass."""
    scope = _scope(expected_scope)
    if diagnostic_amplitude_V not in AMPLITUDES_V:
        raise ValueError("diagnostic amplitude must belong to the declared ladder")
    candidates = []
    for report in reports:
        if (report.get("schema") != "R1TransientLinearityQualificationV1" or report.get("scope") != scope
                or evidence_digest(report) not in verified_report_digests):
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
            "status": "qualified" if candidates else "diagnostic_only"}


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
                                       expected_application=None):
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
    application_valid = (isinstance(expected_application, Mapping)
                         and set(expected_application) == {"step_sha256", "ac_sha256", "conductance_sha256",
                                                          "step_amplitude_V", "times_s", "reconstruction_request_sha256"})
    if application_valid:
        application_times = np.asarray(expected_application["times_s"])
        application_valid = (all(isinstance(expected_application[key], str) and expected_application[key]
                                 for key in ("step_sha256", "ac_sha256", "conductance_sha256", "reconstruction_request_sha256"))
                             and expected_application["step_amplitude_V"] in AMPLITUDES_V
                             and application_times.dtype.kind in "fiu" and application_times.ndim == 1
                             and len(application_times) >= 3 and np.all(np.isfinite(application_times))
                             and application_times[0] == 0 and np.all(np.diff(application_times) > 0))
        if application_valid and report["scope"]["domain"] == "device":
            application_valid = np.array_equal(application_times, observation_times())
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
                if kind in _ERROR_PREREQUISITES and item.get("classification") != "bounded":
                    raise ValueError("prerequisite is not an independently approved error bound")
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
                  missing_prerequisites=missing, invalid_prerequisites=invalid,
                  reasons=missing+list(invalid), frequency_Hz=frequency.tolist(),
                  application=_plain(expected_application) if application_valid else None)
    return report


__all__ = ["SCOPE_FIELDS", "DOUBLE_DOMAIN_PREREQUISITES", "evidence_digest", "qualification_scope",
           "assess_current_error_budget", "assess_transient_linearity", "select_response_amplitude",
           "assess_frequency_coverage", "assess_double_domain_prerequisites"]
