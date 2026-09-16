"""Frequency-window and time/AC study diagnostics on already verified inputs.

These adapters do not establish artifact provenance. The study runner must
verify and bind each input before calling them; every identity used here is
read from those inputs rather than supplied from a caller's live preparation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass

import numpy as np

from perovskite_sim.experiments.one_dimensional_mechanism_r1_admittance import reconstruct_admittance
from perovskite_sim.experiments.one_dimensional_mechanism_r1_convergence import observation_times
from perovskite_sim.experiments.one_dimensional_mechanism_r1_response import (
    _finite_scalar, _frequencies, _response_array, assess_small_signal_response,
    compare_reconstructed_response,
)


def _ready(value):
    if is_dataclass(value):
        return {field.name: _ready(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {key: _ready(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _ready(value.tolist())
    if isinstance(value, np.generic):
        return _ready(value.item())
    if isinstance(value, (tuple, list)):
        return [_ready(item) for item in value]
    if isinstance(value, complex):
        return {"real": _ready(value.real), "imag": _ready(value.imag)}
    if isinstance(value, float) and not np.isfinite(value):
        return {"value": None, "reason": "nonfinite_numeric_evidence", "representation": repr(value)}
    return value


def _bands(frequency, eligible):
    bands = []
    indices = np.flatnonzero(eligible)
    for group in np.split(indices, np.flatnonzero(np.diff(indices) != 1)+1):
        if group.size:
            bands.append({"first_index": int(group[0]), "last_index": int(group[-1]),
                          "minimum_Hz": float(frequency[group[0]]), "maximum_Hz": float(frequency[group[-1]]),
                          "point_count": int(group.size), "scope": "sampled_numeric_eligibility_only"})
    return bands


def frequency_window_report(ac):
    """Report numeric bands and endpoint decades without asserting mode coverage.

    Endpoint flatness is a response diagnostic, not proof that weak or hidden
    modes have been covered. The same-state physical turnover scales required
    by §7.1 remain an independent prerequisite even at the extension limits.
    """
    frequency = _frequencies(ac["frequency_Hz"])
    admittance = _response_array(ac["admittance_S_m2"], "admittance", shape=frequency.shape)
    numeric = assess_small_signal_response(ac)
    protocol = (frequency == 0) | ((frequency >= 1e-6) & (frequency <= 1e10))
    eligible = numeric["numerically_eligible_frequency_points"] & protocol
    positive = np.flatnonzero(frequency > 0)
    reasons = ["same_state_turnover_scales_and_one_decade_margins_not_verified"]
    diagnostics = {}
    low_reached = high_reached = False
    if positive.size:
        first, last = int(positive[0]), int(positive[-1])
        low_reached, high_reached = frequency[first] <= 1e-6, frequency[last] >= 1e10
        # Include a nominal decade endpoint despite floating-point logspace
        # rounding; this selects diagnostic samples, not a physical gate.
        roundoff = 8*np.finfo(float).eps
        for side, chosen in (("lower", positive[frequency[positive] <= frequency[first]*10*(1+roundoff)]),
                             ("upper", positive[frequency[positive] >= frequency[last]/10*(1-roundoff)])):
            a, b = int(chosen[0]), int(chosen[-1])
            capacitance = admittance[[a, b]].imag/(2*np.pi*frequency[[a, b]])
            real = admittance[[a, b]].real
            real_scale, cap_scale = np.max(np.abs(real)), np.max(np.abs(capacitance))
            diagnostics[side] = {
                "frequency_Hz": frequency[[a, b]], "point_count": len(chosen),
                "full_decade_sampled": bool(frequency[b]/frequency[a] >= 10*(1-1e-12)),
                "real_admittance_S_m2": real, "capacitance_F_m2": capacitance,
                "real_relative_change": float(abs(real[1]-real[0])/real_scale) if real_scale else 0.,
                "capacitance_relative_change": float(abs(capacitance[1]-capacitance[0])/cap_scale) if cap_scale else 0.,
                "scope": "endpoint_response_change_not_a_turnover_certificate",
            }
        if not low_reached or not high_reached:
            reasons.append("protocol_decade_extensions_not_exhausted")
        else:
            reasons.append("protocol_extension_limits_reached_without_turnover_coverage_evidence")
    else:
        reasons.append("no_positive_frequency_samples")
    if not np.all(protocol):
        reasons.append("frequency_samples_outside_protocol_limits")
    if not np.all(numeric["numerically_eligible_frequency_points"]):
        reasons.append("one_or_more_frequency_points_failed_numeric_checks")
    return _ready({"schema": "R1FrequencyWindowReportV1", "frequency_Hz": frequency,
                   "numeric_checks_passed": numeric["certified"], "numeric_checks": numeric["checks"],
                   "numerically_eligible_frequency_points": eligible,
                   "valid_contiguous_bands": _bands(frequency, eligible), "endpoint_decades": diagnostics,
                   "protocol_limits_Hz": [1e-6, 1e10], "lower_extension_limit_reached": low_reached,
                   "upper_extension_limit_reached": high_reached, "frequency_window_complete": False,
                   "uncovered_reasons": reasons, "scope": "numeric_band_and_window_diagnostics_only"})


def _matching_identity(record, common, control):
    for key in ("reference_sha256", "intervals"):
        if record.get(key) != common.get(key):
            raise ValueError("study response identity differs: " + key)
    if record.get("prepared_sha256") != common["sha256"] or record.get("control", record.get("control_label")) != control:
        raise ValueError("study response preparation/control identity differs")
    if record.get("source") is None or record["source"] != common.get("source"):
        raise ValueError("study response source identity differs")


def reconstruct_study_response(step, prepared, conductance, ac, *, errors=None, prerequisites=None,
                               quadrature_absolute_tolerance_F_m2=1e-12,
                               quadrature_relative_tolerance=1e-10):
    """Reconstruct the regular response using the saved independent DC ladder.

    Inputs are the exact records verified by the caller. An incomplete or
    failed trajectory may yield a diagnostic curve when samples exist, but
    never yields an eligible dual-domain result. Missing error components are
    preserved as unknown by the integration routine.
    """
    common = prepared.to_dict() if hasattr(prepared, "to_dict") else prepared
    if step.get("schema") != "R1ControlledStepV1" or conductance.get("schema") != "R1DCConductanceStudyV1":
        raise ValueError("study reconstruction requires controlled step and DC conductance records")
    control = step["control_label"]
    baseline = conductance["baseline"]
    for record in (step, baseline, ac):
        _matching_identity(record, common, control)
    if conductance.get("control") != control or baseline.get("voltage_V") != 0. or ac.get("voltage_V") != 0.:
        raise ValueError("study reconstruction requires the same zero-bias operating state")
    widths = _response_array(conductance["half_width_V"], "DC derivative widths", shape=(3,))
    if not np.array_equal(widths, [1e-4, 5e-5, 2.5e-5]) or len(conductance["pairs"]) != 3:
        raise ValueError("study reconstruction requires all three declared DC derivative levels")
    values = _response_array(conductance["conductance_S_m2"], "DC conductance", shape=(3,))
    dc_certified = baseline.get("certified") is True and conductance.get("finest_pair_agrees") is True
    for index, pair in enumerate(conductance["pairs"]):
        width = widths[index]
        if pair.get("half_width_V") != width:
            raise ValueError("DC pair derivative width differs")
        for sign, key in ((-1, "minus"), (1, "plus")):
            _matching_identity(pair[key], common, control)
            if pair[key].get("voltage_V") != sign*width:
                raise ValueError("DC pair voltage differs from its declared derivative width")
            dc_certified &= pair[key].get("certified") is True
        value = (pair["plus"]["terminal_current_A_m2"]-pair["minus"]["terminal_current_A_m2"])/(2*width)
        if values[index] != value:
            raise ValueError("published conductance differs from the independent DC currents")
    times = _response_array(step["times_s"], "step times")
    if times.ndim != 1 or len(times) < 3 or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("study reconstruction needs 0+ and at least two increasing positive-time samples")
    amplitude = _finite_scalar(step["amplitude_V"], "step amplitude")
    if not 0 < abs(amplitude) < .02:
        raise ValueError("step amplitude is outside the declared small-step range")
    if not np.array_equal(_response_array(step["voltage_V"], "step voltage", shape=times.shape), np.full(times.shape, amplitude)):
        raise ValueError("step voltage differs from its declared amplitude")
    event = step["initial_event"]
    operating_voltage = _finite_scalar(event["zero_minus"]["voltage_V"], "initial operating voltage")
    if (operating_voltage != baseline["voltage_V"] or event["zero_plus"]["voltage_V"] != amplitude
            or event["voltage_jump_V"] != amplitude-operating_voltage):
        raise ValueError("saved step event and DC operating voltage identity differs")
    currents = _response_array([item["report_contact_current_A_m2"] for item in step["regular_currents"]],
                               "regular contact currents", shape=(len(times), 2))
    impulse = _finite_scalar(event["impulse_charge_C_m2"], "impulse charge")
    reconstruction = reconstruct_admittance(ac["frequency_Hz"], time_s=times[1:],
        regular_current_A_m2=currents[1:, 0], step_amplitude_V=amplitude,
        baseline_current_A_m2=baseline["terminal_current_A_m2"], dc_conductance_S_m2=values[-1],
        impulse_charge_C_m2=impulse, errors=errors,
        quadrature_absolute_tolerance_F_m2=quadrature_absolute_tolerance_F_m2,
        quadrature_relative_tolerance=quadrature_relative_tolerance)
    try:
        full_window = np.array_equal(times, observation_times(first_time_s=float(times[1]), last_time_s=float(times[-1])))
    except ValueError:
        full_window = False
    physical = step.get("certificate", {}).get("certified") is True and "failure" not in step
    identity = {"prepared_sha256": step["prepared_sha256"], "reference_sha256": step["reference_sha256"],
                "intervals": step["intervals"], "control": control, "source_sha256": step["source"]["sha256"],
                "operating_voltage_V": operating_voltage, "step_amplitude_V": amplitude}
    conditions = dict(prerequisites or {})
    conditions["input_trajectory_certified"] = bool(physical and full_window and dc_certified
                                                   and conditions.get("input_trajectory_certified", False))
    comparison = compare_reconstructed_response(reconstruction, ac, reconstruction_identity=identity,
                                                 prerequisites=conditions)
    return _ready({"schema": "R1StudyReconstructedResponseV1", "published_identity": identity,
                   "input_trajectory_certified": physical, "full_time_window_covered": full_window,
                   "dc_inputs_certified": bool(dc_certified), "reconstruction": reconstruction,
                   "comparison": comparison, "double_domain_consistent": comparison["double_domain_consistent"],
                   "scope": "verified_input_finite_window_reconstruction_with_conditional_error_budget"})


__all__ = ["frequency_window_report", "reconstruct_study_response"]
