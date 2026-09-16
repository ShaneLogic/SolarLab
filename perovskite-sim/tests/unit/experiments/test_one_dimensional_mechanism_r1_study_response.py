"""Study adapters bind identities and keep unmeasured windows/uncertainty open."""

import copy
import json

import numpy as np
import pytest

from perovskite_sim.experiments.one_dimensional_mechanism_r1_admittance import R1AdmittanceErrors
from perovskite_sim.experiments.one_dimensional_mechanism_r1_convergence import observation_times
from perovskite_sim.experiments.one_dimensional_mechanism_r1_study_response import (
    frequency_window_report, reconstruct_study_response,
)
from tests.unit.experiments.test_one_dimensional_mechanism_r1_response import synthetic_ac_record


def inputs(*, times=None):
    ac = synthetic_ac_record()
    common = {"sha256": "prepared", "reference_sha256": "reference", "intervals": 16,
              "source": {"sha256": "source"}}
    identity = {"prepared_sha256": "prepared", "reference_sha256": "reference", "intervals": 16,
                "source": {"sha256": "source"}, "control": "D"}
    ac.update(identity)
    baseline = {**identity, "voltage_V": 0., "terminal_current_A_m2": 0., "certified": True}
    widths = [1e-4, 5e-5, 2.5e-5]
    pairs = [{"half_width_V": width,
              "minus": {**baseline, "voltage_V": -width, "terminal_current_A_m2": -2*width},
              "plus": {**baseline, "voltage_V": width, "terminal_current_A_m2": 2*width}} for width in widths]
    dc = {"schema": "R1DCConductanceStudyV1", "control": "D", "baseline": baseline,
          "half_width_V": widths, "pairs": pairs, "conductance_S_m2": [2., 2., 2.], "finest_pair_agrees": True}
    times = observation_times() if times is None else np.asarray(times)
    step = {**identity, "schema": "R1ControlledStepV1", "control_label": "D", "amplitude_V": .005,
            "times_s": times, "voltage_V": np.full(len(times), .005),
            "certificate": {"certified": True}, "initial_event": {"impulse_charge_C_m2": .0005,
                "zero_minus": {"voltage_V": 0.}, "zero_plus": {"voltage_V": .005}, "voltage_jump_V": .005},
            "regular_currents": [{"report_contact_current_A_m2": [.01, .01]} for _ in times]}
    step.pop("control")
    conditions = {key: True for key in ("finite_amplitude_linearity", "single_axis_convergence", "window_extension",
                  "earlier_start", "stricter_integration", "tail_dc_agreement", "frequency_window_coverage",
                  "input_trajectory_certified", "ac_content_verified")}
    errors = R1AdmittanceErrors(early_omission_integral_F_m2=0., tail_omission_integral_F_m2=0.,
                              interpolation_integral_F_m2=0., current_A_m2=0., baseline_current_A_m2=0.,
                              dc_conductance_S_m2=0., impulse_charge_C_m2=0.)
    return step, common, dc, ac, conditions, errors


def test_frequency_extension_limits_and_numeric_bands_do_not_prove_turnover_coverage():
    ac = synthetic_ac_record()
    report = frequency_window_report(ac)
    assert len(report["valid_contiguous_bands"]) == 1
    assert not report["frequency_window_complete"]
    assert "protocol_decade_extensions_not_exhausted" in report["uncovered_reasons"]
    frequency = np.r_[0., np.logspace(-6, 10, 65)]
    # The same independently specified resistor/capacitor primitive at each point.
    ac["frequency_Hz"] = frequency
    for level in ac["derivative_levels"]:
        old_count = len(level["frequency_Hz"])
        for key, value in tuple(level.items()):
            if isinstance(value, np.ndarray) and value.shape[0] == old_count:
                level[key] = np.repeat(value[:1], len(frequency), axis=0)
        level["frequency_Hz"] = frequency.copy()
        level["admittance_S_m2"] = level["conduction_S_m2"]+2j*np.pi*frequency[:, None]*level["displacement_F_m2"]
    ac["admittance_S_m2"] = ac["derivative_levels"][-1]["admittance_S_m2"][:, 0].copy()
    report = frequency_window_report(ac)
    assert report["lower_extension_limit_reached"] and report["upper_extension_limit_reached"]
    assert not report["frequency_window_complete"]
    assert report["endpoint_decades"]["lower"]["full_decade_sampled"]
    assert report["endpoint_decades"]["upper"]["full_decade_sampled"]
    assert report["endpoint_decades"]["upper"]["capacitance_relative_change"] < 1e-14
    ac["derivative_levels"][-1]["linear_backward_error"][10] = 1e-8
    report = frequency_window_report(ac)
    assert len(report["valid_contiguous_bands"]) == 2
    assert report["valid_contiguous_bands"][0]["last_index"] == 9
    assert report["valid_contiguous_bands"][1]["first_index"] == 11
    json.dumps(report, allow_nan=False)


def test_reconstruction_preserves_unknown_errors_and_conditional_acceptance():
    step, common, dc, ac, conditions, errors = inputs()
    unknown = reconstruct_study_response(step, common, dc, ac)
    assert not unknown["double_domain_consistent"]
    assert "tail_omission" in unknown["reconstruction"]["unknown_error_sources"]
    assert all(item["representation"] == "inf" for item in unknown["reconstruction"]["total_error_estimate_S_m2"])
    good = reconstruct_study_response(step, common, dc, ac, errors=errors, prerequisites=conditions)
    assert good["double_domain_consistent"]
    assert good["published_identity"]["step_amplitude_V"] == .005
    json.dumps(good, allow_nan=False)
    conditions["input_trajectory_certified"] = False
    assert not reconstruct_study_response(step, common, dc, ac, errors=errors, prerequisites=conditions)["double_domain_consistent"]


@pytest.mark.parametrize("which,key,value", [
    ("step", "prepared_sha256", "other"), ("ac", "intervals", 32),
    ("ac", "source", {"sha256": "other"}), ("step", "control_label", "A"),
    ("baseline", "prepared_sha256", "other"), ("baseline", "voltage_V", .005),
])
def test_identity_is_read_from_saved_records_and_mismatches_are_rejected(which, key, value):
    step, common, dc, ac, _, _ = inputs()
    {"step": step, "ac": ac, "baseline": dc["baseline"]}[which][key] = value
    with pytest.raises(ValueError, match="identity|zero-bias"):
        reconstruct_study_response(step, common, dc, ac)


def test_failed_or_short_window_inputs_cannot_be_promoted_by_good_curves():
    step, common, dc, ac, conditions, errors = inputs(times=[0., 1e-9, 1e-4])
    result = reconstruct_study_response(step, common, dc, ac, errors=errors, prerequisites=conditions)
    assert all(result["comparison"]["component_agreement"])
    assert not result["full_time_window_covered"]
    assert not result["double_domain_consistent"]
    step, common, dc, ac, conditions, errors = inputs()
    step["certificate"]["certified"] = False
    assert not reconstruct_study_response(step, common, dc, ac, errors=errors, prerequisites=conditions)["double_domain_consistent"]
    step["certificate"]["certified"] = True
    dc["pairs"][0]["minus"]["certified"] = False
    assert not reconstruct_study_response(step, common, dc, ac, errors=errors, prerequisites=conditions)["double_domain_consistent"]
    dc["conductance_S_m2"][-1] *= 1.001
    with pytest.raises(ValueError, match="published conductance"):
        reconstruct_study_response(step, common, dc, ac, errors=errors, prerequisites=conditions)


def test_step_operating_bias_is_read_from_the_saved_initial_event():
    step, common, dc, ac, _, _ = inputs()
    step["initial_event"]["zero_minus"]["voltage_V"] = .002
    with pytest.raises(ValueError, match="operating voltage identity"):
        reconstruct_study_response(step, common, dc, ac)


def test_study_reconstruction_forwards_and_records_stricter_quadrature_tolerances():
    step, common, dc, ac, conditions, errors = inputs()
    result = reconstruct_study_response(step, common, dc, ac, errors=errors, prerequisites=conditions,
        quadrature_absolute_tolerance_F_m2=1e-14, quadrature_relative_tolerance=1e-12)
    assert result["reconstruction"]["quadrature_absolute_tolerance_F_m2"] == 1e-14
    assert result["reconstruction"]["quadrature_relative_tolerance"] == 1e-12
    assert result["double_domain_consistent"]
    with pytest.raises(ValueError, match="quadrature"):
        reconstruct_study_response(step, common, dc, ac, quadrature_relative_tolerance=-1.)
