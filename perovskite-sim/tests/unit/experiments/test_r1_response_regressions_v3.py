"""Regression cases for the nine non-equivalent response survivors in review 8."""

from types import SimpleNamespace
import numpy as np
import pytest

from perovskite_sim.experiments import one_dimensional_mechanism_r1_response as response_module
from perovskite_sim.experiments.one_dimensional_mechanism_r1_response import (
    assess_small_signal_response, compare_reconstructed_response, dc_amplitude_endpoint_study,
)
from perovskite_sim.experiments.one_dimensional_mechanism_r1_study_response import (
    frequency_window_report, reconstruct_study_response,
)
from tests.unit.experiments.test_one_dimensional_mechanism_r1_response import synthetic_ac_record
from tests.unit.experiments.test_one_dimensional_mechanism_r1_study_response import inputs
from tests.unit.experiments.test_r1_response_qualification import pole_record


def test_m15_each_saved_level_admittance_is_bound_to_its_physical_components():
    ac = synthetic_ac_record()
    assert assess_small_signal_response(ac)["certified"]
    ac["derivative_levels"][0]["admittance_S_m2"][1, 1] += .1
    result = assess_small_signal_response(ac)
    failing = {key for key, mask in result["checks"].items() if not mask[1]}
    assert failing == {"level_admittance_consistent"}
    assert not result["numerically_eligible_frequency_points"][1]


@pytest.mark.parametrize("face", [2, 3])
def test_m18_internal_faces_cannot_be_dropped_from_the_spread_gate(face):
    ac = synthetic_ac_record()
    for level in ac["derivative_levels"]:
        for key in ("conduction_S_m2", "displacement_F_m2", "admittance_S_m2",
                    "electron_admittance_S_m2", "hole_admittance_S_m2", "ion_admittance_S_m2"):
            level[key] = np.repeat(level[key][:, :1], 4, axis=1)
    assert assess_small_signal_response(ac)["certified"]
    delta = 2*5e-4/(1-5e-4)*1.001
    for level in ac["derivative_levels"]:
        for key in ("conduction_S_m2", "electron_admittance_S_m2", "admittance_S_m2"):
            level[key][0, face] += delta
    result = assess_small_signal_response(ac)
    assert {key for key, mask in result["checks"].items() if not mask[0]} == {"physical_face_spread"}
    assert not result["certified"]


def test_m23_endpoint_scale_is_exactly_one_percent_and_remains_diagnostic(monkeypatch):
    calls = []

    def fake_dc(stack, intervals, binding, prepared, *, voltage_V, **kwargs):
        calls.append(voltage_V)
        return SimpleNamespace(evidence={"voltage_V": voltage_V, "terminal_current_A_m2": 2*voltage_V+3*voltage_V**2,
            "control": "D", "prepared_sha256": "prepared", "reference_sha256": "reference", "source": {"sha256": "analytic"},
            "junction_polarity": -1., "current_sign_convention": "analytic-test", "certified": True})

    monkeypatch.setattr(response_module, "solve_controlled_dc", fake_dc)
    report = dc_amplitude_endpoint_study(None, 16, {}, None)
    assert calls == [0., .01, .005, .0025, .00125, .000625, .0003125, .00015625]
    for pair in report["adjacent_halving_diagnostics"]:
        a, b = pair["coarse_amplitude_V"], pair["fine_amplitude_V"]
        expected_scale = .01*max(abs(2+3*a), abs(2+3*b))
        assert pair["one_percent_response_scale_S_m2"] == pytest.approx(expected_scale, rel=1e-14)
        assert pair["difference_to_one_percent_scale_ratio"] == pytest.approx(abs(3*a-3*b)/expected_scale, rel=1e-11)
        assert pair["absolute_numerical_error_budget_S_m2"] is None
        assert not pair["linearity_certified"]
    assert report["absolute_current_error_bounds_A_m2"] is None
    assert not report["full_transient_linearity_certified"]


def comparison_inputs():
    ac = synthetic_ac_record()
    ac.update(prepared_sha256="prepared", reference_sha256="reference", intervals=16,
              control="D", source={"sha256": "source"})
    reconstructed = SimpleNamespace(frequency_Hz=ac["frequency_Hz"].copy(),
        admittance_S_m2=ac["admittance_S_m2"].copy(), total_error_estimate_S_m2=np.zeros(3), unknown_error_sources=[])
    identity = {"prepared_sha256": "prepared", "reference_sha256": "reference", "intervals": 16,
                "control": "D", "source_sha256": "source", "operating_voltage_V": 0.}
    prerequisites = {key: True for key in ("finite_amplitude_linearity", "single_axis_convergence", "window_extension",
        "earlier_start", "stricter_integration", "tail_dc_agreement", "frequency_window_coverage",
        "input_trajectory_certified", "ac_content_verified")}
    return reconstructed, ac, dict(reconstruction_identity=identity, prerequisites=prerequisites)


def test_m26_double_domain_component_budget_is_not_a_ten_percent_budget():
    reconstructed, ac, kwargs = comparison_inputs()
    assert compare_reconstructed_response(reconstructed, ac, **kwargs)["double_domain_consistent"]
    reconstructed.admittance_S_m2[0] += .03
    result = compare_reconstructed_response(reconstructed, ac, **kwargs)
    assert not result["component_agreement"][0]
    assert result["limits_components_S_m2"][0, 0] == pytest.approx(1e-8+.01*2.03)
    assert not result["double_domain_consistent"]


def test_m27_ac_numeric_eligibility_remains_required_even_for_identical_curves():
    reconstructed, ac, kwargs = comparison_inputs()
    assert compare_reconstructed_response(reconstructed, ac, **kwargs)["double_domain_consistent"]
    ac["derivative_levels"][0]["linear_backward_error"][1] = 2e-10
    result = compare_reconstructed_response(reconstructed, ac, **kwargs)
    assert np.all(result["component_agreement"])
    assert not result["double_domain_consistent_frequency_points"][1]


def test_m29_frequency_protocol_limits_filter_otherwise_numeric_points():
    ac, _ = pole_record([0., 1e-7, 1., 1e11])
    assert assess_small_signal_response(ac)["certified"]
    report = frequency_window_report(ac)
    assert report["numerically_eligible_frequency_points"] == [True, False, True, False]
    assert "frequency_samples_outside_protocol_limits" in report["uncovered_reasons"]


def test_m30_frequency_numeric_summary_tracks_the_failing_gate():
    ac = synthetic_ac_record()
    assert frequency_window_report(ac)["numeric_checks_passed"]
    ac["derivative_levels"][1]["capture_storage_relative_error"][1] = .002
    report = frequency_window_report(ac)
    assert not report["numeric_checks_passed"]
    assert report["numerically_eligible_frequency_points"] == [True, False, True]


def test_m33_reconstruction_must_be_about_the_same_zero_bias_state():
    step, common, dc, ac, conditions, errors = inputs()
    # Preserve downstream event identities so the zero-bias guard is the
    # specific rejection, rather than a later unrelated field mismatch.
    dc["baseline"]["voltage_V"] = ac["voltage_V"] = .001
    step["initial_event"]["zero_minus"]["voltage_V"] = .001
    step["initial_event"]["voltage_jump_V"] = .004
    with pytest.raises(ValueError, match="same zero-bias operating state"):
        reconstruct_study_response(step, common, dc, ac, errors=errors, prerequisites=conditions)


@pytest.mark.parametrize("amplitude,valid", [(.019, True), (.02, False), (.020001, False)])
def test_m35_small_step_ceiling_is_twenty_millivolts(amplitude, valid):
    step, common, dc, ac, conditions, errors = inputs()
    step["amplitude_V"] = amplitude
    step["voltage_V"] = np.full(len(step["times_s"]), amplitude)
    step["initial_event"]["zero_plus"]["voltage_V"] = amplitude
    step["initial_event"]["voltage_jump_V"] = amplitude
    if valid:
        assert not reconstruct_study_response(step, common, dc, ac, errors=errors,
                                              prerequisites=conditions)["double_domain_consistent"]
    else:
        with pytest.raises(ValueError, match="small-step range"):
            reconstruct_study_response(step, common, dc, ac, errors=errors, prerequisites=conditions)
