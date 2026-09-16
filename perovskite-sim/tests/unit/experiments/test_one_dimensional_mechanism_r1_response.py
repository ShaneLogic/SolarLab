"""Independent circuit oracles and fail-closed R1 response comparison."""

from dataclasses import replace
import copy

import numpy as np
import pytest

from perovskite_sim.experiments.one_dimensional_mechanism_r1_admittance import (
    R1AdmittanceErrors, reconstruct_admittance,
)
from perovskite_sim.experiments.one_dimensional_mechanism_r1_response import (
    descriptor_frequency_response, compare_reconstructed_response,
    assess_small_signal_response,
)


def descriptor(**changes):
    args = dict(frequency_Hz=[0., .01, 1., 100.], storage=[[0.]], rate=[[-1.]],
                forcing=[1.], storage_voltage=[0.], conduction=[[0.]],
                conduction_voltage=[2.], displacement=[[0.]], displacement_voltage=[0.])
    args.update(changes)
    return descriptor_frequency_response(**args)


def test_pure_resistor_real_conductance_and_zero_displacement():
    result = descriptor()
    np.testing.assert_array_equal(result["admittance_S_m2"], 2.)
    np.testing.assert_array_equal(result["displacement_F_m2"], 0.)


def test_pure_capacitor_has_positive_imaginary_admittance():
    result = descriptor(conduction_voltage=[0.], displacement_voltage=[.25])
    np.testing.assert_allclose(result["admittance_S_m2"][:, 0],
                               2j*np.pi*result["frequency_Hz"]*.25)


def test_single_relaxation_matches_analytic_pole_and_limits():
    frequency = np.array([0., 1e-8, .01, 1., 1e8])
    result = descriptor(frequency_Hz=frequency, storage=[[.25]], displacement=[[.03125]],
                        conduction_voltage=[.125], displacement_voltage=[.0625])
    s = 2j*np.pi*frequency
    expected = .125+s*.0625+s*.03125/(1+s*.25)
    np.testing.assert_allclose(result["admittance_S_m2"][:, 0], expected, rtol=1e-14, atol=1e-14)
    assert result["admittance_S_m2"][0, 0] == .125
    assert result["admittance_S_m2"][-1, 0].real == pytest.approx(.25, rel=1e-12)
    assert max(result["linear_backward_error"]) < 1e-14


def test_voltage_storage_term_and_algebraic_state_are_not_discarded():
    # tau dx/dt = -x + V - mV*dV/dt: direct phasor substitution.
    result = descriptor(storage=[[.25]], storage_voltage=[.125], conduction=[[3.]],
                        conduction_voltage=[.125], displacement=[[.03125]],
                        displacement_voltage=[.0625])
    s = 2j*np.pi*result["frequency_Hz"]
    expected_state = (1-s*.125)/(1+s*.25)
    np.testing.assert_allclose(result["state_per_V"][:, 0], expected_state, rtol=1e-14)
    expected = (3+s*.03125)*expected_state+.125+s*.0625
    np.testing.assert_allclose(result["admittance_S_m2"][:, 0], expected, rtol=1e-14)


def test_algebraic_rows_and_dynamic_rows_solve_one_coupled_system():
    # dx/dt=-x+z and z=2V -> x=2/(s+1), observed x+z.
    result = descriptor(storage=[[1., 0.], [0., 0.]], rate=[[-1., 1.], [0., -1.]],
                        forcing=[0., 2.], storage_voltage=[0., 0.], conduction=[[1., 1.]],
                        conduction_voltage=[0.], displacement=[[0., 0.]])
    s = 2j*np.pi*result["frequency_Hz"]
    np.testing.assert_allclose(result["admittance_S_m2"][:, 0], 2+2/(1+s), rtol=1e-14)


def test_unknown_tail_prevents_error_budget_acceptance_even_when_curves_agree():
    frequency = [0., 1., 10.]
    time = reconstruct_admittance(frequency, time_s=[.1, 1.], regular_current_A_m2=[.25, .25],
                                  step_amplitude_V=.125, baseline_current_A_m2=0.,
                                  dc_conductance_S_m2=2., impulse_charge_C_m2=0.)
    direct = {"frequency_Hz": frequency, "admittance_S_m2": np.full(3, 2.+0j)}
    comparison = compare_reconstructed_response(time, direct)
    assert np.all(comparison["component_agreement"])
    assert not np.any(comparison["error_budget_agreement"])
    assert "tail_omission" in comparison["unknown_error_sources"]
    assert not comparison["double_domain_consistent"]


def test_finite_window_pole_comparison_is_bounded_by_independent_analytic_omissions():
    t = np.geomspace(1e-9, 5., 501)
    f = [0., .01, 1., 100.]
    delta_c, tau = .03125, .25
    direct = descriptor(frequency_Hz=f, storage=[[tau]], displacement=[[delta_c]],
                        conduction_voltage=[.125], displacement_voltage=[.0625])
    errors = R1AdmittanceErrors(
        early_omission_integral_F_m2=delta_c*(-np.expm1(-t[0]/tau)),
        tail_omission_integral_F_m2=delta_c*np.exp(-t[-1]/tau),
        interpolation_integral_F_m2=np.sum(delta_c/tau**3*np.exp(-t[:-1]/tau)*np.diff(t)**3/12),
        current_A_m2=0., baseline_current_A_m2=0., dc_conductance_S_m2=0., impulse_charge_C_m2=0.,
    )
    time = reconstruct_admittance(f, time_s=t, regular_current_A_m2=.125*(.125+delta_c/tau*np.exp(-t/tau)),
                                  step_amplitude_V=.125, baseline_current_A_m2=0.,
                                  dc_conductance_S_m2=.125, impulse_charge_C_m2=.125*.0625, errors=errors)
    comparison = compare_reconstructed_response(time, {
        "frequency_Hz": f, "admittance_S_m2": direct["admittance_S_m2"][:, 0],
    })
    difference = abs(time.admittance_S_m2-direct["admittance_S_m2"][:, 0])
    assert np.all(difference <= time.total_error_estimate_S_m2+1e-14)
    assert np.all(comparison["component_agreement"])
    # Good circuit math does not grant the physical device's other gates.
    assert not comparison["double_domain_consistent"]


@pytest.mark.parametrize("frequency", [[], [-1., 1.], [1., 1.], [1., 0.], [np.nan], [[1.]], [True]])
def test_rejects_bad_frequency_axes(frequency):
    with pytest.raises(ValueError):
        descriptor(frequency_Hz=frequency)


@pytest.mark.parametrize("change", [dict(storage=[[1., 2.]]), dict(rate=[[np.inf]]),
                                   dict(forcing=[1., 2.]), dict(conduction=[[1., 2.]]),
                                   dict(displacement_voltage=[0., 1.])])
def test_rejects_nonfinite_or_incompatible_descriptor(change):
    with pytest.raises(ValueError):
        descriptor(**change)


def test_rejects_comparison_of_different_frequency_axes():
    time = reconstruct_admittance([0., 1.], time_s=[.1, 1.], regular_current_A_m2=[1., 1.],
                                  step_amplitude_V=.125, baseline_current_A_m2=0.,
                                  dc_conductance_S_m2=8., impulse_charge_C_m2=0.)
    with pytest.raises(ValueError, match="frequencies"):
        compare_reconstructed_response(time, {"frequency_Hz": [0., 2.], "admittance_S_m2": [8., 8.]})


def synthetic_ac_record():
    """Two physical faces with separately retained conduction/displacement."""
    frequency = np.array([0., 1., 10.])
    levels = []
    for step in (1e-5, 5e-6, 2.5e-6):
        conduction = np.full((3, 2), 2.+0j)
        displacement = np.full((3, 2), .1+0j)
        levels.append({"derivative_step": step, "frequency_Hz": frequency.copy(),
                       "conduction_S_m2": conduction, "displacement_F_m2": displacement,
                       "admittance_S_m2": conduction+2j*np.pi*frequency[:, None]*displacement,
                       "electron_admittance_S_m2": conduction/2, "hole_admittance_S_m2": conduction/2,
                       "ion_admittance_S_m2": np.zeros((3, 2), dtype=complex),
                       "linear_backward_error": np.zeros(3), "unreplaced_equation_backward_error": np.zeros(3),
                       "inventory_response_thermal_normalized": np.zeros((3, 1)),
                       "legacy_inventory_response_relative": np.zeros(3),
                       "capture_storage_relative_error": np.zeros(3),
                       "state_jacobian_fd_column_relative_error": np.zeros(2)})
    return {"schema": "R1ControlledSmallSignalV1", "frequency_Hz": frequency, "voltage_V": 0.,
            "derivative_levels": levels, "admittance_S_m2": levels[-1]["admittance_S_m2"][:, 0].copy()}


@pytest.mark.parametrize("factor", [-1., .991, 1.009])
def test_published_admittance_cannot_diverge_from_its_physical_components(factor):
    record = synthetic_ac_record()
    record["admittance_S_m2"] *= factor
    report = assess_small_signal_response(record)
    assert not report["certified"]
    assert not np.any(report["checks"]["published_admittance_consistent"])
    if factor < 0:
        assert not np.any(report["checks"]["equilibrium_dissipation_sign"])


def test_dark_dissipation_gate_controls_each_frequency_without_imaginary_sign_rule():
    record = synthetic_ac_record()
    for level in record["derivative_levels"]:
        level["displacement_F_m2"] *= -1
        level["admittance_S_m2"] = level["conduction_S_m2"]+2j*np.pi*record["frequency_Hz"][:, None]*level["displacement_F_m2"]
    record["admittance_S_m2"] = record["derivative_levels"][-1]["admittance_S_m2"][:, 0].copy()
    assert assess_small_signal_response(record)["certified"]
    for level in record["derivative_levels"]:
        for field in ("conduction_S_m2", "electron_admittance_S_m2", "hole_admittance_S_m2", "admittance_S_m2"):
            level[field][1] *= -1
        level["displacement_F_m2"][1] *= -1
    record["admittance_S_m2"] = record["derivative_levels"][-1]["admittance_S_m2"][:, 0].copy()
    report = assess_small_signal_response(record)
    np.testing.assert_array_equal(report["numerically_eligible_frequency_points"], [True, False, True])
    record["voltage_V"] = .005
    assert assess_small_signal_response(record)["certified"]


@pytest.mark.parametrize("field,gate,limit", [
    ("linear_backward_error", "linear_backward_error", 1e-10),
    ("unreplaced_equation_backward_error", "all_equations_backward_error", 1e-10),
    ("inventory_response_thermal_normalized", "inventory", 1e-10),
    ("legacy_inventory_response_relative", "legacy_inventory", 1e-8),
    ("capture_storage_relative_error", "capture_storage", 1e-3),
    ("state_jacobian_fd_column_relative_error", "direct_tangent_difference", 3e-4),
])
def test_each_response_metric_threshold_changes_eligibility(field, gate, limit):
    record = synthetic_ac_record()
    value = record["derivative_levels"][-1][field]
    value.flat[0] = limit*(1-1e-6)
    assert assess_small_signal_response(record)["certified"]
    value.flat[0] = limit*(1+1e-6)
    report = assess_small_signal_response(record)
    assert not report["certified"]
    assert not report["checks"][gate][0]


def test_coarse_derivative_level_cannot_hide_a_failed_linear_solve():
    record = synthetic_ac_record()
    record["derivative_levels"][0]["linear_backward_error"][1] = 1.01e-10
    report = assess_small_signal_response(record)
    np.testing.assert_array_equal(report["numerically_eligible_frequency_points"], [True, False, True])


@pytest.mark.parametrize("gate,limit", [("physical_face_spread", 5e-4), ("current_decomposition", 1e-7)])
def test_face_and_decomposition_limits_use_recomputed_components(gate, limit):
    for factor, expected in ((1-1e-6, True), (1+1e-6, False)):
        record = synthetic_ac_record()
        for level in record["derivative_levels"]:
            if gate == "physical_face_spread":
                delta = 2*limit/(1-limit)*factor
                level["conduction_S_m2"][0, 1] += delta
                level["electron_admittance_S_m2"][0, 1] += delta
                level["admittance_S_m2"][0, 1] += delta
            else:
                level["electron_admittance_S_m2"][0, 1] += 2*limit*factor
        report = assess_small_signal_response(record)
        assert bool(report["checks"][gate][0]) is expected
        assert report["certified"] is expected


def test_derivative_refinement_threshold_and_level_selection_are_binding():
    for factor, expected in ((1-1e-6, True), (1+1e-6, False)):
        record = synthetic_ac_record()
        delta = 2*2e-3/(1-2e-3)*factor
        for level in record["derivative_levels"][:2]:
            level["conduction_S_m2"][0] += delta
            level["electron_admittance_S_m2"][0] += delta
            level["admittance_S_m2"][0] += delta
        assert bool(assess_small_signal_response(record)["checks"]["derivative_refinement"][0]) is expected
    record["admittance_S_m2"] = record["derivative_levels"][0]["admittance_S_m2"][:, 0].copy()
    assert not assess_small_signal_response(record)["certified"]
    record["derivative_levels"][0]["derivative_step"] = 2e-5
    with pytest.raises(ValueError, match="three derivative steps"):
        assess_small_signal_response(record)


def test_small_real_derivative_component_cannot_hide_behind_large_imaginary_response():
    for factor, expected in ((1-1e-6, True), (1+1e-6, False)):
        record = synthetic_ac_record()
        for level in record["derivative_levels"]:
            level["displacement_F_m2"][:] = 1e8
            level["admittance_S_m2"] = level["conduction_S_m2"]+2j*np.pi*record["frequency_Hz"][:, None]*level["displacement_F_m2"]
        delta = (.02+1e-8)/.99*factor
        for level in record["derivative_levels"][:2]:
            level["conduction_S_m2"][1] += delta
            level["electron_admittance_S_m2"][1] += delta
            level["admittance_S_m2"][1] += delta
        record["admittance_S_m2"] = record["derivative_levels"][-1]["admittance_S_m2"][:, 0].copy()
        report = assess_small_signal_response(record)
        assert report["checks"]["derivative_refinement"][1]
        assert bool(report["checks"]["derivative_component_refinement"][1]) is expected


def test_stored_checks_and_eligibility_cannot_disagree_with_the_numbers():
    record = synthetic_ac_record()
    report = assess_small_signal_response(record)
    record.update({key: copy.deepcopy(report[key]) for key in
                   ("checks", "numerically_eligible_frequency_points", "equilibrium_dissipation_sign_observation")})
    record["checks"]["inventory"][1] = False
    result = assess_small_signal_response(record)
    np.testing.assert_array_equal(result["numerically_eligible_frequency_points"], [True, False, True])
    record["checks"].pop("inventory")
    with pytest.raises(ValueError, match="omit or add"):
        assess_small_signal_response(record)


def test_double_domain_requires_matching_identity_and_every_frequency_prerequisite():
    ac = synthetic_ac_record()
    ac.update(prepared_sha256="prepared", reference_sha256="reference", intervals=16,
              control="D", source={"sha256": "source"})
    identity = {"prepared_sha256": "prepared", "reference_sha256": "reference", "intervals": 16,
                "control": "D", "source_sha256": "source", "operating_voltage_V": 0.}
    conditions = {key: True for key in ("finite_amplitude_linearity", "single_axis_convergence", "window_extension",
                  "earlier_start", "stricter_integration", "tail_dc_agreement", "frequency_window_coverage",
                  "input_trajectory_certified", "ac_content_verified")}
    errors = R1AdmittanceErrors(early_omission_integral_F_m2=0., tail_omission_integral_F_m2=0.,
                              interpolation_integral_F_m2=0., current_A_m2=0., baseline_current_A_m2=0.,
                              dc_conductance_S_m2=0., impulse_charge_C_m2=0.)
    time = reconstruct_admittance(ac["frequency_Hz"], time_s=[.1, 1.], regular_current_A_m2=[.25, .25],
                                  step_amplitude_V=.125, baseline_current_A_m2=0., dc_conductance_S_m2=2.,
                                  impulse_charge_C_m2=.0125, errors=errors)
    good = compare_reconstructed_response(time, ac, reconstruction_identity=identity, prerequisites=conditions)
    assert good["double_domain_consistent"]
    assert not compare_reconstructed_response(time, ac, prerequisites=conditions)["double_domain_consistent"]
    for key in conditions:
        changed = {**conditions, key: [True, False, True]}
        result = compare_reconstructed_response(time, ac, reconstruction_identity=identity, prerequisites=changed)
        np.testing.assert_array_equal(result["double_domain_consistent_frequency_points"], [True, False, True])
        assert not result["double_domain_consistent"]
    with pytest.raises(ValueError, match="identity differs"):
        compare_reconstructed_response(time, ac, reconstruction_identity={**identity, "intervals": 32}, prerequisites=conditions)
