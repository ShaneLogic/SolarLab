"""Independent circuit oracles and fail-closed R1 response comparison."""

from dataclasses import replace

import numpy as np
import pytest

from perovskite_sim.experiments.one_dimensional_mechanism_r1_admittance import (
    R1AdmittanceErrors, reconstruct_admittance,
)
from perovskite_sim.experiments.one_dimensional_mechanism_r1_response import (
    descriptor_frequency_response, compare_reconstructed_response,
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
