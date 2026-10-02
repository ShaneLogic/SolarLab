"""Independent R/C/single-pole oracles and deliberate reconstruction faults."""

from dataclasses import replace

import numpy as np
import pytest

import perovskite_sim.experiments.one_dimensional_mechanism_r1_admittance as module
from perovskite_sim.experiments.one_dimensional_mechanism_r1_admittance import (
    R1AdmittanceErrors, R1AdmittanceIntegrationError,
    linear_interpolation_integral_bound, reconstruct_admittance,
)


EXACT_INPUTS = R1AdmittanceErrors(0., 0., 0., 0., 0., 0., 0.)


def constant_case(**changes):
    args = dict(frequency_Hz=[0., .01, 1., 1e4], time_s=[.125, .25, 1.],
                regular_current_A_m2=[.75, .75, .75], step_amplitude_V=.125,
                baseline_current_A_m2=.5, dc_conductance_S_m2=2.,
                impulse_charge_C_m2=0., errors=EXACT_INPUTS)
    args.update(changes)
    return reconstruct_admittance(**args)


def relaxation_case(**changes):
    # Independently declared series-RC branch: Delta C=.03125 F/m2,
    # tau=.25 s, in parallel with G=.125 S/m2 and Cinf=.0625 F/m2.
    # r''=(Delta C/tau**3)exp(-t/tau) decreases on every interval.
    time = np.geomspace(1e-8, 4., 361)
    amplitude, baseline = .125, .0625
    current = baseline+amplitude*(.125+.125*np.exp(-time/.25))
    widths = np.diff(time)
    interpolation_bound = np.sum(2.*np.exp(-time[:-1]/.25)*widths**3/12)
    errors = replace(EXACT_INPUTS,
                     early_omission_integral_F_m2=.03125*(-np.expm1(-time[0]/.25)),
                     tail_omission_integral_F_m2=.03125*np.exp(-time[-1]/.25),
                     interpolation_integral_F_m2=interpolation_bound)
    args = dict(frequency_Hz=[0., 1e-3, .1, 1., 10., 100., 1e4],
                time_s=time, regular_current_A_m2=current, step_amplitude_V=amplitude,
                baseline_current_A_m2=baseline, dc_conductance_S_m2=.125,
                impulse_charge_C_m2=amplitude*.0625, errors=errors)
    args.update(changes)
    return reconstruct_admittance(**args)


def analytic_relaxation(frequency):
    # Analytic transfer function, independently transcribed from §7.3.
    s = 2j*np.pi*np.asarray(frequency)
    return .125+s*.0625+s*.03125/(1+s*.25)


def assert_within_reported_uncertainty(result, expected):
    discrepancy = np.abs(result.admittance_S_m2-expected)
    assert np.all(discrepancy <= result.total_error_estimate_S_m2+1e-14)


def test_pure_resistor_has_g0_without_transient_or_capacitance():
    result = constant_case()
    np.testing.assert_array_equal(result.admittance_S_m2, 2.+0j)
    np.testing.assert_array_equal(result.residual_integral_F_m2, 0.)
    assert result.impulse_capacitance_F_m2 == 0.
    assert result.unknown_error_sources == ()


def test_pure_capacitor_has_positive_imaginary_admittance_and_exact_impulse():
    result = constant_case(regular_current_A_m2=[.5]*3, dc_conductance_S_m2=0.,
                           impulse_charge_C_m2=.125*.75)
    np.testing.assert_allclose(result.admittance_S_m2, 2j*np.pi*result.frequency_Hz*.75,
                               rtol=1e-15, atol=0.)
    assert result.impulse_capacitance_F_m2 == .75
    assert np.all(result.admittance_S_m2.imag[1:] > 0)
    np.testing.assert_array_equal(result.admittance_S_m2.real, 0.)


def test_single_relaxation_matches_independent_analytic_transfer_function():
    result = relaxation_case()
    assert_within_reported_uncertainty(result, analytic_relaxation(result.frequency_Hz))
    # Test actual accuracy separately from the conservative L1 error budget.
    np.testing.assert_allclose(result.admittance_S_m2,
                               analytic_relaxation(result.frequency_Hz), rtol=2e-4, atol=2e-5)
    assert np.all(result.admittance_S_m2.real >= 0)
    assert result.admittance_S_m2[0] == .125


def test_single_relaxation_low_and_high_frequency_limits():
    result = relaxation_case(frequency_Hz=[1e-5, 1e5])
    assert result.admittance_S_m2[0].real == pytest.approx(.125, abs=1e-9)
    assert result.admittance_S_m2[0].imag/(2*np.pi*1e-5) == pytest.approx(.09375, rel=.001)
    # Subtract known ideal capacitor to expose the high-frequency branch.
    branch = result.admittance_S_m2[1]-.125-2j*np.pi*1e5*.0625
    assert branch.real == pytest.approx(.125, rel=.001)
    assert abs(branch.imag) < 1e-3


def test_negative_step_preserves_signed_normalization():
    positive = relaxation_case()
    time = np.geomspace(1e-8, 4., 361)
    negative = relaxation_case(step_amplitude_V=-.125, impulse_charge_C_m2=-.125*.0625,
                               regular_current_A_m2=.0625-.125*(.125+.125*np.exp(-time/.25)))
    np.testing.assert_allclose(negative.admittance_S_m2, positive.admittance_S_m2, atol=1e-12)


def test_voltage_and_charge_normalization_is_invariant_to_step_amplitude():
    first = constant_case(impulse_charge_C_m2=.03125)
    second = constant_case(step_amplitude_V=.25, regular_current_A_m2=[1.]*3,
                           impulse_charge_C_m2=.0625)
    np.testing.assert_array_equal(first.admittance_S_m2, second.admittance_S_m2)


def test_finite_window_is_not_filled_with_an_exponential_tail():
    time = np.linspace(.01, .25, 501)
    result = reconstruct_admittance([.1, 1., 10.], time_s=time,
        regular_current_A_m2=.125*np.exp(-time/.25), step_amplitude_V=1.,
        baseline_current_A_m2=0., dc_conductance_S_m2=0., impulse_charge_C_m2=0.)
    w = 2*np.pi*result.frequency_Hz
    k = 4.+1j*w
    finite_integral = .125*(np.exp(-k*time[0])-np.exp(-k*time[-1]))/k
    np.testing.assert_allclose(result.residual_integral_F_m2, finite_integral, rtol=1e-6)
    infinite_admittance = 1j*w*.03125/(1+1j*w*.25)
    assert np.max(np.abs(result.admittance_S_m2-infinite_admittance)) > .01
    assert "early_omission" in result.unknown_error_sources
    assert "tail_omission" in result.unknown_error_sources
    assert np.all(np.isinf(result.total_error_estimate_S_m2))


def test_early_omission_and_tail_bounds_cover_missing_single_pole_intervals():
    time = np.linspace(.01, .25, 501)
    result = relaxation_case(time_s=time,
        regular_current_A_m2=.0625+.125*(.125+.125*np.exp(-time/.25)),
        errors=replace(EXACT_INPUTS,
            early_omission_integral_F_m2=.03125*(1-np.exp(-time[0]/.25)),
            tail_omission_integral_F_m2=.03125*np.exp(-time[-1]/.25),
            interpolation_integral_F_m2=2.*np.sum(np.diff(time)**3)/12))
    assert_within_reported_uncertainty(result, analytic_relaxation(result.frequency_Hz))
    assert np.all(result.error_components_S_m2["early_omission"][1:] > 0)
    assert np.all(result.error_components_S_m2["tail_omission"][1:] > 0)


def test_curvature_bound_is_exact_for_quadratic_interpolation_error():
    # r=t^2, M=2. Its linear interpolant over [1,3] is 4t-3;
    # integral((4t-3)-t^2,1,3)=4/3 exactly.
    bound = linear_interpolation_integral_bound([1., 3.], 2.)
    assert bound == pytest.approx(4/3)
    assert linear_interpolation_integral_bound([1., 2., 3.], [2., 2.]) == pytest.approx(1/3)


def test_sample_current_bound_is_voltage_normalized_and_time_integrated():
    result = constant_case(errors=replace(EXACT_INPUTS, current_A_m2=[.1, .2, .4]))
    # Trapezoidal integral of error envelope: .125*.15+.75*.3=.24375 A*s/m2.
    np.testing.assert_allclose(result.error_components_S_m2["sample_current"],
                               2*np.pi*result.frequency_Hz*(.24375/.125))


def test_dc_error_includes_dc_term_and_subtracted_residual():
    result = constant_case(errors=replace(EXACT_INPUTS, dc_conductance_S_m2=.03))
    np.testing.assert_allclose(result.error_components_S_m2["dc_conductance"],
                               .03*(1+2*np.pi*result.frequency_Hz*.875))
    assert result.error_components_S_m2["dc_conductance"][0] == .03


def test_baseline_and_impulse_uncertainties_keep_their_dimensions():
    result = constant_case(errors=replace(EXACT_INPUTS, baseline_current_A_m2=.02,
                                          impulse_charge_C_m2=.03))
    w = 2*np.pi*result.frequency_Hz
    np.testing.assert_allclose(result.error_components_S_m2["baseline_current"], w*.875*.02/.125)
    np.testing.assert_allclose(result.error_components_S_m2["impulse_charge"], w*.03/.125)


def test_integral_error_is_converted_to_admittance_error_with_omega(monkeypatch):
    monkeypatch.setattr(module, "_integrate_piecewise_linear", lambda *args: (0j, .123))
    result = constant_case()
    np.testing.assert_allclose(result.error_components_S_m2["quadrature_estimate"],
                               .123*2*np.pi*result.frequency_Hz)
    assert result.error_components_S_m2["quadrature_estimate"][0] == 0.


@pytest.mark.parametrize("name", list(R1AdmittanceErrors.__dataclass_fields__))
def test_each_missing_error_source_stays_unbounded(name):
    result = constant_case(errors=replace(EXACT_INPUTS, **{name: None}))
    assert len(result.unknown_error_sources) == 1
    assert np.all(np.isinf(result.total_error_estimate_S_m2))
    assert result.scope == "finite_window_reconstruction_only"
    assert not hasattr(result, "valid_frequency_band")
    assert not hasattr(result, "passed")


def test_budget_is_sum_of_all_separate_sources():
    result = constant_case(errors=R1AdmittanceErrors(.1, .2, .3, .4, .5, .6, .7))
    np.testing.assert_array_equal(result.total_error_estimate_S_m2,
                                  np.sum(tuple(result.error_components_S_m2.values()), axis=0))
    assert set(result.error_components_S_m2) == {
        "early_omission", "tail_omission", "interpolation", "sample_current",
        "baseline_current", "dc_conductance", "impulse_charge",
        "quadrature_estimate", "floating_point_estimate",
    }


@pytest.mark.parametrize("fault", ["charge_sign", "missing_impulse", "amplitude_mV_as_V"])
def test_pure_capacitor_oracle_detects_sign_impulse_and_voltage_unit_faults(fault):
    args = dict(regular_current_A_m2=[.5]*3, dc_conductance_S_m2=0.,
                impulse_charge_C_m2=.125*.75)
    if fault == "charge_sign":
        args["impulse_charge_C_m2"] *= -1
    elif fault == "missing_impulse":
        args["impulse_charge_C_m2"] = 0.
    else:
        args["step_amplitude_V"] = 125.
    result = constant_case(**args)
    expected = 2j*np.pi*result.frequency_Hz*.75
    assert np.all(np.abs(result.admittance_S_m2[1:]-expected[1:])
                  > result.total_error_estimate_S_m2[1:])


def test_pure_resistor_oracle_detects_missing_dc_term():
    # The test deliberately lies about G0: finite-window transient cannot
    # repair its zero-frequency error, and this must fail the analytic oracle.
    result = constant_case(dc_conductance_S_m2=0.)
    assert abs(result.admittance_S_m2[0]-2.) > result.total_error_estimate_S_m2[0]


def test_single_pole_oracle_detects_fourier_sign_error(monkeypatch):
    real_integrator = module._integrate_piecewise_linear

    def wrong_sign(*args):
        value, error = real_integrator(*args)
        return value.conjugate(), error

    monkeypatch.setattr(module, "_integrate_piecewise_linear", wrong_sign)
    result = relaxation_case(frequency_Hz=[1.])
    assert abs(result.admittance_S_m2[0]-analytic_relaxation([1.])[0]) > result.total_error_estimate_S_m2[0]
    assert result.admittance_S_m2[0].real < .125


def test_using_hz_as_angular_frequency_fails_capacitor_oracle():
    result = constant_case(regular_current_A_m2=[.5]*3, dc_conductance_S_m2=0.,
                           impulse_charge_C_m2=.125*.75)
    wrong_units = 1j*result.frequency_Hz*.75
    assert np.all(np.abs(result.admittance_S_m2[1:]-wrong_units[1:])
                  > result.total_error_estimate_S_m2[1:])


def test_quadrature_tightening_does_not_change_resolved_result():
    first = relaxation_case(frequency_Hz=[.1, 1., 10.])
    second = relaxation_case(frequency_Hz=[.1, 1., 10.],
                             quadrature_absolute_tolerance_F_m2=1e-14,
                             quadrature_relative_tolerance=1e-12)
    np.testing.assert_allclose(first.admittance_S_m2, second.admittance_S_m2, rtol=1e-12, atol=1e-14)


def test_array_inputs_are_not_mutated_and_results_are_readonly():
    frequency = np.array([0., 1.])
    result = constant_case(frequency_Hz=frequency)
    frequency[:] = 100.
    np.testing.assert_array_equal(result.frequency_Hz, [0., 1.])
    for array in (result.admittance_S_m2, result.frequency_Hz, result.residual_integral_F_m2,
                  result.total_error_estimate_S_m2, result.error_components_S_m2["interpolation"]):
        with pytest.raises(ValueError):
            array[0] = 123.
    with pytest.raises(TypeError):
        result.error_components_S_m2["interpolation"] = np.zeros(2)


@pytest.mark.parametrize("changes", [
    {"time_s": [0., .25, 1.]}, {"time_s": [-1., .25, 1.]},
    {"time_s": [.25, .125, 1.]}, {"time_s": [.125, .125, 1.]},
    {"time_s": [.125, .25, np.nan]}, {"time_s": [True, False, True]},
    {"regular_current_A_m2": [1., 2.]}, {"regular_current_A_m2": 1.},
    {"regular_current_A_m2": [1., 2., np.inf]},
    {"regular_current_A_m2": [1j, 2j, 3j]},
    {"frequency_Hz": [2., 1.]}, {"frequency_Hz": [-1., 1.]},
    {"frequency_Hz": [1., 1.]}, {"frequency_Hz": []},
    {"frequency_Hz": [True]}, {"frequency_Hz": [np.nan]},
    {"step_amplitude_V": 0.}, {"step_amplitude_V": True},
    {"step_amplitude_V": "0.125"}, {"step_amplitude_V": np.inf},
    {"baseline_current_A_m2": np.nan}, {"dc_conductance_S_m2": 1j},
    {"impulse_charge_C_m2": np.inf},
    {"quadrature_absolute_tolerance_F_m2": 0.},
    {"quadrature_relative_tolerance": 1.},
    {"quadrature_relative_tolerance": -1.},
])
def test_malformed_inputs_fail_before_success(changes):
    with pytest.raises(ValueError):
        constant_case(**changes)


@pytest.mark.parametrize("field", list(R1AdmittanceErrors.__dataclass_fields__))
@pytest.mark.parametrize("bad", [-1., np.inf, np.nan, True, "0"])
def test_invalid_error_bounds_are_rejected(field, bad):
    with pytest.raises(ValueError):
        constant_case(errors=replace(EXACT_INPUTS, **{field: bad}))


@pytest.mark.parametrize("bad", [[1., 2.], [[1., 2., 3.]], [0., -1., 0.]])
def test_sample_error_requires_nonnegative_matching_shape(bad):
    with pytest.raises(ValueError):
        constant_case(errors=replace(EXACT_INPUTS, current_A_m2=bad))


@pytest.mark.parametrize("changes", [
    {"step_amplitude_V": 1e-320}, {"frequency_Hz": [1e308]},
    {"regular_current_A_m2": [1e308]*3, "baseline_current_A_m2": -1e308},
])
def test_nonfinite_derived_arithmetic_is_rejected(changes):
    with pytest.raises(ValueError, match="arithmetic"):
        constant_case(**changes)


@pytest.mark.parametrize("quad_result", [
    (0., 1., {}, "convergence failed"), (np.nan, 0., {}), (0., np.inf, {}), (0., -1., {}),
])
def test_quadrature_failure_cannot_become_success(monkeypatch, quad_result):
    monkeypatch.setattr(module, "quad", lambda *args, **kwargs: quad_result)
    with pytest.raises(R1AdmittanceIntegrationError):
        constant_case()


@pytest.mark.parametrize("curvature", [-1., np.inf, [1., 2.], True])
def test_interpolation_helper_requires_independent_finite_bound(curvature):
    with pytest.raises(ValueError):
        linear_interpolation_integral_bound([1., 2.], curvature)
