"""Independent exact-input oracles for cancellation in fixed trap captures."""

from dataclasses import replace
from decimal import Decimal, localcontext
import math

import numpy as np
import pytest

from perovskite_sim.physics.two_sided_interface import (
    TwoSidedInterfacePhysics,
    fixed_occupancy_trap_capture_flux_and_log_jacobian,
)


# Exact binary64 inputs retained in V33/Root/PreparationCaptureInputsV1.json.
# This test needs no archived data, material reconstruction or DC preparation.
_N16_STATE = np.array([
    462399226669.298, 13680185118384.672,
    22128050854779.137, 285868243025.3834,
])
_N16_OCCUPANCY = 8.02741239777023e-05


def _physics(**updates):
    return replace(TwoSidedInterfacePhysics(
        thermal_voltage_V=0.025851999786435535,
        temperature_K=300.0,
        D_n_left_m2_s=2.5851999786435536e-05,
        D_n_right_m2_s=2.5851999786435536e-05,
        D_p_left_m2_s=2.5851999786435536e-05,
        D_p_right_m2_s=2.5851999786435536e-05,
        N_C_left_m3=1e25, N_C_right_m3=1e25,
        N_V_left_m3=1e25, N_V_right_m3=1e25,
        richardson_n_A_m2_K2=650768.3018550435,
        richardson_p_A_m2_K2=650768.3018550435,
        conduction_band_step_eV=-0.09999999999999964,
        hole_transport_step_eV=0.09999999999999964,
        surface_recombination_velocity_n_m_s=0.03,
        surface_recombination_velocity_p_m_s=0.049999999999999996,
        n1_left_m3=5759790142394265.0,
        n1_right_m3=2.756339583478704e17,
        p1_left_m3=1098253037.5320292,
        p1_right_m3=22949665.046162695,
    ), **updates)


def _decimal_capture(state, physics, occupancy, multiplier):
    """Evaluate the original formula in Decimal from exact binary64 inputs.

    The public function's existing rounded velocity coefficient is retained;
    the oracle changes neither that boundary nor the physical capture law.
    The caller supplies a high-precision decimal context.
    """
    d = Decimal.from_float
    f = d(float(occupancy))
    vn = d(float(multiplier) * float(physics.surface_recombination_velocity_n_m_s))
    vp = d(float(multiplier) * float(physics.surface_recombination_velocity_p_m_s))
    n_left, p_left, n_right, p_right = state
    return [
        vn * (n_left * (1-f) - d(physics.n1_left_m3) * f),
        vp * (p_left * f - d(physics.p1_left_m3) * (1-f)),
        vn * (n_right * (1-f) - d(physics.n1_right_m3) * f),
        vp * (p_right * f - d(physics.p1_right_m3) * (1-f)),
    ]


def _oracle(state, physics, occupancy, multiplier=1.0):
    # Sufficient for exact binary64 values across the normal/subnormal range,
    # their products, and the deliberately near-cancelling differences.
    with localcontext() as context:
        context.prec = 2000
        d = Decimal.from_float
        values = [d(float(value)) for value in state]
        capture = _decimal_capture(values, physics, occupancy, multiplier)
        f = d(float(occupancy))
        vn = d(float(multiplier) * float(physics.surface_recombination_velocity_n_m_s))
        vp = d(float(multiplier) * float(physics.surface_recombination_velocity_p_m_s))
        diagonal = [vn*(1-f)*values[0], vp*f*values[1],
                    vn*(1-f)*values[2], vp*f*values[3]]
    return np.asarray([float(value) for value in capture]), np.diag([
        float(value) for value in diagonal])


def _assert_ulp_close(actual, expected, ulps=4):
    for actual_value, expected_value in zip(
            np.asarray(actual).flat, np.asarray(expected).flat):
        assert math.isfinite(float(actual_value))
        if expected_value == 0.0:
            assert actual_value == 0.0
        else:
            assert abs(actual_value-expected_value) <= ulps*math.ulp(float(expected_value))


@pytest.fixture(params=("native", "split"), autouse=True)
def product_arithmetic(request, monkeypatch):
    # Exercise the same public capture law on both supported arithmetic paths.
    if request.param == "split":
        monkeypatch.delattr(math, "fma", raising=False)


def test_measured_n16_capture_matches_exact_input_decimal():
    physics = _physics()
    capture, tangent = fixed_occupancy_trap_capture_flux_and_log_jacobian(
        _N16_STATE, physics, _N16_OCCUPANCY)
    expected_capture, expected_tangent = _oracle(_N16_STATE, physics, _N16_OCCUPANCY)
    _assert_ulp_close(capture, expected_capture)
    _assert_ulp_close(tangent, expected_tangent)
    _assert_ulp_close(capture[2], -0.010364032174211803)

    # The original expression is an explicit negative control for the failure.
    old_right_electron = physics.surface_recombination_velocity_n_m_s * (
        _N16_STATE[2]*(1.0-_N16_OCCUPANCY)
        - physics.n1_right_m3*_N16_OCCUPANCY)
    assert old_right_electron == -0.0104296875
    assert abs(old_right_electron-expected_capture[2]) > 1e-6
    assert np.max(np.abs(capture-expected_capture)) <= 1e-6


def test_both_signs_survive_a_complement_rounded_to_one():
    occupancy = math.ldexp(1.0, -55)
    assert 1.0-occupancy == 1.0
    large = math.ldexp(1.0, 55)
    state = np.array([1.0, large, math.nextafter(1.0, math.inf),
                      math.nextafter(large, 0.0)])
    physics = _physics(
        surface_recombination_velocity_n_m_s=1.0,
        surface_recombination_velocity_p_m_s=1.0,
        n1_left_m3=large, n1_right_m3=large,
        p1_left_m3=1.0, p1_right_m3=1.0)
    capture, tangent = fixed_occupancy_trap_capture_flux_and_log_jacobian(
        state, physics, occupancy)
    expected_capture, expected_tangent = _oracle(state, physics, occupancy)
    _assert_ulp_close(capture, expected_capture)
    _assert_ulp_close(tangent, expected_tangent)
    np.testing.assert_array_equal(np.sign(capture), [-1, 1, 1, -1])


@pytest.mark.parametrize("occupancy", [0.0, 0.37, 1.0])
@pytest.mark.parametrize("multiplier", [0.0, 0.37, 1.0])
def test_normal_captures_endpoints_and_disabled_channel(occupancy, multiplier):
    physics = _physics(n1_left_m3=2e17, n1_right_m3=5e17,
                       p1_left_m3=8e16, p1_right_m3=3e17)
    state = np.array([3e20, 7e19, 5e19, 4e20])
    capture, tangent = fixed_occupancy_trap_capture_flux_and_log_jacobian(
        state, physics, occupancy, capture_multiplier=multiplier)
    expected_capture, expected_tangent = _oracle(state, physics, occupancy, multiplier)
    _assert_ulp_close(capture, expected_capture)
    _assert_ulp_close(tangent, expected_tangent)
    if multiplier == 0.0:
        np.testing.assert_array_equal(capture, np.zeros(4))
        np.testing.assert_array_equal(tangent, np.zeros((4, 4)))


@pytest.mark.parametrize("density", [
    math.ldexp(1.0, 1000), math.ldexp(1.0, -1000),
    float(np.finfo(float).max), math.ulp(0.0),
])
@pytest.mark.parametrize("occupancy", [0.0, 0.5, 1.0])
def test_finite_density_domain_includes_values_outside_dd_range(density, occupancy):
    physics = _physics(
        surface_recombination_velocity_n_m_s=1.0,
        surface_recombination_velocity_p_m_s=1.0,
        n1_left_m3=density, n1_right_m3=density,
        p1_left_m3=density, p1_right_m3=density)
    state = np.full(4, density)
    capture, tangent = fixed_occupancy_trap_capture_flux_and_log_jacobian(
        state, physics, occupancy)
    expected_capture, expected_tangent = _oracle(state, physics, occupancy)
    _assert_ulp_close(capture, expected_capture)
    _assert_ulp_close(tangent, expected_tangent)


def test_disabled_capture_at_largest_finite_density():
    largest = float(np.finfo(float).max)
    physics = _physics(
        surface_recombination_velocity_n_m_s=1.0,
        surface_recombination_velocity_p_m_s=1.0,
        n1_left_m3=largest, n1_right_m3=largest,
        p1_left_m3=largest, p1_right_m3=largest)
    capture, tangent = fixed_occupancy_trap_capture_flux_and_log_jacobian(
        np.full(4, largest), physics, 1.0, capture_multiplier=0.0)
    np.testing.assert_array_equal(capture, np.zeros(4))
    np.testing.assert_array_equal(tangent, np.zeros((4, 4)))


def test_subnormal_occupancy_times_large_emission_is_not_lost():
    occupancy = math.ulp(0.0)
    largest = float(np.finfo(float).max)
    matched = occupancy*largest
    lower, upper = math.nextafter(matched, 0.0), math.nextafter(matched, math.inf)
    physics = _physics(
        surface_recombination_velocity_n_m_s=1.0,
        surface_recombination_velocity_p_m_s=1.0,
        n1_left_m3=largest, n1_right_m3=largest,
        p1_left_m3=lower, p1_right_m3=upper)
    state = np.array([upper, largest, lower, largest])
    capture, tangent = fixed_occupancy_trap_capture_flux_and_log_jacobian(
        state, physics, occupancy)
    expected_capture, expected_tangent = _oracle(state, physics, occupancy)
    _assert_ulp_close(capture, expected_capture)
    _assert_ulp_close(tangent, expected_tangent)
    np.testing.assert_array_equal(np.sign(capture), [1, 1, -1, -1])


def test_log_density_tangent_matches_decimal_central_differences():
    physics = _physics()
    _, tangent = fixed_occupancy_trap_capture_flux_and_log_jacobian(
        _N16_STATE, physics, _N16_OCCUPANCY, capture_multiplier=0.37)
    with localcontext() as context:
        context.prec = 150
        values = [Decimal.from_float(float(value)) for value in _N16_STATE]
        step = Decimal("1e-25")
        numerical = np.zeros((4, 4))
        for column in range(4):
            plus, minus = list(values), list(values)
            plus[column] *= step.exp()
            minus[column] *= (-step).exp()
            upper = _decimal_capture(plus, physics, _N16_OCCUPANCY, 0.37)
            lower = _decimal_capture(minus, physics, _N16_OCCUPANCY, 0.37)
            numerical[:, column] = [float((a-b)/(2*step)) for a, b in zip(upper, lower)]
    _assert_ulp_close(tangent, numerical)
