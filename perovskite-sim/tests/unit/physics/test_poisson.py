import numpy as np
import pytest
from perovskite_sim.physics.poisson import (
    factor_poisson,
    factor_poisson_from_finite_volume,
    solve_poisson,
    solve_poisson_prefactored,
)

EPS_0 = 8.854187817e-12
Q = 1.602176634e-19


def test_zero_charge_gives_linear_potential():
    """Zero space charge → linear potential (flat field)."""
    x = np.linspace(0, 400e-9, 51)
    eps_r = 24.1 * np.ones(51)
    rho = np.zeros(51)
    phi = solve_poisson(x, eps_r, rho, phi_left=0.0, phi_right=1.0)
    phi_expected = np.linspace(0, 1, 51)
    np.testing.assert_allclose(phi, phi_expected, atol=1e-8)


def test_positive_charge_creates_concave_potential():
    """Positive uniform charge → concave potential (downward curve)."""
    x = np.linspace(0, 400e-9, 101)
    eps_r = 24.1 * np.ones(101)
    rho_val = Q * 1e22   # uniform positive charge density [C/m³]
    rho = rho_val * np.ones(101)
    phi = solve_poisson(x, eps_r, rho, phi_left=0.0, phi_right=0.0)
    # Maximum should be at centre
    assert np.argmax(phi) == 50


def test_boundary_conditions_enforced():
    x = np.linspace(0, 400e-9, 51)
    eps_r = np.ones(51)
    rho = np.zeros(51)
    phi = solve_poisson(x, eps_r, rho, phi_left=0.3, phi_right=0.7)
    assert phi[0] == pytest.approx(0.3)
    assert phi[-1] == pytest.approx(0.7)


def test_explicit_finite_volume_factor_matches_legacy_factor():
    x = np.array([0.0, 1.0e-9, 3.0e-9, 7.0e-9])
    eps_r = np.array([10.0, 12.0, 20.0, 24.0])
    legacy = factor_poisson(x, eps_r)
    explicit = factor_poisson_from_finite_volume(legacy.C, legacy.h_cell)
    rho = Q * np.array([0.0, 1.0e20, -2.0e20, 0.0])

    np.testing.assert_allclose(
        solve_poisson_prefactored(legacy, rho, 0.0, 0.4),
        solve_poisson_prefactored(explicit, rho, 0.0, 0.4),
        rtol=0.0,
        atol=0.0,
    )


def test_explicit_series_capacitance_resolves_off_midpoint_interface():
    # One charge-free dielectric face split at h_L=1 nm and h_R=3 nm.
    # Its exact series capacitance differs from an equal-half harmonic mean.
    h_left = 1.0e-9
    h_right = 3.0e-9
    eps_left = 10.0
    eps_right = 30.0
    exact_interface_capacitance = EPS_0 / (
        h_left / eps_left + h_right / eps_right
    )
    C = np.array([EPS_0 * eps_left / 1.0e-9, exact_interface_capacitance])
    factor = factor_poisson_from_finite_volume(C, np.array([h_left]))
    phi = solve_poisson_prefactored(
        factor,
        np.zeros(3),
        phi_left=0.0,
        phi_right=1.0,
    )
    expected_middle = C[1] / (C[0] + C[1])

    assert phi[1] == pytest.approx(expected_middle, rel=1.0e-14)
