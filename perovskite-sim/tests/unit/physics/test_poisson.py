import numpy as np
import pytest
from perovskite_sim.physics.poisson import solve_poisson

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
