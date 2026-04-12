import math

import pytest
from perovskite_sim.constants import K_B, Q
from perovskite_sim.discretization.fe_operators import thermionic_emission_flux


T = 300.0
V_T = K_B * T / Q
A_STAR = 1.2e6  # Richardson constant for free electrons [A/(m^2 K^2)]


def test_thermionic_flux_zero_barrier():
    """delta_E=0, equal densities on both sides -> net flux ~ 0."""
    n_left = 1e18
    n_right = 1e18
    J = thermionic_emission_flux(n_left, n_right, delta_E=0.0, T=T, A_star=A_STAR)
    assert abs(J) < 1e-10


def test_thermionic_flux_positive_barrier():
    """delta_E=+0.3 eV step-up barrier limits left->right current via exp(-0.3/V_T)."""
    n_left = 1e22
    n_right = 1e16
    delta_E = 0.3
    J = thermionic_emission_flux(n_left, n_right, delta_E=delta_E, T=T, A_star=A_STAR)
    # Current should be positive (left -> right dominates)
    assert J > 0
    # Magnitude should be limited by exp(-0.3/V_T) ~ 8.5e-6
    boltzmann_factor = math.exp(-delta_E / V_T)
    upper_bound = A_STAR * T**2 * n_left * boltzmann_factor
    assert J < upper_bound


def test_thermionic_flux_negative_barrier():
    """delta_E=-0.3 (step-down) gives much larger |J| than +0.3."""
    n_left = 1e22
    n_right = 1e16
    J_pos = thermionic_emission_flux(n_left, n_right, delta_E=0.3, T=T, A_star=A_STAR)
    J_neg = thermionic_emission_flux(n_left, n_right, delta_E=-0.3, T=T, A_star=A_STAR)
    assert abs(J_neg) > abs(J_pos)


def test_thermionic_flux_units():
    """Result should be finite for typical inputs."""
    J = thermionic_emission_flux(
        n_left=1e18, n_right=1e16, delta_E=0.2, T=300.0, A_star=A_STAR,
    )
    assert math.isfinite(J)
