"""Unit tests for the Padovani-Stratton TFE enhancement factor."""
import math

import pytest

from perovskite_sim.physics.tunneling import tfe_gamma
from perovskite_sim.discretization.fe_operators import thermionic_emission_flux

V_T = 0.025852  # ~kT/q at 300 K


def test_gamma_unity_for_intrinsic_side():
    """Zero (intrinsic) interface doping → no field emission → Gamma = 1."""
    assert tfe_gamma(0.3, 0.0, 0.2, 10.0, V_T) == 1.0


def test_gamma_unity_for_zero_offset():
    assert tfe_gamma(0.0, 1e24, 0.2, 10.0, V_T) == 1.0


def test_gamma_unity_for_nonpositive_mass_or_eps():
    assert tfe_gamma(0.3, 1e24, 0.0, 10.0, V_T) == 1.0
    assert tfe_gamma(0.3, 1e24, 0.2, 0.0, V_T) == 1.0


def test_gamma_above_one_for_doped_offset():
    g = tfe_gamma(0.3, 1e24, 0.2, 10.0, V_T)
    assert g > 1.0


def test_gamma_bounded_below_full_tunnelling():
    """Gamma must stay below exp(|delta_E|/V_T) (clamp delta_tun < |delta_E|)."""
    dE = 0.3
    g = tfe_gamma(dE, 1e27, 0.05, 10.0, V_T)  # very strong field-emission
    assert g < math.exp(dE / V_T)


def test_gamma_monotone_increasing_in_doping():
    g_lo = tfe_gamma(0.3, 1e22, 0.2, 10.0, V_T)
    g_mid = tfe_gamma(0.3, 1e24, 0.2, 10.0, V_T)
    g_hi = tfe_gamma(0.3, 1e26, 0.2, 10.0, V_T)
    assert g_lo <= g_mid <= g_hi


def test_detailed_balance_invariant_under_gamma():
    """The mandatory T1 contract at the flux level: a SYMMETRIC Gamma on A*
    leaves equilibrium J_TE = 0 to machine precision (both legs scale equally).
    """
    dE = 0.3              # eV CB offset (step up left->right)
    A_star = 1.2017e6
    T = 300.0
    # Equilibrium Boltzmann ratio: n_left / n_right = exp(dE / V_T) makes the
    # two-leg bracket vanish.
    n_right = 1e22
    n_left = n_right * math.exp(dE / V_T)
    J_plain = thermionic_emission_flux(n_left, n_right, dE, T, A_star)
    g = tfe_gamma(dE, 1e24, 0.2, 10.0, V_T)
    J_enh = thermionic_emission_flux(n_left, n_right, dE, T, g * A_star)
    # Each current is normalised by its OWN thermionic leg magnitude
    # (A*·T²·n_left). J_enh = Gamma·J_plain exactly, so the RELATIVE equilibrium
    # current is Gamma-invariant and machine-zero — detailed balance preserved.
    scale = A_star * T * T * n_left
    assert abs(J_plain) / scale < 1e-12
    assert abs(J_enh) / (g * scale) < 1e-12
