"""Unit tests for field-dependent mobility (Caughey-Thomas + Poole-Frenkel)."""
from __future__ import annotations

import numpy as np
import pytest

from perovskite_sim.physics.field_mobility import (
    apply_field_mobility,
    caughey_thomas,
    poole_frenkel,
)


# ─── Caughey-Thomas ────────────────────────────────────────────────────


def test_ct_low_field_recovers_mu0():
    """At |E| → 0 the CT formula must return μ₀ exactly."""
    mu0 = np.array([1e-4, 2e-4, 5e-5])
    E = np.zeros_like(mu0)
    v_sat = np.full_like(mu0, 1e5)
    beta = np.full_like(mu0, 2.0)
    mu = caughey_thomas(mu0, E, v_sat, beta)
    np.testing.assert_allclose(mu, mu0, rtol=1e-15)


def test_ct_high_field_drift_saturates_at_v_sat():
    """At |E| >> v_sat / μ₀ the drift velocity μ(E)·|E| → v_sat."""
    mu0 = 2e-4       # typical MAPbI3 electron mobility
    v_sat = 1e5      # 1e5 m/s, silicon-like
    beta = 2.0
    # Saturation threshold is v_sat / μ₀ = 5e8 V/m; pick E two decades
    # above that so the (μ₀E/v_sat)² term dominates the CT denominator.
    E = 1e11
    mu = float(caughey_thomas(mu0, E, v_sat, beta))
    v_drift = mu * E
    # v_drift should match v_sat to within a few percent
    assert abs(v_drift - v_sat) / v_sat < 0.02


def test_ct_zero_v_sat_disables_model():
    """v_sat = 0 must leave μ unchanged (CT disabled at that node)."""
    mu0 = np.array([1e-4, 1e-4, 1e-4])
    E = np.array([0.0, 1e6, 1e8])
    v_sat = np.zeros_like(mu0)
    beta = np.full_like(mu0, 2.0)
    mu = caughey_thomas(mu0, E, v_sat, beta)
    np.testing.assert_allclose(mu, mu0, rtol=1e-15)


def test_ct_zero_beta_disables_model():
    """β ≤ 0 must leave μ unchanged."""
    mu0 = np.array([1e-4, 1e-4])
    E = np.array([1e6, 1e8])
    v_sat = np.full_like(mu0, 1e5)
    beta = np.zeros_like(mu0)
    mu = caughey_thomas(mu0, E, v_sat, beta)
    np.testing.assert_allclose(mu, mu0, rtol=1e-15)


def test_ct_symmetric_in_E_sign():
    """CT depends on |E| only — sign flip must not change μ."""
    mu0 = 1e-4
    v_sat = 1e5
    beta = 2.0
    assert caughey_thomas(mu0, +1e7, v_sat, beta) == pytest.approx(
        caughey_thomas(mu0, -1e7, v_sat, beta), rel=1e-15,
    )


def test_ct_closed_form_beta1_matches_thornber():
    """β = 1 (Thornber form): μ = μ₀ / (1 + μ₀|E|/v_sat)."""
    mu0 = 1e-4
    v_sat = 1e5
    E = 1e7
    expected = mu0 / (1.0 + mu0 * E / v_sat)
    got = float(caughey_thomas(mu0, E, v_sat, 1.0))
    assert got == pytest.approx(expected, rel=1e-12)


def test_ct_closed_form_beta2_matches_canali():
    """β = 2 (Canali form for Si electrons): μ = μ₀ / √(1 + (μ₀|E|/v_sat)²)."""
    mu0 = 1e-4
    v_sat = 1e5
    E = 1e7
    ratio = mu0 * E / v_sat
    expected = mu0 / np.sqrt(1.0 + ratio ** 2)
    got = float(caughey_thomas(mu0, E, v_sat, 2.0))
    assert got == pytest.approx(expected, rel=1e-12)


def test_ct_monotone_in_field():
    """Mobility must decrease monotonically with |E|."""
    mu0 = 1e-4
    v_sat = 1e5
    beta = 2.0
    E = np.array([0.0, 1e5, 1e6, 1e7, 1e8])
    mu = caughey_thomas(np.full_like(E, mu0), E, np.full_like(E, v_sat), np.full_like(E, beta))
    assert np.all(np.diff(mu) <= 0.0)


# ─── Poole-Frenkel ─────────────────────────────────────────────────────


def test_pf_low_field_recovers_mu0():
    """γ_PF · √|E| = 0 at E = 0 → μ = μ₀."""
    mu0 = np.array([1e-7, 5e-8])
    E = np.zeros_like(mu0)
    gamma = np.full_like(mu0, 3e-4)
    mu = poole_frenkel(mu0, E, gamma)
    np.testing.assert_allclose(mu, mu0, rtol=1e-15)


def test_pf_zero_gamma_disables_model():
    """γ_PF = 0 leaves μ untouched at every field."""
    mu0 = np.array([1e-7, 1e-7, 1e-7])
    E = np.array([0.0, 1e6, 1e8])
    gamma = np.zeros_like(mu0)
    mu = poole_frenkel(mu0, E, gamma)
    np.testing.assert_allclose(mu, mu0, rtol=1e-15)


def test_pf_closed_form_matches_exp():
    """μ_PF = μ₀ · exp(γ · √|E|)."""
    mu0 = 1e-7
    gamma = 3e-4
    E = 1e6
    expected = mu0 * np.exp(gamma * np.sqrt(E))
    got = float(poole_frenkel(mu0, E, gamma))
    assert got == pytest.approx(expected, rel=1e-12)


def test_pf_symmetric_in_E_sign():
    """PF uses |E| so the sign of E is irrelevant."""
    mu0 = 1e-7
    gamma = 3e-4
    assert poole_frenkel(mu0, +1e7, gamma) == pytest.approx(
        poole_frenkel(mu0, -1e7, gamma), rel=1e-15,
    )


def test_pf_monotone_increasing_in_field():
    """For γ_PF > 0 μ_PF(E) must increase with |E|."""
    mu0 = 1e-7
    gamma = 3e-4
    E = np.array([0.0, 1e5, 1e6, 1e7, 1e8])
    mu = poole_frenkel(np.full_like(E, mu0), E, np.full_like(E, gamma))
    assert np.all(np.diff(mu) >= 0.0)


def test_pf_handles_huge_field_without_overflow():
    """Very large E should not overflow float64 (arg is clipped)."""
    mu0 = 1e-7
    gamma = 1e-2      # absurdly large
    E = 1e20
    mu = float(poole_frenkel(mu0, E, gamma))
    assert np.isfinite(mu)


# ─── Composition: apply_field_mobility ─────────────────────────────────


def test_composition_zero_both_returns_mu0():
    """v_sat = γ_PF = 0 → μ = μ₀ at any field."""
    mu0 = np.array([1e-4, 1e-4, 1e-4])
    E = np.array([0.0, 1e6, 1e8])
    mu = apply_field_mobility(
        mu0, E, v_sat=np.zeros_like(mu0), beta=np.full_like(mu0, 2.0),
        gamma_pf=np.zeros_like(mu0),
    )
    np.testing.assert_allclose(mu, mu0, rtol=1e-15)


def test_composition_only_pf_equals_pf_alone():
    """When v_sat = 0 the composition must match poole_frenkel alone."""
    mu0 = 1e-7
    E = 1e6
    gamma = 3e-4
    expected = float(poole_frenkel(mu0, E, gamma))
    got = float(apply_field_mobility(mu0, E, v_sat=0.0, beta=2.0, gamma_pf=gamma))
    assert got == pytest.approx(expected, rel=1e-12)


def test_composition_only_ct_equals_ct_alone():
    """When γ_PF = 0 the composition must match caughey_thomas alone."""
    mu0 = 1e-4
    E = 1e7
    v_sat = 1e5
    expected = float(caughey_thomas(mu0, E, v_sat, 2.0))
    got = float(apply_field_mobility(mu0, E, v_sat=v_sat, beta=2.0, gamma_pf=0.0))
    assert got == pytest.approx(expected, rel=1e-12)


def test_composition_pf_then_ct_ordering():
    """Composition applies PF first, then CT on the PF-enhanced mu.

    The high-field asymptote of the composed model is still v_sat / |E|
    because CT saturates the drift velocity regardless of what PF did to
    the low-field mobility.
    """
    mu0 = 1e-4
    gamma = 1e-5      # modest PF so PF·exp stays finite at our E
    v_sat = 1e5
    # At this E the PF-enhanced μ₀' is O(μ₀·e), and μ₀'·E is ~O(1e6) ≫ v_sat,
    # so CT caps v_drift at v_sat regardless of PF.
    E = 1e10
    mu = float(apply_field_mobility(mu0, E, v_sat=v_sat, beta=2.0, gamma_pf=gamma))
    v_drift = mu * E
    assert abs(v_drift - v_sat) / v_sat < 0.05
