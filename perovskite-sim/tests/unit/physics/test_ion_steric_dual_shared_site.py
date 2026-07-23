"""F05 closed-loop: symmetric negative-species diffusion-only steric +
shared-site dual-ion coupling + occupancy sweep.

Covers the gap the review/assessment flagged: the diffusion-only steric form
must be applied symmetrically to BOTH mobile species, and a shared-site dual
ion must crowd against the TOTAL occupancy (P_+ + P_-)/P_lim. Tests:
negative-path default bit-identity and equilibrium; shared-site reduces to
single-species when one density vanishes; total-ion conservation; and an
occupancy sweep theta = 0.01 .. 0.8 for dilute invariance and near-saturation
stability (finite RHS, no NaN/Inf).
"""
from __future__ import annotations

import numpy as np

from perovskite_sim.physics.ion_migration import (
    ion_continuity_rhs,
    ion_continuity_rhs_neg,
)


def _grid(N=61, L=400e-9):
    return np.linspace(0.0, L, N)


# ---------------------------------------------------------------------------
# Negative species: symmetry with the positive path
# ---------------------------------------------------------------------------

def test_neg_default_is_legacy_whole_flux():
    x = _grid()
    P = 1e25 * (1.0 + 0.3 * np.sin(np.linspace(0, 3, len(x))))
    phi = 0.1 * np.linspace(0, 1, len(x))
    D_I, V_T, P_lim = 1e-16, 0.02585, 1.6e27
    from perovskite_sim.discretization.fe_operators import bernoulli
    dx = np.diff(x)
    P_avg = 0.5 * (P[:-1] + P[1:])
    s = 1.0 / np.maximum(1.0 - np.clip(P_avg / P_lim, 0.0, 0.999999), 1e-6)
    xi = -(phi[1:] - phi[:-1]) / V_T
    F = (D_I * s) / dx * (bernoulli(xi) * P[:-1] - bernoulli(-xi) * P[1:])
    F_full = np.concatenate([[0.0], F, [0.0]])
    dxc = np.empty(len(x)); dxc[0] = dx[0]; dxc[-1] = dx[-1]
    dxc[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    ref = -(F_full[1:] - F_full[:-1]) / dxc
    got = ion_continuity_rhs_neg(x, phi, P, D_I, V_T, P_lim)
    np.testing.assert_array_equal(got, ref)


def test_neg_physical_form_zero_field_equilibrium():
    x = _grid()
    P = np.full(len(x), 5e25)
    phi = np.zeros(len(x))
    d = ion_continuity_rhs_neg(x, phi, P, 1e-16, 0.02585, 1.6e27,
                               steric_diffusion_only=True, P_lim_node=1.6e27)
    assert np.max(np.abs(d)) < 1e-6 * (5e25 / 1e-9)


def test_neg_lattice_gas_equilibrium_near_zero():
    # For q_neg = -q, equilibrium is theta/(1-theta) = A e^{+phi/V_T}.
    x = _grid(N=161)
    V_T, P_lim = 0.02585, 1.6e27
    phi = 0.05 * np.linspace(0, 1, len(x))
    A = 0.02
    theta = A * np.exp(phi / V_T) / (1.0 + A * np.exp(phi / V_T))
    P_eq = P_lim * theta
    d_eq = ion_continuity_rhs_neg(x, phi, P_eq, 1e-16, V_T, P_lim,
                                  steric_diffusion_only=True, P_lim_node=P_lim)
    d_neq = ion_continuity_rhs_neg(x, -phi, P_eq, 1e-16, V_T, P_lim,
                                   steric_diffusion_only=True, P_lim_node=P_lim)
    assert np.max(np.abs(d_eq)) < 1e-2 * np.max(np.abs(d_neq))


# ---------------------------------------------------------------------------
# Shared-site dual-ion coupling
# ---------------------------------------------------------------------------

def test_shared_site_reduces_to_single_species_when_other_absent():
    # P_other = 0 -> shared-site total occupancy == own occupancy.
    x = _grid()
    P = 3e26 * (1.0 + 0.2 * np.cos(np.linspace(0, 4, len(x))))
    phi = 0.2 * np.linspace(0, 1, len(x))
    D_I, V_T, P_lim = 1e-16, 0.02585, 1.6e27
    d_solo = ion_continuity_rhs(x, phi, P, D_I, V_T, P_lim,
                                steric_diffusion_only=True, P_lim_node=P_lim)
    d_shared0 = ion_continuity_rhs(x, phi, P, D_I, V_T, P_lim,
                                   steric_diffusion_only=True, P_lim_node=P_lim,
                                   P_other_node=np.zeros_like(P))
    np.testing.assert_allclose(d_shared0, d_solo, rtol=0.0, atol=0.0)


def test_shared_vs_independent_differ_at_high_occupancy():
    # Push the TOTAL occupancy near saturation with a substantial partner
    # density: shared-site crowding (total theta ~ 0.8-0.94) then diverges
    # sharply from independent (own theta ~ 0.31-0.44), so the fluxes must
    # differ unambiguously.
    x = _grid()
    P = np.linspace(5e26, 7e26, len(x))   # strong own gradient
    P_other = np.full(len(x), 8e26)       # pushes total occupancy high
    phi = 0.15 * np.linspace(0, 1, len(x))
    D_I, V_T, P_lim = 1e-16, 0.02585, 1.6e27
    d_indep = ion_continuity_rhs(x, phi, P, D_I, V_T, P_lim,
                                 steric_diffusion_only=True, P_lim_node=P_lim)
    d_shared = ion_continuity_rhs(x, phi, P, D_I, V_T, P_lim,
                                  steric_diffusion_only=True, P_lim_node=P_lim,
                                  P_other_node=P_other)
    assert np.all(np.isfinite(d_shared))
    assert np.max(np.abs(d_shared - d_indep)) > 0.1 * np.max(np.abs(d_indep))


def test_total_ion_conservation_both_species():
    # Zero-flux BCs + divergence form => sum of dP over the cell volumes is
    # zero for each species (no ions created/destroyed) at any occupancy.
    x = _grid()
    dx = np.diff(x)
    dxc = np.empty(len(x)); dxc[0] = dx[0]; dxc[-1] = dx[-1]
    dxc[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    P = np.full(len(x), 5e26) * (1.0 + 0.2 * np.sin(np.linspace(0, 4, len(x))))
    Pn = np.full(len(x), 4e26) * (1.0 + 0.2 * np.cos(np.linspace(0, 3, len(x))))
    phi = 0.2 * np.linspace(0, 1, len(x))
    V_T, P_lim = 0.02585, 1.6e27
    dP = ion_continuity_rhs(x, phi, P, 1e-16, V_T, P_lim,
                            steric_diffusion_only=True, P_lim_node=P_lim,
                            P_other_node=Pn)
    dPn = ion_continuity_rhs_neg(x, phi, Pn, 1e-16, V_T, P_lim,
                                 steric_diffusion_only=True, P_lim_node=P_lim,
                                 P_other_node=P)
    assert abs(np.sum(dP * dxc)) < 1e-6 * np.sum(np.abs(dP) * dxc)
    assert abs(np.sum(dPn * dxc)) < 1e-6 * np.sum(np.abs(dPn) * dxc)


def test_occupancy_sweep_stable_and_dilute_invariant():
    x = _grid()
    phi = 0.2 * np.linspace(0, 1, len(x))
    D_I, V_T, P_lim = 1e-16, 0.02585, 1.6e27
    for theta in (0.01, 0.1, 0.5, 0.8):
        P = np.full(len(x), theta * P_lim) * (
            1.0 + 0.05 * np.sin(np.linspace(0, 4, len(x))))
        Pn = np.zeros_like(P)  # single active species; theta is its own
        d_legacy = ion_continuity_rhs(x, phi, P, D_I, V_T, P_lim)
        d_phys = ion_continuity_rhs(x, phi, P, D_I, V_T, P_lim,
                                    steric_diffusion_only=True, P_lim_node=P_lim,
                                    P_other_node=Pn)
        assert np.all(np.isfinite(d_phys)), f"non-finite RHS at theta={theta}"
        if theta <= 0.01:
            # dilute: physical and legacy forms must nearly coincide.
            assert np.max(np.abs(d_phys - d_legacy)) < 0.03 * np.max(np.abs(d_legacy))
