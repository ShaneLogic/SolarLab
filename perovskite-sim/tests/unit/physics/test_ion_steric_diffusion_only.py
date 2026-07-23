"""Physical diffusion-only steric ion flux (review F05).

The legacy steric factor multiplies the whole SG flux (drift + diffusion);
the physical modified-PNP form corrects only the diffusion term, implemented
here by folding the crowding chemical potential into the SG drift argument.
These tests pin: default bit-identity, equilibrium preservation, and the
dilute-limit agreement (both forms coincide when P << P_lim, which is the
regime of every shipped preset).
"""
from __future__ import annotations

import numpy as np

from perovskite_sim.physics.ion_migration import ion_continuity_rhs


def _grid(N=41, L=400e-9):
    return np.linspace(0.0, L, N)


def test_default_is_legacy_whole_flux():
    x = _grid()
    rng = np.random.default_rng(0)
    P = 1e25 * (1.0 + 0.3 * np.sin(np.linspace(0, 3, len(x))))
    phi = 0.1 * np.linspace(0, 1, len(x))
    D_I, V_T, P_lim = 1e-16, 0.02585, 1.6e27
    # Explicit legacy reference recomputation.
    dx = np.diff(x)
    from perovskite_sim.discretization.fe_operators import bernoulli
    P_avg = 0.5 * (P[:-1] + P[1:])
    s = 1.0 / np.maximum(1.0 - np.clip(P_avg / P_lim, 0.0, 0.999999), 1e-6)
    xi = (phi[1:] - phi[:-1]) / V_T
    F = (D_I * s) / dx * (bernoulli(xi) * P[:-1] - bernoulli(-xi) * P[1:])
    F_full = np.concatenate([[0.0], F, [0.0]])
    dxc = np.empty(len(x)); dxc[0] = dx[0]; dxc[-1] = dx[-1]
    dxc[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    ref = -(F_full[1:] - F_full[:-1]) / dxc
    got = ion_continuity_rhs(x, phi, P, D_I, V_T, P_lim)
    np.testing.assert_allclose(got, ref, rtol=0.0, atol=0.0)


def test_physical_form_preserves_zero_field_equilibrium():
    # Uniform phi + uniform P -> no gradient -> dP/dt = 0 for both forms.
    x = _grid()
    P = np.full(len(x), 5e25)
    phi = np.zeros(len(x))
    d = ion_continuity_rhs(x, phi, P, 1e-16, 0.02585, 1.6e27,
                           steric_diffusion_only=True, P_lim_node=1.6e27)
    assert np.max(np.abs(d)) < 1e-6 * (5e25 / 1e-9)  # ~machine zero on this scale


def test_physical_form_at_lattice_gas_equilibrium_is_near_zero():
    # The lattice-gas (Fermi-Dirac) equilibrium is theta/(1-theta) = A e^{-phi/V_T},
    # theta = P/P_lim. The crowding-in-potential flux must nearly vanish there
    # (up to O(dx^2) from the nodal crowding potential). Compare against a
    # deliberately NON-equilibrium profile on the same grid: the equilibrium
    # divergence must be orders smaller.
    x = _grid(N=161)  # fine grid so the discretization residual is small
    V_T, P_lim = 0.02585, 1.6e27
    phi = 0.05 * np.linspace(0, 1, len(x))
    A = 0.02
    theta = A * np.exp(-phi / V_T) / (1.0 + A * np.exp(-phi / V_T))
    P_eq = P_lim * theta
    d_eq = ion_continuity_rhs(x, phi, P_eq, 1e-16, V_T, P_lim,
                              steric_diffusion_only=True, P_lim_node=P_lim)
    # Non-equilibrium: flip the field sign so the same P is far from balance.
    d_neq = ion_continuity_rhs(x, -phi, P_eq, 1e-16, V_T, P_lim,
                               steric_diffusion_only=True, P_lim_node=P_lim)
    assert np.max(np.abs(d_eq)) < 1e-2 * np.max(np.abs(d_neq))


def test_dilute_limit_forms_agree():
    # P/P_lim ~ 1e-2 (the shipped-preset regime): the two forms must agree
    # to well under 2% (steric factor ~ 1.01).
    x = _grid()
    P = 1.6e25 * (1.0 + 0.2 * np.cos(np.linspace(0, 4, len(x))))  # ~1% of P_lim
    phi = 0.2 * np.linspace(0, 1, len(x))
    D_I, V_T, P_lim = 1e-16, 0.02585, 1.6e27
    d_legacy = ion_continuity_rhs(x, phi, P, D_I, V_T, P_lim)
    d_phys = ion_continuity_rhs(x, phi, P, D_I, V_T, P_lim,
                                steric_diffusion_only=True, P_lim_node=P_lim)
    denom = np.max(np.abs(d_legacy))
    assert np.max(np.abs(d_phys - d_legacy)) < 0.02 * denom
