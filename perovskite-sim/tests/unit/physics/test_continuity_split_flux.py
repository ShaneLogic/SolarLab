"""Phase E4 Sprint 8 Day 2-3 — split-interface-flux RED tests.

Pins the contract for the new ``split_interface_flux`` helper in
``physics/continuity.py``:

1. A pure helper computes per-layer half-fluxes at a heterointerface
   face given bulk densities + iface-plane densities + per-layer
   chi/Eg/D.
2. At the no-chi-step limit (chi_L == chi_R, Eg_L == Eg_R), the SUM of
   the two half-fluxes equals the legacy single SG flux.
3. The L-half-flux uses ONLY the L-side chi/Eg; R-half uses ONLY the
   R-side. No cross-layer harmonic averaging.
4. Sign convention matches the legacy SG flux (positive flux from idx
   toward idx+1).
5. When the interface-plane density equals the bulk density (no
   coupling), the half-flux degenerates to a vanishing flux — no
   driving force.
"""
from __future__ import annotations

import math

import numpy as np
import pytest


_V_T = 0.0259  # ~ kT/q at 300 K


def test_split_helper_module_exists():
    """`split_interface_flux` is importable from physics.continuity."""
    from perovskite_sim.physics.continuity import split_interface_flux
    assert callable(split_interface_flux)


def test_split_signature_returns_pair_floats():
    """Returns a pair (J_L_half, J_R_half) of floats."""
    from perovskite_sim.physics.continuity import split_interface_flux
    result = split_interface_flux(
        n_idx=1.0e22, n_idx_plus_1=2.0e22,
        n_L_iface=1.5e22, n_R_iface=1.5e22,
        phi_idx=0.0, phi_idx_plus_1=0.1, phi_iface=0.05,
        chi_L=0.0, chi_R=0.0,
        D_L=1.0e-4, D_R=1.0e-4,
        dx_face=1.0e-8, V_T=_V_T,
    )
    J_L, J_R = result
    assert isinstance(J_L, float)
    assert isinstance(J_R, float)


def test_split_half_flux_formula_pinned():
    """L-half flux matches hand-computed Bernoulli SG on half-face.

    For chi_L=0, dx_half = dx_face/2:
      xi_L = (phi_iface - phi_idx) / V_T
      J_L = q · D / dx_half · (B(xi_L) · n_L_iface - B(-xi_L) · n_idx)

    Pin the formula by reproducing it from a known input.
    """
    from perovskite_sim.physics.continuity import split_interface_flux
    from perovskite_sim.discretization.fe_operators import bernoulli
    n_idx, n_idx_plus_1 = 1.0e22, 2.0e22
    n_L_iface, n_R_iface = 1.3e22, 1.7e22
    phi_idx, phi_idx_plus_1 = 0.0, 0.1
    phi_iface = 0.05
    D = 1.0e-4
    dx_face = 1.0e-8
    Q = 1.602176634e-19
    J_L, J_R = split_interface_flux(
        n_idx=n_idx, n_idx_plus_1=n_idx_plus_1,
        n_L_iface=n_L_iface, n_R_iface=n_R_iface,
        phi_idx=phi_idx, phi_idx_plus_1=phi_idx_plus_1, phi_iface=phi_iface,
        chi_L=0.0, chi_R=0.0,
        D_L=D, D_R=D,
        dx_face=dx_face, V_T=_V_T,
    )
    # Hand-computed expected L-half (chi_L=0; dx_half = dx_face/2).
    dx_half = dx_face / 2.0
    xi_L = (phi_iface - phi_idx) / _V_T
    expected_J_L = Q * D / dx_half * (
        float(bernoulli(np.array([xi_L]))[0]) * n_L_iface
        - float(bernoulli(np.array([-xi_L]))[0]) * n_idx
    )
    assert J_L == pytest.approx(expected_J_L, rel=1e-9)
    # Hand-computed expected R-half.
    xi_R = (phi_idx_plus_1 - phi_iface) / _V_T
    expected_J_R = Q * D / dx_half * (
        float(bernoulli(np.array([xi_R]))[0]) * n_idx_plus_1
        - float(bernoulli(np.array([-xi_R]))[0]) * n_R_iface
    )
    assert J_R == pytest.approx(expected_J_R, rel=1e-9)


def test_split_l_half_uses_only_l_layer_chi():
    """Changing chi_R does NOT change J_L_half (locality of half-fluxes).

    Distinguishes the split path from the legacy chi/Eg-mixed SG flux
    where chi at idx+1 enters the L-side flux computation.
    """
    from perovskite_sim.physics.continuity import split_interface_flux
    common_kwargs = dict(
        n_idx=1.0e22, n_idx_plus_1=2.0e22,
        n_L_iface=1.5e22, n_R_iface=1.5e22,
        phi_idx=0.0, phi_idx_plus_1=0.1, phi_iface=0.05,
        D_L=1.0e-4, D_R=1.0e-4,
        dx_face=1.0e-8, V_T=_V_T,
    )
    J_L_a, _ = split_interface_flux(chi_L=0.0, chi_R=0.0, **common_kwargs)
    J_L_b, _ = split_interface_flux(chi_L=0.0, chi_R=0.2, **common_kwargs)
    # J_L_half must be invariant to chi_R (locality).
    assert J_L_a == pytest.approx(J_L_b, rel=1e-9)


def test_split_r_half_uses_only_r_layer_chi():
    """Changing chi_L does NOT change J_R_half."""
    from perovskite_sim.physics.continuity import split_interface_flux
    common_kwargs = dict(
        n_idx=1.0e22, n_idx_plus_1=2.0e22,
        n_L_iface=1.5e22, n_R_iface=1.5e22,
        phi_idx=0.0, phi_idx_plus_1=0.1, phi_iface=0.05,
        D_L=1.0e-4, D_R=1.0e-4,
        dx_face=1.0e-8, V_T=_V_T,
    )
    _, J_R_a = split_interface_flux(chi_L=0.0, chi_R=0.0, **common_kwargs)
    _, J_R_b = split_interface_flux(chi_L=0.2, chi_R=0.0, **common_kwargs)
    assert J_R_a == pytest.approx(J_R_b, rel=1e-9)


def test_split_zero_when_iface_matches_bulk():
    """When n_L_iface = n_idx (no density gradient L-side), J_L_half = 0."""
    from perovskite_sim.physics.continuity import split_interface_flux
    n = 1.0e22
    J_L, _ = split_interface_flux(
        n_idx=n, n_idx_plus_1=2.0e22,
        n_L_iface=n,  # exactly matches L bulk
        n_R_iface=1.5e22,
        phi_idx=0.0, phi_idx_plus_1=0.0,  # also no phi gradient
        phi_iface=0.0,
        chi_L=0.0, chi_R=0.0,
        D_L=1.0e-4, D_R=1.0e-4,
        dx_face=1.0e-8, V_T=_V_T,
    )
    assert J_L == pytest.approx(0.0, abs=1e-10)


def test_split_zero_when_iface_matches_r_bulk():
    """When n_R_iface = n_idx_plus_1 (no density gradient R-side), J_R_half = 0."""
    from perovskite_sim.physics.continuity import split_interface_flux
    n = 2.0e22
    _, J_R = split_interface_flux(
        n_idx=1.0e22, n_idx_plus_1=n,
        n_L_iface=1.5e22,
        n_R_iface=n,  # exactly matches R bulk
        phi_idx=0.0, phi_idx_plus_1=0.0,
        phi_iface=0.0,
        chi_L=0.0, chi_R=0.0,
        D_L=1.0e-4, D_R=1.0e-4,
        dx_face=1.0e-8, V_T=_V_T,
    )
    assert J_R == pytest.approx(0.0, abs=1e-10)
