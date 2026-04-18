"""Unit tests for the selective / Schottky contact Robin BC primitives.

These exercise only the pure-function flux helpers in
``perovskite_sim/physics/contacts.py`` — no solver in the loop. The
goal is to pin down the three things that are easy to break when
refactoring sign conventions:

1. Equilibrium flux is zero (``n = n_eq`` → ``J = 0``).
2. The left/right pair of signs has opposite polarity at each side,
   so a node above equilibrium drives *out* of the device at both
   contacts (negative-divergence contribution to the interior cell).
3. The ohmic limit ``S → ∞`` pulls the boundary density to ``n_eq``
   within a few Radau steps on a realistic grid.

Schottky helpers are tested separately because they only compute
``n_eq``; the BC math is reused.
"""
from __future__ import annotations

import numpy as np
import pytest

from perovskite_sim.physics.contacts import (
    apply_selective_contacts,
    schottky_equilibrium_n,
    schottky_equilibrium_p,
    selective_contact_flux,
)

Q = 1.602176634e-19


# -- selective_contact_flux basics --------------------------------------

def test_flux_zero_at_equilibrium_all_sides_and_carriers():
    """n = n_eq (or p = p_eq) → zero flux, regardless of S / side / carrier."""
    for side in ("left", "right"):
        for carrier in ("n", "p"):
            J = selective_contact_flux(
                1e23, 1e23, 1e4, carrier=carrier, side=side,
            )
            assert J == 0.0, (
                f"Equilibrium flux should vanish for carrier={carrier}, "
                f"side={side}; got {J}"
            )


def test_flux_sign_left_vs_right_is_opposite_for_same_excess():
    """Excess density on the left side and the right side produce flux
    that differs only by a sign flip — both contacts carry charge away
    from an excess, not just one of them."""
    n, n_eq, S = 2e23, 1e23, 1e4
    J_left = selective_contact_flux(n, n_eq, S, carrier="n", side="left")
    J_right = selective_contact_flux(n, n_eq, S, carrier="n", side="right")
    assert J_left == pytest.approx(-J_right), (
        "Left / right selective flux should flip sign for the same excess; "
        f"got J_L={J_left}, J_R={J_right}"
    )


def test_flux_sign_electrons_vs_holes_is_opposite():
    """For an identical excess on the same side, electron and hole
    Robin fluxes have opposite polarity because the carrier continuity
    equations carry opposite signs on ∇·J. Flipping one without the
    other would break the ohmic limit."""
    density, density_eq, S = 2e23, 1e23, 1e4
    J_n = selective_contact_flux(density, density_eq, S, carrier="n", side="left")
    J_p = selective_contact_flux(density, density_eq, S, carrier="p", side="left")
    assert J_n == pytest.approx(-J_p)


def test_flux_scales_linearly_with_S():
    """J ∝ S when excess and n_eq are fixed — a basic linearity check."""
    n, n_eq = 2e23, 1e23
    J1 = selective_contact_flux(n, n_eq, 1e3, carrier="n", side="left")
    J2 = selective_contact_flux(n, n_eq, 1e5, carrier="n", side="left")
    assert J2 == pytest.approx(100.0 * J1)


def test_flux_scales_linearly_with_excess():
    """J ∝ (n - n_eq) when S is fixed — linearity in the Robin argument."""
    n_eq, S = 1e23, 1e4
    J1 = selective_contact_flux(1.1e23, n_eq, S, carrier="n", side="left")
    J2 = selective_contact_flux(1.2e23, n_eq, S, carrier="n", side="left")
    assert J2 == pytest.approx(2.0 * J1)


def test_flux_blocking_contact_S_zero_is_no_flux():
    """S = 0 recovers the Neumann (zero-flux) limit for any excess."""
    J = selective_contact_flux(
        5e23, 1e23, 0.0, carrier="n", side="left",
    )
    assert J == 0.0


def test_selective_contact_flux_rejects_bad_side():
    with pytest.raises(ValueError, match="side must be"):
        selective_contact_flux(
            1e23, 1e23, 1e4, carrier="n", side="top",
        )


def test_selective_contact_flux_rejects_bad_carrier():
    with pytest.raises(ValueError, match="carrier must be"):
        selective_contact_flux(
            1e23, 1e23, 1e4, carrier="q", side="left",
        )


# -- apply_selective_contacts: vector-level pad replacement --------------

def test_apply_selective_contacts_only_touches_boundary_entries():
    """The interior pad entries must survive the call unchanged — this
    helper only rewrites indices 0 and -1."""
    N = 8
    J_n = np.linspace(1.0, 8.0, N)
    J_p = np.linspace(-1.0, -8.0, N)
    n = np.full(N, 1e23)
    p = np.full(N, 1e22)

    J_n_out, J_p_out = apply_selective_contacts(
        J_n, J_p, n, p,
        S_n_L=1e4, S_p_L=1e4, S_n_R=1e4, S_p_R=1e4,
        n_L=1e23, p_L=1e22, n_R=1e23, p_R=1e22,
    )
    # Interior untouched.
    assert np.all(J_n_out[1:-1] == J_n[1:-1])
    assert np.all(J_p_out[1:-1] == J_p[1:-1])
    # Boundary at equilibrium → overwritten with zero flux.
    assert J_n_out[0] == 0.0
    assert J_n_out[-1] == 0.0
    assert J_p_out[0] == 0.0
    assert J_p_out[-1] == 0.0


def test_apply_selective_contacts_does_not_mutate_inputs():
    """apply_selective_contacts must return fresh arrays; the caller's
    J_*_full pads (which come from np.concatenate in continuity.py) are
    treated as read-only."""
    J_n = np.array([0.0, 1.0, 2.0, 3.0, 0.0])
    J_p = np.array([0.0, -1.0, -2.0, -3.0, 0.0])
    n = np.array([2e23, 1.5e23, 1.2e23, 1.1e23, 1e23])
    p = np.array([1e22, 1e22, 1e22, 1e22, 2e22])

    J_n_copy = J_n.copy()
    J_p_copy = J_p.copy()

    apply_selective_contacts(
        J_n, J_p, n, p,
        S_n_L=1e4, S_p_L=1e4, S_n_R=1e4, S_p_R=1e4,
        n_L=1e23, p_L=1e22, n_R=1e23, p_R=1e22,
    )
    assert np.all(J_n == J_n_copy)
    assert np.all(J_p == J_p_copy)


# -- Schottky helpers -----------------------------------------------------

def test_schottky_equilibrium_n_matches_exponential():
    """``N_c · exp(-phi_B / V_T)`` for a 0.5 eV barrier at 300 K."""
    N_c = 2.5e25
    V_T = 0.025852
    phi_B = 0.5
    expected = N_c * np.exp(-phi_B / V_T)
    assert schottky_equilibrium_n(N_c, phi_B, V_T) == pytest.approx(expected)


def test_schottky_equilibrium_n_zero_barrier_returns_N_c():
    """φ_B = 0 is a flat-band contact; the thermionic relation returns
    ``N_c`` exactly, which is the degenerate-doping limit."""
    assert schottky_equilibrium_n(2.5e25, 0.0, 0.025852) == pytest.approx(2.5e25)


def test_schottky_equilibrium_p_mirrors_n_helper():
    """The p-side helper must return a numerically identical expression
    when called with N_v in place of N_c; the two helpers share the
    same thermionic formula modulo the DOS argument."""
    N_v = 1.8e25
    V_T = 0.025852
    phi_B = 0.6
    assert schottky_equilibrium_p(N_v, phi_B, V_T) == pytest.approx(
        schottky_equilibrium_n(N_v, phi_B, V_T)
    )
