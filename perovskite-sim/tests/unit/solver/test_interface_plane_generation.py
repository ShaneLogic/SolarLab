"""F-04, step 1: let the plane closure report a SIGNED rate — and measure
why that is necessary but NOT sufficient.

The plane closure computes a physically correct detailed-balance reference,
``ni_s^2 = N_C N_V exp(-Eg_s/V_T)`` on the reduced interface gap, unlike the
bulk cross-carrier path whose reference is bulk-asymptotic and whose
negative excursions are partly a sampling artifact. So of the shipped
formulations it is the one where a negative (generation) rate is
*meaningful*. ``interface_plane_generation`` removes the clamp that was
discarding it.

WHAT THIS BUYS, MEASURED — and it is not what I expected. On
scaps_mirror_v2 in the dark at -0.5 V the signed branch does flow, but the
magnitude is negligible:

    cross-carrier path, clamp lifted     -8.3644e+00  A/m^2
    plane closure + generation           -8.2277e-21  A/m^2

Twenty orders apart. The closure's ceiling explains it: at full depletion
``R -> -ni_s^2 / (n1_s/v_p + p1_s/v_n)``, and with the measured constants at
interface 0 (``ni_s^2 = 1.98e24``, ``n1_s = 8.33e14``, ``v = 0.01 m/s``)
that ceiling is ``-2.4e7 m^-2 s^-1``, i.e. about ``-3.8e-12 A/m^2``. Even a
FULLY depleted plane cannot reach the bulk path's number.

So the NOGEN clamp was never the binding constraint. Two other suppressors
sit underneath it: the trap level is clamped to the plane gap
(``depth_n = clip(E_t - (chi_s - chi_ref), 0, eg_s)``, added to stop a deep
cliff from exploding ``n1_s``), which parks the level near a band edge and
inflates ``n1_s``; and the plane densities are supply-limited, so the plane
sits near quasi-equilibrium and ``n_s p_s - ni_s^2`` stays small.

Recorded so the next attempt starts from the ceiling arithmetic rather than
from the clamp. Default OFF, so nothing shipped changes.
"""
from __future__ import annotations

import dataclasses
import math

import pytest

from perovskite_sim.physics.interface_plane import (
    PlaneInterfaceParams, plane_rate, solve_plane_densities,
)

# Measured constants at interface 0 of scaps_mirror_v2, N_grid=30.
_NI_S_SQ = 1.9821e24
_N1_S = 8.3261e14
_P1_S = 2.3806e09
_V = 0.01


def _prm(**kw):
    base = dict(
        bn_L=1.0, bn_R=1.0, bp_L=1.0, bp_R=1.0,
        ni_s_sq=_NI_S_SQ, n1_s=_N1_S, p1_s=_P1_S,
    )
    base.update(kw)
    return PlaneInterfaceParams(**base)


# ---------------------------------------------------------------------------
# the signed branch itself
# ---------------------------------------------------------------------------

def test_clamp_is_the_default_and_discards_generation():
    """Depleted plane, default flag: the rate is thrown away."""
    prm = _prm()
    n_s = p_s = math.sqrt(_NI_S_SQ) * 1e-3       # deeply depleted
    assert plane_rate(n_s, p_s, prm, _V, _V) == 0.0


def test_generation_flows_when_enabled_and_has_the_right_sign():
    prm = _prm()
    n_s = p_s = math.sqrt(_NI_S_SQ) * 1e-3
    R = plane_rate(n_s, p_s, prm, _V, _V, allow_generation=True)
    assert R < 0.0, f"expected net generation, got {R:.6e}"


def test_recombination_branch_is_untouched_by_the_flag():
    """n_s p_s > ni_s^2 must be bit-identical either way -- the flag may only
    add a branch, never perturb the one that already worked."""
    prm = _prm()
    n_s, p_s = 1e22, 1e22                         # strongly injected
    assert plane_rate(n_s, p_s, prm, _V, _V) == plane_rate(
        n_s, p_s, prm, _V, _V, allow_generation=True,
    )


def test_rate_vanishes_exactly_at_detailed_balance():
    """n_s p_s == ni_s^2 gives exactly zero with or without the flag."""
    prm = _prm()
    n_s = math.sqrt(_NI_S_SQ)
    for flag in (False, True):
        assert plane_rate(n_s, n_s, prm, _V, _V, allow_generation=flag) == 0.0


def test_generation_is_bounded_by_the_depletion_limit():
    """|R| saturates at ni_s^2/(n1_s/v_p + p1_s/v_n) -- no runaway branch.

    This is what makes the signed form safe to enable: driving the plane
    arbitrarily empty cannot produce an arbitrarily large source.
    """
    prm = _prm()
    ceiling = _NI_S_SQ / (_N1_S / _V + _P1_S / _V)
    for scale in (1e-3, 1e-8, 1e-20, 0.0):
        n_s = p_s = math.sqrt(_NI_S_SQ) * scale
        R = plane_rate(n_s, p_s, prm, _V, _V, allow_generation=True)
        assert R >= -ceiling * (1.0 + 1e-12), (
            f"generation {R:.6e} exceeded the depletion limit {-ceiling:.6e}"
        )
    # And it actually approaches it, so the bound is tight rather than vacuous.
    R_empty = plane_rate(0.0, 0.0, prm, _V, _V, allow_generation=True)
    assert R_empty == pytest.approx(-ceiling, rel=1e-12)


# ---------------------------------------------------------------------------
# the finding: the clamp was not the binding constraint
# ---------------------------------------------------------------------------

def test_closure_ceiling_is_orders_below_the_bulk_path_estimate():
    """Pin the arithmetic that says "removing the clamp is not enough".

    The bulk cross-carrier path with its clamp lifted reports -8.36 A/m^2 on
    this interface. The closure's ABSOLUTE ceiling -- what it would give if
    the plane were completely empty -- is ~3.8e-12 A/m^2. If a future change
    to the trap-level clamp or the supply model closes that gap, this test
    fails and the F-04 limitation text can finally be revisited.
    """
    Q = 1.602176634e-19
    ceiling_areal = _NI_S_SQ / (_N1_S / _V + _P1_S / _V)    # m^-2 s^-1
    ceiling_current = Q * ceiling_areal                      # A/m^2
    bulk_path_estimate = 8.3644                              # A/m^2, measured
    assert ceiling_current < 1e-6 * bulk_path_estimate, (
        f"closure ceiling {ceiling_current:.3e} A/m^2 is no longer negligible "
        f"against the bulk-path {bulk_path_estimate} A/m^2 -- re-open F-04"
    )


def test_solver_path_stays_positive_definite_in_the_densities():
    """Enabling generation must not produce negative plane densities.

    The Newton unknowns are logarithms, so this is structural -- asserted
    because a sign-flipping R is exactly what destabilised the earlier
    two-sided mirror pair.
    """
    prm = _prm()
    n_s, p_s, R = solve_plane_densities(
        1e10, 1e10, 1e10, 1e10, prm, _V, _V, allow_generation=True,
    )
    assert n_s > 0.0 and p_s > 0.0 and math.isfinite(R)
