"""Independent Bernoulli and Scharfetter-Gummel oracle for P04-01.

Nothing here imports the legacy engine or the new ``solarlab`` packages.  Two
kinds of reference are provided:

* ``*_hp`` functions evaluate in ``decimal`` at ``HP_DIGITS`` significant
  digits from the *exact* binary64 input (``Decimal(float)`` is exact), so
  their error is far below one binary64 ulp.  They are the oracle.
* ``*_ref`` functions are an independent binary64 construction used where a
  vectorised reference is convenient (SG fluxes, Jacobians).  They are
  themselves checked against the ``*_hp`` oracle by the test suite.

Derivation (all exact identities, no approximation)
---------------------------------------------------
B(x)  = x / (e^x - 1),               B(0) = 1 (removable singularity)
B(-x) = -x/(e^-x - 1) = x e^x/(e^x - 1) = x + B(x)            (I1)
B'(x) = (e^x - 1 - x e^x) / (e^x - 1)^2,  B'(0) = -1/2
Differentiating (I1):  B'(-x) = -1 - B'(x)                     (I2)
B is convex (B'' > 0), so B' increases strictly from -1 (x->-inf) to 0.
Series (Bernoulli numbers B_2k):  B(x) = 1 - x/2 + sum_k B_2k x^2k/(2k)!
  = 1 - x/2 + x^2/12 - x^4/720 + x^6/30240 - x^8/1209600 + ...
For x > 0 with t = e^-x:  B(x) = x t/(1 - t),
  B'(x) = t (1 - x - t)/(1 - t)^2 = -t s/(1 - t)^2,  s = e^-x - 1 + x >= 0.

The P04-01 plan text requires the small-exponent branch to be computed
first and the large branch recovered by a stable addition, i.e. use (I1) as
B(-|x|) = |x| + B(|x|) (a sum of two positives), never B(|x|) = B(-|x|) - |x|
(which loses B(50) ~ 9.6e-21 entirely against B(-50) ~ 50).

Scharfetter-Gummel (Scharfetter & Gummel, IEEE TED 16, 64 (1969)).  For a
species with charge number z (electrons z=-1, holes z=+1) between nodes i and
i+1 at spacing h, with xi = (psi_{i+1} - psi_i)/V_T for the driving
(transport) potential psi, the particle flux directed from i to i+1 is

    F = (D/h) [ B(z xi) c_i - B(-z xi) c_{i+1} ]                 (SG)

Checks: xi = 0 gives Fick's law F = -D (c_{i+1}-c_i)/h.  For |z xi| >> 1 it
is upwind drift F ~ -(D/h) z xi c_up, with c_up = c_{i+1} if z xi > 0 and
c_up = c_i if z xi < 0 (the drift particle flux is -z mu c dpsi/dx).
Zero flux requires c_{i+1}/c_i = B(z xi)/B(-z xi) = e^{-z xi}, i.e. the
Boltzmann equilibrium c ~ exp(-z psi/V_T).  The legacy electric currents are
J_n = -q F_n and J_p = +q F_p (perovskite_sim/discretization/fe_operators.py).
"""

from __future__ import annotations

from decimal import Decimal, localcontext

import numpy as np

HP_DIGITS = 100
_SERIES_SWITCH_HP = Decimal("1e-15")


def _hp(value: float) -> Decimal:
    return Decimal(float(value))


def bernoulli_hp(x: float) -> Decimal:
    """B(x) at ``HP_DIGITS`` digits from the exact binary64 input."""

    with localcontext() as ctx:
        ctx.prec = HP_DIGITS
        d = _hp(x)
        if d == 0:
            return Decimal(1)
        if abs(d) < _SERIES_SWITCH_HP:
            # Truncation after x^4 is below 1e-60 relative here.
            return 1 - d / 2 + d * d / 12 - d**4 / 720
        if d > 0:
            t = (-d).exp()
            return +(d * t / (1 - t))
        return +(d / (d.exp() - 1))


def bernoulli_derivative_hp(x: float) -> Decimal:
    """B'(x) at ``HP_DIGITS`` digits from the exact binary64 input."""

    with localcontext() as ctx:
        ctx.prec = HP_DIGITS
        d = _hp(x)
        if d == 0:
            return Decimal(-1) / 2
        if abs(d) < _SERIES_SWITCH_HP:
            return Decimal(-1) / 2 + d / 6 - d**3 / 180
        if d > 0:
            t = (-d).exp()
            return +(t * (1 - d - t) / (1 - t) ** 2)
        e = d.exp()
        return +((e - 1 - d * e) / (e - 1) ** 2)


# --------------------------------------------------------------------------
# Independent binary64 reference (vectorised)
# --------------------------------------------------------------------------

_SERIES_SWITCH = 0.05


def _bernoulli_nonnegative(y: np.ndarray) -> np.ndarray:
    """B(y) for y >= 0 without overflow or cancellation."""

    out = np.empty_like(y)
    small = y < _SERIES_SWITCH
    ys = y[small]
    # Truncation error < 1e-20 relative for y < 0.05.
    out[small] = (
        1.0 - ys / 2.0 + ys**2 / 12.0 - ys**4 / 720.0 + ys**6 / 30240.0
        - ys**8 / 1209600.0
    )
    large = ~small & np.isfinite(y)
    yl = y[large]
    with np.errstate(over="ignore", under="ignore"):
        out[large] = yl * np.exp(-yl) / -np.expm1(-yl)
    out[np.isposinf(y)] = 0.0
    out[np.isnan(y)] = np.nan
    return out


def bernoulli_ref(x: np.ndarray) -> np.ndarray:
    """Binary64 B(x): small branch first, large branch by stable addition."""

    x = np.asarray(x, dtype=float)
    ax = np.abs(x)
    small_branch = _bernoulli_nonnegative(ax)
    return np.where(x >= 0.0, small_branch, ax + small_branch)


def _shifted_exp_tail(y: np.ndarray) -> np.ndarray:
    """s(y) = e^-y - 1 + y >= 0 for y in [0, 1], by its alternating series."""

    total = np.zeros_like(y)
    term = y * y / 2.0
    for k in range(3, 24):
        total = total + term
        term = -term * y / k
    return total


def bernoulli_derivative_ref(x: np.ndarray) -> np.ndarray:
    """Binary64 B'(x) using (I2) for x < 0 and a cancellation-free x > 0 form."""

    x = np.asarray(x, dtype=float)
    ax = np.abs(x)
    positive = np.empty_like(ax)
    small = ax < _SERIES_SWITCH
    ys = ax[small]
    positive[small] = (
        -0.5 + ys / 6.0 - ys**3 / 180.0 + ys**5 / 5040.0 - ys**7 / 151200.0
    )
    mid = (~small) & (ax <= 1.0)
    ym = ax[mid]
    t = np.exp(-ym)
    u = -np.expm1(-ym)
    positive[mid] = -t * _shifted_exp_tail(ym) / (u * u)
    large = (ax > 1.0) & np.isfinite(ax)
    yl = ax[large]
    with np.errstate(over="ignore", under="ignore"):
        tl = np.exp(-yl)
        ul = -np.expm1(-yl)
        positive[large] = tl * ((1.0 - yl) - tl) / (ul * ul)
    positive[np.isposinf(ax)] = -0.0
    positive[np.isnan(ax)] = np.nan
    return np.where(x >= 0.0, positive, -1.0 - positive)


# --------------------------------------------------------------------------
# Scharfetter-Gummel reference (directed particle flux, node i -> i+1)
# --------------------------------------------------------------------------


def sg_particle_flux_ref(
    psi: np.ndarray,
    c: np.ndarray,
    h: np.ndarray,
    D: np.ndarray | float,
    V_T: float,
    z: int,
) -> np.ndarray:
    """Equation (SG): particle flux [m^-2 s^-1] on every face, +x positive."""

    psi = np.asarray(psi, dtype=float)
    c = np.asarray(c, dtype=float)
    xi = (psi[1:] - psi[:-1]) / V_T
    return (np.asarray(D, dtype=float) / h) * (
        bernoulli_ref(z * xi) * c[:-1] - bernoulli_ref(-z * xi) * c[1:]
    )


def sg_particle_flux_jacobian_ref(
    psi: np.ndarray,
    c: np.ndarray,
    h: np.ndarray,
    D: np.ndarray | float,
    V_T: float,
    z: int,
) -> dict[str, np.ndarray]:
    """Analytic face derivatives of (SG) w.r.t. c_i, c_{i+1}, psi_i, psi_{i+1}."""

    psi = np.asarray(psi, dtype=float)
    c = np.asarray(c, dtype=float)
    xi = (psi[1:] - psi[:-1]) / V_T
    pref = np.asarray(D, dtype=float) / h
    d_xi = pref * z * (
        bernoulli_derivative_ref(z * xi) * c[:-1]
        + bernoulli_derivative_ref(-z * xi) * c[1:]
    )
    return {
        "flux": sg_particle_flux_ref(psi, c, h, D, V_T, z),
        "d_c_left": pref * bernoulli_ref(z * xi),
        "d_c_right": -pref * bernoulli_ref(-z * xi),
        "d_psi_left": -d_xi / V_T,
        "d_psi_right": d_xi / V_T,
    }


def sg_particle_flux_hp(
    psi_left: float,
    psi_right: float,
    c_left: float,
    c_right: float,
    h: float,
    D: float,
    V_T: float,
    z: int,
) -> Decimal:
    """One face of (SG) in ``decimal`` from exact binary64 inputs."""

    with localcontext() as ctx:
        ctx.prec = HP_DIGITS
        xi = (_hp(psi_right) - _hp(psi_left)) / _hp(V_T)
        zx = z * xi
        b_pos = _bernoulli_decimal(zx)
        b_neg = _bernoulli_decimal(-zx)
        return +((_hp(D) / _hp(h)) * (b_pos * _hp(c_left) - b_neg * _hp(c_right)))


def _bernoulli_decimal(d: Decimal) -> Decimal:
    """B on a Decimal argument (used when xi itself is formed in Decimal)."""

    if d == 0:
        return Decimal(1)
    if abs(d) < _SERIES_SWITCH_HP:
        return 1 - d / 2 + d * d / 12 - d**4 / 720
    if d > 0:
        t = (-d).exp()
        return d * t / (1 - t)
    return d / (d.exp() - 1)


def sg_equilibrium_bound(
    psi_left: np.ndarray,
    psi_right: np.ndarray,
    V_T: float,
) -> np.ndarray:
    """A-priori relative bound on the SG equilibrium residual in binary64.

    With c = c0 exp(-z psi/V_T) formed in binary64, the two SG terms differ by
    the rounding of psi/V_T at both nodes, of xi itself and of a handful of
    products: relative mismatch <= eps (|psi_l/V_T| + |psi_r/V_T| + |xi|) plus
    O(10 eps).  A factor 8 margin is applied; the bound is fixed before any
    candidate is evaluated and does not use the candidate's output.
    """

    eps = np.finfo(float).eps
    a = np.abs(np.asarray(psi_left) / V_T)
    b = np.abs(np.asarray(psi_right) / V_T)
    xi = np.abs((np.asarray(psi_right) - np.asarray(psi_left)) / V_T)
    return 8.0 * eps * (16.0 + a + b + xi)
