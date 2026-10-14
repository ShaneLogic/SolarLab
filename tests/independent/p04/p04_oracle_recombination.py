"""Independent bulk-recombination oracle for P04-07 (MB lifetime SRH, B, Auger).

Rates are evaluated in exact rational arithmetic (``fractions.Fraction``) from
the exact binary64 inputs, so the oracle carries no rounding error at all.

Rate laws (net recombination, m^-3 s^-1; Shockley & Read, Phys. Rev. 87,
835 (1952); Hall, Phys. Rev. 87, 387 (1952))::

    U       = n p - ni^2                          (mass-action numerator)
    R_SRH   = U / (tau_p (n + n1) + tau_n (p + p1))
    R_rad   = B U
    R_Auger = (C_n n + C_p p) U

Detailed balance: every channel vanishes when n p = ni^2 (with the SRH
reference densities consistent, n1 p1 = ni^2).  Exact local derivatives::

    dR_SRH/dn = (p - R_SRH tau_p)/Den,   dR_SRH/dp = (n - R_SRH tau_n)/Den
    dR_rad/dn = B p,                     dR_rad/dp = B n
    dR_A/dn   = C_n U + (C_n n + C_p p) p,   dR_A/dp = C_p U + (C_n n + C_p p) n

Injection limits (p-type, p0 = N_A >> n0, n1, p1; excess density Delta)::

    R_SRH  -> Delta / tau_n            (low injection, Delta << p0)
    R_SRH  -> Delta / (tau_n + tau_p)  (high injection, n = p = Delta >> n1, p1)
    R_rad  -> B p0 Delta  |  B Delta^2
    R_A    -> C_p p0^2 Delta  |  (C_n + C_p) Delta^3

The exact low-injection SRH ratio used by the tests is::

    R_SRH tau_n / Delta = (n0 + p0 + Delta)
                          / (p0 + Delta + p1 + (tau_p/tau_n)(n0 + Delta + n1))

Weak-signal rule: near equilibrium the error of a binary64 evaluation is set
by the gross terms, not by the net rate, so tolerances are referred to the
gross scale ``G = (n p + ni^2) * coefficient`` (``gross_scale``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np


def _q(value: float) -> Fraction:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("exact oracle needs finite inputs")
    return Fraction(value)


@dataclass(frozen=True)
class BulkParameters:
    ni_sq: float
    tau_n: float
    tau_p: float
    n1: float
    p1: float
    B_rad: float
    C_n: float
    C_p: float


@dataclass(frozen=True)
class ExactRate:
    rate: Fraction
    d_n: Fraction
    d_p: Fraction
    gross: Fraction  # weak-signal scale of the rate
    gross_d_n: Fraction
    gross_d_p: Fraction


def srh_exact(n: float, p: float, par: BulkParameters) -> ExactRate:
    n_, p_ = _q(n), _q(p)
    ni2, tn, tp = _q(par.ni_sq), _q(par.tau_n), _q(par.tau_p)
    n1, p1 = _q(par.n1), _q(par.p1)
    den = tp * (n_ + n1) + tn * (p_ + p1)
    U = n_ * p_ - ni2
    R = U / den
    gross = (n_ * p_ + ni2) / den
    return ExactRate(
        R,
        (p_ - R * tp) / den,
        (n_ - R * tn) / den,
        gross,
        (p_ + abs(R) * tp) / den,
        (n_ + abs(R) * tn) / den,
    )


def radiative_exact(n: float, p: float, par: BulkParameters) -> ExactRate:
    n_, p_ = _q(n), _q(p)
    ni2, B = _q(par.ni_sq), _q(par.B_rad)
    return ExactRate(
        B * (n_ * p_ - ni2), B * p_, B * n_, abs(B) * (n_ * p_ + ni2), abs(B) * p_, abs(B) * n_
    )


def auger_exact(n: float, p: float, par: BulkParameters) -> ExactRate:
    n_, p_ = _q(n), _q(p)
    ni2, Cn, Cp = _q(par.ni_sq), _q(par.C_n), _q(par.C_p)
    U = n_ * p_ - ni2
    coef = Cn * n_ + Cp * p_
    gross_u = n_ * p_ + ni2
    return ExactRate(
        coef * U,
        Cn * U + coef * p_,
        Cp * U + coef * n_,
        abs(coef) * gross_u,
        abs(Cn) * gross_u + abs(coef) * p_,
        abs(Cp) * gross_u + abs(coef) * n_,
    )


def total_exact(n: float, p: float, par: BulkParameters) -> ExactRate:
    parts = (srh_exact(n, p, par), radiative_exact(n, p, par), auger_exact(n, p, par))
    return ExactRate(*(sum(getattr(r, f) for r in parts) for f in ExactRate.__dataclass_fields__))


def within(
    value: float, exact: Fraction, scale: Fraction, rel_eps: float
) -> bool:
    """|value - exact| <= rel_eps * eps * (|exact| + scale), exactly evaluated."""

    if not math.isfinite(float(value)):
        return False
    eps = Fraction(np.finfo(float).eps)
    return abs(_q(value) - exact) <= Fraction(rel_eps) * eps * (abs(exact) + scale)


def low_injection_srh_ratio(
    n0: float, p0: float, delta: float, par: BulkParameters
) -> Fraction:
    """Exact R_SRH tau_n / Delta for n = n0 + Delta, p = p0 + Delta, ni^2 = n0 p0."""

    n0_, p0_, d = _q(n0), _q(p0), _q(delta)
    tn, tp, n1, p1 = _q(par.tau_n), _q(par.tau_p), _q(par.n1), _q(par.p1)
    return (n0_ + p0_ + d) / (p0_ + d + p1 + (tp / tn) * (n0_ + d + n1))


# --------------------------------------------------------------------------
# Existing trap profiles (Phase 4a; physics/traps.py semantics re-derived)
# --------------------------------------------------------------------------


def exponential_edge_profile_ref(
    x_local: np.ndarray, thickness: float, N_if: float, N_bulk: float, L_d: float,
    edge: str = "both",
) -> np.ndarray:
    """N_t = N_b + (N_if - N_b) S(x); S normalised so that sup S = 1.

    For ``both`` the kernel sum e^{-x/L} + e^{-(d-x)/L} is convex, so its
    supremum is at a face: S_max = 1 + e^{-d/L} (>= 2 e^{-d/2L} by AM-GM),
    hence N_t(0) = N_t(d) = N_if and N_t(d/2) = N_b + (N_if-N_b) sech(d/2L).
    """

    x_local = np.asarray(x_local, dtype=float)
    if L_d <= 0.0:
        return np.full_like(x_local, N_bulk)
    left, right = np.exp(-x_local / L_d), np.exp(-(thickness - x_local) / L_d)
    if edge == "left":
        shape = left
    elif edge == "right":
        shape = right
    elif edge == "both":
        shape = (left + right) / max(1.0 + math.exp(-thickness / L_d), 2.0 * math.exp(-thickness / (2 * L_d)))
    else:
        raise ValueError(edge)
    return N_bulk + (N_if - N_bulk) * shape


def gaussian_edge_profile_ref(
    x_local: np.ndarray, thickness: float, N_if: float, N_bulk: float, sigma: float,
    edge: str = "both",
) -> np.ndarray:
    """Gaussian kernel exp(-(s/sigma)^2); ``both`` uses the documented
    closed-form normaliser max(1 + k(d), 2 k(d/2)) (exact supremum only when
    the face is the maximum; < 3 % low in the sigma ~ 0.64 d band)."""

    x_local = np.asarray(x_local, dtype=float)
    if sigma <= 0.0:
        return np.full_like(x_local, N_bulk)

    def k(s):
        return np.exp(-((np.asarray(s, dtype=float) / sigma) ** 2))

    if edge == "left":
        shape = k(x_local)
    elif edge == "right":
        shape = k(thickness - x_local)
    elif edge == "both":
        shape = (k(x_local) + k(thickness - x_local)) / max(
            float(1.0 + k(thickness)), float(2.0 * k(0.5 * thickness))
        )
    else:
        raise ValueError(edge)
    return N_bulk + (N_if - N_bulk) * shape


def tau_from_trap_density_ref(tau_bulk: float, N_t: np.ndarray, N_bulk: float) -> np.ndarray:
    """tau = tau_b N_b / max(N_t, N_b): inverse-density rule with passivation floor."""

    return tau_bulk * N_bulk / np.maximum(np.asarray(N_t, dtype=float), N_bulk)
