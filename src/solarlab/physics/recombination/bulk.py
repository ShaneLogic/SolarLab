"""Maxwell-Boltzmann bulk recombination rate laws with exact local tangents.

Every law returns the net recombination rate ``R`` [m^-3 s^-1] (positive is
recombination, negative is net thermal generation) together with its exact
partial derivatives with respect to the local electron and hole densities
[s^-1]. The arithmetic, including its evaluation order, is the historical
one, so a migrated consumer reproduces the historical arrays bit for bit:

* lifetime SRH ``(n p - ni^2) / (tau_p (n + n1) + tau_n (p + p1))``
* radiative ``B (n p - ni^2)``
* Auger ``(C_n n + C_p p) (n p - ni^2)``

The kernels do not validate or broadcast. They accept real or complex
arrays (the latter only for complex-step verification); coefficient
validation belongs to the bound reaction in ``reaction.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "RecombinationRate",
    "auger_recombination",
    "lifetime_srh_denominator",
    "lifetime_srh_from_denominator",
    "lifetime_srh_recombination",
    "radiative_recombination",
    "sum_recombination_rates",
]


@dataclass(frozen=True, slots=True)
class RecombinationRate:
    """Net rate [m^-3 s^-1] and local derivatives d/dn, d/dp [s^-1].

    The derivatives are ``None`` when only the value was requested.
    """

    rate: np.ndarray
    d_rate_dn: np.ndarray | None
    d_rate_dp: np.ndarray | None

    @property
    def has_derivatives(self) -> bool:
        return self.d_rate_dn is not None and self.d_rate_dp is not None


def _result(rate: Any, d_dn: Any, d_dp: Any) -> RecombinationRate:
    arrays = [np.asarray(rate)]
    if d_dn is not None:
        arrays += [np.asarray(d_dn), np.asarray(d_dp)]
    for array in arrays:
        if array.ndim:
            array.setflags(write=False)
    if d_dn is None:
        return RecombinationRate(arrays[0], None, None)
    return RecombinationRate(arrays[0], arrays[1], arrays[2])


def lifetime_srh_denominator(n: Any, p: Any, tau_n: Any, tau_p: Any, n1: Any, p1: Any) -> Any:
    """Bulk SRH denominator ``tau_p (n + n1) + tau_n (p + p1)`` [s m^-3]."""

    return tau_p * (n + n1) + tau_n * (p + p1)


def lifetime_srh_recombination(
    n: Any, p: Any, ni_sq: Any, tau_n: Any, tau_p: Any, n1: Any, p1: Any, *, derivatives: bool = True
) -> RecombinationRate:
    """Single-level lifetime-form SRH and its exact local derivatives.

    ``dR/dn = (p - R tau_p) / D`` and ``dR/dp = (n - R tau_n) / D`` follow
    from the quotient rule on ``R = (n p - ni^2) / D``.
    """

    denominator = lifetime_srh_denominator(n, p, tau_n, tau_p, n1, p1)
    return lifetime_srh_from_denominator(n, p, ni_sq, tau_n, tau_p, denominator, derivatives=derivatives)


def lifetime_srh_from_denominator(
    n: Any, p: Any, ni_sq: Any, tau_n: Any, tau_p: Any, denominator: Any, *, derivatives: bool = True
) -> RecombinationRate:
    """Lifetime-form SRH given its already evaluated denominator.

    Lets one evaluation report the denominator it divided by (the SRH
    observation) without computing it twice.
    """

    blocked = np.equal(tau_n, np.inf) | np.equal(tau_p, np.inf)
    if np.any(~np.isfinite(denominator) & ~blocked):
        raise FloatingPointError("finite-lifetime SRH denominator overflow or nonfinite value")
    numerator = n * p - ni_sq
    with np.errstate(invalid="ignore"):
        rate = numerator / denominator
    rate = np.where(blocked, np.copysign(0.0, np.real(numerator)), rate)
    if not derivatives:
        return _result(rate, None, None)
    with np.errstate(invalid="ignore"):
        d_dn = (p - rate * tau_p) / denominator
        d_dp = (n - rate * tau_n) / denominator
    if np.any(blocked):
        # A blocked capture leg (tau = +inf) makes R vanish identically, so
        # both exact partials are 0; the quotient-rule form would evaluate
        # 0 * inf = NaN there (N-class difference, only on these nodes).
        d_dn = np.where(blocked, 0.0, d_dn)
        d_dp = np.where(blocked, 0.0, d_dp)
    return _result(rate, d_dn, d_dp)


def radiative_recombination(n: Any, p: Any, ni_sq: Any, B_rad: Any, *, derivatives: bool = True) -> RecombinationRate:
    """Bimolecular band-to-band radiative recombination."""

    rate = B_rad * (n * p - ni_sq)
    if not derivatives:
        return _result(rate, None, None)
    return _result(rate, B_rad * p, B_rad * n)


def auger_recombination(
    n: Any, p: Any, ni_sq: Any, C_n: Any, C_p: Any, *, derivatives: bool = True
) -> RecombinationRate:
    """Band-to-band Auger recombination ``(C_n n + C_p p)(n p - ni^2)``."""

    numerator = n * p - ni_sq
    coefficient = C_n * n + C_p * p
    rate = coefficient * numerator
    if not derivatives:
        return _result(rate, None, None)
    return _result(
        rate,
        C_n * numerator + coefficient * p,
        C_p * numerator + coefficient * n,
    )


def sum_recombination_rates(*terms: RecombinationRate) -> RecombinationRate:
    """Add channels left to right, in the historical SRH + radiative + Auger order."""

    if not terms:
        raise ValueError("at least one recombination channel is required")
    rate: Any = terms[0].rate
    for term in terms[1:]:
        rate = rate + term.rate
    if not all(term.has_derivatives for term in terms):
        return _result(rate, None, None)
    d_dn: Any = terms[0].d_rate_dn
    d_dp: Any = terms[0].d_rate_dp
    for term in terms[1:]:
        d_dn = d_dn + term.d_rate_dn
        d_dp = d_dp + term.d_rate_dp
    return _result(rate, d_dn, d_dp)
