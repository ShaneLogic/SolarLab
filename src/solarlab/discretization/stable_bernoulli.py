"""Stable paired Bernoulli evaluation (N class).

The historical element-wise kernel is :mod:`solarlab.discretization.bernoulli`
(M class); the two must not be mixed silently.

* ``bernoulli_pair`` evaluates ``B(x)`` and ``B(-x)`` together. It first
  computes the small branch ``B(|x|)`` from ``exp(-|x|)`` only (no overflow,
  no cancellation) and then recovers the large branch with the exact identity
  ``B(-a) = a + B(a)`` -- an addition of two positive numbers. The large value
  therefore never has to be differenced to recover the small one: computing
  ``B(50)`` as ``B(-50) - 50`` would return 0 in binary64, while the pair keeps
  ``B(50) = 9.6e-21``. Derivatives follow ``B'(-a) = -(1 + B'(a))``; on
  ``1e-4 <= a <= 1`` the cancelling numerator ``expm1(a) - a*exp(a)`` is summed
  as its series. This is a new arithmetic (N-class): values agree with the
  historical ones to a few units in the last place (the historical value is
  exactly 0 above x = 700, the pair keeps the normal-range value), and the
  derivative is ~2 ulp accurate where the historical quotient loses ~1e-12.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["BernoulliPair", "bernoulli_pair"]

_VALUE_SERIES_LIMIT = 1e-8
_DERIVATIVE_SERIES_LIMIT = 1.0e-4
_DERIVATIVE_DECAY_LIMIT = 50.0
# a*exp(a) - expm1(a) = sum_{k>=2} (k-1)/k! a^k; coefficients of a^(k-2) for
# Horner evaluation on 0 < a <= 1, highest order first (truncation < 1e-19).
_NUMERATOR_SERIES = tuple(
    (k - 1) / math.factorial(k) for k in range(22, 1, -1)
)


@dataclass(frozen=True)
class BernoulliPair:
    """``B(x)``, ``B(-x)`` and the derivative of ``B`` at ``x`` and at ``-x``.

    ``backward_derivative`` is ``B'(-x)``; the derivative of ``B(-x)`` with
    respect to ``x`` is therefore ``-backward_derivative``.
    """

    forward: NDArray[np.float64]
    backward: NDArray[np.float64]
    forward_derivative: NDArray[np.float64]
    backward_derivative: NDArray[np.float64]


def _small_branch(a: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``B(a)`` and ``B'(a)`` for ``a >= 0`` using only ``exp(-a)``-safe forms."""
    value = np.empty_like(a)
    series = a < _VALUE_SERIES_LIMIT
    value[series] = 1.0 - a[series] / 2.0 + a[series] ** 2 / 12.0
    rest = ~series
    ar = a[rest]
    value[rest] = ar * np.exp(-ar) / -np.expm1(-ar)

    derivative = np.empty_like(a)
    near = a < _DERIVATIVE_SERIES_LIMIT
    moderate = ~near & (a <= 1.0)
    decay_region = a > _DERIVATIVE_DECAY_LIMIT
    regular = ~near & ~moderate & ~decay_region
    an = a[near]
    derivative[near] = -0.5 + an / 6.0 - an**3 / 180.0 + an**5 / 5040.0
    # expm1(a) - a*exp(a) cancels to -a^2/2 for small a; sum its series instead.
    am = a[moderate]
    numerator = np.zeros_like(am)
    for coefficient in _NUMERATOR_SERIES:
        numerator = numerator * am + coefficient
    derivative[moderate] = -(am * am) * numerator / np.expm1(am) ** 2
    ad = a[decay_region]
    decay = np.exp(-ad)
    derivative[decay_region] = decay * (1.0 - ad - decay) / (1.0 - decay) ** 2
    ag = a[regular]
    denominator = np.expm1(ag)
    derivative[regular] = (denominator - ag * np.exp(ag)) / denominator**2
    return value, derivative


def bernoulli_pair(x: ArrayLike) -> BernoulliPair:
    """Stable ``B(x)``/``B(-x)`` and derivatives, small exponential branch first."""
    values = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Bernoulli arguments must be finite")
    a = np.abs(values)
    small, small_derivative = _small_branch(a)
    large = a + small
    large_derivative = -(1.0 + small_derivative)
    nonnegative = values >= 0.0
    return BernoulliPair(
        forward=np.where(nonnegative, small, large),
        backward=np.where(nonnegative, large, small),
        forward_derivative=np.where(nonnegative, small_derivative, large_derivative),
        backward_derivative=np.where(nonnegative, large_derivative, small_derivative),
    )
