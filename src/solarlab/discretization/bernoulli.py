"""Historical element-wise Bernoulli function ``B(x) = x / (exp(x) - 1)``.

``bernoulli`` and ``bernoulli_derivative`` reproduce the legacy kernel
bit-for-bit (M-class migration). The stable paired evaluation that computes the
small exponential branch first lives in
:mod:`solarlab.discretization.stable_bernoulli` (N class).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["bernoulli", "bernoulli_derivative"]


def bernoulli(x: ArrayLike) -> NDArray[np.float64]:
    """Historical element-wise ``B(x)`` (bit-identical to the legacy kernel)."""
    x = np.asarray(x, dtype=float)
    result = np.empty_like(x)
    small = np.abs(x) < 1e-8
    huge_pos = x > 700.0
    large = ~small & ~huge_pos
    result[small] = 1.0 - x[small] / 2.0 + x[small]**2 / 12.0
    result[huge_pos] = 0.0
    result[large] = x[large] / np.expm1(x[large])
    return result


def bernoulli_derivative(x: ArrayLike) -> NDArray[np.float64]:
    """Historical element-wise ``B'(x)`` (bit-identical to the legacy kernel)."""
    x = np.asarray(x, dtype=float)
    result = np.empty_like(x)
    small = np.abs(x) < 1.0e-4
    large_positive = x > 50.0
    regular = ~small & ~large_positive

    xs = x[small]
    result[small] = (
        -0.5
        + xs / 6.0
        - xs**3 / 180.0
        + xs**5 / 5040.0
    )

    xp = x[large_positive]
    decay = np.exp(-xp)
    result[large_positive] = (
        decay * (1.0 - xp - decay) / (1.0 - decay) ** 2
    )

    xr = x[regular]
    exp_x = np.exp(xr)
    denominator = np.expm1(xr)
    result[regular] = (
        denominator - xr * exp_x
    ) / denominator**2
    return result
