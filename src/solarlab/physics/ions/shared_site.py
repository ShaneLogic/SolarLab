"""Shared-site physical derivatives and cancellation-aware paired increments.

The first/second derivatives retain the original dual-ion DAE arithmetic.
The pairwise secant is an algebraically equivalent numerical representation
of a softmax density difference, including motion from nonzero coordinates.
"""

from __future__ import annotations

import numpy as np

from solarlab.physics.ions.finite_site import shared_site_fractions

__all__ = ["shared_site_coordinate_jacobian_m3", "shared_site_coordinate_hessian_m3", "shared_site_secant_m3"]


def _physical_arrays(positive: np.ndarray, negative: np.ndarray, capacity: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    plus, minus, limit = (np.asarray(value, dtype=float) for value in (positive, negative, capacity))
    if (plus.ndim != 1 or plus.size == 0 or minus.shape != plus.shape or limit.shape != plus.shape
            or any(not np.all(np.isfinite(value)) for value in (plus, minus, limit))
            or np.any(plus < 0.0) or np.any(minus < 0.0) or np.any(limit <= 0.0)
            or np.any(plus + minus >= limit)):
        raise ValueError("shared-site physical arrays require matched finite populations and a positive common vacancy")
    return plus, minus, limit


def shared_site_coordinate_jacobian_m3(positive: np.ndarray, negative: np.ndarray, capacity: np.ndarray) -> np.ndarray:
    """Original node-local d(P+,P-)/d(a+,a-), indexed [node,species,coordinate]."""
    plus, minus, limit = _physical_arrays(positive, negative, capacity)
    result: np.ndarray = np.zeros((plus.size, 2, 2), dtype=float)
    plus_fraction, minus_fraction = plus / limit, minus / limit
    coupling = plus * minus_fraction
    result[:, 0, 0] = plus * (1.0 - plus_fraction)
    result[:, 0, 1] = -coupling
    result[:, 1, 0] = -coupling
    result[:, 1, 1] = minus * (1.0 - minus_fraction)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("shared-site coordinate mass matrix is nonfinite")
    return result


def shared_site_coordinate_hessian_m3(positive: np.ndarray, negative: np.ndarray, capacity: np.ndarray) -> np.ndarray:
    """Original complete softmax Hessian [node,species,coordinate,coordinate]."""
    plus, minus, limit = _physical_arrays(positive, negative, capacity)
    fractions = np.stack((plus / limit, minus / limit), axis=1)
    result: np.ndarray = np.zeros((plus.size, 2, 2, 2), dtype=float)
    for species in range(2):
        for first in range(2):
            for second in range(2):
                result[:, species, first, second] = limit * fractions[:, species] * (
                    (float(species == second) - fractions[:, second])
                    * (float(species == first) - fractions[:, first])
                    - fractions[:, first] * (float(first == second) - fractions[:, second])
                )
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("shared-site coordinate Hessian is nonfinite")
    return result


def shared_site_secant_m3(
    capacity: np.ndarray, positive_reference: np.ndarray, negative_reference: np.ndarray,
    old_positive: np.ndarray, old_negative: np.ndarray,
    new_positive: np.ndarray, new_negative: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Stable paired physical increments; no independent logistic fallback.

    The reference arguments are the compiled positive_ion_logit_reference and
    negative_ion_logit_reference: log(species fraction/vacancy fraction).
    Each pair exchanges C*(theta_i_new*theta_j_old-theta_i_old*theta_j_new).
    Vacancy has coordinate increment zero. Every expm1 argument is nonpositive;
    coordinate differences precede reference addition and each exchange is
    added with opposite signs by identity, without a balancing correction.
    """
    arrays = [np.asarray(value, dtype=float) for value in (
        capacity, positive_reference, negative_reference, old_positive, old_negative, new_positive, new_negative,
    )]
    limit, ref_plus, ref_minus, old_plus, old_minus, new_plus, new_minus = arrays
    if (limit.ndim != 1 or limit.size == 0 or any(value.shape != limit.shape or not np.all(np.isfinite(value)) for value in arrays)
            or np.any(limit <= 0.0)):
        raise ValueError("shared-site secant requires matched finite vectors and positive common capacities")
    before = np.stack(shared_site_fractions(ref_plus, ref_minus, old_plus, old_minus))
    after = np.stack(shared_site_fractions(ref_plus, ref_minus, new_plus, new_minus))
    _physical_arrays(limit * before[0], limit * before[1], limit)
    _physical_arrays(limit * after[0], limit * after[1], limit)
    delta = np.stack((new_plus - old_plus, new_minus - old_minus, np.zeros_like(limit)))
    if not np.all(np.isfinite(delta)):
        raise FloatingPointError("shared-site coordinate increment is nonfinite")
    change = np.zeros_like(before)
    for first in range(3):
        for second in range(first + 1, 3):
            difference = delta[first] - delta[second]
            positive = limit * after[first] * before[second] * (-np.expm1(-np.maximum(difference, 0.0)))
            negative = limit * before[first] * after[second] * np.expm1(np.minimum(difference, 0.0))
            pair = np.where(difference >= 0.0, positive, negative)
            change[first] += pair
            change[second] -= pair
    if not np.all(np.isfinite(change)):
        raise FloatingPointError("shared-site physical secant is nonfinite")
    return change[0], change[1]
