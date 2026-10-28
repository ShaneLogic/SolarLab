"""Small array helpers shared by the recombination constitutive modules.

These helpers only validate and freeze inputs. They never change a finite
value, so a law evaluated on a validated array performs exactly the same
floating-point operations as on the caller's array.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "frozen_float_array",
    "frozen_bool_mask",
    "require_finite",
    "finite_positive",
    "finite_nonnegative",
]


def frozen_float_array(value: object, *, name: str) -> np.ndarray:
    """Return a detached read-only float64 copy of ``value``."""

    array = np.array(value, dtype=float, copy=True)
    array.setflags(write=False)
    return array


def frozen_bool_mask(value: object, *, name: str, length: int | None = None) -> np.ndarray:
    """Return a detached read-only 1D boolean mask."""

    raw = np.asarray(value)
    if raw.dtype != np.bool_:
        raise TypeError(f"{name} must be a boolean mask")
    mask = np.array(raw, dtype=bool, copy=True)
    if mask.ndim != 1 or (length is not None and mask.shape != (length,)):
        raise ValueError(f"{name} must be a 1D mask of length {length}")
    mask.setflags(write=False)
    return mask


def require_finite(array: np.ndarray, *, name: str) -> None:
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")


def finite_positive(value: object, name: str) -> float:
    """A finite strictly positive scalar; booleans are not numbers here."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and positive")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def finite_nonnegative(value: object, name: str) -> float:
    """A finite non-negative scalar; booleans are not numbers here."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and non-negative")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and non-negative") from exc
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number
