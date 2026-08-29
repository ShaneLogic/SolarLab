"""Stable physical increments for logarithmic transient coordinates."""

from __future__ import annotations

import numpy as np


class DynamicStorageIncrementError(ValueError):
    """A logarithmic coordinate could not produce a finite physical increment."""


def log_density_increment(
    previous_density: object,
    log_coordinate_increment: object,
) -> np.ndarray:
    """Return ``n_old * expm1(delta_log_n)`` without subtracting two densities."""

    previous = np.asarray(previous_density, dtype=float)
    increment = np.asarray(log_coordinate_increment, dtype=float)
    if previous.shape != increment.shape:
        raise DynamicStorageIncrementError(
            "density and log-coordinate increments must have the same shape"
        )
    if (
        not np.all(np.isfinite(previous))
        or np.any(previous <= 0.0)
        or not np.all(np.isfinite(increment))
    ):
        raise DynamicStorageIncrementError(
            "density increments require positive finite density and finite coordinates"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        result = previous * np.expm1(increment)
    if not np.all(np.isfinite(result)):
        raise DynamicStorageIncrementError("density increment overflowed")
    return result


def logit_occupancy_increment(
    previous_occupancy: object,
    logit_coordinate_increment: object,
) -> np.ndarray:
    """Return the occupancy change from a logit change without cancellation."""

    previous = np.asarray(previous_occupancy, dtype=float)
    increment = np.asarray(logit_coordinate_increment, dtype=float)
    if previous.shape != increment.shape:
        raise DynamicStorageIncrementError(
            "occupancy and logit-coordinate increments must have the same shape"
        )
    if (
        not np.all(np.isfinite(previous))
        or np.any((previous <= 0.0) | (previous >= 1.0))
        or not np.all(np.isfinite(increment))
    ):
        raise DynamicStorageIncrementError(
            "occupancy increments require values inside (0, 1) and finite coordinates"
        )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        odds_increment = np.expm1(increment)
        result = (
            previous
            * (1.0 - previous)
            * odds_increment
            / (1.0 + previous * odds_increment)
        )
    if not np.all(np.isfinite(result)):
        raise DynamicStorageIncrementError("occupancy increment overflowed")
    return result


__all__ = [
    "DynamicStorageIncrementError",
    "log_density_increment",
    "logit_occupancy_increment",
]
