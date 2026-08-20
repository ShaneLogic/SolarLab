"""Opt-in, compact-support regularization for non-smooth RHS closures.

Every width defaults to zero, which selects the historical expression
exactly.  Positive widths only modify a declared neighbourhood of the kink;
outside that neighbourhood the original formula is returned bit-for-bit.
This makes a width ladder an interpretable numerical sensitivity study rather
than a hidden global change to the constitutive law.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


def _finite_nonnegative(name: str, value: float) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and non-negative")
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return 0.0 if value == 0.0 else value


@dataclass(frozen=True)
class RHSRegularization:
    """Research-only widths for RHS kink-sensitivity studies.

    Units are part of the field names.  A zero-valued policy is inert and is
    suitable as the public default; callers must opt in explicitly.
    """

    poole_frenkel_field_width_V_m: float = 0.0
    interface_density_width_m3: float = 0.0
    te_cap_relative_width: float = 0.0

    def __post_init__(self) -> None:
        for field_name in (
            "poole_frenkel_field_width_V_m",
            "interface_density_width_m3",
            "te_cap_relative_width",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_nonnegative(field_name, getattr(self, field_name)),
            )

    @property
    def active(self) -> bool:
        return any(
            value > 0.0
            for value in (
                self.poole_frenkel_field_width_V_m,
                self.interface_density_width_m3,
                self.te_cap_relative_width,
            )
        )

    def refined(self, factor: float) -> "RHSRegularization":
        """Return a policy with every transition width reduced by ``factor``."""
        if isinstance(factor, (bool, np.bool_)):
            raise ValueError("regularization refinement factor must lie in (0, 1]")
        factor = float(factor)
        if not np.isfinite(factor) or not 0.0 < factor <= 1.0:
            raise ValueError("regularization refinement factor must lie in (0, 1]")
        return replace(
            self,
            poole_frenkel_field_width_V_m=(self.poole_frenkel_field_width_V_m * factor),
            interface_density_width_m3=self.interface_density_width_m3 * factor,
            te_cap_relative_width=self.te_cap_relative_width * factor,
        )


def compact_sqrt_abs(
    value: np.ndarray | float,
    width: float = 0.0,
) -> np.ndarray:
    """Return a C2 regularization of ``sqrt(abs(value))`` near zero.

    For ``abs(value) >= width`` the exact NumPy expression is retained.  In
    the transition region a sixth-order even polynomial matches the value and
    first two derivatives of ``sqrt(abs(value))`` at the boundary and has a
    finite zero-field derivative.
    """
    width = _finite_nonnegative("width", width)
    values = np.asarray(value, dtype=float)
    magnitude = np.abs(values)
    exact = np.sqrt(magnitude)
    if width == 0.0:
        return exact

    inside = magnitude < width
    if not np.any(inside):
        return exact
    z = magnitude[inside] / width
    polynomial = (77.0 * z**2 - 66.0 * z**4 + 21.0 * z**6) / 32.0
    result = np.array(exact, copy=True)
    result[inside] = np.sqrt(width) * polynomial
    return result


def compact_positive_part(
    value: np.ndarray | float,
    width: float = 0.0,
) -> np.ndarray:
    """Return a C2 compact smoothing of ``max(value, 0)``.

    The result is exactly zero below ``-width`` and exactly ``value`` above
    ``+width``.  It is intended only for an explicit width ladder; it must not
    be used to hide negative-state diagnostics.
    """
    width = _finite_nonnegative("width", width)
    values = np.asarray(value, dtype=float)
    exact = np.maximum(values, 0.0)
    if width == 0.0:
        return exact

    inside = np.abs(values) < width
    if not np.any(inside):
        return exact
    u = (values[inside] / width + 1.0) / 2.0
    transition = width * (2.0 * u**6 - 6.0 * u**5 + 5.0 * u**4)
    result = np.array(exact, copy=True)
    result[inside] = transition
    return result


def direction_preserving_magnitude_min(
    signed_value: float,
    magnitude_bound: float,
    relative_width: float = 0.0,
) -> float:
    """Smooth ``sign(value) * min(abs(value), abs(bound))`` near crossover.

    The quintic smootherstep has compact support in the relative band
    ``abs(abs(value)-abs(bound)) < relative_width * (abs(value)+abs(bound))``.
    It never imports the sign of the independently evaluated bound.
    """
    relative_width = _finite_nonnegative("relative_width", relative_width)
    value = float(signed_value)
    bound = abs(float(magnitude_bound))
    if not np.isfinite(value) or not np.isfinite(bound):
        raise ValueError("magnitude-min inputs must be finite")

    value_magnitude = abs(value)
    hard_magnitude = min(value_magnitude, bound)
    if relative_width == 0.0:
        return float(np.copysign(hard_magnitude, value))

    scale = relative_width * (value_magnitude + bound)
    if scale == 0.0:
        return float(np.copysign(0.0, value))
    normalized = (value_magnitude - bound) / scale
    if normalized <= -1.0:
        magnitude = value_magnitude
    elif normalized >= 1.0:
        magnitude = bound
    else:
        u = (normalized + 1.0) / 2.0
        weight = 6.0 * u**5 - 15.0 * u**4 + 10.0 * u**3
        magnitude = (1.0 - weight) * value_magnitude + weight * bound
    return float(np.copysign(magnitude, value))


__all__ = [
    "RHSRegularization",
    "compact_positive_part",
    "compact_sqrt_abs",
    "direction_preserving_magnitude_min",
]
