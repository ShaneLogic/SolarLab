"""Continuous composition grading within one layer (M class).

SCAPS material-driven grading (Burgelman & Marlein, 23rd EU PVSEC 2008): a
graded layer mixes material A (front face, the layer's scalar ``chi``/``Eg``)
and material B (back face, ``chi_back``/``Eg_back``) with composition
``y(x) in [0, 1]``. The material laws are

* ``Eg(y) = (1-y) Eg_A + y Eg_B - b y (1-y)`` (bowing ``b``)
* ``chi(y) = (1-y) chi_A + y chi_B`` (Vegard)

and the Eg-derived statistics are anchored to the FRONT value through the
DOS law, ``ni^2(x) = ni_A^2 exp(-(Eg(x) - Eg_A)/V_T)`` and
``n1, p1 *= exp(-(Eg(x) - Eg_A)/2V_T)`` (trap level fixed relative to
midgap), so ``n1 p1 = ni^2`` holds node by node. Grading is a static
coefficient transform; it never sees the state.

Flat grades are exact identities: equal endpoints (and zero bowing) return a
constant front array, never ``(1-y) E + y E`` with its interior rounding.

All arithmetic is the historical one (``perovskite_sim.physics.grading``).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np

__all__ = [
    "GRADING_DIRECTIONS",
    "GRADING_PROFILES",
    "GradingSpec",
    "affinity_profile",
    "band_gap_profile",
    "grade_n1_p1",
    "grade_ni_sq",
    "grading_coordinate",
    "minimum_band_gap_eV",
]

GRADING_PROFILES = ("linear", "parabolic", "exponential")
GRADING_DIRECTIONS = ("front_to_back", "back_to_front")
GradingProfile = Literal["linear", "parabolic", "exponential"]
GradingDirection = Literal["front_to_back", "back_to_front"]


def grading_coordinate(
    x_local: np.ndarray,
    thickness: float,
    profile: str = "linear",
    char_length: float | None = None,
    direction: str = "front_to_back",
) -> np.ndarray:
    """Composition coordinate ``y(x) in [0, 1]`` across one layer.

    ``x_local`` is measured from the layer's front face [m]. ``linear`` gives
    ``y = s``, ``parabolic`` ``y = s^2`` and ``exponential`` the Burgelman
    notch ``(1 - e^{-x/L}) / (1 - e^{-d/L})`` (linear when ``L`` is unset or
    non-positive), with ``s = clip(x/d, 0, 1)``; ``back_to_front`` flips ``y``.
    """

    s = np.clip(x_local / thickness, 0.0, 1.0)
    if profile == "parabolic":
        y = s * s
    elif profile == "exponential":
        if char_length is None or char_length <= 0.0:
            y = s
        else:
            L = char_length
            denom = 1.0 - np.exp(-thickness / L)
            if denom == 0.0:
                y = s
            else:
                y = (1.0 - np.exp(-x_local / L)) / denom
                y = np.clip(y, 0.0, 1.0)
    else:
        y = s
    if direction == "back_to_front":
        y = 1.0 - y
    return y


def band_gap_profile(y: np.ndarray, Eg_front: float, Eg_back: float, bowing: float = 0.0) -> np.ndarray:
    """``Eg(y) = (1-y) Eg_A + y Eg_B - b y (1-y)``; flat grades are exact."""

    if Eg_front == Eg_back and bowing == 0.0:
        return np.full_like(y, float(Eg_front))
    return (1.0 - y) * Eg_front + y * Eg_back - bowing * y * (1.0 - y)


def minimum_band_gap_eV(Eg_front: float, Eg_back: float | None = None, bowing: float = 0.0) -> float:
    """Exact minimum of the graded gap over ``y in [0, 1]``."""

    front = float(Eg_front)
    back = front if Eg_back is None else float(Eg_back)
    curve = float(bowing)
    candidates = [front, back]
    if curve > 0.0:
        stationary = (front + curve - back) / (2.0 * curve)
        if 0.0 < stationary < 1.0:
            candidates.append(float(band_gap_profile(np.asarray([stationary]), front, back, curve)[0]))
    return min(candidates)


def affinity_profile(y: np.ndarray, chi_front: float, chi_back: float) -> np.ndarray:
    """``chi(y) = (1-y) chi_A + y chi_B``; flat grades are exact."""

    if chi_front == chi_back:
        return np.full_like(y, float(chi_front))
    return (1.0 - y) * chi_front + y * chi_back


def grade_ni_sq(ni_front_sq: float, Eg_node: np.ndarray, Eg_front: float, V_T: float) -> np.ndarray:
    """``ni^2(x) = ni_A^2 exp(-(Eg(x) - Eg_A)/V_T)`` (exact where ``Eg = Eg_A``)."""

    return ni_front_sq * np.exp(-(Eg_node - Eg_front) / V_T)


def grade_n1_p1(
    n1_front: float, p1_front: float, Eg_node: np.ndarray, Eg_front: float, V_T: float
) -> tuple[np.ndarray, np.ndarray]:
    """``(n1(x), p1(x))``, both scaled by ``exp(-(Eg(x) - Eg_A)/2V_T)``."""

    r = np.exp(-(Eg_node - Eg_front) / (2.0 * V_T))
    return n1_front * r, p1_front * r


@dataclass(frozen=True, slots=True)
class GradingSpec:
    """A layer's grading declaration: back endpoints and the composition path.

    A layer is graded iff ``band_gap_back_eV`` or ``electron_affinity_back_eV``
    is set (the historical presence rule); a missing endpoint is flat.
    Unknown profile or direction names are rejected here, never read as linear.
    """

    band_gap_back_eV: float | None = None
    electron_affinity_back_eV: float | None = None
    profile: GradingProfile = "linear"
    direction: GradingDirection = "front_to_back"
    bowing_eV: float = 0.0
    characteristic_length_m: float | None = None

    def __post_init__(self) -> None:
        if self.profile not in GRADING_PROFILES:
            raise ValueError(f"grading profile must be one of {GRADING_PROFILES}, got {self.profile!r}")
        if self.direction not in GRADING_DIRECTIONS:
            raise ValueError(f"grading direction must be one of {GRADING_DIRECTIONS}, got {self.direction!r}")
        for name in ("band_gap_back_eV", "electron_affinity_back_eV", "characteristic_length_m"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not math.isfinite(float(value))):
                raise ValueError(f"{name} must be finite when set")
        if isinstance(self.bowing_eV, bool) or not math.isfinite(float(self.bowing_eV)):
            raise ValueError("bowing_eV must be finite")

    @property
    def graded(self) -> bool:
        return self.band_gap_back_eV is not None or self.electron_affinity_back_eV is not None

    def coordinate(self, x_local_m: np.ndarray, thickness_m: float) -> np.ndarray:
        return grading_coordinate(
            x_local_m, thickness_m, self.profile, self.characteristic_length_m, self.direction
        )
