"""Continuous bandgap / electron-affinity grading within a single layer.

SCAPS "material-driven" grading (Burgelman & Marlein, 23rd EU PVSEC 2008):
a graded layer is a mix of two pure materials A (front face, y=0) and B
(back face, y=1) with composition ``y(x)`` varying across the thickness.
All graded material properties are derived from the local composition.

In SolarLab the layer's existing scalar ``chi`` / ``Eg`` are the FRONT
endpoints (material A); the new ``chi_back`` / ``Eg_back`` are the BACK
endpoints (material B). A layer is graded iff ``has_grading_params`` is
True. The build (``solver/mol.build_material_arrays``) calls these pure
functions once to fill the per-node ``chi`` / ``Eg`` / ``ni_sq`` / ``n1`` /
``p1`` arrays; nothing here is state- or time-dependent, so grading is a
static coefficient transform that cannot perturb the per-RHS Newton path
(unlike the dropped per-RHS BBD interface-density term).

Design decisions
----------------
* **Front-anchored derived terms.** Every Eg-derived quantity is anchored
  to the front value and scaled by the SCAPS DOS law
  ``exp(-(Eg(x) - Eg_front) / kT)``. This is the ``ni^2 = Nc·Nv·exp(-Eg/kT)``
  ratio with ``Nc``/``Nv`` cancelled, so it is exact without per-layer DOS
  data and reduces to the original scalar at a flat grade.
* **Flat-grade is byte-identical.** ``band_gap_profile`` / ``affinity_profile``
  short-circuit to a constant-front array when the endpoints are equal (and
  bowing is zero), so a flat grade produces ``np.array_equal`` arrays vs the
  ungraded scalar-broadcast path — interior-node IEEE rounding of
  ``(1-y)·E + y·E`` never leaks in.

Documented limitation
---------------------
The optical absorption (``alpha`` / external TMM ``n,k``) is **not** graded:
a graded absorber's absorption edge does not blue/red-shift spatially. SCAPS
grades ``alpha(lambda, y)``; SolarLab keeps layer-nominal optics. A true CIGS
V-notch is composed of two graded sub-layers (front linear + back exponential
Ga-rich), which the existing multilayer machinery already supports.
"""
from __future__ import annotations

import numpy as np


def has_grading_params(p) -> bool:
    """True if a layer's MaterialParams declares a back endpoint (is graded)."""
    if p is None:
        return False
    return getattr(p, "Eg_back", None) is not None or getattr(p, "chi_back", None) is not None


def grading_coordinate(
    x_local: np.ndarray,
    thickness: float,
    profile: str = "linear",
    char_length: float | None = None,
    direction: str = "front_to_back",
) -> np.ndarray:
    """Composition coordinate ``y(x) in [0, 1]`` across one layer.

    ``x_local`` is the position measured from the layer's front face [m].
    ``profile``:
      - ``"linear"``     : y = s
      - ``"parabolic"``  : y = s²
      - ``"exponential"``: Burgelman notch y = (1 - e^{-x/L}) / (1 - e^{-d/L}),
        L = ``char_length``. Falls back to linear if L is unset / non-positive.
    where ``s = clip(x_local / thickness, 0, 1)``. ``direction="back_to_front"``
    flips the profile (``y -> 1 - y``).
    """
    s = np.clip(x_local / thickness, 0.0, 1.0)
    if profile == "parabolic":
        y = s * s
    elif profile == "exponential":
        if char_length is None or char_length <= 0.0:
            y = s  # degenerate L -> linear
        else:
            L = char_length
            denom = 1.0 - np.exp(-thickness / L)
            if denom == 0.0:
                y = s
            else:
                y = (1.0 - np.exp(-x_local / L)) / denom
                y = np.clip(y, 0.0, 1.0)
    else:  # "linear" (default)
        y = s
    if direction == "back_to_front":
        y = 1.0 - y
    return y


def band_gap_profile(
    y: np.ndarray,
    Eg_front: float,
    Eg_back: float,
    bowing: float = 0.0,
) -> np.ndarray:
    """SCAPS band-gap composition law Eg(y) = (1-y)·Eg_A + y·Eg_B - b·y(1-y).

    Short-circuits to a constant-front array when ``Eg_front == Eg_back`` and
    ``bowing == 0`` so a flat grade is byte-identical to the scalar broadcast.
    """
    if Eg_front == Eg_back and bowing == 0.0:
        return np.full_like(y, float(Eg_front))
    return (1.0 - y) * Eg_front + y * Eg_back - bowing * y * (1.0 - y)


def affinity_profile(
    y: np.ndarray,
    chi_front: float,
    chi_back: float,
) -> np.ndarray:
    """Linear (Vegard) electron-affinity law chi(y) = (1-y)·chi_A + y·chi_B.

    Short-circuits to a constant-front array when ``chi_front == chi_back`` so
    a flat grade is byte-identical to the scalar broadcast.
    """
    if chi_front == chi_back:
        return np.full_like(y, float(chi_front))
    return (1.0 - y) * chi_front + y * chi_back


def grade_ni_sq(
    ni_front_sq: float,
    Eg_node: np.ndarray,
    Eg_front: float,
    V_T: float,
) -> np.ndarray:
    """Per-node ni² anchored to the front value via the DOS law.

    ni²(x) = ni_front² · exp(-(Eg(x) - Eg_front) / V_T)

    This is ``Nc·Nv·exp(-Eg/kT)`` with the (un-graded) DOS factors cancelled,
    so it needs no per-layer Nc/Nv and equals ``ni_front²`` exactly where
    ``Eg(x) == Eg_front`` (exp(0) = 1).
    """
    return ni_front_sq * np.exp(-(Eg_node - Eg_front) / V_T)


def grade_n1_p1(
    n1_front: float,
    p1_front: float,
    Eg_node: np.ndarray,
    Eg_front: float,
    V_T: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-node SRH n1/p1 holding the trap level fixed relative to midgap.

    With E_i = Eg/2 (the srh_n1_p1_from_trap_depth convention) and a fixed
    E_t - E_i, both n1 and p1 scale by ``r = exp(-(Eg(x) - Eg_front) / 2V_T)``,
    so ``n1·p1 = ni_front²·exp(-(Eg(x) - Eg_front)/V_T) = ni²(x)`` exactly
    (detailed balance preserved per node). Returns ``(n1(x), p1(x))``; both
    equal their front scalars where ``Eg(x) == Eg_front``.
    """
    r = np.exp(-(Eg_node - Eg_front) / (2.0 * V_T))
    return n1_front * r, p1_front * r
