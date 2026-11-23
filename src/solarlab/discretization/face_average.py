"""Face averages and face capacitances, one historical expression per channel.

Each channel keeps its own historical mean; no global default is changed:

* ``harmonic_face_mean`` -- unguarded ``2*a*b/(a+b)`` of adjacent nodal
  values: the Poisson face permittivity of the 1D MoL and 2D operators and the
  1D MoL carrier face diffusivities ``D_n``/``D_p``. Two zero neighbours give
  0/0 = NaN; that is the historical carrier behaviour (LD-03) and stays M.
* ``guarded_harmonic_face_mean`` -- the same mean with a zero face kept at
  zero when both sides vanish. Historically used only for the mobile-ion face
  diffusivities (both species) and for the face permittivity of the
  displacement-current and impedance/defect-transient observers; never for the
  carrier ``D_n``/``D_p`` faces (substituting it there is an N-class change).
* ``series_face_capacitance`` -- two-sided geometries, where the true plane
  splits a face into distances ``d_L``/``d_R``: ``eps0 / (d_L/eps_L + d_R/eps_R)``.
  At the face midpoint it equals the harmonic mean over the face spacing.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "DIELECTRIC_RULES", "LAYER_SERIES_CAPACITANCE_V2", "NODAL_HARMONIC_DIELECTRIC_V1",
    "VACUUM_PERMITTIVITY_F_M", "arithmetic_face_mean", "guarded_harmonic_face_mean",
    "harmonic_face_mean", "harmonic_mean", "series_face_capacitance",
    "subface_series_capacitance",
]

# Historical binary64 value of the legacy engine (CODATA 2006). It differs from
# the SI 2019 recommended value in the 10th digit; changing it is a P-class
# correction, not part of this behaviour-preserving migration.
VACUUM_PERMITTIVITY_F_M = 8.854187817e-12

NODAL_HARMONIC_DIELECTRIC_V1 = "nodal_harmonic_v1"
LAYER_SERIES_CAPACITANCE_V2 = "layer_series_capacitance_v2"
DIELECTRIC_RULES = (NODAL_HARMONIC_DIELECTRIC_V1, LAYER_SERIES_CAPACITANCE_V2)


def harmonic_mean(a: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    result: NDArray[np.float64] = 2.0 * left * right / (left + right)
    return result


def harmonic_face_mean(values: ArrayLike) -> NDArray[np.float64]:
    """``2 v_i v_{i+1} / (v_i + v_{i+1})`` on the N-1 faces of N nodal values.

    Unguarded: Poisson permittivity and MoL carrier ``D_n``/``D_p`` faces
    (0/0 = NaN between two zero nodes, LD-03).
    """
    nodal = np.asarray(values, dtype=float)
    result: NDArray[np.float64] = 2.0 * nodal[:-1] * nodal[1:] / (nodal[:-1] + nodal[1:])
    return result


def guarded_harmonic_face_mean(values: ArrayLike) -> NDArray[np.float64]:
    """Harmonic face mean with zero kept where both sides are zero.

    Historical channels: mobile-ion face diffusivities and the observer
    (displacement-current, impedance, defect-transient) face permittivity.
    Not the carrier ``D_n``/``D_p`` faces.
    """
    nodal = np.asarray(values, dtype=float)
    numer = 2.0 * nodal[:-1] * nodal[1:]
    denom = nodal[:-1] + nodal[1:]
    return np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0.0)


def arithmetic_face_mean(values: ArrayLike) -> NDArray[np.float64]:
    nodal = np.asarray(values, dtype=float)
    result: NDArray[np.float64] = 0.5 * (nodal[:-1] + nodal[1:])
    return result


def series_face_capacitance(
    left_distance_m: float,
    eps_r_left: float,
    right_distance_m: float,
    eps_r_right: float,
) -> float:
    """Exact series capacitance [F/m^2] of a face split by a dielectric plane."""
    values = (left_distance_m, eps_r_left, right_distance_m, eps_r_right)
    if not all(np.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("series capacitance needs positive distances and permittivities")
    return float(VACUUM_PERMITTIVITY_F_M / (
        float(left_distance_m) / float(eps_r_left)
        + float(right_distance_m) / float(eps_r_right)
    ))


def subface_series_capacitance(
    left_distance_m: ArrayLike,
    eps_r_left: ArrayLike,
    right_distance_m: ArrayLike,
    eps_r_right: ArrayLike,
) -> NDArray[np.float64]:
    """Sub-face material-extent series capacitance [F/m^2] (P08-02a).

    Each face of total spacing ``d_L + d_R > 0`` is partitioned at the true
    material plane into nonnegative left/right material extents ``d_L >= 0``
    and ``d_R >= 0`` with positive relative permittivities ``eps_L, eps_R``:

        C_f = eps_0 / (d_L / eps_L + d_R / eps_R).

    Unsplit faces (``d_L == 0`` or ``d_R == 0``) and homogeneous faces
    (``eps_L == eps_R``) evaluate directly as ``eps_0 * eps / (d_L + d_R)``
    so homogeneous layers agree with ``eps_0 * eps / h`` to 0-1 ULP.
    """
    d_left, e_left, d_right, e_right = np.broadcast_arrays(
        np.asarray(left_distance_m, dtype=float),
        np.asarray(eps_r_left, dtype=float),
        np.asarray(right_distance_m, dtype=float),
        np.asarray(eps_r_right, dtype=float),
    )
    if not (
        np.all(np.isfinite(d_left))
        and np.all(np.isfinite(e_left))
        and np.all(np.isfinite(d_right))
        and np.all(np.isfinite(e_right))
    ):
        raise ValueError("sub-face series capacitance requires finite distances and permittivities")
    total_distance = d_left + d_right
    if np.any(d_left < 0.0) or np.any(d_right < 0.0) or np.any(total_distance <= 0.0):
        raise ValueError("sub-face distances must be nonnegative with positive face spacing")
    if np.any(e_left <= 0.0) or np.any(e_right <= 0.0):
        raise ValueError("sub-face permittivities must be positive")
    out = np.empty(total_distance.shape, dtype=np.float64)
    left_only = d_right == 0.0
    right_only = (~left_only) & (d_left == 0.0)
    same_eps = (~left_only) & (~right_only) & (e_left == e_right)
    split = ~(left_only | right_only | same_eps)
    if np.any(left_only):
        out[left_only] = VACUUM_PERMITTIVITY_F_M * e_left[left_only] / d_left[left_only]
    if np.any(right_only):
        out[right_only] = VACUUM_PERMITTIVITY_F_M * e_right[right_only] / d_right[right_only]
    if np.any(same_eps):
        out[same_eps] = VACUUM_PERMITTIVITY_F_M * e_left[same_eps] / total_distance[same_eps]
    if np.any(split):
        out[split] = VACUUM_PERMITTIVITY_F_M / (
            d_left[split] / e_left[split] + d_right[split] / e_right[split]
        )
    return out
