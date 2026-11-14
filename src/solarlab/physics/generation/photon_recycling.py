"""Escape-probability kernels and optical bindings for radiative photon recycling.

Models internal photon trapping inside high-index radiative absorbers via two
explicitly labeled escape-probability formulations:

* ``YABLONOVITCH_INTRINSIC_ALPHA_D_V1``: single-pass optical depth at the band
  edge ``OD_gap = alpha(lambda_gap) * d_absorber`` from intrinsic optical
  constants ``(n_gap, k_gap)``, yielding ``P_esc = clip(1 / (4 n_gap^2 OD_gap), 0, 1)``.
* ``TMM_INTEGRATED_ABSORBANCE_TRAPEZOID_V1``: historical stack absorbance
  integral ``OD_gap = trapezoid(A(x, lambda_gap), x_absorber)``.
* ``TMM_INTEGRATED_ABSORBANCE_CONTROL_VOLUME_V1``: discrete control-volume
  absorbance integral ``OD_gap = sum(w_absorber * A(x_absorber, lambda_gap))``.

No mode silently swaps trapezoid and control-volume weights or modifies
non-absorber layers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from solarlab.contracts.arrays import frozen_array
from solarlab.contracts.units import ELEMENTARY_CHARGE_C
from solarlab.physics.generation.optical_binding import OpticalStackBinding

__all__ = [
    "C_LIGHT_M_S",
    "ESCAPE_PROBABILITY_MODES",
    "HC_J_M",
    "H_PLANCK_J_S",
    "TMM_INTEGRATED_ABSORBANCE_CONTROL_VOLUME_V1",
    "TMM_INTEGRATED_ABSORBANCE_TRAPEZOID_V1",
    "YABLONOVITCH_INTRINSIC_ALPHA_D_V1",
    "AbsorberEscapeBinding",
    "AbsorberEscapeInput",
    "PhotonRecyclingCapabilityError",
    "PhotonRecyclingError",
    "bind_absorber_escape_from_optical_stack",
    "compute_p_esc",
    "compute_p_esc_for_absorber",
    "effective_radiative_coefficient",
    "wavelength_at_gap",
]

Vector: TypeAlias = NDArray[np.float64]

H_PLANCK_J_S: float = 6.62607015e-34
C_LIGHT_M_S: float = 2.99792458e8
HC_J_M: float = H_PLANCK_J_S * C_LIGHT_M_S
Q: float = ELEMENTARY_CHARGE_C

YABLONOVITCH_INTRINSIC_ALPHA_D_V1 = "yablonovitch_intrinsic_alpha_d_v1"
TMM_INTEGRATED_ABSORBANCE_TRAPEZOID_V1 = "tmm_integrated_absorbance_trapezoid_v1"
TMM_INTEGRATED_ABSORBANCE_CONTROL_VOLUME_V1 = "tmm_integrated_absorbance_control_volume_v1"
ESCAPE_PROBABILITY_MODES: tuple[str, ...] = (
    YABLONOVITCH_INTRINSIC_ALPHA_D_V1,
    TMM_INTEGRATED_ABSORBANCE_TRAPEZOID_V1,
    TMM_INTEGRATED_ABSORBANCE_CONTROL_VOLUME_V1,
)


class PhotonRecyclingError(ValueError):
    """Invalid input to an escape-probability or photon-recycling kernel."""


class PhotonRecyclingCapabilityError(ValueError):
    """Requested photon-recycling mode or AC linearization exceeds capability."""


def wavelength_at_gap(Eg_eV: float) -> float:
    """Band-edge photon wavelength ``hc / (q Eg)`` in metres."""
    gap = float(Eg_eV)
    if not math.isfinite(gap) or gap <= 0.0:
        raise PhotonRecyclingError(f"Eg_eV must be finite and positive, got {Eg_eV!r}")
    return HC_J_M / (gap * Q)


def compute_p_esc(alpha_gap: float, thickness: float, n_at_gap: float) -> float:
    """Yablonovitch single-pass escape probability ``clip(1 / (4 n^2 alpha d), 0, 1)``.

    Transparent, zero-thickness, non-positive-index, or non-finite inputs return
    ``1.0`` (every internally emitted photon escapes without reabsorption).
    """
    alpha = float(alpha_gap)
    width = float(thickness)
    index = float(n_at_gap)
    if (
        not (math.isfinite(alpha) and math.isfinite(width) and math.isfinite(index))
        or alpha <= 0.0
        or width <= 0.0
        or index <= 0.0
    ):
        return 1.0
    optical_depth = alpha * width
    if optical_depth <= 0.0:
        return 1.0
    p_esc = 1.0 / (4.0 * index * index * optical_depth)
    return float(np.clip(p_esc, 0.0, 1.0))


def compute_p_esc_for_absorber(
    A_xl: ArrayLike,
    wavelengths_m: ArrayLike,
    x_absorber: ArrayLike,
    Eg_eV: float,
    n_at_gap: float,
    *,
    weights_m: ArrayLike | None = None,
    mode: str = TMM_INTEGRATED_ABSORBANCE_TRAPEZOID_V1,
) -> float:
    """Escape probability from a TMM local absorbance profile ``A(x, lambda)`` [m^-1].

    ``mode`` selects the declared spatial quadrature explicitly:
    ``TMM_INTEGRATED_ABSORBANCE_TRAPEZOID_V1`` integrates ``A(x, lambda_gap)``
    with ``np.trapezoid`` over ``x_absorber``, whereas
    ``TMM_INTEGRATED_ABSORBANCE_CONTROL_VOLUME_V1`` integrates with supplied
    discrete control-volume measures ``weights_m``.
    """
    if mode not in (
        TMM_INTEGRATED_ABSORBANCE_TRAPEZOID_V1,
        TMM_INTEGRATED_ABSORBANCE_CONTROL_VOLUME_V1,
    ):
        raise PhotonRecyclingCapabilityError(
            f"unsupported absorbance escape mode {mode!r}; expected trapezoid or control-volume"
        )
    absorbance = np.asarray(A_xl, dtype=float)
    wl = np.asarray(wavelengths_m, dtype=float)
    x_abs = np.asarray(x_absorber, dtype=float)
    index = float(n_at_gap)
    if absorbance.ndim != 2 or wl.ndim != 1 or x_abs.ndim != 1:
        raise PhotonRecyclingError("A_xl must be 2D (N_x, N_wl) with 1D wavelengths_m and x_absorber")
    if absorbance.shape != (x_abs.size, wl.size):
        raise PhotonRecyclingError(
            f"A_xl shape {absorbance.shape} does not match ({x_abs.size}, {wl.size})"
        )
    if wl.size == 0 or x_abs.size < 2 or not math.isfinite(index) or index <= 0.0:
        return 1.0
    if not (np.all(np.isfinite(absorbance)) and np.all(np.isfinite(wl)) and np.all(np.isfinite(x_abs))):
        raise PhotonRecyclingError("A_xl, wavelengths_m, and x_absorber must be finite")
    if np.any(np.diff(wl) <= 0.0):
        raise PhotonRecyclingError("wavelengths_m must be strictly increasing")
    if np.any(np.diff(x_abs) <= 0.0):
        raise PhotonRecyclingError("x_absorber must be strictly increasing")

    lam_gap = float(np.clip(wavelength_at_gap(Eg_eV), float(wl[0]), float(wl[-1])))
    right = int(np.searchsorted(wl, lam_gap))
    if right == 0:
        a_gap = absorbance[:, 0]
    elif right >= wl.size:
        a_gap = absorbance[:, -1]
    else:
        left = right - 1
        weight = (lam_gap - float(wl[left])) / (float(wl[right]) - float(wl[left]))
        a_gap = (1.0 - weight) * absorbance[:, left] + weight * absorbance[:, right]

    if mode == TMM_INTEGRATED_ABSORBANCE_TRAPEZOID_V1:
        if weights_m is not None:
            raise PhotonRecyclingError("weights_m cannot be passed in trapezoid absorbance mode")
        od_gap = float(np.trapezoid(a_gap, x_abs))
    else:
        if weights_m is None:
            raise PhotonRecyclingError("control-volume absorbance mode requires explicit weights_m")
        w = np.asarray(weights_m, dtype=float)
        if w.shape != x_abs.shape or not np.all(np.isfinite(w)) or np.any(w <= 0.0):
            raise PhotonRecyclingError("weights_m must be finite positive and match x_absorber")
        od_gap = float(np.dot(w, a_gap))

    if od_gap <= 0.0:
        return 1.0
    p_esc = 1.0 / (4.0 * index * index * od_gap)
    return float(np.clip(p_esc, 0.0, 1.0))


def effective_radiative_coefficient(
    B_rad_m3_s: ArrayLike,
    absorber_masks: Sequence[ArrayLike],
    absorber_p_esc: Sequence[float],
) -> Vector:
    """Local escape-scaled radiative coefficient ``B_rad_eff = B_rad * P_esc``.

    When ``absorber_masks`` is empty or every ``P_esc == 1.0``, returns a copy
    bit-identical to ``B_rad_m3_s``.
    """
    base = np.array(B_rad_m3_s, dtype=float, copy=True)
    if base.ndim != 1 or not np.all(np.isfinite(base)) or np.any(base < 0.0):
        raise PhotonRecyclingError("B_rad_m3_s must be a finite non-negative 1D array")
    if len(absorber_masks) != len(absorber_p_esc):
        raise PhotonRecyclingError("absorber_masks and absorber_p_esc must have equal length")
    for raw_mask, raw_p_esc in zip(absorber_masks, absorber_p_esc):
        mask = np.asarray(raw_mask, dtype=bool)
        if mask.shape != base.shape:
            raise PhotonRecyclingError("absorber mask shape must match B_rad_m3_s")
        p_esc = float(raw_p_esc)
        if not math.isfinite(p_esc) or p_esc < 0.0 or p_esc > 1.0:
            raise PhotonRecyclingError(f"P_esc must lie in [0, 1], got {raw_p_esc!r}")
        if p_esc != 1.0:
            base[mask] = base[mask] * p_esc
    return base


@dataclass(frozen=True)
class AbsorberEscapeInput:
    """Electrical absorber metadata used to derive band-edge optical escape."""

    layer_id: str
    thickness_m: float
    band_gap_front_eV: float
    band_gap_back_eV: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.layer_id, str) or not self.layer_id.strip():
            raise PhotonRecyclingError("absorber escape input requires a non-empty layer_id")
        thickness = float(self.thickness_m)
        gap_front = float(self.band_gap_front_eV)
        if not math.isfinite(thickness) or thickness <= 0.0:
            raise PhotonRecyclingError(f"{self.layer_id}: thickness_m must be finite positive")
        if not math.isfinite(gap_front) or gap_front <= 0.0:
            raise PhotonRecyclingError(f"{self.layer_id}: band_gap_front_eV must be finite positive")
        object.__setattr__(self, "thickness_m", thickness)
        object.__setattr__(self, "band_gap_front_eV", gap_front)
        if self.band_gap_back_eV is not None:
            gap_back = float(self.band_gap_back_eV)
            if not math.isfinite(gap_back) or gap_back <= 0.0:
                raise PhotonRecyclingError(f"{self.layer_id}: band_gap_back_eV must be finite positive")
            object.__setattr__(self, "band_gap_back_eV", gap_back)

    @property
    def effective_emission_gap_eV(self) -> float:
        """Emission concentrates near the minimum band gap of a graded absorber."""
        if self.band_gap_back_eV is None:
            return self.band_gap_front_eV
        return min(self.band_gap_front_eV, self.band_gap_back_eV)


@dataclass(frozen=True)
class AbsorberEscapeBinding:
    """Resolved optical escape-probability binding for one absorber layer."""

    layer_id: str
    mode: str
    band_gap_eV: float
    wavelength_gap_m: float
    refractive_index_at_gap: float
    extinction_at_gap: float
    alpha_gap_m1: float
    thickness_m: float
    optical_depth: float
    escape_probability: float
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.mode not in ESCAPE_PROBABILITY_MODES:
            raise PhotonRecyclingCapabilityError(f"unknown escape mode {self.mode!r}")
        p_esc = float(self.escape_probability)
        if not math.isfinite(p_esc) or p_esc < 0.0 or p_esc > 1.0:
            raise PhotonRecyclingError("escape_probability must lie in [0, 1]")
        payload = {
            "schema": "solarlab.photon_recycling.escape_binding.v1",
            "layer_id": self.layer_id,
            "mode": self.mode,
            "band_gap_eV": repr(float(self.band_gap_eV)),
            "wavelength_gap_m": repr(float(self.wavelength_gap_m)),
            "refractive_index_at_gap": repr(float(self.refractive_index_at_gap)),
            "extinction_at_gap": repr(float(self.extinction_at_gap)),
            "alpha_gap_m1": repr(float(self.alpha_gap_m1)),
            "thickness_m": repr(float(self.thickness_m)),
            "optical_depth": repr(float(self.optical_depth)),
            "escape_probability": repr(p_esc),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()
        object.__setattr__(self, "sha256", digest)


def bind_absorber_escape_from_optical_stack(
    optical_stack: OpticalStackBinding,
    absorbers: Sequence[AbsorberEscapeInput],
    *,
    mode: str = YABLONOVITCH_INTRINSIC_ALPHA_D_V1,
    electrical_nodes_m: ArrayLike | None = None,
    absorber_node_masks: Sequence[ArrayLike] | None = None,
    absorber_weights_m: Sequence[ArrayLike] | None = None,
) -> tuple[AbsorberEscapeBinding, ...]:
    """Derive per-absorber escape probabilities from a frozen TMM optical stack."""
    if not isinstance(optical_stack, OpticalStackBinding):
        raise TypeError("optical_stack must be an OpticalStackBinding")
    if mode not in ESCAPE_PROBABILITY_MODES:
        raise PhotonRecyclingCapabilityError(f"unsupported escape mode {mode!r}")
    if not absorbers:
        return ()

    wl_m = optical_stack.wavelengths_m
    slice_lookup = dict(optical_stack.physical_layer_slices)
    absorbance_xl: Vector | None = None
    nodes: Vector | None = None
    if mode != YABLONOVITCH_INTRINSIC_ALPHA_D_V1:
        if electrical_nodes_m is None or absorber_node_masks is None:
            raise PhotonRecyclingError(
                f"{mode} requires electrical_nodes_m and absorber_node_masks"
            )
        if len(absorber_node_masks) != len(absorbers):
            raise PhotonRecyclingError("absorber_node_masks must match absorbers")
        if mode == TMM_INTEGRATED_ABSORBANCE_CONTROL_VOLUME_V1:
            if absorber_weights_m is None or len(absorber_weights_m) != len(absorbers):
                raise PhotonRecyclingError(
                    "control-volume absorbance escape mode requires absorber_weights_m per absorber"
                )
        nodes = frozen_array(electrical_nodes_m)
        optical_x = nodes + optical_stack.electrical_offset_m
        from solarlab.physics.generation.transfer_matrix import tmm_absorption_profile

        absorbance_xl = tmm_absorption_profile(
            list(optical_stack.tmm_layers),
            wl_m,
            optical_x,
            optical_stack.boundaries_m,
            optical_stack.ambient_index,
            optical_stack.exit_index,
        )

    bindings: list[AbsorberEscapeBinding] = []
    for idx, absorber in enumerate(absorbers):
        if absorber.layer_id not in slice_lookup:
            raise PhotonRecyclingCapabilityError(
                f"absorber {absorber.layer_id!r} is missing from OpticalStackBinding"
            )
        layer_slice = slice_lookup[absorber.layer_id]
        sublayers = optical_stack.layers[layer_slice]
        if any(not sub.electrical for sub in sublayers):
            raise PhotonRecyclingCapabilityError(
                f"absorber {absorber.layer_id!r} must be an electrical layer"
            )
        weights = np.array([sub.thickness_m for sub in sublayers], dtype=float)
        total_w = float(np.sum(weights))
        n_curve = np.sum([w * sub.n for w, sub in zip(weights, sublayers)], axis=0) / total_w
        k_curve = np.sum([w * sub.k for w, sub in zip(weights, sublayers)], axis=0) / total_w

        eg_eff = absorber.effective_emission_gap_eV
        lam_gap_raw = wavelength_at_gap(eg_eff)
        lam_gap = float(np.clip(lam_gap_raw, float(wl_m[0]), float(wl_m[-1])))
        n_gap = float(np.interp(lam_gap, wl_m, n_curve))
        k_gap = float(np.interp(lam_gap, wl_m, k_curve))
        alpha_gap = 4.0 * math.pi * k_gap / lam_gap if lam_gap > 0.0 else 0.0

        if mode == YABLONOVITCH_INTRINSIC_ALPHA_D_V1:
            od_gap = max(0.0, alpha_gap * absorber.thickness_m)
            p_esc = compute_p_esc(alpha_gap, absorber.thickness_m, n_gap)
        else:
            assert absorbance_xl is not None and nodes is not None and absorber_node_masks is not None
            mask = np.asarray(absorber_node_masks[idx], dtype=bool)
            x_abs = nodes[mask]
            w_abs = None if absorber_weights_m is None else absorber_weights_m[idx]
            p_esc = compute_p_esc_for_absorber(
                absorbance_xl[mask, :],
                wl_m,
                x_abs,
                eg_eff,
                n_gap,
                weights_m=w_abs,
                mode=mode,
            )
            if mode == TMM_INTEGRATED_ABSORBANCE_TRAPEZOID_V1:
                a_interp = np.array(
                    [np.interp(lam_gap, wl_m, absorbance_xl[row, :]) for row in np.flatnonzero(mask)],
                    dtype=float,
                )
                od_gap = max(0.0, float(np.trapezoid(a_interp, x_abs))) if x_abs.size >= 2 else 0.0
            else:
                assert w_abs is not None
                a_interp = np.array(
                    [np.interp(lam_gap, wl_m, absorbance_xl[row, :]) for row in np.flatnonzero(mask)],
                    dtype=float,
                )
                od_gap = max(0.0, float(np.dot(np.asarray(w_abs, dtype=float), a_interp)))

        bindings.append(
            AbsorberEscapeBinding(
                layer_id=absorber.layer_id,
                mode=mode,
                band_gap_eV=eg_eff,
                wavelength_gap_m=lam_gap,
                refractive_index_at_gap=n_gap,
                extinction_at_gap=k_gap,
                alpha_gap_m1=alpha_gap,
                thickness_m=absorber.thickness_m,
                optical_depth=od_gap,
                escape_probability=p_esc,
            )
        )
    return tuple(bindings)
