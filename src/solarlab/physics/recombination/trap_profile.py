"""Position-dependent bulk trap density and the SRH lifetime it implies.

A layer's bulk SRH lifetime describes its cleanest region. With a trap
profile the trap density rises towards the layer faces,
``N_t(x) = N_t,bulk + (N_t,if - N_t,bulk) S(x)``, and the local lifetime
follows the inverse-density rule ``tau(x) = tau_bulk N_t,bulk / N_t(x)``.

``S`` is an exponential (``exp(-s/L_d)``) or Gaussian (``exp(-(s/sigma)^2)``)
kernel of the distance ``s`` to the selected face(s). For ``edge="both"``
the two face kernels are added and divided by the closed-form supremum
``max(1 + k(d), 2 k(d/2))`` so the profile saturates at ``N_t,if`` even in a
thin layer (2026-07 review F-05); a single face already peaks at 1. The
normalisation is exact for the exponential and under-estimates the
Gaussian peak by < 3 % only for ``sigma ~ 0.64 d``. These are build-time
parameter bindings, evaluated once per node set; they never see the state.

The kernels reproduce the historical arithmetic exactly, including the
``L_d <= 0`` "no profile" sentinel returning the bulk density.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Literal

import numpy as np

from solarlab.physics.recombination._validation import frozen_float_array

__all__ = [
    "TRAP_PROFILE_EDGES",
    "TRAP_PROFILE_SHAPES",
    "TrapProfile",
    "TrapProfileBinding",
    "exponential_edge_profile",
    "gaussian_edge_profile",
    "tau_from_trap_density",
]

TRAP_PROFILE_SHAPES = ("exponential", "gaussian")
TRAP_PROFILE_EDGES = ("both", "left", "right")
TrapProfileShape = Literal["exponential", "gaussian"]
TrapProfileEdge = Literal["both", "left", "right"]


def _combine_edge_kernel(
    d_left: np.ndarray,
    d_right: np.ndarray,
    edge: str,
    kernel: Callable[[Any], Any],
    thickness: float,
) -> np.ndarray:
    if edge not in TRAP_PROFILE_EDGES:
        raise ValueError(f"edge must be one of {TRAP_PROFILE_EDGES}, got {edge!r}")
    if edge == "left":
        return kernel(d_left)
    if edge == "right":
        return kernel(d_right)
    both = kernel(d_left) + kernel(d_right)
    # The sum is symmetric about the midpoint, so its extrema sit at the
    # faces or at the midpoint; both candidates are closed form, so the
    # normalisation is independent of the node set.
    peak = max(float(1.0 + kernel(thickness)), float(2.0 * kernel(0.5 * thickness)))
    return both / peak


def exponential_edge_profile(
    x_local: np.ndarray,
    thickness: float,
    N_t_interface: float,
    N_t_bulk: float,
    L_d: float,
    edge: str = "both",
) -> np.ndarray:
    """``N_t(x)`` [m^-3] for an exponential interface-trap distribution.

    ``x_local`` is the position inside the layer (0 at its left face) [m].
    """

    if L_d <= 0.0:
        return np.full_like(x_local, N_t_bulk)
    d_left = x_local
    d_right = thickness - x_local

    def kernel(d: Any) -> Any:
        return np.exp(-d / L_d)

    edge_contribution = _combine_edge_kernel(d_left, d_right, edge, kernel, thickness)
    return N_t_bulk + (N_t_interface - N_t_bulk) * edge_contribution


def gaussian_edge_profile(
    x_local: np.ndarray,
    thickness: float,
    N_t_interface: float,
    N_t_bulk: float,
    sigma: float,
    edge: str = "both",
) -> np.ndarray:
    """``N_t(x)`` [m^-3] for a Gaussian interface-trap distribution."""

    if sigma <= 0.0:
        return np.full_like(x_local, N_t_bulk)
    d_left = x_local
    d_right = thickness - x_local

    def kernel(d: Any) -> Any:
        return np.exp(-((d / sigma) ** 2))

    edge_contribution = _combine_edge_kernel(d_left, d_right, edge, kernel, thickness)
    return N_t_bulk + (N_t_interface - N_t_bulk) * edge_contribution


def tau_from_trap_density(tau_bulk: Any, N_t: Any, N_t_bulk: float) -> Any:
    """SRH inverse-density rule ``tau = tau_bulk N_t,bulk / max(N_t, N_t,bulk, 1)``.

    The floor keeps a passivation profile (``N_t,if < N_t,bulk``) physical
    and monotone instead of lengthening the lifetime above its bulk value.
    """

    N_t_arr = np.asarray(N_t, dtype=float)
    floor = max(float(N_t_bulk), 1.0)
    safe = np.maximum(N_t_arr, floor)
    return tau_bulk * float(N_t_bulk) / safe


@dataclass(frozen=True, slots=True)
class TrapProfile:
    """A layer's resolved trap-profile declaration (SI units).

    ``decay_length_m`` is ``L_d`` for the exponential and ``sigma`` for the
    Gaussian shape; ``0`` is the documented "no profile" sentinel.
    """

    interface_density_m3: float
    bulk_density_m3: float
    decay_length_m: float
    shape: TrapProfileShape = "exponential"
    edge: TrapProfileEdge = "both"

    def __post_init__(self) -> None:
        if self.shape not in TRAP_PROFILE_SHAPES:
            raise ValueError(f"trap profile shape must be one of {TRAP_PROFILE_SHAPES}")
        if self.edge not in TRAP_PROFILE_EDGES:
            raise ValueError(f"trap profile edge must be one of {TRAP_PROFILE_EDGES}")
        for name in ("interface_density_m3", "bulk_density_m3", "decay_length_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, float(value))

    def trap_density(self, x_local_m: np.ndarray, thickness_m: float) -> np.ndarray:
        profile = gaussian_edge_profile if self.shape == "gaussian" else exponential_edge_profile
        return profile(
            x_local_m,
            thickness_m,
            self.interface_density_m3,
            self.bulk_density_m3,
            self.decay_length_m,
            edge=self.edge,
        )

    def bind(
        self,
        x_local_m: np.ndarray,
        thickness_m: float,
        tau_n_bulk_s: np.ndarray | float,
        tau_p_bulk_s: np.ndarray | float,
    ) -> TrapProfileBinding:
        """Bind the profile to a layer's nodes and its bulk lifetimes."""

        x_local = np.asarray(x_local_m, dtype=float)
        if x_local.ndim != 1 or not np.all(np.isfinite(x_local)):
            raise ValueError("x_local_m must be a finite 1D array")
        if not math.isfinite(float(thickness_m)) or float(thickness_m) <= 0.0:
            raise ValueError("thickness_m must be finite and positive")
        density = self.trap_density(x_local, float(thickness_m))
        return TrapProfileBinding(
            profile=self,
            trap_density_m3=frozen_float_array(density, name="trap_density_m3"),
            tau_n_s=frozen_float_array(
                tau_from_trap_density(tau_n_bulk_s, density, self.bulk_density_m3), name="tau_n_s"
            ),
            tau_p_s=frozen_float_array(
                tau_from_trap_density(tau_p_bulk_s, density, self.bulk_density_m3), name="tau_p_s"
            ),
        )


@dataclass(frozen=True, slots=True)
class TrapProfileBinding:
    """Per-node trap density and the lifetimes that replace the bulk values."""

    profile: TrapProfile
    trap_density_m3: np.ndarray
    tau_n_s: np.ndarray
    tau_p_s: np.ndarray
