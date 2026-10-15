"""Static one-dimensional dopant profiles within an electrical layer (M class).

``N_A`` and ``N_D`` stay uniform unless a species declares a bulk asymptote.
For a profiled species the declared density is the value at the selected
layer edge and the bulk density is the deep-layer asymptote; the Gaussian
convention is

    N(d) = N_bulk + (N_edge - N_bulk) exp(-(d / L)^2),

with ``d`` the distance from the selected edge, so ``L`` is a 1/e distance.
Ionization statistics are not part of this module.

Validation and arithmetic are the historical ones (``perovskite_sim.physics.doping``),
including the error messages.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

__all__ = [
    "DOPING_EDGES",
    "DOPING_SHAPES",
    "DopingProfileSpec",
    "doping_at_position",
    "gaussian_doping_profile",
    "layer_doping_profiles",
]

DOPING_SHAPES = ("gaussian",)
DOPING_EDGES = ("front", "back")


@dataclass(frozen=True, slots=True)
class DopingProfileSpec:
    """A layer's dopant declaration [m^-3]; edge densities double as uniform values."""

    acceptor_density_m3: float
    donor_density_m3: float
    acceptor_bulk_density_m3: float | None = None
    donor_bulk_density_m3: float | None = None
    shape: str | None = None
    decay_length_m: float | None = None
    edge: str = "front"

    def __post_init__(self) -> None:
        if not self.profiled:
            if self.shape is not None or self.decay_length_m is not None:
                raise ValueError("doping profile shape/decay length requires N_A_bulk or N_D_bulk")
            return
        if self.shape not in DOPING_SHAPES:
            raise ValueError(f"doping_profile_shape must be one of {DOPING_SHAPES}, got {self.shape!r}")
        if str(self.edge) not in DOPING_EDGES:
            raise ValueError(f"doping_edge must be one of {DOPING_EDGES}, got {str(self.edge)!r}")
        try:
            length = float(self.decay_length_m)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError("doping_decay_length must be finite and positive") from exc
        if not math.isfinite(length) or length <= 0.0:
            raise ValueError("doping_decay_length must be finite and positive")
        for name, value in (
            ("N_A", self.acceptor_density_m3),
            ("N_D", self.donor_density_m3),
            ("N_A_bulk", self.acceptor_bulk_density_m3),
            ("N_D_bulk", self.donor_bulk_density_m3),
        ):
            if value is None:
                continue
            number = float(value)
            if not math.isfinite(number) or number < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

    @property
    def profiled(self) -> bool:
        """Whether either species has a bulk asymptote (historical presence rule)."""

        return self.acceptor_bulk_density_m3 is not None or self.donor_bulk_density_m3 is not None


def gaussian_doping_profile(
    x_local: np.ndarray,
    thickness: float,
    edge_density: float,
    bulk_density: float,
    decay_length: float,
    *,
    edge: str,
) -> np.ndarray:
    """One Gaussian dopant species on local layer coordinates [m^-3]."""

    positions = np.asarray(x_local, dtype=float)
    if edge == "front":
        distance = positions
    elif edge == "back":
        distance = float(thickness) - positions
    else:
        raise ValueError(f"doping edge must be one of {DOPING_EDGES}, got {edge!r}")
    kernel = np.exp(-((distance / float(decay_length)) ** 2))
    return float(bulk_density) + (float(edge_density) - float(bulk_density)) * kernel


def layer_doping_profiles(
    x_local: np.ndarray, thickness: float, spec: DopingProfileSpec
) -> tuple[np.ndarray, np.ndarray]:
    """Per-node ``(N_A, N_D)`` [m^-3] of one layer."""

    positions = np.asarray(x_local, dtype=float)
    acceptors = np.full_like(positions, float(spec.acceptor_density_m3))
    donors = np.full_like(positions, float(spec.donor_density_m3))
    if not spec.profiled:
        return acceptors, donors
    assert spec.decay_length_m is not None
    if spec.acceptor_bulk_density_m3 is not None:
        acceptors = gaussian_doping_profile(
            positions,
            float(thickness),
            float(spec.acceptor_density_m3),
            float(spec.acceptor_bulk_density_m3),
            float(spec.decay_length_m),
            edge=str(spec.edge),
        )
    if spec.donor_bulk_density_m3 is not None:
        donors = gaussian_doping_profile(
            positions,
            float(thickness),
            float(spec.donor_density_m3),
            float(spec.donor_bulk_density_m3),
            float(spec.decay_length_m),
            edge=str(spec.edge),
        )
    return acceptors, donors


def doping_at_position(spec: DopingProfileSpec, position: float, thickness: float) -> tuple[float, float]:
    """Scalar ``(N_A, N_D)`` at one layer position [m^-3]."""

    acceptors, donors = layer_doping_profiles(np.asarray([float(position)]), float(thickness), spec)
    return float(acceptors[0]), float(donors[0])
