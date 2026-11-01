"""Monochromatic Beer-Lambert generation integrated per control volume.

The primary quantity is each cell's EXACT absorbed photon count
[m^-2 s^-1] between its two faces ``b_i < b_{i+1}``::

    absorbed[i] = Phi (exp(-tau(b_i)) - exp(-tau(b_{i+1}))),

which telescopes to ``Phi (1 - exp(-tau_total))`` on ANY mesh. The cell
source is ``absorbed / w`` with ``w`` the geometry's cell measure (see
``profile.py`` for the historical integer-endpoint contact-cell convention).

For the historical integer- and half-endpoint channels, optical depth is
the mid-face trapezoid of the per-node
absorption coefficient (scalar ``alpha``: ``alpha * x``) and is therefore
piecewise linear between nodes. At a face it is evaluated exactly on that
linear interpolant; the historical faces keep their historical expressions
bit for bit -- the node-span ends take the end values and an interval
midpoint takes the mean of its two node values -- so the integer- and
half-endpoint geometries reproduce the historical arrays exactly.

The two-sided channels instead integrate each material's piecewise-constant
absorption on its true extent. Their interface planes and one-sided nodal
coefficients must identify every layer; varying absorption within a layer is
not inferred from incomplete interface traces. This is an N correction, while
the historical nodal optical-depth helpers retain their original arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any

import numpy as np

from solarlab.mesh.geometry import (
    PHYSICAL_BOUNDARY_TWO_SIDED_V1,
    TWO_SIDED_TRACE_V1,
    InterfacePlane,
)
from solarlab.physics.generation._validation import array_digest
from solarlab.physics.generation.profile import (
    ControlVolumes,
    GenerationProfile,
    StaleGenerationProfileError,
    control_volume_digest,
    require_control_volumes,
)

__all__ = [
    "BeerLambertBinding",
    "beer_lambert_absorbed_photon_flux",
    "beer_lambert_generation",
    "beer_lambert_optical_depth",
    "face_optical_depth",
]

_TWO_SIDED_GEOMETRIES = frozenset(
    (TWO_SIDED_TRACE_V1, PHYSICAL_BOUNDARY_TWO_SIDED_V1)
)


def _require_absorption(x: np.ndarray, alpha_m1: Any) -> np.ndarray:
    alpha = np.asarray(alpha_m1, dtype=float)
    if alpha.ndim > 1 or (alpha.ndim == 1 and alpha.shape != x.shape):
        raise ValueError(f"alpha must be scalar or shape {x.shape}, got {alpha.shape}")
    if not np.all(np.isfinite(alpha)) or np.any(alpha < 0.0):
        raise ValueError("alpha must be finite and non-negative")
    return alpha


def _require_photon_flux(photon_flux_m2_s: float) -> float:
    phi = float(photon_flux_m2_s)
    if not math.isfinite(phi) or phi < 0.0:
        raise ValueError("photon_flux_m2_s must be finite and non-negative")
    return phi


def beer_lambert_optical_depth(x_nodes_m: np.ndarray, alpha_m1: Any) -> np.ndarray:
    """Optical depth ``tau(x_i)`` from the first node (mid-face trapezoid)."""

    x = np.asarray(x_nodes_m, dtype=float)
    alpha = np.asarray(alpha_m1, dtype=float)
    if alpha.ndim == 0:
        return float(alpha) * x
    dx = np.diff(x)
    alpha_mid = 0.5 * (alpha[:-1] + alpha[1:])
    cum = np.zeros_like(x)
    cum[1:] = np.cumsum(alpha_mid * dx)
    return cum


def face_optical_depth(x_nodes_m: np.ndarray, alpha_m1: Any, faces_m: np.ndarray) -> np.ndarray:
    """Historical optical depth on the nodal linear interpolant, at each face."""

    x = np.asarray(x_nodes_m, dtype=float)
    faces = np.asarray(faces_m, dtype=float)
    alpha = np.asarray(alpha_m1, dtype=float)
    cum = beer_lambert_optical_depth(x, alpha)
    interval = np.clip(np.searchsorted(x, faces, side="right") - 1, 0, x.size - 2)
    if alpha.ndim == 0:
        slope = np.full(x.size - 1, float(alpha))
    else:
        slope = 0.5 * (alpha[:-1] + alpha[1:])
    tau = cum[interval] + (faces - x[interval]) * slope[interval]
    midpoint = faces == 0.5 * (x[interval] + x[interval + 1])
    tau[midpoint] = 0.5 * (cum[interval[midpoint]] + cum[interval[midpoint] + 1])
    tau[faces == x[0]] = cum[0]
    tau[faces == x[-1]] = cum[-1]
    return tau


def _two_sided_absorbed_photon_flux(
    control_volumes: ControlVolumes, alpha: np.ndarray, photon_flux_m2_s: float
) -> np.ndarray:
    """Integrate constant material coefficients on true layer/cell extents."""
    nodes, faces, _ = require_control_volumes(control_volumes)
    if alpha.ndim == 0 or np.all(alpha == alpha[0]):
        coefficient = float(alpha) if alpha.ndim == 0 else float(alpha[0])
        cell_alpha = np.full(nodes.size, coefficient)
        depth_left = coefficient * (faces[:-1] - nodes[0])
    else:
        planes = getattr(control_volumes, "interfaces", None)
        if planes is None or any(not isinstance(plane, InterfacePlane) for plane in planes):
            raise ValueError(
                "two-sided Beer-Lambert requires explicit interface planes "
                "for nonuniform absorption"
            )
        positions = np.asarray([plane.position_m for plane in planes], dtype=float)
        if (
            not np.all(np.isfinite(positions))
            or np.any(np.diff(positions) <= 0.0)
            or np.any(positions <= nodes[0])
            or np.any(positions >= nodes[-1])
            or np.any(np.isin(nodes, positions))
            or not np.all(np.isin(positions, faces))
        ):
            raise ValueError(
                "two-sided Beer-Lambert requires strictly interior interface "
                "planes on cell faces, without shared nodes"
            )
        boundaries = np.concatenate(([nodes[0]], positions, [nodes[-1]]))
        owner = np.searchsorted(positions, nodes, side="right")
        layer_alpha = np.empty(boundaries.size - 1)
        for layer in range(layer_alpha.size):
            values = alpha[owner == layer]
            if values.size == 0 or not np.all(values == values[0]):
                raise ValueError(
                    "two-sided Beer-Lambert requires piecewise-constant "
                    "absorption in each layer; varying or missing one-sided "
                    "layer coefficients are not supported"
                )
            layer_alpha[layer] = values[0]
        depth_front = np.concatenate(
            ([0.0], np.cumsum(layer_alpha * np.diff(boundaries)))
        )
        cell_owner = np.searchsorted(positions, faces[:-1], side="right")
        cell_alpha = layer_alpha[cell_owner]
        depth_left = (
            depth_front[cell_owner]
            + cell_alpha * (faces[:-1] - boundaries[cell_owner])
        )
    # Each cell uses its actual extent, never its carrier storage weight.
    # expm1 retains thin-cell absorption without subtracting transmissions.
    depth_increment = cell_alpha * np.diff(faces)
    return photon_flux_m2_s * np.exp(-depth_left) * -np.expm1(-depth_increment)


def beer_lambert_absorbed_photon_flux(
    control_volumes: ControlVolumes, alpha_m1: Any, photon_flux_m2_s: float
) -> np.ndarray:
    """Exact absorbed photons of each control volume [m^-2 s^-1], shape ``(N,)``.

    ``alpha_m1`` is a scalar or one value per node [m^-1]; ``photon_flux_m2_s``
    is the above-gap photon flux entering at the first node.

    Named two-sided channels interpret nodal coefficients as one-sided
    piecewise-constant material values. They require explicit interface planes
    and reject within-layer variation whose optical traces are unspecified.
    """

    nodes, faces, _ = require_control_volumes(control_volumes)
    alpha = _require_absorption(nodes, alpha_m1)
    phi = _require_photon_flux(photon_flux_m2_s)
    if control_volumes.name in _TWO_SIDED_GEOMETRIES:
        absorbed = _two_sided_absorbed_photon_flux(control_volumes, alpha, phi)
    else:
        tau_face = face_optical_depth(nodes, alpha, faces)
        absorbed = phi * (np.exp(-tau_face[:-1]) - np.exp(-tau_face[1:]))
    absorbed.setflags(write=False)
    return absorbed


def beer_lambert_generation(control_volumes: ControlVolumes, alpha_m1: Any, photon_flux_m2_s: float) -> np.ndarray:
    """Photon-conserving Beer-Lambert cell source ``absorbed / w`` [m^-3 s^-1]."""

    _, _, measures = require_control_volumes(control_volumes)
    generation = beer_lambert_absorbed_photon_flux(control_volumes, alpha_m1, photon_flux_m2_s) / measures
    generation.setflags(write=False)
    return generation


@dataclass(frozen=True)
class BeerLambertBinding:
    """Frozen monochromatic Beer-Lambert inputs on one node set."""

    x_nodes_m: np.ndarray
    alpha_m1: np.ndarray
    photon_flux_m2_s: float
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        x = np.array(self.x_nodes_m, dtype=float, copy=True)
        if x.ndim != 1 or x.size < 2 or not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0.0):
            raise ValueError("x_nodes_m must be finite and strictly increasing with >= 2 nodes")
        alpha = np.array(_require_absorption(x, self.alpha_m1), copy=True)
        phi = _require_photon_flux(self.photon_flux_m2_s)
        x.setflags(write=False)
        alpha.setflags(write=False)
        object.__setattr__(self, "x_nodes_m", x)
        object.__setattr__(self, "alpha_m1", alpha)
        object.__setattr__(self, "photon_flux_m2_s", phi)
        header = json.dumps(
            {"model": "beer_lambert_monochromatic_v1", "photon_flux_m2_s": repr(phi)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        object.__setattr__(self, "sha256", hashlib.sha256(header + array_digest(x, alpha).encode("ascii")).hexdigest())

    def generation_profile(self, control_volumes: ControlVolumes) -> GenerationProfile:
        nodes, _, measures = require_control_volumes(control_volumes)
        if nodes.tobytes() != self.x_nodes_m.tobytes():
            raise StaleGenerationProfileError("control volumes are not on this binding's nodes")
        absorbed = beer_lambert_absorbed_photon_flux(control_volumes, self.alpha_m1, self.photon_flux_m2_s)
        return GenerationProfile(
            method="beer_lambert",
            geometry_name=str(control_volumes.name),
            source_m3_s=absorbed / measures,
            absorbed_m2_s=absorbed,
            binding_sha256=self.sha256,
            control_volume_sha256=control_volume_digest(control_volumes),
        )
