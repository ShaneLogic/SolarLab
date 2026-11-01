"""Generation profiles bound to the inputs and control volumes that produced them.

Control volumes come from a named geometry (W3a ``solarlab.mesh.geometry``;
e.g. ``legacy_mol_integer_endpoint_v1``): its cell faces bound each cell's
geometric extent and its cell measures are the weights the conservation
law multiplies. Generation is integrated over the faces and divided by the
measures, so ``sum(G * measures)`` is the exact absorbed photon count for
EVERY named geometry. Where a geometry's contact-cell measure is not its
geometric extent (the historical 1D integer endpoint: full ``dx`` against a
``dx/2`` extent), the outer-node G is the cell-integrated count spread over
that wider weight -- a historical convention recorded by the geometry name
on every profile, never silently converted.

A profile also records the identity of its optical inputs. Reusing it after
any input change (thickness, n-k, temperature/composition declaration,
spectrum, mesh, geometry) is rejected instead of returning a stale source.

Interface switch point (S3): ``ControlVolumes`` is the structural subset of
``solarlab.mesh.geometry.ControlVolumeGeometry1D`` this slice reads; the
annotation becomes that class once W3a's mesh package is integrated.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Protocol

import numpy as np

from solarlab.physics.generation._validation import array_digest

__all__ = [
    "ControlVolumes",
    "GENERATION_SOURCE_UNIT",
    "GenerationProfile",
    "StaleGenerationProfileError",
    "control_volume_digest",
    "require_control_volumes",
]

#: Unweighted cell source; the assembly multiplies by the cell measure once.
GENERATION_SOURCE_UNIT = "m^-3 s^-1"


class StaleGenerationProfileError(ValueError):
    """A generation profile was offered for inputs it was not computed from."""


class ControlVolumes(Protocol):
    """Named node-centred control volumes on a 1D node set (unit cross-section)."""

    @property
    def name(self) -> str: ...

    @property
    def nodes_m(self) -> np.ndarray: ...

    @property
    def cell_faces_m(self) -> np.ndarray: ...

    @property
    def cell_measures_m(self) -> np.ndarray: ...


def require_control_volumes(control_volumes: ControlVolumes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(nodes, faces, measures)`` after the checks every integration relies on."""

    name = getattr(control_volumes, "name", None)
    if not isinstance(name, str) or not name:
        raise TypeError("control volumes must carry their geometry name")
    nodes = np.asarray(control_volumes.nodes_m, dtype=float)
    faces = np.asarray(control_volumes.cell_faces_m, dtype=float)
    measures = np.asarray(control_volumes.cell_measures_m, dtype=float)
    if (
        nodes.ndim != 1
        or nodes.size < 2
        or not np.all(np.isfinite(nodes))
        or np.any(np.diff(nodes) <= 0.0)
    ):
        raise ValueError("control-volume nodes must be finite and strictly increasing")
    if faces.shape != (nodes.size + 1,) or not np.all(np.isfinite(faces)) or np.any(np.diff(faces) <= 0.0):
        raise ValueError("control volumes need N+1 finite strictly increasing faces")
    if faces[0] < nodes[0] or faces[-1] > nodes[-1]:
        raise ValueError("cell faces must lie inside the node span")
    if measures.shape != nodes.shape or not np.all(np.isfinite(measures)) or np.any(measures <= 0.0):
        raise ValueError("cell measures must be finite, positive and one per node")
    return nodes, faces, measures


def control_volume_digest(control_volumes: ControlVolumes) -> str:
    nodes, faces, measures = require_control_volumes(control_volumes)
    digest = hashlib.sha256(str(control_volumes.name).encode("utf-8"))
    digest.update(array_digest(nodes, faces, measures).encode("ascii"))
    return digest.hexdigest()


@dataclass(frozen=True)
class GenerationProfile:
    """Cell-averaged generation and the exact absorbed photons of each cell.

    ``source_m3_s * cell_measures == absorbed_m2_s`` cell by cell (to one
    rounding), so the assembled areal generation is ``sum(absorbed_m2_s)``.
    """

    method: str
    geometry_name: str
    source_m3_s: np.ndarray
    absorbed_m2_s: np.ndarray
    binding_sha256: str
    control_volume_sha256: str
    unit: str = GENERATION_SOURCE_UNIT

    def __post_init__(self) -> None:
        for name in ("source_m3_s", "absorbed_m2_s"):
            array = np.array(getattr(self, name), dtype=float, copy=True)
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @property
    def areal_generation_m2_s(self) -> float:
        return float(np.sum(self.absorbed_m2_s))

    def require_inputs(self, binding_sha256: str, control_volumes: ControlVolumes) -> None:
        if binding_sha256 != self.binding_sha256:
            raise StaleGenerationProfileError(
                "generation profile was computed from different optical inputs"
            )
        if control_volume_digest(control_volumes) != self.control_volume_sha256:
            raise StaleGenerationProfileError(
                "generation profile was computed on different control volumes"
            )
