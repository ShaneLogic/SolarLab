"""Rectangular, nonperiodic control volumes per unit out-of-plane depth."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

import numpy as np
from numpy.typing import NDArray

from solarlab.contracts import AREA, LENGTH, Geometry, frozen_array
from solarlab.mesh.geometry import HALF_ENDPOINT_V1, ControlVolumeGeometry1D, half_endpoint_widths

RECTANGULAR_HALF_ENDPOINT_2D_V1 = "rectangular_half_endpoint_neumann_2d_v1"


@dataclass(frozen=True)
class RectangularGeometry2D:
    """x is lateral, y is the stack; flatten (Ny, Nx) with x varying fastest.

    Areas are wx*wy [m2], and x/y face lengths are wy/wx [m]. Fluxes
    retain their three-dimensional densities; integrated rows are per metre
    of out-of-plane depth. There are no exterior x faces: their explicitly
    prescribed displacement and particle fluxes are exactly zero.
    """

    x_m: NDArray[np.float64]
    y_geometry: ControlVolumeGeometry1D
    name: str = field(init=False, default=RECTANGULAR_HALF_ENDPOINT_2D_V1)
    x_widths_m: NDArray[np.float64] = field(init=False)
    face_spacing_m: NDArray[np.float64] = field(init=False)
    contract_geometry: Geometry = field(init=False)
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        x = frozen_array(self.x_m)
        widths = half_endpoint_widths(x)
        y = self.y_geometry
        if y.name != HALF_ENDPOINT_V1 or y.node_count < 3:
            raise ValueError("2D stack requires at least three nodes with physical half-endpoint cells")
        object.__setattr__(self, "x_m", x)
        object.__setattr__(self, "x_widths_m", frozen_array(widths))
        ids = np.arange(self.node_count).reshape(self.shape)
        x_pairs = np.column_stack((ids[:, :-1].ravel(), ids[:, 1:].ravel()))
        y_pairs = np.column_stack((ids[:-1, :].ravel(), ids[1:, :].ravel()))
        pairs = np.concatenate((x_pairs, y_pairs))
        areas = (y.cell_measures_m[:, None] * widths[None, :]).ravel()
        measures = np.r_[np.repeat(y.cell_measures_m, self.nx - 1), np.tile(widths, self.ny - 1)]
        distances = np.r_[np.tile(np.diff(x), self.ny), np.repeat(np.diff(y.nodes_m), self.nx)]
        geometry = Geometry(areas, pairs, measures, AREA, LENGTH)
        object.__setattr__(self, "contract_geometry", geometry)
        object.__setattr__(self, "face_spacing_m", frozen_array(distances))
        digest = hashlib.sha256(self.name.encode())
        for value in (x, y.nodes_m, areas, pairs, measures, distances):
            digest.update(np.ascontiguousarray(value, dtype="<f8").tobytes())
        object.__setattr__(self, "sha256", digest.hexdigest())

    @property
    def nx(self) -> int:
        return int(self.x_m.size)

    @property
    def ny(self) -> int:
        return self.y_geometry.node_count

    @property
    def shape(self) -> tuple[int, int]:
        return self.ny, self.nx

    @property
    def node_count(self) -> int:
        return self.nx * self.ny

    @property
    def x_face_count(self) -> int:
        return self.ny * (self.nx - 1)

    @property
    def lateral_length_m(self) -> float:
        return float(self.x_m[-1] - self.x_m[0])

    @property
    def contact_nodes(self) -> NDArray[np.int64]:
        return np.r_[np.arange(self.nx), np.arange(self.node_count - self.nx, self.node_count)]
