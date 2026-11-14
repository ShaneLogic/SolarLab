"""Rectangular control volumes per unit out-of-plane depth (Neumann or periodic x)."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from solarlab.contracts import AREA, LENGTH, Geometry, frozen_array
from solarlab.mesh.geometry import (
    HALF_ENDPOINT_V1, PHYSICAL_HALF_ENDPOINT_V2, ControlVolumeGeometry1D,
    half_endpoint_widths,
)

RECTANGULAR_HALF_ENDPOINT_2D_V1 = "rectangular_half_endpoint_neumann_2d_v1"
RECTANGULAR_PERIODIC_2D_V1 = "rectangular_half_endpoint_periodic_2d_v1"
RECTANGULAR_2D_NAMES = (RECTANGULAR_HALF_ENDPOINT_2D_V1, RECTANGULAR_PERIODIC_2D_V1)


@dataclass(frozen=True)
class RectangularGeometry2D:
    """x is lateral, y is the stack; flatten (Ny, Nx) with x varying fastest.

    Areas are wx*wy [m2], and x/y face lengths are wy/wx [m]. Fluxes
    retain their three-dimensional densities; integrated rows are per metre
    of out-of-plane depth. Under ``lateral_bc="neumann"``, exterior lateral
    fluxes are identically zero; under ``lateral_bc="periodic"``, each row
    carries an explicit wrap face ``(nx-1) -> 0`` with spacing
    ``0.5 * (dx[0] + dx[-1])`` and matching dual-cell widths at ``x_min/x_max``.
    """

    x_m: NDArray[np.float64]
    y_geometry: ControlVolumeGeometry1D
    lateral_bc: Literal["neumann", "periodic"] = "neumann"
    name: str = field(init=False, default=RECTANGULAR_HALF_ENDPOINT_2D_V1)
    x_widths_m: NDArray[np.float64] = field(init=False)
    face_spacing_m: NDArray[np.float64] = field(init=False)
    contract_geometry: Geometry = field(init=False)
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.lateral_bc not in ("neumann", "periodic"):
            raise ValueError("lateral_bc must be 'neumann' or 'periodic'")
        name = (RECTANGULAR_HALF_ENDPOINT_2D_V1 if self.lateral_bc == "neumann"
                else RECTANGULAR_PERIODIC_2D_V1)
        object.__setattr__(self, "name", name)
        x = frozen_array(self.x_m)
        neumann_widths = half_endpoint_widths(x)
        if self.lateral_bc == "neumann":
            widths = neumann_widths
        else:
            dx = np.diff(x)
            dx_wrap = 0.5 * float(dx[0] + dx[-1])
            widths = np.empty_like(x)
            widths[0] = dx_wrap
            widths[-1] = dx_wrap
            if x.size > 2:
                widths[1:-1] = 0.5 * (dx[:-1] + dx[1:])
        y = self.y_geometry
        if y.name not in {HALF_ENDPOINT_V1, PHYSICAL_HALF_ENDPOINT_V2} or y.node_count < 3:
            raise ValueError("2D stack requires at least three nodes with physical half-endpoint cells")
        object.__setattr__(self, "x_m", x)
        object.__setattr__(self, "x_widths_m", frozen_array(widths))
        ids = np.arange(self.node_count).reshape(self.shape)
        if self.lateral_bc == "neumann":
            x_pairs = np.column_stack((ids[:, :-1].ravel(), ids[:, 1:].ravel()))
            x_measures: np.ndarray = np.repeat(y.cell_measures_m, self.nx - 1)
            x_distances = np.tile(np.diff(x), self.ny)
        else:
            dx = np.diff(x)
            dx_wrap = 0.5 * float(dx[0] + dx[-1])
            x_pairs = np.column_stack((ids.ravel(), np.roll(ids, -1, axis=1).ravel()))
            x_measures = np.repeat(y.cell_measures_m, self.nx)
            x_distances = np.tile(np.r_[dx, dx_wrap], self.ny)
        y_pairs = np.column_stack((ids[:-1, :].ravel(), ids[1:, :].ravel()))
        pairs = np.concatenate((x_pairs, y_pairs))
        areas = (y.cell_measures_m[:, None] * widths[None, :]).ravel()
        measures = np.r_[x_measures, np.tile(widths, self.ny - 1)]
        distances = np.r_[x_distances, np.repeat(np.diff(y.nodes_m), self.nx)]
        geometry = Geometry(areas, pairs, measures, AREA, LENGTH)
        object.__setattr__(self, "contract_geometry", geometry)
        object.__setattr__(self, "face_spacing_m", frozen_array(distances))
        digest = hashlib.sha256(self.name.encode())
        if y.name != HALF_ENDPOINT_V1:
            digest.update(y.name.encode())
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
    def x_faces_per_row(self) -> int:
        return self.nx if self.lateral_bc == "periodic" else self.nx - 1

    @property
    def x_face_count(self) -> int:
        return self.ny * self.x_faces_per_row

    @property
    def domain_span_m(self) -> float:
        return float(self.x_m[-1] - self.x_m[0])

    @property
    def lateral_length_m(self) -> float:
        if self.lateral_bc == "neumann":
            return float(self.x_m[-1] - self.x_m[0])
        return float(np.sum(self.x_widths_m))

    @property
    def wrap_face_indices(self) -> NDArray[np.intp]:
        if self.lateral_bc == "neumann":
            return np.empty(0, dtype=np.intp)
        return np.arange(self.nx - 1, self.x_face_count, self.nx, dtype=np.intp)

    @property
    def contact_nodes(self) -> NDArray[np.int64]:
        return np.r_[np.arange(self.nx), np.arange(self.node_count - self.nx, self.node_count)]
