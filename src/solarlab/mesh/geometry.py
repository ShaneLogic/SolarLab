"""Explicitly named historical 1D control-volume geometries.

Each historical consumer used its own node-centred control volumes. They are
kept as separate, named geometries so a change of geometry can never hide in a
"unified" endpoint rule (V1 P04-01). Every builder reproduces its source's
arithmetic, so the arrays are bit-identical to the historical ones:

``INTEGER_ENDPOINT_V1``
    1D method of lines (``solver/mol.py``, ``physics/continuity.py``,
    ``physics/generation.dual_cell_widths``). Interior nodes own half of each
    neighbouring interval; the two contact nodes own the FULL adjacent
    interval, so the conservation weights sum to ``L + (dx[0]+dx[-1])/2``. The
    shared layer-boundary node stays on the grid; the right layer owns it for
    both material values and role masks.
``HALF_ENDPOINT_V1``
    2D transport axis (``twod/continuity_2d.py``, ``twod/poisson_2d.py``) and
    the 2D non-periodic lateral axis. Contact cells are geometric half cells,
    so the weights sum to ``L``. The shared node stays on the grid. Its
    material values (eps_r, doping, chi, Eg, A*, lifetimes) are extruded from
    the 1D arrays, so the right layer owns them; only role tagging (absorber
    range, grain-boundary painting) gives the node to the left layer.
``TWO_SIDED_TRACE_V1``
    Quasi-Fermi ``two_sided_trace`` topology without physical boundary volumes
    (``quasi_fermi_steady_state._prepare_two_sided_material``). Shared nodes
    are removed; the two cells beside each interface extend to the true plane;
    contact cells keep the full integer-endpoint width.
``PHYSICAL_BOUNDARY_TWO_SIDED_V1``
    R1 physical lane (``physics/physical_control_volume.physical_cell_faces``).
    Faces are midpoints, clipped at the contacts and moved to the true plane
    at every interface gap; the weights are the face differences and sum to
    ``L``.

The periodic 2D lateral wrap cell is a 2D geometry and belongs to P06-01.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from solarlab.contracts import (
    LENGTH, ONE, Geometry, ImmutableArrays, TerminalPort, frozen_array,
)
from solarlab.mesh.grid import (
    LEFT_LAYER_WINS, RIGHT_LAYER_WINS, InterfaceStencil, build_interface_stencils,
)

__all__ = [
    "ControlVolumeGeometry1D", "GEOMETRY_NAMES", "HALF_ENDPOINT_V1", "INTEGER_ENDPOINT_V1",
    "InterfacePlane", "NO_SHARED_NODES", "PHYSICAL_BOUNDARY_TWO_SIDED_V1", "PortFace",
    "TWO_SIDED_TRACE_V1", "build_geometry", "dual_cell_faces", "half_endpoint_widths",
    "half_endpoint_geometry", "integer_endpoint_geometry", "integer_endpoint_widths",
    "physical_boundary_two_sided_geometry", "physical_cell_faces", "two_sided_trace_geometry",
    "two_sided_trace_widths",
]

INTEGER_ENDPOINT_V1 = "legacy_mol_integer_endpoint_v1"
HALF_ENDPOINT_V1 = "legacy_twod_half_endpoint_v1"
TWO_SIDED_TRACE_V1 = "qf_two_sided_trace_integer_endpoint_v1"
PHYSICAL_BOUNDARY_TWO_SIDED_V1 = "physical_boundary_two_sided_v1"
GEOMETRY_NAMES = (
    INTEGER_ENDPOINT_V1, HALF_ENDPOINT_V1, TWO_SIDED_TRACE_V1, PHYSICAL_BOUNDARY_TWO_SIDED_V1,
)
NO_SHARED_NODES = "no_shared_nodes"
# Geometries whose contact cells are geometric half cells; the other two weight
# each contact node by its full adjacent interval.
_GEOMETRIC_CONTACT_CELLS = frozenset({HALF_ENDPOINT_V1, PHYSICAL_BOUNDARY_TWO_SIDED_V1})

# 1D devices are evaluated per unit cross-section: ports and faces have area
# 1 m^2, cell measures are lengths and stored particles are per m^2.
UNIT_CROSS_SECTION_M2 = 1.0


def _strict_grid(x: ArrayLike, *, minimum: int) -> NDArray[np.float64]:
    grid = np.asarray(x, dtype=float)
    if (
        grid.ndim != 1
        or grid.size < minimum
        or not np.all(np.isfinite(grid))
        or np.any(np.diff(grid) <= 0)
    ):
        raise ValueError("grid must be finite and strictly increasing")
    return grid


def integer_endpoint_widths(x: ArrayLike) -> NDArray[np.float64]:
    """Historical MoL dual-cell weights (full interval at both contact nodes)."""
    grid = _strict_grid(x, minimum=2)
    dx = np.diff(grid)
    w = np.empty(grid.shape[0], dtype=float)
    w[0] = dx[0]
    w[-1] = dx[-1]
    w[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    return w


def dual_cell_faces(x: ArrayLike) -> NDArray[np.float64]:
    """The ``N+1`` geometric faces bounding the node-centred dual cells."""
    grid = _strict_grid(x, minimum=2)
    faces = np.empty(grid.shape[0] + 1, dtype=float)
    faces[0] = grid[0]
    faces[-1] = grid[-1]
    faces[1:-1] = 0.5 * (grid[:-1] + grid[1:])
    return faces


def half_endpoint_widths(x: ArrayLike) -> NDArray[np.float64]:
    """Historical 2D dual-cell widths along a non-periodic axis."""
    grid = _strict_grid(x, minimum=2)
    dx = np.diff(grid)
    w = np.empty(grid.shape[0])
    w[0] = dx[0] / 2.0
    w[-1] = dx[-1] / 2.0
    if grid.shape[0] > 2:
        w[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    return w


def physical_cell_faces(x: ArrayLike, interface_positions: ArrayLike = ()) -> NDArray[np.float64]:
    """Clip outer cells at contacts; split unlike materials at the true plane."""
    grid = np.asarray(x, dtype=float)
    positions = np.asarray(interface_positions, dtype=float)
    if (
        grid.ndim != 1
        or grid.size < 3
        or not np.all(np.isfinite(grid))
        or np.any(np.diff(grid) <= 0)
    ):
        raise ValueError("physical grid must be finite and strictly increasing")
    if (
        positions.ndim != 1
        or not np.all(np.isfinite(positions))
        or np.any(np.diff(positions) <= 0)
    ):
        raise ValueError("interface positions must be finite and ordered")
    faces = np.r_[grid[0], (grid[:-1] + grid[1:]) / 2, grid[-1]]
    used = set()
    for position in positions:
        right = int(np.searchsorted(grid, position))
        if right == 0 or right == grid.size or grid[right] == position or right in used:
            raise ValueError("each interface needs its own two-sided grid gap")
        faces[right] = position
        used.add(right)
    if np.any(np.diff(faces) <= 0):
        raise ValueError("nonpositive physical control volume")
    faces.setflags(write=False)
    return faces


def _two_sided_stencils(grid: NDArray[np.float64],
                        positions: ArrayLike) -> tuple[InterfaceStencil, ...]:
    stencils = build_interface_stencils(grid, positions)
    if any(stencil.shared_boundary_node is not None for stencil in stencils):
        raise ValueError(
            "two-sided geometry requires a grid without shared interface nodes"
        )
    for stencil in stencils:
        if stencil.right_bulk_node != stencil.left_bulk_node + 1:
            raise ValueError("two-sided interface bulk nodes must share one grid face")
    return stencils


def two_sided_trace_widths(x: ArrayLike, interface_positions: ArrayLike) -> NDArray[np.float64]:
    """QF two-sided-trace carrier volumes: integer endpoints, true-plane interiors."""
    grid = _strict_grid(x, minimum=3)
    widths = integer_endpoint_widths(grid).copy()
    for stencil in _two_sided_stencils(grid, interface_positions):
        left = int(stencil.left_bulk_node)
        right = int(stencil.right_bulk_node)
        face_width = float(grid[right] - grid[left])
        half_width = 0.5 * face_width
        widths[left] += float(stencil.left_distance_m) - half_width
        widths[right] += float(stencil.right_distance_m) - half_width
    if np.any(~np.isfinite(widths)) or np.any(widths <= 0.0):
        raise ValueError("two-sided interface produced an invalid carrier control volume")
    return widths


@dataclass(frozen=True)
class InterfacePlane:
    """A true material interface and the nodes that bracket it."""

    index: int
    position_m: float
    left_node: int
    right_node: int
    left_distance_m: float
    right_distance_m: float
    shared_node: int | None

    @classmethod
    def from_stencil(cls, index: int, stencil: InterfaceStencil) -> InterfacePlane:
        return cls(index, stencil.interface_position_m, stencil.left_bulk_node,
                   stencil.right_bulk_node, stencil.left_distance_m,
                   stencil.right_distance_m, stencil.shared_boundary_node)


@dataclass(frozen=True)
class PortFace:
    """A contact face: its node, position, outward normal and area."""

    id: str
    node: int
    position_m: float
    outward_normal: int
    area_m2: float = UNIT_CROSS_SECTION_M2

    def terminal_port(self) -> TerminalPort:
        return TerminalPort(self.id, self.outward_normal, self.area_m2)


@dataclass(frozen=True)
class ControlVolumeGeometry1D(ImmutableArrays):
    """Named node-centred control volumes on a 1D grid (unit cross-section).

    ``cell_measures_m`` are the weights the conservation laws multiply.
    ``cell_faces_m`` are the geometric extents ``[faces[i], faces[i+1]]`` of
    each cell; for the integer-endpoint geometry the contact cells' weights
    deliberately differ from their extents (``geometric_cells=False``).
    Transport faces join consecutive nodes and are oriented along +x.

    A node on a layer boundary has two independent historical owners:
    ``material_owner_convention`` picks the layer whose material values
    (eps_r, doping, bands, DOS, mobilities) the node carries, and
    ``role_owner_convention`` the layer whose role tag (absorber range,
    grain-boundary painting) it carries. They differ for ``HALF_ENDPOINT_V1``.
    """

    name: str
    nodes_m: NDArray[np.float64]
    cell_measures_m: NDArray[np.float64]
    cell_faces_m: NDArray[np.float64]
    interfaces: tuple[InterfacePlane, ...]
    material_owner_convention: str
    role_owner_convention: str
    face_normal: int = 1
    face_area_m2: float = UNIT_CROSS_SECTION_M2
    face_pairs: NDArray[np.int64] = field(init=False)
    face_spacing_m: NDArray[np.float64] = field(init=False)
    ports: tuple[PortFace, PortFace] = field(init=False)
    extent_m: float = field(init=False)
    total_measure_m: float = field(init=False)
    geometric_cells: bool = field(init=False)

    def __post_init__(self) -> None:
        if self.name not in GEOMETRY_NAMES:
            raise ValueError(f"unknown historical geometry {self.name!r}")
        conventions = {RIGHT_LAYER_WINS, LEFT_LAYER_WINS, NO_SHARED_NODES}
        if (self.material_owner_convention not in conventions
                or self.role_owner_convention not in conventions):
            raise ValueError("unknown node ownership convention")
        if self.face_normal != 1 or not self.face_area_m2 > 0.0:
            raise ValueError("1D faces are oriented along +x with a positive area")
        nodes = frozen_array(self.nodes_m)
        measures = frozen_array(self.cell_measures_m)
        faces = frozen_array(self.cell_faces_m)
        if (nodes.ndim != 1 or nodes.size < 2 or np.any(np.diff(nodes) <= 0)
                or measures.shape != nodes.shape or np.any(measures <= 0)
                or faces.shape != (nodes.size + 1,) or np.any(np.diff(faces) <= 0)):
            raise ValueError("inconsistent control-volume geometry")
        pairs = np.column_stack((np.arange(nodes.size - 1), np.arange(1, nodes.size)))
        pairs = np.frombuffer(pairs.astype(np.int64).tobytes(), dtype=np.int64).reshape(pairs.shape)
        object.__setattr__(self, "nodes_m", nodes)
        object.__setattr__(self, "cell_measures_m", measures)
        object.__setattr__(self, "cell_faces_m", faces)
        object.__setattr__(self, "face_pairs", pairs)
        object.__setattr__(self, "face_spacing_m", frozen_array(np.diff(nodes)))
        object.__setattr__(self, "ports", (
            PortFace("left", 0, float(nodes[0]), -1, self.face_area_m2),
            PortFace("right", int(nodes.size - 1), float(nodes[-1]), 1, self.face_area_m2),
        ))
        geometric = self.name in _GEOMETRIC_CONTACT_CELLS
        # Measures and extents come from different historical expressions, so
        # they agree to rounding (not bitwise) wherever the cell is geometric.
        extents = np.diff(faces)
        checked = slice(None) if geometric else slice(1, -1)
        tolerance = 16.0 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(faces))))
        if np.any(np.abs(measures[checked] - extents[checked]) > tolerance):
            raise ValueError("cell measures disagree with their geometric extents")
        object.__setattr__(self, "extent_m", float(nodes[-1] - nodes[0]))
        object.__setattr__(self, "total_measure_m", math.fsum(measures.tolist()))
        object.__setattr__(self, "geometric_cells", geometric)

    @property
    def node_count(self) -> int:
        return int(self.nodes_m.size)

    def to_contract_geometry(self, *, cell_support: str = "cells",
                             face_support: str = "faces") -> Geometry:
        """Contract geometry: measures are lengths, faces are dimensionless."""
        return Geometry(self.cell_measures_m, self.face_pairs,
                        np.full(self.node_count - 1, self.face_area_m2), LENGTH, ONE,
                        cell_support, face_support)


def _planes(grid: NDArray[np.float64], positions: ArrayLike) -> tuple[InterfacePlane, ...]:
    stencils = build_interface_stencils(grid, positions) if np.asarray(positions).size else ()
    return tuple(InterfacePlane.from_stencil(i, stencil) for i, stencil in enumerate(stencils))


def integer_endpoint_geometry(x: ArrayLike,
                              interface_positions: ArrayLike = ()) -> ControlVolumeGeometry1D:
    grid = _strict_grid(x, minimum=2)
    planes = _planes(grid, interface_positions)
    if any(plane.shared_node is None for plane in planes):
        raise ValueError("the integer-endpoint grid keeps a node on every interface")
    return ControlVolumeGeometry1D(INTEGER_ENDPOINT_V1, grid, integer_endpoint_widths(grid),
                                   dual_cell_faces(grid), planes,
                                   material_owner_convention=RIGHT_LAYER_WINS,
                                   role_owner_convention=RIGHT_LAYER_WINS)


def half_endpoint_geometry(x: ArrayLike,
                           interface_positions: ArrayLike = ()) -> ControlVolumeGeometry1D:
    grid = _strict_grid(x, minimum=2)
    planes = _planes(grid, interface_positions)
    if any(plane.shared_node is None for plane in planes):
        raise ValueError("the half-endpoint grid keeps a node on every interface")
    # Material arrays are extruded from the 1D (right-owned) arrays; roles are left-owned.
    return ControlVolumeGeometry1D(HALF_ENDPOINT_V1, grid, half_endpoint_widths(grid),
                                   dual_cell_faces(grid), planes,
                                   material_owner_convention=RIGHT_LAYER_WINS,
                                   role_owner_convention=LEFT_LAYER_WINS)


def two_sided_trace_geometry(x: ArrayLike,
                             interface_positions: ArrayLike) -> ControlVolumeGeometry1D:
    grid = _strict_grid(x, minimum=3)
    planes = tuple(InterfacePlane.from_stencil(i, stencil) for i, stencil
                   in enumerate(_two_sided_stencils(grid, interface_positions)))
    faces = dual_cell_faces(grid).copy()
    for plane in planes:
        faces[plane.right_node] = plane.position_m
    return ControlVolumeGeometry1D(TWO_SIDED_TRACE_V1, grid,
                                   two_sided_trace_widths(grid, interface_positions),
                                   faces, planes, material_owner_convention=NO_SHARED_NODES,
                                   role_owner_convention=NO_SHARED_NODES)


def physical_boundary_two_sided_geometry(
        x: ArrayLike, interface_positions: ArrayLike) -> ControlVolumeGeometry1D:
    grid = _strict_grid(x, minimum=3)
    planes = tuple(InterfacePlane.from_stencil(i, stencil) for i, stencil
                   in enumerate(_two_sided_stencils(grid, interface_positions)))
    faces = physical_cell_faces(grid, interface_positions)
    return ControlVolumeGeometry1D(PHYSICAL_BOUNDARY_TWO_SIDED_V1, grid, np.diff(faces),
                                   faces, planes, material_owner_convention=NO_SHARED_NODES,
                                   role_owner_convention=NO_SHARED_NODES)


def build_geometry(name: str, x: ArrayLike,
                   interface_positions: ArrayLike = ()) -> ControlVolumeGeometry1D:
    """Build one named historical geometry; there is no default geometry."""
    builders = {
        INTEGER_ENDPOINT_V1: integer_endpoint_geometry,
        HALF_ENDPOINT_V1: half_endpoint_geometry,
        TWO_SIDED_TRACE_V1: two_sided_trace_geometry,
        PHYSICAL_BOUNDARY_TWO_SIDED_V1: physical_boundary_two_sided_geometry,
    }
    if name not in builders:
        raise ValueError(f"unknown historical geometry {name!r}")
    return builders[name](x, interface_positions)
