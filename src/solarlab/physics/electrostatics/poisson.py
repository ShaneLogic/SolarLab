"""1D Poisson/Gauss law as an algebraic residual on named control volumes.

The electrostatic potential is a constrained variable: on every interior node

    C_-(phi_{i-1} - phi_i) + C_+(phi_{i+1} - phi_i) + rho_i w_i = 0

(net inflow of displacement plus enclosed charge, per unit cross-section),
with the original Dirichlet contact rows ``phi_0 = phi_left`` and
``phi_{N-1} = phi_right``. Face capacitances ``C = eps0*eps_face/h`` and the
interior widths ``w`` are taken from the channel's own historical geometry;
they are never replaced by a global default.

:class:`PoissonFactorization` eliminates the constraint for a given charge with
the same LAPACK tridiagonal factorisation as the legacy MoL. It is an exact
elimination tool for the algebraic constraint, not a new default solve path.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg.lapack import dgttrf, dgttrs
from scipy.sparse import csc_matrix, diags

from solarlab.contracts.arrays import ImmutableArrays, frozen_array
from solarlab.discretization.face_average import (
    DIELECTRIC_RULES, LAYER_SERIES_CAPACITANCE_V2, NODAL_HARMONIC_DIELECTRIC_V1,
    VACUUM_PERMITTIVITY_F_M, harmonic_face_mean, harmonic_mean,
    series_face_capacitance, subface_series_capacitance,
)
from solarlab.mesh.geometry import (
    HALF_ENDPOINT_V1, INTEGER_ENDPOINT_V1, NO_SHARED_NODES,
    PHYSICAL_BOUNDARY_TWO_SIDED_V1, TWO_SIDED_TRACE_V1, ControlVolumeGeometry1D,
    dual_cell_faces,
)
from solarlab.mesh.geometry_2d import RectangularGeometry2D
from solarlab.mesh.grid import LEFT_LAYER_WINS, RIGHT_LAYER_WINS, build_interface_stencils

__all__ = [
    "PoissonFactorization", "PoissonOperator1D", "add_interface_sheet_charge",
    "cell_parallel_permittivities", "contact_displacement",
    "layer_permittivities_from_geometry", "layer_series_face_capacitances",
    "rectangular_2d_face_capacitances",
]

EPS_0 = VACUUM_PERMITTIVITY_F_M


def _validate_nodal_permittivity(
    nodes: NDArray[np.float64], eps_r_node: ArrayLike
) -> NDArray[np.float64]:
    permittivity = np.asarray(eps_r_node, dtype=float)
    if (
        permittivity.shape != nodes.shape
        or not np.all(np.isfinite(permittivity))
        or np.any(permittivity <= 0.0)
    ):
        raise ValueError("permittivity must be finite and positive on the geometry nodes")
    return permittivity


def layer_permittivities_from_geometry(
    geometry: ControlVolumeGeometry1D,
    eps_r_node: ArrayLike | None = None,
    *,
    layer_eps_r: Sequence[float] | ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Resolve the positive relative permittivity of each geometry layer (P08-02a)."""
    layer_count = len(geometry.interfaces) + 1
    if eps_r_node is not None:
        permittivity = _validate_nodal_permittivity(geometry.nodes_m, eps_r_node)
    else:
        permittivity = None
    if layer_eps_r is not None:
        layers: NDArray[np.float64] = np.asarray(layer_eps_r, dtype=np.float64)
        if (
            layers.shape != (layer_count,)
            or not np.all(np.isfinite(layers))
            or np.any(layers <= 0.0)
        ):
            raise ValueError("layer_eps_r must supply one finite positive value per layer")
        return frozen_array(layers)
    if permittivity is None:
        raise ValueError("either eps_r_node or layer_eps_r is required")
    if layer_count == 1:
        if not np.all(permittivity == permittivity[0]):
            raise ValueError(
                "single-layer nodal permittivity is non-uniform; supply explicit layer/sub-face permittivities"
            )
        return frozen_array(np.array([permittivity[0]], dtype=np.float64))
    nodes = geometry.nodes_m
    positions = [plane.position_m for plane in geometry.interfaces]
    convention = geometry.material_owner_convention
    layers = np.empty(layer_count, dtype=np.float64)
    for m in range(layer_count):
        lower = -math.inf if m == 0 else positions[m - 1]
        upper = math.inf if m == layer_count - 1 else positions[m]
        if convention == RIGHT_LAYER_WINS:
            mask = (nodes >= lower) & (nodes < upper) if m < layer_count - 1 else (nodes >= lower)
        elif convention == LEFT_LAYER_WINS:
            if m == 0:
                mask = nodes <= upper
            elif m < layer_count - 1:
                mask = (nodes > lower) & (nodes <= upper)
            else:
                mask = nodes > lower
        elif convention == NO_SHARED_NODES:
            mask = (nodes > lower) & (nodes < upper)
        else:
            raise ValueError(f"unsupported material owner convention {convention!r}")
        if not np.any(mask):
            raise ValueError(f"layer {m} owns no grid nodes; pass layer_eps_r explicitly")
        layer_values = permittivity[mask]
        if not np.all(layer_values == layer_values[0]):
            raise ValueError(
                f"layer {m} nodal permittivity is non-uniform; pass explicit sub-face permittivities"
            )
        layers[m] = float(layer_values[0])
    return frozen_array(layers)


def _axis_series_capacitances(
    nodes: NDArray[np.float64],
    positions: Sequence[float],
    layers: NDArray[np.float64],
) -> NDArray[np.float64]:
    h = np.diff(nodes)
    if len(positions) == 0:
        return frozen_array(subface_series_capacitance(h, layers[0], np.zeros_like(h), layers[0]))
    pos_arr: NDArray[np.float64] = np.asarray(positions, dtype=np.float64)
    stencils = build_interface_stencils(nodes, pos_arr)
    layer_index: NDArray[np.intp] = np.asarray(
        np.searchsorted(pos_arr, nodes[:-1], side="right"), dtype=np.intp
    )
    d_left = h.copy()
    d_right = np.zeros_like(h)
    e_left = layers[layer_index].copy()
    e_right = e_left.copy()
    used_faces: set[int] = set()
    for m, stencil in enumerate(stencils):
        if stencil.shared_boundary_node is not None:
            continue
        if stencil.right_bulk_node != stencil.left_bulk_node + 1:
            raise ValueError("split dielectric face requires adjacent bulk nodes")
        face = int(stencil.left_bulk_node)
        if face in used_faces:
            raise ValueError("multiple material planes inside one face require multi-slab integration")
        used_faces.add(face)
        d_left[face] = float(stencil.left_distance_m)
        d_right[face] = float(stencil.right_distance_m)
        e_left[face] = float(layers[m])
        e_right[face] = float(layers[m + 1])
    return frozen_array(subface_series_capacitance(d_left, e_left, d_right, e_right))


def layer_series_face_capacitances(
    geometry: ControlVolumeGeometry1D,
    eps_r_node: ArrayLike | None = None,
    *,
    layer_eps_r: Sequence[float] | ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Sub-face material-extent series capacitances [F/m^2] on a 1D geometry (P08-02a).

    On shared-node grids (``x_k = X_if``), face ``[x_{k-1}, x_k]`` lies wholly in
    the left layer (``eps_L``) and face ``[x_k, x_{k+1}]`` lies wholly in the
    right layer (``eps_R``). On two-sided grids, the split face uses exact
    sub-face extents ``d_L = X_if - x_L`` and ``d_R = x_R - X_if``.
    """
    layers = layer_permittivities_from_geometry(geometry, eps_r_node, layer_eps_r=layer_eps_r)
    positions = [plane.position_m for plane in geometry.interfaces]
    return _axis_series_capacitances(geometry.nodes_m, positions, layers)


def _cell_parallel_from_faces(
    cell_faces: NDArray[np.float64],
    positions: Sequence[float],
    layers: NDArray[np.float64],
) -> NDArray[np.float64]:
    cell_count = cell_faces.size - 1
    if len(positions) == 0:
        return frozen_array(np.full(cell_count, float(layers[0]), dtype=float))
    pos_arr: NDArray[np.float64] = np.asarray(positions, dtype=np.float64)
    left_layer: NDArray[np.intp] = np.asarray(
        np.searchsorted(pos_arr, cell_faces[:-1], side="right"), dtype=np.intp
    )
    right_layer: NDArray[np.intp] = np.asarray(
        np.searchsorted(pos_arr, cell_faces[1:], side="left"), dtype=np.intp
    )
    parallel = layers[left_layer].copy()
    bounds = np.r_[cell_faces[0], pos_arr, cell_faces[-1]]
    straddling = np.flatnonzero(left_layer != right_layer)
    for raw_idx in straddling:
        idx = int(raw_idx)
        lo = float(cell_faces[idx])
        hi = float(cell_faces[idx + 1])
        m_start = int(left_layer[idx])
        m_stop = int(right_layer[idx])
        involved = layers[m_start : m_stop + 1]
        if np.all(involved == involved[0]):
            parallel[idx] = float(involved[0])
            continue
        weighted = 0.0
        for m in range(m_start, m_stop + 1):
            overlap = min(hi, float(bounds[m + 1])) - max(lo, float(bounds[m]))
            if overlap > 0.0:
                weighted += overlap * float(layers[m])
        parallel[idx] = weighted / (hi - lo)
    return frozen_array(parallel)


def cell_parallel_permittivities(
    geometry: ControlVolumeGeometry1D,
    eps_r_node: ArrayLike | None = None,
    *,
    layer_eps_r: Sequence[float] | ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Thickness-weighted parallel permittivity on each 1D control volume (P08-02a)."""
    layers = layer_permittivities_from_geometry(geometry, eps_r_node, layer_eps_r=layer_eps_r)
    positions = [plane.position_m for plane in geometry.interfaces]
    return _cell_parallel_from_faces(geometry.cell_faces_m, positions, layers)


def add_interface_sheet_charge(
    interior_residual_C_m2: NDArray[np.float64],
    geometry: ControlVolumeGeometry1D,
    sheet_charge_C_m2: ArrayLike,
    *,
    eps_r_node: ArrayLike | None = None,
    layer_eps_r: Sequence[float] | ArrayLike | None = None,
) -> None:
    """Add sheet charge ``sigma`` [C/m^2] so ``D_R - D_L = sigma`` at every interface.

    Shared-node interfaces place node ``k`` on the material plane, so ``sigma``
    enters interior Gauss row ``k - 1`` directly. Two-sided split faces partition
    ``sigma`` by exact sub-face series elimination weights ``(c_L, c_R) / (c_L + c_R)``.
    """
    sheets = np.asarray(sheet_charge_C_m2, dtype=float)
    if sheets.shape != (len(geometry.interfaces),) or not np.all(np.isfinite(sheets)):
        raise ValueError("sheet_charge_C_m2 must provide one finite value per interface")
    if interior_residual_C_m2.shape != (geometry.node_count - 2,):
        raise ValueError("interior_residual_C_m2 must match interior Poisson rows")
    if len(geometry.interfaces) == 0:
        return
    needs_layers = any(plane.shared_node is None for plane in geometry.interfaces)
    layers = (
        layer_permittivities_from_geometry(geometry, eps_r_node, layer_eps_r=layer_eps_r)
        if needs_layers
        else None
    )
    for plane, sigma in zip(geometry.interfaces, sheets):
        if plane.shared_node is not None:
            if plane.shared_node <= 0 or plane.shared_node >= geometry.node_count - 1:
                raise ValueError("shared interface node cannot lie on a Dirichlet contact")
            interior_residual_C_m2[plane.shared_node - 1] += float(sigma)
        else:
            assert layers is not None
            c_left = float(layers[plane.index]) / float(plane.left_distance_m)
            c_right = float(layers[plane.index + 1]) / float(plane.right_distance_m)
            total = c_left + c_right
            if 0 < plane.left_node < geometry.node_count - 1:
                interior_residual_C_m2[plane.left_node - 1] += (c_left / total) * float(sigma)
            if 0 < plane.right_node < geometry.node_count - 1:
                interior_residual_C_m2[plane.right_node - 1] += (c_right / total) * float(sigma)


def rectangular_2d_face_capacitances(
    geometry: RectangularGeometry2D,
    eps_r_y: ArrayLike,
    *,
    dielectric_rule: str = NODAL_HARMONIC_DIELECTRIC_V1,
    layer_eps_r_y: Sequence[float] | ArrayLike | None = None,
    lateral_eps_r_x: ArrayLike | None = None,
    lateral_interfaces_m: ArrayLike = (),
    lateral_layer_eps_r_x: Sequence[float] | ArrayLike | None = None,
) -> NDArray[np.float64]:
    """2D graph face capacitances [F/m^2] on ``x`` faces then ``y`` faces (P08-02a)."""
    if dielectric_rule not in DIELECTRIC_RULES:
        raise ValueError(f"unknown dielectric rule {dielectric_rule!r}")
    eps_y = _validate_nodal_permittivity(geometry.y_geometry.nodes_m, eps_r_y)
    dx_row = (
        np.diff(geometry.x_m)
        if geometry.lateral_bc == "neumann"
        else geometry.face_spacing_m[: geometry.nx]
    )
    lat_positions = np.asarray(lateral_interfaces_m, dtype=float)
    if lat_positions.ndim != 1 or not np.all(np.isfinite(lat_positions)) or np.any(np.diff(lat_positions) <= 0.0):
        raise ValueError("lateral_interfaces_m must be finite and strictly increasing")

    if lateral_eps_r_x is None and lat_positions.size == 0 and lateral_layer_eps_r_x is None:
        if dielectric_rule == NODAL_HARMONIC_DIELECTRIC_V1:
            horizontal = (EPS_0 * eps_y[:, None] / dx_row[None, :]).ravel()
            vertical_1d = PoissonOperator1D.from_harmonic_nodes(
                geometry.y_geometry.nodes_m, eps_y, geometry_name=geometry.y_geometry.name
            ).face_capacitance_F_m2
            return frozen_array(np.r_[horizontal, np.repeat(vertical_1d, geometry.nx)])
        vertical_1d = layer_series_face_capacitances(
            geometry.y_geometry, eps_y, layer_eps_r=layer_eps_r_y
        )
        eps_parallel_y = cell_parallel_permittivities(
            geometry.y_geometry, eps_y, layer_eps_r=layer_eps_r_y
        )
        horizontal = (EPS_0 * eps_parallel_y[:, None] / dx_row[None, :]).ravel()
        return frozen_array(np.r_[horizontal, np.repeat(vertical_1d, geometry.nx)])

    if not np.all(eps_y == eps_y[0]):
        raise ValueError("lateral permittivity profile requires uniform stack eps_r_y")
    eps_x = (
        _validate_nodal_permittivity(geometry.x_m, lateral_eps_r_x)
        if lateral_eps_r_x is not None
        else None
    )
    if dielectric_rule == NODAL_HARMONIC_DIELECTRIC_V1:
        if eps_x is None:
            raise ValueError("nodal_harmonic_v1 lateral profile requires nodal lateral_eps_r_x")
        interior_x = EPS_0 * harmonic_face_mean(eps_x) / np.diff(geometry.x_m)
        if geometry.lateral_bc == "periodic":
            wrap_eps = float(harmonic_mean(eps_x[-1], eps_x[0]))
            row_x = np.r_[interior_x, EPS_0 * wrap_eps / dx_row[-1]]
        else:
            row_x = interior_x
        dy = np.diff(geometry.y_geometry.nodes_m)
        vertical = (EPS_0 * eps_x[None, :] / dy[:, None]).ravel()
        return frozen_array(np.r_[np.tile(row_x, geometry.ny), vertical])

    # LAYER_SERIES_CAPACITANCE_V2 with lateral dielectric interfaces along x.
    x_layer_count = int(lat_positions.size) + 1
    layers_x: NDArray[np.float64]
    if lateral_layer_eps_r_x is not None:
        layers_x = np.asarray(lateral_layer_eps_r_x, dtype=np.float64)
        if layers_x.shape != (x_layer_count,) or not np.all(np.isfinite(layers_x)) or np.any(layers_x <= 0.0):
            raise ValueError("lateral_layer_eps_r_x must supply one finite positive value per lateral region")
    else:
        if eps_x is None:
            raise ValueError("either lateral_eps_r_x or lateral_layer_eps_r_x is required")
        if x_layer_count == 1:
            if not np.all(eps_x == eps_x[0]):
                raise ValueError("single-region lateral_eps_r_x must be uniform")
            layers_x = np.array([float(eps_x[0])], dtype=np.float64)
        else:
            stencils_x = build_interface_stencils(geometry.x_m, lat_positions)
            has_shared = any(s.shared_boundary_node is not None for s in stencils_x)
            layers_x = np.empty(x_layer_count, dtype=np.float64)
            for m in range(x_layer_count):
                lo = -math.inf if m == 0 else float(lat_positions[m - 1])
                hi = math.inf if m == x_layer_count - 1 else float(lat_positions[m])
                if has_shared:
                    mask = (geometry.x_m >= lo) & (geometry.x_m < hi) if m < x_layer_count - 1 else (geometry.x_m >= lo)
                else:
                    mask = (geometry.x_m > lo) & (geometry.x_m < hi)
                if not np.any(mask) or not np.all(eps_x[mask] == eps_x[mask][0]):
                    raise ValueError("lateral_eps_r_x must be piecewise-constant per lateral region")
                layers_x[m] = float(eps_x[mask][0])
    interior_x = _axis_series_capacitances(geometry.x_m, lat_positions.tolist(), layers_x)
    x_faces: NDArray[np.float64]
    if geometry.lateral_bc == "periodic":
        dx = np.diff(geometry.x_m)
        wrap_cap = float(
            subface_series_capacitance(0.5 * dx[-1], layers_x[-1], 0.5 * dx[0], layers_x[0])
        )
        row_x = np.r_[interior_x, wrap_cap]
        x_faces = np.empty(geometry.nx + 1, dtype=np.float64)
        x_faces[0] = geometry.x_m[0] - 0.5 * dx[-1]
        x_faces[1:-1] = 0.5 * (geometry.x_m[:-1] + geometry.x_m[1:])
        x_faces[-1] = geometry.x_m[-1] + 0.5 * dx[0]
    else:
        row_x = interior_x
        x_faces = dual_cell_faces(geometry.x_m).copy()
        if lat_positions.size:
            for stencil in build_interface_stencils(geometry.x_m, lat_positions):
                if stencil.shared_boundary_node is None:
                    x_faces[stencil.right_bulk_node] = stencil.interface_position_m
    eps_parallel_x = _cell_parallel_from_faces(x_faces, lat_positions.tolist(), layers_x)
    dy = np.diff(geometry.y_geometry.nodes_m)
    vertical = (EPS_0 * eps_parallel_x[None, :] / dy[:, None]).ravel()
    return frozen_array(np.r_[np.tile(row_x, geometry.ny), vertical])


@dataclass(frozen=True)
class PoissonOperator1D(ImmutableArrays):
    """Face capacitances [F/m^2] and interior control widths [m] of one channel."""

    face_capacitance_F_m2: NDArray[np.float64]
    interior_widths_m: NDArray[np.float64]
    geometry_name: str
    dielectric_rule: str = NODAL_HARMONIC_DIELECTRIC_V1

    def __post_init__(self) -> None:
        C = frozen_array(self.face_capacitance_F_m2)
        widths = frozen_array(self.interior_widths_m)
        if C.ndim != 1 or C.size < 2 or np.any(C <= 0.0):
            raise ValueError("face capacitances must be a finite positive vector")
        if widths.shape != (C.size - 1,) or np.any(widths <= 0.0):
            raise ValueError("interior cell widths must be finite, positive, and match faces")
        if not self.geometry_name:
            raise ValueError("a Poisson operator must name its geometry channel")
        if self.dielectric_rule not in DIELECTRIC_RULES:
            raise ValueError(f"unknown dielectric rule {self.dielectric_rule!r}")
        object.__setattr__(self, "face_capacitance_F_m2", C)
        object.__setattr__(self, "interior_widths_m", widths)

    @property
    def node_count(self) -> int:
        return int(self.face_capacitance_F_m2.size + 1)

    @classmethod
    def from_harmonic_nodes(cls, x: ArrayLike, eps_r: ArrayLike, *,
                            geometry_name: str = INTEGER_ENDPOINT_V1) -> PoissonOperator1D:
        """Historical MoL/2D channel: harmonic face permittivity of nodal values."""
        grid = np.asarray(x, dtype=float)
        permittivity = np.asarray(eps_r, dtype=float)
        if grid.ndim != 1 or permittivity.shape != grid.shape or grid.size < 3:
            raise ValueError("grid and nodal permittivity must be matching 1-D arrays")
        if np.any(np.diff(grid) <= 0.0) or np.any(permittivity <= 0.0):
            raise ValueError("grid must increase and permittivity must be positive")
        h = np.diff(grid)
        eps_face = harmonic_face_mean(permittivity)
        C = EPS_0 * eps_face / h
        h_cell = 0.5 * (h[:-1] + h[1:])
        return cls(C, h_cell, geometry_name)

    @classmethod
    def from_finite_volume(cls, face_capacitance_F_m2: ArrayLike,
                           interior_widths_m: ArrayLike, *,
                           geometry_name: str,
                           dielectric_rule: str = LAYER_SERIES_CAPACITANCE_V2) -> PoissonOperator1D:
        """Explicit series capacitances and charge-control volumes."""
        return cls(np.asarray(face_capacitance_F_m2, dtype=float),
                   np.asarray(interior_widths_m, dtype=float), geometry_name,
                   dielectric_rule=dielectric_rule)

    @classmethod
    def from_layer_series_capacitance(
        cls,
        geometry: ControlVolumeGeometry1D,
        eps_r_node: ArrayLike | None = None,
        *,
        layer_eps_r: Sequence[float] | ArrayLike | None = None,
    ) -> PoissonOperator1D:
        """Sub-face material-extent series capacitance on a named 1D geometry (P08-02a)."""
        if geometry.node_count < 3:
            raise ValueError("Poisson operator requires at least three grid nodes")
        capacitance = layer_series_face_capacitances(
            geometry, eps_r_node, layer_eps_r=layer_eps_r
        )
        return cls(
            capacitance,
            geometry.cell_measures_m[1:-1],
            geometry.name,
            dielectric_rule=LAYER_SERIES_CAPACITANCE_V2,
        )

    @classmethod
    def for_geometry(
        cls,
        geometry: ControlVolumeGeometry1D,
        eps_r_node: ArrayLike,
        *,
        dielectric_rule: str | None = None,
        layer_eps_r: Sequence[float] | ArrayLike | None = None,
    ) -> PoissonOperator1D:
        """The channel each named geometry historically used, or an explicit dielectric rule.

        Integer/half-endpoint grids keep the harmonic nodal mean by default
        (``dielectric_rule=None``). Passing ``dielectric_rule=LAYER_SERIES_CAPACITANCE_V2``
        opts into sub-face material-extent series capacitance while preserving the
        geometry's interior control widths.
        """
        nodes = geometry.nodes_m
        permittivity = np.asarray(eps_r_node, dtype=float)
        if permittivity.shape != nodes.shape:
            raise ValueError("permittivity must be given on the geometry nodes")
        if dielectric_rule is not None and dielectric_rule not in DIELECTRIC_RULES:
            raise ValueError(f"unknown dielectric rule {dielectric_rule!r}")
        if dielectric_rule == LAYER_SERIES_CAPACITANCE_V2:
            return cls.from_layer_series_capacitance(
                geometry, permittivity, layer_eps_r=layer_eps_r
            )
        base = cls.from_harmonic_nodes(nodes, permittivity, geometry_name=geometry.name)
        if dielectric_rule == NODAL_HARMONIC_DIELECTRIC_V1:
            if geometry.name in {INTEGER_ENDPOINT_V1, HALF_ENDPOINT_V1}:
                return base
            return cls(
                base.face_capacitance_F_m2,
                geometry.cell_measures_m[1:-1],
                geometry.name,
                dielectric_rule=NODAL_HARMONIC_DIELECTRIC_V1,
            )
        if geometry.name in {INTEGER_ENDPOINT_V1, HALF_ENDPOINT_V1}:
            return base
        if geometry.name not in {TWO_SIDED_TRACE_V1, PHYSICAL_BOUNDARY_TWO_SIDED_V1}:
            raise ValueError(f"no historical Poisson channel for {geometry.name!r}")
        capacitance = np.asarray(base.face_capacitance_F_m2, dtype=float).copy()
        for plane in geometry.interfaces:
            capacitance[plane.left_node] = series_face_capacitance(
                plane.left_distance_m, float(permittivity[plane.left_node]),
                plane.right_distance_m, float(permittivity[plane.right_node]),
            )
        return cls(
            capacitance,
            geometry.cell_measures_m[1:-1],
            geometry.name,
            dielectric_rule=LAYER_SERIES_CAPACITANCE_V2,
        )

    def interior_residual(self, phi: ArrayLike, rho_C_m3: ArrayLike) -> NDArray[np.float64]:
        """Gauss balance of nodes ``1..N-2`` [C/m^2] (historical expression)."""
        potential = self._nodal(phi, "phi")
        charge = self._nodal(rho_C_m3, "rho")
        C = self.face_capacitance_F_m2
        residual: NDArray[np.float64] = (
            C[:-1] * (potential[:-2] - potential[1:-1])
            + C[1:] * (potential[2:] - potential[1:-1])
            + charge[1:-1] * self.interior_widths_m
        )
        return residual

    def residual(self, phi: ArrayLike, rho_C_m3: ArrayLike, phi_left: float,
                 phi_right: float) -> NDArray[np.float64]:
        """All N rows: Dirichlet constraint rows around the interior Gauss rows."""
        potential = self._nodal(phi, "phi")
        if not (math.isfinite(phi_left) and math.isfinite(phi_right)):
            raise ValueError("contact potentials must be finite")
        rows = np.empty(self.node_count)
        rows[0] = potential[0] - phi_left
        rows[-1] = potential[-1] - phi_right
        rows[1:-1] = self.interior_residual(potential, rho_C_m3)
        return rows

    def jacobian(self) -> csc_matrix:
        """``d residual / d phi`` (constant, tridiagonal) for :meth:`residual`."""
        C = self.face_capacitance_F_m2
        n = self.node_count
        diagonal = np.ones(n)
        diagonal[1:-1] = -(C[:-1] + C[1:])
        lower = np.zeros(n - 1)
        upper = np.zeros(n - 1)
        lower[:-1] = C[:-1]     # row i, column i-1, for i = 1..N-2
        upper[1:] = C[1:]       # row i, column i+1, for i = 1..N-2
        matrix: csc_matrix = diags([lower, diagonal, upper], [-1, 0, 1], format="csc")
        return matrix

    def charge_jacobian(self) -> NDArray[np.float64]:
        """``d residual_i / d rho_i`` (diagonal): interior widths, zero on contact rows."""
        derivative = np.zeros(self.node_count)
        derivative[1:-1] = self.interior_widths_m
        return derivative

    def face_displacement(self, phi: ArrayLike) -> NDArray[np.float64]:
        """Displacement ``D = -C*dphi`` [C/m^2] on the N-1 faces, oriented along +x."""
        potential = self._nodal(phi, "phi")
        displacement: NDArray[np.float64] = -self.face_capacitance_F_m2 * np.diff(potential)
        return displacement

    def _nodal(self, value: ArrayLike, label: str) -> NDArray[np.float64]:
        array = np.asarray(value, dtype=float)
        if array.shape != (self.node_count,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{label} must be a finite array on the operator nodes")
        return array


def contact_displacement(face_displacement_C_m2: ArrayLike, rho_C_m3: ArrayLike,
                         widths_m: ArrayLike) -> NDArray[np.float64]:
    """Gauss extension from the first/last internal faces to the two contacts."""
    face_d = np.asarray(face_displacement_C_m2, dtype=float)
    rho = np.asarray(rho_C_m3, dtype=float)
    widths = np.asarray(widths_m, dtype=float)
    return np.asarray(
        [
            face_d[0] - rho[0] * widths[0],
            face_d[-1] + rho[-1] * widths[-1],
        ]
    )


@dataclass(frozen=True)
class PoissonFactorization:
    """Precomputed LAPACK LU of the interior Poisson tridiagonal (legacy identical).

    SciPy's ``dgttrf`` wrapper cannot infer ``n < 3``; such tiny operators are
    embedded in a decoupled 3x3 block whose dummy rows are discarded.
    """

    C: NDArray[np.float64]
    h_cell: NDArray[np.float64]
    dl: NDArray[np.float64]
    d: NDArray[np.float64]
    du: NDArray[np.float64]
    du2: NDArray[np.float64]
    ipiv: NDArray[np.int32]
    N: int

    @classmethod
    def factor(cls, operator: PoissonOperator1D) -> PoissonFactorization:
        C = np.asarray(operator.face_capacitance_F_m2, dtype=float)
        h_cell = np.asarray(operator.interior_widths_m, dtype=float)
        N = C.size + 1
        dia = -(C[:-1] + C[1:])                       # (N-2,)  main diagonal
        off = C[1:-1].copy()                          # (N-3,)  sub- and super-diagonals
        physical_interior = dia.size
        if physical_interior < 3:
            padded_dia: NDArray[np.float64] = -np.ones(3, dtype=float)
            padded_dia[:physical_interior] = dia
            padded_off: NDArray[np.float64] = np.zeros(2, dtype=float)
            if physical_interior == 2:
                padded_off[0] = off[0]
            dia = padded_dia
            off = padded_off
        dl_in = off.astype(np.float64)
        d_in = dia.astype(np.float64)
        du_in = off.astype(np.float64)
        dl, d, du, du2, ipiv, info = dgttrf(dl_in, d_in, du_in)
        if info != 0:
            raise RuntimeError(f"dgttrf failed with info={info}")
        return cls(C.copy(), h_cell.copy(), dl, d, du, du2, ipiv, N)

    def solve(self, rho_C_m3: ArrayLike, phi_left: float,
              phi_right: float) -> NDArray[np.float64]:
        """Potential satisfying every Poisson row for this charge and contacts."""
        rho = np.asarray(rho_C_m3, dtype=float)
        N = self.N
        C = self.C
        rhs = -rho[1:-1] * self.h_cell
        rhs[0] -= C[0] * phi_left
        rhs[-1] -= C[-1] * phi_right
        factor_size = self.d.size
        if factor_size == rhs.size:
            b = rhs.reshape(-1, 1)
        else:
            b = np.zeros((factor_size, 1), dtype=float)
            b[: rhs.size, 0] = rhs
        phi_int, info = dgttrs(self.dl, self.d, self.du, self.du2, self.ipiv, b)
        if info != 0:
            raise RuntimeError(f"dgttrs failed with info={info}")
        phi = np.empty(N)
        phi[0] = phi_left
        phi[-1] = phi_right
        phi[1:-1] = phi_int[: N - 2, 0]
        return phi
