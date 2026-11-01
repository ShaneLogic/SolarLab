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

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg.lapack import dgttrf, dgttrs
from scipy.sparse import csc_matrix, diags

from solarlab.contracts.arrays import ImmutableArrays, frozen_array
from solarlab.discretization.face_average import (
    VACUUM_PERMITTIVITY_F_M, harmonic_face_mean, series_face_capacitance,
)
from solarlab.mesh.geometry import (
    HALF_ENDPOINT_V1, INTEGER_ENDPOINT_V1, PHYSICAL_BOUNDARY_TWO_SIDED_V1,
    TWO_SIDED_TRACE_V1, ControlVolumeGeometry1D,
)

__all__ = ["PoissonFactorization", "PoissonOperator1D", "contact_displacement"]

EPS_0 = VACUUM_PERMITTIVITY_F_M


@dataclass(frozen=True)
class PoissonOperator1D(ImmutableArrays):
    """Face capacitances [F/m^2] and interior control widths [m] of one channel."""

    face_capacitance_F_m2: NDArray[np.float64]
    interior_widths_m: NDArray[np.float64]
    geometry_name: str

    def __post_init__(self) -> None:
        C = frozen_array(self.face_capacitance_F_m2)
        widths = frozen_array(self.interior_widths_m)
        if C.ndim != 1 or C.size < 2 or np.any(C <= 0.0):
            raise ValueError("face capacitances must be a finite positive vector")
        if widths.shape != (C.size - 1,) or np.any(widths <= 0.0):
            raise ValueError("interior cell widths must be finite, positive, and match faces")
        if not self.geometry_name:
            raise ValueError("a Poisson operator must name its geometry channel")
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
                           geometry_name: str) -> PoissonOperator1D:
        """Explicit series capacitances and charge-control volumes."""
        return cls(np.asarray(face_capacitance_F_m2, dtype=float),
                   np.asarray(interior_widths_m, dtype=float), geometry_name)

    @classmethod
    def for_geometry(cls, geometry: ControlVolumeGeometry1D,
                     eps_r_node: ArrayLike) -> PoissonOperator1D:
        """The channel each named geometry historically used.

        Integer/half-endpoint grids keep the harmonic nodal mean. Two-sided
        geometries replace each interface face by its exact series capacitance
        at the true plane and use their own carrier control volumes.
        """
        nodes = geometry.nodes_m
        permittivity = np.asarray(eps_r_node, dtype=float)
        if permittivity.shape != nodes.shape:
            raise ValueError("permittivity must be given on the geometry nodes")
        base = cls.from_harmonic_nodes(nodes, permittivity, geometry_name=geometry.name)
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
        return cls(capacitance, geometry.cell_measures_m[1:-1], geometry.name)

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
