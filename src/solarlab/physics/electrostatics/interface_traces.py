"""Two-sided interface declarations and the electrostatic (Gauss-jump) traces.

At a two-sided heterointerface three different kinds of local unknowns exist
and are declared separately, with different supports, units and domains:

* carrier traces ``n_L, p_L, n_R, p_R`` -- volumetric densities [m^-3] at the
  two sides of the plane;
* electrostatic potential traces ``phi_L, phi_R`` [V];
* trap occupancy ``f`` -- dimensionless, in ``[0, 1]``.

A carrier trace is never an occupancy. The interface sheet charge enters only
through the Gauss jump ``C_L(phi_L - phi_bulk,L) + C_R(phi_R - phi_bulk,R) = sigma``
with the potential jump ``phi_R - phi_L = Delta``. The functions below reproduce
the historical trace residual, its exact solution and the row weights used to
fold an eliminated sheet charge into deduplicated bulk Poisson rows.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from solarlab.contracts import (
    ONE, PARTICLE, VOLT, VOLUME, Layout, Support, VariableSpec,
)
from solarlab.discretization.face_average import VACUUM_PERMITTIVITY_F_M
from solarlab.mesh.geometry import InterfacePlane

__all__ = [
    "CARRIER_TRACE_UNIT", "InterfaceTraceDeclaration", "OCCUPANCY_UNIT", "TraceElectrostatics",
    "add_sheet_charge_to_interior_residual", "electrostatic_trace_residual_and_jacobian",
    "interface_trace_layout", "require_carrier_trace", "require_occupancy",
    "sheet_charge_row_weights", "solve_electrostatic_traces",
]

EPS_0 = VACUUM_PERMITTIVITY_F_M
CARRIER_TRACE_UNIT = PARTICLE / VOLUME
OCCUPANCY_UNIT = ONE

LEFT_TRACE_SUPPORT = "interfaces.left_trace"
RIGHT_TRACE_SUPPORT = "interfaces.right_trace"
PLANE_SUPPORT = "interfaces.plane"


@dataclass(frozen=True)
class InterfaceTraceDeclaration:
    """The separately owned variables of a set of two-sided interfaces."""

    carrier_traces: tuple[VariableSpec, ...]
    potential_traces: tuple[VariableSpec, ...]
    trap_occupancy: VariableSpec
    layout: Layout


def interface_trace_layout(interface_count: int, *,
                           owner: str = "interfaces") -> InterfaceTraceDeclaration:
    """Declare carrier traces, potential traces and occupancy for each interface."""
    if type(interface_count) is not int or interface_count < 1:
        raise ValueError("at least one interface is required")
    shape = (interface_count,)
    supports = (
        Support(LEFT_TRACE_SUPPORT, "face", shape),
        Support(RIGHT_TRACE_SUPPORT, "face", shape),
        Support(PLANE_SUPPORT, "face", shape),
    )
    carriers = tuple(
        VariableSpec(f"{owner}.{name}", owner, support, shape, CARRIER_TRACE_UNIT, lower=0.0)
        for name, support in (("n_left", LEFT_TRACE_SUPPORT), ("p_left", LEFT_TRACE_SUPPORT),
                              ("n_right", RIGHT_TRACE_SUPPORT), ("p_right", RIGHT_TRACE_SUPPORT))
    )
    potentials = (
        VariableSpec(f"{owner}.phi_left", owner, LEFT_TRACE_SUPPORT, shape, VOLT,
                     role="constraint"),
        VariableSpec(f"{owner}.phi_right", owner, RIGHT_TRACE_SUPPORT, shape, VOLT,
                     role="constraint"),
    )
    occupancy = VariableSpec(f"{owner}.trap_occupancy", owner, PLANE_SUPPORT, shape,
                             OCCUPANCY_UNIT, lower=0.0, upper=1.0)
    layout = Layout(supports, carriers + potentials + (occupancy,), ())
    return InterfaceTraceDeclaration(carriers, potentials, occupancy, layout)


def require_occupancy(spec: VariableSpec) -> VariableSpec:
    """Accept only a dimensionless variable bounded to [0, 1] as an occupancy."""
    if spec.unit != OCCUPANCY_UNIT or spec.lower != 0.0 or spec.upper != 1.0:
        raise ValueError(f"{spec.id} is not a trap occupancy in [0, 1]")
    return spec


def require_carrier_trace(spec: VariableSpec) -> VariableSpec:
    """Accept only a non-negative volumetric density as a carrier trace."""
    if spec.unit != CARRIER_TRACE_UNIT or spec.lower != 0.0:
        raise ValueError(f"{spec.id} is not a carrier trace density [m^-3]")
    return spec


@dataclass(frozen=True)
class TraceElectrostatics:
    """Geometry and electrostatic jump data for one physical interface."""

    left_distance_m: float
    right_distance_m: float
    eps_r_left: float
    eps_r_right: float
    potential_jump_right_minus_left_V: float = 0.0
    fixed_sheet_charge_C_m2: float = 0.0

    def __post_init__(self) -> None:
        for name in ("left_distance_m", "right_distance_m", "eps_r_left", "eps_r_right"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("potential_jump_right_minus_left_V", "fixed_sheet_charge_C_m2"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")

    @classmethod
    def from_plane(cls, plane: InterfacePlane, eps_r_left: float, eps_r_right: float, *,
                   potential_jump_V: float = 0.0,
                   sheet_charge_C_m2: float = 0.0) -> TraceElectrostatics:
        return cls(plane.left_distance_m, plane.right_distance_m, eps_r_left, eps_r_right,
                   potential_jump_V, sheet_charge_C_m2)


def electrostatic_trace_residual_and_jacobian(
    trace_potentials_V: ArrayLike,
    geometry: TraceElectrostatics,
    phi_bulk_left_V: float,
    phi_bulk_right_V: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Potential-jump and Gauss-law residuals plus their Jacobian (historical)."""
    values = np.asarray(trace_potentials_V, dtype=float)
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise ValueError("trace_potentials_V must contain two finite values")
    phi_left, phi_right = (float(value) for value in values)
    capacitance_left = EPS_0 * float(geometry.eps_r_left) / float(geometry.left_distance_m)
    capacitance_right = EPS_0 * float(geometry.eps_r_right) / float(geometry.right_distance_m)
    residual = np.array(
        [
            phi_right - phi_left - float(geometry.potential_jump_right_minus_left_V),
            capacitance_left * (phi_left - float(phi_bulk_left_V))
            + capacitance_right * (phi_right - float(phi_bulk_right_V))
            - float(geometry.fixed_sheet_charge_C_m2),
        ],
        dtype=float,
    )
    jacobian = np.array([[-1.0, 1.0], [capacitance_left, capacitance_right]], dtype=float)
    return residual, jacobian


def solve_electrostatic_traces(
    geometry: TraceElectrostatics,
    phi_bulk_left_V: float,
    phi_bulk_right_V: float,
) -> tuple[float, float]:
    """Solve the two linear electrostatic interface constraints exactly."""
    capacitance_left = EPS_0 * float(geometry.eps_r_left) / float(geometry.left_distance_m)
    capacitance_right = EPS_0 * float(geometry.eps_r_right) / float(geometry.right_distance_m)
    jump = float(geometry.potential_jump_right_minus_left_V)
    denominator = capacitance_left + capacitance_right
    phi_left = (
        capacitance_left * float(phi_bulk_left_V)
        + capacitance_right * float(phi_bulk_right_V)
        + float(geometry.fixed_sheet_charge_C_m2)
        - capacitance_right * jump
    ) / denominator
    phi_right = phi_left + jump
    return phi_left, phi_right


def sheet_charge_row_weights(eps_r_left: float, left_distance_m: float,
                             eps_r_right: float, right_distance_m: float) -> tuple[float, float]:
    """Share of an eliminated sheet charge that each adjacent bulk row receives."""
    capacitance_left = EPS_0 * float(eps_r_left) / float(left_distance_m)
    capacitance_right = EPS_0 * float(eps_r_right) / float(right_distance_m)
    capacitance_sum = capacitance_left + capacitance_right
    return capacitance_left / capacitance_sum, capacitance_right / capacitance_sum


def add_sheet_charge_to_interior_residual(
    interior_residual: NDArray[np.float64],
    planes: Sequence[InterfacePlane],
    eps_r_node: ArrayLike,
    sheet_charge_C_m2: ArrayLike,
) -> None:
    """Fold eliminated sheet charges into interior Poisson rows, in place.

    Row ``k`` of ``interior_residual`` belongs to node ``k + 1``. Each plane must
    have adjacent interior bulk nodes, as on a two-sided geometry.
    """
    permittivity = np.asarray(eps_r_node, dtype=float)
    sheets = np.asarray(sheet_charge_C_m2, dtype=float)
    node_count = interior_residual.size + 2
    if sheets.shape != (len(planes),) or permittivity.shape != (node_count,):
        raise ValueError("sheet charges and permittivity must match the planes and nodes")
    for index, plane in enumerate(planes):
        left, right = plane.left_node, plane.right_node
        if left <= 0 or right >= node_count - 1 or right != left + 1:
            raise ValueError("charged two-sided interfaces require adjacent interior nodes")
        weight_left, weight_right = sheet_charge_row_weights(
            float(permittivity[left]), plane.left_distance_m,
            float(permittivity[right]), plane.right_distance_m,
        )
        sheet = sheets[index]
        interior_residual[left - 1] += weight_left * sheet
        interior_residual[right - 1] += weight_right * sheet
