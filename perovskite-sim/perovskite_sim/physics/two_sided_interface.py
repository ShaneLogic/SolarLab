"""Zero-volume, two-sided heterointerface control element.

The production grid currently assigns a shared layer-boundary node to the
right material.  This module defines the local physics needed to replace that
topology: independent left/right carrier traces, explicit electrostatic trace
constraints, one-material Scharfetter-Gummel half-fluxes, reciprocal
Fermi-Dirac Richardson exchange, and shared-occupancy interface SRH.

The element is dispatched only by the opt-in ``two_sided_trace`` quasi-Fermi
and CBO paths. The default deduplicated grid remains unchanged. Its local
carrier state is statically eliminated on every outer residual evaluation, so
no duplicate Scharfetter-Gummel bypass remains at the material boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import least_squares

from perovskite_sim.constants import EPS_0, Q
from perovskite_sim.discretization.fe_operators import bernoulli
from perovskite_sim.physics.fermi_dirac import (
    fermi_dirac_half_log_derivative,
    fermi_dirac_one,
    fermi_dirac_zero,
    inverse_fermi_dirac_half,
)


_STATE_SIZE = 4
_N_LEFT = 0
_P_LEFT = 1
_N_RIGHT = 2
_P_RIGHT = 3

DEDUPLICATED_QSS = "deduplicated_qss"
TWO_SIDED_TRACE = "two_sided_trace"
INTERFACE_TOPOLOGIES = (DEDUPLICATED_QSS, TWO_SIDED_TRACE)


@dataclass(frozen=True)
class TwoSidedInterfaceGeometry:
    """Geometry and electrostatic jump data for one physical interface."""

    left_distance_m: float
    right_distance_m: float
    eps_r_left: float
    eps_r_right: float
    potential_jump_right_minus_left_V: float = 0.0
    fixed_sheet_charge_C_m2: float = 0.0


@dataclass(frozen=True)
class TwoSidedInterfaceStencil:
    """Grid indices bracketing a material boundary with interior reservoirs.

    ``shared_boundary_node`` records the node produced by the current
    deduplicated grid, when present. It is deliberately not used as either
    reservoir because its distance to the physical interface is zero.
    """

    interface_position_m: float
    left_bulk_node: int
    right_bulk_node: int
    left_distance_m: float
    right_distance_m: float
    shared_boundary_node: int | None


@dataclass(frozen=True)
class TwoSidedInterfacePhysics:
    """Carrier parameters on the two material sides.

    ``conduction_band_step_eV`` is ``E_C,R - E_C,L`` at equal electrostatic
    potential. ``hole_transport_step_eV`` is the corresponding right-minus-
    left step of the hole particle barrier, ``(-E_V)_R - (-E_V)_L``.
    """

    thermal_voltage_V: float
    temperature_K: float
    D_n_left_m2_s: float
    D_n_right_m2_s: float
    D_p_left_m2_s: float
    D_p_right_m2_s: float
    N_C_left_m3: float
    N_C_right_m3: float
    N_V_left_m3: float
    N_V_right_m3: float
    richardson_n_A_m2_K2: float
    richardson_p_A_m2_K2: float
    conduction_band_step_eV: float = 0.0
    hole_transport_step_eV: float = 0.0
    transmission: float = 1.0
    surface_recombination_velocity_n_m_s: float = 0.0
    surface_recombination_velocity_p_m_s: float = 0.0
    n1_left_m3: float = 0.0
    n1_right_m3: float = 0.0
    p1_left_m3: float = 0.0
    p1_right_m3: float = 0.0


@dataclass(frozen=True)
class TwoSidedBulkState:
    """Adjacent bulk-reservoir values bracketing the interface."""

    phi_left_V: float
    phi_right_V: float
    n_left_m3: float
    p_left_m3: float
    n_right_m3: float
    p_right_m3: float


@dataclass(frozen=True)
class InterfaceTracePotentials:
    """Independent electrostatic traces on the two interface sides."""

    phi_left_V: float
    phi_right_V: float


@dataclass(frozen=True)
class EquilibriumReferencedSheetCharge:
    """Research-only incremental interface-charge closure.

    The physical convention is fixed to ``Delta sigma = -q Nt (f - f_eq)``.
    No trap-character sign is accepted because donor-like and acceptor-like
    traps have the same equilibrium-referenced increment.
    """

    equilibrium_occupancy: float
    trap_density_m2: float

    def __post_init__(self) -> None:
        occupancy = float(self.equilibrium_occupancy)
        density = float(self.trap_density_m2)
        if not math.isfinite(occupancy) or not 0.0 <= occupancy <= 1.0:
            raise ValueError("equilibrium_occupancy must lie in [0, 1]")
        if not math.isfinite(density) or density < 0.0:
            raise ValueError("trap_density_m2 must be finite and non-negative")


@dataclass(frozen=True)
class TwoSidedCarrierBalance:
    """Raw carrier balance and its log-density Jacobian.

    All vectors use ``(n_L, p_L, n_R, p_R)`` ordering. Fluxes are positive
    into the corresponding zero-volume state. Captures are positive out of a
    state, so ``residual = bulk + cross - capture``.
    """

    state_m3: np.ndarray
    bulk_flux_m2_s: np.ndarray
    cross_flux_m2_s: np.ndarray
    capture_flux_m2_s: np.ndarray
    residual_m2_s: np.ndarray
    jacobian_log_state_m2_s: np.ndarray
    one_way_cross_scale_m2_s: np.ndarray


@dataclass(frozen=True)
class ChargedElectrostaticTraceBalance:
    """Potential-jump/Gauss balance with occupancy-dependent sheet charge.

    Jacobian columns use ``(phi_L, phi_R, log n_L, log p_L, log n_R,
    log p_R)`` ordering. The incremental charge is reported separately from
    any fixed sheet charge already carried by the geometry.
    """

    occupancy: float
    incremental_sheet_charge_C_m2: float
    residual: np.ndarray
    jacobian_trace_and_log_state: np.ndarray


@dataclass(frozen=True)
class TwoSidedInterfaceResult:
    """Locally eliminated two-sided trace state and certificate."""

    potentials: InterfaceTracePotentials
    state_m3: np.ndarray
    bulk_flux_m2_s: np.ndarray
    cross_flux_m2_s: np.ndarray
    capture_flux_m2_s: np.ndarray
    residual_m2_s: np.ndarray
    jacobian_log_state_m2_s: np.ndarray
    normalized_residual: float
    evaluations: int
    converged: bool


@dataclass(frozen=True)
class TwoSidedMaterialQSSResult:
    """Multi-interface adapter using the legacy right-first block layout."""

    state_m3: np.ndarray
    bulk_flux_m2_s: np.ndarray
    cross_flux_m2_s: np.ndarray
    state_flux_m2_s: np.ndarray
    normalized_residual: float
    evaluations: int
    transport_model: str
    capture_flux_m2_s: np.ndarray | None = None
    occupancy: np.ndarray | None = None


def validate_interface_topology(topology: str) -> str:
    """Return a normalized, supported interface topology name."""
    normalized = str(topology).strip().lower()
    if normalized not in INTERFACE_TOPOLOGIES:
        raise ValueError(
            "interface_topology must be one of "
            f"{INTERFACE_TOPOLOGIES}; got {topology!r}"
        )
    return normalized


def _coordinate_tolerance(grid: np.ndarray) -> float:
    return 32.0 * np.finfo(float).eps * max(
        1.0,
        abs(float(grid[0])),
        abs(float(grid[-1])),
        abs(float(grid[-1] - grid[0])),
    )


def remove_shared_interface_nodes(
    grid_m: np.ndarray,
    interface_positions_m: np.ndarray | tuple[float, ...] | list[float],
) -> np.ndarray:
    """Return a strict grid with deduplicated material-boundary nodes removed.

    Only nodes coincident with declared internal interfaces are removed. Outer
    contacts and every intra-layer point remain unchanged. The returned grid
    is suitable for a two-sided trace element whose left and right reservoirs
    must lie strictly inside their respective materials.
    """
    grid = np.asarray(grid_m, dtype=float)
    positions = np.asarray(interface_positions_m, dtype=float)
    if (
        grid.ndim != 1
        or grid.size < 3
        or not np.all(np.isfinite(grid))
        or np.any(np.diff(grid) <= 0.0)
    ):
        raise ValueError("grid_m must be finite and strictly increasing")
    if positions.ndim != 1 or not np.all(np.isfinite(positions)):
        raise ValueError("interface_positions_m must be a finite vector")
    if positions.size and np.any(np.diff(positions) <= 0.0):
        raise ValueError("interface_positions_m must be strictly increasing")
    tolerance = _coordinate_tolerance(grid)
    keep = np.ones(grid.size, dtype=bool)
    for position_value in positions:
        position = float(position_value)
        if not float(grid[0]) < position < float(grid[-1]):
            raise ValueError("each interface position must lie inside grid_m")
        nearest = int(np.argmin(np.abs(grid - position)))
        if abs(float(grid[nearest]) - position) <= tolerance:
            keep[nearest] = False
    reduced = grid[keep]
    if reduced.size < 3:
        raise ValueError("removing interface nodes leaves fewer than three nodes")
    stencils = build_two_sided_interface_stencils(reduced, positions)
    if any(stencil.shared_boundary_node is not None for stencil in stencils):
        raise RuntimeError("shared interface node remained after grid reduction")
    return reduced


def build_two_sided_interface_stencils(
    grid_m: np.ndarray,
    interface_positions_m: np.ndarray | tuple[float, ...] | list[float],
) -> tuple[TwoSidedInterfaceStencil, ...]:
    """Map physical interfaces to the nearest strict left/right bulk nodes.

    A node lying exactly on an interface is classified separately as the
    legacy shared boundary node. The two reservoirs are always selected with
    ``x_left < x_interface < x_right``, which prevents a zero-distance SG
    half-cell and makes the geometry independent of layer ownership.
    """
    grid = np.asarray(grid_m, dtype=float)
    positions = np.asarray(interface_positions_m, dtype=float)
    if (
        grid.ndim != 1
        or grid.size < 3
        or not np.all(np.isfinite(grid))
        or np.any(np.diff(grid) <= 0.0)
    ):
        raise ValueError("grid_m must be finite and strictly increasing")
    if positions.ndim != 1 or not np.all(np.isfinite(positions)):
        raise ValueError("interface_positions_m must be a finite vector")
    if positions.size and np.any(np.diff(positions) <= 0.0):
        raise ValueError("interface_positions_m must be strictly increasing")

    tolerance = _coordinate_tolerance(grid)
    stencils: list[TwoSidedInterfaceStencil] = []
    for position_value in positions:
        position = float(position_value)
        if not float(grid[0]) < position < float(grid[-1]):
            raise ValueError("each interface position must lie inside grid_m")
        nearest = int(np.argmin(np.abs(grid - position)))
        shared = (
            nearest if abs(float(grid[nearest]) - position) <= tolerance else None
        )
        if shared is None:
            right = int(np.searchsorted(grid, position, side="right"))
            left = right - 1
        else:
            left = shared - 1
            right = shared + 1
        if left < 0 or right >= grid.size:
            raise ValueError("interface lacks strict left/right bulk nodes")
        left_distance = position - float(grid[left])
        right_distance = float(grid[right]) - position
        if left_distance <= 0.0 or right_distance <= 0.0:
            raise ValueError("interface stencil contains a zero-distance half-cell")
        stencils.append(
            TwoSidedInterfaceStencil(
                interface_position_m=position,
                left_bulk_node=left,
                right_bulk_node=right,
                left_distance_m=left_distance,
                right_distance_m=right_distance,
                shared_boundary_node=shared,
            )
        )
    return tuple(stencils)


def _require_positive(name: str, value: float) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _require_nonnegative(name: str, value: float) -> None:
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


def _validate_inputs(
    geometry: TwoSidedInterfaceGeometry,
    physics: TwoSidedInterfacePhysics,
    bulk: TwoSidedBulkState,
) -> None:
    for name in (
        "left_distance_m",
        "right_distance_m",
        "eps_r_left",
        "eps_r_right",
    ):
        _require_positive(name, getattr(geometry, name))
    for name in (
        "potential_jump_right_minus_left_V",
        "fixed_sheet_charge_C_m2",
    ):
        if not math.isfinite(float(getattr(geometry, name))):
            raise ValueError(f"{name} must be finite")
    for name in (
        "thermal_voltage_V",
        "temperature_K",
        "D_n_left_m2_s",
        "D_n_right_m2_s",
        "D_p_left_m2_s",
        "D_p_right_m2_s",
        "N_C_left_m3",
        "N_C_right_m3",
        "N_V_left_m3",
        "N_V_right_m3",
        "richardson_n_A_m2_K2",
        "richardson_p_A_m2_K2",
    ):
        _require_positive(name, getattr(physics, name))
    if (
        not math.isfinite(float(physics.transmission))
        or not 0.0 < float(physics.transmission) <= 1.0
    ):
        raise ValueError("transmission must lie in (0, 1]")
    for name in (
        "conduction_band_step_eV",
        "hole_transport_step_eV",
    ):
        if not math.isfinite(float(getattr(physics, name))):
            raise ValueError(f"{name} must be finite")
    for name in (
        "surface_recombination_velocity_n_m_s",
        "surface_recombination_velocity_p_m_s",
        "n1_left_m3",
        "n1_right_m3",
        "p1_left_m3",
        "p1_right_m3",
    ):
        _require_nonnegative(name, getattr(physics, name))
    for name in (
        "n_left_m3",
        "p_left_m3",
        "n_right_m3",
        "p_right_m3",
    ):
        _require_positive(name, getattr(bulk, name))
    for name in ("phi_left_V", "phi_right_V"):
        if not math.isfinite(float(getattr(bulk, name))):
            raise ValueError(f"{name} must be finite")


def electrostatic_trace_residual_and_jacobian(
    trace_potentials_V: np.ndarray,
    geometry: TwoSidedInterfaceGeometry,
    bulk: TwoSidedBulkState,
) -> tuple[np.ndarray, np.ndarray]:
    """Return potential-jump and Gauss-law residuals plus their Jacobian."""
    values = np.asarray(trace_potentials_V, dtype=float)
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise ValueError("trace_potentials_V must contain two finite values")
    phi_left, phi_right = (float(value) for value in values)
    capacitance_left = (
        EPS_0 * float(geometry.eps_r_left) / float(geometry.left_distance_m)
    )
    capacitance_right = (
        EPS_0 * float(geometry.eps_r_right) / float(geometry.right_distance_m)
    )
    residual = np.array(
        [
            phi_right
            - phi_left
            - float(geometry.potential_jump_right_minus_left_V),
            capacitance_left * (phi_left - float(bulk.phi_left_V))
            + capacitance_right * (phi_right - float(bulk.phi_right_V))
            - float(geometry.fixed_sheet_charge_C_m2),
        ],
        dtype=float,
    )
    jacobian = np.array(
        [
            [-1.0, 1.0],
            [capacitance_left, capacitance_right],
        ],
        dtype=float,
    )
    return residual, jacobian


def solve_electrostatic_traces(
    geometry: TwoSidedInterfaceGeometry,
    bulk: TwoSidedBulkState,
) -> InterfaceTracePotentials:
    """Solve the two linear electrostatic interface constraints exactly."""
    capacitance_left = (
        EPS_0 * float(geometry.eps_r_left) / float(geometry.left_distance_m)
    )
    capacitance_right = (
        EPS_0 * float(geometry.eps_r_right) / float(geometry.right_distance_m)
    )
    jump = float(geometry.potential_jump_right_minus_left_V)
    denominator = capacitance_left + capacitance_right
    phi_left = (
        capacitance_left * float(bulk.phi_left_V)
        + capacitance_right * float(bulk.phi_right_V)
        + float(geometry.fixed_sheet_charge_C_m2)
        - capacitance_right * jump
    ) / denominator
    phi_right = phi_left + jump
    return InterfaceTracePotentials(phi_left, phi_right)


def _bernoulli_pair(value: float) -> tuple[float, float]:
    pair = bernoulli(np.array([float(value), -float(value)]))
    return float(pair[0]), float(pair[1])


def _bulk_flux_and_log_jacobian(
    state: np.ndarray,
    traces: InterfaceTracePotentials,
    geometry: TwoSidedInterfaceGeometry,
    physics: TwoSidedInterfacePhysics,
    bulk: TwoSidedBulkState,
) -> tuple[np.ndarray, np.ndarray]:
    n_left, p_left, n_right, p_right = state
    thermal_voltage = float(physics.thermal_voltage_V)
    xi_left = (
        float(traces.phi_left_V) - float(bulk.phi_left_V)
    ) / thermal_voltage
    xi_right = (
        float(bulk.phi_right_V) - float(traces.phi_right_V)
    ) / thermal_voltage
    b_left, b_minus_left = _bernoulli_pair(xi_left)
    b_right, b_minus_right = _bernoulli_pair(xi_right)
    k_n_left = float(physics.D_n_left_m2_s) / float(geometry.left_distance_m)
    k_n_right = float(physics.D_n_right_m2_s) / float(
        geometry.right_distance_m
    )
    k_p_left = float(physics.D_p_left_m2_s) / float(geometry.left_distance_m)
    k_p_right = float(physics.D_p_right_m2_s) / float(
        geometry.right_distance_m
    )

    # Convert conventional SG currents to particle fluxes into each trace.
    flux = np.array(
        [
            k_n_left * (b_minus_left * bulk.n_left_m3 - b_left * n_left),
            k_p_left * (b_left * bulk.p_left_m3 - b_minus_left * p_left),
            k_n_right * (b_right * bulk.n_right_m3 - b_minus_right * n_right),
            k_p_right * (b_minus_right * bulk.p_right_m3 - b_right * p_right),
        ],
        dtype=float,
    )
    jacobian = np.diag(
        [
            -k_n_left * b_left * n_left,
            -k_p_left * b_minus_left * p_left,
            -k_n_right * b_minus_right * n_right,
            -k_p_right * b_right * p_right,
        ]
    )
    return flux, jacobian


def _one_way_fermi_supply(
    density_m3: float,
    density_of_states_m3: float,
    barrier_eV: float,
    thermal_voltage_V: float,
) -> tuple[float, float]:
    ratio = float(density_m3) / float(density_of_states_m3)
    eta = inverse_fermi_dirac_half(ratio)
    argument = eta - float(barrier_eV) / float(thermal_voltage_V)
    supply = fermi_dirac_one(argument)
    log_slope = fermi_dirac_half_log_derivative(eta)
    if not math.isfinite(log_slope) or log_slope <= 0.0:
        raise FloatingPointError("invalid Fermi-Dirac constitutive derivative")
    derivative_log_density = fermi_dirac_zero(argument) / log_slope
    return supply, derivative_log_density


def _cross_flux_and_log_derivatives(
    state: np.ndarray,
    traces: InterfaceTracePotentials,
    physics: TwoSidedInterfacePhysics,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    potential_jump = float(traces.phi_right_V - traces.phi_left_V)
    electron_step = float(physics.conduction_band_step_eV) - potential_jump
    hole_step = float(physics.hole_transport_step_eV) + potential_jump
    thermal_voltage = float(physics.thermal_voltage_V)

    n_left_supply, dn_left = _one_way_fermi_supply(
        state[_N_LEFT],
        physics.N_C_left_m3,
        max(electron_step, 0.0),
        thermal_voltage,
    )
    n_right_supply, dn_right = _one_way_fermi_supply(
        state[_N_RIGHT],
        physics.N_C_right_m3,
        max(-electron_step, 0.0),
        thermal_voltage,
    )
    p_left_supply, dp_left = _one_way_fermi_supply(
        state[_P_LEFT],
        physics.N_V_left_m3,
        max(hole_step, 0.0),
        thermal_voltage,
    )
    p_right_supply, dp_right = _one_way_fermi_supply(
        state[_P_RIGHT],
        physics.N_V_right_m3,
        max(-hole_step, 0.0),
        thermal_voltage,
    )
    prefactor_n = (
        float(physics.transmission)
        * float(physics.richardson_n_A_m2_K2)
        * float(physics.temperature_K) ** 2
        / Q
    )
    prefactor_p = (
        float(physics.transmission)
        * float(physics.richardson_p_A_m2_K2)
        * float(physics.temperature_K) ** 2
        / Q
    )
    cross_n = prefactor_n * (n_left_supply - n_right_supply)
    cross_p = prefactor_p * (p_left_supply - p_right_supply)
    derivative = np.zeros((2, _STATE_SIZE), dtype=float)
    derivative[0, _N_LEFT] = prefactor_n * dn_left
    derivative[0, _N_RIGHT] = -prefactor_n * dn_right
    derivative[1, _P_LEFT] = prefactor_p * dp_left
    derivative[1, _P_RIGHT] = -prefactor_p * dp_right
    one_way_scale = np.array(
        [
            prefactor_n * max(n_left_supply, n_right_supply),
            prefactor_p * max(p_left_supply, p_right_supply),
        ],
        dtype=float,
    )
    return np.array([cross_n, cross_p]), derivative, one_way_scale


def _shared_trap_occupancy_and_denominator(
    state: np.ndarray,
    physics: TwoSidedInterfacePhysics,
) -> tuple[float, float]:
    velocity_n = float(physics.surface_recombination_velocity_n_m_s)
    velocity_p = float(physics.surface_recombination_velocity_p_m_s)
    if velocity_n == 0.0 and velocity_p == 0.0:
        raise ValueError("shared-trap occupancy is undefined without capture")

    n1 = np.array([physics.n1_left_m3, physics.n1_right_m3], dtype=float)
    p1 = np.array([physics.p1_left_m3, physics.p1_right_m3], dtype=float)
    n_total = state[_N_LEFT] + state[_N_RIGHT]
    p_total = state[_P_LEFT] + state[_P_RIGHT]
    numerator = velocity_n * n_total + velocity_p * float(np.sum(p1))
    denominator = velocity_n * (n_total + float(np.sum(n1))) + velocity_p * (
        p_total + float(np.sum(p1))
    )
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise FloatingPointError("invalid shared-trap occupancy denominator")
    occupancy = numerator / denominator
    if not math.isfinite(occupancy) or not 0.0 <= occupancy <= 1.0:
        raise FloatingPointError("shared-trap occupancy left [0, 1]")
    return occupancy, denominator


def shared_trap_occupancy(
    state_m3: np.ndarray,
    physics: TwoSidedInterfacePhysics,
) -> float:
    """Return the common electron occupancy of one two-sided interface trap."""
    state = np.asarray(state_m3, dtype=float)
    if state.shape != (_STATE_SIZE,) or not np.all(np.isfinite(state)):
        raise ValueError("state_m3 must contain four finite values")
    if np.any(state < 0.0):
        raise ValueError("state_m3 must be non-negative")
    occupancy, _ = _shared_trap_occupancy_and_denominator(state, physics)
    return occupancy


def _shared_trap_occupancy_and_linear_jacobian(
    state: np.ndarray,
    physics: TwoSidedInterfacePhysics,
) -> tuple[float, np.ndarray]:
    velocity_n = float(physics.surface_recombination_velocity_n_m_s)
    velocity_p = float(physics.surface_recombination_velocity_p_m_s)
    occupancy, denominator = _shared_trap_occupancy_and_denominator(state, physics)
    derivative = np.array(
        [
            velocity_n * (1.0 - occupancy) / denominator,
            -velocity_p * occupancy / denominator,
            velocity_n * (1.0 - occupancy) / denominator,
            -velocity_p * occupancy / denominator,
        ],
        dtype=float,
    )
    return occupancy, derivative


def shared_trap_occupancy_and_log_jacobian(
    state_m3: np.ndarray,
    physics: TwoSidedInterfacePhysics,
) -> tuple[float, np.ndarray]:
    """Return shared occupancy and ``d f / d log(state_m3)``."""
    state = np.asarray(state_m3, dtype=float)
    if state.shape != (_STATE_SIZE,) or not np.all(np.isfinite(state)):
        raise ValueError("state_m3 must contain four finite values")
    if np.any(state < 0.0):
        raise ValueError("state_m3 must be non-negative")
    occupancy, derivative_linear = _shared_trap_occupancy_and_linear_jacobian(
        state, physics
    )
    return occupancy, derivative_linear * state


def _capture_flux_and_log_jacobian(
    state: np.ndarray,
    physics: TwoSidedInterfacePhysics,
) -> tuple[np.ndarray, np.ndarray]:
    velocity_n = float(physics.surface_recombination_velocity_n_m_s)
    velocity_p = float(physics.surface_recombination_velocity_p_m_s)
    if velocity_n == 0.0 and velocity_p == 0.0:
        return np.zeros(_STATE_SIZE), np.zeros((_STATE_SIZE, _STATE_SIZE))

    n1 = np.array([physics.n1_left_m3, physics.n1_right_m3], dtype=float)
    p1 = np.array([physics.p1_left_m3, physics.p1_right_m3], dtype=float)
    occupancy, derivative_occupancy = (
        _shared_trap_occupancy_and_linear_jacobian(state, physics)
    )

    capture = np.empty(_STATE_SIZE, dtype=float)
    capture[_N_LEFT] = velocity_n * (
        state[_N_LEFT] * (1.0 - occupancy) - n1[0] * occupancy
    )
    capture[_P_LEFT] = velocity_p * (
        state[_P_LEFT] * occupancy - p1[0] * (1.0 - occupancy)
    )
    capture[_N_RIGHT] = velocity_n * (
        state[_N_RIGHT] * (1.0 - occupancy) - n1[1] * occupancy
    )
    capture[_P_RIGHT] = velocity_p * (
        state[_P_RIGHT] * occupancy - p1[1] * (1.0 - occupancy)
    )

    jacobian_linear = np.empty((_STATE_SIZE, _STATE_SIZE), dtype=float)
    for row, trap_density in ((_N_LEFT, n1[0]), (_N_RIGHT, n1[1])):
        for column in range(_STATE_SIZE):
            direct = 1.0 - occupancy if column == row else 0.0
            jacobian_linear[row, column] = velocity_n * (
                direct
                - (state[row] + trap_density) * derivative_occupancy[column]
            )
    for row, trap_density in ((_P_LEFT, p1[0]), (_P_RIGHT, p1[1])):
        for column in range(_STATE_SIZE):
            direct = occupancy if column == row else 0.0
            jacobian_linear[row, column] = velocity_p * (
                direct
                + (state[row] + trap_density) * derivative_occupancy[column]
            )
    return capture, jacobian_linear * state[np.newaxis, :]


def equilibrium_referenced_electrostatic_trace_balance(
    trace_and_log_state: np.ndarray,
    geometry: TwoSidedInterfaceGeometry,
    physics: TwoSidedInterfacePhysics,
    bulk: TwoSidedBulkState,
    charge: EquilibriumReferencedSheetCharge,
) -> ChargedElectrostaticTraceBalance:
    """Evaluate the charged two-sided Gauss law and its analytic tangent.

    This is a local research primitive. It does not enable interface charge in
    a device solve; the outer Poisson system must consume this residual and
    tangent explicitly before the closure is self-consistent.
    """
    _validate_inputs(geometry, physics, bulk)
    values = np.asarray(trace_and_log_state, dtype=float)
    if values.shape != (2 + _STATE_SIZE,) or not np.all(np.isfinite(values)):
        raise ValueError("trace_and_log_state must contain six finite values")
    state = np.exp(values[2:])
    if not np.all(np.isfinite(state)) or np.any(state <= 0.0):
        raise FloatingPointError("interface trace densities left finite range")
    occupancy, occupancy_jacobian = shared_trap_occupancy_and_log_jacobian(
        state, physics
    )
    from perovskite_sim.physics.interface_plane import (
        equilibrium_referenced_interface_trap_charge,
    )

    incremental_charge = float(
        equilibrium_referenced_interface_trap_charge(
            occupancy,
            charge.equilibrium_occupancy,
            charge.trap_density_m2,
        )
    )
    residual, trace_jacobian = electrostatic_trace_residual_and_jacobian(
        values[:2], geometry, bulk
    )
    residual[1] -= incremental_charge
    jacobian = np.zeros((2, 2 + _STATE_SIZE), dtype=float)
    jacobian[:, :2] = trace_jacobian
    jacobian[1, 2:] = Q * charge.trap_density_m2 * occupancy_jacobian
    return ChargedElectrostaticTraceBalance(
        occupancy=occupancy,
        incremental_sheet_charge_C_m2=incremental_charge,
        residual=residual,
        jacobian_trace_and_log_state=jacobian,
    )


def carrier_balance_and_jacobian(
    log_state: np.ndarray,
    geometry: TwoSidedInterfaceGeometry,
    physics: TwoSidedInterfacePhysics,
    bulk: TwoSidedBulkState,
    traces: InterfaceTracePotentials | None = None,
) -> TwoSidedCarrierBalance:
    """Evaluate the four algebraic carrier balances and analytic Jacobian."""
    values = np.asarray(log_state, dtype=float)
    if values.shape != (_STATE_SIZE,) or not np.all(np.isfinite(values)):
        raise ValueError("log_state must contain four finite values")
    trace_values = traces or solve_electrostatic_traces(geometry, bulk)
    state = np.exp(values)
    if not np.all(np.isfinite(state)) or np.any(state <= 0.0):
        raise FloatingPointError("interface trace densities left finite range")
    bulk_flux, bulk_jacobian = _bulk_flux_and_log_jacobian(
        state, trace_values, geometry, physics, bulk
    )
    cross_pair, cross_derivative, one_way_scale = (
        _cross_flux_and_log_derivatives(state, trace_values, physics)
    )
    cross_flux = np.array(
        [-cross_pair[0], -cross_pair[1], cross_pair[0], cross_pair[1]],
        dtype=float,
    )
    cross_jacobian = np.zeros((_STATE_SIZE, _STATE_SIZE), dtype=float)
    cross_jacobian[_N_LEFT] = -cross_derivative[0]
    cross_jacobian[_N_RIGHT] = cross_derivative[0]
    cross_jacobian[_P_LEFT] = -cross_derivative[1]
    cross_jacobian[_P_RIGHT] = cross_derivative[1]
    capture_flux, capture_jacobian = _capture_flux_and_log_jacobian(
        state, physics
    )
    residual = bulk_flux + cross_flux - capture_flux
    jacobian = bulk_jacobian + cross_jacobian - capture_jacobian
    return TwoSidedCarrierBalance(
        state_m3=state,
        bulk_flux_m2_s=bulk_flux,
        cross_flux_m2_s=cross_flux,
        capture_flux_m2_s=capture_flux,
        residual_m2_s=residual,
        jacobian_log_state_m2_s=jacobian,
        one_way_cross_scale_m2_s=one_way_scale,
    )


def _zero_bulk_flux_seed(
    traces: InterfaceTracePotentials,
    physics: TwoSidedInterfacePhysics,
    bulk: TwoSidedBulkState,
) -> np.ndarray:
    thermal_voltage = float(physics.thermal_voltage_V)
    left_drop = (float(traces.phi_left_V) - bulk.phi_left_V) / thermal_voltage
    right_drop = (float(traces.phi_right_V) - bulk.phi_right_V) / thermal_voltage
    return np.array(
        [
            bulk.n_left_m3 * math.exp(np.clip(left_drop, -100.0, 100.0)),
            bulk.p_left_m3 * math.exp(np.clip(-left_drop, -100.0, 100.0)),
            bulk.n_right_m3 * math.exp(np.clip(right_drop, -100.0, 100.0)),
            bulk.p_right_m3 * math.exp(np.clip(-right_drop, -100.0, 100.0)),
        ],
        dtype=float,
    )


def solve_two_sided_interface(
    geometry: TwoSidedInterfaceGeometry,
    physics: TwoSidedInterfacePhysics,
    bulk: TwoSidedBulkState,
    *,
    initial_state_m3: np.ndarray | None = None,
    residual_tolerance: float = 1.0e-9,
    max_evaluations: int = 200,
    fail_on_residual: bool = True,
) -> TwoSidedInterfaceResult:
    """Eliminate both electrostatic traces and four carrier traces locally."""
    _validate_inputs(geometry, physics, bulk)
    _require_positive("residual_tolerance", residual_tolerance)
    if int(max_evaluations) <= 0:
        raise ValueError("max_evaluations must be positive")
    traces = solve_electrostatic_traces(geometry, bulk)
    seed = (
        _zero_bulk_flux_seed(traces, physics, bulk)
        if initial_state_m3 is None
        else np.asarray(initial_state_m3, dtype=float)
    )
    if seed.shape != (_STATE_SIZE,) or not np.all(np.isfinite(seed)):
        raise ValueError("initial_state_m3 must contain four finite values")
    if np.any(seed <= 0.0):
        raise ValueError("initial_state_m3 must be strictly positive")
    density_reference = max(
        float(np.max(seed)),
        physics.N_C_left_m3,
        physics.N_C_right_m3,
        physics.N_V_left_m3,
        physics.N_V_right_m3,
        physics.n1_left_m3,
        physics.n1_right_m3,
        physics.p1_left_m3,
        physics.p1_right_m3,
        1.0,
    )
    lower = math.log(1.0e-100)
    upper = math.log(density_reference) + math.log(1.0e12)
    log_seed = np.clip(np.log(seed), lower + 1.0, upper - 1.0)

    seed_balance = carrier_balance_and_jacobian(
        log_seed, geometry, physics, bulk, traces
    )
    electron_scale = max(
        abs(seed_balance.bulk_flux_m2_s[_N_LEFT]),
        abs(seed_balance.bulk_flux_m2_s[_N_RIGHT]),
        abs(seed_balance.capture_flux_m2_s[_N_LEFT]),
        abs(seed_balance.capture_flux_m2_s[_N_RIGHT]),
        seed_balance.one_way_cross_scale_m2_s[0],
        1.0,
    )
    hole_scale = max(
        abs(seed_balance.bulk_flux_m2_s[_P_LEFT]),
        abs(seed_balance.bulk_flux_m2_s[_P_RIGHT]),
        abs(seed_balance.capture_flux_m2_s[_P_LEFT]),
        abs(seed_balance.capture_flux_m2_s[_P_RIGHT]),
        seed_balance.one_way_cross_scale_m2_s[1],
        1.0,
    )
    row_scale = np.array(
        [electron_scale, hole_scale, electron_scale, hole_scale], dtype=float
    )

    def residual(log_state: np.ndarray) -> np.ndarray:
        balance = carrier_balance_and_jacobian(
            log_state, geometry, physics, bulk, traces
        )
        return balance.residual_m2_s / row_scale

    def jacobian(log_state: np.ndarray) -> np.ndarray:
        balance = carrier_balance_and_jacobian(
            log_state, geometry, physics, bulk, traces
        )
        return balance.jacobian_log_state_m2_s / row_scale[:, np.newaxis]

    solution = least_squares(
        residual,
        log_seed,
        jac=jacobian,
        bounds=(lower, upper),
        x_scale="jac",
        ftol=1.0e-13,
        xtol=1.0e-13,
        gtol=1.0e-13,
        max_nfev=int(max_evaluations),
    )
    balance = carrier_balance_and_jacobian(
        solution.x, geometry, physics, bulk, traces
    )
    normalized_residual = float(
        np.max(np.abs(balance.residual_m2_s / row_scale))
    )
    converged = bool(solution.success and normalized_residual <= residual_tolerance)
    if fail_on_residual and not converged:
        raise RuntimeError(
            "two-sided interface QSS did not certify: "
            f"solver_success={solution.success}, "
            f"normalized_residual={normalized_residual:.6g}, "
            f"evaluations={solution.nfev}"
        )
    return TwoSidedInterfaceResult(
        potentials=traces,
        state_m3=balance.state_m3,
        bulk_flux_m2_s=balance.bulk_flux_m2_s,
        cross_flux_m2_s=balance.cross_flux_m2_s,
        capture_flux_m2_s=balance.capture_flux_m2_s,
        residual_m2_s=balance.residual_m2_s,
        jacobian_log_state_m2_s=balance.jacobian_log_state_m2_s,
        normalized_residual=normalized_residual,
        evaluations=int(solution.nfev),
        converged=converged,
    )


def solve_material_two_sided_interfaces_qss(
    mat,
    stack,
    n: np.ndarray,
    p: np.ndarray,
    phi: np.ndarray,
    *,
    cross_transmission: float,
    interface_transport_model: str,
    residual_tolerance: float = 1.0e-7,
    max_evaluations: int = 200,
    fail_on_residual: bool = True,
) -> TwoSidedMaterialQSSResult:
    """Eliminate every configured two-sided material interface locally.

    The local primitive currently implements the reciprocal 3D Fermi-Dirac
    Richardson law only. Returning the established right-first block layout
    lets the device continuity assembly reuse its audited conservative drain
    mapping while the grid topology itself changes.
    """
    from perovskite_sim.models.device import electrical_interfaces
    from perovskite_sim.physics.interface_plane import FERMI_DIRAC_RICHARDSON

    model = str(interface_transport_model).strip().lower()
    if model != FERMI_DIRAC_RICHARDSON:
        raise ValueError(
            "two_sided_trace currently requires "
            f"interface_transport_model={FERMI_DIRAC_RICHARDSON!r}"
        )
    left_nodes = tuple(int(value) for value in mat.iface_qss_left_nodes)
    right_nodes = tuple(int(value) for value in mat.iface_qss_right_nodes)
    faces = tuple(int(value) for value in mat.iface_qss_interface_faces)
    positions = tuple(float(value) for value in mat.iface_qss_interface_positions_m)
    left_distances = tuple(
        float(value) for value in mat.iface_qss_left_distances_m
    )
    right_distances = tuple(
        float(value) for value in mat.iface_qss_right_distances_m
    )
    interface_count = len(left_nodes)
    aligned = (
        right_nodes,
        faces,
        positions,
        left_distances,
        right_distances,
    )
    if any(len(values) != interface_count for values in aligned):
        raise ValueError("two-sided interface topology arrays are not aligned")
    if interface_count == 0:
        empty = np.zeros(0, dtype=float)
        return TwoSidedMaterialQSSResult(
            state_m3=empty,
            bulk_flux_m2_s=empty,
            cross_flux_m2_s=empty,
            state_flux_m2_s=empty,
            normalized_residual=0.0,
            evaluations=0,
            transport_model=model,
            capture_flux_m2_s=empty,
            occupancy=empty,
        )
    if mat.D_n_node is None or mat.D_p_node is None:
        raise ValueError("two-sided interface transport requires nodal diffusivity")
    required_arrays = (
        mat.N_C_physical,
        mat.N_V_physical,
        mat.A_star_n,
        mat.A_star_p,
    )
    if any(values is None for values in required_arrays):
        raise ValueError("two-sided interface transport requires DOS and A-star data")
    chi = mat.chi_phys if mat.chi_phys is not None else mat.chi
    Eg = mat.Eg_phys if mat.Eg_phys is not None else mat.Eg
    interface_velocities = electrical_interfaces(stack)

    size = 4 * interface_count
    state = np.empty(size, dtype=float)
    bulk_flux = np.empty(size, dtype=float)
    cross_flux = np.empty(size, dtype=float)
    capture_flux = np.empty(size, dtype=float)
    residual = np.empty(size, dtype=float)
    occupancy = np.empty(interface_count, dtype=float)
    maximum_residual = 0.0
    evaluations = 0
    for index, (left, right) in enumerate(zip(left_nodes, right_nodes)):
        if right != left + 1 or faces[index] != left:
            raise ValueError("two-sided interface reservoirs must share one face")
        velocity_n = velocity_p = 0.0
        if index < len(interface_velocities):
            velocity_n, velocity_p = (
                float(value) for value in interface_velocities[index]
            )
        if index < len(mat.interface_calibration_factor):
            calibration = float(mat.interface_calibration_factor[index])
            velocity_n *= calibration
            velocity_p *= calibration

        def trap_value(name: str) -> float:
            values = getattr(mat, name)
            return float(values[index]) if index < len(values) else 0.0

        geometry = TwoSidedInterfaceGeometry(
            left_distance_m=left_distances[index],
            right_distance_m=right_distances[index],
            eps_r_left=float(mat.eps_r[left]),
            eps_r_right=float(mat.eps_r[right]),
        )
        physics = TwoSidedInterfacePhysics(
            thermal_voltage_V=float(mat.V_T_device),
            temperature_K=float(mat.T_device),
            D_n_left_m2_s=float(mat.D_n_node[left]),
            D_n_right_m2_s=float(mat.D_n_node[right]),
            D_p_left_m2_s=float(mat.D_p_node[left]),
            D_p_right_m2_s=float(mat.D_p_node[right]),
            N_C_left_m3=float(mat.N_C_physical[left]),
            N_C_right_m3=float(mat.N_C_physical[right]),
            N_V_left_m3=float(mat.N_V_physical[left]),
            N_V_right_m3=float(mat.N_V_physical[right]),
            richardson_n_A_m2_K2=min(
                float(mat.A_star_n[left]), float(mat.A_star_n[right])
            ),
            richardson_p_A_m2_K2=min(
                float(mat.A_star_p[left]), float(mat.A_star_p[right])
            ),
            conduction_band_step_eV=float(chi[left] - chi[right]),
            hole_transport_step_eV=float(
                chi[right] + Eg[right] - chi[left] - Eg[left]
            ),
            transmission=float(cross_transmission),
            surface_recombination_velocity_n_m_s=velocity_n,
            surface_recombination_velocity_p_m_s=velocity_p,
            n1_left_m3=trap_value("interface_n1_L"),
            n1_right_m3=trap_value("interface_n1_R"),
            p1_left_m3=trap_value("interface_p1_L"),
            p1_right_m3=trap_value("interface_p1_R"),
        )
        bulk = TwoSidedBulkState(
            phi_left_V=float(phi[left]),
            phi_right_V=float(phi[right]),
            n_left_m3=max(float(n[left]), 1.0e-100),
            p_left_m3=max(float(p[left]), 1.0e-100),
            n_right_m3=max(float(n[right]), 1.0e-100),
            p_right_m3=max(float(p[right]), 1.0e-100),
        )
        local = solve_two_sided_interface(
            geometry,
            physics,
            bulk,
            residual_tolerance=residual_tolerance,
            max_evaluations=max_evaluations,
            fail_on_residual=fail_on_residual,
        )
        base = 4 * index
        right_first = np.array([2, 3, 0, 1])
        state[base : base + 4] = local.state_m3[right_first]
        bulk_flux[base : base + 4] = local.bulk_flux_m2_s[right_first]
        cross_flux[base : base + 4] = local.cross_flux_m2_s[right_first]
        capture_flux[base : base + 4] = local.capture_flux_m2_s[right_first]
        residual[base : base + 4] = local.residual_m2_s[right_first]
        occupancy[index] = (
            shared_trap_occupancy(local.state_m3, physics)
            if velocity_n > 0.0 or velocity_p > 0.0
            else np.nan
        )
        maximum_residual = max(maximum_residual, local.normalized_residual)
        evaluations += local.evaluations
    return TwoSidedMaterialQSSResult(
        state_m3=state,
        bulk_flux_m2_s=bulk_flux,
        cross_flux_m2_s=cross_flux,
        state_flux_m2_s=residual,
        normalized_residual=maximum_residual,
        evaluations=evaluations,
        transport_model=model,
        capture_flux_m2_s=capture_flux,
        occupancy=occupancy,
    )


__all__ = [
    "ChargedElectrostaticTraceBalance",
    "DEDUPLICATED_QSS",
    "EquilibriumReferencedSheetCharge",
    "INTERFACE_TOPOLOGIES",
    "InterfaceTracePotentials",
    "TWO_SIDED_TRACE",
    "TwoSidedBulkState",
    "TwoSidedCarrierBalance",
    "TwoSidedInterfaceGeometry",
    "TwoSidedInterfacePhysics",
    "TwoSidedInterfaceResult",
    "TwoSidedInterfaceStencil",
    "TwoSidedMaterialQSSResult",
    "build_two_sided_interface_stencils",
    "carrier_balance_and_jacobian",
    "electrostatic_trace_residual_and_jacobian",
    "equilibrium_referenced_electrostatic_trace_balance",
    "remove_shared_interface_nodes",
    "shared_trap_occupancy",
    "shared_trap_occupancy_and_log_jacobian",
    "solve_electrostatic_traces",
    "solve_material_two_sided_interfaces_qss",
    "solve_two_sided_interface",
    "validate_interface_topology",
]
