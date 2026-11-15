"""Two-sided heterointerface QSS trace elimination and IFT sensitivities (P04-09).

Eliminates the local algebraic trace variables at a physical two-sided interface:
* Charge-off (:func:`solve_two_sided_interface_qss`): solves the 2x2 linear
  electrostatic trace system exactly and the 4x4 nonlinear log-density carrier
  trace balance ``bulk_flux - P cross_flux - capture == 0``, returning exact
  4x6 Implicit-Function-Theorem sensitivities wrt bulk coordinates
  ``(phi_L, phi_R, log n_L, log p_L, log n_R, log p_R)``.
* Equilibrium-referenced sheet charge (:func:`solve_charged_two_sided_interface_qss`):
  solves the coupled 6x6 electrostatic + carrier trace system where the Gauss
  jump and the four carrier capture fluxes share ONE constitutive evaluation of
  :func:`~solarlab.physics.recombination.interface.evaluate_shared_interface_trap`,
  returning exact IFT sensitivities of ``bulk_flux``, ``occupancy``, and
  ``incremental_sheet_charge_C_m2`` wrt all six bulk coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from solarlab.contracts.units import ELEMENTARY_CHARGE_C
from solarlab.discretization.face_average import VACUUM_PERMITTIVITY_F_M
from solarlab.discretization.stable_bernoulli import bernoulli_pair
from solarlab.physics.electrostatics.interface_traces import (
    TraceElectrostatics,
    sheet_charge_row_weights,
    solve_electrostatic_traces,
)
from solarlab.physics.interface_transport.thermionic import fermi_dirac_richardson_flux
from solarlab.physics.recombination.interface import (
    SharedInterfaceTrapEvaluation,
    evaluate_shared_interface_trap,
)

__all__ = [
    "ChargedTwoSidedInterfaceQSSResult",
    "EquilibriumReferencedSheetCharge",
    "FixedOccupancyTwoSidedResult",
    "TwoSidedBulkState",
    "TwoSidedInterfacePhysics",
    "TwoSidedInterfaceQSSResult",
    "remove_shared_interface_nodes",
    "solve_charged_two_sided_interface_qss",
    "solve_fixed_occupancy_two_sided_interface",
    "solve_two_sided_interface_qss",
]

Q: float = ELEMENTARY_CHARGE_C
EPS_0: float = VACUUM_PERMITTIVITY_F_M
RealArray: TypeAlias = NDArray[np.float64]

_N_LEFT = 0
_P_LEFT = 1
_N_RIGHT = 2
_P_RIGHT = 3
_STATE_SIZE = 4


def _positive(value: float, name: str) -> float:
    scalar = float(value)
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return scalar


def _nonnegative(value: float, name: str) -> float:
    scalar = float(value)
    if not math.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return scalar


def remove_shared_interface_nodes(
    grid_m: ArrayLike,
    interface_positions_m: Sequence[float] | ArrayLike,
) -> RealArray:
    """Remove deduplicated shared interface nodes while keeping adjacent bulk cells."""
    grid = np.asarray(grid_m, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 3 or not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0.0):
        raise ValueError("grid_m must be finite, strictly increasing, and have at least 3 nodes")
    positions = [float(p) for p in np.asarray(interface_positions_m, dtype=np.float64).ravel()]
    if not positions:
        return grid.copy()
    if any(not math.isfinite(p) for p in positions) or any(
        right <= left for left, right in zip(positions[:-1], positions[1:])
    ):
        raise ValueError("interface_positions_m must be finite and strictly increasing")

    tol = 32.0 * float(np.finfo(np.float64).eps) * max(
        1.0, abs(float(grid[0])), abs(float(grid[-1])), abs(float(grid[-1] - grid[0]))
    )
    keep = np.ones(grid.size, dtype=bool)
    for pos in positions:
        matches = np.flatnonzero(np.abs(grid - pos) <= tol)
        if matches.size != 1:
            raise ValueError(f"expected exactly one shared node at interface {pos}")
        idx = int(matches[0])
        if idx == 0 or idx == grid.size - 1:
            raise ValueError("cannot remove an outer contact node")
        keep[idx] = False
    reduced = grid[keep]
    if reduced.size < 3 or np.any(np.diff(reduced) <= 0.0):
        raise ValueError("two-sided grid reduction collapsed bulk support")
    return reduced


@dataclass(frozen=True)
class TwoSidedInterfacePhysics:
    """Constitutive parameters for one physical two-sided heterointerface."""

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
    conduction_band_step_eV: float
    hole_transport_step_eV: float
    transmission: float = 1.0
    surface_recombination_velocity_n_m_s: float = 0.0
    surface_recombination_velocity_p_m_s: float = 0.0
    n1_left_m3: float = 0.0
    n1_right_m3: float = 0.0
    p1_left_m3: float = 0.0
    p1_right_m3: float = 0.0

    def __post_init__(self) -> None:
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
            _positive(getattr(self, name), name)
        for name in (
            "transmission",
            "surface_recombination_velocity_n_m_s",
            "surface_recombination_velocity_p_m_s",
            "n1_left_m3",
            "n1_right_m3",
            "p1_left_m3",
            "p1_right_m3",
        ):
            _nonnegative(getattr(self, name), name)
        if float(self.transmission) > 1.0:
            raise ValueError("transmission must lie in [0, 1]")
        for name in ("conduction_band_step_eV", "hole_transport_step_eV"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class TwoSidedBulkState:
    """Adjacent left and right bulk cell potentials and carrier densities."""

    phi_left_V: float
    phi_right_V: float
    n_left_m3: float
    p_left_m3: float
    n_right_m3: float
    p_right_m3: float
    electron_electrochemical_step: float | None = None
    hole_electrochemical_step: float | None = None

    def __post_init__(self) -> None:
        if not (math.isfinite(float(self.phi_left_V)) and math.isfinite(float(self.phi_right_V))):
            raise ValueError("bulk potentials must be finite")
        for name in ("n_left_m3", "p_left_m3", "n_right_m3", "p_right_m3"):
            _positive(getattr(self, name), name)
        for step_name in ("electron_electrochemical_step", "hole_electrochemical_step"):
            step_val = getattr(self, step_name)
            if step_val is not None and not math.isfinite(float(step_val)):
                raise ValueError(f"{step_name} must be finite when supplied")


@dataclass(frozen=True)
class EquilibriumReferencedSheetCharge:
    """Certified dark-equilibrium reference parameters for static sheet charge."""

    trap_density_m2: float
    equilibrium_occupancy: float

    def __post_init__(self) -> None:
        _nonnegative(self.trap_density_m2, "trap_density_m2")
        f_ref = float(self.equilibrium_occupancy)
        if not (math.isfinite(f_ref) and 0.0 <= f_ref <= 1.0):
            raise ValueError("equilibrium_occupancy must lie in [0, 1]")


@dataclass(frozen=True)
class TwoSidedInterfaceQSSResult:
    """Charge-off two-sided trace elimination result and 4x6 bulk IFT Jacobian.

    ``bulk_flux_m2_s`` has ordering ``(n_L, p_L, n_R, p_R)`` representing
    particle flux from each adjacent bulk node INTO its corresponding trace
    (so bulk continuity row ``L`` receives ``-bulk_flux[n_L]`` and row ``R``
    receives ``-bulk_flux[n_R]``).
    ``bulk_flux_jacobian_bulk`` has shape ``(4, 6)`` with columns
    ``(phi_L, phi_R, log n_L, log p_L, log n_R, log p_R)``.
    """

    phi_trace_left_V: float
    phi_trace_right_V: float
    state_m3: RealArray
    bulk_flux_m2_s: RealArray
    cross_flux_m2_s: RealArray
    capture_flux_m2_s: RealArray
    occupancy: float | None
    occupancy_jacobian_bulk: RealArray
    bulk_flux_jacobian_bulk: RealArray
    normalized_residual: float
    evaluations: int
    converged: bool


@dataclass(frozen=True)
class ChargedTwoSidedInterfaceQSSResult:
    """Self-consistent equilibrium-referenced charged two-sided QSS result.

    Shares ONE single constitutive evaluation of the shared trap at the converged
    state for both ``capture_flux_m2_s`` and ``incremental_sheet_charge_C_m2``.
    All bulk Jacobians use column ordering
    ``(phi_L, phi_R, log n_L, log p_L, log n_R, log p_R)``.
    """

    phi_trace_left_V: float
    phi_trace_right_V: float
    state_m3: RealArray
    bulk_flux_m2_s: RealArray
    cross_flux_m2_s: RealArray
    capture_flux_m2_s: RealArray
    occupancy: float
    occupancy_jacobian_bulk: RealArray
    incremental_sheet_charge_C_m2: float
    sheet_charge_jacobian_bulk: RealArray
    bulk_flux_jacobian_bulk: RealArray
    normalized_electrostatic_residual: float
    normalized_carrier_residual: float
    evaluations: int
    converged: bool


def _two_prod(a: float, b: float) -> tuple[float, float]:
    """Exact Veltkamp-Dekker error-free product ``p + err == a * b`` in IEEE-754 float64."""
    p = a * b
    if not math.isfinite(p) or abs(a) > 1.0e145 or abs(b) > 1.0e145 or abs(p) < 1.0e-290:
        return p, 0.0
    split = 134217729.0  # 2**27 + 1
    ca = split * a
    a_hi = ca - (ca - a)
    a_lo = a - a_hi
    cb = split * b
    b_hi = cb - (cb - b)
    b_lo = b - b_hi
    err = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo
    return p, err


def _product_difference(a: float, b: float, c: float, d: float) -> float:
    """Compensated ``a * b - c * d`` retaining full precision under cancellation."""
    p1, e1 = _two_prod(a, b)
    p2, e2 = _two_prod(c, d)
    return math.fsum((p1, -p2, e1, -e2))


def _safe_scaled_density(bulk_density: float, shift: float) -> float:
    """Evaluate ``bulk_density * exp(shift)`` safely across extreme bias ranges."""
    log_target = math.log(bulk_density) + shift
    if log_target >= 700.0:
        return math.exp(700.0)
    if log_target <= -690.0:
        return math.exp(-690.0)
    if abs(shift) <= 600.0:
        return bulk_density * math.exp(shift)
    return math.exp(log_target)


def _trace_and_bulk_log_maps(thermal_voltage_V: float) -> tuple[RealArray, RealArray]:
    """Chain-rule maps ``∂log(state)/∂phi_t`` (4x2) and ``∂log(state)/∂bulk`` (4x6) at fixed ``u_delta``."""
    inv_vt = 1.0 / float(thermal_voltage_V)
    m_trace = np.array(
        [
            [inv_vt, 0.0],
            [-inv_vt, 0.0],
            [0.0, inv_vt],
            [0.0, -inv_vt],
        ],
        dtype=np.float64,
    )
    m_bulk = np.array(
        [
            [-inv_vt, 0.0, 1.0, 0.0, 0.0, 0.0],
            [inv_vt, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, -inv_vt, 0.0, 0.0, 1.0, 0.0],
            [0.0, inv_vt, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return m_trace, m_bulk


def _electrostatic_seed_drops(
    geometry: TraceElectrostatics,
    bulk: TwoSidedBulkState,
) -> tuple[float, float, float, float, float]:
    """Return ``(drop_left_0, drop_right_0, jump_0, phi_seed_l, phi_seed_r)`` without bulk-potential cancellation."""
    c_l = EPS_0 * float(geometry.eps_r_left) / float(geometry.left_distance_m)
    c_r = EPS_0 * float(geometry.eps_r_right) / float(geometry.right_distance_m)
    jump = float(geometry.potential_jump_right_minus_left_V)
    fixed_q = float(geometry.fixed_sheet_charge_C_m2)
    delta_bulk = float(bulk.phi_right_V) - float(bulk.phi_left_V)
    denom = c_l + c_r
    drop_left_0 = (c_r * (delta_bulk - jump) + fixed_q) / denom
    drop_right_0 = (c_l * (delta_bulk - jump) - fixed_q) / denom
    phi_seed_l = float(bulk.phi_left_V) + drop_left_0
    phi_seed_r = phi_seed_l + jump
    return drop_left_0, drop_right_0, jump, phi_seed_l, phi_seed_r


def _state_from_u_delta(
    u_delta: RealArray,
    drop_left_V: float,
    drop_right_V: float,
    physics: TwoSidedInterfacePhysics,
    bulk: TwoSidedBulkState,
) -> RealArray:
    """Resolve positive trace densities from incremental quasi-Fermi shifts ``u_delta``."""
    vt = float(physics.thermal_voltage_V)
    xi_left = float(drop_left_V) / vt
    xi_right = float(drop_right_V) / vt
    return np.array(
        [
            _safe_scaled_density(float(bulk.n_left_m3), xi_left + float(u_delta[_N_LEFT])),
            _safe_scaled_density(float(bulk.p_left_m3), -xi_left + float(u_delta[_P_LEFT])),
            _safe_scaled_density(float(bulk.n_right_m3), -xi_right + float(u_delta[_N_RIGHT])),
            _safe_scaled_density(float(bulk.p_right_m3), xi_right + float(u_delta[_P_RIGHT])),
        ],
        dtype=np.float64,
    )


def _u_delta_from_log_state(
    log_state: RealArray,
    drop_left_V: float,
    drop_right_V: float,
    physics: TwoSidedInterfacePhysics,
    bulk: TwoSidedBulkState,
) -> RealArray:
    """Map ``log(state_m3)`` to incremental quasi-Fermi shifts ``u_delta``."""
    vt = float(physics.thermal_voltage_V)
    xi_left = float(drop_left_V) / vt
    xi_right = float(drop_right_V) / vt
    return np.array(
        [
            float(log_state[_N_LEFT]) - math.log(float(bulk.n_left_m3)) - xi_left,
            float(log_state[_P_LEFT]) - math.log(float(bulk.p_left_m3)) + xi_left,
            float(log_state[_N_RIGHT]) - math.log(float(bulk.n_right_m3)) + xi_right,
            float(log_state[_P_RIGHT]) - math.log(float(bulk.p_right_m3)) - xi_right,
        ],
        dtype=np.float64,
    )


def _bulk_sg_flux_and_derivatives_u(
    u_delta: RealArray,
    state_m3: RealArray,
    drop_left_V: float,
    drop_right_V: float,
    geometry: TraceElectrostatics,
    physics: TwoSidedInterfacePhysics,
    bulk: TwoSidedBulkState,
) -> tuple[RealArray, RealArray, RealArray, RealArray]:
    """Half-cell SG particle fluxes and cancellation-free partials in ``(u_delta, phi_t, bulk)``."""
    n_l, p_l, n_r, p_r = (float(v) for v in state_m3)
    vt = float(physics.thermal_voltage_V)
    xi_left = float(drop_left_V) / vt
    xi_right = float(drop_right_V) / vt
    pair_l = bernoulli_pair(np.array([xi_left], dtype=np.float64))
    pair_r = bernoulli_pair(np.array([xi_right], dtype=np.float64))
    b_l = float(pair_l.forward[0])
    bm_l = float(pair_l.backward[0])
    db_l = float(pair_l.forward_derivative[0])
    dbm_l = float(pair_l.backward_derivative[0])
    b_r = float(pair_r.forward[0])
    bm_r = float(pair_r.backward[0])
    db_r = float(pair_r.forward_derivative[0])
    dbm_r = float(pair_r.backward_derivative[0])

    kn_l = float(physics.D_n_left_m2_s) / float(geometry.left_distance_m)
    kn_r = float(physics.D_n_right_m2_s) / float(geometry.right_distance_m)
    kp_l = float(physics.D_p_left_m2_s) / float(geometry.left_distance_m)
    kp_r = float(physics.D_p_right_m2_s) / float(geometry.right_distance_m)

    d_u = np.diag(
        [
            -kn_l * b_l * n_l,
            -kp_l * bm_l * p_l,
            -kn_r * bm_r * n_r,
            -kp_r * b_r * p_r,
        ]
    )

    u0, u1, u2, u3 = (float(v) for v in u_delta)
    log_n_l = math.log(float(bulk.n_left_m3)) + xi_left + u0
    log_p_l = math.log(float(bulk.p_left_m3)) - xi_left + u1
    log_n_r = math.log(float(bulk.n_right_m3)) - xi_right + u2
    log_p_r = math.log(float(bulk.p_right_m3)) + xi_right + u3
    unclipped = (
        -690.0 < log_n_l < 700.0
        and -690.0 < log_p_l < 700.0
        and -690.0 < log_n_r < 700.0
        and -690.0 < log_p_r < 700.0
    )

    if unclipped and max(abs(xi_left), abs(xi_right), abs(u0), abs(u1), abs(u2), abs(u3)) <= 500.0:
        # Exact identity B(x) * exp(x) == B(-x) eliminates both ULP quantization of
        # log(n_trace) and two-term Bernoulli subtraction in fluxes and IFT Jacobians.
        em0 = math.expm1(u0)
        em1 = math.expm1(u1)
        em2 = math.expm1(u2)
        em3 = math.expm1(u3)
        flux = np.array(
            [
                -kn_l * bm_l * bulk.n_left_m3 * em0,
                -kp_l * b_l * bulk.p_left_m3 * em1,
                -kn_r * b_r * bulk.n_right_m3 * em2,
                -kp_r * bm_r * bulk.p_right_m3 * em3,
            ],
            dtype=np.float64,
        )
        d_trace_u: RealArray = np.zeros((_STATE_SIZE, 2), dtype=np.float64)
        d_trace_u[_N_LEFT, 0] = (kn_l / vt) * dbm_l * bulk.n_left_m3 * em0
        d_trace_u[_P_LEFT, 0] = -(kp_l / vt) * db_l * bulk.p_left_m3 * em1
        d_trace_u[_N_RIGHT, 1] = (kn_r / vt) * db_r * bulk.n_right_m3 * em2
        d_trace_u[_P_RIGHT, 1] = -(kp_r / vt) * dbm_r * bulk.p_right_m3 * em3

        d_bulk_u: RealArray = np.zeros((_STATE_SIZE, 6), dtype=np.float64)
        d_bulk_u[_N_LEFT, 0] = -d_trace_u[_N_LEFT, 0]
        d_bulk_u[_P_LEFT, 0] = -d_trace_u[_P_LEFT, 0]
        d_bulk_u[_N_RIGHT, 1] = -d_trace_u[_N_RIGHT, 1]
        d_bulk_u[_P_RIGHT, 1] = -d_trace_u[_P_RIGHT, 1]
        d_bulk_u[_N_LEFT, 2 + _N_LEFT] = flux[_N_LEFT]
        d_bulk_u[_P_LEFT, 2 + _P_LEFT] = flux[_P_LEFT]
        d_bulk_u[_N_RIGHT, 2 + _N_RIGHT] = flux[_N_RIGHT]
        d_bulk_u[_P_RIGHT, 2 + _P_RIGHT] = flux[_P_RIGHT]
        return flux, d_u, d_trace_u, d_bulk_u

    small_l = b_l if xi_left >= 0.0 else bm_l
    small_r = b_r if xi_right >= 0.0 else bm_r
    flux = np.array(
        [
            kn_l
            * _product_difference(
                small_l,
                bulk.n_left_m3 - n_l,
                -xi_left,
                bulk.n_left_m3 if xi_left >= 0.0 else n_l,
            ),
            kp_l
            * _product_difference(
                small_l,
                bulk.p_left_m3 - p_l,
                xi_left,
                p_l if xi_left >= 0.0 else bulk.p_left_m3,
            ),
            kn_r
            * _product_difference(
                small_r,
                bulk.n_right_m3 - n_r,
                xi_right,
                n_r if xi_right >= 0.0 else bulk.n_right_m3,
            ),
            kp_r
            * _product_difference(
                small_r,
                bulk.p_right_m3 - p_r,
                -xi_right,
                bulk.p_right_m3 if xi_right >= 0.0 else p_r,
            ),
        ],
        dtype=np.float64,
    )
    d_trace: RealArray = np.zeros((_STATE_SIZE, 2), dtype=np.float64)
    d_trace[_N_LEFT, 0] = (kn_l / vt) * (-dbm_l * bulk.n_left_m3 - db_l * n_l)
    d_trace[_P_LEFT, 0] = (kp_l / vt) * (db_l * bulk.p_left_m3 + dbm_l * p_l)
    d_trace[_N_RIGHT, 1] = -(kn_r / vt) * (db_r * bulk.n_right_m3 + dbm_r * n_r)
    d_trace[_P_RIGHT, 1] = (kp_r / vt) * (dbm_r * bulk.p_right_m3 + db_r * p_r)

    d_bulk: RealArray = np.zeros((_STATE_SIZE, 6), dtype=np.float64)
    d_bulk[_N_LEFT, 0] = -d_trace[_N_LEFT, 0]
    d_bulk[_P_LEFT, 0] = -d_trace[_P_LEFT, 0]
    d_bulk[_N_RIGHT, 1] = -d_trace[_N_RIGHT, 1]
    d_bulk[_P_RIGHT, 1] = -d_trace[_P_RIGHT, 1]
    d_bulk[_N_LEFT, 2 + _N_LEFT] = kn_l * bm_l * bulk.n_left_m3
    d_bulk[_P_LEFT, 2 + _P_LEFT] = kp_l * b_l * bulk.p_left_m3
    d_bulk[_N_RIGHT, 2 + _N_RIGHT] = kn_r * b_r * bulk.n_right_m3
    d_bulk[_P_RIGHT, 2 + _P_RIGHT] = kp_r * bm_r * bulk.p_right_m3
    m_trace, m_bulk = _trace_and_bulk_log_maps(vt)
    return flux, d_u, d_trace + d_u @ m_trace, d_bulk + d_u @ m_bulk


def _evaluate_trap_for_two_sided(
    state_m3: RealArray,
    physics: TwoSidedInterfacePhysics,
    charge: EquilibriumReferencedSheetCharge | None = None,
) -> SharedInterfaceTrapEvaluation | None:
    vn = float(physics.surface_recombination_velocity_n_m_s)
    vp = float(physics.surface_recombination_velocity_p_m_s)
    if vn == 0.0 and vp == 0.0:
        if charge is not None:
            raise ValueError("charged two-sided interface requires positive capture velocity")
        return None
    return evaluate_shared_interface_trap(
        float(state_m3[_N_LEFT]),
        float(state_m3[_P_LEFT]),
        float(state_m3[_N_RIGHT]),
        float(state_m3[_P_RIGHT]),
        v_n_left_m_s=vn,
        v_p_left_m_s=vp,
        v_n_right_m_s=vn,
        v_p_right_m_s=vp,
        n1_left_m3=physics.n1_left_m3,
        p1_left_m3=physics.p1_left_m3,
        n1_right_m3=physics.n1_right_m3,
        p1_right_m3=physics.p1_right_m3,
        areal_density_m2=0.0 if charge is None else charge.trap_density_m2,
        equilibrium_occupancy=None if charge is None else charge.equilibrium_occupancy,
    )


def _carrier_balance_terms(
    u_delta: RealArray,
    drop_left_V: float,
    drop_right_V: float,
    potential_jump_V: float,
    geometry: TraceElectrostatics,
    physics: TwoSidedInterfacePhysics,
    bulk: TwoSidedBulkState,
    charge: EquilibriumReferencedSheetCharge | None = None,
) -> tuple[
    RealArray,
    RealArray,
    RealArray,
    RealArray,
    RealArray,
    RealArray,
    RealArray,
    RealArray,
    RealArray,
    RealArray,
    RealArray,
    RealArray,
    SharedInterfaceTrapEvaluation | None,
]:
    state = _state_from_u_delta(u_delta, drop_left_V, drop_right_V, physics, bulk)
    bulk_flux, d_bulk_u, d_bulk_trace_u, d_bulk_coords_u = _bulk_sg_flux_and_derivatives_u(
        u_delta, state, drop_left_V, drop_right_V, geometry, physics, bulk
    )
    vt_local = float(physics.thermal_voltage_V)
    xi_l = float(drop_left_V) / vt_local
    xi_r_from_bulk = -float(drop_right_V) / vt_local
    base_log_to_dos = np.array(
        [
            math.log(float(bulk.n_left_m3) / float(physics.N_C_left_m3)) + xi_l,
            math.log(float(bulk.p_left_m3) / float(physics.N_V_left_m3)) - xi_l,
            math.log(float(bulk.n_right_m3) / float(physics.N_C_right_m3)) + xi_r_from_bulk,
            math.log(float(bulk.p_right_m3) / float(physics.N_V_right_m3)) - xi_r_from_bulk,
        ],
        dtype=np.float64,
    )
    cross_eval = fermi_dirac_richardson_flux(
        state,
        phi_trace_left_V=0.0,
        phi_trace_right_V=float(potential_jump_V),
        conduction_band_step_eV=physics.conduction_band_step_eV,
        hole_transport_step_eV=physics.hole_transport_step_eV,
        N_C_left_m3=physics.N_C_left_m3,
        N_C_right_m3=physics.N_C_right_m3,
        N_V_left_m3=physics.N_V_left_m3,
        N_V_right_m3=physics.N_V_right_m3,
        richardson_n_A_m2_K2=physics.richardson_n_A_m2_K2,
        richardson_p_A_m2_K2=physics.richardson_p_A_m2_K2,
        temperature_K=physics.temperature_K,
        thermal_voltage_V=physics.thermal_voltage_V,
        transmission=physics.transmission,
        base_log_to_dos=base_log_to_dos,
        u_delta=u_delta,
        electrochemical_steps=(
            bulk.electron_electrochemical_step,
            bulk.hole_electrochemical_step,
        ),
    )
    trap = _evaluate_trap_for_two_sided(state, physics, charge)
    if trap is None:
        capture: RealArray = np.zeros(_STATE_SIZE, dtype=np.float64)
        d_cap_log: RealArray = np.zeros((_STATE_SIZE, _STATE_SIZE), dtype=np.float64)
    else:
        capture = trap.capture_rates_m2_s
        d_cap_log = trap.d_capture_d_densities_m_s * state[np.newaxis, :]

    m_trace, m_bulk = _trace_and_bulk_log_maps(physics.thermal_voltage_V)
    cross = cross_eval.cross_flux_m2_s
    residual = np.array(
        [
            bulk_flux[_N_LEFT] - cross[0] - capture[_N_LEFT],
            bulk_flux[_P_LEFT] - cross[1] - capture[_P_LEFT],
            bulk_flux[_N_RIGHT] + cross[0] - capture[_N_RIGHT],
            bulk_flux[_P_RIGHT] + cross[1] - capture[_P_RIGHT],
        ],
        dtype=np.float64,
    )
    j_u = d_bulk_u - d_cap_log
    j_u[_N_LEFT, :] -= cross_eval.d_cross_d_log_state_m2_s[0, :]
    j_u[_P_LEFT, :] -= cross_eval.d_cross_d_log_state_m2_s[1, :]
    j_u[_N_RIGHT, :] += cross_eval.d_cross_d_log_state_m2_s[0, :]
    j_u[_P_RIGHT, :] += cross_eval.d_cross_d_log_state_m2_s[1, :]

    d_cross_trace_u = (
        cross_eval.d_cross_d_trace_potentials_m2_s_V
        + cross_eval.d_cross_d_log_state_m2_s @ m_trace
    )
    j_trace_u = d_bulk_trace_u - d_cap_log @ m_trace
    j_trace_u[_N_LEFT, :] -= d_cross_trace_u[0, :]
    j_trace_u[_P_LEFT, :] -= d_cross_trace_u[1, :]
    j_trace_u[_N_RIGHT, :] += d_cross_trace_u[0, :]
    j_trace_u[_P_RIGHT, :] += d_cross_trace_u[1, :]

    d_cross_bulk_u = cross_eval.d_cross_d_log_state_m2_s @ m_bulk
    j_coords_u = d_bulk_coords_u - d_cap_log @ m_bulk
    j_coords_u[_N_LEFT, :] -= d_cross_bulk_u[0, :]
    j_coords_u[_P_LEFT, :] -= d_cross_bulk_u[1, :]
    j_coords_u[_N_RIGHT, :] += d_cross_bulk_u[0, :]
    j_coords_u[_P_RIGHT, :] += d_cross_bulk_u[1, :]

    return (
        state,
        bulk_flux,
        cross,
        capture,
        residual,
        j_u,
        j_trace_u,
        j_coords_u,
        d_bulk_u,
        d_bulk_trace_u,
        d_bulk_coords_u,
        cross_eval.one_way_scale_m2_s,
        trap,
    )


def _equilibrated_solve(matrix: RealArray, rhs: RealArray) -> RealArray:
    """Column-equilibrated dense linear solve preserving accuracy across 30+ decades."""
    col_scale = np.max(np.abs(matrix), axis=0)
    col_scale = np.where(col_scale > 0.0, col_scale, 1.0)
    scaled_sol = np.linalg.solve(matrix / col_scale[np.newaxis, :], rhs)
    if rhs.ndim == 1:
        return scaled_sol / col_scale
    return scaled_sol / col_scale[:, np.newaxis]


def _zero_bulk_flux_seed(
    drop_left_V: float,
    drop_right_V: float,
    physics: TwoSidedInterfacePhysics,
    bulk: TwoSidedBulkState,
) -> RealArray:
    return _state_from_u_delta(
        np.zeros(_STATE_SIZE, dtype=np.float64),
        drop_left_V,
        drop_right_V,
        physics,
        bulk,
    )


def solve_two_sided_interface_qss(
    geometry: TraceElectrostatics,
    physics: TwoSidedInterfacePhysics,
    bulk: TwoSidedBulkState,
    *,
    initial_state_m3: ArrayLike | None = None,
    residual_tolerance: float = 1.0e-9,
    max_evaluations: int = 200,
    fail_on_residual: bool = True,
) -> TwoSidedInterfaceQSSResult:
    """Eliminate charge-off two-sided electrostatic and carrier traces with exact IFT."""
    _positive(residual_tolerance, "residual_tolerance")
    if int(max_evaluations) <= 0:
        raise ValueError("max_evaluations must be positive")

    phi_t_l, phi_t_r = solve_electrostatic_traces(
        geometry, bulk.phi_left_V, bulk.phi_right_V
    )
    drop_l_0, drop_r_0, jump_0, _, _ = _electrostatic_seed_drops(geometry, bulk)
    zero_flux_seed = _zero_bulk_flux_seed(drop_l_0, drop_r_0, physics, bulk)
    u_seed_raw: RealArray
    if initial_state_m3 is None:
        seed = zero_flux_seed
        u_seed_raw = np.zeros(_STATE_SIZE, dtype=np.float64)
    else:
        seed = np.asarray(initial_state_m3, dtype=np.float64)
        if seed.shape != (_STATE_SIZE,) or not np.all(np.isfinite(seed)) or np.any(seed <= 0.0):
            raise ValueError("initial_state_m3 must contain four positive finite densities")
        u_seed_raw = _u_delta_from_log_state(np.log(seed), drop_l_0, drop_r_0, physics, bulk)

    density_ref = max(
        float(np.max(seed)),
        float(np.max(zero_flux_seed)),
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
    vt = float(physics.thermal_voltage_V)
    xi_left = drop_l_0 / vt
    xi_right = drop_r_0 / vt
    base_log = np.array(
        [
            math.log(float(bulk.n_left_m3)) + xi_left,
            math.log(float(bulk.p_left_m3)) - xi_left,
            math.log(float(bulk.n_right_m3)) - xi_right,
            math.log(float(bulk.p_right_m3)) + xi_right,
        ],
        dtype=np.float64,
    )
    log_lower = math.log(1.0e-100)
    log_upper = math.log(density_ref) + math.log(1.0e12)
    u_lower = np.maximum(log_lower - base_log, -490.0)
    u_upper = np.minimum(log_upper - base_log, 490.0)
    u_upper = np.maximum(u_upper, u_lower + 2.0)
    u_seed = np.clip(u_seed_raw, u_lower + 0.5, u_upper - 0.5)

    (
        _,
        seed_bulk_flux,
        _,
        seed_cap,
        _,
        _,
        _,
        _,
        seed_d_bulk_u,
        _,
        _,
        seed_one_way,
        _,
    ) = _carrier_balance_terms(u_seed, drop_l_0, drop_r_0, jump_0, geometry, physics, bulk)

    e_scale = max(
        abs(float(seed_bulk_flux[_N_LEFT])),
        abs(float(seed_bulk_flux[_N_RIGHT])),
        abs(float(seed_d_bulk_u[_N_LEFT, _N_LEFT])),
        abs(float(seed_d_bulk_u[_N_RIGHT, _N_RIGHT])),
        abs(float(seed_cap[_N_LEFT])),
        abs(float(seed_cap[_N_RIGHT])),
        float(seed_one_way[0]),
        1.0,
    )
    h_scale = max(
        abs(float(seed_bulk_flux[_P_LEFT])),
        abs(float(seed_bulk_flux[_P_RIGHT])),
        abs(float(seed_d_bulk_u[_P_LEFT, _P_LEFT])),
        abs(float(seed_d_bulk_u[_P_RIGHT, _P_RIGHT])),
        abs(float(seed_cap[_P_LEFT])),
        abs(float(seed_cap[_P_RIGHT])),
        float(seed_one_way[1]),
        1.0,
    )
    row_scale = np.array([e_scale, h_scale, e_scale, h_scale], dtype=np.float64)

    def scaled_res(u_vec: RealArray) -> RealArray:
        terms = _carrier_balance_terms(u_vec, drop_l_0, drop_r_0, jump_0, geometry, physics, bulk)
        return terms[4] / row_scale

    def scaled_jac(u_vec: RealArray) -> RealArray:
        terms = _carrier_balance_terms(u_vec, drop_l_0, drop_r_0, jump_0, geometry, physics, bulk)
        return terms[5] / row_scale[:, np.newaxis]

    sol = least_squares(
        scaled_res,
        u_seed,
        jac=scaled_jac,
        bounds=(u_lower, u_upper),
        x_scale="jac",
        ftol=3.0e-16,
        xtol=3.0e-16,
        gtol=3.0e-16,
        max_nfev=int(max_evaluations),
    )
    x_best = sol.x.copy()
    eval_count = int(sol.nfev)

    # Polish to machine precision with per-row equilibrated Newton steps.
    best_terms = _carrier_balance_terms(x_best, drop_l_0, drop_r_0, jump_0, geometry, physics, bulk)
    for _ in range(6):
        p_row = np.maximum(np.max(np.abs(best_terms[5]), axis=1), 1.0)
        p_norm = float(np.max(np.abs(best_terms[4] / p_row)))
        if p_norm <= 1.0e-16:
            break
        try:
            delta = -_equilibrated_solve(
                best_terms[5] / p_row[:, np.newaxis],
                best_terms[4] / p_row,
            )
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(delta)):
            break
        cand = np.clip(x_best + delta, u_lower, u_upper)
        cand_terms = _carrier_balance_terms(cand, drop_l_0, drop_r_0, jump_0, geometry, physics, bulk)
        eval_count += 1
        cand_norm = float(np.max(np.abs(cand_terms[4] / p_row)))
        if cand_norm < p_norm or float(np.max(np.abs(delta))) <= 1.0e-15:
            x_best = cand
            best_terms = cand_terms
            if float(np.max(np.abs(delta))) <= 1.0e-15:
                break
        else:
            break
    best_norm = float(np.max(np.abs(best_terms[4] / row_scale)))

    (
        state,
        bulk_flux,
        cross_flux,
        capture_flux,
        _,
        j_u,
        j_trace_u,
        j_coords_u,
        d_bulk_u,
        d_bulk_trace_u,
        d_bulk_coords_u,
        _,
        trap,
    ) = best_terms

    w_l, w_r = sheet_charge_row_weights(
        geometry.eps_r_left,
        geometry.left_distance_m,
        geometry.eps_r_right,
        geometry.right_distance_m,
    )
    d_trace_d_bulk: RealArray = np.zeros((2, 6), dtype=np.float64)
    d_trace_d_bulk[0, 0] = w_l
    d_trace_d_bulk[0, 1] = w_r
    d_trace_d_bulk[1, 0] = w_l
    d_trace_d_bulk[1, 1] = w_r

    j_bulk_total_u = j_coords_u + j_trace_u @ d_trace_d_bulk
    ift_row = np.maximum(np.max(np.abs(j_u), axis=1), float(np.finfo(np.float64).tiny))
    du_d_bulk = _equilibrated_solve(
        j_u / ift_row[:, np.newaxis],
        -(j_bulk_total_u / ift_row[:, np.newaxis]),
    )
    bulk_flux_jac = d_bulk_coords_u + d_bulk_trace_u @ d_trace_d_bulk + d_bulk_u @ du_d_bulk

    if trap is None:
        occupancy: float | None = None
        occ_jac: RealArray = np.zeros(6, dtype=np.float64)
    else:
        m_trace, m_bulk = _trace_and_bulk_log_maps(physics.thermal_voltage_V)
        d_log_state_d_bulk = m_bulk + m_trace @ d_trace_d_bulk + du_d_bulk
        occupancy = trap.occupancy
        df_dlog = trap.d_occupancy_d_densities_m3 * state
        occ_jac = df_dlog @ d_log_state_d_bulk

    converged = bool(
        (sol.success or best_norm <= residual_tolerance)
        and best_norm <= residual_tolerance
        and np.all(np.isfinite(du_d_bulk))
        and np.all(np.isfinite(bulk_flux_jac))
    )
    if fail_on_residual and not converged:
        raise RuntimeError(
            f"two-sided interface QSS did not certify: residual={best_norm:.6g}"
        )
    return TwoSidedInterfaceQSSResult(
        phi_trace_left_V=phi_t_l,
        phi_trace_right_V=phi_t_r,
        state_m3=state,
        bulk_flux_m2_s=bulk_flux,
        cross_flux_m2_s=cross_flux,
        capture_flux_m2_s=capture_flux,
        occupancy=occupancy,
        occupancy_jacobian_bulk=occ_jac,
        bulk_flux_jacobian_bulk=bulk_flux_jac,
        normalized_residual=best_norm,
        evaluations=eval_count,
        converged=converged,
    )


def _charged_balance_terms(
    local_coords_u: RealArray,
    drop_l_0: float,
    drop_r_0: float,
    jump_0: float,
    geometry: TraceElectrostatics,
    physics: TwoSidedInterfacePhysics,
    bulk: TwoSidedBulkState,
    charge: EquilibriumReferencedSheetCharge,
) -> tuple[
    RealArray,
    RealArray,
    RealArray,
    RealArray,
    RealArray,
    RealArray,
    RealArray,
    RealArray,
    RealArray,
    RealArray,
    RealArray,
    SharedInterfaceTrapEvaluation,
]:
    s_l = float(local_coords_u[0])
    s_r = float(local_coords_u[1])
    u_delta = local_coords_u[2:]

    drop_l = drop_l_0 + s_l
    drop_r = drop_r_0 - s_r
    pot_jump = jump_0 + (s_r - s_l)

    (
        state,
        bulk_flux,
        cross_flux,
        capture_flux,
        carrier_res,
        j_carrier_u,
        j_carrier_trace_u,
        j_carrier_coords_u,
        d_bulk_u,
        d_bulk_trace_u,
        d_bulk_coords_u,
        one_way_scale,
        trap,
    ) = _carrier_balance_terms(
        u_delta, drop_l, drop_r, pot_jump, geometry, physics, bulk, charge
    )
    assert trap is not None

    c_l = EPS_0 * float(geometry.eps_r_left) / float(geometry.left_distance_m)
    c_r = EPS_0 * float(geometry.eps_r_right) / float(geometry.right_distance_m)
    elec_res = np.array(
        [
            s_r - s_l,
            c_l * s_l + c_r * s_r - trap.incremental_sheet_charge_C_m2,
        ],
        dtype=np.float64,
    )
    elec_jac_trace = np.array([[-1.0, 1.0], [c_l, c_r]], dtype=np.float64)

    m_trace, m_bulk = _trace_and_bulk_log_maps(physics.thermal_voltage_V)
    d_sheet_d_log = trap.d_sheet_charge_d_densities_C_m * state

    residual = np.concatenate((elec_res, carrier_res))
    j_local_u: RealArray = np.zeros((6, 6), dtype=np.float64)
    j_local_u[:2, :2] = elec_jac_trace
    j_local_u[1, :2] -= d_sheet_d_log @ m_trace
    j_local_u[1, 2:] = -d_sheet_d_log
    j_local_u[2:, :2] = j_carrier_trace_u
    j_local_u[2:, 2:] = j_carrier_u

    j_bulk_u: RealArray = np.zeros((6, 6), dtype=np.float64)
    j_bulk_u[1, 0] = -c_l
    j_bulk_u[1, 1] = -c_r
    j_bulk_u[1, :] -= d_sheet_d_log @ m_bulk
    j_bulk_u[2:, :] = j_carrier_coords_u

    return (
        state,
        bulk_flux,
        cross_flux,
        capture_flux,
        residual,
        j_local_u,
        j_bulk_u,
        d_bulk_u,
        d_bulk_trace_u,
        d_bulk_coords_u,
        one_way_scale,
        trap,
    )


def solve_charged_two_sided_interface_qss(
    geometry: TraceElectrostatics,
    physics: TwoSidedInterfacePhysics,
    bulk: TwoSidedBulkState,
    charge: EquilibriumReferencedSheetCharge,
    *,
    initial_coordinates: ArrayLike | None = None,
    residual_tolerance: float = 1.0e-9,
    max_evaluations: int = 300,
    fail_on_residual: bool = True,
) -> ChargedTwoSidedInterfaceQSSResult:
    """Solve coupled 6x6 equilibrium-referenced charged two-sided QSS + exact IFT."""
    _positive(residual_tolerance, "residual_tolerance")
    if int(max_evaluations) <= 0:
        raise ValueError("max_evaluations must be positive")

    phi_seed_l, phi_seed_r = solve_electrostatic_traces(
        geometry, bulk.phi_left_V, bulk.phi_right_V
    )
    drop_l_0, drop_r_0, jump_0, _, _ = _electrostatic_seed_drops(geometry, bulk)
    c_l = EPS_0 * float(geometry.eps_r_left) / float(geometry.left_distance_m)
    c_r = EPS_0 * float(geometry.eps_r_right) / float(geometry.right_distance_m)
    phi_margin = Q * float(charge.trap_density_m2) / (c_l + c_r) + max(
        20.0 * float(physics.thermal_voltage_V), 2.0
    )
    u_bound = 480.0
    lower = np.array(
        [
            -phi_margin,
            -phi_margin,
            -u_bound,
            -u_bound,
            -u_bound,
            -u_bound,
        ],
        dtype=np.float64,
    )
    upper = np.array(
        [
            phi_margin,
            phi_margin,
            u_bound,
            u_bound,
            u_bound,
            u_bound,
        ],
        dtype=np.float64,
    )

    if initial_coordinates is None:
        state_seed = solve_two_sided_interface_qss(
            geometry,
            physics,
            bulk,
            residual_tolerance=max(residual_tolerance, 1.0e-9),
            max_evaluations=max_evaluations,
        ).state_m3
        u_seed = _u_delta_from_log_state(
            np.log(state_seed), drop_l_0, drop_r_0, physics, bulk
        )
        coords = np.concatenate(
            (
                np.zeros(2, dtype=np.float64),
                np.clip(u_seed, -u_bound + 1.0, u_bound - 1.0),
            )
        )
    else:
        raw_coords = np.asarray(initial_coordinates, dtype=np.float64)
        if raw_coords.shape != (6,) or not np.all(np.isfinite(raw_coords)):
            raise ValueError("initial_coordinates must contain 6 finite values")
        s_l_init = float(raw_coords[0]) - phi_seed_l
        s_r_init = float(raw_coords[1]) - phi_seed_r
        u_seed = _u_delta_from_log_state(
            raw_coords[2:], drop_l_0 + s_l_init, drop_r_0 - s_r_init, physics, bulk
        )
        coords = np.concatenate(
            (
                np.clip(np.array([s_l_init, s_r_init], dtype=np.float64), lower[:2] + 1.0e-6, upper[:2] - 1.0e-6),
                np.clip(u_seed, -u_bound + 1.0, u_bound - 1.0),
            )
        )

    seed_terms = _charged_balance_terms(coords, drop_l_0, drop_r_0, jump_0, geometry, physics, bulk, charge)
    seed_bulk_flux = seed_terms[1]
    seed_cap = seed_terms[3]
    seed_d_bulk_u = seed_terms[7]
    seed_one_way = seed_terms[10]

    e_scale = max(
        abs(float(seed_bulk_flux[_N_LEFT])),
        abs(float(seed_bulk_flux[_N_RIGHT])),
        abs(float(seed_d_bulk_u[_N_LEFT, _N_LEFT])),
        abs(float(seed_d_bulk_u[_N_RIGHT, _N_RIGHT])),
        abs(float(seed_cap[_N_LEFT])),
        abs(float(seed_cap[_N_RIGHT])),
        float(seed_one_way[0]),
        1.0,
    )
    h_scale = max(
        abs(float(seed_bulk_flux[_P_LEFT])),
        abs(float(seed_bulk_flux[_P_RIGHT])),
        abs(float(seed_d_bulk_u[_P_LEFT, _P_LEFT])),
        abs(float(seed_d_bulk_u[_P_RIGHT, _P_RIGHT])),
        abs(float(seed_cap[_P_LEFT])),
        abs(float(seed_cap[_P_RIGHT])),
        float(seed_one_way[1]),
        1.0,
    )
    cap_scale = (c_l + c_r) * float(physics.thermal_voltage_V)
    gauss_scale = max(
        cap_scale,
        abs(float(geometry.fixed_sheet_charge_C_m2)),
        Q * float(charge.trap_density_m2),
        float(np.finfo(np.float64).tiny),
    )
    row_scale = np.array(
        [physics.thermal_voltage_V, gauss_scale, e_scale, h_scale, e_scale, h_scale],
        dtype=np.float64,
    )

    def scaled_res(y_loc: RealArray) -> RealArray:
        return _charged_balance_terms(y_loc, drop_l_0, drop_r_0, jump_0, geometry, physics, bulk, charge)[4] / row_scale

    def scaled_jac(y_loc: RealArray) -> RealArray:
        return (
            _charged_balance_terms(y_loc, drop_l_0, drop_r_0, jump_0, geometry, physics, bulk, charge)[5]
            / row_scale[:, np.newaxis]
        )

    sol = least_squares(
        scaled_res,
        coords,
        jac=scaled_jac,
        bounds=(lower, upper),
        x_scale="jac",
        ftol=3.0e-16,
        xtol=3.0e-16,
        gtol=3.0e-16,
        max_nfev=int(max_evaluations),
    )
    y_best = sol.x.copy()
    eval_count = int(sol.nfev)

    # Polish coupled 6x6 system to full binary64 precision using per-row Jacobian scaling.
    best_terms = _charged_balance_terms(y_best, drop_l_0, drop_r_0, jump_0, geometry, physics, bulk, charge)
    vt_coord = np.array(
        [physics.thermal_voltage_V, physics.thermal_voltage_V, 1.0, 1.0, 1.0, 1.0],
        dtype=np.float64,
    )
    for _ in range(8):
        p_row = np.maximum(np.max(np.abs(best_terms[5]), axis=1), float(np.finfo(np.float64).tiny))
        p_norm = float(np.max(np.abs(best_terms[4] / p_row)))
        if p_norm <= 1.0e-16:
            break
        try:
            delta = -_equilibrated_solve(
                best_terms[5] / p_row[:, np.newaxis],
                best_terms[4] / p_row,
            )
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(delta)):
            break
        step_norm = float(np.max(np.abs(delta / vt_coord)))
        cand = np.clip(y_best + delta, lower, upper)
        cand_terms = _charged_balance_terms(cand, drop_l_0, drop_r_0, jump_0, geometry, physics, bulk, charge)
        eval_count += 1
        cand_norm = float(np.max(np.abs(cand_terms[4] / p_row)))
        if cand_norm < p_norm or step_norm <= 1.0e-13:
            y_best = cand
            best_terms = cand_terms
            if step_norm <= 1.0e-15:
                break
        else:
            break

    (
        state,
        bulk_flux,
        cross_flux,
        capture_flux,
        residual,
        j_local_u,
        j_bulk_u,
        d_bulk_u,
        d_bulk_trace_u,
        d_bulk_coords_u,
        _,
        trap,
    ) = best_terms

    norm_vec = np.abs(residual / row_scale)
    norm_elec = float(np.max(norm_vec[:2]))
    norm_carrier = float(np.max(norm_vec[2:]))

    ift_row = np.maximum(np.max(np.abs(j_local_u), axis=1), float(np.finfo(np.float64).tiny))
    sensitivity_u = _equilibrated_solve(
        j_local_u / ift_row[:, np.newaxis],
        -(j_bulk_u / ift_row[:, np.newaxis]),
    )
    d_phi_t_d_bulk = sensitivity_u[:2, :]
    d_u_d_bulk = sensitivity_u[2:, :]
    bulk_flux_jac = (
        d_bulk_coords_u
        + d_bulk_trace_u @ d_phi_t_d_bulk
        + d_bulk_u @ d_u_d_bulk
    )
    m_trace, m_bulk = _trace_and_bulk_log_maps(physics.thermal_voltage_V)
    d_log_state_d_bulk = m_bulk + m_trace @ d_phi_t_d_bulk + d_u_d_bulk
    d_sheet_d_log = trap.d_sheet_charge_d_densities_C_m * state
    sheet_jac_bulk = d_sheet_d_log @ d_log_state_d_bulk
    d_occ_d_log = trap.d_occupancy_d_densities_m3 * state
    occ_jac_bulk = d_occ_d_log @ d_log_state_d_bulk

    converged = bool(
        (sol.success or max(norm_elec, norm_carrier) <= residual_tolerance)
        and max(norm_elec, norm_carrier) <= residual_tolerance
        and np.all(np.isfinite(sensitivity_u))
        and np.all(np.isfinite(bulk_flux_jac))
        and np.all(np.isfinite(sheet_jac_bulk))
    )
    if fail_on_residual and not converged:
        raise RuntimeError(
            "charged two-sided interface QSS did not certify: "
            f"electrostatic={norm_elec:.6g}, carrier={norm_carrier:.6g}"
        )

    return ChargedTwoSidedInterfaceQSSResult(
        phi_trace_left_V=float(phi_seed_l + y_best[0]),
        phi_trace_right_V=float(phi_seed_r + y_best[1]),
        state_m3=state,
        bulk_flux_m2_s=bulk_flux,
        cross_flux_m2_s=cross_flux,
        capture_flux_m2_s=capture_flux,
        occupancy=trap.occupancy,
        occupancy_jacobian_bulk=occ_jac_bulk,
        incremental_sheet_charge_C_m2=trap.incremental_sheet_charge_C_m2,
        sheet_charge_jacobian_bulk=sheet_jac_bulk,
        bulk_flux_jacobian_bulk=bulk_flux_jac,
        normalized_electrostatic_residual=norm_elec,
        normalized_carrier_residual=norm_carrier,
        evaluations=eval_count,
        converged=converged,
    )


@dataclass(frozen=True)
class FixedOccupancyTwoSidedResult:
    """Two-sided trace elimination at an explicit dynamic trap occupancy ``f``.

    Used by unified dynamic-interface assembly (`P04-10`) where ``f`` is a
    storage-bearing state variable rather than locally eliminated by QSS.
    Provides exact IFT sensitivities wrt both the 6 bulk coordinates
    ``(phi_L, phi_R, log n_L, log p_L, log n_R, log p_R)`` AND ``occupancy``.
    """

    phi_trace_left_V: float
    phi_trace_right_V: float
    state_m3: RealArray
    bulk_flux_m2_s: RealArray
    cross_flux_m2_s: RealArray
    capture_flux_m2_s: RealArray
    occupancy: float
    qss_occupancy: float
    relaxation_rate_s1: float
    trap_storage_rate_m2_s: float
    incremental_sheet_charge_C_m2: float
    d_sheet_charge_df_C_m2: float
    bulk_flux_jacobian_bulk: RealArray  # (4, 6)
    bulk_flux_derivative_occupancy: RealArray  # (4,)
    trap_storage_rate_jacobian_bulk: RealArray  # (6,)
    trap_storage_rate_derivative_occupancy: float
    normalized_residual: float
    evaluations: int
    converged: bool
    one_minus_occupancy: float = 0.0


def _fixed_occupancy_balance_terms(
    u_delta: RealArray,
    drop_left_V: float,
    drop_right_V: float,
    potential_jump_V: float,
    geometry: TraceElectrostatics,
    physics: TwoSidedInterfacePhysics,
    bulk: TwoSidedBulkState,
    occupancy: float,
    one_minus_occupancy: float,
) -> tuple[
    RealArray,
    RealArray,
    RealArray,
    RealArray,
    RealArray,
    RealArray,
    RealArray,
    RealArray,
    RealArray,
    RealArray,
    RealArray,
    RealArray,
    RealArray,
]:
    state = _state_from_u_delta(u_delta, drop_left_V, drop_right_V, physics, bulk)
    bulk_flux, d_bulk_u, d_bulk_trace_u, d_bulk_coords_u = _bulk_sg_flux_and_derivatives_u(
        u_delta, state, drop_left_V, drop_right_V, geometry, physics, bulk
    )
    vt_local = float(physics.thermal_voltage_V)
    xi_l = float(drop_left_V) / vt_local
    xi_r_from_bulk = -float(drop_right_V) / vt_local
    base_log_to_dos = np.array(
        [
            math.log(float(bulk.n_left_m3) / float(physics.N_C_left_m3)) + xi_l,
            math.log(float(bulk.p_left_m3) / float(physics.N_V_left_m3)) - xi_l,
            math.log(float(bulk.n_right_m3) / float(physics.N_C_right_m3)) + xi_r_from_bulk,
            math.log(float(bulk.p_right_m3) / float(physics.N_V_right_m3)) - xi_r_from_bulk,
        ],
        dtype=np.float64,
    )
    cross_eval = fermi_dirac_richardson_flux(
        state,
        phi_trace_left_V=0.0,
        phi_trace_right_V=float(potential_jump_V),
        conduction_band_step_eV=physics.conduction_band_step_eV,
        hole_transport_step_eV=physics.hole_transport_step_eV,
        N_C_left_m3=physics.N_C_left_m3,
        N_C_right_m3=physics.N_C_right_m3,
        N_V_left_m3=physics.N_V_left_m3,
        N_V_right_m3=physics.N_V_right_m3,
        richardson_n_A_m2_K2=physics.richardson_n_A_m2_K2,
        richardson_p_A_m2_K2=physics.richardson_p_A_m2_K2,
        temperature_K=physics.temperature_K,
        thermal_voltage_V=physics.thermal_voltage_V,
        transmission=physics.transmission,
        base_log_to_dos=base_log_to_dos,
        u_delta=u_delta,
        electrochemical_steps=(
            bulk.electron_electrochemical_step,
            bulk.hole_electrochemical_step,
        ),
    )

    vn = float(physics.surface_recombination_velocity_n_m_s)
    vp = float(physics.surface_recombination_velocity_p_m_s)
    f = float(occupancy)
    omf = float(one_minus_occupancy)
    n_l, p_l, n_r, p_r = (float(v) for v in state)
    capture = np.array(
        [
            vn * _product_difference(n_l, omf, physics.n1_left_m3, f),
            vp * _product_difference(p_l, f, physics.p1_left_m3, omf),
            vn * _product_difference(n_r, omf, physics.n1_right_m3, f),
            vp * _product_difference(p_r, f, physics.p1_right_m3, omf),
        ],
        dtype=np.float64,
    )
    d_cap_log = np.diag(
        [
            vn * omf * n_l,
            vp * f * p_l,
            vn * omf * n_r,
            vp * f * p_r,
        ]
    )
    d_cap_df = np.array(
        [
            -vn * (n_l + physics.n1_left_m3),
            vp * (p_l + physics.p1_left_m3),
            -vn * (n_r + physics.n1_right_m3),
            vp * (p_r + physics.p1_right_m3),
        ],
        dtype=np.float64,
    )

    m_trace, m_bulk = _trace_and_bulk_log_maps(physics.thermal_voltage_V)
    cross = cross_eval.cross_flux_m2_s
    residual = np.array(
        [
            bulk_flux[_N_LEFT] - cross[0] - capture[_N_LEFT],
            bulk_flux[_P_LEFT] - cross[1] - capture[_P_LEFT],
            bulk_flux[_N_RIGHT] + cross[0] - capture[_N_RIGHT],
            bulk_flux[_P_RIGHT] + cross[1] - capture[_P_RIGHT],
        ],
        dtype=np.float64,
    )
    j_u = d_bulk_u - d_cap_log
    j_u[_N_LEFT, :] -= cross_eval.d_cross_d_log_state_m2_s[0, :]
    j_u[_P_LEFT, :] -= cross_eval.d_cross_d_log_state_m2_s[1, :]
    j_u[_N_RIGHT, :] += cross_eval.d_cross_d_log_state_m2_s[0, :]
    j_u[_P_RIGHT, :] += cross_eval.d_cross_d_log_state_m2_s[1, :]

    d_cross_trace_u = (
        cross_eval.d_cross_d_trace_potentials_m2_s_V
        + cross_eval.d_cross_d_log_state_m2_s @ m_trace
    )
    j_trace_u = d_bulk_trace_u - d_cap_log @ m_trace
    j_trace_u[_N_LEFT, :] -= d_cross_trace_u[0, :]
    j_trace_u[_P_LEFT, :] -= d_cross_trace_u[1, :]
    j_trace_u[_N_RIGHT, :] += d_cross_trace_u[0, :]
    j_trace_u[_P_RIGHT, :] += d_cross_trace_u[1, :]

    d_cross_bulk_u = cross_eval.d_cross_d_log_state_m2_s @ m_bulk
    j_coords_u = d_bulk_coords_u - d_cap_log @ m_bulk
    j_coords_u[_N_LEFT, :] -= d_cross_bulk_u[0, :]
    j_coords_u[_P_LEFT, :] -= d_cross_bulk_u[1, :]
    j_coords_u[_N_RIGHT, :] += d_cross_bulk_u[0, :]
    j_coords_u[_P_RIGHT, :] += d_cross_bulk_u[1, :]

    return (
        state,
        bulk_flux,
        cross,
        capture,
        residual,
        j_u,
        j_trace_u,
        j_coords_u,
        d_bulk_u,
        d_bulk_trace_u,
        d_bulk_coords_u,
        d_cap_df,
        cross_eval.one_way_scale_m2_s,
    )


def solve_fixed_occupancy_two_sided_interface(
    geometry: TraceElectrostatics,
    physics: TwoSidedInterfacePhysics,
    bulk: TwoSidedBulkState,
    charge: EquilibriumReferencedSheetCharge,
    *,
    occupancy: float,
    one_minus_occupancy: float | None = None,
    occupancy_minus_equilibrium: float | None = None,
    initial_state_m3: ArrayLike | None = None,
    residual_tolerance: float = 1.0e-9,
    max_evaluations: int = 200,
    fail_on_residual: bool = True,
) -> FixedOccupancyTwoSidedResult:
    """Eliminate two-sided electrostatic + carrier traces at explicit occupancy ``f``."""
    _positive(residual_tolerance, "residual_tolerance")
    f = float(occupancy)
    if not math.isfinite(f) or not 0.0 <= f <= 1.0:
        raise ValueError("occupancy must lie in [0, 1]")
    omf = 1.0 - f if one_minus_occupancy is None else float(one_minus_occupancy)
    nt = float(charge.trap_density_m2)
    df_eq = (
        f - float(charge.equilibrium_occupancy)
        if occupancy_minus_equilibrium is None
        else float(occupancy_minus_equilibrium)
    )
    delta_sigma = -Q * nt * df_eq
    d_sigma_df = -Q * nt

    eff_geometry = TraceElectrostatics(
        eps_r_left=geometry.eps_r_left,
        eps_r_right=geometry.eps_r_right,
        left_distance_m=geometry.left_distance_m,
        right_distance_m=geometry.right_distance_m,
        potential_jump_right_minus_left_V=geometry.potential_jump_right_minus_left_V,
        fixed_sheet_charge_C_m2=float(geometry.fixed_sheet_charge_C_m2) + delta_sigma,
    )
    phi_t_l, phi_t_r = solve_electrostatic_traces(
        eff_geometry, bulk.phi_left_V, bulk.phi_right_V
    )
    drop_l_0, drop_r_0, jump_0, _, _ = _electrostatic_seed_drops(eff_geometry, bulk)
    zero_flux_seed = _zero_bulk_flux_seed(drop_l_0, drop_r_0, physics, bulk)

    if initial_state_m3 is None:
        seed = zero_flux_seed
        u_seed_raw: NDArray[np.float64] = np.zeros(_STATE_SIZE, dtype=np.float64)
    else:
        seed = np.asarray(initial_state_m3, dtype=np.float64)
        if seed.shape != (_STATE_SIZE,) or not np.all(np.isfinite(seed)) or np.any(seed <= 0.0):
            raise ValueError("initial_state_m3 must contain four positive finite densities")
        u_seed_raw = _u_delta_from_log_state(np.log(seed), drop_l_0, drop_r_0, physics, bulk)

    density_ref = max(
        float(np.max(seed)),
        float(np.max(zero_flux_seed)),
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
    vt = float(physics.thermal_voltage_V)
    xi_left = drop_l_0 / vt
    xi_right = drop_r_0 / vt
    base_log = np.array(
        [
            math.log(float(bulk.n_left_m3)) + xi_left,
            math.log(float(bulk.p_left_m3)) - xi_left,
            math.log(float(bulk.n_right_m3)) - xi_right,
            math.log(float(bulk.p_right_m3)) + xi_right,
        ],
        dtype=np.float64,
    )
    log_lower = math.log(1.0e-100)
    log_upper = math.log(density_ref) + math.log(1.0e12)
    u_lower = np.maximum(log_lower - base_log, -490.0)
    u_upper = np.minimum(log_upper - base_log, 490.0)
    u_upper = np.maximum(u_upper, u_lower + 2.0)
    u_seed = np.clip(u_seed_raw, u_lower + 0.5, u_upper - 0.5)

    seed_terms = _fixed_occupancy_balance_terms(
        u_seed, drop_l_0, drop_r_0, jump_0, eff_geometry, physics, bulk, f, omf
    )
    e_scale = max(
        abs(float(seed_terms[1][_N_LEFT])),
        abs(float(seed_terms[1][_N_RIGHT])),
        abs(float(seed_terms[8][_N_LEFT, _N_LEFT])),
        abs(float(seed_terms[8][_N_RIGHT, _N_RIGHT])),
        abs(float(seed_terms[3][_N_LEFT])),
        abs(float(seed_terms[3][_N_RIGHT])),
        float(seed_terms[12][0]),
        1.0,
    )
    h_scale = max(
        abs(float(seed_terms[1][_P_LEFT])),
        abs(float(seed_terms[1][_P_RIGHT])),
        abs(float(seed_terms[8][_P_LEFT, _P_LEFT])),
        abs(float(seed_terms[8][_P_RIGHT, _P_RIGHT])),
        abs(float(seed_terms[3][_P_LEFT])),
        abs(float(seed_terms[3][_P_RIGHT])),
        float(seed_terms[12][1]),
        1.0,
    )
    row_scale = np.array([e_scale, h_scale, e_scale, h_scale], dtype=np.float64)

    def scaled_res(u_vec: RealArray) -> RealArray:
        return (
            _fixed_occupancy_balance_terms(
                u_vec, drop_l_0, drop_r_0, jump_0, eff_geometry, physics, bulk, f, omf
            )[4]
            / row_scale
        )

    def scaled_jac(u_vec: RealArray) -> RealArray:
        return (
            _fixed_occupancy_balance_terms(
                u_vec, drop_l_0, drop_r_0, jump_0, eff_geometry, physics, bulk, f, omf
            )[5]
            / row_scale[:, np.newaxis]
        )

    sol = least_squares(
        scaled_res,
        u_seed,
        jac=scaled_jac,
        bounds=(u_lower, u_upper),
        x_scale="jac",
        ftol=3.0e-16,
        xtol=3.0e-16,
        gtol=3.0e-16,
        max_nfev=int(max_evaluations),
    )
    x_best = sol.x.copy()
    eval_count = int(sol.nfev)

    best_terms = _fixed_occupancy_balance_terms(
        x_best, drop_l_0, drop_r_0, jump_0, eff_geometry, physics, bulk, f, omf
    )
    for _ in range(8):
        p_row = np.maximum(np.max(np.abs(best_terms[5]), axis=1), float(np.finfo(np.float64).tiny))
        p_norm = float(np.max(np.abs(best_terms[4] / p_row)))
        if p_norm <= 1.0e-16:
            break
        try:
            delta = -_equilibrated_solve(
                best_terms[5] / p_row[:, np.newaxis],
                best_terms[4] / p_row,
            )
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(delta)):
            break
        cand = np.clip(x_best + delta, u_lower, u_upper)
        cand_terms = _fixed_occupancy_balance_terms(
            cand, drop_l_0, drop_r_0, jump_0, eff_geometry, physics, bulk, f, omf
        )
        eval_count += 1
        cand_norm = float(np.max(np.abs(cand_terms[4] / p_row)))
        if cand_norm < p_norm or float(np.max(np.abs(delta))) <= 1.0e-15:
            x_best = cand
            best_terms = cand_terms
            if float(np.max(np.abs(delta))) <= 1.0e-15:
                break
        else:
            break

    (
        state,
        bulk_flux,
        cross_flux,
        capture_flux,
        residual,
        j_u,
        j_trace_u,
        j_coords_u,
        d_bulk_u,
        d_bulk_trace_u,
        d_bulk_coords_u,
        d_cap_df,
        _,
    ) = best_terms
    best_norm = float(np.max(np.abs(residual / row_scale)))

    c_l = EPS_0 * float(geometry.eps_r_left) / float(geometry.left_distance_m)
    c_r = EPS_0 * float(geometry.eps_r_right) / float(geometry.right_distance_m)
    c_sum = c_l + c_r
    w_l = c_l / c_sum
    w_r = c_r / c_sum

    d_trace_d_bulk = np.array(
        [[w_l, w_r, 0.0, 0.0, 0.0, 0.0], [w_l, w_r, 0.0, 0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    d_trace_df = np.array([d_sigma_df / c_sum, d_sigma_df / c_sum], dtype=np.float64)

    ift_row = np.maximum(np.max(np.abs(j_u), axis=1), float(np.finfo(np.float64).tiny))
    rhs_bulk = -(j_coords_u + j_trace_u @ d_trace_d_bulk)
    rhs_f = -(j_trace_u @ d_trace_df - d_cap_df)
    du_all = _equilibrated_solve(
        j_u / ift_row[:, np.newaxis],
        np.column_stack((rhs_bulk, rhs_f)) / ift_row[:, np.newaxis],
    )
    du_d_bulk = du_all[:, :6]
    du_df = du_all[:, 6]

    bulk_flux_jac = d_bulk_coords_u + d_bulk_trace_u @ d_trace_d_bulk + d_bulk_u @ du_d_bulk
    bulk_flux_df = d_bulk_trace_u @ d_trace_df + d_bulk_u @ du_df

    m_trace, m_bulk = _trace_and_bulk_log_maps(physics.thermal_voltage_V)
    d_log_state_d_bulk = m_bulk + m_trace @ d_trace_d_bulk + du_d_bulk
    d_log_state_df = m_trace @ d_trace_df + du_df

    vn = float(physics.surface_recombination_velocity_n_m_s)
    vp = float(physics.surface_recombination_velocity_p_m_s)
    n_l, p_l, n_r, p_r = (float(v) for v in state)
    filled = vn * (n_l + n_r) + vp * (physics.p1_left_m3 + physics.p1_right_m3)
    empty = vn * (physics.n1_left_m3 + physics.n1_right_m3) + vp * (p_l + p_r)
    denom = filled + empty
    f_qss = filled / denom if denom > 0.0 else f
    relax = denom / nt if nt > 0.0 else 0.0
    storage_rate = _product_difference(filled, omf, empty, f)

    d_sr_d_log = np.array(
        [vn * omf * n_l, -vp * f * p_l, vn * omf * n_r, -vp * f * p_r],
        dtype=np.float64,
    )
    sr_jac_bulk = d_sr_d_log @ d_log_state_d_bulk
    sr_df = float(-denom + d_sr_d_log @ d_log_state_df)

    converged = bool(
        (sol.success or best_norm <= residual_tolerance)
        and best_norm <= residual_tolerance
        and np.all(np.isfinite(bulk_flux_jac))
        and np.all(np.isfinite(bulk_flux_df))
        and np.all(np.isfinite(sr_jac_bulk))
        and math.isfinite(sr_df)
    )
    if fail_on_residual and not converged:
        raise RuntimeError(
            f"fixed-occupancy two-sided interface did not certify: residual={best_norm:.6g}"
        )

    return FixedOccupancyTwoSidedResult(
        phi_trace_left_V=float(phi_t_l),
        phi_trace_right_V=float(phi_t_r),
        state_m3=state,
        bulk_flux_m2_s=bulk_flux,
        cross_flux_m2_s=cross_flux,
        capture_flux_m2_s=capture_flux,
        occupancy=f,
        qss_occupancy=float(f_qss),
        relaxation_rate_s1=float(relax),
        trap_storage_rate_m2_s=float(storage_rate),
        incremental_sheet_charge_C_m2=float(delta_sigma),
        d_sheet_charge_df_C_m2=float(d_sigma_df),
        bulk_flux_jacobian_bulk=bulk_flux_jac,
        bulk_flux_derivative_occupancy=bulk_flux_df,
        trap_storage_rate_jacobian_bulk=sr_jac_bulk,
        trap_storage_rate_derivative_occupancy=sr_df,
        normalized_residual=best_norm,
        evaluations=eval_count,
        converged=converged,
        one_minus_occupancy=float(omf),
    )
