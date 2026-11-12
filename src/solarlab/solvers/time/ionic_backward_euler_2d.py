"""Conservative 2D single-mobile-ion Backward Euler stepper and port total current.

Solves ``F(u_n; V_n, g_n) - (S(u_n) - S(u_{n-1})) / dt = 0`` on the unified
:class:`AssembledIonic2DProblem` preallocated sparse graph (``M = -S_u``,
``J_BE = J + M / dt``) with cancellation-free ``expm1`` carrier storage
increments and ``finite_site_secant_2d_m3`` ion storage increments.

Provides BOTH:
1. :func:`ionic_step_accounting_2d`: conservative BE interval Maxwell port &
   cross-section current (``J_n + J_p + J_ion + (D_new - D_old)/dt``).
2. :func:`evaluate_mobile_ion_current_components_2d` /
   :func:`compute_mobile_ion_current_components_2d`: instantaneous semidiscrete
   differentiated-Poisson Maxwell current decomposition with explicit
   ``applied_voltage_rate_V_s`` (true ``dV/dt`` during ramps and ``0`` at dwell
   endpoints).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import math
import time
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import spsolve

from solarlab.contracts import frozen_array
from solarlab.contracts.units import ELEMENTARY_CHARGE_C as Q
from solarlab.discretization.face_average import VACUUM_PERMITTIVITY_F_M
from solarlab.equations.assembly.ionic_2d import AssembledIonic2DProblem
from solarlab.equations.ionic_2d import (
    CompiledIonicDevice2D, IonicState2D, MobileIonDiagnostics2D,
    assess_mobile_ion_terminal_2d, positive_ion_fluxes_2d,
)
from solarlab.physics.contacts import junction_polarity
from solarlab.solvers.nonlinear.ionic_2d import _poisson_weights_2d
from solarlab.solvers.nonlinear.newton import NewtonResult, NewtonSettings, solve_newton
from solarlab.solvers.nonlinear.problem import CallableProblem, GateOutcome, ResidualScaling, SolveBudget, StepControl
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions

__all__ = [
    "IonicBackwardEuler2DOptions",
    "IonicBackwardEuler2DRunResult",
    "IonicBackwardEuler2DStep",
    "IonicBackwardEuler2DStepAccounting",
    "IonicBackwardEuler2DStepResult",
    "MobileIonCurrentComponents2D",
    "compute_mobile_ion_current_components_2d",
    "evaluate_mobile_ion_current_components_2d",
    "ionic_step_accounting_2d",
    "run_ionic_backward_euler_2d",
    "step_ionic_backward_euler_2d",
]

Vector: TypeAlias = NDArray[np.float64]


def _read_only(values: ArrayLike) -> Vector:
    arr = np.asarray(values, dtype=float).copy()
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True, slots=True)
class MobileIonCurrentComponents2D:
    """Instantaneous 2D vertical Maxwell current decomposition [A/m^2]."""

    electron_y_A_m2: Vector
    hole_y_A_m2: Vector
    positive_ion_y_A_m2: Vector
    displacement_y_A_m2: Vector
    total_y_A_m2: Vector
    lateral_average_electron_A_m2: Vector
    lateral_average_hole_A_m2: Vector
    lateral_average_positive_ion_A_m2: Vector
    lateral_average_displacement_A_m2: Vector
    lateral_average_total_A_m2: Vector
    terminal_electron_A_m2: float
    terminal_hole_A_m2: float
    terminal_positive_ion_A_m2: float
    terminal_displacement_A_m2: float
    terminal_total_A_m2: float
    max_face_spread_A_m2: float
    max_relative_face_spread: float
    applied_voltage_rate_V_s: float


def _terminal_decomposition_is_roundoff_consistent(total: float, components: tuple[float, ...]) -> bool:
    comp_sum = sum(components)
    scale = max(abs(float(total)), sum(abs(float(v)) for v in components), float(np.finfo(float).tiny))
    return bool(abs(float(total) - comp_sum) <= 64.0 * float(np.finfo(float).eps) * scale)


def _solve_poisson_rate_2d(
    device: CompiledIonicDevice2D,
    charge_rate_C_m3: Vector,
    *,
    phi_bottom_rate_V_s: float,
    phi_top_rate_V_s: float,
) -> Vector:
    """Solve linear differentiated 2D Gauss system ``div(eps grad(dphi/dt)) = -drho/dt``."""
    comp = device.electronic_component
    g = comp.geometry
    cg = g.contract_geometry
    count = comp.node_count
    left, right = cg.pairs[:, 0], cg.pairs[:, 1]
    cap_w = comp.face_capacitance_F_m2 * cg.face_measures

    contact_mask: NDArray[np.bool_] = np.zeros(count, dtype=np.bool_)
    contact_nodes = g.contact_nodes.astype(np.intp)
    contact_mask[contact_nodes] = True

    r_chunks = [left, left, right, right]
    c_chunks = [left, right, left, right]
    # Standard discrete Laplacian -div(D) = sum_{faces} cap*(phi_node - phi_nbr)
    v_chunks = [cap_w, -cap_w, -cap_w, cap_w]
    all_r = np.concatenate(r_chunks)
    all_c = np.concatenate(c_chunks)
    all_v = np.concatenate(v_chunks)
    keep = ~contact_mask[all_r]

    mat = csc_matrix(
        (
            np.concatenate((all_v[keep], np.ones(contact_nodes.size, dtype=float))),
            (
                np.concatenate((all_r[keep], contact_nodes)),
                np.concatenate((all_c[keep], contact_nodes)),
            ),
        ),
        shape=(count, count),
    )
    rhs = cg.volumes * charge_rate_C_m3.ravel()
    rhs[:g.nx] = phi_bottom_rate_V_s
    rhs[-g.nx:] = phi_top_rate_V_s
    solved = np.asarray(spsolve(mat, rhs), dtype=float)
    return solved.reshape(g.shape)


def compute_mobile_ion_current_components_2d(
    device: CompiledIonicDevice2D,
    state: IonicState2D,
    *,
    generation_scale: float = 0.0,
    applied_voltage_rate_V_s: float = 0.0,
) -> MobileIonCurrentComponents2D:
    """Evaluate instantaneous semidiscrete 2D Maxwell current components from an ``IonicState2D``."""
    if not isinstance(device, CompiledIonicDevice2D) or not isinstance(state, IonicState2D):
        raise TypeError("CompiledIonicDevice2D and IonicState2D are required")
    if state.device_sha256 != device.identity:
        raise ValueError("IonicState2D does not match CompiledIonicDevice2D identity")
    voltage_rate = float(applied_voltage_rate_V_s)
    if not math.isfinite(voltage_rate):
        raise ValueError("applied_voltage_rate_V_s must be finite")

    problem = AssembledIonic2DProblem(device, state.voltage_V, generation_scale)
    system = problem.evaluate(state.u, derivatives=False)
    if not system.admissible:
        raise ValueError("IonicState2D is outside the admissible domain")

    comp = device.electronic_component
    g = comp.geometry
    ny, nx = g.shape
    count = comp.node_count
    elec_vol = g.contract_geometry.volumes

    # Semidiscrete continuity rates: area * dn/dt = balance_n (zero on ohmic Dirichlet contact rows)
    dn_dt = (system.balance[count:2 * count] / elec_vol).copy()
    dp_dt = (system.balance[2 * count:3 * count] / elec_vol).copy()
    dn_dt[comp.fixed_carrier_mask[0]] = 0.0
    dp_dt[comp.fixed_carrier_mask[1]] = 0.0

    ion_vol = device.ion_geometry.contract_geometry.volumes
    dp_ion_domain_dt = system.balance[3 * count:] / ion_vol
    dp_ion_elec: Vector = np.zeros(count, dtype=np.float64)
    dp_ion_elec[device.ion_node_indices] = dp_ion_domain_dt * device.ion_charge_weight_ratio

    charge_rate = Q * (dp_dt - dn_dt + dp_ion_elec)
    polarity = float(junction_polarity(comp.column.built_in_potential_V))
    phi_rate = _solve_poisson_rate_2d(
        device,
        charge_rate,
        phi_bottom_rate_V_s=0.0,
        phi_top_rate_V_s=-polarity * voltage_rate,
    )

    dy = np.diff(g.y_geometry.nodes_m)[:, None]
    field_rate_y = -(phi_rate[1:, :] - phi_rate[:-1, :]) / dy
    eps_r = comp.relative_permittivity_y
    eps_face_y = (2.0 * eps_r[:-1] * eps_r[1:] / (eps_r[:-1] + eps_r[1:]))[:, None]
    displacement_y = VACUUM_PERMITTIVITY_F_M * eps_face_y * field_rate_y

    c_fluxes = problem.carrier_fluxes(state.u)
    electron_y = c_fluxes.electron_current()[g.x_face_count:].reshape(ny - 1, nx)
    hole_y = c_fluxes.hole_current()[g.x_face_count:].reshape(ny - 1, nx)

    ion_eval = problem.ion_flux_evaluation(state.u)
    ion_g = device.ion_geometry
    ion_y_domain = Q * ion_eval.flux[ion_g.x_face_count:].reshape(ion_g.ny - 1, nx)
    positive_ion_y: Vector = np.zeros((ny - 1, nx), dtype=np.float64)
    row_start = int(device.ion_row_indices[0])
    positive_ion_y[row_start:row_start + ion_g.ny - 1, :] = ion_y_domain

    total_y = electron_y + hole_y + positive_ion_y + displacement_y
    if not all(np.all(np.isfinite(arr)) for arr in (electron_y, hole_y, positive_ion_y, displacement_y, total_y)):
        raise ValueError("2D mobile-ion current decomposition produced non-finite values")

    widths = g.x_widths_m
    domain_w = g.lateral_length_m
    avg_e = (electron_y @ widths) / domain_w
    avg_h = (hole_y @ widths) / domain_w
    avg_ion = (positive_ion_y @ widths) / domain_w
    avg_disp = (displacement_y @ widths) / domain_w
    avg_tot = (total_y @ widths) / domain_w

    term_idx = -1
    term_comps = (
        float(avg_e[term_idx]),
        float(avg_h[term_idx]),
        float(avg_ion[term_idx]),
        float(avg_disp[term_idx]),
    )
    term_tot = float(avg_tot[term_idx])
    if not _terminal_decomposition_is_roundoff_consistent(term_tot, term_comps):
        raise RuntimeError("2D mobile-ion terminal current decomposition is inconsistent")

    spread = float(np.ptp(avg_tot))
    scale = max(float(np.max(np.abs(avg_tot))), 1.0e-30)

    return MobileIonCurrentComponents2D(
        electron_y_A_m2=_read_only(electron_y),
        hole_y_A_m2=_read_only(hole_y),
        positive_ion_y_A_m2=_read_only(positive_ion_y),
        displacement_y_A_m2=_read_only(displacement_y),
        total_y_A_m2=_read_only(total_y),
        lateral_average_electron_A_m2=_read_only(avg_e),
        lateral_average_hole_A_m2=_read_only(avg_h),
        lateral_average_positive_ion_A_m2=_read_only(avg_ion),
        lateral_average_displacement_A_m2=_read_only(avg_disp),
        lateral_average_total_A_m2=_read_only(avg_tot),
        terminal_electron_A_m2=term_comps[0],
        terminal_hole_A_m2=term_comps[1],
        terminal_positive_ion_A_m2=term_comps[2],
        terminal_displacement_A_m2=term_comps[3],
        terminal_total_A_m2=term_tot,
        max_face_spread_A_m2=spread,
        max_relative_face_spread=spread / scale,
        applied_voltage_rate_V_s=voltage_rate,
    )


def evaluate_mobile_ion_current_components_2d(
    device: CompiledIonicDevice2D,
    phi_2d: ArrayLike,
    Jy_n_2d: ArrayLike,
    Jy_p_2d: ArrayLike,
    P_ion_2d: ArrayLike,
    state_derivative_3block: ArrayLike,
    *,
    applied_voltage_rate_V_s: float = 0.0,
) -> MobileIonCurrentComponents2D:
    """Evaluate 2D Maxwell current components from explicit snapshot arrays & 3-block density RHS."""
    comp = device.electronic_component
    g = comp.geometry
    ny, nx = g.shape
    voltage_rate = float(applied_voltage_rate_V_s)
    if not math.isfinite(voltage_rate):
        raise ValueError("applied_voltage_rate_V_s must be finite")

    phi = np.asarray(phi_2d, dtype=float)
    jy_n = np.asarray(Jy_n_2d, dtype=float)
    jy_p = np.asarray(Jy_p_2d, dtype=float)
    p_ion = np.asarray(P_ion_2d, dtype=float)
    deriv = np.asarray(state_derivative_3block, dtype=float)
    if phi.shape != (ny, nx) or p_ion.shape != (ny, nx):
        raise ValueError("phi_2d and P_ion_2d must have shape (Ny, Nx)")
    if jy_n.shape != (ny - 1, nx) or jy_p.shape != (ny - 1, nx):
        raise ValueError("Jy_n_2d and Jy_p_2d must have shape (Ny - 1, Nx)")
    if deriv.shape != (3 * ny * nx,) or not np.all(np.isfinite(deriv)):
        raise ValueError("state_derivative_3block must be finite with shape (3 * Ny * Nx,)")

    dn = deriv[:ny * nx]
    dp = deriv[ny * nx:2 * ny * nx]
    d_pion = deriv[2 * ny * nx:]
    charge_rate = Q * (dp - dn + d_pion)
    polarity = float(junction_polarity(comp.column.built_in_potential_V))
    phi_rate = _solve_poisson_rate_2d(
        device,
        charge_rate,
        phi_bottom_rate_V_s=0.0,
        phi_top_rate_V_s=-polarity * voltage_rate,
    )
    dy = np.diff(g.y_geometry.nodes_m)[:, None]
    field_rate_y = -(phi_rate[1:, :] - phi_rate[:-1, :]) / dy
    eps_r = comp.relative_permittivity_y
    eps_face_y = (2.0 * eps_r[:-1] * eps_r[1:] / (eps_r[:-1] + eps_r[1:]))[:, None]
    displacement_y = VACUUM_PERMITTIVITY_F_M * eps_face_y * field_rate_y

    from solarlab.physics.ions import DIFFUSION_ONLY_STERIC_V1
    ion_flux = positive_ion_fluxes_2d(
        g.x_m,
        g.y_geometry.nodes_m,
        phi,
        p_ion,
        device.full_grid_ion_diffusivity_m2_s(),
        comp.thermal_voltage_V,
        device.full_grid_ion_site_limit_m3(),
        steric_diffusion_only=(device.steric_model == DIFFUSION_ONLY_STERIC_V1),
    )
    positive_ion_y = Q * ion_flux.y
    total_y = jy_n + jy_p + positive_ion_y + displacement_y

    widths = g.x_widths_m
    domain_w = g.lateral_length_m
    avg_e = (jy_n @ widths) / domain_w
    avg_h = (jy_p @ widths) / domain_w
    avg_ion = (positive_ion_y @ widths) / domain_w
    avg_disp = (displacement_y @ widths) / domain_w
    avg_tot = (total_y @ widths) / domain_w
    term_comps = (float(avg_e[-1]), float(avg_h[-1]), float(avg_ion[-1]), float(avg_disp[-1]))
    term_tot = float(avg_tot[-1])
    if not _terminal_decomposition_is_roundoff_consistent(term_tot, term_comps):
        raise RuntimeError("2D mobile-ion terminal current decomposition is inconsistent")
    spread = float(np.ptp(avg_tot))
    scale = max(float(np.max(np.abs(avg_tot))), 1.0e-30)

    return MobileIonCurrentComponents2D(
        electron_y_A_m2=_read_only(jy_n),
        hole_y_A_m2=_read_only(jy_p),
        positive_ion_y_A_m2=_read_only(positive_ion_y),
        displacement_y_A_m2=_read_only(displacement_y),
        total_y_A_m2=_read_only(total_y),
        lateral_average_electron_A_m2=_read_only(avg_e),
        lateral_average_hole_A_m2=_read_only(avg_h),
        lateral_average_positive_ion_A_m2=_read_only(avg_ion),
        lateral_average_displacement_A_m2=_read_only(avg_disp),
        lateral_average_total_A_m2=_read_only(avg_tot),
        terminal_electron_A_m2=term_comps[0],
        terminal_hole_A_m2=term_comps[1],
        terminal_positive_ion_A_m2=term_comps[2],
        terminal_displacement_A_m2=term_comps[3],
        terminal_total_A_m2=term_tot,
        max_face_spread_A_m2=spread,
        max_relative_face_spread=spread / scale,
        applied_voltage_rate_V_s=voltage_rate,
    )


@dataclass(frozen=True)
class IonicBackwardEuler2DStep:
    """One conservative 2D single-mobile-ion Backward Euler nonlinear algebraic problem."""

    current: AssembledIonic2DProblem
    previous: AssembledIonic2DProblem
    previous_u: Vector
    dt_s: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.dt_s) or self.dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        if self.current.binding.identity != self.previous.binding.identity:
            raise ValueError("current and previous 2D ionic problems must share one bound system")
        object.__setattr__(self, "previous_u", frozen_array(self.previous_u))

    def residual(self, u: ArrayLike) -> Vector:
        res = self.current.residual(u)
        if not np.all(np.isfinite(res)):
            return res
        try:
            delta_s = self.current.storage_increment(u, self.previous, self.previous_u)
        except FloatingPointError:
            return np.full_like(res, np.nan)
        return res - delta_s / self.dt_s

    def jacobian(self, u: ArrayLike) -> csc_matrix:
        jac = self.current.jacobian(u)
        mass = self.current.mass_matrix(u)
        return cast(csc_matrix, (jac + mass / self.dt_s).tocsc())


@dataclass(frozen=True)
class IonicBackwardEuler2DOptions:
    """Explicit numerical controls and conservation gates for 2D mobile-ion BE."""

    electronic: SteadyOptions = field(
        default_factory=lambda: SteadyOptions(continuity_tolerance_A_m2=1.0e-5)
    )
    ion_inventory_rtol: float = 1.0e-12
    ion_step_rtol: float = 1.0e-13
    ion_logit_step: float = 1.0
    budget: SolveBudget = field(default_factory=lambda: SolveBudget(65, 650, 70, 24))
    max_bisect: int = 8
    wall_s: float = 90.0

    def __post_init__(self) -> None:
        if not isinstance(self.electronic, SteadyOptions) or not isinstance(self.budget, SolveBudget):
            raise ValueError("SteadyOptions and SolveBudget are required")
        if type(self.max_bisect) is not int or self.max_bisect < 0:
            raise ValueError("max_bisect must be a non-negative integer")
        for name in ("ion_inventory_rtol", "ion_step_rtol", "ion_logit_step", "wall_s"):
            val = float(getattr(self, name))
            if not math.isfinite(val) or val <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class IonicBackwardEuler2DStepAccounting:
    """Conservative BE interval Maxwell port & cross-section total current [A/m^2]."""

    dt_s: float
    terminal_carrier_A_m2: Vector
    terminal_ion_A_m2: Vector
    terminal_displacement_A_m2: Vector
    terminal_total_A_m2: Vector
    cross_section_carrier_A_m2: Vector
    cross_section_ion_A_m2: Vector
    cross_section_displacement_A_m2: Vector
    cross_section_total_A_m2: Vector
    max_total_current_spread_A_m2: float
    ion_inventory_m1: float
    relative_ion_inventory_step_drift: float


def ionic_step_accounting_2d(
    current: AssembledIonic2DProblem,
    u_new: ArrayLike,
    previous: AssembledIonic2DProblem,
    u_old: ArrayLike,
    dt_s: float,
) -> IonicBackwardEuler2DStepAccounting:
    """Compute conservative BE interval carrier + ion + displacement Maxwell current."""
    d = current.device
    comp = d.electronic_component
    g = comp.geometry
    width = g.lateral_length_m

    delta_s_phys = current.physical_increment(u_new, previous, u_old)
    storage_rate = delta_s_phys / dt_s

    electron_port, hole_port = current.contact_particle_fluxes(u_new, storage_rate=storage_rate)
    j_carrier_ports = np.sum((-Q * electron_port + Q * hole_port) * g.x_widths_m[None, :], axis=1) / width
    # Exterior electrodes block ions; if ion domain reaches y_min/y_max, reaction_flux on ion is 0
    j_ion_ports: Vector = np.zeros(2, dtype=float)

    d_port_new = current.contact_displacement_C_m2(u_new)
    d_port_old = previous.contact_displacement_C_m2(u_old)
    j_disp_ports = np.sum(((d_port_new - d_port_old) / dt_s) * g.x_widths_m[None, :], axis=1) / width
    j_tot_ports = j_carrier_ports + j_ion_ports + j_disp_ports

    c_flux = current.carrier_fluxes(u_new)
    j_carrier_y = (-Q * c_flux.electron.flux[g.x_face_count:] + Q * c_flux.hole.flux[g.x_face_count:]).reshape(
        g.ny - 1, g.nx
    )
    ion_eval = current.ion_flux_evaluation(u_new)
    ion_g = d.ion_geometry
    j_ion_y: Vector = np.zeros((g.ny - 1, g.nx), dtype=np.float64)
    row0 = int(d.ion_row_indices[0])
    j_ion_y[row0:row0 + ion_g.ny - 1, :] = Q * ion_eval.flux[ion_g.x_face_count:].reshape(ion_g.ny - 1, g.nx)

    phi_new, _, _, ion_p_new = d.fields(u_new, voltage_V=current.voltage_V)
    phi_old, _, _, ion_p_old = d.fields(u_old, voltage_V=previous.voltage_V)
    left, right = g.contract_geometry.pairs[g.x_face_count:].T
    cap_y = comp.face_capacitance_F_m2[g.x_face_count:]
    d_y_new = (-cap_y * (phi_new[right] - phi_new[left])).reshape(g.ny - 1, g.nx)
    d_y_old = (-cap_y * (phi_old[right] - phi_old[left])).reshape(g.ny - 1, g.nx)
    j_disp_y = (d_y_new - d_y_old) / dt_s

    sec_carrier = np.sum(j_carrier_y * g.x_widths_m[None, :], axis=1) / width
    sec_ion = np.sum(j_ion_y * g.x_widths_m[None, :], axis=1) / width
    sec_disp = np.sum(j_disp_y * g.x_widths_m[None, :], axis=1) / width
    sec_tot = sec_carrier + sec_ion + sec_disp
    all_levels = np.r_[j_tot_ports[:1], sec_tot, j_tot_ports[1:]]

    ion_vol = ion_g.contract_geometry.volumes
    inv_new = math.fsum((ion_vol * ion_p_new).tolist())
    inv_old = math.fsum((ion_vol * ion_p_old).tolist())
    step_drift = abs(inv_new - inv_old) / max(abs(inv_old), 1.0e-30)

    return IonicBackwardEuler2DStepAccounting(
        dt_s=dt_s,
        terminal_carrier_A_m2=frozen_array(j_carrier_ports),
        terminal_ion_A_m2=frozen_array(j_ion_ports),
        terminal_displacement_A_m2=frozen_array(j_disp_ports),
        terminal_total_A_m2=frozen_array(j_tot_ports),
        cross_section_carrier_A_m2=frozen_array(sec_carrier),
        cross_section_ion_A_m2=frozen_array(sec_ion),
        cross_section_displacement_A_m2=frozen_array(sec_disp),
        cross_section_total_A_m2=frozen_array(sec_tot),
        max_total_current_spread_A_m2=float(np.ptp(all_levels)),
        ion_inventory_m1=inv_new,
        relative_ion_inventory_step_drift=step_drift,
    )


@dataclass(frozen=True)
class IonicBackwardEuler2DStepResult:
    state: IonicState2D | None
    accepted: bool
    termination: str
    dt_s: float
    newton: NewtonResult | None
    accounting: IonicBackwardEuler2DStepAccounting | None
    wall_s: float
    error: str | None = None


@dataclass(frozen=True)
class IonicBackwardEuler2DRunResult:
    times_s: Vector
    states: tuple[IonicState2D, ...]
    accountings: tuple[IonicBackwardEuler2DStepAccounting, ...]
    ion_diagnostics: MobileIonDiagnostics2D
    accepted: bool
    termination: str
    attempted_steps: int
    accepted_steps: int
    rejected_steps: int
    bisections: int
    wall_s: float
    error: str | None = None
    scientific_qualification: bool = False
    total_newton_iterations: int = 0


def _rebase_ionic_u(device: CompiledIonicDevice2D, u_old: Vector, old_V: float, new_V: float) -> Vector:
    u_guess = np.array(u_old, copy=True)
    if old_V == new_V:
        return u_guess
    comp = device.electronic_component
    n_elec = comp.node_count
    old_e = comp.column.carrier_reference_energies(old_V)
    new_e = comp.column.carrier_reference_energies(new_V)
    for offset, old_val, new_val in zip((n_elec, 2 * n_elec), old_e, new_e):
        u_guess[offset:offset + n_elec] += (old_val - new_val) / comp.thermal_voltage_V
    resistance = 1.0 / comp.column.poisson.face_capacitance_F_m2
    profile = np.r_[0.0, np.cumsum(resistance)]
    profile /= profile[-1]
    change = -comp.column.voltage_polarity * (new_V - old_V)
    before = u_guess[:n_elec].copy()
    u_guess[:n_elec] += comp.extrude(change * profile)
    bottom, top = comp.column.contact_potentials(new_V)
    u_guess[:comp.geometry.nx] = bottom
    u_guess[n_elec - comp.geometry.nx:n_elec] = top
    shift = (u_guess[:n_elec] - before) / comp.thermal_voltage_V
    u_guess[n_elec:2 * n_elec] -= shift
    u_guess[2 * n_elec:3 * n_elec] -= shift
    return u_guess


def _generation_inventory_ionic_seed_2d(
    device: CompiledIonicDevice2D,
    step_prob: IonicBackwardEuler2DStep,
    u_base: Vector,
    previous_state: IonicState2D,
    *,
    dt_s: float,
    voltage_V: float,
    generation_scale: float,
) -> Vector:
    comp = device.electronic_component
    n_elec = comp.node_count
    try:
        _, n_cur, p_cur, _ = device.fields(u_base, voltage_V=voltage_V)
        res_c = step_prob.residual(u_base)[n_elec:3 * n_elec]
        if np.all(np.isfinite(res_c)) and np.all(n_cur > 0.0) and np.all(p_cur > 0.0):
            inv_d = np.r_[1.0 / n_cur, -1.0 / p_cur]
            mat = step_prob.jacobian(u_base)[n_elec:3 * n_elec, n_elec:3 * n_elec] @ diags(inv_d)
            inc = np.asarray(spsolve(mat.tocsc(), -res_c), dtype=float)
            trial_n, trial_p = n_cur + inc[:n_elec], p_cur + inc[n_elec:]
            if np.all(np.isfinite(trial_n)) and np.all(np.isfinite(trial_p)) and np.all(trial_n > 0.0) and np.all(trial_p > 0.0):
                u_lin = np.array(u_base, copy=True)
                for offset, dens, trial, delta, sign in (
                    (n_elec, n_cur, trial_n, inc[:n_elec], 1.0),
                    (2 * n_elec, p_cur, trial_p, inc[n_elec:], -1.0),
                ):
                    with np.errstate(over="ignore", divide="ignore"):
                        chg = np.log1p(delta / dens)
                    bad = ~np.isfinite(chg)
                    chg[bad] = np.log(trial[bad]) - np.log(dens[bad])
                    u_lin[offset:offset + n_elec] += sign * chg
                if np.all(np.isfinite(u_lin)):
                    return u_lin
    except Exception:
        pass
    u_seed = np.array(u_base, copy=True)
    if generation_scale <= 0.0:
        return u_seed
    increment = dt_s * generation_scale * comp.generation_source_m3_s
    fixed = comp.fixed_carrier_mask
    elec_state = previous_state.electronic_state
    for species, (density_2d, offset, sign) in enumerate((
        (elec_state.electron_density_m3, n_elec, 1.0),
        (elec_state.hole_density_m3, 2 * n_elec, -1.0),
    )):
        density = density_2d.ravel()
        addition = np.where(~fixed[species], increment, 0.0)
        u_seed[offset:offset + n_elec] += sign * np.log1p(addition / np.maximum(density, 1.0e-300))
    return u_seed


def step_ionic_backward_euler_2d(
    device: CompiledIonicDevice2D,
    previous_state: IonicState2D,
    *,
    dt_s: float,
    voltage_V: float,
    generation_scale: float,
    previous_generation_scale: float | None = None,
    options: IonicBackwardEuler2DOptions | None = None,
) -> IonicBackwardEuler2DStepResult:
    """Advance one conservative 2D single-mobile-ion Backward Euler step."""
    started = time.monotonic()
    if previous_state.device_sha256 != device.identity:
        raise ValueError("previous_state must match CompiledIonicDevice2D identity")
    controls = IonicBackwardEuler2DOptions() if options is None else options
    prev_g = generation_scale if previous_generation_scale is None else previous_generation_scale
    prev_prob = AssembledIonic2DProblem(device, previous_state.voltage_V, prev_g)
    curr_prob = AssembledIonic2DProblem(device, voltage_V, generation_scale)
    step_prob = IonicBackwardEuler2DStep(curr_prob, prev_prob, previous_state.u, dt_s)

    c = controls.electronic
    comp = device.electronic_component
    n_elec = comp.node_count
    n_ion = device.ion_node_count
    weights: Vector = np.empty(device.state_size, dtype=float)
    weights[:n_elec] = _poisson_weights_2d(device) / c.poisson_tolerance
    weights[n_elec:3 * n_elec] = 2.0 * n_elec * Q / (
        comp.geometry.lateral_length_m * c.continuity_tolerance_A_m2
    )
    weights[n_elec:3 * n_elec].reshape(2, n_elec)[comp.fixed_carrier_mask] = 1.0 / c.contact_log_tolerance
    # Tight per-row ion balance scaling guarantees dt * sum(|R_ion|) <= ion_step_rtol * I_ion
    ion_row_tol = max(device.reference_inventory_m1 * controls.ion_step_rtol / (n_ion * dt_s), 1.0e-30)
    weights[3 * n_elec:] = 1.0 / ion_row_tol
    steps = np.r_[
        np.full(n_elec, c.potential_step_V),
        np.full(2 * n_elec, c.quasi_fermi_step),
        np.full(n_ion, controls.ion_logit_step),
    ]

    def gate_fn(cand_u: Vector) -> GateOutcome:
        if time.monotonic() - started > controls.wall_s:
            return GateOutcome("wall_budget", False, time.monotonic() - started, controls.wall_s)
        res = step_prob.residual(cand_u)
        if not np.all(np.isfinite(res)):
            return GateOutcome("finite_ionic_be_residual", False, math.inf, 1.0)
        scaled = float(np.max(np.abs(res * weights)))
        return GateOutcome("ionic_be_2d_scaled_max", scaled <= 1.0, scaled, 1.0)

    settings = NewtonSettings(ResidualScaling(weights), 1.0, StepControl(steps), controls.budget, (gate_fn,), False)
    u_rebased = _rebase_ionic_u(device, previous_state.u, previous_state.voltage_V, voltage_V)
    u_guess = (
        _generation_inventory_ionic_seed_2d(
            device, step_prob, u_rebased, previous_state,
            dt_s=dt_s, voltage_V=voltage_V, generation_scale=generation_scale,
        )
        if generation_scale > prev_g
        else u_rebased
    )
    prob = CallableProblem(step_prob.residual, step_prob.jacobian)
    try:
        newton = solve_newton(prob, u_guess, settings)
        if not newton.accepted and generation_scale <= prev_g:
            u_alt = _generation_inventory_ionic_seed_2d(
                device, step_prob, u_rebased, previous_state,
                dt_s=dt_s, voltage_V=voltage_V, generation_scale=generation_scale,
            )
            newton_alt = solve_newton(prob, u_alt, settings)
            if newton_alt.accepted:
                newton = newton_alt
    except Exception as exc:
        return IonicBackwardEuler2DStepResult(
            None, False, "exception", dt_s, None, None, time.monotonic() - started, str(exc)
        )
    if not newton.accepted:
        return IonicBackwardEuler2DStepResult(
            None, False, newton.termination, dt_s, newton, None, time.monotonic() - started,
            "2D mobile-ion Backward Euler step did not converge",
        )
    state = device.state(newton.state, voltage_V=voltage_V)
    acct = ionic_step_accounting_2d(curr_prob, newton.state, prev_prob, previous_state.u, dt_s)
    if acct.relative_ion_inventory_step_drift > controls.ion_inventory_rtol:
        return IonicBackwardEuler2DStepResult(
            None, False, "inventory_drift_exceeded", dt_s, newton, acct, time.monotonic() - started,
            f"step ion inventory drift {acct.relative_ion_inventory_step_drift:.3e} exceeded {controls.ion_inventory_rtol:.3e}",
        )
    return IonicBackwardEuler2DStepResult(state, True, "converged", dt_s, newton, acct, time.monotonic() - started)


def run_ionic_backward_euler_2d(
    device: CompiledIonicDevice2D,
    initial_state: IonicState2D,
    time_grid_s: ArrayLike,
    *,
    voltage_fn: Callable[[float], float],
    generation_fn: Callable[[float], float],
    options: IonicBackwardEuler2DOptions | None = None,
) -> IonicBackwardEuler2DRunResult:
    """Advance 2D single-mobile-ion transient across ``time_grid_s`` with bisection recovery."""
    started = time.monotonic()
    controls = IonicBackwardEuler2DOptions() if options is None else options
    grid = np.asarray(time_grid_s, dtype=float)
    if grid.ndim != 1 or grid.size < 2 or not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0.0):
        raise ValueError("time_grid_s must be 1D, finite, and strictly increasing")

    times = [float(grid[0])]
    states = [initial_state]
    accountings: list[IonicBackwardEuler2DStepAccounting] = []
    attempted = accepted = rejected = bisections = total_newton = 0
    curr_state = initial_state
    prev_g = float(generation_fn(float(grid[0])))

    def make_diag(final_state: IonicState2D) -> MobileIonDiagnostics2D:
        return assess_mobile_ion_terminal_2d(
            device.ion_geometry.x_m,
            device.ion_geometry.y_geometry.nodes_m,
            initial_state.positive_ion_density_m3,
            final_state.positive_ion_density_m3,
            device.positive_ion_site_limit_m3.reshape(device.ion_geometry.shape),
            terminal_electron_density=final_state.electronic_state.electron_density_m3,
            terminal_hole_density=final_state.electronic_state.hole_density_m3,
            inventory_rtol=controls.ion_inventory_rtol,
            control_volume_areas_m2=device.ion_geometry.contract_geometry.volumes.reshape(device.ion_geometry.shape),
        )

    for seg_idx in range(grid.size - 1):
        t_cur = float(grid[seg_idx])
        t_target = float(grid[seg_idx + 1])
        dt = t_target - t_cur
        depth = 0
        while t_cur < t_target - 1.0e-15 * max(1.0, abs(t_target)):
            if time.monotonic() - started > controls.wall_s:
                return IonicBackwardEuler2DRunResult(
                    frozen_array(times), tuple(states), tuple(accountings), make_diag(curr_state),
                    False, "wall_budget_exceeded", attempted, accepted, rejected, bisections,
                    time.monotonic() - started, "2D mobile-ion BE wall budget exceeded",
                    total_newton_iterations=total_newton,
                )
            dt_try = min(dt, t_target - t_cur)
            t_next = t_cur + dt_try
            v_next = float(voltage_fn(t_next))
            g_next = float(generation_fn(t_next))
            attempted += 1
            res = step_ionic_backward_euler_2d(
                device, curr_state, dt_s=dt_try, voltage_V=v_next,
                generation_scale=g_next, previous_generation_scale=prev_g, options=controls,
            )
            if res.newton is not None:
                total_newton += len(res.newton.iterations)
            if res.accepted and res.state is not None and res.accounting is not None:
                accepted += 1
                curr_state = res.state
                prev_g = g_next
                t_cur = t_next
                times.append(t_cur)
                states.append(curr_state)
                accountings.append(res.accounting)
            else:
                rejected += 1
                if depth >= controls.max_bisect:
                    return IonicBackwardEuler2DRunResult(
                        frozen_array(times), tuple(states), tuple(accountings), make_diag(curr_state),
                        False, res.termination, attempted, accepted, rejected, bisections,
                        time.monotonic() - started, res.error or "2D mobile-ion BE bisection budget exhausted",
                        total_newton_iterations=total_newton,
                    )
                depth += 1
                bisections += 1
                dt *= 0.5

    diag = make_diag(curr_state)
    ok = diag.passed
    return IonicBackwardEuler2DRunResult(
        frozen_array(times), tuple(states), tuple(accountings), diag,
        ok, "completed" if ok else "terminal_gate_failed",
        attempted, accepted, rejected, bisections, time.monotonic() - started,
        None if ok else ("; ".join(diag.violations) or "terminal mobile-ion gate failed"),
        total_newton_iterations=total_newton,
    )
