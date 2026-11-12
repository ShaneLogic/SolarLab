"""Conservative 2D Backward Euler stepper and Gauss-Maxwell port accounting (ion-free/frozen).

Solves ``F(u_n; V_n, g_n) - (S(u_n) - S(u_{n-1})) / dt = 0`` on the unified
:class:`AssembledStorage2D` preallocated sparse graph (``M = -S_u``,
``J_BE = J + M / dt``). Carrier-only steady current readers explicitly refuse
active mobile-ion states so incomplete terminal currents can never be read out
silently.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import math
import time
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import spsolve

from solarlab.contracts import frozen_array
from solarlab.contracts.units import ELEMENTARY_CHARGE_C as Q
from solarlab.equations.assembly.ionic_2d import AssembledStorage2D
from solarlab.equations.assembly.two_dimensional import ELECTRON, HOLE, POTENTIAL, SIDES
from solarlab.equations.dc_1d import CapabilityError
from solarlab.equations.dc_2d import CompiledDevice2D, DC2DProblem, State2D
from solarlab.equations.ionic_2d import (
    IonicState2D, RadiativeReabsorptionSpec2D, TwoSidedInterfaceSRHCoupling2D,
)
from solarlab.mesh.geometry_2d import RECTANGULAR_2D_NAMES
from solarlab.solvers.nonlinear.newton import NewtonResult, NewtonSettings, solve_newton
from solarlab.solvers.nonlinear.problem import CallableProblem, GateOutcome, ResidualScaling, SolveBudget, StepControl
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions

__all__ = [
    "BackwardEuler2DOptions",
    "BackwardEuler2DRunResult",
    "BackwardEuler2DStep",
    "BackwardEuler2DStepAccounting",
    "BackwardEuler2DStepResult",
    "compute_carrier_only_terminal_current_2d",
    "run_backward_euler_2d",
    "step_accounting_2d",
    "step_backward_euler_2d",
]

Vector: TypeAlias = NDArray[np.float64]


def compute_carrier_only_terminal_current_2d(
    device: CompiledDevice2D,
    state: Any,
    *,
    generation_scale: float = 0.0,
    radiative_reabsorption: tuple[RadiativeReabsorptionSpec2D, ...] = (),
    interface_srh_couplings: tuple[TwoSidedInterfaceSRHCoupling2D, ...] = (),
) -> Vector:
    """Carrier-only 2D terminal current density [A/m^2]; refuses mobile-ion states."""
    if isinstance(state, IonicState2D) or getattr(state, "positive_ion_density_m3", None) is not None:
        raise CapabilityError(
            "carrier_only_current_refuses_mobile_ion_2d: use ionic_step_accounting_2d or "
            "evaluate_mobile_ion_current_components_2d for mobile-ion states"
        )
    if getattr(state, "P_ion", None) is not None:
        raise CapabilityError(
            "carrier_only_current_refuses_mobile_ion_2d: snapshot carries P_ion; use complete 2D ion+displacement current"
        )
    if not isinstance(state, State2D) or state.device_sha256 != device.identity:
        raise ValueError("State2D bound to this exact CompiledDevice2D is required")
    if not radiative_reabsorption and not interface_srh_couplings:
        problem = DC2DProblem(device, state.voltage_V, generation_scale)
        return problem.terminal_current_density_A_m2(state.u)
    storage_prob = AssembledStorage2D(
        device,
        state.voltage_V,
        generation_scale,
        radiative_reabsorption=radiative_reabsorption,
        interface_srh_couplings=interface_srh_couplings,
    )
    sys_eval = storage_prob.evaluate(state.u, derivatives=False)
    if not sys_eval.admissible:
        raise ValueError("State2D is outside the admissible domain")
    assembler = storage_prob.binding.assembler
    g = device.geometry
    sides = g.contact_nodes.reshape(2, g.nx)
    ports: list[Vector] = []
    for block, carrier_code in ((ELECTRON, "n"), (HOLE, "p")):
        rows: list[Vector] = []
        for (side_name, port_id), nodes in zip(SIDES, sides):
            ch = device.column.channel(carrier_code, side_name)
            if ch.kind == "fixed_reservoir":
                rows.append(assembler.reaction_flux(sys_eval, block, port_id))
            else:
                dc_prob = DC2DProblem(device, state.voltage_V, generation_scale)
                exch, _ = dc_prob._contact_exchange(np.asarray(state.u, dtype=np.float64), ch, nodes)
                rows.append(exch)
        ports.append(np.stack(rows))
    return np.sum((-Q * ports[0] + Q * ports[1]) * g.x_widths_m[None, :], axis=1) / g.lateral_length_m


@dataclass(frozen=True)
class BackwardEuler2DStep:
    """One conservative 2D Backward Euler nonlinear algebraic problem."""

    current: AssembledStorage2D
    previous: AssembledStorage2D
    previous_u: Vector
    dt_s: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.dt_s) or self.dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        if self.current.binding.identity != self.previous.binding.identity:
            raise ValueError("current and previous 2D problems must share one bound system")
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
class BackwardEuler2DOptions:
    """Explicit numerical controls for 2D conservative Backward Euler."""

    electronic: SteadyOptions = field(
        default_factory=lambda: SteadyOptions(continuity_tolerance_A_m2=1.0e-5)
    )
    budget: SolveBudget = field(default_factory=lambda: SolveBudget(60, 600, 65, 24))
    max_bisect: int = 8
    wall_s: float = 60.0
    radiative_reabsorption: tuple[RadiativeReabsorptionSpec2D, ...] = ()
    interface_srh_couplings: tuple[TwoSidedInterfaceSRHCoupling2D, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.electronic, SteadyOptions) or not isinstance(self.budget, SolveBudget):
            raise ValueError("SteadyOptions and SolveBudget are required")
        if type(self.max_bisect) is not int or self.max_bisect < 0:
            raise ValueError("max_bisect must be a non-negative integer")
        if not math.isfinite(self.wall_s) or self.wall_s <= 0.0:
            raise ValueError("wall_s must be finite and positive")


@dataclass(frozen=True)
class BackwardEuler2DStepAccounting:
    """Conservative BE interval terminal & cross-section total Maxwell current [A/m^2]."""

    dt_s: float
    terminal_conduction_A_m2: Vector
    terminal_displacement_A_m2: Vector
    terminal_total_A_m2: Vector
    cross_section_total_A_m2: Vector
    max_total_current_spread_A_m2: float


def step_accounting_2d(
    current: AssembledStorage2D,
    u_new: ArrayLike,
    previous: AssembledStorage2D,
    u_old: ArrayLike,
    dt_s: float,
) -> BackwardEuler2DStepAccounting:
    """Compute conservative BE interval conduction + displacement port current."""
    d = current.device
    g = d.geometry
    width = g.lateral_length_m
    sys_new = current.evaluate(u_new, derivatives=False)
    sys_old = previous.evaluate(u_old, derivatives=False)
    if not sys_new.admissible or not sys_old.admissible:
        raise ValueError("2D BE step accounting requires admissible states")

    assembler = current.binding.assembler
    delta_s_phys = current.physical_increment(u_new, previous, u_old)
    storage_rate = delta_s_phys / dt_s

    # Port carrier particle fluxes along +y at y_min and y_max
    sides = g.contact_nodes.reshape(2, g.nx)
    particle_ports: list[Vector] = []
    for block, carrier_code in ((ELECTRON, "n"), (HOLE, "p")):
        rows: list[Vector] = []
        for (side_name, port_id), nodes in zip(SIDES, sides):
            ch = d.column.channel(carrier_code, side_name)
            if ch.kind == "fixed_reservoir":
                rows.append(assembler.reaction_flux(sys_new, block, port_id, storage_rate=storage_rate))
            else:
                prob = DC2DProblem(d, current.voltage_V, current.generation_scale)
                exch, _ = prob._contact_exchange(np.asarray(u_new, dtype=float), ch, nodes)
                rows.append(exch)
        particle_ports.append(np.stack(rows))
    j_cond_ports = np.sum((-Q * particle_ports[0] + Q * particle_ports[1]) * g.x_widths_m[None, :], axis=1) / width

    # Port electric displacement D_y along +y closing Gauss boundary cells
    d_port_new = np.stack([assembler.reaction_flux(sys_new, POTENTIAL, port_id) for _, port_id in SIDES])
    d_port_old = np.stack([assembler.reaction_flux(sys_old, POTENTIAL, port_id) for _, port_id in SIDES])
    j_disp_ports = np.sum(((d_port_new - d_port_old) / dt_s) * g.x_widths_m[None, :], axis=1) / width
    j_tot_ports = j_cond_ports + j_disp_ports

    # Interior horizontal cross-sections
    state_new = assembler.state(u_new, current.inputs)
    state_old = assembler.state(u_old, previous.inputs)
    assert state_new is not None and state_old is not None
    c_flux = current.binding.dc.transport.fluxes(state_new)
    j_cond_y = (-Q * c_flux.electron.flux[g.x_face_count:] + Q * c_flux.hole.flux[g.x_face_count:]).reshape(
        g.ny - 1, g.nx
    )
    phi_new, phi_old = state_new.field("potential_V"), state_old.field("potential_V")
    left, right = g.contract_geometry.pairs[g.x_face_count:].T
    cap_y = d.face_capacitance_F_m2[g.x_face_count:]
    d_y_new = (-cap_y * (phi_new[right] - phi_new[left])).reshape(g.ny - 1, g.nx)
    d_y_old = (-cap_y * (phi_old[right] - phi_old[left])).reshape(g.ny - 1, g.nx)
    j_tot_sections = np.sum((j_cond_y + (d_y_new - d_y_old) / dt_s) * g.x_widths_m[None, :], axis=1) / width
    all_levels = np.r_[j_tot_ports[:1], j_tot_sections, j_tot_ports[1:]]

    return BackwardEuler2DStepAccounting(
        dt_s=dt_s,
        terminal_conduction_A_m2=frozen_array(j_cond_ports),
        terminal_displacement_A_m2=frozen_array(j_disp_ports),
        terminal_total_A_m2=frozen_array(j_tot_ports),
        cross_section_total_A_m2=frozen_array(j_tot_sections),
        max_total_current_spread_A_m2=float(np.ptp(all_levels)),
    )


@dataclass(frozen=True)
class BackwardEuler2DStepResult:
    state: State2D | None
    accepted: bool
    termination: str
    dt_s: float
    newton: NewtonResult | None
    accounting: BackwardEuler2DStepAccounting | None
    wall_s: float
    error: str | None = None


@dataclass(frozen=True)
class BackwardEuler2DRunResult:
    times_s: Vector
    states: tuple[State2D, ...]
    accountings: tuple[BackwardEuler2DStepAccounting, ...]
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


def _poisson_weights(device: CompiledDevice2D) -> Vector:
    g = device.geometry.contract_geometry
    reference_V = max(abs(device.column.built_in_potential_V), device.thermal_voltage_V)
    density = device.extrude(np.maximum(np.maximum(
        device.column.parameters.acceptor_density_m3,
        device.column.parameters.donor_density_m3,
    ), 1.0))
    capacitance = np.zeros(device.node_count)
    face = device.face_capacitance_F_m2 * g.face_measures
    np.add.at(capacitance, g.pairs[:, 0], face)
    np.add.at(capacitance, g.pairs[:, 1], face)
    scale = Q * density * g.volumes + reference_V * capacitance
    result = 1.0 / scale
    result[device.geometry.contact_nodes] = 1.0 / reference_V
    return result


def _rebase_electronic_u(device: CompiledDevice2D, u_old: Vector, old_V: float, new_V: float) -> Vector:
    u_guess = np.array(u_old, copy=True)
    if old_V == new_V:
        return u_guess
    n_nodes = device.node_count
    old_e = device.column.carrier_reference_energies(old_V)
    new_e = device.column.carrier_reference_energies(new_V)
    for offset, old_val, new_val in zip((n_nodes, 2 * n_nodes), old_e, new_e):
        u_guess[offset:offset + n_nodes] += (old_val - new_val) / device.thermal_voltage_V
    resistance = 1.0 / device.column.poisson.face_capacitance_F_m2
    profile = np.r_[0.0, np.cumsum(resistance)]
    profile /= profile[-1]
    change = -device.column.voltage_polarity * (new_V - old_V)
    before = u_guess[:n_nodes].copy()
    u_guess[:n_nodes] += device.extrude(change * profile)
    bottom, top = device.column.contact_potentials(new_V)
    u_guess[:device.geometry.nx] = bottom
    u_guess[n_nodes - device.geometry.nx:n_nodes] = top
    shift = (u_guess[:n_nodes] - before) / device.thermal_voltage_V
    u_guess[n_nodes:2 * n_nodes] -= shift
    u_guess[2 * n_nodes:3 * n_nodes] -= shift
    return u_guess


def _generation_inventory_seed_2d(
    device: CompiledDevice2D,
    step_prob: BackwardEuler2DStep,
    u_base: Vector,
    previous_state: State2D,
    *,
    dt_s: float,
    voltage_V: float,
    generation_scale: float,
) -> Vector:
    n_nodes = device.node_count
    try:
        _, n_cur, p_cur = device.fields(u_base, voltage_V=voltage_V)
        res_c = step_prob.residual(u_base)[n_nodes:3 * n_nodes]
        if np.all(np.isfinite(res_c)) and np.all(n_cur > 0.0) and np.all(p_cur > 0.0):
            inv_d = np.r_[1.0 / n_cur, -1.0 / p_cur]
            mat = step_prob.jacobian(u_base)[n_nodes:3 * n_nodes, n_nodes:3 * n_nodes] @ diags(inv_d)
            inc = np.asarray(spsolve(mat.tocsc(), -res_c), dtype=float)
            trial_n, trial_p = n_cur + inc[:n_nodes], p_cur + inc[n_nodes:]
            if np.all(np.isfinite(trial_n)) and np.all(np.isfinite(trial_p)) and np.all(trial_n > 0.0) and np.all(trial_p > 0.0):
                u_lin = np.array(u_base, copy=True)
                for offset, dens, trial, delta, sign in (
                    (n_nodes, n_cur, trial_n, inc[:n_nodes], 1.0),
                    (2 * n_nodes, p_cur, trial_p, inc[n_nodes:], -1.0),
                ):
                    with np.errstate(over="ignore", divide="ignore"):
                        chg = np.log1p(delta / dens)
                    bad = ~np.isfinite(chg)
                    chg[bad] = np.log(trial[bad]) - np.log(dens[bad])
                    u_lin[offset:offset + n_nodes] += sign * chg
                if np.all(np.isfinite(u_lin)):
                    return u_lin
    except Exception:
        pass
    u_seed = np.array(u_base, copy=True)
    if generation_scale <= 0.0:
        return u_seed
    increment = dt_s * generation_scale * device.generation_source_m3_s
    fixed = device.fixed_carrier_mask
    for species, (density_2d, offset, sign) in enumerate((
        (previous_state.electron_density_m3, n_nodes, 1.0),
        (previous_state.hole_density_m3, 2 * n_nodes, -1.0),
    )):
        density = density_2d.ravel()
        addition = np.where(~fixed[species], increment, 0.0)
        u_seed[offset:offset + n_nodes] += sign * np.log1p(addition / np.maximum(density, 1.0e-300))
    return u_seed


def step_backward_euler_2d(
    device: CompiledDevice2D,
    previous_state: State2D,
    *,
    dt_s: float,
    voltage_V: float,
    generation_scale: float,
    previous_generation_scale: float | None = None,
    options: BackwardEuler2DOptions | None = None,
) -> BackwardEuler2DStepResult:
    """Advance one conservative 2D Backward Euler step."""
    started = time.monotonic()
    if device.geometry.name not in RECTANGULAR_2D_NAMES:
        raise ValueError("2D BE requires Neumann or periodic half-endpoint geometry")
    if previous_state.device_sha256 != device.identity:
        raise ValueError("previous_state must match CompiledDevice2D identity")
    controls = BackwardEuler2DOptions() if options is None else options
    prev_g = generation_scale if previous_generation_scale is None else previous_generation_scale
    prev_prob = AssembledStorage2D(
        device,
        previous_state.voltage_V,
        prev_g,
        radiative_reabsorption=controls.radiative_reabsorption,
        interface_srh_couplings=controls.interface_srh_couplings,
    )
    curr_prob = AssembledStorage2D(
        device,
        voltage_V,
        generation_scale,
        radiative_reabsorption=controls.radiative_reabsorption,
        interface_srh_couplings=controls.interface_srh_couplings,
    )
    step_prob = BackwardEuler2DStep(curr_prob, prev_prob, previous_state.u, dt_s)

    c = controls.electronic
    n_nodes = device.node_count
    weights = np.full(3 * n_nodes, 2.0 * n_nodes * Q / (device.geometry.lateral_length_m * c.continuity_tolerance_A_m2))
    weights[:n_nodes] = _poisson_weights(device) / c.poisson_tolerance
    weights[n_nodes:].reshape(2, n_nodes)[device.fixed_carrier_mask] = 1.0 / c.contact_log_tolerance
    steps = np.r_[np.full(n_nodes, c.potential_step_V), np.full(2 * n_nodes, c.quasi_fermi_step)]

    def gate_fn(cand_u: Vector) -> GateOutcome:
        if time.monotonic() - started > controls.wall_s:
            return GateOutcome("wall_budget", False, time.monotonic() - started, controls.wall_s)
        res = step_prob.residual(cand_u)
        if not np.all(np.isfinite(res)):
            return GateOutcome("finite_be_residual", False, math.inf, 1.0)
        scaled = float(np.max(np.abs(res * weights)))
        return GateOutcome("be_2d_scaled_max", scaled <= 1.0, scaled, 1.0)

    settings = NewtonSettings(ResidualScaling(weights), 1.0, StepControl(steps), controls.budget, (gate_fn,), False)
    u_rebased = _rebase_electronic_u(device, previous_state.u, previous_state.voltage_V, voltage_V)
    u_guess = (
        _generation_inventory_seed_2d(
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
            u_alt = _generation_inventory_seed_2d(
                device, step_prob, u_rebased, previous_state,
                dt_s=dt_s, voltage_V=voltage_V, generation_scale=generation_scale,
            )
            newton_alt = solve_newton(prob, u_alt, settings)
            if newton_alt.accepted:
                newton = newton_alt
    except Exception as exc:
        return BackwardEuler2DStepResult(None, False, "exception", dt_s, None, None, time.monotonic() - started, str(exc))
    if not newton.accepted:
        return BackwardEuler2DStepResult(
            None, False, newton.termination, dt_s, newton, None, time.monotonic() - started,
            "2D Backward Euler step did not converge",
        )
    state = device.state(newton.state, voltage_V=voltage_V)
    acct = step_accounting_2d(curr_prob, newton.state, prev_prob, previous_state.u, dt_s)
    return BackwardEuler2DStepResult(state, True, "converged", dt_s, newton, acct, time.monotonic() - started)


def run_backward_euler_2d(
    device: CompiledDevice2D,
    initial_state: State2D,
    time_grid_s: ArrayLike,
    *,
    voltage_fn: Callable[[float], float],
    generation_fn: Callable[[float], float],
    options: BackwardEuler2DOptions | None = None,
) -> BackwardEuler2DRunResult:
    """Advance across ``time_grid_s`` with bounded step-bisection recovery."""
    started = time.monotonic()
    controls = BackwardEuler2DOptions() if options is None else options
    grid = np.asarray(time_grid_s, dtype=float)
    if grid.ndim != 1 or grid.size < 2 or not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0.0):
        raise ValueError("time_grid_s must be 1D, finite, and strictly increasing")

    times = [float(grid[0])]
    states = [initial_state]
    accountings: list[BackwardEuler2DStepAccounting] = []
    attempted = accepted = rejected = bisections = total_newton = 0
    curr_state = initial_state
    prev_g = float(generation_fn(float(grid[0])))

    for seg_idx in range(grid.size - 1):
        t_cur = float(grid[seg_idx])
        t_target = float(grid[seg_idx + 1])
        dt = t_target - t_cur
        depth = 0
        while t_cur < t_target - 1.0e-15 * max(1.0, abs(t_target)):
            if time.monotonic() - started > controls.wall_s:
                return BackwardEuler2DRunResult(
                    frozen_array(times), tuple(states), tuple(accountings), False, "wall_budget_exceeded",
                    attempted, accepted, rejected, bisections, time.monotonic() - started,
                    "2D BE wall budget exceeded", total_newton_iterations=total_newton,
                )
            dt_try = min(dt, t_target - t_cur)
            t_next = t_cur + dt_try
            v_next = float(voltage_fn(t_next))
            g_next = float(generation_fn(t_next))
            attempted += 1
            res = step_backward_euler_2d(
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
                    return BackwardEuler2DRunResult(
                        frozen_array(times), tuple(states), tuple(accountings), False, res.termination,
                        attempted, accepted, rejected, bisections, time.monotonic() - started,
                        res.error or "2D BE step bisection budget exhausted",
                        total_newton_iterations=total_newton,
                    )
                depth += 1
                bisections += 1
                dt *= 0.5

    return BackwardEuler2DRunResult(
        frozen_array(times), tuple(states), tuple(accountings), True, "completed",
        attempted, accepted, rejected, bisections, time.monotonic() - started,
        total_newton_iterations=total_newton,
    )
