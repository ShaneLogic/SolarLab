"""Steady-state and physical-storage BDF adapters for 1D iodine aging systems.

Reuses SolarLab's shared :func:`~solarlab.solvers.nonlinear.newton.solve_newton`
and :func:`~solarlab.solvers.time.bdf.solve_bdf` drivers without duplicating
any linear, nonlinear, or multistep time-integration solver logic.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csc_matrix, diags

from solarlab.contracts import Vector, frozen_array
from solarlab.contracts.units import ELEMENTARY_CHARGE_C as Q
from solarlab.equations.assembly.iodine_1d import (
    AssembledConstrainedIodine1DProblem,
    AssembledIodine1DProblem,
    IodineSystem,
    iodine_system,
)
from solarlab.equations.dc_1d import DC1DProblem
from solarlab.equations.iodine_1d import CompiledIodineDevice1D, IodineState1D
from solarlab.solvers.nonlinear.newton import (
    NewtonResult,
    NewtonSettings,
    Termination,
    factorize,
    solve_newton,
)
from solarlab.solvers.nonlinear.problem import (
    CallableProblem,
    GateOutcome,
    PhysicalGate,
    ResidualScaling,
    SolveBudget,
    SteadyProblem,
    StepControl,
)
from solarlab.solvers.nonlinear.production import (
    POISSON_CONTACT_BALANCE_V1,
    production_convergence_norm,
    production_coordinates,
    production_residual_scaling,
)
from solarlab.solvers.nonlinear.steady_1d import (
    SteadyOptions,
    SteadyStateError,
    solve_equilibrium,
)
from solarlab.solvers.time.bdf import (
    Assessment,
    BDFEvent,
    BDFOptions,
    BDFResult,
    BDFStep,
    solve_bdf,
)

__all__ = [
    "IodineSteadyOptions",
    "IodineSteadyResult",
    "IodineStepAccounting",
    "iodine_bdf_newton_settings",
    "iodine_bdf_seed",
    "iodine_convergence_norm",
    "iodine_steady_newton_settings",
    "iodine_step_accounting",
    "solve_frozen_chemistry_electronic_1d",
    "solve_iodine_bdf_transient",
    "solve_iodine_steady_state",
]


@dataclass(frozen=True, slots=True)
class IodineSteadyOptions:
    """Controls for closed-domain 1D iodine steady-state solves."""

    poisson_tolerance: float = 5.0e-9
    continuity_tolerance_A_m2: float = 2.0e-6
    chem_flux_tolerance_A_m2: float = 5.0e-9
    inventory_relative_tolerance: float = 1.0e-12
    contact_log_tolerance: float = 1.0e-11
    potential_step_V: float = 0.15
    quasi_fermi_step: float = 2.5
    chem_coordinate_step: float = 40.0
    max_iterations: int = 120
    max_backtracks: int = 16
    wall_s: float = 30.0
    continuation_steps: int = 6


@dataclass(frozen=True, slots=True)
class IodineSteadyResult:
    """Converged or failed closed-domain 1D iodine steady-state result."""

    converged: bool
    state: IodineState1D | None
    current_density_A_m2: float
    termination: str
    scaled_residual: float
    iterations: int
    newton: NewtonResult


@dataclass(frozen=True, slots=True)
class IodineStepAccounting:
    """Per-step conservation and Maxwell current diagnostics."""

    start_s: float
    end_s: float
    dt_s: float
    in_device_iodine_m2: float
    step_escaped_iodine_atoms_m2: float
    step_immobilized_iodine_atoms_m2: float
    step_conservation_residual_m2: float
    terminal_current_A_m2: float
    max_total_current_spread_A_m2: float


def _poisson_weights_with_ions(device: CompiledIodineDevice1D) -> np.ndarray:
    comp = device.electronic_component
    count = comp.node_count
    v_ref = max(abs(comp.built_in_potential_V), comp.thermal_voltage_V)
    ion_bg = device.project_chem_density(device.compensating_donor_m3)
    ion_peak = float(np.max(device.initial_iodide_m3)) * 0.1
    d_ref = np.maximum(
        np.maximum(
            comp.parameters.acceptor_density_m3[1:-1],
            comp.parameters.donor_density_m3[1:-1],
        ),
        np.maximum(ion_bg[1:-1], max(ion_peak, 1.0)),
    )
    cap = comp.poisson.face_capacitance_F_m2
    scale = (
        Q * d_ref * comp.poisson.interior_widths_m
        + (cap[:-1] + cap[1:]) * v_ref
    )
    weights: NDArray[np.float64] = np.full(count, 1.0 / v_ref, dtype=np.float64)
    weights[1:-1] = 1.0 / scale
    return weights


def iodine_convergence_norm(
    device: CompiledIodineDevice1D,
    *,
    constrained_steady: bool = False,
    inventory_node: int = 0,
) -> Callable[[NDArray[np.float64]], float]:
    """L1-integrated balance norm matching POISSON_CONTACT_BALANCE_V1."""
    comp = device.electronic_component
    n_count = device.node_count
    m_count = device.chem_node_count
    pointwise: NDArray[np.bool_] = np.zeros(device.state_size, dtype=bool)
    pointwise[:n_count] = True
    for carrier, offset in (("n", n_count), ("p", 2 * n_count)):
        for side, node in (("left", 0), ("right", n_count - 1)):
            if comp.channel(carrier, side).kind == "fixed_reservoir":
                pointwise[offset + node] = True
    if constrained_steady:
        slices = device.chem_slices
        if device.is_unreactive_limit:
            for sp in ("iodide", "iodine_atom", "iodine_molecule"):
                pointwise[slices[sp].start + inventory_node] = True
        else:
            pointwise[slices["iodide"].start + inventory_node] = True

    def norm(scaled_residual: NDArray[np.float64]) -> float:
        vals = np.asarray(scaled_residual, dtype=np.float64)
        if not np.all(np.isfinite(vals)):
            return math.inf
        abs_v = np.abs(vals)
        max_fixed = float(np.max(abs_v[pointwise]))
        carrier_rows = abs_v[n_count : 3 * n_count].copy()
        carrier_rows[pointwise[n_count : 3 * n_count]] = 0.0
        carrier_bal = float(np.sum(carrier_rows)) / (2.0 * n_count)
        chem_rows = abs_v[3 * n_count :].copy()
        chem_rows[pointwise[3 * n_count :]] = 0.0
        chem_bal = float(np.sum(chem_rows)) / (3.0 * m_count)
        return max(max_fixed, carrier_bal, chem_bal)

    return norm


def _positive_species_gate(device: CompiledIodineDevice1D, voltage_V: float) -> PhysicalGate:
    def check(u: Vector) -> GateOutcome:
        try:
            phi, n, p, cm, c0, c2, _, _, _ = device.fields_and_tangents(
                u, voltage_V=voltage_V
            )
        except (ValueError, FloatingPointError):
            return GateOutcome("positive_finite_densities", False, math.inf, 0.5)
        ok = bool(
            np.all(np.isfinite(phi))
            and np.all(n > 0.0)
            and np.all(p > 0.0)
            and np.all(cm > 0.0)
            and np.all(c0 > 0.0)
            and np.all(c2 > 0.0)
        )
        return GateOutcome("positive_finite_densities", ok, 0.0 if ok else 1.0, 0.5)

    return check


def iodine_steady_newton_settings(
    device: CompiledIodineDevice1D,
    *,
    voltage_V: float,
    inventory_node: int = 0,
    options: IodineSteadyOptions = IodineSteadyOptions(),
) -> NewtonSettings:
    comp = device.electronic_component
    n_count = device.node_count
    m_count = device.chem_node_count
    weights: NDArray[np.float64] = np.empty(device.state_size, dtype=np.float64)

    weights[:n_count] = _poisson_weights_with_ions(device) / options.poisson_tolerance
    weights[n_count : 3 * n_count] = (
        2.0 * n_count * Q / options.continuity_tolerance_A_m2
    )
    for carrier, offset in (("n", n_count), ("p", 2 * n_count)):
        for side, node in (("left", 0), ("right", n_count - 1)):
            if comp.channel(carrier, side).kind == "fixed_reservoir":
                weights[offset + node] = 1.0 / options.contact_log_tolerance

    chem_w = m_count * Q / options.chem_flux_tolerance_A_m2
    weights[3 * n_count :] = chem_w

    inv_ref = max(device.reference_iodine_inventory_m2, 1.0)
    inv_weight = 1.0 / (options.inventory_relative_tolerance * inv_ref)
    slices = device.chem_slices
    if device.is_unreactive_limit:
        sp_refs = device.reference_species_inventories_m2
        for sp in ("iodide", "iodine_atom", "iodine_molecule"):
            ref_sp = max(sp_refs[sp], 1.0)
            weights[slices[sp].start + inventory_node] = 1.0 / (
                options.inventory_relative_tolerance * ref_sp
            )
    else:
        weights[slices["iodide"].start + inventory_node] = inv_weight

    iodide_step = (
        10.0 if device.steric_site_limit_m3 is not None else options.chem_coordinate_step
    )
    max_step = np.concatenate(
        (
            np.full(n_count, options.potential_step_V),
            np.full(2 * n_count, options.quasi_fermi_step),
            np.full(m_count, iodide_step),
            np.full(2 * m_count, options.chem_coordinate_step),
        )
    )
    budget = SolveBudget(
        max_iterations=options.max_iterations,
        max_residual_evaluations=options.max_iterations * (options.max_backtracks + 2),
        max_jacobian_evaluations=options.max_iterations + 2,
        max_backtracks=options.max_backtracks,
    )
    lower = np.concatenate(
        (
            np.full(3 * n_count, -math.inf),
            np.full(3 * m_count, -220.0),
        )
    )
    return NewtonSettings(
        scaling=ResidualScaling(weights),
        tolerance=1.0,
        step_control=StepControl(max_step, lower=lower),
        budget=budget,
        gates=(_positive_species_gate(device, voltage_V),),
        reuse_jacobian=False,
    )


def iodine_bdf_newton_settings(
    step: BDFStep,
    *,
    poisson_tolerance: float = 5.0e-9,
    continuity_tolerance_A_m2: float = 2.0e-6,
    chem_flux_tolerance_A_m2: float = 2.0e-8,
    inventory_relative_tolerance: float = 1.0e-10,
    potential_step_V: float = 0.15,
    quasi_fermi_step: float = 2.5,
    chem_coordinate_step: float = 40.0,
    max_iterations: int = 80,
) -> NewtonSettings:
    if not isinstance(step.endpoint, AssembledIodine1DProblem):
        raise TypeError("iodine_bdf_newton_settings requires AssembledIodine1DProblem")
    device = step.endpoint.device
    comp = device.electronic_component
    n_count = device.node_count
    m_count = device.chem_node_count
    dt_s = max(step.dt_s, 1.0e-30)

    weights: NDArray[np.float64] = np.empty(device.state_size, dtype=np.float64)
    weights[:n_count] = _poisson_weights_with_ions(device) / poisson_tolerance
    w_cont = 2.0 * n_count * Q / continuity_tolerance_A_m2
    old_c_s = float(
        np.sum(np.abs(step.history[0].storage_value[n_count : 3 * n_count]))
    )
    carrier_s_ref = max(old_c_s, 1.0e14)
    w_c_step = (2.0 * n_count * dt_s) / (1.0e-9 * carrier_s_ref)
    weights[n_count : 3 * n_count] = min(w_cont, max(w_c_step, 1.0e-20))
    for carrier, offset in (("n", n_count), ("p", 2 * n_count)):
        for side, node in (("left", 0), ("right", n_count - 1)):
            if comp.channel(carrier, side).kind == "fixed_reservoir":
                weights[offset + node] = 1.0e11

    inv_ref = max(device.reference_iodine_inventory_m2, 1.0)
    # Per-species diagonal flux scaling aligned with merit = 0.5 * ||scaled||_2^2:
    # monomer rows stay >=20x above PVK fast-redox ULP (~1.6e-12 A/m^2), while trace I_2
    # enforces 1e-13 A/m^2 (~75x above I_2 ULP) so outgassing equilibrates at every step.
    if step.endpoint.generation_scale > 0.0:
        w_flux = m_count * Q / chem_flux_tolerance_A_m2
        w_step = (m_count * dt_s) / (inventory_relative_tolerance * inv_ref)
        weights[3 * n_count :] = min(w_flux, max(w_step, 1.0e-20))
        assert device.volumetric_mol_escape_s_1 is not None
        vol_mask = device.volumetric_mol_escape_s_1 > 0.0
        if np.any(vol_mask):
            w_vol = min(
                3.0 * m_count * Q / min(chem_flux_tolerance_A_m2, 5.0e-13),
                max(w_step * 4.0e3, 1.0e-20),
            )
            weights[3 * n_count + 2 * m_count :][vol_mask] = np.maximum(
                weights[3 * n_count + 2 * m_count :][vol_mask], w_vol
            )
    else:
        w_flux = 3.0 * m_count * Q / 1.0e-12
        w_step = (6.0 * m_count * dt_s) / (inventory_relative_tolerance * inv_ref)
        weights[3 * n_count :] = min(w_flux, max(w_step, 1.0e-20))

    iodide_step = (
        10.0 if device.steric_site_limit_m3 is not None else chem_coordinate_step
    )
    max_step = np.concatenate(
        (
            np.full(n_count, potential_step_V),
            np.full(2 * n_count, quasi_fermi_step),
            np.full(m_count, iodide_step),
            np.full(2 * m_count, chem_coordinate_step),
        )
    )
    lower = np.concatenate(
        (
            np.full(3 * n_count, -math.inf),
            np.full(3 * m_count, -220.0),
        )
    )
    return NewtonSettings(
        scaling=ResidualScaling(weights),
        tolerance=1.0,
        step_control=StepControl(max_step, lower=lower),
        budget=SolveBudget(
            max_iterations=max_iterations,
            max_residual_evaluations=max_iterations * 18,
            max_jacobian_evaluations=max_iterations + 2,
            max_backtracks=16,
        ),
        gates=(_positive_species_gate(device, step.endpoint.voltage_V),),
        reuse_jacobian=False,
    )


def _carrier_density_basin_lift(
    problem: SteadyProblem,
    device: CompiledIodineDevice1D,
    u: Vector,
    voltage_V: float,
) -> Vector:
    """Linear-density carrier seed eliminating wide-gap log-coordinate stagnation."""
    n_count = device.node_count
    comp = device.electronic_component
    u_cur = np.array(u, dtype=np.float64, copy=True)
    fixed_local: list[int] = []
    for carrier, offset in (("n", 0), ("p", n_count)):
        for side, node in (("left", 0), ("right", n_count - 1)):
            if comp.channel(carrier, side).kind == "fixed_reservoir":
                fixed_local.append(offset + node)

    for _ in range(4):
        _, n, p = comp.fields(u_cur[: 3 * n_count], voltage_V=voltage_V)
        f_all = problem.residual(u_cur)
        if float(np.max(np.abs(f_all[n_count : 3 * n_count]))) * Q < 1.0e-6:
            break
        j_all = problem.jacobian(u_cur)
        f_np = np.array(f_all[n_count : 3 * n_count], dtype=np.float64, copy=True)
        inv_scale = np.concatenate((1.0 / n, -1.0 / p))
        j_np = (
            j_all[n_count : 3 * n_count, n_count : 3 * n_count] @ diags(inv_scale)
        ).tolil()
        for r_idx in fixed_local:
            j_np.rows[r_idx] = [r_idx]
            j_np.data[r_idx] = [1.0]
            f_np[r_idx] = 0.0
        try:
            inc = factorize(j_np.tocsc()).solve(-f_np)
        except Exception:
            break
        if not np.all(np.isfinite(inc)):
            break
        trial_n = np.maximum(n + inc[:n_count], 0.1 * n)
        trial_p = np.maximum(p + inc[n_count:], 0.1 * p)
        u_cur[n_count : 2 * n_count] += np.log(trial_n) - np.log(n)
        u_cur[2 * n_count : 3 * n_count] -= np.log(trial_p) - np.log(p)
    return frozen_array(u_cur)


def _chem_density_basin_lift(
    problem: SteadyProblem,
    device: CompiledIodineDevice1D,
    u: Vector,
    voltage_V: float,
    *,
    freeze_iodide: bool = False,
) -> Vector:
    """Linear-density chemical seed eliminating trace-species Softplus stagnation."""
    n_elec = 3 * device.node_count
    m = device.chem_node_count
    u_cur = np.array(u, dtype=np.float64, copy=True)
    for _ in range(8):
        try:
            _, _, _, cm, c0, c2, mm, m0, m2 = device.fields_and_tangents(
                u_cur, voltage_V=voltage_V
            )
            f_chem = problem.residual(u_cur)[n_elec:]
            if float(np.max(np.abs(f_chem))) * Q < 1.0e-8:
                break
            if freeze_iodide:
                inv_tan_n = 1.0 / np.concatenate((m0, m2))
                j_neut = problem.jacobian(u_cur)[n_elec + m :, n_elec + m :] @ diags(
                    inv_tan_n
                )
                dc_neut = factorize(j_neut.tocsc()).solve(-f_chem[m:])
                dc = np.concatenate((np.zeros(m, dtype=np.float64), dc_neut))
            else:
                inv_tan = 1.0 / np.concatenate((mm, m0, m2))
                j_chem = problem.jacobian(u_cur)[n_elec:, n_elec:] @ diags(inv_tan)
                dc = factorize(j_chem.tocsc()).solve(-f_chem)
        except Exception:
            break
        if not np.all(np.isfinite(dc)):
            break
        cm_n = cm if freeze_iodide else np.maximum(cm + dc[:m], 0.1 * cm)
        c0_n = np.maximum(c0 + dc[m : 2 * m], 0.1 * c0)
        c2_n = np.maximum(c2 + dc[2 * m :], 0.1 * c2)
        if device.steric_site_limit_m3 is not None:
            cm_n = np.minimum(cm_n, 0.999999 * device.steric_site_limit_m3)
        u_cur[n_elec:] = np.maximum(
            np.concatenate(
                (
                    device.iodide_map.coordinate_from_density(cm_n),
                    device.atom_map.coordinate_from_density(c0_n),
                    device.mol_map.coordinate_from_density(c2_n),
                )
            ),
            -115.0,
        )
    return frozen_array(u_cur)


def _inferred_voltage_V(device: CompiledIodineDevice1D, u: Vector) -> float:
    comp = device.electronic_component
    n_c = comp.node_count
    dphi_contacts = float(u[n_c - 1] - u[0])
    return float((comp.built_in_potential_V - dphi_contacts) / comp.voltage_polarity)


def iodine_bdf_seed(step: BDFStep) -> Vector:
    """Harmonic potential & BDF-step homotopy lift across voltage/light jumps."""
    if not isinstance(step.endpoint, AssembledIodine1DProblem):
        raise TypeError("iodine_bdf_seed requires AssembledIodine1DProblem")
    prev = step.history[0]
    u_cur = np.array(prev.u, dtype=np.float64, copy=True)
    dev = step.endpoint.device
    u_cur[3 * dev.node_count :] = np.maximum(u_cur[3 * dev.node_count :], -115.0)
    v_prev = _inferred_voltage_V(dev, u_cur)
    v_next = step.endpoint.voltage_V
    if abs(v_next - v_prev) > 1.0e-13:
        u_cur = np.array(
            _harmonic_voltage_lift(dev, u_cur, v_prev, v_next),
            dtype=np.float64,
            copy=True,
        )

    bdf_norm = iodine_convergence_norm(dev, constrained_steady=False)
    base_settings = iodine_bdf_newton_settings(step)
    f_try = step.residual(u_cur)
    scaled_try = base_settings.scaling.scaled(f_try)
    if float(np.max(np.abs(scaled_try[3 * dev.node_count :]))) > 10.0:
        p_comp = dev.electronic_component.parameters
        undoped_bl = (
            max(float(p_comp.donor_density_m3[0]), float(p_comp.acceptor_density_m3[0]))
            < 1.0e22
            and max(
                float(p_comp.donor_density_m3[-1]),
                float(p_comp.acceptor_density_m3[-1]),
            )
            < 1.0e22
        )
        u_cur = np.array(
            _chem_density_basin_lift(
                CallableProblem(step.residual, step.jacobian),
                dev,
                u_cur,
                v_next,
                freeze_iodide=undoped_bl and abs(v_next - v_prev) <= 1.0e-12,
            ),
            dtype=np.float64,
            copy=True,
        )
        f_try = step.residual(u_cur)
        scaled_try = base_settings.scaling.scaled(f_try)

    # Full-Jacobian elemental conservation polish when long BDF steps accumulate
    # open outgassing / contact oxidation below the per-row monomer ULP flux cap.
    if dev.has_open_iodine_loss and step.dt_s >= 1.0 and bdf_norm(scaled_try) <= 1.2:
        m_c = dev.chem_node_count
        n_e = 3 * dev.node_count
        inv_ref = max(dev.reference_iodine_inventory_m2, 1.0)
        tot_imb = abs(
            math.fsum(
                (
                    f_try[n_e : n_e + m_c]
                    + f_try[n_e + m_c : n_e + 2 * m_c]
                    + 2.0 * f_try[n_e + 2 * m_c :]
                ).tolist()
            )
        )
        if tot_imb * step.dt_s > 5.0e-11 * inv_ref:
            for _ in range(2):
                try:
                    du_p = factorize(step.jacobian(u_cur).tocsc()).solve(-f_try)
                except Exception:
                    break
                if not np.all(np.isfinite(du_p)):
                    break
                alpha_p = base_settings.step_control.step_fraction(u_cur, du_p)
                u_next = u_cur + alpha_p * du_p
                if not base_settings.step_control.feasible(u_next):
                    break
                try:
                    f_next = step.residual(u_next)
                except Exception:
                    break
                if not np.all(np.isfinite(f_next)):
                    break
                u_cur = u_next
                f_try = f_next
                tot_imb = abs(
                    math.fsum(
                        (
                            f_try[n_e : n_e + m_c]
                            + f_try[n_e + m_c : n_e + 2 * m_c]
                            + 2.0 * f_try[n_e + 2 * m_c :]
                        ).tolist()
                    )
                )
                if tot_imb * step.dt_s <= 1.0e-11 * inv_ref:
                    break
            scaled_try = base_settings.scaling.scaled(f_try)
    scaled_norm = bdf_norm(scaled_try)
    if len(step.history) > 1 or abs(v_next - v_prev) <= 0.05 or scaled_norm <= 1.0e4:
        return frozen_array(u_cur)

    # Discontinuous drive step: homotopy from (0 V, 0 Sun) -> (0 V, g_next) -> (v_next, g_next)
    # on the exact BDFStep while keeping step.history completely untouched!
    g_next = step.endpoint.generation_scale
    cur_v = v_next
    if abs(cur_v) > 1.0e-13:
        u_cur = np.array(
            _harmonic_voltage_lift(dev, u_cur, cur_v, 0.0),
            dtype=np.float64,
            copy=True,
        )
        cur_v = 0.0
    cur_g = 0.0
    stages: list[tuple[float, float]] = [(0.0, 0.0)]
    if g_next > 0.0:
        for frac in (1.0e-5, 1.0e-3, 0.03, 0.25, 1.0):
            stages.append((0.0, g_next * frac))
    if abs(v_next) > 1.0e-13:
        n_v = max(1, int(math.ceil(abs(v_next) / 0.08)))
        for k in range(1, n_v + 1):
            stages.append((v_next * (k / n_v), g_next))

    queue: list[tuple[float, float, int]] = [
        (v_s, g_s, 0) for v_s, g_s in reversed(stages)
    ]
    while queue:
        v_s, g_s, depth = queue.pop()
        sub_end = AssembledIodine1DProblem(
            device=dev,
            voltage_V=v_s,
            generation_scale=g_s,
            binding=step.endpoint.binding,
        )
        sub_step = BDFStep(sub_end, step.history, step.end_s, step.order)
        u_try = u_cur
        if abs(v_s - cur_v) > 1.0e-13:
            u_try = np.array(
                _harmonic_voltage_lift(dev, u_try, cur_v, v_s),
                dtype=np.float64,
                copy=True,
            )
        if g_s > cur_g:
            u_try = np.array(
                _carrier_density_basin_lift(
                    CallableProblem(sub_step.residual, sub_step.jacobian),
                    dev,
                    u_try,
                    v_s,
                ),
                dtype=np.float64,
                copy=True,
            )
        sub_res = solve_newton(
            CallableProblem(sub_step.residual, sub_step.jacobian),
            u_try,
            iodine_bdf_newton_settings(sub_step),
            convergence_norm=bdf_norm,
        )
        if sub_res.accepted:
            u_cur = np.array(sub_res.state, dtype=np.float64, copy=True)
            cur_v, cur_g = v_s, g_s
        elif depth < 6 and (
            abs(v_s - cur_v) > 2.0e-3 or abs(g_s - cur_g) > 1.0e-8
        ):
            v_mid = 0.5 * (cur_v + v_s)
            g_mid = (
                math.sqrt(cur_g * g_s)
                if (cur_g > 0.0 and g_s / cur_g > 4.0)
                else 0.5 * (cur_g + g_s)
            )
            queue.append((v_s, g_s, depth + 1))
            queue.append((v_mid, g_mid, depth + 1))
        else:
            break
    if abs(v_next - cur_v) > 1.0e-13:
        u_cur = np.array(
            _harmonic_voltage_lift(dev, u_cur, cur_v, v_next),
            dtype=np.float64,
            copy=True,
        )
    return frozen_array(u_cur)


def iodine_step_accounting(step: BDFStep, u: Vector) -> IodineStepAccounting:
    """Compute exact BDF step iodine inventory balance and Maxwell current spread."""
    if not isinstance(step.endpoint, AssembledIodine1DProblem):
        raise TypeError("iodine_step_accounting requires AssembledIodine1DProblem")
    endpoint = step.endpoint
    device = endpoint.device
    old = step.history[0]
    if not isinstance(old.problem, AssembledIodine1DProblem):
        raise TypeError("history problem must be AssembledIodine1DProblem")

    phys_rate = (
        step.coefficients[0] * endpoint.physical_increment(u, old.problem, old.u) / step.dt_s
    )
    if step.order == 2:
        older = step.history[1]
        assert isinstance(older.problem, AssembledIodine1DProblem)
        phys_rate -= (
            step.coefficients[2]
            * old.problem.physical_increment(old.u, older.problem, older.u)
            / step.dt_s
        )

    slices = device.chem_slices
    d_cm_dt = phys_rate[slices["iodide"]]
    d_c0_dt = phys_rate[slices["iodine_atom"]]
    d_c2_dt = phys_rate[slices["iodine_molecule"]]
    d_iodine_rate_m2_s = math.fsum((d_cm_dt + d_c0_dt + 2.0 * d_c2_dt).tolist())

    brates = endpoint.boundary_rates_m2_s(u)
    esc_atoms_rate = 2.0 * (
        brates["mol_escape_left_m2_s"]
        + brates["mol_escape_right_m2_s"]
        + brates["volumetric_mol_escape_m2_s"]
    )
    immob_rate = 0.0
    if device.left_boundary.contact_reaction_mode in (
        "immobilized_product_sheet",
        "metal_faradaic_corrosion",
    ):
        immob_rate += brates["contact_oxidation_left_m2_s"]
    if device.right_boundary.contact_reaction_mode in (
        "immobilized_product_sheet",
        "metal_faradaic_corrosion",
    ):
        immob_rate += brates["contact_oxidation_right_m2_s"]

    # Discrete BDF balance: dM_I/dt + esc_atoms_rate + immob_rate == 0
    step_cons_res_m2 = (d_iodine_rate_m2_s + esc_atoms_rate + immob_rate) * step.dt_s

    # Maxwell total current across every interior face: J_cond + dD_face/dt
    j_cond = endpoint.face_conduction_current_A_m2(u)
    cap = device.electronic_component.poisson.face_capacitance_F_m2
    n_count = device.node_count
    dphi_new = np.diff(u[:n_count])
    dphi_old = np.diff(old.u[:n_count])
    dd_dt = step.coefficients[0] * (-cap * (dphi_new - dphi_old)) / step.dt_s
    if step.order == 2:
        dphi_older = np.diff(step.history[1].u[:n_count])
        dd_dt -= step.coefficients[2] * (-cap * (dphi_old - dphi_older)) / step.dt_s
    j_total_faces = j_cond + dd_dt
    dx = np.diff(device.electronic_component.nodes_m)
    j_term = float(math.fsum((dx * j_total_faces).tolist()) / math.fsum(dx.tolist()))
    spread = float(np.max(j_total_faces) - np.min(j_total_faces))

    state = device.state(u, voltage_V=endpoint.voltage_V)
    return IodineStepAccounting(
        start_s=old.time_s,
        end_s=step.end_s,
        dt_s=step.dt_s,
        in_device_iodine_m2=state.total_iodine_inventory_m2,
        step_escaped_iodine_atoms_m2=esc_atoms_rate * step.dt_s,
        step_immobilized_iodine_atoms_m2=immob_rate * step.dt_s,
        step_conservation_residual_m2=step_cons_res_m2,
        terminal_current_A_m2=j_term,
        max_total_current_spread_A_m2=spread,
    )


def _harmonic_voltage_lift(
    device: CompiledIodineDevice1D, u: Vector, v_from: float, v_to: float
) -> Vector:
    if v_to == v_from:
        return u
    comp = device.electronic_component
    n_c = comp.node_count
    u_mut = np.array(u, dtype=np.float64, copy=True)
    values = comp.rebase_coordinates(u_mut[: 3 * n_c], v_from, v_to)
    resistance = 1.0 / comp.poisson.face_capacitance_F_m2
    profile = np.r_[0.0, np.cumsum(resistance)]
    profile /= profile[-1]
    change = -comp.voltage_polarity * (v_to - v_from)
    before = values[:n_c].copy()
    values[:n_c] += change * profile
    shift = (values[:n_c] - before) / comp.thermal_voltage_V
    values[n_c : 2 * n_c] -= shift
    values[2 * n_c : 3 * n_c] -= shift
    u_mut[: 3 * n_c] = values
    return frozen_array(u_mut)


def _local_steady_iodine_coordinates(
    device: CompiledIodineDevice1D,
    elec_u: Vector,
    voltage_V: float,
    *,
    cell_total_iodine_m3: Vector | None = None,
) -> Vector:
    """Closed-form local photochemical steady seed conserving cell iodine."""
    if device.is_unreactive_limit:
        return device.initial_coordinates(voltage_V=voltage_V, electronic_u=elec_u)
    _, n_full, p_full = device.electronic_component.fields(elec_u, voltage_V=voltage_V)
    idx = device.chem_node_indices
    n = n_full[idx]
    p = p_full[idx]
    kin = device.kinetics
    assert kin.n_1_iodine_m3 is not None and kin.k_e_3body_m6_s is not None
    reversible = kin.thermodynamic_mode == "wegscheider_detailed_balance"
    n1 = kin.n_1_iodine_m3 if reversible else np.zeros_like(n)
    k_diss = kin.K_diss_m3

    # Monomer quasi-steady ratio r0 = c_0 / c_-
    denom_0 = kin.k_b_s_1 + kin.k_c_m3_s * n
    r0 = np.where(
        denom_0 > 0.0,
        (kin.k_a_m3_s * p + kin.k_c_m3_s * n1) / np.maximum(denom_0, 1.0e-30),
        0.0,
    )

    # Dimer quasi-steady quadratic coefficient q2 = c_2 / c_-^2
    if kin.reaction_e_order == "bimolecular_n":
        if reversible:
            ke_eff = kin.k_e_m3_s / np.maximum(n + kin.n_e_star_m3, 1.0e-30)
            ke_fwd = ke_eff * (n * n)
        else:
            ke_eff = np.zeros_like(n)
            ke_fwd = kin.k_e_m3_s * n
    else:
        ke_eff = kin.k_e_3body_m6_s if reversible else np.zeros_like(n)
        ke_fwd = kin.k_e_3body_m6_s * (n * n)

    assert device.volumetric_mol_escape_s_1 is not None
    alpha_e = (n1 * n1) / k_diss if reversible else np.zeros_like(n)
    num_2 = kin.k_d_m3_s * (r0 * r0) + ke_eff * alpha_e
    den_2 = (
        (kin.k_d_m3_s * k_diss if reversible else 0.0)
        + ke_fwd
        + device.volumetric_mol_escape_s_1
    )
    q2 = np.where(den_2 > 0.0, num_2 / np.maximum(den_2, 1.0e-30), 0.0)

    c_tot = (
        device.initial_total_iodine_atoms_m3
        if cell_total_iodine_m3 is None
        else cell_total_iodine_m3
    )
    lin_b = 1.0 + r0
    disc = np.sqrt(lin_b * lin_b + 8.0 * q2 * c_tot)
    cm = (2.0 * c_tot) / (lin_b + disc)
    # Clamp trace c0, c2 at vacuum_floor_m3 so coordinates stay O(1) while
    # keeping residual << 1.0 and adjusting cm to conserve c_tot!
    floor_0 = getattr(device.atom_map, "vacuum_floor_m3", 1.0)
    floor_2 = getattr(device.mol_map, "vacuum_floor_m3", 1.0)
    c0 = np.maximum(r0 * cm, floor_0 * 1.0e-3)
    c2 = np.maximum(q2 * (cm * cm), floor_2 * 1.0e-3)
    cm = np.maximum(c_tot - c0 - 2.0 * c2, c_tot * 0.5)

    u_chem = np.concatenate(
        (
            device.iodide_map.coordinate_from_density(cm),
            device.atom_map.coordinate_from_density(c0),
            device.mol_map.coordinate_from_density(c2),
        )
    )
    return frozen_array(np.concatenate((elec_u, u_chem)))


def _poisson_boltzmann_dark_seed(device: CompiledIodineDevice1D) -> Vector:
    """Grand-canonical 1D Poisson-Boltzmann + Wegscheider dark equilibrium seed."""
    comp = device.electronic_component
    n_count = device.node_count
    idx = device.chem_node_indices
    w_chem = device.chem_geometry.cell_measures_m
    w_e = comp.geometry.cell_measures_m
    vt = comp.thermal_voltage_V
    l_pot, r_pot = comp.contact_potentials(0.0)
    phi = l_pot + (r_pot - l_pot) * (
        (comp.nodes_m - comp.nodes_m[0]) / comp.geometry.extent_m
    )
    k_part_m = np.exp(
        np.concatenate(([0.0], np.cumsum(device.log_partition_minus_faces)))
    )
    k_part_0 = np.exp(
        np.concatenate(([0.0], np.cumsum(device.log_partition_atom_faces)))
    )
    k_part_2 = np.exp(
        np.concatenate(([0.0], np.cumsum(device.log_partition_mol_faces)))
    )
    m_tot = device.reference_iodine_inventory_m2
    m_minus_only = device.reference_species_inventories_m2["iodide"]
    j_disp = comp.poisson.jacobian().toarray()
    ratio = device.chem_weight_ratio
    kin = device.kinetics
    use_weg = (
        kin.thermodynamic_mode == "wegscheider_detailed_balance"
        and not device.is_unreactive_limit
        and kin.n_1_iodine_m3 is not None
        and bool(np.any(kin.n_1_iodine_m3 > 0.0))
    )

    for _ in range(120):
        u_e = np.concatenate((phi, np.zeros(2 * n_count)))
        _, n, p = comp.fields(u_e, voltage_V=0.0)
        phi_ref_max = float(np.max(phi[idx]))
        bm = k_part_m * np.exp((phi[idx] - phi_ref_max) / vt)
        if use_weg:
            assert kin.n_1_iodine_m3 is not None
            act = kin.n_1_iodine_m3 > 0.0
            first_act = int(np.flatnonzero(act)[0])
            b0_unit = float(
                (bm[first_act] * kin.n_1_iodine_m3[first_act])
                / (n[idx[first_act]] * k_part_0[first_act])
            )
            b0 = b0_unit * k_part_0
            kdiss_0 = float(kin.K_diss_m3[first_act]) / (
                (k_part_0[first_act] ** 2) / k_part_2[first_act]
            )
            b2 = ((b0_unit * b0_unit) / kdiss_0) * k_part_2
            a1 = float(np.sum(w_chem * (bm + b0)))
            q2 = float(np.sum(2.0 * w_chem * b2))
            disc = math.sqrt(a1 * a1 + 4.0 * q2 * m_tot)
            alpha = (2.0 * m_tot) / (a1 + disc) if (a1 + disc) > 0.0 else 0.0
            d_alpha_dphi = -(alpha / max(a1 + 2.0 * q2 * alpha, 1.0e-30)) * (
                w_chem * bm / vt
            )
        else:
            z_norm = max(float(np.sum(w_chem * bm)), 1.0e-30)
            alpha = m_minus_only / z_norm
            d_alpha_dphi = -(alpha / z_norm) * (w_chem * bm / vt)
        cm = alpha * bm
        rho = device.charge_ledger(n, p, cm).space_charge_C_m3()
        f_phi = comp.poisson.residual(phi, rho, l_pot, r_pot)
        if float(np.max(np.abs(f_phi[1:-1]))) < 1.0e-13:
            break
        j_pb = j_disp.copy()
        diag_sc = -Q * w_e * (n + p) / vt
        diag_sc[idx] -= Q * w_e[idx] * ratio * cm / vt
        diag_sc[0] = 0.0
        diag_sc[-1] = 0.0
        j_pb += np.diag(diag_sc)
        row_vec = -Q * w_e[idx] * ratio * bm
        if int(idx[0]) == 0:
            row_vec[0] = 0.0
        if int(idx[-1]) == n_count - 1:
            row_vec[-1] = 0.0
        j_pb[np.ix_(idx, idx)] += np.outer(row_vec, d_alpha_dphi)
        dphi = -np.linalg.solve(j_pb, f_phi)
        step = min(1.0, 0.15 / max(float(np.max(np.abs(dphi))), 1.0e-30))
        phi = phi + step * dphi

    u_e = np.concatenate((phi, np.zeros(2 * n_count)))
    _, n, _ = comp.fields(u_e, voltage_V=0.0)
    bm = k_part_m * np.exp((phi[idx] - float(np.max(phi[idx]))) / vt)
    if use_weg:
        assert kin.n_1_iodine_m3 is not None
        act = kin.n_1_iodine_m3 > 0.0
        first_act = int(np.flatnonzero(act)[0])
        b0_unit = float(
            (bm[first_act] * kin.n_1_iodine_m3[first_act])
            / (n[idx[first_act]] * k_part_0[first_act])
        )
        b0 = b0_unit * k_part_0
        kdiss_0 = float(kin.K_diss_m3[first_act]) / (
            (k_part_0[first_act] ** 2) / k_part_2[first_act]
        )
        b2 = ((b0_unit * b0_unit) / kdiss_0) * k_part_2
        a1 = float(np.sum(w_chem * (bm + b0)))
        q2 = float(np.sum(2.0 * w_chem * b2))
        disc = math.sqrt(a1 * a1 + 4.0 * q2 * m_tot)
        alpha = (2.0 * m_tot) / (a1 + disc)
        cm = alpha * bm
        c0 = np.maximum(alpha * b0, 1.0)
        c2 = np.maximum((alpha * alpha) * b2, 1.0)
    else:
        z_norm = max(float(np.sum(w_chem * bm)), 1.0e-30)
        cm = np.maximum((m_minus_only / z_norm) * bm, 1.0)
        c0 = np.maximum(device.initial_atom_m3, 1.0)
        c2 = np.maximum(device.initial_mol_m3, 1.0)
    return frozen_array(
        np.concatenate(
            (
                u_e,
                device.iodide_map.coordinate_from_density(cm),
                device.atom_map.coordinate_from_density(c0),
                device.mol_map.coordinate_from_density(c2),
            )
        )
    )


def solve_iodine_steady_state(
    device: CompiledIodineDevice1D,
    *,
    voltage_V: float = 0.0,
    generation_scale: float = 0.0,
    initial_u: Vector | None = None,
    inventory_node: int = 0,
    options: IodineSteadyOptions = IodineSteadyOptions(),
) -> IodineSteadyResult:
    """Solve closed-domain coupled steady state from exact PB dark equilibrium."""
    system = iodine_system(
        device, constrained_steady=True, inventory_node=inventory_node
    )
    unconstrained = iodine_system(device, constrained_steady=False)

    if initial_u is None:
        if device.steric_site_limit_m3 is None:
            u_curr = _poisson_boltzmann_dark_seed(device)
        else:
            eq_seed = solve_equilibrium(
                device.electronic_component,
                options=SteadyOptions(convergence_policy=POISSON_CONTACT_BALANCE_V1),
            )
            u_curr = _local_steady_iodine_coordinates(device, eq_seed.state.u, 0.0)
        prev_v = 0.0
        stages: list[tuple[float, float]] = [(0.0, 0.0)]
        comp_p = device.electronic_component.parameters
        undoped_contacts = bool(
            (comp_p.donor_density_m3[0] + comp_p.acceptor_density_m3[0]) < 1.0e15
            or (comp_p.donor_density_m3[-1] + comp_p.acceptor_density_m3[-1]) < 1.0e15
        )
        if undoped_contacts and voltage_V > 0.3 and generation_scale > 0.0:
            # Ramp forward bias in the dark first to reduce junction barrier and
            # avoid SCLC extraction bottlenecks in undoped low-mobility blocking layers.
            n_v = max(1, int(math.ceil(abs(voltage_V) / 0.10)))
            for k in range(1, n_v + 1):
                stages.append((voltage_V * (k / n_v), 0.0))
            for frac in (1.0e-5, 1.0e-3, 0.03, 0.15, 0.45, 0.75, 1.0):
                stages.append((voltage_V, generation_scale * frac))
        else:
            if generation_scale > 0.0:
                for frac in (1.0e-5, 1.0e-3, 0.03, 0.25, 1.0):
                    stages.append((0.0, generation_scale * frac))
            if voltage_V != 0.0:
                n_v = max(1, int(math.ceil(abs(voltage_V) / 0.10)))
                for k in range(1, n_v + 1):
                    stages.append((voltage_V * (k / n_v), generation_scale))
    else:
        u_curr = frozen_array(initial_u)
        prev_v = voltage_V
        stages = [(voltage_V, generation_scale)]

    last_newton: NewtonResult | None = None
    total_iters = 0
    prev_g = 0.0 if initial_u is None else generation_scale

    steady_norm = iodine_convergence_norm(
        device, constrained_steady=True, inventory_node=inventory_node
    )
    bdf_norm = iodine_convergence_norm(device, constrained_steady=False)
    stage_stack: list[tuple[float, float, int]] = [
        (v_s, g_s, 0) for v_s, g_s in reversed(stages)
    ]
    while stage_stack:
        v_step, g_step, depth = stage_stack.pop()
        u_trial = _harmonic_voltage_lift(device, u_curr, prev_v, v_step)
        prob = AssembledConstrainedIodine1DProblem(
            device=device,
            voltage_V=v_step,
            generation_scale=g_step,
            inventory_node=inventory_node,
            binding=system,
        )
        if g_step > prev_g:
            u_trial = _carrier_density_basin_lift(
                CallableProblem(prob.residual, prob.jacobian),
                device,
                u_trial,
                v_step,
            )
        settings = iodine_steady_newton_settings(
            device,
            voltage_V=v_step,
            inventory_node=inventory_node,
            options=options,
        )
        res = solve_newton(
            CallableProblem(prob.residual, prob.jacobian),
            u_trial,
            settings,
            convergence_norm=steady_norm,
        )
        total_iters += res.iteration_count
        if (
            (not res.accepted or res.termination != Termination.CONVERGED)
            and depth < 6
            and (abs(v_step - prev_v) > 2.0e-3 or abs(g_step - prev_g) > 1.0e-8)
        ):
            v_mid = 0.5 * (prev_v + v_step)
            g_mid = (
                math.sqrt(prev_g * g_step)
                if (prev_g > 0.0 and g_step / prev_g > 4.0)
                else 0.5 * (prev_g + g_step)
            )
            stage_stack.append((v_step, g_step, depth + 1))
            stage_stack.append((v_mid, g_mid, depth + 1))
            continue
        if not res.accepted or res.termination != Termination.CONVERGED:
            # Mass-regularized BDF relaxation into steady basin followed by exact polish
            ptc_opts = BDFOptions(
                initial_step_s=1.0e-6,
                min_step_s=1.0e-14,
                max_step_s=200.0,
                relative_tolerance=5.0e-3,
                absolute_tolerance=1.0e14,
                max_order=1,
                max_factor=2.0,
                max_steps=250,
                wall_s=options.wall_s,
            )

            def _ptc_factory(_t: float, _side: str) -> AssembledIodine1DProblem:
                return AssembledIodine1DProblem(
                    device=device,
                    voltage_V=v_step,
                    generation_scale=g_step,
                    binding=unconstrained,
                )

            ptc_bdf = solve_bdf(
                _ptc_factory,
                u_trial,
                (0.0, 200.0),
                options=ptc_opts,
                settings=iodine_bdf_newton_settings,
                seed=iodine_bdf_seed,
                convergence_norm=bdf_norm,
            )
            total_iters += ptc_bdf.nonlinear_calls
            if ptc_bdf.accepted_coordinates:
                u_warm = ptc_bdf.accepted_coordinates[-1]
                res = solve_newton(
                    CallableProblem(prob.residual, prob.jacobian),
                    u_warm,
                    settings,
                    convergence_norm=steady_norm,
                )
                total_iters += res.iteration_count

        last_newton = res
        if not res.accepted or res.termination != Termination.CONVERGED:
            return IodineSteadyResult(
                converged=False,
                state=None,
                current_density_A_m2=math.nan,
                termination=res.termination.value,
                scaled_residual=res.scaled_residual,
                iterations=total_iters,
                newton=res,
            )
        u_curr = res.state
        prev_v = v_step
        prev_g = g_step

    assert last_newton is not None
    state = device.state(u_curr, voltage_V=voltage_V)
    obs = AssembledIodine1DProblem(
        device=device,
        voltage_V=voltage_V,
        generation_scale=generation_scale,
        binding=unconstrained,
    )
    j_steady = obs.steady_current_density_A_m2(u_curr)
    return IodineSteadyResult(
        converged=True,
        state=state,
        current_density_A_m2=j_steady,
        termination=last_newton.termination.value,
        scaled_residual=last_newton.scaled_residual,
        iterations=total_iters,
        newton=last_newton,
    )


def solve_frozen_chemistry_electronic_1d(
    device: CompiledIodineDevice1D,
    u: Vector,
    *,
    voltage_V: float,
    generation_scale: float,
    prev_voltage_V: float | None = None,
    binding: IodineSystem | None = None,
) -> tuple[Vector, float]:
    """Solve stationary ``3 * N`` electronic rows with frozen chemical coordinates.

    Preserves ``u[3 * N :]`` bit-for-bit (0-ULP iodine species invariance) while
    relaxing ``(phi, z_n, z_p)`` to ``(voltage_V, generation_scale)``. Returns
    ``(u_relaxed, electronic_current_density_A_m2)``.
    """
    comp = device.electronic_component
    n_c = device.node_count
    n_elec = 3 * n_c
    u_chem = np.array(u[n_elec:], dtype=np.float64, copy=True)
    v_from = _inferred_voltage_V(device, u) if prev_voltage_V is None else float(prev_voltage_V)
    v_target = float(voltage_V)
    g_target = float(generation_scale)
    _ = binding

    weights = production_residual_scaling(
        comp,
        poisson_tolerance=5.0e-9,
        continuity_tolerance_A_m2=2.0e-6,
        contact_log_tolerance=1.0e-11,
    ).row_weights.copy()
    weights[:n_c] = _poisson_weights_with_ions(device) / 5.0e-9
    steps = production_coordinates(
        comp,
        potential_step_V=0.15,
        quasi_fermi_step=2.5,
    ).max_step
    settings = NewtonSettings(
        scaling=ResidualScaling(weights),
        tolerance=1.0,
        step_control=StepControl(steps),
        budget=SolveBudget(80, 1400, 82, 16),
        gates=(),
        reuse_jacobian=False,
    )
    elec_norm = production_convergence_norm(
        comp, convergence_policy=POISSON_CONTACT_BALANCE_V1
    )

    # Static space-charge contribution of frozen iodide profile in interior Gauss cells
    _, _, _, cm_frozen, _, _, _, _, _ = device.fields_and_tangents(u, voltage_V=v_from)
    w_e = comp.geometry.cell_measures_m
    net_ion_m3 = device.project_chem_density(cm_frozen) - device.project_chem_density(
        device.compensating_donor_m3
    )
    static_ion_gauss: NDArray[np.float64] = np.zeros(n_c, dtype=np.float64)
    static_ion_gauss[1:-1] = -Q * w_e[1:-1] * net_ion_m3[1:-1]

    def _f_dc_frozen(ue: Vector, dc_p: DC1DProblem) -> Vector:
        res_v = dc_p.residual(ue).copy()
        res_v[:n_c] += static_ion_gauss
        return res_v

    u_cur = np.array(u, dtype=np.float64, copy=True)
    p_check = DC1DProblem(comp, v_from, g_target)
    need_light_ramp = (
        float(np.max(np.abs(_f_dc_frozen(u_cur[:n_elec], p_check)[n_c:n_elec]))) * Q
        > 1.0e-2
    )
    stages: list[tuple[float, float]] = []
    if need_light_ramp:
        if g_target > 0.0:
            for frac in (1.0e-5, 1.0e-3, 0.03, 0.25, 1.0):
                stages.append((v_from, g_target * frac))
        else:
            for frac in (0.2, 0.02, 1.0e-3, 1.0e-5, 0.0):
                stages.append((v_from, frac))
    if abs(v_target - v_from) > 1.0e-13:
        n_v = max(1, int(math.ceil(abs(v_target - v_from) / 0.10)))
        for k in range(1, n_v + 1):
            stages.append((v_from + (v_target - v_from) * (k / n_v), g_target))
    if not stages:
        stages.append((v_target, g_target))

    cur_v = v_from
    cur_g = (0.0 if g_target > 0.0 else 1.0) if need_light_ramp else g_target
    stack: list[tuple[float, float, int]] = [
        (vs, gs, 0) for vs, gs in reversed(stages)
    ]
    while stack:
        vs, gs, depth = stack.pop()
        u_try = _harmonic_voltage_lift(device, u_cur, cur_v, vs)
        dc_prob = DC1DProblem(comp, vs, gs)

        def _f_e(ue: Vector, _dp: DC1DProblem = dc_prob) -> Vector:
            return _f_dc_frozen(ue, _dp)

        def _j_e(ue: Vector, _dp: DC1DProblem = dc_prob) -> csc_matrix:
            return _dp.jacobian(ue)

        if gs > cur_g:
            # Lift 3N carriers in linear density space against frozen iodide field
            def _f_full(uf: Vector, _dp: DC1DProblem = dc_prob) -> Vector:
                out = np.zeros_like(uf)
                out[:n_elec] = _f_dc_frozen(uf[:n_elec], _dp)
                return out

            def _j_full(uf: Vector, _dp: DC1DProblem = dc_prob) -> csc_matrix:
                from scipy.sparse import block_diag, eye

                return csc_matrix(
                    block_diag(
                        (_dp.jacobian(uf[:n_elec]), eye(uf.size - n_elec, format="csc")),
                        format="csc",
                    )
                )

            u_try = _carrier_density_basin_lift(
                CallableProblem(_f_full, _j_full), device, u_try, vs
            )

        res = solve_newton(
            CallableProblem(_f_e, _j_e),
            u_try[:n_elec],
            settings,
            convergence_norm=elec_norm,
        )
        if res.accepted:
            u_cur = np.concatenate((res.state, u_chem))
            cur_v, cur_g = vs, gs
        elif depth < 6 and (abs(vs - cur_v) > 2.0e-3 or abs(gs - cur_g) > 1.0e-8):
            stack.append((vs, gs, depth + 1))
            stack.append((0.5 * (cur_v + vs), 0.5 * (cur_g + gs), depth + 1))
        else:
            raise SteadyStateError(
                f"Frozen-chemistry electronic solve failed at V={vs:.4f} V: {res.termination.value}",
                res,
            )

    final_dc = DC1DProblem(comp, v_target, g_target)
    j_elec_faces = final_dc.current_density(u_cur[:n_elec])
    dx = np.diff(comp.nodes_m)
    j_mean = float(math.fsum((dx * j_elec_faces).tolist()) / math.fsum(dx.tolist()))
    return frozen_array(u_cur), j_mean


def solve_iodine_bdf_transient(
    device: CompiledIodineDevice1D,
    initial_u: Vector,
    interval_s: tuple[float, float],
    *,
    voltage_fn: Callable[[float], float],
    generation_fn: Callable[[float], float],
    options: BDFOptions,
    output_times_s: Sequence[float] | Vector | None = None,
    breakpoints_s: tuple[float, ...] = (),
    events: tuple[BDFEvent, ...] = (),
) -> tuple[BDFResult, tuple[IodineStepAccounting, ...]]:
    """Integrate the assembled 1D iodine system via adaptive BDF1/2."""
    binding = iodine_system(device, constrained_steady=False)
    step_ledger: list[IodineStepAccounting] = []

    start_s = float(interval_s[0])
    v_u0 = _inferred_voltage_V(device, initial_u)
    v_start_target = float(voltage_fn(start_s))
    g_start_target = float(generation_fn(start_s))
    all_events = list(events)
    has_start_event = any(ev.time_s == start_s for ev in all_events)
    n_c = device.node_count
    probe_right = AssembledIodine1DProblem(
        device=device,
        voltage_V=v_start_target,
        generation_scale=g_start_target,
        binding=binding,
    )
    elec_jump = False
    if abs(v_u0 - v_start_target) > 1.0e-12:
        elec_jump = True
    else:
        f_e0 = probe_right.residual(initial_u)[n_c : 3 * n_c]
        if float(np.max(np.abs(f_e0))) * Q > 1.0e-3:
            elec_jump = True
    use_auto_start_jump = not has_start_event and elec_jump

    def factory(t_s: float, side: str) -> AssembledIodine1DProblem:
        if use_auto_start_jump and t_s == start_s and side == "left":
            return AssembledIodine1DProblem(
                device=device,
                voltage_V=v_u0,
                generation_scale=0.0,
                binding=binding,
            )
        return AssembledIodine1DProblem(
            device=device,
            voltage_V=float(voltage_fn(t_s)),
            generation_scale=float(generation_fn(t_s)),
            binding=binding,
        )

    if use_auto_start_jump:
        def _start_restart(
            _old_p: object, _new_p: object, left_u: Vector
        ) -> tuple[Vector, Assessment]:
            relaxed_u, _ = solve_frozen_chemistry_electronic_1d(
                device,
                left_u,
                voltage_V=v_start_target,
                generation_scale=g_start_target,
                prev_voltage_V=v_u0,
                binding=binding,
            )
            chem_unchanged = bool(np.array_equal(relaxed_u[3 * n_c :], left_u[3 * n_c :]))
            return relaxed_u, Assessment(
                accounting={
                    "voltage_jump_V": v_start_target - v_u0,
                    "g_jump": g_start_target,
                },
                checks=(
                    GateOutcome(
                        "frozen_chemistry_start_relaxation",
                        chem_unchanged,
                        0.0 if chem_unchanged else 1.0,
                        0.5,
                    ),
                ),
            )

        all_events.insert(
            0, BDFEvent(time_s=start_s, restart=_start_restart, name="initial_drive_step")
        )

    def assess(step: BDFStep, u: Vector) -> Assessment:
        acct = iodine_step_accounting(step, u)
        inv_ref = max(device.reference_iodine_inventory_m2, 1.0)
        rel_err = abs(acct.step_conservation_residual_m2) / inv_ref
        step_lim = 1.0e-9
        if (
            isinstance(step.endpoint, AssembledIodine1DProblem)
            and step.endpoint.generation_scale > 0.0
        ):
            step_lim = max(1.0e-9, (6.0e-8 / Q) * step.dt_s / inv_ref)
        gate = GateOutcome(
            "step_iodine_conservation", rel_err <= step_lim, rel_err, step_lim
        )
        return Assessment(
            accounting={
                "in_device_iodine_m2": acct.in_device_iodine_m2,
                "step_escaped_iodine_atoms_m2": acct.step_escaped_iodine_atoms_m2,
                "step_immobilized_iodine_atoms_m2": acct.step_immobilized_iodine_atoms_m2,
                "step_conservation_residual_m2": acct.step_conservation_residual_m2,
                "terminal_current_A_m2": acct.terminal_current_A_m2,
                "max_total_current_spread_A_m2": acct.max_total_current_spread_A_m2,
            },
            checks=(gate,),
        )

    out_arr: NDArray[np.float64] | None = (
        None
        if output_times_s is None
        else np.asarray(output_times_s, dtype=np.float64)
    )
    eff_options = options
    if use_auto_start_jump and not device.is_unreactive_limit:
        # Match initial step to fastest active electron-capture rate k_c * n + k_b
        # so step-doubling accepts immediately without burning rejection budget.
        kin = device.kinetics
        comp_f = device.electronic_component
        n_est = max(
            float(
                np.max(comp_f.parameters.conduction_dos_m3[device.chem_node_indices])
            )
            * 1.0e-3,
            1.0e21,
        )
        fast_rate = float(np.max(kin.k_b_s_1 + kin.k_c_m3_s * n_est))
        if fast_rate > 0.0:
            auto_h0 = max(options.min_step_s * 10.0, 0.02 / fast_rate)
            if auto_h0 < options.initial_step_s:
                from dataclasses import replace as _dc_replace

                eff_options = _dc_replace(options, initial_step_s=auto_h0)

    bdf_res = solve_bdf(
        factory,
        initial_u,
        interval_s,
        options=eff_options,
        settings=iodine_bdf_newton_settings,
        assess=assess,
        seed=iodine_bdf_seed,
        breakpoints_s=breakpoints_s,
        events=tuple(all_events),
        output_times_s=out_arr,
        convergence_norm=iodine_convergence_norm(device, constrained_steady=False),
    )
    prev_esc_step = 0.0
    prev_immob_step = 0.0
    for attempt in bdf_res.accepted_steps:
        assert attempt.assessment is not None
        acc = attempt.assessment.accounting
        c0_bdf = float(attempt.coefficients[0])
        c2_bdf = float(attempt.coefficients[2]) if attempt.order == 2 else 0.0
        raw_esc = float(acc["step_escaped_iodine_atoms_m2"])
        raw_immob = float(acc["step_immobilized_iodine_atoms_m2"])
        # Exact BDF1/2 flux-storage consistent step increments:
        eff_esc = (c2_bdf * prev_esc_step + raw_esc) / c0_bdf
        eff_immob = (c2_bdf * prev_immob_step + raw_immob) / c0_bdf
        prev_esc_step = eff_esc
        prev_immob_step = eff_immob
        step_ledger.append(
            IodineStepAccounting(
                start_s=attempt.start_s,
                end_s=attempt.end_s,
                dt_s=attempt.end_s - attempt.start_s,
                in_device_iodine_m2=float(acc["in_device_iodine_m2"]),
                step_escaped_iodine_atoms_m2=eff_esc,
                step_immobilized_iodine_atoms_m2=eff_immob,
                step_conservation_residual_m2=float(
                    acc["step_conservation_residual_m2"]
                ),
                terminal_current_A_m2=float(acc["terminal_current_A_m2"]),
                max_total_current_spread_A_m2=float(
                    acc["max_total_current_spread_A_m2"]
                ),
            )
        )
    return bdf_res, tuple(step_ledger)
