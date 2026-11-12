"""Coupled 2D steady solver with whole-domain finite-site ion inventory closure.

Replaces one ion balance row (``3*N + inventory_row_node``) with the physical 2D
domain inventory constraint ``fsum(area * P) - fsum(area * P0) = 0`` (or closes
every ion cell to ``P0`` when ``D_ion == 0`` everywhere). Every original ion row
is checked independently in ``diagnostics`` and ``checks``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import math
import time
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csc_matrix

from solarlab.contracts import frozen_array
from solarlab.contracts.units import ELEMENTARY_CHARGE_C as Q
from solarlab.equations.assembly.ionic_2d import AssembledConstrainedIonic2DProblem, AssembledIonic2DProblem
from solarlab.equations.dc_1d import CapabilityError
from solarlab.equations.implementation import use_assembled
from solarlab.equations.ionic_2d import CompiledIonicDevice2D, Ionic2DProblem, IonicState2D
from solarlab.reproducibility.job_provenance import admitted_source_pins
from solarlab.solvers.nonlinear.newton import NewtonResult, NewtonSettings, solve_newton
from solarlab.solvers.nonlinear.problem import CallableProblem, GateOutcome, ResidualScaling, SolveBudget, StepControl, state_digest
from solarlab.solvers.nonlinear.production import HISTORICAL_MAX_ROW_V1
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions

__all__ = [
    "ConstrainedIonic2DProblem",
    "IonicSteady2DOptions",
    "IonicSteady2DResult",
    "checks",
    "diagnostics",
    "poisson_weights_2d",
    "solve_ionic_steady_state_2d",
]


@dataclass(frozen=True)
class ConstrainedIonic2DProblem:
    """Reference 2D ionic steady problem with mobile inventory or immobile profile closure."""

    original: Ionic2DProblem
    inventory_row_node: int = 0

    def __post_init__(self) -> None:
        d = self.original.device
        if type(self.inventory_row_node) is not int or not 0 <= self.inventory_row_node < d.ion_node_count:
            raise ValueError("inventory_row_node must name a 2D ion-domain node")
        manifest = d.binding_manifest
        if hashlib.sha256(d.binding_manifest_json.encode()).hexdigest() != d.identity:
            raise ValueError("2D ionic binding manifest hash mismatch")
        domain = manifest.get("ion_domain")
        contacts = domain.get("contacts") if isinstance(domain, dict) else None
        if not isinstance(contacts, list) or contacts[:2] != ["blocking", "blocking"]:
            raise CapabilityError("2D ionic steady closure requires four-side blocking domain")

    @property
    def is_immobile(self) -> bool:
        return bool(np.all(self.original.device.positive_ion_diffusivity_faces_m2_s == 0.0))

    @property
    def constraint_row(self) -> int:
        return 3 * self.original.device.node_count + self.inventory_row_node

    @property
    def inventory_target_m1(self) -> float:
        return self.original.device.reference_inventory_m1

    def residual(self, u: ArrayLike) -> np.ndarray:
        res = self.original.residual(u)
        if not np.all(np.isfinite(res)):
            return res
        d = self.original.device
        _, _, _, ion_p = d.fields(u, voltage_V=self.original.voltage_V)
        vol = d.ion_geometry.contract_geometry.volumes
        if self.is_immobile:
            res[3 * d.node_count:] = vol * (ion_p - d.positive_ion_reference_m3)
            return res
        inventory = math.fsum((vol * ion_p).tolist())
        res[self.constraint_row] = inventory - self.inventory_target_m1
        return res

    def jacobian(self, u: ArrayLike) -> csc_matrix:
        jac = self.original.jacobian(u).tolil()
        d = self.original.device
        _, _, _, ion_p = d.fields(u, voltage_V=self.original.voltage_V)
        vol = d.ion_geometry.contract_geometry.volumes
        dp_dy = vol * ion_p * (1.0 - ion_p / d.positive_ion_site_limit_m3)
        start = 3 * d.node_count
        if self.is_immobile:
            for idx in range(d.ion_node_count):
                row = start + idx
                jac.rows[row] = [row]
                jac.data[row] = [float(dp_dy[idx])]
            return jac.tocsc()
        row = self.constraint_row
        jac.rows[row] = list(range(start, start + d.ion_node_count))
        jac.data[row] = dp_dy.tolist()
        return jac.tocsc()


@dataclass(frozen=True)
class IonicSteady2DOptions:
    """Explicit numerical controls and conservation gates for 2D coupled ionic steady."""

    electronic: SteadyOptions = field(default_factory=SteadyOptions)
    inventory_relative_tolerance: float = 1.0e-12
    ion_balance_tolerance_m1_s: float = 1.0e3
    ion_flux_tolerance_m2_s: float = 1.0e9
    ion_logit_step: float = 1.0
    inventory_row_node: int = 0
    budget: SolveBudget = field(default_factory=lambda: SolveBudget(60, 600, 60, 24))
    wall_s: float = 30.0

    def __post_init__(self) -> None:
        if not isinstance(self.electronic, SteadyOptions) or not isinstance(self.budget, SolveBudget):
            raise ValueError("SteadyOptions and SolveBudget are required")
        for name in (
            "inventory_relative_tolerance",
            "ion_balance_tolerance_m1_s",
            "ion_flux_tolerance_m2_s",
            "ion_logit_step",
            "wall_s",
        ):
            val = float(getattr(self, name))
            if not math.isfinite(val) or val <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class IonicSteady2DResult:
    """Coupled 2D ionic steady solve outcome and independent gate diagnostics."""

    state: IonicState2D | None
    accepted: bool
    termination: str
    newton: NewtonResult | None
    initial_coordinates: np.ndarray
    candidate_coordinates: np.ndarray
    diagnostics: dict[str, Any] | None
    checks: tuple[GateOutcome, ...]
    compiled_identity: str
    operating_voltage_V: float
    generation_scale: float
    solver_controls: dict[str, Any]
    source_sha256: dict[str, str]
    wall_s: float
    error: str | None = None
    scientific_qualification: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "initial_coordinates", frozen_array(self.initial_coordinates))
        object.__setattr__(self, "candidate_coordinates", frozen_array(self.candidate_coordinates))


_Unconstrained2D: TypeAlias = Ionic2DProblem | AssembledIonic2DProblem


def poisson_weights_2d(device: CompiledIonicDevice2D) -> np.ndarray:
    comp = device.electronic_component
    g = comp.geometry.contract_geometry
    reference_V = max(abs(comp.column.built_in_potential_V), comp.thermal_voltage_V)
    density = comp.extrude(np.maximum(np.maximum(
        comp.column.parameters.acceptor_density_m3,
        comp.column.parameters.donor_density_m3,
    ), 1.0))
    density = density.copy()
    density[device.ion_node_indices] = np.maximum(
        density[device.ion_node_indices],
        device.positive_ion_reference_m3,
    )
    capacitance = np.zeros(comp.node_count)
    face = comp.face_capacitance_F_m2 * g.face_measures
    np.add.at(capacitance, g.pairs[:, 0], face)
    np.add.at(capacitance, g.pairs[:, 1], face)
    scale = Q * density * g.volumes + reference_V * capacitance
    result = 1.0 / scale
    result[comp.geometry.contact_nodes] = 1.0 / reference_V
    return result


def _newton_settings(device: CompiledIonicDevice2D, options: IonicSteady2DOptions, gate: Any) -> NewtonSettings:
    c = options.electronic
    comp = device.electronic_component
    n_elec = comp.node_count
    n_ion = device.ion_node_count
    weights: NDArray[np.float64] = np.empty(device.state_size, dtype=np.float64)
    weights[:n_elec] = poisson_weights_2d(device) / c.poisson_tolerance
    weights[n_elec:3 * n_elec] = 2.0 * n_elec * Q / (
        comp.geometry.lateral_length_m * c.continuity_tolerance_A_m2
    )
    weights[n_elec:3 * n_elec].reshape(2, n_elec)[comp.fixed_carrier_mask] = 1.0 / c.contact_log_tolerance
    if np.all(device.positive_ion_diffusivity_faces_m2_s == 0.0):
        weights[3 * n_elec:] = 1.0 / (device.reference_inventory_m1 * options.inventory_relative_tolerance / n_ion)
    else:
        weights[3 * n_elec:] = 1.0 / options.ion_balance_tolerance_m1_s
        weights[3 * n_elec + options.inventory_row_node] = 1.0 / (
            device.reference_inventory_m1 * options.inventory_relative_tolerance
        )
    steps = np.r_[
        np.full(n_elec, c.potential_step_V),
        np.full(2 * n_elec, c.quasi_fermi_step),
        np.full(n_ion, options.ion_logit_step),
    ]
    return NewtonSettings(ResidualScaling(weights), 1.0, StepControl(steps), options.budget, (gate,), False)


def diagnostics(problem: _Unconstrained2D, u: np.ndarray, options: IonicSteady2DOptions) -> dict[str, Any]:
    d = problem.device
    comp = d.electronic_component
    n_elec = comp.node_count
    raw_res = problem.residual(u)
    phi, n, p, ion_p = d.fields(u, voltage_V=problem.voltage_V)
    if not (np.all(np.isfinite(raw_res)) and np.all(np.isfinite(n)) and np.all(np.isfinite(p)) and np.all(np.isfinite(ion_p))):
        raise ValueError("2D ionic diagnostic fields or original residual are non-finite")
    poisson_norm = float(np.max(np.abs(raw_res[:n_elec] * poisson_weights_2d(d))))
    carrier_rows = raw_res[n_elec:3 * n_elec].reshape(2, n_elec).copy()
    contact_logs = carrier_rows[comp.fixed_carrier_mask].copy()
    carrier_rows[comp.fixed_carrier_mask] = 0.0
    width = comp.geometry.lateral_length_m
    carrier_bound = Q * float(np.sum(np.abs(carrier_rows))) / width

    ion_rows = raw_res[3 * n_elec:]
    ion_eval = problem.ion_flux_evaluation(u)
    inventory = math.fsum((d.ion_geometry.contract_geometry.volumes * ion_p).tolist())
    target = d.reference_inventory_m1
    inv_rel_err = abs(inventory - target) / max(abs(target), 1.0e-30)
    max_site_frac = float(np.max(ion_p / d.positive_ion_site_limit_m3))

    return {
        "original_residual": raw_res,
        "poisson_normalized_max": poisson_norm,
        "contact_log_max": float(np.max(np.abs(contact_logs), initial=0.0)),
        "carrier_balance_bound_A_m2": carrier_bound,
        "ion_balance_max_m1_s": float(np.max(np.abs(ion_rows))),
        "ion_flux_max_m2_s": float(np.max(np.abs(ion_eval.flux), initial=0.0)),
        "domain_inventory_m1": inventory,
        "reference_inventory_m1": target,
        "inventory_relative_error": inv_rel_err,
        "max_site_fraction": max_site_frac,
        "minimum_electron_density_m3": float(np.min(n)),
        "minimum_hole_density_m3": float(np.min(p)),
        "minimum_ion_density_m3": float(np.min(ion_p)),
        "finite_positive_fields": bool(
            np.all(np.isfinite(phi))
            and np.all(n > 0.0)
            and np.all(p > 0.0)
            and np.all(ion_p > 0.0)
            and np.all(ion_p < d.positive_ion_site_limit_m3)
        ),
    }


def checks(values: dict[str, Any], options: IonicSteady2DOptions) -> tuple[GateOutcome, ...]:
    c = options.electronic
    limits = {
        "poisson_normalized_max": c.poisson_tolerance,
        "contact_log_max": c.contact_log_tolerance,
        "carrier_balance_bound_A_m2": c.continuity_tolerance_A_m2,
        "ion_balance_max_m1_s": options.ion_balance_tolerance_m1_s,
        "ion_flux_max_m2_s": options.ion_flux_tolerance_m2_s,
        "inventory_relative_error": options.inventory_relative_tolerance,
    }
    outcomes = [
        GateOutcome(name, math.isfinite(values[name]) and values[name] <= limit, float(values[name]), limit)
        for name, limit in limits.items()
    ]
    ok_fields = bool(values["finite_positive_fields"])
    outcomes.append(GateOutcome("finite_positive_fields", ok_fields, 0.0 if ok_fields else math.inf, 0.0))
    return tuple(outcomes)


def solve_ionic_steady_state_2d(
    device: CompiledIonicDevice2D,
    *,
    voltage_V: float,
    generation_scale: float,
    options: IonicSteady2DOptions | None = None,
    initial_state: IonicState2D | None = None,
) -> IonicSteady2DResult:
    """Solve coupled 2D steady state with exact 2D ion inventory conservation."""
    started = time.monotonic()
    if not isinstance(device, CompiledIonicDevice2D):
        raise TypeError("CompiledIonicDevice2D is required")
    controls = IonicSteady2DOptions() if options is None else options
    if controls.electronic.convergence_policy != HISTORICAL_MAX_ROW_V1:
        from dataclasses import replace as dc_replace
        controls = dc_replace(
            controls,
            electronic=dc_replace(controls.electronic, convergence_policy=HISTORICAL_MAX_ROW_V1),
        )

    immobile = bool(np.all(device.positive_ion_diffusivity_faces_m2_s == 0.0))
    unconstrained: _Unconstrained2D
    if use_assembled() and not immobile:
        unconstrained = AssembledIonic2DProblem(device, voltage_V, generation_scale)
        constrained: Any = AssembledConstrainedIonic2DProblem(unconstrained, controls.inventory_row_node)
    else:
        unconstrained = Ionic2DProblem(device, voltage_V, generation_scale)
        constrained = ConstrainedIonic2DProblem(unconstrained, controls.inventory_row_node)

    comp = device.electronic_component
    if initial_state is None:
        u_elec = comp.extrude_coordinates(
            np.r_[comp.column.reference_potential_V, np.zeros(2 * comp.geometry.ny)]
        )
        phi = u_elec[:comp.node_count]
        bottom, top = comp.column.contact_potentials(voltage_V)
        phi[:comp.geometry.nx], phi[-comp.geometry.nx:] = bottom, top
        for offset, ref, mask in zip(
            (comp.node_count, 2 * comp.node_count),
            comp.reference_potentials(voltage_V),
            comp.fixed_carrier_mask,
        ):
            u_elec[offset:offset + comp.node_count][mask] = (ref[mask] - phi[mask]) / comp.thermal_voltage_V
        u0 = np.r_[u_elec, np.zeros(device.ion_node_count)]
    else:
        if not isinstance(initial_state, IonicState2D) or initial_state.device_sha256 != device.identity:
            raise ValueError("initial_state must belong to the same CompiledIonicDevice2D")
        u0 = np.array(initial_state.u, copy=True)
        # Rebase contact potentials & quasi-Fermi shifts if voltage changed
        if initial_state.voltage_V != voltage_V:
            n_elec = comp.node_count
            old_e = comp.column.carrier_reference_energies(initial_state.voltage_V)
            new_e = comp.column.carrier_reference_energies(voltage_V)
            for offset, old_v, new_v in zip((n_elec, 2 * n_elec), old_e, new_e):
                u0[offset:offset + n_elec] += comp.extrude((old_v - new_v) / comp.thermal_voltage_V)
            bottom, top = comp.column.contact_potentials(voltage_V)
            u0[:comp.geometry.nx] = bottom
            u0[n_elec - comp.geometry.nx:n_elec] = top

    def gate_fn(candidate_u: np.ndarray) -> GateOutcome:
        if time.monotonic() - started > controls.wall_s:
            return GateOutcome("wall_budget", False, time.monotonic() - started, controls.wall_s)
        try:
            diag = diagnostics(unconstrained, candidate_u, controls)
            gate_list = checks(diag, controls)
            ok = all(g.passed for g in gate_list)
            worst = max((g.value / max(g.limit, 1e-30) for g in gate_list if g.limit > 0), default=0.0)
            return GateOutcome("ionic_steady_2d_gates", ok, worst, 1.0)
        except (ValueError, FloatingPointError):
            return GateOutcome("ionic_steady_2d_gates", False, math.inf, 1.0)

    settings = _newton_settings(device, controls, gate_fn)
    problem = CallableProblem(constrained.residual, constrained.jacobian)
    pins = admitted_source_pins()
    try:
        newton = solve_newton(problem, u0, settings)
    except Exception as exc:
        return IonicSteady2DResult(
            state=None,
            accepted=False,
            termination="exception",
            newton=None,
            initial_coordinates=u0,
            candidate_coordinates=u0,
            diagnostics=None,
            checks=(),
            compiled_identity=device.identity,
            operating_voltage_V=voltage_V,
            generation_scale=generation_scale,
            solver_controls={"options": asdict(controls), "initial_sha256": state_digest(u0)},
            source_sha256=pins,
            wall_s=time.monotonic() - started,
            error=str(exc),
        )

    cand = newton.state
    try:
        diag = diagnostics(unconstrained, cand, controls)
        gate_outcomes = checks(diag, controls)
    except (ValueError, FloatingPointError) as exc:
        return IonicSteady2DResult(
            state=None,
            accepted=False,
            termination=newton.termination,
            newton=newton,
            initial_coordinates=u0,
            candidate_coordinates=cand,
            diagnostics=None,
            checks=(),
            compiled_identity=device.identity,
            operating_voltage_V=voltage_V,
            generation_scale=generation_scale,
            solver_controls={"options": asdict(controls), "initial_sha256": state_digest(u0)},
            source_sha256=pins,
            wall_s=time.monotonic() - started,
            error=str(exc),
        )

    accepted = bool(newton.accepted and all(g.passed for g in gate_outcomes))
    state = device.state(cand, voltage_V=voltage_V) if accepted else None
    return IonicSteady2DResult(
        state=state,
        accepted=accepted,
        termination=newton.termination if accepted else (newton.termination if not newton.accepted else "gate_rejected"),
        newton=newton,
        initial_coordinates=u0,
        candidate_coordinates=cand,
        diagnostics=diag,
        checks=gate_outcomes,
        compiled_identity=device.identity,
        operating_voltage_V=voltage_V,
        generation_scale=generation_scale,
        solver_controls={"options": asdict(controls), "initial_sha256": state_digest(u0)},
        source_sha256=pins,
        wall_s=time.monotonic() - started,
        error=None if accepted else "ionic steady 2D did not satisfy all acceptance gates",
    )
