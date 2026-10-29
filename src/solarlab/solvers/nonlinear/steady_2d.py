"""Bounded unified Newton for genuine connected classical 2D DC.

Integrated Gauss/carrier rows are C/m and particles/(m*s), per unit depth.
Current checks divide A/m by the actual lateral width, retaining the existing
1D physical A/m2 tolerances. Fixed stack contacts retain unweighted V/log rows.
No column solve, seed solver, continuation or legacy fallback is hidden here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from solarlab.contracts import frozen_array
from solarlab.contracts.units import ELEMENTARY_CHARGE_C as Q
from solarlab.equations.dc_2d import CompiledDevice2D, DC2DProblem, State2D
from solarlab.mesh.geometry_2d import RECTANGULAR_HALF_ENDPOINT_2D_V1
from solarlab.solvers.nonlinear.newton import NewtonResult, NewtonSettings, solve_newton
from solarlab.solvers.nonlinear.problem import CallableProblem, GateOutcome, ResidualScaling, StepControl, state_digest
from solarlab.solvers.nonlinear.production import HISTORICAL_MAX_ROW_V1, POISSON_CONTACT_BALANCE_V1
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions

__all__ = ["Steady2DOptions", "Steady2DResult", "solve_steady_state_2d"]


@dataclass(frozen=True)
class Steady2DOptions:
    electronic: SteadyOptions = field(default_factory=SteadyOptions)
    wall_s: float = 30.0

    def __post_init__(self) -> None:
        if not isinstance(self.electronic, SteadyOptions):
            raise ValueError("explicit existing SteadyOptions are required")
        if not math.isfinite(self.wall_s) or self.wall_s <= 0.0:
            raise ValueError("2D owned wall budget must be finite and positive")


def _scope(device: CompiledDevice2D) -> None:
    if not isinstance(device, CompiledDevice2D):
        raise ValueError("an actual fully compiled classical 2D device is required")
    if device.geometry.name != RECTANGULAR_HALF_ENDPOINT_2D_V1:
        raise ValueError("2D steady requires physical half-endpoint Neumann geometry")
    if (device.column.field_transport is not None or device.column.recombination.explicit_defects is not None
            or device.column.additional_recombination is not None
            or any(channel.kind != "fixed_reservoir" for channel in device.column.channels)):
        raise ValueError("2D steady supports constant mobility, lifetime SRH and fixed stack reservoirs")


def _poisson_weights(device: CompiledDevice2D) -> np.ndarray:
    """Inverse physical C/m scale, with unweighted contact voltage rows."""
    g = device.geometry.contract_geometry
    reference_V = max(abs(device.column.built_in_potential_V), device.thermal_voltage_V)
    density = device.extrude(np.maximum(np.maximum(device.column.parameters.acceptor_density_m3,
        device.column.parameters.donor_density_m3), 1.0))
    capacitance = np.zeros(device.node_count)
    face = device.face_capacitance_F_m2 * g.face_measures
    np.add.at(capacitance, g.pairs[:, 0], face)
    np.add.at(capacitance, g.pairs[:, 1], face)
    scale = Q * density * g.volumes + reference_V * capacitance
    result = 1.0 / scale
    result[device.geometry.contact_nodes] = 1.0 / reference_V
    return result


def _settings(device: CompiledDevice2D, options: Steady2DOptions, gate: Any) -> tuple[NewtonSettings, Any]:
    c, N = options.electronic, device.node_count
    weights = np.full(3 * N, 2 * N * Q / (device.geometry.lateral_length_m * c.continuity_tolerance_A_m2))
    weights[:N] = _poisson_weights(device) / c.poisson_tolerance
    contacts = device.geometry.contact_nodes
    for offset in (N, 2 * N):
        weights[offset + contacts] = 1.0 / c.contact_log_tolerance
    steps = np.r_[np.full(N, c.potential_step_V), np.full(2 * N, c.quasi_fermi_step)]
    settings = NewtonSettings(ResidualScaling(weights), 1.0, StepControl(steps), c.budget(), (gate,), False)
    norm = None
    if c.convergence_policy == POISSON_CONTACT_BALANCE_V1:
        fixed: np.ndarray = np.zeros(3 * N, dtype=bool)
        fixed[:N] = True
        fixed[N + contacts] = fixed[2 * N + contacts] = True
        def balance_norm(scaled: np.ndarray) -> float:
            if scaled.shape != (3 * N,) or not np.all(np.isfinite(scaled)):
                return math.inf
            return max(float(np.max(abs(scaled[fixed]))), float(np.sum(abs(scaled[~fixed]))) / (2 * N))
        norm = balance_norm
    elif c.convergence_policy != HISTORICAL_MAX_ROW_V1:
        raise ValueError("unsupported 2D convergence policy")
    return settings, norm


def _sources() -> dict[str, str]:
    names = (__name__, "solarlab.equations.dc_2d", "solarlab.equations.dc_1d", "solarlab.mesh.geometry_2d",
        "solarlab.physics.transport.electrochemical_flux", "solarlab.physics.recombination.reaction",
        "solarlab.contracts.terms", "solarlab.solvers.nonlinear.newton", "solarlab.solvers.nonlinear.problem")
    result = {}
    for name in names:
        module = sys.modules.get(name)
        filename = None if module is None else getattr(module, "__file__", None)
        if filename is None:
            raise ValueError("2D steady source must be bound to actual source files: " + name)
        result[name] = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    return result


def _diagnostics(problem: DC2DProblem, u: np.ndarray, options: Steady2DOptions) -> dict[str, Any]:
    d, g, N = problem.device, problem.device.geometry, problem.device.node_count
    residual = problem.residual(u)
    phi, n, p = d.fields(u, voltage_V=problem.voltage_V)
    if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(n)) or not np.all(np.isfinite(p)):
        raise ValueError("2D diagnostic fields/residual are nonfinite")
    fluxes = problem.fluxes(u)
    source = (problem.generation_scale * d.generation_source_m3_s
              - problem.recombination_rate(n, p, derivatives=False).rate) * g.contract_geometry.volumes
    contacts = g.contact_nodes
    carrier_rows = residual[N:].reshape(2, N).copy()
    contact_logs = carrier_rows[:, contacts].copy()
    carrier_rows[:, contacts] = 0.0
    terminal_particles = np.stack(problem.contact_particle_fluxes(u))
    rates = np.array([[math.fsum((values * g.x_widths_m).tolist()) for values in carrier]
                      for carrier in terminal_particles])
    net_source = math.fsum(source.tolist())
    source_balance = net_source + rates[:, 0] - rates[:, 1]
    sections, terminals = problem.cross_section_current_A_m(u), problem.terminal_current_A_m(u)
    integrated = np.r_[terminals[:1], sections, terminals[1:]]
    current = problem.current_density(u)
    physical = np.tile(source, (2, 1))
    for row, flux, contact in zip(physical, (fluxes.electron.flux, fluxes.hole.flux), terminal_particles):
        weighted = flux * g.contract_geometry.face_measures
        np.add.at(row, g.contract_geometry.pairs[:, 0], -weighted)
        np.add.at(row, g.contract_geometry.pairs[:, 1], weighted)
        row[:g.nx] += contact[0] * g.x_widths_m
        row[-g.nx:] -= contact[1] * g.x_widths_m
    controls, width = options.electronic, g.lateral_length_m
    normalized_phi = residual[:N] * _poisson_weights(d)
    settings, norm = _settings(d, options, lambda values: GateOutcome("unused", True, 0.0, 1.0))
    scaled = settings.scaling.scaled(residual)
    row_norm = float(np.max(abs(scaled))) if norm is None else norm(scaled)
    return {"original_residual": residual, "scaled_residual": scaled, "original_rows": tuple(range(3 * N)),
        "convergence_norm": row_norm, "poisson_normalized_max": float(np.max(abs(normalized_phi))),
        "contact_log_max": float(np.max(abs(contact_logs))),
        "carrier_balance_bound_A_m2": Q * float(np.sum(abs(carrier_rows))) / width,
        "physical_carrier_balance_per_depth_m_s": physical,
        "source_terminal_particle_balance_per_depth_m_s": source_balance,
        "source_terminal_current_bound_A_m2": Q * float(np.sum(abs(source_balance))) / width,
        "net_pair_source_per_depth_m_s": net_source, "terminal_particle_rates_per_depth_m_s": rates,
        "terminal_particle_flux_density_m2_s": terminal_particles,
        "terminal_current_per_depth_A_m": terminals, "cross_section_current_per_depth_A_m": sections,
        "terminal_mean_current_density_A_m2": terminals / width,
        "cross_section_terminal_spread_A_m2": float(np.ptp(integrated)) / width,
        "face_current_density_A_m2": current,
        "lateral_current_density_A_m2": current[:g.x_face_count].reshape(g.ny, g.nx - 1),
        "stack_current_density_A_m2": current[g.x_face_count:].reshape(g.ny - 1, g.nx),
        "carrier_face_particle_flux_m2_s": np.stack((fluxes.electron.flux, fluxes.hole.flux)),
        "face_displacement_C_m2": problem.displacement(u),
        "lateral_boundary_particle_flux_m2_s": np.zeros((2, g.ny, 2)),
        "lateral_boundary_displacement_C_m2": np.zeros((g.ny, 2)),
        "finite_positive_fields": bool(np.all(n > 0.0) and np.all(p > 0.0) and np.all(np.isfinite(phi))),
        "minimum_electron_density_m3": float(np.min(n)), "minimum_hole_density_m3": float(np.min(p)),
        "generation_current_limit_per_depth_A_m": d.generation_limit_A_m * problem.generation_scale,
        "physical_limits": {"poisson": controls.poisson_tolerance, "contact_log": controls.contact_log_tolerance,
            "continuity_A_m2": controls.continuity_tolerance_A_m2, "current_spread_A_m2": controls.current_spread_tolerance_A_m2}}


def _checks(values: dict[str, Any], options: Steady2DOptions) -> tuple[GateOutcome, ...]:
    c = options.electronic
    limits = {"convergence_norm": 1.0, "poisson_normalized_max": c.poisson_tolerance,
        "contact_log_max": c.contact_log_tolerance, "carrier_balance_bound_A_m2": c.continuity_tolerance_A_m2,
        "source_terminal_current_bound_A_m2": c.continuity_tolerance_A_m2,
        "cross_section_terminal_spread_A_m2": c.current_spread_tolerance_A_m2}
    result = [GateOutcome(name, math.isfinite(values[name]) and values[name] <= limit, values[name], limit,
        "original numerical tolerances, physical2D per-depth mapping; qualification false") for name, limit in limits.items()]
    result.append(GateOutcome("finite_positive_fields", values["finite_positive_fields"],
        0.0 if values["finite_positive_fields"] else math.inf, 0.0))
    return tuple(result)


@dataclass(frozen=True)
class Steady2DResult:
    state: State2D | None
    accepted: bool
    termination: str
    newton: NewtonResult | None
    initial_coordinates: np.ndarray
    candidate_coordinates: np.ndarray
    initial_provenance: dict[str, Any]
    initial_diagnostics: dict[str, Any] | None
    diagnostics: dict[str, Any] | None
    checks: tuple[GateOutcome, ...]
    compiled_identity: str
    operating_voltage_V: float
    generation_scale: float
    solver_controls: dict[str, Any]
    source_sha256: dict[str, str]
    evaluation_counts: dict[str, int]
    wall_s: float
    error: str | None = None
    physical_certificate_status: str = "not_evaluated"

    def __post_init__(self) -> None:
        object.__setattr__(self, "initial_coordinates", frozen_array(self.initial_coordinates))
        object.__setattr__(self, "candidate_coordinates", frozen_array(self.candidate_coordinates))


def solve_steady_state_2d(device: CompiledDevice2D, *, voltage_V: float, generation_scale: float,
                          options: Steady2DOptions | None = None, initial_state: State2D | None = None,
                          initial_provenance: dict[str, Any] | None = None) -> Steady2DResult:
    """One full2D Newton, preserving its first failure and every original row."""
    started = time.monotonic()
    _scope(device)
    controls = Steady2DOptions() if options is None else options
    if not isinstance(controls, Steady2DOptions):
        raise ValueError("Steady2DOptions are required")
    problem = DC2DProblem(device, voltage_V, generation_scale)
    if initial_state is None:
        u = device.extrude_coordinates(np.r_[device.column.reference_potential_V, np.zeros(2 * device.geometry.ny)])
        phi = u[:device.node_count]
        bottom, top = device.column.contact_potentials(voltage_V)
        phi[:device.geometry.nx], phi[-device.geometry.nx:] = bottom, top
        for offset, ref in zip((device.node_count, 2 * device.node_count), device.reference_potentials(voltage_V)):
            contact = device.geometry.contact_nodes
            u[offset + contact] = (ref[contact] - phi[contact]) / device.thermal_voltage_V
        provenance = {"kind": "compiler physical reference; no seed solve"}
    else:
        if not isinstance(initial_state, State2D) or initial_state.device_sha256 != device.identity or initial_state.voltage_V != voltage_V:
            raise ValueError("initial2D state must have the same device identity and voltage")
        expected = device.state(initial_state.u, voltage_V=voltage_V)
        if any(not np.array_equal(getattr(initial_state, name), getattr(expected, name))
               for name in ("potential_V", "electron_density_m3", "hole_density_m3")):
            raise ValueError("initial physical2D fields disagree with their coordinates")
        u = np.array(initial_state.u, copy=True)
        provenance = {"kind": "provided full2D numerical guess"}
    provenance.update({} if initial_provenance is None else initial_provenance)
    provenance["state_sha256"] = state_digest(u)
    pins = _sources()
    deadline = started + controls.wall_s
    candidate = u.copy()
    counts = {"residual": 0, "jacobian": 0, "diagnostics": 0}
    newton_calls = 0
    def checkpoint() -> None:
        if time.monotonic() >= deadline:
            raise TimeoutError("2D steady owned wall budget exhausted; no retry")
        if _sources() != pins:
            raise RuntimeError("2D steady physical source changed during the solve")
    def evaluate(kind: str, values: np.ndarray) -> Any:
        nonlocal candidate
        candidate = np.array(values, copy=True)
        checkpoint()
        counts[kind] += 1
        output = problem.residual(values) if kind == "residual" else problem.jacobian(values)
        checkpoint()
        return output
    def gate(values: np.ndarray) -> GateOutcome:
        checkpoint()
        counts["diagnostics"] += 1
        checks = _checks(_diagnostics(problem, values, controls), controls)
        checkpoint()
        return GateOutcome("all_original_2d_physical_checks", all(check.passed for check in checks),
            max(check.value / check.limit if check.limit > 0.0 else check.value for check in checks), 1.0)
    settings, norm = _settings(device, controls, gate)
    state, newton, diagnostics, initial_data, error = None, None, None, None, None
    checks: tuple[GateOutcome, ...] = ()
    try:
        checkpoint()
        counts["diagnostics"] += 1
        initial_data = _diagnostics(problem, u, controls)
        checkpoint()
        newton_calls += 1
        newton = solve_newton(CallableProblem(lambda values: evaluate("residual", values),
            lambda values: evaluate("jacobian", values)), u, settings, convergence_norm=norm)
        candidate = newton.state
        checkpoint()
        counts["diagnostics"] += 1
        diagnostics = _diagnostics(problem, candidate, controls)
        checks = _checks(diagnostics, controls)
        state = device.state(candidate, voltage_V=voltage_V)
        checkpoint()
        accepted = newton.accepted and all(check.passed for check in checks)
        termination = newton.termination.value if accepted or not newton.accepted else "physical_checks_rejected"
    except Exception as exc:
        accepted = False
        termination = "wall_budget_exhausted" if isinstance(exc, TimeoutError) else "evaluation_error"
        error = f"{type(exc).__name__}: {str(exc)[:600]}"
    recorded = {"numerical": asdict(controls), "row_units": {"gauss": "C/m depth", "carrier": "particles/(m depth*s)",
        "contact_phi": "V", "contact_carrier": "dimensionless log"},
        "current_gate_normalization": "raw A/m divided by actual lateral width[m] gives A/m2",
        "lateral_boundary": "zero normal displacement and particle flux", "full_2d_newton_calls": newton_calls,
        "evaluation_count_semantics": "residual/jacobian count Newton callbacks; diagnostics count separate initial/gate/final physical rechecks",
        "hidden_seed_or_fallback_calls": 0}
    return Steady2DResult(state, accepted, termination, newton, u, candidate, provenance, initial_data,
        diagnostics, checks, device.identity, voltage_V, generation_scale, recorded, pins, counts,
        time.monotonic() - started, error)
