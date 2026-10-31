"""BDF experiment admission and history orchestration; unchanged solver kernels."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import math
import time
from typing import Any, cast

import numpy as np

from solarlab.contracts import frozen_array
from solarlab.equations.assembly.classical_1d import AssembledDC1DProblem
from solarlab.equations.assembly.ionic_1d import AssembledIonic1DProblem, AssembledMultipleIonic1DProblem
from solarlab.equations.dc_1d import CompiledDevice1D, DC1DProblem
from solarlab.equations.implementation import use_assembled
from solarlab.equations.ionic_1d import CompiledIonicDevice1D, Ionic1DProblem
from solarlab.equations.multiple_ionic_1d import CompiledMultipleIonicDevice1D
from solarlab.experiments import ionic_transient, multiple_ionic_transient
from solarlab.experiments.ionic_transient import (
    IonicTransientInitial, IonicTransientOptions, _admit_evidence, _finite_accounting, _newton_settings, _same_state,
)
from solarlab.experiments.jv.operating_points import _verify_point
from solarlab.experiments.multiple_ionic_transient import MultipleIonicTransientInitial
from solarlab.experiments.transient import TransientOptions, TransientProtocol
from solarlab.solvers.nonlinear.newton import NewtonSettings
from solarlab.solvers.nonlinear.problem import GateOutcome, Vector
from solarlab.solvers.nonlinear.production import (
    production_convergence_norm, production_coordinates, production_residual_scaling,
)
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions, SteadyState1D
from solarlab.solvers.time.assembled_ionic_bdf import (
    IonicDevice, IonicState, assembled_ionic_bdf_accounting, assembled_ionic_bdf_seed,
)
from solarlab.solvers.time.assembled_storage import AssembledIonicStorage1D
from solarlab.solvers.time.bdf import (
    Assessment, BDFEvent, BDFOptions, BDFResult, BDFStep, HistoryPoint, StorageProblem, solve_bdf,
)
from solarlab.solvers.time.bdf_1d import BDFDrive, ElectronicStorage1D, IonicStorage1D, bdf_accounting, bdf_seed
from solarlab.solvers.time.carrier_seed import DensitySeedContext

__all__ = ["solve_electronic_bdf", "solve_ionic_bdf", "solve_assembled_ionic_bdf", "solve_assembled_multiple_ionic_bdf"]



def _events(drive: BDFDrive, factory: Any, controls: TransientOptions) -> tuple[BDFEvent, ...]:
    def restart(old: StorageProblem, new: StorageProblem, u: Vector) -> tuple[Vector, Assessment]:
        assert isinstance(old, (ElectronicStorage1D, IonicStorage1D))
        assert isinstance(new, type(old))
        if isinstance(new, IonicStorage1D) and isinstance(old, IonicStorage1D):
            e = new.problem.device.electronic_component
        elif isinstance(new, ElectronicStorage1D) and isinstance(old, ElectronicStorage1D):
            e = new.problem.device
        else:
            raise ValueError("restart must retain the same storage adapter kind")
        right = bdf_seed(BDFStep(new, (HistoryPoint(0.0, u, old.storage(u), old),), 1.0, 1),
                         "previous", voltage_lift=True)
        tangent = new.storage_tangent(right)
        algebraic = np.asarray(abs(tangent).sum(axis=1)).ravel() == 0
        weights = production_residual_scaling(e, poisson_tolerance=controls.poisson_tolerance,
            continuity_tolerance_A_m2=controls.continuity_tolerance_A_m2,
            contact_log_tolerance=controls.contact_log_tolerance).row_weights
        weights = np.r_[weights, np.ones(len(u) - len(weights))]
        algebraic_norm = float(np.max(abs(new.residual(right)[algebraic] * weights[algebraic]), initial=0.0))
        if isinstance(new, ElectronicStorage1D) and isinstance(old, ElectronicStorage1D):
            delta = new.physical_increment(right, old, u)
        elif isinstance(new, IonicStorage1D) and isinstance(old, IonicStorage1D):
            delta = new.physical_increment(right, old, u)
        else:
            raise ValueError("physical inventory history must keep the same adapter kind")
        relative = float(np.max(abs(delta) / np.maximum(abs(old.physical_storage(u)), 1.0)))
        phi_change = right[:e.node_count] - u[:e.node_count]
        impulse = -e.poisson.face_capacitance_F_m2 * np.diff(phi_change)
        assessment = Assessment({"left_voltage_V": old.problem.voltage_V, "right_voltage_V": new.problem.voltage_V,
            "physical_inventory_change": delta, "relative_inventory_change": relative,
            "displacement_charge_impulse_C_m2": impulse, "terminal_charge_impulse_C_m2": impulse[[0, -1]]},
            (GateOutcome("restart_algebraic_rows", algebraic_norm <= 1, algebraic_norm, 1.0),
             GateOutcome("restart_frozen_physical_inventory", relative <= controls.contact_log_tolerance,
                         relative, controls.contact_log_tolerance)))
        return right, assessment
    events = [BDFEvent(float(t), restart, "voltage_generation_protocol_jump") for t, a, b in
              zip(drive.times_s, drive.left_values, drive.right_values) if not np.array_equal(a, b)]
    for t in drive.slope_breakpoints_s:
        index = int(np.flatnonzero(drive.times_s == t)[0])
        left_slope = ((drive.left_values[index] - drive.right_values[index - 1])
                      / (drive.times_s[index] - drive.times_s[index - 1]))
        right_slope = ((drive.left_values[index + 1] - drive.right_values[index])
                       / (drive.times_s[index + 1] - drive.times_s[index]))
        def slope_restart(old: StorageProblem, new: StorageProblem, u: Vector,
                          before: Vector = left_slope, after: Vector = right_slope) -> tuple[Vector, Assessment]:
            right, assessed = restart(old, new, u)
            if right.tobytes() != u.tobytes():
                raise ValueError("a continuous slope knot must preserve the represented state")
            return right, Assessment({**assessed.accounting, "left_drive_slope": before,
                                      "right_drive_slope": after}, assessed.checks)
        events.append(BDFEvent(t, slope_restart, "drive_slope_discontinuity"))
    return tuple(sorted(events, key=lambda event: event.time_s))


def _adapter_step_budget(
    protocol: TransientProtocol, max_sample_intervals: int, bdf_options: BDFOptions,
    adaptive_max_steps: int | None,
) -> int:
    """Keep legacy clipping unless an independent adaptive work cap is explicit.

    The explicit route retains the production limit on requested sample
    intervals. Its accepted-step work cap is independent and still bounded
    by BDFOptions.max_steps; no physical or nonlinear controls change.
    """
    if adaptive_max_steps is None:
        return min(bdf_options.max_steps, max_sample_intervals)
    if type(adaptive_max_steps) is not int or adaptive_max_steps < 1:
        raise ValueError("adaptive_max_steps must be an explicit positive integer")
    if len(protocol.times_s) - 1 > max_sample_intervals:
        raise ValueError("requested timepoints exceed max_steps sample admission")
    return min(bdf_options.max_steps, adaptive_max_steps)


def _generation_density_context(drive: BDFDrive, step: BDFStep) -> DensitySeedContext | None:
    """Only a complete BDF1 target starting at a dark-to-light value jump."""
    if step.order != 1 or len(step.history) != 1:
        return None
    old = step.history[0]
    if not isinstance(step.endpoint, ElectronicStorage1D) or not isinstance(old.problem, ElectronicStorage1D):
        return None
    exact = np.flatnonzero(drive.times_s == old.time_s)
    if exact.size != 1:
        return None
    index = int(exact[0])
    left, right = drive.left_values[index], drive.right_values[index]
    if left[1] != 0.0 or right[1] <= 0.0 or left[0] != right[0]:
        return None
    endpoint, previous = step.endpoint.problem, old.problem.problem
    if (endpoint.voltage_V != right[0] or previous.voltage_V != right[0]
            or endpoint.generation_scale != right[1] or previous.generation_scale != right[1]):
        return None
    device = endpoint.device

    def validate() -> None:
        if (type(device) is not CompiledDevice1D or previous.device.identity != device.identity
                or old.u.shape != (3 * device.node_count,)
                or not {"classical_1d_dc", "maxwell_boltzmann", "ion_free"} <= set(device.capabilities)):
            raise ValueError("density activation requires the matching compiled electronic log-coordinate target")
        if (step.dt_s <= 0 or not math.isfinite(step.dt_s)
                or not np.array_equal(step.coefficients, np.array([1.0, -1.0]))):
            raise ValueError("density activation requires a complete positive-interval BDF1 target")
        if endpoint.device.identity != device.identity or endpoint.voltage_V != right[0]:
            raise ValueError("density activation device/reference changed")

    def fingerprint() -> str:
        digest = hashlib.sha256()
        for point in step.history:
            assert isinstance(point.problem, ElectronicStorage1D)
            digest.update(point.time_s.hex().encode())
            digest.update(point.u.tobytes())
            digest.update(point.storage_value.tobytes())
            digest.update(point.problem.problem.device.identity.encode())
            digest.update(float(point.problem.problem.voltage_V).hex().encode())
            digest.update(float(point.problem.problem.generation_scale).hex().encode())
        return digest.hexdigest()

    return DensitySeedContext(device, endpoint.voltage_V, {
        "policy": "dark_generation_stagnation_density_v1", "device_identity": device.identity,
        "coordinate_form": "electrode_quasi_fermi_1d_v1", "source_knot_s": old.time_s,
        "left_source": frozen_array(left), "right_source": frozen_array(right),
        "voltage_reference_V": endpoint.voltage_V, "complete_target": "BDFStep order1",
    }, validate, fingerprint)


def solve_electronic_bdf(
    compiled: CompiledDevice1D, initial: SteadyState1D, protocol: TransientProtocol, *,
    options: TransientOptions, initial_options: SteadyOptions, bdf_options: BDFOptions,
    drive: BDFDrive | None = None, adaptive_max_steps: int | None = None,
    density_seed_activation: bool = False,
) -> BDFResult:
    if type(density_seed_activation) is not bool:
        raise ValueError("density_seed_activation must be an explicit boolean")
    started = time.monotonic()
    work_steps = _adapter_step_budget(protocol, options.max_steps, bdf_options, adaptive_max_steps)
    _verify_point(compiled, initial, initial.generation_scale, initial_options)
    source = BDFDrive.from_protocol(protocol) if drive is None else drive
    if source.values(float(protocol.times_s[0]), "left") != (initial.voltage_V, initial.generation_scale):
        raise ValueError("BDF initial drive must match the admitted steady state")
    base = NewtonSettings(production_residual_scaling(compiled, poisson_tolerance=options.poisson_tolerance,
        continuity_tolerance_A_m2=options.continuity_tolerance_A_m2, contact_log_tolerance=options.contact_log_tolerance),
        1.0, production_coordinates(compiled, potential_step_V=options.potential_step_V,
        quasi_fermi_step=options.quasi_fermi_step), options.newton_budget, (), False)
    norm = production_convergence_norm(compiled, convergence_policy=options.convergence_policy)
    bound = AssembledDC1DProblem(compiled, initial.voltage_V, initial.generation_scale) if use_assembled() else None
    def factory(t: float, side: str) -> ElectronicStorage1D:
        voltage, generation = source.values(t, side)
        return ElectronicStorage1D(bound.at(voltage, generation) if bound is not None
                                   else DC1DProblem(compiled, voltage, generation))
    def assessment(step: BDFStep, u: Vector) -> Assessment:
        data = bdf_accounting(step, u)
        finite = _finite_accounting(data)
        return Assessment(data, (GateOutcome("finite_original_accounting", finite, 0.0 if finite else math.inf, 0.0),))
    preparation_s = time.monotonic() - started
    remaining = min(bdf_options.wall_s, options.wall_s) - preparation_s
    if remaining <= 0:
        raise TimeoutError("BDF initial admission exhausted the whole-history wall budget")
    result = solve_bdf(factory, initial.state.u, (float(protocol.times_s[0]), float(protocol.times_s[-1])),
        options=replace(bdf_options, wall_s=remaining, max_steps=work_steps), settings=lambda step: base,
        convergence_norm=norm, assess=assessment,
        seed=lambda step: bdf_seed(step, options.predictor, voltage_lift=True),
        density_seed_context=(lambda step, failed: _generation_density_context(source, step)) if density_seed_activation else None,
        breakpoints_s=tuple(float(t) for t in source.times_s[1:-1]),
        events=_events(source, factory, options), output_times_s=protocol.times_s)
    return replace(result, wall_s=time.monotonic() - started, controls={**result.controls, "physical_controls": asdict(options),
        "initial_controls": asdict(initial_options), "compiled_identity": compiled.identity,
        "drive": asdict(source), "initial_admission_wall_s": preparation_s, "method": "classical_physical_storage_bdf12",
        **({"adapter_adaptive_max_steps": adaptive_max_steps, "adapter_sample_interval_limit": options.max_steps}
           if adaptive_max_steps is not None else {})})


def solve_ionic_bdf(
    compiled: CompiledIonicDevice1D, initial: IonicTransientInitial, protocol: TransientProtocol, *,
    options: IonicTransientOptions, bdf_options: BDFOptions, drive: BDFDrive | None = None,
    adaptive_max_steps: int | None = None,
) -> BDFResult:
    started = time.monotonic()
    work_steps = _adapter_step_budget(protocol, options.electronic.max_steps, bdf_options, adaptive_max_steps)
    admitted, generation, checks = _admit_evidence(compiled, initial.evidence_json, initial.source_sha256,
                                                  initial.steady_controls)
    if not _same_state(initial.state, admitted) or initial.generation_scale != generation:
        raise ValueError("BDF ionic initial object disagrees with its original pinned evidence")
    source = BDFDrive.from_protocol(protocol) if drive is None else drive
    if source.values(float(protocol.times_s[0]), "left") != (admitted.voltage_V, generation):
        raise ValueError("BDF initial drive must match the admitted ionic steady state")
    base, norm = _newton_settings(compiled, options)
    bound = AssembledIonic1DProblem(compiled, admitted.voltage_V, generation) if use_assembled() else None
    def factory(t: float, side: str) -> IonicStorage1D:
        voltage, scale = source.values(t, side)
        return IonicStorage1D(bound.at(voltage, scale) if bound is not None else Ionic1DProblem(compiled, voltage, scale))
    def assessment(step: BDFStep, u: Vector) -> Assessment:
        data = bdf_accounting(step, u)
        finite = _finite_accounting(data)
        residual_norm = norm(base.scaling.scaled(step.residual(u))) if finite else math.inf
        return Assessment(data, (GateOutcome("finite_original_accounting", finite, 0.0 if finite else math.inf, 0.0),
            GateOutcome("all_original_dynamic_rows", residual_norm <= 1, residual_norm, 1.0)))
    preparation_s = time.monotonic() - started
    remaining = min(bdf_options.wall_s, options.electronic.wall_s) - preparation_s
    if remaining <= 0:
        raise TimeoutError("BDF ionic admission exhausted the whole-history wall budget")
    result = solve_bdf(factory, admitted.u, (float(protocol.times_s[0]), float(protocol.times_s[-1])),
        options=replace(bdf_options, wall_s=remaining, max_steps=work_steps),
        settings=lambda step: base, convergence_norm=norm, assess=assessment,
        seed=lambda step: bdf_seed(step, options.electronic.predictor, voltage_lift=True),
        breakpoints_s=tuple(float(t) for t in source.times_s[1:-1]),
        events=_events(source, factory, options.electronic), output_times_s=protocol.times_s)
    return replace(result, wall_s=time.monotonic() - started, controls={**result.controls, "physical_controls": asdict(options),
        "initial_source_sha256": initial.source_sha256, "initial_checks": checks,
        "compiled_identity": compiled.identity, "drive": asdict(source), "initial_admission_wall_s": preparation_s,
        "method": "classical_physical_storage_bdf12",
        **({"adapter_adaptive_max_steps": adaptive_max_steps,
            "adapter_sample_interval_limit": options.electronic.max_steps}
           if adaptive_max_steps is not None else {})})


def _assembled_events(source: BDFDrive, settings: Any, factory: Any, controls: IonicTransientOptions) -> tuple[BDFEvent, ...]:
    def restart(old: Any, new: Any, u: Vector) -> tuple[Vector, Assessment]:
        right = assembled_ionic_bdf_seed(BDFStep(new, (HistoryPoint(0.0, u, old.storage(u), old),), 1.0, 1))
        tangent = new.storage_tangent(right)
        algebraic = np.asarray(abs(tangent).sum(axis=1)).ravel() == 0
        residual = new.residual(right)
        value = float(np.max(abs(residual[algebraic] * settings.scaling.row_weights[algebraic]), initial=0.0))
        change = new.physical_increment(right, old, u)
        relative = float(np.max(abs(change) / np.maximum(abs(old.physical_storage(u)), 1.0)))
        return right, Assessment({"physical_inventory_change": change, "relative_inventory_change": relative},
            (GateOutcome("restart_algebraic_rows", value <= 1, value, 1.0),
             GateOutcome("restart_frozen_physical_inventory", relative <= controls.electronic.contact_log_tolerance,
                         relative, controls.electronic.contact_log_tolerance)))
    return tuple(BDFEvent(float(t), restart, "voltage_generation_protocol_jump") for t, left, right in
                 zip(source.times_s, source.left_values, source.right_values) if not np.array_equal(left, right)) + tuple(
        BDFEvent(t, restart, "drive_slope_discontinuity") for t in source.slope_breakpoints_s)


def _solve_assembled(
    device: IonicDevice, admitted: IonicState, initial: IonicTransientInitial | MultipleIonicTransientInitial,
    protocol: TransientProtocol, options: IonicTransientOptions, bdf_options: BDFOptions,
    source: BDFDrive, adaptive_max_steps: int | None, checks: Any, started: float,
) -> BDFResult:
    if len(protocol.times_s) - 1 > options.electronic.max_steps:
        raise ValueError("requested timepoints exceed max_steps sample admission")
    if adaptive_max_steps is not None and (type(adaptive_max_steps) is not int or adaptive_max_steps < 1):
        raise ValueError("adaptive_max_steps must be an explicit positive integer")
    work = min(bdf_options.max_steps, options.electronic.max_steps if adaptive_max_steps is None else adaptive_max_steps)
    if source.values(float(protocol.times_s[0]), "left") != (admitted.voltage_V, initial.generation_scale):
        raise ValueError("BDF drive must begin at the original admitted ionic state")
    multiple = isinstance(device, CompiledMultipleIonicDevice1D)
    settings, norm = (multiple_ionic_transient._settings(cast(CompiledMultipleIonicDevice1D, device), options) if multiple
                      else ionic_transient._newton_settings(cast(CompiledIonicDevice1D, device), options))
    original_problem = (AssembledMultipleIonic1DProblem(device, admitted.voltage_V, initial.generation_scale)
                        if isinstance(device, CompiledMultipleIonicDevice1D)
                        else AssembledIonic1DProblem(device, admitted.voltage_V, initial.generation_scale))
    def factory(t: float, side: str) -> AssembledIonicStorage1D:
        voltage, generation = source.values(t, side)
        return AssembledIonicStorage1D(original_problem.at(voltage, generation))
    def assessment(step: BDFStep, u: Vector) -> Assessment:
        data = assembled_ionic_bdf_accounting(step, u)
        finite = ionic_transient._finite_accounting(data)
        value = norm(settings.scaling.row_weights * step.residual(u)) if finite else math.inf
        gates = [GateOutcome("finite_original_accounting", finite, 0.0 if finite else math.inf, 0.0),
                 GateOutcome("all_original_dynamic_rows", value <= 1, value, 1.0)]
        if multiple:
            for name, domain in data["domains"].items():
                for species, values in domain["species"].items():
                    value = float(values["balance_l1_m2_s"])
                    gates.append(GateOutcome(name + "." + species + ".all_native_balances",
                        value <= options.ion_balance_tolerance_m2_s, value, options.ion_balance_tolerance_m2_s))
            limit = initial.steady_controls.electronic.current_spread_tolerance_A_m2
            for key in ("maxwell_spread_A_m2", "maxwell_identity_error_A_m2"):
                value = float(np.max(abs(np.asarray(data[key]))))
                gates.append(GateOutcome(key, value <= limit, value, limit))
        return Assessment(data, tuple(gates))
    admission_wall = time.monotonic() - started
    remaining = min(options.electronic.wall_s, bdf_options.wall_s) - admission_wall
    if remaining <= 0:
        raise TimeoutError("BDF whole-history budget exhausted during original initial admission")
    result = solve_bdf(factory, admitted.u, (float(protocol.times_s[0]), float(protocol.times_s[-1])),
        options=replace(bdf_options, max_steps=work, wall_s=remaining), settings=lambda step: settings,
        convergence_norm=norm, assess=assessment, seed=assembled_ionic_bdf_seed,
        breakpoints_s=tuple(float(t) for t in source.times_s[1:-1]),
        events=_assembled_events(source, settings, factory, options), output_times_s=protocol.times_s)
    return replace(result, wall_s=time.monotonic() - started, controls={**result.controls,
        "method": "assembled_ionic_physical_storage_bdf12", "compiled_identity": device.identity,
        "physical_controls": asdict(options), "initial_source_sha256": initial.source_sha256,
        "initial_checks": checks, "drive": asdict(source), "initial_admission_wall_s": admission_wall,
        "adapter_adaptive_max_steps": adaptive_max_steps, "adapter_sample_interval_limit": options.electronic.max_steps})


def solve_assembled_ionic_bdf(
    device: CompiledIonicDevice1D, initial: IonicTransientInitial, protocol: TransientProtocol, *,
    options: IonicTransientOptions, bdf_options: BDFOptions, drive: BDFDrive | None = None,
    adaptive_max_steps: int | None = None,
) -> BDFResult:
    """Single-domain positive/dual/shared-site scope, with original admission."""
    started = time.monotonic()
    admitted, generation, checks = ionic_transient._admit_evidence(
        device, initial.evidence_json, initial.source_sha256, initial.steady_controls)
    if not ionic_transient._same_state(initial.state, admitted) or initial.generation_scale != generation:
        raise ValueError("ionic initial disagrees with the original pinned evidence")
    return _solve_assembled(device, admitted, initial, protocol, options, bdf_options,
                  BDFDrive.from_protocol(protocol) if drive is None else drive, adaptive_max_steps, checks, started)


def solve_assembled_multiple_ionic_bdf(
    device: CompiledMultipleIonicDevice1D, initial: MultipleIonicTransientInitial, protocol: TransientProtocol, *,
    options: IonicTransientOptions, bdf_options: BDFOptions, drive: BDFDrive | None = None,
    adaptive_max_steps: int | None = None,
) -> BDFResult:
    """Multiple-domain scope with independent domain/species balances and gates."""
    started = time.monotonic()
    admitted, generation, checks = multiple_ionic_transient._admit(
        device, initial.evidence_json, initial.source_sha256, initial.steady_controls)
    if not multiple_ionic_transient._same_record(
        multiple_ionic_transient._decode(multiple_ionic_transient._encode(admitted)),
        multiple_ionic_transient._decode(multiple_ionic_transient._encode(initial.state)),
    ) or initial.generation_scale != generation:
        raise ValueError("multiple-domain initial disagrees with the original pinned evidence")
    return _solve_assembled(device, admitted, initial, protocol, options, bdf_options,
                  BDFDrive.from_protocol(protocol) if drive is None else drive, adaptive_max_steps, checks, started)
