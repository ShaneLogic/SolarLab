"""Physical BDF on the canonical assembled single/multiple ionic BE storage.

Every species secant and tangent is supplied by AssembledIonicStorage1D.
D0 species retain their full physical mass, as in the original BE time rows;
only genuinely algebraic/constraint rows have zero equation storage. The
original BE observations supply fluxes, sources, charge and domain geometry.
No experiment selection or production default is changed by this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
import time
from typing import Any, TypeAlias, cast

import numpy as np

from solarlab.contracts.units import ELEMENTARY_CHARGE_C as Q
from solarlab.equations.assembly.ionic_1d import AssembledIonic1DProblem, AssembledMultipleIonic1DProblem
from solarlab.equations.dc_1d import DC1DProblem, State1D
from solarlab.equations.ionic_1d import CompiledIonicDevice1D, IonicState1D
from solarlab.equations.multiple_ionic_1d import CompiledMultipleIonicDevice1D, MultipleIonicState1D
from solarlab.experiments import ionic_transient, multiple_ionic_transient
from solarlab.experiments.ionic_transient import IonicTransientInitial, IonicTransientOptions
from solarlab.experiments.multiple_ionic_transient import MultipleIonicTransientInitial
from solarlab.experiments.transient import TransientProtocol
from solarlab.physics.electrostatics import contact_displacement
from solarlab.solvers.nonlinear.problem import GateOutcome, Vector
from solarlab.solvers.time.assembled_ionic_backward_euler import (
    AssembledIonicBackwardEuler1DStep, AssembledMultipleIonicBackwardEuler1DStep,
)
from solarlab.solvers.time.assembled_storage import AssembledIonicStorage1D
from solarlab.solvers.time.backward_euler import BackwardEuler1DStep
from solarlab.solvers.time.bdf import Assessment, BDFEvent, BDFOptions, BDFResult, BDFStep, HistoryPoint, solve_bdf
from solarlab.solvers.time.bdf_1d import BDFDrive
from solarlab.solvers.time.ionic_backward_euler import IonicBackwardEuler1DStep, ionic_step_accounting
from solarlab.solvers.time.multiple_ionic_backward_euler import MultipleIonicBackwardEuler1DStep, multiple_ionic_step_accounting

__all__ = ["assembled_ionic_bdf_accounting", "assembled_ionic_bdf_seed",
           "solve_assembled_ionic_bdf", "solve_assembled_multiple_ionic_bdf"]

IonicDevice: TypeAlias = CompiledIonicDevice1D | CompiledMultipleIonicDevice1D
IonicState: TypeAlias = IonicState1D | MultipleIonicState1D
IonicBEStep: TypeAlias = AssembledIonicBackwardEuler1DStep | AssembledMultipleIonicBackwardEuler1DStep


def _adapter(step: BDFStep) -> AssembledIonicStorage1D:
    if not isinstance(step.endpoint, AssembledIonicStorage1D):
        raise TypeError("canonical assembled ionic storage is required")
    return step.endpoint


def _be(step: BDFStep) -> IonicBEStep:
    endpoint = _adapter(step)
    old = step.history[0]
    if not isinstance(old.problem, AssembledIonicStorage1D):
        raise TypeError("ionic history must retain the canonical assembled storage")
    previous = old.problem.device.state(old.u, voltage_V=old.problem.voltage_V)
    if isinstance(endpoint.problem, AssembledIonic1DProblem):
        return AssembledIonicBackwardEuler1DStep(endpoint.problem, cast(IonicState1D, previous), step.dt_s)
    return AssembledMultipleIonicBackwardEuler1DStep(endpoint.problem, cast(MultipleIonicState1D, previous), step.dt_s)


def assembled_ionic_bdf_seed(step: BDFStep) -> Vector:
    """Reuse the original harmonic carrier lift and unchanged ion coordinates."""
    be = _be(step)
    return (be.initial_coordinates("previous", voltage_lift=True)
            if isinstance(be, AssembledIonicBackwardEuler1DStep) else be.initial_coordinates())


def _physical_rate(step: BDFStep, u: Vector) -> Vector:
    adapter = _adapter(step)
    old = step.history[0]
    rate = step.coefficients[0] * adapter.physical_increment(u, old.problem, old.u) / step.dt_s
    if step.order == 2:
        older = step.history[1]
        if not isinstance(old.problem, AssembledIonicStorage1D):
            raise TypeError("ionic history must retain its storage binding")
        rate -= step.coefficients[2] * old.problem.physical_increment(old.u, older.problem, older.u) / step.dt_s
    return rate


@dataclass(frozen=True)
class _CarrierRateStep(BackwardEuler1DStep):
    """Readout-only view with the complete BDF carrier inventory rate."""

    physical_rate: np.ndarray

    def inventory_rate_m2_s(self, u: np.ndarray) -> np.ndarray:
        return self.physical_rate


@dataclass(frozen=True)
class _AccountingStep:
    """Supply canonical BDF rates to the unchanged BE flux/charge readers."""

    step: BDFStep
    be: IonicBEStep
    rate: Vector

    @property
    def endpoint(self) -> AssembledIonic1DProblem | AssembledMultipleIonic1DProblem:
        return self.be.endpoint

    @property
    def previous(self) -> IonicState:
        return self.be.previous

    @property
    def dt_s(self) -> float:
        return self.step.dt_s

    @property
    def electronic_step(self) -> _CarrierRateStep:
        original = self.be.electronic_step
        n = self.endpoint.device.node_count
        return _CarrierRateStep(cast(DC1DProblem, original.endpoint), cast(State1D, original.previous),
                                self.dt_s, self.rate[n:3 * n].reshape(2, n))

    def inventory_rates_m2_s(self, u: Vector) -> Any:
        device = self.endpoint.device
        n = device.node_count
        if isinstance(device, CompiledIonicDevice1D):
            ions: Any = tuple(self.rate[block] for block in device.ion_species_slices)
        else:
            ions = {name: {species: self.rate[block] for species, block in blocks.items()}
                    for name, blocks in device.ion_species_slices.items()}
        return self.rate[n:3 * n].reshape(2, n), ions

    def residual(self, u: Vector) -> Vector:
        return self.step.residual(u)


def _domains(device: IonicDevice) -> tuple[tuple[CompiledIonicDevice1D, dict[str, slice]], ...]:
    if isinstance(device, CompiledIonicDevice1D):
        return ((device, dict(zip(device.ion_species_names, device.ion_species_slices))),)
    return tuple((domain, dict(device.ion_species_slices[domain.domain_layer_id])) for domain in device.domains)


def _electrostatic(adapter: AssembledIonicStorage1D, u: Vector) -> tuple[Vector, Vector, Vector]:
    device = adapter.device
    state = device.state(u, voltage_V=adapter.voltage_V)
    e = state.electronic_state
    if isinstance(device, CompiledIonicDevice1D):
        assert isinstance(state, IonicState1D)
        ledger = device.charge_ledger(e.electron_density_m3, e.hole_density_m3,
            state.positive_ion_density_m3,
            state.negative_ion_density_m3 if device.ion_species_count == 2 else None)
    else:
        populations = device.fields_by_domain(u, voltage_V=adapter.voltage_V)[3]
        ledger = device.charge_ledger(e.electron_density_m3, e.hole_density_m3, populations)
    geometry, poisson = device.electronic_component.geometry, device.electronic_component.poisson
    rho = ledger.space_charge_C_m3()
    interior = poisson.face_displacement(e.potential_V)
    ports = contact_displacement(interior, rho, geometry.cell_measures_m)
    return e.potential_V, np.r_[ports[0], interior, ports[1]], rho * geometry.cell_measures_m


def _domain_records(device: IonicDevice, data: dict[str, Any]) -> dict[str, Any]:
    if isinstance(device, CompiledMultipleIonicDevice1D):
        return cast(dict[str, Any], data["domains"])
    names = data.get("ion_species", {"positive": data})
    records = {}
    for species in device.ion_species_names:
        values = names[species]
        records[species] = {
            "inventory_m2": values["ion_inventory_m2"],
            "inventory_rate_m2_s": values["ion_inventory_rate_m2_s"],
            "balance_m2_s": values["ion_balance_m2_s"],
            "balance_l1_m2_s": math.fsum(abs(np.asarray(values["ion_balance_m2_s"])).tolist()),
            "domain_flux_m2_s": np.r_[0.0, values["ion_domain_fluxes_m2_s"], 0.0],
        }
    return {device.domain_layer_id: {"species": records,
            "electrical_node_indices": device.ion_node_indices,
            "physical_weights_m": device.ion_geometry.cell_measures_m}}


def assembled_ionic_bdf_accounting(step: BDFStep, u: Vector) -> dict[str, Any]:
    """Original BE readouts at order one; full coefficient-consistent BDF2."""
    adapter, be = _adapter(step), _be(step)
    device = adapter.device
    original = dict(be.accounting(u))
    if step.order == 1:
        original["domains"] = _domain_records(device, original)
        return original
    rate = _physical_rate(step, u)
    view = _AccountingStep(step, be, rate)
    data = dict(ionic_step_accounting(cast(IonicBackwardEuler1DStep, view), u)
                if isinstance(be, AssembledIonicBackwardEuler1DStep)
                else multiple_ionic_step_accounting(cast(MultipleIonicBackwardEuler1DStep, view), u))
    domains = _domain_records(device, data)
    old, older = step.history[:2]
    assert isinstance(old.problem, AssembledIonicStorage1D)
    assert isinstance(older.problem, AssembledIonicStorage1D)
    coefficients, h = step.coefficients, step.dt_s
    new_s, old_s, older_s = adapter.physical_storage(u), old.problem.physical_storage(old.u), older.problem.physical_storage(older.u)
    rounded_rate = (coefficients[0] * (new_s - old_s) - coefficients[2] * (old_s - older_s)) / h
    actual_increment = adapter.physical_increment(u, old.problem, old.u)
    ion_balance = np.zeros(device.node_count)
    for domain, blocks in _domains(device):
        for species, block in blocks.items():
            record = domains[domain.domain_layer_id]["species"][species]
            sign = 1.0 if species == "positive" else -1.0
            ion_balance[domain.ion_node_indices] += sign * np.asarray(record["balance_m2_s"])
            record.update(inventory_rate_m2_s=rate[block], rounded_inventory_rate_m2_s=rounded_rate[block],
                inventory_readback_error_m2_s=rate[block] - rounded_rate[block],
                physical_inventory_increment_m2=actual_increment[block],
                bdf_weighted_inventory_increment_m2=h * rate[block],
                native_BDF_rows_m2_s=step.residual(u)[block])
            if isinstance(device, CompiledMultipleIonicDevice1D):
                record["native_BE_rows_m2_s"] = original["domains"][domain.domain_layer_id]["species"][species]["native_BE_rows_m2_s"]
            else:
                named = data.get("ion_species", {"positive": data})[species]
                named["rounded_ion_inventory_rate_m2_s"] = rounded_rate[block]
                named["ion_inventory_readback_error_m2_s"] = rate[block] - rounded_rate[block]
    phi, displacement, charge = _electrostatic(adapter, u)
    old_phi, old_d, old_charge = _electrostatic(old.problem, old.u)
    older_phi, older_d, older_charge = _electrostatic(older.problem, older.u)
    phi_rate = (coefficients[0] * (phi - old_phi) - coefficients[2] * (old_phi - older_phi)) / h
    interior = -device.electronic_component.poisson.face_capacitance_F_m2 * np.diff(phi_rate)
    charge_rate = np.asarray(data["charge_inventory_rate_A_m2"])
    displacement_rate = np.r_[interior[0] - charge_rate[0], interior, interior[-1] + charge_rate[-1]]
    rounded_d_rate = (coefficients[0] * (displacement - old_d) - coefficients[2] * (old_d - older_d)) / h
    rounded_charge_rate = (coefficients[0] * (charge - old_charge) - coefficients[2] * (old_charge - older_charge)) / h
    d_error, q_error = displacement_rate - rounded_d_rate, charge_rate - rounded_charge_rate
    gauss = displacement_rate[:-1] - displacement_rate[1:] + charge_rate
    rounded_gauss = rounded_d_rate[:-1] - rounded_d_rate[1:] + rounded_charge_rate
    carrier_balance = np.asarray(data["carrier_balance_m2_s"])
    defect = Q * (carrier_balance[1] - carrier_balance[0] + ion_balance)
    current = np.asarray(data["conduction_current_density_A_m2"]) + displacement_rate
    jump = current[:-1] - current[1:]
    data.update(domains=domains, bdf_order=2, bdf_coefficients=coefficients, bdf_residual=step.residual(u),
        backward_euler_comparison=original, backward_euler_residual=original["backward_euler_residual"],
        bdf_physical_inventory_rate_m2_s=rate, bdf_rounded_inventory_rate_m2_s=rounded_rate,
        carrier_inventory_rate_m2_s=rate[device.node_count:3 * device.node_count].reshape(2, device.node_count),
        rounded_carrier_inventory_rate_m2_s=rounded_rate[device.node_count:3 * device.node_count].reshape(2, device.node_count),
        carrier_inventory_readback_error_m2_s=(rate - rounded_rate)[device.node_count:3 * device.node_count].reshape(2, device.node_count),
        bdf_displacement_rate_A_m2=displacement_rate, bdf_rounded_displacement_rate_A_m2=rounded_d_rate,
        bdf_displacement_readback_error_A_m2=d_error,
        bdf_weighted_displacement_increment_C_m2=h * displacement_rate,
        bdf_actual_physical_inventory_increment_m2=actual_increment,
        displacement_C_m2=displacement, charge_inventory_C_m2=charge,
        charge_inventory_rate_A_m2=charge_rate, charge_readback_error_A_m2=q_error,
        charge_subtraction_roundoff_A_m2=q_error,
        interval_displacement_C_m2=original["interval_displacement_C_m2"],
        rounded_interval_displacement_C_m2=original["rounded_interval_displacement_C_m2"],
        displacement_readback_error_C_m2=original["displacement_readback_error_C_m2"],
        displacement_current_density_A_m2=displacement_rate, total_current_density_A_m2=current,
        terminal_total_current_density_A_m2=current[[0, -1]],
        stable_gauss_defect_rate_A_m2=gauss, rounded_gauss_defect_rate_A_m2=rounded_gauss,
        gauss_defect_rate_A_m2=rounded_gauss,
        maxwell_cell_jump_A_m2=jump, predicted_maxwell_cell_jump_A_m2=defect + gauss,
        readback_predicted_maxwell_cell_jump_A_m2=defect + rounded_gauss + q_error + d_error[:-1] - d_error[1:],
        maxwell_identity_error_A_m2=jump - defect - gauss,
        readback_maxwell_identity_error_A_m2=jump - defect - rounded_gauss - q_error - d_error[:-1] + d_error[1:],
        maxwell_spread_A_m2=float(np.ptp(current)))
    if isinstance(device, CompiledIonicDevice1D):
        positive = device.ion_species_slices[0]
        data["rounded_ion_inventory_rate_m2_s"] = rounded_rate[positive]
        data["ion_inventory_readback_error_m2_s"] = rate[positive] - rounded_rate[positive]
    return data


def _events(source: BDFDrive, settings: Any, factory: Any, controls: IonicTransientOptions) -> tuple[BDFEvent, ...]:
    def restart(old: Any, new: Any, u: Vector) -> tuple[Vector, Assessment]:
        be = _be(BDFStep(new, (HistoryPoint(0.0, u, old.storage(u), old),), 1.0, 1))
        right = (be.initial_coordinates("previous", voltage_lift=True)
                 if isinstance(be, AssembledIonicBackwardEuler1DStep) else be.initial_coordinates())
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


def _solve(
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
        events=_events(source, settings, factory, options), output_times_s=protocol.times_s)
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
    return _solve(device, admitted, initial, protocol, options, bdf_options,
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
    return _solve(device, admitted, initial, protocol, options, bdf_options,
                  BDFDrive.from_protocol(protocol) if drive is None else drive, adaptive_max_steps, checks, started)
