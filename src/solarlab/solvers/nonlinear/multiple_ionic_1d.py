"""One bounded joint steady solve with separate closed-domain inventories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
from scipy.sparse import csc_matrix

from solarlab.contracts import frozen_array
from solarlab.contracts.units import ELEMENTARY_CHARGE_C as Q
from solarlab.equations.dc_1d import DC1DProblem
from solarlab.equations.ionic_1d import Ionic1DProblem
from solarlab.equations.ionic_closure import immobile_profile_binding, ion_mobility_partition
from solarlab.equations.multiple_ionic_1d import CompiledMultipleIonicDevice1D, MultipleIonic1DProblem, MultipleIonicState1D
from solarlab.solvers.nonlinear.ionic_1d import ConstrainedIonic1DProblem, IonicSteadyOptions
from solarlab.solvers.nonlinear.newton import NewtonResult, NewtonSettings, solve_newton
from solarlab.solvers.nonlinear.problem import CallableProblem, GateOutcome, ResidualScaling, StepControl
from solarlab.solvers.nonlinear.production import poisson_row_weights, production_coordinates, production_residual_scaling
from solarlab.solvers.nonlinear.steady_1d import physical_gates

__all__ = ["ConstrainedMultipleIonic1DProblem", "MultipleIonicSteadyResult",
           "multiple_ionic_diagnostics", "multiple_ionic_checks", "solve_multiple_ionic_steady_state"]


@dataclass(frozen=True)
class ConstrainedMultipleIonic1DProblem:
    """Own mobile inventories and original per-cell immobile physical closure."""

    original: MultipleIonic1DProblem
    inventory_row_node: int

    def __post_init__(self) -> None:
        if not isinstance(self.original, MultipleIonic1DProblem):
            raise TypeError("the original multiple-domain equations are required")
        for domain in self.original.device.domains:
            # Preserve the accepted local source/geometry/mobility validation.
            ConstrainedIonic1DProblem(Ionic1DProblem(domain, self.original.voltage_V,
                                                    self.original.generation_scale), self.inventory_row_node)

    @property
    def immobile_species(self) -> dict[str, tuple[str, ...]]:
        return {domain.domain_layer_id: ion_mobility_partition(domain)[1] for domain in self.original.device.domains}

    @property
    def profile_constraint_rows(self) -> dict[str, dict[str, tuple[int, ...]]]:
        return {name: {species: tuple(range(cast(int, block.start), cast(int, block.stop)))
                       for species, block in blocks.items() if species in self.immobile_species[name]}
                for name, blocks in self.original.device.ion_species_slices.items()}

    @property
    def constraint_rows(self) -> dict[str, dict[str, int]]:
        return {name: {species: cast(int, block.start) + self.inventory_row_node
                       for species, block in blocks.items() if species not in self.immobile_species[name]}
                for name, blocks in self.original.device.ion_species_slices.items()}

    def residual(self, y: np.ndarray) -> np.ndarray:
        device = self.original.device
        result = self.original.residual(y)
        populations = device.fields_by_domain(y, voltage_V=self.original.voltage_V)[3]
        for domain in device.domains:
            name = domain.domain_layer_id
            local_physical = None
            if self.immobile_species[name]:
                local = ConstrainedIonic1DProblem(Ionic1DProblem(domain, self.original.voltage_V,
                    self.original.generation_scale), self.inventory_row_node)
                local_physical = local.residual(device.local_coordinates(y, name))
            for index, species in enumerate(domain.ion_species_names):
                if species in self.immobile_species[name]:
                    assert local_physical is not None
                    result[device.ion_species_slices[name][species]] = local_physical[domain.ion_species_slices[index]]
                    continue
                cells = domain.ion_geometry.cell_measures_m * (
                    populations[name][index] - getattr(domain, species + "_ion_reference_m3"))
                result[self.constraint_rows[name][species]] = math.fsum(cells.tolist())
        return result

    def jacobian(self, y: np.ndarray) -> csc_matrix:
        matrix = self.original.jacobian(y).tolil()
        storage = self.original.storage_jacobian(y)
        device = self.original.device
        for domain in device.domains:
            name, blocks = domain.domain_layer_id, device.ion_species_slices[domain.domain_layer_id]
            if self.immobile_species[name]:
                local = ConstrainedIonic1DProblem(Ionic1DProblem(domain, self.original.voltage_V,
                    self.original.generation_scale), self.inventory_row_node)
                local_jacobian = local.jacobian(device.local_coordinates(y, name))
                domain_block = device.domain_slices[name]
                global_columns = np.r_[np.arange(3 * device.node_count),
                    np.arange(cast(int, domain_block.start), cast(int, domain_block.stop))]
                for index, species in enumerate(domain.ion_species_names):
                    if species not in self.immobile_species[name]:
                        continue
                    for local_row, global_row in zip(
                        range(cast(int, domain.ion_species_slices[index].start), cast(int, domain.ion_species_slices[index].stop)),
                        self.profile_constraint_rows[name][species]):
                        gradient = local_jacobian.getrow(local_row).toarray().ravel()
                        columns = np.flatnonzero(gradient)
                        matrix.rows[global_row], matrix.data[global_row] = global_columns[columns].tolist(), gradient[columns].tolist()
            for species, block in blocks.items():
                if species in self.immobile_species[name]:
                    continue
                row = self.constraint_rows[name][species]
                gradient = -np.asarray(storage[block, :].sum(axis=0)).ravel()
                columns = np.flatnonzero(gradient)
                matrix.rows[row], matrix.data[row] = columns.tolist(), gradient[columns].tolist()
        return matrix.tocsc()


@dataclass(frozen=True)
class MultipleIonicSteadyResult:
    """Raw candidate/evidence; numerical acceptance is not scientific qualification."""

    state: MultipleIonicState1D | None
    accepted: bool
    termination: str
    newton: NewtonResult | None
    initial_state: MultipleIonicState1D
    candidate_coordinates: np.ndarray
    diagnostics: dict[str, Any] | None
    checks: tuple[GateOutcome, ...]
    solver_controls: dict[str, Any]
    compiled_identity: str
    original_prepared_identity: str
    binding_manifest: dict[str, Any]
    evaluation_counts: dict[str, int]
    wall_s: float
    error: str | None = None
    qualification: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_coordinates", frozen_array(self.candidate_coordinates))
        if self.accepted and (self.state is None or self.newton is None or not self.newton.accepted
                              or not self.checks or not all(check.passed for check in self.checks)):
            raise ValueError("acceptance requires the joint Newton result and every original check")


def multiple_ionic_diagnostics(problem: ConstrainedMultipleIonic1DProblem, y: np.ndarray) -> dict[str, Any]:
    original, device = problem.original, problem.original.device
    count = device.node_count
    state = device.state(y, voltage_V=original.voltage_V)
    full = original.residual(y)
    electronic = original.electronic_problem
    carrier = full[count:3 * count].reshape(2, count).copy()
    fixed = []
    for index, species in enumerate(("n", "p")):
        for side, node in (("left", 0), ("right", count - 1)):
            if device.electronic_component.channel(species, side).kind == "fixed_reservoir":
                fixed.append(abs(carrier[index, node]))
                carrier[index, node] = 0.0
    current = np.r_[electronic.contact_current_density(y[:3 * count])[0],
                    electronic.current_density(y[:3 * count]), electronic.contact_current_density(y[:3 * count])[1]]
    ionic = np.r_[0.0, original.ion_current_density_A_m2(y), 0.0]
    populations = device.fields_by_domain(y, voltage_V=original.voltage_V)[3]
    ledger = device.charge_ledger(state.electronic_state.electron_density_m3,
                                 state.electronic_state.hole_density_m3, populations)
    domains: dict[str, Any] = {}
    for domain in device.domains:
        name = domain.domain_layer_id
        species_records = {}
        for index, species in enumerate(domain.ion_species_names):
            density = populations[name][index]
            reference = getattr(domain, species + "_ion_reference_m3")
            capacity = getattr(domain, species + "_ion_site_limit_m3")
            target = device.reference_inventories_m2[name][species]
            error = math.fsum((domain.ion_geometry.cell_measures_m * (density - reference)).tolist())
            block = device.ion_species_slices[name][species]
            flux = original.ion_fluxes(y, domain_id=name, species=species)
            sign = 1.0 if species == "positive" else -1.0
            physical = np.zeros(count)
            physical[domain.ion_node_indices] = sign * Q * domain.ion_geometry.cell_measures_m * (density - reference)
            deposited = sign * Q * device.electronic_component.geometry.cell_measures_m * ledger.entry(
                species + "_ion." + name).net_population()
            species_records[species] = {
                "density_m3": density, "reference_density_m3": reference, "site_capacity_m3": capacity,
                "inventory_m2": state.domain_inventories_m2[name][species], "inventory_target_m2": target,
                "inventory_error_m2": error, "inventory_relative_error": abs(error) / target,
                "gross_inventory_subtraction_error_m2": state.domain_inventories_m2[name][species] - target,
                "ion_balance_m2_s": full[block], "ion_balance_l1_m2_s": math.fsum(abs(full[block]).tolist()),
                "ion_balance_sum_m2_s": math.fsum(full[block].tolist()),
                "ion_flux_m2_s": np.r_[0.0, flux, 0.0], "ion_flux_max_m2_s": float(np.max(abs(flux))),
                "global_face_flux_m2_s": original.ion_fluxes_on_electronic_faces(y, domain_id=name, species=species),
                "minimum_density_m3": float(density.min()), "maximum_site_fraction": float(np.max(density / capacity)),
                "physical_excess_charge_C_m2": physical, "deposited_excess_charge_C_m2": deposited,
                "charge_deposition_error_C_m2": deposited - physical,
                "closure": "declared_per_cell_profile" if species in problem.immobile_species[name] else "own_domain_inventory",
                "dropped_row": None if species in problem.immobile_species[name] else problem.constraint_rows[name][species],
                "dropped_balance_m2_s": None if species in problem.immobile_species[name] else float(full[problem.constraint_rows[name][species]]),
                "maximum_relative_change_from_reference": float(np.max(abs(density - reference) / reference)),
            }
            if species in problem.immobile_species[name]:
                species_records[species].update(
                    profile_constraint_rows=problem.profile_constraint_rows[name][species],
                    profile_relative_error=float(np.max(abs(density - reference) / reference)),
                    profile_cell_error_m2=domain.ion_geometry.cell_measures_m * (density - reference),
                    reference_source_parameter="P0" if species == "positive" else "P0_neg")
        domains[name] = {
            "species": species_records, "electrical_node_indices": domain.ion_node_indices,
            "physical_weights_m": domain.ion_geometry.cell_measures_m, "site_contract": domain.ion_site_contract,
            "maximum_summed_site_fraction": float(np.max(sum(populations[name]) / domain.positive_ion_site_limit_m3))
                if domain.ion_site_contract == "shared_finite_sites" else None,
        }
    return {
        "original_residual": full, "constrained_residual": problem.residual(y), "domains": domains,
        "electrical_node_count": count, "state_size": device.state_size,
        "carrier_balance_A_m2": float(Q * np.sum(abs(carrier))),
        "fixed_contact_log_max": max(fixed, default=0.0),
        "poisson_normalized_max": float(np.max(abs(full[:count] * poisson_row_weights(device.electronic_component)))),
        "electronic_current_density_A_m2": current, "ionic_current_density_A_m2": ionic,
        "total_current_density_A_m2": current + ionic, "current_spread_A_m2": float(np.ptp(current + ionic)),
        "charge_density_C_m3": ledger.space_charge_C_m3(), "potential_V": state.electronic_state.potential_V,
    }


def multiple_ionic_checks(
    problem: ConstrainedMultipleIonic1DProblem, y: np.ndarray, options: IonicSteadyOptions,
) -> tuple[GateOutcome, ...]:
    """Recheck original electronic gates and each complete closed-domain balance."""
    original, device = problem.original, problem.original.device
    size = 3 * device.node_count
    electronic = original.electronic_problem
    # Existing gates use the actual full ionic candidate for Poisson/current;
    # their individual-carrier equilibrium checks still use original fluxes.
    def full(u: np.ndarray) -> np.ndarray:
        return np.r_[u, y[size:]]
    gate_view = SimpleNamespace(device=device.electronic_component, voltage_V=original.voltage_V,
        generation_scale=original.generation_scale,
        residual=lambda u: original.residual(full(u))[:size],
        current_density=lambda u: electronic.current_density(u) + original.ion_current_density_A_m2(full(u)),
        contact_current_density=electronic.contact_current_density, fluxes=electronic.fluxes)
    gates = physical_gates(cast(DC1DProblem, gate_view), options.electronic,
                            equilibrium=original.voltage_V == 0.0 and original.generation_scale == 0.0)
    outcomes = [gate(y[:size]) for gate in gates]
    data = multiple_ionic_diagnostics(problem, y)
    log = float(data["fixed_contact_log_max"])
    outcomes.append(GateOutcome("fixed_contact_log_max", log <= options.electronic.contact_log_tolerance,
                                log, options.electronic.contact_log_tolerance))
    for name, record in data["domains"].items():
        for species, values in record["species"].items():
            prefix = name + "." + species + "."
            limits = {"inventory_relative_error": options.inventory_relative_tolerance,
                      "ion_balance_l1_m2_s": options.ion_balance_tolerance_m2_s,
                      "ion_flux_max_m2_s": options.ion_flux_tolerance_m2_s}
            if species in problem.immobile_species[name]:
                limits["profile_relative_error"] = options.inventory_relative_tolerance
            for key, limit in limits.items():
                value = float(values[key])
                outcomes.append(GateOutcome(prefix + key, math.isfinite(value) and value <= limit, value, limit,
                    "all original rows and compiled reference; qualification remains false"))
            fraction = float(values["maximum_site_fraction"])
            valid = float(values["minimum_density_m3"]) > 0.0 and fraction < 1.0
            if record["maximum_summed_site_fraction"] is not None:
                valid = valid and float(record["maximum_summed_site_fraction"]) < 1.0
            outcomes.append(GateOutcome(prefix + "finite_site_admissibility", valid, fraction, 1.0))
    return tuple(outcomes)


def _settings(problem: ConstrainedMultipleIonic1DProblem, options: IonicSteadyOptions) -> NewtonSettings:
    device = problem.original.device
    weights = np.r_[production_residual_scaling(device.electronic_component,
        poisson_tolerance=options.electronic.poisson_tolerance,
        continuity_tolerance_A_m2=options.electronic.continuity_tolerance_A_m2,
        contact_log_tolerance=options.electronic.contact_log_tolerance).row_weights,
        np.zeros(device.state_size - 3 * device.node_count)]
    for domain in device.domains:
        name = domain.domain_layer_id
        for species, block in device.ion_species_slices[name].items():
            if species in problem.immobile_species[name]:
                weights[block] = 1.0 / (options.inventory_relative_tolerance * domain.ion_geometry.cell_measures_m
                                        * getattr(domain, species + "_ion_reference_m3"))
                continue
            weights[block] = 2 * domain.ion_node_count / options.ion_balance_tolerance_m2_s
            weights[problem.constraint_rows[name][species]] = 1.0 / (
                options.inventory_relative_tolerance * device.reference_inventories_m2[name][species])
    steps = np.r_[production_coordinates(device.electronic_component,
        potential_step_V=options.electronic.potential_step_V,
        quasi_fermi_step=options.electronic.quasi_fermi_step).max_step,
        np.full(device.state_size - 3 * device.node_count, options.ion_logit_step)]
    return NewtonSettings(ResidualScaling(weights), 1.0, StepControl(steps), options.budget, (), False)


def solve_multiple_ionic_steady_state(
    device: CompiledMultipleIonicDevice1D, *, voltage_V: float, generation_scale: float,
    options: IonicSteadyOptions, initial_state: MultipleIonicState1D | None = None,
) -> MultipleIonicSteadyResult:
    """One unchanged-control coupled solve; no retry, preparation or projection."""
    started = time.monotonic()
    if not isinstance(device, CompiledMultipleIonicDevice1D) or not isinstance(options, IonicSteadyOptions):
        raise TypeError("full multiple-domain device and existing explicit ionic controls are required")
    original = MultipleIonic1DProblem(device, voltage_V, generation_scale)
    problem = ConstrainedMultipleIonic1DProblem(original, options.inventory_row_node)
    initial = device.initial_state(voltage_V=voltage_V) if initial_state is None else initial_state
    device.validate_state(initial)
    if initial.voltage_V != voltage_V:
        raise ValueError("initial voltage must match; no implicit voltage/reference conversion")
    for name, targets in device.reference_inventories_m2.items():
        for species, target in targets.items():
            if species not in problem.immobile_species[name] and abs(initial.domain_inventories_m2[name][species] - target) > options.inventory_relative_tolerance * target:
                raise ValueError("each mobile domain/species initial inventory must match its own compiled target")
    deadline = started + options.wall_s
    last = initial.u.copy()
    counts = {"residual": 0, "jacobian": 0}
    def checkpoint() -> None:
        if time.monotonic() >= deadline:
            raise TimeoutError("multiple-domain steady wall budget exhausted")
    def evaluate(kind: str, y: np.ndarray) -> Any:
        nonlocal last
        last = np.array(y, copy=True)
        checkpoint()
        counts[kind] += 1
        value = problem.residual(y) if kind == "residual" else problem.jacobian(y)
        checkpoint()
        return value
    def gate(y: np.ndarray) -> GateOutcome:
        checkpoint()
        checks = multiple_ionic_checks(problem, y, options)
        checkpoint()
        return GateOutcome("all_original_multiple_domain_checks", all(item.passed for item in checks),
            max(item.value / item.limit for item in checks), 1.0,
            "; ".join(item.name for item in checks if not item.passed))
    base = _settings(problem, options)
    settings = NewtonSettings(base.scaling, base.tolerance, base.step_control, base.budget, (gate,), False)
    newton, state, diagnostics, error = None, None, None, None
    checks: tuple[GateOutcome, ...] = ()
    accepted = False
    try:
        newton = solve_newton(CallableProblem(lambda y: evaluate("residual", y),
                                             lambda y: evaluate("jacobian", y)), last, settings)
        last = newton.state
        checkpoint()
        state = device.state(last, voltage_V=voltage_V)
        diagnostics = multiple_ionic_diagnostics(problem, last)
        checks = multiple_ionic_checks(problem, last, options)
        checkpoint()
        accepted = newton.accepted and all(item.passed for item in checks)
        termination = newton.termination.value
    except Exception as exc:
        termination = "wall_budget_exhausted" if isinstance(exc, TimeoutError) else "evaluation_error"
        error = f"{type(exc).__name__}: {str(exc)[:500]}"
    control = {"numerical": asdict(options), "constraint_rows": problem.constraint_rows,
        "inventory_targets_m2": device.reference_inventories_m2,
        "supported_mobility": "every_domain_species_connected_positive_D",
        "convergence_norm": "historical_max_row_v1", "merit": "original_weighted_l2_armijo"}
    if any(problem.immobile_species.values()):
        control["supported_mobility"] = "connected_positive_D_or_declared_per_cell_D0"
        control["profile_constraint_rows"] = problem.profile_constraint_rows
        control["immobile_profile_binding"] = {domain.domain_layer_id: immobile_profile_binding(domain)
            for domain in device.domains if problem.immobile_species[domain.domain_layer_id]}
    return MultipleIonicSteadyResult(state, accepted, termination, newton, initial, np.array(last, copy=True),
        diagnostics, checks, control, device.identity, device.original_prepared_sha256,
        device.binding_manifest, counts, time.monotonic() - started, error)
