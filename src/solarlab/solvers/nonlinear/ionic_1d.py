"""Mobile inventory and per-cell immobile closure for coupled finite-site ions.

One balance per mobile species or every cell of an immobile species is closed
by its declared physical inventory/profile. All original balances remain diagnostic and
are checked independently. No density projection or electronic-only solve
occurs here; every Newton system contains 3N electronic coordinates and M
coordinates per species, with explicit mapping into the electrical grid.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
import hashlib
import math
import time
from typing import cast

import numpy as np
from scipy.sparse import csc_matrix

from solarlab.contracts.units import ELEMENTARY_CHARGE_C as Q
from solarlab.equations.dc_1d import CapabilityError
from solarlab.equations.ionic_1d import CompiledIonicDevice1D, Ionic1DProblem, IonicState1D
from solarlab.equations.ionic_closure import (
    declared_immobile_profiles, immobile_profile_binding, ion_mobility_partition,
    ionic_closure_residual, ionic_closure_jacobian,
)
from solarlab.solvers.nonlinear.newton import NewtonResult, NewtonSettings, solve_newton
from solarlab.solvers.nonlinear.problem import CallableProblem, GateOutcome, Matrix, ResidualScaling, SolveBudget, StepControl
from solarlab.solvers.nonlinear.production import (
    HISTORICAL_MAX_ROW_V1, poisson_row_weights, production_coordinates, production_residual_scaling,
)
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions

__all__ = ["ConstrainedIonic1DProblem", "IonicSteadyOptions", "IonicSteadyResult", "solve_ionic_steady_state", "ionic_steady_control_record", "checks", "diagnostics"]


@dataclass(frozen=True)
class ConstrainedIonic1DProblem:
    """Original rows with mobile inventory or immobile per-cell physical closure."""

    original: Ionic1DProblem
    inventory_row_node: int

    def __post_init__(self) -> None:
        device = self.original.device
        if type(self.inventory_row_node) is not int or not 0 <= self.inventory_row_node < device.ion_node_count:
            raise ValueError("inventory_row_node must name an ion-domain node")
        ion_mobility_partition(device)
        manifest = device.binding_manifest
        domain = manifest.get("ion_domain")
        component_binding = manifest.get("electronic_component")
        ion_model = manifest.get("ion_model")
        if not isinstance(domain, dict) or domain.get("contacts") != ["blocking", "blocking"]:
            raise CapabilityError("ionic steady closure requires the declared closed blocking domain")
        if not isinstance(component_binding, dict) or not isinstance(ion_model, dict):
            raise ValueError("complete component and ionic policy bindings are required")
        if (hashlib.sha256(device.binding_manifest_json.encode()).hexdigest() != device.identity
                or manifest["original_prepared_sha256"] != device.original_prepared_sha256
                or component_binding["compiled_sha256"] != device.electronic_component.identity
                or ion_model["steric_model"] != device.steric_model
                or not np.array_equal(ion_model["face_diffusivity_m2_s"], device.positive_ion_diffusivity_faces_m2_s)):
            raise ValueError("the full ionic/component provenance binding is inconsistent")
        if (domain["layer"] != device.domain_layer_id
                or not np.array_equal(domain.get("electrical_node_indices"), device.ion_node_indices)
                or not np.array_equal(domain["nodes_m"], device.ion_geometry.nodes_m)
                or not np.array_equal(domain["cell_measures_m"], device.ion_geometry.cell_measures_m)
                or float(domain["reference_inventory_m2"]) != device.reference_inventory_m2):
            raise ValueError("physical inventory must match the original prepared domain and weights")
        if device.ion_species_count == 2:
            if (ion_model.get("site_contract") != device.ion_site_contract
                    or device.ion_site_contract not in {"independent_finite_sites", "shared_finite_sites"}
                    or not np.array_equal(ion_model.get("negative_face_diffusivity_m2_s"),
                                          device.negative_ion_diffusivity_faces_m2_s)):
                raise ValueError("the negative ionic provenance binding is inconsistent")
            if domain.get("negative_reference_inventory_m2") != device.negative_reference_inventory_m2:
                raise ValueError("negative physical inventory must match the original prepared domain and weights")

    @property
    def constraint_row(self) -> int:
        return 3 * self.original.device.node_count + self.inventory_row_node

    @property
    def immobile_species(self) -> tuple[str, ...]:
        return ion_mobility_partition(self.original.device)[1]

    @property
    def profile_constraint_rows(self) -> dict[str, tuple[int, ...]]:
        return {species: tuple(range(block.start, block.stop))
                for species, block in zip(self.original.device.ion_species_names, self.original.device.ion_species_slices)
                if species in self.immobile_species}

    @property
    def inventory_target_m2(self) -> float:
        return self.original.device.reference_inventory_m2

    @property
    def constraint_rows(self) -> dict[str, int]:
        device = self.original.device
        return {species: block.start + self.inventory_row_node
                for species, block in zip(device.ion_species_names, device.ion_species_slices)}

    @property
    def inventory_targets_m2(self) -> dict[str, float]:
        device = self.original.device
        targets = {"positive": self.inventory_target_m2}
        if device.ion_species_count == 2:
            targets["negative"] = device.negative_reference_inventory_m2
        return targets

    def residual(self, y: np.ndarray) -> np.ndarray:
        if self.immobile_species:
            return ionic_closure_residual(self.original, y, self.inventory_row_node)
        result = self.original.residual(y)
        device = self.original.device
        _, _, _, populations = device.fields_by_species(y, voltage_V=self.original.voltage_V)
        for species, density in zip(device.ion_species_names, populations):
            inventory = math.fsum((device.ion_geometry.cell_measures_m * density).tolist())
            result[self.constraint_rows[species]] = inventory - self.inventory_targets_m2[species]
        return result

    def jacobian(self, y: np.ndarray) -> csc_matrix:
        if self.immobile_species:
            return ionic_closure_jacobian(self.original, y, self.inventory_row_node)
        matrix = self.original.jacobian(y).tolil()
        device = self.original.device
        # The actual physical storage derivative supplies d(sum(W_P*P))/da.
        storage = self.original.storage_jacobian(y)
        derivative = -storage.diagonal()
        for species, block in zip(device.ion_species_names, device.ion_species_slices):
            row = self.constraint_rows[species]
            if device.ion_site_contract == "shared_finite_sites":
                # Each inventory depends on both shared-site coordinates.
                first_ion = 3 * device.node_count
                gradient = -np.asarray(storage[block, first_ion:].sum(axis=0)).ravel()
                matrix.rows[row] = list(range(first_ion, device.state_size))
                matrix.data[row] = gradient.tolist()
            else:
                matrix.rows[row] = list(range(block.start, block.stop))
                matrix.data[row] = derivative[block].tolist()
        return matrix.tocsc()


@dataclass(frozen=True)
class IonicSteadyOptions:
    """Explicit internal row controls and independent numerical checks.

    electronic supplies physical limits and phi/quasi-Fermi step bounds only;
    budget controls the one coupled solve. Ionic limits apply independently
    to every configured species. No electronic continuation runs.
    """

    electronic: SteadyOptions
    inventory_relative_tolerance: float
    ion_balance_tolerance_m2_s: float
    ion_flux_tolerance_m2_s: float
    ion_logit_step: float
    inventory_row_node: int
    budget: SolveBudget
    wall_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.electronic, SteadyOptions) or not isinstance(self.budget, SolveBudget):
            raise ValueError("explicit electronic controls and a Newton budget are required")
        if self.electronic.convergence_policy != HISTORICAL_MAX_ROW_V1:
            raise ValueError("ionic steady supports only historical_max_row_v1; no other convergence policy is implemented")
        for name in ("inventory_relative_tolerance", "ion_balance_tolerance_m2_s",
                     "ion_flux_tolerance_m2_s", "ion_logit_step", "wall_s"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class IonicSteadyResult:
    """Candidate state and original evidence; accepted is numerical only."""

    state: IonicState1D | None
    accepted: bool
    termination: str
    newton: NewtonResult | None
    initial_state: IonicState1D
    candidate_coordinates: np.ndarray
    diagnostics: dict[str, object] | None
    checks: tuple[GateOutcome, ...]
    solver_controls: dict[str, object]
    compiled_identity: str
    original_prepared_identity: str
    binding_manifest: dict[str, object]
    evaluation_counts: dict[str, int]
    wall_s: float
    error: str | None = None
    qualification: bool = False


def ionic_steady_control_record(device: CompiledIonicDevice1D, options: IonicSteadyOptions) -> dict[str, object]:
    """Retain old positive-D controls; bind declared profiles only for zero D."""
    recorded = asdict(options)
    binding = immobile_profile_binding(device)
    if binding is not None:
        recorded["ion_closure"] = binding
    return recorded


def diagnostics(problem: ConstrainedIonic1DProblem, y: np.ndarray) -> dict[str, object]:
    """Keep historical positive aliases; dual evidence also names each species."""
    original, device = problem.original, problem.original.device
    count = device.node_count
    phi, n, p, populations = device.fields_by_species(y, voltage_V=original.voltage_V)
    density = populations[0]
    positive_rows = device.ion_species_slices[0]
    full = original.residual(y)
    component = device.electronic_component
    electronic = original.electronic_problem
    carrier_rows = full[count:3 * count].reshape(2, count).copy()
    fixed_logs = []
    for species, carrier in enumerate(("n", "p")):
        for side, node in (("left", 0), ("right", count - 1)):
            if component.channel(carrier, side).kind == "fixed_reservoir":
                fixed_logs.append(abs(carrier_rows[species, node]))
                carrier_rows[species, node] = 0.0
    flux = original.ion_fluxes(y)
    electronic_current = np.r_[electronic.contact_current_density(y[:3 * count])[0],
                               electronic.current_density(y[:3 * count]),
                               electronic.contact_current_density(y[:3 * count])[1]]
    ionic_current = np.r_[0.0, original.ion_current_density_A_m2(y), 0.0]
    inventory = math.fsum((device.ion_geometry.cell_measures_m * density).tolist())
    ledger = device.charge_ledger(n, p, density, populations[1] if len(populations) == 2 else None)
    deposited = component.geometry.cell_measures_m * Q * ledger.entry("positive_ion").net_population()
    domain_charge = Q * device.ion_geometry.cell_measures_m * (density - device.positive_ion_reference_m3)
    physical = np.zeros(count)
    physical[device.ion_node_indices] = domain_charge
    result = {
        "electrical_node_count": count, "ion_node_count": device.ion_node_count,
        "ion_node_indices": device.ion_node_indices, "ion_nodes_m": device.ion_geometry.nodes_m,
        "original_residual": full, "constrained_residual": problem.residual(y),
        "constraint_row": problem.constraint_row, "constraint_unit": "particles/m^2",
        "inventory_target_m2": problem.inventory_target_m2, "inventory_m2": inventory,
        "inventory_error_m2": inventory - problem.inventory_target_m2,
        "inventory_relative_error": abs(inventory - problem.inventory_target_m2) / problem.inventory_target_m2,
        "ion_flux_m2_s": flux, "ion_balance_m2_s": full[positive_rows],
        "ion_balance_l1_m2_s": math.fsum(abs(full[positive_rows]).tolist()),
        "ion_balance_sum_m2_s": math.fsum(full[positive_rows].tolist()),
        "ion_flux_max_m2_s": float(np.max(abs(flux))),
        "electronic_current_density_A_m2": electronic_current,
        "ionic_current_density_A_m2": ionic_current,
        "total_current_density_A_m2": electronic_current + ionic_current,
        "current_spread_A_m2": float(np.ptp(electronic_current + ionic_current)),
        "carrier_balance_A_m2": float(Q * np.sum(abs(carrier_rows))),
        "poisson_normalized_max": float(np.max(abs(full[:count] * poisson_row_weights(component)))),
        "fixed_contact_log_max": max(fixed_logs, default=0.0),
        "charge_density_C_m3": ledger.space_charge_C_m3(),
        "physical_ion_net_charge_C_m2": physical,
        "domain_ion_net_charge_C_m2": domain_charge,
        "deposited_ion_net_charge_C_m2": deposited,
        "charge_deposition_error_C_m2": deposited - physical,
        "ion_density_change_from_reference_m3": density - device.positive_ion_reference_m3,
        "minimum_ion_density_m3": float(np.min(density)),
        "maximum_site_fraction": float(np.max(density / device.positive_ion_site_limit_m3)),
        "potential_V": phi,
    }
    if device.ion_species_count == 2:
        species_keys = (
            "constraint_row", "inventory_target_m2", "inventory_m2", "inventory_error_m2",
            "inventory_relative_error", "ion_flux_m2_s", "ion_balance_m2_s", "ion_balance_l1_m2_s",
            "ion_balance_sum_m2_s", "ion_flux_max_m2_s", "domain_ion_net_charge_C_m2",
            "physical_ion_net_charge_C_m2", "deposited_ion_net_charge_C_m2", "charge_deposition_error_C_m2",
            "ion_density_change_from_reference_m3", "minimum_ion_density_m3", "maximum_site_fraction",
        )
        positive = {key: result[key] for key in species_keys}
        positive["original_dropped_balance_m2_s"] = float(full[problem.constraint_row])
        negative_density = populations[1]
        negative_rows = device.ion_species_slices[1]
        negative_flux = original.ion_fluxes(y, species="negative")
        negative_inventory = math.fsum((device.ion_geometry.cell_measures_m * negative_density).tolist())
        negative_target = problem.inventory_targets_m2["negative"]
        negative_domain_charge = -Q * device.ion_geometry.cell_measures_m * (
            negative_density - device.negative_ion_reference_m3)
        negative_physical = np.zeros(count)
        negative_physical[device.ion_node_indices] = negative_domain_charge
        negative_deposited = -component.geometry.cell_measures_m * Q * ledger.entry("negative_ion").net_population()
        negative = {
            "constraint_row": problem.constraint_rows["negative"], "inventory_target_m2": negative_target,
            "inventory_m2": negative_inventory, "inventory_error_m2": negative_inventory - negative_target,
            "inventory_relative_error": abs(negative_inventory - negative_target) / negative_target,
            "ion_flux_m2_s": negative_flux, "ion_balance_m2_s": full[negative_rows],
            "ion_balance_l1_m2_s": math.fsum(abs(full[negative_rows]).tolist()),
            "ion_balance_sum_m2_s": math.fsum(full[negative_rows].tolist()),
            "ion_flux_max_m2_s": float(np.max(abs(negative_flux))),
            "original_dropped_balance_m2_s": float(full[problem.constraint_rows["negative"]]),
            "domain_ion_net_charge_C_m2": negative_domain_charge,
            "physical_ion_net_charge_C_m2": negative_physical,
            "deposited_ion_net_charge_C_m2": negative_deposited,
            "charge_deposition_error_C_m2": negative_deposited - negative_physical,
            "ion_density_change_from_reference_m3": negative_density - device.negative_ion_reference_m3,
            "minimum_ion_density_m3": float(np.min(negative_density)),
            "maximum_site_fraction": float(np.max(negative_density / device.negative_ion_site_limit_m3)),
        }
        result["ion_species"] = {"positive": positive, "negative": negative}
        for key in ("domain_ion_net_charge_C_m2", "physical_ion_net_charge_C_m2",
                    "deposited_ion_net_charge_C_m2", "charge_deposition_error_C_m2"):
            result[key] = positive[key] + negative[key]
    if device.ion_site_contract == "shared_finite_sites":
        occupied = (populations[0] + populations[1]) / device.positive_ion_site_limit_m3
        result["shared_site"] = {
            "summed_site_fraction": occupied, "vacancy_fraction": 1.0 - occupied,
            "maximum_summed_site_fraction": float(np.max(occupied)),
            "minimum_vacancy_fraction": float(np.min(1.0 - occupied)),
        }
    profiles = declared_immobile_profiles(device)
    if profiles:
        result["immobile_profiles"] = {"binding": immobile_profile_binding(device), "species": {
            species: {"constraint_rows": problem.profile_constraint_rows[species],
                      "target_density_m3": target,
                      "actual_density_m3": populations[device.ion_species_names.index(species)],
                      "cell_particle_error_m2": device.ion_geometry.cell_measures_m *
                          (populations[device.ion_species_names.index(species)] - target),
                      "relative_error": float(np.max(abs(populations[device.ion_species_names.index(species)] - target) / target))}
            for species, target in profiles.items()}}
    return result


def checks(diagnostics: dict[str, object], options: IonicSteadyOptions) -> tuple[GateOutcome, ...]:
    limits = {
        "inventory_relative_error": options.inventory_relative_tolerance,
        "ion_balance_l1_m2_s": options.ion_balance_tolerance_m2_s,
        "ion_flux_max_m2_s": options.ion_flux_tolerance_m2_s,
        "carrier_balance_A_m2": options.electronic.continuity_tolerance_A_m2,
        "poisson_normalized_max": options.electronic.poisson_tolerance,
        "fixed_contact_log_max": options.electronic.contact_log_tolerance,
        "current_spread_A_m2": options.electronic.current_spread_tolerance_A_m2,
    }
    outcomes = []
    for name, limit in limits.items():
        measured = diagnostics[name]
        if not isinstance(measured, (int, float)):
            raise TypeError("numerical check must have a scalar measurement")
        value = float(measured)
        outcomes.append(GateOutcome(name, math.isfinite(value) and value <= limit, value, limit,
                                    "prospective numerical research control; qualification remains false"))
    if "ion_species" in diagnostics:
        negative = cast(dict[str, dict[str, object]], diagnostics["ion_species"])["negative"]
        for name in ("inventory_relative_error", "ion_balance_l1_m2_s", "ion_flux_max_m2_s"):
            value, limit = float(cast(float, negative[name])), limits[name]
            outcomes.append(GateOutcome("negative_" + name, math.isfinite(value) and value <= limit,
                                        value, limit, "independent negative-species numerical control; qualification remains false"))
    if "immobile_profiles" in diagnostics:
        local = cast(dict[str, dict[str, object]], diagnostics["immobile_profiles"])
        for species, values in cast(dict[str, dict[str, object]], local["species"]).items():
            value = float(cast(float, values["relative_error"]))
            limit = options.inventory_relative_tolerance
            outcomes.append(GateOutcome(species + "_immobile_profile_relative_error", math.isfinite(value) and value <= limit,
                                        value, limit, "declared physical target per cell; no average or profile projection"))
    return tuple(outcomes)


class _WallLimit(RuntimeError):
    pass


def solve_ionic_steady_state(
    device: CompiledIonicDevice1D, *, voltage_V: float, generation_scale: float,
    options: IonicSteadyOptions, initial_state: IonicState1D | None = None,
) -> IonicSteadyResult:
    """One bounded joint solve, retaining the first failure without a retry."""
    started = time.monotonic()
    if not isinstance(device, CompiledIonicDevice1D) or not isinstance(options, IonicSteadyOptions):
        raise ValueError("an actual full ionic device and explicit ionic steady controls are required")
    original = Ionic1DProblem(device, voltage_V=voltage_V, generation_scale=generation_scale)
    problem = ConstrainedIonic1DProblem(original, options.inventory_row_node)
    initial = device.initial_state(voltage_V=voltage_V) if initial_state is None else initial_state
    if (not isinstance(initial, IonicState1D) or initial.device_sha256 != device.identity
            or initial.voltage_V != voltage_V
            or initial.electronic_state.device_sha256 != device.electronic_component.identity):
        raise ValueError("initial state must carry the full same-device ionic identity and voltage")
    checked = device.state(initial.u, voltage_V=voltage_V)
    for name in ("positive_ion_density_m3", "negative_ion_density_m3"):
        if not np.array_equal(getattr(checked, name), getattr(initial, name)):
            raise ValueError("initial ionic fields disagree with the bound coordinates")
    for name in ("potential_V", "electron_density_m3", "hole_density_m3"):
        if not np.array_equal(getattr(checked.electronic_state, name), getattr(initial.electronic_state, name)):
            raise ValueError("initial electronic fields disagree with the bound full state")
    if (initial.domain_inventories_m2 != checked.domain_inventories_m2
            or any(abs(checked.domain_inventories_m2[species] - target)
                   > options.inventory_relative_tolerance * target
                   for species, target in problem.inventory_targets_m2.items() if species not in problem.immobile_species)):
        raise ValueError("initial physical inventory must match the declared prepared-domain inventory")
    ion_count = device.ion_node_count
    electronic = options.electronic
    weights = np.r_[production_residual_scaling(
        device.electronic_component, poisson_tolerance=electronic.poisson_tolerance,
        continuity_tolerance_A_m2=electronic.continuity_tolerance_A_m2,
        contact_log_tolerance=electronic.contact_log_tolerance,
    ).row_weights, np.full(ion_count * device.ion_species_count, 2 * ion_count / options.ion_balance_tolerance_m2_s)]
    for species, row in problem.constraint_rows.items():
        weights[row] = 1.0 / (options.inventory_relative_tolerance * problem.inventory_targets_m2[species])
    for species, block in zip(device.ion_species_names, device.ion_species_slices):
        if species in problem.immobile_species:
            target = getattr(device, species + "_ion_reference_m3")
            weights[block] = 1.0 / (options.inventory_relative_tolerance * device.ion_geometry.cell_measures_m * target)
    steps = np.r_[production_coordinates(device.electronic_component,
                                         potential_step_V=electronic.potential_step_V,
                                         quasi_fermi_step=electronic.quasi_fermi_step).max_step,
                  np.full(ion_count * device.ion_species_count, options.ion_logit_step)]
    deadline = started + options.wall_s
    last = np.array(initial.u, copy=True)
    counts = {"residual": 0, "jacobian": 0}

    def checkpoint() -> None:
        if time.monotonic() >= deadline:
            raise _WallLimit("ionic steady wall budget exhausted")

    def evaluate(kind: str, function: Callable[[np.ndarray], Matrix], y: np.ndarray) -> Matrix:
        nonlocal last
        last = np.array(y, copy=True)
        checkpoint()
        counts[kind] += 1
        value = function(y)
        checkpoint()
        return value

    def gate(y: np.ndarray) -> GateOutcome:
        checkpoint()
        outcomes = checks(diagnostics(problem, y), options)
        checkpoint()
        ratio = max(item.value / item.limit for item in outcomes)
        return GateOutcome("original_ionic_inventory_flux_current_checks", all(item.passed for item in outcomes),
                           ratio, 1.0, "; ".join(item.name for item in outcomes if not item.passed))

    settings = NewtonSettings(ResidualScaling(weights), 1.0, StepControl(steps), options.budget, (gate,), False)
    newton, error, state, diagnostic_values = None, None, None, None
    check_outcomes: tuple[GateOutcome, ...] = ()
    try:
        newton = solve_newton(CallableProblem(lambda y: evaluate("residual", problem.residual, y),
                                             lambda y: evaluate("jacobian", problem.jacobian, y)), last, settings)
        last = newton.state
        checkpoint()
        state = device.state(last, voltage_V=voltage_V)
        diagnostic_values = diagnostics(problem, last)
        check_outcomes = checks(diagnostic_values, options)
        checkpoint()
        accepted = newton.accepted and all(item.passed for item in check_outcomes)
        termination = newton.termination.value
    except Exception as exc:
        accepted = False
        termination = "wall_budget_exhausted" if isinstance(exc, _WallLimit) else "evaluation_error"
        error = f"{type(exc).__name__}: {str(exc)[:400]}"
    candidate = np.array(last, copy=True)
    candidate.setflags(write=False)
    return IonicSteadyResult(state, accepted, termination, newton, initial, candidate, diagnostic_values, check_outcomes,
                             ionic_steady_control_record(device, options), device.identity, device.original_prepared_sha256,
                             device.binding_manifest, counts, time.monotonic() - started, error)
