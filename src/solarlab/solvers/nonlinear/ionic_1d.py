"""Inventory-constrained, jointly coupled single-ion research steady solve.

One dependent blocking-ion balance is replaced by the physical inventory
declared in the compiled input. All original balances remain diagnostic and
are checked independently. No density projection or electronic-only solve
occurs here; every Newton system contains phi, both carriers and the ions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
import hashlib
import math
import time

import numpy as np
from scipy.sparse import csc_matrix

from solarlab.contracts.units import ELEMENTARY_CHARGE_C as Q
from solarlab.equations.dc_1d import CapabilityError
from solarlab.equations.ionic_1d import CompiledIonicDevice1D, Ionic1DProblem, IonicState1D
from solarlab.solvers.nonlinear.newton import NewtonResult, NewtonSettings, solve_newton
from solarlab.solvers.nonlinear.problem import CallableProblem, GateOutcome, Matrix, ResidualScaling, SolveBudget, StepControl
from solarlab.solvers.nonlinear.production import (
    HISTORICAL_MAX_ROW_V1, poisson_row_weights, production_coordinates, production_residual_scaling,
)
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions

__all__ = ["ConstrainedIonic1DProblem", "IonicSteadyOptions", "IonicSteadyResult", "solve_ionic_steady_state"]


@dataclass(frozen=True)
class ConstrainedIonic1DProblem:
    """Original coupled steady rows plus one physical number-inventory row."""

    original: Ionic1DProblem
    inventory_row_node: int

    def __post_init__(self) -> None:
        device = self.original.device
        if type(self.inventory_row_node) is not int or not 0 <= self.inventory_row_node < device.node_count:
            raise ValueError("inventory_row_node must name an ion-domain node")
        if np.any(device.positive_ion_diffusivity_faces_m2_s <= 0.0):
            raise CapabilityError("one-constraint ionic steady solve requires one connected positive-D domain; zero D is underdetermined")
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
                or not np.array_equal(domain["nodes_m"], device.nodes_m)
                or not np.array_equal(domain["cell_measures_m"], device.ion_geometry.cell_measures_m)
                or float(domain["reference_inventory_m2"]) != device.reference_inventory_m2):
            raise ValueError("physical inventory must match the original prepared domain and weights")

    @property
    def constraint_row(self) -> int:
        return 3 * self.original.device.node_count + self.inventory_row_node

    @property
    def inventory_target_m2(self) -> float:
        return self.original.device.reference_inventory_m2

    def residual(self, y: np.ndarray) -> np.ndarray:
        result = self.original.residual(y)
        device = self.original.device
        _, _, _, density = device.fields(y, voltage_V=self.original.voltage_V)
        inventory = math.fsum((device.ion_geometry.cell_measures_m * density).tolist())
        result[self.constraint_row] = inventory - self.inventory_target_m2
        return result

    def jacobian(self, y: np.ndarray) -> csc_matrix:
        matrix = self.original.jacobian(y).tolil()
        count = self.original.device.node_count
        # The actual physical storage derivative supplies d(sum(W_P*P))/da.
        derivative = -self.original.storage_jacobian(y).diagonal()[3 * count:]
        matrix.rows[self.constraint_row] = list(range(3 * count, 4 * count))
        matrix.data[self.constraint_row] = derivative.tolist()
        return matrix.tocsc()


@dataclass(frozen=True)
class IonicSteadyOptions:
    """Explicit internal row controls and independent numerical checks.

    electronic supplies physical limits and phi/quasi-Fermi step bounds only;
    budget controls the one coupled solve. No electronic continuation runs.
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


def _diagnostics(problem: ConstrainedIonic1DProblem, y: np.ndarray) -> dict[str, object]:
    original, device = problem.original, problem.original.device
    count = device.node_count
    phi, n, p, density = device.fields(y, voltage_V=original.voltage_V)
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
    ionic_current = Q * np.r_[0.0, flux, 0.0]
    inventory = math.fsum((device.ion_geometry.cell_measures_m * density).tolist())
    ledger = device.charge_ledger(n, p, density)
    deposited = component.geometry.cell_measures_m * Q * ledger.entry("positive_ion").net_population()
    physical = Q * device.ion_geometry.cell_measures_m * (density - device.positive_ion_reference_m3)
    return {
        "original_residual": full, "constrained_residual": problem.residual(y),
        "constraint_row": problem.constraint_row, "constraint_unit": "particles/m^2",
        "inventory_target_m2": problem.inventory_target_m2, "inventory_m2": inventory,
        "inventory_error_m2": inventory - problem.inventory_target_m2,
        "inventory_relative_error": abs(inventory - problem.inventory_target_m2) / problem.inventory_target_m2,
        "ion_flux_m2_s": flux, "ion_balance_m2_s": full[3 * count:],
        "ion_balance_l1_m2_s": math.fsum(abs(full[3 * count:]).tolist()),
        "ion_balance_sum_m2_s": math.fsum(full[3 * count:].tolist()),
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
        "deposited_ion_net_charge_C_m2": deposited,
        "charge_deposition_error_C_m2": deposited - physical,
        "ion_density_change_from_reference_m3": density - device.positive_ion_reference_m3,
        "minimum_ion_density_m3": float(np.min(density)),
        "maximum_site_fraction": float(np.max(density / device.positive_ion_site_limit_m3)),
        "potential_V": phi,
    }


def _checks(diagnostics: dict[str, object], options: IonicSteadyOptions) -> tuple[GateOutcome, ...]:
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
    for name in ("positive_ion_density_m3",):
        if not np.array_equal(getattr(checked, name), getattr(initial, name)):
            raise ValueError("initial ionic fields disagree with the bound coordinates")
    for name in ("potential_V", "electron_density_m3", "hole_density_m3"):
        if not np.array_equal(getattr(checked.electronic_state, name), getattr(initial.electronic_state, name)):
            raise ValueError("initial electronic fields disagree with the bound full state")
    if (initial.domain_inventory_m2 != checked.domain_inventory_m2
            or abs(checked.domain_inventory_m2 - problem.inventory_target_m2)
            > options.inventory_relative_tolerance * problem.inventory_target_m2):
        raise ValueError("initial physical inventory must match the declared prepared-domain inventory")
    count = device.node_count
    electronic = options.electronic
    weights = np.r_[production_residual_scaling(
        device.electronic_component, poisson_tolerance=electronic.poisson_tolerance,
        continuity_tolerance_A_m2=electronic.continuity_tolerance_A_m2,
        contact_log_tolerance=electronic.contact_log_tolerance,
    ).row_weights, np.full(count, 2 * count / options.ion_balance_tolerance_m2_s)]
    weights[problem.constraint_row] = 1.0 / (options.inventory_relative_tolerance * problem.inventory_target_m2)
    steps = np.r_[production_coordinates(device.electronic_component,
                                         potential_step_V=electronic.potential_step_V,
                                         quasi_fermi_step=electronic.quasi_fermi_step).max_step,
                  np.full(count, options.ion_logit_step)]
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
        outcomes = _checks(_diagnostics(problem, y), options)
        checkpoint()
        ratio = max(item.value / item.limit for item in outcomes)
        return GateOutcome("original_ionic_inventory_flux_current_checks", all(item.passed for item in outcomes),
                           ratio, 1.0, "; ".join(item.name for item in outcomes if not item.passed))

    settings = NewtonSettings(ResidualScaling(weights), 1.0, StepControl(steps), options.budget, (gate,), False)
    newton, error, state, diagnostics = None, None, None, None
    checks: tuple[GateOutcome, ...] = ()
    try:
        newton = solve_newton(CallableProblem(lambda y: evaluate("residual", problem.residual, y),
                                             lambda y: evaluate("jacobian", problem.jacobian, y)), last, settings)
        last = newton.state
        checkpoint()
        state = device.state(last, voltage_V=voltage_V)
        diagnostics = _diagnostics(problem, last)
        checks = _checks(diagnostics, options)
        checkpoint()
        accepted = newton.accepted and all(item.passed for item in checks)
        termination = newton.termination.value
    except Exception as exc:
        accepted = False
        termination = "wall_budget_exhausted" if isinstance(exc, _WallLimit) else "evaluation_error"
        error = f"{type(exc).__name__}: {str(exc)[:400]}"
    candidate = np.array(last, copy=True)
    candidate.setflags(write=False)
    return IonicSteadyResult(state, accepted, termination, newton, initial, candidate, diagnostics, checks,
                             asdict(options), device.identity, device.original_prepared_sha256,
                             device.binding_manifest, counts, time.monotonic() - started, error)
