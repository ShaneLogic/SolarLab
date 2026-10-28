"""Explicit driven ionic histories from a pinned, accepted same-device steady state.

Every carrier/ion continuity row remains active in the physical BE solve.
The steady inventory replacement is used only when admitting the initial
artifact. Histories retain their accepted prefix and first failure, without
substeps, normalization, retries, or physical qualification claims.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from solarlab.contracts import frozen_array
from solarlab.equations.ionic_1d import CompiledIonicDevice1D, Ionic1DProblem, IonicState1D
from solarlab.equations.ionic_closure import immobile_profile_binding
from solarlab.experiments.transient import TransientOptions, TransientProtocol
from solarlab.solvers.nonlinear.ionic_1d import (
    ConstrainedIonic1DProblem, IonicSteadyOptions, _checks, _diagnostics, ionic_steady_control_record,
)
from solarlab.solvers.nonlinear.newton import NewtonResult, NewtonSettings, solve_newton
from solarlab.solvers.nonlinear.problem import (
    CallableProblem, GateOutcome, Matrix, ResidualScaling, StepControl, state_digest,
)
from solarlab.solvers.nonlinear.production import (
    POISSON_CONTACT_BALANCE_V1, production_convergence_norm,
    production_coordinates, production_residual_scaling,
)
from solarlab.solvers.time.ionic_backward_euler import IonicBackwardEuler1DStep, step_accounting

__all__ = [
    "IonicTransientInitial", "IonicTransientOptions", "IonicTransientAttempt",
    "IonicTransientResult", "load_ionic_steady_initial", "solve_ionic_transient",
]


@dataclass(frozen=True)
class IonicTransientInitial:
    """Exact state and retained source bytes of an accepted steady artifact."""

    state: IonicState1D
    generation_scale: float
    steady_controls: IonicSteadyOptions
    source_path: str
    source_sha256: str
    evidence_json: str


def _same_state(left: IonicState1D, right: IonicState1D) -> bool:
    if not isinstance(left, IonicState1D) or not isinstance(right, IonicState1D):
        return False
    if (left.device_sha256 != right.device_sha256 or left.voltage_V != right.voltage_V
            or left.domain_inventories_m2 != right.domain_inventories_m2
            or left.electronic_state.device_sha256 != right.electronic_state.device_sha256):
        return False
    pairs = [(left.u, right.u), (left.positive_ion_density_m3, right.positive_ion_density_m3),
             (left.negative_ion_density_m3, right.negative_ion_density_m3)]
    pairs.extend((getattr(left.electronic_state, name), getattr(right.electronic_state, name))
                 for name in ("potential_V", "electron_density_m3", "hole_density_m3"))
    return all(np.array_equal(a, b) for a, b in pairs)


def _admit_evidence(
    compiled: CompiledIonicDevice1D, text: str, sha256: str, controls: IonicSteadyOptions,
) -> tuple[IonicState1D, float, tuple[GateOutcome, ...]]:
    """Recheck the saved initial steady point, without solving or changing it."""
    if not isinstance(controls, IonicSteadyOptions):
        raise ValueError("initial steady controls must be explicit IonicSteadyOptions")
    if hashlib.sha256(text.encode()).hexdigest() != sha256:
        raise ValueError("initial steady artifact SHA-256 mismatch")
    try:
        document = json.loads(text)
        summary, result = document["summary"], document["result"]
        if (summary["accepted"] is not True or result["accepted"] is not True
                or result["newton"]["accepted"] is not True or result["termination"] != "converged"
                or result["newton"]["termination"] != "converged"):
            raise ValueError("initial artifact must retain an accepted steady solve")
        if (summary["compiled_identity"] != compiled.identity
                or result["compiled_identity"] != compiled.identity
                or result["original_prepared_identity"] != compiled.original_prepared_sha256
                or result["binding_manifest"] != compiled.binding_manifest):
            raise ValueError("initial artifact belongs to a different compiled device or source binding")
        if result["solver_controls"] != ionic_steady_control_record(compiled, controls):
            raise ValueError("initial steady controls differ from the recorded solve")
        voltage, generation = summary["voltage_V"], summary["generation_scale"]
        if (type(voltage) not in (int, float) or type(generation) not in (int, float)
                or not math.isfinite(voltage) or not math.isfinite(generation) or generation < 0):
            raise ValueError("initial voltage/generation must be explicit finite numbers")
        saved = result["state"]
        state = compiled.state(np.asarray(saved["u"], dtype=float), voltage_V=voltage)
        electronic = saved["electronic_state"]
        if (saved["device_sha256"] != compiled.identity
                or electronic["device_sha256"] != compiled.electronic_component.identity
                or electronic["voltage_V"] != voltage
                or saved["domain_inventory_m2"] != state.domain_inventory_m2
                or not np.array_equal(saved["positive_ion_density_m3"], state.positive_ion_density_m3)
                or saved.get("negative_domain_inventory_m2", 0.0) != state.negative_domain_inventory_m2
                or not np.array_equal(saved.get("negative_ion_density_m3", []), state.negative_ion_density_m3)
                or not np.array_equal(electronic["u"], state.electronic_state.u)
                or not np.array_equal(result["candidate_coordinates"], state.u)
                or not np.array_equal(result["newton"]["state"], state.u)
                or any(not np.array_equal(electronic[name], getattr(state.electronic_state, name))
                       for name in ("potential_V", "electron_density_m3", "hole_density_m3"))):
            raise ValueError("initial recorded physical fields disagree with their coordinates")
        recorded_checks = result["checks"]
        if not recorded_checks or any(check["passed"] is not True for check in recorded_checks):
            raise ValueError("initial steady artifact contains unaccepted checks")
        saved_checks = tuple(GateOutcome(**check) for check in recorded_checks)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("incomplete or malformed initial steady artifact") from exc
    # Initial admission only. The time loop below never constructs this problem.
    steady = ConstrainedIonic1DProblem(Ionic1DProblem(compiled, voltage, generation), controls.inventory_row_node)
    checks = _checks(_diagnostics(steady, state.u), controls)
    saved_limits = {check.name: check.limit for check in saved_checks}
    if len(saved_limits) != len(saved_checks) or saved_limits != {check.name: check.limit for check in checks}:
        raise ValueError("initial recorded checks do not match the declared steady controls")
    if not all(check.passed for check in checks):
        raise ValueError("initial steady admission failed: " + ", ".join(check.name for check in checks if not check.passed))
    return state, float(generation), checks


def load_ionic_steady_initial(
    compiled: CompiledIonicDevice1D, path: str | Path, sha256: str, *, controls: IonicSteadyOptions,
) -> IonicTransientInitial:
    """Load and validate a pinned steady example artifact; never rerun its solve."""
    source = Path(path)
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("initial steady artifact SHA-256 mismatch")
    text = raw.decode()
    state, generation, _ = _admit_evidence(compiled, text, sha256, controls)
    return IonicTransientInitial(state, generation, controls, str(source.resolve()), sha256, text)


@dataclass(frozen=True)
class IonicTransientOptions:
    """Existing electronic controls plus independent per-species L1 and step bounds."""

    electronic: TransientOptions
    ion_balance_tolerance_m2_s: float
    ion_logit_step: float

    def __post_init__(self) -> None:
        if not isinstance(self.electronic, TransientOptions):
            raise ValueError("explicit electronic TransientOptions are required")
        if self.electronic.convergence_policy != POISSON_CONTACT_BALANCE_V1:
            raise ValueError("ionic transient supports only poisson_contact_balance_v1")
        if self.electronic.predictor != "previous":
            raise ValueError("ionic transient currently supports only the previous-state predictor")
        for value in (self.ion_balance_tolerance_m2_s, self.ion_logit_step):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("ion residual tolerance and logit step must be finite and positive")


@dataclass(frozen=True)
class IonicTransientAttempt:
    start_s: float
    end_s: float
    voltage_V: float
    generation_scale: float
    accepted: bool
    termination: str
    initial_coordinates: np.ndarray
    candidate_coordinates: np.ndarray
    newton: NewtonResult | None
    accounting: dict[str, object] | None
    evaluation_counts: dict[str, int]
    wall_s: float
    error: str | None = None

    def __post_init__(self) -> None:
        for name in ("initial_coordinates", "candidate_coordinates"):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class IonicTransientResult:
    protocol: TransientProtocol
    accepted_times_s: np.ndarray
    states: tuple[IonicState1D, ...]
    attempts: tuple[IonicTransientAttempt, ...]
    initial_provenance: dict[str, object]
    compiled_identity: str
    original_prepared_identity: str
    binding_manifest: dict[str, object]
    solver_controls: dict[str, object]
    evaluation_counts: dict[str, int]
    wall_s: float
    physical_certificate_status: str = "not_evaluated"

    def __post_init__(self) -> None:
        object.__setattr__(self, "accepted_times_s", frozen_array(self.accepted_times_s))

    @property
    def completed(self) -> bool:
        return (len(self.states) == self.protocol.times_s.size
                and len(self.attempts) == self.protocol.times_s.size - 1
                and all(attempt.accepted for attempt in self.attempts))

    @property
    def failed_attempts(self) -> tuple[IonicTransientAttempt, ...]:
        return tuple(attempt for attempt in self.attempts if not attempt.accepted)


class _WallBudgetExceeded(RuntimeError):
    pass


def _finite_accounting(value: object) -> bool:
    if isinstance(value, dict):
        return all(_finite_accounting(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_accounting(item) for item in value)
    if isinstance(value, np.ndarray):
        return bool(np.all(np.isfinite(value)))
    if isinstance(value, (float, int, np.number)):
        return bool(np.isfinite(value))
    return True


def _newton_settings(
    compiled: CompiledIonicDevice1D, options: IonicTransientOptions,
) -> tuple[NewtonSettings, Callable[[np.ndarray], float]]:
    electronic = options.electronic
    count, ion_count = compiled.node_count, compiled.ion_node_count
    weights = np.r_[production_residual_scaling(
        compiled.electronic_component, poisson_tolerance=electronic.poisson_tolerance,
        continuity_tolerance_A_m2=electronic.continuity_tolerance_A_m2,
        contact_log_tolerance=electronic.contact_log_tolerance,
    ).row_weights, np.full(ion_count * compiled.ion_species_count, ion_count / options.ion_balance_tolerance_m2_s)]
    steps = np.r_[production_coordinates(compiled.electronic_component,
        potential_step_V=electronic.potential_step_V, quasi_fermi_step=electronic.quasi_fermi_step).max_step,
        np.full(ion_count * compiled.ion_species_count, options.ion_logit_step)]
    settings = NewtonSettings(ResidualScaling(weights), 1.0, StepControl(steps), electronic.newton_budget, (), False)
    electronic_norm = production_convergence_norm(compiled.electronic_component,
                                                   convergence_policy=electronic.convergence_policy)
    assert electronic_norm is not None

    def convergence_norm(scaled: np.ndarray) -> float:
        if scaled.shape != (compiled.state_size,) or not np.all(np.isfinite(scaled)):
            return math.inf
        return max(electronic_norm(scaled[:3 * count]),
                   *(math.fsum(abs(scaled[block]).tolist()) / ion_count for block in compiled.ion_species_slices))

    return settings, convergence_norm


def solve_ionic_transient(
    compiled: CompiledIonicDevice1D, initial: IonicTransientInitial, protocol: TransientProtocol,
    *, options: IonicTransientOptions,
) -> IonicTransientResult:
    """Run requested physical BE steps once, preserving the first partial outcome.

The one history wall budget includes initial admission and all attempted steps.
Newton call budgets are explicit per step, with the total bounded by max_steps;
accounting is called at most once per attempt and counted separately. Individual
NumPy/SciPy calls are not preempted. Initial-only steady checks never gate the
time steps, and all dynamic ion continuity rows remain in residual and J.
"""
    started = time.monotonic()
    if not isinstance(compiled, CompiledIonicDevice1D) or not isinstance(initial, IonicTransientInitial):
        raise ValueError("an actual compiled ionic device and pinned steady initial are required")
    if not isinstance(protocol, TransientProtocol) or not isinstance(options, IonicTransientOptions):
        raise ValueError("explicit transient protocol and ionic controls are required")
    electronic = options.electronic
    if protocol.times_s.size - 1 > electronic.max_steps:
        raise ValueError("requested timepoints exceed max_steps")
    admitted, generation0, initial_checks = _admit_evidence(
        compiled, initial.evidence_json, initial.source_sha256, initial.steady_controls,
    )
    if not _same_state(initial.state, admitted) or initial.generation_scale != generation0:
        raise ValueError("initial object disagrees with its pinned steady evidence")
    if admitted.voltage_V != protocol.voltage_V[0] or generation0 != protocol.generation_scale[0]:
        raise ValueError("initial voltage/generation must equal the first protocol sample")
    settings, convergence_norm = _newton_settings(compiled, options)
    deadline = started + electronic.wall_s
    states, accepted_times, attempts = [admitted], [float(protocol.times_s[0])], []
    total_counts = {"residual": 0, "jacobian": 0, "accounting": 0}

    def checkpoint() -> None:
        if time.monotonic() >= deadline:
            raise _WallBudgetExceeded("ionic transient history wall budget exhausted; no retry")

    for index in range(1, protocol.times_s.size):
        start, end = float(protocol.times_s[index - 1]), float(protocol.times_s[index])
        voltage, generation = float(protocol.voltage_V[index]), float(protocol.generation_scale[index])
        attempt_started = time.monotonic()
        candidate = np.array(states[-1].u, copy=True)
        seed = candidate.copy()
        counts = {"residual": 0, "jacobian": 0, "accounting": 0}
        newton, accounting, error = None, None, None
        next_state = None

        def evaluate(kind: str, function: Callable[[np.ndarray], Matrix], y: np.ndarray) -> Matrix:
            nonlocal candidate
            candidate = np.array(y, copy=True)
            checkpoint()
            counts[kind] += 1
            value = function(y)
            checkpoint()
            return value

        try:
            checkpoint()
            step = IonicBackwardEuler1DStep(Ionic1DProblem(compiled, voltage, generation), states[-1], end - start)
            candidate = step.initial_coordinates(electronic.predictor, voltage_lift=True)
            seed = candidate.copy()
            checkpoint()
            newton = solve_newton(CallableProblem(lambda y: evaluate("residual", step.residual, y),
                                                   lambda y: evaluate("jacobian", step.jacobian, y)),
                                  candidate, settings, convergence_norm=convergence_norm)
            candidate = newton.state
            checkpoint()
            if np.all(np.isfinite(candidate)):
                next_state = compiled.state(candidate, voltage_V=voltage)
                counts["accounting"] += 1
                accounting = step_accounting(step, candidate)
            checkpoint()
            accepted, termination = newton.accepted, newton.termination.value
            if accepted and (accounting is None or not _finite_accounting(accounting)):
                accepted, termination = False, "nonfinite_accounting"
            if accepted:
                assert accounting is not None
                if convergence_norm(settings.scaling.row_weights * np.asarray(accounting["backward_euler_residual"])) > 1:
                    accepted, termination = False, "residual_check_failed"
        except Exception as exc:
            accepted = False
            termination = "wall_budget_exhausted" if isinstance(exc, _WallBudgetExceeded) else "evaluation_error"
            error = f"{type(exc).__name__}: {str(exc)[:400]}"
        attempts.append(IonicTransientAttempt(start, end, voltage, generation, accepted, termination,
            seed, candidate, newton, accounting, counts.copy(), time.monotonic() - attempt_started, error))
        for kind in total_counts:
            total_counts[kind] += counts[kind]
        if not accepted:
            break
        assert next_state is not None
        states.append(next_state)
        accepted_times.append(end)
    provenance = {"source_path": initial.source_path, "source_sha256": initial.source_sha256,
                  "state_sha256": state_digest(admitted.u), "generation_scale": generation0,
                  "steady_controls": asdict(initial.steady_controls), "admission_checks": initial_checks}
    profile_binding = immobile_profile_binding(compiled)
    if profile_binding is not None:
        provenance["ion_closure"] = profile_binding
    recorded = asdict(options)
    recorded["stopping_norm"] = "max(electronic_balance_norm(scaled_BE[:3N]),sum(abs(scaled_BE[3N:]))/M)"
    if compiled.ion_species_count == 2:
        recorded["stopping_norm"] = "max(electronic_balance_norm(scaled_BE[:3N]),L1(scaled_BE[positive])/M,L1(scaled_BE[negative])/M)"
        recorded["ion_species_controls"] = {species: {
            "ion_balance_tolerance_m2_s": options.ion_balance_tolerance_m2_s,
            "ion_logit_step": options.ion_logit_step,
        } for species in compiled.ion_species_names}
    if compiled.ion_site_contract == "shared_finite_sites":
        recorded["ion_site_contract"] = compiled.ion_site_contract
        recorded["ion_inventory_increment"] = "coupled_shared_site_physical_secant"
    recorded["voltage_seed_policy"] = "harmonic_potential_preserve_carrier_and_ion_densities"
    recorded["max_total_newton_residual_calls"] = electronic.max_steps * electronic.newton_budget.max_residual_evaluations
    recorded["max_total_newton_jacobian_calls"] = electronic.max_steps * electronic.newton_budget.max_jacobian_evaluations
    return IonicTransientResult(protocol, np.asarray(accepted_times), tuple(states), tuple(attempts), provenance,
        compiled.identity, compiled.original_prepared_sha256, compiled.binding_manifest, recorded, total_counts,
        time.monotonic() - started)
