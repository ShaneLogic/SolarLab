"""One explicit native two-domain history from pinned accepted steady evidence."""

from __future__ import annotations

from solarlab.reproducibility.job_provenance import (
    JobProvenance, current_provenance, job_provenance, result_provenance,
)

from dataclasses import asdict, dataclass, field
from collections.abc import Callable
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, cast

import numpy as np

from solarlab.contracts import frozen_array
from solarlab.equations.assembly.ionic_1d import (
    AssembledConstrainedMultipleIonic1DProblem, AssembledMultipleIonic1DProblem,
)
from solarlab.equations.implementation import provenance_binding, use_assembled
from solarlab.equations.ionic_closure import immobile_profile_binding, ion_mobility_partition
from solarlab.equations.multiple_ionic_1d import CompiledMultipleIonicDevice1D, MultipleIonic1DProblem, MultipleIonicState1D
from solarlab.experiments.ionic_transient import IonicTransientOptions, _finite_accounting
from solarlab.experiments.transient import TransientProtocol
from solarlab.experiments.transient_results import _decode, _encode
from solarlab.solvers.nonlinear.ionic_1d import IonicSteadyOptions
from solarlab.solvers.nonlinear.multiple_ionic_1d import ConstrainedMultipleIonic1DProblem, multiple_ionic_checks
from solarlab.solvers.nonlinear.newton import NewtonResult, NewtonSettings, solve_newton
from solarlab.solvers.nonlinear.problem import CallableProblem, GateOutcome, Matrix, ResidualScaling, StepControl, state_digest
from solarlab.solvers.nonlinear.production import production_convergence_norm, production_coordinates, production_residual_scaling
from solarlab.solvers.time.assembled_ionic_backward_euler import AssembledMultipleIonicBackwardEuler1DStep
from solarlab.solvers.time.multiple_ionic_backward_euler import MultipleIonicBackwardEuler1DStep, multiple_ionic_step_accounting

__all__ = ["MultipleIonicTransientInitial", "MultipleIonicTransientAttempt", "MultipleIonicTransientResult",
           "load_multiple_ionic_steady_initial", "solve_multiple_ionic_transient",
           "write_multiple_ionic_transient", "read_multiple_ionic_transient"]


@dataclass(frozen=True)
class MultipleIonicTransientInitial:
    state: MultipleIonicState1D
    generation_scale: float
    source_path: str
    source_sha256: str
    evidence_json: str
    steady_controls: IonicSteadyOptions


def _same_record(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_same_record(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_same_record(a, b) for a, b in zip(left, right))
    return bool(left == right)


def _admit(
    device: CompiledMultipleIonicDevice1D, text: str, sha256: str, controls: IonicSteadyOptions,
) -> tuple[MultipleIonicState1D, float, tuple[GateOutcome, ...]]:
    if not isinstance(device, CompiledMultipleIonicDevice1D) or not isinstance(controls, IonicSteadyOptions):
        raise ValueError("a complete device and explicit initial steady controls are required")
    immobile = {domain.domain_layer_id: ion_mobility_partition(domain)[1] for domain in device.domains}
    if hashlib.sha256(text.encode()).hexdigest() != sha256:
        raise ValueError("initial evidence SHA256 mismatch")
    document = _decode(json.loads(text))
    if document.get("schema") != "solarlab.multiple_ionic_steady.v1":
        raise ValueError("an actual multiple-domain steady result is required")
    record = document["result"]
    if (record["accepted"] is not True or record["newton"] is None or record["newton"]["accepted"] is not True
            or not record["checks"] or not all(item["passed"] for item in record["checks"])):
        raise ValueError("missing or failed steady evidence cannot initialize a history")
    if (record["compiled_identity"] != device.identity
            or record["original_prepared_identity"] != device.original_prepared_sha256
            or record["binding_manifest"] != device.binding_manifest
            or record["solver_controls"]["numerical"] != asdict(controls)):
        raise ValueError("initial model/source/control binding mismatch")
    if any(immobile.values()):
        binding = {domain.domain_layer_id: immobile_profile_binding(domain)
                   for domain in device.domains if immobile[domain.domain_layer_id]}
        if record["solver_controls"].get("immobile_profile_binding") != binding:
            raise ValueError("initial declared D0 profile binding mismatch")
    saved = record["state"]
    if saved is None:
        raise ValueError("accepted evidence must contain its actual state")
    voltage = saved["electronic_state"]["voltage_V"]
    state = device.state(saved["u"], voltage_V=voltage)
    if (not _same_record(_decode(_encode(state)), saved)
            or not np.array_equal(record["newton"]["state"], state.u)
            or not np.array_equal(record["candidate_coordinates"], state.u)):
        raise ValueError("saved physical fields, coordinates or state identity disagree")
    # This reader explicitly admits the dark initial scope by rechecking G=0;
    # the old saved envelope did not record a standalone generation field.
    generation = 0.0
    problem: ConstrainedMultipleIonic1DProblem | AssembledConstrainedMultipleIonic1DProblem
    if use_assembled():
        problem = AssembledConstrainedMultipleIonic1DProblem(AssembledMultipleIonic1DProblem(device, voltage, generation),
                                                             controls.inventory_row_node)
    else:
        problem = ConstrainedMultipleIonic1DProblem(MultipleIonic1DProblem(device, voltage, generation),
                                                    controls.inventory_row_node)
    checks = multiple_ionic_checks(problem, state.u, controls)
    if not all(check.passed for check in checks):
        raise ValueError("saved state fails current full original steady revalidation")
    return state, generation, checks


def load_multiple_ionic_steady_initial(
    device: CompiledMultipleIonicDevice1D, path: str | Path, sha256: str, *, controls: IonicSteadyOptions,
) -> MultipleIonicTransientInitial:
    source = Path(path)
    text = source.read_text()
    state, generation, _ = _admit(device, text, sha256, controls)
    return MultipleIonicTransientInitial(state, generation, str(source.resolve()), sha256, text, controls)


@dataclass(frozen=True)
class MultipleIonicTransientAttempt:
    start_s: float
    end_s: float
    voltage_V: float
    generation_scale: float
    accepted: bool
    termination: str
    initial_coordinates: np.ndarray
    candidate_coordinates: np.ndarray
    newton: NewtonResult | None
    accounting: dict[str, Any] | None
    checks: tuple[GateOutcome, ...]
    evaluation_counts: dict[str, int]
    wall_s: float
    error: str | None = None


@dataclass(frozen=True)
class MultipleIonicTransientResult:
    protocol: TransientProtocol
    accepted_times_s: np.ndarray
    states: tuple[MultipleIonicState1D, ...]
    attempts: tuple[MultipleIonicTransientAttempt, ...]
    initial_provenance: dict[str, Any]
    compiled_identity: str
    original_prepared_identity: str
    binding_manifest: dict[str, Any]
    solver_controls: dict[str, Any]
    evaluation_counts: dict[str, int]
    wall_s: float
    physical_certificate_status: str = "not_evaluated"

    job_provenance: dict[str, Any] | None = field(default_factory=current_provenance)

    def __post_init__(self) -> None:
        object.__setattr__(self, "accepted_times_s", frozen_array(self.accepted_times_s))

    @property
    def completed(self) -> bool:
        return (len(self.states) == self.protocol.times_s.size and len(self.attempts) == self.protocol.times_s.size - 1
                and all(attempt.accepted for attempt in self.attempts))

    @property
    def failed_attempts(self) -> tuple[MultipleIonicTransientAttempt, ...]:
        return tuple(attempt for attempt in self.attempts if not attempt.accepted)


def _settings(device: CompiledMultipleIonicDevice1D, options: IonicTransientOptions) -> tuple[NewtonSettings, Callable[[np.ndarray], float]]:
    electronic, size = options.electronic, 3 * device.node_count
    weights = np.r_[production_residual_scaling(device.electronic_component,
        poisson_tolerance=electronic.poisson_tolerance, continuity_tolerance_A_m2=electronic.continuity_tolerance_A_m2,
        contact_log_tolerance=electronic.contact_log_tolerance).row_weights, np.zeros(device.state_size - size)]
    for domain in device.domains:
        for block in device.ion_species_slices[domain.domain_layer_id].values():
            weights[block] = domain.ion_node_count / options.ion_balance_tolerance_m2_s
    steps = np.r_[production_coordinates(device.electronic_component,
        potential_step_V=electronic.potential_step_V, quasi_fermi_step=electronic.quasi_fermi_step).max_step,
        np.full(device.state_size - size, options.ion_logit_step)]
    settings = NewtonSettings(ResidualScaling(weights), 1.0, StepControl(steps), electronic.newton_budget, (), False)
    electronic_norm = production_convergence_norm(device.electronic_component, convergence_policy=electronic.convergence_policy)
    assert electronic_norm is not None
    def norm(scaled: np.ndarray) -> float:
        if scaled.shape != (device.state_size,) or not np.all(np.isfinite(scaled)):
            return math.inf
        return max(electronic_norm(scaled[:size]),
            *(math.fsum(abs(scaled[block]).tolist()) / domain.ion_node_count
              for domain in device.domains for block in device.ion_species_slices[domain.domain_layer_id].values()))
    return settings, norm


def solve_multiple_ionic_transient(
    device: CompiledMultipleIonicDevice1D, initial: MultipleIonicTransientInitial, protocol: TransientProtocol,
    *, options: IonicTransientOptions,
) -> MultipleIonicTransientResult:
    """Exactly the requested native time steps, stopping at the first failure."""
    started = time.monotonic()
    if not isinstance(initial, MultipleIonicTransientInitial) or not isinstance(protocol, TransientProtocol) or not isinstance(options, IonicTransientOptions):
        raise ValueError("explicit bound initial evidence, protocol and reviewed ionic controls are required")
    with job_provenance({"compiled_identity": device.identity, "initial_sha256": initial.source_sha256, "steady_controls": asdict(initial.steady_controls), "protocol": _encode(protocol), "controls": asdict(options)},
            **provenance_binding("MultipleIonicBackwardEuler1DStep", device,
                inventory_row_node=initial.steady_controls.inventory_row_node)):
        admitted, generation, admission_checks = _admit(device, initial.evidence_json, initial.source_sha256, initial.steady_controls)
        if not _same_record(_decode(_encode(admitted)), _decode(_encode(initial.state))) or initial.generation_scale != generation:
            raise ValueError("initial object disagrees with pinned evidence")
        if admitted.voltage_V != protocol.voltage_V[0] or generation != protocol.generation_scale[0]:
            raise ValueError("initial voltage/generation must match the first protocol sample")
        if protocol.times_s.size - 1 > options.electronic.max_steps:
            raise ValueError("requested steps exceed the existing max_steps control")
        settings, norm = _settings(device, options)
        deadline = started + options.electronic.wall_s
        assembled = use_assembled()
        states, times, attempts = [admitted], [float(protocol.times_s[0])], []
        total = {"residual": 0, "jacobian": 0, "accounting": 0}
        def checkpoint() -> None:
            if time.monotonic() >= deadline:
                raise TimeoutError("two-domain history wall budget exhausted; no retry")
        for index in range(1, protocol.times_s.size):
            began = time.monotonic()
            start, end = float(protocol.times_s[index - 1]), float(protocol.times_s[index])
            voltage, light = float(protocol.voltage_V[index]), float(protocol.generation_scale[index])
            seed = states[-1].u.copy()
            candidate = seed.copy()
            counts = {"residual": 0, "jacobian": 0, "accounting": 0}
            newton, accounting, next_state, error = None, None, None, None
            checks: tuple[GateOutcome, ...] = ()
            def evaluate(kind: str, y: np.ndarray) -> Matrix:
                nonlocal candidate
                candidate = np.array(y, copy=True)
                checkpoint()
                counts[kind] += 1
                value = step.residual(y) if kind == "residual" else step.jacobian(y)
                checkpoint()
                return value
            try:
                checkpoint()
                # The assembled step is constructor-compatible with the original one.
                step = (cast(MultipleIonicBackwardEuler1DStep, AssembledMultipleIonicBackwardEuler1DStep(
                            AssembledMultipleIonic1DProblem(device, voltage, light), states[-1], end - start))
                        if assembled else
                        MultipleIonicBackwardEuler1DStep(MultipleIonic1DProblem(device, voltage, light), states[-1],
                                                         end - start))
                seed = step.initial_coordinates()
                candidate = seed.copy()
                newton = solve_newton(CallableProblem(lambda y: evaluate("residual", y),
                                                      lambda y: evaluate("jacobian", y)), seed, settings, convergence_norm=norm)
                candidate = newton.state
                checkpoint()
                next_state = device.state(candidate, voltage_V=voltage)
                counts["accounting"] += 1
                accounting = multiple_ionic_step_accounting(step, candidate)
                for domain in device.domains:
                    name = domain.domain_layer_id
                    for species in ion_mobility_partition(domain)[1]:
                        index = device.domain_ids.index(name)
                        before = getattr(states[-1].domain_component_states[index], species + "_ion_density_m3")
                        after = getattr(next_state.domain_component_states[index], species + "_ion_density_m3")
                        reference = getattr(domain, species + "_ion_reference_m3")
                        block = device.ion_species_slices[name][species]
                        values = accounting["domains"][name]["species"][species]
                        values["immobile_profile"] = {
                            "compiled_reference_m3": reference, "previous_density_m3": before, "new_density_m3": after,
                            "physical_density_increment_m3": values["physical_inventory_increment_m2"] / domain.ion_geometry.cell_measures_m,
                            "rounded_density_increment_m3": after - before,
                            "profile_relative_error": float(np.max(abs(after - reference) / reference)),
                            "step_density_relative_change": float(np.max(abs(after - before) / before)),
                            "coordinate_increment": candidate[block] - states[-1].u[block],
                        }
                checkpoint()
                accepted, termination = newton.accepted, newton.termination.value
                if not _finite_accounting(accounting):
                    accepted, termination = False, "nonfinite_accounting"
                residual_norm = norm(settings.scaling.row_weights * accounting["backward_euler_residual"])
                checks_list = [GateOutcome("full_native_BE_norm", residual_norm <= 1.0, residual_norm, 1.0)]
                for name, record in accounting["domains"].items():
                    for species, values in record["species"].items():
                        value = float(values["balance_l1_m2_s"])
                        checks_list.append(GateOutcome(name + "." + species + ".all_native_balances",
                            value <= options.ion_balance_tolerance_m2_s, value, options.ion_balance_tolerance_m2_s))
                current_limit = initial.steady_controls.electronic.current_spread_tolerance_A_m2
                for key in ("maxwell_spread_A_m2", "maxwell_identity_error_A_m2"):
                    value = float(np.max(abs(np.asarray(accounting[key]))))
                    checks_list.append(GateOutcome(key, value <= current_limit, value, current_limit))
                checks = tuple(checks_list)
                if accepted and not all(check.passed for check in checks):
                    accepted, termination = False, "original_time_accounting_rejected"
            except Exception as exc:
                accepted = False
                termination = "wall_budget_exhausted" if isinstance(exc, TimeoutError) else "evaluation_error"
                error = f"{type(exc).__name__}: {str(exc)[:500]}"
            attempts.append(MultipleIonicTransientAttempt(start, end, voltage, light, accepted, termination,
                seed, candidate, newton, accounting, checks, counts.copy(), time.monotonic() - began, error))
            for key in total:
                total[key] += counts[key]
            if not accepted:
                break
            assert next_state is not None
            states.append(next_state)
            times.append(end)
        recorded = {"numerical": asdict(options), "stopping_norm": "max(original_electronic_norm,each_domain_species_L1/M_domain)",
            "operator": "physical_backward_euler_all_native_rows", "inventory_row_replacement": False,
            "zero_flux_time_gate": False, "voltage_seed": "original_harmonic_lift_preserving_populations",
            "current_accuracy_A_m2": initial.steady_controls.electronic.current_spread_tolerance_A_m2}
        provenance = {"source_path": initial.source_path, "source_sha256": initial.source_sha256,
            "state_sha256": state_digest(admitted.u), "steady_controls": asdict(initial.steady_controls),
            "generation_scale": generation, "generation_admission": "explicit_dark_G0_revalidation",
            "admission_checks": admission_checks}
        if any(ion_mobility_partition(domain)[1] for domain in device.domains):
            provenance["immobile_profile_binding"] = {
                domain.domain_layer_id: immobile_profile_binding(domain)
                for domain in device.domains if ion_mobility_partition(domain)[1]}
        return MultipleIonicTransientResult(protocol, np.asarray(times), tuple(states), tuple(attempts), provenance,
            device.identity, device.original_prepared_sha256, device.binding_manifest, recorded, total,
            time.monotonic() - started)


def write_multiple_ionic_transient(result: MultipleIonicTransientResult, path: str | Path) -> str:
    output = Path(path)
    if output.exists():
        raise FileExistsError("preserve the original history; no replacement or retry")
    output.parent.mkdir(parents=True, exist_ok=True)
    document = _encode({"schema": "solarlab.multiple_ionic_transient.v1", "completed": result.completed, "result": result})
    output.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    return hashlib.sha256(output.read_bytes()).hexdigest()


def read_multiple_ionic_transient(path: str | Path, sha256: str, *,
    current_identity: JobProvenance | dict[str, Any] | None = None,
    strict_same_source: bool = False,
) -> dict[str, Any]:
    """Verify a saved requested/accepted/failed prefix without invoking any solver."""
    text = Path(path).read_text()
    if hashlib.sha256(text.encode()).hexdigest() != sha256:
        raise ValueError("history SHA256 mismatch")
    document = _decode(json.loads(text))
    if document.get("schema") != "solarlab.multiple_ionic_transient.v1":
        raise ValueError("wrong native multiple-domain history schema")
    result = document["result"]
    requested = result["protocol"]["times_s"]
    states, attempts = result["states"], result["attempts"]
    if (not states or len(attempts) > requested.size - 1
            or not np.array_equal(result["accepted_times_s"], requested[:len(states)])
            or len(states) != 1 + sum(attempt["accepted"] for attempt in attempts)
            or any(not attempt["accepted"] for attempt in attempts[:-1])):
        raise ValueError("invalid requested/accepted/first-failed prefix")
    for index, attempt in enumerate(attempts):
        if attempt["start_s"] != requested[index] or attempt["end_s"] != requested[index + 1]:
            raise ValueError("attempt times disagree with the explicit request")
    if any(state["device_sha256"] != result["compiled_identity"] for state in states):
        raise ValueError("saved history contains an unbound state")
    completed = len(states) == requested.size and all(attempt["accepted"] for attempt in attempts)
    if document["completed"] != completed or result["solver_controls"]["inventory_row_replacement"] is not False:
        raise ValueError("completion/native-row contract mismatch")
    document["provenance_comparison"] = result_provenance(result, current_identity,
        strict_same_source=strict_same_source, expected_inputs=None if result.get("job_provenance") is None else {"compiled_identity": result["compiled_identity"], "initial_sha256": result["initial_provenance"]["source_sha256"],
            "steady_controls": result["initial_provenance"]["steady_controls"], "protocol": _encode(result["protocol"]),
            "controls": result["solver_controls"]["numerical"]})
    return document
