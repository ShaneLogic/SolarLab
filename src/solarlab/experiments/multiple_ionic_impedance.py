"""One bounded two-domain AC sweep from admitted source-bound dark DC."""

from __future__ import annotations

from solarlab.reproducibility.job_provenance import (
    JobProvenance, admitted_source_pins, current_provenance, job_provenance, result_provenance,
)

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from solarlab.equations.multiple_ionic_1d import CompiledMultipleIonicDevice1D
from solarlab.experiments.ionic_impedance import IonicImpedanceAttempt, IonicImpedanceOptions, _decoded, _encoded, _finite, _frequencies
from solarlab.experiments.multiple_ionic_transient import MultipleIonicTransientInitial, _admit, _same_record
from solarlab.experiments.transient_results import _decode, _encode
from solarlab.solvers.linear import FrequencyOptions, FrequencySolution, solve_frequency_systems
from solarlab.solvers.nonlinear.problem import GateOutcome, state_digest

if TYPE_CHECKING:
    from solarlab.equations.assembly.small_signal import AssembledMultipleIonicSmallSignal1D
    from solarlab.equations.multiple_ionic_small_signal import MultipleIonicSmallSignal1D

__all__ = ["MultipleIonicImpedanceResult", "solve_multiple_ionic_impedance",
           "write_multiple_ionic_impedance_result", "read_multiple_ionic_impedance_result"]
SCHEMA = "solarlab.multiple_ionic_impedance.v1"


@dataclass(frozen=True)
class MultipleIonicImpedanceResult:
    requested_frequencies_Hz: np.ndarray
    voltage_phasor_V: complex
    attempts: tuple[IonicImpedanceAttempt, ...]
    initial_provenance: dict[str, Any]
    compiled_identity: str
    original_prepared_identity: str
    binding_manifest: dict[str, Any]
    source_sha256: dict[str, str]
    solver_controls: dict[str, Any]
    linear_solver_calls: int
    wall_s: float
    ion_mode: str = "mobile"
    phasor_convention: str = "exp(+i omega t)"
    physical_certificate_status: str = "not_evaluated"

    job_provenance: dict[str, Any] | None = field(default_factory=current_provenance)

    def __post_init__(self) -> None:
        values = np.array(self.requested_frequencies_Hz, dtype=float, copy=True)
        values.setflags(write=False)
        object.__setattr__(self, "requested_frequencies_Hz", values)

    @property
    def completed(self) -> bool:
        return len(self.attempts) == self.requested_frequencies_Hz.size and all(attempt.accepted for attempt in self.attempts)

    @property
    def failed_attempts(self) -> tuple[IonicImpedanceAttempt, ...]:
        return tuple(attempt for attempt in self.attempts if not attempt.accepted)




def _checks(device: CompiledMultipleIonicDevice1D, observed: dict[str, Any], linear: FrequencySolution,
            voltage: complex, options: IonicImpedanceOptions) -> tuple[GateOutcome, ...]:
    def check(name: str, value: float, limit: float) -> GateOutcome:
        return GateOutcome(name, math.isfinite(value) and value <= limit, value, limit,
                           "existing reviewed AC numerical control; qualification not_evaluated")
    current = np.asarray(observed["total_current_density_A_m2"], dtype=complex)
    if current.shape != (device.node_count + 1,) or set(observed["domains"]) != set(device.domain_ids):
        raise ValueError("AC current/domain observation must cover the full assembled device")
    normalized = current / voltage
    spread, scale = float(np.max(abs(normalized - normalized[0]))), float(np.max(abs(normalized)))
    results = [check("linear_system_backward_error", linear.componentwise_backward_error, options.linear.relative_tolerance),
        check("original_linear_rows_backward_error", observed["original_componentwise_backward_error"], options.linear.relative_tolerance),
        check("all_face_Maxwell_spread_A_m2_V", spread,
              options.current_absolute_tolerance_A_m2_V + options.current_relative_tolerance * scale)]
    inventories: dict[str, dict[str, complex]] = {}
    ratios: dict[str, dict[str, float]] = {}
    for domain in device.domains:
        name = domain.domain_layer_id
        values = observed["domains"][name]["species"]["positive"]
        density = np.asarray(values["density_perturbation_m3"], dtype=complex)
        if density.shape != (domain.ion_node_count,):
            raise ValueError("every original domain population must retain its physical nodes")
        weighted = domain.ion_geometry.cell_measures_m * density
        inventory = complex(math.fsum(weighted.real.tolist()), math.fsum(weighted.imag.tolist()))
        ratio = abs(inventory) / (device.reference_inventories_m2[name]["positive"] * abs(voltage))
        inventories[name], ratios[name] = {"positive": inventory}, {"positive": ratio}
        results.append(check(name + ".positive_inventory_relative_per_V", ratio, options.inventory_relative_tolerance_per_V))
    observed["consumer_inventory_perturbation_m2"], observed["consumer_inventory_relative_per_V"] = inventories, ratios
    observed["consumer_Maxwell_spread_A_m2_V"] = spread
    return tuple(results)


def solve_multiple_ionic_impedance(device: CompiledMultipleIonicDevice1D, initial: MultipleIonicTransientInitial,
                                  frequencies_Hz: ArrayLike, *, voltage_phasor_V: complex = 1.0,
                                  ion_mode: str = "mobile", options: IonicImpedanceOptions | None = None) -> MultipleIonicImpedanceResult:
    """No DC solve or retry; preserve all requested/accepted/first-failed frequencies."""
    started = time.monotonic()
    from solarlab.equations.assembly.small_signal import assembled_multiple_ionic_small_signal
    from solarlab.equations.implementation import observation_binding, provenance_binding, use_assembled
    from solarlab.equations.multiple_ionic_small_signal import scope, build_multiple_ionic_small_signal
    scope(device, ion_mode)
    controls = IonicImpedanceOptions() if options is None else options
    if not isinstance(controls, IonicImpedanceOptions) or not isinstance(initial, MultipleIonicTransientInitial):
        raise ValueError("explicit reviewed controls and pinned accepted multiple-domain DC evidence are required")
    requested = _frequencies(frequencies_Hz, controls.linear)
    voltage = complex(voltage_phasor_V)
    if not np.isfinite(voltage) or voltage == 0.0:
        raise ValueError("voltage normalization must be finite and nonzero")
    with job_provenance({"compiled_identity": device.identity, "initial_sha256": initial.source_sha256, "steady_controls": asdict(initial.steady_controls), "frequencies_Hz": requested.tolist(), "voltage_phasor_V": [voltage.real, voltage.imag], "ion_mode": ion_mode, "controls": asdict(controls)},
            **provenance_binding("MultipleIonicSmallSignal1D", device,
                inventory_row_node=initial.steady_controls.inventory_row_node)):
        pins = admitted_source_pins()
        state, generation, initial_checks = _admit(device, initial.evidence_json, initial.source_sha256, initial.steady_controls)
        if not _same_record(_decode(_encode(state)), _decode(_encode(initial.state))) or initial.generation_scale != generation:
            raise ValueError("initial object disagrees with its exact admitted evidence")
        operator: MultipleIonicSmallSignal1D | AssembledMultipleIonicSmallSignal1D
        if use_assembled():
            operator = assembled_multiple_ionic_small_signal(device, state, generation_scale=generation)
        else:
            operator = build_multiple_ionic_small_signal(device, state, generation_scale=generation)
            pins.update(operator.source_sha256)
        deadline = started + controls.wall_s
        attempts, calls = [], 0
        def checkpoint() -> None:
            if time.monotonic() >= deadline:
                raise TimeoutError("multiple-domain AC wall budget exhausted; no retry")
        for supplied in requested:
            frequency, began = float(supplied), time.monotonic()
            solved, observed, error, representation = None, None, None, None
            checks: tuple[GateOutcome, ...] = ()
            try:
                checkpoint()
                system = operator.system(frequency, voltage_phasor_V=voltage)
                representation = system.coordinate_form
                if system.original_rows_enforced != tuple(range(device.state_size)):
                    raise ValueError("AC system omitted an original two-domain balance row")
                calls += 1
                solved = solve_frequency_systems(system.jacobian, system.storage_jacobian, system.rhs, [frequency],
                    options=controls.linear, fixed_coordinates=system.fixed_coordinates, rhs_rate=system.rhs_rate)[0]
                checkpoint()
                observed = operator.observe(solved.solution, frequency, voltage_phasor_V=voltage,
                                            **observation_binding(operator, system))
                if (not _finite(observed) or observed["original_rows_enforced"] != tuple(range(device.state_size))
                        or not np.array_equal(observed["solve_coordinate_perturbation"], solved.solution)
                        or observed["coordinate_form"] != representation):
                    raise ValueError("AC observation lost finite original rows or raw solve coordinates")
                observed["inventory_constraint_rows"] = system.constraint_rows
                checks = _checks(device, observed, solved, voltage, controls)
                checkpoint()
                accepted = all(item.passed for item in checks)
                termination = "converged" if accepted else "numerical_check_failed"
            except Exception as exc:
                accepted = False
                termination = "wall_budget_exhausted" if isinstance(exc, TimeoutError) else "evaluation_error"
                error = f"{type(exc).__name__}: {str(exc)[:400]}"
            attempts.append(IonicImpedanceAttempt(frequency, accepted, termination, solved, observed, checks,
                                                  time.monotonic() - began, error, representation))
            if not accepted:
                break
        provenance = {"source_path": initial.source_path, "source_sha256": initial.source_sha256,
            "state_sha256": state_digest(state.u), "voltage_V": state.voltage_V, "generation_scale": generation,
            "steady_controls": asdict(initial.steady_controls), "initial_admission_checks": initial_checks}
        recorded = asdict(controls)
        recorded["voltage_normalization"] = "linear derivative; not a finite-amplitude voltage experiment"
        return MultipleIonicImpedanceResult(requested, voltage, tuple(attempts), provenance, device.identity,
            device.original_prepared_sha256, device.binding_manifest, pins, recorded, calls, time.monotonic() - started)


def write_multiple_ionic_impedance_result(result: MultipleIonicImpedanceResult, path: str | Path) -> str:
    if not isinstance(result, MultipleIonicImpedanceResult):
        raise ValueError("an actual multiple-domain AC result is required")
    record = {"schema": SCHEMA, "completed": result.completed, "result": _encoded(result)}
    raw = (json.dumps(record, indent=2, allow_nan=False) + "\n").encode()
    with Path(path).open("xb") as stream:
        stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


def read_multiple_ionic_impedance_result(path: str | Path, sha256: str, *,
    current_identity: JobProvenance | dict[str, Any] | None = None,
    strict_same_source: bool = False,
) -> dict[str, Any]:
    """Checksum-checked prefix validation without compiling, solving or an operator."""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("multiple-domain AC artifact SHA-256 mismatch")
    try:
        record = _decoded(json.loads(raw))
        if record["schema"] != SCHEMA:
            raise ValueError("unsupported multiple-domain AC schema")
        result = record["result"]
        numerical = {name: result["solver_controls"][name] for name in (
            "current_absolute_tolerance_A_m2_V", "current_relative_tolerance", "inventory_relative_tolerance_per_V", "wall_s")}
        controls = IonicImpedanceOptions(linear=FrequencyOptions(**result["solver_controls"]["linear"]), **numerical)
        requested, attempts = _frequencies(result["requested_frequencies_Hz"], controls.linear), result["attempts"]
        voltage = complex(result["voltage_phasor_V"])
        if (result["ion_mode"] != "mobile" or len(attempts) > requested.size
                or result["physical_certificate_status"] != "not_evaluated"
                or result["phasor_convention"] != "exp(+i omega t)" or not np.isfinite(voltage) or voltage == 0.0):
            raise ValueError("inconsistent multiple-domain AC metadata")
        for index, attempt in enumerate(attempts):
            if (attempt["frequency_Hz"] != requested[index] or type(attempt["accepted"]) is not bool
                    or (not attempt["accepted"] and index != len(attempts) - 1)):
                raise ValueError("AC attempts must retain the requested prefix and first failure")
            if attempt["accepted"]:
                linear, observed = attempt["linear_solution"], attempt["observation"]
                if (attempt["termination"] != "converged" or observed is None or linear is None
                        or not math.isfinite(linear["componentwise_backward_error"])
                        or linear["componentwise_backward_error"] > controls.linear.relative_tolerance
                        or not attempt["checks"] or not all(item["passed"] is True and math.isfinite(item["value"])
                            and item["value"] <= item["limit"] for item in attempt["checks"])):
                    raise ValueError("contradictory accepted multiple-domain AC evidence")
                if (not _finite(observed) or observed["coordinate_form"] != attempt["coordinate_representation"]
                        or not np.array_equal(observed["solve_coordinate_perturbation"], linear["solution"])):
                    raise ValueError("AC evidence lost its finite raw solve representation")
        if record["completed"] != (len(attempts) == requested.size and all(item["accepted"] for item in attempts)):
            raise ValueError("AC completion contradicts its retained attempts")
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError("incomplete or malformed multiple-domain AC artifact") from exc
    record["provenance_comparison"] = result_provenance(result, current_identity,
        strict_same_source=strict_same_source, expected_inputs=None if result.get("job_provenance") is None else {"compiled_identity": result["compiled_identity"], "initial_sha256": result["initial_provenance"]["source_sha256"],
            "steady_controls": result["initial_provenance"]["steady_controls"], "frequencies_Hz": result["requested_frequencies_Hz"].tolist(),
            "voltage_phasor_V": [voltage.real, voltage.imag], "ion_mode": result["ion_mode"],
            "controls": {k: v for k, v in result["solver_controls"].items() if k != "voltage_normalization"}})
    return record
