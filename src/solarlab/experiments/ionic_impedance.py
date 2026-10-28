"""Bounded ionic AC sweeps from an accepted, source-bound steady artifact.

Phasors use exp(+i omega t); voltage amplitude is a linear normalization.
Mobile and explicitly frozen-ion responses retain the same operating ionic
profile. Result reading is checksum-checked data access without a solve.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from solarlab.equations.ionic_1d import CompiledIonicDevice1D
from solarlab.experiments.ionic_transient import IonicTransientInitial, _admit_evidence, _same_state
from solarlab.solvers.linear.frequency_domain import FrequencyOptions, FrequencySolution, solve_frequency_systems
from solarlab.solvers.nonlinear.problem import GateOutcome, state_digest

__all__ = [
    "IonicImpedanceOptions", "IonicImpedanceAttempt", "IonicImpedanceResult",
    "solve_ionic_impedance", "write_ionic_impedance_result", "read_ionic_impedance_result",
]

RESULT_SCHEMA = "solarlab.ionic_impedance.v1"


@dataclass(frozen=True)
class IonicImpedanceOptions:
    linear: FrequencyOptions = field(default_factory=FrequencyOptions)
    current_absolute_tolerance_A_m2_V: float = 1e-8
    current_relative_tolerance: float = 1e-10
    inventory_relative_tolerance_per_V: float = 1e-12
    wall_s: float = 30.0

    def __post_init__(self) -> None:
        if not isinstance(self.linear, FrequencyOptions):
            raise ValueError("explicit FrequencyOptions are required")
        for name in ("current_absolute_tolerance_A_m2_V", "current_relative_tolerance",
                     "inventory_relative_tolerance_per_V", "wall_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class IonicImpedanceAttempt:
    frequency_Hz: float
    accepted: bool
    termination: str
    linear_solution: FrequencySolution | None
    observation: dict[str, Any] | None
    checks: tuple[GateOutcome, ...]
    wall_s: float
    error: str | None = None
    coordinate_representation: str | None = None


@dataclass(frozen=True)
class IonicImpedanceResult:
    requested_frequencies_Hz: np.ndarray
    voltage_phasor_V: complex
    ion_mode: str
    attempts: tuple[IonicImpedanceAttempt, ...]
    initial_provenance: dict[str, Any]
    compiled_identity: str
    original_prepared_identity: str
    binding_manifest: dict[str, Any]
    source_sha256: dict[str, str]
    solver_controls: dict[str, Any]
    linear_solver_calls: int
    wall_s: float
    phasor_convention: str = "exp(+i omega t)"
    physical_certificate_status: str = "not_evaluated"

    def __post_init__(self) -> None:
        frequencies = np.array(self.requested_frequencies_Hz, dtype=float, copy=True)
        frequencies.setflags(write=False)
        object.__setattr__(self, "requested_frequencies_Hz", frequencies)

    @property
    def completed(self) -> bool:
        return (len(self.attempts) == self.requested_frequencies_Hz.size
                and all(attempt.accepted for attempt in self.attempts))

    @property
    def failed_attempts(self) -> tuple[IonicImpedanceAttempt, ...]:
        return tuple(attempt for attempt in self.attempts if not attempt.accepted)


def _frequencies(values: ArrayLike, controls: FrequencyOptions) -> np.ndarray:
    supplied = np.asarray(values)
    if np.iscomplexobj(supplied):
        raise ValueError("frequencies must be real Hz")
    result = np.asarray(supplied, dtype=float)
    if (result.ndim != 1 or result.size == 0 or result.size > controls.max_frequencies
            or not np.all(np.isfinite(result)) or np.any(result < 0.0)
            or np.unique(result).size != result.size):
        raise ValueError("frequencies must be a bounded distinct list of finite nonnegative Hz")
    return result.copy()


def _source_pins() -> dict[str, str]:
    names = (
        __name__, "solarlab.equations.ionic_small_signal", "solarlab.equations.ionic_1d",
        "solarlab.equations.ionic_closure",
        "solarlab.equations.dc_1d", "solarlab.device.compile_ionic_1d",
        "solarlab.physics.ions.migration", "solarlab.physics.ions.shared_site",
        "solarlab.physics.ions.finite_site", "solarlab.discretization.bernoulli",
        "solarlab.physics.electrostatics.poisson", "solarlab.physics.electrostatics.charge",
        "solarlab.physics.transport.electrochemical_flux", "solarlab.physics.recombination.bulk",
        "solarlab.physics.recombination.reaction", "solarlab.discretization.stable_bernoulli",
        "solarlab.physics.contacts.boundary_flux", "solarlab.solvers.linear.frequency_domain",
        "solarlab.experiments.ionic_transient", "solarlab.solvers.nonlinear.ionic_1d",
    )
    result = {}
    for name in names:
        path = getattr(sys.modules.get(name), "__file__", None)
        if path is None:
            raise RuntimeError(f"ionic AC source cannot be pinned: {name}")
        result[name] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    optional = "solarlab.experiments.impedance"
    if optional in sys.modules:
        path = getattr(sys.modules[optional], "__file__", None)
        if path is None:
            raise RuntimeError(f"ionic AC source cannot be pinned: {optional}")
        result[optional] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return result


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite(item) for item in value)
    if isinstance(value, (np.ndarray, complex, float, int, np.number)):
        return bool(np.all(np.isfinite(value)))
    return True


def _checks(
    compiled: CompiledIonicDevice1D, observed: dict[str, Any], linear: FrequencySolution,
    voltage: complex, ion_mode: str, controls: IonicImpedanceOptions,
) -> tuple[GateOutcome, ...]:
    from solarlab.equations.ionic_closure import declared_immobile_profiles

    def check(name: str, value: float, limit: float) -> GateOutcome:
        return GateOutcome(name, math.isfinite(value) and value <= limit, value, limit,
                           "prospective numerical AC control; physical status not_evaluated")

    current = np.asarray(observed["total_current_density_A_m2"], dtype=complex)
    if current.shape != (compiled.node_count + 1,):
        raise ValueError("ionic AC accounting requires both external and all internal current faces")
    normalized = current / voltage
    spread = float(np.max(abs(normalized - normalized[0])))
    scale = float(np.max(abs(normalized)))
    results = [
        check("linear_system_backward_error", linear.componentwise_backward_error, controls.linear.relative_tolerance),
        check("original_linear_rows_backward_error", float(observed["original_componentwise_backward_error"]),
              controls.linear.relative_tolerance),
        check("all_face_Maxwell_spread_A_m2_V", spread,
              controls.current_absolute_tolerance_A_m2_V + controls.current_relative_tolerance * scale),
    ]
    inventories, ratios = {}, {}
    immobile_profiles = declared_immobile_profiles(compiled)
    for species in compiled.ion_species_names:
        density = np.asarray(observed["ion_density_perturbation_m3"][species], dtype=complex)
        if density.shape != (compiled.ion_node_count,):
            raise ValueError("ionic AC density perturbation must cover each species domain")
        weighted = compiled.ion_geometry.cell_measures_m * density
        inventory = complex(math.fsum(weighted.real.tolist()), math.fsum(weighted.imag.tolist()))
        target = math.fsum((compiled.ion_geometry.cell_measures_m
                           * getattr(compiled, f"{species}_ion_reference_m3")).tolist())
        ratio = abs(inventory) / (target * abs(voltage))
        inventories[species], ratios[species] = inventory, ratio
        results.append(check(f"{species}_inventory_relative_per_V", ratio, controls.inventory_relative_tolerance_per_V))
        if species in immobile_profiles:
            local_ratio = float(np.max(abs(density) / immobile_profiles[species])) / abs(voltage)
            results.append(check(f"{species}_immobile_density_relative_per_V", local_ratio,
                                 controls.inventory_relative_tolerance_per_V))
        if ion_mode == "frozen":
            results.append(check(f"{species}_frozen_density_response", float(np.max(abs(density))), 0.0))
    if ion_mode == "frozen":
        ionic_current = np.asarray(observed["ionic_conduction_current_density_A_m2"], dtype=complex)
        results.append(check("frozen_ionic_current_response", float(np.max(abs(ionic_current))), 0.0))
    observed["consumer_inventory_perturbation_m2"] = inventories
    observed["consumer_inventory_relative_per_V"] = ratios
    observed["consumer_Maxwell_spread_A_m2_V"] = spread
    return tuple(results)


class _SweepWallLimit(RuntimeError):
    pass


def solve_ionic_impedance(
    compiled: CompiledIonicDevice1D, initial: IonicTransientInitial, frequencies_Hz: ArrayLike,
    *, ion_mode: str = "mobile", voltage_phasor_V: complex = 1.0,
    options: IonicImpedanceOptions | None = None,
) -> IonicImpedanceResult:
    """One bounded sweep using the existing complex solver and a pinned steady input.

    The shared steady-artifact reader performs admission only. No nonlinear
    solve, time advance or ionic relaxation is run here. A frequency failure
    stops the sweep and preserves its preceding responses and failed attempt.
    The wall limit is cooperative between bounded native operations.
    """
    started = time.monotonic()
    controls = IonicImpedanceOptions() if options is None else options
    if not isinstance(controls, IonicImpedanceOptions):
        raise ValueError("IonicImpedanceOptions are required")
    requested = _frequencies(frequencies_Hz, controls.linear)
    voltage = complex(voltage_phasor_V)
    if not np.isfinite(voltage) or voltage == 0.0:
        raise ValueError("voltage normalization must be finite and nonzero")
    if ion_mode not in {"mobile", "frozen"}:
        raise ValueError("ion_mode must be explicitly mobile or frozen")
    if not isinstance(compiled, CompiledIonicDevice1D) or not isinstance(initial, IonicTransientInitial):
        raise ValueError("an actual ionic device and pinned accepted steady input are required")
    if any(channel.kind != "fixed_reservoir" for channel in compiled.electronic_component.channels):
        raise ValueError("ionic AC currently supports only fixed-reservoir carrier contacts")

    # Kept lazy so checksum-checked result reading never loads the AC operator.
    from solarlab.equations.ionic_small_signal import build_ionic_small_signal

    pins = _source_pins()
    admitted, generation, initial_checks = _admit_evidence(
        compiled, initial.evidence_json, initial.source_sha256, initial.steady_controls,
    )
    if not _same_state(initial.state, admitted) or initial.generation_scale != generation:
        raise ValueError("ionic AC initial object disagrees with its pinned steady evidence")
    operator = build_ionic_small_signal(compiled, admitted, generation_scale=generation)
    deadline = started + controls.wall_s
    attempts = []
    calls = 0

    def checkpoint() -> None:
        if time.monotonic() >= deadline:
            raise _SweepWallLimit("ionic AC sweep wall budget exhausted; no retry")
        if _source_pins() != pins:
            raise RuntimeError("ionic AC source changed during the sweep")

    for supplied in requested:
        frequency = float(supplied)
        attempt_started = time.monotonic()
        solved, observed, error = None, None, None
        representation = None
        checks: tuple[GateOutcome, ...] = ()
        try:
            checkpoint()
            system = operator.system(frequency, voltage_phasor_V=voltage, ion_mode=ion_mode)
            representation = system.coordinate_form
            if not isinstance(representation, str) or not representation:
                raise ValueError("ionic AC solve requires an explicit coordinate representation")
            expected_rows = tuple(range(compiled.state_size if ion_mode == "mobile" else 3 * compiled.node_count))
            if tuple(system.original_rows_enforced) != expected_rows:
                raise ValueError("ionic AC original-row coverage disagrees with its response model")
            checkpoint()
            calls += 1
            solved = solve_frequency_systems(
                system.jacobian, system.storage_jacobian, system.rhs, [frequency],
                options=controls.linear,
                fixed_coordinates=system.fixed_coordinates,
                rhs_rate=system.rhs_rate,
            )[0]
            checkpoint()
            observed = operator.observe(solved.solution, frequency, voltage_phasor_V=voltage, ion_mode=ion_mode)
            if not isinstance(observed, dict) or not _finite(observed):
                raise ValueError("ionic AC accounting is malformed or nonfinite")
            observed_rows = observed["original_rows_enforced"]
            if not isinstance(observed_rows, (tuple, list)) or tuple(observed_rows) != expected_rows:
                raise ValueError("ionic AC observation omits required original rows")
            if (observed["coordinate_form"] != representation
                    or not np.array_equal(observed["solve_coordinate_perturbation"], solved.solution)):
                raise ValueError("ionic AC observation must retain its exact raw solve coordinates")
            observed["inventory_constraint_rows"] = system.constraint_rows
            if system.immobile_constraint_rows:
                observed["immobile_constraint_rows"] = system.immobile_constraint_rows
            checks = _checks(compiled, observed, solved, voltage, ion_mode, controls)
            checkpoint()
            accepted = all(item.passed for item in checks)
            termination = "converged" if accepted else "numerical_check_failed"
        except Exception as exc:
            accepted = False
            termination = "wall_budget_exhausted" if isinstance(exc, _SweepWallLimit) else "evaluation_error"
            error = f"{type(exc).__name__}: {str(exc)[:400]}"
        attempts.append(IonicImpedanceAttempt(frequency, accepted, termination, solved, observed, checks,
                                             time.monotonic() - attempt_started, error, representation))
        if not accepted:
            break
    provenance = {"source_path": initial.source_path, "source_sha256": initial.source_sha256,
                  "state_sha256": state_digest(admitted.u), "voltage_V": admitted.voltage_V,
                  "generation_scale": generation, "steady_controls": asdict(initial.steady_controls),
                  "initial_admission_checks": initial_checks}
    recorded = asdict(controls)
    recorded["voltage_normalization"] = "linear derivative; not a finite-amplitude voltage experiment"
    return IonicImpedanceResult(requested, voltage, ion_mode, tuple(attempts), provenance, compiled.identity,
        compiled.original_prepared_sha256, compiled.binding_manifest, pins, recorded, calls, time.monotonic() - started)


def _encoded(value: Any) -> Any:
    """Lossless binary64 JSON roundtrip, with explicit failed nonfinite values."""
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _encoded(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, np.ndarray):
        return {"__array__": _encoded(value.tolist()), "dtype": value.dtype.str, "shape": list(value.shape)}
    if isinstance(value, np.generic):
        return _encoded(value.item())
    if isinstance(value, complex):
        return {"__complex__": [_encoded(value.real), _encoded(value.imag)]}
    if isinstance(value, float) and not math.isfinite(value):
        return {"__nonfinite__": "nan" if math.isnan(value) else "inf" if value > 0 else "-inf"}
    if isinstance(value, dict):
        return {key: _encoded(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_encoded(item) for item in value]
    return value


def _decoded(value: Any) -> Any:
    if isinstance(value, list):
        return [_decoded(item) for item in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {"__nonfinite__"}:
        if value["__nonfinite__"] not in {"nan", "inf", "-inf"}:
            raise ValueError("invalid nonfinite encoding")
        return float(value["__nonfinite__"])
    if set(value) == {"__complex__"}:
        components = value["__complex__"]
        if not isinstance(components, list) or len(components) != 2:
            raise ValueError("invalid complex encoding")
        return complex(_decoded(components[0]), _decoded(components[1]))
    if set(value) == {"__array__", "dtype", "shape"}:
        dtype = np.dtype(value["dtype"])
        if dtype.kind not in "biufc":
            raise ValueError("result arrays require numerical dtypes")
        shape = value["shape"]
        if not isinstance(shape, list) or any(type(size) is not int or size < 0 for size in shape):
            raise ValueError("invalid result array shape")
        result = np.asarray(_decoded(value["__array__"]), dtype=dtype).reshape(shape)
        result.setflags(write=False)
        return result
    return {key: _decoded(item) for key, item in value.items()}


def write_ionic_impedance_result(result: IonicImpedanceResult, path: str | Path) -> str:
    """Write one result exclusively and return the hash of its complete bytes."""
    if not isinstance(result, IonicImpedanceResult):
        raise ValueError("an actual IonicImpedanceResult is required")
    record = {"schema": RESULT_SCHEMA, "completed": result.completed, "result": _encoded(result)}
    raw = (json.dumps(record, indent=2, allow_nan=False) + "\n").encode()
    with Path(path).open("xb") as output:
        output.write(raw)
    return hashlib.sha256(raw).hexdigest()


def read_ionic_impedance_result(path: str | Path, sha256: str) -> dict[str, Any]:
    """Read and validate recorded data without an operator, model rebuild or solve."""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("ionic impedance artifact SHA-256 mismatch")
    try:
        record = _decoded(json.loads(raw))
        if record["schema"] != RESULT_SCHEMA:
            raise ValueError("unsupported ionic impedance result schema")
        result = record["result"]
        controls = FrequencyOptions(**result["solver_controls"]["linear"])
        requested = _frequencies(result["requested_frequencies_Hz"], controls)
        attempts = result["attempts"]
        voltage = complex(result["voltage_phasor_V"])
        if (result["ion_mode"] not in {"mobile", "frozen"}
                or not np.isfinite(voltage) or voltage == 0.0
                or result["phasor_convention"] != "exp(+i omega t)"
                or result["physical_certificate_status"] != "not_evaluated"
                or len(attempts) > requested.size):
            raise ValueError("inconsistent ionic impedance result metadata")
        for index, attempt in enumerate(attempts):
            if (attempt["frequency_Hz"] != requested[index] or type(attempt["accepted"]) is not bool
                    or (not attempt["accepted"] and index != len(attempts) - 1)):
                raise ValueError("ionic impedance attempts must retain the requested prefix and first failure")
            if attempt["accepted"]:
                linear = attempt["linear_solution"]
                error = linear["componentwise_backward_error"]
                if (attempt["termination"] != "converged" or attempt["observation"] is None
                        or not math.isfinite(error) or error > controls.relative_tolerance
                        or not attempt["checks"] or any(
                            check["passed"] is not True or not math.isfinite(check["value"])
                            or check["value"] > check["limit"] for check in attempt["checks"])):
                    raise ValueError("contradictory accepted ionic impedance evidence")
                observed = attempt["observation"]
                representation = attempt["coordinate_representation"]
                if (not isinstance(representation, str) or not representation
                        or observed["coordinate_form"] != representation
                        or not np.array_equal(observed["solve_coordinate_perturbation"], linear["solution"])
                        or not _finite(observed)):
                    raise ValueError("ionic impedance evidence lost its finite raw solve representation")
        completed = len(attempts) == requested.size and all(item["accepted"] for item in attempts)
        if record["completed"] is not completed:
            raise ValueError("ionic impedance completion contradicts its retained attempts")
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError("incomplete or malformed ionic impedance artifact") from exc
    return record
