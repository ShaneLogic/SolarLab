"""Equilibrated ionic J-V, with explicit guesses and retained first failures.

Each requested voltage gets at most one call to the coupled new-core steady
solver. This experiment has no time, scan-rate or hysteresis interpretation.
Saved data can be read without constructing a device or invoking a solver.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from solarlab.equations.ionic_1d import CompiledIonicDevice1D, IonicState1D
from solarlab.experiments.ionic_transient import IonicTransientInitial, _admit_evidence, _same_state
from solarlab.experiments.jv.steady import SteadyJVMetrics, steady_jv_metrics
from solarlab.experiments.transient_results import _decode, _encode
from solarlab.solvers.nonlinear.ionic_1d import IonicSteadyOptions, IonicSteadyResult, solve_ionic_steady_state
from solarlab.solvers.nonlinear.problem import state_digest

__all__ = ["IonicSteadyJVOptions", "IonicSteadyJVAttempt", "IonicSteadyJVResult",
           "run_ionic_steady_jv", "write_ionic_steady_jv", "read_ionic_steady_jv"]

SCHEMA = "solarlab.ionic_steady_jv.v1"


@dataclass(frozen=True)
class IonicSteadyJVOptions:
    steady: IonicSteadyOptions
    max_points: int = 32
    wall_s: float = 30.0

    def __post_init__(self) -> None:
        if not isinstance(self.steady, IonicSteadyOptions):
            raise ValueError("explicit IonicSteadyOptions are required")
        if type(self.max_points) is not int or self.max_points < 1:
            raise ValueError("max_points must be a positive integer")
        if not math.isfinite(self.wall_s) or self.wall_s <= 0.0:
            raise ValueError("wave wall_s must be finite and positive")


@dataclass(frozen=True)
class IonicSteadyJVAttempt:
    index: int
    voltage_V: float
    accepted: bool
    termination: str
    initial_guess: IonicState1D | None
    seed_provenance: dict[str, Any]
    solution: IonicSteadyResult | None
    solver_called: bool
    wall_s: float
    error: str | None = None


@dataclass(frozen=True)
class IonicSteadyJVResult:
    requested_voltage_V: np.ndarray
    voltage_V: np.ndarray
    current_density_A_m2: np.ndarray
    output_current_A_m2: np.ndarray
    attempts: tuple[IonicSteadyJVAttempt, ...]
    metrics: SteadyJVMetrics
    generation_scale: float
    incident_power_W_m2: float | None
    voltage_polarity: float
    initial_provenance: dict[str, Any]
    compiled_identity: str
    original_prepared_identity: str
    binding_manifest: dict[str, Any]
    source_sha256: dict[str, str]
    source_identity: str
    controls_sha256: str
    request_sha256: str
    solver_controls: dict[str, Any]
    ion_species: tuple[str, ...]
    reference_inventories_m2: dict[str, float]
    ion_weights_m: np.ndarray
    nodes_m: np.ndarray
    solver_calls: int
    evaluation_counts: dict[str, int]
    wall_s: float
    experiment_kind: str = "equilibrated_ionic_steady_jv"
    physical_certificate_status: str = "not_evaluated"

    def __post_init__(self) -> None:
        for name in ("requested_voltage_V", "voltage_V", "current_density_A_m2", "output_current_A_m2",
                     "ion_weights_m", "nodes_m"):
            array = np.array(getattr(self, name), copy=True)
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @property
    def completed(self) -> bool:
        return (len(self.attempts) == self.requested_voltage_V.size
                and all(attempt.accepted for attempt in self.attempts))

    @property
    def failed_attempts(self) -> tuple[IonicSteadyJVAttempt, ...]:
        return tuple(attempt for attempt in self.attempts if not attempt.accepted)


def _voltages(value: ArrayLike, max_points: int) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError("requested voltages must be real")
    result = np.asarray(raw, dtype=float)
    if (result.ndim != 1 or not 1 <= result.size <= max_points or not np.all(np.isfinite(result))
            or (result.size > 1 and not (np.all(np.diff(result) > 0.0) or np.all(np.diff(result) < 0.0)))):
        raise ValueError("requested voltages must be a bounded strictly monotonic finite vector")
    return result.copy()


def _identity(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _sources() -> dict[str, str]:
    names = [__name__, "solarlab.experiments.ionic_transient", "solarlab.experiments.jv.steady",
        "solarlab.experiments.transient_results", "solarlab.experiments.results",
        "solarlab.device.compile_ionic_1d", "solarlab.device.compile_1d", "solarlab.equations.ionic_1d",
        "solarlab.equations.dc_1d", "solarlab.solvers.nonlinear.ionic_1d", "solarlab.solvers.nonlinear.newton",
        "solarlab.solvers.nonlinear.production", "solarlab.physics.ions.migration", "solarlab.physics.ions.finite_site",
        "solarlab.physics.ions.shared_site", "solarlab.physics.transport.electrochemical_flux",
        "solarlab.physics.electrostatics.poisson", "solarlab.physics.electrostatics.charge",
        "solarlab.physics.recombination.bulk", "solarlab.physics.recombination.reaction",
        "solarlab.physics.contacts.exchange", "solarlab.discretization.bernoulli", "solarlab.discretization.stable_bernoulli"]
    if "solarlab.equations.ionic_closure" in sys.modules:
        names.append("solarlab.equations.ionic_closure")
    result = {}
    for name in names:
        filename = importlib.import_module(name).__file__
        if filename is None:
            raise ValueError("steady J-V requires actual source files for provenance")
        result[name] = hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    return result


def _rebase_guess(device: CompiledIonicDevice1D, state: IonicState1D, voltage: float) -> IonicState1D:
    if not _same_state(state, device.state(state.u, voltage_V=state.voltage_V)):
        raise ValueError("previous accepted seed fields disagree with their bound coordinates")
    if state.voltage_V == voltage:
        return state
    values = state.u.copy()
    count = 3 * device.node_count
    values[:count] = device.electronic_component.rebase_coordinates(values[:count], state.voltage_V, voltage)
    return device.state(values, voltage_V=voltage)


def _metrics(voltage: np.ndarray, current: np.ndarray, incident: float | None) -> SteadyJVMetrics:
    reason = ("insufficient_accepted_points" if voltage.size < 2 else
              "zero_voltage_not_bracketed" if not np.min(voltage) <= 0.0 <= np.max(voltage) else None)
    if reason is not None:
        return SteadyJVMetrics(None, None, None, None, None, reason)
    return steady_jv_metrics(voltage, current, incident_power_W_m2=incident)


class _WaveWallLimit(RuntimeError):
    pass


def run_ionic_steady_jv(
    compiled: CompiledIonicDevice1D, voltage_V: ArrayLike, *, generation_scale: float,
    options: IonicSteadyJVOptions, initial_policy: str,
    initial: IonicTransientInitial | None = None, incident_power_W_m2: float | None = None,
) -> IonicSteadyJVResult:
    """One coupled steady solve per point; only an accepted point may seed the next.

    The sole cold/saved policy concerns the first numerical guess. Subsequent
    guesses use the previous accepted full state, rebased by the existing
    electronic reference map while retaining ionic coordinates. The remaining
    wave wall budget caps each solver call; physical controls never change.
    """
    started = time.monotonic()
    if not isinstance(compiled, CompiledIonicDevice1D) or not isinstance(options, IonicSteadyJVOptions):
        raise ValueError("an actual ionic device and explicit steady J-V options are required")
    requested = _voltages(voltage_V, options.max_points)
    generation = float(generation_scale)
    if not math.isfinite(generation) or generation < 0.0:
        raise ValueError("generation_scale must be finite and nonnegative")
    if incident_power_W_m2 is not None and (not math.isfinite(incident_power_W_m2) or incident_power_W_m2 <= 0.0):
        raise ValueError("incident power must be finite and positive when supplied")
    if initial_policy not in {"compiled_cold", "saved_accepted"}:
        raise ValueError("initial_policy must explicitly be compiled_cold or saved_accepted")
    if any(channel.kind != "fixed_reservoir" for channel in compiled.electronic_component.channels):
        raise ValueError("ionic steady J-V currently requires fixed-reservoir contacts")
    domain = compiled.binding_manifest.get("ion_domain")
    if not isinstance(domain, dict) or domain.get("contacts") != ["blocking", "blocking"]:
        raise ValueError("ionic steady J-V requires one declared blocking domain")
    for species in compiled.ion_species_names:
        diffusivity = getattr(compiled, species + "_ion_diffusivity_faces_m2_s")
        if not np.all(np.isfinite(diffusivity)) or np.any(diffusivity <= 0.0):
            raise ValueError("ionic steady J-V requires connected positive-D species; D0 is unsupported here")
    polarity = float(compiled.electronic_component.voltage_polarity)
    if polarity not in {-1.0, 1.0}:
        raise ValueError("terminal voltage polarity must be explicitly signed")
    pins = _sources()
    previous = None
    initial_provenance: dict[str, Any] = {"policy": initial_policy}
    if initial_policy == "saved_accepted":
        if not isinstance(initial, IonicTransientInitial) or initial.steady_controls != options.steady:
            raise ValueError("saved initial requires an accepted artifact and matching declared steady controls")
        previous, initial_generation, checks = _admit_evidence(
            compiled, initial.evidence_json, initial.source_sha256, initial.steady_controls)
        if not _same_state(previous, initial.state) or initial_generation != initial.generation_scale:
            raise ValueError("initial object disagrees with its pinned steady evidence")
        initial_provenance.update(source_path=initial.source_path, source_sha256=initial.source_sha256,
            state=previous, state_sha256=state_digest(previous.u), generation_scale=initial_generation,
            admission_checks=checks)
    elif initial is not None:
        raise ValueError("compiled_cold policy cannot accept a hidden supplied initial state")
    controls = asdict(options)
    request = {"voltage_V": requested.tolist(), "generation_scale": generation,
        "incident_power_W_m2": incident_power_W_m2, "initial_policy": initial_policy,
        "initial_source_sha256": initial_provenance.get("source_sha256"), "controls": controls}
    deadline = started + options.wall_s
    attempts: list[IonicSteadyJVAttempt] = []
    accepted_voltages, currents = [], []
    calls, counts = 0, {"residual": 0, "jacobian": 0}

    def checkpoint() -> None:
        if time.monotonic() >= deadline:
            raise _WaveWallLimit("ionic J-V wave wall budget exhausted; no retry")
        if _sources() != pins:
            raise RuntimeError("ionic J-V source changed during the wave")

    for index, supplied in enumerate(requested):
        voltage = float(supplied)
        attempt_started = time.monotonic()
        seed, solution, error = None, None, None
        called = False
        seed_provenance: dict[str, Any] = {"policy": initial_policy if index == 0 else "previous_accepted"}
        try:
            checkpoint()
            if previous is None:
                seed = compiled.initial_state(voltage_V=voltage)
            else:
                seed_provenance.update(previous_voltage_V=previous.voltage_V,
                    previous_state_sha256=state_digest(previous.u), previous_accepted_point=None if index == 0 else index - 1)
                seed = _rebase_guess(compiled, previous, voltage)
            seed_provenance["rebased_state_sha256"] = state_digest(seed.u)
            checkpoint()
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise _WaveWallLimit("ionic J-V wave wall budget exhausted before solver call")
            effective = replace(options.steady, wall_s=min(options.steady.wall_s, remaining))
            calls += 1
            called = True
            solution = solve_ionic_steady_state(compiled, voltage_V=voltage, generation_scale=generation,
                                               options=effective, initial_state=seed)
            for name, number in solution.evaluation_counts.items():
                counts[name] = counts.get(name, 0) + number
            checkpoint()
            accepted = solution.accepted
            termination, error = solution.termination, solution.error
            if accepted:
                if (solution.state is None or solution.newton is None or not solution.newton.accepted
                        or not solution.checks or not all(check.passed for check in solution.checks)
                        or solution.compiled_identity != compiled.identity
                        or solution.original_prepared_identity != compiled.original_prepared_sha256
                        or solution.binding_manifest != compiled.binding_manifest
                        or solution.solver_controls != asdict(effective)
                        or solution.state.voltage_V != voltage
                        or not _same_state(solution.state, compiled.state(solution.state.u, voltage_V=voltage))
                        or solution.diagnostics is None):
                    raise ValueError("accepted solver result has inconsistent state, controls or provenance")
                current = np.asarray(solution.diagnostics["total_current_density_A_m2"], dtype=float)
                if current.shape != (compiled.node_count + 1,) or not np.all(np.isfinite(current)):
                    raise ValueError("accepted J-V point must retain all internal and external current faces")
                currents.append(current.copy())
                accepted_voltages.append(voltage)
                previous = solution.state
        except Exception as exc:
            accepted = False
            termination = "wall_budget_exhausted" if isinstance(exc, _WaveWallLimit) else "evaluation_error"
            error = f"{type(exc).__name__}: {str(exc)[:400]}"
        attempts.append(IonicSteadyJVAttempt(index, voltage, accepted, termination, seed,
            seed_provenance, solution, called, time.monotonic() - attempt_started, error))
        if not accepted:
            break
    voltage_array: np.ndarray = np.asarray(accepted_voltages, dtype=float)
    current_array = np.asarray(currents, dtype=float).reshape((-1, compiled.node_count + 1))
    output = -polarity * current_array[:, 0]
    targets = {"positive": compiled.reference_inventory_m2}
    if compiled.ion_species_count == 2:
        targets["negative"] = compiled.negative_reference_inventory_m2
    return IonicSteadyJVResult(requested, voltage_array, current_array, output, tuple(attempts),
        _metrics(voltage_array, output, incident_power_W_m2), generation, incident_power_W_m2, polarity,
        initial_provenance, compiled.identity, compiled.original_prepared_sha256, compiled.binding_manifest,
        pins, _identity(pins), _identity(controls), _identity(request), controls, compiled.ion_species_names,
        targets, compiled.ion_geometry.cell_measures_m, compiled.electronic_component.nodes_m,
        calls, counts, time.monotonic() - started)


def write_ionic_steady_jv(result: IonicSteadyJVResult, path: str | Path) -> str:
    """Create one result exclusively, preserving array words and all failed data."""
    if not isinstance(result, IonicSteadyJVResult):
        raise ValueError("an actual IonicSteadyJVResult is required")
    raw = (json.dumps({"schema": SCHEMA, "completed": result.completed, "result": _encode(result)},
                     indent=2, allow_nan=False) + "\n").encode()
    with Path(path).open("xb") as output:
        output.write(raw)
    return hashlib.sha256(raw).hexdigest()


def read_ionic_steady_jv(path: str | Path, sha256: str) -> dict[str, Any]:
    """Validate saved checks, hashes and prefix consistency without a model solve."""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("ionic steady J-V artifact SHA-256 mismatch")
    try:
        record = _decode(json.loads(raw))
        result = record["result"]
        if (record["schema"] != SCHEMA or result["experiment_kind"] != "equilibrated_ionic_steady_jv"
                or result["physical_certificate_status"] != "not_evaluated"
                or result["controls_sha256"] != _identity(result["solver_controls"])
                or result["source_identity"] != _identity(result["source_sha256"])
                or result["compiled_identity"] != _identity(result["binding_manifest"])
                or result["original_prepared_identity"] != result["binding_manifest"]["original_prepared_sha256"]):
            raise ValueError("inconsistent ionic steady J-V metadata")
        requested = _voltages(result["requested_voltage_V"], result["solver_controls"]["max_points"])
        generation, incident = result["generation_scale"], result["incident_power_W_m2"]
        policy = result["initial_provenance"]["policy"]
        if (type(generation) not in {int, float} or not math.isfinite(generation) or generation < 0.0
                or (incident is not None and (type(incident) not in {int, float} or not math.isfinite(incident) or incident <= 0.0))
                or policy not in {"compiled_cold", "saved_accepted"} or result["voltage_polarity"] not in {-1.0, 1.0}):
            raise ValueError("invalid recorded ionic J-V generation, incident power, initial policy or polarity")
        request = {"voltage_V": requested.tolist(), "generation_scale": result["generation_scale"],
            "incident_power_W_m2": result["incident_power_W_m2"], "initial_policy": result["initial_provenance"]["policy"],
            "initial_source_sha256": result["initial_provenance"].get("source_sha256"), "controls": result["solver_controls"]}
        if result["request_sha256"] != _identity(request):
            raise ValueError("ionic J-V request identity differs from its recorded protocol")
        attempts = result["attempts"]
        if not 1 <= len(attempts) <= requested.size:
            raise ValueError("ionic J-V must retain its first attempted point")
        accepted = []
        for index, attempt in enumerate(attempts):
            if (attempt["index"] != index or attempt["voltage_V"] != requested[index]
                    or type(attempt["accepted"]) is not bool
                    or (not attempt["accepted"] and index != len(attempts) - 1)):
                raise ValueError("ionic J-V attempts must retain the requested prefix and first failure")
            if attempt["accepted"]:
                point = attempt["solution"]
                if (point["accepted"] is not True or point["newton"]["accepted"] is not True
                        or attempt["solver_called"] is not True or attempt["termination"] != "converged"
                        or point["compiled_identity"] != result["compiled_identity"]
                        or point["binding_manifest"] != result["binding_manifest"]
                        or point["state"]["electronic_state"]["voltage_V"] != requested[index]
                        or not point["checks"] or any(check["passed"] is not True
                            or not math.isfinite(check["value"]) or check["value"] > check["limit"] for check in point["checks"])):
                    raise ValueError("contradictory accepted ionic steady evidence")
                actual_controls = dict(point["solver_controls"])
                actual_wall = actual_controls["wall_s"]
                declared = result["solver_controls"]["steady"]
                actual_controls["wall_s"] = declared["wall_s"]
                if (actual_controls != declared or not 0.0 < actual_wall <= min(
                        declared["wall_s"], result["solver_controls"]["wall_s"])):
                    raise ValueError("point controls differ from the declared controls/remaining wall cap")
                accepted.append(attempt)
        completed = len(attempts) == requested.size and len(accepted) == len(attempts)
        if not completed and attempts[-1]["accepted"]:
            raise ValueError("an incomplete ionic J-V curve must retain its first failed point")
        if record["completed"] is not completed or result["solver_calls"] != sum(a["solver_called"] for a in attempts):
            raise ValueError("ionic J-V completion or call count contradicts its attempts")
        voltage, current, output = (result[key] for key in ("voltage_V", "current_density_A_m2", "output_current_A_m2"))
        if (not np.array_equal(voltage, requested[:len(accepted)])
                or current.shape != (len(accepted), result["nodes_m"].size + 1)
                or not np.all(np.isfinite(current))
                or not np.array_equal(output, -result["voltage_polarity"] * current[:, 0])):
            raise ValueError("ionic J-V accepted currents/voltages do not match its recorded prefix")
        for index, attempt in enumerate(accepted):
            if not np.array_equal(current[index], attempt["solution"]["diagnostics"]["total_current_density_A_m2"]):
                raise ValueError("ionic J-V current differs from its original point diagnostics")
            state = attempt["solution"]["state"]
            for species in result["ion_species"]:
                density = state[species + "_ion_density_m3"]
                if density.shape != result["ion_weights_m"].shape or not np.all(np.isfinite(density)) or np.any(density <= 0.0):
                    raise ValueError("ionic J-V species profile is incomplete or invalid")
                inventory = math.fsum((result["ion_weights_m"] * density).tolist())
                key = "domain_inventory_m2" if species == "positive" else "negative_domain_inventory_m2"
                if inventory != state[key]:
                    raise ValueError("ionic J-V inventory differs from its physical profile")
        if result["metrics"] != asdict(_metrics(voltage, output, result["incident_power_W_m2"])):
            raise ValueError("ionic J-V metrics contradict the accepted voltage brackets")
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise ValueError("incomplete or malformed ionic steady J-V artifact") from exc
    return record
