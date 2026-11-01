"""Accepted-only steady voltage continuation of connected classical2D models.

Every voltage invokes the existing full2D Newton once. Seeds are rebased
accepted2D states; failed states, independent-column solves and retries are
never used. Sampled PV metrics retain the existing interpolation limitations.
"""

from __future__ import annotations

from solarlab.reproducibility.job_provenance import (
    JobProvenance, admitted_source_pins, current_provenance, job_provenance, result_provenance,
)

from dataclasses import asdict, dataclass, field, replace
from collections.abc import Callable
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from solarlab.device.grain_2d import GRAIN_SRH_2D_V1, GrainRecombination2D
from solarlab.equations.assembly.two_dimensional import AssembledDC2DProblem
from solarlab.equations.dc_2d import CompiledDevice2D, DC2DProblem, State2D
from solarlab.equations.implementation import provenance_binding, use_assembled
from solarlab.experiments.jv.steady import SteadyJVMetrics, steady_jv_metrics
from solarlab.experiments.transient_results import _decode, _encode
from solarlab.experiments.worker import _atomic_json
from solarlab.io.artifacts import bytes_artifact
from solarlab.solvers.nonlinear.problem import GateOutcome, state_digest
from solarlab.solvers.nonlinear.steady_2d import (
    Steady2DOptions, Steady2DResult, checks as steady_checks, diagnostics as steady_diagnostics, scope as steady_scope,
    solve_steady_state_2d,
)

__all__ = ["SteadyJV2DOptions", "SteadyJV2DInitial", "SteadyJV2DAttempt", "SteadyJV2DResult",
           "load_steady_jv_2d_initial", "run_steady_jv_2d", "write_steady_jv_2d", "read_steady_jv_2d",
           "read_steady_jv_2d_checkpoint", "admit_steady_jv_2d_checkpoint", "continue_steady_jv_2d"]
LEGACY_SCHEMA = "solarlab.steady_jv_2d.v1"
SCHEMA = "solarlab.steady_jv_2d.v2"


@dataclass(frozen=True)
class SteadyJV2DOptions:
    steady: Steady2DOptions = field(default_factory=Steady2DOptions)
    max_points: int = 51
    wall_s: float = 30.0

    def __post_init__(self) -> None:
        if not isinstance(self.steady, Steady2DOptions):
            raise ValueError("existing Steady2DOptions are required")
        if type(self.max_points) is not int or self.max_points < 1:
            raise ValueError("max_points must be a positive integer")
        if not math.isfinite(self.wall_s) or self.wall_s <= 0.0:
            raise ValueError("wave wall budget must be finite and positive")


@dataclass(frozen=True)
class SteadyJV2DInitial:
    state: State2D
    generation_scale: float
    source_path: str
    source_sha256: str
    evidence_json: str


@dataclass(frozen=True)
class SteadyJV2DAttempt:
    index: int
    voltage_V: float
    accepted: bool
    termination: str
    initial_guess: State2D | None
    seed_provenance: dict[str, Any]
    solution: Steady2DResult | None
    solver_called: bool
    wall_s: float
    error: str | None = None


@dataclass(frozen=True)
class SteadyJV2DResult:
    requested_voltage_V: np.ndarray
    voltage_V: np.ndarray
    output_current_A_m2: np.ndarray
    terminal_current_per_depth_A_m: np.ndarray
    attempts: tuple[SteadyJV2DAttempt, ...]
    metrics: SteadyJVMetrics
    generation_scale: float
    incident_power_W_m2: float | None
    voltage_polarity: float
    lateral_width_m: float
    initial_provenance: dict[str, Any]
    compiled_identity: str
    geometry: dict[str, Any]
    solver_controls: dict[str, Any]
    source_sha256: dict[str, str]
    solver_calls: int
    wall_s: float
    physical_certificate_status: str = "not_evaluated"

    job_provenance: dict[str, Any] | None = field(default_factory=current_provenance)

    def __post_init__(self) -> None:
        for name in ("requested_voltage_V", "voltage_V", "output_current_A_m2", "terminal_current_per_depth_A_m"):
            array = np.array(getattr(self, name), dtype=float, copy=True)
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @property
    def completed(self) -> bool:
        return len(self.attempts) == self.requested_voltage_V.size and all(item.accepted for item in self.attempts)

    @property
    def unattempted_voltage_V(self) -> np.ndarray:
        return self.requested_voltage_V[len(self.attempts):]


def _problem(device: CompiledDevice2D, voltage_V: float, generation_scale: float) -> DC2DProblem | AssembledDC2DProblem:
    """The selected 2D problem for admission and point checks (equations.implementation)."""
    if use_assembled():
        return AssembledDC2DProblem(device, voltage_V, generation_scale)
    return DC2DProblem(device, voltage_V, generation_scale)


def _scope(device: CompiledDevice2D) -> None:
    steady_scope(device)
    if device.field_transport is not None:
        raise ValueError("unsupported_field_transport_2d_jv: CT source binding is not admitted for 2D J-V")
    grain = getattr(device, "grain_recombination", None)
    if grain is not None and (not isinstance(grain, GrainRecombination2D) or not grain.regions
            or grain.geometry_sha256 != device.geometry.sha256
            or not grain.model_id.startswith(GRAIN_SRH_2D_V1 + ":")):
        raise ValueError("steady2D J-V requires the declared neutral grain-region model")


def _grain_model(device: CompiledDevice2D) -> dict[str, Any] | None:
    """Record the existing exact compiled binding; introduce no grain law."""
    grain = getattr(device, "grain_recombination", None)
    if grain is None:
        return None
    _scope(device)
    names = ("solarlab.device.grain_2d", "solarlab.device.compile_2d", "solarlab.equations.dc_2d",
             "solarlab.physics.recombination.bulk", "solarlab.mesh.grid")
    sources = {name: pin for name, pin in admitted_source_pins().items() if name in names}
    return {"model_id": grain.model_id, "geometry_sha256": grain.geometry_sha256,
        "effective_origin": grain.effective_origin,
        "regions": [{"declaration": r.declaration.normalized_data(), "x_overlap_fraction": r.x_overlap_fraction,
            "y_mask": r.y_mask, "fractions": r.fractions.reshape(device.geometry.shape),
            "represented_width_m": r.represented_width_m, "represented_height_m": r.represented_height_m,
            "represented_area_m2": r.represented_area_m2} for r in grain.regions],
        "rule": "R_total += f * (R_grain_SRH - R_bulk_SRH); exact density partials use the same f",
        "no_grain_charge_or_extra_state": True,
        "source_sha256": sources}


def _read_grain_model(geometry: dict[str, Any]) -> None:
    """Verify the saved exact binding, without compiling or solving."""
    model = geometry.get("grain_model")
    if model is None:
        return
    if (model["geometry_sha256"] != geometry["identity"] or not model["regions"]
            or model["no_grain_charge_or_extra_state"] is not True):
        raise ValueError("saved neutral grain2D model/geometry identity changed")
    digest = hashlib.sha256(json.dumps((GRAIN_SRH_2D_V1, model["geometry_sha256"], model["effective_origin"],
        [r["declaration"] for r in model["regions"]]), sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    for region in model["regions"]:
        f = np.asarray(region["fractions"])
        expected = np.asarray(region["y_mask"])[:, None] * np.asarray(region["x_overlap_fraction"])[None, :]
        if (f.shape != tuple(geometry["shape_Ny_Nx"]) or not np.array_equal(f, expected)
                or not np.all(np.isfinite(f)) or np.any(f < 0) or np.any(f > 1)
                or float(f.ravel() @ np.asarray(geometry["cell_areas_m2"]).ravel()) != region["represented_area_m2"]):
            raise ValueError("saved neutral grain2D physical fraction/area changed")
        digest.update(f.ravel().astype("<f8").tobytes())
    if model["model_id"] != GRAIN_SRH_2D_V1 + ":" + digest.hexdigest():
        raise ValueError("saved neutral grain2D declaration/model hash changed")


def _state(device: CompiledDevice2D, state: State2D) -> State2D:
    if not isinstance(state, State2D) or state.device_sha256 != device.identity:
        raise ValueError("continuation requires a complete same-device2D state")
    expected = device.state(state.u, voltage_V=state.voltage_V)
    if any(not np.array_equal(getattr(state, name), getattr(expected, name))
           for name in ("potential_V", "electron_density_m3", "hole_density_m3")):
        raise ValueError("2D physical fields disagree with their bound coordinates")
    return expected


def _admit(device: CompiledDevice2D, text: str, sha256: str,
           options: Steady2DOptions) -> tuple[State2D, float, tuple[GateOutcome, ...]]:
    _scope(device)
    if hashlib.sha256(text.encode()).hexdigest() != sha256:
        raise ValueError("accepted initial artifact SHA-256 mismatch")
    try:
        document = _decode(json.loads(text))
        if document["completed"] is not True:
            raise ValueError("an actually completed accepted2D steady artifact is required")
        if document["schema"] == "solarlab.classical_2d_steady.research.v1":
            if getattr(device, "grain_recombination", None) is not None:
                raise ValueError("a declared grain model requires its explicit grain initial artifact")
            attempt = document["attempts"][-1]
            result = attempt["result_2d"]
        elif document["schema"] == "solarlab.neutral_grain_2d.research.v1":
            model = _grain_model(device)
            if (model is None or document["mode"] != "steady" or document["accepted"] is not True
                    or document["error"] is not None or document["full_2d_newton_calls"] != 1):
                raise ValueError("initial artifact requires actual accepted neutral grain2D Newton evidence")
            binding = document["grain_binding"]
            result_provenance(document)
            result_provenance(document["result_2d"])
            if any(document["source_sha256"].get(name, pin) != pin
                   for name, pin in document["result_2d"]["source_sha256"].items()):
                raise ValueError("saved grain initial source records contradict each other")
            for name in ("model_id", "geometry_sha256", "effective_origin"):
                if binding[name] != model[name]:
                    raise ValueError("initial physical grain declaration/model differs from the compiled device")
            if len(binding["regions"]) != len(model["regions"]):
                raise ValueError("initial physical grain region inventory differs")
            for saved_region, expected_region in zip(binding["regions"], model["regions"]):
                for name, expected in expected_region.items():
                    if not np.array_equal(saved_region[name], expected):
                        raise ValueError("initial physical grain declaration/fraction differs: " + name)
            attempt, result = {"accepted": document["accepted"]}, document["result_2d"]
        else:
            raise ValueError("an actually completed accepted2D steady artifact is required")
        if (attempt["accepted"] is not True or result["accepted"] is not True
                or result["newton"]["accepted"] is not True or result["termination"] != "converged"
                or result["compiled_identity"] != device.identity or document["compiled_identity"] != device.identity
                or result["solver_controls"]["numerical"]["electronic"] != asdict(options.electronic)
                or not result["checks"] or not all(item["passed"] is True for item in result["checks"])):
            raise ValueError("initial2D acceptance, model or physical controls disagree")
        saved = result["state"]
        voltage, generation = float(result["operating_voltage_V"]), float(result["generation_scale"])
        if not math.isfinite(generation) or generation < 0.0:
            raise ValueError("initial generation must be explicit finite nonnegative")
        state = device.state(saved["u"], voltage_V=voltage)
        if (saved["device_sha256"] != device.identity or saved["voltage_V"] != voltage
                or not np.array_equal(result["newton"]["state"], state.u)
                or not np.array_equal(result["candidate_coordinates"], state.u)
                or any(not np.array_equal(saved[name], getattr(state, name))
                       for name in ("potential_V", "electron_density_m3", "hole_density_m3"))):
            raise ValueError("initial accepted2D fields/coordinates are inconsistent")
        checks = steady_checks(steady_diagnostics(_problem(device, voltage, generation), state.u, options), options)
        if ({item.name: item.limit for item in checks} != {item["name"]: item["limit"] for item in result["checks"]}
                or not all(item.passed for item in checks)):
            raise ValueError("initial2D state fails current full original-gate admission")
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError("incomplete accepted2D initial evidence") from exc
    return state, generation, checks


def load_steady_jv_2d_initial(device: CompiledDevice2D, path: str | Path, sha256: str,
                             *, options: Steady2DOptions | None = None) -> SteadyJV2DInitial:
    controls = Steady2DOptions() if options is None else options
    source = Path(path)
    text = source.read_text()
    state, generation, _ = _admit(device, text, sha256, controls)
    return SteadyJV2DInitial(state, generation, str(source.resolve()), sha256, text)


def _rebase(device: CompiledDevice2D, state: State2D, voltage: float) -> State2D:
    _state(device, state)
    if state.voltage_V == voltage:
        return state
    values, N = state.u.copy(), device.node_count
    before, after = device.column.carrier_reference_energies(state.voltage_V), device.column.carrier_reference_energies(voltage)
    for offset, old, new in zip((N, 2 * N), before, after):
        values[offset:offset + N] += (old - new) / device.thermal_voltage_V
    return device.state(values, voltage_V=voltage)


def _voltages(value: ArrayLike, maximum: int) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError("2D J-V voltages must be real")
    result = np.asarray(raw, dtype=float)
    if (result.ndim != 1 or not 1 <= result.size <= maximum or not np.all(np.isfinite(result))
            or (result.size > 1 and not (np.all(np.diff(result) > 0) or np.all(np.diff(result) < 0)))):
        raise ValueError("requested2D voltages must be a bounded strictly monotonic finite vector")
    return result.copy()




def _metrics(voltage: np.ndarray, current: np.ndarray, incident: float | None) -> SteadyJVMetrics:
    if voltage.size < 2:
        return SteadyJVMetrics(None, None, None, None, None, "insufficient_accepted_points")
    return steady_jv_metrics(voltage, current, incident_power_W_m2=incident)


def _checkpoint_write(path: Path, value: dict[str, Any], *, exclusive: bool = False) -> str:
    if exclusive and path.exists():
        raise FileExistsError("preserve the existing2D point checkpoint")
    _atomic_json(path, _encode(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint_artifact(path: Path, raw: bytes) -> str:
    """Scoped immutable large input/point artifact; metadata limits stay intact."""
    if len(raw) > 32 * 2**20:
        raise ValueError("scoped2D checkpoint payload exceeds32MiB")
    payload = bytes_artifact(raw, metadata={"role": "steady2d_jv_checkpoint_payload", "unit": "raw lossless JSON bytes"})
    if path.exists():
        raise FileExistsError("preserve immutable2D checkpoint artifact")
    partial = path.with_suffix(path.suffix + ".part")
    fd = os.open(partial, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload.data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return payload.sha256


def read_steady_jv_2d_checkpoint(directory: str | Path, manifest_sha256: str) -> dict[str, Any]:
    """Read immutable point records and an interrupted active marker, no solve."""
    root = Path(directory)
    raw = (root / "Manifest.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest_sha256:
        raise ValueError("2D checkpoint manifest SHA-256 mismatch")
    manifest = _decode(json.loads(raw))
    def read_bound(name: str, sha: str) -> Any:
        if name not in {"Request.json", "Active.json"} and not (
                name.startswith("Points/") and len(name.split("/")) == 2 and name.endswith(".json")):
            raise ValueError("invalid scoped2D checkpoint member")
        data = (root / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != sha:
            raise ValueError("2D checkpoint member SHA-256 mismatch")
        return _decode(json.loads(data))
    request = read_bound("Request.json", manifest["request_sha256"])
    result_provenance(request)
    initial = request["initial_evidence_artifact"]
    if initial["path"] != "Initial.json" or initial["sha256"] != request["initial_source_sha256"]:
        raise ValueError("initial2D checkpoint artifact descriptor changed")
    initial_raw = (root / "Initial.json").read_bytes()
    if len(initial_raw) != initial["nbytes"] or hashlib.sha256(initial_raw).hexdigest() != initial["sha256"]:
        raise ValueError("initial2D checkpoint artifact SHA-256/size mismatch")
    requested = _voltages(request["requested_voltage_V"], request["controls"]["max_points"])
    if len(manifest["points"]) > requested.size:
        raise ValueError("2D checkpoint points exceed the bounded request")
    attempts = []
    for index, item in enumerate(manifest["points"]):
        point = read_bound(item["path"], item["sha256"])
        attempt = point["attempt"]
        if (point["request_sha256"] != manifest["request_sha256"] or attempt["index"] != index
                or attempt["voltage_V"] != requested[index]):
            raise ValueError("2D durable point does not match its request/prefix")
        if not attempt["accepted"] and index != len(manifest["points"]) - 1:
            raise ValueError("failure-state2D continuation cannot be resumed")
        attempts.append(attempt)
    active = None if manifest["active"] is None else read_bound("Active.json", manifest["active"])
    if manifest["finished"] and active is not None:
        raise ValueError("a finished2D checkpoint cannot contain an unresolved active point")
    if active is not None and (type(active["index"]) is not int or not 0 <= active["index"] < requested.size
            or active["request_sha256"] != manifest["request_sha256"]
            or active["voltage_V"] != requested[active["index"]]):
        raise ValueError("active2D point does not match its exact requested index/voltage")
    if active is not None and active["index"] < len(attempts):
        active = None  # A completed immutable record wins over a stale active marker.
    if active is not None and active["index"] != len(attempts):
        raise ValueError("interrupted2D point must follow the exact complete prefix")
    next_index = len(attempts)
    if active is not None:
        next_index = active["index"]
    unattempted_index = len(attempts) if active is None else active["index"] + 1
    return {"request": request, "attempts": attempts, "active_point": active,
        "active_status": None if active is None else "unresolved_or_interrupted",
        "unattempted_voltage_V": request["requested_voltage_V"][unattempted_index:],
        "next_index": next_index, "finished": manifest["finished"],
        "completed": manifest["finished"] and len(attempts) == len(request["requested_voltage_V"])
            and all(item["accepted"] for item in attempts)}


def admit_steady_jv_2d_checkpoint(device: CompiledDevice2D, directory: str | Path, manifest_sha256: str,
                                 *, options: SteadyJV2DOptions | None = None) -> tuple[State2D, int]:
    """Validate the last accepted raw state for explicit resume; never retry here."""
    controls = SteadyJV2DOptions() if options is None else options
    checkpoint = read_steady_jv_2d_checkpoint(directory, manifest_sha256)
    request = checkpoint["request"]
    _scope(device)
    if "continuation_prefix" in request:
        raise ValueError("a wall-tail checkpoint requires the explicit continuation entry; no automatic retry")
    if (request["compiled_identity"] != device.identity or request["controls"] != asdict(controls)
           ):
        raise ValueError("2D checkpoint model/control binding changed")
    text = (Path(directory) / request["initial_evidence_artifact"]["path"]).read_text()
    state, generation, _ = _admit(device, text, request["initial_source_sha256"], controls.steady)
    for attempt in checkpoint["attempts"]:
        if not attempt["accepted"]:
            raise ValueError("a failed2D point requires explicit recovery; never failure-state seeding")
        point = attempt["solution"]
        if (point["accepted"] is not True or point["newton"]["accepted"] is not True
                or point["compiled_identity"] != device.identity
                or point["operating_voltage_V"] != attempt["voltage_V"]
                or not np.array_equal(point["state"]["u"], point["newton"]["state"])):
            raise ValueError("checkpoint lost actual accepted2D Newton identity")
        state = device.state(point["state"]["u"], voltage_V=attempt["voltage_V"])
        if any(not np.array_equal(point["state"][name], getattr(state, name))
               for name in ("potential_V", "electron_density_m3", "hole_density_m3")):
            raise ValueError("checkpoint physical fields changed")
        if not all(item.passed for item in steady_checks(steady_diagnostics(_problem(device, state.voltage_V, generation), state.u,
                                                             controls.steady), controls.steady)):
            raise ValueError("checkpoint failed current original2D physical gate admission")
    return state, checkpoint["next_index"]


def run_steady_jv_2d(device: CompiledDevice2D, initial: SteadyJV2DInitial, voltage_V: ArrayLike,
                     *, options: SteadyJV2DOptions | None = None, incident_power_W_m2: float | None = None,
                     on_point: Callable[[SteadyJV2DAttempt], None] | None = None,
                     checkpoint_directory: str | Path | None = None,
                     absolute_deadline_s: float | None = None) -> SteadyJV2DResult:
    """At most one original full2D solve per voltage; stop at first failure."""
    return _run_steady_jv_2d(device, initial, voltage_V, options=options,
        incident_power_W_m2=incident_power_W_m2, on_point=on_point,
        checkpoint_directory=checkpoint_directory, absolute_deadline_s=absolute_deadline_s)


def _read_wall_prefix(device: CompiledDevice2D, path: str | Path, sha256: str, controls: SteadyJV2DOptions,
                      ) -> tuple[SteadyJV2DInitial, np.ndarray, Callable[[], tuple[State2D, float, tuple[GateOutcome, ...]]]]:
    """Only an exact wall-stopped prefix may supply its last accepted2D state.

    Reading parses and hash-checks the prefix and evaluates no equation. It returns the tail's initial
    (the last accepted state, from its saved coordinates), the remaining requested voltages and the
    prefix's numerical readmission, which the tail job runs once its equation binding is admitted.
    """
    _scope(device)
    source = Path(path)
    record = read_steady_jv_2d(source, sha256)
    result, attempts = record["result"], record["result"]["attempts"]
    accepted = len(result["voltage_V"])
    if (record["completed"] or accepted < 1
            or len(attempts) != accepted + 1 or attempts[-1]["accepted"]
            or attempts[-1]["termination"] not in {"wall_budget_exhausted", "wave_wall_budget_exhausted"}):
        raise ValueError("explicit2D continuation requires an original wall-stopped accepted prefix")
    linked = record.get("continuation")
    requested = result["requested_voltage_V"]
    points = list(attempts[:accepted])
    original = result["initial_provenance"]
    if linked is not None:
        requested = linked["original_requested_voltage_V"]
        if linked.get("endpoint_only") or requested.size - linked["combined_voltage_V"].size != 1:
            raise ValueError("linked2D recovery supports only the one final original endpoint")
        binding = linked["prefix_artifact"]
        earlier = read_steady_jv_2d(binding["path"], binding["sha256"])
        if "continuation" in earlier:
            raise ValueError("no continuation beyond the single final2D endpoint")
        points = list(earlier["result"]["attempts"][:linked["prefix_accepted_points"]]) + points
        accepted = len(points)
        original = {"source_path": linked["original_initial_artifact"]["path"],
                    "source_sha256": linked["original_initial_artifact"]["sha256"]}
    if result["compiled_identity"] != device.identity or result["solver_controls"] != asdict(controls):
        raise ValueError("wall-prefix2D model or controls changed")
    last = points[-1]
    initial = SteadyJV2DInitial(device.state(last["solution"]["state"]["u"], voltage_V=last["voltage_V"]),
                                result["generation_scale"], str(source.resolve()), sha256, source.read_text())

    def readmit() -> tuple[State2D, float, tuple[GateOutcome, ...]]:
        _, generation, _ = _admit(device, Path(original["source_path"]).read_text(),
                                  original["source_sha256"], controls.steady)
        if result["generation_scale"] != generation:
            raise ValueError("wall-prefix2D generation identity changed")
        admission: tuple[GateOutcome, ...] = ()
        state = None
        for attempt in points:
            point = attempt["solution"]
            if (point["solver_controls"]["numerical"]["electronic"] != asdict(controls.steady.electronic)):
                raise ValueError("wall-prefix2D point physical source or controls changed")
            state = device.state(point["state"]["u"], voltage_V=attempt["voltage_V"])
            if any(not np.array_equal(point["state"][name], getattr(state, name))
                   for name in ("potential_V", "electron_density_m3", "hole_density_m3")):
                raise ValueError("wall-prefix2D accepted physical fields changed")
            admission = steady_checks(steady_diagnostics(_problem(device, state.voltage_V, generation), state.u,
                                              controls.steady), controls.steady)
            if not all(item.passed for item in admission):
                raise ValueError("wall-prefix2D accepted state fails original physical gates")
        if state is None:
            raise ValueError("wall-prefix2D contains no accepted seed")
        return state, generation, admission
    return initial, np.array(requested[accepted:], copy=True), readmit


def continue_steady_jv_2d(device: CompiledDevice2D, prefix_path: str | Path, prefix_sha256: str,
                          *, options: SteadyJV2DOptions | None = None,
                          on_point: Callable[[SteadyJV2DAttempt], None] | None = None,
                          checkpoint_directory: str | Path | None = None,
                          absolute_deadline_s: float | None = None) -> SteadyJV2DResult:
    """Explicit wall-prefix recovery: solve only its exact remaining voltage tail.

    The original failed artifact and driver-source identity remain unchanged.
    All physical sources/controls and accepted rows are readmitted before using
    the last accepted state, inside the tail job once its equation binding is
    admitted. This entry performs no retry of a numerical failure.
    A stopped linked tail may supply only its single final original endpoint.
    """
    controls = SteadyJV2DOptions() if options is None else options
    initial, requested, readmit = _read_wall_prefix(device, prefix_path, prefix_sha256, controls)
    prefix = read_steady_jv_2d(prefix_path, prefix_sha256)
    return _run_steady_jv_2d(device, initial, requested, options=controls,
        incident_power_W_m2=prefix["result"]["incident_power_W_m2"], on_point=on_point,
        checkpoint_directory=checkpoint_directory, absolute_deadline_s=absolute_deadline_s,
        wall_prefix_admission=readmit)


def _run_steady_jv_2d(device: CompiledDevice2D, initial: SteadyJV2DInitial, voltage_V: ArrayLike,
                       *, options: SteadyJV2DOptions | None = None, incident_power_W_m2: float | None = None,
                       on_point: Callable[[SteadyJV2DAttempt], None] | None = None,
                       checkpoint_directory: str | Path | None = None,
                       absolute_deadline_s: float | None = None,
                       wall_prefix_admission: Callable[[], tuple[State2D, float, tuple[GateOutcome, ...]]] | None = None,
                       ) -> SteadyJV2DResult:
    started = time.monotonic()
    _scope(device)
    controls = SteadyJV2DOptions() if options is None else options
    if not isinstance(controls, SteadyJV2DOptions) or not isinstance(initial, SteadyJV2DInitial):
        raise ValueError("reviewed2D controls and a hash-bound accepted2D initial are required")
    requested = _voltages(voltage_V, controls.max_points)
    if incident_power_W_m2 is not None and (not math.isfinite(incident_power_W_m2) or incident_power_W_m2 <= 0.0):
        raise ValueError("efficiency requires explicitly supplied finite positive irradiance W/m2")
    with job_provenance({"compiled_identity": device.identity, "geometry_identity": device.geometry.sha256, "initial_sha256": initial.source_sha256, "voltage_V": requested.tolist(), "controls": asdict(controls), "incident_power_W_m2": incident_power_W_m2},
            **provenance_binding("DC2DProblem.steady_jv", device)):
        pins = admitted_source_pins()
        if wall_prefix_admission is None:
            previous, generation, admission = _admit(device, initial.evidence_json, initial.source_sha256, controls.steady)
        else:
            previous, generation, admission = wall_prefix_admission()
        if not np.array_equal(_state(device, initial.state).u, previous.u) or initial.generation_scale != generation:
            raise ValueError("initial object differs from its admitted saved state")
        deadline = started + controls.wall_s
        if absolute_deadline_s is not None:
            if not math.isfinite(absolute_deadline_s):
                raise ValueError("parent-owned absolute monotonic deadline must be finite")
            deadline = min(deadline, absolute_deadline_s)
        journal = None
        manifest: dict[str, Any] | None = None
        if checkpoint_directory is not None:
            journal = Path(checkpoint_directory)
            journal.mkdir(parents=True, exist_ok=False)
            (journal / "Points").mkdir()
            raw_initial = initial.evidence_json.encode()
            initial_digest = _checkpoint_artifact(journal / "Initial.json", raw_initial)
            request = {"schema": "solarlab.steady_jv_2d.checkpoint_request.v1", "requested_voltage_V": requested,
                "compiled_identity": device.identity, "generation_scale": generation,
                "initial_source_sha256": initial.source_sha256,
                "initial_evidence_artifact": {"path": "Initial.json", "sha256": initial_digest, "nbytes": len(raw_initial)},
                "controls": asdict(controls), "source_sha256": pins, "job_provenance": current_provenance(),
                "incident_power_W_m2": incident_power_W_m2,
                "absolute_compute_deadline_s": deadline}
            grain_model = _grain_model(device)
            if grain_model is not None:
                request["grain_model"] = grain_model
            if wall_prefix_admission is not None:
                request["continuation_prefix"] = {"path": initial.source_path, "sha256": initial.source_sha256}
            digest = _checkpoint_write(journal / "Request.json", request, exclusive=True)
            manifest = {"request_sha256": digest, "points": [], "active": None, "finished": False}
            _checkpoint_write(journal / "Manifest.json", manifest)
        attempts: list[SteadyJV2DAttempt] = []
        voltages, output, terminals = [], [], []
        calls = 0
        for index, supplied in enumerate(requested):
            voltage, began = float(supplied), time.monotonic()
            seed, solution, error = None, None, None
            called = False
            first_kind = "accepted_saved_initial" if wall_prefix_admission is None else "accepted_wall_prefix"
            provenance = {"kind": first_kind if index == 0 else "previous_accepted2D",
                "previous_voltage_V": previous.voltage_V, "previous_state_sha256": state_digest(previous.u)}
            try:
                if time.monotonic() >= deadline:
                    raise TimeoutError("2D J-V wave wall budget exhausted before next point")
                seed = _rebase(device, previous, voltage)
                provenance["rebased_state_sha256"] = state_digest(seed.u)
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError("2D J-V wave wall budget exhausted before solver call")
                effective = replace(controls.steady, wall_s=min(controls.steady.wall_s, remaining))
                if journal is not None and manifest is not None:
                    marker = {"index": index, "voltage_V": voltage, "initial_guess": seed,
                        "seed_provenance": provenance, "request_sha256": manifest["request_sha256"],
                        "status": "solver_entry_pending_or_running", "solver_called": None}
                    manifest["active"] = _checkpoint_write(journal / "Active.json", marker)
                    _checkpoint_write(journal / "Manifest.json", manifest)
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        raise TimeoutError("2D compute deadline reached after durable solver-entry marker")
                    effective = replace(controls.steady, wall_s=min(controls.steady.wall_s, remaining))
                called, calls = True, calls + 1
                solution = solve_steady_state_2d(device, voltage_V=voltage, generation_scale=generation,
                    options=effective, initial_state=seed, initial_provenance=provenance)
                accepted, termination, error = solution.accepted, solution.termination, solution.error
                if accepted:
                    if solution.state is None or solution.diagnostics is None:
                        raise ValueError("accepted2D point must retain its original state and current ledger")
                    checks = steady_checks(steady_diagnostics(_problem(device, voltage, generation), solution.state.u, effective), effective)
                    if not all(item.passed for item in checks):
                        raise ValueError("accepted2D point fails original physical gates on readback")
                    previous = _state(device, solution.state)
                    terminal = np.asarray(solution.diagnostics["terminal_current_per_depth_A_m"])
                    voltages.append(voltage)
                    terminals.append(terminal)
                    output.append(-device.column.voltage_polarity * terminal[0] / device.geometry.lateral_length_m)
            except Exception as exc:
                accepted = False
                termination = "wave_wall_budget_exhausted" if isinstance(exc, TimeoutError) else "evaluation_error"
                error = f"{type(exc).__name__}: {str(exc)[:600]}"
            attempt = SteadyJV2DAttempt(index, voltage, accepted, termination, seed, provenance, solution, called,
                                        time.monotonic() - began, error)
            attempts.append(attempt)
            if journal is not None and manifest is not None:
                name = f"Points/{index:05d}.json"
                raw_point = json.dumps(_encode({"request_sha256": manifest["request_sha256"], "attempt": attempt}),
                                       sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
                digest = _checkpoint_artifact(journal / name, raw_point)
                manifest["points"].append({"path": name, "sha256": digest})
                manifest["active"] = None
                _checkpoint_write(journal / "Manifest.json", manifest)
            if on_point is not None:
                on_point(attempt)
            if not accepted:
                break
        if journal is not None and manifest is not None:
            manifest["finished"] = True
            _checkpoint_write(journal / "Manifest.json", manifest)
        accepted_v: np.ndarray = np.asarray(voltages, dtype=float)
        accepted_j: np.ndarray = np.asarray(output, dtype=float)
        g = device.geometry
        geometry = {"identity": g.sha256, "shape_Ny_Nx": g.shape, "x_m": g.x_m, "y_m": device.column.nodes_m,
            "cell_areas_m2": g.contract_geometry.volumes, "face_lengths_m": g.contract_geometry.face_measures,
            "face_pairs": g.contract_geometry.pairs, "lateral_width_m": g.lateral_length_m}
        grain_model = _grain_model(device)
        if grain_model is not None:
            geometry["grain_model"] = grain_model
        initial_provenance = {"source_path": initial.source_path, "source_sha256": initial.source_sha256,
            "state_sha256": state_digest(initial.state.u), "operating_voltage_V": initial.state.voltage_V,
            "admission_checks": admission, "no_new_cold_or_1d_seed": True}
        if wall_prefix_admission is not None:
            initial_provenance["continuation_prefix"] = {"path": initial.source_path, "sha256": initial.source_sha256}
        elif grain_model is not None:
            original = _decode(json.loads(initial.evidence_json))["result_2d"]["source_sha256"]
            current = admitted_source_pins()
            initial_provenance["grain_source_admission"] = {"saved_result_source_sha256": original,
                "current_result_source_sha256": current, "exact_source_match": original == current,
                "current_original_gate_revalidation": True, "original_saved_acceptance_preserved": True}
        return SteadyJV2DResult(requested, accepted_v, accepted_j, np.asarray(terminals, dtype=float).reshape(-1, 2),
            tuple(attempts), _metrics(accepted_v, accepted_j, incident_power_W_m2), generation, incident_power_W_m2,
            device.column.voltage_polarity, g.lateral_length_m, initial_provenance, device.identity, geometry,
            asdict(controls), pins, calls, time.monotonic() - started)


def write_steady_jv_2d(result: SteadyJV2DResult, path: str | Path) -> str:
    if not isinstance(result, SteadyJV2DResult):
        raise ValueError("an actual steady2D J-V result is required")
    document = {"schema": SCHEMA, "completed": result.completed, "qualification": False,
        "unattempted_voltage_V": result.unattempted_voltage_V, "result": result}
    if "continuation_prefix" in result.initial_provenance:
        binding = result.initial_provenance["continuation_prefix"]
        prefix = read_steady_jv_2d(binding["path"], binding["sha256"])
        document["continuation"] = _linked_tail(prefix, asdict(result))
    raw = (json.dumps(_encode(document), indent=2, allow_nan=False) + "\n").encode()
    with Path(path).open("xb") as stream:
        stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


def _linked_tail(prefix: dict[str, Any], tail: dict[str, Any]) -> dict[str, Any]:
    """Bind one new tail to preserved raw prefix evidence and sampled metrics."""
    old = prefix["result"]
    local_accepted = len(old["voltage_V"])
    prior = prefix.get("continuation")
    requested = old["requested_voltage_V"] if prior is None else prior["original_requested_voltage_V"]
    prior_voltage = old["voltage_V"] if prior is None else prior["combined_voltage_V"]
    offset = prior_voltage.size
    if prior is not None and (prior.get("endpoint_only") or requested.size - offset != 1):
        raise ValueError("linked2D recovery supports only the one final original endpoint")
    if (prefix["completed"] or local_accepted < 1
            or len(old["attempts"]) != local_accepted + 1 or old["attempts"][-1]["accepted"]
            or old["attempts"][-1]["termination"] not in {"wall_budget_exhausted", "wave_wall_budget_exhausted"}
            or not np.array_equal(tail["requested_voltage_V"], requested[offset:])):
        raise ValueError("linked2D continuation must be only the original wall-stopped remaining tail")
    for name in ("compiled_identity", "generation_scale", "voltage_polarity", "lateral_width_m",
                 "incident_power_W_m2", "solver_controls"):
        if tail[name] != old[name]:
            raise ValueError("linked2D continuation model, generation, units or controls changed")
    if tail.get("job_provenance") is None and {
            k: v for k, v in tail["source_sha256"].items() if k != __name__} != {
            k: v for k, v in old["source_sha256"].items() if k != __name__}:
        raise ValueError("linked2D v1 continuation changed a physical source")
    last = old["attempts"][local_accepted - 1]["solution"]["state"]
    seed = tail["initial_provenance"]
    if seed["operating_voltage_V"] != last["voltage_V"] or seed["state_sha256"] != state_digest(last["u"]):
        raise ValueError("linked2D continuation must use the last accepted state, never the failed candidate")
    prior_current = old["output_current_A_m2"] if prior is None else prior["combined_output_current_A_m2"]
    prior_terminals = old["terminal_current_per_depth_A_m"] if prior is None else prior["combined_terminal_current_per_depth_A_m"]
    voltage = np.r_[prior_voltage, tail["voltage_V"]]
    current = np.r_[prior_current, tail["output_current_A_m2"]]
    terminals = np.concatenate((prior_terminals, tail["terminal_current_per_depth_A_m"]), axis=0)
    failure = old["attempts"][-1]
    linked: dict[str, Any] = {"prefix_artifact": seed["continuation_prefix"], "prefix_completed": False,
        "preserved_prefix_failure": {key: failure[key] for key in ("index", "voltage_V", "termination", "error")},
        "prefix_accepted_points": offset, "tail_index_offset": offset,
        "original_initial_artifact": {"path": old["initial_provenance"]["source_path"],
                                      "sha256": old["initial_provenance"]["source_sha256"]} if prior is None
            else prior["original_initial_artifact"],
        "original_requested_voltage_V": requested, "combined_voltage_V": voltage,
        "combined_output_current_A_m2": current, "combined_terminal_current_per_depth_A_m": terminals,
        "metrics": asdict(_metrics(voltage, current, tail["incident_power_W_m2"])),
        "completed": voltage.size == requested.size,
        "total_actual_solver_calls": (old["solver_calls"] if prior is None else prior["total_actual_solver_calls"]) + tail["solver_calls"],
        "prefix_driver_sha256": old["source_sha256"].get(__name__),
        "tail_driver_sha256": tail["source_sha256"].get(__name__), "qualification": False}
    if tail.get("job_provenance") is not None:
        linked["schema"] = "solarlab.steady_jv_2d.continuation.v2"
        linked["producer_provenance"] = {"prefix": old.get("job_provenance"), "tail": tail["job_provenance"]}
        linked["source_comparison"] = result_provenance(old, tail["job_provenance"])["status"]
    if prior is not None:
        linked["preserved_prefix_failure"]["index"] += offset - local_accepted
        linked["endpoint_only"] = True
        linked["preserved_earlier_failure"] = prior["preserved_prefix_failure"]
        linked["preserved_earlier_artifact"] = prior["prefix_artifact"]
    return linked


def read_steady_jv_2d(path: str | Path, sha256: str, *,
    current_identity: JobProvenance | dict[str, Any] | None = None,
    strict_same_source: bool = False,
) -> dict[str, Any]:
    """Validate saved identity and accepted/failed prefix without any solver."""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("steady2D J-V artifact SHA-256 mismatch")
    try:
        document = _decode(json.loads(raw))
        if document["schema"] not in {LEGACY_SCHEMA, SCHEMA} or document["qualification"] is not False:
            raise ValueError("unsupported steady2D J-V result schema")
        result = document["result"]
        if document["schema"] == LEGACY_SCHEMA and result.get("job_provenance") is not None:
            raise ValueError("legacy2D producer identity cannot be relabelled with new job provenance")
        _read_grain_model(result["geometry"])
        requested = _voltages(result["requested_voltage_V"], result["solver_controls"]["max_points"])
        attempts = result["attempts"]
        count = int(np.prod(result["geometry"]["shape_Ny_Nx"]))
        if len(attempts) > requested.size:
            raise ValueError("attempts exceed the requested2D voltage wave")
        accepted = []
        for index, attempt in enumerate(attempts):
            if (attempt["index"] != index or attempt["voltage_V"] != requested[index]
                    or type(attempt["accepted"]) is not bool
                    or (not attempt["accepted"] and index != len(attempts) - 1)):
                raise ValueError("2D J-V attempts must retain the exact prefix and first failure")
            point = attempt["solution"]
            if point is not None:
                for name, pin in point["source_sha256"].items():
                    if name in result["source_sha256"] and result["source_sha256"][name] != pin:
                        raise ValueError("saved2D point source differs from its recorded producer")
            if attempt["accepted"]:
                point = attempt["solution"]
                if (point["accepted"] is not True or point["newton"]["accepted"] is not True
                        or point["termination"] != "converged" or point["compiled_identity"] != result["compiled_identity"]
                        or point["operating_voltage_V"] != requested[index]
                        or point["generation_scale"] != result["generation_scale"]
                        or tuple(point["diagnostics"]["original_rows"]) != tuple(range(3 * count))
                        or not np.array_equal(point["state"]["u"], point["newton"]["state"])
                        or not point["checks"] or any(item["passed"] is not True or not math.isfinite(item["value"])
                            or item["value"] > item["limit"] for item in point["checks"])):
                    raise ValueError("accepted2D J-V point lost original fields/rows/gates")
                accepted.append(attempt["voltage_V"])
        if not np.array_equal(result["voltage_V"], accepted) or not np.array_equal(
                document["unattempted_voltage_V"], requested[len(attempts):]):
            raise ValueError("accepted/unattempted2D voltage prefixes disagree")
        if document["completed"] is not (len(attempts) == requested.size and all(item["accepted"] for item in attempts)):
            raise ValueError("completed2D voltage result contradicts its attempts")
        terminals = np.asarray(result["terminal_current_per_depth_A_m"])
        expected = -result["voltage_polarity"] * terminals[:, 0] / result["lateral_width_m"]
        if not np.array_equal(result["output_current_A_m2"], expected):
            raise ValueError("output current sign or per-depth width normalization changed")
        if "continuation_prefix" in result["initial_provenance"]:
            binding = result["initial_provenance"]["continuation_prefix"]
            prefix = read_steady_jv_2d(binding["path"], binding["sha256"])
            linked = _linked_tail(prefix, result)
            if json.dumps(_encode(linked), sort_keys=True) != json.dumps(_encode(document.get("continuation")), sort_keys=True):
                raise ValueError("linked2D continuation lost its preserved source, prefix or combined metrics")
        elif "continuation" in document:
            raise ValueError("linked2D continuation requires a bound original prefix artifact")
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError("incomplete steady2D J-V artifact") from exc
    document["provenance_comparison"] = result_provenance(result, current_identity,
        strict_same_source=strict_same_source, expected_inputs=None if result.get("job_provenance") is None else {"compiled_identity": result["compiled_identity"], "geometry_identity": result["geometry"]["identity"],
            "initial_sha256": result["initial_provenance"]["source_sha256"], "voltage_V": result["requested_voltage_V"].tolist(),
            "controls": result["solver_controls"], "incident_power_W_m2": result["incident_power_W_m2"]})
    return document
