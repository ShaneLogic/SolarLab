"""Transient history publication on RunStore, with a solver-free reader.

Primary physical arrays use the existing immutable NPY artifacts. Raw Newton
and accounting diagnostics use the existing lossless numeric codec. A returned
TransientResult1D has no historical source/runtime or node-grid provenance;
only the explicit computation wrapper can supply a contemporaneous capture.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import importlib.metadata
import math
from pathlib import Path
import platform
import re
import sys
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from solarlab.experiments.results import decode_array, encode_array
from solarlab.io.artifacts import (
    ArtifactPayload, _json_document, _parse_document, array_artifact, bytes_artifact,
)

if TYPE_CHECKING:
    from solarlab.equations.dc_1d import CompiledDevice1D
    from solarlab.experiments.transient import TransientOptions, TransientProtocol, TransientResult1D
    from solarlab.io.run_store import AttemptToken, RunStore
    from solarlab.solvers.nonlinear.steady_1d import SteadyOptions, SteadyState1D

__all__ = ["CapturedTransientRun", "run_transient_for_publication",
           "publish_transient_result", "read_transient_result", "transient_result_payload"]

_SCHEMA = "solarlab.transient-result.v1"
_CURRENTS = tuple(prefix + "current_density_A_m2" for prefix in (
    "conduction_", "displacement_", "total_", "terminal_conduction_",
    "terminal_displacement_", "terminal_total_"))
_UNITS = {"requested_times_s": "s", "requested_voltage_V": "V", "requested_generation_scale": "1",
          "accepted_times_s": "s", "nodes_m": "m", "potential_V": "V",
          "electron_density_m3": "m^-3", "hole_density_m3": "m^-3",
          **{name: "A/m^2" for name in _CURRENTS}}
_TAGS = {"$array", "$scalar", "$float", "$tuple"}
_SOURCE_MODULES = (__name__, "solarlab.experiments.transient", "solarlab.solvers.time.backward_euler",
    "solarlab.solvers.nonlinear.newton", "solarlab.solvers.nonlinear.problem",
    "solarlab.solvers.nonlinear.production", "solarlab.solvers.nonlinear.steady_1d",
    "solarlab.experiments.jv.operating_points", "solarlab.equations.dc_1d",
    "solarlab.physics.transport.electrochemical_flux", "solarlab.physics.recombination.bulk",
    "solarlab.physics.recombination.reaction", "solarlab.physics.electrostatics.poisson",
    "solarlab.physics.contacts.boundary_flux", "solarlab.discretization.stable_bernoulli",
    "solarlab.contracts.units")


def _digest(value: Any) -> str:
    return hashlib.sha256(_json_document(value).encode()).hexdigest()


def _fingerprint(array: np.ndarray) -> dict[str, Any]:
    return {"dtype": array.dtype.str, "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}


def _same(left: np.ndarray, right: np.ndarray) -> bool:
    return _fingerprint(left) == _fingerprint(right)


def _encode(value: Any) -> Any:
    """Plain dataclass fields, explicit tuples and original nonfinite words."""
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _encode(getattr(value, field.name)) for field in fields(value)}
    if type(value) is np.ndarray:
        return {"$array": encode_array(value, unit="raw diagnostic; units follow the named producer field")}
    if isinstance(value, np.generic):
        return {"$scalar": encode_array(np.asarray(value), unit="raw diagnostic")}
    if isinstance(value, Enum):
        return _encode(value.value)
    if type(value) is tuple:
        return {"$tuple": [_encode(item) for item in value]}
    if type(value) is list:
        return [_encode(item) for item in value]
    if type(value) is dict:
        if any(type(key) is not str for key in value) or set(value) & _TAGS:
            raise ValueError("diagnostics require plain string keys without codec tags")
        return {key: _encode(item) for key, item in value.items()}
    if type(value) is float and not math.isfinite(value):
        return {"$float": encode_array(np.asarray(value), unit="raw diagnostic")}
    if value is None or type(value) in {str, bool, int, float}:
        return value
    raise ValueError("unsupported raw diagnostic type: " + type(value).__name__)


def _decode(value: Any) -> Any:
    if type(value) is list:
        return [_decode(item) for item in value]
    if type(value) is dict:
        tags = set(value) & _TAGS
        if tags:
            if len(value) != 1:
                raise ValueError("ambiguous raw diagnostic tag")
            tag, = tags
            if tag == "$tuple":
                if type(value[tag]) is not list:
                    raise ValueError("invalid diagnostic tuple")
                return tuple(_decode(item) for item in value[tag])
            array = decode_array(value[tag])
            if tag != "$array":
                if array.shape != ():
                    raise ValueError("diagnostic scalar must have scalar shape")
                return array.item() if tag == "$float" else array[()]
            return array
        return {key: _decode(item) for key, item in value.items()}
    return value


def _validate(data: dict[str, Any], arrays: dict[str, np.ndarray], diagnostics: dict[str, Any]) -> None:
    """Validate relationships and completeness, not physical convergence gates."""
    if (data["schema"] != _SCHEMA or data["engine"] != "solarlab"
            or data["physical_certificate_status"] != "not_evaluated"
            or re.fullmatch(r"[0-9a-f]{64}", data["compiled_identity"]) is None
            or set(arrays) != set(_UNITS)):
        raise ValueError("invalid transient result domain or physical status")
    if any(a.dtype != np.dtype("float64") or not np.all(np.isfinite(a)) for a in arrays.values()):
        raise ValueError("accepted physical arrays must be finite float64 values")
    times, accepted, nodes = arrays["requested_times_s"], arrays["accepted_times_s"], arrays["nodes_m"]
    if (times.ndim != 1 or times.size < 2 or not np.all(np.isfinite(np.diff(times))) or np.any(np.diff(times) <= 0)
            or nodes.ndim != 1 or nodes.size < 2 or np.any(np.diff(nodes) <= 0)
            or accepted.ndim != 1 or not 1 <= accepted.size <= times.size
            or not _same(accepted, times[:accepted.size])):
        raise ValueError("accepted times must be an exact prefix on the declared time/grid vectors")
    if (arrays["requested_voltage_V"].shape != times.shape
            or arrays["requested_generation_scale"].shape != times.shape
            or np.any(arrays["requested_generation_scale"] < 0)):
        raise ValueError("incomplete requested protocol")
    count, width = accepted.size, nodes.size
    for name in ("potential_V", "electron_density_m3", "hole_density_m3"):
        if arrays[name].shape != (count, width):
            raise ValueError("accepted field shape differs from time/node grid: " + name)
    if any(np.any(arrays[name] <= 0) for name in ("electron_density_m3", "hole_density_m3")):
        raise ValueError("accepted carrier densities must be positive")
    for name in _CURRENTS:
        expected = (count - 1, 2 if name.startswith("terminal_") else width - 1)
        if arrays[name].shape != expected:
            raise ValueError("step-current shape differs from accepted right endpoints: " + name)
    attempts = diagnostics["attempts"]
    accepted_flags = [a["accepted"] for a in attempts]
    if (any(type(flag) is not bool for flag in accepted_flags)
            or accepted_flags not in ([True] * (count - 1), [True] * (count - 1) + [False])):
        raise ValueError("attempts must preserve the accepted prefix and at most one final failure")
    completed = count == times.size and accepted_flags == [True] * (count - 1)
    if not completed and accepted_flags != [True] * (count - 1) + [False]:
        raise ValueError("an incomplete result must retain its first failed attempt")
    expected_outcome = "completed" if completed else "partial"
    reason = None if completed else attempts[-1]["termination"]
    if (data["outcome"] != expected_outcome or data["reason"] != reason
            or data["requested_count"] != times.size or data["accepted_count"] != count
            or data["attempt_count"] != len(attempts)):
        raise ValueError("transient outcome/counts differ from retained history")
    if len(diagnostics["accepted_state_coordinates"]) != count or len(diagnostics["state_device_identities"]) != count:
        raise ValueError("missing accepted state diagnostics")
    if any(identity != data["compiled_identity"] for identity in diagnostics["state_device_identities"]):
        raise ValueError("accepted state belongs to another compiled device")
    for u in diagnostics["accepted_state_coordinates"]:
        if u.shape != (3 * width,) or not np.all(np.isfinite(u)):
            raise ValueError("invalid accepted solver coordinates")
    if not _same(np.asarray(diagnostics["accepted_state_voltage_V"]), arrays["requested_voltage_V"][:count]):
        raise ValueError("state reference voltages differ from accepted protocol")
    initial = diagnostics["initial"]
    if (initial["state"]["device_sha256"] != data["compiled_identity"]
            or initial["newton"]["accepted"] is not True
            or initial["voltage_V"] != arrays["requested_voltage_V"][0]
            or initial["generation_scale"] != arrays["requested_generation_scale"][0]):
        raise ValueError("initial steady state differs from the transient protocol/device")
    for name in ("potential_V", "electron_density_m3", "hole_density_m3"):
        if not _same(initial["state"][name], arrays[name][0]):
            raise ValueError("initial field differs from first accepted state")
    if not _same(initial["state"]["u"], diagnostics["accepted_state_coordinates"][0]):
        raise ValueError("initial coordinate words differ from first accepted state")
    for index, attempt in enumerate(attempts):
        if (attempt["start_s"] != times[index] or attempt["end_s"] != times[index + 1]
                or attempt["voltage_V"] != arrays["requested_voltage_V"][index + 1]
                or attempt["generation_scale"] != arrays["requested_generation_scale"][index + 1]):
            raise ValueError("attempt timing/driving values differ from requested protocol")
        for key in ("initial_coordinates", "candidate_coordinates"):
            if attempt[key].shape != (3 * width,):
                raise ValueError("attempt coordinates have wrong shape")
        if attempt["accepted"]:
            if attempt["newton"] is None or attempt["newton"]["accepted"] is not True or attempt["accounting"] is None:
                raise ValueError("accepted attempt lacks raw Newton/accounting evidence")
            if attempt["accounting"]["physical_certificate_status"] != "not_evaluated":
                raise ValueError("transient accounting cannot promote physical status")
            for name in _CURRENTS:
                if not _same(attempt["accounting"][name], arrays[name][index]):
                    raise ValueError("stored current differs from raw accepted-step accounting")


def _payload(result: TransientResult1D, nodes_m: np.ndarray) -> tuple[dict[str, Any], dict[str, ArtifactPayload]]:
    from solarlab.experiments.transient import TransientResult1D
    if type(result) is not TransientResult1D or type(nodes_m) is not np.ndarray:
        raise ValueError("an actual TransientResult1D and explicit numpy node grid are required")
    arrays = {"requested_times_s": result.protocol.times_s, "requested_voltage_V": result.protocol.voltage_V,
              "requested_generation_scale": result.protocol.generation_scale, "accepted_times_s": result.accepted_times_s,
              "nodes_m": nodes_m}
    for name in ("potential_V", "electron_density_m3", "hole_density_m3"):
        arrays[name] = np.stack([getattr(state, name) for state in result.states])
    for name in _CURRENTS:
        values = [cast(dict[str, Any], attempt.accounting)[name] for attempt in result.attempts if attempt.accepted]
        arrays[name] = (np.stack(values) if values else np.empty((0, 2 if name.startswith("terminal_") else nodes_m.size - 1)))
    diagnostics = _encode({"initial": result.initial, "attempts": result.attempts,
        "accepted_state_coordinates": tuple(state.u for state in result.states),
        "accepted_state_voltage_V": tuple(state.voltage_V for state in result.states),
        "state_device_identities": tuple(state.device_sha256 for state in result.states),
        "solver_controls": result.solver_controls, "initial_controls": result.initial_controls, "wall_s": result.wall_s})
    data = {"schema": _SCHEMA, "engine": "solarlab", "compiled_identity": result.compiled_identity,
        "requested_count": result.protocol.times_s.size, "accepted_count": result.accepted_times_s.size,
        "attempt_count": len(result.attempts), "outcome": "completed" if result.completed else "partial",
        "reason": result.failed_attempts[0].termination if result.failed_attempts else None,
        "physical_certificate_status": result.physical_certificate_status}
    _validate(data, arrays, _decode(diagnostics))
    payloads = {name: array_artifact(array, unit=_UNITS[name], metadata={"schema": _SCHEMA,
                 "compiled_identity": result.compiled_identity}) for name, array in arrays.items()}
    payloads["transient_diagnostics"] = bytes_artifact(_json_document(diagnostics).encode(),
        metadata={"schema": _SCHEMA, "role": "raw_diagnostics", "compiled_identity": result.compiled_identity})
    return data, payloads


def _result_digest(data: dict[str, Any], payloads: dict[str, ArtifactPayload]) -> str:
    return _digest({"summary": data, "payloads": {name: value.sha256 for name, value in payloads.items()}})


def _sources() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    result = {}
    for name in _SOURCE_MODULES:
        origin = getattr(sys.modules.get(name), "__file__", None)
        if origin is None or not Path(origin).resolve().is_relative_to(root):
            raise ValueError("transient execution source cannot be bound: " + name)
        result["solarlab/" + Path(origin).resolve().relative_to(root).as_posix()] = hashlib.sha256(Path(origin).read_bytes()).hexdigest()
    return result


def _inputs(compiled_identity: str, arrays: dict[str, np.ndarray], initial: dict[str, Any], controls: Any) -> dict[str, Any]:
    return {"compiled_identity": compiled_identity,
        "protocol_and_grid": {name: _fingerprint(arrays[name]) for name in
                              ("requested_times_s", "requested_voltage_V", "requested_generation_scale", "nodes_m")},
        "initial_state": {name: _fingerprint(initial["state"][name]) for name in
                          ("u", "potential_V", "electron_density_m3", "hole_density_m3")},
        "initial_controls": _encode(controls)}


@dataclass(frozen=True, slots=True)
class CapturedTransientRun:
    result: TransientResult1D
    nodes_m: np.ndarray
    capture_json: str

    @property
    def identity(self) -> dict[str, str]:
        return dict(_parse_document(self.capture_json)["identity"])


def run_transient_for_publication(
    compiled: CompiledDevice1D, initial: SteadyState1D, protocol: TransientProtocol, *,
    options: TransientOptions, initial_options: SteadyOptions,
) -> CapturedTransientRun:
    """Capture this call's actual inputs/source/runtime, then run the existing solver.

    It makes no claim about when/how the supplied compiled device or initial
    DC state was produced. No HEAD, build commit, release or certificate is
    inferred. Sources are contemporaneous file hashes, not a source archive.
    """
    from solarlab.experiments.transient import solve_transient
    from threadpoolctl import threadpool_info
    nodes = np.frombuffer(compiled.nodes_m.tobytes(), dtype=compiled.nodes_m.dtype).reshape(compiled.nodes_m.shape)
    arrays = {"requested_times_s": protocol.times_s, "requested_voltage_V": protocol.voltage_V,
              "requested_generation_scale": protocol.generation_scale, "nodes_m": nodes}
    inputs = _inputs(compiled.identity, arrays, _decode(_encode(initial)), asdict(initial_options))
    description = {"scope": "solve_transient_call", "inputs": inputs, "numerics": _encode(asdict(options)),
        "sources": _sources(), "runtime": {"python": sys.version, "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "threadpoolctl")},
            "threadpools": threadpool_info()},
        "source_content": "selected execution/admission file hashes; not a complete dependency or release attestation",
        "build_source_commit": None,
        "compiled_device_history": None, "initial_dc_history": None}
    started = datetime.now(timezone.utc).isoformat()
    result = solve_transient(compiled, initial, protocol, options=options, initial_options=initial_options)
    if _sources() != description["sources"]:
        raise ValueError("loaded core source changed during transient computation")
    data, payloads = _payload(result, nodes)
    identity = {"H_input": _digest(inputs), "H_physics": compiled.identity, "H_execution": _digest(description)}
    record = {"status": "captured_at_computation", "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(), "description": description,
        "identity": identity, "result_sha256": _result_digest(data, payloads)}
    return CapturedTransientRun(result, nodes, _json_document(record))


def publish_transient_result(
    store: RunStore, token: AttemptToken, result: TransientResult1D, *, nodes_m: np.ndarray,
    capture: CapturedTransientRun | None = None,
) -> dict[str, Any]:
    """Publish into an already claimed attempt; its owner still commits the terminal."""
    data, payloads = _payload(result, nodes_m)
    identity = store.get_run(token.run_id)["request"]["identity"]
    provenance: dict[str, Any] = {"status": "not_recorded",
        "reason": "returned TransientResult1D has no historical source/runtime or node-grid provenance",
        "node_grid": "caller_supplied; shape_checked_only"}
    if capture is not None:
        if type(capture) is not CapturedTransientRun:
            raise ValueError("capture must come from the explicit computation wrapper")
        record = _parse_document(capture.capture_json)
        if not _same(capture.nodes_m, nodes_m) or record["result_sha256"] != _result_digest(data, payloads):
            raise ValueError("computation capture belongs to different result/grid words")
        if identity != record["identity"]:
            raise ValueError("RunStore identity differs from the captured computation")
        provenance = record
    records = {name: store.publish(token, name, payload) for name, payload in payloads.items()}
    manifest = {"summary": data, "store_id": token.store_id, "run_id": token.run_id, "attempt_id": token.attempt_id, "generation": token.generation,
        "identity": identity, "provenance": provenance,
        "current_alignment": "accepted_times_s[1:]; no displacement current exists at the initial sample",
        "payloads": {name: {"artifact_id": item["artifact_id"], "sha256": item["sha256"],
                     "metadata": item["metadata"]} for name, item in records.items()}}
    raw = _json_document(manifest).encode()
    record = store.publish(token, "transient_manifest", bytes_artifact(raw,
        metadata={"schema": _SCHEMA, "role": "result_manifest", "compiled_identity": result.compiled_identity}))
    return {**data, "store_id": token.store_id, "run_id": token.run_id, "attempt_id": token.attempt_id, "generation": token.generation,
        "manifest_artifact_id": record["artifact_id"], "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "artifact_ids": [*[item["artifact_id"] for item in records.values()], record["artifact_id"]]}


def read_transient_result(store: RunStore, run_id: str, *, attempt_id: str | None = None) -> dict[str, Any]:
    """Read committed complete/partial results and raw diagnostics without a solver."""
    run = store.get_run(run_id)
    selected = run["current_attempt_id"] if attempt_id is None else attempt_id
    attempt = next((a for a in store.attempts(run_id) if a["attempt_id"] == selected), None)
    if attempt is None or attempt["state"] not in {"succeeded", "failed", "cancelled"} or attempt["terminal"] is None:
        raise ValueError("a committed terminal attempt of this run is required")
    document = attempt["result"]
    if type(document) is not dict or document.get("schema") != _SCHEMA:
        raise ValueError("attempt does not contain a transient publication")
    origin = {"run_id": run_id, "attempt_id": selected, "generation": attempt["generation"]}
    if document.get("store_id") != store.store_id or any(document.get(key) != value for key, value in origin.items()):
        raise ValueError("transient result belongs to a different attempt/generation")
    terminal_ids = [item["artifact_id"] for item in attempt["terminal"]["artifacts"]]
    if (len(document["artifact_ids"]) != 16 or len(set(document["artifact_ids"])) != 16
            or len(terminal_ids) != 16 or set(terminal_ids) != set(document["artifact_ids"])):
        raise ValueError("terminal lacks the exact transient artifact set")
    terminal_references = {item["artifact_id"]: item for item in attempt["terminal"]["artifacts"]}

    def record(identifier: str, name: str) -> dict[str, Any]:
        item = store.artifact(identifier)
        if (identifier not in terminal_ids or item["name"] != name or item["phase"] != "committed"
                or any(item[key] != value for key, value in origin.items())
                or item["metadata"].get("schema") != _SCHEMA
                or item["metadata"].get("compiled_identity") != document["compiled_identity"]):
            raise ValueError("transient artifact role/attempt/compiled identity mismatch")
        if item["manifest_sha256"] != terminal_references[identifier]["manifest_sha256"]:
            raise ValueError("terminal reference differs from the committed artifact manifest")
        return item

    manifest_record = record(document["manifest_artifact_id"], "transient_manifest")
    raw = store.read_artifact(manifest_record["artifact_id"])
    if hashlib.sha256(raw).hexdigest() != document["manifest_sha256"]:
        raise ValueError("transient manifest hash mismatch")
    manifest = _parse_document(raw.decode())
    if (manifest["store_id"] != store.store_id or any(manifest[key] != value for key, value in origin.items())
            or manifest["identity"] != run["request"]["identity"]
            or any(document.get(key) != value for key, value in manifest["summary"].items())
            or set(manifest["payloads"]) != {*_UNITS, "transient_diagnostics"}):
        raise ValueError("transient manifest differs from its terminal result")
    arrays, payloads = {}, {}
    for name, bound in manifest["payloads"].items():
        item = record(bound["artifact_id"], name)
        if item["sha256"] != bound["sha256"] or item["metadata"] != bound["metadata"]:
            raise ValueError("artifact hash/unit/shape metadata differs from its transient manifest")
        content = store.read_artifact(item["artifact_id"])
        payloads[name] = ArtifactPayload(content, item["format"], _json_document(item["metadata"]))
        if name in _UNITS:
            if item["metadata"].get("unit") != _UNITS[name]:
                raise ValueError("unexpected transient physical unit: " + name)
            arrays[name] = store.read_array(item["artifact_id"])
    diagnostics = _decode(_parse_document(payloads["transient_diagnostics"].data.decode()))
    _validate(manifest["summary"], arrays, diagnostics)
    if document["outcome"] != "completed" and attempt["state"] == "succeeded":
        raise ValueError("an incomplete transient cannot claim a succeeded terminal")
    provenance = manifest["provenance"]
    if provenance["status"] == "captured_at_computation":
        description = provenance["description"]
        inputs = _inputs(document["compiled_identity"], arrays, diagnostics["initial"], diagnostics["initial_controls"])
        expected = {"H_input": _digest(inputs), "H_physics": document["compiled_identity"], "H_execution": _digest(description)}
        if (description["scope"] != "solve_transient_call"
                or any(description[name] is not None for name in ("build_source_commit", "compiled_device_history", "initial_dc_history"))
                or description["inputs"] != inputs or provenance["identity"] != expected or manifest["identity"] != expected
                or provenance["result_sha256"] != _result_digest(manifest["summary"], payloads)):
            raise ValueError("transient computation capture differs from the saved inputs/result")
    elif provenance["status"] != "not_recorded":
        raise ValueError("unknown transient provenance status")
    return {"document": document, "manifest": manifest, "arrays": arrays, "diagnostics": diagnostics,
            "run_state": attempt["state"], "provenance": provenance}

# Public entry point for trusted companion preparation.
transient_result_payload = _payload
