"""Immutable 2D steady payloads and readers; no compiler/equation/solver imports."""
from __future__ import annotations

import hashlib
import json
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from solarlab.config.identity import ExecutionManifest
from solarlab.experiments.transient_results import _decode, _encode, _fingerprint, _same
from solarlab.experiments.worker import WorkerRequest
from solarlab.io.artifacts import ArtifactPayload, _json_document, array_artifact, bytes_artifact

if TYPE_CHECKING:
    from solarlab.device.resolved import PreparedDevice
    from solarlab.equations.dc_2d import CompiledDevice2D
    from solarlab.io.run_store import RunStore

__all__ = ["steady_2d_payload", "read_steady_2d_job"]
SCHEMA = "solarlab.steady-2d.result.v1"
RAW = "steady_2d_diagnostics"
COORDINATE_UNIT = "mixed: phi[V], electron/hole dimensionless log coordinates; C order"
RESIDUAL_UNIT = "mixed: Gauss[C/m depth], carrier[particles/(m depth*s)], contact[V or dimensionless log]"
UNITS = {"x_m": "m", "y_m": "m", "cell_areas_m2": "m^2", "face_lengths_m": "m", "face_spacing_m": "m",
    "face_pairs": "1", "accepted_indices": "1", "accepted_voltage_V": "V", "accepted_generation_scale": "1",
    "state_coordinates": COORDINATE_UNIT, "potential_V": "V", "electron_density_m3": "m^-3", "hole_density_m3": "m^-3",
    "face_current_density_A_m2": "A/m^2", "lateral_current_density_A_m2": "A/m^2", "stack_current_density_A_m2": "A/m^2",
    "terminal_current_per_depth_A_m": "A/m", "cross_section_current_per_depth_A_m": "A/m",
    "terminal_mean_current_density_A_m2": "A/m^2", "original_residual": RESIDUAL_UNIT,
    "face_displacement_C_m2": "C/m^2", "carrier_face_particle_flux_m2_s": "m^-2 s^-1"}
STORED_UNITS = {"x_m": "m", "y_m": "m", "cell_areas_m2": "m^2", "face_geometry_m": "m",
    "face_pairs": "1", "accepted_indices": "1", "state_coordinates": COORDINATE_UNIT,
    "potential_V": "V", "carrier_density_m3": "m^-3", "face_current_density_A_m2": "A/m^2",
    "section_terminal_current_A_m": "A/m", "original_residual": RESIDUAL_UNIT}
COMPONENTS = {"face_geometry_m": ["face_lengths_m", "face_spacing_m"],
    "carrier_density_m3": ["electron_density_m3", "hole_density_m3"],
    "section_terminal_current_A_m": ["lower_terminal", "stack_cross_sections", "upper_terminal"]}


def _raw_bytes(raw: dict[str, Any]) -> bytes:
    # Payload bytes use the supervisor artifact limit, not SourceDocument or metadata limits.
    return json.dumps(_encode(raw), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _load_raw(raw: bytes) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        if len({k for k, _ in items}) != len(items):
            raise ValueError("duplicate raw 2D diagnostic key")
        return dict(items)
    def invalid(value: str) -> None:
        raise ValueError("nonfinite raw JSON must use the lossless codec: " + value)
    data = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(data, dict):
        raise ValueError("raw 2D diagnostics must be an object")
    return dict(_decode(data))


def _arrays(raw: dict[str, Any]) -> dict[str, np.ndarray]:
    g = raw["geometry"]
    ny, nx = g["shape_Ny_Nx"]
    n, f = ny * nx, g["face_pairs"].shape[0]
    accepted = [(i, a) for i, a in enumerate(raw["attempts"]) if a["accepted"]]
    def collect(key: str, shape: tuple[int, ...], *, state: bool = False) -> np.ndarray:
        if not accepted:
            return np.empty((0, *shape), dtype=np.float64)
        values = [a["result_2d"]["state" if state else "diagnostics"][key] for _, a in accepted]
        return np.stack(values)
    arrays = {key: np.asarray(g[key]) for key in ("x_m", "y_m", "cell_areas_m2", "face_lengths_m", "face_spacing_m", "face_pairs")}
    arrays.update(accepted_indices=np.asarray([i for i, _ in accepted], dtype=np.int64),
        accepted_voltage_V=np.asarray([a["target"]["voltage_V"] for _, a in accepted], dtype=np.float64),
        accepted_generation_scale=np.asarray([a["target"]["generation_scale"] for _, a in accepted], dtype=np.float64),
        state_coordinates=collect("u", (3 * n,), state=True),
        potential_V=collect("potential_V", (ny, nx), state=True),
        electron_density_m3=collect("electron_density_m3", (ny, nx), state=True),
        hole_density_m3=collect("hole_density_m3", (ny, nx), state=True))
    shapes = {"face_current_density_A_m2": (f,), "lateral_current_density_A_m2": (ny, nx - 1),
        "stack_current_density_A_m2": (ny - 1, nx), "terminal_current_per_depth_A_m": (2,),
        "cross_section_current_per_depth_A_m": (ny - 1,), "terminal_mean_current_density_A_m2": (2,),
        "original_residual": (3 * n,), "face_displacement_C_m2": (f,),
        "carrier_face_particle_flux_m2_s": (2, f)}
    arrays.update({key: collect(key, shape) for key, shape in shapes.items()})
    return arrays


def _pack(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = {key: arrays[key] for key in STORED_UNITS if key in arrays}
    result.update(face_geometry_m=np.stack((arrays["face_lengths_m"], arrays["face_spacing_m"])),
        carrier_density_m3=np.stack((arrays["electron_density_m3"], arrays["hole_density_m3"])),
        section_terminal_current_A_m=np.concatenate((arrays["terminal_current_per_depth_A_m"][:, :1],
            arrays["cross_section_current_per_depth_A_m"], arrays["terminal_current_per_depth_A_m"][:, 1:]), axis=1))
    return result


def _counts(raw: dict[str, Any]) -> dict[str, Any]:
    seeds, full = [], []
    for attempt in raw["attempts"]:
        seed, failure, result = attempt["auxiliary_1d_seed"], attempt["failed_auxiliary_seed_newton"], attempt["result_2d"]
        if seed is not None:
            seeds.extend([*seed["preparation"], seed["newton"]])
        elif failure is not None:
            seeds.append(failure)
        if result is not None:
            full.append(result)
    return {"auxiliary_1d_seed_calls": sum(a["auxiliary_1d_seed_called"] for a in raw["attempts"]),
        "auxiliary_1d_observed_newton_results": len(seeds),
        "auxiliary_1d_observed_residual_evaluations": sum(r["residual_evaluations"] for r in seeds),
        "auxiliary_1d_observed_jacobian_evaluations": sum(r["jacobian_evaluations"] for r in seeds),
        "auxiliary_1d_complete_counts_observed": all(not a["auxiliary_1d_seed_called"] or a["auxiliary_1d_seed"] is not None for a in raw["attempts"]),
        "density_basin_initializations": sum(a["auxiliary_1d_seed"]["density_basin_initializations"]
            for a in raw["attempts"] if a["auxiliary_1d_seed"] is not None),
        "full_2d_adapter_calls": sum(a["full_2d_adapter_called"] for a in raw["attempts"]),
        "full_2d_newton_calls": sum(r["solver_controls"]["full_2d_newton_calls"] for r in full),
        "full_2d_evaluation_counts": {name: sum(r["evaluation_counts"][name] for r in full)
            for name in ("residual", "jacobian", "diagnostics")},
        "scope": "separate auxiliary1D invocation/observed Newton records and actual full2D callbacks; no inferred fallback calls"}


def _summary(raw: dict[str, Any]) -> dict[str, Any]:
    attempts = raw["attempts"]
    failed = next(({"index": i, "target": a["target"], "termination": a["termination"], "error": a["error"],
                   "full_2d_adapter_called": a["full_2d_adapter_called"]} for i, a in enumerate(attempts) if not a["accepted"]), None)
    count = sum(a["accepted"] for a in attempts)
    complete = len(attempts) == len(raw["requested_targets"]) and all(a["accepted"] for a in attempts)
    return {"schema": SCHEMA, "dimension": 2, "engine": "solarlab", "completed": complete,
        "reason": None if complete else "no_attempt" if failed is None else failed["termination"],
        "requested_count": len(raw["requested_targets"]), "attempt_count": len(attempts),
        "accepted_count": count, "unattempted_count": len(raw["requested_targets"]) - len(attempts),
        "first_failure": failed, "prepared_sha256": raw["prepared_identity"], "compiled_identity": raw["compiled_identity"],
        "geometry_identity": raw["geometry"]["identity"], "shape_Ny_Nx": list(raw["geometry"]["shape_Ny_Nx"]),
        "flatten_order": "C: y-major, x-fastest; coordinates [phi,z_n,z_p]",
        "lateral_width_m": raw["geometry"]["lateral_width_m"], "face_order": "x faces then y faces; positive coordinate direction",
        "array_alignment": "accepted target prefix only; failed candidate/state/diagnostics retained exclusively in raw attempts",
        "computation_counts": _counts(raw), "physical_certificate_status": "not_evaluated",
        "scientific_qualification": False, "mesh_qualification": "not_evaluated", "photovoltaic_metrics": None}


def _validate(raw: dict[str, Any], summary: dict[str, Any], arrays: dict[str, np.ndarray]) -> None:
    if raw.get("schema") != SCHEMA or raw.get("qualification") is not False or summary != _summary(raw):
        raise ValueError("2D summary differs from original raw outcome")
    g = raw["geometry"]
    if tuple(g["shape_Ny_Nx"]) != (13, 5) or g["lateral_width_m"] != 1e-6:
        raise ValueError("2D result is outside the declared fixed geometry")
    ny, nx = g["shape_Ny_Nx"]
    n, f, k = ny * nx, ny * (nx - 1) + (ny - 1) * nx, summary["accepted_count"]
    expected_shapes = {"x_m": (nx,), "y_m": (ny,), "cell_areas_m2": (ny, nx), "face_lengths_m": (f,),
        "face_spacing_m": (f,), "face_pairs": (f, 2), "accepted_indices": (k,), "accepted_voltage_V": (k,),
        "accepted_generation_scale": (k,), "state_coordinates": (k, 3 * n), "potential_V": (k, ny, nx),
        "electron_density_m3": (k, ny, nx), "hole_density_m3": (k, ny, nx),
        "face_current_density_A_m2": (k, f), "lateral_current_density_A_m2": (k, ny, nx - 1),
        "stack_current_density_A_m2": (k, ny - 1, nx), "terminal_current_per_depth_A_m": (k, 2),
        "cross_section_current_per_depth_A_m": (k, ny - 1), "terminal_mean_current_density_A_m2": (k, 2),
        "original_residual": (k, 3 * n), "face_displacement_C_m2": (k, f), "carrier_face_particle_flux_m2_s": (k, 2, f)}
    if set(arrays) != set(UNITS):
        raise ValueError("missing logical 2D arrays")
    for name, shape in expected_shapes.items():
        array = arrays[name]
        kind = "i" if name in {"accepted_indices", "face_pairs"} else "f"
        if array.shape != shape or array.dtype.kind != kind or not np.all(np.isfinite(array)):
            raise ValueError("2D shape/dtype/nonfinite field mismatch: " + name)
    expected_pairs = [(iy * nx + ix, iy * nx + ix + 1) for iy in range(ny) for ix in range(nx - 1)]
    expected_pairs += [(iy * nx + ix, (iy + 1) * nx + ix) for iy in range(ny - 1) for ix in range(nx)]
    if not np.array_equal(arrays["face_pairs"], expected_pairs) or g["x_face_count"] != ny * (nx - 1):
        raise ValueError("2D directed-face connectivity/order differs")
    if (np.any(np.diff(arrays["x_m"]) <= 0) or np.any(np.diff(arrays["y_m"]) <= 0)
            or any(np.any(arrays[name] <= 0) for name in ("cell_areas_m2", "face_lengths_m", "face_spacing_m"))):
        raise ValueError("invalid physical 2D axes/measures")
    flags = [a["accepted"] for a in raw["attempts"]]
    if (not flags or len(flags) > len(raw["requested_targets"]) or any(type(flag) is not bool for flag in flags)
            or flags != ([True] * k if summary["completed"] else [True] * k + [False])):
        raise ValueError("2D accepted prefix/first failure was changed")
    for i, attempt in enumerate(raw["attempts"]):
        if (attempt["target"] != raw["requested_targets"][i] or attempt["auxiliary_seed_is_2d_completion"] is not False
                or type(attempt["auxiliary_1d_seed_called"]) is not bool or type(attempt["full_2d_adapter_called"]) is not bool):
            raise ValueError("2D attempt target/seed provenance differs")
        result = attempt["result_2d"]
        if result is None:
            if attempt["accepted"]:
                raise ValueError("accepted 2D attempt has no actual 2D result")
            continue
        if (result["compiled_identity"] != raw["compiled_identity"] or result["accepted"] != attempt["accepted"]
                or result["termination"] != attempt["termination"] or result["error"] != attempt["error"]
                or result["operating_voltage_V"] != attempt["target"]["voltage_V"]
                or result["generation_scale"] != attempt["target"]["generation_scale"]
                or result["physical_certificate_status"] != "not_evaluated"):
            raise ValueError("original 2D state/outcome identity differs")
        for name in ("initial_coordinates", "candidate_coordinates"):
            if result[name].shape != (3 * n,):
                raise ValueError("2D original coordinates lost their flattening")
        if result["initial_provenance"]["state_sha256"] != hashlib.sha256(result["initial_coordinates"].tobytes()).hexdigest():
            raise ValueError("2D numerical initial identity differs")
        actual_controls = result["solver_controls"]["numerical"]
        requested_controls = raw["request"]["controls"]["full_2d"]
        if (actual_controls["electronic"] != requested_controls["electronic"]
                or not 0 < actual_controls["wall_s"] <= requested_controls["wall_s"]
                or result["solver_controls"]["hidden_seed_or_fallback_calls"] != 0):
            raise ValueError("2D controls/hidden solver accounting differs")
        state, newton, diagnostics = result["state"], result["newton"], result["diagnostics"]
        if newton is not None and not _same(newton["state"], result["candidate_coordinates"]):
            raise ValueError("2D candidate differs from original Newton state")
        if state is not None and (state["device_sha256"] != raw["compiled_identity"]
                or not _same(state["u"], result["candidate_coordinates"])):
            raise ValueError("2D field state differs from candidate/source")
        if diagnostics is not None and tuple(diagnostics["original_rows"]) != tuple(range(3 * n)):
            raise ValueError("2D diagnostic original rows were omitted")
        if attempt["accepted"] and (state is None or newton is None or diagnostics is None
                or not newton["accepted"] or result["termination"] != "converged" or not result["checks"]
                or any(check["passed"] is not True or not math.isfinite(check["value"]) or check["value"] > check["limit"] for check in result["checks"])):
            raise ValueError("accepted state lacks original full-2D Newton/gate evidence")
        if attempt["accepted"]:
            if not _same(state["u"][:n].reshape(ny, nx), state["potential_V"]):
                raise ValueError("2D potential flattening differs")
            if not np.array_equal(diagnostics["terminal_current_per_depth_A_m"] / g["lateral_width_m"],
                                  diagnostics["terminal_mean_current_density_A_m2"]):
                raise ValueError("2D terminal current lost width normalization")


def result_digest(summary: dict[str, Any], payloads: dict[str, ArtifactPayload]) -> str:
    return hashlib.sha256(_json_document({"summary": summary,
        "payloads": {key: value.sha256 for key, value in payloads.items()}}).encode()).hexdigest()


def steady_2d_payload(device: CompiledDevice2D, prepared: PreparedDevice, body: dict[str, Any],
                      attempts: list[dict[str, Any]], wall_s: float) -> tuple[dict[str, Any], dict[str, ArtifactPayload]]:
    g = device.geometry
    raw = {"schema": SCHEMA, "qualification": False, "request": body,
        "requested_targets": [{"voltage_V": body["protocol"]["voltage_V"], "generation_scale": scale}
                              for scale in body["protocol"]["generation_scales"]],
        "attempts": attempts, "prepared_identity": prepared.content_sha256, "prepared_declaration": prepared.to_mapping(),
        "compiled_identity": device.identity, "column_identity": device.column.identity,
        "generation_identity": device.generation_y.binding_sha256, "wall_s": wall_s,
        "geometry": {"identity": g.sha256, "shape_Ny_Nx": g.shape, "x_m": g.x_m, "y_m": device.column.nodes_m,
            "cell_areas_m2": g.contract_geometry.volumes.reshape(g.shape), "face_lengths_m": g.contract_geometry.face_measures,
            "face_pairs": g.contract_geometry.pairs, "face_spacing_m": g.face_spacing_m,
            "x_widths_m": g.x_widths_m, "x_face_count": g.x_face_count, "lateral_width_m": g.lateral_length_m},
        "generation_source_m3_s": device.generation_source_m3_s.reshape(g.shape)}
    content = _raw_bytes(raw)
    original = _load_raw(content)
    arrays = _arrays(original)
    summary = _summary(original)
    _validate(original, summary, arrays)
    metadata = {"schema": SCHEMA, "compiled_identity": device.identity, "geometry_identity": g.sha256,
        "array_alignment": "accepted target prefix", "flatten_order": summary["flatten_order"]}
    payloads = {name: array_artifact(a, unit=STORED_UNITS[name],
        metadata={**metadata, "components": COMPONENTS.get(name, [])}) for name, a in _pack(arrays).items()}
    payloads[RAW] = bytes_artifact(content, metadata={**metadata, "role": "original_auxiliary_and_2d_attempts"})
    return summary, payloads


def read_steady_2d_job(store: RunStore, run_id: str, *, attempt_id: str | None = None) -> dict[str, Any]:
    from solarlab.experiments.two_dimensional.steady_worker import REGISTRY, REQUEST_SCHEMA, RESULT_SCHEMA
    run = store.get_run(run_id)
    selected = run["current_attempt_id"] if attempt_id is None else attempt_id
    attempt = next((a for a in store.attempts(run_id) if a["attempt_id"] == selected), None)
    if attempt is None or attempt["state"] not in {"succeeded", "failed"} or attempt["terminal"] is None:
        raise ValueError("committed completed/failed 2D result required; cancellation has no invented fields")
    worker = (attempt["result"] or {}).get("worker_result", {})
    if worker.get("schema") != RESULT_SCHEMA or worker.get("engine") != "solarlab" or worker.get("scientific_qualification") is not False:
        raise ValueError("attempt has no returned new-core 2D result")
    identity = run["request"]["identity"]
    metadata = run["request"]["input_metadata"]["execution_manifest"]
    if (any(worker[k] != identity[k] for k in ("H_input", "H_physics", "H_execution"))
            or worker["manifest_sha256"] != metadata["manifest_sha256"]):
        raise ValueError("2D result belongs to another input/physics/execution")
    refs = attempt["terminal"]["artifacts"]
    records = [store.artifact(ref["artifact_id"]) for ref in refs]
    by_name = {r["name"]: r for r in records}
    if (len(records) != len(STORED_UNITS) + 2 or set(by_name) != {*STORED_UNITS, RAW, "execution_manifest"}
            or set(worker["payloads"]) != {*STORED_UNITS, RAW}):
        raise ValueError("2D terminal does not own exactly the required payloads")
    for ref, item in zip(refs, records):
        if (item["phase"] != "committed" or item["run_id"] != run_id or item["attempt_id"] != selected
                or item["generation"] != attempt["generation"] or item["manifest_sha256"] != ref["manifest_sha256"]):
            raise ValueError("2D artifact ownership/generation differs")
    # ExecutionManifest keeps its 64MiB aggregate domain, distinct from SourceDocument.
    manifest = ExecutionManifest(store.read_artifact(by_name["execution_manifest"]["artifact_id"]), metadata["current_repository_head"])
    document = manifest.export()["document"]
    bound = document["execution_identity"]["payload"]
    if (manifest.sha256 != worker["manifest_sha256"] or manifest.execution_sha256 != identity["H_execution"]
            or any(bound[k] != identity[k] for k in ("H_input", "H_physics")) or bound["engine"]["name"] != "solarlab"):
        raise ValueError("2D execution manifest identity differs")
    summary = worker["steady_2d_result"]
    payloads, packed = {}, {}
    for name, data in worker["payloads"].items():
        item = by_name[name]
        if (data != {"sha256": item["sha256"], "format": item["format"], "metadata": item["metadata"]}
                or item["metadata"].get("schema") != SCHEMA
                or item["metadata"].get("compiled_identity") != summary["compiled_identity"]
                or item["metadata"].get("geometry_identity") != summary["geometry_identity"]):
            raise ValueError("2D immutable payload metadata differs")
        payloads[name] = ArtifactPayload(store.read_artifact(item["artifact_id"]), item["format"], _json_document(item["metadata"]))
        if name in STORED_UNITS:
            if item["metadata"].get("unit") != STORED_UNITS[name] or item["metadata"].get("components") != COMPONENTS.get(name, []):
                raise ValueError("2D artifact unit/components differ")
            packed[name] = store.read_array(item["artifact_id"])
    raw = _load_raw(payloads[RAW].data)
    arrays = _arrays(raw)
    _validate(raw, summary, arrays)
    expected = _pack(arrays)
    if set(packed) != set(expected) or any(not _same(packed[k], expected[k]) for k in expected):
        raise ValueError("2D NPY payload differs from original producer words")
    if attempt["state"] != ("succeeded" if summary["completed"] else "failed") or worker["reason"] != summary["reason"]:
        raise ValueError("2D terminal outcome differs from original numerical truth")
    transport = WorkerRequest.from_bytes(_json_document(run["request"]["input_metadata"]["transport"]).encode())
    body = json.loads(transport.request.content)
    roles = {name: json.loads(bytes.fromhex(v["content_hex"])) for name, v in document["content_roles"].items()}
    if (transport.registry != REGISTRY or transport.schema != REQUEST_SCHEMA
            or transport.execution_identity.sha256 != identity["H_execution"]
            or transport.H_input != identity["H_input"] or transport.H_physics != identity["H_physics"]
            or body != raw["request"] or body["schema"] != REQUEST_SCHEMA.id or body["engine"] != "solarlab"
            or body["prepared_sha256"] != summary["prepared_sha256"] or worker["prepared_sha256"] != summary["prepared_sha256"]
            or roles["preparation"]["dimension"] != 2 or roles["preparation"]["resolved_device"] != raw["prepared_declaration"]
            or roles["preparation"]["prepared_sha256"] != summary["prepared_sha256"]
            or any(body[key] != roles[role] for key, role in (("protocol", "protocol"), ("spatial", "grid"),
                ("controls", "numeric_controls"), ("initial", "initial_state")))):
        raise ValueError("2D request/source/grid/controls do not match the saved producer")
    sources = {s["id"]: s["sha256"] for s in document["implementation_content"]}
    sources.update({s["id"]: s["sha256"] for s in roles["preparation"]["adapter_sources"]})
    capture = worker["capture"]
    if (capture["source_sha256"] != sources or capture["build_source_commit"] != bound["implementation"]["build_source_commit"]
            or capture["runtime"] != bound["environment"] or capture["manifest_sha256"] != manifest.sha256
            or capture["request_sha256"] != transport.request.sha256 or capture["prepared_sha256"] != summary["prepared_sha256"]
            or capture["compiled_identity"] != summary["compiled_identity"] or capture["geometry_identity"] != summary["geometry_identity"]
            or capture["result_sha256"] != result_digest(summary, payloads)):
        raise ValueError("2D computation capture differs from bound sources/runtime/result")
    for item in raw["attempts"]:
        result = item["result_2d"]
        if result is not None and any(sources.get(name.replace(".", "/") + ".py") != digest for name, digest in result["source_sha256"].items()):
            raise ValueError("original 2D numerical source differs from execution binding")
    return {"document": {**summary, "run_id": run_id, "attempt_id": selected, "generation": attempt["generation"],
        "logical_array_units": UNITS, "arrays": {name: by_name[name]["artifact_id"] for name in STORED_UNITS}},
        "run_state": attempt["state"], "arrays": arrays, "stored_arrays": packed, "diagnostics": raw,
        "worker_result": worker, "manifest": manifest.export(),
        "array_fingerprints": {name: _fingerprint(a) for name, a in arrays.items()}}
