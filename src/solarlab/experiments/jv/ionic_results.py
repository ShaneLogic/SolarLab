"""Immutable ionic steady J-V payloads and data-only committed readback.

Raw attempts retain original finite/nonfinite failure words. Arrays contain
accepted points only; steady currents have no displacement/time interpretation.
"""
from __future__ import annotations

from solarlab.reproducibility.job_provenance import (
    JobProvenance, validate_recorded_sources, worker_result_provenance,
)

from dataclasses import asdict
import hashlib
import json
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from solarlab.config.identity import ExecutionManifest
from solarlab.experiments.ionic_transient_results import _finite, _plain, result_digest
from solarlab.experiments.jv.steady import SteadyJVMetrics, steady_jv_metrics
from solarlab.experiments.transient_results import _decode, _encode, _fingerprint, _same
from solarlab.experiments.worker import WorkerRequest
from solarlab.io.artifacts import ArtifactPayload, _json_document, array_artifact, bytes_artifact

if TYPE_CHECKING:
    from solarlab.equations.ionic_1d import CompiledIonicDevice1D
    from solarlab.experiments.ionic_initialization import IonicInitializationResult
    from solarlab.experiments.jv.ionic_steady import IonicSteadyJVResult
    from solarlab.io.run_store import RunStore
    from solarlab.solvers.nonlinear.ionic_1d import IonicSteadyResult

SCHEMA = "solarlab.ionic-jv.result.v1"
INITIAL = "ionic_jv_initial"
RAW = "ionic_jv_diagnostics"
CURRENTS = ("electronic_current_density_A_m2", "ionic_current_density_A_m2", "total_current_density_A_m2")


def _identity(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def stored_units() -> dict[str, str]:
    return {"requested_voltage_V": "V", "voltage_V": "V", "nodes_m": "m", "ion_node_indices": "1",
        "ion_cell_measures_m": "m", "state_coordinates": "mixed: first N in V; remaining coordinates dimensionless",
        "potential_V": "V", "carrier_density_m3": "m^-3", "ion_density_m3": "m^-3", "ion_inventory_m2": "m^-2",
        "current_density_A_m2": "A/m^2", "output_current_A_m2": "A/m^2"}


def components(species: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    return {"carrier_density_m3": ("electron_density_m3", "hole_density_m3"),
        "ion_density_m3": tuple(s + "_ion_density_m3" for s in species),
        "ion_inventory_m2": tuple(s + "_ion_inventory_m2" for s in species), "current_density_A_m2": CURRENTS}


def units(species: tuple[str, ...]) -> dict[str, str]:
    groups = components(species)
    return {**{k: v for k, v in stored_units().items() if k not in groups}, "ion_nodes_m": "m",
        **{key: stored_units()[name] for name, keys in groups.items() for key in keys},
        **{"terminal_" + name: "A/m^2" for name in CURRENTS},
        **{s + "_ion_cell_inventory_m2": "m^-2" for s in species}}


def initial_document(compiled_identity: str, result: IonicSteadyResult | None, voltage: float, generation: float) -> bytes:
    return _json_document({"summary": {"accepted": result is not None and result.accepted,
        "compiled_identity": compiled_identity, "voltage_V": voltage, "generation_scale": generation,
        "qualification": False}, "result": _plain(result)}).encode()


def _diagnostic_payload(raw: dict[str, Any], metadata: dict[str, Any]) -> ArtifactPayload:
    """Full histories use the artifact byte ceiling; metadata keeps its own cap."""
    return bytes_artifact(json.dumps(_encode(raw), sort_keys=True, separators=(",", ":"), allow_nan=False).encode(),
                          metadata={**metadata, "role": "original_ionic_jv_attempts"})


def _parse_diagnostics(data: bytes) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        if len({k for k, _ in items}) != len(items):
            raise ValueError("duplicate raw diagnostic key")
        return dict(items)

    def check(value: Any, depth: int = 0) -> None:
        if depth > 32:
            raise ValueError("raw diagnostics exceed supported nesting depth")
        if isinstance(value, dict):
            for k, v in value.items():
                k.encode("utf-8")
                check(v, depth + 1)
        elif isinstance(value, list):
            for v in value:
                check(v, depth + 1)
        elif isinstance(value, str):
            value.encode("utf-8")
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("nonfinite raw JSON must use the lossless numerical codec")

    value = json.loads(data, object_pairs_hook=pairs)
    check(value)
    if not isinstance(value, dict):
        raise ValueError("raw diagnostic artifact must be an object")
    return _decode(value)


def _arrays(raw: dict[str, Any]) -> dict[str, np.ndarray]:
    curve, grid = raw["curve"], raw["grid"]
    accepted = [] if curve is None else [a for a in curve["attempts"] if a["accepted"]]
    states = [a["solution"]["state"] for a in accepted]
    N, M, K = len(grid["nodes_m"]), len(grid["ion_node_indices"]), len(states)
    a = {name: np.asarray(value, dtype=np.int64 if name == "ion_node_indices" else np.float64) for name, value in grid.items()}
    a.update(requested_voltage_V=np.asarray(raw["protocol"]["voltage_V"], dtype=np.float64),
        voltage_V=np.asarray([] if curve is None else curve["voltage_V"], dtype=np.float64),
        output_current_A_m2=np.asarray([] if curve is None else curve["output_current_A_m2"], dtype=np.float64),
        state_coordinates=np.stack([s["u"] for s in states]) if K else np.empty((0, raw["state_size"])))
    a["ion_nodes_m"] = a["nodes_m"][a["ion_node_indices"]]
    for name in ("potential_V", "electron_density_m3", "hole_density_m3"):
        a[name] = np.stack([s["electronic_state"][name] for s in states]) if K else np.empty((0, N))
    for species in raw["ion_species"]:
        a[species + "_ion_density_m3"] = np.stack([s[species + "_ion_density_m3"] for s in states]) if K else np.empty((0, M))
        a[species + "_ion_cell_inventory_m2"] = a[species + "_ion_density_m3"] * a["ion_cell_measures_m"]
        key = "domain_inventory_m2" if species == "positive" else "negative_domain_inventory_m2"
        a[species + "_ion_inventory_m2"] = np.asarray([s[key] for s in states], dtype=np.float64)
    for name in CURRENTS:
        a[name] = np.stack([p["solution"]["diagnostics"][name] for p in accepted]) if K else np.empty((0, N + 1))
        a["terminal_" + name] = a[name][:, [0, -1]]
    return a


def _pack(arrays: dict[str, np.ndarray], species: tuple[str, ...]) -> dict[str, np.ndarray]:
    groups = components(species)
    return {name: np.stack([arrays[k] for k in groups[name]]) if name in groups else arrays[name] for name in stored_units()}


def _metrics(voltage: np.ndarray, current: np.ndarray, incident: float | None) -> dict[str, Any]:
    reason = "insufficient_accepted_points" if voltage.size < 2 else "zero_voltage_not_bracketed" if not min(voltage) <= 0 <= max(voltage) else None
    return asdict(SteadyJVMetrics(None, None, None, None, None, reason) if reason else
                  steady_jv_metrics(voltage, current, incident_power_W_m2=incident))


def _counts(raw: dict[str, Any]) -> dict[str, Any]:
    dark, light, curve = (raw[k] for k in ("dark_steady", "illumination", "curve"))
    cold = {"solver_calls": int(dark is not None), "evaluation_counts": {} if dark is None else dark["evaluation_counts"]}
    initialization = {k: 0 for k in ("density_preparation_calls", "density_residual", "density_jacobian", "component_solver_calls",
        "component_residual", "component_jacobian", "coupled_solver_calls", "coupled_residual", "coupled_jacobian")}
    if light is not None:
        initialization = light["evaluation_counts"]
    sweep = {"solver_calls": 0 if curve is None else curve["solver_calls"],
             "evaluation_counts": {} if curve is None else curve["evaluation_counts"]}
    return {"cold_steady": cold, "illumination": initialization, "curve": sweep,
        "total_solver_calls": cold["solver_calls"] + initialization["component_solver_calls"] + initialization["coupled_solver_calls"] + sweep["solver_calls"],
        "evaluation_scope": "producer-reported solver/preparation evaluations; admission diagnostics excluded",
        "saved_initial_producer_calls_in_current_job": 0}


def _summary(raw: dict[str, Any]) -> dict[str, Any]:
    curve, light, dark = raw["curve"], raw["illumination"], raw["dark_steady"]
    attempts = [] if curve is None else curve["attempts"]
    count = sum(a["accepted"] for a in attempts)
    complete = curve is not None and count == len(raw["protocol"]["voltage_V"])
    failure = None if curve is None else next((a for a in attempts if not a["accepted"]), None)
    if curve is not None and not complete and failure is None:
        raise ValueError("incomplete curve lacks its first failed attempt")
    reason = (light["termination"] if light is not None else dark["termination"]) if curve is None else None if failure is None else failure["termination"]
    return {"schema": SCHEMA, "engine": "solarlab", "compiled_identity": _identity(raw["compiled_binding"]),
        "prepared_sha256": raw["compiled_binding"]["original_prepared_sha256"], "ion_species": list(raw["ion_species"]),
        "ion_site_contract": raw["compiled_binding"]["ion_model"].get("site_contract", "independent_finite_sites"),
        "requested_count": len(raw["protocol"]["voltage_V"]), "accepted_count": count, "attempt_count": len(attempts),
        "outcome": "initial_failed" if curve is None else "completed" if complete else "partial", "reason": reason,
        "first_failure": None if failure is None else {k: failure[k] for k in ("index", "voltage_V", "termination", "solver_called", "error")},
        "metrics": _metrics(np.asarray([]), np.asarray([]), raw["protocol"]["incident_power_W_m2"]) if curve is None else curve["metrics"],
        "incident_power_W_m2": raw["protocol"]["incident_power_W_m2"], "generation_scale": raw["protocol"]["generation_scale"],
        "initial_policy": raw["initial_recipe"]["policy"], "computation_counts": _counts(raw),
        "physical_certificate_status": "not_evaluated", "voltage_grid_qualification": "not_evaluated",
        "current_alignment": "accepted voltage samples; all faces include both ports; steady conduction only",
        "metric_basis": "piecewise-linear accepted samples; fill factor and efficiency are fractions"}


def _validate_point(point: dict[str, Any], binding: dict[str, Any], controls: dict[str, Any], *, exact_wall: bool) -> None:
    if point["compiled_identity"] != _identity(binding) or point["binding_manifest"] != binding or point["qualification"] is not False:
        raise ValueError("point identity/physical status differs from actual compiled input")
    declared = dict(point["solver_controls"])
    actual_wall = declared["wall_s"]
    if not exact_wall:
        declared["wall_s"] = controls["wall_s"]
    if declared != controls or not 0 < actual_wall <= controls["wall_s"]:
        raise ValueError("point controls differ from declared physical controls/remaining wall cap")
    if not point["accepted"]:
        return
    state, newton = point["state"], point["newton"]
    if (newton is None or newton["accepted"] is not True or point["termination"] != "converged"
            or not point["checks"] or any(c["passed"] is not True or not math.isfinite(c["value"])
                or not math.isfinite(c["limit"]) or c["value"] > c["limit"] for c in point["checks"])
            or state is None or not _finite(state) or not _finite(point["diagnostics"])
            or state["device_sha256"] != point["compiled_identity"]
            or not _same(newton["state"], state["u"]) or not _same(point["candidate_coordinates"], state["u"])):
        raise ValueError("contradictory accepted steady state/checks/diagnostics")


def validate_result(summary: dict[str, Any], raw: dict[str, Any], arrays: dict[str, np.ndarray], initial_bytes: bytes) -> None:
    """Verify stored relationships; never reevaluate a physical residual."""
    binding, curve, species = raw["compiled_binding"], raw["curve"], tuple(raw["ion_species"])
    if summary != _summary(raw) or species not in (("positive",), ("positive", "negative")):
        raise ValueError("ionic J-V summary differs from original attempts")
    if set(arrays) != set(units(species)) or any(not np.all(np.isfinite(a)) or a.dtype != np.dtype(
            "int64" if k == "ion_node_indices" else "float64") for k, a in arrays.items()):
        raise ValueError("accepted ionic arrays must be complete finite original words")
    a, layout, domain = arrays, binding["state_layout"], binding["ion_domain"]
    N, M, K = len(a["nodes_m"]), len(a["ion_node_indices"]), len(a["voltage_V"])
    if (N != layout["electrical_node_count"] or M != layout["ion_node_count"] or raw["state_size"] != layout["size"]
            or raw["state_size"] != 3 * N + len(species) * M or list(species) != layout.get("ion_species", ["positive"])
            or domain["contacts"] != ["blocking", "blocking"] or np.any(np.diff(a["nodes_m"]) <= 0)
            or np.any(np.diff(a["ion_node_indices"]) != 1) or np.any(a["ion_cell_measures_m"] <= 0)
            or not np.array_equal(a["ion_node_indices"], domain["electrical_node_indices"])
            or not np.array_equal(a["ion_cell_measures_m"], domain["cell_measures_m"])):
        raise ValueError("ionic domain/grid/state layout differs from declared compiled binding")
    protocol, recipe = raw["protocol"], raw["initial_recipe"]
    requested = a["requested_voltage_V"]
    if (requested.ndim != 1 or not 1 <= requested.size <= raw["controls"]["max_points"]
            or not (np.all(np.diff(requested) > 0) or np.all(np.diff(requested) < 0))
            or not _same(a["voltage_V"], requested[:K]) or protocol["generation_scale"] <= 0
            or (protocol["incident_power_W_m2"] is not None and protocol["incident_power_W_m2"] <= 0)):
        raise ValueError("invalid ionic J-V requested/accepted protocol")
    initial = json.loads(initial_bytes)
    initial_sha = hashlib.sha256(initial_bytes).hexdigest()
    if initial["summary"]["compiled_identity"] != summary["compiled_identity"] or initial["summary"]["qualification"] is not False:
        raise ValueError("initial artifact lacks exact compiled identity/qualification")
    dark, light = raw["dark_steady"], raw["illumination"]
    if recipe["policy"] == "saved_accepted":
        if (dark is not None or light is not None or initial_sha != recipe["artifact_sha256"]
                or initial_bytes.hex() != recipe["artifact_content_hex"]):
            raise ValueError("supplied initial bytes changed or were silently recomputed")
    elif recipe["policy"] == "dark_direct_density_seed_v1":
        if dark is None:
            raise ValueError("cold recipe lacks its real first solve")
        _validate_point(dark, binding, raw["controls"]["steady"], exact_wall=True)
        expected_initial = dark if light is None else light["coupled_result"]
        if initial["result"] != _plain(expected_initial):
            raise ValueError("initial artifact differs from raw producer result")
        if light is not None:
            if (dark["accepted"] is not True or light["compiled_identity"] != summary["compiled_identity"]
                    or light["binding_manifest"] != binding or light["qualification"] is not False
                    or light["physical_certificate_status"] != "not_evaluated"
                    or light["solver_controls"] != {"steady": raw["controls"]["steady"], "policy": "direct_density_seed_v1", "wall_s": recipe["wall_s"]}
                    or light["evaluation_counts"]["component_solver_calls"] != 0
                    or light["generation_scale"] != protocol["generation_scale"] or light["voltage_V"] != recipe["voltage_V"]
                    or json.loads(light["initial"]["evidence_json"])["result"] != _plain(dark)
                    or hashlib.sha256(light["initial"]["evidence_json"].encode()).hexdigest() != light["initial"]["source_sha256"]):
                raise ValueError("direct-density initialization provenance/controls/counts differ")
            if light["coupled_result"] is not None:
                _validate_point(light["coupled_result"], binding, raw["controls"]["steady"], exact_wall=True)
    else:
        raise ValueError("unsupported initial policy")
    if curve is None:
        if K or summary["outcome"] != "initial_failed" or (dark["accepted"] and (light is None or light["accepted"])):
            raise ValueError("initial failure cannot invent a curve")
        return
    if (initial["summary"]["accepted"] is not True or initial["result"]["accepted"] is not True
            or initial["result"]["solver_controls"] != raw["controls"]["steady"]
            or initial["summary"]["generation_scale"] != protocol["generation_scale"]
            or initial["summary"]["voltage_V"] != recipe["voltage_V"]
            or (light is not None and light["accepted"] is not True)
            or curve["physical_certificate_status"] != "not_evaluated" or curve["experiment_kind"] != "equilibrated_ionic_steady_jv"
            or curve["compiled_identity"] != summary["compiled_identity"] or curve["binding_manifest"] != binding
            or curve["original_prepared_identity"] != summary["prepared_sha256"]
            or curve["solver_controls"] != raw["controls"] or curve["controls_sha256"] != _identity(raw["controls"])
            or curve["source_identity"] != _identity(curve["source_sha256"])):
        raise ValueError("curve initial/source/control identity mismatch")
    provenance = curve["initial_provenance"]
    if (provenance["policy"] != "saved_accepted" or provenance["source_sha256"] != initial_sha
            or _plain(provenance["state"]) != initial["result"]["state"]
            or provenance["state_sha256"] != hashlib.sha256(provenance["state"]["u"].tobytes()).hexdigest()
            or provenance["generation_scale"] != protocol["generation_scale"]):
        raise ValueError("curve initial admission differs from exact initial artifact")
    expected_request = {**protocol, "initial_policy": "saved_accepted", "initial_source_sha256": initial_sha, "controls": raw["controls"]}
    if (curve["request_sha256"] != _identity(expected_request) or curve["generation_scale"] != protocol["generation_scale"]
            or curve["incident_power_W_m2"] != protocol["incident_power_W_m2"] or curve["voltage_polarity"] not in {-1., 1.}
            or not _same(curve["requested_voltage_V"], requested) or not _same(curve["nodes_m"], a["nodes_m"])
            or not _same(curve["ion_weights_m"], a["ion_cell_measures_m"]) or tuple(curve["ion_species"]) != species):
        raise ValueError("curve request/grid/polarity differs from declared values")
    attempts = curve["attempts"]
    flags = [p["accepted"] for p in attempts]
    if (not attempts or len(attempts) > requested.size or any(type(flag) is not bool for flag in flags)
            or flags != ([True] * K if K == len(requested) else [True] * K + [False])
            or curve["solver_calls"] != sum(p["solver_called"] for p in attempts)):
        raise ValueError("J-V must retain exact accepted prefix, first failure and actual calls")
    counts = {"residual": 0, "jacobian": 0}
    for i, attempt in enumerate(attempts):
        point = attempt["solution"]
        if attempt["index"] != i or attempt["voltage_V"] != requested[i] or type(attempt["solver_called"]) is not bool:
            raise ValueError("attempt differs from requested voltage/index")
        if point is not None:
            if attempt["solver_called"] is not True:
                raise ValueError("returned numerical point lacks its solver call")
            _validate_point(point, binding, raw["controls"]["steady"], exact_wall=False)
            if point["solver_controls"]["wall_s"] > raw["controls"]["wall_s"]:
                raise ValueError("point exceeds the declared curve wall budget")
            for key, number in point["evaluation_counts"].items():
                if type(number) is not int or number < 0:
                    raise ValueError("invalid actual evaluation count")
                counts[key] = counts.get(key, 0) + number
        if attempt["accepted"]:
            if (point is None or point["accepted"] is not True or attempt["termination"] != "converged"
                    or point["state"]["electronic_state"]["voltage_V"] != requested[i]):
                raise ValueError("accepted attempt lacks actual converged solution")
            if not _same(point["state"]["u"][:3 * N], point["state"]["electronic_state"]["u"]):
                raise ValueError("ionic/electronic state coordinates disagree")
            for s in species:
                density = a[s + "_ion_density_m3"][i]
                if density.shape != (M,) or np.any(density <= 0) or math.fsum((density * a["ion_cell_measures_m"]).tolist()) != a[s + "_ion_inventory_m2"][i]:
                    raise ValueError("species inventory differs from accepted physical profile")
    if (counts != curve["evaluation_counts"] or a["state_coordinates"].shape != (K, raw["state_size"])
            or any(a[k].shape != (K, N) for k in ("potential_V", "electron_density_m3", "hole_density_m3"))
            or any(a[k].shape != (K, N + 1) for k in CURRENTS)
            or not _same(a[CURRENTS[2]], a[CURRENTS[0]] + a[CURRENTS[1]])
            or not _same(curve["current_density_A_m2"], a[CURRENTS[2]])
            or not _same(a["output_current_A_m2"], -curve["voltage_polarity"] * a[CURRENTS[2]][:, 0])
            or curve["metrics"] != _metrics(a["voltage_V"], a["output_current_A_m2"], protocol["incident_power_W_m2"])):
        raise ValueError("counts/current components/output polarity/accepted-only metrics differ")


def ionic_jv_payload(compiled: CompiledIonicDevice1D, body: dict[str, Any], dark: IonicSteadyResult | None,
    illumination: IonicInitializationResult | None, curve: IonicSteadyJVResult | None, initial_bytes: bytes,
) -> tuple[dict[str, Any], dict[str, ArtifactPayload]]:
    raw = _decode(_encode({"protocol": body["protocol"], "controls": body["controls"], "initial_recipe": body["initial"],
        "dark_steady": dark, "illumination": illumination, "curve": curve, "compiled_binding": compiled.binding_manifest,
        "state_size": compiled.state_size, "ion_species": compiled.ion_species_names,
        "grid": {"nodes_m": compiled.electronic_component.nodes_m, "ion_node_indices": np.asarray(compiled.ion_node_indices, dtype=np.int64),
                 "ion_cell_measures_m": compiled.ion_geometry.cell_measures_m}}))
    arrays, summary = _arrays(raw), _summary(raw)
    validate_result(summary, raw, arrays, initial_bytes)
    metadata = {"schema": SCHEMA, "compiled_identity": compiled.identity}
    payloads = {name: array_artifact(value, unit=stored_units()[name],
        metadata={**metadata, "components": list(components(compiled.ion_species_names).get(name, ()))})
        for name, value in _pack(arrays, compiled.ion_species_names).items()}
    payloads[INITIAL] = bytes_artifact(initial_bytes, metadata={**metadata, "role": "exact_initial_evidence"})
    payloads[RAW] = _diagnostic_payload(raw, metadata)
    return summary, payloads


def read_ionic_jv_job(store: RunStore, run_id: str, *, attempt_id: str | None = None,
    current_identity: JobProvenance | dict[str, Any] | None = None, strict_same_source: bool = False) -> dict[str, Any]:
    """Validate exact terminal, manifest, request and immutable physical payloads."""
    from solarlab.experiments.jv.ionic_worker import REGISTRY, REQUEST_SCHEMA, RESULT_SCHEMA

    run = store.get_run(run_id)
    selected = run["current_attempt_id"] if attempt_id is None else attempt_id
    attempt = next((a for a in store.attempts(run_id) if a["attempt_id"] == selected), None)
    if attempt is None or attempt["state"] not in {"succeeded", "failed"} or attempt["terminal"] is None:
        raise ValueError("a committed completed/failed ionic J-V attempt is required; cancellation has no invented curve")
    result = (attempt["result"] or {}).get("worker_result", {})
    if result.get("schema") != RESULT_SCHEMA or result.get("engine") != "solarlab" or result.get("scientific_qualification") is not False:
        raise ValueError("attempt has no returned ionic J-V result")
    identity, metadata = run["request"]["identity"], run["request"]["input_metadata"]["execution_manifest"]
    if any(result[k] != identity[k] for k in ("H_input", "H_physics", "H_execution")) or result["manifest_sha256"] != metadata["manifest_sha256"]:
        raise ValueError("ionic J-V result identity differs from its attempt")
    summary = result["ionic_jv_result"]
    references = attempt["terminal"]["artifacts"]
    records = [store.artifact(ref["artifact_id"]) for ref in references]
    by_name = {item["name"]: item for item in records}
    expected = {*stored_units(), INITIAL, RAW}
    if len(records) != len(expected) + 1 or set(by_name) != {*expected, "execution_manifest"} or set(result["payloads"]) != expected:
        raise ValueError("ionic J-V terminal lacks the exact artifact roles")
    for ref, item in zip(references, records):
        if (item["phase"] != "committed" or item["run_id"] != run_id or item["attempt_id"] != selected
                or item["generation"] != attempt["generation"] or item["manifest_sha256"] != ref["manifest_sha256"]):
            raise ValueError("artifact differs from committed attempt/reference")
    # ExecutionManifest has a separate64MiB domain; never route it through SourceDocument.
    manifest = ExecutionManifest(store.read_artifact(by_name["execution_manifest"]["artifact_id"]), metadata["current_repository_head"])
    document = manifest.export()["document"]
    bound = document["execution_identity"]["payload"]
    if (manifest.sha256 != result["manifest_sha256"] or manifest.execution_sha256 != identity["H_execution"]
            or any(bound[k] != identity[k] for k in ("H_input", "H_physics")) or bound["engine"]["name"] != "solarlab"):
        raise ValueError("ionic J-V manifest differs from request/result")
    species = tuple(summary["ion_species"])
    stored, payloads = {}, {}
    for name, data in result["payloads"].items():
        item = by_name[name]
        if (data != {"sha256": item["sha256"], "format": item["format"], "metadata": item["metadata"]}
                or item["metadata"].get("schema") != SCHEMA or item["metadata"].get("compiled_identity") != summary["compiled_identity"]):
            raise ValueError("ionic J-V payload metadata differs from result")
        payloads[name] = ArtifactPayload(store.read_artifact(item["artifact_id"]), item["format"], _json_document(item["metadata"]))
        if name in stored_units():
            if (item["metadata"].get("unit") != stored_units()[name]
                    or item["metadata"].get("components") != list(components(species).get(name, ()))):
                raise ValueError("unexpected ionic J-V physical unit/components")
            stored[name] = store.read_array(item["artifact_id"])
    raw = _parse_diagnostics(payloads[RAW].data)
    arrays = _arrays(raw)
    packed = _pack(arrays, species)
    if set(stored) != set(packed) or any(not _same(stored[k], packed[k]) for k in packed):
        raise ValueError("packed physical arrays differ from raw producer words")
    validate_result(summary, raw, arrays, payloads[INITIAL].data)
    if attempt["state"] != ("succeeded" if summary["outcome"] == "completed" else "failed") or result["reason"] != summary["reason"]:
        raise ValueError("terminal state differs from original numerical outcome")
    roles = {name: json.loads(bytes.fromhex(v["content_hex"])) for name, v in document["content_roles"].items()}
    transport = WorkerRequest.from_bytes(_json_document(run["request"]["input_metadata"]["transport"]).encode())
    submitted = json.loads(transport.request.content)
    request_contract = transport.registry == REGISTRY and transport.schema == REQUEST_SCHEMA
    if not request_contract:
        # HTTP retains the actual core request body and supplies an explicit
        # installed catalog. Check its stored contract without server imports.
        catalog = json.loads(transport.registry.content)
        entries = [e for e in catalog.get("bindings", ()) if e.get("driver") == "ionic_steady_jv"]
        request_contract = (transport.registry.id == "solarlab.execution-catalog.v1"
            and transport.schema.id == "solarlab.job-submission.schema.v1"
            and catalog.get("schema") == "solarlab.execution-catalog.v1"
            and catalog.get("job_schema_sha256") == transport.schema.sha256 and len(entries) == 1
            and entries[0].get("experiment_id") == "ionic_steady_jv" and entries[0].get("version") == 1
            and entries[0].get("engine") == "solarlab" and entries[0].get("result_schema") == RESULT_SCHEMA
            and entries[0].get("worker_source_sha256") == result["capture"]["source_sha256"].get("solarlab/experiments/jv/ionic_worker.py")
            and result.get("execution_catalog_sha256") == transport.registry.sha256
            and result.get("experiment_id") == "ionic_steady_jv" and result.get("experiment_version") == 1)
    if (not request_contract
            or transport.H_input != identity["H_input"] or transport.H_physics != identity["H_physics"]
            or transport.execution_identity.sha256 != identity["H_execution"]
            or submitted["schema"] != REQUEST_SCHEMA.id or submitted["engine"] != "solarlab"
            or submitted["prepared_sha256"] != summary["prepared_sha256"]
            or any(submitted[k] != roles[role] for k, role in (("protocol", "protocol"), ("grid", "grid"),
                ("controls", "numeric_controls"), ("initial", "initial_state")))
            or raw["protocol"] != roles["protocol"] or raw["controls"] != roles["numeric_controls"]
            or raw["initial_recipe"] != roles["initial_state"] or result["prepared_sha256"] != summary["prepared_sha256"]
            or roles["preparation"]["prepared_sha256"] != summary["prepared_sha256"]):
        raise ValueError("saved curve differs from exact request/configuration/controls/initial")
    grid = roles["grid"]
    layers = [p for p in roles["preparation"]["resolved_experiment"]["layers"] if p["role"] != "substrate"]
    domain = next((i for i, p in enumerate(layers) if p["id"] == grid["ion_domain_layer"]), None)
    if (domain is None or grid["ion_domain_endpoints"] != ["blocking", "blocking"]
            or raw["compiled_binding"]["ion_domain"]["layer"] != grid["ion_domain_layer"]
            or len(grid["intervals_per_layer"]) != len(layers) or len(arrays["nodes_m"]) != 1 + sum(grid["intervals_per_layer"])
            or len(arrays["ion_nodes_m"]) != 1 + grid["intervals_per_layer"][domain]):
        raise ValueError("compiled grid differs from explicit requested domain")
    capture = result["capture"]
    sources = {s["id"]: s["sha256"] for s in document["implementation_content"]}
    sources.update({s["id"]: s["sha256"] for s in roles["preparation"]["adapter_sources"]})
    curve = raw["curve"]
    for producer in (raw["illumination"], curve):
        if producer is not None:
            if producer.get("job_provenance") is not None:
                if producer["job_provenance"] != result.get("job_provenance"):
                    raise ValueError("numerical producer differs from admitted job provenance")
                validate_recorded_sources(producer)
            elif any(sources.get(name.replace(".", "/") + ".py") != sha for name, sha in producer["source_sha256"].items()):
                raise ValueError("original numerical producer source differs from computation-time binding")
    initial_state = None if curve is None else curve["initial_provenance"]["state"]["u"]
    if (capture["scope"] != "ionic_compile_initialization_and_steady_jv_in_worker"
            or capture["build_source_commit"] != bound["implementation"]["build_source_commit"]
            or capture["source_sha256"] != sources or capture["runtime"] != bound["environment"]
            or capture["manifest_sha256"] != manifest.sha256 or capture["compiled_identity"] != summary["compiled_identity"]
            or capture["request_sha256"] != transport.request.sha256 or capture["prepared_sha256"] != summary["prepared_sha256"]
            or capture["initial_artifact_sha256"] != payloads[INITIAL].sha256
            or capture["initial_state"] != (None if initial_state is None else _fingerprint(initial_state))
            or capture["curve_started"] != (curve is not None) or capture["result_sha256"] != result_digest(summary, payloads)):
        raise ValueError("computation capture differs from actual source/inputs/result")
    return {"document": {**summary, "run_id": run_id, "attempt_id": selected, "generation": attempt["generation"],
        "arrays": {name: by_name[name]["artifact_id"] for name in stored_units()}, "logical_array_units": units(species)},
        "arrays": arrays, "stored_arrays": stored, "diagnostics": raw, "initial_bytes": payloads[INITIAL].data,
        "worker_result": result, "manifest": manifest.export(), "run_state": attempt["state"], "provenance_comparison": worker_result_provenance(result, run, attempt, store_id=store.store_id,
            current=current_identity, strict_same_source=strict_same_source,
            provenance_ref=by_name["execution_manifest"]["metadata"].get("job_provenance_sha256"), execution_document=document)}
