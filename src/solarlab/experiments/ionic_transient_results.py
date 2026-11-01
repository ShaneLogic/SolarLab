"""Ionic history payloads and solver-free structural validation for real workers.

All-face currents include the two ports; terminal conduction/displacement are
exact endpoint selections. Raw diagnostics retain every producer field and
nonfinite failure word using the existing transient numeric codec.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from solarlab.experiments.transient_results import _decode, _digest, _encode, _fingerprint, _same
from solarlab.io.artifacts import ArtifactPayload, _json_document, array_artifact, bytes_artifact

if TYPE_CHECKING:
    from solarlab.equations.ionic_1d import CompiledIonicDevice1D
    from solarlab.experiments.ionic_transient import IonicTransientResult
    from solarlab.experiments.transient import TransientProtocol
    from solarlab.solvers.nonlinear.ionic_1d import IonicSteadyResult

__all__ = ["ionic_initial_document", "ionic_transient_payload", "validate_ionic_history"]
SCHEMA = "solarlab.ionic-transient-result.v1"
INITIAL_ROLE = "worker-produced:ionic_initial_steady"
CURRENT_FIELDS = tuple(prefix + "current_density_A_m2" for prefix in
    ("carrier_conduction_", "ion_conduction_", "conduction_", "displacement_", "total_"))
TERMINAL_FIELDS = tuple("terminal_" + prefix + "current_density_A_m2" for prefix in
                        ("conduction_", "displacement_", "total_"))


def units(species: tuple[str, ...]) -> dict[str, str]:
    return {"requested_times_s": "s", "requested_voltage_V": "V", "requested_generation_scale": "1",
        "accepted_times_s": "s", "nodes_m": "m", "ion_nodes_m": "m", "ion_node_indices": "1",
        "ion_cell_measures_m": "m", "state_coordinates": "mixed: first N in V; remaining coordinates dimensionless", "potential_V": "V",
        "electron_density_m3": "m^-3", "hole_density_m3": "m^-3",
        **{name: "A/m^2" for name in (*CURRENT_FIELDS, *TERMINAL_FIELDS)},
        **{name + "_ion_density_m3": "m^-3" for name in species},
        **{name + "_ion_cell_inventory_m2": "m^-2" for name in species},
        **{name + "_ion_inventory_m2": "m^-2" for name in species}}


def storage_components(species: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    return {"carrier_density_m3": ("electron_density_m3", "hole_density_m3"),
        "ion_density_m3": tuple(name + "_ion_density_m3" for name in species),
        "ion_inventory_m2": tuple(name + "_ion_inventory_m2" for name in species),
        "all_face_current_density_A_m2": CURRENT_FIELDS,
        "terminal_current_density_A_m2": TERMINAL_FIELDS}


def stored_units() -> dict[str, str]:
    """Thirteen fixed NPY roles keep the existing worker's artifact bound."""
    return {"requested_times_s": "s", "requested_voltage_V": "V", "requested_generation_scale": "1",
        "accepted_times_s": "s", "nodes_m": "m", "ion_node_indices": "1", "ion_cell_measures_m": "m",
        "potential_V": "V", "carrier_density_m3": "m^-3", "ion_density_m3": "m^-3",
        "ion_inventory_m2": "m^-2", "all_face_current_density_A_m2": "A/m^2", "terminal_current_density_A_m2": "A/m^2"}


def _pack(arrays: dict[str, np.ndarray], species: tuple[str, ...]) -> dict[str, np.ndarray]:
    components = storage_components(species)
    return {name: np.stack([arrays[key] for key in components[name]]) if name in components else arrays[name]
            for name in stored_units()}


def unpack_ionic_arrays(summary: dict[str, Any], stored: dict[str, np.ndarray], raw: dict[str, Any]) -> dict[str, np.ndarray]:
    """Restore logical views from committed raw words and check every NPY group."""
    species = tuple(summary["ion_species"])
    arrays = _arrays(raw, species)
    expected = _pack(arrays, species)
    if set(stored) != set(expected) or any(not _same(stored[name], expected[name]) for name in expected):
        raise ValueError("packed ionic arrays differ from their raw producer words/components")
    return arrays


def _plain(value: Any) -> Any:
    """The existing initial loader's example-compatible JSON, never its raw codec."""
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    return value


def ionic_initial_document(compiled: CompiledIonicDevice1D, initial: IonicSteadyResult,
                           protocol: TransientProtocol) -> bytes:
    """Retain the actual cold result in the accepted initial-admission format."""
    return _json_document({"summary": {"accepted": initial.accepted,
        "compiled_identity": compiled.identity, "voltage_V": float(protocol.voltage_V[0]),
        "generation_scale": float(protocol.generation_scale[0]), "qualification": False},
        "result": _plain(initial)}).encode()


def _arrays(raw: dict[str, Any], species: tuple[str, ...]) -> dict[str, np.ndarray]:
    grid, history = raw["grid"], raw["history"]
    states = () if history is None else history["states"]
    accepted = [] if history is None else [a for a in history["attempts"] if a["accepted"]]
    N, M, size = len(grid["nodes_m"]), len(grid["ion_nodes_m"]), raw["state_size"]
    arrays = {"requested_" + name: np.asarray(raw["protocol"][name], dtype=np.float64)
              for name in ("times_s", "voltage_V", "generation_scale")}
    arrays.update({name: np.asarray(value, dtype=np.int64 if name == "ion_node_indices" else np.float64)
                   for name, value in grid.items()})
    arrays["accepted_times_s"] = np.asarray([] if history is None else history["accepted_times_s"], dtype=np.float64)
    arrays["state_coordinates"] = np.stack([s["u"] for s in states]) if states else np.empty((0, size))
    for name in ("potential_V", "electron_density_m3", "hole_density_m3"):
        arrays[name] = np.stack([s["electronic_state"][name] for s in states]) if states else np.empty((0, N))
    for name in species:
        density = np.stack([s[name + "_ion_density_m3"] for s in states]) if states else np.empty((0, M))
        arrays[name + "_ion_density_m3"] = density
        arrays[name + "_ion_cell_inventory_m2"] = density * arrays["ion_cell_measures_m"]
        key = "domain_inventory_m2" if name == "positive" else "negative_domain_inventory_m2"
        arrays[name + "_ion_inventory_m2"] = np.asarray([s[key] for s in states], dtype=np.float64)
    for name in CURRENT_FIELDS:
        arrays[name] = np.stack([a["accounting"][name] for a in accepted]) if accepted else np.empty((0, N + 1))
    for name in TERMINAL_FIELDS:
        arrays[name] = arrays[name.removeprefix("terminal_")][:, [0, -1]]
    return arrays


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite(item) for item in value)
    if isinstance(value, (np.ndarray, np.number, float, int)):
        return bool(np.all(np.isfinite(value)))
    return True


def validate_ionic_history(summary: dict[str, Any], arrays: dict[str, np.ndarray], raw: dict[str, Any],
                           initial_bytes: bytes) -> None:
    """Check exact histories, units/layout relationships and truth, not solve again."""
    species = tuple(summary["ion_species"])
    if (summary["schema"] != SCHEMA or summary["engine"] != "solarlab"
            or summary["physical_certificate_status"] != "not_evaluated"
            or species not in (("positive",), ("positive", "negative")) or set(arrays) != set(units(species))):
        raise ValueError("invalid ionic history scope or physical status")
    binding = raw["compiled_binding"]
    if (hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            != summary["compiled_identity"] or binding["original_prepared_sha256"] != summary["prepared_sha256"]):
        raise ValueError("ionic compiled/prepared identity differs from the retained binding")
    expected = _arrays(raw, species)
    if any(not _same(arrays[name], expected[name]) for name in expected):
        raise ValueError("physical arrays differ from raw ionic history")
    if any(not np.all(np.isfinite(a)) or a.dtype != np.dtype("int64" if name == "ion_node_indices" else "float64")
           for name, a in arrays.items()):
        raise ValueError("accepted physical arrays must retain finite exact words")
    times, accepted, nodes, ion_nodes = (arrays[name] for name in
        ("requested_times_s", "accepted_times_s", "nodes_m", "ion_nodes_m"))
    N, M, count = len(nodes), len(ion_nodes), len(accepted)
    indices, widths = arrays["ion_node_indices"], arrays["ion_cell_measures_m"]
    if (times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0) or not np.all(np.isfinite(np.diff(times)))
            or nodes.ndim != 1 or N < 2 or np.any(np.diff(nodes) <= 0)
            or ion_nodes.ndim != 1 or M < 2 or indices.shape != (M,) or widths.shape != (M,)
            or np.any(widths <= 0) or np.any(indices < 0) or np.any(indices >= N)
            or not _same(nodes[indices], ion_nodes) or np.any(np.diff(indices) != 1)
            or count > len(times) or not _same(accepted, times[:count])):
        raise ValueError("ionic history grids or accepted-prefix alignment differ")
    if (not np.array_equal(binding["ion_domain"]["electrical_node_indices"], indices)
            or not np.array_equal(binding["ion_domain"]["cell_measures_m"], widths)
            or binding["ion_domain"]["contacts"] != ["blocking", "blocking"]
            or raw["state_size"] != 3 * N + len(species) * M
            or raw["state_size"] != binding["state_layout"]["size"]
            or N != binding["state_layout"]["electrical_node_count"] or M != binding["state_layout"]["ion_node_count"]
            or list(species) != binding["state_layout"].get("ion_species", ["positive"])
            or summary["ion_site_contract"] != binding["ion_model"].get("site_contract", "independent_finite_sites")):
        raise ValueError("ionic domain/coordinate layout differs from compiled binding")
    if any(arrays['requested_' + name].shape != times.shape for name in ("voltage_V", "generation_scale")):
        raise ValueError("incomplete requested ionic protocol")
    if np.any(arrays["requested_generation_scale"] < 0):
        raise ValueError("negative requested generation")
    if summary["requested_count"] != len(times) or summary["accepted_count"] != count:
        raise ValueError("ionic result counts differ from actual arrays")
    initial = raw["initial"]
    initial_document = json.loads(initial_bytes)
    if (initial_document["result"] != _plain(initial)
            or initial_document["summary"] != {"accepted": initial["accepted"],
                "compiled_identity": summary["compiled_identity"], "voltage_V": float(arrays["requested_voltage_V"][0]),
                "generation_scale": float(arrays["requested_generation_scale"][0]), "qualification": False}
            or initial["compiled_identity"] != summary["compiled_identity"] or initial["qualification"] is not False):
        raise ValueError("cold initial artifact differs from raw producer result")
    history = raw["history"]
    if history is None:
        if (initial["accepted"] is not False or count != 0 or summary["outcome"] != "initial_failed"
                or summary["reason"] != initial["termination"] or summary["attempt_count"] != 0):
            raise ValueError("failed cold start cannot fabricate an accepted history")
        return
    if (initial["accepted"] is not True or initial["newton"]["accepted"] is not True
            or history["physical_certificate_status"] != "not_evaluated"
            or history["compiled_identity"] != summary["compiled_identity"]
            or history["original_prepared_identity"] != summary["prepared_sha256"]
            or history["binding_manifest"] != binding or not count):
        raise ValueError("ionic history lacks its accepted same-device cold start")
    attempts = history["attempts"]
    flags = [a["accepted"] for a in attempts]
    complete = count == len(times) and flags == [True] * (count - 1)
    if (any(type(flag) is not bool for flag in flags)
            or (not complete and flags != [True] * (count - 1) + [False])
            or summary["outcome"] != ("completed" if complete else "partial")
            or summary["reason"] != (None if complete else attempts[-1]["termination"])
            or summary["requested_count"] != len(times) or summary["accepted_count"] != count
            or summary["attempt_count"] != len(attempts)):
        raise ValueError("ionic result must retain exactly its accepted prefix and first failure")
    if not _same(initial["state"]["u"], arrays["state_coordinates"][0]):
        raise ValueError("initial ionic coordinate words differ")
    for name in ("potential_V", "electron_density_m3", "hole_density_m3"):
        if not _same(initial["state"]["electronic_state"][name], arrays[name][0]):
            raise ValueError("initial electronic fields differ from the accepted first state")
    for name in species:
        if not _same(initial["state"][name + "_ion_density_m3"], arrays[name + "_ion_density_m3"][0]):
            raise ValueError("initial ionic fields differ from the accepted first state")
    for name in ("times_s", "voltage_V", "generation_scale"):
        if not _same(history["protocol"][name], arrays["requested_" + name]):
            raise ValueError("raw ionic result protocol differs from requested samples")
    provenance = history["initial_provenance"]
    if (provenance["source_sha256"] != hashlib.sha256(initial_bytes).hexdigest()
            or provenance["source_path"] != INITIAL_ROLE or provenance["steady_controls"] != initial["solver_controls"]
            or provenance["state_sha256"] != _fingerprint(initial["state"]["u"])["sha256"]):
        raise ValueError("initial provenance is not bound to the actual cold artifact")
    for index, state in enumerate(history["states"]):
        if (state["device_sha256"] != summary["compiled_identity"]
                or state["electronic_state"]["voltage_V"] != arrays["requested_voltage_V"][index]
                or state["u"].shape != (raw["state_size"],)
                or state["electronic_state"]["device_sha256"] != binding["electronic_component"]["compiled_sha256"]
                or not _same(state["electronic_state"]["u"], state["u"][:3 * N])):
            raise ValueError("accepted ionic state belongs to another device/protocol")
        for name in ("electron_density_m3", "hole_density_m3"):
            if arrays[name].shape != (count, N) or np.any(arrays[name] <= 0):
                raise ValueError("invalid accepted electronic density")
        for name in species:
            if arrays[name + "_ion_density_m3"].shape != (count, M) or np.any(arrays[name + "_ion_density_m3"] <= 0):
                raise ValueError("invalid accepted ionic density")
            if math.fsum(arrays[name + "_ion_cell_inventory_m2"][index].tolist()) != arrays[name + "_ion_inventory_m2"][index]:
                raise ValueError("ionic inventory differs from its physical cell measures")
    for index, attempt in enumerate(attempts):
        if (attempt["start_s"] != times[index] or attempt["end_s"] != times[index + 1]
                or attempt["voltage_V"] != arrays["requested_voltage_V"][index + 1]
                or attempt["generation_scale"] != arrays["requested_generation_scale"][index + 1]
                or attempt["initial_coordinates"].shape != (raw["state_size"],)
                or attempt["candidate_coordinates"].shape != (raw["state_size"],)):
            raise ValueError("ionic attempt differs from requested protocol/layout")
        if attempt["accepted"]:
            accounting = attempt["accounting"]
            if (attempt["newton"] is None or attempt["newton"]["accepted"] is not True or accounting is None
                    or not _finite(accounting)
                    or accounting["physical_certificate_status"] != "not_evaluated"
                    or accounting["compiled_identity"] != summary["compiled_identity"]
                    or not _same(attempt["candidate_coordinates"], arrays["state_coordinates"][index + 1])
                    or not _same(attempt["newton"]["state"], attempt["candidate_coordinates"])
                    or not _same(accounting["terminal_total_current_density_A_m2"], arrays["terminal_total_current_density_A_m2"][index])):
                raise ValueError("accepted ionic step lacks consistent raw solver/current evidence")
    if any(arrays[name].shape != (count - 1, N + 1) for name in CURRENT_FIELDS):
        raise ValueError("all-face ionic current layout must include both ports")
    if (not _same(arrays["conduction_current_density_A_m2"], arrays["carrier_conduction_current_density_A_m2"] + arrays["ion_conduction_current_density_A_m2"])
            or not _same(arrays["total_current_density_A_m2"], arrays["conduction_current_density_A_m2"] + arrays["displacement_current_density_A_m2"])):
        raise ValueError("ionic current components do not reproduce their recorded total")


def ionic_transient_payload(compiled: CompiledIonicDevice1D, initial: IonicSteadyResult,
                            protocol: TransientProtocol, history: IonicTransientResult | None,
                            initial_bytes: bytes) -> tuple[dict[str, Any], dict[str, ArtifactPayload]]:
    """Encode actual producer objects; ProcessSupervisor owns durable publication."""
    species = compiled.ion_species_names
    raw = _decode(_encode({"initial": initial, "history": history,
        "protocol": {name: getattr(protocol, name) for name in ("times_s", "voltage_V", "generation_scale")},
        "compiled_binding": compiled.binding_manifest, "state_size": compiled.state_size,
        "grid": {"nodes_m": compiled.electronic_component.nodes_m, "ion_nodes_m": compiled.ion_geometry.nodes_m,
                 "ion_node_indices": np.asarray(compiled.ion_node_indices, dtype=np.int64),
                 "ion_cell_measures_m": compiled.ion_geometry.cell_measures_m}}))
    arrays = _arrays(raw, species)
    summary = {"schema": SCHEMA, "engine": "solarlab", "compiled_identity": compiled.identity,
        "prepared_sha256": compiled.original_prepared_sha256, "ion_species": list(species),
        "ion_site_contract": compiled.ion_site_contract, "requested_count": protocol.times_s.size,
        "accepted_count": len(arrays["accepted_times_s"]), "attempt_count": 0 if history is None else len(history.attempts),
        "outcome": "initial_failed" if history is None else "completed" if history.completed else "partial",
        "reason": initial.termination if history is None else history.failed_attempts[0].termination if history.failed_attempts else None,
        "physical_certificate_status": "not_evaluated",
        "current_alignment": "accepted_times_s[1:]; full faces include both ports; no initial displacement current"}
    validate_ionic_history(summary, arrays, raw, initial_bytes)
    metadata = {"schema": SCHEMA, "compiled_identity": compiled.identity}
    payloads = {name: array_artifact(value, unit=stored_units()[name],
        metadata={**metadata, "components": list(storage_components(species).get(name, ()))})
        for name, value in _pack(arrays, species).items()}
    payloads["ionic_initial_steady"] = bytes_artifact(initial_bytes, metadata={**metadata, "role": "actual_cold_initial"})
    payloads["ionic_transient_diagnostics"] = bytes_artifact(_json_document(_encode(raw)).encode(),
        metadata={**metadata, "role": "raw_ionic_history"})
    return summary, payloads


def result_digest(summary: dict[str, Any], payloads: dict[str, ArtifactPayload]) -> str:
    return _digest({"summary": summary, "payloads": {name: value.sha256 for name, value in payloads.items()}})
