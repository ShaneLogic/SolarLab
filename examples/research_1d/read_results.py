"""Read a saved research sweep through RunStore; never import or call a solver."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from solarlab.experiments.jv.steady import read_steady_jv
from solarlab.io.run_store import RunStore

SCOPE = "solarlab.research-1d.execution.v1"


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def verify_identity(request: dict[str, Any]) -> dict[str, Any]:
    identity, metadata = request["identity"], request["input_metadata"]
    if identity.get("scope") != SCOPE or metadata.get("schema") != "solarlab.research-1d.point.v1":
        raise ValueError("not a research-1d result binding")
    payloads = metadata["identity_payloads"]
    if set(payloads) != {"H_input", "H_physics", "H_execution"}:
        raise ValueError("incomplete research identity payloads")
    if any(digest(payload) != identity[name] for name, payload in payloads.items()):
        raise ValueError("research identity digest mismatch")
    execution, physics = payloads["H_execution"], payloads["H_physics"]
    if (physics["H_input"] != identity["H_input"] or execution["H_input"] != identity["H_input"]
            or execution["H_physics"] != identity["H_physics"]):
        raise ValueError("research identity chain mismatch")
    sources, runtime = metadata["sources"], metadata["runtime"]
    if (sources["sha256"] != digest(sources["files"])
            or runtime["sha256"] != digest({key: value for key, value in runtime.items() if key != "sha256"})
            or execution["source_sha256"] != sources["sha256"]
            or execution["runtime_sha256"] != runtime["sha256"]):
        raise ValueError("saved source/runtime identity mismatch")
    return metadata


def read_study(store_path: Path, study_id: str) -> dict[str, Any]:
    store_path = store_path.expanduser().resolve()
    points = []
    with RunStore(store_path, read_only=True) as store:
        rows = store.sweep_points(study_id)
        if not rows:
            raise ValueError("unknown or empty saved study")
        for association in rows:
            run = store.get_run(association["run_id"])
            metadata = verify_identity(run["request"])
            identity = run["request"]["identity"]
            if (metadata["study_id"] != study_id or metadata["point_id"] != association["point_id"]
                    or association["opaque_execution_identity"] != identity["H_execution"]):
                raise ValueError("study association differs from the saved identity")
            attempts = []
            for attempt in store.attempts(association["run_id"]):
                document, arrays = attempt["result"], {}
                if attempt["state"] == "succeeded":
                    physics = metadata["identity_payloads"]["H_physics"]
                    protocol = metadata["identity_payloads"]["H_execution"]["protocol"]
                    if (document["compiled_identity"] != physics["compiled_identity"]
                            or document["solver_controls"] != protocol["solver_controls"]
                            or document["generation_scale"] != protocol["generation_scale"]):
                        raise ValueError("result differs from its actual execution binding")
                    expected = {item["artifact_id"] for item in attempt["terminal"]["artifacts"]}
                    if expected != set(document["arrays"].values()) or expected != set(document["artifact_ids"]):
                        raise ValueError("result references differ from the committed terminal")
                    for identifier in expected:
                        record = store.artifact(identifier)
                        if (record["phase"] != "committed" or record["run_id"] != association["run_id"]
                                or record["attempt_id"] != attempt["attempt_id"]
                                or record["generation"] != attempt["generation"]):
                            raise ValueError("artifact belongs to a different attempt")
                    arrays = read_steady_jv(store, document)
                    if arrays["voltage_V"].tolist() != protocol["voltage_V"]:
                        raise ValueError("saved voltages differ from the requested protocol")
                    for name, array in arrays.items():
                        expected_array = document["research_array_fingerprints"][name]
                        if (array.dtype.str != expected_array["dtype"] or list(array.shape) != expected_array["shape"]
                                or hashlib.sha256(array.tobytes(order="C")).hexdigest() != expected_array["sha256"]):
                            raise ValueError("roundtrip differs from the original numerical array")
                attempts.append({"attempt_id": attempt["attempt_id"], "generation": attempt["generation"],
                                 "state": attempt["state"], "result": document, "arrays": arrays})
            current = next(item for item in attempts if item["attempt_id"] == association["current_attempt_id"])
            points.append({"point_id": association["point_id"], "point_index": metadata["point_index"],
                           "run_id": association["run_id"], "coordinates": association["coordinates"],
                           "identity": identity, "metadata": metadata, "attempts": attempts, "current": current})
        return {"schema": "solarlab.research-1d.readback.v1", "study_id": study_id,
                "store": str(store_path), "store_id": store.store_id,
                "points": sorted(points, key=lambda point: point["point_index"])}


def summarize(study: dict[str, Any], *, include_arrays: bool = False) -> dict[str, Any]:
    points = []
    for point in study["points"]:
        current = point["current"]
        document = current["result"] or {}
        metrics = document.get("metrics")
        metrics_status = ("estimated" if metrics is not None and metrics["open_circuit_voltage_V"] is not None
                          else "unavailable")
        value = {"point_id": point["point_id"], "run_id": point["run_id"], "coordinates": point["coordinates"],
                 "state": current["state"], "metrics_status": metrics_status, "metrics": metrics,
                 "reason": document.get("reason"), "identity": point["identity"],
                 "compiled_identity": document.get("compiled_identity"),
                 "attempts": [{k: a[k] for k in ("attempt_id", "generation", "state")} for a in point["attempts"]],
                 "qualification": document.get("qualification"),
                 "array_shapes": {name: list(array.shape) for name, array in current["arrays"].items()}}
        if include_arrays:
            value["arrays"] = {name: array.tolist() for name, array in current["arrays"].items()}
        points.append(value)
    states = {p["state"] for p in points}
    outcome = ("incomplete" if states & {"queued", "running"}
               else "completed_with_failed_points" if states & {"failed", "cancelled"}
               else "completed_with_unavailable_metrics" if any(p["metrics_status"] != "estimated" for p in points)
               else "completed_with_estimated_metrics")
    first = study["points"][0]["metadata"]
    return {"schema": "solarlab.research-1d.summary.v1", "study_id": study["study_id"],
            "store": study["store"], "store_id": study["store_id"], "outcome": outcome,
            "identity_scope": SCOPE, "source_sha256": first["sources"]["sha256"],
            "runtime": first["runtime"], "calculation_window": first["calculation_window"],
            "attempt_phase": first["attempt_phase"], "scientific_qualification": False, "points": points}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--study-id", default="lifetime")
    parser.add_argument("--include-arrays", action="store_true")
    args = parser.parse_args(argv)
    result = summarize(read_study(args.store, args.study_id), include_arrays=args.include_arrays)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["outcome"] == "completed_with_estimated_metrics" else 2


if __name__ == "__main__":
    raise SystemExit(main())
