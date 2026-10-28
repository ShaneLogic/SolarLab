"""Run and retain a small, explicitly steady new-core research sweep."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

import solarlab
from solarlab.experiments.jv.steady import SteadyJVResult, publish_steady_jv
from solarlab.io.run_store import RunStore
from solarlab.io.storage import require_local_storage
from solarlab.solvers.nonlinear.steady_1d import SteadyOptions
from solarlab.sweeps.inputs import SweepInput
from solarlab.sweeps.preparation import PreparedSweep, digest, prepare_sweep
from solarlab.sweeps.steady import SteadySweepPoint, run_prepared_steady_sweep

if __package__:
    from .steady import load_device
    from .read_results import read_study, summarize
else:
    from steady import load_device
    from read_results import read_study, summarize

ROOT = Path(__file__).resolve().parents[2]
SCOPE = "solarlab.research-1d.execution.v1"
THREAD_NAMES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def source_record(config: Path) -> dict[str, Any]:
    """Actual source bytes, including uncommitted code; HEAD is not a proxy."""
    package = Path(solarlab.__file__).resolve().parent
    files = {"solarlab/" + str(path.relative_to(package)): path
             for path in package.rglob("*.py")}
    files.update({"example/" + name: Path(__file__).with_name(name)
                  for name in ("steady.py", "jv_scan.py", "read_results.py")})
    files.update({name: ROOT / name for name in ("pyproject.toml", "uv.lock")})
    files["supplied_configuration"] = config
    records = [{"name": name, "path": str(path), "sha256": file_sha256(path)}
               for name, path in sorted(files.items())]
    return {"scope": "actual_python_sources_and_input; not a release attestation",
            "files": records, "sha256": digest(records)}


def runtime_record() -> dict[str, Any]:
    pools = [{**item, "binary_sha256": file_sha256(Path(item["filepath"]))}
             for item in threadpool_info()]
    value = {
        "python": sys.version, "python_executable": sys.executable,
        "python_executable_sha256": file_sha256(Path(sys.executable)),
        "platform": platform.platform(), "machine": platform.machine(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ("solarlab", "numpy", "scipy", "pydantic", "threadpoolctl")},
        "thread_environment": {name: os.environ.get(name) for name in THREAD_NAMES},
        "threadpools": pools,
    }
    return {**value, "sha256": digest(value)}


def make_sweep(config: Path, *, study_id: str, parameter: str,
               values: list[float | str], layer: str = "absorber") -> PreparedSweep:
    if parameter not in {"tau_n", "tau_p", "mu_n", "mu_p"}:
        raise ValueError("this example exposes only explicit lifetime or mobility fields")
    prepared = load_device(config)
    declaration = SweepInput.model_validate({
        "schema_version": "solarlab.sweep-preparation.v1", "id": study_id,
        "base": prepared.to_input().editing_data(),
        "axes": [{"id": parameter,
                  "target": {"family": "layer_parameter", "owner_id": layer,
                             "parameter": [parameter]},
                  "coordinates": [{"kind": "value", "value": value} for value in values]}],
    })
    return prepare_sweep(declaration, prepared.defaults, prepared.resources, sources=prepared.sources)


def identity_payloads(point: dict[str, Any], sweep: PreparedSweep,
                      result: SteadyJVResult | None, protocol: dict[str, Any],
                      sources: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    """Research-scoped identities; no claim to a frozen registry release."""
    inputs = {"scope": SCOPE + ":input", "supplied_point": point["applied_input"],
              "declared_coordinates": point["coordinates"],
              "prepared_content_sha256": point.get("content_sha256"),
              "resolved_sha256": point.get("resolved_sha256"),
              "defaults_sha256": sweep.defaults.content_sha256,
              "resources_sha256": sweep.resources.content_sha256,
              "configuration_sources": [{"id": s.id, "sha256": s.sha256} for s in sweep.sources]}
    physics = {"scope": SCOPE + ":physics", "H_input": digest(inputs),
               "compiled_identity": None if result is None else result.compiled_identity,
               "binding_status": "no_completed_curve" if result is None else "observed_in_completed_result",
               "generation_scale": protocol["generation_scale"]}
    execution = {"scope": SCOPE, "H_input": digest(inputs), "H_physics": digest(physics),
                 "protocol": protocol, "source_sha256": sources["sha256"],
                 "runtime_sha256": runtime["sha256"], "engine": "solarlab",
                 "cross_parameter_state_reuse": False,
                 "initialization": "new-core cold point; continuation only within voltage curve"}
    return {"H_input": inputs, "H_physics": physics, "H_execution": execution}


def array_fingerprints(result: SteadyJVResult) -> dict[str, Any]:
    arrays = {"nodes_m": result.nodes_m, "voltage_V": result.voltage_V,
              "current_density_A_m2": result.current_density_A_m2,
              "contact_current_density_A_m2": np.asarray([point.contact_current_density_A_m2 for point in result.solutions]),
              "output_current_A_m2": result.output_current_A_m2}
    for name in ("potential_V", "electron_density_m3", "hole_density_m3"):
        arrays[name] = np.asarray([getattr(point.state, name) for point in result.solutions])
    return {name: {"dtype": value.dtype.str, "shape": list(value.shape),
                   "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}
            for name, value in arrays.items()}


def run_study(store_path: Path, *, config: Path, study_id: str = "lifetime",
              parameter: str = "tau_n", values: list[float | str] | None = None,
              layer: str = "absorber", intervals_per_layer: int = 12,
              voltage_points: int = 51, voltage_max_V: float = 1.25,
              generation_scale: float = 1.0,
              incident_power_W_m2: float | None = None) -> dict[str, Any]:
    """Compute once, then publish the returned outcomes under actual identities.

    RunStore attempts describe result publication, not a fictional pre-solve
    claim. The measured calculation window is saved separately. This small
    example does not implement supervisor crash recovery or automatic retries.
    """
    store_path, config = store_path.expanduser().resolve(), config.expanduser().resolve()
    if store_path.exists():
        raise FileExistsError("choose a fresh store; existing attempts are never overwritten")
    require_local_storage(store_path)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    if not 2 <= voltage_points <= 101 or not np.isfinite(voltage_max_V) or voltage_max_V <= 0:
        raise ValueError("request 2..101 voltage samples and a positive finite maximum")
    if not np.isfinite(generation_scale) or generation_scale < 0:
        raise ValueError("generation_scale must be finite and nonnegative")
    if incident_power_W_m2 is not None and (not np.isfinite(incident_power_W_m2) or incident_power_W_m2 <= 0):
        raise ValueError("incident power must be finite and positive when supplied")
    sweep = make_sweep(config, study_id=study_id, parameter=parameter,
                       values=["1 us", "2 us"] if values is None else values, layer=layer)
    if len(sweep.to_mapping()["points"]) > 8:
        raise ValueError("this small serial example is bounded to eight parameter points")
    voltages = np.linspace(0.0, voltage_max_V, voltage_points)
    options = SteadyOptions()  # Preserve the production physical/numerical limits.
    protocol = {"kind": "steady_voltage_continuation", "voltage_V": voltages.tolist(),
                "intervals_per_layer": intervals_per_layer, "generation_scale": generation_scale,
                "incident_power_W_m2": incident_power_W_m2, "solver_controls": asdict(options)}
    with threadpool_limits(limits=1), RunStore(store_path) as store:
        sources, runtime = source_record(config), runtime_record()
        started = datetime.now(timezone.utc).isoformat()
        start = time.monotonic()
        fatal = None
        results: tuple[SteadySweepPoint, ...] = ()
        try:
            results = run_prepared_steady_sweep(
                sweep, voltages, intervals_per_layer=intervals_per_layer,
                generation_scale=generation_scale, incident_power_W_m2=incident_power_W_m2,
                options=options)
            if source_record(config)["sha256"] != sources["sha256"]:
                raise RuntimeError("source bytes changed during the calculation; publication refused")
        except Exception as failure:
            fatal = f"{type(failure).__name__}: {failure}"
        window = {"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
                  "elapsed_seconds": time.monotonic() - start, "scope": "whole serial calculation"}
        declarations = sweep.to_mapping()["points"]
        if fatal is None and [p.point_id for p in results] != [p["id"] for p in declarations]:
            fatal = "runner returned a different point sequence"
        for index, point in enumerate(declarations):
            outcome = results[index] if fatal is None else None
            result = None if outcome is None else outcome.result
            error = fatal if outcome is None else outcome.error
            if outcome is not None and (outcome.status == "completed") != (result is not None):
                result, error = None, "runner status and result presence disagree"
            if result is not None and result.solver_controls != protocol["solver_controls"]:
                result, error = None, "result solver controls differ from the requested controls"
            payloads = identity_payloads(point, sweep, result, protocol, sources, runtime)
            identity = {key: digest(value) for key, value in payloads.items()}
            identity["scope"] = SCOPE
            metadata = {"schema": "solarlab.research-1d.point.v1", "study_id": study_id,
                        "point_id": point["id"], "point_index": index, "identity_payloads": payloads,
                        "sources": sources, "runtime": runtime, "calculation_window": window,
                        "attempt_phase": "publication_of_returned_calculation_outcome",
                        "automatic_retry": False}
            run_id = "research:" + digest([study_id, point["id"]])
            row = store.enqueue(run_id, opaque_identity=identity, input_metadata=metadata)
            store.bind_sweep_point(study_id, point["id"], run_id=run_id,
                                   coordinates=point["normalized_coordinates"])
            token = store.claim(run_id, expected_attempt_id=row["current_attempt_id"],
                                expected_generation=row["generation"], owner_id="research_1d_example")
            if result is None:
                store.complete(token, state="failed", result={
                    "schema": "solarlab.research-1d.failure.v1", "point_status": "failed",
                    "reason": error or "runner returned no complete result", "compiled_identity": None,
                    "qualification": "no_completed_curve"})
                continue
            try:
                document = publish_steady_jv(store, token, result)
                document["research_array_fingerprints"] = array_fingerprints(result)
                document["metrics_status"] = ("estimated" if result.metrics.open_circuit_voltage_V is not None
                                               else "unavailable")
                store.complete(token, state="succeeded", result=document,
                               artifacts=document["artifact_ids"])
            except Exception as failure:
                if store.get_run(run_id)["state"] == "running":
                    store.complete(token, state="failed", result={
                        "schema": "solarlab.research-1d.failure.v1", "point_status": "failed",
                        "reason": f"publication: {type(failure).__name__}: {failure}",
                        "compiled_identity": result.compiled_identity})
                raise
    return summarize(read_study(store_path, study_id))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True, help="fresh absolute local RunStore directory")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("device.json"))
    parser.add_argument("--study-id", default="lifetime")
    parser.add_argument("--layer", default="absorber")
    parser.add_argument("--parameter", choices=("tau_n", "tau_p", "mu_n", "mu_p"), default="tau_n")
    parser.add_argument("--values", nargs="+", default=None, help="SI numbers or quoted unit strings")
    parser.add_argument("--intervals-per-layer", type=int, default=12)
    parser.add_argument("--voltage-points", type=int, default=51)
    parser.add_argument("--voltage-max", type=float, default=1.25)
    parser.add_argument("--generation-scale", type=float, default=1.0)
    parser.add_argument("--incident-power", type=float, default=None, help="explicit W/m^2; omit to leave efficiency null")
    args = parser.parse_args(argv)
    if args.values is None and args.parameter not in {"tau_n", "tau_p"}:
        parser.error("mobility sweeps require explicit --values")
    values: list[float | str] | None = None
    if args.values is not None:
        values = []
        for value in args.values:
            try:
                values.append(float(value))
            except ValueError:
                values.append(value)
    result = run_study(args.store, config=args.config, study_id=args.study_id,
        parameter=args.parameter, values=values, layer=args.layer,
        intervals_per_layer=args.intervals_per_layer, voltage_points=args.voltage_points,
        voltage_max_V=args.voltage_max, generation_scale=args.generation_scale,
        incident_power_W_m2=args.incident_power)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["outcome"] == "completed_with_estimated_metrics" else 2


if __name__ == "__main__":
    raise SystemExit(main())
