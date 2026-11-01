"""Time the identical ionic J-V child outside the supervisor's whole-job wall.

Each sample uses a fresh interpreter and the real spawn entry, provenance and
artifact writer. By default the installed request and every limit are unchanged.
The explicit curve measurement mode changes only the curve ceiling to 64 seconds
in a separately source-bound diagnostic worker; it never recommends a limit.
This is diagnostic execution, not a store submission or a scientific qualification.
The workflow timeout bounds the diagnostic; there is no per-sample wall censor.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import runpy
import statistics
import subprocess
import sys
import time
from types import FunctionType
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from solarlab.config.identity import BuildSource
    from solarlab.device.resolved import PreparedDevice
    from solarlab.experiments.jv.ionic_steady import IonicSteadyJVOptions
    from solarlab.experiments.jv.ionic_worker import IonicJVInitial, IonicJVJob, IonicJVProtocol
    from solarlab.experiments.worker import ManifestBinding, WorkerBinding, WorkerContext, WorkerRequest, WorkerResult
    from solarlab.materials.source import SourceDocument


ROOT = Path(__file__).resolve().parents[2]
CURVE_MEASUREMENT = "diagnostic_ionic_jv_curve_64s"
CURVE_MEASUREMENT_SECONDS = 64.


def _diagnostic_source() -> SourceDocument:
    from solarlab.materials.source import SourceDocument

    return SourceDocument("scripts/ci/ionic_jv_timing.py", Path(__file__).read_bytes())


def _diagnostic_documents() -> tuple[SourceDocument, SourceDocument]:
    from solarlab.experiments.jv import ionic_worker as core

    schema = core._document("solarlab.diagnostic-ionic-jv.request.v1",
        {**json.loads(core.REQUEST_SCHEMA.content), "diagnostic_measurement": CURVE_MEASUREMENT})
    registry = core._document("solarlab.diagnostic-ionic-jv.registry.v1",
        {"engine": "solarlab", "worker": CURVE_MEASUREMENT,
         "curve_wall_seconds": CURVE_MEASUREMENT_SECONDS, "source_sha256": _diagnostic_source().sha256})
    return schema, registry


def _diagnostic_validate(prepared: PreparedDevice, protocol: IonicJVProtocol, intervals: tuple[int, ...],
    domain: str, endpoints: tuple[str, str], options: IonicSteadyJVOptions, initial: IonicJVInitial,
) -> None:
    """Admit only the installed example with the single approved measurement delta."""
    from solarlab.experiments.jv import ionic_worker as core
    from solarlab.experiments.jv.ionic_steady import IonicSteadyJVOptions
    from solarlab_server.ionic_jv_preparation import (
        IonicJVJobInput, _REGISTERED_WALL_PROFILE_SHA256, _initial, _wall_profile_sha256,
    )

    if (type(options) is not IonicSteadyJVOptions or type(options.wall_s) not in {int, float}
            or options.wall_s != CURVE_MEASUREMENT_SECONDS or initial != _initial()):
        raise ValueError("diagnostic measurement requires exactly the 64s curve and installed initial recipe")
    shipping_options = replace(options, wall_s=25.)
    IonicJVJobInput.model_validate({"device_id": "research_1d_shared", **asdict(protocol),
        "voltage_V": list(protocol.voltage_V), "intervals_per_layer": list(intervals),
        "controls": asdict(shipping_options),
        "initial": {key: asdict(initial)[key] for key in ("policy", "voltage_V", "wall_s")}})
    if _wall_profile_sha256(prepared, domain, endpoints) != _REGISTERED_WALL_PROFILE_SHA256:
        raise ValueError("diagnostic measurement requires the approved installed device and domain")
    core._validate(prepared, protocol, intervals, domain, endpoints, shipping_options, initial)


def _diagnostic_roles(prepared: PreparedDevice, body: dict[str, Any],
                      implementation: BuildSource) -> dict[str, SourceDocument]:
    from solarlab.experiments.jv import ionic_worker as core

    roles = core._roles(prepared, body, implementation)
    source = _diagnostic_source()
    roles["preparation"] = core._document("solarlab.diagnostic-ionic-jv.preparation.v1",
        {**json.loads(roles["preparation"].content), "source_scope": "committed_core_plus_explicit_diagnostic_adapter",
         "diagnostic_measurement": {
            "kind": CURVE_MEASUREMENT, "shipping_curve_wall_seconds": 25.,
            "measurement_curve_wall_seconds": CURVE_MEASUREMENT_SECONDS,
            "source": {"id": source.id, "sha256": source.sha256, "content_hex": source.content.hex()},
            "recommendation": False}})
    return roles


def _diagnostic_callable(function: Callable[..., Any]) -> FunctionType:
    """Reuse worker bytecode with private globals; never patch an installed module.

    The core validator rejects curves above 25s. Its diagnostic replacement
    checks the exact installed recipe with only that one scalar projected back
    to 25s for admission. Execution still receives the source-bound 64s request.
    Separate schema/registry and preparation identities prevent production use.
    """
    if not isinstance(function, FunctionType):
        raise TypeError("diagnostic rebinding requires a Python function")
    schema, registry = _diagnostic_documents()
    namespace = {**function.__globals__, "_validate": _diagnostic_validate, "_roles": _diagnostic_roles,
                 "REQUEST_SCHEMA": schema, "REGISTRY": registry}
    result = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def prepare_curve_measurement(request: WorkerRequest, execution: ManifestBinding) -> IonicJVJob:
    from solarlab.experiments.jv import ionic_worker as core
    from solarlab_server.ionic_jv_preparation import _initial, _options

    execution.validate_request(request)
    body = json.loads(request.request.content)
    grid = body["grid"]
    job: IonicJVJob = _diagnostic_callable(core.prepare_ionic_jv_job)(
        execution.inputs.prepared, core.IonicJVProtocol(**body["protocol"]),
        intervals_per_layer=tuple(grid["intervals_per_layer"]), ion_domain_layer=grid["ion_domain_layer"],
        ion_domain_endpoints=tuple(grid["ion_domain_endpoints"]),
        options=replace(_options(), wall_s=CURVE_MEASUREMENT_SECONDS), initial=_initial(),
        implementation=execution.execution.implementation,
        current_repository_head=execution.execution.current_repository_head, engine=execution.engine)
    body["schema"] = _diagnostic_documents()[0].id
    body["controls"]["wall_s"] = CURVE_MEASUREMENT_SECONDS
    if json.loads(job.request.request.content) != body:
        raise ValueError("diagnostic preparation changed more than the curve ceiling and diagnostic schema")
    return job


def curve_measurement_binding() -> WorkerBinding:
    from solarlab.experiments.worker import WorkerBinding
    # The canonical import remains transportable when this script is __main__.
    from scripts.ci.ionic_jv_timing import run_curve_measurement

    schema, registry = _diagnostic_documents()
    return WorkerBinding.from_function(CURVE_MEASUREMENT, run_curve_measurement,
        registry_sha256=registry.sha256, schema_sha256=schema.sha256)


def run_curve_measurement(request: WorkerRequest, worker: WorkerContext) -> WorkerResult:
    from examples.research_1d.ionic_jv_service import resource_limits
    from solarlab.experiments.jv import ionic_worker as core
    from solarlab.experiments.worker import WorkerResult

    curve_measurement_binding().validate_request(request)
    if worker._limits != resource_limits():
        raise ValueError("diagnostic measurement must keep every installed resource limit")
    result = _diagnostic_callable(core.run_ionic_jv_worker)(
        request, worker, wall_ceiling_seconds=resource_limits().wall_seconds)
    return WorkerResult.from_dict({**json.loads(result.result_json), "diagnostic_measurement": {
        "kind": CURVE_MEASUREMENT, "source_sha256": _diagnostic_source().sha256, "recommendation": False}},
        artifacts=result.artifacts, state=result.state)


def numerical_measurements(history: dict[str, Any]) -> dict[str, Any]:
    """Keep failed phases and every attempted point, including clipped budgets."""
    seconds: dict[str, float | None] = {}
    stages: dict[str, Any] = {}
    events: list[dict[str, Any]] = []
    points = []
    for name in ("dark_steady", "illumination", "curve"):
        phase = history.get(name)
        seconds[name] = None if phase is None else phase["wall_s"]
        if phase is not None:
            stages[name] = {key: phase.get(key) for key in ("accepted", "termination", "error")}
            if phase.get("termination") == "wall_budget_exhausted":
                events.append({"phase": name, "limit": "initialization_wall_s" if name == "illumination" else "point_wall_s",
                               "reason": phase["termination"]})
    curve = history.get("curve")
    if curve is not None:
        for attempt in curve["attempts"]["$tuple"]:
            solution = attempt["solution"]
            point = {key: attempt[key] for key in
                     ("index", "voltage_V", "accepted", "termination", "solver_called", "wall_s", "error")}
            point["solver_wall_seconds"] = None if solution is None else solution["wall_s"]
            point["effective_point_wall_seconds"] = None if solution is None else solution["solver_controls"]["wall_s"]
            point["solver_termination"] = None if solution is None else solution["termination"]
            point["solver_error"] = None if solution is None else solution["error"]
            point["wall_censored"] = (point["termination"] == "wall_budget_exhausted"
                                      or point["solver_termination"] == "wall_budget_exhausted")
            if point["wall_censored"]:
                clipped = (solution is not None and solution["solver_controls"]["wall_s"]
                           < history["controls"]["steady"]["wall_s"])
                events.append({"phase": "curve", "index": point["index"],
                    "limit": "curve_wall_s" if clipped or solution is None else "point_wall_s",
                    "reason": "wall_budget_exhausted"})
            points.append(point)
    return {"numerical_seconds": seconds, "numerical_stages": stages, "point_timings": points,
        "censoring": {"whole_job_wall": False,
                     "numerical_wall": bool(events) if any(value is not None for value in seconds.values()) else None,
                     "events": events, "unobserved_phases": [name for name, value in seconds.items() if value is None]}}


def distribution(values: list[float]) -> dict[str, float]:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("positive finite uncensored measurements are required")
    ordered = sorted(values)
    return {"p50": statistics.median(ordered), "p95": ordered[math.ceil(.95 * len(ordered)) - 1],
            "max": ordered[-1]}


def summarize(rows: list[dict[str, Any]], *, curve_measurement: bool = False) -> dict[str, Any]:
    complete = bool(rows) and all(row.get("status") == "passed" for row in rows)
    result: dict[str, Any] = {"status": "passed" if complete else "failed",
        "requested": len(rows), "completed": sum(row.get("status") == "passed" for row in rows),
        "percentiles": "p50: median; p95: nearest rank; seconds",
        "recommended_wall_seconds": None}
    measurement_only = curve_measurement or any(
        row.get("measurement", {}).get("measurement_kind") == CURVE_MEASUREMENT for row in rows)
    if measurement_only:
        result.update(measurement_kind=CURVE_MEASUREMENT, recommendation_status="measurement_only")
    observed = {}
    for phase in ("dark_steady", "illumination", "curve"):
        values = [row["measurement"].get("numerical_seconds", {}).get(phase) for row in rows]
        available = [value for value in values if value is not None]
        if available:
            observed[phase] = {**distribution(available), "observed": len(available),
                              "unobserved": len(values) - len(available), "includes_failed_samples": True}
    result["observed_numerical_seconds"] = observed
    if complete and not measurement_only:
        timings = distribution([row["measurement"]["child_wall_seconds"] for row in rows])
        result["child_wall_seconds"] = timings
        result["recommended_wall_seconds"] = math.ceil(max(128., 4 * timings["p95"], 2 * timings["max"]))
        ceilings = {row["measurement"]["request"]["limits"]["wall_seconds"] for row in rows}
        result["installed_wall_seconds"] = sorted(ceilings)
        result["allowance_sufficient"] = len(ceilings) == 1 and min(ceilings) >= result["recommended_wall_seconds"]
        if not result["allowance_sufficient"]:
            result.update(status="failed", reason="installed_allowance_below_uncensored_derivation")
    # Failed/numerically capped samples stay visible and never silently become
    # an uncensored success-only percentile used to approve the ceiling.
    return result


def measure(output: Path, *, curve_measurement: bool = False) -> dict[str, Any]:
    imports_started = time.perf_counter()
    import multiprocessing

    from examples.research_1d.ionic_jv_service import example_submission, installed_catalog
    from examples.research_1d.process_ionic_transient import source_build
    # Deliberately reuse the production bootstrap and artifact writer, rather
    # than timing a simplified numerical surrogate. No supervisor is created.
    from solarlab.experiments.worker import PROGRESS_BYTES, _child_main
    from solarlab_server.job_service import JobSubmission

    imports_seconds = time.perf_counter() - imports_started
    preparation_started = time.perf_counter()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    implementation = source_build(commit)
    catalog = installed_catalog(implementation, commit)
    submitted = example_submission("uncensored-ionic-jv")
    spec, execution = catalog.prepare(JobSubmission.model_validate(submitted))
    binding = catalog.worker_binding("solarlab", experiment_id=submitted["experiment_id"])
    request = spec.request
    if curve_measurement:
        diagnostic = prepare_curve_measurement(request, execution)
        request, execution = diagnostic.request, diagnostic.execution
        binding = curve_measurement_binding()
    preparation_seconds = time.perf_counter() - preparation_started
    directory = output / "child"
    directory.mkdir()
    context = multiprocessing.get_context("spawn")
    stop, lock = context.Event(), context.Lock()
    header, buffer = context.RawArray("q", 4), context.RawArray("B", PROGRESS_BYTES)
    process = context.Process(target=_child_main, args=(
        request.to_bytes(), binding, spec.limits, str(directory),
        {"diagnostic": CURVE_MEASUREMENT if curve_measurement else "uncensored_ionic_jv",
         "sample": output.name}, stop, lock, header, buffer, execution),
        daemon=True)
    started = time.perf_counter()
    try:
        process.start()
        process.join()
        child_wall_seconds = time.perf_counter() - started
    finally:
        if process.is_alive():
            process.kill()
            process.join()
    terminal_path = directory / "terminal.json"
    terminal = json.loads(terminal_path.read_text())
    result = terminal["result"]
    summary = result.get("ionic_jv_result", {})
    artifacts = terminal["artifacts"]
    measurement: dict[str, Any] = {"status": "failed", "source_commit": commit,
        "measurement_kind": CURVE_MEASUREMENT if curve_measurement else "installed_recipe_wall_timing",
        "python_version": sys.version, "parent_imports_seconds": imports_seconds,
        "parent_preparation_seconds": preparation_seconds, "child_wall_seconds": child_wall_seconds,
        "scope": "spawn, engine imports, actual worker, provenance and child artifact fsync; no supervisor/store commit",
        "whole_job_wall_enforced": False, "numerical_stage_limits_unchanged": not curve_measurement,
        "installed_request": submitted, "core_request": json.loads(request.request.content),
        "numerical_limits_seconds": {"curve": CURVE_MEASUREMENT_SECONDS if curve_measurement else 25.,
                                    "initialization": 25., "point": 30.},
        "request": {**submitted, "input": {**submitted["input"],
            "controls": json.loads(request.request.content)["controls"]}},
        "request_sha256": request.request.sha256,
        "diagnostic_source_sha256": _diagnostic_source().sha256 if curve_measurement else None,
        "manifest_sha256": execution.expected_sha256, "exitcode": process.exitcode,
        "process_reaped": not process.is_alive(), "terminal_state": terminal["state"],
        "reason": result.get("reason"), "outcome": summary.get("outcome"),
        "accepted_count": summary.get("accepted_count"), "artifact_count": len(artifacts),
        "terminal_sha256": hashlib.sha256(terminal_path.read_bytes()).hexdigest(),
        "child_peak_rss_bytes": terminal["child_peak_rss_bytes"],
        **numerical_measurements({})}
    for artifact in artifacts:
        raw = (directory / artifact["file"]).read_bytes()
        if len(raw) != artifact["bytes"] or hashlib.sha256(raw).hexdigest() != artifact["sha256"]:
            raise ValueError("direct child artifact integrity failed")
        if artifact["name"] == "ionic_jv_diagnostics":
            history = json.loads(raw)
            measurement.update(numerical_measurements(history))
    measurement["artifact_integrity_verified"] = True
    if capture := result.get("capture"):
        measurement.update(runtime=capture["runtime"], computation_counts=summary["computation_counts"],
            worker_body_seconds=(datetime.fromisoformat(capture["finished_utc"])
                                 - datetime.fromisoformat(capture["started_utc"])).total_seconds())
    if (process.exitcode == 0 and terminal["state"] == "succeeded" and summary.get("outcome") == "completed"
            and summary.get("accepted_count") == summary.get("requested_count") == 51
            and len(artifacts) == 15 and len(measurement["point_timings"]) == 51
            and not measurement["censoring"]["unobserved_phases"]):
        measurement["status"] = "passed"
    return measurement


def diagnose(repetitions: int, output: Path, *, curve_measurement: bool = False) -> dict[str, Any]:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    if output.exists() and any(output.iterdir()):
        raise ValueError("diagnostic output must be empty; previous evidence is preserved")
    output.mkdir(parents=True, exist_ok=True)
    environment = runpy.run_path(str(ROOT / "scripts/ci/ionic_jv_diagnostics.py"))["environment"]
    rows: list[dict[str, Any]] = []
    for index in range(1, repetitions + 1):
        directory = output / f"attempt_{index:02d}"
        directory.mkdir()
        report = directory / "Timing.json"
        print(f"[{index}/{repetitions}] uncensored ionic child: {directory}", flush=True)
        command = [sys.executable, "-m", "scripts.ci.ionic_jv_timing", "--single", str(directory)]
        if curve_measurement:
            command.append("--diagnostic-curve-64")
        with (directory / "Timing.log").open("w") as log:
            finished = subprocess.run(command, cwd=ROOT, env=environment(directory),
                stdout=log, stderr=subprocess.STDOUT, check=False)
        measurement = json.loads(report.read_text()) if report.exists() else {
            "status": "failed", "error": "missing timing receipt", **numerical_measurements({})}
        row = {"attempt": index, "exit_code": finished.returncode, "measurement": measurement,
               "status": "passed" if finished.returncode == 0 and measurement["status"] == "passed" else "failed"}
        rows.append(row)
        summary = {**summarize(rows, curve_measurement=curve_measurement), "attempts": rows}
        (output / "Summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        print(f"[{index}/{repetitions}] {row['status']}: {measurement.get('child_wall_seconds')} s", flush=True)
    print("Ionic diagnostic timing: " + json.dumps(summarize(rows, curve_measurement=curve_measurement), sort_keys=True), flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--single", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--diagnostic-curve-64", action="store_true",
        help="measure the installed curve with a source-bound 64s ceiling; never recommend a limit")
    args = parser.parse_args(argv)
    if args.single is not None:
        try:
            result = measure(args.single, curve_measurement=args.diagnostic_curve_64)
        except Exception as error:
            result = {"status": "failed", "error": f"{type(error).__name__}: {error}",
                "measurement_kind": CURVE_MEASUREMENT if args.diagnostic_curve_64 else "installed_recipe_wall_timing",
                **numerical_measurements({})}
        (args.single / "Timing.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        return 0 if result["status"] == "passed" else 1
    if args.output_dir is None or args.repetitions < 1:
        parser.error("--output-dir and a positive --repetitions are required")
    return 0 if diagnose(args.repetitions, args.output_dir.resolve(),
                        curve_measurement=args.diagnostic_curve_64)["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
