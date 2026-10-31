"""Time the identical ionic J-V child outside the supervisor's whole-job wall.

Each sample uses a fresh interpreter and the real spawn entry, request, controls,
provenance and artifact writer. Numerical-stage and output limits are unchanged.
This is diagnostic execution, not a store submission or a scientific qualification.
The workflow timeout bounds the diagnostic; there is no per-sample wall censor.
"""
from __future__ import annotations

import argparse
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
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def distribution(values: list[float]) -> dict[str, float]:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("positive finite uncensored measurements are required")
    ordered = sorted(values)
    return {"p50": statistics.median(ordered), "p95": ordered[math.ceil(.95 * len(ordered)) - 1],
            "max": ordered[-1]}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete = bool(rows) and all(row.get("status") == "passed" for row in rows)
    result: dict[str, Any] = {"status": "passed" if complete else "failed",
        "requested": len(rows), "completed": sum(row.get("status") == "passed" for row in rows),
        "percentiles": "p50: median; p95: nearest rank; seconds",
        "recommended_wall_seconds": None}
    if complete:
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


def measure(output: Path) -> dict[str, Any]:
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
    preparation_seconds = time.perf_counter() - preparation_started
    directory = output / "child"
    directory.mkdir()
    context = multiprocessing.get_context("spawn")
    stop, lock = context.Event(), context.Lock()
    header, buffer = context.RawArray("q", 4), context.RawArray("B", PROGRESS_BYTES)
    process = context.Process(target=_child_main, args=(
        spec.request.to_bytes(), binding, spec.limits, str(directory),
        {"diagnostic": "uncensored_ionic_jv", "sample": output.name}, stop, lock, header, buffer, execution),
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
        "python_version": sys.version, "parent_imports_seconds": imports_seconds,
        "parent_preparation_seconds": preparation_seconds, "child_wall_seconds": child_wall_seconds,
        "scope": "spawn, engine imports, actual worker, provenance and child artifact fsync; no supervisor/store commit",
        "whole_job_wall_enforced": False, "numerical_stage_limits_unchanged": True,
        "request": submitted, "request_sha256": spec.request.request.sha256,
        "manifest_sha256": execution.expected_sha256, "exitcode": process.exitcode,
        "process_reaped": not process.is_alive(), "terminal_state": terminal["state"],
        "reason": result.get("reason"), "outcome": summary.get("outcome"),
        "accepted_count": summary.get("accepted_count"), "artifact_count": len(artifacts),
        "terminal_sha256": hashlib.sha256(terminal_path.read_bytes()).hexdigest(),
        "child_peak_rss_bytes": terminal["child_peak_rss_bytes"]}
    if (process.exitcode != 0 or terminal["state"] != "succeeded" or summary.get("outcome") != "completed"
            or summary.get("accepted_count") != summary.get("requested_count") or summary.get("accepted_count") != 51
            or len(artifacts) != 15):
        return measurement
    for artifact in artifacts:
        raw = (directory / artifact["file"]).read_bytes()
        if len(raw) != artifact["bytes"] or hashlib.sha256(raw).hexdigest() != artifact["sha256"]:
            raise ValueError("direct child artifact integrity failed")
        if artifact["name"] == "ionic_jv_diagnostics":
            history = json.loads(raw)
            measurement["numerical_seconds"] = {key: history[key]["wall_s"]
                for key in ("dark_steady", "illumination", "curve")}
    capture = result["capture"]
    measurement.update(status="passed", runtime=capture["runtime"],
        computation_counts=summary["computation_counts"],
        worker_body_seconds=(datetime.fromisoformat(capture["finished_utc"])
                             - datetime.fromisoformat(capture["started_utc"])).total_seconds())
    return measurement


def diagnose(repetitions: int, output: Path) -> dict[str, Any]:
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
        with (directory / "Timing.log").open("w") as log:
            finished = subprocess.run(command, cwd=ROOT, env=environment(directory),
                stdout=log, stderr=subprocess.STDOUT, check=False)
        measurement = json.loads(report.read_text()) if report.exists() else {"status": "failed", "error": "missing timing receipt"}
        row = {"attempt": index, "exit_code": finished.returncode, "measurement": measurement,
               "status": "passed" if finished.returncode == 0 and measurement["status"] == "passed" else "failed"}
        rows.append(row)
        summary = {**summarize(rows), "attempts": rows}
        (output / "Summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        print(f"[{index}/{repetitions}] {row['status']}: {measurement.get('child_wall_seconds')} s", flush=True)
    print("Uncensored ionic timing: " + json.dumps(summarize(rows), sort_keys=True), flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--single", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.single is not None:
        try:
            result = measure(args.single)
        except Exception as error:
            result = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
        (args.single / "Timing.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        return 0 if result["status"] == "passed" else 1
    if args.output_dir is None or args.repetitions < 1:
        parser.error("--output-dir and a positive --repetitions are required")
    return 0 if diagnose(args.repetitions, args.output_dir.resolve())["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
