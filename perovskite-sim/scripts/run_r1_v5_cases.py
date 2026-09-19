"""Run an externally selected, bounded V5 short-window development case set.

Invoke on a clean commit with an isolated interpreter and explicit dependency
paths. Results are development numerical evidence, never R1-2 qualification.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
TIMES = [0.0, 1e-9, 1e-8, 1e-6, 1e-4]
MAX_CASES = 32
TIME_LADDERS = ([1, 2, 4], [2, 4, 8], [4, 8, 16], [8, 16, 32], [16, 32, 64])
CASE_FIELDS = {"control", "intervals", "nonlinear_factor", "time_substeps", "amplitude_V", "times_s"}
THREAD_KEYS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    raw = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".pending-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def case_name(case):
    factor = str(float(case["nonlinear_factor"])).replace(".", "p")
    return f"{case['control']}_N{case['intervals']}_F{factor}_T{case['time_substeps'][0]}"


def load_request(path, expected_sha256, output):
    path, output = Path(path).resolve(), Path(output).resolve()
    if path.is_relative_to(output):
        raise ValueError("case request must be held outside the result directory")
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(c not in "0123456789abcdef" for c in expected_sha256)):
        raise ValueError("a full external request SHA-256 is required")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("case request differs from the supplied external digest")
    request = json.loads(raw)
    if (not isinstance(request, dict) or set(request) != {"schema", "cases"}
            or request["schema"] != "R1V5CaseRequestV1" or not isinstance(request["cases"], list)
            or not 1 <= len(request["cases"]) <= MAX_CASES):
        raise ValueError("invalid bounded V5 case request")
    names = set()
    for case in request["cases"]:
        if not isinstance(case, dict) or set(case) != CASE_FIELDS:
            raise ValueError("each case requires the six explicit physical/numerical axes")
        if (case["control"] not in ("A", "B", "C", "D") or type(case["intervals"]) is not int
                or case["intervals"] not in (16, 32, 64, 128, 256)):
            raise ValueError("unsupported case control or grid")
        if (type(case["nonlinear_factor"]) not in (int, float)
                or case["nonlinear_factor"] not in (1.0, 0.1, 0.01, 0.001)):
            raise ValueError("unsupported nonlinear factor")
        steps = case["time_substeps"]
        if (not isinstance(steps, list) or any(type(v) is not int for v in steps)
                or steps not in TIME_LADDERS):
            raise ValueError("unsupported time-substep ladder")
        times = case["times_s"]
        if (type(case["amplitude_V"]) not in (int, float) or case["amplitude_V"] != 0.005
                or not isinstance(times, list) or any(type(v) not in (int, float) for v in times)
                or times != TIMES):
            raise ValueError("V5 cases require the fixed five-point short window and +5 mV")
        name = case_name(case)
        if name in names:
            raise ValueError("duplicate requested case: " + name)
        names.add(name)
    return request, raw


def git(*args):
    return subprocess.check_output(["git", *args], cwd=PROJECT, text=True).strip()


def source_files():
    root = Path(git("rev-parse", "--show-toplevel"))
    names = subprocess.check_output(["git", "ls-files"], cwd=root, text=True).splitlines()
    return {name: digest(root / name) for name in names}


def seal(directory):
    write(directory / "ManifestV1.json", {path.relative_to(directory).as_posix(): {
        "sha256": digest(path), "bytes": path.stat().st_size}
        for path in sorted(directory.rglob("*")) if path.is_file() and path != directory / "ManifestV1.json"})


def exception_record(exc):
    return {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}


def run(args):
    output = args.output.resolve()
    request, request_bytes = load_request(args.request_file, args.request_sha256, output)
    if output.exists():
        raise FileExistsError("case output already exists; use a new output for a new run")
    if (len(args.expected_commit) != 40 or any(c not in "0123456789abcdef" for c in args.expected_commit)
            or git("rev-parse", "HEAD") != args.expected_commit or git("status", "--porcelain")):
        raise ValueError("development cases require the specified full commit and a clean frozen source")
    for name in THREAD_KEYS:
        os.environ[name] = "1"
    sys.path.insert(0, str(PROJECT))
    import numpy as np
    import scipy.linalg
    from threadpoolctl import threadpool_info, threadpool_limits
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_checkout import require_r1_checkout
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_binding import standard_binding_record
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import prepare_common_state, json_data, execution_source
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import run_r1_step, r1_policy
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_physics_validation import verify_r1_step_physics
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_failure_witness import build_failure_witness
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_failure_reconstruction import rebuild_failure_witness
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_result_validation import failure_scope_report
    from perovskite_sim.models.config_loader import load_device_from_yaml

    context = require_r1_checkout(project=PROJECT, formal=True, source_commit=args.expected_commit)
    source, initial_files = execution_source(), source_files()
    standard = standard_binding_record(context)
    fixture = context.read_bytes("tests/fixtures/configs/dynamic_interface_defect_ion_transient_absorber_only.yaml")
    binding_raw = context.read_bytes("tests/fixtures/OneDimensionalMechanismR1ReferenceBindingV1.json")
    binding = json.loads(binding_raw)

    def source_unchanged():
        if (git("rev-parse", "HEAD") != args.expected_commit or git("status", "--porcelain")
                or digest(args.request_file) != args.request_sha256):
            raise ValueError("source or external request changed during the run")
        require_r1_checkout(project=PROJECT, formal=True, source_commit=args.expected_commit,
                            expected_source_sha256=context.source_content_sha256)
        if execution_source() != source:
            raise ValueError("executed source identity changed during the run")

    output.mkdir(parents=True)
    (output / "RequestV1.json").write_bytes(request_bytes)
    (output / "SourceFixtureV1.yaml").write_bytes(fixture)
    (output / "ReferenceBindingV1.json").write_bytes(binding_raw)
    write(output / "StandardBindingV1.json", standard)
    write(output / "SourceReceiptV1.json", {"source_commit": args.expected_commit,
        "source_content_sha256": context.source_content_sha256, "actual_source": source,
        "script_sha256": digest(__file__), "tracked_files": initial_files})
    stack = load_device_from_yaml(output / "SourceFixtureV1.yaml")
    summary = {"schema": "R1V5CaseRunV1", "scope": "development_short_window_evidence_not_R1_2_qualification",
        "source_commit": args.expected_commit, "request_sha256": args.request_sha256,
        "started_utc": datetime.now(timezone.utc).isoformat(), "run_status": "running",
        "scientifically_accepted": False, "cases": [{"case": case_name(case), "request": case,
            "status": "not_run", "execution_status": "not_run", "accepted_rows": 0,
            "reason": "awaiting_requested_execution", "short_case_checks_passed": False}
            for case in request["cases"]]}
    write(output / "Summary.json", summary)
    preparations, preparation_errors = {}, {}
    current = None
    with threadpool_limits(limits=1, user_api="blas"):
        runtime = {"executable": sys.executable, "python": sys.version, "numpy": np.__version__,
            "scipy": scipy.__version__, "isolated": bool(sys.flags.isolated), "no_site": bool(sys.flags.no_site),
            "threads": {key: os.environ[key] for key in THREAD_KEYS}, "blas": threadpool_info()}
        if any(item["num_threads"] != 1 for item in runtime["blas"]):
            raise ValueError("BLAS single-thread limit did not take effect")
        write(output / "EnvironmentV1.json", runtime)
        try:
            for current in summary["cases"]:
                source_unchanged()
                case, n = current["request"], current["request"]["intervals"]
                if n in preparation_errors:
                    current.update(reason="required_preparation_failed", preparation_error=preparation_errors[n])
                    continue
                directory = output / current["case"]
                directory.mkdir()
                current.update(status="running", phase="preparation", reason=None)
                write(directory / "RequestV1.json", case)
                write(output / "Summary.json", summary)
                start = time.monotonic()
                try:
                    if n not in preparations:
                        preparations[n] = prepare_common_state(stack, n, binding, policy=r1_policy())
                    prepared = preparations[n]
                    write(directory / "Prepared.json", json_data(prepared.to_dict()))
                except Exception as exc:
                    error = exception_record(exc)
                    preparation_errors[n] = error
                    current.update(status="failed", preparation_error=error, elapsed_s=time.monotonic() - start)
                    write(directory / "PreparationFailureV1.json", error)
                    write(directory / "RecordV1.json", current)
                    seal(directory)
                    write(output / "Summary.json", summary)
                    continue
                current.update(phase="integration", execution_status="running")
                policy = r1_policy(case["nonlinear_factor"], time_substeps=tuple(case["time_substeps"]))
                observer = directory / "AcceptedStepsV1.jsonl"
                observer.touch()
                def observe(row):
                    with observer.open("a") as stream:
                        stream.write(json.dumps(json_data(row), allow_nan=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                error = None
                try:
                    result = run_r1_step(stack, n, binding, prepared, control=case["control"],
                        amplitude_V=case["amplitude_V"], policy=policy, times_s=case["times_s"],
                        physics_evidence=True, accepted_step_observer=observe)
                    current["execution_status"] = "completed"
                except Exception as exc:
                    error = exception_record(exc)
                    result = getattr(exc, "result", None)
                    current.update(execution_status="failed", execution_error=error)
                result = json_data(result) if isinstance(result, dict) else {}
                write(directory / "Result.json", result)
                current.update(elapsed_s=time.monotonic() - start, accepted_rows=len(result.get("accepted_steps", [])))
                replay, terminal, rows = None, None, []
                try:
                    rows = [json.loads(line) for line in observer.read_text().splitlines()]
                    current["observer_matches_result"] = rows == result.get("accepted_steps", [])
                    replay = verify_r1_step_physics(stack, n, binding, prepared, result,
                        allow_incomplete=current["execution_status"] != "completed")
                    write(directory / "PhysicsReplay.json", {"schema": "R1V5CasePhysicsReplayV1",
                        "scope": "complete_short_case" if error is None else "saved_failed_prefix_only",
                        "run_completed": error is None, "scientifically_accepted": False, "reconstruction": replay})
                except Exception as exc:
                    current["physics_replay_error"] = exception_record(exc)
                if error is not None:
                    write(directory / "FailureV1.json", error)
                    protocol = {"stage": "controlled-step", "intervals": n, "control": case["control"],
                        "amplitude_V": case["amplitude_V"], "times_s": case["times_s"], "policy": json_data(policy)}
                    try:
                        write(directory / "FailureWitnessV1.json", build_failure_witness(
                            source_commit=args.expected_commit, protocol=protocol, failure=error,
                            result=result, persisted_rows=rows))
                        terminal = rebuild_failure_witness(stack, n, binding, prepared, result)
                        write(directory / "FailureReplay.json", terminal)
                    except Exception as exc:
                        current["terminal_replay_error"] = exception_record(exc)
                    if replay is not None:
                        write(directory / "FailureScopeV1.json", failure_scope_report(result, replay,
                            persisted_rows=rows, failure=error, terminal_reconstruction=terminal))
                current["short_case_checks_passed"] = bool(error is None and replay is not None
                    and result.get("certificate", {}).get("certified") is True and replay.get("certified") is True
                    and current.get("observer_matches_result") is True)
                current["status"] = "completed" if current["short_case_checks_passed"] else "failed"
                source_unchanged()
                current["source_identity_unchanged"] = True
                write(directory / "RecordV1.json", current)
                seal(directory)
                write(output / "Summary.json", summary)
                print(current["case"], current["status"], current["accepted_rows"], flush=True)
            summary["run_status"] = "completed"
        except (Exception, KeyboardInterrupt) as exc:
            summary.update(run_status="interrupted" if isinstance(exc, KeyboardInterrupt) else "aborted",
                           run_error=exception_record(exc))
            if current is not None and current["status"] != "not_run":
                current.update(status="interrupted", short_case_checks_passed=False, stop_error=summary["run_error"])
                if current["execution_status"] == "running":
                    current["execution_status"] = "interrupted"
                directory = output / current["case"]
                if directory.is_dir():
                    write(directory / "RecordV1.json", current)
                    seal(directory)
            for item in summary["cases"]:
                if item["status"] == "not_run":
                    item["reason"] = "batch_stopped_before_case"
        finally:
            try:
                source_unchanged()
                summary["source_files_unchanged"] = source_files() == initial_files
            except Exception as exc:
                summary.update(source_files_unchanged=False, final_identity_error=exception_record(exc))
            summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
            summary["all_requested_cases_passed"] = bool(summary["source_files_unchanged"]
                and summary["run_status"] == "completed"
                and all(item["short_case_checks_passed"] for item in summary["cases"]))
            summary["not_run_cases"] = [item["case"] for item in summary["cases"] if item["status"] == "not_run"]
            summary["exit_code"] = (1 if summary["run_status"] != "completed" or not summary["source_files_unchanged"]
                or any(item["status"] == "failed" for item in summary["cases"])
                else 2 if summary["not_run_cases"] else 0)
            write(output / "Summary.json", summary)
            seal(output)
    return summary["exit_code"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--request-file", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    try:
        return run(parser.parse_args(argv))
    except Exception as exc:
        print(f"V5 CASE RUN STOPPED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
