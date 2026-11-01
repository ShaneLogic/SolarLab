"""One short compute-only profile of the unchanged saved-history observer.

This entry records timings, call counts and bounded stack names, never charge
history, a verifier receipt, or qualification. It requires the existing outer
launcher and a fresh diagnostic operation; it never calls a producer.
"""
from __future__ import annotations

import argparse
from collections import Counter
import cProfile
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pstats
import resource
import signal
import sys
import time

from scripts.benchmarks import g2_route as route
from scripts.benchmarks import g2_route_launcher as launcher


def run(config_path):
    started = time.monotonic()
    config = json.loads(Path(config_path).read_bytes())
    execution = config["execution"]
    limits = launcher.Limits.from_execution(execution)
    if not (0 < limits.wall_s <= 60 and limits.rss_bytes <= 1024**3
            and limits.artifact_bytes <= 32*1024**2
            and config["observer_window_s"] == 20
            and config["task_id"] == "task_92ddca5bf4ba"):
        raise ValueError("diagnostic scope/caps")
    root = Path(execution["directory"])
    if root.exists() or root.resolve().is_relative_to(launcher.REPOSITORY):
        raise ValueError("fresh external diagnostic directory required")
    helper_path = Path(__file__).with_name("g2_saved_observe.py")
    if hashlib.sha256(helper_path.read_bytes()).hexdigest() != config["helper_sha256"]:
        raise ValueError("helper source changed")
    spec = importlib.util.spec_from_file_location("_profile_control", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    # The kernel hard limit includes import/setup and reporting CPU too. Even
    # an uncooperative C call cannot acquire a new 30-second CPU allowance.
    resource.setrlimit(resource.RLIMIT_CPU, (25, 28))
    parent = helper.pinned(config["producer_plan"])
    route.validate(config["producer_plan"]["path"], executable=True, case_id=config["case_id"])
    case = next(c for c in parent["cases"] if c["id"] == config["case_id"])
    original = case["execution"]
    request = helper.pinned(original["request"])
    for name in ("producer_closure", "failed_observe_closure", "saved_observe_closure"):
        if not helper.processes_absent(helper.pinned(config[name])):
            raise ValueError("old process/group still present")
    if Path(original["directory"]).resolve().is_relative_to(root.resolve()) or root.resolve().is_relative_to(Path(original["directory"]).resolve()):
        raise ValueError("source/output overlap")
    if not all(os.environ.get(name) == "1" for name in launcher.THREAD_VARIABLES):
        raise ValueError("BLAS1 thread environment")
    from scripts.benchmarks.coupled_device_prototype import SlabDefinition, AffineCoupledSlab, AffineVoltageMap
    from scripts.benchmarks.native_observation import SealedChargeObserver
    from scripts.benchmarks.native_readback import verify_history

    helper._blas_one()
    clock = launcher._pipeline_deadline(execution["supervision"], started, limits)
    model = AffineCoupledSlab(SlabDefinition(**request["numeric_packet"]["definition"]), case["mesh_intervals"])
    mapping = AffineVoltageMap(model, mapped_input_profile=request["voltage_lift_map"]["mapped_input_profile"])
    gate = parent["source_freeze"]["frozen_gate_spec"]
    profiler = cProfile.Profile()
    counts, stacks = Counter(), []
    cpu_start, wall_start = time.process_time(), time.monotonic()
    work_end = min(wall_start+20, launcher._refresh_pipeline_deadline(clock)-limits.cleanup_s)
    stop = None

    def checkpoint():
        nonlocal work_end
        counts["checkpoints"] += 1
        # Retain the real wrapper's clock mapping/RSS-check cost in the sample.
        work_end = min(work_end, launcher._refresh_pipeline_deadline(clock)-limits.cleanup_s)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        if Path(execution["supervision"]["cancel_path"]).exists():
            raise helper.ObservationStopped("cancelled", "profile_cancel")
        if time.monotonic() >= work_end or time.process_time()-cpu_start >= 20:
            raise helper.ObservationStopped("deadline", "profile_20s_window")
        if rss > limits.rss_bytes:
            raise helper.ObservationStopped("resource_limit", "profile_RSS")

    def sample(number, frame):
        if number == signal.SIGXCPU:
            raise helper.ObservationStopped("resource_limit", "profile_CPU_soft_limit")
        if len(stacks) < 40:
            names = []
            while frame is not None and len(names) < 12:
                names.append({"file": Path(frame.f_code.co_filename).name[:120],
                              "function": frame.f_code.co_name[:100], "line": frame.f_lineno})
                frame = frame.f_back
            stacks.append({"elapsed_s": time.monotonic()-wall_start, "frames": names})
        checkpoint()

    def emit(row):
        counts["emitted_"+row["kind"]] += 1
        checkpoint()

    def sample_sink(*args):
        counts["requested_samples"] += 1
        checkpoint()

    observer = SealedChargeObserver(mapping, request, emit, checkpoint,
        t1_context={"spec": helper.pinned(gate), "spec_sha256": gate["sha256"],
                    "run_id": original["native_admission"]["operation_id"], "case_id": case["id"]})
    with launcher.inherited_channel(parent["artifacts"]["channel_directory"], execution["channel"]), helper.cancellation():
        root.mkdir(parents=True, exist_ok=False)
        previous = {sig: signal.signal(sig, sample) for sig in (signal.SIGALRM, signal.SIGXCPU)}
        signal.setitimer(signal.ITIMER_REAL, .5, .5)
        try:
            profiler.enable()
            verify_history(Path(original["directory"]), max_record_bytes=original["max_record_bytes"],
                max_records=original["max_records"], max_logical_bytes=parent["resources"]["reader_logical_byte_cap"],
                deadline_monotonic=work_end+.25, integer_io_policy=original.get("integer_io_policy"),
                execution_admission=original["native_admission"], execution_provenance=original.get("execution_provenance"),
                charge_observer=observer, sample_sink=sample_sink)
        except BaseException as error:
            stop = helper.exception_evidence(error)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            profiler.disable()
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        wall_elapsed, cpu_elapsed = time.monotonic()-wall_start, time.process_time()-cpu_start
        profiler.dump_stats(root / "Calls.pstats")
        entries = []
        for (filename, line, function), (primitive, total, own, cumulative, _) in pstats.Stats(profiler).stats.items():
            entries.append({"file": Path(filename).name, "line": line, "function": function[:160],
                            "primitive_calls": primitive, "calls": total, "self_s": own, "cumulative_s": cumulative})
        result = {"scope": "bounded performance sample only", "task_id": config["task_id"],
            "operation_id": config["operation_id"], "configuration_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
            "source_commit": helper.LEGACY_COMMIT, "source_plan": config["producer_plan"],
            "pid": os.getpid(), "observer_window_s": 20, "profile_wall_s": wall_elapsed,
            "profile_CPU_s": cpu_elapsed, "whole_process_elapsed_s": time.monotonic()-started,
            "stop": stop, "counts": dict(counts), "stack_samples": stacks,
            "top_self": sorted(entries, key=lambda x:x["self_s"], reverse=True)[:40],
            "top_cumulative": sorted(entries, key=lambda x:x["cumulative_s"], reverse=True)[:40],
            "profiled_functions": len(entries), "sink": "counts only; output serialization/GZIP excluded",
            "qualification_output": False, "EOF_CRC_coverage_or_T1_pass_asserted": False,
            "producer_calls": 0, "retry_authorized": False}
        raw = (json.dumps(result, indent=2, sort_keys=True)+"\n").encode()
        if len(raw)+(root/"Calls.pstats").stat().st_size > 30*1024**2:
            raise ValueError("profile output reserve exceeded")
        with (root/"Profile.json").open("xb") as stream:
            stream.write(raw)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config)
    print(json.dumps({"profile_wall_s": result["profile_wall_s"], "profile_CPU_s": result["profile_CPU_s"],
                      "qualification_output": False, "stop": result["stop"]}))
