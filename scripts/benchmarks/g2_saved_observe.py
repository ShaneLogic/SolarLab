"""Observe one sealed G2 probe into a fresh directory under separate Root authority.

The manifest binds producer/failed-observer plans and receipts, the immutable
producer result, an accepted source review, source hashes, and a fresh ordinary
launcher execution. The legacy 20751ff tree stays read-only. This module never
calls a producer, changes a scientific threshold, or resets an old clock.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import sys
import time

from scripts.benchmarks import g2_route as route
from scripts.benchmarks import g2_route_launcher as launcher

SCHEMA = "solarlab.g2-saved-observe.v1"
LEGACY_COMMIT = "20751ff9bacd4040d930a9e794b4348094d4fda7"
GIB = 1024**3


class ObservationStopped(RuntimeError):
    """An operational stop, deliberately outside the reader's ValueError catch."""

    def __init__(self, kind, detail):
        self.kind = kind
        super().__init__("g2_saved_observe_" + detail)


def exception_evidence(error):
    """Bounded exception/call-site facts, without locals or native operands.

    Generic reader ValueErrors remain unclassified unless their retained chain
    identifies our own stop or a definite JSON/container decoding error.
    """
    chain, seen, current = [], set(), error
    classification = "unclassified_error"
    while current is not None and id(current) not in seen and len(chain) < 4:
        seen.add(id(current))
        if isinstance(current, ObservationStopped):
            classification = current.kind
        elif classification == "unclassified_error" and isinstance(
                current, (json.JSONDecodeError, UnicodeDecodeError, gzip.BadGzipFile, EOFError)):
            classification = "artifact_decode_error"
        parts = []
        for arg in current.args[:3]:
            if type(arg) is str:
                # Truncate before formatting; never stringify arrays/large ints.
                parts.append(re.sub(r"(?:0x)?[0-9a-fA-F]{32,}", "<numeric-word>", arg[:1024]))
            elif arg is None or type(arg) is bool or type(arg) is int and arg.bit_length() <= 63:
                parts.append(str(arg))
            else:
                parts.append("<" + type(arg).__name__[:80] + ">")
        frames, tb = [], current.__traceback__
        while tb is not None:
            frames.append({"file": Path(tb.tb_frame.f_code.co_filename).name[:120],
                           "function": tb.tb_frame.f_code.co_name[:100], "line": tb.tb_lineno})
            frames = frames[-8:]
            tb = tb.tb_next
        chain.append({"type": type(current).__name__[:80], "message": "; ".join(parts)[:1024],
                      "frames": frames})
        current = current.__cause__ if current.__cause__ is not None else (
            None if current.__suppress_context__ else current.__context__)
    return {"exception": chain[0]["type"], "reason": chain[0]["message"],
            "classification": classification, "chain": chain,
            "chain_truncated": current is not None}


def require(condition, reason):
    if not condition:
        raise ValueError("g2_saved_observe_" + reason)


def pinned(reference, *, decode=True):
    require(route._resolved_descriptor(reference), "file_descriptor")
    absolute(reference["path"])
    value = route._pinned(reference, decode=decode)
    require(not decode or type(value) is dict, "metadata_object")
    return value


def absolute(value):
    require(type(value) is str, "absolute_path")
    path = Path(value)
    require(path.is_absolute() and str(path.resolve()) == value, "absolute_path")
    return path


def processes_absent(receipt):
    pids = receipt.get("observed_pids")
    closure = receipt.get("closure")
    require(type(pids) is list and pids and all(type(p) is int and p > 0 for p in pids)
            and pids == sorted(set(pids)) and receipt.get("pid") in pids
            and receipt["pid"] == receipt.get("pgid") == receipt.get("session_id"), "owned_pids")
    require(type(closure) is dict and closure.get("schema") == "solarlab.owned-tool-process-closure.v1"
            and closure.get("pgid") == receipt["pgid"]
            and closure.get("after_tool_exit_code") == receipt.get("exit_code")
            and closure.get("process_group_absent") is True
            and closure.get("all_named_owned_processes_and_groups_absent") is True
            and closure.get("remaining_group_members") == closure.get("remaining_owned_processes") == []
            and closure.get("error") is None, "owned_closure")
    checks = closure.get("checks")
    require(type(checks) is list and all(type(row) is dict and type(row.get("pid")) is int
            and row.get("pid_absent") is True and row.get("same_id_process_group_absent") is True for row in checks)
            and [row["pid"] for row in checks] == pids, "owned_closure_checks")
    for pid in pids:
        for check in (os.kill, os.killpg):
            try:
                check(pid, 0)
            except ProcessLookupError:
                continue
            except PermissionError:
                return False
            return False
    return True


def _validate(manifest_path):
    raw = route._read(manifest_path)
    manifest = json.loads(raw)
    require(type(manifest) is dict and manifest.get("schema") == SCHEMA, "manifest_schema")
    authority = manifest.get("authority")
    require(type(authority) is dict and authority.get("role") == "Root"
            and authority.get("authorized") is True
            and authority.get("scope") == "saved-observation-only"
            and type(authority.get("coordinator_message")) is str
            and authority["coordinator_message"].startswith("msg_"), "Root_authority")
    operation = manifest.get("operation_id")
    require(type(operation) is str and bool(operation), "operation_identity")
    source, target = (absolute(manifest[k]) for k in ("source_directory", "output_directory"))
    require(source.is_dir() and not target.exists()
            and not source.is_relative_to(target) and not target.is_relative_to(source)
            and not target.is_relative_to(launcher.REPOSITORY), "fresh_separate_output")
    review = pinned(manifest["source_review"])
    sources = manifest["source_sha256"]
    require(type(sources) is dict and sources and type(review) is dict
            and review.get("accepted") is True and review.get("source_sha256") == sources
            and review.get("legacy_execution_commit") == LEGACY_COMMIT, "source_review")
    required = {str(Path(__file__).resolve()), str(Path(route.__file__).resolve()),
                str(Path(launcher.__file__).resolve())}
    required.update(str(Path(route.__file__).with_name(name).resolve()) for name in route.EVALUATOR_FILES)
    require(required <= sources.keys(), "required_sources")
    for path, digest in sources.items():
        require(hashlib.sha256(route._read(absolute(path), 16*1024**2)).hexdigest() == digest, "source_changed")
    pinned(manifest["producer_plan"])
    producer = route.validate(manifest["producer_plan"]["path"], executable=True,
                              case_id=manifest["case_id"])
    require(producer["plan_sha256"] == manifest["producer_plan"]["sha256"], "producer_plan_changed")
    old = producer["plan"]
    require(old["source_freeze"]["future_execution_commit"] == LEGACY_COMMIT, "legacy_commit")
    case = next(c for c in old["cases"] if c["id"] == manifest["case_id"])
    execution = case["execution"]
    require(absolute(execution["directory"]) == source, "producer_directory")
    require(operation != execution["native_admission"]["operation_id"], "distinct_recovery_operation")
    require(all(sources.get(path) == digest for path, digest in old["source_freeze"]["source_sha256"].items()),
            "legacy_sources_changed")
    receipt = pinned(manifest["producer_closure"])
    require(absolute(manifest["producer_closure"]["path"]) == absolute(execution["supervision"]["closure_path"]),
            "producer_closure_path")
    route._probe_runtime_receipt(receipt, execution, manifest["producer_closure"],
                                 manifest["producer_plan"], case["id"])
    require(processes_absent(receipt), "producer_still_present")
    result = pinned(manifest["producer_result"])
    history = result.get("history")
    require(absolute(manifest["producer_result"]["path"]) == source / "NativeResult.json"
            and result.get("complete_run_window") is True
            and type(history) is dict and history.get("closed") is True
            and history.get("container_complete") is True and history.get("error") is None
            and result.get("request_sha256") == execution["effective_input_digest"], "sealed_producer_result")
    request = pinned(execution["request"])
    require(route._read(source / "NativeRequest.json") == pinned(execution["request"], decode=False), "source_request_changed")
    failed_plan = pinned(manifest["failed_observe_plan"])
    evaluated = route.validate(manifest["failed_observe_plan"]["path"], executable=True,
                               case_id=case["id"], purpose="evaluate")
    require(evaluated["plan_sha256"] == manifest["failed_observe_plan"]["sha256"]
            and failed_plan["source_freeze"] == old["source_freeze"]
            and failed_plan["definitions"] == old["definitions"], "original_evaluation_binding")
    prior = next((c["execution"] for c in failed_plan["cases"] if c["id"] == case["id"]), None)
    failed = pinned(manifest["failed_observe_closure"])
    require(type(prior) is dict and prior["directory"] == str(source)
            and prior["native_admission"] == execution["native_admission"]
            and prior.get("execution_provenance") == execution.get("execution_provenance")
            and prior["request"] == execution["request"]
            and failed.get("schema") == "solarlab.g2-route.launcher-result.v1"
            and failed.get("case_id") == case["id"]
            and failed.get("plan_sha256") == manifest["failed_observe_plan"]["sha256"]
            and failed.get("command") == prior["command"]
            and type(failed.get("channel")) is int and failed["channel"] == prior["channel"]
            and failed.get("channel_protocol") == launcher.CHANNEL_PROTOCOL
            and failed.get("cwd") == str(launcher.REPOSITORY)
            and absolute(manifest["failed_observe_closure"]["path"]) == absolute(prior["supervision"]["closure_path"])
            and failed.get("process_outcome") == failed.get("termination_reason") == "wall_limit"
            and failed.get("closure", {}).get("all_named_owned_processes_and_groups_absent") is True,
            "failed_observe_binding")
    require(processes_absent(failed), "observer_still_present")
    recovery = manifest["execution"]
    require(type(recovery.get("channel")) is int and 0 <= recovery["channel"] < 4
            and recovery.get("directory") == str(target)
            and recovery.get("cwd") == str(launcher.REPOSITORY), "recovery_execution")
    policy = route._validate_supervision(old, recovery)
    limits = launcher.Limits.from_execution(recovery)
    require(limits.wall_s <= 3600 and limits.rss_bytes <= 4*GIB and limits.artifact_bytes <= 4*GIB,
            "recovery_resource_ceiling")
    require(set(("pipeline_started_unix_ns", "pipeline_wall_s")) <= policy.keys()
            and policy["pipeline_wall_s"] == recovery["wall_s"]
            and policy["pipeline_started_unix_ns"] != prior["supervision"]["pipeline_started_unix_ns"], "new_operation_clock")
    for key in ("closure_path", "cancel_path"):
        path = absolute(policy[key])
        require(not path.is_relative_to(source) and not path.is_relative_to(target)
                and not path.is_relative_to(launcher.REPOSITORY) and not path.exists(), "recovery_control_path")
    caps = manifest["read_limits"]
    require(type(caps) is dict and set(caps) == {"max_record_bytes", "max_records", "max_logical_bytes"},
            "read_limit_fields")
    for name, maximum in (("max_record_bytes", execution["max_record_bytes"]),
                          ("max_records", execution["max_records"]),
                          ("max_logical_bytes", old["resources"]["reader_logical_byte_cap"])):
        require(type(caps.get(name)) is int and 0 < caps[name] <= maximum, "read_limit_" + name)
    return {"manifest": manifest, "manifest_sha256": hashlib.sha256(raw).hexdigest(), "plan": old,
            "case": case, "request": request, "source": source, "target": target,
            "limits": limits, "producer_result": result}


def validate(manifest_path):
    """Validate metadata only; malformed containers are ordinary refusals."""
    try:
        return _validate(manifest_path)
    except (KeyError, TypeError, AttributeError, StopIteration, OverflowError) as error:
        raise ValueError("g2_saved_observe_metadata_shape: " + type(error).__name__) from error


@contextmanager
def cancellation():
    def stop(number, frame):
        raise ObservationStopped("cancelled", "signal_" + str(number))
    saved = {number: signal.signal(number, stop) for number in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield
    finally:
        for number, handler in saved.items():
            signal.signal(number, handler)


def _blas_one():
    library = ctypes.CDLL("/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/libBLAS.dylib")
    library.BLASSetThreading.argtypes = [ctypes.c_int]
    library.BLASSetThreading.restype = ctypes.c_int
    library.BLASGetThreading.restype = ctypes.c_int
    library.BLASSetThreading(1)
    require(library.BLASGetThreading() == 1, "actual_BLAS_thread_mode")


def _observe(context, checkpoint):
    from scripts.benchmarks.coupled_device_prototype import SlabDefinition, AffineCoupledSlab, AffineVoltageMap
    from scripts.benchmarks.g2_compare import case_metadata, normalize_ida_sample
    from scripts.benchmarks.native_history import HistoryWriter, integer_json_dumps
    from scripts.benchmarks.native_observation import SealedChargeObserver
    from scripts.benchmarks.native_readback import verify_history
    from scripts.benchmarks.native_runner import output_bytes

    m, plan, case, request = (context[k] for k in ("manifest", "plan", "case", "request"))
    source, target, execution = context["source"], context["target"], case["execution"]
    source_bytes = output_bytes(source, ())
    cap = min(context["limits"].artifact_bytes, plan["resources"]["combined_persisted_case_bytes"]-source_bytes)
    if cap <= 2*1024**2:
        raise ObservationStopped("resource_limit", "combined_output_cap")
    model = AffineCoupledSlab(SlabDefinition(**request["numeric_packet"]["definition"]), case["mesh_intervals"])
    mapping = AffineVoltageMap(model, mapped_input_profile=request["voltage_lift_map"]["mapped_input_profile"])
    target.mkdir(parents=True, exist_ok=False)
    writer = HistoryWriter(target / "Charge.jsonl.gz", encoding="gzip", total_output_bytes=cap,
        max_record_bytes=m["read_limits"]["max_record_bytes"], integer_io_policy=execution.get("integer_io_policy"))
    try:
        checkpoint()
        run_id = execution["native_admission"]["operation_id"]
        gate_ref = plan["source_freeze"]["frozen_gate_spec"]
        observer = SealedChargeObserver(mapping, request, writer.append, checkpoint,
            t1_context={"spec": pinned(gate_ref), "spec_sha256": gate_ref["sha256"],
                        "run_id": run_id, "case_id": case["id"]})
        comparison = case_metadata(request, case["id"])
        comparison["run_id"] = run_id
        writer.append({"kind": "sealed_g2_context", "request_sha256": route._digest(request), "evidence": comparison})
        def retain(tag, sample, certificate):
            checkpoint()
            writer.append({"kind": "sealed_g2_sample", "request_sha256": route._digest(request),
                "evidence": normalize_ida_sample(request, tag, sample, certificate)})
        result = verify_history(source, **m["read_limits"], deadline_monotonic=context["work_deadline"],
            integer_io_policy=execution.get("integer_io_policy"), execution_admission=execution["native_admission"],
            execution_provenance=execution.get("execution_provenance"), charge_observer=observer, sample_sink=retain)
        checkpoint()
        writer.finish()
        # Scientific identities remain those of the original producer. The
        # new operation and its budget are recorded separately below.
        result.update(plan_sha256=m["producer_plan"]["sha256"], case_id=case["id"], methods=route.METHODS,
            observed_history={"path": "Charge.jsonl.gz", "records": writer.records,
                "logical_sha256": writer.logical_sha256, "logical_bytes": writer.logical_bytes,
                "encoded_bytes": writer.encoded_bytes, "closed": writer.closed, "container_complete": writer.closed})
        raw = (integer_json_dumps(result, execution.get("integer_io_policy"), indent=2)+"\n").encode()
        if output_bytes(target, ())+len(raw)+65536 > cap:
            raise ObservationStopped("resource_limit", "output_cap")
        require(pinned(m["producer_result"]) == context["producer_result"], "producer_changed_during_read")
        with (target / "RunResult.json").open("xb") as stream:
            stream.write(raw)
        return result
    except BaseException as error:
        try:
            writer.abort(error)
            writer.publish_first_failure(exception_evidence(error))
        except BaseException as secondary:
            error.add_note("saved observer closeout: " + type(secondary).__name__)
        raise


def run(manifest_path):
    started = time.monotonic()
    context = validate(manifest_path)
    m, limits = context["manifest"], context["limits"]
    policy = m["execution"]["supervision"]
    clock = launcher._pipeline_deadline(policy, started, limits)
    deadline = launcher._require_pipeline_launch_budget(clock, "saved_observe")
    context["work_deadline"] = deadline-limits.cleanup_s
    def checkpoint():
        context["work_deadline"] = min(context["work_deadline"],
            launcher._refresh_pipeline_deadline(clock)-limits.cleanup_s)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        if Path(policy["cancel_path"]).exists():
            raise ObservationStopped("cancelled", "cancel_path")
        if time.perf_counter() >= context["work_deadline"]:
            raise ObservationStopped("deadline", "work_deadline")
        if rss > limits.rss_bytes:
            raise ObservationStopped("resource_limit", "RSS_limit")
    checkpoint()
    require(all(os.environ.get(k) == "1" for k in launcher.THREAD_VARIABLES), "thread_environment")
    _blas_one()
    with launcher.inherited_channel(context["plan"]["artifacts"]["channel_directory"], m["execution"]["channel"]), cancellation():
        result = _observe(context, checkpoint)
    receipt = {"operation_id": m["operation_id"], "manifest_sha256": context["manifest_sha256"],
        "source_directory": str(context["source"]), "output_directory": str(context["target"]),
        "producer_plan": m["producer_plan"], "failed_observe_closure": m["failed_observe_closure"],
        "producer_operation_id": context["case"]["execution"]["native_admission"]["operation_id"],
        "execution": m["execution"], "elapsed_s": time.perf_counter()-started,
        "old_failures_retained": True, "producer_runs": 0}
    with (context["target"] / "RecoveryReceipt.json").open("x") as stream:
        raw = json.dumps(receipt, indent=2)+"\n"
        require(len(raw.encode()) <= 65536, "recovery_receipt_reserve")
        stream.write(raw)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.validate_only:
            value = validate(args.manifest)
            print(json.dumps({"status": "validated_not_executed", "operation_id": value["manifest"]["operation_id"]}))
        else:
            result = run(args.manifest)
            print(json.dumps({"status": "returned", "scientific_status": result.get("scientific_status"),
                              "scientific_pass": result.get("scientific_pass")}))
        return 0
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        print(json.dumps({"status": "refused_or_failed", **exception_evidence(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
