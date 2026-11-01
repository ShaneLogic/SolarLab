"""One bounded saved G2 observation with durable, explicitly partial output.

The existing reader, observer, integer codec and history writer are reused.
A healthy output writer may close a *prefix* after an operational stop; that
closure never grants input EOF/CRC, interval coverage, T1 or resume authority.
No producer is called, and the historical frozen execution tree is read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import cProfile
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pstats
import resource
import signal
import sys
import time

from scripts.benchmarks import g2_route as route
from scripts.benchmarks import g2_route_launcher as launcher
from scripts.benchmarks import g2_saved_observe as saved
from scripts.benchmarks.native_history import HistoryLimitError, HistoryReader, HistoryWriter, _line

SCHEMA = "solarlab.g2-observer-prefix-progress.v1"
MIB = 1024**2
KINDS = frozenset({"sealed_g2_context", "sealed_g2_sample", "sealed_voltage_lift_charge", "t1_fixed_window_piece"})


def _publish(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()
    if len(raw) > 256*1024:
        raise HistoryLimitError("bounded progress metadata exceeds reserve")
    temporary = path.with_suffix(".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _stop(error):
    result = saved.exception_evidence(error)
    current, seen = error, set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, HistoryLimitError):
            result["classification"] = "resource_limit"
            break
        current = current.__cause__ if current.__cause__ is not None else (
            None if current.__suppress_context__ else current.__context__)
    return result


class PrefixProgress:
    """Keep exact completed callback bytes and an atomic descriptive pointer."""

    def __init__(self, directory, *, binding, artifact_bytes, max_record_bytes, integer_io_policy=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.binding = binding
        self.counts = Counter()
        self.last = None
        self.saving = False
        # Reserve space for old/new progress pointers, outcome and bounded profile.
        if artifact_bytes <= 4*MIB:
            raise HistoryLimitError("prefix output requires its metadata/profile reserve")
        self.writer = HistoryWriter(self.directory/"ChargePrefix.jsonl.gz", encoding="gzip",
            total_output_bytes=artifact_bytes-4*MIB, max_record_bytes=max_record_bytes,
            integer_io_policy=integer_io_policy)
        self.publish()

    def snapshot(self, *, stop=None, reader_returned=False):
        writer = self.writer
        return {"schema": SCHEMA, "binding": self.binding,
            "status": "reader_returned" if reader_returned else "bounded_stop" if stop is not None else "running",
            "stop": stop, "records": writer.records, "counts_by_kind": dict(self.counts), "last_record": self.last,
            "output_prefix": {"path": writer.path.name, "logical_bytes": writer.logical_bytes,
                "encoded_bytes": writer.encoded_bytes, "logical_sha256": writer.logical_sha256,
                "writer_closed": writer.closed, "writer_container_complete": writer.container_complete,
                "writer_error": writer.error},
            "input_verifier_returned": reader_returned,
            "input_EOF_CRC_coverage_asserted": False, "T1_pass_asserted": False,
            "resume_eligible": False,
            "resume_blocker": "no serialized reader protocol, charge prefixes, active frame or partial T1 window state",
            "scientific_qualification": False, "producer_calls": 0}

    def publish(self):
        _publish(self.directory/"Progress.json", self.snapshot())

    def append(self, row):
        self.saving = True
        try:
            self._append(row)
        finally:
            self.saving = False

    def _append(self, row):
        if row.get("kind") not in KINDS:
            raise ValueError("unsupported observer callback kind")
        # Identical logical record codec/order to HistoryWriter.append. Passing
        # the line avoids a second encoding or mutation of exact numeric words.
        line = _line(row, self.writer.integer_io_policy)
        position = self.writer.append_line(line)
        self.counts[row["kind"]] += 1
        interval = row.get("interval", {})
        evidence = row.get("evidence", {})
        self.last = {"index": self.writer.records-1, "kind": row["kind"],
            "record_sha256": hashlib.sha256(line).hexdigest(),
            "history_byte_offset_after_record": position.logical_bytes,
            "source_interval_sha256": row.get("source_interval_sha256"),
            "path_identity": evidence.get("path_identity"), "piece_id": evidence.get("piece_id"),
            "window_index": evidence.get("window_index"), "tag": evidence.get("tag"),
            "segment_id": evidence.get("segment_id", interval.get("segment_id")),
            "t_left_hex": evidence.get("t_left_hex"), "t_right_hex": evidence.get("t_right_hex")}
        self.publish()

    def finish_prefix(self, *, stop=None, reader_returned=False):
        """Finalize only a healthy output stream; failed writes remain partial."""
        close_error = None
        if self.writer.error is None:
            try:
                self.writer.finish()
            except BaseException as error:
                close_error = _stop(error)
                if stop is None:
                    stop = close_error
        outcome = self.snapshot(stop=stop, reader_returned=reader_returned)
        outcome["prefix_close_error"] = close_error
        if stop is not None:
            self.writer.publish_first_failure(stop)
        _publish(self.directory/"Progress.json", outcome)
        _publish(self.directory/"PrefixOutcome.json", outcome)
        return outcome


def read_progress(directory, *, max_logical_bytes=256*MIB, max_record_bytes=16*MIB):
    """Read the new closed prefix only; never restore a numerical observer."""
    directory = Path(directory)
    value = json.loads((directory/"PrefixOutcome.json").read_bytes())
    if (value.get("schema") != SCHEMA or value.get("resume_eligible") is not False
            or value.get("input_EOF_CRC_coverage_asserted") is not False
            or value.get("T1_pass_asserted") is not False or value.get("scientific_qualification") is not False):
        raise ValueError("unsupported or overclaimed observation prefix")
    history = value["output_prefix"]
    if history["path"] != "ChargePrefix.jsonl.gz":
        raise ValueError("unexpected prefix path")
    if history["writer_container_complete"] and not history["writer_closed"]:
        raise ValueError("prefix cannot be complete while its writer is open")
    checked = {"output_prefix_EOF_CRC_verified": False, "input_EOF_CRC_coverage_asserted": False,
               "resume_eligible": False, "scientific_qualification": False}
    if history["writer_error"] is not None or not history["writer_container_complete"]:
        return {"progress": value, "read_checks": checked}
    if (directory/history["path"]).stat().st_size != history["encoded_bytes"]:
        raise ValueError("prefix encoded byte count differs")
    digest, counts, count, last_hash = hashlib.sha256(), Counter(), 0, None
    policy = value["binding"].get("integer_io_policy")
    with HistoryReader(directory/history["path"], encoding="gzip", max_record_bytes=max_record_bytes,
                       max_logical_bytes=max_logical_bytes, integer_io_policy=policy) as reader:
        for line in reader:
            if not line.endswith(b"\n") or count >= value["records"]:
                raise ValueError("prefix record count or completeness differs")
            row = json.loads(line)
            if row.get("kind") not in KINDS:
                raise ValueError("prefix callback kind differs")
            digest.update(line)
            last_hash = hashlib.sha256(line).hexdigest()
            counts[row["kind"]] += 1
            count += 1
        if (not reader.eof_seen or not reader.container_complete or reader.partial_tail
                or count != value["records"] or dict(counts) != value["counts_by_kind"]
                or digest.hexdigest() != history["logical_sha256"]
                or reader.logical_bytes != history["logical_bytes"]
                or (count > 0 and last_hash != value["last_record"]["record_sha256"])):
            raise ValueError("prefix digest/count/closure disagrees with progress")
    checked["output_prefix_EOF_CRC_verified"] = True
    return {"progress": value, "read_checks": checked}


def producer_metadata(reference, case_id):
    """Validate historical producer authority with its exact pinned entry.

    The old RunPlan rightly refuses a relocated/current entry as a producer.
    Load that unchanged metadata-only validator at its declared path instead;
    numerical observation remains bound separately to this current snapshot.
    """
    parent = saved.pinned(reference)
    case = next(case for case in parent["cases"] if case["id"] == case_id)
    entry = saved.absolute(case["execution"]["cwd"])/"scripts/benchmarks/g2_route.py"
    pin = parent["source_freeze"]["source_sha256"].get(str(entry))
    saved.require(hashlib.sha256(entry.read_bytes()).hexdigest() == pin, "producer_metadata_entry_changed")
    spec = importlib.util.spec_from_file_location("_g2_prefix_producer_metadata", entry)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    validated = module.validate(reference["path"], executable=True, case_id=case_id)
    saved.require(validated["plan_sha256"] == reference["sha256"], "producer_plan_binding")
    return parent, case


def backend_preflight(request, environment_reference):
    """Require the already frozen Arb runtime before any history is consumed.

    Paths are supplied by the launch environment, never discovered/installed
    or silently changed here. Constructing the existing adapter performs no
    ball evaluation, integral, observer callback or native producer operation.
    """
    from scripts.benchmarks.interval_observation import BallIntegrator
    try:
        lock = saved.pinned(environment_reference)
        saved.require(lock["python_version"] == sys.version, "observer_python_version")
        arithmetic = request["interval_observation"]["arithmetic"]
        adapter = BallIntegrator(**{name: arithmetic[name] for name in ("bits", "evaluations", "depth")})
        saved.require(adapter.flint.__version__ == lock["flint_versions"]["__version__"]
            and adapter.flint.__FLINT_VERSION__ == lock["flint_versions"]["__FLINT_VERSION__"], "observer_flint_version")
        pins = {name: ref for name, ref in lock["module_files"].items() if name == "flint" or name.startswith("flint.")}
        saved.require("flint" in pins, "observer_flint_source_pin")
        for name, reference in pins.items():
            module = importlib.import_module(name)
            saved.require(str(Path(module.__file__).resolve()) == reference["path"], "observer_flint_module_path")
            saved.pinned(reference, decode=False)
        return {"status": "bound_without_evaluation", "environment_lock": environment_reference,
            "python_version": sys.version, "flint_versions": lock["flint_versions"], "module_files": pins,
            "bits": adapter.bits, "evaluations": adapter.evaluations, "depth": adapter.depth,
            "ball_evaluations": 0, "integrations": 0, "history_reads": 0, "producer_calls": 0}
    except Exception as error:
        raise saved.ObservationStopped("runtime_binding_failure", "required_arb_preflight") from error


def validate(config_path):
    config = json.loads(Path(config_path).read_bytes())
    execution = config["execution"]
    limits = launcher.Limits.from_execution(execution)
    saved.require(0 < limits.wall_s <= 60 and limits.rss_bytes <= 1024**3
        and limits.artifact_bytes <= 32*MIB and config["observer_window_s"] == 20, "prefix_caps")
    saved.require(config["task_id"] == "task_53aa88c9ed09" and config["quiet_window_message"].startswith("msg_"),
                  "prefix_current_task_and_quiet_window")
    root = saved.absolute(execution["directory"])
    saved.require(not root.exists() and not root.is_relative_to(launcher.REPOSITORY), "fresh_prefix_directory")
    parent, case = producer_metadata(config["producer_plan"], config["case_id"])
    old = case["execution"]
    saved.require(not Path(old["directory"]).is_relative_to(root)
                  and not root.is_relative_to(Path(old["directory"])), "prefix_source_overlap")
    review = saved.pinned(config["flux_reuse_review"])
    saved.require(review["verdict"] == "accepted" and not review["blocking_findings"], "flux_reuse_review")
    sources = config["source_sha256"]
    check = saved.pinned(config["progress_source_check"])
    saved.require(check.get("accepted") is True and check.get("source_sha256") == sources, "prefix_source_review")
    for name, pin in sources.items():
        path = (launcher.REPOSITORY/name).resolve()
        saved.require(path.is_relative_to(launcher.REPOSITORY) and hashlib.sha256(path.read_bytes()).hexdigest() == pin,
                      "prefix_source_changed")
    saved.require(str(Path(__file__).resolve().relative_to(launcher.REPOSITORY)) in sources, "prefix_helper_pin")
    for path, pin in parent["source_freeze"]["source_sha256"].items():
        name = str(Path(path).relative_to(old["cwd"]))
        expected = review["source_sha256"][name] if name == "scripts/benchmarks/interval_observation.py" else pin
        saved.require(sources.get(name) == expected, "unreviewed_observer_source_delta")
    for name in ("producer_closure", "failed_observe_closure", "saved_observe_closure"):
        saved.require(saved.processes_absent(saved.pinned(config[name])), "old_process_still_present")
    result = saved.pinned(config["producer_result"])
    saved.require(Path(config["producer_result"]["path"]) == Path(old["directory"])/"NativeResult.json"
        and result["complete_run_window"] is True and result["history"]["closed"] is True
        and result["history"]["container_complete"] is True, "sealed_producer_metadata")
    request = saved.pinned(old["request"])
    saved.require((Path(old["directory"])/"NativeRequest.json").read_bytes() == saved.pinned(old["request"], decode=False),
                  "saved_request_changed")
    return config, limits, parent, case, request


def run(config_path):
    started = time.monotonic()
    config, limits, parent, case, request = validate(config_path)
    resource.setrlimit(resource.RLIMIT_CPU, (25, 28))
    backend = backend_preflight(request, parent["source_freeze"]["environment_lock"])
    execution, original = config["execution"], case["execution"]
    saved.require(all(os.environ.get(key) == "1" for key in launcher.THREAD_VARIABLES), "prefix_BLAS1")
    from scripts.benchmarks.coupled_device_prototype import SlabDefinition, AffineCoupledSlab, AffineVoltageMap
    from scripts.benchmarks.g2_compare import case_metadata, normalize_ida_sample
    from scripts.benchmarks.native_observation import SealedChargeObserver
    from scripts.benchmarks.native_readback import verify_history

    saved._blas_one()
    clock = launcher._pipeline_deadline(execution["supervision"], started, limits)
    model = AffineCoupledSlab(SlabDefinition(**request["numeric_packet"]["definition"]), case["mesh_intervals"])
    mapping = AffineVoltageMap(model, mapped_input_profile=request["voltage_lift_map"]["mapped_input_profile"])
    run_id = original["native_admission"]["operation_id"]
    binding = {"operation_id": config["operation_id"], "producer_operation_id": run_id,
        "producer_plan": config["producer_plan"], "producer_result": config["producer_result"],
        "request_sha256": route._digest(request), "source_sha256": config["source_sha256"],
        "source_snapshot_sha256": config["source_snapshot_sha256"],
        "integer_io_policy": original.get("integer_io_policy"), "progress_source_check": config["progress_source_check"],
        "arithmetic_backend_preflight": backend,
        "producer_metadata_validator": {"path": str(Path(original["cwd"])/"scripts/benchmarks/g2_route.py"),
            "sha256": parent["source_freeze"]["source_sha256"][str(Path(original["cwd"])/"scripts/benchmarks/g2_route.py")]}}
    cpu_start, wall_start = time.process_time(), time.monotonic()
    end = min(wall_start+20, launcher._refresh_pipeline_deadline(clock)-limits.cleanup_s)
    counts, stacks = Counter(), []
    pending_stop = None
    def checkpoint():
        nonlocal end
        counts["checkpoints"] += 1
        if pending_stop is not None:
            raise pending_stop
        end = min(end, launcher._refresh_pipeline_deadline(clock)-limits.cleanup_s)
        if Path(execution["supervision"]["cancel_path"]).exists():
            raise saved.ObservationStopped("cancelled", "prefix_cancel")
        if time.monotonic() >= end or time.process_time()-cpu_start >= 20:
            raise saved.ObservationStopped("deadline", "prefix_20s_window")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        if rss > limits.rss_bytes:
            raise saved.ObservationStopped("resource_limit", "prefix_RSS")
    def sample(number, frame):
        nonlocal pending_stop
        # Finish the existing bounded record transaction before raising our
        # cooperative stop. The outer watchdog and 28 s kernel CPU cap remain
        # in force; every callback checks the deadline again after publication.
        if progress.saving:
            if number == signal.SIGXCPU:
                pending_stop = saved.ObservationStopped("resource_limit", "prefix_CPU_soft_limit")
            elif time.monotonic() >= end or time.process_time()-cpu_start >= 20:
                pending_stop = saved.ObservationStopped("deadline", "prefix_20s_window")
            return
        if number == signal.SIGXCPU:
            raise saved.ObservationStopped("resource_limit", "prefix_CPU_soft_limit")
        if len(stacks) < 40:
            names = []
            while frame is not None and len(names) < 12:
                names.append({"file": Path(frame.f_code.co_filename).name[:120],
                              "function": frame.f_code.co_name[:100], "line": frame.f_lineno})
                frame = frame.f_back
            stacks.append({"elapsed_s": time.monotonic()-wall_start, "frames": names})
        checkpoint()
    profiler, stop, result = cProfile.Profile(), None, None
    with launcher.inherited_channel(parent["artifacts"]["channel_directory"], execution["channel"]), saved.cancellation():
        progress = PrefixProgress(execution["directory"], binding=binding, artifact_bytes=limits.artifact_bytes,
                                  max_record_bytes=original["max_record_bytes"], integer_io_policy=original.get("integer_io_policy"))
        def emit(row):
            checkpoint()
            progress.append(row)
            checkpoint()
        def retain(tag, sample, certificate):
            emit({"kind": "sealed_g2_sample", "request_sha256": binding["request_sha256"],
                  "evidence": normalize_ida_sample(request, tag, sample, certificate)})
        gate = parent["source_freeze"]["frozen_gate_spec"]
        observer = SealedChargeObserver(mapping, request, emit, checkpoint,
            t1_context={"spec": saved.pinned(gate), "spec_sha256": gate["sha256"], "run_id": run_id, "case_id": case["id"]})
        metadata = case_metadata(request, case["id"])
        metadata["run_id"] = run_id
        emit({"kind": "sealed_g2_context", "request_sha256": binding["request_sha256"], "evidence": metadata})
        previous = {sig: signal.signal(sig, sample) for sig in (signal.SIGALRM, signal.SIGXCPU)}
        signal.setitimer(signal.ITIMER_REAL, .5, .5)
        try:
            profiler.enable()
            result = verify_history(Path(original["directory"]), max_record_bytes=original["max_record_bytes"],
                max_records=original["max_records"], max_logical_bytes=parent["resources"]["reader_logical_byte_cap"],
                deadline_monotonic=end+.25, integer_io_policy=original.get("integer_io_policy"),
                execution_admission=original["native_admission"], execution_provenance=original.get("execution_provenance"),
                charge_observer=observer, sample_sink=retain)
        except BaseException as error:
            stop = _stop(error)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            profiler.disable()
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        outcome = progress.finish_prefix(stop=stop, reader_returned=result is not None)
        profiler.dump_stats(progress.directory/"Calls.pstats")
        entries = [{"file": Path(filename).name, "line": line, "function": function[:160],
                    "calls": total, "self_s": own, "cumulative_s": cumulative}
                   for (filename, line, function), (_, total, own, cumulative, _) in pstats.Stats(profiler).stats.items()]
        profile = {"observer_wall_s": time.monotonic()-wall_start, "observer_CPU_s": time.process_time()-cpu_start,
            "whole_process_s": time.monotonic()-started, "counts": dict(counts), "stack_samples": stacks,
            "observer_completed_intervals": observer.intervals, "observer_completed_t1_pieces": observer.t1_pieces,
            "top_self": sorted(entries, key=lambda row: row["self_s"], reverse=True)[:30],
            "top_cumulative": sorted(entries, key=lambda row: row["cumulative_s"], reverse=True)[:30],
            "scope": "real saved prefix including exact output serialization; no baseline speedup inference",
            "scientific_qualification": False, "producer_calls": 0}
        _publish(progress.directory/"Profile.json", profile)
    return {"status": outcome["status"], "stop": stop, "records": outcome["records"],
            "writer_container_complete": outcome["output_prefix"]["writer_container_complete"],
            "observer_wall_s": profile["observer_wall_s"], "producer_calls": 0, "scientific_qualification": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", type=Path)
    mode.add_argument("--read", type=Path)
    args = parser.parse_args(argv)
    try:
        result = read_progress(args.read) if args.read is not None else run(args.config)
    except BaseException as error:
        print(json.dumps({"status": "refused_or_failed", **_stop(error)}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
