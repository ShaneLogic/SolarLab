"""Thin G2 RunPlan entry; never promote a candidate definition into authority.

The materialized plan supplies each case's execution.request (path/bytes/SHA),
directory, native_admission, wall_s, max_record_bytes and integer_io_policy.
source_freeze supplies the immutable source_sha256 map and environment_lock
descriptor. Existing producer, recording and independent reader APIs own the
numerical work. Commander owns admission, channel reservations and the outer
process watchdog. No candidate T1 constants are implemented by this entry.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time


METHODS = {"certificate": "solarlab.certificate-failure-policy.v1",
           "transfer": "solarlab.reference-state-transfer.v2", "observer": "solarlab.sealed-charge-observation.v1"}
RUN_PLAN_SCHEMA = "solarlab.g2-route.run-plan.v3"
T1_SPEC_SCHEMA = "solarlab.g2-route.t1-gatespec.v3"
EVIDENCE_METHOD_SCHEMA = "solarlab.g2-reference-method.v1"
# A run binds frozen definitions. Evaluation separately binds reviewed code.
EVALUATOR_FILES = ("g2_route.py", "t1_gate.py", "native_readback.py", "native_observation.py", "interval_observation.py")


def _require(value, reason):
    if not value:
        raise ValueError("g2_route_" + reason)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _read(path, limit=4*1024*1024):
    with Path(path).open("rb") as stream:
        raw = stream.read(limit+1)
    _require(len(raw) <= limit, "metadata_size")
    return raw


def _pinned(reference, *, decode=True):
    _require(type(reference) is dict and {"path", "sha256"} <= reference.keys(), "unresolved_reference")
    raw = _read(reference["path"])
    _require(hashlib.sha256(raw).hexdigest() == reference["sha256"]
             and ("bytes" not in reference or reference["bytes"] == len(raw)), "reference_changed")
    return json.loads(raw) if decode else raw


def _labelled_samples(request):
    """The V2 ordered tags of an actual request, including both event sides."""
    segments, sampling = request["segments"], request["observation_times"]
    _require(set(sampling) == {s["id"] for s in segments}, "sample_segments")
    result = []
    for ordinal, segment in enumerate(segments):
        times = sampling[segment["id"]]
        _require(type(times) is list and len(times) >= 2
                 and all(type(t) in (int, float) and math.isfinite(t) for t in times)
                 and times[0] == segment["start"] and times[-1] == segment["end"]
                 and all(a < b for a, b in zip(times, times[1:])), "sample_times")
        for index, time_s in enumerate(times):
            result.append({"segment": segment["id"], "index": index, "time_hex": float(time_s).hex(),
                "event_side": "right" if index == 0 else "left" if index == len(times)-1 else "interior",
                "protocol_initial": ordinal == index == 0,
                "protocol_terminal": ordinal == len(segments)-1 and index == len(times)-1})
    return result


def _validate_supervision(plan, execution):
    """Bind the outer launcher's caps and receipt paths; never launch it."""
    value = execution.get("supervision")
    fields = {"poll_s", "term_grace_s", "kill_wait_s", "receipt_reserve_s",
              "rss_bytes", "artifact_bytes", "closure_path", "cancel_path"}
    _require(type(value) is dict and set(value) == fields, "supervision_fields")
    for name in ("poll_s", "term_grace_s", "kill_wait_s", "receipt_reserve_s"):
        _require(type(value[name]) in (int, float) and math.isfinite(value[name]) and value[name] > 0,
                 "supervision_" + name)
    for name, cap in (("rss_bytes", plan["resources"]["rss_bytes_per_case"]),
                      ("artifact_bytes", plan["resources"]["combined_persisted_case_bytes"])):
        _require(type(value[name]) is int and 0 < value[name] <= cap, "supervision_" + name)
    for name in ("closure_path", "cancel_path"):
        path = value[name]
        _require(type(path) is str and Path(path).is_absolute() and str(Path(path).resolve()) == path,
                 "supervision_" + name)
    _require(value["closure_path"] != value["cancel_path"], "supervision_distinct_paths")
    wall = execution.get("wall_s")
    _require(type(wall) in (int, float) and math.isfinite(wall) and wall > 0
             and value["receipt_reserve_s"] + value["term_grace_s"] + value["kill_wait_s"] < wall,
             "supervision_wall_reserve")
    return value


def _resolved_descriptor(value):
    return (type(value) is dict and type(value.get("path")) is str
            and type(value.get("bytes")) is int and value["bytes"] >= 0
            and type(value.get("sha256")) is str and len(value["sha256"]) == 64
            and all(c in "0123456789abcdef" for c in value["sha256"]))


def _frozen_definition(value, schema, version):
    """Verify Commander's copy against the exact reviewed author document."""
    _require(type(value) is dict and value.get("schema") == schema and type(value.get("version")) is int
             and value["version"] == version and value.get("status") == "frozen", "frozen_definition_schema")
    freeze = value.get("freeze")
    keys = {"frozen_at", "frozen_by", "reviewed_path", "reviewed_sha256", "reviewed_bytes", "review_records", "user_records"}
    explanations = {"notes", "plan_rule"}
    _require(type(freeze) is dict and keys <= set(freeze) <= keys | explanations
             and type(freeze["frozen_at"]) is str and freeze["frozen_at"]
             and type(freeze["frozen_by"]) in (str, dict) and bool(freeze["frozen_by"]), "definition_freeze_record")
    _require(all(type(freeze[k]) is str or type(freeze[k]) is list
                 and all(type(item) is str for item in freeze[k])
                 for k in explanations & set(freeze)), "definition_freeze_explanation_type")
    reviewed_ref = {"path": freeze["reviewed_path"], "sha256": freeze["reviewed_sha256"], "bytes": freeze["reviewed_bytes"]}
    _require(_resolved_descriptor(reviewed_ref), "definition_reviewed_descriptor")
    reviewed = _pinned(reviewed_ref)
    _require(type(reviewed) is dict and _digest({k: v for k, v in value.items() if k not in {"status", "freeze"}})
             == _digest({k: v for k, v in reviewed.items() if k != "status"}), "definition_reviewed_body_changed")
    _require(type(freeze["review_records"]) is list and freeze["review_records"]
             and type(freeze["user_records"]) is list and freeze["user_records"], "definition_review_user_records")
    for review in freeze["review_records"]:
        _require(type(review) is dict and set(review) == {"path", "sha256"}, "definition_review_record_fields")
        _pinned(review, decode=False)
    return value


def _definition_bindings(plan, blockers, *, include_plan=True):
    """Pending author documents cannot supply frozen run authority."""
    bound = {}
    for key in ("plan", "gate_spec", "evidence_method_contract"):
        if key == "plan" and not include_plan:
            continue
        ref = plan["definitions"].get(key)
        if not _resolved_descriptor(ref):
            _require(ref is None or type(ref) is dict and
                     (ref.get("status", "").startswith("pending") or ref.get("sha256") is None),
                     "invalid_definition_descriptor_" + key)
            blockers.append("unresolved_definition_" + key)
            continue
        value = _pinned(ref, decode=key != "plan")
        if key != "plan":
            schema = T1_SPEC_SCHEMA if key == "gate_spec" else EVIDENCE_METHOD_SCHEMA
            _require(value.get("schema") == schema
                     and ref.get("schema", schema) == schema, "definition_schema_" + key)
            if value.get("status") != "frozen":
                blockers.append("definition_not_frozen_" + key)
            else:
                _frozen_definition(value, schema, 3 if key == "gate_spec" else 1)
        bound[key] = value
    return bound


def _evaluation_blockers(plan):
    """No evaluator code is imported before its separate source review."""
    freeze = plan.get("evaluation_freeze")
    if type(freeze) is not dict or freeze.get("status") != "admitted":
        return ["evaluation_not_admitted"]
    if not _resolved_descriptor(freeze.get("review_record")) or type(freeze.get("source_sha256")) is not dict:
        return ["evaluation_review_or_source_pending"]
    review = _pinned(freeze["review_record"])
    _require(type(review) is dict, "evaluation_review_root")
    sources = freeze["source_sha256"]
    root = Path(__file__).resolve().parent
    _require(all(str(root / name) in sources for name in EVALUATOR_FILES), "evaluator_required_sources")
    _require(review.get("accepted") is True and review.get("evaluation_source_review") is True
             and review.get("source_sha256") == sources
             and review.get("gate_spec_sha256") == plan["definitions"]["gate_spec"]["sha256"]
             and review.get("evidence_method_contract_sha256") == plan["definitions"]["evidence_method_contract"]["sha256"],
             "evaluation_review_binding")
    for path, expected in sources.items():
        _require(hashlib.sha256(_read(path, 16*1024*1024)).hexdigest() == expected, "evaluator_source_changed")
    return []


def validate(plan_path, *, executable=False, case_id=None, purpose="run"):
    """Metadata-only validation, including all labelled samples and case slots."""
    raw = _read(plan_path)
    plan = json.loads(raw)
    _require(purpose in {"run", "evaluate"}, "admission_purpose")
    _require(plan.get("schema") == RUN_PLAN_SCHEMA and type(plan.get("version")) is int
             and plan["version"] == 3, "plan_schema_V3_required")
    resources = plan["resources"]
    _require(resources["WS1_channels"] == 4 and resources["threads_per_case"] == 1
             and resources["rss_bytes_per_case"] <= 4*1024**3
             and resources["WS1_max_declared_rss_bytes"] <= 16*1024**3
             and resources["combined_persisted_case_bytes"] <= 4*1024**3
             and resources["reader_logical_byte_cap"] <= 32*1024**3, "four_channel_resource_contract")
    labels = plan["protocol"]["requested_sample_records"]
    _require(len(labels) == 131 and len({r["time_hex"] for r in labels}) == 129, "labelled_sample_contract")
    segments = plan["protocol"]["segments"]
    _require([s["id"] for s in segments] == ["dark_equilibrium_hold", "voltage_ramp", "slow_state_hold"]
             and [s["start"] for s in segments] == [0., .1, .2]
             and [s["end"] for s in segments] == [.1, .2, 9.2]
             and [s["voltage"] for s in segments] == [[0, 0], [0, .1], [.1, .1]]
             and all(s["photons"] == [0, 0] for s in segments)
             and plan["protocol"]["max_step_s"] == .02, "declared_B_protocol")
    expected_ids = {f"B_N8_{profile}_{stage}" for profile in ("Coarse", "Main", "Strict") for stage in ("Probe", "Full")}
    expected_ids.add("B_N16_Strict_Full")
    cases = {c["id"]: c for c in plan["cases"]}
    _require(set(cases) == expected_ids and len(plan["cases"]) == len(cases), "seven_declared_cases")
    from scripts.benchmarks.physical_tolerance import PROFILES
    ancestors, case_stages = {}, {}
    for name, case in cases.items():
        declared_stage = {"Probe": "probe", "Full": "full"}[name.rsplit("_", 1)[1]]
        _require(case.get("stage") in ("probe", "full") and case["stage"] == declared_stage,
                 "case_id_stage_binding")
        case_stages[name] = declared_stage
        probe = declared_stage == "probe"
        profile = name.split("_")[2]
        n = 16 if "N16" in name else 8
        count = sum(row["segment"] != "slow_state_hold" for row in labels) if probe else len(labels)
        _require(all(type(case[k]) is int for k in ("mesh_intervals", "nodes", "coordinates", "sample_records", "native_execution_count"))
                 and case["mesh_intervals"] == n and case["nodes"] == n+1 and case["coordinates"] == 5*(n+1)
                 and tuple(case["profile"][k] for k in ("r", "alpha")) == PROFILES[("Coarse", "Main", "Strict").index(profile)]
                 and case["sample_records"] == count == (66 if probe else 131)
                 and case["horizon_s"] == [0, .2 if probe else 9.2]
                 and case["native_execution_count"] == 1, "case_geometry_profile_or_labels")
        reference = case["input_ancestor"]
        key = (reference["path"], reference["sha256"])
        if key not in ancestors:
            ancestors[key] = _pinned(reference)
        ancestor = ancestors[key]
        _require(ancestor["numeric_packet"]["nodes"] == case["nodes"]
                 and ancestor["physical_tolerance_policy"]["profile"] == case["profile"], "input_ancestor_case")
        _require(_digest(ancestor["segments"]) == _digest(segments)
                 and _digest(_labelled_samples(ancestor)) == _digest(labels), "input_ancestor_ordered_sample_tags")
        refresh = case["reference_refresh"]
        _require(type(refresh.get("enabled")) is bool and refresh["enabled"] is (not probe)
                 and refresh.get("record_schema") == "solarlab.reference-state-transfer.v2"
                 and refresh.get("max_refreshes") == (0 if probe else 1)
                 and refresh.get("schedule_ref") == "/protocol/reference_refresh_schedule", "case_refresh_scope")
        if "supervision" in case.get("execution", {}):
            _validate_supervision(plan, case["execution"])
    schedule = plan["protocol"]["reference_refresh_schedule"]
    _require(schedule.get("time_hex") == "0x1.7a4c1c85a8b06p-2"
             and schedule.get("segment_id") == "slow_state_hold" and schedule.get("physical_segment_ordinal") == 3
             and schedule.get("max_refreshes") == 1 and schedule.get("native_api") == "init_step"
             and schedule.get("record_schema") == "solarlab.reference-state-transfer.v2"
             and not any("sha256" in key or "map_identity" in key for key in schedule), "shared_refresh_schedule")
    _require(case_id is None or case_id in cases, "unknown_case")
    for stage in plan["stages"]:
        if "max_channels" in stage:
            _require(0 < stage["max_channels"] <= 4 and len(stage.get("parallel_cases", [])) <= 4, "stage_channel_cap")
    # These definitions are inputs, never imports from historical source_path.
    freeze = plan["source_freeze"]
    blockers = []
    definitions = _definition_bindings(plan, blockers)
    if plan.get("status") != "admitted" or plan["authority"].get("scientific_execution_authorized") is not True:
        blockers.append("run_plan_not_admitted")
    if plan["implementation"].get("available_now") is not True:
        blockers.append("implementation_not_reviewed")
    for key in ("future_execution_commit", "environment_lock", "review_record", "frozen_gate_spec",
                "frozen_evidence_method_contract"):
        if not freeze.get(key):
            blockers.append("unresolved_"+key)
    if not blockers:
        commit = freeze["future_execution_commit"]
        _require(type(commit) is str and len(commit) == 40 and all(c in "0123456789abcdef" for c in commit), "execution_commit")
        gate, review = _pinned(freeze["frozen_gate_spec"]), _pinned(freeze["review_record"])
        method = _pinned(freeze["frozen_evidence_method_contract"])
        _require(gate == definitions["gate_spec"] and method == definitions["evidence_method_contract"]
                 and freeze["frozen_gate_spec"]["sha256"] == plan["definitions"]["gate_spec"]["sha256"]
                 and freeze["frozen_evidence_method_contract"]["sha256"] == plan["definitions"]["evidence_method_contract"]["sha256"],
                 "frozen_definition_binding")
        _require(freeze.get("method_versions") == METHODS and review.get("accepted") is True
                 and review.get("execution_commit") == commit and review.get("method_versions") == METHODS
                 and review.get("gate_spec_sha256") == freeze["frozen_gate_spec"]["sha256"]
                 and review.get("evidence_method_contract_sha256") == freeze["frozen_evidence_method_contract"]["sha256"]
                 and review.get("source_sha256") == freeze.get("source_sha256"), "source_review_binding")
        if purpose == "run":
            _require(str(Path(__file__).resolve()) in freeze["source_sha256"], "actual_entry_not_source_bound")
        for path, expected in freeze["source_sha256"].items():
            _require(hashlib.sha256(_read(path, 16*1024*1024)).hexdigest() == expected, "actual_source_changed")
        _pinned(freeze["environment_lock"])
        for case in (cases.values() if case_id is None else [cases[case_id]]):
            stage = case_stages[case["id"]]
            execution = case.get("execution")
            _require(type(execution) is dict, "case_execution_unresolved")
            _validate_supervision(plan, execution)
            request = _pinned(execution["request"])
            from scripts.benchmarks.physical_tolerance import validate_certificate_policy, run_segment_count
            policy = validate_certificate_policy(request)
            _require(policy is not None and policy["observation"] == "sealed-history"
                     and policy["T1"] == {"schema": T1_SPEC_SCHEMA, "sha256": freeze["frozen_gate_spec"]["sha256"]}
                     and request["physical_tolerance_policy"]["profile"] == case["profile"]
                     and request["numeric_packet"]["nodes"] == case["nodes"]
                     and len(request["z0"]) == case["coordinates"] and request["controls"]["max_step"] == .02
                     and request["voltage_lift_map"].get("mapped_input_words") == 12
                     and run_segment_count(request) == (2 if stage == "probe" else 3), "effective_case_input")
            _require(_digest(request["segments"]) == _digest(segments)
                     and _digest(_labelled_samples(request)) == _digest(labels), "effective_ordered_sample_tags")
            from scripts.benchmarks.g2_route_bindings import effective_issues
            ancestor = ancestors[(case["input_ancestor"]["path"], case["input_ancestor"]["sha256"])]
            _require(not effective_issues(request, ancestor, case, plan), "effective_request_derivation")
            _require(execution["effective_input_digest"] == _digest(request)
                     and review.get("request_sha256s", {}).get(case["id"]) == _digest(request), "effective_input_review")
            _require(type(execution["channel"]) is int and 0 <= execution["channel"] < 4
                     and "Execution12" not in Path(execution["directory"]).parts, "case_channel_or_namespace")
            maximum = resources["probe_wall_cap_s"] if stage == "probe" else resources["full_wall_cap_max_s"]
            _require(type(execution["wall_s"]) in (int, float) and math.isfinite(execution["wall_s"])
                     and 0 < execution["wall_s"] <= maximum, "case_wall_cap")
            if stage == "full":
                forecast = execution["forecast"]
                _require(forecast.get("feasible") is True and forecast.get("slow_hold_model")
                         and execution["wall_s"] == math.ceil(3*forecast["estimated_s"]), "full_forecast_required")
    try:
        evaluation = _evaluation_blockers(plan)
    except (ValueError, KeyError, OSError) as error:
        if purpose == "evaluate":
            raise
        evaluation = [str(error)]  # A bad evaluator binding cannot authorize evaluation or block a producer.
    selected = blockers + (evaluation if purpose == "evaluate" else [])
    if executable:
        _require(not selected, ",".join(selected))
    return {"plan": plan, "plan_sha256": hashlib.sha256(raw).hexdigest(), "ready": not selected,
            "run_ready": not blockers, "evaluation_ready": not blockers and not evaluation,
            "blockers": selected, "evaluation_blockers": evaluation,
            "case_ids": list(cases), "sample_records": {k: v["sample_records"] for k, v in cases.items()}}


def _runtime(context, case_id):
    plan = context["plan"]
    case = next(c for c in plan["cases"] if c["id"] == case_id)
    execution = case["execution"]
    deadline = time.perf_counter()+execution["wall_s"]
    lock = _pinned(plan["source_freeze"]["environment_lock"])
    _require(lock.get("python_executable") == str(Path(sys.executable).resolve())
             and lock.get("python_version") == sys.version, "environment_lock_mismatch")
    _require(all(os.environ.get(k) == v == "1" for k, v in plan["resources"]["thread_environment"].items()), "thread_environment")
    # Same actual Accelerate getter as the maintained native caller; no thread pool.
    import ctypes
    blas = ctypes.CDLL("/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/libBLAS.dylib")
    blas.BLASSetThreading.argtypes = [ctypes.c_int]
    blas.BLASSetThreading.restype = ctypes.c_int
    blas.BLASGetThreading.restype = ctypes.c_int
    blas.BLASSetThreading(1)
    _require(blas.BLASGetThreading() == 1, "actual_BLAS_thread_mode")
    request = _pinned(execution["request"])
    from scripts.benchmarks.coupled_device_prototype import SlabDefinition, AffineCoupledSlab, AffineVoltageMap, validate_voltage_lift_native_request, ProtocolSegment
    definition = SlabDefinition(**request["numeric_packet"]["definition"])
    model = AffineCoupledSlab(definition, case["mesh_intervals"])
    mapping = AffineVoltageMap(model, mapped_input_profile=request["voltage_lift_map"]["mapped_input_profile"])
    segments = tuple(ProtocolSegment(**s) for s in request["segments"])
    validate_voltage_lift_native_request(mapping, segments, request)
    def checkpoint():
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        _require(time.perf_counter() < deadline and rss <= plan["resources"]["rss_bytes_per_case"], "owned_wall_or_RSS_cap")
    return case, request, mapping, segments, checkpoint


@contextmanager
def _channel(plan, execution):
    """Use the launcher's inherited slot lock through producer and observer."""
    import fcntl
    root = Path(plan["artifacts"]["channel_directory"])
    _require(root.is_dir(), "commander_channel_directory_required")
    slot = root / (str(execution["channel"]) + ".lock")
    inherited = os.environ.get("SOLARLAB_G2_ROUTE_CHANNEL_FD")
    if inherited is not None:
        _require(inherited.isdecimal(), "inherited_channel_fd")
        try:
            held, expected = os.fstat(int(inherited)), slot.stat()
        except (OSError, OverflowError) as error:
            raise ValueError("g2_route_inherited_channel_fd") from error
        _require((held.st_dev, held.st_ino) == (expected.st_dev, expected.st_ino), "inherited_channel_identity")
        yield
        # Do not close or unlock the shared open-file description. The
        # launcher releases it only after confirming all owned processes exit.
        return
    _require("supervision" not in execution and plan.get("status") != "admitted",
             "supervised_channel_fd_required")
    with slot.open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _observe(context, case, request, mapping, checkpoint):
    from scripts.benchmarks.native_history import HistoryWriter, integer_json_dumps
    from scripts.benchmarks.native_readback import verify_history
    from scripts.benchmarks.native_observation import SealedChargeObserver
    from scripts.benchmarks.native_runner import output_bytes
    execution, resources = case["execution"], context["plan"]["resources"]
    folder = Path(execution["directory"])
    remaining = resources["combined_persisted_case_bytes"]-output_bytes(folder, ())
    _require(remaining > 2*1048576, "combined_output_cap")
    target = folder / "Observation"
    target.mkdir(exist_ok=False)
    writer = HistoryWriter(target / "Charge.jsonl.gz", encoding="gzip", total_output_bytes=remaining,
                           max_record_bytes=execution["max_record_bytes"], integer_io_policy=execution.get("integer_io_policy"))
    try:
        run_id = execution["native_admission"].get("operation_id")
        _require(type(run_id) is str and run_id, "observation_operation_identity")
        gate_ref = context["plan"]["source_freeze"]["frozen_gate_spec"]
        observer = SealedChargeObserver(mapping, request, writer.append, checkpoint,
            t1_context={"spec":_pinned(gate_ref),"spec_sha256":gate_ref["sha256"],"run_id":run_id,"case_id":case["id"]})
        result = verify_history(folder, max_record_bytes=execution["max_record_bytes"],
            max_logical_bytes=resources["reader_logical_byte_cap"], max_records=execution["max_records"],
            deadline_monotonic=time.perf_counter()+execution["wall_s"], integer_io_policy=execution.get("integer_io_policy"),
            execution_admission=execution["native_admission"], execution_provenance=execution.get("execution_provenance"),
            charge_observer=observer)
        checkpoint()
        writer.finish()
        result.update(plan_sha256=context["plan_sha256"], case_id=case["id"], methods=METHODS,
                      observed_history={"path": "Observation/Charge.jsonl.gz", "records": writer.records,
                          "logical_sha256": writer.logical_sha256, "logical_bytes": writer.logical_bytes,
                          "encoded_bytes": writer.encoded_bytes, "closed": writer.closed})
        raw = (integer_json_dumps(result, execution.get("integer_io_policy"), indent=2)+"\n").encode()
        _require(output_bytes(folder, ())+len(raw) <= resources["combined_persisted_case_bytes"], "combined_output_cap")
        with (folder / "RunResult.json").open("xb") as stream:
            stream.write(raw)
        return result
    except BaseException as error:
        try:
            writer.abort(error)
            writer.publish_first_failure({"exception": type(error).__name__, "reason": str(error)[:2048]})
        except BaseException as secondary:
            error.add_note("observer closeout: "+type(secondary).__name__)
        raise


def run(plan_path, case_id):
    context = validate(plan_path, executable=True, case_id=case_id)
    execution = next(c for c in context["plan"]["cases"] if c["id"] == case_id)["execution"]
    with _channel(context["plan"], execution):
        case, request, mapping, segments, checkpoint = _runtime(context, case_id)
        from scripts.benchmarks.native_runner import run_recorded
        from scripts.benchmarks.coupled_device_prototype import run_voltage_lift_native_pilot
        folder = Path(execution["directory"])
        folder.mkdir(exist_ok=False)
        (folder / "NativeRequest.json").write_bytes(_read(execution["request"]["path"]))
        result = run_recorded(folder, lambda writer: run_voltage_lift_native_pilot(
            mapping, segments, request, execution["native_admission"], writer,
            execution_provenance=execution.get("execution_provenance")),
            total_output_bytes=context["plan"]["resources"]["combined_persisted_case_bytes"],
            input_names=("NativeRequest.json",), max_record_bytes=execution["max_record_bytes"],
            integer_io_policy=execution.get("integer_io_policy"))
        checkpoint()
        return result  # Sealed producer evidence; observe has separate admission.


def observe(plan_path, case_id):
    context = validate(plan_path, executable=True, case_id=case_id, purpose="evaluate")
    execution = next(c for c in context["plan"]["cases"] if c["id"] == case_id)["execution"]
    with _channel(context["plan"], execution):
        case, request, mapping, _, checkpoint = _runtime(context, case_id)
        return _observe(context, case, request, mapping, checkpoint)


def compare(plan_path):
    context = validate(plan_path, executable=True, purpose="evaluate")
    return {"plan_sha256": context["plan_sha256"], "scientific_status": "not_evaluated",
            "scientific_pass": False, "reason": "comparison implementation is a separate future task"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate", "run", "observe", "compare"))
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--case")
    args = parser.parse_args(argv)
    try:
        if args.action == "validate":
            result = validate(args.plan)
            result.pop("plan")
        elif args.action in {"run", "observe"}:
            _require(args.case is not None, "case_required")
            result = globals()[args.action](args.plan, args.case)
        else:
            result = compare(args.plan)
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    except (ValueError, KeyError, OSError) as error:
        print(json.dumps({"status": "refused", "reason": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
