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
EVALUATOR_FILES = ("g2_route.py", "t1_gate.py", "native_readback.py", "native_observation.py", "interval_observation.py",
                   "g2_compare.py", "qualification_comparison.py", "g2_reference_be.py", "g2_reference_error.py")


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
    pipeline_fields = {"pipeline_started_unix_ns", "pipeline_wall_s"}
    _require(type(value) is dict and set(value) in (fields, fields | pipeline_fields), "supervision_fields")
    if pipeline_fields <= set(value):
        start = value["pipeline_started_unix_ns"]
        _require(type(start) is int and start >= 0, "supervision_pipeline_started_unix_ns")
        pipeline_wall = value["pipeline_wall_s"]
        try:
            valid_wall = type(pipeline_wall) in (int, float) and math.isfinite(pipeline_wall) and pipeline_wall > 0
        except OverflowError:
            valid_wall = False
        _require(valid_wall, "supervision_pipeline_wall_s")
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


def _admitted_cases(plan, cases):
    """A phase grants exactly its declared cases; dormant execution is null."""
    admission = plan.get("admission")
    if admission is None:
        selected = []
    else:
        _require(type(admission) is dict and set(admission) == {"stage", "admitted_case_ids"}
                 and type(admission["stage"]) is str and admission["stage"] in {"probes", "full"}, "admission_fields")
        selected = admission["admitted_case_ids"]
        expected = {name for name, case in cases.items()
                    if case["stage"] == ("probe" if admission["stage"] == "probes" else "full")}
        _require(type(selected) is list and all(type(name) is str for name in selected)
                 and len(selected) == len(set(selected)) and set(selected) == expected,
                 "admission_case_ids")
    for name, case in cases.items():
        if name not in selected:
            _require((admission is None or "execution" in case) and case.get("execution") is None,
                     "unadmitted_case_execution:" + name)
    return selected


def _positive_pin(reference, reason):
    _require(_resolved_descriptor(reference) and reference["bytes"] > 0
             and Path(reference["path"]).is_absolute(), reason)
    value = _pinned(reference)
    _require(type(value) is dict, reason + "_root")
    return value


def _probe_runtime_receipt(receipt, execution, receipt_pin, producer_pin, case_id):
    """Verify the existing launcher's saved facts without inspecting processes."""
    from scripts.benchmarks.g2_route_launcher import CHANNEL_PROTOCOL, Limits, REPOSITORY
    limits = vars(Limits.from_execution(execution))
    command = execution.get("command")
    if command is None:
        command = [str(REPOSITORY / ".venv/bin/python"), "-m", "scripts.benchmarks.g2_route", "run",
                   "--plan", str(Path(producer_pin["path"]).resolve()), "--case", case_id]
    _require(receipt.get("schema") == "solarlab.g2-route.launcher-result.v1"
             and receipt.get("case_id") == case_id and receipt.get("plan_sha256") == producer_pin["sha256"]
             and type(receipt.get("channel")) is int and receipt["channel"] == execution["channel"]
             and receipt.get("channel_protocol") == CHANNEL_PROTOCOL
             and receipt.get("command") == command and receipt.get("cwd") == str(REPOSITORY),
             "full_probe_runtime_binding")
    _require(receipt.get("process_outcome") == receipt.get("termination_reason") == "completed"
             and type(receipt.get("exit_code")) is int and receipt["exit_code"] == 0
             and all(key in receipt and receipt[key] is None for key in ("exit_signal", "cancellation_signal", "error"))
             and receipt.get("signals_sent") == []
             and receipt.get("artifact_limit_exceeded", False) is False
             and receipt.get("channel_reusable", True) is True
             and receipt.get("quarantine_path") is None, "full_probe_runtime_outcome")
    observed, closure = receipt.get("observed_pids"), receipt.get("closure")
    _require(all(type(receipt.get(key)) is int and receipt[key] > 0 for key in ("pid", "pgid", "session_id"))
             and receipt["pid"] == receipt["pgid"] == receipt["session_id"]
             and type(observed) is list and all(type(pid) is int and pid > 0 for pid in observed)
             and observed == sorted(set(observed)) and receipt["pid"] in observed
             and type(closure) is dict and closure.get("schema") == "solarlab.owned-tool-process-closure.v1"
             and type(closure.get("pgid")) is int and closure["pgid"] == receipt["pgid"]
             and type(closure.get("after_tool_exit_code")) is int and closure["after_tool_exit_code"] == 0
             and closure.get("process_group_absent") is True
             and closure.get("all_named_owned_processes_and_groups_absent") is True
             and closure.get("remaining_group_members") == [] and closure.get("remaining_owned_processes") == []
             and closure.get("error") is None, "full_probe_runtime_closure")
    checks = closure.get("checks")
    _require(type(checks) is list and all(type(row) is dict and type(row.get("pid")) is int
                 and row.get("pid_absent") is True and row.get("same_id_process_group_absent") is True for row in checks)
             and [row["pid"] for row in checks] == observed, "full_probe_runtime_closure_checks")
    recorded_limits = receipt.get("limits")
    _require(type(recorded_limits) is dict and set(recorded_limits) == set(limits)
             and all(type(recorded_limits[key]) is int if key in {"rss_bytes", "artifact_bytes"}
                     else type(recorded_limits[key]) in (int, float) for key in limits)
             and all(recorded_limits[key] == value for key, value in limits.items()), "full_probe_runtime_limits")
    wall = receipt.get("wall_s")
    _require(type(wall) in (int, float) and 0 < wall < limits["wall_s"] <= 3600
             and type(receipt.get("receipt_initial_write_s")) in (int, float)
             and 0 <= receipt["receipt_initial_write_s"] <= wall, "full_probe_runtime_wall")
    if "wall_through_receipt_s" in receipt:
        _require(type(receipt["wall_through_receipt_s"]) in (int, float)
                 and wall <= receipt["wall_through_receipt_s"] < limits["wall_s"], "full_probe_runtime_complete_wall")
    _require(type(receipt.get("peak_group_rss_bytes")) is int
             and 0 <= receipt["peak_group_rss_bytes"] <= limits["rss_bytes"]
             and all(type(receipt.get(key)) is int and 0 <= receipt[key] <= limits["artifact_bytes"] for key in
                     ("peak_artifact_bytes", "artifact_bytes_before_receipt", "receipt_bytes", "artifact_bytes"))
             and receipt["receipt_bytes"] == receipt_pin["bytes"]
             and receipt["artifact_bytes"] == receipt["artifact_bytes_before_receipt"] + receipt["receipt_bytes"]
             and receipt["peak_artifact_bytes"] >= receipt["artifact_bytes"], "full_probe_runtime_resources")
    last, rss_peak, artifact_peak = (receipt.get(key) for key in ("last_samples", "rss_peak_sample", "artifact_peak_sample"))
    _require(type(last) is list and len(last) <= 2
             and ((not last and rss_peak is artifact_peak is None and receipt["peak_group_rss_bytes"] == 0)
                  or (last and type(rss_peak) is dict and type(artifact_peak) is dict)), "full_probe_runtime_samples")
    for sample in last + ([] if rss_peak is None else [rss_peak, artifact_peak]):
        _require(type(sample) is dict and type(sample.get("elapsed_s")) in (int, float)
                 and 0 <= sample["elapsed_s"] <= wall
                 and type(sample.get("rss_bytes")) is int and 0 <= sample["rss_bytes"] <= receipt["peak_group_rss_bytes"]
                 and type(sample.get("artifact_bytes")) is int and 0 <= sample["artifact_bytes"] <= receipt["peak_artifact_bytes"]
                 and type(sample.get("members")) is list, "full_probe_runtime_sample_bounds")
        _require(all(type(member) is dict and type(member.get("pid")) is int and member["pid"] in observed
                     and type(member.get("pgid")) is int and member["pgid"] == receipt["pgid"]
                     and type(member.get("rss_bytes")) is int and member["rss_bytes"] >= 0 for member in sample["members"])
                 and sum(member["rss_bytes"] for member in sample["members"]) == sample["rss_bytes"],
                 "full_probe_runtime_sample_members")
    _require(receipt["peak_group_rss_bytes"] == (0 if rss_peak is None else rss_peak["rss_bytes"])
             and receipt["peak_artifact_bytes"] == max(receipt["artifact_bytes"],
                 0 if artifact_peak is None else artifact_peak["artifact_bytes"]), "full_probe_runtime_sample_maxima")


def _full_probe_evidence(plan, cases):
    """Read existing receipts/JSON only, never the sealed history stream."""
    evidence = plan.get("probe_evidence")
    probes = {name for name, case in cases.items() if case["stage"] == "probe"}
    _require(type(evidence) is dict and set(evidence) == probes, "full_probe_receipts_required")
    checked = {}
    for name in sorted(probes):
        refs = evidence[name]
        required = {"runtime_receipt", "sealed_result", "producer_plan", "independent_readback"}
        _require(type(refs) is dict and required <= set(refs)
                 and set(refs) <= required | {"runtime_timing"},
                 "full_probe_evidence_fields")
        values = {key: _positive_pin(ref, "full_probe_" + key) for key, ref in refs.items()}
        receipt, result, producer = (values[key] for key in ("runtime_receipt", "sealed_result", "producer_plan"))
        source = producer.get("source_freeze")
        _require(type(source) is dict and type(producer.get("authority")) is dict
                 and type(producer.get("admission")) is dict
                 and producer.get("schema") == RUN_PLAN_SCHEMA and type(producer.get("version")) is int
                 and producer["version"] == 3
                 and producer.get("status") == "admitted"
                 and producer.get("authority", {}).get("scientific_execution_authorized") is True
                 and producer.get("admission", {}).get("stage") == "probes"
                 and set(producer["admission"].get("admitted_case_ids", [])) == probes
                 and source.get("future_execution_commit") == plan["source_freeze"].get("future_execution_commit")
                 and source.get("source_sha256") == plan["source_freeze"].get("source_sha256"),
                 "full_probe_producer_source")
        prior = validate(refs["producer_plan"]["path"], executable=True, case_id=name)
        _require(prior["plan_sha256"] == refs["producer_plan"]["sha256"], "full_probe_plan_changed")
        matches = [case for case in producer.get("cases", []) if case.get("id") == name]
        _require(len(matches) == 1 and type(matches[0].get("execution")) is dict,
                 "full_probe_producer_case")
        execution = matches[0]["execution"]
        _require(type(execution.get("directory")) is str and Path(execution["directory"]).is_absolute()
                 and type(execution.get("supervision")) is dict, "full_probe_execution")
        directory = Path(execution["directory"])
        _require(Path(refs["runtime_receipt"]["path"]).resolve()
                 == Path(execution["supervision"]["closure_path"]).resolve()
                 and Path(refs["sealed_result"]["path"]).resolve() == (directory / "NativeResult.json").resolve(),
                 "full_probe_receipt_paths")
        request = _positive_pin(execution.get("request"), "full_probe_request")
        _require(_digest(request) == execution.get("effective_input_digest")
                 and result.get("request_sha256") == execution["effective_input_digest"]
                 and _digest(request["segments"]) == _digest(plan["protocol"]["segments"])
                 and _digest(_labelled_samples(request)) == _digest(plan["protocol"]["requested_sample_records"]),
                 "full_probe_input_and_labels")
        from scripts.benchmarks.g2_route_bindings import effective_issues
        ancestor = _positive_pin(cases[name]["input_ancestor"], "full_probe_ancestor")
        _require(not effective_issues(request, ancestor, cases[name], producer), "full_probe_effective_request")
        _probe_runtime_receipt(receipt, execution, refs["runtime_receipt"], refs["producer_plan"], name)
        _require(result.get("status") == "completed_bounded_voltage_lift_native_pilot"
                 and result.get("complete_run_window") is True and result.get("complete_protocol") is False
                 and _digest(result.get("execution_window")) == _digest(request.get("execution_window"))
                 and request.get("execution_window", {}).get("sample_records") == 66,
                 "full_probe_complete_window")
        history = result.get("history", {})
        _require(type(history) is dict and history.get("closed") is True and history.get("container_complete") is True
                 and history.get("writer_error") is None
                 and all(type(history.get(key)) is int and history[key] > 0
                         for key in ("logical_bytes", "encoded_bytes", "records"))
                 and type(history.get("logical_sha256")) is str and len(history["logical_sha256"]) == 64
                 and all(c in "0123456789abcdef" for c in history["logical_sha256"]), "full_probe_history_seal")
        if "runtime_timing" in values:
            timing = values["runtime_timing"]
            _require(timing.get("schema") == "solarlab.native-aggregate-timing.v1"
                     and timing.get("recorded_call_outcome") == "returned_completed"
                     and Path(refs["runtime_timing"]["path"]).resolve() == (directory / "RuntimeTiming.json").resolve(),
                     "full_probe_timing")
        if "independent_readback" in values:
            reader = values["independent_readback"]
            _require(reader.get("schema") == "solarlab.native-history-verification.v1"
                     and reader.get("verified") is True and reader.get("complete_readback") is True
                     and reader.get("complete_run_window") is True
                     and reader.get("request_sha256") == result["request_sha256"]
                     and Path(refs["independent_readback"]["path"]).resolve() == (directory / "RunResult.json").resolve()
                     and reader.get("history", {}).get("eof_seen") is True
                     and reader["history"].get("crc_verified") is True
                     and all(type(reader["history"].get(key)) is type(history[key])
                             and reader["history"][key] == history[key] for key in
                             ("logical_sha256", "logical_bytes", "encoded_bytes", "records")),
                     "full_probe_independent_readback")
        checked[name] = values
    return checked


def _measurement_value(record, path):
    _require(type(path) is list and 0 < len(path) <= 8
             and all(type(key) is str and key for key in path), "forecast_measurement_path")
    for key in path:
        _require(type(record) is dict and key in record, "forecast_measurement_missing")
        record = record[key]
    _require(type(record) in (int, float) and math.isfinite(record) and record >= 0,
             "forecast_measurement_number")
    return record


def _full_forecast(plan, case, probes):
    """Bind actual phase clocks; whole-call bootstrap/closeout proxies are conservative."""
    from scripts.benchmarks.g2_route_bindings import forecast_cap
    execution, phases = case["execution"], {"bootstrap", "native", "online_safety", "observer", "io", "closeout"}
    forecast = execution.get("forecast")
    _require(type(forecast) is dict and forecast.get("feasible") is True
             and type(forecast.get("mesh_intervals")) is int
             and forecast["mesh_intervals"] == case["mesh_intervals"]
             and type(forecast.get("slow_hold_model")) is str and forecast["slow_hold_model"].strip()
             and _digest(forecast.get("probe_evidence")) == _digest(plan["probe_evidence"]),
             "full_forecast_evidence")
    measured, work = forecast.get("measured_throughput"), forecast.get("phase_work_units")
    _require(type(measured) is dict and set(measured) == phases and type(work) is dict and set(work) == phases,
             "full_forecast_measured_phases")
    clocks = {
        "bootstrap": {("runtime_receipt", ("wall_s",), "s")},
        "native": {("sealed_result", ("elapsed_s",), "s"),
                   ("runtime_timing", ("phases", "ida_calls", "exclusive_ns"), "ns")},
        "online_safety": {("sealed_result", ("elapsed_s",), "s"),
                          ("runtime_timing", ("phases", "observation", "exclusive_ns"), "ns")},
        "observer": {("independent_readback", ("elapsed_s",), "s")},
        "io": {("runtime_timing", ("phases", name, "exclusive_ns"), "ns")
               for name in ("history_io", "artifact_io")},
        "closeout": {("runtime_receipt", ("wall_s",), "s")},
    }
    for phase in phases:
        row = measured[phase]
        _require(type(row) is dict and set(row) == {"probe_case_id", "source", "seconds_path", "time_unit",
                 "work_path", "seconds", "work_units", "units_per_s"}
                 and all(type(row[key]) is str for key in ("probe_case_id", "source", "time_unit"))
                 and row["probe_case_id"] in probes
                 and row["probe_case_id"] == "B_N8_" + case["id"].split("_")[2] + "_Probe"
                 and row["source"] in {"runtime_receipt", "sealed_result", "runtime_timing", "independent_readback"}
                 and row["source"] in probes[row["probe_case_id"]] and row["time_unit"] in {"s", "ns"},
                 "full_forecast_throughput_fields")
        _require(type(row["seconds_path"]) is list and all(type(key) is str for key in row["seconds_path"])
                 and (row["source"], tuple(row["seconds_path"]), row["time_unit"]) in clocks[phase],
                 "full_forecast_phase_clock")
        source = probes[row["probe_case_id"]][row["source"]]
        seconds = _measurement_value(source, row["seconds_path"])
        if row["time_unit"] == "ns":
            _require(type(seconds) is int, "forecast_integer_nanoseconds")
            seconds = seconds / 10**9
        units = 1 if row["work_path"] is None else _measurement_value(source, row["work_path"])
        _require(seconds > 0 and units > 0
                 and all(type(row[key]) in (int, float) and math.isfinite(row[key]) for key in
                         ("seconds", "work_units", "units_per_s"))
                 and row["seconds"] == seconds and row["work_units"] == units
                 and row["units_per_s"] == units / seconds,
                 "full_forecast_actual_throughput")
        _require(type(work[phase]) in (int, float) and math.isfinite(work[phase]) and work[phase] >= units
                 and type(forecast.get("phase_seconds", {}).get(phase)) in (int, float)
                 and forecast["phase_seconds"][phase] >= work[phase] / row["units_per_s"],
                 "full_forecast_phase_work")
    cap = forecast_cap(forecast)
    _require(forecast["work_counts"]["accepted_steps"] >=
             math.ceil((case["horizon_s"][1] - case["horizon_s"][0]) / plan["protocol"]["max_step_s"]),
             "full_forecast_step_lower_bound")
    _require(work["native"] >= forecast["work_counts"]["accepted_steps"], "full_forecast_native_work")
    if case["mesh_intervals"] == 16:
        _require(type(forecast.get("N16_uncertainty")) in (str, dict) and bool(forecast["N16_uncertainty"]),
                 "full_forecast_N16_uncertainty")
    _require(type(forecast.get("estimated_s")) in (int, float)
             and forecast["estimated_s"] == sum(forecast["phase_seconds"].values())
             and execution["wall_s"] == cap, "full_forecast_wall_binding")
    return cap


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
    admitted = _admitted_cases(plan, cases)
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
        if name in admitted and type(case.get("execution")) is dict and "supervision" in case["execution"]:
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
    if plan.get("admission") is None:
        blockers.append("unresolved_admission_stage")
    elif case_id is not None:
        _require(case_id in admitted, "case_not_in_admitted_stage")
    probes = _full_probe_evidence(plan, cases) if (plan.get("admission") or {}).get("stage") == "full" else None
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
        for case in (cases[name] for name in admitted):
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
                _full_forecast(plan, case, probes)
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
            "case_ids": list(cases), "admitted_case_ids": admitted,
            "admission_stage": None if plan.get("admission") is None else plan["admission"]["stage"],
            "sample_records": {k: v["sample_records"] for k, v in cases.items()}}


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
    from scripts.benchmarks.g2_compare import case_metadata, normalize_ida_sample
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
        comparison_context = case_metadata(request, case["id"])
        comparison_context["run_id"] = run_id
        writer.append({"kind":"sealed_g2_context","request_sha256":_digest(request),"evidence":comparison_context})
        def retain_sample(tag, sample, certificate):
            checkpoint()
            writer.append({"kind":"sealed_g2_sample","request_sha256":_digest(request),
                           "evidence":normalize_ida_sample(request,tag,sample,certificate)})
        result = verify_history(folder, max_record_bytes=execution["max_record_bytes"],
            max_logical_bytes=resources["reader_logical_byte_cap"], max_records=execution["max_records"],
            deadline_monotonic=time.perf_counter()+execution["wall_s"], integer_io_policy=execution.get("integer_io_policy"),
            execution_admission=execution["native_admission"], execution_provenance=execution.get("execution_provenance"),
            charge_observer=observer, sample_sink=retain_sample)
        checkpoint()
        writer.finish()
        result.update(plan_sha256=context["plan_sha256"], case_id=case["id"], methods=METHODS,
                      observed_history={"path": "Observation/Charge.jsonl.gz", "records": writer.records,
                          "logical_sha256": writer.logical_sha256, "logical_bytes": writer.logical_bytes,
                          "encoded_bytes": writer.encoded_bytes, "closed": writer.closed,
                          "container_complete":writer.closed})
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


def compare(plan_path, *, be_layout=None):
    context = validate(plan_path, executable=True, purpose="evaluate")
    from scripts.benchmarks.g2_compare import compare_plan
    return compare_plan(context, be_layout=be_layout)


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
        if args.action == "compare":
            from scripts.benchmarks.native_history import integer_json_dumps
            from scripts.benchmarks.t1_gate import INTEGER_POLICY
            print(integer_json_dumps(result, INTEGER_POLICY, sort_keys=True, indent=2))
        else:
            print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    except (ValueError, KeyError, OSError) as error:
        print(json.dumps({"status": "refused", "reason": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
