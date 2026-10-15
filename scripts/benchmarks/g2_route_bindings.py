"""Read-only G2-R V2 binding preflight; no solver, history reader or admission.

Saved inputs are ancestry until a case supplies execution.request.  This module
never edits a request, promotes a source snapshot to a commit, or freezes T1.
All hashes named *_sha256 below use canonical JSON except file descriptors.
The sampling hash includes BOTH observation_times and quadrature, as the
producer's reference-refresh and observer contracts require.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys


LIMIT = 4 * 1024**2
PLAN_SCHEMA = "solarlab.g2-route.run-plan.v2"
PROFILES = {"Coarse": {"r": 1e-4, "alpha": 1e-8},
            "Main": {"r": 1e-5, "alpha": 1e-8},
            "Strict": {"r": 1e-6, "alpha": 1e-10}}
CASE_IDS = {f"B_N8_{p}_{s}" for p in PROFILES for s in ("Probe", "Full")} | {"B_N16_Strict_Full"}


def require(condition, reason):
    if not condition:
        raise ValueError("g2_bindings_" + reason)


def _case_stage(case):
    require(case.get("id") in CASE_IDS, "case_id")
    declared = {"Probe": "probe", "Full": "full"}[case["id"].rsplit("_", 1)[1]]
    require(case.get("stage") in ("probe", "full") and case["stage"] == declared, "case_id_stage_binding")
    return declared


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate_json_key:" + key)
        result[key] = value
    return result


def read_json(path, expected=None):
    path = Path(path)
    require(path.suffix == ".json" and path.is_file(), "json_metadata_only:" + str(path))
    with path.open("rb") as stream:
        raw = stream.read(LIMIT + 1)
    require(len(raw) <= LIMIT, "metadata_size")
    record = {"path": str(path.resolve()), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    if expected is not None:
        require(expected.get("bytes") == len(raw) and expected.get("sha256") == record["sha256"], "file_pin:" + str(path))
    value = json.loads(raw, object_pairs_hook=_object,
                       parse_constant=lambda s: require(False, "nonfinite_json:" + s))
    require(type(value) is dict, "json_object")
    return value, record


def pinned(reference):
    require(type(reference) is dict and {"path", "bytes", "sha256"} <= reference.keys(), "unresolved_descriptor")
    return read_json(reference["path"], reference)


def sample_labels(request, count=None):
    """Keep event sides; a probe retains only the ramp-side 0.2 endpoint."""
    segments = request["segments"] if count is None else request["segments"][:count]
    require(set(request["observation_times"]) == {s["id"] for s in request["segments"]}, "sampling_segments")
    labels = []
    for i, segment in enumerate(segments):
        times = request["observation_times"][segment["id"]]
        require(len(times) >= 2 and times[0] == segment["start"] and times[-1] == segment["end"]
                and all(type(t) in (int, float) and math.isfinite(t) for t in times)
                and all(a < b for a, b in zip(times, times[1:])), "sampling_order_or_endpoints")
        for j, t in enumerate(times):
            labels.append({"segment": segment["id"], "index": j, "time_hex": float(t).hex(),
                           "event_side": "right" if j == 0 else "left" if j == len(times)-1 else "interior",
                           "protocol_initial": i == j == 0,
                           "protocol_terminal": i == len(request["segments"])-1 and j == len(times)-1})
    return labels


def inspect_request(request):
    """Check saved identities and shapes without constructing a physical model."""
    require(request.get("schema") == "solarlab.voltage-lift-native-request.v1", "request_schema")
    packet, mapping = request["numeric_packet"], request["voltage_lift_map"]
    profile, size = request["physical_tolerance_policy"], len(request["z0"])
    require(size == profile["size"] == len(request["zdot0"]) == len(request["controls"]["atol"]), "coordinate_shape")
    require(all(type(x) in (int, float) and math.isfinite(x) for k in ("z0", "zdot0") for x in request[k]), "initial_finite")
    require(mapping["mapped_input_words"] == 12 and request["frame_input_policy"]["raw_parent_words"] == 4,
            "raw4_mapped12")
    require(digest(mapping) == request["map_identity"], "map_identity")
    require(profile["numeric_packet_sha256"] == digest(packet) and request["weight_certificate"] == profile,
            "physical_weight_identity")
    require(request["budgets"] == request["original_budgets"], "historical_budgets_changed")
    hashes = {"parent_map_identity": request["map_identity"], "map_sha256": digest(mapping),
              "controls_sha256": digest(request["controls"]), "budgets_sha256": digest(request["budgets"]),
              "segments_sha256": digest(request["segments"]),
              "sampling_sha256": digest((request["observation_times"], request["quadrature"]))}
    for name in ("interval_observation", "segment_frame_policy", "frame_input_policy", "reference_refresh_policy"):
        policy = request.get(name, {})
        for key, expected in hashes.items():
            if key in policy:
                require(policy[key] == expected, name + ":" + key)
        for key, expected in (("map_identity", request["map_identity"]), ("protocol_sha256", hashes["segments_sha256"])):
            if key in policy:
                require(policy[key] == expected, name + ":" + key)
    frame = request["initial_segment_frame"]
    for key in ("physical_handoff_words_hex", "physical_rate_handoff_words_hex"):
        require(len(frame[key]) == 12 and all(len(row) == size for row in frame[key]), "initial_word_shape")
        require(all(isinstance(x, str) and math.isfinite(float.fromhex(x)) for row in frame[key] for x in row), "initial_word_finite")
    labels = sample_labels(request)
    return {"canonical_request_sha256": digest(request), **hashes,
            "nodes": packet["nodes"], "mesh_intervals": packet["intervals"], "coordinates": size,
            "profile": deepcopy(profile["profile"]), "controls": deepcopy(request["controls"]),
            "budgets": deepcopy(request["budgets"]), "raw_components": 4, "mapped_components": 12,
            "physical_charge_policy_observed": deepcopy(request["interval_observation"]["physical_charge"]),
            "budget_scope": "Saved budgets and signed-charge diagnostics; no unfrozen T1 values are activated.",
            "floor_basis": profile["floor_basis"], "physical_policy_sha256": digest(profile),
            "protocol": deepcopy(request["segments"]), "sampling": {"full_labels": len(labels),
                "unique_times": len({row["time_hex"] for row in labels}),
                "positive_windows": len(labels)-len(request["segments"]), "labels_sha256": digest(labels)},
            "initial": {"actual_initial_identity": request["actual_initial_identity"],
                "numeric_packet_sha256": digest(packet),
                "physical_reference_fields_sha256": digest(packet["physical_reference_fields"]),
                "z0_words_sha256": digest([float(x).hex() for x in request["z0"]]),
                "zdot0_words_sha256": digest([float(x).hex() for x in request["zdot0"]]),
                **{key + "_sha256": digest(frame[key]) for key in
                   ("physical_handoff_words_hex", "physical_rate_handoff_words_hex")},
                "preparation_record_sha256": digest(request["initial_preparation"]),
                "preparation_context_sha256": request["preparation_context_sha256"]}}


def effective_issues(request, ancestor, case, plan):
    """Check real supplied effective metadata; return every missing adaptation."""
    issues = []
    def check(value, reason):
        if not value:
            issues.append(reason)
    check(request["controls"]["max_step"] == plan["protocol"]["max_step_s"], "controls.max_step_requires_0.02")
    check(request["physical_tolerance_policy"]["profile"] == case["profile"], "profile_mismatch")
    expected_controls = dict(ancestor["controls"], max_step=plan["protocol"]["max_step_s"])
    check(request["controls"] == expected_controls, "only_declared_max_step_control_change")
    for key in ("numeric_packet", "voltage_lift_map", "z0", "zdot0", "budgets", "original_budgets",
                "segments", "observation_times", "quadrature", "physical_tolerance_policy"):
        check(digest(request[key]) == digest(ancestor[key]), "saved_physical_input_changed:" + key)
    for key in ("physical_handoff_words_hex", "physical_rate_handoff_words_hex"):
        check(request["initial_segment_frame"][key] == ancestor["initial_segment_frame"][key], "initial_words_changed:" + key)
    policy = request.get("certificate_failure_policy", {})
    check(policy.get("schema") == "solarlab.certificate-failure-policy.v1"
          and policy.get("scientific_failure") == "latch-and-continue"
          and policy.get("safety_failure") == "stop"
          and policy.get("observation") == "sealed-history", "D14_D25_certificate_policy")
    check(bool(policy.get("T1")), "frozen_T1_request_binding")
    probe = _case_stage(case) == "probe"
    window = request.get("execution_window")
    if probe:
        expected_window = {"schema": "solarlab.protocol-run-window.v1", "last_segment": "voltage_ramp",
            "segment_count": 2, "protocol_sha256": digest(request["segments"]),
            "start_hex": float(0).hex(), "end_hex": float(.2).hex(), "full_end_hex": float(9.2).hex(),
            "sample_records": 66, "initial_state": "original-protocol-initial-state", "multistep_history_restored": False}
        check(window == expected_window, "probe_window_requires_66_ramp_side_labels")
    else:
        check(window is None, "full_protocol_required")
    refresh = request.get("reference_refresh_policy")
    if probe:
        check(refresh is None, "probe_refresh_must_be_disabled")
    else:
        refresh = refresh or {}
        schedule = plan["protocol"]["reference_refresh_schedule"]
        for key in ("time_hex", "physical_segment_ordinal", "max_refreshes", "native_api",
                    "same_owner_generation_increment", "preserve_original_segment_inputs",
                    "discard_native_multistep_history", "restored_seed_prefix_credit"):
            check(refresh.get(key) == schedule[key], "refresh:" + key)
        check(refresh.get("schema") == "solarlab.reference-state-transfer-policy.v2"
              and refresh.get("trigger") == "exact_domain_safe_native_endpoint"
              and refresh.get("certificate_failure_policy_sha256") == digest(policy), "v2_refresh_source_binding")
    return issues


def forecast_cap(forecast):
    phases = ("bootstrap", "native", "online_safety", "observer", "io", "closeout")
    require(type(forecast) is dict and set(forecast.get("phase_seconds", {})) == set(phases), "forecast_phases")
    values = forecast["phase_seconds"].values()
    require(all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in values), "forecast_finite")
    estimate = sum(values)
    require(estimate > 0 and forecast.get("slow_hold_model") and forecast.get("work_counts")
            and forecast.get("probe_evidence"), "forecast_unmeasured_phase_or_evidence")
    require(all(type(forecast["work_counts"].get(k)) is int and forecast["work_counts"][k] >= 0
                for k in ("accepted_steps", "residual_calls", "jacobian_calls")), "forecast_work_counts")
    if forecast.get("mesh_intervals") == 16:
        require(forecast.get("N16_uncertainty"), "N16_forecast_uncertainty")
    cap = math.ceil(3 * estimate)
    require(cap <= 43200, "forecast_infeasible_no_silent_cap")
    return cap


def resource_preflight(plan):
    """Capacity snapshot, not a reservation; no directories or runs are created."""
    resources = plan["resources"]
    require(resources["WS1_channels"] == 4 and resources["threads_per_case"] == 1
            and resources["rss_bytes_per_case"] == 4*1024**3
            and resources["combined_persisted_case_bytes"] == 4*1024**3
            and resources["reader_logical_byte_cap"] == 32*1024**3
            and resources["probe_wall_cap_s"] == 3600, "resource_contract")
    active, archive = (Path(plan["artifacts"][k]) for k in ("active_root", "archive_root"))
    # One complete temporary copy per case is a conservative preparation
    # allowance, not a claim that streaming readers need 32 GiB scratch files.
    quantities = [("active", active, 16*1024**3), ("temporary_copy_allowance", active, 16*1024**3),
                  ("completed_archive_copy", archive, 16*1024**3), ("workspace_headroom", active, 8*1024**3)]
    devices = {}
    for name, path, amount in quantities:
        nearest = path
        while not nearest.exists():
            nearest = nearest.parent
        usage, device = shutil.disk_usage(nearest), os.stat(nearest).st_dev
        entry = devices.setdefault(device, {"device": device, "total_bytes": usage.total,
                    "free_bytes": usage.free, "required_bytes": 0, "allocations": []})
        entry["free_bytes"] = min(entry["free_bytes"], usage.free)
        entry["required_bytes"] += amount
        entry["allocations"].append({"purpose": name, "planned_root": str(path),
                                      "measured_at": str(nearest), "bytes": amount})
    for entry in devices.values():
        entry["capacity_sufficient"] = entry["free_bytes"] >= entry["required_bytes"]
    case_stages = [(case, _case_stage(case)) for case in plan["cases"]]
    return {"schema": "solarlab.g2-route.resource-preflight.v1", "reservation_made": False,
        "WS1_channels": 4, "global_channel_cap": resources["global_scientific_channel_cap"],
        "rss_bytes_per_case": resources["rss_bytes_per_case"], "four_channel_rss_bytes": 16*1024**3,
        "combined_persisted_bytes_per_case": 4*1024**3, "reader_logical_bytes_per_case": 32*1024**3,
        "four_channel_persisted_bytes": 16*1024**3, "peak_disk_allowance_bytes": 56*1024**3,
        "filesystems": list(devices.values()), "disk_capacity_sufficient": all(d["capacity_sufficient"] for d in devices.values()),
        "accounting": "Group free capacity by st_dev once; sum physically separate active/temp/archive copies even on the same filesystem.",
        "temporary_allowance_basis": "up to one full persisted-case copy per concurrent case; no decompressed-history file",
        "workspace_headroom_basis": "8 GiB prospective engineering headroom; final source/runtime staging must fit or be re-estimated",
        "per_case": [{"case_id": c["id"], "rss_bytes": resources["rss_bytes_per_case"],
            "persisted_bytes": resources["combined_persisted_case_bytes"],
            "wall_cap_s": 3600 if stage == "probe" else None,
            "wall_cap_status": "declared_runaway_guard" if stage == "probe" else "pending_probe_forecast",
            "native_steps": resources["native_step_cap"], "residual_calls": resources["residual_call_cap"]}
            for c, stage in case_stages],
        "conditional_S0": {"shares_one_of_four_channels": True, "rss_bytes": resources["rss_bytes_per_case"],
                            "persisted_bytes": resources["combined_persisted_case_bytes"], "wall_s": None,
                            "reason": "Reuse/revalidation source decision pending; actual saved S0 budgets remain separate."},
        "full_forecast": {"estimated_seconds": None, "cap_seconds": None, "maximum_seconds": 43200,
            "rule": "ceil(3 * sum(bootstrap,native,online_safety,observer,io,closeout)); reject when above43200",
            "requirements": deepcopy(resources["forecast_requirements"]),
            "probe_coverage": "0-0.2s excludes slow-hold startup, refresh and full throughput",
            "N16_proxy": "4x N8 Strict work forecast only as explicit uncertain engineering proxy",
            "rolling_forecast": resources["rolling_forecast"]},
        "pending": ["actual channel reservation and host-wide memory availability",
                    "recheck free capacity immediately before admission",
                    "genuine per-case source-bound wall/output policies and immutable execution directories"]}


def route_capabilities(path):
    """Inspect reviewed entry text, without importing its runtime paths."""
    raw = Path(path).read_bytes()
    require(len(raw) <= LIMIT, "source_size")
    tree = ast.parse(raw)
    schemas = sorted({n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
                      and isinstance(n.value, str) and n.value.startswith("solarlab.g2-route.run-plan.")})
    t1 = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "T1_IMPLEMENTATION" for t in node.targets):
            t1 = ast.literal_eval(node.value)
    return {"source": {"path": str(Path(path).resolve()), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()},
            "literal_plan_schemas": schemas, "T1_IMPLEMENTATION": t1,
            "scope": "source metadata observation; not implementation review or runtime validation"}


def resolve(plan_path, *, repository, n16_acceptance, environment_lock=None, resource_preflight=None):
    plan, plan_pin = read_json(plan_path)
    require(plan.get("schema") == PLAN_SCHEMA and plan.get("version") == 2, "plan_schema")
    require({c["id"] for c in plan["cases"]} == CASE_IDS and len(plan["cases"]) == 7, "seven_cases")
    schedule = plan["protocol"]["reference_refresh_schedule"]
    require(not any("sha256" in k or "map_identity" in k for k in schedule), "shared_case_specific_identity")
    require(schedule["time_hex"] == "0x1.7a4c1c85a8b06p-2" and schedule["segment_id"] == "slow_state_hold"
            and schedule["max_refreshes"] == 1 and schedule["native_api"] == "init_step", "refresh_schedule")
    missing = []
    freeze = plan["source_freeze"]
    for key in ("future_execution_commit", "review_record", "frozen_gate_spec"):
        if not freeze.get(key):
            missing.append("source_freeze." + key)
    if freeze.get("future_execution_commit"):
        commit = freeze["future_execution_commit"]
        require(type(commit) is str and len(commit) == 40 and all(c in "0123456789abcdef" for c in commit), "commit_format")
    if not freeze.get("environment_lock"):
        missing.append("source_freeze.environment_lock_not_commander_bound")
    if plan.get("status") != "admitted" or plan["authority"].get("scientific_execution_authorized") is not True:
        missing.append("run_plan_not_admitted")
    gate, gate_pin = pinned(plan["definitions"]["gate_spec"])
    if gate.get("status") != "frozen":
        missing.append("T1_not_frozen")
    if freeze.get("frozen_gate_spec"):
        frozen, frozen_pin = pinned(freeze["frozen_gate_spec"])
        require(frozen.get("status") == "frozen" and frozen_pin == gate_pin, "gate_freeze_mismatch")
    capabilities = route_capabilities(Path(repository) / plan["implementation"]["entrypoint_proposal"])
    if PLAN_SCHEMA not in capabilities["literal_plan_schemas"]:
        missing.append("reviewed_entry_does_not_support_RunPlanV2")
    if capabilities["T1_IMPLEMENTATION"] is None:
        missing.append("reviewed_frozen_T1_implementation_pending")
    accepted, acceptance_pin = read_json(n16_acceptance)
    require(accepted.get("accepted") is True and accepted.get("source_input_acceptance_only") is True,
            "N16_saved_input_acceptance")
    n16 = accepted["cases"]["B"]
    cases = []
    for case in plan["cases"]:
        name = case["id"]
        declared_stage = _case_stage(case)
        probe = declared_stage == "probe"
        mesh = 16 if "N16" in name else 8
        require(all(type(case[k]) is int for k in ("nodes", "coordinates", "mesh_intervals", "sample_records"))
                and case["profile"] == PROFILES[name.split("_")[2]] and case["nodes"] == mesh+1
                and case["coordinates"] == 5*(mesh+1) and case["mesh_intervals"] == mesh
                and case["horizon_s"] == [0, .2 if probe else 9.2]
                and case["sample_records"] == (66 if probe else 131), "case_definition:" + name)
        ancestor, ancestor_pin = pinned(case["input_ancestor"])
        observed = inspect_request(ancestor)
        require(observed["nodes"] == case["nodes"] and observed["coordinates"] == case["coordinates"]
                and observed["profile"] == case["profile"], "ancestor_case:" + name)
        require(digest(ancestor["segments"]) == digest(plan["protocol"]["segments"])
                and digest(sample_labels(ancestor)) == digest(plan["protocol"]["requested_sample_records"]), "original_protocol_labels")
        if mesh == 16:
            require(ancestor_pin == n16["request"] and observed["canonical_request_sha256"] == n16["canonical_request_sha256"]
                    and observed["initial"]["actual_initial_identity"] == n16["actual_initial_identity"], "N16_actual_accepted_input")
        execution = case.get("execution") or {}
        effective, effective_pin, binding = None, None, None
        issues = []
        if execution.get("request"):
            effective, effective_pin = pinned(execution["request"])
            binding = inspect_request(effective)
            require(digest(sample_labels(effective)) == digest(plan["protocol"]["requested_sample_records"]), "effective_protocol_labels")
            issues.extend(effective_issues(effective, ancestor, case, plan))
            if execution.get("effective_input_digest") != binding["canonical_request_sha256"]:
                issues.append("effective_input_digest")
        else:
            issues.append("effective_request_not_materialized")
        for key in ("directory", "channel", "wall_s", "max_record_bytes", "max_records", "integer_io_policy", "native_admission", "supervision"):
            if execution.get(key) is None:
                issues.append("execution." + key)
        if execution.get("supervision") is not None:
            from scripts.benchmarks.g2_route import _validate_supervision
            _validate_supervision(plan, execution)
        if probe and execution.get("wall_s") is not None:
            require(type(execution["wall_s"]) in (int, float) and 0 < execution["wall_s"] <= 3600, "probe_wall_guard")
        if not probe:
            if not execution.get("forecast"):
                issues.append("phase_and_work_forecast_after_probe")
            else:
                require(execution.get("wall_s") == forecast_cap(execution["forecast"]), "forecast_cap_binding")
        missing.extend(name + ":" + issue for issue in issues)
        planned_labels = sample_labels(ancestor, 2 if probe else 3)
        cases.append({"case_id": name, "stage": case["stage"], "profile": case["profile"],
            "ancestry": {"request": ancestor_pin, "resolved": observed,
                         "old_reference_refresh_policy": deepcopy(ancestor.get("reference_refresh_policy")),
                         "saved_input_acceptance": acceptance_pin if mesh == 16 else None},
            "effective_request": effective_pin, "effective_case_binding": binding,
            "binding_status": "pending" if issues else "metadata_resolved_not_admitted",
            "required_adaptations_from_saved_input": effective_issues(ancestor, ancestor, case, plan),
            "planned_active_sampling": {"records": len(planned_labels),
                "unique_times": len({row["time_hex"] for row in planned_labels}),
                "positive_windows": len(planned_labels)-(2 if probe else 3),
                "labels_sha256": digest(planned_labels), "last_label": planned_labels[-1]},
            "reference_refresh": {"enabled": not probe, "max_refreshes": 0 if probe else 1,
                                  "schedule_ref": "/shared_reference_refresh_schedule"},
            "missing": issues})
    acceptance, s0_pin = pinned(plan["conditional_S0"]["reference"])
    require(acceptance.get("accepted") is True and acceptance.get("complete_protocol_accepted") is True, "S0_acceptance")
    s0, s0_input = pinned(acceptance["actual_input"])
    s0_binding = inspect_request(s0)
    require(s0_binding["canonical_request_sha256"] == acceptance["request_sha256"], "S0_actual_request")
    missing.append("conditional_S0.current_source_reuse_or_revalidation_review")
    environment = read_json(environment_lock) if environment_lock else None
    if not environment or environment[0].get("loaded_thread_verification", {}).get("passed") is not True:
        missing.append("actual_loaded_environment_lock")
    resources = read_json(resource_preflight) if resource_preflight else None
    if not resources or resources[0].get("disk_capacity_sufficient") is not True:
        missing.append("disk_resource_preflight")
    return {"schema": "solarlab.g2-route.bindings.v1", "status": "not_admitted" if missing else "metadata_validated",
        "execution_authorized": False, "metadata_only": True, "source_plan": plan_pin,
        "definition_observed": gate_pin, "source_freeze": deepcopy(freeze), "source_capabilities": capabilities,
        "shared_reference_refresh_schedule": deepcopy(schedule), "cases": cases,
        "conditional_S0": {"case_id": "S0_Revalidation", "acceptance": s0_pin, "actual_input": s0_input,
            "observed": s0_binding, "decision": "pending_independent_source_review",
            "conditions": deepcopy(plan["conditional_S0"])},
        "environment_lock_observed": environment[1] if environment else None,
        "resource_preflight_observed": resources[1] if resources else None,
        "missing": missing, "scientific_qualification": False,
        "scope": "Parsing and identity checks only; future effective requests and source admission are separate."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--n16-acceptance", required=True)
    parser.add_argument("--environment-lock")
    parser.add_argument("--resource-preflight")
    args = parser.parse_args(argv)
    try:
        result = resolve(args.plan, repository=args.repository, n16_acceptance=args.n16_acceptance,
                         environment_lock=args.environment_lock, resource_preflight=args.resource_preflight)
    except (ValueError, KeyError, TypeError, OSError, SyntaxError) as error:
        print(json.dumps({"status": "invalid_metadata", "execution_authorized": False,
                          "error": type(error).__name__, "reason": str(error)}, indent=2))
        return 2
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0 if result["status"] == "metadata_validated" else 1


if __name__ == "__main__":
    sys.exit(main())
