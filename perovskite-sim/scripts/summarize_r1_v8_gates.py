"""Read-only aggregation of original gates in a sealed complete V8 run.

All accepted rows are inspected where the original quantity is applicable.
The saved finest-tier certificate remains distinct from all-row maxima. This
does not refit, integrate, independently re-evaluate physics, or grant P1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

from scripts.run_r1_v8_prototype import LIMITS, canonical, sha, source_snapshot, write


FINITE_METRICS = {
    "nonlinear_residual": ("physics_reconstruction", "scaled_nonlinear_residual"),
    "charge_balance_relative": ("physics_reconstruction", "charge_balance_relative"),
    "full_charge_balance_normalized": ("physical", "charge_balance_normalized"),
    "physical_current_spread_relative": ("physical", "contact_internal_current_spread_relative"),
    "trap_storage_normalized_error": ("physical", "trap_storage_check", "normalized_error"),
}
ALL_METRICS = {
    "local_carrier_residual": ("physics_reconstruction", "local_carrier_residual"),
    "local_gauss_residual": ("physics_reconstruction", "local_gauss_residual"),
    "eliminated_operator_error": ("physics_reconstruction", "eliminated_operator_error"),
    "inventory_relative_drift": ("physical", "inventory_relative_drift"),
    "full_gauss_normalized": ("physical", "gauss_normalized"),
}
FINEST_CERTIFICATE = {"nonlinear_residual", "local_carrier_residual", "local_gauss_residual",
    "analytic_jacobian_error", "charge_balance_relative", "all_face_current_relative",
    "interface_current_relative", "eliminated_operator_error"}
REFINEMENT = {"refinement_state_change", "refinement_current_change"}


def load(path):
    return json.loads(Path(path).read_text())


def at(value, path):
    for name in path:
        value = value[name]
    return value


def location(index, row):
    return {"row": index, "substeps": row["substeps"], "time_s": row["time_s"], "phase": row["phase"]}


def metric_values(row):
    finite = row["phase"] == "accepted_regular_step"
    if not finite and row["phase"] != "0+":
        raise ValueError("unrecognized accepted-row phase")
    values = {name: at(row, path) for name, path in ALL_METRICS.items()}
    if finite:
        values.update({name: at(row, path) for name, path in FINITE_METRICS.items()})
        values["all_face_current_relative"] = row["physics_reconstruction"]["all_face_current_relative"]
        values["interface_current_relative"] = row["physics_reconstruction"]["interface_current_relative"]
    else:
        # These are computed derivative right limits, not finite-step zeros.
        regular = row["physical"]["regular_right_limit"]
        values["all_face_current_relative"] = regular["face_current_spread_relative"]
        values["interface_current_relative"] = regular["interface_current_spread_relative"]
    physics = row["physics_reconstruction"]
    if physics["jacobian_checked"]:
        if not finite:
            raise ValueError("initial row cannot claim a finite-step Jacobian check")
        values["analytic_jacobian_error"] = physics["analytic_jacobian_error"]
    currents = [physics[name] for name in ("conduction_A_m2", "carrier_conduction_A_m2", "positive_ion_current_A_m2")]
    if not currents[0] or not len(currents[0]) == len(currents[1]) == len(currents[2]):
        raise ValueError("current decomposition arrays do not align")
    # Preserve the original s.conduction-(s.carrier+s.positive) operation order.
    values["current_decomposition_error"] = max(abs(a - (b + c)) for a, b, c in zip(*currents)) / max(
        max(abs(value) for value in currents[0]), 1.)
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError("nonfinite, negative, or missing original gate observation")
    return values


def peak(entries):
    item = max(entries, key=lambda entry: entry[0])
    return {"maximum": item[0], "location": item[1], "evaluated_rows": len(entries)}


def margin(value, limit):
    return {"limit": limit, "absolute_headroom": limit - value,
            "headroom_fraction": (limit - value) / limit, "headroom_percent": 100 * (limit - value) / limit,
            "limit_to_maximum_ratio": None if value == 0 else limit / value}


def aggregate(rows, certificate):
    if set(certificate["metrics"]) != set(LIMITS) or certificate["limits"] != LIMITS:
        raise ValueError("certificate does not retain the original sixteen metrics and limits")
    entries = {name: [] for name in LIMITS if name not in REFINEMENT}
    component_entries = {name: [] for name in ("positive_ion_flux", "positive_ion_rate")}
    failures, initial_checks, tiers = [], [], {}
    for index, row in enumerate(rows):
        point, values = location(index, row), metric_values(row)
        tier = tiers.setdefault(str(row["substeps"]), {"rows": 0, "anchors": 0, "finite_rows": 0,
                                                       "jacobian_checked_rows": 0, "failed_rows": 0})
        tier["rows"] += 1
        tier["anchors" if row["phase"] == "0+" else "finite_rows"] += 1
        tier["jacobian_checked_rows"] += int(row["physics_reconstruction"]["jacobian_checked"])
        reasons = [name for name, value in values.items() if value > LIMITS[name]]
        if not row["physical_checks_passed"]:
            reasons.extend(row["physical_failure_reasons"] or ["saved_physical_checks_failed"])
        for name, value in values.items():
            entries[name].append((value, point))
        components = {}
        for name in component_entries:
            detail = row["physics_reconstruction"]["eliminated_operator"][name]
            keys = ("maximum_absolute_difference", "normalization_scale", "normalization_floor",
                    "relative_error", "floor_active", "unit", "direct_maximum_absolute", "eliminated_maximum_absolute")
            item = {key: detail[key] for key in keys}
            if (not math.isfinite(item["normalization_floor"]) or item["normalization_floor"] <= 0.
                    or not item["normalization_scale"] >= item["normalization_floor"]
                    or name == "positive_ion_flux" and item["normalization_floor"] != 1.):
                raise ValueError("original ion component denominator/floor changed")
            item["location"] = point
            component_entries[name].append(item)
            components[name] = item
        if reasons:
            tier["failed_rows"] += 1
            failures.append({**point, "reasons": sorted(set(reasons)), "metrics": values, "ion_components": components})
        if row["phase"] == "0+":
            initial_checks.append({**point, "physical_checks": row["physical_checks"],
                "applicable_metric_names": sorted(values),
                "finite_step_placeholders_excluded": sorted(FINITE_METRICS) + ["analytic_jacobian_error"]})
    finest = max(row["substeps"] for row in rows)
    gates = {}
    for name, values in entries.items():
        if not values:
            raise ValueError("required gate has no actual observations: " + name)
        maximum = peak(values)
        saved = certificate["metrics"][name]
        by_tier = {tier: peak([item for item in values if str(item[1]["substeps"]) == tier])
                   for tier in tiers if any(str(item[1]["substeps"]) == tier for item in values)}
        gates[name] = {**maximum, **margin(maximum["maximum"], LIMITS[name]), "rowwise_aggregation_applicable": True,
            "rows_exceeding_original_limit": sum(value > LIMITS[name] for value, _ in values),
            "per_substeps": by_tier, "finest_saved_row_maximum": by_tier[str(finest)],
            "reported_certificate_value": saved, "certificate_value_is_all_row_maximum": saved == maximum["maximum"],
            "original_certificate_aggregation": ("finest_tier_including_initial_right_limits_where_defined" if name in FINEST_CERTIFICATE
                else "all_tier_output_states" if name in {"inventory_relative_drift", "current_decomposition_error"}
                else "all_finite_accepted_rows"),
            "observation_scope": ("actually_jacobian_checked_rows_only" if name == "analytic_jacobian_error"
                else "finite_accepted_rows_only" if name in FINITE_METRICS
                else "all_accepted_rows_including_actual_initial_quantity"),
            "source": ("initial physical.regular_right_limit; finite physics_reconstruction" if name in {
                "all_face_current_relative", "interface_current_relative"}
                else "original conduction-(carrier+ion) binary64 arithmetic" if name == "current_decomposition_error"
                else ".".join((ALL_METRICS | FINITE_METRICS).get(name, ("physics_reconstruction", name))))}
    for name in sorted(REFINEMENT):
        saved = certificate["metrics"][name]
        gates[name] = {"reported_certificate_value": saved, **margin(saved, LIMITS[name]),
            "rowwise_aggregation_applicable": False, "evaluated_rows": None, "location": None,
            "source": "sealed ResultV1.certificate.metrics; original nested output-trajectory comparison",
            "independently_recomputed_here": False, "passed_recorded_limit": saved <= LIMITS[name]}
    components = {name: {"evaluated_rows": len(values),
        "maximum_relative_error": max(values, key=lambda item: item["relative_error"]),
        "maximum_absolute_difference": max(values, key=lambda item: item["maximum_absolute_difference"]),
        "floor_active_rows": sum(item["floor_active"] for item in values),
        "minimum_recorded_floor": min(item["normalization_floor"] for item in values),
        "maximum_recorded_floor": max(item["normalization_floor"] for item in values),
        "rows_exceeding_original_limit": sum(item["relative_error"] > LIMITS["eliminated_operator_error"] for item in values),
        "scope": "original_saved_difference_and_normalization_not_subtraction_of_rounded_side_arrays"}
        for name, values in component_entries.items()}
    return {"original_gates": gates, "ion_components": components, "rows": len(rows), "per_substeps": tiers,
        "failed_rows": len(failures), "failures": failures,
        "first_failure_in_file_order": failures[0] if failures else None,
        "earliest_failure_by_physical_time": min(failures, key=lambda item: (item["time_s"], item["substeps"], item["row"])) if failures else None,
        "initial_gate_observations": initial_checks, "reported_certificate_certified": certificate["certified"]}


def summarize(directory, output):
    start = time.perf_counter()
    directory, output = Path(directory).resolve(), Path(output).resolve()
    if output == directory or output.is_relative_to(directory):
        raise ValueError("summary must not change the sealed trajectory")
    from scripts.analyze_r1_v7_prototype import verify_manifest
    manifest = verify_manifest(directory)
    required = {"ResultV1.json", "SummaryV1.json", "AcceptedStepsV1.jsonl", "SourceReceiptV1.json", "RequestV1.json"}
    if not required <= set(manifest):
        raise ValueError("gate summary input lacks sealed original artifacts")
    record, summary, receipt, request = [load(directory / name) for name in
        ("ResultV1.json", "SummaryV1.json", "SourceReceiptV1.json", "RequestV1.json")]
    if summary.get("schema") != "R1V8PrototypeRunV1" or summary.get("source_unchanged") is not True:
        raise ValueError("gate summary requires an unchanged V8 execution source")
    current = source_snapshot(summary["source_commit"])
    if (receipt.get("source_commit") != summary["source_commit"]
            or receipt.get("r1_source_content_sha256") != summary["source_content_sha256"]
            or any(receipt.get(name) != current[name] for name in ("all_tracked_files", "all_tracked_content_sha256"))
            or sha(directory / "RequestV1.json") != summary["request_sha256"]):
        raise ValueError("gate summary source/request differs from the frozen run")
    if record["sha256"] != hashlib.sha256(canonical({key: value for key, value in record.items() if key != "sha256"}).encode()).hexdigest():
        raise ValueError("controlled result content seal mismatch")
    prefix = "perovskite-sim/perovskite_sim/"
    package_files = {name.removeprefix(prefix): value for name, value in current["all_tracked_files"].items()
                     if name.startswith(prefix) and name.endswith(".py")}
    if record.get("source", {}).get("files") != package_files:
        raise ValueError("saved result equations differ from the frozen V8 package source")
    rows = [json.loads(line) for line in (directory / "AcceptedStepsV1.jsonl").read_text().splitlines()]
    schedule = []
    case = request["case"]
    for level in case["time_substeps"]:
        schedule.append((level, 0., 0.))
        for left, right in zip(case["times_s"][:-1], case["times_s"][1:]):
            dt = float(right - left) / level
            schedule.extend((level, float(left + step * dt), dt) for step in range(1, level + 1))
    if rows != record["accepted_steps"] or len(rows) != 794 or len(schedule) != 794:
        raise ValueError("gate summary requires the full matching 794-row execution")
    for row, expected in zip(rows, schedule):
        if (row["substeps"], row["time_s"], row["dt_s"]) != expected:
            raise ValueError("accepted-row schedule differs from the original request")
    result = {"schema": "R1V8OriginalGatesSummaryV1", "mode": summary["mode"],
        "source_commit": summary["source_commit"], "source_content_sha256": summary["source_content_sha256"],
        "request_sha256": summary["request_sha256"], "run_manifest_sha256": sha(directory / "ManifestV1.json"),
        "script_sha256": sha(__file__), "all_source_files_match_frozen_run": True,
        "P1_qualified": False, "formal_qualification": False, "full_100s_sentinel": False,
        "scope": "read_only_original_gate_aggregation_not_new_physics_replay_or_acceptance",
        "four_predicates_passed": summary.get("four_predicates_passed"), **aggregate(rows, record["certificate"])}
    if source_snapshot(summary["source_commit"]) != current:
        raise ValueError("source changed during gate aggregation")
    result["elapsed_s"] = time.perf_counter() - start
    output.mkdir(parents=True, exist_ok=False)
    write(output / "ResultV1.json", result)
    write(output / "ManifestV1.json", {"ResultV1.json": {"sha256": sha(output / "ResultV1.json"),
                                                      "bytes": (output / "ResultV1.json").stat().st_size}})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.run_dir, args.output)
    print(json.dumps({key: result[key] for key in ("rows", "failed_rows", "P1_qualified")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
