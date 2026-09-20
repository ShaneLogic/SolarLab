"""Offline V8 two-side, initialization and regular-step evidence analysis.

The run is never changed and no trajectory is generated.  The external
precision-budget digest is mandatory unless already bound by the run summary.
Historical inputs may be inspected explicitly but never qualify as a final
V8 trajectory.  The tool reports its evidence subset and never grants P1.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path

from scripts.analyze_r1_v7_prototype import (
    canonical, compare_state_arithmetic, expected_schedule, find_budget, load,
    original_gate, sha, verify_manifest, write,
)
from scripts.verify_r1_v7_precision import _check_frozen, dec
from scripts.verify_r1_v8_precision import precision_fields, state_identity, verify_initial_arithmetic, verify_two_sides


PROJECT = Path(__file__).resolve().parents[1]
LATE_START_S = 0.7173200981104755
EVIDENCE_TOOLS = ("scripts/analyze_r1_v8_prototype.py", "scripts/verify_r1_v8_precision.py",
                  "scripts/analyze_r1_v7_prototype.py", "scripts/verify_r1_v7_precision.py")


def validate_budget(budget):
    expected = {
        ("local_state_arithmetic", "potential_qf_trace_absolute_error_V"): Decimal("1e-26"),
        ("local_state_arithmetic", "population_occupancy_relative_error"): Decimal("1e-27"),
        ("independent_side_operator", "symmetric_relative_target"): Decimal("1e-10"),
        ("independent_side_operator", "floor"): Decimal(1),
        ("independent_side_operator", "precision_stability_target"): Decimal("1e-20"),
        ("comparison", "original_gate"): Decimal("1e-6"),
    }
    for (section, key), value in expected.items():
        # JSON numeric literals preserve their declared decimal policy value;
        # binary coefficients in physical equations are handled separately.
        if Decimal(str(budget.get(section, {}).get(key))) != value:
            raise ValueError("frozen numerical target changed: " + section + "." + key)


def verify_inputs(directory, budget_file=None, budget_sha256=None, *, historical=False, evidence_tools=EVIDENCE_TOOLS):
    directory = Path(directory).resolve()
    manifest = verify_manifest(directory)
    names = ("RequestV1.json", "SummaryV1.json", "SourceReceiptV1.json", "FrozenHealthyMaterialV1.json",
             "PreparedV1.json", "ResultV1.json", "AcceptedStepsV1.jsonl", "SourceFixtureV1.yaml", "ReferenceBindingV1.json")
    if not set(names) <= set(manifest):
        raise ValueError("required run evidence is missing")
    request, summary, receipt, frozen, prepared, result = (load(directory / name) for name in names[:6])
    if summary.get("mode") != "compensated" or summary.get("source_unchanged") is not True:
        raise ValueError("analysis requires an unchanged compensated run source")
    if not historical and summary.get("schema") != "R1V8PrototypeRunV1":
        raise ValueError("historical run inputs require explicit diagnostic mode")
    if sha(directory / "RequestV1.json") != summary.get("request_sha256"):
        raise ValueError("request/summary identity mismatch")
    if (receipt.get("source_commit") != summary.get("source_commit")
            or receipt.get("r1_source_content_sha256") != summary.get("source_content_sha256")):
        raise ValueError("source receipt and summary disagree")
    tracked = receipt.get("all_tracked_files", {})
    if not tracked or hashlib.sha256(canonical(tracked).encode()).hexdigest() != receipt.get("all_tracked_content_sha256"):
        raise ValueError("source receipt content map mismatch")
    tool_matches = {name: sha(PROJECT / name) == tracked.get("perovskite-sim/" + name) for name in evidence_tools}
    if not historical and not all(tool_matches.values()):
        raise ValueError("final evidence tools must belong to the same frozen V8 source; historical inspection must be explicit")
    _check_frozen(frozen)
    if (frozen.get("source_identity") != summary.get("source_content_sha256")
            or summary.get("healthy_material_frozen_before_precision_context") is not True):
        raise ValueError("healthy material is not bound before fault/precision execution")
    for key, filename in (("fixture", "SourceFixtureV1.yaml"), ("reference", "ReferenceBindingV1.json")):
        if sha(directory / filename) != request["inputs"][key]["sha256"]:
            raise ValueError("input differs from request: " + key)
    budget_file = Path(budget_file) if budget_file is not None else find_budget(directory)
    expected = budget_sha256 or summary.get("precision_budget_sha256")
    if not expected or sha(budget_file) != expected:
        raise ValueError("an independently supplied/bound precision-budget digest is required")
    budget = load(budget_file)
    if budget.get("prototype_request_sha256") != summary["request_sha256"]:
        raise ValueError("precision budget is for another request")
    validate_budget(budget)
    schedule = expected_schedule(request["case"])
    if len(schedule) != 794 or request["execution"]["expected_total_rows"] != 794:
        raise ValueError("V8 normal analysis is scoped to the fixed 794-row request, not the ablation")
    return request, summary, frozen, prepared, result, budget, {
        "manifest_sha256": sha(directory / "ManifestV1.json"), "manifest_files_verified": len(manifest),
        "source_commit": summary["source_commit"], "r1_source_content_sha256": summary["source_content_sha256"],
        "request_sha256": summary["request_sha256"], "precision_budget_sha256": expected,
        "analysis_tools_match_run_source": tool_matches, "historical_input": historical,
        "final_source_eligible": not historical and all(tool_matches.values()),
        "scope": "sealed_run_integrity_and_recorded_source_binding"}


def initial_checks(frozen, prepared, result):
    envelope = result.get("initial_event", {}).get("precision_arithmetic_context", {})
    zero_minus = envelope.get("zero_minus_state", prepared.get("state"))
    zero_plus = result.get("physics_reconstruction", {}).get("initial_state")
    contexts = (envelope.get("zero_minus_context", envelope), envelope.get("zero_plus_context", {}))
    reports = {}
    for phase, state, upstream in zip(("zero_minus", "zero_plus"), (zero_minus, zero_plus), contexts):
        try:
            if state is None:
                raise ValueError("actual initial state missing")
            reports[phase] = verify_initial_arithmetic(frozen, state, upstream, phase=phase,
                                                      zero_minus_state=zero_minus if phase == "zero_plus" else None)
            identity_passed = envelope.get(phase + "_state_identity") == state_identity(state)
            if phase == "zero_plus":
                identity_passed = identity_passed and state_identity(envelope["zero_plus_state"]) == state_identity(state)
            reports[phase]["saved_initial_identity_passed"] = identity_passed
            reports[phase]["qualified"] = reports[phase]["qualified"] and identity_passed
        except Exception as exc:
            reports[phase] = {"qualified": False, "error": {"type": type(exc).__name__, "message": str(exc)}}
    return {"qualified": all(report["qualified"] for report in reports.values()), "phases": reports,
            "zero_plus_fields": None if zero_plus is None else precision_fields(zero_plus),
            "scope": "one_actual_zero_minus_and_zero_plus_three_tier_anchors_checked_by_identity"}


def analyze(directory, output, *, budget_file=None, budget_sha256=None, historical=False):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    if output == directory or output.is_relative_to(directory):
        raise ValueError("analysis cannot write into the sealed run")
    request, summary, frozen, prepared, raw_result, budget, identity = verify_inputs(
        directory, budget_file, budget_sha256, historical=historical)
    output.mkdir(parents=True, exist_ok=False)
    initial = initial_checks(frozen, prepared, raw_result)
    upstream = raw_result.get("initial_event", {}).get("precision_arithmetic_context", {})
    schedule = expected_schedule(request["case"])
    report = {"schema": "R1V8PrototypeOfflineAnalysisV1", "run_class": "development", "P1_qualified": False,
              "input_identity": identity, "initial_arithmetic": initial, "rows": 0, "expected_rows": 794,
              "state_arithmetic_checked_rows": 0, "initial_anchor_rows": 0, "late_rows": 0,
              "failed_rows": 0, "first_failure": None, "maxima": {"direct": {}, "eliminated": {}},
              "state_field_maxima": {}, "per_substeps": {}, "original_jacobian_checked_rows": 0,
              "execution_status_preserved": summary.get("execution_status"),
              "scope": "recorded_two_side_and_local_arithmetic_evidence_not_full_P1_or_global_error"}
    previous = None
    with (output / "RowSummaryV1.jsonl").open("w") as row_file, (output / "FailuresV1.jsonl").open("w") as failure_file, (
            output / "LateRowsV1.jsonl").open("w") as late_file, (directory / "AcceptedStepsV1.jsonl").open() as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            item = {"row": index, "time_s": row.get("time_s"), "substeps": row.get("substeps")}
            reasons, details = [], {}
            report["rows"] += 1
            tier = report["per_substeps"].setdefault(str(row.get("substeps")),
                {"rows": 0, "failed_rows": 0, "maxima": {"direct": {}, "eliminated": {}}, "state_field_maxima": {}})
            tier["rows"] += 1
            report["original_jacobian_checked_rows"] += row.get("physics_reconstruction", {}).get("jacobian_checked") is True
            try:
                if index >= len(schedule) or (row["substeps"], row["time_s"]) != schedule[index]:
                    raise ValueError("row sequence is not the exact declared request prefix")
                sides = verify_two_sides(frozen, row, upstream=upstream)
                item["sides_qualified"] = sides["qualified"]
                item["side_checks"] = {name: None if sides.get(name) is None else sides[name]["checks"]
                                       for name in ("direct", "eliminated")}
                item["upstream_construction"] = sides.get("upstream_construction")
                if not sides["qualified"]:
                    reasons.append("two_sides_failed_or_incomplete")
                    details["side_oracle"] = sides
                for name in ("direct", "eliminated"):
                    if sides.get(name) is None:
                        continue
                    for field, check in sides[name]["checks"].items():
                        for target in (report, tier):
                            current = target["maxima"][name].get(field)
                            if current is None or dec(check["relative"]) > dec(current["relative"]):
                                target["maxima"][name][field] = {"row": index, **check}
                item["original_gate"] = original_gate(row, budget["comparison"]["original_gate"])
                if not item["original_gate"]["passed"]:
                    reasons.append("original_path_gate_failed")
                if row["time_s"] == 0:
                    report["initial_anchor_rows"] += 1
                    same_anchor = canonical(precision_fields(row["state"])) == canonical(initial["zero_plus_fields"])
                    item["initial_anchor_identity_passed"] = same_anchor
                    if not same_anchor:
                        reasons.append("zero_plus_anchor_differs_from_actual_initial_state")
                else:
                    if previous is None:
                        raise ValueError("regular row has no actual previous fine state")
                    arithmetic = compare_state_arithmetic(previous, row, frozen, budget)
                    report["state_arithmetic_checked_rows"] += 1
                    item["state_arithmetic_passed"] = arithmetic["passed"]
                    if not arithmetic["passed"]:
                        reasons.append("regular_state_arithmetic_failed")
                        details["state_arithmetic"] = arithmetic
                    for field, check in arithmetic["fields"].items():
                        for target in (report, tier):
                            current = target["state_field_maxima"].get(field)
                            if current is None or dec(check["maximum_error"]) > dec(current["maximum_error"]):
                                target["state_field_maxima"][field] = {"row": index, **check}
            except Exception as exc:
                reasons.append("analysis_error")
                details["error"] = {"type": type(exc).__name__, "message": str(exc)}
            item.update(passed_local_checks=not reasons, reasons=reasons)
            row_file.write(canonical(item) + "\n")
            if type(row.get("time_s")) in (int, float) and row["time_s"] >= LATE_START_S:
                report["late_rows"] += 1
                late_file.write(canonical(item) + "\n")
            if reasons:
                report["failed_rows"] += 1
                tier["failed_rows"] += 1
                report["first_failure"] = report["first_failure"] or item
                failure_file.write(canonical({**item, "details": details, "saved_row": row,
                                              "previous_saved_state": None if previous is None else previous["state"]}) + "\n")
            previous = row
    report["row_count_matches_summary"] = report["rows"] == summary.get("extent", {}).get("accepted_rows")
    report["complete_trajectory_preserved"] = (report["rows"] == 794 and report["row_count_matches_summary"]
        and summary.get("execution_status") == "completed" and summary.get("four_predicates_passed") is True)
    report["prefix_only"] = not report["complete_trajectory_preserved"]
    report["observed_local_checks_passed"] = report["rows"] > 0 and report["failed_rows"] == 0
    report["arithmetic_evidence_subset_qualified"] = (identity["final_source_eligible"] and initial["qualified"]
        and report["complete_trajectory_preserved"] and report["observed_local_checks_passed"]
        and report["state_arithmetic_checked_rows"] == 791 and report["initial_anchor_rows"] == 3)
    verify_manifest(directory)
    write(output / "ResultV1.json", report)
    write(output / "ManifestV1.json", {path.name: {"sha256": sha(path), "bytes": path.stat().st_size}
                                        for path in sorted(output.iterdir()) if path.is_file()})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-file", type=Path)
    parser.add_argument("--budget-sha256")
    parser.add_argument("--allow-historical-input", action="store_true")
    args = parser.parse_args()
    result = analyze(args.run_dir, args.output, budget_file=args.budget_file,
                     budget_sha256=args.budget_sha256, historical=args.allow_historical_input)
    print(json.dumps({name: result[name] for name in (
        "rows", "failed_rows", "prefix_only", "arithmetic_evidence_subset_qualified", "P1_qualified")}, indent=2))
    return 0 if result["arithmetic_evidence_subset_qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
