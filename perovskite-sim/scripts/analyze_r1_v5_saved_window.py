"""Inspect an already saved window without integrating or changing its gates."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    attempt = args.attempt.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((attempt / "ManifestV1.json").read_text())
    hashes = {}
    for name, expected in manifest.items():
        path = (attempt / name).resolve()
        if not path.is_relative_to(attempt):
            raise ValueError("manifest path leaves attempt")
        hashes[name] = digest(path)
        if hashes[name] != expected:
            raise ValueError("saved window digest mismatch: " + name)
    audit = json.loads((attempt / "FailedPrefixPhysicsAuditV1.json").read_text())
    request = json.loads((attempt / "RequestV1.json").read_text())
    completion = json.loads((attempt / "CompletionV1.json").read_text())
    witness = json.loads((attempt / "FailureWitnessV1.json").read_text())
    violations = audit["physical_limit_violations"]
    indices = {entry["row"] for entry in violations}
    selected = {0, audit["checked_row_count"] - 1}
    if indices:
        first = min(indices)
        worst = max(violations, key=lambda entry: entry["value"])["row"]
        selected.update({max(0, first - 1), first, first + 1, max(0, worst - 1), worst, worst + 1})
    maximum = {}
    counts = Counter()
    by_level = {}
    details = []
    selected_rows = []
    actual_violations = []
    first_failure = None
    total = 0
    for index, line in enumerate((attempt / "AcceptedStepsV1.jsonl").open()):
        row = json.loads(line)
        total += 1
        components = row["physics_reconstruction"]["eliminated_operator"]
        for name, value in components.items():
            if name not in maximum or value["relative_error"] > maximum[name]["relative_error"]:
                maximum[name] = {key: value[key] for key in (
                    "relative_error", "maximum_absolute_difference", "normalization_scale", "normalization_floor", "floor_active", "unit")}
                maximum[name].update(row=index, time_s=row["time_s"], substeps=row["substeps"])
        peak = max(value["relative_error"] for value in components.values())
        if peak > audit["limits"]["eliminated_operator_error"]:
            actual_violations.append({"row": index, "metric": "eliminated_operator_error",
                                      "limit": audit["limits"]["eliminated_operator_error"], "value": peak})
            first_failure = first_failure or {"row": index, "time_s": row["time_s"], "substeps": row["substeps"]}
            for name, value in components.items():
                if value["relative_error"] > audit["limits"]["eliminated_operator_error"]:
                    counts[name] += 1
            by_level[str(row["substeps"])] = by_level.get(str(row["substeps"]), 0) + 1
        if index in selected:
            selected_rows.append({"original_line_1based": index + 1, "record_index": index, "row": row})
            failing = {name: value for name, value in components.items()
                       if value["relative_error"] > audit["limits"]["eliminated_operator_error"]}
            details.append({"row": index, "time_s": row["time_s"], "substeps": row["substeps"],
                            "solver_accepted": row["solver_accepted"], "components_exceeding_limit": failing,
                            "reported_physical": row["physical"],
                            "operator_error": peak})
    if total != audit["checked_row_count"] or total != audit["expected_row_count"]:
        raise ValueError("saved rows and audit count differ")
    if actual_violations != violations:
        raise ValueError("saved component violations differ from independently archived audit")
    args.output.mkdir(parents=True)
    report = {"schema": "R1V5SavedWindowDiagnosisV1", "analysis": "saved_component_audit_no_new_physical_solves",
              "source_attempt": str(attempt), "source_manifest_sha256": digest(attempt / "ManifestV1.json"),
              "source_files_sha256": hashes, "script_sha256": digest(Path(__file__)),
              "request": request, "recorded_status": completion["status"],
              "checked_rows": total, "violation_rows": len(actual_violations),
              "first_failure": first_failure, "violations_by_component": dict(counts),
              "violations_by_substeps": by_level, "component_maxima": maximum, "selected": details,
              "last_persisted_row": witness["last_persisted_row"],
              "terminal_response_error_budget": "unknown",
              "limits_changed": False,
              "scope": "verifies saved arithmetic and internal manifest; source attestation remains the original review evidence"}
    write(args.output / "DiagnosisV1.json", report)
    write(args.output / "SelectedRowsV1.json", selected_rows)
    write(args.output / "SourceManifestV1.json", manifest)
    print(json.dumps({"checked_rows": total, "violation_rows": len(actual_violations),
                      "first_failure": first_failure, "violations_by_component": dict(counts),
                      "violations_by_substeps": by_level, "component_maxima": maximum}, indent=2))


if __name__ == "__main__":
    main()
