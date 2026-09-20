"""Verify historical R1 evidence and prepare unchanged, bounded V6 inputs.

This program reads old deliveries without rewriting them.  Temporary research
data needed by the independent comparison are materialized under the new
evidence output; third-party solver source is identified but is not copied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

BASE = "a71f1fa6adef719202913ef3e2fc4c8c9297f7f2"
COPY_DIRECTORIES = {"Spec", "ConclusionValidity", "TransFid", "ConventionAlignment",
                    "DifferenceBudget", "EquilibriumCrossCheck", "DeviceTranslation"}
EXCLUDED_DIRECTORIES = {"tree", "v5tree", "df", "im", "__pycache__", ".git", "ionmongermodel"}
FOUR_PROBES = (("D", 256, .1, 1), ("A", 256, .1, 4),
               ("D", 256, .1, 2), ("B", 256, .1, 4))
SEVEN = (("D", 64, 1., 1), ("D", 32, .01, 1), ("D", 64, .01, 1),
         ("D", 256, .01, 1), ("A", 256, .01, 4),
         ("D", 256, 1., 2), ("B", 256, 1., 4))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path):
    return {"sha256": sha(path), "bytes": Path(path).stat().st_size}


def verify_delivery(directory):
    manifest_path = directory / "ManifestV1.json"
    manifest = read(manifest_path)
    entries = manifest["files"]
    failures = []
    for name, expected in entries.items():
        path = directory / name
        if not path.is_file():
            failures.append({"file": name, "reason": "missing"})
        elif sha(path) != expected["sha256"]:
            failures.append({"file": name, "reason": "sha256_mismatch"})
        elif "bytes" in expected and path.stat().st_size != expected["bytes"]:
            failures.append({"file": name, "reason": "length_mismatch"})
    return {"directory": str(directory), "manifest": identity(manifest_path),
            "checked_files": len(entries), "failures": failures, "passed": not failures}


def case_key(spec):
    return (spec["control"], spec["intervals"], float(spec["nonlinear_factor"]),
            spec["time_substeps"][0])


def case_name(key):
    control, intervals, factor, level = key
    return f"{control}_N{intervals}_F{str(float(factor)).replace('.', 'p')}_T{level}"


def case_spec(key):
    control, intervals, factor, level = key
    return {"control": control, "intervals": intervals, "nonlinear_factor": factor,
            "time_substeps": [level, 2 * level, 4 * level], "amplitude_V": .005,
            "times_s": [0., 1e-9, 1e-8, 1e-6, 1e-4]}


def summarize_f01(out, ledger):
    available = dict(ledger)
    supplement = out / "F01MissingBaselineV1"
    if supplement.exists():
        for name, expected in read(supplement / "ManifestV1.json").items():
            if identity(supplement / name) != expected:
                raise ValueError("F0.1 supplemental manifest mismatch")
        summary = read(supplement / "Summary.json")
        if summary["source_commit"] != BASE or not summary["source_files_unchanged"]:
            raise ValueError("F0.1 supplement has a different source")
        for row in summary["cases"]:
            key = case_key(row["request"])
            if key in available:
                raise ValueError("F0.1 supplement repeats an existing formal case")
            available[key] = {**row, "directory": str(supplement / row["case"])}
    report = []
    for key in FOUR_PROBES:
        if key not in available:
            report.append({"case": case_name(key), "status": "not_run"})
            continue
        entry = available[key]
        directory = Path(entry["directory"])
        result = read(directory / "Result.json")
        certificate = result["certificate"]
        metrics = certificate["metrics"]
        gates = ("physical_current_spread_relative", "nonlinear_residual", "eliminated_operator_error")
        row = {"case": case_name(key), "directory": str(directory),
               "status": entry["status"], "accepted_rows": entry["accepted_rows"],
               "short_case_checks_passed": entry["short_case_checks_passed"],
               "result_identity": identity(directory / "Result.json"),
               "gates": {name: {"value": metrics[name], "limit": certificate["limits"][name],
                                  "fraction_of_limit": metrics[name] / certificate["limits"][name]}
                         for name in gates}}
        strict = (key[0], key[1], .01, key[3])
        if strict in ledger:
            other_path = Path(ledger[strict]["directory"]) / "Result.json"
            other = read(other_path)
            if result["times_s"] != other["times_s"] or result["source"] != other["source"]:
                raise ValueError("F0.1/F0.01 pair source or output times differ")
            currents = []
            for time_s, left, right in zip(result["times_s"], result["regular_currents"], other["regular_currents"]):
                a, b = left["contact_maxwell_A_m2"], right["contact_maxwell_A_m2"]
                currents.append({"time_s": time_s, "F01_contact_current_A_m2": a,
                                 "F001_contact_current_A_m2": b,
                                 "difference_A_m2": [x - y for x, y in zip(a, b)]})
            charges = result["charge_integral"]["regular_by_substeps_C_m2"]
            strict_charges = other["charge_integral"]["regular_by_substeps_C_m2"]
            row["strict_pair"] = {"case": case_name(strict), "result_identity": identity(other_path),
                "requested_output_currents": currents,
                "maximum_absolute_contact_current_difference_A_m2": max(abs(v) for pair in currents for v in pair["difference_A_m2"]),
                "regular_charge_difference_by_substeps_C_m2": {key: value - strict_charges[key] for key, value in charges.items()}}
        else:
            row["strict_pair"] = {"case": case_name(strict), "status": "not_in_V5_29_case_ledger"}
        report.append(row)
    write(out / "F01ProbeClosureV1.json", {"schema": "R1V6F01ProbeClosureV1", "source_commit": BASE,
          "cases": report, "all_four_formal_configurations_completed": all(row["status"] == "completed" for row in report),
          "scope": "formal short-run records and descriptive F01/F001 differences; no new error budget or long-window acceptance"})


def prepare(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    archive = args.archive.resolve()
    v5 = archive / "R1PhysicsDevelopmentV5"
    verification = {name: verify_delivery(archive / name) for name in (
        "R1PhysicsDevelopmentV5", "R1PhysicsDevelopmentV4", "R1PhysicsAcceptanceV4")}
    write(out / "HistoricalDeliveryVerificationV1.json", verification)
    if not all(result["passed"] for result in verification.values()):
        raise ValueError("historical delivery identity changed; see verification report")

    source = args.baseline_project.resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True)
    previous = read(v5 / "Root/FinalShardV1ACommand.json")["executed_file_hashes"]
    mismatches = [name for name, expected in previous.items()
                  if not (source.parent / name).is_file() or sha(source.parent / name) != expected]
    write(out / "V5SourceVerificationV1.json", {
        "project": str(source), "commit": commit, "expected_commit": BASE,
        "clean": not status, "git_status": status, "checked_source_files": len(previous),
        "mismatches": mismatches, "passed": commit == BASE and not status and not mismatches})
    if commit != BASE or status or mismatches:
        raise ValueError("the frozen V5 baseline differs from its final regression snapshot")

    ledger = {}
    for batch in sorted(v5.glob("Root/RecoveryBatch[123]V1")):
        summary = read(batch / "Summary.json")
        if summary["source_commit"] != BASE or not summary["source_files_unchanged"]:
            raise ValueError("V5 batch source mismatch")
        for item in summary["cases"]:
            key = case_key(item["request"])
            if key in ledger:
                raise ValueError("duplicate V5 ledger case")
            ledger[key] = {"case": item["case"], "directory": str(batch / item["case"]),
                           "request": item["request"], "status": item["status"],
                           "accepted_rows": item["accepted_rows"],
                           "short_case_checks_passed": item["short_case_checks_passed"],
                           "record": identity(batch / item["case"] / "RecordV1.json")}
    expected = read(v5 / "Root/RecoveryRequestV1.json")
    if set(ledger) != {case_key(row) for row in expected["cases"]} or len(ledger) != 29:
        raise ValueError("V5 29-case coverage does not match its sealed request")
    missing = [key for key in FOUR_PROBES if key not in ledger]
    write(out / "V5ShortCaseLedgerV1.json", {
        "source_commit": BASE, "case_count": len(ledger),
        "accepted_rows": sum(row["accepted_rows"] for row in ledger.values()),
        "all_completed": all(row["short_case_checks_passed"] for row in ledger.values()),
        "cases": list(ledger.values())})
    write(out / "F01CoverageV1.json", {
        "schema": "R1V6F01CoverageV1", "four_probes": [case_name(key) for key in FOUR_PROBES],
        "formally_covered": [ledger[key] for key in FOUR_PROBES if key in ledger],
        "missing_cases": [case_name(key) for key in missing],
        "reason": "Only the missing original configuration needs a new baseline run; F0.01 remains required.",
        "development_probe_record": str(archive / "R1CrossCodeReferenceV1/TierProbeV1/TierProbeResultV1.json")})
    requests = out / "Requests"
    for name, cases in (("SevenRepresentativesV1", [case_spec(key) for key in SEVEN]),
                        ("Short29V1", expected["cases"]),
                        ("F01MissingV1", [case_spec(key) for key in missing])):
        write(requests / (name + ".json"), {"schema": "R1V5CaseRequestV1", "cases": cases})
    for index in range(1, 4):
        shutil.copyfile(v5 / f"Root/RecoveryBatch{index}RequestV1.json", requests / f"ShortBatch{index}V1.json")
        remaining = [row for row in read(v5 / f"Root/RecoveryBatch{index}RequestV1.json")["cases"]
                     if case_key(row) not in SEVEN]
        write(requests / f"RemainingBatch{index}V1.json", {"schema": "R1V5CaseRequestV1", "cases": remaining})
    summarize_f01(out, ledger)
    shutil.copyfile(v5 / "Root/ResponseMethodologyV1.json", requests / "ResponseMethodologyV1.json")
    shutil.copyfile(v5 / "Root/FinalAffectedSelectionV1.json", out / "V5AffectedSelectionV1.json")
    shutil.copyfile(v5 / "Root/FinalShardPlanV1.json", out / "V5ShardPlanV1.json")

    crosscode = archive / "R1CrossCodeReferenceV1"
    archived = {str(path): identity(path) for path in sorted(crosscode.rglob("*")) if path.is_file()}
    # Existing /tmp records retain their V4/V5 identities; only copies are new.
    temporary = args.crosscode_temporary.resolve()
    inventory = {}
    for directory, children, files in os.walk(temporary):
        children[:] = [name for name in children if name not in EXCLUDED_DIRECTORIES]
        for name in sorted(files):
            path = Path(directory) / name
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(temporary)
            record = identity(path)
            copy = relative.parts[0] in COPY_DIRECTORIES or path.suffix in (".py", ".m", ".json", ".csv", ".log", ".txt", ".md")
            if copy:
                target = out / "HistoricalCrossCodeV1" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copyfile(path, target)
                if identity(target) != record:
                    raise ValueError("existing historical copy differs: " + str(target))
                record["preserved_file"] = str(target)
            record["preserved"] = copy
            inventory[str(path)] = record
    solver = {str(path): identity(path) for path in sorted((temporary / "df").rglob("*")) if path.is_file()}
    refs = {}
    for path in crosscode.rglob("*"):
        if path.suffix not in (".json", ".md"):
            continue
        for value in set(re.findall(r"/tmp/r1xc/[A-Za-z0-9_./+@%-]+", path.read_text())):
            candidate = Path(value.rstrip("."))
            refs.setdefault(str(candidate), {"exists": candidate.exists(), "mentioned_by": []})["mentioned_by"].append(str(path))
    write(out / "CrossCodeHistoricalIdentityV1.json", {
        "schema": "R1V6CrossCodeHistoricalIdentityV1", "archive_has_no_total_manifest": True,
        "archived_files": archived, "temporary_files": inventory, "driftfusion_source_files": solver,
        "references": refs, "copied_file_count": sum(row["preserved"] for row in inventory.values()),
        "identity_only_file_count": sum(not row["preserved"] for row in inventory.values()),
        "limitations": ["Path fragments in narrative text are retained as unresolved references, not fabricated files.",
                        "Third-party Driftfusion source remains external and is not copied.",
                        "Historical results retain their original source/version; this inventory grants no V6 result."]})
    print(json.dumps({"deliveries": {k: v["checked_files"] for k, v in verification.items()},
                      "v5_cases": len(ledger), "missing_F01": [case_name(k) for k in missing],
                      "crosscode_files": len(inventory), "copied": sum(r["preserved"] for r in inventory.values())}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--baseline-project", type=Path, required=True)
    parser.add_argument("--crosscode-temporary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
