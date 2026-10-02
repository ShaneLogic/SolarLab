"""Offline Decimal analysis of a sealed V7 trajectory or retained prefix.

No solver is run. Manifest integrity, request, source receipts, frozen healthy
material and the pre-run precision budget are checked before reading rows.
Initial 0+ rows are anchors, not fabricated previous-state reconstruction.
All conclusions remain development diagnostics; P1 is never granted here.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

from scripts.verify_r1_v7_precision import _check_frozen, dec, verify_side


EXPECTED_BUDGET_SHA256 = "6f5d02730bd682839e20896ba8cd55567d8f1b7edae93e4188bb54ffacfcd231"
LATE_START_S = 0.7173200981104755
VOLTAGE_FIELDS = ("phi_V", "dqfn_V", "dqfp_V", "trace_potential_V")
POPULATION_FIELDS = ("n_m3", "p_m3", "positive_m3", "occupancy", "trace_state_m3")
STATE_FIELDS = VOLTAGE_FIELDS + POPULATION_FIELDS


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError("nonfinite JSON token: " + value)))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify_manifest(directory):
    directory = Path(directory).resolve()
    manifest = load(directory / "ManifestV1.json")
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("run manifest must be a nonempty path map")
    for name, entry in manifest.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or name == "ManifestV1.json":
            raise ValueError("unsafe manifest path")
        path = directory / name
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(directory):
            raise ValueError("manifest file is missing or not a regular contained file: " + name)
        if path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
            raise ValueError("manifest content mismatch: " + name)
    actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*")
              if path.is_file() and path.name != "ManifestV1.json"}
    if actual != set(manifest):
        raise ValueError("run contains unmanifested or missing files")
    return manifest


def find_budget(directory):
    for parent in (Path(directory).resolve(), *Path(directory).resolve().parents):
        candidate = parent / "Contract/PrecisionBudgetV1.json"
        if candidate.is_file():
            return candidate
    raise ValueError("PrecisionBudgetV1.json not found; provide --budget-file")


def verify_run_inputs(directory, budget_path=None):
    directory = Path(directory).resolve()
    manifest = verify_manifest(directory)
    required = {"RequestV1.json", "SummaryV1.json", "SourceReceiptV1.json", "FrozenHealthyMaterialV1.json",
                "AcceptedStepsV1.jsonl", "SourceFixtureV1.yaml", "ReferenceBindingV1.json"}
    if not required <= set(manifest):
        raise ValueError("run lacks required identity/evidence files: " + str(sorted(required - set(manifest))))
    request, summary, receipt, frozen = (load(directory / name) for name in (
        "RequestV1.json", "SummaryV1.json", "SourceReceiptV1.json", "FrozenHealthyMaterialV1.json"))
    if summary.get("schema") != "R1V7PrototypeRunV1" or summary.get("mode") != "compensated":
        raise ValueError("only the compensated V7 prototype schema is supported")
    if sha(directory / "RequestV1.json") != summary.get("request_sha256"):
        raise ValueError("summary request hash does not match the preserved request")
    if summary.get("source_commit") != receipt.get("source_commit") or summary.get(
            "source_content_sha256") != receipt.get("r1_source_content_sha256"):
        raise ValueError("summary and source receipt identities disagree")
    if summary.get("source_unchanged") is not True:
        raise ValueError("run did not establish an unchanged execution source")
    tracked = receipt.get("all_tracked_files")
    if not isinstance(tracked, dict) or not tracked or hashlib.sha256(canonical(tracked).encode()).hexdigest() != receipt.get(
            "all_tracked_content_sha256"):
        raise ValueError("source receipt content map/hash mismatch")
    _check_frozen(frozen)
    if frozen.get("source_identity") != summary.get("source_content_sha256") or summary.get(
            "healthy_material_frozen_before_precision_context") is not True:
        raise ValueError("healthy material was not bound to the declared pre-fault execution source")
    for name, filename in (("fixture", "SourceFixtureV1.yaml"), ("reference", "ReferenceBindingV1.json")):
        if sha(directory / filename) != request["inputs"][name]["sha256"]:
            raise ValueError("preserved input differs from request: " + name)
    budget_path = Path(budget_path) if budget_path is not None else find_budget(directory)
    if sha(budget_path) != EXPECTED_BUDGET_SHA256:
        raise ValueError("precision budget differs from the pre-trajectory frozen budget")
    budget = load(budget_path)
    if budget.get("prototype_request_sha256") != summary["request_sha256"]:
        raise ValueError("precision budget belongs to another request")
    case = request["case"]
    expected = {str(level): 1 + (len(case["times_s"]) - 1) * level for level in case["time_substeps"]}
    if expected != request["execution"]["expected_rows_by_substeps"] or sum(expected.values()) != request[
            "execution"]["expected_total_rows"]:
        raise ValueError("request row counts do not match its time schedule")
    if sum(expected.values()) != 794:
        raise ValueError("this analyzer is scoped to the frozen 794-row request")
    return request, summary, frozen, budget, {
        "manifest_sha256": sha(directory / "ManifestV1.json"), "manifest_files_verified": len(manifest),
        "request_sha256": summary["request_sha256"], "source_commit": summary["source_commit"],
        "r1_source_content_sha256": summary["source_content_sha256"],
        "frozen_material_sha256": frozen["content_sha256"], "precision_budget_sha256": sha(budget_path),
        "scope": "manifest_integrity_and_cross_document_source_identity_not_new_execution_attestation"}


def _shape_flat(values):
    if not isinstance(values, list):
        return (), [dec(values)]
    if not values:
        return (0,), []
    rows = [_shape_flat(value) for value in values]
    if any(shape != rows[0][0] for shape, _ in rows):
        raise ValueError("ragged precision field")
    return (len(values), *rows[0][0]), [item for _, flat in rows for item in flat]


def precise_field(state, name):
    high, low = "precision_" + name + "_hi", "precision_" + name + "_lo"
    if high not in state or low not in state:
        raise ValueError("missing saved precision words: " + name)
    shape_h, h = _shape_flat(state[high])
    shape_l, l = _shape_flat(state[low])
    if shape_h != shape_l:
        raise ValueError("precision word shapes differ: " + name)
    return shape_h, [a + b for a, b in zip(h, l)]


def reconstruct_expected(previous, coordinate, frozen):
    """Recompute the recorded local rebase operation; no nonlinear solve."""
    with localcontext() as context:
        context.prec = 100
        result = {name: precise_field(previous, name)[1] for name in STATE_FIELDS}
        size = len(result["phi_V"])
        interior = size - 2
        interfaces = len(frozen["coefficients"]["interface_nodes"])
        active_faces = [j for j, value in enumerate(frozen["coefficients"]["D_ion_face_m2_s"]) if dec(value) > 0]
        active = sorted({node for face in active_faces for node in (face, face + 1)})
        if any(len(result[name]) != size for name in ("phi_V", "dqfn_V", "dqfp_V", "n_m3", "p_m3", "positive_m3")):
            raise ValueError("node fields have inconsistent lengths")
        if (len(result["occupancy"]) != interfaces or len(result["trace_potential_V"]) != 2 * interfaces
                or len(result["trace_state_m3"]) != 4 * interfaces):
            raise ValueError("interface fields have inconsistent lengths")
        z = [dec(value) for value in coordinate]
        ion_start = 2 * interior + interfaces
        phi_start = ion_start + len(active)
        trace_start = phi_start + interior
        if len(z) != trace_start + 6 * interfaces:
            raise ValueError("recorded coordinate does not match the frozen single-ion layout")
        vt = dec(frozen["constants"]["thermal_voltage_V"])
        for j, node in enumerate(range(1, size - 1)):
            dphi, dn, dp = z[phi_start + j], z[j], z[interior + j]
            result["phi_V"][node] += vt * dphi
            result["dqfn_V"][node] += vt * dn
            result["dqfp_V"][node] += vt * dp
            result["n_m3"][node] *= (dn + dphi).exp()
            result["p_m3"][node] *= (dp - dphi).exp()
        for j, node in enumerate(active):
            result["positive_m3"][node] *= z[ion_start + j].exp()
        for j in range(interfaces):
            old = result["occupancy"][j]
            if not Decimal(0) < old < Decimal(1):
                raise ValueError("occupancy outside the un-clipped logistic domain")
            exponent = z[2 * interior + j].exp()
            result["occupancy"][j] = old * exponent / (1 - old + old * exponent)
            start = trace_start + 6 * j
            for side in range(2):
                result["trace_potential_V"][2 * j + side] += vt * z[start + side]
            for carrier in range(4):
                result["trace_state_m3"][4 * j + carrier] *= z[start + 2 + carrier].exp()
        return result


def compare_state_arithmetic(previous_row, row, frozen, budget):
    with localcontext() as context:
        context.prec = 100
        if row["substeps"] != previous_row["substeps"] or not row["time_s"] > previous_row["time_s"]:
            raise ValueError("previous-state pairing crosses a tier or is not increasing")
        previous, current = previous_row["state"], row["state"]
        before_phi = precise_field(previous, "phi_V")[1]
        current_phi = precise_field(current, "phi_V")[1]
        if before_phi[0] != current_phi[0] or before_phi[-1] != current_phi[-1]:
            raise ValueError("nonzero boundary lift requires its own recorded arithmetic input")
        expected = reconstruct_expected(previous, row["physics_reconstruction"]["coordinate"], frozen)
        result = {}
        for name, wanted in expected.items():
            actual = precise_field(current, name)[1]
            if len(actual) != len(wanted):
                raise ValueError("state arithmetic shape mismatch: " + name)
            errors = [abs(a - b) for a, b in zip(actual, wanted)]
            if name in VOLTAGE_FIELDS:
                scores = errors
                limit = dec(budget["local_state_arithmetic"]["potential_qf_trace_absolute_error_V"])
                criterion = "absolute_V"
            else:
                scores = [error / max(abs(a), abs(b)) if a != 0 or b != 0 else Decimal(0)
                          for a, b, error in zip(actual, wanted, errors)]
                limit = dec(budget["local_state_arithmetic"]["population_occupancy_relative_error"])
                criterion = "symmetric_relative_no_arbitrary_floor"
            worst = max(range(len(scores)), key=scores.__getitem__) if scores else None
            maximum = Decimal(0) if worst is None else scores[worst]
            entry = {"criterion": criterion, "maximum_error": str(maximum), "limit": str(limit),
                     "passed": maximum <= limit, "worst_flat_index": worst,
                     "maximum_absolute_error": str(max(errors, default=Decimal(0)))}
            if not entry["passed"]:
                entry.update(actual=[str(x) for x in actual], expected=[str(x) for x in wanted],
                             absolute_errors=[str(x) for x in errors])
            result[name] = entry
        return {"status": "checked", "passed": all(item["passed"] for item in result.values()), "fields": result}


def observed_words(state):
    result = {}
    for name in ("positive_flux_m2_s", "positive_rate_m3_s", "boundary_flux_m2_s"):
        high, low = "precision_" + name + "_hi", "precision_" + name + "_lo"
        if high in state and low in state:
            result[name] = {"hi": state[high], "lo": state[low]}
        elif high in state or low in state:
            raise ValueError("incomplete observed precision words: " + name)
    return result


def original_gate(row, limit):
    evidence = row.get("physics_reconstruction", {})
    components, complete = {}, True
    for name in ("positive_ion_flux", "positive_ion_rate"):
        item = evidence.get("eliminated_operator", {}).get(name)
        if not item:
            complete = False
            continue
        absolute, scale, relative = (dec(item[key]) for key in (
            "maximum_absolute_difference", "normalization_scale", "relative_error"))
        if absolute < 0 or scale <= 0 or relative < 0:
            raise ValueError("invalid original gate numerator or denominator")
        if not math.isclose(float(absolute / scale), float(relative), rel_tol=1e-12, abs_tol=0):
            raise ValueError("stored original gate ratio contradicts its numerator/denominator")
        components[name] = {"absolute": str(absolute), "scale": str(scale), "relative": str(relative),
                            "floor": item.get("normalization_floor"), "units": item.get("unit"),
                            "passed": relative <= dec(limit)}
    overall = evidence.get("eliminated_operator_error")
    if overall is None:
        complete = False
    return {"complete": complete, "overall": overall, "components": components,
            "passed": complete and dec(overall) <= dec(limit) and all(x["passed"] for x in components.values())}


def expected_schedule(case):
    result = []
    for level in case["time_substeps"]:
        result.append((level, float(case["times_s"][0])))
        for left, right in zip(case["times_s"], case["times_s"][1:]):
            dt = float(right - left) / level
            result.extend((level, float(left + step * dt)) for step in range(1, level + 1))
    return result


def analyze(directory, output, *, budget_path=None):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    if output == directory or output.is_relative_to(directory):
        raise ValueError("analysis output must be outside the sealed run directory")
    request, summary, frozen, budget, identity = verify_run_inputs(directory, budget_path)
    schedule = expected_schedule(request["case"])
    output.mkdir(parents=True, exist_ok=False)
    result = {"schema": "R1V7PrototypeOfflineAnalysisV1", "run_class": "development", "P1_qualified": False,
              "formal_qualification": False, "input_identity": identity,
              "execution_status_preserved": summary.get("execution_status"), "rows": 0,
              "expected_rows": request["execution"]["expected_total_rows"], "initial_anchor_rows": 0,
              "state_arithmetic_checked_rows": 0, "late_rows": 0, "failed_rows": 0,
              "first_failure": None, "maxima": {}, "state_field_maxima": {},
              "missing_boundary_rows": 0, "source_analyzer_sha256": sha(__file__),
              "scope": "per_saved_state_local_arithmetic_and_internal_operators_not_global_error_or_new_trajectory"}
    previous = None
    with (output / "RowSummaryV1.jsonl").open("w") as rows_out, (output / "FailuresV1.jsonl").open("w") as failures_out, (
            output / "LateRowsV1.jsonl").open("w") as late_out, (directory / "AcceptedStepsV1.jsonl").open() as stream:
        for index, line in enumerate(stream):
            row = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite row " + value)))
            result["rows"] += 1
            record = {"row": index, "time_s": row.get("time_s"), "substeps": row.get("substeps")}
            reasons, detail = [], {}
            try:
                if index >= len(schedule) or (row["substeps"], row["time_s"]) != schedule[index]:
                    raise ValueError("row is not the next exact point in the frozen request prefix")
                with localcontext() as context:
                    context.prec = 100
                    side = verify_side(frozen, row["state"], observed_words(row["state"]))
                    record["oracle"] = {name: check for name, check in side["checks"].items()}
                    record["poisson_maximum_absolute_C_m2"] = side["poisson"]["maximum_absolute_C_m2"]
                    record["boundary_independently_measured"] = "boundary_flux_m2_s" not in side["missing_observed"]
                    if not record["boundary_independently_measured"]:
                        result["missing_boundary_rows"] += 1
                        reasons.append("boundary_observation_missing_requires_separate_evidence")
                    if not side["qualified"]:
                        reasons.append("independent_side_oracle_failed_or_incomplete")
                        detail["side_oracle"] = side
                    for name, check in side["checks"].items():
                        current = result["maxima"].get(name)
                        if current is None or dec(check["relative"]) > dec(current["relative"]):
                            result["maxima"][name] = {"row": index, **check}
                    record["original_gate"] = original_gate(row, budget["comparison"]["original_gate"])
                    if not record["original_gate"]["passed"]:
                        reasons.append("original_gate_failed_or_missing")
                    if row["time_s"] == 0:
                        result["initial_anchor_rows"] += 1
                        arithmetic = {"status": "initial_anchor_no_previous_row", "passed": None}
                    else:
                        if previous is None:
                            raise ValueError("regular row has no previous fine state")
                        arithmetic = compare_state_arithmetic(previous, row, frozen, budget)
                        result["state_arithmetic_checked_rows"] += 1
                        if not arithmetic["passed"]:
                            reasons.append("local_state_arithmetic_budget_exceeded")
                            detail["state_arithmetic"] = arithmetic
                        for name, check in arithmetic["fields"].items():
                            current = result["state_field_maxima"].get(name)
                            if current is None or dec(check["maximum_error"]) > dec(current["maximum_error"]):
                                result["state_field_maxima"][name] = {"row": index, **check}
                    record["state_arithmetic"] = {key: arithmetic[key] for key in ("status", "passed")}
            except Exception as exc:
                reasons.append("row_analysis_error")
                detail["error"] = {"type": type(exc).__name__, "message": str(exc)}
            record["passed_local_checks"] = not reasons
            record["reasons"] = reasons
            rows_out.write(canonical(record) + "\n")
            if type(row.get("time_s")) in (int, float) and row["time_s"] >= LATE_START_S:
                result["late_rows"] += 1
                late_out.write(canonical(record) + "\n")
            if reasons:
                result["failed_rows"] += 1
                if result["first_failure"] is None:
                    result["first_failure"] = record
                failures_out.write(canonical({**record, "details": detail, "saved_row": row,
                                               "previous_saved_state": None if previous is None else previous["state"]}) + "\n")
            previous = row
    result["row_count_matches_summary"] = result["rows"] == summary.get("extent", {}).get("accepted_rows")
    result["full_request_observed"] = result["rows"] == result["expected_rows"]
    result["completed_trajectory_preserved"] = (summary.get("execution_status") == "completed"
                                                and summary.get("four_predicates_passed") is True
                                                and result["full_request_observed"] and result["row_count_matches_summary"])
    result["prefix_only"] = not result["completed_trajectory_preserved"]
    result["all_observed_local_checks_passed"] = result["rows"] > 0 and result["failed_rows"] == 0
    result["boundary_observations_complete"] = result["rows"] > 0 and result["missing_boundary_rows"] == 0
    result["initial_arithmetic_limit"] = "0+ anchors have no preceding saved fine row; initialization needs its own evidence"
    result["run_inputs_unchanged_during_analysis"] = sha(directory / "ManifestV1.json") == identity["manifest_sha256"]
    # Revalidate every listed file, not only the manifest's bytes, on exit.
    verify_manifest(directory)
    write(output / "ResultV1.json", result)
    write(output / "ManifestV1.json", {path.name: {"sha256": sha(path), "bytes": path.stat().st_size}
                                        for path in sorted(output.iterdir()) if path.is_file()})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--budget-file", type=Path)
    args = parser.parse_args()
    result = analyze(args.run_dir, args.output, budget_path=args.budget_file)
    print(json.dumps({key: result[key] for key in (
        "rows", "expected_rows", "prefix_only", "failed_rows", "all_observed_local_checks_passed", "P1_qualified")}, indent=2))
    return 0 if result["all_observed_local_checks_passed"] and result["row_count_matches_summary"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
