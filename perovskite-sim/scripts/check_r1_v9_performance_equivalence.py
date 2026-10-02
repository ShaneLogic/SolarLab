"""Frozen saved-point numerical equivalence for the V9 overhead changes.

No time integration or profiling is performed. The original saved solve
summary is compared; unsaved iteration traces are not inferred or invented.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import struct
import time

import numpy as np
from threadpoolctl import threadpool_limits

from scripts.analyze_r1_v7_prototype import sha, write
from scripts.check_r1_v8_precision_consumers import reconstruct_selected, select_rows, snapshot
from scripts.check_r1_v9_precision_consumers import (
    actual_sides, campaign, current_source, load_contract, load_runtime, runtime_identity,
)
from scripts.verify_r1_v7_precision import freeze_healthy_material
from scripts.verify_r1_v8_precision import digest, precision_fields

CONTRACT_SHA = "f4439235dcdd0348ed10954a912fb652646f0ca5e8467a68281288e2db7e5df0"
PARENT_COMMIT = "ceec959a90d219c4ced06485c6d13ee734b685ad"
PARENT_CASE_MANIFEST = "a2ac0c4bde71a99ae3350c66c65714f774931b35c647e808c34a55aeddbf515a"
FIXED_ROWS = [0, 341, 342, 791, 205, 338, 655]
FINE_FIELDS = frozenset((
    "boundary_flux_m2_s", "dqfn_V", "dqfp_V", "electron_current_A_m2",
    "hole_current_A_m2", "n_m3", "occupancy", "p_m3", "phi_V",
    "poisson_residual_C_m2", "positive_flux_m2_s", "positive_m3",
    "positive_rate_m3_s", "sheet_charge_C_m2", "storage",
    "trace_potential_V", "trace_state_m3",
))


def bit_tree(value):
    """Type/shape-sensitive JSON identity retaining every binary64 bit."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("nonfinite value in numerical equivalence evidence")
        return ["binary64", struct.pack(">d", value).hex()]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["integer", str(value)]
    if isinstance(value, str) or value is None:
        return [type(value).__name__, value]
    if isinstance(value, list):
        return ["list", [bit_tree(x) for x in value]]
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise TypeError("equivalence dictionaries require string keys")
        return ["mapping", [[k, bit_tree(value[k])] for k in sorted(value)]]
    raise TypeError("unsupported equivalence evidence type: " + type(value).__name__)


def bit_digest(value):
    return hashlib.sha256(json.dumps(bit_tree(value), separators=(",", ":")).encode()).hexdigest()


def compare(actual, expected):
    observed, reference = bit_digest(actual), bit_digest(expected)
    return {"bit_identical": observed == reference, "actual_bit_sha256": observed,
            "parent_bit_sha256": reference}


def fine_words(value):
    fields = precision_fields(value)
    if set(fields) != FINE_FIELDS:
        raise ValueError("saved-point comparison requires exactly all 17 fine fields")
    return fields


def source_identity():
    value = current_source()
    value["equivalence_checker_sha256"] = sha(Path(__file__))
    return value


def verify_case_manifest(directory):
    """Verify the V9 case inventory, including its nested analysis manifest."""
    directory = Path(directory).resolve()
    manifest_path = directory / "ManifestV1.json"
    entries = json.loads(manifest_path.read_text())
    if not isinstance(entries, dict) or not entries:
        raise ValueError("case manifest must be a nonempty path map")
    for name, entry in entries.items():
        relative = PurePosixPath(name)
        path = directory / name
        if relative.is_absolute() or ".." in relative.parts or name == "ManifestV1.json":
            raise ValueError("unsafe case manifest path")
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(directory):
            raise ValueError("case manifest file is missing or unsafe: " + name)
        if path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
            raise ValueError("case manifest content mismatch: " + name)
    actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*")
              if path.is_file() and path != manifest_path}
    if actual != set(entries):
        raise ValueError("case contains unmanifested or missing files")
    return entries


def read_parent(directory, contract_path, expected_sha256):
    directory, contract_path = Path(directory), Path(contract_path)
    if expected_sha256 != CONTRACT_SHA or sha(contract_path) != CONTRACT_SHA:
        raise ValueError("performance contract differs from the frozen identity")
    contract = json.loads(contract_path.read_text())
    if (contract.get("schema") != "R1V9PerformanceContractV1"
            or contract.get("fixed_rows") != FIXED_ROWS
            or contract.get("parent_source_commit") != PARENT_COMMIT):
        raise ValueError("performance comparison scope differs from its frozen contract")
    parent_root = directory.parents[1]
    outer = parent_root / "ManifestV1.json"
    if sha(outer) != contract["parent_manifest_sha256"]:
        raise ValueError("parent delivery manifest differs from the contract")
    if sha(directory / "ManifestV1.json") != PARENT_CASE_MANIFEST:
        raise ValueError("parent pair case manifest differs from its frozen identity")
    verify_case_manifest(directory)
    for name, expected in contract["fixed_parent_case_files"].items():
        if sha(directory / name) != expected:
            raise ValueError("parent fixed file differs from contract: " + name)
    read = lambda name: json.loads((directory / name).read_text())
    result, request = read("ResultV1.json"), read("RequestV1.json")
    rows = [json.loads(line) for line in (directory / "AcceptedStepsV1.jsonl").read_text().splitlines()]
    # This is whole-file integrity, not the bounded numerical comparison below.
    # Canonical JSON preserves the serialized binary64 values without expanding
    # all 794 rows into the tagged, per-float comparison representation.
    if digest(rows) != digest(result["accepted_steps"]):
        raise ValueError("parent accepted rows disagree with the sealed result")
    near_path = parent_root / "Root/FinalNearSharedV1/ResultV1.json"
    outer_entries = json.loads(outer.read_text())
    if sha(near_path) != outer_entries[near_path.relative_to(parent_root).as_posix()]["sha256"]:
        raise ValueError("parent near-threshold report differs from its delivery manifest")
    return {"contract": contract, "request": request, "result": result, "rows": rows,
            "prepared": read("PreparedV1.json"), "frozen": read("FrozenHealthyMaterialV1.json"),
            "near_reference": json.loads(near_path.read_text()),
            "near_reference_sha256": sha(near_path)}


def side_projection(side):
    """Only explicit numerical state/operator evidence; execution context is separate."""
    return {"actual_row": side["actual_row"],
            "original_ion_components": side["original_ion_components"]}


def near_projection(value):
    return {
        "healthy": {key: side_projection(row["result"]) for key, row in value["healthy"].items()},
        "near_threshold": [{"case": row["case"], "multiplier_binary64": row["multiplier_binary64"],
                            "actual_diffusion_m2_s": row["actual_diffusion_m2_s"],
                            "result": side_projection(row["result"]), "detected": row["detected"]}
                           for row in value["near_threshold"]],
        "shared_input": [{"case": row["case"], "multiplier_binary64": row["multiplier_binary64"],
                          "actual_diffusion_m2_s": row["actual_diffusion_m2_s"],
                          "result": side_projection(row["result"]),
                          "expected_behavior_passed": row["expected_behavior_passed"]}
                         for row in value["shared_input"]],
        "floor_specimens": value["floor_specimens"],
    }


def evaluate_points(directory, parent, validation, context):
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import json_data

    system, minus, event, preparation = load_runtime(
        directory, parent["request"], parent["prepared"], historical=True)
    frozen, rows = parent["frozen"], parent["rows"]
    upstream = system.precision_arithmetic_context()
    actual_material = freeze_healthy_material(system.grid, system.material,
                                             source_identity=frozen["source_identity"])
    material_check = compare(actual_material, frozen)
    if not material_check["bit_identical"]:
        raise ValueError("fixed material differs from the parent case")
    event_check = compare(json_data(event.event), parent["result"]["initial_event"])
    points = []
    for index in FIXED_ROWS:
        row = rows[index]
        evidence = row["physics_reconstruction"]
        voltage = evidence["voltage_V"]
        if row["phase"] == "0+":
            owner, state = event.system, event.zero_plus
            previous_row = None
            residual = None
            input_state = {"previous": None, "reference": snapshot(state),
                           "coordinate": evidence["coordinate"], "dt_s": row["dt_s"],
                           "voltage_V": voltage, "material": frozen, "scales": None}
            previous_check = None
        else:
            previous_row = rows[index - 1]
            if previous_row["substeps"] != row["substeps"] or previous_row["time_s"] >= row["time_s"]:
                raise ValueError("frozen point has no exact same-tier physical previous")
            owner, state, _ = reconstruct_selected(event.system, previous_row, row, require_exact=True)
            previous_check = compare(fine_words(snapshot(owner._step_reference)), fine_words(previous_row["state"]))
            scales = tuple(np.asarray(evidence[key]) for key in (
                "storage_scale", "poisson_scale_C_m2", "local_algebraic_scale"))
            input_state = {"previous": previous_row["state"],
                           "reference": {key: {"hi": value.hi.tolist(), "lo": value.lo.tolist()}
                                         for key, value in owner._fine_reference.items()},
                           "coordinate": evidence["coordinate"], "dt_s": row["dt_s"],
                           "voltage_V": voltage, "material": frozen,
                           "scales": {key: evidence[key] for key in (
                               "storage_scale", "poisson_scale_C_m2", "local_algebraic_scale")}}
            residual, _, state = owner.residual_and_jacobian(
                np.asarray(evidence["coordinate"]), voltage, owner._step_reference, row["dt_s"], *scales)
            residual = residual.tolist()
        sides = actual_sides(owner, state, voltage, frozen, upstream, context)
        actual_record = sides["actual_row"]["physics_reconstruction"]["eliminated_precision"]
        # Capture the complete original operator channels in the same fixed state.
        operators = json_data(dict(owner.eliminated_operator_diagnostics(state, voltage)))
        checks = {"all_17_fine_fields": compare(fine_words(snapshot(state)), fine_words(row["state"])),
                  "coordinate": compare(state.coordinate.tolist(), evidence["coordinate"]),
                  "eliminated_operator": compare(operators, evidence["eliminated_operator"]),
                  "independent_solve_saved_summary": compare(actual_record["solve"], evidence["eliminated_precision"]["solve"]),
                  "complete_eliminated_record": compare(actual_record, evidence["eliminated_precision"])}
        if residual is not None:
            checks["complete_scaled_residual_vector"] = compare(residual, evidence["scaled_residual_vector"])
            checks["physical_previous_17_fine_fields"] = previous_check
        else:
            checks["initial_event_at_zero_plus"] = event_check
        points.append({"row": index, "substeps": row["substeps"], "time_s": row["time_s"],
                       "fixed_inputs": input_state, "fixed_inputs_bit_sha256": bit_digest(input_state),
                       "checks": checks, "passed": all(x["bit_identical"] for x in checks.values())
                       and sides["verification"]["qualified"],
                       "actual_fine_fields": fine_words(snapshot(state)),
                       "actual_scaled_residual_vector": residual,
                       "actual_eliminated_operator": operators, "actual_eliminated_record": actual_record,
                       "independent_verification": sides["verification"],
                       "residual_scope": "saved_finite_step_scales" if residual is not None else "undefined_at_zero_plus"})
    near = campaign(event.system, system, select_rows(directory), frozen, validation,
                    context=context, require_exact=True)
    near_check = compare(near_projection(near), near_projection(parent["near_reference"]))
    return {"points": points, "material_check": material_check, "initial_event_check": event_check,
            "actual_preparation_sha256": preparation, "near_shared_campaign": near,
            "near_shared_parent_comparison": near_check,
            "passed": event_check["bit_identical"] and all(x["passed"] for x in points)
            and near["passed"] and near_check["bit_identical"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", default=CONTRACT_SHA)
    parser.add_argument("--validation-contract", type=Path, required=True)
    args = parser.parse_args()
    directory, output = args.run_dir.resolve(), args.output.resolve()
    if output == directory or output.is_relative_to(directory):
        raise ValueError("output must be outside the sealed parent case")
    parent = read_parent(directory, args.contract, args.contract_sha256)
    validation = load_contract(args.validation_contract)
    output.mkdir(parents=True, exist_ok=False)
    before, start = source_identity(), time.monotonic()
    context = {"parent_source_commit": PARENT_COMMIT, "parent_manifest_sha256": PARENT_CASE_MANIFEST,
               "performance_contract_sha256": CONTRACT_SHA, "execution_source": before["source_files_sha256"]}
    report = {"schema": "R1V9PerformanceSavedPointEquivalenceV1", "passed": False,
              "context": context, "parent_near_reference_sha256": parent["near_reference_sha256"],
              "new_trajectory_steps": 0, "new_target_point_Newton_iterations": 0,
              "preparation_scope": "fresh_initial_constraint_reconstruction_may_iterate",
              "solve_trace_scope": "original_saved_summary_only_no_unsaved_iteration_trace_claim",
              "copy_campaign_required_separately": True, "P2_qualified": False}
    try:
        with threadpool_limits(1):
            report["runtime_identity"] = runtime_identity()
            report.update(evaluate_points(directory, parent, validation, context))
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    after = source_identity()
    report.update(source_before=before, source_after=after, source_unchanged=before == after,
                  wall_seconds=time.monotonic() - start)
    report["passed"] &= before == after
    write(output / "ResultV1.json", report)
    write(output / "ManifestV1.json", {"ResultV1.json": {"sha256": sha(output / "ResultV1.json"),
                                                        "bytes": (output / "ResultV1.json").stat().st_size}})
    print(json.dumps({key: report.get(key) for key in ("passed", "error", "source_unchanged", "wall_seconds")}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
