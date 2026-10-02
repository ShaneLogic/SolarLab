"""Real pair-preparation and saved-trajectory low-word tampering checks.

The two-row prefix is extracted from an actual accepted trajectory; it is not
integrated again or advertised as a complete experiment. Both negative inputs
are resealed and read back before the unmodified production verifier is called.
The saved coordinates and physical observations remain independent witnesses.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time


PROJECT = Path(__file__).resolve().parents[1]
LOW_FIELD = "precision_phi_V_lo"
HIGH_FIELD = "precision_phi_V_hi"
NODE = 1
DELTA_V = 1e-23
TARGET_ROW = 1
PREFIX_ROWS = 2


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def reseal(record):
    record["sha256"] = digest({key: value for key, value in record.items() if key != "sha256"})
    return record


def require_sealed(record):
    if record.get("sha256") != digest({key: value for key, value in record.items() if key != "sha256"}):
        raise ValueError("replay campaign input has an invalid content seal")


def mutation_contract():
    """Fixed before execution, including a nonzero low-word-only perturbation."""
    return {"schema": "R1V8LowWordMutationContractV1", "field": LOW_FIELD,
            "node": NODE, "delta_V": DELTA_V, "trajectory_row": TARGET_ROW,
            "prefix_rows": PREFIX_ROWS, "sign": "positive",
            "target_selection": "first_interior_node_of_preparation_and_first_finite_tier1_step",
            "preserved": ["high_word", "binary64_phi_V", "saved_solver_coordinate",
                          "independent_eliminated_input", "source", "request"],
            "synchronized": ["all_matching_direct_snapshot_copies", "accepted_state_arrays"],
            "resealed": ["record_sha256", "new_artifact_manifest"],
            "required_rejections": {
                "preparation": "pair common-state physical fields differ from deterministic reconstruction",
                "trajectory": "physical reconstruction mismatch: accepted state 1"}}


def _mutate_snapshot(snapshot, *, old=None, new=None):
    high = snapshot[HIGH_FIELD][NODE]
    low = snapshot[LOW_FIELD][NODE]
    if old is not None and low != old:
        raise ValueError("duplicate low-word representation disagrees before injection")
    changed = low + DELTA_V if new is None else new
    if changed == low or high + changed != high + low:
        raise ValueError("fixed low-word perturbation is not finite, nonzero, and binary64-invisible")
    snapshot[LOW_FIELD][NODE] = changed
    return low, changed


def _matching_snapshot_paths(value, original, path=()):
    """Locate exact repeated direct snapshots, not independently solved states."""
    if isinstance(value, dict):
        if value == original:
            yield path
            return
        for name, child in value.items():
            if name not in ("eliminated_precision", "independent_eliminated"):
                yield from _matching_snapshot_paths(child, original, (*path, name))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _matching_snapshot_paths(child, original, (*path, index))


def _at(value, path):
    for key in path:
        value = value[key]
    return value


def mutate_preparation(prepared):
    result = copy.deepcopy(prepared)
    old, new = _mutate_snapshot(result["state"])
    reseal(result)
    require_sealed(result)
    return result, {"old_low_V": old, "new_low_V": new,
                    "paths": [["state", LOW_FIELD, NODE]],
                    "seed_unchanged": result["seed_preparation"] == prepared["seed_preparation"],
                    "source_unchanged": result["source"] == prepared["source"]}


def bounded_prefix(record):
    """Keep actual first two rows under the original request's prefix API."""
    result = copy.deepcopy(record)
    rows = result["accepted_steps"][:PREFIX_ROWS]
    if (len(rows) != PREFIX_ROWS or rows[0]["phase"] != "0+"
            or rows[1]["phase"] != "accepted_regular_step"
            or rows[0]["substeps"] != rows[1]["substeps"]):
        raise ValueError("actual trajectory lacks the fixed initial/first-finite prefix")
    result["accepted_steps"] = rows
    result["accepted_state_arrays"] = {
        key: [row["state"][key] for row in rows] for key in rows[0]["state"]}
    for name in ("output_states", "regular_currents", "finite_step_averages", "charge_integral",
                 "failure", "persistence_failure"):
        result.pop(name, None)
    result["certificate"] = {"certified": False,
        "scope": record["scope"], "reasons": ["bounded_saved_prefix_not_complete_experiment"]}
    return reseal(result)


def mutate_trajectory(record):
    result = copy.deepcopy(record)
    original = copy.deepcopy(result["accepted_steps"][TARGET_ROW]["state"])
    paths = list(_matching_snapshot_paths(result, original))
    if ("accepted_steps", TARGET_ROW, "state") not in paths:
        raise ValueError("fixed trajectory snapshot is absent")
    old, new = original[LOW_FIELD][NODE], original[LOW_FIELD][NODE] + DELTA_V
    for path in paths:
        _mutate_snapshot(_at(result, path), old=old, new=new)
    arrays = result["accepted_state_arrays"]
    if arrays[LOW_FIELD][TARGET_ROW][NODE] not in (old, new):
        raise ValueError("accepted-state array disagrees before injection")
    arrays[LOW_FIELD][TARGET_ROW][NODE] = new
    # Equality explicitly proves that no duplicate-array mismatch can be the
    # rejection reason, even though the physical reconstruction runs earlier.
    expected_arrays = {key: [row["state"][key] for row in result["accepted_steps"]]
                       for key in result["accepted_steps"][0]["state"]}
    if arrays != expected_arrays:
        raise ValueError("mutation failed to synchronize accepted-state arrays")
    reseal(result)
    require_sealed(result)
    return result, {"old_low_V": old, "new_low_V": new,
        "paths": [list(path) + [LOW_FIELD, NODE] for path in paths]
                 + [["accepted_state_arrays", LOW_FIELD, TARGET_ROW, NODE]],
        "duplicated_arrays_synchronized": True,
        "source_unchanged": result["source"] == record["source"],
        "coordinate_unchanged": result["accepted_steps"][TARGET_ROW]["physics_reconstruction"]["coordinate"]
                                == record["accepted_steps"][TARGET_ROW]["physics_reconstruction"]["coordinate"]}


def save_case(directory, prepared, result, mutation):
    directory.mkdir(parents=True, exist_ok=False)
    write(directory / "PreparedV1.json", prepared)
    write(directory / "MutationV1.json", mutation)
    if result is not None:
        write(directory / "ResultV1.json", result)
        rows = result["accepted_steps"]
        (directory / "AcceptedStepsV1.jsonl").write_text("".join(canonical(row) + "\n" for row in rows))
    manifest = {path.name: {"sha256": sha(path), "bytes": path.stat().st_size}
                for path in sorted(directory.iterdir())}
    write(directory / "ManifestV1.json", manifest)
    # Normal disk round trip, including all duplicated rows and fresh seals.
    from scripts.analyze_r1_v7_prototype import verify_manifest
    verify_manifest(directory)
    saved_prepared = read(directory / "PreparedV1.json")
    require_sealed(saved_prepared)
    saved_result = None
    if result is not None:
        saved_result = read(directory / "ResultV1.json")
        require_sealed(saved_result)
        if [json.loads(line) for line in (directory / "AcceptedStepsV1.jsonl").read_text().splitlines()] != saved_result["accepted_steps"]:
            raise ValueError("mutation case observer copies are not synchronized")
    return saved_prepared, saved_result


def expect_reconstruction_rejection(call, expected):
    try:
        call()
    except Exception as exc:
        return {"rejected": True, "type": type(exc).__name__, "message": str(exc),
                "required_physical_rejection": str(exc) == expected,
                "hash_or_duplicate_rejection": "hash" in str(exc).lower() or "accepted state arrays" in str(exc)}
    return {"rejected": False, "required_physical_rejection": False,
            "hash_or_duplicate_rejection": False}


def run_campaign(run_dir, output, *, allow_development=False):
    start = time.perf_counter()
    from scripts.analyze_r1_v7_prototype import verify_manifest
    from scripts.run_r1_v7_prototype import source_snapshot
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_checkout import require_r1_checkout
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_precision import precision_context
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_precision_state import R1CommonStatePair
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from threadpoolctl import threadpool_limits

    run_dir, output = Path(run_dir).resolve(), Path(output).resolve()
    if output == run_dir or output.is_relative_to(run_dir):
        raise ValueError("negative evidence must not overwrite or extend the sealed source run")
    verify_manifest(run_dir)
    prepared, result, summary = [read(run_dir / name) for name in
                                 ("PreparedV1.json", "ResultV1.json", "SummaryV1.json")]
    receipt = read(run_dir / "SourceReceiptV1.json")
    if (summary.get("mode") != "compensated" or summary.get("source_unchanged") is not True
            or summary.get("request_sha256") != sha(run_dir / "RequestV1.json")
            or summary.get("source_commit") != receipt.get("source_commit")
            or summary.get("source_content_sha256") != receipt.get("r1_source_content_sha256")):
        raise ValueError("source run does not bind a completed compensated execution to its request and source receipt")
    require_sealed(prepared)
    require_sealed(result)
    rows = [json.loads(line) for line in (run_dir / "AcceptedStepsV1.jsonl").read_text().splitlines()]
    if rows != result["accepted_steps"]:
        raise ValueError("source run observer rows differ from the saved result")
    source = prepared["source"]
    # Prototype runs have an outer committed receipt; their ordinary in-process
    # preparation source can still truthfully say development/commit=None.
    # Anchor execution to the outer receipt rather than relabeling that source.
    context = require_r1_checkout(project=PROJECT, formal=not allow_development,
        source_commit=summary["source_commit"] if not allow_development else None,
        expected_source_sha256=summary["source_content_sha256"] if not allow_development else None)
    before_source = source_snapshot(summary["source_commit"]) if not allow_development else context.to_dict()
    output.mkdir(parents=True, exist_ok=False)
    write(output / "MutationContractV1.json", mutation_contract())
    write(output / "SourceIdentityV1.json", {"source_run": str(run_dir),
        "run_manifest_sha256": sha(run_dir / "ManifestV1.json"),
        "request_sha256": sha(run_dir / "RequestV1.json"),
        "source": source, "execution": before_source, "script_sha256": sha(__file__),
        "development_only": allow_development})
    prefix = bounded_prefix(result)
    bad_prepared, prep_change = mutate_preparation(prepared)
    bad_prefix, row_change = mutate_trajectory(prefix)
    healthy_prepared, healthy_prefix = save_case(output / "HealthyPrefixV1", prepared, prefix,
        {"operation": "actual_saved_prefix", "new_trajectory_integrated": False})
    saved_bad_prepared, _ = save_case(output / "PreparationMutationV1", bad_prepared, None, prep_change)
    _, saved_bad_prefix = save_case(output / "TrajectoryMutationV1", prepared, bad_prefix, row_change)
    stack = load_device_from_yaml(run_dir / "SourceFixtureV1.yaml")
    binding = read(run_dir / "ReferenceBindingV1.json")
    physics_start = time.perf_counter()
    with threadpool_limits(1), precision_context():
        from perovskite_sim.experiments import one_dimensional_mechanism_r1_state as states
        from perovskite_sim.experiments import one_dimensional_mechanism_r1_physics_validation as replay
        pair = R1CommonStatePair.from_dict(healthy_prepared)
        states.verify_prepared_physics(pair, stack, binding)
        healthy = replay.verify_r1_step_physics(stack, result["intervals"], binding, pair,
                                               healthy_prefix, allow_incomplete=True)
        contract = mutation_contract()["required_rejections"]
        prep_rejection = expect_reconstruction_rejection(
            lambda: states.verify_prepared_physics(R1CommonStatePair.from_dict(saved_bad_prepared), stack, binding),
            contract["preparation"])
        row_rejection = expect_reconstruction_rejection(
            lambda: replay.verify_r1_step_physics(stack, result["intervals"], binding, pair,
                                                   saved_bad_prefix, allow_incomplete=True),
            contract["trajectory"])
    healthy_passed = (healthy.get("evidence_matches_equations") is True
        and healthy.get("content_matches_recomputed") is True
        and healthy.get("physical_limits_satisfied") is True
        and healthy.get("checked_row_count") == PREFIX_ROWS)
    after_source = source_snapshot(summary["source_commit"]) if not allow_development else require_r1_checkout(project=PROJECT).to_dict()
    source_unchanged = before_source == after_source
    report = {"schema": "R1V8RealLowWordReplayV1", "elapsed_s": time.perf_counter() - start,
        "physics_elapsed_s": time.perf_counter() - physics_start,
        "timing_scope": "campaign_entry_through_checks_including_input_validation_and_mutation_persistence_before_final_report_write",
        "scope": "real_preparation_and_two_actual_saved_rows_under_original_prefix_replay_entry",
        "source_run_rows": len(rows), "checked_prefix_rows": PREFIX_ROWS,
        "new_trajectory_integrated": False, "source_unchanged": source_unchanged,
        "source_commit": summary["source_commit"], "request_sha256": sha(run_dir / "RequestV1.json"),
        "healthy_preparation_passed": True, "healthy_prefix": healthy,
        "preparation_mutation": prep_rejection, "trajectory_mutation": row_rejection,
        "development_only": allow_development,
        "passed": healthy_passed and source_unchanged and all(
            item["required_physical_rejection"] and not item["hash_or_duplicate_rejection"]
            for item in (prep_rejection, row_rejection)),
        "qualification": "replay_fault_requirement_only_not_full_P1"}
    write(output / "ResultV1.json", report)
    write(output / "ManifestV1.json", {path.relative_to(output).as_posix():
        {"sha256": sha(path), "bytes": path.stat().st_size} for path in sorted(output.rglob("*"))
        if path.is_file() and path != output / "ManifestV1.json"})
    return report


def main():
    start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_campaign(args.run_dir, args.output)
    print(json.dumps({**{key: report[key] for key in ("passed", "elapsed_s", "checked_prefix_rows")},
                      "entry_elapsed_s_including_final_report_writes": time.perf_counter() - start}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
