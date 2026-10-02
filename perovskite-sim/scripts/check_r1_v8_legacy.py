"""V8 audit entry for the original V5 934-row binary64 saved trajectory.

This script deliberately loads the original clean V5 package in a child
interpreter. The full original physical replay runs once, followed by one real
same-tier stale-predecessor injection into its rebase call. No transient steps
are integrated and no missing low words are inferred.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


V5_COMMIT = "a71f1fa6adef719202913ef3e2fc4c8c9297f7f2"
V5_SOURCE_SHA256 = "006fbeaefbd9671b384c12e9562449bfa63093753125f64ccac658da03deeea1"
ATTEMPT_MANIFEST_SHA256 = "9fef888269c5771f28ff8b9ba5b92b8e278cbc080e0d690a5f97275553f1fb02"
PREPARED_FILE_SHA256 = "8ae7796d6c6892ce1b885dd1ce95fe44a0d1043815e38ce10dc603c9d5a15a2f"
EXPECTED_ROWS = 934
COUNTS = {1: 134, 2: 267, 4: 533}
TARGET_ROW = 2
CORRECT_PREVIOUS_ROW = 1
WRONG_PREVIOUS_ROW = 0


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def git(project, *arguments):
    return subprocess.check_output(["git", *arguments], cwd=project, text=True).strip()


def clean_source(project, expected):
    if git(project, "rev-parse", "HEAD") != expected or git(project, "status", "--porcelain"):
        raise ValueError("legacy reconstruction requires the specified clean original checkout")
    return {"source_commit": expected, "clean": True}


def validate_attempt(attempt, prepared):
    attempt, prepared = Path(attempt).resolve(), Path(prepared).resolve()
    if sha(attempt / "ManifestV1.json") != ATTEMPT_MANIFEST_SHA256:
        raise ValueError("legacy attempt differs from the fixed V5 manifest")
    if sha(prepared) != PREPARED_FILE_SHA256:
        raise ValueError("legacy preparation differs from the fixed V5 artifact")
    manifest = read(attempt / "ManifestV1.json")
    if not {"FailureV1.json", "AcceptedStepsV1.jsonl", "CompletionV1.json"} <= set(manifest):
        raise ValueError("legacy manifest does not cover required original artifacts")
    for name, expected in manifest.items():
        path = (attempt / name).resolve()
        if not path.is_relative_to(attempt) or path.is_symlink() or not path.is_file() or sha(path) != expected:
            raise ValueError("legacy artifact manifest mismatch: " + name)
    record = read(attempt / "FailureV1.json")["partial_result"]
    rows = [json.loads(line) for line in (attempt / "AcceptedStepsV1.jsonl").read_text().splitlines()]
    if rows != record["accepted_steps"] or len(rows) != EXPECTED_ROWS:
        raise ValueError("legacy saved rows do not match the complete 934-row record")
    counts = {level: sum(row["substeps"] == level for row in rows) for level in COUNTS}
    if counts != COUNTS or record["control_label"] != "D" or record["intervals"] != 16:
        raise ValueError("legacy record is not the fixed D/N16 full-window case")
    expected_source = record["source"]
    if (expected_source.get("source_commit") != V5_COMMIT
            or expected_source.get("source_content_sha256") != V5_SOURCE_SHA256):
        raise ValueError("legacy record is bound to a different source")
    return record, rows


def predecessor_contract(rows):
    chosen = [rows[index] for index in (WRONG_PREVIOUS_ROW, CORRECT_PREVIOUS_ROW, TARGET_ROW)]
    if (len({row["substeps"] for row in chosen}) != 1
            or not chosen[0]["time_s"] < chosen[1]["time_s"] < chosen[2]["time_s"]):
        raise ValueError("fixed stale predecessor must be earlier in the same actual tier")
    # Flat field lengths are sufficient here because the actual rebase receives
    # the fully typed original state, not these serialized arrays.
    for field in ("phi_V", "n_m3", "p_m3", "positive_m3", "occupancy", "sheet_charge_C_m2"):
        if len(chosen[0]["state"][field]) != len(chosen[1]["state"][field]):
            raise ValueError("stale predecessor must retain the real state's shape")
    return {"schema": "R1V8LegacyPredecessorMutationV1", "target_row": TARGET_ROW,
        "correct_previous_row": CORRECT_PREVIOUS_ROW, "wrong_previous_row": WRONG_PREVIOUS_ROW,
        "substeps": chosen[2]["substeps"], "target_time_s": chosen[2]["time_s"],
        "correct_previous_time_s": chosen[1]["time_s"], "wrong_previous_time_s": chosen[0]["time_s"],
        "injection": "replace_real_previous_state_at_original_rebase_entry_once",
        "shape_and_tier_preserved": True,
        "states": {str(index): {"time_s": rows[index]["time_s"], "substeps": rows[index]["substeps"],
                   "snapshot_sha256": hashlib.sha256(canonical(rows[index]["state"]).encode()).hexdigest()}
                   for index in (WRONG_PREVIOUS_ROW, CORRECT_PREVIOUS_ROW, TARGET_ROW)},
        "required_rejection": "physical reconstruction mismatch: accepted state 2"}


@contextmanager
def stale_predecessor(system_class, *, original_rebase, snapshot, expected, replacement):
    """Instrument the actual legacy entry, restoring its class unconditionally."""
    previous_attribute = system_class.__dict__.get("rebase")
    had_attribute = "rebase" in system_class.__dict__
    events = []

    def injected(system, previous):
        if not events and canonical(snapshot(system, previous)) == canonical(expected):
            events.append({"entry": "original_system.rebase", "injected": True})
            return original_rebase(system, replacement)
        return original_rebase(system, previous)

    system_class.rebase = injected
    try:
        yield events
    finally:
        if had_attribute:
            system_class.rebase = previous_attribute
        else:
            delattr(system_class, "rebase")


def worker(args):
    """Run only after the subprocess import path has selected the V5 package."""
    start = time.perf_counter()
    historical = args.historical_project.resolve()
    initial_source = clean_source(historical, V5_COMMIT)
    record, rows = validate_attempt(args.attempt, args.prepared)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    contract = predecessor_contract(rows)
    write(output / "PredecessorContractV1.json", contract)
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_checkout import require_r1_checkout
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_state as states
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_physics_validation as replay
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_step import build_initial_step
    from perovskite_sim.models.config_loader import load_device_from_yaml
    import numpy as np
    import scipy.linalg  # Both numerical libraries load before the thread check.
    from threadpoolctl import threadpool_info, threadpool_limits
    actual_module = Path(states.__file__).resolve()
    if not actual_module.is_relative_to(historical):
        raise ValueError("legacy child imported equations from the wrong source directory")
    context = require_r1_checkout(project=historical, formal=True,
                                  source_commit=V5_COMMIT, expected_source_sha256=V5_SOURCE_SHA256)
    actual_source = states.execution_source()
    if (actual_source["files"] != record["source"]["files"]
            or actual_source["study_input"] != record["source"]["study_input"]
            or context.source_commit != record["source"]["source_commit"]
            or context.source_content_sha256 != record["source"]["source_content_sha256"]):
        raise ValueError("original saved and executed equation/source identities differ")
    fixture = historical / "tests/fixtures/configs/dynamic_interface_defect_ion_transient_absorber_only.yaml"
    binding_file = historical / "tests/fixtures/OneDimensionalMechanismR1ReferenceBindingV1.json"
    stack, binding = load_device_from_yaml(fixture), read(binding_file)
    prepared = states.R1PreparedState.from_dict(read(args.prepared))
    if prepared.sha256 != record["prepared_sha256"]:
        raise ValueError("legacy original preparation identity disagrees with the trajectory")
    with threadpool_limits(1):
        np.dot(np.ones((2, 2)), np.ones((2, 2)))
        pools = threadpool_info()
        if not pools or any(pool["num_threads"] != 1 for pool in pools):
            raise ValueError("legacy audit numerical libraries are not single-threaded")
        # Complete original physical verifier: all 934 states, observations,
        # physical checks, summaries and original failed certificate are read.
        healthy = replay.verify_r1_step_physics(stack, 16, binding, prepared, record,
            expected_prepared_sha256=prepared.sha256, allow_incomplete=True)
        write(output / "OriginalPhysicsReplayV1.json", healthy)
        policy = states._preparation_policy(record["policy"])
        base, before = states.verify_prepared_physics(prepared, stack, binding, policy=policy)
        initial = build_initial_step(base, before, record["amplitude_V"], policy=policy)
        if states.canonical(states.snapshot(initial.system, initial.zero_plus)) != states.canonical(rows[0]["state"]):
            raise ValueError("stale predecessor is not the actual original same-tier anchor")
        write(output / "PredecessorInputsV1.json", {
            "contract": contract,
            "correct_saved_previous_state": rows[CORRECT_PREVIOUS_ROW]["state"],
            "wrong_actual_previous_state": states.snapshot(initial.system, initial.zero_plus),
            "target_saved_state": rows[TARGET_ROW]["state"],
            "target_actual_coordinate": rows[TARGET_ROW]["physics_reconstruction"]["coordinate"]})
        cls = type(initial.system)
        original_rebase = cls.rebase
        rejection = {"rejected": False, "required_physical_rejection": False}
        with stale_predecessor(cls, original_rebase=original_rebase, snapshot=states.snapshot,
                               expected=rows[CORRECT_PREVIOUS_ROW]["state"], replacement=initial.zero_plus) as events:
            try:
                replay.verify_r1_step_physics(stack, 16, binding, prepared, record,
                    expected_prepared_sha256=prepared.sha256, allow_incomplete=True)
            except replay.R1PhysicsValidationError as exc:
                rejection = {"rejected": True, "type": type(exc).__name__, "message": str(exc),
                             "required_physical_rejection": str(exc) == contract["required_rejection"]}
        rejection["actual_injections"] = events
    after_source = clean_source(historical, V5_COMMIT)
    healthy_passed = (healthy.get("evidence_matches_equations") is True
        and healthy.get("content_matches_recomputed") is True
        and healthy.get("checked_row_count") == EXPECTED_ROWS and healthy.get("complete") is True)
    report = {"schema": "R1V8Legacy934ReplayV1", "elapsed_s": time.perf_counter() - start,
        "scope": "original_V5_source_and_format_offline_reconstruction_not_new_trajectory",
        "original_source": context.to_dict(), "script_sha256": sha(__file__),
        "recorded_source": record["source"], "executed_equation_files": actual_source["files"],
        "source_unchanged": initial_source == after_source,
        "attempt_manifest_sha256": sha(args.attempt / "ManifestV1.json"),
        "prepared_file_sha256": sha(args.prepared), "fixture_sha256": sha(fixture),
        "binding_sha256": sha(binding_file), "blas": pools,
        "checked_rows": EXPECTED_ROWS, "anchor_rows": 3, "regular_increment_rows": EXPECTED_ROWS - 3,
        "rows_per_tier": COUNTS, "original_replay_exact": healthy_passed,
        "original_physical_certificate_passed": healthy.get("certified"),
        "historical_failed_physical_status_preserved": healthy.get("certified") is False,
        "predecessor_mutation": rejection,
        "new_trajectory_integrated": False, "missing_low_words_inferred": False,
        "passed": healthy_passed and initial_source == after_source
                  and rejection["required_physical_rejection"] and len(events) == 1,
        "qualification": "legacy_reconstruction_requirement_only_no_V8_or_long_window_physical_acceptance"}
    write(output / "ResultV1.json", report)
    return report


def main():
    start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-project", type=Path, required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        report = worker(args)
        print(json.dumps({key: report[key] for key in ("passed", "elapsed_s", "checked_rows")}))
        return 0 if report["passed"] else 1
    project = Path(__file__).resolve().parents[1]
    wrapper_commit = git(project, "rev-parse", "HEAD")
    clean_source(project, wrapper_commit)
    clean_source(args.historical_project, V5_COMMIT)
    if args.output.exists():
        raise FileExistsError(args.output)
    # Keep installed dependencies but remove every current project path. The
    # child reads this V8 script while all numerical imports come from V5.
    dependency_paths = [path for path in sys.path if path and
                        ("site-packages" in Path(path).parts or "dist-packages" in Path(path).parts)]
    bootstrap = "import runpy,sys;sys.path[:0]=" + repr([str(args.historical_project.resolve()), *dependency_paths])
    bootstrap += ";sys.argv=" + repr([str(Path(__file__).resolve()), "--worker",
        "--historical-project", str(args.historical_project.resolve()), "--attempt", str(args.attempt.resolve()),
        "--prepared", str(args.prepared.resolve()), "--output", str(args.output.resolve())])
    bootstrap += ";runpy.run_path(" + repr(str(Path(__file__).resolve())) + ",run_name='__main__')"
    environment = dict(os.environ)
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[key] = "1"
    argv = [sys.executable, "-I", "-S", "-B", "-c", bootstrap]
    completed = subprocess.run(argv, env=environment, text=True, capture_output=True, check=False)
    if args.output.exists():
        (args.output / "WorkerV1.log").write_text(completed.stdout + completed.stderr)
        write(args.output / "V8EntrySourceV1.json", {"source_commit": wrapper_commit,
            "script_sha256": sha(__file__), "historical_source_commit": V5_COMMIT,
            "source_unchanged": clean_source(project, wrapper_commit)["clean"],
            "argv": argv, "exit_code": completed.returncode,
            "entry_elapsed_s_through_worker_completion": time.perf_counter() - start,
            "exception": "V8_audit_entry_executes_original_V5_numerical_equations"})
        write(args.output / "ManifestV1.json", {path.name: {"sha256": sha(path), "bytes": path.stat().st_size}
              for path in sorted(args.output.iterdir()) if path.is_file() and path.name != "ManifestV1.json"})
    print(completed.stdout, end="")
    print(completed.stderr, end="", file=sys.stderr)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
