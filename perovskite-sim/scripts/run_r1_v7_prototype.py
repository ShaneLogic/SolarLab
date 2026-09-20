"""Run the externally frozen V7 N16 prefix as bounded development evidence.

The original equations and certificate limits remain in force. A completed
trajectory, an exact replay, and a precision qualification are separate results.
No output from this runner grants formal R1-2 or P1 scientific acceptance.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
import time
import traceback


PROJECT = Path(__file__).resolve().parents[1]
SCHEMA = "R1V7PrototypeContractV1"
SCOPE = "development_prototype_not_R1_2_qualification"
TIMES_SHA256 = "763ab13ddb16966cf4d533894b852544119b383499070fef3076bac5fea9e04f"
HISTORICAL_REQUEST_SHA256 = "68fcb503c692413187cb981900157b601c7d8d97efe43b36784fc5eb432c0eef"
THREAD_KEYS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
INPUTS = {
    "fixture": {"path": "tests/fixtures/configs/dynamic_interface_defect_ion_transient_absorber_only.yaml",
                "sha256": "f617f230b2d9c144573394e38fcc313225dc84c7b38f7670b97e9a0a7cc12a24"},
    "reference": {"path": "tests/fixtures/OneDimensionalMechanismR1ReferenceBindingV1.json",
                  "sha256": "780e0275a3278e2d0ef2eb0ddf748ac763a7cc38ac0d03796312a952e0a01b4e"},
}
LIMITS = {
    "all_face_current_relative": 2e-6, "analytic_jacobian_error": 3e-4,
    "charge_balance_relative": 1e-10, "current_decomposition_error": 1e-14,
    "eliminated_operator_error": 1e-6, "full_charge_balance_normalized": 1e-10,
    "full_gauss_normalized": 1e-10, "interface_current_relative": 2e-6,
    "inventory_relative_drift": 1e-10, "local_carrier_residual": 1e-7,
    "local_gauss_residual": 1e-7, "nonlinear_residual": 0.05,
    "physical_current_spread_relative": 2e-6, "refinement_current_change": 0.05,
    "refinement_state_change": 0.02, "trap_storage_normalized_error": 1.0,
}
ENGINEERING_LIMITS = {"elapsed_ratio": 5.0, "peak_rss_ratio": 2.0, "bytes_per_row_ratio": 3.0}
CASE_AXES = {"control": "D", "intervals": 16, "nonlinear_factor": 0.1,
             "time_substeps": [1, 2, 4], "amplitude_V": 0.005, "fault": "none"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_digest(value, name, lengths=(64,)):
    if (not isinstance(value, str) or len(value) not in lengths
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(name + " must be a full lowercase digest")


def write(path, value):
    path = Path(path)
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pending-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def error_record(exc):
    return {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}


def fault_contract():
    common = {"responsibility": "independent_per_side_formula_and_boundary_oracle",
              "injection_scope": "implementation_on_both_paths; healthy_material_frozen_before_injection",
              "healthy_predicate": "absolute_error <= predeclared_budget",
              "fault_predicate": "absolute_error > same_predeclared_budget",
              "qualified": False, "evidence_status": "required_not_yet_measured"}
    return {
        "drift_sign": {**common, "injection": "ion drive sign reversed in implementation"},
        "diffusion_multiplier": {**common, "injection": "Dion multiplied by 1.01 in implementation"},
        "thermal_voltage_multiplier": {**common, "injection": "VT multiplied by 1.01 in implementation"},
        "omitted_ion_contribution": {**common, "injection": "ion contribution omitted in implementation"},
        "one_face_sign": {**common, "injection": "one interior face sign reversed"},
        "boundary_implementation": {**common, "injection": "blocking boundary implemented incorrectly"},
        "path_potential_inconsistency": {
            "responsibility": "original_direct_eliminated_input_consistency_gate",
            "injection_scope": "one_path_only", "injection": "named nonuniform potential perturbation",
            "healthy_predicate": "original_relative_error <= 1e-6",
            "fault_predicate": "original_relative_error > 1e-6 at frozen identifiable perturbation",
            "qualified": False, "evidence_status": "direction_and_amplitude_require_oracle_freeze"},
    }


def make_contract(historical_request):
    """Build the fixed physical contract; this never starts a computation."""
    path = Path(historical_request)
    if sha(path) != HISTORICAL_REQUEST_SHA256:
        raise ValueError("historical request differs from the V5 N16 sentinel")
    historical = json.loads(path.read_text())
    times = historical["times_s"][:114]
    if hashlib.sha256(canonical(times).encode()).hexdigest() != TIMES_SHA256:
        raise ValueError("historical prefix differs from the declared 114 points")
    return {
        "schema": SCHEMA, "scope": SCOPE, "case_id": "D_N16_F0p1_T1_To2p154434690031884s",
        "case": {**CASE_AXES, "times_s": times}, "inputs": INPUTS,
        "historical_origin": {"source_commit": "a71f1fa6adef719202913ef3e2fc4c8c9297f7f2",
                              "request_sha256": HISTORICAL_REQUEST_SHA256,
                              "time_selection": "first_114_values_including_zero_plus",
                              "times_sha256": TIMES_SHA256},
        "execution": {"modes": ["baseline", "compensated"], "start": "new_common_equilibrium_then_new_zero_plus",
                      "expected_rows_by_substeps": {"1": 114, "2": 227, "4": 453},
                      "expected_total_rows": 794, "output_time_count": 114,
                      "single_thread": True, "requires_clean_full_commit": True,
                      "resume_from_old_state": False, "may_replace_100s_sentinel": False},
        "certificate_limits": LIMITS,
        "engineering_limits": ENGINEERING_LIMITS,
        "engineering_measurement": {
            "elapsed": "wall time from preparation through trajectory and replay; includes evidence writes",
            "rss": "fresh-process peak resident set bytes; same interpreter/dependencies/threads",
            "payload": "AcceptedStepsV1.jsonl bytes divided by actual persisted rows",
            "comparison": "complete baseline and compensated runs with identical external request digest",
            "missing_comparable_run": "unknown; never zero and never qualified"},
        "fault_contract": fault_contract(),
        "precision_requirements": {
            "state_fields": {"phi_V": "V", "n_m3": "m^-3", "p_m3": "m^-3",
                             "positive_m3": "m^-3", "dqfn_V": "V", "dqfp_V": "V"},
            "composition": "independent_oracle compares exact hi+lo, never a float64-rounded sum",
            "consumer_checks": ["rebase", "flux", "storage", "charge", "current", "analytic_derivative"],
            "side_oracle_budget": "freeze per-state absolute budgets before fault measurements; original gates remain unchanged",
            "directional_potential_target": "late-state ~1e-23 V is a calibration estimate, not a universal bound",
            "numeric_budget_freeze_complete": False,
            "qualification": "requires external independent oracle, actual low-bit consumption and fault evidence"},
        "scientifically_accepted": False, "formal_qualification": False,
    }


def load_contract(path, digest, output):
    path, output = Path(path).resolve(), Path(output).resolve()
    if path.is_relative_to(output):
        raise ValueError("external request must be outside the output directory")
    validate_digest(digest, "request_sha256")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("request differs from its external SHA256")
    contract = json.loads(raw)
    if contract.get("schema") != SCHEMA or contract.get("scope") != SCOPE:
        raise ValueError("unsupported V7 prototype contract")
    case = contract.get("case", {})
    if set(case) != set(CASE_AXES) | {"times_s"} or any(case.get(k) != v for k, v in CASE_AXES.items()):
        raise ValueError("prototype case identity differs from the fixed N16 prefix")
    times = case["times_s"]
    if (not isinstance(times, list) or len(times) != 114
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in times)
            or hashlib.sha256(canonical(times).encode()).hexdigest() != TIMES_SHA256):
        raise ValueError("prototype requires the exact 114-point historical time prefix")
    if contract.get("certificate_limits") != LIMITS or contract.get("engineering_limits") != ENGINEERING_LIMITS:
        raise ValueError("prototype certificate or engineering limits changed")
    if contract.get("inputs") != INPUTS:
        raise ValueError("prototype fixture or reference identity changed")
    execution = contract.get("execution", {})
    if (execution.get("expected_rows_by_substeps") != {"1": 114, "2": 227, "4": 453}
            or execution.get("expected_total_rows") != 794 or execution.get("output_time_count") != 114
            or execution.get("modes") != ["baseline", "compensated"]
            or execution.get("start") != "new_common_equilibrium_then_new_zero_plus"
            or execution.get("resume_from_old_state") is not False
            or execution.get("single_thread") is not True):
        raise ValueError("prototype execution identity or extent changed")
    if contract.get("fault_contract") != fault_contract():
        raise ValueError("prototype fault responsibility contract changed")
    if contract.get("formal_qualification") is not False or contract.get("scientifically_accepted") is not False:
        raise ValueError("prototype request cannot grant scientific acceptance")
    return contract, raw


def git(*args):
    return subprocess.check_output(["git", *args], cwd=PROJECT, text=True).strip()


def source_snapshot(expected_commit):
    validate_digest(expected_commit, "expected_commit", (40, 64))
    if git("rev-parse", "HEAD") != expected_commit or git("status", "--porcelain"):
        raise ValueError("prototype requires its exact full commit and a clean source tree")
    root = Path(git("rev-parse", "--show-toplevel"))
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    files = {name: sha(root / name) for name in sorted(names) if name}
    tree = subprocess.check_output(["git", "ls-tree", "-r", "-z", expected_commit], cwd=root).split(b"\0")
    tree_paths = set()
    object_hash = hashlib.sha1 if git("rev-parse", "--show-object-format") == "sha1" else hashlib.sha256
    for entry in tree:
        if not entry:
            continue
        metadata, raw_name = entry.split(b"\t", 1)
        mode, kind, object_id = metadata.split()
        name = raw_name.decode()
        tree_paths.add(name)
        path = root / name
        if kind != b"blob":
            raise ValueError("prototype does not silently traverse an unfrozen submodule: " + name)
        if mode == b"120000":
            raw_bytes = os.readlink(path).encode()
        else:
            if path.is_symlink():
                raise ValueError("tracked source became a symlink: " + name)
            raw_bytes = path.read_bytes()
        blob = b"blob " + str(len(raw_bytes)).encode() + b"\0" + raw_bytes
        if object_hash(blob).hexdigest() != object_id.decode():
            raise ValueError("source bytes differ from the full commit even if Git status is clean: " + name)
    if tree_paths != set(files):
        raise ValueError("tracked source paths differ from the full commit")
    return {"source_commit": expected_commit, "all_tracked_files": files,
            "all_tracked_content_sha256": hashlib.sha256(canonical(files).encode()).hexdigest()}


def extent(rows, case):
    """A prefix or duplicate count can never be reported as a complete level."""
    result = []
    expected_levels = case["time_substeps"]
    for level in expected_levels:
        part = [row for row in rows if row.get("substeps") == level]
        expected = 1 + (len(case["times_s"]) - 1) * level
        observed_times = [row.get("time_s") for row in part]
        result.append({"substeps": level, "accepted_rows": len(part), "expected_rows": expected,
                       "last_time_s": observed_times[-1] if part else None,
                       "row_count_complete": len(part) == expected,
                       "zero_plus_present": bool(part and observed_times[0] == 0.0),
                       "reached_declared_end": bool(part and observed_times[-1] == case["times_s"][-1]),
                       "strictly_increasing": all(isinstance(a, (int, float)) and isinstance(b, (int, float))
                                                   and math.isfinite(a) and math.isfinite(b) and a < b
                                                   for a, b in zip(observed_times[:-1], observed_times[1:]))})
    unassigned = sum(row.get("substeps") not in expected_levels for row in rows)
    complete = unassigned == 0 and all(all(item[k] for k in (
        "row_count_complete", "zero_plus_present", "reached_declared_end", "strictly_increasing")) for item in result)
    return {"accepted_rows": len(rows), "expected_rows": 794, "levels": result,
            "unassigned_rows": unassigned, "complete": complete}


def certificate_check(result):
    certificate = result.get("certificate", {})
    actual_limits, metrics = certificate.get("limits"), certificate.get("metrics", {})
    limits_match = actual_limits == LIMITS
    values_pass = (set(metrics) == set(LIMITS) and all(type(metrics[k]) in (int, float)
                   and math.isfinite(metrics[k]) and 0.0 <= metrics[k] <= limit for k, limit in LIMITS.items()))
    return {"limits_match_contract": limits_match, "all_original_metrics_within_limits": values_pass,
            "certificate_certified": certificate.get("certified") is True,
            "passed": limits_match and values_pass and certificate.get("certified") is True}


def precision_record_check(rows, mode):
    fields = {"phi_V", "n_m3", "p_m3", "positive_m3", "dqfn_V", "dqfp_V"}
    missing, invalid = [], []
    for index, row in enumerate(rows):
        data = row.get("state", {})
        expected = {"precision_" + name + "_" + side for name in fields for side in ("hi", "lo")}
        if not expected <= set(data):
            missing.append(index)
            continue
        for name in fields:
            hi, lo = (data["precision_" + name + "_" + side] for side in ("hi", "lo"))
            if (not isinstance(hi, list) or not isinstance(lo, list) or len(hi) != len(lo)
                    or any(type(v) not in (int, float) or not math.isfinite(v) for side in (hi, lo) for v in side)):
                invalid.append({"row": index, "field": name})
    return {"required": mode == "compensated", "checked_rows": len(rows),
            "record_fields_present": bool(rows) and not missing and not invalid,
            "missing_rows": missing, "invalid_fields": invalid,
            "proves_low_bit_consumption": False,
            "scope": "numeric field coverage only; consumer and oracle tests are separate"}


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def cost_comparison(current, baseline, contract_digest):
    if baseline is None:
        return {"qualified": False, "status": "missing_comparable_baseline", "ratios": {}}
    reasons = []
    if (baseline.get("schema") != "R1V7PrototypeRunV1" or baseline.get("mode") != "baseline"
            or baseline.get("request_sha256") != contract_digest
            or baseline.get("source_unchanged") is not True
            or baseline.get("extent", {}).get("complete") is not True
            or current.get("extent", {}).get("complete") is not True):
        reasons.append("baseline_not_complete_and_comparable")
    if baseline.get("runtime_identity") != current.get("runtime_identity"):
        reasons.append("runtime_identity_differs")
    ratios = {}
    for key, metric in (("elapsed_ratio", "elapsed_s"), ("peak_rss_ratio", "peak_rss_bytes"),
                        ("bytes_per_row_ratio", "bytes_per_row")):
        a, b = current.get("cost", {}).get(metric), baseline.get("cost", {}).get(metric)
        if (type(a) not in (int, float) or type(b) not in (int, float)
                or not math.isfinite(a) or not math.isfinite(b) or b <= 0):
            reasons.append("unavailable_" + metric)
        else:
            ratios[key] = a/b
            if ratios[key] > ENGINEERING_LIMITS[key]:
                reasons.append("exceeded_" + key)
    return {"qualified": not reasons, "status": "within_limits" if not reasons else "not_qualified",
            "ratios": ratios, "limits": ENGINEERING_LIMITS, "reasons": reasons}


def run_trajectory(directory, case, mode, api, source_guard):
    """Persist a run and every accepted prefix. api is explicit for bounded tests."""
    observer_path = directory / "AcceptedStepsV1.jsonl"
    observer_path.touch(exist_ok=False)
    rows, result, replay, failure = [], {}, None, None
    prepared = None
    report = {"execution_status": "not_started", "physical_execution_started": False}
    started = time.monotonic()

    def observe(row):
        value = api["json_data"](row)
        raw = canonical(value) + "\n"
        with observer_path.open("a") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        rows.append(value)

    try:
        source_guard()
        report.update(execution_status="preparing", physical_execution_started=True)
        prepared = api["prepare"]()
        write(directory / "PreparedV1.json", api["json_data"](prepared.to_dict()))
        report["execution_status"] = "integrating"
        try:
            result = api["run"](prepared, observe)
            report["execution_status"] = "completed"
        except (Exception, KeyboardInterrupt) as exc:
            failure = error_record(exc)
            report["integration_failure"] = {key: failure[key] for key in ("type", "message")}
            result = getattr(exc, "result", {})
            report["execution_status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        result = api["json_data"](result) if isinstance(result, dict) else {}
        write(directory / "ResultV1.json", result)
        if prepared is not None and result:
            try:
                replay = api["replay"](prepared, result, report["execution_status"] != "completed")
                write(directory / "PhysicsReplayV1.json", api["json_data"](replay))
            except Exception as exc:
                report["replay_error"] = error_record(exc)
        if failure is not None and prepared is not None and result and "failure_witness" in api:
            try:
                witness = api["failure_witness"](prepared, result, rows, failure)
                write(directory / "FailureWitnessV1.json", api["json_data"](witness))
            except Exception as exc:
                report["failure_witness_error"] = error_record(exc)
        source_guard()
    except (Exception, KeyboardInterrupt) as exc:
        failure = error_record(exc)
        payload = getattr(exc, "result", None)
        if payload is not None:
            try:
                filename = ("PreparationFailureV1.json" if report["execution_status"] == "preparing"
                            else "PhaseFailureResultV1.json")
                write(directory / filename, api["json_data"](payload))
            except Exception as payload_exc:
                report["failure_payload_persistence_error"] = error_record(payload_exc)
        report["execution_status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
    finally:
        if failure is not None:
            write(directory / "FailureV1.json", failure)
        persisted = [json.loads(line) for line in observer_path.read_text().splitlines()]
        report["extent"] = extent(persisted, case)
        report["certificate_check"] = certificate_check(result)
        report["precision_records"] = precision_record_check(persisted, mode)
        predicates = {
            "run_completed": report["execution_status"] == "completed" and failure is None,
            "certificate_certified": report["certificate_check"]["passed"],
            "replay_certified": isinstance(replay, dict) and replay.get("certified") is True,
            "observer_matches_result": persisted == result.get("accepted_steps", []) and persisted == rows,
        }
        report["four_predicates"] = predicates
        report["four_predicates_passed"] = all(predicates.values()) and report["extent"]["complete"]
        report["cost"] = {"elapsed_s": time.monotonic()-started, "peak_rss_bytes": peak_rss_bytes(),
                          "accepted_rows_bytes": observer_path.stat().st_size,
                          "bytes_per_row": observer_path.stat().st_size/len(persisted) if persisted else None}
        report["failure"] = None if failure is None else {k: failure[k] for k in ("type", "message")}
    return report


def run(args):
    output = args.output.resolve()
    contract, raw = load_contract(args.request_file, args.request_sha256, output)
    if output.exists():
        raise FileExistsError("prototype output exists; use a new run directory")
    if not (sys.flags.isolated and sys.flags.no_site):
        raise ValueError("start the prototype with an isolated Python -I -S interpreter")
    validate_digest(args.source_sha256, "source_sha256")
    initial_source = source_snapshot(args.expected_commit)
    for key in THREAD_KEYS:
        os.environ[key] = "1"
    sys.path.insert(0, str(PROJECT))
    import numpy as np
    import scipy
    from threadpoolctl import threadpool_info, threadpool_limits
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_checkout import require_r1_checkout
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import json_data
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import r1_policy
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_failure_witness import build_failure_witness
    from perovskite_sim.models.config_loader import load_device_from_yaml
    context = require_r1_checkout(project=PROJECT, formal=True, source_commit=args.expected_commit,
                                  expected_source_sha256=args.source_sha256)
    input_bytes = {name: context.read_bytes(spec["path"]) for name, spec in INPUTS.items()}
    if any(hashlib.sha256(input_bytes[k]).hexdigest() != INPUTS[k]["sha256"] for k in INPUTS):
        raise ValueError("frozen fixture or binding differs from the P0 request")
    baseline = None
    if args.baseline_summary is not None:
        validate_digest(args.baseline_sha256, "baseline_sha256")
        if sha(args.baseline_summary) != args.baseline_sha256:
            raise ValueError("baseline summary differs from its external digest")
        baseline = json.loads(args.baseline_summary.read_text())
    elif args.baseline_sha256 is not None:
        raise ValueError("baseline digest requires a baseline summary")
    output.mkdir(parents=True, exist_ok=False)
    (output / "RequestV1.json").write_bytes(raw)
    (output / "SourceFixtureV1.yaml").write_bytes(input_bytes["fixture"])
    (output / "ReferenceBindingV1.json").write_bytes(input_bytes["reference"])
    write(output / "SourceReceiptV1.json", {**initial_source,
          "r1_source_content_sha256": context.source_content_sha256, "runner_sha256": sha(__file__),
          "scope": "clean committed bytes verified; prototype is not a controlled formal qualification"})
    stack = load_device_from_yaml(output / "SourceFixtureV1.yaml")
    binding = json.loads(input_bytes["reference"])
    case = contract["case"]
    summary = {"schema": "R1V7PrototypeRunV1", "scope": SCOPE, "mode": args.mode,
               "request_sha256": args.request_sha256, "source_commit": args.expected_commit,
               "source_content_sha256": args.source_sha256,
               "started_utc": datetime.now(timezone.utc).isoformat(), "scientifically_accepted": False,
               "formal_qualification": False, "P1_qualified": False,
               "independent_requirements": {"side_oracle": "not_yet_evaluated",
                                            "implementation_fault_detection": "not_yet_evaluated",
                                            "low_bit_consumption": "not_yet_evaluated"}}
    write(output / "SummaryV1.json", summary)

    def source_guard():
        if source_snapshot(args.expected_commit) != initial_source or sha(args.request_file) != args.request_sha256:
            raise ValueError("source or external request changed during prototype execution")
        require_r1_checkout(project=PROJECT, formal=True, source_commit=args.expected_commit,
                            expected_source_sha256=args.source_sha256)

    try:
        # Freeze healthy material before entering any compensated/fault context.
        # Missing oracle support is explicit and does not prevent a real run.
        try:
            from scripts.verify_r1_v7_precision import freeze_healthy_material
            from perovskite_sim.experiments.one_dimensional_mechanism_r1 import build_r1_material
            grid, material = build_r1_material(stack, case["intervals"])
            frozen_material = freeze_healthy_material(grid, material, source_identity=args.source_sha256)
            write(output / "FrozenHealthyMaterialV1.json", frozen_material)
            summary["healthy_material_frozen_before_precision_context"] = True
        except Exception as oracle_exc:
            summary["healthy_material_frozen_before_precision_context"] = False
            summary["oracle_freeze_error"] = error_record(oracle_exc)
        precision = None
        if args.mode == "compensated":
            precision = importlib.import_module("perovskite_sim.experiments.one_dimensional_mechanism_r1_precision")
        manager = nullcontext() if precision is None else precision.precision_context(mode="compensated")
        with threadpool_limits(limits=1), manager:
            # Resolve hooks after entering the opt-in context. Earlier aliases
            # would silently bypass the versioned pair preparation codec.
            from perovskite_sim.experiments import one_dimensional_mechanism_r1_state as state_api
            from perovskite_sim.experiments import one_dimensional_mechanism_r1_protocol as protocol_api
            from perovskite_sim.experiments import one_dimensional_mechanism_r1_physics_validation as replay_api
            # Force BLAS loading before collecting its state, rather than saving [].
            np.dot(np.ones((2, 2)), np.ones((2, 2)))
            pools = threadpool_info()
            if not pools or any(pool.get("num_threads") != 1 for pool in pools):
                raise ValueError("runtime does not evidence single-thread numerical libraries")
            runtime = {"executable": sys.executable, "python": sys.version, "numpy": np.__version__,
                       "scipy": scipy.__version__, "platform": sys.platform,
                       "machine": platform.machine(), "host": platform.node(),
                       "threads": {key: os.environ[key] for key in THREAD_KEYS}, "blas": pools}
            write(output / "EnvironmentV1.json", runtime)
            summary["runtime_identity"] = runtime
            summary["precision_capabilities"] = ({"representation_id": "float64-baseline",
                "state_low_fields_persisted": False, "low_bits_consumed": False,
                "replay_supported": True, "original_checks_preserved": True}
                if precision is None else json_data(precision.precision_capabilities()))
            api = {
                "json_data": json_data,
                "prepare": lambda: state_api.prepare_common_state(stack, case["intervals"], binding, policy=r1_policy()),
                "run": lambda prepared, observer: protocol_api.run_r1_step(stack, case["intervals"], binding, prepared,
                    control=case["control"], amplitude_V=case["amplitude_V"], times_s=case["times_s"],
                    policy=r1_policy(case["nonlinear_factor"], time_substeps=tuple(case["time_substeps"])),
                    physics_evidence=True, accepted_step_observer=observer, expected_prepared_sha256=prepared.sha256),
                "replay": lambda prepared, result, incomplete: replay_api.verify_r1_step_physics(
                    stack, case["intervals"], binding, prepared, result, allow_incomplete=incomplete),
                "failure_witness": lambda prepared, result, rows, failure: build_failure_witness(
                    source_commit=args.expected_commit,
                    protocol={"stage": "controlled-step", "intervals": case["intervals"],
                              "control": case["control"], "amplitude_V": case["amplitude_V"],
                              "times_s": case["times_s"], "policy": json_data(r1_policy(
                                  case["nonlinear_factor"], time_substeps=tuple(case["time_substeps"])))},
                    failure=failure, result=result, persisted_rows=rows),
            }
            summary.update(run_trajectory(output, case, args.mode, api, source_guard))
            summary["precision_capabilities_after_run"] = (summary["precision_capabilities"] if precision is None
                                                           else json_data(precision.precision_capabilities()))
        source_guard()
        summary["source_unchanged"] = True
    except (Exception, KeyboardInterrupt) as exc:
        summary.update(execution_status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                       runner_error=error_record(exc), four_predicates_passed=False)
        write(output / "RunnerFailureV1.json", summary["runner_error"])
        try:
            source_guard()
            summary["source_unchanged"] = True
        except Exception as source_exc:
            summary["source_unchanged"] = False
            summary["source_error"] = error_record(source_exc)
    summary["engineering_check"] = cost_comparison(summary, baseline, args.request_sha256)
    summary["P1_blockers"] = ["independent_side_oracle_evidence_not_attached",
                              "implementation_fault_detection_not_attached",
                              "low_bit_consumption_evidence_not_attached"]
    if not summary.get("four_predicates_passed"):
        summary["P1_blockers"].append("trajectory_or_replay_not_passed")
    if args.mode == "compensated" and not summary.get("precision_records", {}).get("record_fields_present"):
        summary["P1_blockers"].append("complete_precision_fields_not_persisted")
    if not summary["engineering_check"]["qualified"]:
        summary["P1_blockers"].append("engineering_limits_not_qualified")
    summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
    summary["exit_code"] = 0 if summary.get("four_predicates_passed") and summary.get("source_unchanged") else 1
    summary["exit_code_scope"] = "trajectory/replay execution only; inspect P1_qualified separately"
    write(output / "SummaryV1.json", summary)
    write(output / "ManifestV1.json", {path.relative_to(output).as_posix(): {
        "sha256": sha(path), "bytes": path.stat().st_size}
        for path in sorted(output.rglob("*")) if path.is_file() and path.name != "ManifestV1.json"})
    print(json.dumps({"status": summary.get("execution_status"), "mode": args.mode,
                      "accepted_rows": summary.get("extent", {}).get("accepted_rows", 0),
                      "four_predicates_passed": summary.get("four_predicates_passed", False),
                      "P1_qualified": False, "exit_code": summary["exit_code"]}))
    return summary["exit_code"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("baseline", "compensated"))
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--source-sha256", required=True, help="R1 required-source content digest")
    parser.add_argument("--request-file", required=True, type=Path)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline-summary", type=Path)
    parser.add_argument("--baseline-sha256")
    try:
        return run(parser.parse_args(argv))
    except Exception as exc:
        print("V7 PROTOTYPE NOT STARTED: " + type(exc).__name__ + ": " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
