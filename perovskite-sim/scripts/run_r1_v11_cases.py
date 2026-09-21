"""Execute one preregistered P3/P4 case without changing the physical solver.

The default V9 window remains fixed. This entry selects an explicit new case,
uses the production backend, and gives incomplete references no cost ratio.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys

from scripts.run_r1_v9_prototype import (
    INPUTS, LIMITS, THREAD_KEYS, PROJECT, canonical, sha, write, error_record,
    source_snapshot, function_snapshot, check_functions, tag_nonfinite,
    run_trajectory,
)
from scripts.r1_v10_extent import extent, expected_schedule

SCHEMA = "R1V11CaseRequestV1"
SCOPE = "bounded_P3_P4_existing_physics_validation_no_new_physics_qualification"
LONG_TIMES_SHA256 = "ca50ba91f43c5c0da49996feb65556e19c21658a6ffe0cb95d88cda810de0872"
SHORT_TIMES = [0., 1e-9, 1e-8, 1e-6, 1e-4]
RELATIVE_LIMITS = {"elapsed_ratio": 5., "peak_rss_ratio": 2., "bytes_per_row_ratio": 3.}
SOLVER_LIMITS = {"newton": 100, "line_search": 40, "near_acceptance_nonmonotone": 2,
                 "joint_corrector": 8, "independent_poisson": 12}
HISTORICAL_MATRIX_SHA256 = "7d751b619f95f85229cc045aac6124e64dfa35fbc752cb2d3d9e2b90a9881120"
# Exact ordered axes from V5 Root/RecoveryRequestV1.json, not a Cartesian guess.
ORIGINAL_AXES = (('D', 32, 0.01, 1), ('D', 32, 0.01, 2), ('D', 32, 0.01, 4), ('A', 64, 0.01, 2), ('B', 64, 0.01, 2), ('B', 64, 1.0, 2), ('B', 64, 1.0, 4), ('C', 64, 0.01, 2), ('D', 64, 0.01, 1), ('D', 64, 0.01, 2), ('D', 64, 0.01, 4), ('D', 64, 1.0, 1), ('D', 64, 1.0, 2), ('D', 64, 1.0, 4), ('D', 128, 0.01, 4), ('A', 256, 0.01, 1), ('A', 256, 0.01, 2), ('A', 256, 0.01, 4), ('A', 256, 0.1, 4), ('B', 256, 0.1, 4), ('B', 256, 1.0, 4), ('D', 256, 0.01, 1), ('D', 256, 0.01, 2), ('D', 256, 0.01, 4), ('D', 256, 0.1, 1), ('D', 256, 0.1, 4), ('D', 256, 1.0, 1), ('D', 256, 1.0, 2), ('D', 256, 1.0, 4))


def case_name(control, n, factor, tier):
    return f"{control}_N{n}_F{str(float(factor)).replace('.', 'p')}_T{tier}"


CASES = {case_name(c, n, f, t): ("original_matrix", c, n, f, [t, 2*t, 4*t])
         for c, n, f, t in ORIGINAL_AXES}
CASES.update({case_name(c, 256, .01, 8): ("strict_extension", c, 256, .01, [8, 16, 32])
              for c in ("D", "B")})
CASES.update({"P4LongD16": ("long_window", "D", 16, .1, [1, 2, 4]),
              "P4LongD256": ("long_window", "D", 256, .01, [4, 8, 16])})
ABSOLUTE_KEYS = ("elapsed_s", "peak_rss_bytes", "row_payload_bytes", "case_artifact_bytes")


def validate_absolute_limits(limits):
    if (not isinstance(limits, dict) or set(limits) != set(ABSOLUTE_KEYS)
            or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                   for value in limits.values())
            or any(type(limits[key]) is not int for key in ABSOLUTE_KEYS[1:])):
        raise ValueError("four finite positive absolute resource limits are required; bytes are integers")
    return limits


def make_request(case_id, budget_sha256, absolute_limits, *, long_times=None,
                 required_pair_preparation_identity_sha256=None):
    """Materialize one named request; callers freeze it before any execution."""
    kind, control, n, factor, steps = CASES[case_id]
    times = list(long_times) if kind == "long_window" and long_times is not None else list(SHORT_TIMES)
    case = {"control": control, "intervals": n, "nonlinear_factor": factor,
            "time_substeps": list(steps), "amplitude_V": .005, "fault": "none", "times_s": times}
    return {"schema": SCHEMA, "scope": SCOPE, "case_id": case_id, "kind": kind,
            "case": case, "inputs": INPUTS, "certificate_limits": LIMITS,
            "solver_limits": SOLVER_LIMITS, "relative_engineering_limits": RELATIVE_LIMITS,
            "absolute_engineering_limits": dict(validate_absolute_limits(absolute_limits)),
            "preparation": {"nonlinear_factor": .1, "time_substeps": [1, 2, 4]},
            "expected_rows": len(expected_schedule(case)), "source_requires_clean_commit": True,
            "numerical_retries": 0, "baseline_repeats": 1, "pair_repeats": 1,
            "absolute_engineering_qualification": True, "scientifically_accepted": False,
            "formal_R1_2_qualification": False, "precision_budget_sha256": budget_sha256,
            "historical_matrix_sha256": HISTORICAL_MATRIX_SHA256,
            "required_pair_preparation_identity_sha256": required_pair_preparation_identity_sha256}


def absolute_cost(cost, limits):
    validate_absolute_limits(limits)
    names = {"row_payload_bytes": "accepted_row_payload_bytes"}
    measurements = {key: cost.get(names.get(key, key)) for key in ABSOLUTE_KEYS}
    invalid = [key for key, value in measurements.items()
               if type(value) not in (int, float) or not math.isfinite(value) or value <= 0]
    failed = [key for key, value in measurements.items() if key not in invalid and value > limits[key]]
    return {"applicable": True, "qualified": not invalid and not failed, "limits": dict(limits),
            "measurements": measurements, "failed_metrics": invalid + failed,
            "invalid_metrics": invalid, "scope": "preregistered_case_absolute_resource_allocation"}



def external_json(path, expected):
    path = Path(path)
    if (not isinstance(expected, str) or len(expected) != 64
            or any(c not in "0123456789abcdef" for c in expected) or sha(path) != expected):
        raise ValueError("external artifact identity differs: " + path.name)
    return json.loads(path.read_text())


def load_request(path, expected, output):
    path = Path(path).resolve()
    if path.is_relative_to(Path(output).resolve()):
        raise ValueError("request must be held outside the case directory")
    request = external_json(path, expected)
    if request.get("schema") != SCHEMA or request.get("scope") != SCOPE:
        raise ValueError("not a bounded V11 case request")
    if request.get("case_id") not in CASES:
        raise ValueError("case is outside the approved V11 set")
    kind, control, n, factor, steps = CASES[request["case_id"]]
    case = request.get("case", {})
    axes = {"control": control, "intervals": n, "nonlinear_factor": factor,
            "time_substeps": steps, "amplitude_V": .005, "fault": "none"}
    if (request.get("kind") != kind or set(case) != set(axes) | {"times_s"}
            or any(case.get(key) != value for key, value in axes.items())):
        raise ValueError("V11 physical axes differ from the named case")
    if (type(case["intervals"]) is not int or type(case["nonlinear_factor"]) not in (int, float)
            or type(case["amplitude_V"]) not in (int, float)):
        raise ValueError("V11 physical axes cannot use boolean or string coercion")
    schedule = expected_schedule(case)
    times = case["times_s"]
    if kind == "long_window":
        if (len(times) != 134 or times[-1] != 100.
                or hashlib.sha256(canonical(times).encode()).hexdigest() != LONG_TIMES_SHA256):
            raise ValueError("long window requires the exact 134 historical output times")
    elif times != SHORT_TIMES:
        raise ValueError("strict cases require the original five observation times")
    for key, value in (("inputs", INPUTS), ("certificate_limits", LIMITS),
                       ("solver_limits", SOLVER_LIMITS),
                       ("relative_engineering_limits", RELATIVE_LIMITS),
                       ("preparation", {"nonlinear_factor": .1, "time_substeps": [1, 2, 4]}),
                       ("expected_rows", len(schedule)), ("source_requires_clean_commit", True),
                       ("numerical_retries", 0), ("baseline_repeats", 1), ("pair_repeats", 1),
                       ("absolute_engineering_qualification", True),
                       ("scientifically_accepted", False), ("formal_R1_2_qualification", False)):
        if request.get(key) != value:
            raise ValueError("V11 frozen requirement changed: " + key)
    validate_absolute_limits(request.get("absolute_engineering_limits"))
    if request.get("historical_matrix_sha256") != HISTORICAL_MATRIX_SHA256:
        raise ValueError("V11 original matrix provenance differs")
    required = request.get("required_pair_preparation_identity_sha256")
    if "required_pair_preparation_identity_sha256" not in request:
        raise ValueError("V11 explicit preparation identity requirement is absent")
    if request["case_id"] == "P4LongD256":
        if (not isinstance(required, str) or len(required) != 64
                or any(c not in "0123456789abcdef" for c in required)):
            raise ValueError("N256 long window requires its prior pair physical preparation identity")
    elif required is not None:
        raise ValueError("only N256 long window imports a required pair preparation identity")
    return request


def validate_required_preparation(prepared, request, *, pair):
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import physical_preparation_identity
    required = request["required_pair_preparation_identity_sha256"]
    if not pair or required is None:
        return {"applicable": False, "matched": None, "required_sha256": required,
                "reason": "float_baseline_not_pair_identity" if not pair else "no_external_preparation_required"}
    actual = physical_preparation_identity(prepared)["sha256"]
    if actual != required:
        raise ValueError("N256 pair physical preparation differs from its completed DC/D2 dependencies")
    return {"applicable": True, "matched": True, "required_sha256": required, "actual_sha256": actual,
            "source_binding": "separate_clean_commit_and_runtime_guards"}


def relative_cost(current, baseline):
    if baseline is None:
        return {"applicable": False, "qualified": None, "status": "reference_only", "ratios": {}}
    identity = ("schema", "case_id", "source_commit", "source_content_sha256",
                "request_sha256", "precision_budget_sha256", "runtime_identity")
    if (any(current.get(key) != baseline.get(key) for key in identity)
            or baseline.get("mode") != "baseline" or not baseline.get("baseline_usable")):
        raise ValueError("cost reference is not the same-source same-request baseline")
    if not baseline.get("extent", {}).get("complete") or not current.get("extent", {}).get("complete"):
        return {"applicable": False, "qualified": None, "status": "unavailable_incomplete_extent",
                "ratios": {}, "reason": "a failed prefix is not a denominator for a complete run"}
    ratios = {}
    for name, metric in (("elapsed_ratio", "elapsed_s"), ("peak_rss_ratio", "peak_rss_bytes"),
                         ("bytes_per_row_ratio", "bytes_per_row")):
        a, b = current["cost"][metric], baseline["cost"][metric]
        if not all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in (a, b)):
            raise ValueError("invalid measured cost: " + metric)
        ratios[name] = a / b
    reasons = [key for key, value in ratios.items() if value > RELATIVE_LIMITS[key]]
    return {"applicable": True, "qualified": not reasons,
            "status": "exceeded" if reasons else "within_limits", "ratios": ratios,
            "limits": RELATIVE_LIMITS, "failed_metrics": reasons,
            "scope": "one_preregistered_same_source_same_extent_pair_not_a_population_confidence_bound"}


def validate_result_request(result, case, prepared_sha256, policy, *, pair):
    """Bind actual producer axes to the request, including retained failures."""
    expected = {"schema": "R1ControlledStepV2" if pair else "R1ControlledStepV1",
                "intervals": case["intervals"], "control_label": case["control"],
                "amplitude_V": case["amplitude_V"], "times_s": case["times_s"],
                "prepared_sha256": prepared_sha256, "policy": policy,
                "execution_axes": {"intervals": case["intervals"], "time_substeps": case["time_substeps"]}}
    for key, value in expected.items():
        if canonical(result.get(key)) != canonical(value):
            raise ValueError("actual producer differs from the V11 request: " + key)


def finalize_manifest(directory, summary, limits):
    """Seal all files and check the exact case allocation, including metadata."""
    for _ in range(10):
        summary["absolute_engineering_check"] = absolute_cost(summary["cost"], limits)
        summary["absolute_engineering_qualified"] = summary["absolute_engineering_check"]["qualified"]
        write(directory / "SummaryV1.json", summary)
        entries = {p.relative_to(directory).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size}
                   for p in sorted(directory.rglob("*")) if p.is_file() and p != directory / "ManifestV1.json"}
        write(directory / "ManifestV1.json", entries)
        size = sum(v["bytes"] for v in entries.values()) + (directory / "ManifestV1.json").stat().st_size
        if summary["cost"].get("case_artifact_bytes") == size:
            return
        summary["cost"]["case_artifact_bytes"] = size
    raise ValueError("V11 artifact accounting did not stabilize")


def run(args):
    if not (sys.flags.isolated and sys.flags.no_site):
        raise ValueError("V11 requires an isolated Python -I -S process")
    if any(os.environ.get(key) != "1" for key in THREAD_KEYS):
        raise ValueError("V11 requires all five numerical thread limits to be 1")
    output = args.output.resolve()
    request = load_request(args.request_file, args.request_sha256, output)
    budget = external_json(args.budget_file, args.budget_sha256)
    from scripts.analyze_r1_v8_prototype import validate_budget
    validate_budget(budget)
    if (budget.get("schema") != "R1V11PrecisionBudgetV1"
            or request["precision_budget_sha256"] != args.budget_sha256
            or budget.get("engineering", {}).get("relative_limits") != RELATIVE_LIMITS
            or budget["engineering"].get("absolute_limits") is not None):
        raise ValueError("V11 numerical/engineering budget binding differs")
    baseline = None
    if args.mode == "compensated":
        if args.baseline_summary is None or args.baseline_sha256 is None:
            raise ValueError("pair run requires its one preregistered baseline result")
        baseline = external_json(args.baseline_summary, args.baseline_sha256)
        if (not baseline.get("baseline_usable") or baseline.get("mode") != "baseline"
                or baseline.get("case_id") != request["case_id"]
                or baseline.get("source_commit") != args.expected_commit
                or baseline.get("source_content_sha256") != args.source_sha256
                or baseline.get("request_sha256") != args.request_sha256):
            raise ValueError("baseline identity or retained-prefix integrity differs")
    elif args.baseline_summary is not None or args.baseline_sha256 is not None:
        raise ValueError("baseline cannot import another baseline")
    initial_source = source_snapshot(args.expected_commit)
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_checkout import require_r1_checkout
    context = require_r1_checkout(project=PROJECT, formal=True, source_commit=args.expected_commit,
                                  expected_source_sha256=args.source_sha256)
    inputs = {name: context.read_bytes(spec["path"]) for name, spec in INPUTS.items()}
    if any(hashlib.sha256(inputs[name]).hexdigest() != spec["sha256"] for name, spec in INPUTS.items()):
        raise ValueError("frozen physical input changed")
    output.mkdir(parents=True, exist_ok=False)
    (output / "RequestV1.json").write_bytes(Path(args.request_file).read_bytes())
    (output / "SourceFixtureV1.yaml").write_bytes(inputs["fixture"])
    (output / "ReferenceBindingV1.json").write_bytes(inputs["reference"])
    write(output / "SourceReceiptV1.json", {**initial_source, "r1_source_content_sha256": args.source_sha256,
                                            "runner_sha256": sha(__file__)})
    if baseline is not None:
        (output / "InputBaselineSummaryV1.json").write_bytes(args.baseline_summary.read_bytes())
    import numpy as np
    import scipy
    from threadpoolctl import threadpool_info, threadpool_limits
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_backend import get_backend
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_state as states
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_protocol as protocol
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_physics_validation as physics
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_pair_codec as codec
    from perovskite_sim.experiments.one_dimensional_mechanism_r1 import build_r1_material
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_failure_reconstruction import rebuild_failure_witness
    from scripts.verify_r1_v7_precision import freeze_healthy_material
    backend = get_backend("pair" if args.mode == "compensated" else "float64")
    stack = load_device_from_yaml(output / "SourceFixtureV1.yaml")
    binding, case = json.loads(inputs["reference"]), request["case"]
    n = case["intervals"]
    grid, material = build_r1_material(stack, n)
    frozen = freeze_healthy_material(grid, material, source_identity=args.source_sha256,
                                     control_label=case["control"])
    write(output / "FrozenHealthyMaterialV1.json", frozen)
    functions_before = function_snapshot()
    summary = {"schema": "R1V11CaseRunV1", "scope": SCOPE, "case_id": request["case_id"],
               "kind": request["kind"], "case": case, "mode": args.mode,
               "source_commit": args.expected_commit, "source_content_sha256": args.source_sha256,
               "request_sha256": args.request_sha256, "precision_budget_sha256": args.budget_sha256,
               "backend_representation": backend.representation_id,
               "started_utc": datetime.now(timezone.utc).isoformat(),
               "scientifically_accepted": False, "formal_R1_2_qualification": False,
               "absolute_engineering_qualified": False,
               "execution_source_scope": "clean_commit_bound_by_outer_receipt; internal_API_records_keep_development_identity",
               "baseline_sha256": args.baseline_sha256}
    write(output / "SummaryV1.json", summary)

    def guard():
        if source_snapshot(args.expected_commit) != initial_source:
            raise ValueError("source changed during V11 execution")
        for path, digest in ((args.request_file, args.request_sha256), (args.budget_file, args.budget_sha256)):
            if sha(path) != digest:
                raise ValueError("external V11 contract changed")
        require_r1_checkout(project=PROJECT, formal=True, source_commit=args.expected_commit,
                            expected_source_sha256=args.source_sha256)

    try:
        with threadpool_limits(1):
            np.dot(np.ones((2, 2)), np.ones((2, 2)))
            pools = threadpool_info()
            if not pools or any(p["num_threads"] != 1 for p in pools):
                raise ValueError("numerical libraries are not demonstrably single-threaded")
            runtime = {"executable": sys.executable, "python": sys.version, "numpy": np.__version__,
                       "scipy": scipy.__version__, "platform": sys.platform, "machine": platform.machine(),
                       "host": platform.node(), "threads": {k: os.environ[k] for k in THREAD_KEYS}, "blas": pools}
            if runtime["numpy"] != "2.1.3" or runtime["scipy"] != "1.15.3":
                raise ValueError("runtime differs from the preregistered V9 dependency lineage")
            if baseline is not None and baseline.get("runtime_identity") != runtime:
                raise ValueError("baseline runtime differs")
            summary["runtime_identity"] = runtime
            write(output / "EnvironmentV1.json", runtime)

            def persist_numeric(path, result):
                if protocol.nonfinite_numeric_paths(result):
                    with Path(path).open("wb") as stream:
                        np.savez_compressed(stream, **codec.numeric_arrays(result, allow_nonfinite=True))
                else:
                    codec.write_numeric_sidecar(path, result)

            def verify_numeric(path, result):
                return (codec.verify_failed_numeric_sidecar if codec.contains_nonfinite_tags(result)
                        else codec.verify_numeric_sidecar)(path, result)

            step_policy = protocol.r1_policy(case["nonlinear_factor"], time_substeps=tuple(case["time_substeps"]))

            def integrate(prepared, observer):
                try:
                    result = protocol.run_r1_step(stack, n, binding, prepared,
                        control=case["control"], amplitude_V=case["amplitude_V"], times_s=case["times_s"],
                        policy=step_policy, physics_evidence=True, accepted_step_observer=observer,
                        expected_prepared_sha256=prepared.sha256, backend=backend)
                except Exception as exc:
                    partial = getattr(exc, "result", None)
                    if isinstance(partial, dict):
                        validate_result_request(states.json_data(partial), case, prepared.sha256,
                            states.json_data(step_policy), pair=backend.is_pair)
                        summary["result_request_binding_passed"] = True
                    raise
                validate_result_request(states.json_data({key: result[key] for key in (
                    "schema", "intervals", "control_label", "amplitude_V", "times_s", "prepared_sha256",
                    "policy", "execution_axes")}), case, prepared.sha256,
                    states.json_data(step_policy), pair=backend.is_pair)
                summary["result_request_binding_passed"] = True
                return result

            def prepare():
                prepared = states.prepare_common_state(stack, n, binding,
                    policy=protocol.r1_policy(), backend=backend)
                summary["required_pair_preparation_identity"] = validate_required_preparation(
                    prepared, request, pair=backend.is_pair)
                return prepared

            api = {"json_data": lambda v: tag_nonfinite(states.json_data(v)),
                   "prepare": prepare,
                   "run": integrate,
                   "persist_numeric": persist_numeric, "verify_numeric": verify_numeric,
                   "replay": lambda prepared, result, incomplete, observer: physics.verify_r1_step_physics(
                       stack, n, binding, prepared, result, allow_incomplete=incomplete, backend=backend,
                       row_observer=observer),
                   "failure_witness": lambda prepared, result, rows, failure: rebuild_failure_witness(
                       stack, n, binding, prepared, result, backend=backend)}
            if backend.is_pair:
                from scripts.analyze_r1_v9_prototype import analyze_records
                from scripts.verify_r1_v9_precision import digest

                def analyze(prepared, result, replay):
                    saved = states.json_data(prepared.to_dict())
                    return analyze_records(result["accepted_steps"], frozen, saved, result,
                        context={"source_digest": digest(result["source"]), "prepared_sha256": saved["sha256"],
                                 "result_sha256": result["sha256"], "saved_rows_digest": digest(result["accepted_steps"]),
                                 "source_commit": args.expected_commit, "request_sha256": args.request_sha256},
                        output=output / "IndependentAnalysis", budget=budget,
                        replay_receipt=getattr(replay, "replay_receipt", None))
                api["after_main"] = analyze
            summary.update(run_trajectory(output, case, args.mode, api, guard, extent_check=extent))
        guard()
        summary["source_unchanged"] = True
    except (Exception, KeyboardInterrupt) as exc:
        summary.update(execution_status="failed", runner_error=error_record(exc), four_predicates_passed=False)
        write(output / "RunnerFailureV1.json", summary["runner_error"])
        try:
            guard()
            summary["source_unchanged"] = True
        except Exception as source_exc:
            summary.update(source_unchanged=False, source_error=error_record(source_exc))
    summary["module_function_identity"] = check_functions(functions_before)
    summary.setdefault("cost", {})
    summary["integrity_passed"] = bool(summary.get("source_unchanged")
        and summary["module_function_identity"]["unchanged"] and not summary.get("runner_error")
        and not summary.get("replay_error") and not summary.get("failure_witness_error")
        and summary.get("result_request_binding_passed") and summary.get("numeric_sidecar_exact")
        and summary.get("replay_completed") and summary.get("extent", {}).get("prefix_valid")
        and summary.get("four_predicates", {}).get("observer_matches_result"))
    summary["baseline_usable"] = args.mode == "baseline" and summary["integrity_passed"]
    summary["numerical_passed"] = bool(summary["integrity_passed"] and summary.get("four_predicates_passed")
        and (args.mode == "baseline" or (summary.get("precision_records", {}).get("record_fields_present")
            and summary.get("independent_analysis", {}).get("passed"))))
    summary["engineering_check"] = relative_cost(summary, baseline)
    write(output / "PhaseTimingV1.json", summary.get("phase_timing", {}))
    finalize_manifest(output, summary, request["absolute_engineering_limits"])
    print(json.dumps({key: summary.get(key) for key in ("case_id", "mode", "execution_status", "extent",
        "integrity_passed", "numerical_passed", "baseline_usable", "engineering_check")}, indent=2))
    return 0 if (summary["numerical_passed"] and summary["engineering_check"]["qualified"] is not False
                 and summary["absolute_engineering_qualified"]) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("expected-commit", "source-sha256", "request-sha256", "budget-sha256"):
        parser.add_argument("--" + name, required=True)
    for name in ("request-file", "budget-file", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--mode", choices=("baseline", "compensated"), required=True)
    parser.add_argument("--baseline-summary", type=Path)
    parser.add_argument("--baseline-sha256")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
