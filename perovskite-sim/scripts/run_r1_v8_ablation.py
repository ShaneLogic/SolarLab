"""One-tier rebase-reference ablation with truthful physical history and replay.

The ordinary R1 multilevel runner is not used to pretend a single tier has
refinement evidence. All applicable original row gates are retained. A failed
ablated trajectory can be a valid diagnostic, but never a P1 qualification.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

from scripts import run_r1_v8_prototype as prototype


SCHEMA = "R1V8ReferenceAblationV1"
REFERENCE_FIELDS = ("phi_V", "dqfn_V", "dqfp_V", "n_m3", "p_m3", "positive_m3",
                    "occupancy", "trace_state_m3", "trace_potential_V")


def contract():
    return {"schema": "R1V8ReferenceAblationContractV1", "substeps": 4,
            "maximum_rows": 453, "first_intervention": "rebase_after_first_regular_accepted_step",
            "reference_fields": list(REFERENCE_FIELDS), "operation": "reference_low_words_to_zero",
            "physical_previous": "unchanged_full_precision",
            "history_consumers": ["storage", "charge_density_rate", "integrated_charge", "displacement_current"],
            "stop": "first_applicable_original_gate_failure_or_declared_end",
            "refinement_state_change": "not_applicable_single_tier",
            "refinement_current_change": "not_applicable_single_tier",
            "P1_qualified": False}


class AblationRowFailure(RuntimeError):
    pass


def validate_times(times):
    values = list(times)
    if (not 2 <= len(values) <= 114 or any(type(value) not in (float, int) or not math.isfinite(value)
            for value in values) or values[0] != 0. or values[-1] > 2.154434690031884
            or any(right <= left for left, right in zip(values[:-1], values[1:]))):
        raise ValueError("Invalid bounded ablation time schedule")
    return [float(value) for value in values]


def failure_record(exc, *, stage):
    """Keep a failed numeric witness JSON-safe without claiming it is finite."""
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_state as states
    from perovskite_sim.experiments.interface_defect_transient import InterfaceDefectTransientError
    markers = []
    def safe(value, path="numerical_evidence"):
        if isinstance(value, float) and not math.isfinite(value):
            markers.append(path)
            return {"nonfinite_float": repr(value)}
        if isinstance(value, dict):
            return {key: safe(item, path + "." + key) for key, item in value.items()}
        if isinstance(value, list):
            return [safe(item, f"{path}[{index}]") for index, item in enumerate(value)]
        return value
    evidence = safe(states.json_data(getattr(exc, "result", None)))
    return {"type": type(exc).__name__, "message": str(exc), "stage": stage,
            "original_numerical_solver_error": isinstance(exc, InterfaceDefectTransientError),
            "traceback": traceback.format_exc(), "numerical_evidence": evidence,
            "nonfinite_evidence_paths": markers,
            "nonfinite_encoding": "tagged_original_failed_values_not_replacement_numeric_evidence"}


def bounded_scientific_outcome(result):
    if result is None:
        return False
    if result.get("execution_status") == "completed" and result.get("complete") is True:
        return True
    failure = result.get("failure", {})
    if failure.get("stage") == "original_row_gate" and failure.get("type") == "AblationRowFailure":
        return True
    return (failure.get("stage") == "integration"
            and failure.get("original_numerical_solver_error") is True
            and isinstance(failure.get("numerical_evidence"), dict))


def time_schedule(times):
    schedule = [(0., 0.)]
    for left, right in zip(times[:-1], times[1:]):
        dt = (right - left) / 4
        schedule.extend((left + (step + 1) * dt, dt) for step in range(4))
    return schedule


def attempt_inputs(working, previous, coordinate, voltage, dt, index, when):
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_state as states
    return states.json_data({"schema": "R1V8AblationSolverAttemptV1", "row_index": index,
        "time_s": when, "dt_s": dt, "voltage_V": voltage, "substeps": 4,
        "starting_coordinate": coordinate, "physical_previous": states.snapshot(working, previous),
        "reference_quantization": working.rebase_evidence,
        "scope": "actual_solver_input_not_an_accepted_state_or_reexecuted_Newton_path"})


@contextmanager
def capture_solver_attempt(record, times):
    from perovskite_sim.experiments import interface_defect_transient as transient
    original = transient._solve_step
    schedule = time_schedule(times)
    def capture(system, coordinate, previous, voltage, dt, policy, **kwargs):
        index = len(record["accepted_steps"])
        when, expected_dt = schedule[index]
        if dt != expected_dt:
            raise ValueError("Ablation solver attempted an undeclared duration")
        record["last_solver_attempt"] = attempt_inputs(system, previous, coordinate, voltage, dt, index, when)
        return original(system, coordinate, previous, voltage, dt, policy, **kwargs)
    transient._solve_step = capture
    try:
        yield
    finally:
        transient._solve_step = original


def row_gates(row, policy):
    """Apply unchanged original limits, with explicit initial-row applicability."""
    import numpy as np
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_protocol as protocol
    diagnostic, physical = row["physics_reconstruction"], row["physical"]
    finite = row["phase"] == "accepted_regular_step"
    metrics = {
        "local_carrier_residual": diagnostic["local_carrier_residual"],
        "local_gauss_residual": diagnostic["local_gauss_residual"],
        "eliminated_operator_error": diagnostic["eliminated_operator_error"],
        "inventory_relative_drift": physical["inventory_relative_drift"],
        "full_gauss_normalized": physical["gauss_normalized"],
    }
    current = np.asarray(diagnostic["conduction_A_m2"])
    expected = np.asarray(diagnostic["carrier_conduction_A_m2"]) + np.asarray(diagnostic["positive_ion_current_A_m2"])
    metrics["current_decomposition_error"] = float(np.max(np.abs(current - expected))) / max(float(np.max(np.abs(current))), 1.)
    if finite:
        metrics.update({
            "nonlinear_residual": diagnostic["scaled_nonlinear_residual"],
            "charge_balance_relative": diagnostic["charge_balance_relative"],
            "all_face_current_relative": diagnostic["all_face_current_relative"],
            "interface_current_relative": diagnostic["interface_current_relative"],
            "full_charge_balance_normalized": physical["charge_balance_normalized"],
            "physical_current_spread_relative": physical["contact_internal_current_spread_relative"],
            "trap_storage_normalized_error": physical["trap_storage_check"]["normalized_error"],
        })
        if diagnostic["jacobian_checked"]:
            metrics["analytic_jacobian_error"] = diagnostic["analytic_jacobian_error"]
    else:
        # _trace_certificate includes the initial derivative-current maxima;
        # retain those two original gates in this single-tier route as well.
        regular = physical["regular_right_limit"]
        metrics["all_face_current_relative"] = regular["face_current_spread_relative"]
        metrics["interface_current_relative"] = regular["interface_current_spread_relative"]
    violations = [name for name, value in metrics.items()
                  if not np.isfinite(value) or value > prototype.LIMITS[name]]
    paths = protocol.nonfinite_numeric_paths(row)
    return {"metrics": metrics, "limits": {name: prototype.LIMITS[name] for name in metrics},
            "passed": not violations and not paths and row["physical_checks_passed"],
            "reasons": violations + list(row["physical_failure_reasons"]) + (["nonfinite_row"] if paths else []),
            "unavailable": ["refinement_state_change", "refinement_current_change"],
            "scope": "applicable_original_row_gates_only_no_multilevel_certificate"}


def setup(stack, binding, prepared, policy, amplitude):
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_state as states
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_step import build_initial_step
    system, before = states.restore_common_state(prepared, stack, 16, binding,
        policy=policy, expected_prepared_sha256=prepared.sha256)
    initial = build_initial_step(system, before, amplitude, policy=policy)
    if not hasattr(initial.system, "enable_reference_quantization"):
        raise RuntimeError("Explicit reference-only ablation API is unavailable")
    return initial


def make_row(initial, working, state, previous, voltage, dt, when, policy, *, first_finite, integrated):
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_state as states
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_protocol as protocol
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_physics_validation as physics
    physical = physics._physical(initial.system, working, state, previous, dt, policy, initial)
    if previous is not None:
        integrated += float(working.polarity * physical["contact_maxwell_A_m2"][0] * dt)
    diagnostic = physics.capture_r1_physics_row(initial.system, working, state, previous,
        voltage, dt, policy, check_jacobian=first_finite)
    row = {"phase": "0+" if previous is None else "accepted_regular_step",
           "time_s": float(when), "dt_s": float(dt), "substeps": 4,
           "scaled_nonlinear_residual": diagnostic["scaled_nonlinear_residual"],
           "state": states.snapshot(working, state), "physical": physical,
           "physics_reconstruction": diagnostic, "solver_accepted": previous is not None,
           "regular_integrated_charge_C_m2": integrated}
    evidence = dict(row)
    if previous is None:
        evidence["initial_event"] = initial.event
    checks = protocol._physical_step_checks(physical, finite_step=previous is not None,
                                            policy=policy, evidence=evidence)
    row.update(physical_checks=checks, physical_checks_passed=checks["passed"],
               physical_failure_reasons=list(checks["reasons"]))
    row["applicable_original_gates"] = row_gates(row, policy)
    return states.json_data(row)


def run_trace(stack, binding, prepared, times, policy, *, observer=None, request_sha256=None):
    import numpy as np
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_state as states
    from perovskite_sim.experiments.interface_defect_transient import _integrate_trace
    times = np.asarray(validate_times(times), dtype=float)
    if request_sha256 is not None:
        prototype.validate_digest(request_sha256, "ablation request_sha256")
    record = {"schema": SCHEMA, "contract": contract(), "prepared_sha256": prepared.sha256,
              "reference_sha256": binding["sha256"], "times_s": times.tolist(),
              "policy": states.json_data(policy), "amplitude_V": .005, "control": "D",
              "request_sha256": request_sha256, "initial_event": None, "accepted_steps": [],
              "expected_rows": 1 + 4 * (len(times) - 1), "P1_qualified": False,
              "refinement_evidence": {"available": False, "reason": "single_tier"}}
    integrated = 0.
    first_regular = True
    stage = "initialization"

    def observe(working, state, previous, dt, when, substeps, residual):
        nonlocal integrated, first_regular, stage
        if substeps != 4:
            raise ValueError("Ablation runs only the declared tier4")
        stage = "accepted_state_diagnostics"
        try:
            row = make_row(initial, working, state, previous, .005, dt, when, policy,
                           first_finite=previous is not None and first_regular, integrated=integrated)
        except (Exception, KeyboardInterrupt):
            record["failed_observer_state"] = states.json_data({
                "scope": "actual_solver_state_not_fully_observed_or_physically_accepted",
                "time_s": when, "dt_s": dt, "substeps": substeps,
                "state": states.snapshot(working, state), "coordinate": state.coordinate,
                "physical_previous": None if previous is None else states.snapshot(working, previous)})
            raise
        integrated = row["regular_integrated_charge_C_m2"]
        record["accepted_steps"].append(row)
        if observer is not None:
            stage = "observer_persistence"
            observer(row)
        if not row["applicable_original_gates"]["passed"]:
            stage = "original_row_gate"
            raise AblationRowFailure(", ".join(row["applicable_original_gates"]["reasons"]))
        if previous is not None and first_regular:
            initial.system.enable_reference_quantization()
            first_regular = False
        stage = "integration"

    try:
        initial = setup(stack, binding, prepared, policy, .005)
        record["initial_event"] = states.json_data(initial.event)
        stage = "integration"
        with capture_solver_attempt(record, times):
            trace = _integrate_trace(initial.system, times, np.full(times.size, .005), 4, policy,
                accepted_step_observer=observe, initial_state=initial.zero_plus,
                initial_current_metrics=initial.initial_current_metrics)
        record["execution_status"] = "completed"
        record["last_output_time_s"] = float(trace.times[-1])
        record.pop("last_solver_attempt", None)
    except (Exception, KeyboardInterrupt) as exc:
        record["execution_status"] = "failed"
        record["failure"] = failure_record(exc, stage=stage)
        if isinstance(exc, AblationRowFailure):
            record["failure"].update(record_index=len(record["accepted_steps"]) - 1,
                reasons=record["accepted_steps"][-1]["applicable_original_gates"]["reasons"])
    record["observed_rows"] = len(record["accepted_steps"])
    record["intervention_rows"] = sum("reference_quantization" in r["physics_reconstruction"]
                                      for r in record["accepted_steps"])
    record["complete"] = record["execution_status"] == "completed" and record["observed_rows"] == record["expected_rows"]
    record["regular_integrated_charge_C_m2"] = integrated
    record["source"] = states.execution_source()
    try:
        record["sha256"] = states.digest(record)
    except Exception as exc:
        # A nonfinite failed observer row cannot be a normal sealed record,
        # but its complete raw values must remain attached to the failure.
        exc.result = record
        raise
    return states.json_data(record)


def validate_replay_header(prepared, binding, record, *, expected_request_sha256=None, expected_times=None):
    """Close claims that otherwise survive a resealed row-exact replay."""
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_state as states
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_protocol as protocol
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_physics_validation as physics
    required = {"schema", "contract", "prepared_sha256", "reference_sha256", "times_s", "policy",
        "amplitude_V", "control", "request_sha256", "initial_event", "accepted_steps", "expected_rows",
        "P1_qualified", "refinement_evidence", "execution_status", "observed_rows", "intervention_rows",
        "complete", "regular_integrated_charge_C_m2", "source", "sha256"}
    if not required <= set(record) or set(record) - required - {
            "failure", "last_output_time_s", "failed_observer_state", "last_solver_attempt"}:
        raise ValueError("Unclassified or missing ablation replay metadata")
    physics._same(record["schema"], SCHEMA, "ablation schema")
    physics._same(record["contract"], contract(), "ablation contract")
    physics._same(record["prepared_sha256"], prepared.sha256, "preparation")
    physics._same(record["reference_sha256"], binding["sha256"], "reference")
    physics._same(record["sha256"], states.digest({k: v for k, v in record.items() if k != "sha256"}), "record hash")
    physics._same(record["source"], states.execution_source(), "ablation execution source")
    physics._same(record["amplitude_V"], .005, "ablation voltage")
    physics._same(record["control"], "D", "ablation control")
    physics._same(record["P1_qualified"], False, "ablation qualification scope")
    physics._same(record["refinement_evidence"], {"available": False, "reason": "single_tier"}, "refinement scope")
    if record["request_sha256"] is not None:
        prototype.validate_digest(record["request_sha256"], "record request_sha256")
    if expected_request_sha256 is not None:
        physics._same(record["request_sha256"], expected_request_sha256, "external ablation request")
    times = validate_times(record["times_s"])
    if expected_times is not None:
        physics._same(times, validate_times(expected_times), "external ablation time array")
    rows = record["accepted_steps"]
    if not isinstance(rows, list):
        raise ValueError("Ablation accepted steps must be a list")
    expected_count = 1 + 4 * (len(times) - 1)
    physics._same(record["expected_rows"], expected_count, "expected ablation rows")
    physics._same(record["observed_rows"], len(rows), "observed ablation rows")
    interventions = sum("reference_quantization" in row["physics_reconstruction"] for row in rows)
    physics._same(record["intervention_rows"], interventions, "ablation intervention count")
    if len(rows) > expected_count:
        raise ValueError("Ablation contains more rows than its bounded schedule")
    if record["execution_status"] == "completed":
        physics._same(record["complete"], True, "completed ablation extent")
        physics._same(len(rows), expected_count, "completed ablation row count")
        physics._same(record.get("last_output_time_s"), times[-1], "completed ablation endpoint")
        if "failure" in record or "failed_observer_state" in record or "last_solver_attempt" in record:
            raise ValueError("Completed ablation contains failure evidence")
    elif record["execution_status"] == "failed":
        physics._same(record["complete"], False, "failed ablation extent")
        if not isinstance(record.get("failure"), dict) or "last_output_time_s" in record:
            raise ValueError("Failed ablation lacks failure evidence or claims a completed endpoint")
    else:
        raise ValueError("Unrecognized ablation execution status")
    policy = protocol.r1_policy(.1)
    physics._same(record["policy"], states.json_data(policy), "unchanged policy")
    return policy, times


def replay_trace(stack, binding, prepared, record, *, expected_request_sha256=None, expected_times=None):
    """Rebuild each accepted coordinate with the same explicit reference intervention."""
    import numpy as np
    from dataclasses import replace
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_state as states
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_physics_validation as physics
    policy, times = validate_replay_header(prepared, binding, record,
                                           expected_request_sha256=expected_request_sha256,
                                           expected_times=expected_times)
    initial = setup(stack, binding, prepared, policy, .005)
    physics._same(record["initial_event"], states.json_data(initial.event), "initial event")
    schedule = time_schedule(times)
    rows = record["accepted_steps"]
    if len(rows) > len(schedule) or (record["complete"] and len(rows) != len(schedule)):
        raise ValueError("Ablation row coverage is inconsistent")
    previous, integrated, passed = None, 0., True
    first_failure = None
    reconstructed_attempts = {}
    for index, (row, (when, dt)) in enumerate(zip(rows, schedule)):
        physics._same([row["time_s"], row["dt_s"], row["substeps"]], [when, dt, 4], "ablation schedule")
        coordinate = np.asarray(row["physics_reconstruction"]["coordinate"], dtype=float)
        if index == 0:
            physics._same(coordinate, np.zeros(initial.system.dimension), "initial coordinate")
            working, local_previous = initial.system, None
            state = replace(initial.zero_plus, coordinate=coordinate.copy())
        else:
            working, local_previous = initial.system.rebase(previous)
            working.set_voltage_lift(.005, local_previous)
            reconstructed_attempts[index] = attempt_inputs(working, local_previous,
                np.zeros(initial.system.dimension), .005, dt, index, when)
            state = working.evaluate(coordinate, .005)
        rebuilt = make_row(initial, working, state, local_previous, .005, dt, when, policy,
                           first_finite=index == 1, integrated=integrated)
        integrated = rebuilt["regular_integrated_charge_C_m2"]
        physics._same(row, rebuilt, f"ablation accepted row {index}")
        passed &= rebuilt["applicable_original_gates"]["passed"]
        intervention = rebuilt["physics_reconstruction"].get("reference_quantization")
        if (index < 2 and intervention is not None) or (index >= 2 and intervention is None):
            raise ValueError("Ablation did not apply its frozen intervention schedule")
        if not rebuilt["applicable_original_gates"]["passed"]:
            first_failure = index
            if index != len(rows) - 1:
                raise ValueError("Ablation continued after its first original-gate failure")
        if index == 1:
            initial.system.enable_reference_quantization()
        previous = state
    physics._same(record["regular_integrated_charge_C_m2"], integrated, "ablation final integrated charge")
    if first_failure is not None:
        if record["execution_status"] != "failed":
            raise ValueError("Ablation with an original-gate failure cannot be completed")
        failure = record["failure"]
        physics._same(failure.get("type"), "AblationRowFailure", "terminal original-gate failure type")
        physics._same(failure.get("stage"), "original_row_gate", "terminal original-gate stage")
        physics._same(failure.get("record_index"), first_failure, "terminal failure row")
        physics._same(failure.get("reasons"), rows[first_failure]["applicable_original_gates"]["reasons"],
                      "terminal original-gate reasons")
    elif record.get("failure", {}).get("type") == "AblationRowFailure":
        raise ValueError("Ablation claims an original-gate failure without a violating row")
    attempt = record.get("last_solver_attempt")
    attempt_reconstructed = False
    if attempt is not None:
        index = attempt.get("row_index")
        if type(index) is not int or index < 1 or index >= len(schedule) or index not in (len(rows) - 1, len(rows)):
            raise ValueError("Failed ablation's attempted step is not adjacent to its saved prefix")
        if index not in reconstructed_attempts:
            if previous is None:
                raise ValueError("Failed solver attempt has no reconstructed physical previous")
            when, dt = schedule[index]
            working, local_previous = initial.system.rebase(previous)
            working.set_voltage_lift(.005, local_previous)
            reconstructed_attempts[index] = attempt_inputs(working, local_previous,
                np.zeros(initial.system.dimension), .005, dt, index, when)
        physics._same(attempt, reconstructed_attempts[index], "failed solver actual reference and previous inputs")
        attempt_reconstructed = True
    return {"schema": "R1V8AblationReplayV1", "evidence_exact": True, "rows": len(rows),
            "complete": record["complete"], "applicable_original_gates_passed": passed,
            "failed_prefix_preserved": not record["complete"], "P1_qualified": False,
            "first_original_gate_failure_row": first_failure,
            "last_solver_attempt_inputs_reconstructed": attempt_reconstructed,
            "intervention_attempt_reconstructed": bool(attempt_reconstructed and attempt["reference_quantization"]),
            "failure_provenance_reexecuted": False,
            "external_time_array_bound": expected_times is not None,
            "refinement_evidence": "unavailable_single_tier"}


def run(args):
    for key in prototype.THREAD_KEYS:
        os.environ[key] = "1"
    if not (sys.flags.isolated and sys.flags.no_site):
        raise ValueError("Use isolated Python -I -S -B")
    output = args.output.resolve()
    request, request_bytes = prototype.load_contract(args.request_file, args.request_sha256, output)
    if output.exists():
        raise FileExistsError(output)
    source = prototype.source_snapshot(args.expected_commit)
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_checkout import require_r1_checkout
    checkout = require_r1_checkout(project=prototype.PROJECT, formal=True, source_commit=args.expected_commit,
                                   expected_source_sha256=args.source_sha256)
    from threadpoolctl import threadpool_limits, threadpool_info
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_precision import precision_context
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_state as states
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_protocol as protocol
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    accepted_path = output / "AcceptedStepsV1.jsonl"
    stage = "input_persistence"
    result, replay, error = None, None, None
    summary = {"schema": SCHEMA, "source_commit": args.expected_commit,
               "source_content_sha256": args.source_sha256, "request_sha256": args.request_sha256,
               "execution_status": "not_started", "rows": 0, "expected_rows": 453,
               "complete": False, "intervention_rows": 0, "P1_qualified": False,
               "scope": "single_tier_reference_precision_ablation_not_engineering_comparator"}
    def persist(row):
        with accepted_path.open("a") as stream:
            stream.write(prototype.canonical(row) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    try:
        (output / "RequestV1.json").write_bytes(request_bytes)
        prototype.write(output / "AblationContractV1.json", contract())
        prototype.write(output / "SourceReceiptV1.json", {**source, "r1_source_content_sha256": checkout.source_content_sha256})
        for key, filename in (("fixture", "SourceFixtureV1.yaml"), ("reference", "ReferenceBindingV1.json")):
            raw = checkout.read_bytes(prototype.INPUTS[key]["path"])
            if hashlib.sha256(raw).hexdigest() != prototype.INPUTS[key]["sha256"]:
                raise ValueError("Ablation input differs from its frozen request: " + key)
            (output / filename).write_bytes(raw)
        stack = load_device_from_yaml(output / "SourceFixtureV1.yaml")
        binding = json.loads((output / "ReferenceBindingV1.json").read_text())
        accepted_path.touch()
        with threadpool_limits(1), precision_context():
            pools = threadpool_info()
            if not pools or any(pool["num_threads"] != 1 for pool in pools):
                raise ValueError("Ablation libraries do not evidence single-thread execution")
            prototype.write(output / "EnvironmentV1.json", {"python": sys.version, "blas": pools})
            stage = "preparation"
            prepared = states.prepare_common_state(stack, 16, binding, policy=protocol.r1_policy())
            prototype.write(output / "PreparedV1.json", prepared.to_dict())
            stage = "trajectory"
            result = run_trace(stack, binding, prepared, request["case"]["times_s"], protocol.r1_policy(.1),
                               observer=persist, request_sha256=args.request_sha256)
            stage = "result_persistence"
            prototype.write(output / "ResultV1.json", result)
            if "failure" in result:
                prototype.write(output / "TrajectoryFailureV1.json", result["failure"])
            stage = "replay"
            replay = replay_trace(stack, binding, prepared, result, expected_request_sha256=args.request_sha256,
                                  expected_times=request["case"]["times_s"])
            prototype.write(output / "PhysicsReplayV1.json", replay)
    except (Exception, KeyboardInterrupt) as exc:
        error = failure_record(exc, stage=stage)
        prototype.write(output / "RunnerFailureV1.json", error)
    if result is not None:
        summary.update(execution_status=result["execution_status"], complete=result["complete"],
                       intervention_rows=result["intervention_rows"],
                       trajectory_failure=result.get("failure"))
    elif error is not None:
        summary["execution_status"] = "failed"
    observed = []
    try:
        if accepted_path.exists():
            observed = [json.loads(line) for line in accepted_path.read_text().splitlines()]
        summary["observer_matches"] = result is not None and observed == result["accepted_steps"]
    except Exception as exc:
        summary["observer_matches"] = False
        summary["observer_readback_error"] = failure_record(exc, stage="observer_readback")
    summary["rows"] = len(observed)
    try:
        summary["source_unchanged"] = prototype.source_snapshot(args.expected_commit) == source
        require_r1_checkout(project=prototype.PROJECT, formal=True, source_commit=args.expected_commit,
                            expected_source_sha256=args.source_sha256)
        summary["request_unchanged"] = (prototype.sha(args.request_file) == args.request_sha256
                                         and prototype.sha(output / "RequestV1.json") == args.request_sha256)
        summary["contract_unchanged"] = json.loads((output / "AblationContractV1.json").read_text()) == contract()
    except Exception as exc:
        summary["source_unchanged"] = False
        summary["final_identity_error"] = failure_record(exc, stage="final_source_request_contract_check")
    summary.update(replay=replay, runner_failure=error, elapsed_s=time.monotonic() - started)
    intervention_observed = (result is not None and result["intervention_rows"] > 0
        or replay is not None and replay.get("intervention_attempt_reconstructed") is True)
    summary["bounded_scientific_outcome_recorded"] = bounded_scientific_outcome(result)
    summary["diagnostic_evidence_complete"] = bool(error is None and summary["bounded_scientific_outcome_recorded"]
        and summary["source_unchanged"]
        and summary.get("request_unchanged") and summary.get("contract_unchanged")
        and summary["observer_matches"] and replay is not None
        and replay["evidence_exact"] and replay.get("external_time_array_bound") is True and intervention_observed)
    prototype.write(output / "SummaryV1.json", summary)
    prototype.write(output / "ManifestV1.json", {p.relative_to(output).as_posix(): {
        "sha256": prototype.sha(p), "bytes": p.stat().st_size}
        for p in sorted(output.rglob("*")) if p.is_file() and p.name != "ManifestV1.json"})
    print(json.dumps({k: summary[k] for k in ("execution_status", "rows", "intervention_rows", "diagnostic_evidence_complete")}))
    return 0 if summary["diagnostic_evidence_complete"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--request-file", required=True, type=Path)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        return run(args)
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
