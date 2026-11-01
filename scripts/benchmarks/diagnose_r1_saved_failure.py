"""Freeze and diagnose R1 saved inputs; this tool never advances time.

Historical source/environment identities belong to the archived producer.
Current equation evaluations, when requested, have a separate report identity.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch


REPOSITORY = Path(__file__).resolve().parents[2]
PROJECT = REPOSITORY / "perovskite-sim"
FIXTURE = REPOSITORY / "tests/fixtures/refactor/R1FailureWitnessV1.json"
THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)
SCHEMA = "SolarLabR1FailureFixtureV1"


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def content_sha256(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def read_json(path, maximum_bytes=512 * 1024 * 1024):
    if Path(path).stat().st_size > maximum_bytes:
        raise ValueError(f"input exceeds bounded read size: {path}")
    with Path(path).open() as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def current_runtime():
    import numpy as np
    import scipy
    import platform
    import perovskite_sim
    from threadpoolctl import threadpool_info

    return {
        "python": sys.version, "executable": sys.executable,
        "numpy": np.__version__, "scipy": scipy.__version__,
        "platform": platform.platform(), "package_file": perovskite_sim.__file__,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True,
        ).strip(),
        "git_branch": subprocess.check_output(
            ["git", "symbolic-ref", "--short", "HEAD"], cwd=REPOSITORY, text=True,
        ).strip(),
        "thread_environment": {name: os.environ.get(name) for name in THREAD_VARIABLES},
        "observed_blas_pools": threadpool_info(),
        "actual_blas_threads_verified": bool(threadpool_info()),
        "historical_execution_identity_claimed": False,
    }


def load_fixture(path=FIXTURE):
    value = read_json(path, 16 * 1024 * 1024)
    if value.get("schema") != SCHEMA:
        raise ValueError("unsupported saved R1 fixture schema")
    body = {key: item for key, item in value.items() if key != "sha256"}
    if value.get("sha256") != content_sha256(body):
        raise ValueError("saved R1 fixture content hash mismatch")
    if len(value["request"]["certificate_limits"]) != 16:
        raise ValueError("original certificate threshold coverage changed")
    if len(value["operator_channels"]) != 13:
        raise ValueError("original physical channel coverage changed")
    if value["prefix"]["accepted_rows"] != 309 or value["request"]["expected_rows"] != 3727:
        raise ValueError("saved prefix extent differs from original request")
    if value["prefix"]["bilateral_100s_qualified"] or value["prefix"]["compensated_admitted"]:
        raise ValueError("a failed prefix cannot grant full-window qualification")
    return value


@contextmanager
def forbid_trajectory_work():
    """Fail closed if a diagnostic accidentally invokes preparation or time stepping."""
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_state as state
    from perovskite_sim.experiments import interface_defect_transient as transient

    def forbidden(*args, **kwargs):
        raise AssertionError("saved-state diagnosis cannot prepare DC or advance time")

    with ExitStack() as patches:
        for module, name in ((state, "prepare_common_state"), (state, "solve_r1_dc"),
                             (transient, "_solve_step")):
            patches.enter_context(patch.object(module, name, forbidden))
        yield


def array_comparison(actual, expected):
    import numpy as np

    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    result = {"actual_shape": list(actual.shape), "expected_shape": list(expected.shape)}
    if actual.shape != expected.shape:
        return {**result, "exact_bytes": False, "shape_mismatch": True}
    words = actual.view(np.uint64).ravel() != expected.view(np.uint64).ravel()
    sign = np.uint64(1 << 63)
    a_bits, e_bits = actual.view(np.uint64), expected.view(np.uint64)
    a_order = np.where(a_bits & sign, ~a_bits, a_bits | sign)
    e_order = np.where(e_bits & sign, ~e_bits, e_bits | sign)
    ulps = np.maximum(a_order, e_order) - np.minimum(a_order, e_order)
    delta = np.abs(actual - expected)
    scale = np.maximum(np.abs(expected), np.finfo(float).tiny)
    indices = np.flatnonzero(words)
    result.update(
        exact_bytes=actual.tobytes() == expected.tobytes(),
        different_words=int(np.count_nonzero(words)),
        maximum_absolute=float(np.max(delta)) if delta.size else 0.,
        maximum_relative=float(np.max(delta / scale)) if delta.size else 0.,
        maximum_ulp=int(np.max(ulps)) if ulps.size else 0,
        first_differences=[{
            "flat_index": int(i), "actual": float(actual.ravel()[i]),
            "saved": float(expected.ravel()[i]),
            "actual_hex": float(actual.ravel()[i]).hex(),
            "saved_hex": float(expected.ravel()[i]).hex(),
        } for i in indices[:6]],
    )
    return result


def snapshot_comparison(actual, expected):
    return {name: array_comparison(actual[name], expected[name]) for name in expected}


@dataclass
class SavedPreviousState:
    """Only bound saved fields needed as the predecessor of a point evaluation.

    This is deliberately not a complete production state: an unexpected read
    of an unrecorded field raises AttributeError instead of using a fabricated
    Jacobian, flux, optimizer result, or an earlier state's cached value.
    """

    coordinate: object
    dqfn: object
    dqfp: object
    n: object
    p: object
    occupancy: object
    positive: object
    negative: object
    phi: object
    local: tuple
    storage: object
    rate: object
    sheet_charge: object
    poisson_residual: object
    local_residual: object
    direct_poisson_residual: object


def direct_saved_reference(base, row):
    """Copy original primary arrays, without exp/log reconstruction or solving."""
    import numpy as np

    raw, equation = row["state"], row["physics_reconstruction"]

    def value(name):
        return np.asarray(raw[name], dtype=float).copy()

    local = tuple(SimpleNamespace(
        trace_potential=np.asarray(trace, dtype=float).copy(),
        state_m3=np.asarray(density, dtype=float).copy(),
    ) for trace, density in zip(raw["trace_potential_V"], raw["trace_state_m3"]))
    return SavedPreviousState(
        coordinate=np.asarray(equation["coordinate"], dtype=float).copy(),
        dqfn=np.asarray(equation["electron_qf_increment_V"], dtype=float).copy(),
        dqfp=np.asarray(equation["hole_qf_increment_V"], dtype=float).copy(),
        n=value("n_m3"), p=value("p_m3"), occupancy=value("occupancy"),
        positive=value("positive_m3"), negative=None, phi=value("phi_V"), local=local,
        storage=np.asarray(equation["storage"], dtype=float).copy(),
        rate=np.asarray(equation["rate"], dtype=float).copy(),
        sheet_charge=value("sheet_charge_C_m2"), poisson_residual=value("poisson_residual_C_m2"),
        local_residual=value("local_residual"),
        direct_poisson_residual=np.asarray(equation["direct_poisson_residual_C_m2"], dtype=float).copy(),
    )


def saved_input_system(fixture):
    """Build coefficients from the sealed preparation, retaining import failure.

    The private constructor is used only for same-input diagnostics after the
    public historical verifier is attempted. No source metadata or saved
    certificate is edited, and this route never grants preparation acceptance.
    """
    import numpy as np
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_state as state
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_dynamics import R1DynamicsControls
    from perovskite_sim.models.config_loader import load_device_from_yaml

    raw = fixture["prepared"]
    prepared = state.R1PreparedState.from_dict(raw)
    policy = state._preparation_policy(fixture["policy"])
    source = fixture["request"]["inputs"]["fixture"]
    source_path = PROJECT / source["path"]
    if file_sha256(source_path) != source["sha256"]:
        raise ValueError("local material fixture differs from bound saved input")
    stack = load_device_from_yaml(source_path)
    verification = {"historical_prepared_verified": False}
    try:
        base, initial = state.verify_prepared_physics(
            prepared, stack, fixture["reference"], policy=policy,
        )
        verification["historical_prepared_verified"] = True
    except state.R1StateError as error:
        verification["original_verifier_error"] = str(error)
        grid, material = state.build_r1_material(stack, raw["intervals"])
        base = state._make_system(
            stack, grid, material, state._decode_dc(raw["dc_state"]), fixture["reference"],
            R1DynamicsControls.from_label(fixture["control_label"]),
            state._preparation_policy(raw["preparation_policy"]),
        )
        initial = base.evaluate(base.initial_coordinate(), 0.)
    verification["prepared_evaluation_differences"] = snapshot_comparison(
        state.snapshot(base, initial), raw["state"],
    )
    context = fixture["initial_context"]
    references = {
        "electron_qf_reference_V": base.qfn_reference,
        "hole_qf_reference_V": base.qfp_reference,
        "initial_reference_n_m3": base.reference_n,
        "initial_reference_p_m3": base.reference_p,
        "initial_reference_positive_m3": base.reference_positive,
        "initial_reference_occupancy": base.reference_occupancy,
        "fixed_equilibrium_occupancy": base.equilibrium_occupancy,
    }
    verification["reference_differences"] = {
        name: array_comparison(value, context[name]) for name, value in references.items()
    }
    if not all(item["exact_bytes"] for item in verification["reference_differences"].values()):
        raise ValueError("same-input point construction has changed a bound initial reference")
    verification["scope"] = "point operators on copied saved input; no historical execution certification"
    return base, policy, verification


def restored_point(fixture, case_name, base, policy):
    import numpy as np
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import snapshot

    case = fixture["cases"][case_name]
    raw = case.get("saved_witness")
    row = case.get("saved_row")
    voltage = float(fixture["request"]["case"]["amplitude_V"])
    previous = direct_saved_reference(base, case["previous_row"])
    working, previous = base.rebase(previous)
    working.set_voltage_lift(voltage, previous)
    dt = float(raw["dt_s"] if raw is not None else row["dt_s"])
    if raw is not None:
        coordinate = np.asarray(raw["coordinate"], dtype=float)
        expected_residual = raw["scaled_residual_vector"]
        expected_scales = [raw[name] for name in ("storage_scale", "poisson_scale", "local_scale")]
        expected_snapshot = raw["attempted_state"]
    else:
        equation = row["physics_reconstruction"]
        coordinate = np.asarray(equation["coordinate"], dtype=float)
        expected_residual = equation["scaled_residual_vector"]
        expected_scales = [equation[name] for name in ("storage_scale", "poisson_scale_C_m2", "local_algebraic_scale")]
        expected_snapshot = row["state"]
    recomputed_previous = working.evaluate(np.zeros(working.dimension), voltage)
    previous_checks = snapshot_comparison(
        snapshot(working, recomputed_previous), case["previous_row"]["state"],
    )
    primary_names = ("n_m3", "p_m3", "positive_m3", "occupancy", "phi_V",
                     "trace_potential_V", "trace_state_m3")
    if not all(previous_checks[name]["exact_bytes"] for name in primary_names):
        raise ValueError("zero-coordinate evaluation changed a copied predecessor primary word")
    computed_scales = [base.storage_scale(previous.storage, recomputed_previous, dt, policy),
                       base.poisson_scale(policy), base.local_algebraic_scale(policy)]
    scale_checks = [array_comparison(actual, saved) for actual, saved in zip(computed_scales, expected_scales)]
    # The saved scale is part of the frozen input. A different current-platform
    # derivative-derived scale is reported, never silently substituted into it.
    scales = [np.asarray(value, dtype=float).copy() for value in expected_scales]
    residual, jacobian, attempted = working.residual_and_jacobian(
        coordinate, voltage, previous, dt, *scales,
    )
    comparisons = {
        "scales": scale_checks,
        "evaluation_scale_source": "unaltered bound witness; recomputed scales reported separately",
        "predecessor_zero_evaluation": previous_checks,
        "residual": array_comparison(residual, expected_residual),
        "attempted_snapshot": snapshot_comparison(snapshot(working, attempted), expected_snapshot),
        "previous_coordinate": array_comparison(previous.coordinate, raw["previous_coordinate"])
        if raw is not None else {"exact_bytes": True, "scope": "zero coordinate assigned by original rebase"},
        "saved_predecessor_primaries_copied_exactly": True,
        "copied_state_is_not_a_new_preparation": True,
    }
    return working, previous, coordinate, voltage, dt, scales, residual, jacobian, attempted, comparisons


def local_input_words(state):
    values = state.local[0].carrier_data.inputs
    return {name: {word: getattr(getattr(values, name), word).tolist() for word in ("hi", "lo")}
            for name in ("state_density", "trace_potential", "bulk_density", "bulk_potential", "occupancy")}


def representability_guards(system, state, rejected, coordinate, direction, residual, scales, limit):
    """Explain the production predicate without changing any input or state."""
    import numpy as np
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_dynamics import ControlledPhysicalInterfaceIonSystem

    storage, poisson, local = scales
    checks, values = {}, {}

    def check(name, condition):
        checks[name] = bool(condition)
        return bool(condition)

    def finish(candidate=None):
        return {"checks": checks, "values": values,
                "first_rejected_guard": next((name for name, passed in checks.items() if not passed), None),
                "proposal_returned": candidate is not None}, candidate

    check("exact_controlled_system_type", type(system) is ControlledPhysicalInterfaceIonSystem)
    check("step_reference_present", system._step_reference is not None)
    check("single_interface", system.interface_count == 1)
    if not all(checks.values()):
        return finish()
    residual, coordinate, direction = (np.asarray(x, dtype=float) for x in (residual, coordinate, direction))
    stop = len(storage) + len(poisson)
    check("residual_shape", residual.shape == (stop + len(local),))
    check("six_local_scales", len(local) == 6)
    check("coordinate_shape", coordinate.shape == (system.dimension,))
    check("direction_shape", direction.shape == coordinate.shape)
    check("finite_residual_coordinate_direction", all(np.all(np.isfinite(x)) for x in (residual, coordinate, direction)))
    check("finite_positive_residual_limit", np.isfinite(limit) and limit > 0.)
    if not all(checks.values()):
        return finish()
    row = int(np.argmax(np.abs(residual)))
    component = row - stop - 2
    values.update(dominant_row=row, component=component, residual=float(residual[row]), limit=float(limit))
    check("dominant_local_carrier_row", component in range(4))
    check("dominant_strictly_exceeds_limit", abs(residual[row]) > limit)
    check("only_one_exceeding_row", not np.any(np.abs(np.delete(residual, row)) > limit))
    if component not in range(4):
        return finish()
    before, after = state.local[0].carrier_data.inputs, rejected.local[0].carrier_data.inputs
    support = {}
    for name in ("state_density", "trace_potential", "bulk_density", "bulk_potential", "occupancy"):
        first, second = getattr(before, name), getattr(after, name)
        check("same_" + name + "_hi_lo", np.array_equal(first.hi, second.hi) and np.array_equal(first.lo, second.lo))
        support[name] = {word: array_comparison(getattr(first, word), getattr(second, word)) for word in ("hi", "lo")}
    values["support_word_comparisons"] = support
    column = system._local_block_slice(0).start + 2 + component
    density, change = float(state.local[0].state_m3[component]), float(direction[column])
    values.update(column=column, density=density, direction=change, coordinate=float(coordinate[column]))
    check("finite_positive_density", np.isfinite(density) and density > 0.)
    check("nonzero_direction_including_signed_zero", change != 0.)
    if not checks["finite_positive_density"] or not checks["nonzero_direction_including_signed_zero"]:
        return finish()
    target = np.nextafter(density, np.inf if change > 0. else -np.inf)
    check("finite_positive_neighbor", np.isfinite(target) and target > 0.)
    if not checks["finite_positive_neighbor"]:
        return finish()
    gap = abs(target - density)
    with np.errstate(over="ignore", invalid="ignore"):
        density_change = density * np.expm1(change)
    values.update(target_density=float(target), directional_gap=float(gap),
                  upward_gap=float(np.nextafter(density, np.inf) - density),
                  downward_gap=float(density - np.nextafter(density, -np.inf)),
                  density_expm1_change=float(density_change),
                  magnitude_in_directional_gaps=float(abs(density_change) / gap))
    check("finite_density_expm1_change", np.isfinite(density_change))
    check("density_change_half_to_two_directional_gaps", .5 * gap <= abs(density_change) <= 2. * gap)
    reference = float(system._step_reference.local[0].state_m3[component])
    values["reference_density"] = reference
    check("finite_positive_reference_density", np.isfinite(reference) and reference > 0.)
    if not checks["finite_positive_reference_density"]:
        return finish()
    relative = (target - reference) / reference
    values["relative_change_to_reference"] = float(relative)
    check("finite_relative_reference_change", np.isfinite(relative))
    check("relative_reference_change_at_most_half", abs(relative) <= .5)
    if not checks["finite_relative_reference_change"] or relative <= -1.:
        return finish()
    mapped = np.log1p(relative)
    delta = mapped - coordinate[column]
    values.update(neighbor_coordinate=float(mapped), neighbor_coordinate_change=float(delta))
    check("finite_log1p_coordinate", np.isfinite(mapped))
    check("log1p_change_agrees_with_direction", delta * change > 0.)
    check("log1p_change_at_most_twice_direction", abs(delta) <= 2. * abs(change))
    proposed = coordinate.copy()
    proposed[column] = mapped
    expected = np.asarray([state.local[0].state_m3]).copy()
    expected[0, component] = target
    forward = system._trace_density_coordinates(proposed)
    check("forward_map_exact_neighbor_only", np.array_equal(forward, expected))
    values["forward_trace_density"] = forward.tolist()
    actual = system.representable_line_search_candidate(
        state, rejected, coordinate, direction, residual, storage, poisson, local, limit,
    )
    if (actual is not None) != all(checks.values()):
        raise AssertionError("diagnostic explanation disagrees with the original candidate guard")
    if actual is not None and not np.array_equal(actual, proposed):
        raise AssertionError("diagnostic coordinate differs from original proposal")
    return finish(actual)


def original_acceptance_metrics(system, state, previous, dt, residual, settings):
    import numpy as np

    current = system.solver_current_metrics(state, previous, dt)
    metrics = {
        "nonlinear_residual": float(np.max(np.abs(residual))),
        "charge_balance_relative": float(system.charge_balance_metrics(state, previous, dt)[1]),
        "all_face_current_relative": float(current[4]),
        "interface_current_relative": float(current[5]),
    }
    limits = {
        "nonlinear_residual": settings["maximum_scaled_nonlinear_residual"],
        "charge_balance_relative": settings["maximum_charge_balance_relative_error"],
        "all_face_current_relative": settings["maximum_all_face_current_spread_relative"],
        "interface_current_relative": settings["maximum_two_sided_interface_total_current_relative_error"],
    }
    passed = {name: bool(np.isfinite(value) and value <= limits[name]) for name, value in metrics.items()}
    closure = max(metrics[name] / limits[name] for name in metrics if name != "nonlinear_residual")
    return {"metrics": metrics, "limits": limits, "passed": passed,
            "all_original_solver_gates_passed": all(passed.values()), "closure_ratio": float(closure)}


def fixed_iterate_line_search(point, policy, expected_slots):
    """Use the original slot budget for fixed-iterate candidates, never advance.

    After the first candidate the live algorithm would accept, remaining
    damping samples are explicitly counterfactual point evaluations. They
    never update the reference/trial or become a historical Newton trace.
    """
    import numpy as np
    from scipy.sparse.linalg import spsolve
    from perovskite_sim.experiments.interface_defect_transient import (
        InterfaceDefectTransientError, effective_newton_acceptance_settings,
    )

    system, previous, coordinate, voltage, dt, scales, residual, jacobian, state, _ = point
    settings = effective_newton_acceptance_settings(policy)
    budget = policy.maximum_line_search_steps
    if budget != expected_slots:
        raise ValueError("candidate budget differs from the original request")
    current = original_acceptance_metrics(system, state, previous, dt, residual, settings)
    report = {
        "scope": "current-platform reconstruction at one saved coordinate; not recorded historical Newton path",
        "budget": budget, "starting_acceptance": current, "slots": [],
        "native_solver_calls": 0, "trajectory_steps_advanced": 0,
        "inputs": local_input_words(state),
    }
    if current["all_original_solver_gates_passed"]:
        report["disposition"] = "current_platform_saved_coordinate_already_meets_solver_gates"
        report["evaluations"] = 0
        report["slots"] = [{"slot": i, "evaluated": False, "reason": "initial_convergence_gate"} for i in range(budget)]
        return report
    limit = settings["maximum_scaled_nonlinear_residual"]
    norm = current["metrics"]["nonlinear_residual"]
    eligible = norm <= limit or system.newton_direction_eligible(residual, *scales, limit)
    rhs = residual.copy()
    target = system.newton_residual_target(previous, *scales)
    target_valid = (target.shape == residual.shape and np.all(np.isfinite(target))
                    and float(np.max(np.abs(target))) <= limit)
    if eligible and target_valid:
        rhs = system.newton_direction_rhs(state, previous, residual, target, *scales)
    direction = np.asarray(spsolve(jacobian, -rhs), dtype=float)
    if direction.shape != coordinate.shape or not np.all(np.isfinite(direction)):
        raise ValueError("reconstructed Newton direction is not finite")
    linear_error = np.max(np.abs(jacobian @ direction + rhs)) / max(
        float(np.max(np.abs(rhs))), float(np.max(abs(jacobian) @ np.abs(direction))), np.finfo(float).tiny,
    )
    report.update(direction=direction.tolist(), direction_rhs=rhs.tolist(),
                  stable_direction_eligible=bool(eligible), previous_target_valid=bool(target_valid),
                  linear_backward_error=float(linear_error), jacobian_nnz=int(jacobian.nnz))
    damping, neighbor, neighbor_used = 1., None, False
    native_stop_slot = None
    for index in range(budget):
        representable = neighbor is not None
        candidate = neighbor if representable else coordinate + damping * direction
        neighbor = None
        entry = {"slot": index, "evaluated": True, "kind": "neighbor" if representable else "damped",
                 "counterfactual_after_native_stop": native_stop_slot is not None,
                 "damping": damping, "coordinate_sha256": hashlib.sha256(candidate.tobytes()).hexdigest(),
                 "coordinate_changed_columns": np.flatnonzero(candidate != coordinate).tolist()}
        try:
            candidate_residual, _, candidate_state = system.residual_and_jacobian(
                candidate, voltage, previous, dt, *scales,
            )
            candidate_metrics = original_acceptance_metrics(
                system, candidate_state, previous, dt, candidate_residual, settings,
            )
        except (InterfaceDefectTransientError, ValueError, FloatingPointError) as error:
            entry["evaluation_error"] = str(error)
            report["slots"].append(entry)
            if not representable:
                damping *= .5
            continue
        next_norm = candidate_metrics["metrics"]["nonlinear_residual"]
        fraction = 1. if representable else damping
        descent = next_norm < norm * (1. - 1.e-4 * fraction)
        closure_improved = (
            norm <= policy.maximum_scaled_nonlinear_residual
            and next_norm <= policy.maximum_scaled_nonlinear_residual
            and (candidate_metrics["closure_ratio"] <= 1.
                 or candidate_metrics["closure_ratio"] < current["closure_ratio"] * (1. - 1.e-4 * damping))
        )
        accepted = descent or closure_improved
        if representable:
            accepted = descent and next_norm <= limit and candidate_metrics["closure_ratio"] <= 1.
        entry.update(acceptance=candidate_metrics, residual_descent=bool(descent),
                     closure_improved=bool(closure_improved), line_search_accepted=bool(accepted),
                     target_row_895=float(candidate_residual[895]),
                     residual_sha256=hashlib.sha256(candidate_residual.tobytes()).hexdigest(),
                     candidate_trace_density=[local.state_m3.tolist() for local in candidate_state.local])
        if accepted and native_stop_slot is None:
            native_stop_slot = index
        caller_eligible = (native_stop_slot is None and not accepted
                           and not representable and not neighbor_used and index + 1 < budget
                           and current["closure_ratio"] <= 1.)
        entry["representable_caller_eligible"] = bool(caller_eligible)
        if not representable:
            guards, proposed = representability_guards(
                system, state, candidate_state, coordinate, direction, residual, scales, limit,
            )
            entry["representability"] = guards
            entry["guard_was_only_diagnostically_evaluated"] = not caller_eligible
            if caller_eligible and proposed is not None:
                neighbor, neighbor_used = proposed, True
        report["slots"].append(entry)
        if not representable:
            damping *= .5
    report["native_stop_slot"] = native_stop_slot
    report["disposition"] = (
        "first_line_search_acceptance_recorded_remaining_slots_are_counterfactual_fixed_input_probes"
        if native_stop_slot is not None else "original_slot_budget_exhausted_on_current_platform"
    )
    report["evaluations"] = sum(slot["evaluated"] for slot in report["slots"])
    report["near_acceptance_fallback_eligible_by_residual"] = bool(norm <= policy.maximum_scaled_nonlinear_residual)
    report["near_acceptance_fallback_executed"] = False
    return report


def _read_bound_archive_json(archive, fixture, relative):
    path = Path(archive) / relative
    expected = fixture["sources"][relative]
    if path.stat().st_size != expected["bytes"] or file_sha256(path) != expected["sha256"]:
        raise ValueError("historical diagnostic input changed: " + relative)
    return read_json(path)


def _load_decimal_oracle(archive, fixture):
    import importlib.util

    root = Path(archive) / "results/OneDimensionalMechanism/R1PhysicsDevelopmentV52"
    relative = "results/OneDimensionalMechanism/R1PhysicsDevelopmentV52/ManifestV1.json"
    manifest = _read_bound_archive_json(archive, fixture, relative)
    path = root / "Root/LocalCarrierOracleV1.py"
    record = manifest["Root/LocalCarrierOracleV1.py"]
    if file_sha256(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
        raise ValueError("archived Decimal oracle is not bound by its original manifest")
    name = "solarlab_archived_r1_decimal_oracle"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, {"path": str(path), **record}


def same_input_local_oracle(system, state, oracle):
    from decimal import Decimal, localcontext
    import numpy as np

    item = state.local[0].carrier_data
    identity, coefficients = system._r1_local_carrier_coefficients[0]
    if identity != item.coefficient_identity:
        raise ValueError("local state and live cached coefficient identities differ")
    payload = oracle.freeze_inputs(item.inputs, coefficients, system.capture_multiplier)
    result = oracle.evaluate_frozen(payload, precision=80)
    errors = {}
    references = {**result["balance"],
                  "residual_occupancy_derivative_m2_s": result["residual_occupancy_derivative_m2_s"]}
    evaluated = {**item.balance,
                 "residual_occupancy_derivative_m2_s": item.tangent["residual_occupancy_derivative_m2_s"]}
    with localcontext() as context:
        context.prec = 80
        for name, reference in references.items():
            pair = evaluated[name]
            high, low = np.asarray(pair.hi).ravel(), np.asarray(pair.lo).ravel()
            refs = np.asarray(reference, dtype=object).ravel()
            differences = [abs(Decimal.from_float(float(h)) + Decimal.from_float(float(l)) - Decimal(str(r)))
                           for h, l, r in zip(high, low, refs)]
            relative = [d / max(abs(Decimal(str(r))), Decimal(1)) for d, r in zip(differences, refs)]
            errors[name] = {"maximum_absolute": str(max(differences)), "maximum_relative": str(max(relative))}
    return payload, {"coefficient_identity": identity, "input_sha256": result["input_sha256"],
                     "oracle": result, "DD_vs_decimal_errors": errors,
                     "local_derivative_coordinates_checked": len(result["coordinate_order"]),
                     "scope": "same current inputs and frozen current coefficients; no historical table substitution"}


def diagnose(archive, fixture_path, output):
    import numpy as np
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_independent_physics import independent_physics_row
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_local_carrier import float_local_carrier_inputs
    from perovskite_sim.physics.two_sided_interface_compensated import default_fd_table

    fixture = load_fixture(fixture_path)
    launch = _read_bound_archive_json(
        archive, fixture,
        "results/OneDimensionalMechanism/R1PhysicsDevelopmentV53/LongWindowV1/BaselineV1/Root/LaunchContractV1.json",
    )
    source_mismatches = [relative for relative, expected in launch["source_files"].items()
                         if file_sha256(PROJECT / relative) != expected]
    if source_mismatches:
        raise ValueError("current bound computational source changed: " + ", ".join(source_mismatches))
    old_relative = "results/OneDimensionalMechanism/R1PhysicsDevelopmentV52/DiagnosticsV1/RepresentabilityV1.json"
    historical = _read_bound_archive_json(archive, fixture, old_relative)
    oracle, oracle_source = _load_decimal_oracle(archive, fixture)
    table, table_hash = default_fd_table()
    report = {
        "schema": "SolarLabR1SavedFailureDiagnosisV1",
        "scope": "P01-01/P01-02 bounded saved-input diagnosis only",
        "fixture": {"path": str(fixture_path), "sha256": file_sha256(fixture_path)},
        "tool_sha256": file_sha256(Path(__file__)),
        "native_step_calls": 0, "prefix_integration_calls": 0, "new_DC_optimizations": 0,
        "historical_Newton_path_claimed": False, "full_100s_or_R1_qualification": False,
        "original_thresholds": fixture["request"]["certificate_limits"],
        "original_channel_identities": fixture["operator_channels"],
        "cases": {}, "local_oracle_source": oracle_source,
        "current_required_source_files_match_original": len(launch["source_files"]),
        "diagnosis_completed": False, "passed": False,
        "fermi_table": {
            "current_sha256": table_hash, "original_V52_sha256": historical["same_input_decimal_oracle"]["fd_table_sha256"],
            "identical": table_hash == historical["same_input_decimal_oracle"]["fd_table_sha256"],
            "nodes": len(table["eta"]),
            "source": "physics/fermi_dirac.py::_half_table generates the table using current NumPy Gaussian quadrature",
            "original_table_numeric_payload_available_in_fixture": False,
        },
        "original_environment_route": fixture["original_environment"],
    }
    write_json(output, report)
    oracle_inputs = {"schema": "SolarLabR1CurrentConstitutiveInputsV1", "coefficients": {}, "cases": {}}
    with forbid_trajectory_work():
        base, policy, verification = saved_input_system(fixture)
        report["preparation_verification"] = verification
        for name in fixture["cases"]:
            point = restored_point(fixture, name, base, policy)
            working, previous, coordinate, voltage, dt, scales, residual, jacobian, attempted, comparisons = point
            item = {
                "time_s": fixture["cases"][name].get("saved_witness", fixture["cases"][name].get("saved_row"))["time_s"],
                "dt_s": dt, "dimension": int(working.dimension),
                "comparison": comparisons, "current_scaled_residual_vector": residual.tolist(),
                "current_jacobian_nnz": int(jacobian.nnz),
                "signed_row_895": float(residual[895]),
                "current_maximum_residual": float(np.max(np.abs(residual))),
                "physical_primary_input_words": local_input_words(attempted),
            }
            case = fixture["cases"][name]
            snapshot = case["saved_witness"]["attempted_state"] if "saved_witness" in case else case["saved_row"]["state"]
            saved_inputs = float_local_carrier_inputs(
                working, 0, np.asarray(snapshot["n_m3"]), np.asarray(snapshot["p_m3"]), np.asarray(snapshot["phi_V"]),
                np.asarray(snapshot["occupancy"]), np.asarray(snapshot["trace_potential_V"]), None,
                trace_density_m3=np.asarray(snapshot["trace_state_m3"]),
            )
            actual_inputs = attempted.local[0].carrier_data.inputs
            item["local_primary_words_equal_to_saved_attempt"] = {
                key: {word: array_comparison(getattr(getattr(actual_inputs, key), word), getattr(getattr(saved_inputs, key), word))
                      for word in ("hi", "lo")}
                for key in ("state_density", "trace_potential", "bulk_density", "bulk_potential", "occupancy")
            }
            item["independent_physics"] = independent_physics_row(working, attempted, previous, dt)
            operators = base.eliminated_operator_diagnostics(attempted, voltage)
            if sorted(operators) != fixture["operator_channels"]:
                raise ValueError("recomputed operator channel set differs from the original thirteen")
            item["operator_channels"] = {
                channel: {"relative_error": float(values["relative_error"]),
                          "limit": fixture["request"]["certificate_limits"]["eliminated_operator_error"],
                          "passed": bool(values["relative_error"] <= fixture["request"]["certificate_limits"]["eliminated_operator_error"])}
                for channel, values in operators.items()
            }
            if "failed" in name:
                payload, oracle_report = same_input_local_oracle(working, attempted, oracle)
                key = oracle_report["coefficient_identity"]
                oracle_inputs["coefficients"][key] = payload["coefficients"]
                oracle_inputs["cases"][name] = {**{k: v for k, v in payload.items() if k != "coefficients"},
                                               "coefficient_identity": key,
                                               "full_payload_sha256": content_sha256(payload)}
                item["local_same_input_decimal"] = oracle_report
            item["line_search"] = fixed_iterate_line_search(point, policy, fixture["request"]["solver_limits"]["line_search"])
            if name == "step183_failed":
                # The archived diagnostic provides this actual vector. It is
                # an input-only guard check, not the current platform's Newton
                # direction or a replay of unrecorded rejected states.
                guards, candidate = representability_guards(
                    working, attempted, attempted, coordinate,
                    np.asarray(historical["original_direction"]),
                    np.asarray(case["saved_witness"]["scaled_residual_vector"]), scales,
                    fixture["request"]["certificate_limits"]["nonlinear_residual"],
                )
                item["historical_183_guard_input_check"] = {
                    "source": old_relative, "rejected_state_assumption": "same-support plateau predicate input",
                    "guards": guards,
                    "matches_recorded_candidate": candidate is not None and np.array_equal(
                        candidate, fixture["historical_v52_candidate_evidence"][0]["coordinate"]),
                    "not_a_historical_Newton_path_replay": True,
                }
            report["cases"][name] = item
            write_json(output, report)
    inputs_path = Path(output).parent / "CurrentConstitutiveInputsV1.json"
    write_json(inputs_path, oracle_inputs)
    report["current_constitutive_inputs"] = {"path": str(inputs_path), "sha256": file_sha256(inputs_path)}
    report["diagnosis_completed"] = True
    report["passed"] = True
    report["passed_means"] = "bounded diagnosis completed with limitations; not historical reproduction or solver qualification"
    report["disposition"] = {
        "historical_step309_root_cause": "not established on the changed runtime/table",
        "local_fix_supported_for_native_trial": False,
        "old_path_status": "historical failure preserved",
        "next_branch": "P02 numerical contracts; original-runtime saved-state analysis if needed, separately dispatched",
        "P01_03_native_trial_started": False,
    }
    return report


def _selected_row(row):
    physics = row["physics_reconstruction"]
    keep = (
        "coordinate", "electron_qf_increment_V", "hole_qf_increment_V",
        "storage", "rate", "direct_poisson_residual_C_m2",
        "scaled_residual_vector", "storage_scale", "poisson_scale_C_m2",
        "local_algebraic_scale", "scaled_nonlinear_residual",
        "eliminated_operator", "independent_physics",
    )
    return {
        "phase": row["phase"], "time_s": row["time_s"], "dt_s": row["dt_s"],
        "substeps": row["substeps"], "solver_accepted": row["solver_accepted"],
        "state": row["state"],
        "physics_reconstruction": {key: physics[key] for key in keep if key in physics},
    }


def freeze(archive, fixture_path, output):
    """Read original manifests/codec and copy only the needed frozen inputs."""
    import numpy as np
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_pair_codec import (
        verify_numeric_sidecar,
    )
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import (
        R1PreparedState, physical_preparation_identity,
    )
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_binding import (
        validate_r1_study_binding,
    )
    from perovskite_sim.models.config_loader import load_device_from_yaml

    archive = Path(archive).resolve()
    history = archive / "results/OneDimensionalMechanism"
    v52, v53 = history / "R1PhysicsDevelopmentV52", history / "R1PhysicsDevelopmentV53"
    case_root = v53 / "LongWindowV1/BaselineV1"
    case = case_root / "OutputV1/CaseV1"
    sources = {}

    def bound(path, manifest=None, relative=None):
        record = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
        if manifest is not None and manifest[relative] != record:
            raise ValueError(f"original manifest binding mismatch: {path}")
        sources[path.relative_to(archive).as_posix()] = record
        return record

    case_manifest = read_json(case / "ManifestV1.json")
    v52_manifest = read_json(v52 / "ManifestV1.json")
    for name in (
        "RequestV1.json", "PreparedV1.json", "ReferenceBindingV1.json", "SourceFixtureV1.yaml",
        "SourceReceiptV1.json", "StateArraysV1.npz", "FailureV1.json", "FailureWitnessV1.json",
        "ResultV1.json", "EnvironmentV1.json", "SummaryV1.json",
    ):
        bound(case / name, case_manifest, name)
    for name in (
        "Inputs/SavedStepInputV1.json", "Inputs/PreparedV1.json", "Inputs/ReferenceBindingV1.json",
        "Inputs/SourceFixtureV1.yaml", "OutputV1/ConvergedCheckpointV1.json",
        "OutputV1/ValidationResultV1.json", "OutputV1/RepresentableCandidatesV1.json",
        "DiagnosticsV1/RepresentabilityV1.json", "Root/LaunchContractV1.json",
    ):
        bound(v52 / name, v52_manifest, name)
    for path in (
        case / "ManifestV1.json", v52 / "ManifestV1.json", v53 / "ManifestV1.json",
        v53 / "Root/RecoveryScopeProposalV1.json", case_root / "Root/RequestV1.json",
        case_root / "Root/LaunchContractV1.json", v53 / "Root/SourceBindingV1.json",
    ):
        bound(path)

    request = read_json(case / "RequestV1.json")
    if sources[(case / "RequestV1.json").relative_to(archive).as_posix()] != sources[
        (case_root / "Root/RequestV1.json").relative_to(archive).as_posix()
    ]:
        raise ValueError("original launch and saved request differ")
    source = read_json(case / "SourceReceiptV1.json")
    launch = read_json(case_root / "Root/LaunchContractV1.json")
    if source["source_commit"] != launch["source_commit"] or source["source_commit"] != "12869a4f90d6d08af2d24af902838d8a4d5c00c4":
        raise ValueError("original launch/source receipt commit mismatch")
    if source["r1_source_content_sha256"] != launch["source_content_sha256"]:
        raise ValueError("original launch/source receipt content mismatch")
    if source["all_tracked_files"] != launch["all_tracked_files"]:
        raise ValueError("original tracked source manifests differ")
    checked_source = {}
    for relative, expected in launch["source_files"].items():
        # LaunchContract.source_files is relative to its declared project;
        # all_tracked_files is relative to the containing Git repository.
        path = PROJECT / relative
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"current computational source differs from original: {relative}")
        checked_source[path.relative_to(REPOSITORY).as_posix()] = actual

    prepared_raw = read_json(case / "PreparedV1.json")
    prepared = R1PreparedState.from_dict(prepared_raw)
    reference = read_json(case / "ReferenceBindingV1.json")
    stack = load_device_from_yaml(case / "SourceFixtureV1.yaml")
    validate_r1_study_binding(reference, stack)
    prepared_v52 = read_json(v52 / "Inputs/PreparedV1.json")
    R1PreparedState.from_dict(prepared_v52)
    preparation_identity = physical_preparation_identity(prepared)
    if preparation_identity != physical_preparation_identity(prepared_v52):
        raise ValueError("V52/V53 physical preparations differ")
    if prepared_raw["reference_sha256"] != reference["sha256"]:
        raise ValueError("prepared state reference binding mismatch")
    result = read_json(case / "ResultV1.json")
    numeric = verify_numeric_sidecar(case / "StateArraysV1.npz", result)
    old = read_json(v52 / "Inputs/SavedStepInputV1.json")
    rows = result["accepted_steps"]
    checkpoint = read_json(v52 / "OutputV1/ConvergedCheckpointV1.json")
    validation = read_json(v52 / "OutputV1/ValidationResultV1.json")
    failed = result["failure"]["numerical_evidence"]
    old_failed = old["failure"]["numerical_evidence"]
    if len(rows) != 309 or failed["previous_state"] != rows[-1]["state"]:
        raise ValueError("failed step does not follow the exact saved prefix")
    if rows[183]["state"] != checkpoint["state"] or not rows[183]["solver_accepted"]:
        raise ValueError("V53 step183 does not match the validated V52 checkpoint")
    if rows[183]["physics_reconstruction"]["scaled_nonlinear_residual"] != validation["scaled_nonlinear_residual"]:
        raise ValueError("V52/V53 successful step183 residual differs")
    if old_failed["previous_state"] != rows[182]["state"]:
        raise ValueError("V52 failed step183 predecessor differs from the V53 prefix")
    channels = sorted(rows[183]["physics_reconstruction"]["eliminated_operator"])
    if any(sorted(row["physics_reconstruction"]["eliminated_operator"]) != channels for row in rows):
        raise ValueError("historical accepted-row channel coverage differs")
    fixture = {
        "schema": SCHEMA, "sources": sources,
        "source_commit": source["source_commit"],
        "source_content_sha256": source["r1_source_content_sha256"],
        "prepared": prepared_raw, "prepared_v52": prepared_v52, "reference": reference,
        "physical_preparation_identity": preparation_identity,
        "source_fixture_yaml": (case / "SourceFixtureV1.yaml").read_text(),
        "request": request, "policy": result["policy"],
        "control_label": result["control_label"],
        "initial_context": result["physics_reconstruction"],
        "initial_event": result["initial_event"],
        "initial_context_v52": old["physics_reconstruction"],
        "initial_event_v52": old["initial_event"], "policy_v52": old["policy"],
        "operator_channels": channels,
        "original_environment": read_json(case / "EnvironmentV1.json"),
        "prefix": {
            "accepted_rows": len(rows), "finite_accepted_steps": len(rows) - 1,
            "expected_rows": request["expected_rows"], "substeps_reached": [4],
            "last_accepted_time_s": rows[-1]["time_s"],
            "bilateral_100s_qualified": False, "compensated_admitted": False,
            "complete_R1_2_qualification": False,
        },
        "cases": {
            "step183_failed": {
                "saved_witness": old_failed, "previous_row": _selected_row(rows[182]),
                "original_failure": {key: value for key, value in old["failure"].items() if key != "numerical_evidence"},
            },
            "step183_accepted": {
                "saved_row": _selected_row(rows[183]), "previous_row": _selected_row(rows[182]),
                "historical_validation": validation,
            },
            "step309_failed": {
                "saved_witness": failed, "previous_row": _selected_row(rows[308]),
                "original_failure": {key: value for key, value in result["failure"].items() if key != "numerical_evidence"},
            },
        },
        "historical_v52_candidate_evidence": read_json(v52 / "OutputV1/RepresentableCandidatesV1.json"),
        "representation_scope": {
            "primary_words": "original binary64 baseline arrays",
            "persistent_primary_low_words": "not part of this historical baseline representation",
            "local_carrier_DD_low_words": "recomputed intermediates from exact saved binary64 inputs; not invented persistent state",
            "newton_candidate_history_309": "not recorded; future candidate evaluations must be labelled reconstructed",
        },
    }
    fixture["sha256"] = content_sha256(fixture)
    write_json(fixture_path, fixture)
    load_fixture(fixture_path)
    report = {
        "schema": "SolarLabR1FrozenWitnessReportV1", "passed": True,
        "scope": "P00-04 source/input integrity; no state equation evaluation or trajectory",
        "sources": sources, "current_source_files_checked": checked_source,
        "numeric_sidecar": numeric, "prepared_sha256": prepared.sha256,
        "reference_sha256": reference["sha256"],
        "preparation_identities": {
            "v52_artifact": prepared_v52["sha256"], "v53_artifact": prepared.sha256,
            "shared_physical_identity": preparation_identity,
            "distinct_provenance_fields": [key for key in prepared_raw if prepared_raw[key] != prepared_v52[key]],
            "execution_or_approval_transfer": False,
        },
        "certificate_limits": request["certificate_limits"],
        "operator_channels": channels, "prefix": fixture["prefix"],
        "step183_checkpoint_equal": True,
        "residual_exceedances": {
            key: [{"row": index, "value": value} for index, value in enumerate(case_data["saved_witness"]["scaled_residual_vector"])
                  if abs(value) > request["certificate_limits"]["nonlinear_residual"]]
            for key, case_data in fixture["cases"].items() if "saved_witness" in case_data
        },
        "fixture": {"path": str(fixture_path), "sha256": file_sha256(fixture_path),
                    "bytes": Path(fixture_path).stat().st_size},
        "original_artifacts_modified": False, "native_step_calls": 0,
    }
    write_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("freeze", "diagnose"))
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(os.environ.get(name) != "1" for name in THREAD_VARIABLES):
        raise SystemExit("all five documented BLAS/thread environment variables must be 1")
    sys.path.insert(0, str(PROJECT))
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        operation = freeze if args.operation == "freeze" else diagnose
        report = operation(args.archive, args.fixture, args.output)
    except Exception as error:
        import traceback
        report = read_json(args.output) if args.output.is_file() and args.operation == "diagnose" else {}
        report.update(passed=False, diagnosis_completed=False,
                      exception_type=type(error).__name__, exception=str(error),
                      traceback=traceback.format_exc())
    report["command"] = [sys.executable, *sys.argv]
    report["started_at"] = started_at
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["elapsed_s"] = time.monotonic() - started
    report["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    report["runtime"] = current_runtime()
    write_json(args.output, report)
    print(json.dumps({key: report.get(key) for key in (
        "passed", "elapsed_s", "peak_rss_bytes", "fixture", "exception", "disposition",
    )}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
