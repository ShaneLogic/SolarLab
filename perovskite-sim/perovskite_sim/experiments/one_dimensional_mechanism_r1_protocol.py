"""R1-1 common-state, controlled ideal-step execution and evidence gates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from decimal import Decimal
from numbers import Integral, Number, Rational
import numpy as np

from perovskite_sim.constants import Q
from perovskite_sim.experiments.interface_defect_ion_transient import (
    InterfaceDefectIonTransientPolicy, _refinement_changes,
)
from perovskite_sim.experiments.interface_defect_transient import _integrate_trace
from perovskite_sim.experiments.one_dimensional_mechanism_r1 import physical_step_record
from perovskite_sim.experiments.one_dimensional_mechanism_r1_dynamics import R1DynamicsControls
from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import (
    R1PreparedState, digest, equilibrium_checks, execution_source, json_data,
    restore_common_state, snapshot,
)
from perovskite_sim.experiments.one_dimensional_mechanism_r1_step import (
    build_initial_step, regular_current_at_state,
)


SCOPE = "research_r1_1_common_state_controlled_ideal_step"
VERSION = "r1-1-controls-and-initial-charge-v4"
DEFAULT_TIMES_S = (0.0, 1e-9, 1e-8, 1e-6, 1e-4)
_TRAP_STORAGE_FAILURE_REASON = "trap_storage_balance_exceeds_budget"
_PERSISTENCE_FAILURE_REASON = "accepted_step_persistence_failed"
_NONFINITE_EVIDENCE_REASON = "nonfinite_numeric_evidence"
_PHYSICAL_LIMITS = (
    ("gauss_normalized", 1e-10), ("charge_balance_normalized", 1e-10),
    ("inventory_relative_drift", 1e-10),
    ("contact_internal_current_spread_relative", 2e-6),
)
_BUDGET_BRANCHES = ("newton", "local_charge", "equal", "invalid")
_TOLERANCE_FIELDS = (
    "storage_relative_tolerance", "carrier_storage_atol_m3", "interface_storage_atol_m2",
    "ion_storage_atol_m3", "poisson_relative_tolerance", "poisson_atol_C_m2",
    "interface_algebraic_relative_tolerance", "interface_potential_atol_V",
    "interface_gauss_atol_C_m2", "interface_flux_atol_m2_s",
)


class R1RunError(RuntimeError):
    def __init__(self, message, result):
        self.result = result
        super().__init__(message)


def r1_policy(nonlinear_factor=0.1):
    factor = float(nonlinear_factor)
    if factor not in (1.0, 0.1, 0.01, 0.001):
        raise ValueError("R1 nonlinear factor must be 1, .1, .01 or .001")
    policy = InterfaceDefectIonTransientPolicy(
        maximum_newton_iterations=100, maximum_line_search_steps=40,
        maximum_near_acceptance_nonmonotone_steps=2,
        maximum_ion_inventory_relative_drift=1e-10,
    )
    return replace(policy, **{k: getattr(policy, k)*factor for k in _TOLERANCE_FIELDS})


def _prepared(value):
    return value if isinstance(value, R1PreparedState) else R1PreparedState.from_dict(value)


def nonfinite_numeric_paths(value, *, prefix=""):
    """Locate numeric NaN/Inf leaves without changing the original evidence.

    Complex components and array indices are named separately.  ``None`` and
    descriptive strings (including N/A labels) are not numerical observations.
    """
    paths = []

    def visit(item, path):
        if is_dataclass(item) and not isinstance(item, type):
            for field in fields(item):
                visit(getattr(item, field.name), f"{path}.{field.name}" if path else field.name)
        elif isinstance(item, Mapping):
            for key, child in item.items():
                name = str(key)
                child_path = (f"{path}.{name}" if path else name) if name.isidentifier() else f"{path}[{key!r}]"
                visit(child, child_path)
        elif isinstance(item, np.ndarray):
            for index in np.ndindex(item.shape):
                visit(item[index], path + "".join(f"[{part}]" for part in index))
        elif isinstance(item, (tuple, list)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
        elif isinstance(item, (complex, np.complexfloating)):
            for name in ("real", "imag"):
                if not np.isfinite(getattr(item, name)):
                    paths.append(f"{path}.{name}" if path else name)
        elif isinstance(item, (Number, Decimal)) and not isinstance(item, (Integral, Rational)):
            finite = item.is_finite() if isinstance(item, Decimal) else bool(np.isfinite(item))
            if not finite:
                paths.append(path or "$")

    visit(value, prefix)
    return paths


def _require_finite_result(record):
    """Reject a completed numerical result before strict digest serialization.

    Failure records intentionally retain raw values and are never rescanned:
    their diagnostic copies must not become a second numerical incident.
    """
    paths = nonfinite_numeric_paths(record)
    certificate = record["certificate"]
    certificate["finite_numeric_evidence"] = {
        "passed": not paths, "nonfinite_numeric_paths": paths,
        "scope": "all_numeric_leaves_before_digest",
    }
    if not paths:
        return
    reasons = list(certificate["reasons"])
    covered = {
        "certificate.metrics." + key
        for key in certificate.get("metrics", {})
        if (key in reasons or (key == "trap_storage_normalized_error"
                              and _TRAP_STORAGE_FAILURE_REASON in reasons))
    }
    if any(path not in covered for path in paths) and _NONFINITE_EVIDENCE_REASON not in reasons:
        reasons.append(_NONFINITE_EVIDENCE_REASON)
    certificate.update({"certified": False, "reasons": reasons, "nonfinite_numeric_paths": paths})
    record["failure"] = {
        "type": "PhysicalCheckFailure", "message": ", ".join(reasons),
        "reasons": reasons, "nonfinite_numeric_paths": paths,
        "scope": "completed_result",
    }
    raise R1RunError("R1 result failed physical gates: " + ", ".join(reasons), record)


def _trap_storage_check(scaling_system, state, previous, dt, policy, physical):
    """Check a single interface against existing storage and charge budgets.

    ``scaling_system`` is the original integration system, just as in
    ``_solve_step``; using the rebased working system would reset the declared
    storage accuracy reference. The whole-device charge budget is applied
    conservatively to the one supported interface, without cancellation with
    carrier or ion storage residuals.
    """
    if previous is None:
        return {
            "applicable": False, "dt_s": 0.0, "certified": None,
            "reason": "no_positive_duration_time_step",
            "legacy_error_is_placeholder": True,
            "legacy_error_note": "0.0 is a compatibility placeholder; no finite-step error was computed",
        }
    if scaling_system.interface_count != 1:
        raise ValueError("R1 trap-storage charge budget is defined for one interface")
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("R1 trap-storage check requires a finite positive time step")
    scale = scaling_system.storage_scale(previous.storage, previous, dt, policy)
    storage_scale = float(scale[2 * scaling_system.interior_count])
    # Match physical_step_record's original charge-balance normalization.
    contact = np.asarray(physical["contact_conduction_A_m2"], dtype=float)
    charge_scale_before_floor = max(
        abs(float(physical["charge_rate_A_m2"])),
        abs(float(contact[0] - contact[1])),
        float(np.max(np.abs(state.conduction))),
    )
    charge_scale = max(charge_scale_before_floor, 1.0)
    error = float(physical["trap_storage_error_A_m2"])
    charge_error = error * dt
    nonlinear_budget = Q * policy.maximum_scaled_nonlinear_residual * storage_scale
    charge_relative_limit = min(policy.maximum_charge_balance_relative_error, 1e-10)
    conservation_budget = dt * charge_relative_limit * charge_scale
    charge_limit = min(nonlinear_budget, conservation_budget)
    current_limit = charge_limit / dt
    valid = (
        all(np.isfinite(value) for value in (
            error, charge_error, storage_scale, charge_scale,
            nonlinear_budget, conservation_budget, charge_limit, current_limit,
        ))
        and error >= 0.0 and storage_scale > 0.0 and charge_scale > 0.0
        and nonlinear_budget > 0.0 and conservation_budget > 0.0
        and charge_limit > 0.0 and current_limit > 0.0
    )
    ratio = error / current_limit if valid else float("inf")
    newton_ratio = error / (nonlinear_budget / dt) if valid else float("inf")
    local_charge_ratio = error / (conservation_budget / dt) if valid else float("inf")
    if not valid:
        dominant_budget = "invalid"
    elif nonlinear_budget == conservation_budget:
        dominant_budget = "equal"
    else:
        dominant_budget = "newton" if nonlinear_budget < conservation_budget else "local_charge"
    return {
        "applicable": True, "dt_s": dt,
        "error_A_m2": error, "charge_error_C_m2": charge_error,
        "storage_scale_m2": storage_scale,
        "maximum_scaled_nonlinear_residual": policy.maximum_scaled_nonlinear_residual,
        "nonlinear_tolerances": {key: float(getattr(policy, key)) for key in _TOLERANCE_FIELDS},
        "nonlinear_charge_budget_C_m2": nonlinear_budget,
        "newton_consistency_ratio": newton_ratio,
        "newton_consistency_role": "consistency_with_existing_nonlinear_storage_tolerance",
        "charge_balance_scale_A_m2": charge_scale,
        "charge_scale_before_floor_A_m2": charge_scale_before_floor,
        "charge_scale_floor_A_m2": 1.0,
        "charge_scale_floor_active": bool(charge_scale_before_floor < 1.0),
        "charge_balance_relative_limit": charge_relative_limit,
        "conservation_charge_budget_C_m2": conservation_budget,
        "local_charge_ratio": local_charge_ratio,
        "local_charge_role": "single_interface_application_of_existing_whole_device_charge_budget",
        "dominant_budget": dominant_budget,
        "trap_dynamics_active": bool(scaling_system.controls.nu_t),
        "charge_error_limit_C_m2": charge_limit,
        "current_error_limit_A_m2": current_limit,
        "normalized_error": ratio, "normalized_limit": 1.0,
        "certified": bool(valid and ratio <= 1.0),
        "failure_reason": None if valid and ratio <= 1.0 else _TRAP_STORAGE_FAILURE_REASON,
        "budget_scope": "single_interface_storage_and_whole_device_charge_budget",
    }


def _trap_storage_summary(physical_records):
    checks = [r["trap_storage_check"] for r in physical_records]
    if not checks or any(not check.get("applicable") for check in checks):
        raise ValueError("R1 trap-storage certificate requires finite-step evidence")
    ratios = np.asarray([check["normalized_error"] for check in checks], dtype=float)
    raw_errors = np.asarray([r["trap_storage_error_A_m2"] for r in physical_records], dtype=float)
    finite_ratios = bool(np.all(np.isfinite(ratios)))
    subsets = {
        "all": checks,
        "active_trap": [c for c in checks if c["trap_dynamics_active"]],
        "nontrivial": [c for c in checks if np.isfinite(c["error_A_m2"]) and c["error_A_m2"] > 0.0],
        "active_trap_nontrivial": [c for c in checks if c["trap_dynamics_active"]
                                   and np.isfinite(c["error_A_m2"]) and c["error_A_m2"] > 0.0],
    }
    branch_counts = {}
    branch_maxima = {}
    for name, subset in subsets.items():
        branch_counts[name] = {}
        branch_maxima[name] = {}
        for branch in _BUDGET_BRANCHES:
            selected = [c for c in subset if c["dominant_budget"] == branch]
            branch_counts[name][branch] = len(selected)
            branch_maxima[name][branch] = {
                "maximum_" + metric: (
                    float(np.max([c[metric] for c in selected])) if selected else None
                )
                for metric in ("newton_consistency_ratio", "local_charge_ratio", "normalized_error")
            }
    return {
        "checked_finite_step_count": len(checks),
        "maximum_error_A_m2": float(np.max(raw_errors)),
        "maximum_step_charge_error_C_m2": float(np.max([c["charge_error_C_m2"] for c in checks])),
        "maximum_normalized_error": float(np.max(ratios)) if finite_ratios else float("inf"),
        "normalized_limit": 1.0,
        "minimum_current_error_limit_A_m2": float(min(c["current_error_limit_A_m2"] for c in checks)),
        "maximum_current_error_limit_A_m2": float(max(c["current_error_limit_A_m2"] for c in checks)),
        "certified": bool(finite_ratios and np.all(np.isfinite(raw_errors))
                          and all(c["certified"] for c in checks)),
        "branch_counts": branch_counts,
        "branch_maxima": branch_maxima,
        "nontrivial_definition": "finite strictly positive raw trap-storage error",
        "charge_scale_floor_active_step_count": sum(c["charge_scale_floor_active"] for c in checks),
        "scope": "all_finite_accepted_steps_across_all_nested_levels",
    }


def _physical_step_checks(physical, *, finite_step, policy, evidence=None):
    """Collect every applicable physical violation before observation or failure."""
    checks = {}
    covered_paths = set()

    def scalar_check(name, value, limit, *, source, path):
        nonfinite = nonfinite_numeric_paths(value, prefix=path)
        if nonfinite:
            reason = name + "_nonfinite"
            covered_paths.update(nonfinite)
            value = json_data(value)
        else:
            value = float(value)
            reason = name + "_exceeds_limit" if value > limit else None
        checks[name] = {
            "applicable": True, "value": value, "limit": float(limit),
            "passed": reason is None, "failure_reason": reason, "source": source,
        }

    for key, limit in _PHYSICAL_LIMITS:
        if not finite_step and key in ("charge_balance_normalized", "contact_internal_current_spread_relative"):
            checks[key] = {
                "applicable": False, "value": None, "limit": limit, "passed": None,
                "failure_reason": None, "reason": "no_positive_duration_time_step",
            }
        else:
            scalar_check(key, physical[key], limit,
                         source="finite_step" if finite_step else "initial_algebraic_state",
                         path="physical." + key)
    trap = physical["trap_storage_check"]
    checks["trap_storage"] = {
        "applicable": trap["applicable"], "value": trap.get("normalized_error"),
        "limit": 1.0, "passed": trap["certified"],
        "failure_reason": trap.get("failure_reason"),
        "source": "finite_step" if finite_step else "compatibility_placeholder_not_computed",
    }
    if trap.get("failure_reason"):
        # These fields already participate in the trap gate's finite-value
        # predicate or are invalid ratios produced by that same failed gate.
        covered_paths.add("physical.trap_storage_error_A_m2")
        covered_paths.update("physical.trap_storage_check." + key for key in (
            "error_A_m2", "charge_error_C_m2", "storage_scale_m2", "charge_balance_scale_A_m2",
            "nonlinear_charge_budget_C_m2", "conservation_charge_budget_C_m2",
            "charge_error_limit_C_m2", "current_error_limit_A_m2", "normalized_error",
            "newton_consistency_ratio", "local_charge_ratio",
        ))
    if not finite_step:
        # These are the evaluated derivative right limits, with their original
        # current-at-state policy limits, not the finite-step placeholders.
        for key, limit in (
            ("charge_balance_normalized", policy.maximum_charge_balance_relative_error),
            ("contact_internal_current_spread_relative", policy.maximum_all_face_current_spread_relative),
        ):
            scalar_check("regular_right_limit_" + key, physical["regular_right_limit"][key],
                         limit, source="initial_event.regular_current",
                         path="physical.regular_right_limit." + key)
    paths = nonfinite_numeric_paths({"physical": physical} if evidence is None else evidence)
    uncovered = [path for path in paths if path.replace(
        "initial_event.regular_current.", "physical.regular_right_limit.", 1,
    ) not in covered_paths]
    checks["finite_numeric_evidence"] = {
        "applicable": True, "passed": not paths, "nonfinite_numeric_paths": paths,
        "failure_reason": _NONFINITE_EVIDENCE_REASON if uncovered else None,
        "scope": "all_numeric_leaves_in_row_and_initial_event",
    }
    reasons = [check["failure_reason"] for check in checks.values() if check["failure_reason"]]
    return {"checks": checks, "passed": not reasons and not paths, "reasons": reasons,
            "nonfinite_numeric_paths": paths}


def check_zero_excitation(stack, intervals, binding, prepared, *, controls="ABCD", policy=None,
                          expected_prepared_sha256=None):
    """Check A-D remaining equilibrium equations; do not invent a zero-current relative pass."""
    prepared = _prepared(prepared)
    policy = policy or r1_policy()
    results = {}
    try:
        for label in controls:
            choice = R1DynamicsControls.from_label(label)
            system, state = restore_common_state(prepared, stack, intervals, binding,
                                                 controls=choice, policy=policy,
                                                 expected_prepared_sha256=expected_prepared_sha256)
            checks = equilibrium_checks(system, state, policy)
            initial = build_initial_step(system, state, 0.0, policy=policy)
            results[label] = {
                "controls": json_data(choice), "remaining_equations": checks,
                "initial_event": initial.event,
                "population_identity": snapshot(system, state),
            }
    except Exception as exc:
        partial = {
            "schema": "R1ZeroExcitationV1", "scope": SCOPE, "version": VERSION,
            "prepared_sha256": prepared.sha256, "reference_sha256": binding["sha256"],
            "controls": results,
            "certificate": {"certified": False, "reasons": [str(exc)], "scope": SCOPE},
            "failure": {"type": type(exc).__name__, "message": str(exc),
                        "numerical_evidence": getattr(exc, "result", None)},
        }
        raise R1RunError(f"R1 zero-excitation check failed: {exc}", partial) from exc
    certified = bool(results) and all(r["remaining_equations"]["certified"] for r in results.values())
    record = {
        "schema": "R1ZeroExcitationV1", "scope": SCOPE, "version": VERSION,
        "prepared_sha256": prepared.sha256, "reference_sha256": binding["sha256"],
        "controls": results,
        "certificate": {"certified": certified,
            "reasons": [] if certified else ["zero_excitation_equations_failed"],
            "scope": "remaining_equilibrium_equations_and_no_impulse",
            "relative_dynamic_current_certified": False},
        "source": execution_source(),
    }
    _require_finite_result(record)
    record["sha256"] = digest(record)
    if not certified:
        raise R1RunError("R1 zero-excitation check failed", record)
    return record


def _trace_certificate(system, zero_minus, levels, policy, physical_records):
    final = levels[-1]
    state_change, current_change = _refinement_changes(levels[-2], levels[-1], system)
    all_states = [state for level in levels for state in level.states]
    inventories = np.array([
        [np.dot(s.positive[np.asarray(nodes)], system.widths[np.asarray(nodes)])
         for nodes in system.ion_layout.positive_components]
        for s in all_states
    ])
    inventory_error = float(np.max(np.abs(inventories/system.positive_targets-1.0)))
    decomposition_error = max(
        float(np.max(np.abs(s.conduction-(s.carrier_conduction+s.positive_current))))
        / max(float(np.max(np.abs(s.conduction))), 1.0)
        for s in all_states
    )
    trap_storage = _trap_storage_summary(physical_records)
    metrics = {
        "nonlinear_residual": final.maximum_scaled_residual,
        "local_carrier_residual": final.maximum_local_carrier_residual,
        "local_gauss_residual": final.maximum_local_gauss_residual,
        "analytic_jacobian_error": final.maximum_jacobian_error,
        "charge_balance_relative": final.maximum_charge_balance_error,
        "all_face_current_relative": final.maximum_face_spread,
        "interface_current_relative": final.maximum_interface_current_error,
        "eliminated_operator_error": final.maximum_operator_error,
        "inventory_relative_drift": inventory_error,
        "current_decomposition_error": decomposition_error,
        "refinement_state_change": state_change,
        "refinement_current_change": current_change,
        "full_gauss_normalized": max(r["gauss_normalized"] for r in physical_records),
        "full_charge_balance_normalized": max(r["charge_balance_normalized"] for r in physical_records),
        "physical_current_spread_relative": max(r["contact_internal_current_spread_relative"] for r in physical_records),
        "trap_storage_normalized_error": trap_storage["maximum_normalized_error"],
    }
    limits = {
        "nonlinear_residual": policy.maximum_scaled_nonlinear_residual,
        "local_carrier_residual": policy.maximum_local_carrier_normalized_residual,
        "local_gauss_residual": policy.maximum_local_gauss_normalized_residual,
        "analytic_jacobian_error": policy.maximum_jacobian_column_relative_error,
        "charge_balance_relative": policy.maximum_charge_balance_relative_error,
        "all_face_current_relative": policy.maximum_all_face_current_spread_relative,
        "interface_current_relative": policy.maximum_two_sided_interface_total_current_relative_error,
        "eliminated_operator_error": policy.maximum_eliminated_operator_relative_error,
        "inventory_relative_drift": min(policy.maximum_ion_inventory_relative_drift, 1e-10),
        "current_decomposition_error": policy.maximum_current_decomposition_relative_error,
        "refinement_state_change": policy.maximum_refinement_state_change,
        "refinement_current_change": policy.maximum_refinement_current_relative_change,
        "full_gauss_normalized": 1e-10, "full_charge_balance_normalized": 1e-10,
        "physical_current_spread_relative": 2e-6,
        "trap_storage_normalized_error": 1.0,
    }
    reasons = [
        _TRAP_STORAGE_FAILURE_REASON if k == "trap_storage_normalized_error" else k
        for k in metrics if not np.isfinite(metrics[k]) or metrics[k] > limits[k]
    ]
    if not trap_storage["certified"] and _TRAP_STORAGE_FAILURE_REASON not in reasons:
        reasons.append(_TRAP_STORAGE_FAILURE_REASON)
    if final.maximum_nnz >= system.dimension**2:
        reasons.append("analytic_jacobian_not_sparse")
    frozen = {}
    if not system.controls.nu_I:
        frozen["positive_population"] = all(np.array_equal(s.positive, zero_minus.positive) for s in all_states)
        frozen["ion_flux"] = all(np.count_nonzero(s.positive_flux) == 0 for s in all_states)
    if not system.controls.nu_t:
        frozen["trap_population"] = all(np.array_equal(s.occupancy, zero_minus.occupancy) for s in all_states)
        frozen["trap_charge"] = all(np.array_equal(s.sheet_charge, zero_minus.sheet_charge) for s in all_states)
        frozen["capture_flux"] = all(
            np.count_nonzero(local.tangent.balance.capture_flux_m2_s) == 0
            for s in all_states for local in s.local)
    reasons.extend(k + "_not_frozen" for k, value in frozen.items() if not value)
    return {
        "certified": not reasons, "reasons": reasons, "metrics": metrics, "limits": limits,
        "exact_freeze_checks": frozen, "prepared_D_equilibrium_certified": True,
        "zero_plus_is_equilibrium": False,
        "maximum_absolute_charge_balance_error_A_m2": final.maximum_charge_balance_absolute_error,
        "maximum_absolute_poisson_residual_C_m2": final.maximum_poisson_residual,
        "maximum_trap_storage_error_A_m2": trap_storage["maximum_error_A_m2"],
        "trap_storage": trap_storage,
        "site_occupancy_fraction": max(system._site_fraction(s.positive, s.negative, reject=False) for s in all_states),
        "analytic_jacobian_nnz": final.maximum_nnz, "dense_entries": system.dimension**2,
        "scope": SCOPE,
    }


def run_r1_step(stack, intervals, binding, prepared, *, control="D", amplitude_V=0.005,
                times_s=None, policy=None, accepted_step_observer=None,
                expected_prepared_sha256=None):
    """Integrate from 0+; finite steps contain no ideal charging impulse."""
    prepared = _prepared(prepared)
    choice = control if isinstance(control, R1DynamicsControls) else R1DynamicsControls.from_label(control)
    policy = policy or r1_policy()
    times = np.asarray(DEFAULT_TIMES_S if times_s is None else times_s, dtype=float)
    amplitude = float(amplitude_V)
    if not np.isfinite(amplitude) or not 0 < abs(amplitude) < 0.02:
        raise ValueError("R1 step requires 0 < abs(amplitude_V) < .02; use zero-excitation check for zero")
    if (times.ndim != 1 or times.size < 2 or times[0] != 0 or not np.all(np.isfinite(times))
            or np.any(np.diff(times) <= 0) or times[1] < 1e-12):
        raise ValueError("R1 times must start at 0+ then increase, with first positive time >=1e-12 s")
    if len(policy.refinement_substeps) < 2:
        raise ValueError("R1 requires independent nested time-step comparison")
    accepted = []
    record = {
        "schema": "R1ControlledStepV1", "scope": SCOPE, "version": VERSION,
        "prepared_sha256": prepared.sha256, "reference_sha256": binding["sha256"],
        "controls": json_data(choice), "control_label": choice.label,
        "intervals": int(intervals), "amplitude_V": amplitude, "times_s": times,
        "voltage_V": np.full(times.size, amplitude), "policy": json_data(policy),
        "accepted_steps": accepted,
        "scope_note": "R1-1 controlled short-time execution; no full-window or three-axis convergence claim",
    }
    try:
        system, before = restore_common_state(prepared, stack, intervals, binding,
                                              controls=choice, policy=policy,
                                              expected_prepared_sha256=expected_prepared_sha256)
        record["junction_polarity"] = system.polarity
        record["current_sign_convention"] = "J_x along +x; reported j = junction_polarity * J_x"
        initial = build_initial_step(system, before, amplitude, policy=policy)
        record["initial_event"] = initial.event
        integrated = {}
        physical_records = []

        def observe(working, state, previous, dt, time, substeps, residual):
            physical = physical_step_record(working, state, previous, dt)
            raw_initial_physical = {}
            if previous is None:
                # Replacing/removing finite-step placeholders must not conceal
                # invalid raw numerical evidence on an initial observation.
                raw_initial_physical = {
                    key: physical[key] for key in (
                        "trap_storage_error_A_m2", "contact_maxwell_A_m2", "contact_displacement_A_m2",
                        "internal_maxwell_A_m2", "contact_internal_current_spread_relative",
                    )
                    if key in physical and nonfinite_numeric_paths(physical[key])
                }
            physical["trap_storage_check"] = _trap_storage_check(
                initial.system, state, previous, dt, policy, physical,
            )
            if previous is None:
                physical = dict(physical)
                # Preserve the legacy shape without claiming a measured zero.
                physical["trap_storage_error_A_m2"] = 0.0
                # At 0+ the derivative displacement is nonzero; its truthful
                # observation is the separately evaluated right-limit record.
                for key in ("contact_maxwell_A_m2", "contact_displacement_A_m2", "internal_maxwell_A_m2",
                            "contact_internal_current_spread_relative"):
                    physical.pop(key, None)
                physical["regular_right_limit"] = initial.event["regular_current"]
                physical["initial_algebraic_certificate"] = initial.event["algebraic_certificate"]
                integrated[substeps] = 0.0
            else:
                physical_records.append(physical)
                integrated[substeps] += float(working.polarity * physical["contact_maxwell_A_m2"][0] * dt)
            item = {
                "phase": "0+" if previous is None else "accepted_regular_step",
                "time_s": float(time), "dt_s": float(dt), "substeps": int(substeps),
                "scaled_nonlinear_residual": float(residual), "state": snapshot(working, state),
                "physical": json_data(physical),
                "solver_accepted": previous is not None,
                "regular_integrated_charge_C_m2": integrated[substeps],
            }
            if raw_initial_physical:
                item["raw_initial_physical"] = json_data(raw_initial_physical)
            evidence = dict(item)
            if previous is None:
                evidence["initial_event"] = initial.event
            physical_checks = _physical_step_checks(
                physical, finite_step=previous is not None, policy=policy, evidence=evidence,
            )
            item.update({
                "physical_checks": json_data(physical_checks),
                "physical_checks_passed": physical_checks["passed"],
                "physical_failure_reasons": list(physical_checks["reasons"]),
            })
            accepted.append(item)
            reasons = physical_checks["reasons"]
            if reasons:
                record["certificate"] = {
                    "certified": False, "reasons": list(reasons), "scope": SCOPE,
                    "failed_record_index": len(accepted) - 1,
                    "physical_checks": json_data(physical_checks),
                    "nonfinite_numeric_paths": list(physical_checks["nonfinite_numeric_paths"]),
                }
                if physical_records:
                    trap_storage = _trap_storage_summary(physical_records)
                    record["certificate"].update({
                        "maximum_trap_storage_error_A_m2": trap_storage["maximum_error_A_m2"],
                        "trap_storage": json_data(trap_storage),
                    })
                record["failure"] = {
                    "type": "PhysicalCheckFailure", "message": ", ".join(reasons),
                    "reasons": list(reasons), "record_index": len(accepted) - 1,
                    "nonfinite_numeric_paths": list(physical_checks["nonfinite_numeric_paths"]),
                }
            persistence_error = None
            if accepted_step_observer is not None:
                try:
                    # The observer receives every complete row, including a
                    # solver-accepted row that fails several physical gates.
                    accepted_step_observer(item)
                except Exception as exc:
                    persistence_error = exc
                    persistence_failure = {
                        "reason": _PERSISTENCE_FAILURE_REASON,
                        "type": type(exc).__name__, "message": str(exc),
                        "record_index": len(accepted) - 1, "time_s": float(time),
                        "substeps": int(substeps),
                    }
                    item["persistence_failure"] = persistence_failure
                    record["persistence_failure"] = persistence_failure
            if reasons:
                if persistence_error is not None:
                    record["certificate"]["secondary_reasons"] = [_PERSISTENCE_FAILURE_REASON]
                    record["failure"]["persistence_failure"] = record["persistence_failure"]
                raise R1RunError("R1 row failed physical gates: " + ", ".join(reasons), record) from persistence_error
            if persistence_error is not None:
                record["certificate"] = {
                    "certified": False, "reasons": [_PERSISTENCE_FAILURE_REASON], "scope": SCOPE,
                    "failed_record_index": len(accepted) - 1,
                }
                record["failure"] = record["persistence_failure"]
                raise R1RunError("R1 row failed " + _PERSISTENCE_FAILURE_REASON, record) from persistence_error

        levels = tuple(
            _integrate_trace(initial.system, times, np.full(times.size, amplitude), substeps, policy,
                             accepted_step_observer=observe, initial_state=initial.zero_plus,
                             initial_current_metrics=initial.initial_current_metrics)
            for substeps in policy.refinement_substeps
        )
        final = levels[-1]
        record["certificate"] = _trace_certificate(initial.system, before, levels, policy, physical_records)
        record["output_states"] = {
            name: np.asarray([getattr(s, attribute) for s in final.states])
            for name, attribute in (("n_m3", "n"), ("p_m3", "p"), ("positive_m3", "positive"),
                                    ("occupancy", "occupancy"), ("phi_V", "phi"), ("sheet_charge_C_m2", "sheet_charge"))
        }
        record["regular_currents"] = [
            regular_current_at_state(initial.system, s, policy=policy).evidence for s in final.states
        ]
        record["finite_step_averages"] = {
            "internal_total_A_m2": final.total_current, "internal_displacement_A_m2": final.displacement,
            "interface_total_A_m2": final.interface_total_current,
            "note": "positive-time values are backward-Euler last-substep averages; row0 is the regular right limit",
        }
        record["accepted_state_arrays"] = {
            key: np.asarray([r["state"][key] for r in accepted]) for key in accepted[0]["state"]
        }
        record["charge_integral"] = {
            "impulse_charge_C_m2": initial.event["impulse_charge_C_m2"],
            "regular_by_substeps_C_m2": integrated,
            "complete_by_substeps_C_m2": {k: initial.event["impulse_charge_C_m2"]+v for k, v in integrated.items()},
            "quadrature": "sum of each accepted backward-Euler physical-contact current times dt",
        }
        record["source"] = execution_source()
        if record["source"] != prepared.to_dict()["source"]:
            raise R1RunError("source changed during the controlled experiment", record)
        _require_finite_result(record)
        record["sha256"] = digest(record)
        if not record["certificate"]["certified"]:
            raise R1RunError("R1 controlled-step certificate failed: " + ", ".join(record["certificate"]["reasons"]), record)
        return record
    except R1RunError as exc:
        if isinstance(exc.result, dict):
            certificate = exc.result.get("certificate")
            if not isinstance(certificate, dict) or certificate.get("certified"):
                exc.result["certificate"] = {"certified": False, "reasons": [str(exc)], "scope": SCOPE}
        raise
    except Exception as exc:
        record["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        evidence = getattr(exc, "result", None)
        if evidence is not None and evidence is not record:
            record["failure"]["numerical_evidence"] = evidence
        record["certificate"] = {"certified": False, "reasons": [str(exc)], "scope": SCOPE}
        raise R1RunError(f"R1 controlled step failed: {exc}", record) from exc


__all__ = ["run_r1_step", "check_zero_excitation", "r1_policy", "R1RunError"]
