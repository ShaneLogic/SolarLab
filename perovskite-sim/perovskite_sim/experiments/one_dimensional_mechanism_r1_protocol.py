"""R1-1 common-state, controlled ideal-step execution and evidence gates."""

from __future__ import annotations

from dataclasses import replace
import numpy as np

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
VERSION = "r1-1-controls-and-initial-charge-v1"
DEFAULT_TIMES_S = (0.0, 1e-9, 1e-8, 1e-6, 1e-4)
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


def check_zero_excitation(stack, intervals, binding, prepared, *, controls="ABCD", policy=None):
    """Check A-D remaining equilibrium equations; do not invent a zero-current relative pass."""
    prepared = _prepared(prepared)
    policy = policy or r1_policy()
    results = {}
    try:
        for label in controls:
            choice = R1DynamicsControls.from_label(label)
            system, state = restore_common_state(prepared, stack, intervals, binding,
                                                 controls=choice, policy=policy)
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
    }
    reasons = [k for k in metrics if not np.isfinite(metrics[k]) or metrics[k] > limits[k]]
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
        "site_occupancy_fraction": max(system._site_fraction(s.positive, s.negative, reject=False) for s in all_states),
        "analytic_jacobian_nnz": final.maximum_nnz, "dense_entries": system.dimension**2,
        "scope": SCOPE,
    }


def run_r1_step(stack, intervals, binding, prepared, *, control="D", amplitude_V=0.005,
                times_s=None, policy=None, accepted_step_observer=None):
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
                                              controls=choice, policy=policy)
        record["junction_polarity"] = system.polarity
        record["current_sign_convention"] = "J_x along +x; reported j = junction_polarity * J_x"
        initial = build_initial_step(system, before, amplitude, policy=policy)
        record["initial_event"] = initial.event
        integrated = {}
        physical_records = []

        def observe(working, state, previous, dt, time, substeps, residual):
            physical = physical_step_record(working, state, previous, dt)
            if previous is None:
                physical = dict(physical)
                # At 0+ the derivative displacement is nonzero; its truthful
                # observation is the separately evaluated right-limit record.
                for key in ("contact_maxwell_A_m2", "contact_displacement_A_m2", "internal_maxwell_A_m2",
                            "contact_internal_current_spread_relative"):
                    physical.pop(key, None)
                physical["regular_right_limit"] = initial.event["regular_current"]
                integrated[substeps] = 0.0
            else:
                physical_records.append(physical)
                integrated[substeps] += float(working.polarity * physical["contact_maxwell_A_m2"][0] * dt)
            item = {
                "phase": "0+" if previous is None else "accepted_regular_step",
                "time_s": float(time), "dt_s": float(dt), "substeps": int(substeps),
                "scaled_nonlinear_residual": float(residual), "state": snapshot(working, state),
                "physical": json_data(physical),
                "regular_integrated_charge_C_m2": integrated[substeps],
            }
            accepted.append(item)
            if accepted_step_observer is not None:
                accepted_step_observer(item)
            if previous is not None and any(
                physical[key] > limit or not np.isfinite(physical[key])
                for key, limit in (("gauss_normalized", 1e-10), ("charge_balance_normalized", 1e-10),
                                   ("inventory_relative_drift", 1e-10),
                                   ("contact_internal_current_spread_relative", 2e-6))
            ):
                raise R1RunError("R1 accepted step failed physical contact/charge gates", record)

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
