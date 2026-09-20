"""Re-evaluate saved R1 states against transport, storage and current equations.

Saved local solver coordinates retain changes smaller than the spacing of an
absolute density/potential. They are evaluated about the preceding reconstructed
state. No transient integration or optimization is performed by this verifier.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from perovskite_sim.experiments.interface_defect_transient import _jacobian_error
from perovskite_sim.experiments.one_dimensional_mechanism_r1 import physical_step_record
from perovskite_sim.experiments.one_dimensional_mechanism_r1_dynamics import R1DynamicsControls, CURRENT_METRIC_SEMANTICS
from perovskite_sim.experiments.one_dimensional_mechanism_r1_independent_physics import independent_physics_row
from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import (
    R1PreparedState, _make_system, _preparation_policy, canonical, json_data,
    snapshot, verify_prepared_physics,
)
from perovskite_sim.experiments.one_dimensional_mechanism_r1_step import (
    build_initial_step, regular_current_at_state,
)

SCHEMA = "R1StateEquationReconstructionV1"
ROW_SCHEMA = "R1StateEquationRowV1"
SCOPE = "saved_accepted_states_transport_storage_current_and_declared_discrete_equations"


class R1PhysicsValidationError(ValueError):
    """Saved physical evidence does not match a fresh equation evaluation."""

    def __init__(self, message, report=None):
        self.report = report or {"certified": False, "scope": SCOPE, "reasons": [message]}
        super().__init__(message)


def _same(actual, expected, label):
    # Strict serialization equality is deliberate: this is replay under the
    # anchored numerical implementation, not a physical error tolerance.
    if canonical(actual) != canonical(expected):
        raise R1PhysicsValidationError("physical reconstruction mismatch: " + label)


def reconstruction_context(system, state, prepared_sha256, reference_sha256):
    """Record the starting reference representation, including exact QF values."""
    return json_data({
        "schema": SCHEMA, "scope": SCOPE,
        "current_metric_semantics": CURRENT_METRIC_SEMANTICS,
        "prepared_sha256": prepared_sha256, "reference_sha256": reference_sha256,
        "coordinate_representation": "dimensionless_local_increments_rebased_after_each_accepted_step",
        "dimension": system.dimension,
        "initial_coordinate": state.coordinate,
        "initial_state": snapshot(system, state),
        "initial_electron_qf_increment_V": state.dqfn,
        "initial_hole_qf_increment_V": state.dqfp,
        "electron_qf_reference_V": system.qfn_reference,
        "hole_qf_reference_V": system.qfp_reference,
        "initial_reference_n_m3": system.reference_n,
        "initial_reference_p_m3": system.reference_p,
        "initial_reference_positive_m3": system.reference_positive,
        "initial_reference_occupancy": system.reference_occupancy,
        "fixed_equilibrium_occupancy": system.equilibrium_occupancy,
        "jacobian_scope": "first_finite_step_of_each_nested_level_at_accepted_state",
        "provenance_only": ["Newton_iteration_path", "Newton_iteration_counts",
                            "earlier_iterate_maximum_jacobian_nnz", "execution_environment"],
        "numerical_comparison": "exact_saved_float64_values_under_anchored_source",
        "physical_limits": "unchanged_R1_policy_and_original_component_normalizations",
    })


def capture_r1_physics_row(scaling_system, working, state, previous, voltage, dt,
                           policy, *, check_jacobian=False):
    """Record enough equation detail to audit an accepted state independently."""
    local_carrier, local_gauss = scaling_system.local_normalized_residuals(state)
    details = scaling_system.eliminated_operator_diagnostics(state, voltage)
    result = {
        "schema": ROW_SCHEMA, "voltage_V": float(voltage),
        "coordinate": state.coordinate, "electron_qf_increment_V": state.dqfn,
        "hole_qf_increment_V": state.dqfp, "storage": state.storage, "rate": state.rate,
        "poisson_residual_C_m2": state.poisson_residual,
        "direct_poisson_residual_C_m2": state.direct_poisson_residual,
        "local_residual": state.local_residual,
        "electron_current_A_m2": state.current_n,
        "hole_current_A_m2": state.current_p,
        "positive_ion_flux_m2_s": state.positive_flux,
        "positive_ion_rate_m3_s": state.positive_rate,
        "positive_ion_current_A_m2": state.positive_current,
        "carrier_conduction_A_m2": state.carrier_conduction,
        "conduction_A_m2": state.conduction,
        "local_carrier_residual": float(local_carrier),
        "local_gauss_residual": float(local_gauss),
        "eliminated_operator": details,
        "eliminated_operator_error": max(v["relative_error"] for v in details.values()),
        "finite_step": previous is not None,
        "jacobian_checked": bool(check_jacobian),
        "analytic_jacobian_error": None,
        "accepted_state_jacobian_nnz": None,
        "scaled_nonlinear_residual": 0.0,
    }
    # Opt-in precision diagnostics carry the actual independent solve without
    # introducing a non-component key into the legacy operator comparisons.
    if hasattr(details, "precision_evidence"):
        result["eliminated_precision"] = details.precision_evidence
    reference_quantization = getattr(working, "rebase_evidence", None)
    if reference_quantization is not None:
        result["reference_quantization"] = reference_quantization
    if previous is not None:
        storage_scale = scaling_system.storage_scale(previous.storage, previous, dt, policy)
        poisson_scale = scaling_system.poisson_scale(policy)
        local_scale = scaling_system.local_algebraic_scale(policy)
        residual, jacobian, evaluated = working.residual_and_jacobian(
            state.coordinate, voltage, previous, dt, storage_scale, poisson_scale, local_scale,
        )
        # A state's diagnostic must refer to that very state, not a second
        # representation silently rounded to a different physical population.
        _same(snapshot(working, state), snapshot(working, evaluated), "recorded state evaluation")
        transient = working.transient_current_metrics(state, previous, dt)
        charge_absolute, charge_relative = working.charge_balance_metrics(state, previous, dt)
        increment = working.storage_increment(state, previous)
        result.update({
            "storage_increment": increment,
            "storage_residual": increment - float(dt) * state.rate,
            "storage_scale": storage_scale, "poisson_scale_C_m2": poisson_scale,
            "local_algebraic_scale": local_scale, "scaled_residual_vector": residual,
            "scaled_nonlinear_residual": float(np.max(np.abs(residual))),
            "accepted_state_jacobian_nnz": int(jacobian.nnz),
            "charge_balance_absolute_A_m2": float(charge_absolute),
            "charge_balance_relative": float(charge_relative),
            "all_face_current_relative": float(transient[4]),
            "interface_current_relative": float(transient[5]),
            "internal_displacement_A_m2": transient[0], "internal_total_A_m2": transient[1],
            "interface_conduction_A_m2": transient[2],
            "interface_displacement_A_m2": transient[3],
            "interface_total_A_m2": transient[2] + transient[3],
        })
        if check_jacobian:
            result["analytic_jacobian_error"] = _jacobian_error(
                working, state.coordinate, voltage, previous, dt, storage_scale,
                poisson_scale, local_scale, jacobian, policy.jacobian_check_step,
            )
    result["independent_physics"] = independent_physics_row(
        working, state, previous, dt, reported=result)
    return json_data(result)


def _physical(scaling, working, state, previous, dt, policy, initial):
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import _trap_storage_check
    physical = physical_step_record(working, state, previous, dt)
    physical["trap_storage_check"] = _trap_storage_check(
        scaling, state, previous, dt, policy, physical,
    )
    if previous is None:
        physical["trap_storage_error_A_m2"] = 0.0
        for key in ("contact_maxwell_A_m2", "contact_displacement_A_m2", "internal_maxwell_A_m2",
                    "contact_internal_current_spread_relative"):
            physical.pop(key, None)
        physical["regular_right_limit"] = initial.event["regular_current"]
        physical["initial_algebraic_certificate"] = initial.event["algebraic_certificate"]
    return json_data(physical)


def _level(states, observations, initial):
    """Recreate accepted-state summaries used by the original certificate."""
    finite = [v for v in observations if v["finite_step"]]
    maximum = lambda name: max((v[name] for v in finite), default=0.0)
    initial_current = initial.initial_current_metrics
    poisson = max(float(np.max(np.abs(v["poisson_residual_C_m2"]))) for v in observations)
    poisson = max(poisson, max((float(np.max(np.abs(v["direct_poisson_residual_C_m2"])))
                               for v in finite if v["direct_poisson_residual_C_m2"] is not None), default=0.0))
    return SimpleNamespace(
        states=tuple(s for s, _ in states),
        total_current=np.asarray([initial_current[1]] + [v["internal_total_A_m2"] for _, v in states[1:]]),
        displacement=np.asarray([initial_current[0]] + [v["internal_displacement_A_m2"] for _, v in states[1:]]),
        interface_total_current=np.asarray([initial_current[2] + initial_current[3]]
                                           + [v["interface_total_A_m2"] for _, v in states[1:]]),
        maximum_scaled_residual=maximum("scaled_nonlinear_residual"),
        maximum_local_carrier_residual=max(v["local_carrier_residual"] for v in observations),
        maximum_local_gauss_residual=max(v["local_gauss_residual"] for v in observations),
        maximum_jacobian_error=max((v["analytic_jacobian_error"] for v in finite
                                   if v["jacobian_checked"]), default=0.0),
        maximum_charge_balance_error=maximum("charge_balance_relative"),
        maximum_charge_balance_absolute_error=maximum("charge_balance_absolute_A_m2"),
        maximum_face_spread=max(initial_current[4], maximum("all_face_current_relative")),
        maximum_interface_current_error=max(initial_current[5], maximum("interface_current_relative")),
        maximum_operator_error=max(v["eliminated_operator_error"] for v in observations),
        maximum_poisson_residual=poisson,
        maximum_nnz=max((v["accepted_state_jacobian_nnz"] for v in finite), default=0),
    )


def verify_r1_step_physics(stack, intervals, binding, prepared, record, *,
                           expected_prepared_sha256=None, allow_incomplete=False):
    """Re-evaluate a bounded complete trajectory or a saved failed prefix.

    The caller must separately anchor source and policy. Legacy rows without
    exact coordinates are unavailable, never upgraded to scientific acceptance.
    A successful failed-prefix audit means recorded rows are truthful; it never
    means that the failed experiment is physically certified or complete.
    """
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import (
        _physical_step_checks, _trace_certificate, nonfinite_numeric_paths,
    )
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_result_contract import (
        RESULT_ARRAY_FIELDS, verify_step_metadata,
    )
    if not allow_incomplete and nonfinite_numeric_paths(record):
        raise R1PhysicsValidationError("nonfinite numeric evidence in controlled result")
    prepared = prepared if isinstance(prepared, R1PreparedState) else R1PreparedState.from_dict(prepared)
    _same(record.get("prepared_sha256"), prepared.sha256, "preparation identity")
    if expected_prepared_sha256 is not None:
        _same(prepared.sha256, expected_prepared_sha256, "external preparation identity")
    _same(record.get("reference_sha256"), binding["sha256"], "reference identity")
    _same(record.get("intervals"), int(intervals), "interval count")
    policy = _preparation_policy(record["policy"])
    controls = R1DynamicsControls.from_label(record["control_label"])
    _same(record["controls"], json_data(controls), "rate controls")
    base, before_D = verify_prepared_physics(prepared, stack, binding, policy=policy)
    if controls == R1DynamicsControls():
        system, before = base, before_D
    else:
        system = _make_system(stack, base.grid, base.material, base.common_dc_state,
                              binding, controls, policy)
        before = system.evaluate(system.initial_coordinate(), 0.0)
        for name in ("n", "p", "positive", "occupancy", "phi", "sheet_charge"):
            _same(getattr(before, name), getattr(before_D, name), "initial control population " + name)
    verify_step_metadata(record, intervals=intervals, policy=policy, polarity=system.polarity, same=_same)
    amplitude = float(record["amplitude_V"])
    times = np.asarray(record["times_s"], dtype=float)
    if (not np.isfinite(amplitude) or not 0 < abs(amplitude) < 0.02
            or times.ndim != 1 or len(times) < 2 or times[0] != 0.0
            or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0)
            or times[1] < 1e-12):
        raise R1PhysicsValidationError("invalid physical reconstruction voltage/time domain")
    _same(record["voltage_V"], np.full(times.size, amplitude), "constant step voltage")
    initial = build_initial_step(system, before, amplitude, policy=policy)
    _same(record.get("initial_event"), initial.event, "initial charging and right-limit current")
    _same(record.get("physics_reconstruction"), reconstruction_context(
        initial.system, initial.zero_plus, prepared.sha256, binding["sha256"]), "initial reconstruction references")
    rows = record.get("accepted_steps", [])
    if not isinstance(rows, list):
        raise R1PhysicsValidationError("accepted rows must be a list")
    expected_count = sum(1 + (times.size - 1) * count for count in policy.refinement_substeps)
    if len(rows) > expected_count or (not allow_incomplete and len(rows) != expected_count):
        raise R1PhysicsValidationError("physical reconstruction accepted-row coverage mismatch")
    if not rows and not allow_incomplete:
        raise R1PhysicsValidationError("no accepted states to reconstruct")
    output_levels, all_physical, summaries, integrals = [], [], [], {}
    available_levels = {}
    consumed = 0
    all_recomputed_rows = []
    any_physical_failure = False
    independent_physics_passed = True
    limit_violations = []
    for count in policy.refinement_substeps:
        previous, state_outputs, observations = None, [], []
        integrated = 0.0
        schedule = [(0.0, 0.0, True)]
        for point in range(1, times.size):
            dt = float(times[point] - times[point-1]) / count
            schedule.extend((float(times[point-1] + (step+1) * dt), dt, step == count-1)
                            for step in range(count))
        for local_index, (time, dt, is_output) in enumerate(schedule):
            if consumed == len(rows):
                break
            row = rows[consumed]
            _same(row["substeps"], int(count), "nested level")
            _same(row["time_s"], time, "accepted time")
            _same(row["dt_s"], dt, "accepted duration")
            _same(row["phase"], "0+" if local_index == 0 else "accepted_regular_step", "row phase")
            _same(row["solver_accepted"], local_index != 0, "finite-step applicability")
            saved = row.get("physics_reconstruction")
            if (not isinstance(saved, dict) or saved.get("schema") != ROW_SCHEMA
                    or saved.get("available") is False):
                raise R1PhysicsValidationError("exact accepted-state reconstruction evidence unavailable")
            coordinate = np.asarray(saved["coordinate"], dtype=float)
            if coordinate.shape != (initial.system.dimension,) or not np.all(np.isfinite(coordinate)):
                raise R1PhysicsValidationError("invalid saved solver coordinates")
            _same(saved["voltage_V"], amplitude, "row voltage")
            if local_index == 0:
                _same(coordinate, np.zeros(initial.system.dimension), "initial integration coordinate")
                working, local_previous = initial.system, None
                state = replace(initial.zero_plus, coordinate=coordinate.copy())
            else:
                working, local_previous = initial.system.rebase(previous)
                working.set_voltage_lift(amplitude, local_previous)
                state = working.evaluate(coordinate, amplitude)
            _same(row["state"], snapshot(working, state), f"accepted state {consumed}")
            diagnostic = capture_r1_physics_row(initial.system, working, state, local_previous,
                amplitude, dt, policy, check_jacobian=local_index == 1)
            independent = diagnostic["independent_physics"]
            independent_physics_passed &= independent["passed"]
            if not independent["passed"]:
                limit_violations.append({"row":consumed, "substeps":count,
                    "independent_physics_reasons":independent["reasons"]})
            _same(saved, diagnostic, f"state equations {consumed}")
            _same(row["scaled_nonlinear_residual"], diagnostic["scaled_nonlinear_residual"], "nonlinear residual")
            physical = _physical(initial.system, working, state, local_previous, dt, policy, initial)
            _same(row["physical"], physical, f"state-to-current and charge {consumed}")
            recomputed = dict(row, state=snapshot(working, state), physical=physical,
                              physics_reconstruction=diagnostic)
            evidence = dict(recomputed)
            for key in ("physical_checks", "physical_checks_passed", "physical_failure_reasons"):
                evidence.pop(key, None)
            if local_index == 0:
                evidence["initial_event"] = initial.event
            checks = _physical_step_checks(physical, finite_step=local_index != 0,
                                           policy=policy, evidence=evidence)
            _same(row["physical_checks"], checks, "physical checks")
            _same(row["physical_checks_passed"], checks["passed"], "physical pass")
            _same(row["physical_failure_reasons"], checks["reasons"], "physical failure reasons")
            any_physical_failure |= not checks["passed"]
            row_limits = {
                "local_carrier_residual": policy.maximum_local_carrier_normalized_residual,
                "local_gauss_residual": policy.maximum_local_gauss_normalized_residual,
                "eliminated_operator_error": policy.maximum_eliminated_operator_relative_error,
            }
            if local_index:
                row_limits.update({
                    "scaled_nonlinear_residual": policy.maximum_scaled_nonlinear_residual,
                    "charge_balance_relative": policy.maximum_charge_balance_relative_error,
                    "all_face_current_relative": policy.maximum_all_face_current_spread_relative,
                    "interface_current_relative": policy.maximum_two_sided_interface_total_current_relative_error,
                })
                if diagnostic["jacobian_checked"]:
                    row_limits["analytic_jacobian_error"] = policy.maximum_jacobian_column_relative_error
            for name, limit in row_limits.items():
                if not np.isfinite(diagnostic[name]) or diagnostic[name] > limit:
                    limit_violations.append({"row": consumed, "metric": name,
                                             "value": diagnostic[name], "limit": limit})
            if not checks["passed"]:
                limit_violations.append({"row": consumed, "physical_reasons": checks["reasons"]})
            if local_index:
                all_physical.append(physical)
                integrated += float(working.polarity * physical["contact_maxwell_A_m2"][0] * dt)
            _same(row["regular_integrated_charge_C_m2"], integrated, "integrated physical current")
            observations.append(diagnostic)
            if is_output:
                state_outputs.append((state, diagnostic))
            summaries.append({"row": consumed, "substeps": count, "time_s": time,
                              "eliminated_operator": {name: {key: detail[key] for key in (
                                  "maximum_absolute_difference", "normalization_scale", "normalization_floor",
                                  "relative_error", "floor_active", "unit")}
                                  for name, detail in diagnostic["eliminated_operator"].items()}})
            all_recomputed_rows.append(recomputed)
            previous, consumed = state, consumed + 1
        if observations:
            integrals[count] = integrated
            available_levels[count] = _level(state_outputs, observations, initial)
        if len(observations) == len(schedule):
            output_levels.append(available_levels[count])
        else:
            break
    complete = consumed == expected_count
    certificate = None
    required_result_fields = RESULT_ARRAY_FIELDS
    has_complete_result = complete and all(key in record for key in required_result_fields)
    if complete:
        certificate = _trace_certificate(initial.system, before, output_levels, policy, all_physical)
        # The original count includes transient Newton iterates that were not
        # accepted. Accepted-state sparse structure is independently checked;
        # the historical iteration maximum stays explicitly outside this claim.
        stored_certificate = dict(record["certificate"])
        expected_certificate = dict(certificate)
        if "metrics" in stored_certificate:
            nnz = stored_certificate.get("analytic_jacobian_nnz")
            if type(nnz) is not int or not 0 < nnz < initial.system.dimension**2:
                raise R1PhysicsValidationError("physical certificate lacks a sparse Jacobian count")
            if nnz < expected_certificate["analytic_jacobian_nnz"]:
                raise R1PhysicsValidationError("historical Jacobian maximum is below accepted-state structure")
            finite_record = dict(record, certificate=dict(stored_certificate))
            finite_record["certificate"].pop("finite_numeric_evidence", None)
            paths = nonfinite_numeric_paths(finite_record)
            _same(stored_certificate.get("finite_numeric_evidence"), {
                "passed": not paths, "nonfinite_numeric_paths": paths,
                "scope": "all_numeric_leaves_before_digest",
            }, "finite numeric certificate")
            if paths:
                raise R1PhysicsValidationError("nonfinite numeric evidence in completed controlled result")
            for value in (stored_certificate, expected_certificate):
                value.pop("analytic_jacobian_nnz", None)
                value.pop("finite_numeric_evidence", None)
            _same(stored_certificate, expected_certificate, "recomputed physical certificate")
        elif not allow_incomplete:
            raise R1PhysicsValidationError("physical certificate metrics unavailable")
    elif record.get("certificate", {}).get("certified") is not False:
        raise R1PhysicsValidationError("incomplete failed trajectory must retain a failed certificate")
    elif "metrics" in record.get("certificate", {}):
        raise R1PhysicsValidationError("incomplete trajectory cannot claim a full-trace metric certificate")
    final = available_levels.get(policy.refinement_substeps[-1])
    sampled_fields = ("output_states", "regular_currents", "finite_step_averages")
    if final is None and any(name in record for name in sampled_fields):
        raise R1PhysicsValidationError("result samples have no reconstructed finest-level prefix")
    if final is not None:
        output_states = {name: np.asarray([getattr(s, attribute) for s in final.states])
            for name, attribute in (("n_m3", "n"), ("p_m3", "p"), ("positive_m3", "positive"),
                                    ("occupancy", "occupancy"), ("phi_V", "phi"), ("sheet_charge_C_m2", "sheet_charge"))}
        if "output_states" in record:
            _same(record["output_states"], output_states, "output state samples")
        if "regular_currents" in record:
            currents = [regular_current_at_state(initial.system, state, policy=policy).evidence for state in final.states]
            _same(record["regular_currents"], currents, "instantaneous currents from each output state")
        if "finite_step_averages" in record:
            from perovskite_sim.experiments.one_dimensional_mechanism_r1_result_contract import AVERAGE_NOTE
            _same(record["finite_step_averages"], {
                "internal_total_A_m2": final.total_current,
                "internal_displacement_A_m2": final.displacement,
                "interface_total_A_m2": final.interface_total_current, "note": AVERAGE_NOTE,
            }, "finite-step current")
    if "accepted_state_arrays" in record:
        if not rows:
            raise R1PhysicsValidationError("accepted state arrays have no reconstructed rows")
        _same(record["accepted_state_arrays"], {key: np.asarray([r["state"][key] for r in all_recomputed_rows])
                                              for key in rows[0]["state"]}, "accepted state arrays")
    if "charge_integral" in record:
        from perovskite_sim.experiments.one_dimensional_mechanism_r1_result_contract import CHARGE_QUADRATURE
        impulse = initial.event["impulse_charge_C_m2"]
        _same(record["charge_integral"], {"regular_by_substeps_C_m2": integrals,
            "impulse_charge_C_m2": impulse,
            "complete_by_substeps_C_m2": {key: impulse + value for key, value in integrals.items()},
            "quadrature": CHARGE_QUADRATURE}, "complete charge integrals")
    if not has_complete_result and not allow_incomplete:
        raise R1PhysicsValidationError("complete result fields unavailable")
    if "failure" in record and rows and rows[-1].get("physical_failure_reasons"):
        from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import _trap_storage_summary, SCOPE as PROTOCOL_SCOPE
        last = rows[-1]
        expected_failure_certificate = {
            "certified": False, "reasons": last["physical_failure_reasons"], "scope": PROTOCOL_SCOPE,
            "failed_record_index": len(rows) - 1,
            "physical_checks": last["physical_checks"],
            "nonfinite_numeric_paths": last["physical_checks"]["nonfinite_numeric_paths"],
        }
        if all_physical:
            trap = _trap_storage_summary(all_physical)
            expected_failure_certificate.update(maximum_trap_storage_error_A_m2=trap["maximum_error_A_m2"],
                                                trap_storage=trap)
        actual_failure_certificate = dict(record["certificate"])
        # Persistence is a historical I/O claim with separate outer verification.
        actual_failure_certificate.pop("secondary_reasons", None)
        _same(actual_failure_certificate, expected_failure_certificate, "failed physical certificate")
        _same(record["failure"].get("reasons"), last["physical_failure_reasons"], "failed physical reasons")
        _same(record["failure"].get("record_index"), len(rows)-1, "failed physical row")
    limits_satisfied = independent_physics_passed and not any_physical_failure and not limit_violations
    if certificate is not None:
        limits_satisfied = limits_satisfied and certificate["certified"]
    successful = bool(has_complete_result and certificate is not None and certificate["certified"]
                      and limits_satisfied and "failure" not in record)
    return {
        "schema": SCHEMA, "scope": SCOPE, "certified": successful,
        "evidence_matches_equations": True, "content_matches_recomputed": True,
        "physical_limits_satisfied": bool(limits_satisfied),
        "independent_physics_passed": bool(independent_physics_passed),
        "physical_limit_violations": limit_violations, "complete": complete,
        "checked_row_count": consumed, "expected_row_count": expected_count,
        "checked_result_fields": [key for key in RESULT_ARRAY_FIELDS if key in record],
        "unavailable_result_fields": [key for key in RESULT_ARRAY_FIELDS if key not in record],
        "content_scope": "present_scientific_fields_and_saved_rows; failure_messages_and_solver_history_are_provenance",
        "metrics": None if certificate is None else certificate["metrics"],
        "limits": None if certificate is None else certificate["limits"],
        "operator_diagnostics": summaries,
        "reasons": [] if successful else ["failed_or_incomplete_experiment_not_scientifically_certified"],
        "provenance_only": reconstruction_context(initial.system, initial.zero_plus,
                             prepared.sha256, binding["sha256"])["provenance_only"],
    }
