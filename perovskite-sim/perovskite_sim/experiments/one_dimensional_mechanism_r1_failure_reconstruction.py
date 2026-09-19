"""Re-evaluate a saved R1 failed Newton iterate, never retry time integration."""
from __future__ import annotations

import numpy as np

from perovskite_sim.experiments.one_dimensional_mechanism_r1_failure_witness import newton_witness_available

WITNESS_FIELDS = frozenset({"schema", "terminal_state_available", "scope", "voltage_V", "dt_s",
    "coordinate", "previous_coordinate", "previous_state", "attempted_state", "scaled_residual_vector",
    "storage_scale", "poisson_scale", "local_scale", "diagnostics", "independent_physics",
    "time_s", "previous_time_s", "substeps"})
DIAGNOSTIC_FIELDS = frozenset({"iteration", "scaled_nonlinear_residual", "charge_balance_relative",
    "solver_current_spread_relative", "interface_current_spread_relative", "linear_backward_error"})


def rebuild_failure_witness(stack, intervals, binding, prepared, result):
    """Rebuild saved prefix and terminal state with independently derived scales.

    Call only after the result and preparation's source/request/manifests have
    been anchored. This is content replay, not a second solution-error oracle
    or reconstruction of historical Newton/line-search iterations.
    """
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import (
        R1PreparedState, restore_common_state, _preparation_policy, snapshot, json_data)
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_dynamics import R1DynamicsControls
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_step import build_initial_step
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_independent_physics import independent_physics_row
    witness = result.get("failure", {}).get("numerical_evidence")
    if not isinstance(witness, dict) or witness.get("schema") != "R1NewtonFailureWitnessV1":
        return {"available": False, "terminal_state_physics_recomputed": None,
                "reason": "no_supported_failed_newton_witness"}
    if not newton_witness_available(witness):
        return {"available": False, "terminal_state_physics_recomputed": None,
                "reason": "failed_newton_witness_collection_unavailable",
                "witness_collection_error": witness["witness_collection_error"]}
    if set(witness) != WITNESS_FIELDS or set(witness.get("diagnostics", {})) != DIAGNOSTIC_FIELDS:
        raise ValueError("failed Newton witness fields differ from the declared schema")
    if witness["scope"] != "failed_iterate_not_accepted_step_or_complete_newton_history":
        raise ValueError("failed Newton witness scope mismatch")
    def same(actual, expected, name):
        if json_data(actual) != json_data(expected):
            raise ValueError("failed Newton witness mismatch: " + name)
    policy = _preparation_policy(result["policy"])
    prepared = prepared if isinstance(prepared, R1PreparedState) else R1PreparedState.from_dict(prepared)
    system, before = restore_common_state(prepared, stack, intervals, binding,
        controls=R1DynamicsControls.from_label(result["control_label"]), policy=policy,
        expected_prepared_sha256=prepared.sha256)
    voltage = float(result["amplitude_V"])
    same(witness["voltage_V"], voltage, "voltage")
    initial = build_initial_step(system, before, voltage, policy=policy)
    rows = result.get("accepted_steps", [])
    if not rows:
        raise ValueError("failed Newton witness requires the saved initial/accepted prefix")
    previous = None
    for row in rows:
        if row["phase"] == "0+":
            previous = initial.zero_plus
        else:
            if previous is None:
                raise ValueError("failed Newton prefix lacks its initial state")
            working, reference = initial.system.rebase(previous)
            working.set_voltage_lift(voltage, reference)
            previous = working.evaluate(np.asarray(row["physics_reconstruction"]["coordinate"]), voltage)
        same(snapshot(initial.system, previous), row["state"], "accepted prefix state")
    same(witness["previous_time_s"], rows[-1]["time_s"], "last accepted time")
    same(witness["substeps"], rows[-1]["substeps"], "refinement level")
    substeps = witness["substeps"]
    if type(substeps) is not int or substeps not in policy.refinement_substeps:
        raise ValueError("failed Newton witness has invalid subdivision")
    times = np.asarray(result["times_s"], dtype=float)
    point = int(np.searchsorted(times, witness["previous_time_s"], side="right"))
    if not 0 < point < len(times):
        raise ValueError("failed Newton time is outside requested intervals")
    dt = float((times[point] - times[point - 1]) / substeps)
    same(witness["dt_s"], dt, "requested dt")
    # Use the same requested-grid arithmetic as the producer, not addition
    # to an already rounded saved time.
    local_step = int(round((witness["previous_time_s"] - times[point - 1]) / dt))
    same(witness["time_s"], float(times[point - 1] + (local_step + 1) * dt), "attempted time")
    working, previous = initial.system.rebase(previous)
    working.set_voltage_lift(voltage, previous)
    same(snapshot(working, previous), witness["previous_state"], "previous physical state")
    same(previous.coordinate, witness["previous_coordinate"], "previous coordinate")
    storage = initial.system.storage_scale(previous.storage, previous, dt, policy)
    poisson = initial.system.poisson_scale(policy)
    local = initial.system.local_algebraic_scale(policy)
    for actual, name in ((storage, "storage_scale"), (poisson, "poisson_scale"), (local, "local_scale")):
        same(actual, witness[name], name)
    coordinate = np.asarray(witness["coordinate"], dtype=float)
    if coordinate.shape != (working.dimension,) or not np.all(np.isfinite(coordinate)):
        raise ValueError("failed Newton coordinate is invalid")
    residual, _, state = working.residual_and_jacobian(coordinate, voltage, previous, dt, storage, poisson, local)
    same(snapshot(working, state), witness["attempted_state"], "attempted physical state")
    same(residual, witness["scaled_residual_vector"], "scaled residual")
    same(float(np.max(np.abs(residual))), witness["diagnostics"]["scaled_nonlinear_residual"], "residual norm")
    independent = independent_physics_row(working, state, previous, dt)
    same(independent, witness["independent_physics"], "independent physical assembly")
    return {"available": True, "terminal_state_physics_recomputed": True,
            "terminal_state_physics_passed": independent["passed"],
            "scaled_nonlinear_residual": float(np.max(np.abs(residual))),
            "metrics": independent["metrics"], "reasons": independent["reasons"],
            "not_recomputed": ["historical_Newton_path", "linear_solve_backward_error", "iteration_count"],
            "accepted": False}
