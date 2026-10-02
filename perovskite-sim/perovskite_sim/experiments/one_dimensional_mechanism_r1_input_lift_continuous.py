"""Accepted-state continuation for the explicit R1 input-lift representation.

This entry point resumes a supplied state. It does not prepare equilibrium or
claim the refinement, eliminated-operator and cold-start protocol checks. The
caller supplies a fixed schedule and the original, unre-based scaling system.
Every step uses the existing Newton solver and its original acceptance policy.
"""
from dataclasses import dataclass
import math

import numpy as np

from . import interface_defect_transient as transient
from .one_dimensional_mechanism_r1_input_lift import (
    REPRESENTATION, RebasedInputLiftR1System, cat, DD,
    primary_inputs, independent_physics_row,
)


@dataclass(frozen=True)
class StepResult:
    system: object
    previous: object
    state: object
    step: dict
    solver_metadata: tuple
    assessment: dict
    scales: dict


def validate_schedule(steps):
    """Validate a finite, contiguous prescribed schedule before any solve."""
    schedule = tuple(dict(step) for step in steps)
    if not schedule:
        raise ValueError("continuation requires at least one prescribed step")
    previous_end = None
    for step in schedule:
        for name in ("previous_time_s", "time_s", "dt_s", "voltage_V"):
            value = step[name]
            if isinstance(value, (bool, np.bool_)) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite numeric data")
            step[name] = float(value)
        start, end, dt = (step[key] for key in ("previous_time_s", "time_s", "dt_s"))
        # Original interval/substep arithmetic can round the endpoint
        # subtraction differently from the original prescribed dt.
        tolerance = 8 * np.finfo(float).eps * max(abs(start), abs(end), abs(dt))
        if start < 0 or end <= start or dt <= 0 or abs((end-start)-dt) > tolerance:
            raise ValueError("step times and prescribed dt disagree")
        if previous_end is not None and start != previous_end:
            raise ValueError("continuation schedule is not contiguous")
        previous_end = end
    return schedule


def assess_step(system, state, previous, dt_s, policy, *, scales):
    """Read accepted words and independently assemble physics; never solve."""
    current, before = primary_inputs(state), primary_inputs(previous)
    residual = cat(
        (current["storage"]-before["storage"]-DD(dt_s)*current["rate"])/DD(scales["storage_scale"]),
        current["poisson_residual_C_m2"]/DD(scales["poisson_scale"]),
        current["local_residual"]/DD(scales["local_scale"]),
    ).hi
    settings = transient.effective_newton_acceptance_settings(policy)
    limits = {
        "scaled_residual": settings["maximum_scaled_nonlinear_residual"],
        "charge_balance_relative": settings["maximum_charge_balance_relative_error"],
        "all_face_current_spread_relative": settings["maximum_all_face_current_spread_relative"],
        "interface_current_spread_relative": settings["maximum_two_sided_interface_total_current_relative_error"],
    }
    _, charge = system.charge_balance_metrics(state, previous, dt_s)
    *_, faces, interface = system.solver_current_metrics(state, previous, dt_s)
    metrics = {"scaled_residual": float(np.max(np.abs(residual))),
               "charge_balance_relative": float(charge),
               "all_face_current_spread_relative": float(faces),
               "interface_current_spread_relative": float(interface)}
    checks = {name: math.isfinite(value) and 0 <= value <= limits[name]
              for name, value in metrics.items()}
    checks["full_residual_finite_and_in_bound"] = bool(
        residual.shape == (system.dimension,) and np.isfinite(residual).all()
        and (np.abs(residual) <= limits["scaled_residual"]).all())
    independent = independent_physics_row(system, state, previous, dt_s)
    checks["independent_physics"] = independent.get("passed") is True
    return {"schema": "R1InputLiftContinuousStepAssessmentV1",
            "operator_representation": REPRESENTATION,
            "scaled_residual": residual.tolist(), "residual_count": int(residual.size),
            "metrics": metrics, "limits": limits, "checks": checks,
            "independent_physics": independent, "passed": all(checks.values()),
            "full_protocol_qualified": False}


def advance_step(system, previous, step, policy, *, scaling_system, check_jacobian=False):
    """Advance once, preserving DD words through the accepted-state rebase.

    ``scaling_system`` must retain the initial reference arrays. Actual scales
    still depend on the current previous state and prescribed dt, as in the
    original integrator. A rejected step is never retried or subdivided here.
    """
    if not isinstance(system, RebasedInputLiftR1System):
        raise TypeError("continuation requires an explicit input-lift system")
    step, = validate_schedule((step,))
    working, local_previous = system.rebase(previous)
    working.set_voltage_lift(step["voltage_V"], local_previous)
    scales = {
        "storage_scale": scaling_system.storage_scale(local_previous.storage, local_previous, step["dt_s"], policy),
        "poisson_scale": scaling_system.poisson_scale(policy),
        "local_scale": scaling_system.local_algebraic_scale(policy),
    }
    for values in scales.values():
        if not np.isfinite(values).all() or not (np.asarray(values) > 0).all():
            raise ValueError("continuation scales must be positive finite arrays")
    try:
        result = transient._solve_step(working, np.zeros(working.dimension), local_previous,
            step["voltage_V"], step["dt_s"], policy,
            check_jacobian=check_jacobian, scaling_system=scaling_system)
    except transient.InterfaceDefectTransientError as error:
        if isinstance(getattr(error, "result", None), dict):
            error.result["continuation_step"] = step
        raise
    state, *metadata = result
    assessment = assess_step(working, state, local_previous, step["dt_s"], policy, scales=scales)
    assessment["native_residual_readout_matches"] = assessment["metrics"]["scaled_residual"] == result[2]
    assessment["passed"] &= assessment["native_residual_readout_matches"]
    if not assessment["passed"]:
        error = transient.InterfaceDefectTransientError("accepted continuation step failed saved-state assessment")
        error.result = {"step": step, "assessment": assessment}
        raise error
    return StepResult(working, local_previous, state, step, tuple(metadata), assessment, scales)


def advance_steps(system, previous, steps, policy, *, scaling_system, accepted_step_observer=None):
    """Yield sequential accepted results; stop immediately on a failed step."""
    for step in validate_schedule(steps):
        result = advance_step(system, previous, step, policy, scaling_system=scaling_system)
        if accepted_step_observer is not None:
            accepted_step_observer(result)
        yield result
        previous = result.state


__all__ = ["StepResult", "validate_schedule", "assess_step", "advance_step", "advance_steps"]
