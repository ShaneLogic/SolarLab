"""Unwired N proposal: a physical-density seed for a complete transient step."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any, Callable, Mapping, Protocol

import numpy as np
from scipy.sparse import diags

from solarlab.contracts import frozen_array
from solarlab.solvers.nonlinear.newton import NewtonResult, NewtonSettings, Termination, factorize, solve_newton
from solarlab.solvers.nonlinear.problem import Matrix, SolveBudget, SteadyProblem, Vector, state_digest

__all__ = ["CarrierDensityDevice", "DensitySeedContext", "recover_density_seed",
           "stagnation_basin_signature", "transient_density_basin_seed"]


class CarrierDensityDevice(Protocol):
    @property
    def node_count(self) -> int: ...

    def fields(self, u: Vector, *, voltage_V: float = 0.0) -> tuple[Vector, Vector, Vector]: ...


@dataclass(frozen=True)
class DensitySeedContext:
    """Caller-certified full target and immutable-history scope for one retry.

    The caller names its deadline exception types so its original exception can
    propagate unchanged, including a deadline derived from RuntimeError.
    """

    device: CarrierDensityDevice
    voltage_V: float
    scope: Mapping[str, Any]
    validate: Callable[[], None]
    history_fingerprint: Callable[[], str]
    checkpoint: Callable[[], None] = lambda: None
    deadline_exceptions: tuple[type[Exception], ...] = (TimeoutError,)


def transient_density_basin_seed(
    problem: SteadyProblem, u: Vector, *, device: CarrierDensityDevice, voltage_V: float,
    work_observer: Callable[[str], None] | None = None,
) -> Vector:
    """One carrier density linearization using the full BE/BDF residual and J.

    The supplied problem must retain the actual time-step storage and history.
    Its potential stays frozen only while constructing this unaccepted guess;
    the original coupled Newton and physical/LTE gates still judge the result.
    No default, waveform, old state, inventory, or generation stage is changed.

    The caller must charge the one residual, Jacobian and factorization to its
    existing work/deadline accounting. This helper grants no solve or retry.
    Nonpositive/nonfinite density proposals fail through the existing seeder.
    """
    if np.asarray(u).shape != (3 * device.node_count,):
        raise ValueError("transient carrier seed requires an electronic 3N state")
    observe = (lambda phase: None) if work_observer is None else work_observer
    count = device.node_count
    observe("field_map")
    _, n, p = device.fields(u, voltage_V=voltage_V)
    observe("checkpoint")
    if not all(np.all(np.isfinite(v)) and np.all(v > 0) for v in (n, p)):
        raise ValueError("density seed requires finite strictly positive input populations")
    residual = problem.residual(u)[count:]
    density_inverse = np.r_[1.0 / n, -1.0 / p]
    matrix = problem.jacobian(u)[count:, count:] @ diags(density_inverse)
    observe("factorization")
    factor = factorize(matrix)
    observe("linear_solve")
    increment = factor.solve(-residual)
    observe("checkpoint")
    trial_n, trial_p = n + increment[:count], p + increment[count:]
    if not all(np.all(np.isfinite(v)) and np.all(v > 0) for v in (trial_n, trial_p)):
        raise ValueError("density preparation did not produce finite positive carriers")
    observe("conversion")
    result = np.array(u, copy=True)
    for offset, density, trial, delta, sign in (
        (count, n, trial_n, increment[:count], 1.0),
        (2 * count, p, trial_p, increment[count:], -1.0),
    ):
        with np.errstate(over="ignore", divide="ignore"):
            relative = delta / density
            change = np.log1p(relative)
        large = ~np.isfinite(change)
        change[large] = np.log(trial[large]) - np.log(density[large])
        result[offset:offset + count] += sign * change
    observe("checkpoint")
    if not np.all(np.isfinite(result)):
        raise ValueError("density seed coordinates are nonfinite; no clipping or fallback")
    return result


def stagnation_basin_signature(result: NewtonResult, tolerance: float) -> bool:
    """The frozen three-iteration, sub-epsilon whole-direction plateau only."""
    if result.accepted or result.termination != Termination.STAGNATED or len(result.iterations) < 3:
        return False
    return all(row.fresh_jacobian and row.backtracks == 0
               and math.isfinite(row.damping) and 0 < row.damping <= np.finfo(float).eps
               and math.isfinite(row.step_inf) and row.step_inf > 0
               and math.isfinite(row.scaled_residual) and row.scaled_residual > tolerance
               for row in result.iterations[-3:])


def recover_density_seed(
    problem: SteadyProblem, initial: Vector, original: NewtonResult, settings: NewtonSettings,
    context: DensitySeedContext, record: dict[str, Any], *,
    convergence_norm: Callable[[Vector], float] | None = None,
) -> NewtonResult:
    """One explicit density retry within the remaining original call budget.

    Caller eligibility is separate from the signature. The record is appended
    before calling this function so failures/timeouts retain all available work.
    Original and following Newton records are separate; only the latter is the
    returned candidate. A failure never grants another initializer or acceptance.
    """
    diagnostic_initial = np.asarray(initial, dtype=np.float64)
    diagnostic_initial = np.frombuffer(diagnostic_initial.tobytes(), dtype=np.float64).reshape(diagnostic_initial.shape)
    record.update(original_newton=original, original_initial=diagnostic_initial,
        scope=dict(context.scope), original_budget=asdict(settings.budget),
        seed=None, retry_newton=None, retry_attempted=False, status="checking_binding",
        seed_work={"residual": 0, "jacobian": 0, "field_map": 0, "factorization": 0,
                   "linear_solve": 0, "conversion": 0})
    work = record["seed_work"]
    if not stagnation_basin_signature(original, settings.tolerance):
        raise ValueError("density recovery requires the scoped recorded stagnation signature")
    before = context.history_fingerprint()
    initial_bytes = np.asarray(initial).tobytes()
    record["history_before"] = before
    budget = settings.budget
    used_i = len(original.iterations)
    left_i = budget.max_iterations - used_i
    left_r = budget.max_residual_evaluations - original.residual_evaluations
    left_j = budget.max_jacobian_evaluations - original.jacobian_evaluations
    if left_i < 1 or left_r < 2 or left_j < 2:
        record.update(status="remaining_budget_exhausted", history_after=before, error="no seed and following Newton allocation remains")
        return original

    class CountedSeed:
        def residual(self, u: Vector) -> Vector:
            context.checkpoint()
            if work["residual"] >= left_r - 1:
                raise ArithmeticError("density seed residual budget exhausted")
            work["residual"] += 1
            value = problem.residual(u)
            context.checkpoint()
            return value

        def jacobian(self, u: Vector) -> Matrix:
            context.checkpoint()
            if work["jacobian"] >= left_j - 1:
                raise ArithmeticError("density seed Jacobian budget exhausted")
            work["jacobian"] += 1
            value = problem.jacobian(u)
            context.checkpoint()
            return value

    def observe(phase: str) -> None:
        context.checkpoint()
        if phase != "checkpoint":
            work[phase] += 1

    try:
        context.checkpoint()
        context.validate()
        if original.start_digest != state_digest(initial):
            raise ValueError("original Newton start differs from the recorded initial guess")
        record["status"] = "constructing_seed"
        proposed = transient_density_basin_seed(CountedSeed(), initial,
            device=context.device, voltage_V=context.voltage_V, work_observer=observe)
        if initial_bytes != np.asarray(initial).tobytes() or context.history_fingerprint() != before:
            raise RuntimeError("density seed mutated the frozen initial/history")
        record["seed"] = frozen_array(proposed)
        remaining = SolveBudget(left_i, left_r - work["residual"], left_j - work["jacobian"], budget.max_backtracks)
        record.update(remaining_budget=asdict(remaining), status="retrying", retry_attempted=True)
        following = solve_newton(problem, proposed, replace(settings, budget=remaining), convergence_norm=convergence_norm)
        record["retry_newton"] = following
        context.checkpoint()
        after = context.history_fingerprint()
        record["history_after"] = after
        if before != after or initial_bytes != np.asarray(initial).tobytes():
            raise RuntimeError("density retry mutated the frozen initial/history")
        record["status"] = "recovered" if following.accepted else "retry_rejected"
        return following
    except context.deadline_exceptions as exc:
        record.update(status="wall_budget_exhausted", error=str(exc), history_after=context.history_fingerprint())
        raise
    except RuntimeError as exc:
        record.update(status="frozen_history_violation", error=str(exc), history_after=context.history_fingerprint())
        raise
    except Exception as exc:
        record.update(status="seed_or_binding_failed", error=type(exc).__name__ + ": " + str(exc),
                      history_after=context.history_fingerprint())
        return original
