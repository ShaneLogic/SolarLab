"""Variable-step classical BDF1/2 in physical equation storage.

Only F, F_u, S and positive S_u are required. The sign is F-dS/dt=0.
Optional cancellation-safe storage_increment methods are algebraically the
same S difference, never coordinate time differences. Accepted history,
rejected trials, protocol restarts and interpolated samples are distinct.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
import math
import time
from typing import Any, Protocol

import numpy as np
from scipy.sparse import csc_matrix, diags, issparse

from solarlab.contracts import frozen_array
from solarlab.solvers.nonlinear.newton import NewtonResult, NewtonSettings, solve_newton
from solarlab.solvers.nonlinear.problem import CallableProblem, GateOutcome, Matrix, Vector


class StorageProblem(Protocol):
    """Square residual and physical equation storage in caller coordinates."""

    def residual(self, u: Vector) -> Vector: ...
    def jacobian(self, u: Vector) -> Matrix: ...
    def storage(self, u: Vector) -> Vector: ...
    def storage_tangent(self, u: Vector) -> Matrix: ...


class _StartupNewtonRejected(ArithmeticError):
    """A returned auxiliary Newton failure rejects a trial, not the history."""


@dataclass(frozen=True)
class BDFOptions:
    initial_step_s: float
    min_step_s: float = 1e-16
    max_step_s: float = 1e30
    relative_tolerance: float = 1e-4
    absolute_tolerance: float | np.ndarray = 1e-12
    max_order: int = 2
    adaptive: bool = True
    safety: float = .8
    min_factor: float = .2
    max_factor: float = 2.0
    max_steps: int = 10000
    max_attempts: int = 20000
    max_consecutive_rejections: int = 12
    wall_s: float = 60.0

    def __post_init__(self) -> None:
        for name in ("initial_step_s", "min_step_s", "max_step_s", "relative_tolerance", "wall_s"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name + " must be finite and positive")
        if self.min_step_s > self.initial_step_s or self.initial_step_s > self.max_step_s:
            raise ValueError("min_step <= initial_step <= max_step is required")
        absolute = np.asarray(self.absolute_tolerance, dtype=float)
        if absolute.ndim > 1 or not np.all(np.isfinite(absolute)) or np.any(absolute <= 0):
            raise ValueError("absolute_tolerance must be positive in physical storage units")
        if absolute.ndim == 1:
            object.__setattr__(self, "absolute_tolerance", frozen_array(absolute))
        if self.max_order not in (1, 2) or not .1 <= self.safety < 1:
            raise ValueError("orders1/2 and a conservative safety factor are required")
        if not 0 < self.min_factor < 1 < self.max_factor <= 2:
            raise ValueError("step growth is limited to2, below the BDF2 zero-stability ratio")
        for name in ("max_steps", "max_attempts", "max_consecutive_rejections"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + " must be a positive integer")


@dataclass(frozen=True)
class HistoryPoint:
    time_s: float
    u: Vector
    storage_value: Vector
    problem: StorageProblem = field(repr=False, compare=False)


def storage_increment(problem: StorageProblem, u: Vector, old: HistoryPoint) -> Vector:
    """Physical S difference; an adapter may preserve small differences."""
    function = getattr(problem, "storage_increment", None)
    value = (function(u, old.problem, old.u) if callable(function)
             else problem.storage(u) - old.storage_value)
    result = np.asarray(value, dtype=float)
    if result.shape != u.shape or not np.all(np.isfinite(result)):
        raise ValueError("storage increment must be finite on every equation row")
    return result


def bdf_coefficients(step_s: float, previous_step_s: float | None, order: int) -> Vector:
    if not math.isfinite(step_s) or step_s <= 0:
        raise ValueError("positive finite BDF step required")
    if order == 1:
        return np.array([1.0, -1.0])
    if order != 2 or previous_step_s is None or previous_step_s <= 0:
        raise ValueError("BDF2 needs its previous accepted interval")
    ratio = step_s / previous_step_s
    if not math.isfinite(ratio) or ratio > 2.0 * (1 + 8 * np.finfo(float).eps):
        raise ValueError("BDF2 step growth cannot exceed2")
    return np.array([(1 + 2 * ratio) / (1 + ratio), -(1 + ratio), ratio**2 / (1 + ratio)])


@dataclass(frozen=True)
class BDFStep:
    endpoint: StorageProblem
    history: tuple[HistoryPoint, ...]  # newest first; rejected states never enter
    end_s: float
    order: int

    @property
    def dt_s(self) -> float:
        return self.end_s - self.history[0].time_s

    @property
    def coefficients(self) -> Vector:
        old_step = None if len(self.history) < 2 else self.history[0].time_s - self.history[1].time_s
        return bdf_coefficients(self.dt_s, old_step, self.order)

    def storage_rate(self, u: Vector) -> Vector:
        coefficients = self.coefficients
        rate = coefficients[0] * storage_increment(self.endpoint, u, self.history[0]) / self.dt_s
        if self.order == 2:
            old = storage_increment(self.history[0].problem, self.history[0].u, self.history[1])
            rate -= coefficients[2] * old / self.dt_s
        return rate

    def residual(self, u: Vector) -> Vector:
        value = self.endpoint.residual(u)
        rate = self.storage_rate(u)
        result = value - rate
        tangent = self.endpoint.storage_tangent(u)
        if tangent.shape != (u.size, u.size):
            raise ValueError("physical storage tangent must match the equation rows")
        row_size = (np.asarray(abs(tangent).sum(axis=1)).ravel() if issparse(tangent)
                    else np.sum(abs(np.asarray(tangent)), axis=1))
        algebraic = (row_size == 0.0) & (rate == 0.0)
        # An algebraic row has no physical storage subtraction. Preserve its
        # represented value, including -0; nonzero history rates are retained
        # even when the current storage tangent happens to vanish.
        result[algebraic] = value[algebraic]
        return result

    def jacobian(self, u: Vector) -> Matrix:
        return self.endpoint.jacobian(u) - self.coefficients[0] * self.endpoint.storage_tangent(u) / self.dt_s


@dataclass(frozen=True)
class Assessment:
    accounting: dict[str, Any] = field(default_factory=dict)
    checks: tuple[GateOutcome, ...] = ()


@dataclass(frozen=True)
class BDFEvent:
    time_s: float
    restart: Callable[[StorageProblem, StorageProblem, Vector], tuple[Vector, Assessment]]
    name: str = "protocol_discontinuity"


@dataclass(frozen=True)
class BDFAttempt:
    start_s: float
    end_s: float
    order: int
    coefficients: Vector
    accepted: bool
    termination: str
    initial_coordinates: Vector
    candidate_coordinates: Vector
    error_norm: float
    error_estimate_storage: Vector | None
    estimator: str
    newton: NewtonResult | None
    auxiliary_newton: tuple[NewtonResult, ...]
    assessment: Assessment | None
    wall_s: float
    error: str | None = None


@dataclass(frozen=True)
class BDFSample:
    time_s: float
    u: Vector
    storage_value: Vector
    interpolant_order: int
    accepted_step_endpoint: bool
    projection: NewtonResult | None = None


@dataclass(frozen=True)
class RestartRecord:
    time_s: float
    name: str
    left_u: Vector
    right_u: Vector
    left_storage: Vector
    right_storage: Vector
    assessment: Assessment
    accepted: bool


@dataclass(frozen=True)
class BDFResult:
    initial_time_s: float
    initial_u: Vector
    final_time_s: float
    accepted_times_s: Vector
    accepted_coordinates: tuple[Vector, ...]
    attempts: tuple[BDFAttempt, ...]
    samples: tuple[BDFSample, ...]
    restarts: tuple[RestartRecord, ...]
    controls: dict[str, Any]
    completed: bool
    termination: str
    wall_s: float
    nonlinear_calls: int
    evaluation_counts: dict[str, int]
    error: str | None = None
    requested_output_times_s: Vector | None = None
    sample_failure: dict[str, Any] | None = None
    physical_certificate_status: str = "not_evaluated"
    schema: str = "solarlab.physical_storage_bdf12.v1"

    @property
    def accepted_steps(self) -> tuple[BDFAttempt, ...]:
        return tuple(item for item in self.attempts if item.accepted)

    @property
    def rejected_steps(self) -> tuple[BDFAttempt, ...]:
        return tuple(item for item in self.attempts if not item.accepted)


def lagrange_weights(nodes: Vector, target: float) -> Vector:
    result = np.ones(len(nodes))
    for j in range(len(nodes)):
        for k in range(len(nodes)):
            if j != k:
                result[j] *= (target - nodes[k]) / (nodes[j] - nodes[k])
    return result


def _prediction_change(points: tuple[HistoryPoint, ...], target: float) -> Vector:
    weights = lagrange_weights(np.array([point.time_s for point in points]), target)
    change = np.zeros_like(points[0].u)
    for weight, point in zip(weights[1:], points[1:]):
        change -= weight * storage_increment(points[0].problem, points[0].u, point)
    return change


def solve_bdf(
    factory: Callable[[float, str], StorageProblem], initial_u: Vector, interval_s: tuple[float, float], *,
    options: BDFOptions, settings: Callable[[BDFStep], NewtonSettings],
    convergence_norm: Callable[[Vector], float] | None = None,
    assess: Callable[[BDFStep, Vector], Assessment] | None = None,
    seed: Callable[[BDFStep], Vector] | None = None,
    breakpoints_s: tuple[float, ...] = (), events: tuple[BDFEvent, ...] = (),
    output_times_s: Vector | None = None,
) -> BDFResult:
    """Bounded physical BDF; samples never feed the accepted trajectory.

    Residual/Jacobian counters include projection work; sample_* are subsets.
    """
    started = time.monotonic()
    start, finish = interval_s
    if not math.isfinite(start) or not math.isfinite(finish) or finish <= start:
        raise ValueError("finite increasing integration interval required")
    u0 = frozen_array(initial_u)
    if u0.ndim != 1 or u0.size == 0:
        raise ValueError("a complete initial coordinate vector is required")
    output = np.array([start, finish]) if output_times_s is None else frozen_array(output_times_s)
    if (output.ndim != 1 or np.any(np.diff(output) < 0) or np.any(output < start) or np.any(output > finish)):
        raise ValueError("output samples must be ordered within the integration interval")
    event_map = {event.time_s: event for event in events}
    if len(event_map) != len(events) or any(not start <= t <= finish for t in event_map):
        raise ValueError("protocol events must be unique and within the interval")
    boundaries = sorted(set((finish, *breakpoints_s, *event_map)))
    if any(not math.isfinite(t) or t < start or t > finish for t in boundaries):
        raise ValueError("protocol segment breakpoints must lie within the interval")
    counts = {"residual": 0, "jacobian": 0, "storage": 0, "assessment": 0, "sample_projection": 0,
              "sample_residual": 0, "sample_jacobian": 0}
    nonlinear_calls = 0
    deadline = started + options.wall_s

    def checkpoint() -> None:
        if time.monotonic() >= deadline:
            raise TimeoutError("BDF whole-history wall budget exhausted")

    def point(t: float, u: Vector, problem: StorageProblem) -> HistoryPoint:
        checkpoint()
        counts["storage"] += 1
        value = np.asarray(problem.storage(u), dtype=float)
        if value.shape != u.shape or not np.all(np.isfinite(value)):
            raise ValueError("finite physical storage is required on every equation row")
        checkpoint()
        return HistoryPoint(t, frozen_array(u), frozen_array(value), problem)

    first = point(start, u0, factory(start, "left" if start in event_map else "right"))
    all_points = [first]
    history = [first]
    attempts: list[BDFAttempt] = []
    restarts: list[RestartRecord] = []
    dense: list[tuple[float, float, tuple[HistoryPoint, ...], int]] = []
    accepted_times, accepted_u = [start], [u0]
    t, h, consecutive, next_order = start, options.initial_step_s, 0, 1
    termination, error = "completed", None

    def perform(step: BDFStep, initial: Vector | None = None) -> NewtonResult:
        nonlocal nonlinear_calls
        guess = (step.history[0].u.copy() if seed is None else seed(step)) if initial is None else initial
        control = settings(step)
        def evaluate(kind: str, value: Vector) -> Any:
            checkpoint()
            counts[kind] += 1
            result = step.residual(value) if kind == "residual" else step.jacobian(value)
            checkpoint()
            return result
        nonlinear_calls += 1
        return solve_newton(CallableProblem(lambda value: evaluate("residual", value),
            lambda value: evaluate("jacobian", value)), guess, control, convergence_norm=convergence_norm)

    def restart(event: BDFEvent) -> bool:
        nonlocal history, next_order
        old = history[0]
        new_problem = factory(t, "right")
        right, assessment = event.restart(old.problem, new_problem, old.u.copy())
        new = point(t, right, new_problem)
        passed = all(check.passed for check in assessment.checks)
        restarts.append(RestartRecord(t, event.name, old.u, new.u, old.storage_value,
                                     new.storage_value, assessment, passed))
        if passed:
            history, next_order = [new], 1
            all_points.append(new)
        return passed

    try:
        if start in event_map and not restart(event_map[start]):
            raise ValueError("initial protocol restart physical checks failed")
        while t < finish:
            checkpoint()
            if len(accepted_times) - 1 >= options.max_steps or len(attempts) >= options.max_attempts:
                termination = "step_budget_exhausted"
                break
            boundary = next(value for value in boundaries if value > t)
            h = min(h, options.max_step_s, boundary - t)
            if h < options.min_step_s or t + h == t:
                termination = "step_underflow"
                break
            # Snap only a few representable time ulps, avoiding a spurious
            # near-zero last interval from floating-point accumulation.
            candidate_end = t + h
            time_roundoff = 8 * np.finfo(float).eps * max(abs(boundary), abs(t), h)
            end = boundary if boundary - candidate_end <= time_roundoff else candidate_end
            h = end - t
            order = next_order if len(history) >= 2 else 1
            if order == 2 and h > 2 * (history[0].time_s - history[1].time_s):
                order = 1
            endpoint = factory(end, "left" if end in event_map else "right")
            step = BDFStep(endpoint, tuple(history[:3]), end, order)
            guess = history[0].u.copy() if seed is None else seed(step)
            attempt_start = time.monotonic()
            newton, assessment, estimate = None, None, None
            auxiliary: list[NewtonResult] = []
            estimator, norm, accepted, why, message = "disabled_fixed_step", 0.0, False, "evaluation_error", None
            candidate = guess.copy()
            selected: HistoryPoint | None = None
            try:
                newton = perform(step, guess)
                candidate = newton.state
                checkpoint()
                if not newton.accepted:
                    why = "newton_" + newton.termination.value
                else:
                    counts["assessment"] += 1
                    assessment = Assessment() if assess is None else assess(step, candidate)
                    checkpoint()
                    if not all(check.passed for check in assessment.checks):
                        why = "physical_gate_rejected"
                    else:
                        if options.adaptive:
                            if len(history) < order + 1:
                                estimator = "startup_step_doubling_physical_storage"
                                middle = t + h / 2
                                half = BDFStep(factory(middle, "right"), tuple(history), middle, 1)
                                a = perform(half)
                                auxiliary.append(a)
                                if not a.accepted:
                                    raise _StartupNewtonRejected("startup error-estimate half step failed")
                                mid = point(middle, a.state, half.endpoint)
                                half2 = BDFStep(endpoint, (mid,), end, 1)
                                b = perform(half2)
                                auxiliary.append(b)
                                if not b.accepted:
                                    raise _StartupNewtonRejected("startup error-estimate second half step failed")
                                fine = point(end, b.state, endpoint)
                                estimate = 2 * storage_increment(endpoint, candidate, fine)
                            else:
                                estimator = "physical_storage_divided_difference"
                                change = storage_increment(endpoint, candidate, history[0])
                                prediction = _prediction_change(tuple(history[:order + 1]), end)
                                span = end - history[order].time_s
                                estimate = (h / (step.coefficients[0] * span)) * (change - prediction)
                            counts["storage"] += 1
                            new_s = np.asarray(endpoint.storage(candidate))
                            scale = np.asarray(options.absolute_tolerance) + options.relative_tolerance * np.maximum(abs(new_s), abs(history[0].storage_value))
                            if scale.shape not in ((), candidate.shape):
                                raise ValueError("physical absolute tolerances must match storage rows")
                            norm = float(np.max(abs(estimate) / scale))
                            if not math.isfinite(norm):
                                raise FloatingPointError("physical BDF error estimate is nonfinite")
                        accepted = norm <= 1
                        why = "accepted" if accepted else "local_error_rejected"
                        if accepted:
                            selected = point(end, candidate, endpoint)
            except _StartupNewtonRejected as exc:
                accepted = False
                why = "startup_newton_" + auxiliary[-1].termination.value
                message = f"ArithmeticError: {exc}"
            except Exception as exc:
                accepted = False
                why = "wall_budget_exhausted" if isinstance(exc, TimeoutError) else "evaluation_error"
                message = f"{type(exc).__name__}: {exc}"
            attempts.append(BDFAttempt(t, end, order, step.coefficients, accepted, why, frozen_array(guess),
                frozen_array(candidate), norm, None if estimate is None else frozen_array(estimate), estimator,
                newton, tuple(auxiliary), assessment, time.monotonic() - attempt_start, message))
            if accepted:
                assert selected is not None
                new = selected
                dense.append((t, end, (new, *tuple(history[:order])), order))
                history.insert(0, new)
                history = history[:4]
                all_points.append(new)
                t = end
                accepted_times.append(t)
                accepted_u.append(new.u)
                consecutive = 0
                next_order = options.max_order if len(history) >= (3 if options.adaptive else 2) else 1
                if t in event_map and not restart(event_map[t]):
                    termination = "restart_physical_gate_rejected"
                    break
                if options.adaptive:
                    factor = options.max_factor if norm == 0 else options.safety * norm**(-1 / (order + 1))
                    h *= min(options.max_factor, max(options.min_factor, factor))
                else:
                    h = options.initial_step_s
            else:
                if why in ("physical_gate_rejected", "evaluation_error", "wall_budget_exhausted") or not options.adaptive:
                    termination, error = why, message
                    break
                consecutive += 1
                if consecutive > options.max_consecutive_rejections:
                    termination = "rejection_budget_exhausted"
                    break
                factor = .5 if norm <= 1 else options.safety * norm**(-1 / (order + 1))
                h *= max(options.min_factor, min(.8, factor))
                next_order = 1 if order == 2 else order
    except Exception as exc:
        termination = "wall_budget_exhausted" if isinstance(exc, TimeoutError) else "evaluation_error"
        error = f"{type(exc).__name__}: {exc}"

    samples: list[BDFSample] = []
    sample_failure: dict[str, Any] | None = None
    projection: NewtonResult | None = None
    if t == finish and termination == "completed":
        try:
            for target in output:
                projection = None
                exact = next((p for p in reversed(all_points) if p.time_s == target), None)
                if exact is not None:
                    samples.append(BDFSample(float(target), exact.u, exact.storage_value, 0, True))
                    continue
                left, right, nodes, order = next(item for item in dense if item[0] < target < item[1])
                problem = factory(float(target), "right")
                anchor = nodes[0]
                target_delta = _prediction_change(nodes, float(target))
                weights = lagrange_weights(np.array([p.time_s for p in nodes]), float(target))
                guess = sum((weight * p.u for weight, p in zip(weights, nodes)), np.zeros_like(u0))
                tangent = problem.storage_tangent(guess)
                tangent_values = tangent.data if issparse(tangent) else np.asarray(tangent)
                if tangent.shape != (len(u0), len(u0)) or not np.all(np.isfinite(tangent_values)):
                    raise ValueError("sample physical storage tangent is nonfinite or has the wrong shape")
                mask = (np.asarray(abs(tangent).sum(axis=1)).ravel() > 0 if issparse(tangent)
                        else np.any(np.asarray(tangent) != 0, axis=1))
                h_sample = right - left
                def sample_residual(value: Vector) -> Vector:
                    checkpoint()
                    counts["residual"] += 1
                    counts["sample_residual"] += 1
                    base = problem.residual(value)
                    delta = storage_increment(problem, value, anchor)
                    return np.where(mask, (target_delta - delta) / h_sample, base)
                def sample_jacobian(value: Vector) -> Matrix:
                    checkpoint()
                    counts["jacobian"] += 1
                    counts["sample_jacobian"] += 1
                    return (diags((~mask).astype(np.float64)) @ csc_matrix(problem.jacobian(value))
                            - diags(mask.astype(np.float64)) @ csc_matrix(problem.storage_tangent(value)) / h_sample)
                control = settings(BDFStep(problem, (anchor,), float(target) + h_sample, 1))
                nonlinear_calls += 1
                counts["sample_projection"] += 1
                projection = solve_newton(CallableProblem(sample_residual, sample_jacobian), guess,
                                           control, convergence_norm=convergence_norm)
                if not projection.accepted:
                    raise ArithmeticError("storage-interpolant sample projection failed: " + projection.termination.value)
                sampled = point(float(target), projection.state, problem)
                samples.append(BDFSample(float(target), sampled.u, sampled.storage_value, order, False, projection))
        except Exception as exc:
            termination, error = "sample_projection_failed", f"{type(exc).__name__}: {exc}"
            sample_failure = {"time_s": float(target), "newton": projection, "error": error,
                              "accepted_sample_prefix": len(samples)}
    return BDFResult(start, u0, finish, frozen_array(accepted_times), tuple(accepted_u), tuple(attempts),
        tuple(samples), tuple(restarts), asdict(options), t == finish and termination == "completed",
        termination, time.monotonic() - started, nonlinear_calls, counts, error,
        requested_output_times_s=output, sample_failure=sample_failure)
