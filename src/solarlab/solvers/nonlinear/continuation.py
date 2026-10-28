"""Bounded parameter continuation for staged preparation and bias sweeps (N class).

``solve_continuation`` marches one scalar parameter (bias, generation scale,
contact homotopy, ...) from an accepted start to each requested target. A
failed attempt halves the step; the number of consecutive cuts, the smallest
step and the total number of attempts are bounded, and running out of any of
them is a reported failure -- no further fallback is tried.

Every attempt starts from a guess built only from accepted states (the last
accepted state, or a secant through the last two), so a failed or rejected
attempt can never seed the next one. Each attempt records its parameters, the
digest of the accepted state it was built from, the digest of its actual start,
its predictor, its termination and its cost.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math

import numpy as np

from solarlab.solvers.nonlinear.newton import NewtonResult, Termination
from solarlab.solvers.nonlinear.problem import SteadyProblem, Vector, state_digest

__all__ = [
    "AttemptRecord", "ContinuationPoint", "ContinuationResult", "ContinuationSettings",
    "PREVIOUS", "SECANT", "solve_continuation",
]

PREVIOUS = "previous"
SECANT = "secant"
INITIAL = "initial"


@dataclass(frozen=True)
class ContinuationSettings:
    max_step: float
    min_step: float
    max_consecutive_cuts: int
    max_attempts: int
    predictor: str
    growth: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.max_step) and math.isfinite(self.min_step)
                and 0.0 < self.min_step <= self.max_step):
            raise ValueError("steps must satisfy 0 < min_step <= max_step")
        if type(self.max_consecutive_cuts) is not int or self.max_consecutive_cuts < 0:
            raise ValueError("max_consecutive_cuts must be a non-negative integer")
        if type(self.max_attempts) is not int or self.max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        if self.predictor not in (PREVIOUS, SECANT):
            raise ValueError("predictor must be 'previous' or 'secant'")
        if not (math.isfinite(self.growth) and self.growth >= 1.0):
            raise ValueError("growth must be finite and at least 1")


@dataclass(frozen=True)
class AttemptRecord:
    index: int
    start_parameter: float
    target_parameter: float
    base_digest: str
    start_digest: str
    predictor: str
    termination: Termination
    accepted: bool
    iterations: int
    residual_evaluations: int
    jacobian_evaluations: int
    scaled_residual: float
    message: str


@dataclass(frozen=True)
class ContinuationPoint:
    parameter: float
    state: Vector
    result: NewtonResult
    requested: bool


@dataclass(frozen=True)
class ContinuationResult:
    completed: bool
    points: tuple[ContinuationPoint, ...]
    attempts: tuple[AttemptRecord, ...]
    failure: str | None

    def requested_points(self) -> tuple[ContinuationPoint, ...]:
        return tuple(point for point in self.points if point.requested)


def solve_continuation(
    problem_at: Callable[[float], SteadyProblem],
    initial_guess: Vector,
    start_parameter: float,
    targets: Sequence[float],
    newton: Callable[[SteadyProblem, Vector], NewtonResult],
    settings: ContinuationSettings,
) -> ContinuationResult:
    """Accept ``start_parameter``, then march to every target in order."""
    goals = [float(value) for value in targets]
    start = float(start_parameter)
    if not goals or not all(math.isfinite(value) for value in [start, *goals]):
        raise ValueError("continuation needs finite start and target parameters")
    direction = math.copysign(1.0, goals[0] - start) if goals[0] != start else 1.0
    path = [start, *goals]
    if any(direction * (right - left) < 0.0 for left, right in zip(path[:-1], path[1:])) or \
            any(right == left for left, right in zip(path[1:-1], path[2:])):
        raise ValueError("targets must move monotonically away from the start")
    attempts: list[AttemptRecord] = []
    points: list[ContinuationPoint] = []

    def attempt(parameter: float, base: Vector | None, guess: Vector, predictor: str,
                origin: float) -> NewtonResult:
        result = newton(problem_at(parameter), guess.copy())
        attempts.append(AttemptRecord(
            len(attempts), origin, parameter,
            state_digest(base) if base is not None else "", result.start_digest, predictor,
            result.termination, result.accepted, result.iteration_count,
            result.residual_evaluations, result.jacobian_evaluations,
            result.scaled_residual, result.message))
        return result

    def failed(message: str) -> ContinuationResult:
        return ContinuationResult(False, tuple(points), tuple(attempts), message)

    first = attempt(start, None, np.asarray(initial_guess, dtype=float), INITIAL, start)
    if not first.accepted:
        return failed(f"start parameter {start:g} not accepted: {first.termination.value}: "
                      f"{first.message}")
    points.append(ContinuationPoint(start, first.state, first, True))
    accepted_parameter, accepted_state = start, first.state
    previous: tuple[float, Vector] | None = None
    step = settings.max_step
    for goal in goals:
        cuts = 0
        while accepted_parameter != goal:
            if len(attempts) >= settings.max_attempts:
                return failed(f"attempt budget {settings.max_attempts} exhausted before "
                              f"{goal:g} (last accepted {accepted_parameter:g})")
            remaining = abs(goal - accepted_parameter)
            parameter = goal if remaining <= step else accepted_parameter + direction * step
            guess = accepted_state
            predictor = PREVIOUS
            if settings.predictor == SECANT and previous is not None:
                old_parameter, old_state = previous
                ratio = (parameter - accepted_parameter) / (accepted_parameter - old_parameter)
                guess = accepted_state + ratio * (accepted_state - old_state)
                predictor = SECANT
            result = attempt(parameter, accepted_state, guess, predictor, accepted_parameter)
            if result.accepted:
                points.append(ContinuationPoint(parameter, result.state, result,
                                                parameter == goal))
                previous = (accepted_parameter, accepted_state)
                accepted_parameter, accepted_state = parameter, result.state
                cuts = 0
                step = min(settings.max_step, step * settings.growth)
                continue
            cuts += 1
            step *= 0.5
            if cuts > settings.max_consecutive_cuts or step < settings.min_step:
                return failed(
                    f"continuation stopped at {accepted_parameter:g} toward {goal:g}: "
                    f"{cuts} consecutive cuts, step {step:.3g}; last attempt "
                    f"{result.termination.value}: {result.message}")
    return ContinuationResult(True, tuple(points), tuple(attempts), None)
