"""Bounded pseudo-transient continuation toward a steady state (N class).

Each pseudo-time step solves ``(diag(m) / dtau + J) du = -F`` with a positive
caller-given pseudo mass ``m``, shortens the step to the caller's step limits
and feasible bounds, and adapts ``dtau`` by switched evolution relaxation
(``dtau *= |wF|_old / |wF|_new``, clipped to ``[min_dtau, max_dtau]``); a
non-finite or infeasible trial is rejected and ``dtau`` is quartered.

PTC is only a numerical path into the Newton basin. When the scaled residual
reaches ``handoff_tolerance`` the steady Newton (with its gates) solves the
unmodified system; only its accepted result is a steady state. The recorded
pseudo-time states are numerical, never a physical preparation history
(``physical_history`` is always ``False``). Step count and ``dtau`` range are
bounded; running out is a reported failure.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

import numpy as np
from scipy.sparse import diags, issparse

from solarlab.solvers.nonlinear.newton import LinearSolveError, NewtonResult, factorize
from solarlab.solvers.nonlinear.problem import (
    ResidualScaling, SteadyProblem, StepControl, Vector, state_digest,
)

__all__ = ["PtcResult", "PtcSettings", "PtcStepRecord", "solve_ptc"]


@dataclass(frozen=True)
class PtcSettings:
    pseudo_mass: Vector
    initial_dtau: float
    min_dtau: float
    max_dtau: float
    max_steps: int
    handoff_tolerance: float
    scaling: ResidualScaling
    step_control: StepControl

    def __post_init__(self) -> None:
        mass = np.array(self.pseudo_mass, dtype=float, copy=True)
        if mass.shape != self.scaling.row_weights.shape or not np.all(np.isfinite(mass)) \
                or np.any(mass <= 0.0):
            raise ValueError("pseudo_mass must be positive and match the system size")
        mass.setflags(write=False)
        object.__setattr__(self, "pseudo_mass", mass)
        if not (0.0 < self.min_dtau <= self.initial_dtau <= self.max_dtau < math.inf):
            raise ValueError("dtau must satisfy 0 < min <= initial <= max < inf")
        if type(self.max_steps) is not int or self.max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if not math.isfinite(self.handoff_tolerance) or self.handoff_tolerance <= 0.0:
            raise ValueError("handoff_tolerance must be finite and positive")


@dataclass(frozen=True)
class PtcStepRecord:
    step: int
    dtau: float
    scaled_residual: float
    accepted: bool


@dataclass(frozen=True)
class PtcResult:
    accepted: bool
    termination: str
    newton: NewtonResult | None
    pseudo_history: tuple[PtcStepRecord, ...]
    start_digest: str
    message: str
    physical_history: bool = False

    def __post_init__(self) -> None:
        if self.physical_history:
            raise ValueError("pseudo-time states are never a physical preparation history")

    @property
    def state(self) -> Vector | None:
        return self.newton.state if self.accepted and self.newton is not None else None


def solve_ptc(problem: SteadyProblem, u0: Vector, settings: PtcSettings,
              newton: Callable[[SteadyProblem, Vector], NewtonResult]) -> PtcResult:
    """Pseudo-time march to the Newton basin, then a steady Newton acceptance."""
    u = np.array(u0, dtype=float, copy=True)
    start = state_digest(u)
    history: list[PtcStepRecord] = []
    control = settings.step_control

    def residual(point: Vector) -> Vector | None:
        with np.errstate(all="ignore"):
            values = np.asarray(problem.residual(point), dtype=float)
        return values if np.all(np.isfinite(values)) else None

    def failure(termination: str, message: str) -> PtcResult:
        return PtcResult(False, termination, None, tuple(history), start, message)

    if u.shape != settings.pseudo_mass.shape or not np.all(np.isfinite(u)) \
            or not control.feasible(u):
        return failure("infeasible_start", "the PTC start state is not admissible")
    F = residual(u)
    if F is None:
        return failure("nonfinite_residual", "the residual is not finite at the PTC start")
    norm = float(np.linalg.norm(settings.scaling.scaled(F)))
    dtau = settings.initial_dtau
    for step in range(1, settings.max_steps + 1):
        if float(np.max(np.abs(settings.scaling.scaled(F)))) <= settings.handoff_tolerance:
            break
        jacobian = problem.jacobian(u.copy())
        shift = settings.pseudo_mass / dtau
        if issparse(jacobian):
            matrix = jacobian + diags(shift)
        else:
            matrix = np.asarray(jacobian, dtype=float) + np.diag(shift)
        try:
            direction = factorize(matrix).solve(-F)
        except LinearSolveError as error:
            return failure("singular_jacobian", f"PTC step {step}: {error}")
        alpha = control.step_fraction(u, direction)
        trial = u + alpha * direction
        trial_residual = residual(trial) if control.feasible(trial) else None
        if trial_residual is None:
            history.append(PtcStepRecord(step, dtau, math.nan, False))
            dtau *= 0.25
            if dtau < settings.min_dtau:
                return failure("stagnated", f"PTC dtau fell below {settings.min_dtau:g}")
            continue
        new_norm = float(np.linalg.norm(settings.scaling.scaled(trial_residual)))
        u, F = trial, trial_residual
        history.append(PtcStepRecord(step, dtau, float(np.max(np.abs(
            settings.scaling.scaled(F)))), True))
        ratio = norm / new_norm if new_norm > 0.0 else settings.max_dtau / dtau
        dtau = min(settings.max_dtau, max(settings.min_dtau, dtau * ratio))
        norm = new_norm
    else:
        if float(np.max(np.abs(settings.scaling.scaled(F)))) > settings.handoff_tolerance:
            return failure("budget_exhausted",
                           f"PTC step budget {settings.max_steps} exhausted at scaled "
                           f"residual {np.max(np.abs(settings.scaling.scaled(F))):.6g}")
    final = newton(problem, u.copy())
    if not final.accepted:
        return PtcResult(False, "handoff_rejected", final, tuple(history), start,
                         f"steady Newton after PTC: {final.termination.value}: "
                         f"{final.message}")
    return PtcResult(True, "converged", final, tuple(history), start, "converged")
