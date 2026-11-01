"""Unified damped Newton for ``F(u) = 0`` in caller coordinates (N class).

Each iteration solves ``J du = -F``, shortens the whole step to the caller's
step limits and feasible bounds (the direction is kept), and backtracks on the
merit ``phi = |w F|_2^2 / 2`` with the Armijo condition
``phi(a) <= (1 - 2 c a) phi(0)``; a non-finite trial is rejected like any
other. A reused (stale) factorization that fails the line search, or needs
deep backtracking, is rebuilt before the iteration is declared failed; a
reused factor that does not halve the scaled residual is rebuilt next time.

A state is **accepted** only when the convergence norm (``max |w F|`` by
default) is within the tolerance *and* every physical gate passes. An explicit
caller callback may measure the scaled rows; it does not change the merit or
the row-max factor refresh rule. A converged residual whose
gates fail is a rejected (wrong-branch) state, never a solution. Every exit
names its reason, and all work is bounded by a :class:`SolveBudget`; there is
no hidden fallback. Steps that no longer lower the merit end the solve as
stagnated rather than consuming the budget.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
import math
from typing import Any
import warnings

import numpy as np
from scipy.linalg import LinAlgWarning, lu_factor, lu_solve
from scipy.sparse import diags, issparse
from scipy.sparse.linalg import splu

from solarlab.solvers.nonlinear.problem import (
    GateOutcome, Matrix, PhysicalGate, ResidualScaling, SolveBudget, SteadyProblem,
    StepControl, Vector, state_digest,
)

__all__ = [
    "IterationRecord", "LinearSolveError", "NewtonResult", "NewtonSettings", "Termination",
    "factorize", "gate_from", "solve_newton",
]

_ARMIJO = 1.0e-4
# A reused factor must at least halve the scaled residual, or it is rebuilt.
_CHORD_CONTRACTION = 0.5
# Accepted steps that lower the merit by less than this fraction, this many
# times in a row, end the solve as stagnated instead of draining the budget.
_STAGNATION_DECREASE = 1.0e-10
_STAGNATION_WINDOW = 3


class Termination(str, Enum):
    CONVERGED = "converged"
    GATE_REJECTED = "gate_rejected"
    LINE_SEARCH_FAILED = "line_search_failed"
    SINGULAR_JACOBIAN = "singular_jacobian"
    NONFINITE_RESIDUAL = "nonfinite_residual"
    BUDGET_EXHAUSTED = "budget_exhausted"
    STAGNATED = "stagnated"
    INFEASIBLE_START = "infeasible_start"


class LinearSolveError(ArithmeticError):
    """The Jacobian cannot be factorized or its solve is not finite."""


@dataclass(frozen=True)
class NewtonSettings:
    """All controls of one Newton solve; none of them has a default value."""

    scaling: ResidualScaling
    tolerance: float
    step_control: StepControl
    budget: SolveBudget
    gates: tuple[PhysicalGate, ...]
    reuse_jacobian: bool

    def __post_init__(self) -> None:
        if not math.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive")
        if self.scaling.row_weights.shape != self.step_control.max_step.shape:
            raise ValueError("residual weights and step limits must have one size")
        object.__setattr__(self, "gates", tuple(self.gates))


@dataclass(frozen=True)
class IterationRecord:
    iteration: int
    scaled_residual: float
    step_inf: float
    damping: float
    backtracks: int
    fresh_jacobian: bool


@dataclass(frozen=True)
class NewtonResult:
    """Solve evidence; scaled_residual reports the selected convergence norm."""

    state: Vector
    accepted: bool
    termination: Termination
    scaled_residual: float
    gates: tuple[GateOutcome, ...]
    iterations: tuple[IterationRecord, ...]
    residual_evaluations: int
    jacobian_evaluations: int
    start_digest: str
    message: str

    @property
    def iteration_count(self) -> int:
        return len(self.iterations)


class _Factor:
    """LU of the row- and column-equilibrated Jacobian.

    Equilibration (max-norm rows, then columns) changes only the rounding of
    the linear solve, not the Newton direction: ``J du = -F`` is solved as
    ``(R J C) v = -R F``, ``du = C v``. It keeps partial pivoting meaningful
    when rows and columns span many decades (e.g. log-density coordinates).
    """

    def __init__(self, matrix: Matrix):
        self.sparse = bool(issparse(matrix))
        if self.sparse:
            csr = matrix.tocsr().astype(float)
            if csr.shape[0] != csr.shape[1] or not np.all(np.isfinite(csr.data)):
                raise LinearSolveError("the Jacobian must be square and finite")
            row_max = np.asarray(abs(csr).max(axis=1).todense()).ravel()
            if np.any(row_max == 0.0):
                raise LinearSolveError("the Jacobian has an all-zero row")
            self._rows = 1.0 / row_max
            scaled = diags(self._rows) @ csr
            column_max = np.asarray(abs(scaled).max(axis=0).todense()).ravel()
            if np.any(column_max == 0.0):
                raise LinearSolveError("the Jacobian has an all-zero column")
            self._columns = 1.0 / column_max
            try:
                self._lu: Any = splu((scaled @ diags(self._columns)).tocsc())
            except RuntimeError as error:
                raise LinearSolveError(str(error)) from error
        else:
            dense: Vector = np.asarray(matrix, dtype=float)
            if dense.ndim != 2 or dense.shape[0] != dense.shape[1]:
                raise LinearSolveError("the Jacobian must be square")
            if not np.all(np.isfinite(dense)):
                raise LinearSolveError("the Jacobian is not finite")
            row_max = np.max(np.abs(dense), axis=1)
            if np.any(row_max == 0.0):
                raise LinearSolveError("the Jacobian has an all-zero row")
            self._rows = 1.0 / row_max
            scaled = dense * self._rows[:, None]
            column_max = np.max(np.abs(scaled), axis=0)
            if np.any(column_max == 0.0):
                raise LinearSolveError("the Jacobian has an all-zero column")
            self._columns = 1.0 / column_max
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", LinAlgWarning)
                self._lu = lu_factor(scaled * self._columns[None, :], check_finite=False)
            pivots = np.abs(np.diag(self._lu[0]))
            if np.any(pivots == 0.0) or not np.all(np.isfinite(pivots)):
                raise LinearSolveError("the Jacobian is exactly singular")

    def solve(self, rhs: Vector) -> Vector:
        scaled_rhs = self._rows * np.asarray(rhs, dtype=float)
        with np.errstate(all="ignore"):
            result = (self._lu.solve(scaled_rhs) if self.sparse
                      else lu_solve(self._lu, scaled_rhs, check_finite=False))
        solution: Vector = self._columns * np.asarray(result, dtype=float)
        if not np.all(np.isfinite(solution)):
            raise LinearSolveError("the Newton step is not finite")
        return solution


def factorize(matrix: Matrix) -> _Factor:
    """LU factorization of a dense or sparse Jacobian (raises LinearSolveError)."""
    return _Factor(matrix)


def solve_newton(
    problem: SteadyProblem, u0: Vector, settings: NewtonSettings, *,
    convergence_norm: Callable[[Vector], float] | None = None,
) -> NewtonResult:
    """Damped Newton with explicit acceptance; see the module docstring."""
    if convergence_norm is not None and not callable(convergence_norm):
        raise ValueError("convergence_norm must be callable or None")
    budget = settings.budget
    u = np.array(u0, dtype=float, copy=True)
    start = state_digest(u)
    records: list[IterationRecord] = []
    counts = {"residual": 0, "jacobian": 0}
    gates: tuple[GateOutcome, ...] = ()

    def finish(termination: Termination, residual_norm: float, message: str,
               accepted: bool = False) -> NewtonResult:
        state = u.copy()
        state.setflags(write=False)
        return NewtonResult(state, accepted, termination, residual_norm, gates,
                            tuple(records), counts["residual"], counts["jacobian"], start,
                            message)

    if u.shape != settings.scaling.row_weights.shape or not np.all(np.isfinite(u)):
        return finish(Termination.INFEASIBLE_START, math.inf,
                      "the start state is not finite or has the wrong size")
    if not settings.step_control.feasible(u):
        return finish(Termination.INFEASIBLE_START, math.inf,
                      "the start state violates the feasible bounds")

    def evaluate(point: Vector) -> Vector | None:
        counts["residual"] += 1
        with np.errstate(all="ignore"):
            values = np.asarray(problem.residual(point), dtype=float)
        if values.shape != u.shape:
            raise ValueError("the residual has the wrong size")
        return values if np.all(np.isfinite(values)) else None

    def measure_norm(values: Vector) -> float:
        if convergence_norm is None:
            return float(np.max(np.abs(values)))
        value = float(convergence_norm(values.copy()))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("convergence_norm must return a finite nonnegative scalar")
        return value

    F = evaluate(u)
    if F is None:
        return finish(Termination.NONFINITE_RESIDUAL, math.inf,
                      "the residual is not finite at the start state")
    scaled = settings.scaling.scaled(F)
    norm = measure_norm(scaled)
    factor: _Factor | None = None
    fresh = False
    iteration = 0
    stagnant = 0
    while True:
        if norm <= settings.tolerance:
            gates = tuple(gate(u.copy()) for gate in settings.gates)
            failed = [gate for gate in gates if not gate.passed]
            if failed:
                return finish(Termination.GATE_REJECTED, norm,
                              "residual converged but gates failed: "
                              + "; ".join(f"{gate.name} {gate.value:.6g} > {gate.limit:.6g}"
                                          f" {gate.message}".rstrip() for gate in failed))
            return finish(Termination.CONVERGED, norm, "converged", accepted=True)
        if iteration >= budget.max_iterations:
            return finish(Termination.BUDGET_EXHAUSTED, norm,
                          f"iteration budget {budget.max_iterations} exhausted at "
                          f"scaled residual {norm:.6g}")
        iteration += 1
        if factor is None or not settings.reuse_jacobian:
            if counts["jacobian"] >= budget.max_jacobian_evaluations:
                return finish(Termination.BUDGET_EXHAUSTED, norm,
                              f"Jacobian budget {budget.max_jacobian_evaluations} exhausted")
            counts["jacobian"] += 1
            try:
                factor = _Factor(problem.jacobian(u.copy()))
            except LinearSolveError as error:
                return finish(Termination.SINGULAR_JACOBIAN, norm, str(error))
            fresh = True
        try:
            step = factor.solve(-F)
        except LinearSolveError as error:
            if not fresh:
                factor = None
                continue
            return finish(Termination.SINGULAR_JACOBIAN, norm, str(error))
        step_inf = float(np.max(np.abs(step)))
        alpha = settings.step_control.step_fraction(u, step)
        merit = 0.5 * float(np.dot(scaled, scaled))
        accepted_trial: tuple[Vector, Vector, Vector] | None = None
        backtracks = 0
        while backtracks <= budget.max_backtracks:
            if counts["residual"] >= budget.max_residual_evaluations:
                return finish(Termination.BUDGET_EXHAUSTED, norm,
                              "residual-evaluation budget "
                              f"{budget.max_residual_evaluations} exhausted")
            trial = u + alpha * step
            values = evaluate(trial) if settings.step_control.feasible(trial) else None
            if values is not None:
                trial_scaled = settings.scaling.scaled(values)
                trial_merit = 0.5 * float(np.dot(trial_scaled, trial_scaled))
                if trial_merit <= (1.0 - 2.0 * _ARMIJO * alpha) * merit:
                    accepted_trial = (trial, values, trial_scaled)
                    stagnant = (stagnant + 1
                                if trial_merit > (1.0 - _STAGNATION_DECREASE) * merit else 0)
                    break
            alpha *= 0.5
            backtracks += 1
        if accepted_trial is None:
            if not fresh:      # a stale factor failed: rebuild before giving up
                factor = None
                continue
            return finish(Termination.LINE_SEARCH_FAILED, norm,
                          f"line search failed after {budget.max_backtracks} backtracks at "
                          f"scaled residual {norm:.6g}")
        previous_row_norm = float(np.max(np.abs(scaled)))
        u, F, scaled = accepted_trial
        norm = measure_norm(scaled)
        records.append(IterationRecord(iteration, norm, step_inf, alpha, backtracks, fresh))
        # Refresh after any backtracking, and when a reused factor contracts poorly.
        if backtracks > 0 or (
            not fresh and float(np.max(np.abs(scaled))) > _CHORD_CONTRACTION * previous_row_norm
        ):
            factor = None
        fresh = False
        if stagnant >= _STAGNATION_WINDOW and norm > settings.tolerance:
            return finish(Termination.STAGNATED, norm,
                          f"no merit progress in {_STAGNATION_WINDOW} consecutive iterations "
                          f"at scaled residual {norm:.6g} (last damping {alpha:.3g})")


def gate_from(name: str, measure: Callable[[Vector], float], limit: float,
              message: str = "") -> PhysicalGate:
    """Gate passing when ``measure(u) <= limit`` (non-finite values fail)."""
    def gate(u: Vector) -> GateOutcome:
        value = float(measure(u))
        return GateOutcome(name, bool(math.isfinite(value) and value <= limit), value, limit,
                           message)
    return gate
