"""Solver-facing steady-state problem interface and its controls (N class).

The nonlinear layer sees only a residual ``F(u)`` and its Jacobian ``dF/du`` in
the caller's coordinates ``u``. It never chooses coordinates, norms or
scalings itself: the row weights of the scaled residual, the step limits and
feasible bounds, the budgets and the physical acceptance gates are explicit
arguments without defaults. The production residual, linearization, scaling
and coordinates are G2-R/P04-B interfaces
(:mod:`solarlab.solvers.nonlinear.production`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import math
from typing import Any, Protocol, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "CallableProblem", "GateOutcome", "Matrix", "PhysicalGate", "ResidualScaling",
    "SolveBudget", "SteadyProblem", "StepControl", "Vector", "state_digest",
]

Vector: TypeAlias = NDArray[np.float64]
Matrix: TypeAlias = Any  # dense ndarray or scipy.sparse matrix


class SteadyProblem(Protocol):
    """A square nonlinear system ``F(u) = 0`` with its Jacobian."""

    def residual(self, u: Vector) -> Vector: ...

    def jacobian(self, u: Vector) -> Matrix: ...


@dataclass(frozen=True)
class CallableProblem:
    """Adapter from two callables to :class:`SteadyProblem`."""

    residual_function: Callable[[Vector], ArrayLike]
    jacobian_function: Callable[[Vector], Matrix]

    def residual(self, u: Vector) -> Vector:
        return np.asarray(self.residual_function(u), dtype=float)

    def jacobian(self, u: Vector) -> Matrix:
        return self.jacobian_function(u)


def _positive_vector(value: ArrayLike, name: str, allow_inf: bool = False) -> Vector:
    array = np.array(value, dtype=float, copy=True)
    finite = np.isfinite(array) | (allow_inf & np.isposinf(array))
    if array.ndim != 1 or array.size == 0 or not np.all(finite) or np.any(array <= 0.0):
        raise ValueError(f"{name} must be a non-empty 1-D array of positive values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class ResidualScaling:
    """Row weights ``w``: the scaled residual is ``w * F`` (no default exists)."""

    row_weights: Vector

    def __post_init__(self) -> None:
        object.__setattr__(self, "row_weights", _positive_vector(self.row_weights,
                                                                 "row_weights"))

    def scaled(self, residual: Vector) -> Vector:
        if residual.shape != self.row_weights.shape:
            raise ValueError("residual and row weights have different shapes")
        result: Vector = self.row_weights * residual
        return result


@dataclass(frozen=True)
class StepControl:
    """Largest allowed ``|du_i|`` per iteration and optional feasible bounds.

    A Newton direction longer than ``max_step`` is shortened as a whole, so its
    direction is kept. With bounds, the step is also shortened so the trial
    point keeps ``fraction_to_boundary`` of the distance to every bound.
    """

    max_step: Vector
    lower: Vector | None = None
    upper: Vector | None = None
    fraction_to_boundary: float = 0.99

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_step", _positive_vector(self.max_step, "max_step",
                                                              allow_inf=True))
        for name in ("lower", "upper"):
            bound = getattr(self, name)
            if bound is not None:
                array = np.array(bound, dtype=float, copy=True)
                if array.shape != self.max_step.shape or np.any(np.isnan(array)):
                    raise ValueError(f"{name} must match max_step and not be NaN")
                array.setflags(write=False)
                object.__setattr__(self, name, array)
        if self.lower is not None and self.upper is not None and np.any(self.lower >= self.upper):
            raise ValueError("lower bounds must lie below upper bounds")
        if not 0.0 < self.fraction_to_boundary < 1.0:
            raise ValueError("fraction_to_boundary must lie in (0, 1)")

    def feasible(self, u: Vector) -> bool:
        if self.lower is not None and np.any(u <= self.lower):
            return False
        if self.upper is not None and np.any(u >= self.upper):
            return False
        return True

    def step_fraction(self, u: Vector, step: Vector) -> float:
        """Largest ``alpha <= 1`` keeping the step bounded and the trial feasible."""
        magnitude = np.abs(step)
        with np.errstate(divide="ignore"):
            limits = np.where(magnitude > self.max_step, self.max_step / magnitude, 1.0)
        alpha = float(np.min(limits))
        tau = self.fraction_to_boundary
        if self.lower is not None:
            falling = step < 0.0
            if np.any(falling):
                alpha = min(alpha, float(np.min(tau * (self.lower[falling] - u[falling])
                                                / step[falling])))
        if self.upper is not None:
            rising = step > 0.0
            if np.any(rising):
                alpha = min(alpha, float(np.min(tau * (self.upper[rising] - u[rising])
                                                / step[rising])))
        return min(alpha, 1.0)


@dataclass(frozen=True)
class SolveBudget:
    """Hard limits of one solve; exhausting any of them is a reported failure."""

    max_iterations: int
    max_residual_evaluations: int
    max_jacobian_evaluations: int
    max_backtracks: int

    def __post_init__(self) -> None:
        for name in ("max_iterations", "max_residual_evaluations",
                     "max_jacobian_evaluations"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.max_backtracks) is not int or self.max_backtracks < 0:
            raise ValueError("max_backtracks must be a non-negative integer")


@dataclass(frozen=True)
class GateOutcome:
    """One physical acceptance gate evaluated on a candidate solution."""

    name: str
    passed: bool
    value: float
    limit: float
    message: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a gate needs a name")
        if self.passed and not (math.isfinite(self.value) and self.value <= self.limit):
            raise ValueError(f"gate {self.name!r} cannot pass with value above its limit")


PhysicalGate: TypeAlias = Callable[[Vector], GateOutcome]


def state_digest(u: ArrayLike) -> str:
    """sha256 of the binary64 bytes of a state (start-state traceability)."""
    array = np.ascontiguousarray(np.asarray(u, dtype=np.float64))
    return hashlib.sha256(array.tobytes()).hexdigest()

