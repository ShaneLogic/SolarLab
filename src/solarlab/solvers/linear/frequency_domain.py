"""Bounded complex sparse solves of (J + i*omega*K) x = b.

The caller supplies the actual residual and storage Jacobians. Row/column
equilibration and bounded refinement change no equation or right-hand side.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import splu

__all__ = ["FrequencyOptions", "FrequencySolution", "FrequencySolveError", "solve_frequency_systems"]


@dataclass(frozen=True)
class FrequencyOptions:
    max_frequencies: int = 32
    relative_tolerance: float = 1e-11
    max_refinements: int = 2

    def __post_init__(self) -> None:
        if type(self.max_frequencies) is not int or self.max_frequencies < 1:
            raise ValueError("max_frequencies must be a positive integer")
        if not math.isfinite(self.relative_tolerance) or self.relative_tolerance <= 0.0:
            raise ValueError("relative_tolerance must be finite and positive")
        if type(self.max_refinements) is not int or self.max_refinements < 0:
            raise ValueError("max_refinements must be a nonnegative integer")


@dataclass(frozen=True)
class FrequencySolution:
    frequency_Hz: float
    solution: np.ndarray
    residual: np.ndarray
    componentwise_backward_error: float
    refinement_steps: int

    def __post_init__(self) -> None:
        for name in ("solution", "residual"):
            value = np.array(getattr(self, name), dtype=complex, copy=True)
            if value.ndim != 1 or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite complex vector")
            value.setflags(write=False)
            object.__setattr__(self, name, value)


class FrequencySolveError(ArithmeticError):
    """A complex solve failed; no alternative equation/solver is attempted."""


def solve_frequency_systems(
    jacobian: csc_matrix,
    storage_jacobian: csc_matrix,
    rhs: ArrayLike,
    frequencies_Hz: ArrayLike,
    *,
    options: FrequencyOptions | None = None,
    fixed_coordinates: dict[int, complex] | None = None,
    rhs_rate: ArrayLike | None = None,
) -> tuple[FrequencySolution, ...]:
    """exp(+i*omega*t) systems with rhs+i*omega*rhs_rate.

    Prescribed coordinates eliminate an invertible algebraic row block;
    every result is checked against all original rows, not just the reduction.
    """
    controls = FrequencyOptions() if options is None else options
    supplied = np.asarray(frequencies_Hz)
    if np.iscomplexobj(supplied):
        raise ValueError("frequencies must be real Hz")
    frequencies = np.asarray(supplied, dtype=float)
    if (
        frequencies.ndim != 1 or frequencies.size == 0
        or frequencies.size > controls.max_frequencies
        or not np.all(np.isfinite(frequencies)) or np.any(frequencies < 0.0)
        or np.unique(frequencies).size != frequencies.size
    ):
        raise ValueError("frequencies must be a bounded nonempty list of distinct finite nonnegative Hz")
    J = csc_matrix(jacobian, dtype=complex)
    K = csc_matrix(storage_jacobian, dtype=complex)
    b_static = np.asarray(rhs, dtype=complex)
    b_rate = np.zeros_like(b_static) if rhs_rate is None else np.asarray(rhs_rate, dtype=complex)
    if (
        J.shape[0] == 0 or J.shape[0] != J.shape[1] or K.shape != J.shape
        or b_static.shape != (J.shape[0],) or b_rate.shape != b_static.shape
        or not np.all(np.isfinite(J.data)) or not np.all(np.isfinite(K.data))
        or not np.all(np.isfinite(b_static)) or not np.all(np.isfinite(b_rate))
    ):
        raise ValueError("finite square Jacobians and a matching finite complex rhs are required")
    known = {} if fixed_coordinates is None else dict(fixed_coordinates)
    if any(
        type(index) is not int or not 0 <= index < J.shape[0] or not np.isfinite(value)
        for index, value in known.items()
    ):
        raise ValueError("fixed coordinates must be finite and index the system")
    fixed: np.ndarray = np.asarray(sorted(known), dtype=int)
    free: np.ndarray = np.asarray([index for index in range(J.shape[0]) if index not in known], dtype=int)
    prescribed: np.ndarray = np.asarray([known[index] for index in fixed], dtype=complex)
    if fixed.size:
        if np.any(K[fixed].data != 0.0) or np.any(J[fixed][:, free].data != 0.0):
            raise ValueError("fixed coordinate rows must be algebraic and contain no free variables")
        try:
            splu(J[fixed][:, fixed].tocsc())
        except RuntimeError as error:
            raise FrequencySolveError("fixed algebraic coordinate block is singular") from error
    solutions = []
    for frequency in frequencies:
        omega = 2.0 * math.pi * float(frequency)
        if not math.isfinite(omega):
            raise ValueError("angular frequency must be finite")
        b = b_static + 1j * omega * b_rate
        if not np.all(np.isfinite(b)):
            raise FrequencySolveError(f"nonfinite frequency rhs at {frequency:g} Hz")
        matrix = (J + 1j * omega * K).tocsc()
        if not np.all(np.isfinite(matrix.data)):
            raise FrequencySolveError(f"nonfinite frequency matrix at {frequency:g} Hz")
        reduced = matrix[free][:, free]
        reduced_rhs = b[free] - matrix[free][:, fixed] @ prescribed
        if free.size == 0:
            x = np.zeros(J.shape[0], dtype=complex)
            x[fixed] = prescribed
            residual = matrix @ x - b
            if np.any(residual != 0.0):
                raise FrequencySolveError("fully prescribed coordinates do not satisfy the original equations")
            solutions.append(FrequencySolution(float(frequency), x, residual, 0.0, 0))
            continue
        row_max = np.asarray(abs(reduced).max(axis=1).todense()).ravel()
        if np.any(row_max == 0.0):
            raise FrequencySolveError(f"singular zero row at {frequency:g} Hz")
        rows = 1.0 / row_max
        scaled = diags(rows) @ reduced
        column_max = np.asarray(abs(scaled).max(axis=0).todense()).ravel()
        if np.any(column_max == 0.0):
            raise FrequencySolveError(f"singular zero column at {frequency:g} Hz")
        columns = 1.0 / column_max
        try:
            factor = splu((scaled @ diags(columns)).tocsc())
            x = np.zeros(J.shape[0], dtype=complex)
            x[fixed] = prescribed
            x[free] = columns * factor.solve(rows * reduced_rhs)
        except RuntimeError as error:
            raise FrequencySolveError(f"factorization failed at {frequency:g} Hz: {error}") from error
        for refinement in range(controls.max_refinements + 1):
            residual = matrix @ x - b
            denominator = abs(matrix) @ abs(x) + abs(b)
            error_rows = np.divide(
                abs(residual), denominator, out=np.zeros_like(denominator), where=denominator != 0.0,
            )
            error_rows[(denominator == 0.0) & (abs(residual) != 0.0)] = math.inf
            backward_error = float(np.max(error_rows))
            if np.all(np.isfinite(x)) and math.isfinite(backward_error) and backward_error <= controls.relative_tolerance:
                solutions.append(FrequencySolution(float(frequency), x, residual, backward_error, refinement))
                break
            if refinement == controls.max_refinements:
                raise FrequencySolveError(
                    f"linear residual failed at {frequency:g} Hz after {refinement} refinements: {backward_error:g}"
                )
            x[free] += columns * factor.solve(-rows * residual[free])
    return tuple(solutions)
