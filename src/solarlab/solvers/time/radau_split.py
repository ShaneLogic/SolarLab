"""Radau IIA with binary64 stage departures from two-word inventories.

The unknown v is dimensionless: Z_i = z_start_hi + z_start_lo + s*v_i.
All three summands survive into the independent Arb reconstruction. The
stage equation v_i - h*sum_j(A_ij*f_j/s) is evaluated in Arb, so neither a
large-inventory subtraction nor a rounded RHS dot product sets its floor.
Only the small Newton departures and the equilibrated linear solve use
binary64. Accepted inventories and cubic coefficients retain two words;
any remaining encoding error belongs to the separate trajectory defect.

This changes reference arithmetic, not the tableau, controls or physics.
It shares no BDF/BE step, history, interpolation or accounting. A successful
numerical trajectory is unqualified until its independent certificates pass.

Integration conditions: optional, unqualified reference infrastructure for
restricted declared E/I inputs and analytic unit fixtures only. Independence
is static-only: no BDF stepping, history, interpolation or accounting is used.
Retained failed attempts stay failures and qualification flags stay false.
Any future nonzero initial uncertainty must be expressed in the selected
weighted log norm; a numerical trajectory alone does not qualify a reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from functools import cache
from importlib import import_module
import math
from types import ModuleType
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from solarlab.solvers.time.radau_enclosure import (
    EnclosureBlocked,
    NonlinearInventoryEnclosure,
    NonlinearTubeResult,
    nonlinear_tube_bound,
)
from solarlab.solvers.time.radau_reference import (
    InventoryDAE,
    InventoryPoint,
    RadauControls,
    RadauCounters,
    RadauStep,
    RadauTrajectory,
)

__all__ = [
    "SplitInventory",
    "SplitRadauModel",
    "SplitRadauPoint",
    "SplitRadauStep",
    "SplitRadauTrajectory",
    "SplitRadauFailure",
    "integrate_split_radau",
]

Vector: TypeAlias = NDArray[np.float64]
Ball: TypeAlias = Any
PRECISION = 192


@cache
def _flint() -> ModuleType:
    """Load optional Arb arithmetic only when a reference operation needs it."""
    try:
        return import_module("flint")
    except ImportError as error:
        raise ImportError(
            "split-inventory Radau arithmetic require the optional python-flint backend "
            "(install solarlab[arb])"
        ) from error


def _frozen(value: ArrayLike) -> Vector:
    result = np.array(value, dtype=float, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError("radau_split_nonfinite_array")
    result.setflags(write=False)
    return result


def _q(value: float | int | Fraction) -> Ball:
    fraction = value if isinstance(value, Fraction) else Fraction(value)
    return _flint().arb(_flint().fmpq(fraction.numerator, fraction.denominator))


def _upper(value: Ball) -> float:
    if not value.is_finite():
        raise EnclosureBlocked("radau_split_nonfinite_enclosure")
    if value.is_zero():
        return 0.0
    result = float(value.upper())
    if not math.isfinite(result):
        raise EnclosureBlocked("radau_split_unrepresentable_enclosure")
    return math.nextafter(result, math.inf)


def _maximum(values: tuple[Ball, ...]) -> float:
    if not values:
        return 0.0
    result = values[0]
    for value in values[1:]:
        result = result.union(value)
    return _upper(result.upper())


# The exact same binary64 tableau words and integrated Lagrange basis as the
# stage-1 implementation. Unit tests bind every word; no BDF source is used.
_ROOT6 = math.sqrt(6.0)
_C = _frozen([(4 - _ROOT6) / 10, (4 + _ROOT6) / 10, 1.0])
_A = _frozen(
    [
        [(88 - 7 * _ROOT6) / 360, (296 - 169 * _ROOT6) / 1800, (-2 + 3 * _ROOT6) / 225],
        [(296 + 169 * _ROOT6) / 1800, (88 + 7 * _ROOT6) / 360, (-2 - 3 * _ROOT6) / 225],
        [(16 - _ROOT6) / 36, (16 + _ROOT6) / 36, 1 / 9],
    ]
)
_BASIS = _frozen(np.linalg.solve(np.vander(_C, 3, increasing=True), np.eye(3)))


@dataclass(frozen=True)
class SplitInventory:
    """The exact sum of two immutable binary64 reference vectors."""

    high: Vector
    low: Vector

    def __post_init__(self) -> None:
        high, low = _frozen(self.high), _frozen(self.low)
        if high.ndim != 1 or high.shape != low.shape:
            raise ValueError("radau_split_inventory_shape")
        object.__setattr__(self, "high", high)
        object.__setattr__(self, "low", low)

    @classmethod
    def from_values(cls, values: tuple[Ball, ...]) -> tuple[SplitInventory, Vector]:
        """Encode a midpoint as two words and enclose the discarded remainder."""
        with _flint().ctx.workprec(PRECISION):
            high = _frozen([float(value.mid()) for value in values])
            low = _frozen(
                [float((value - _q(hi)).mid()) for value, hi in zip(values, high)]
            )
            errors = _frozen(
                [
                    _upper((value - _q(hi) - _q(lo)).abs_upper())
                    for value, hi, lo in zip(values, high, low)
                ]
            )
        return cls(high, low), errors

    def values(self) -> tuple[Ball, ...]:
        with _flint().ctx.workprec(PRECISION):
            return tuple(_q(hi) + _q(lo) for hi, lo in zip(self.high, self.low))

    def shifted(self, departure: Vector, scale: Vector) -> tuple[Ball, ...]:
        if departure.shape != self.high.shape or scale.shape != self.high.shape:
            raise ValueError("radau_split_departure_shape")
        with _flint().ctx.workprec(PRECISION):
            return tuple(
                value + _q(s) * _q(v)
                for value, s, v in zip(self.values(), scale, departure)
            )


@dataclass
class SplitCounters(RadauCounters):
    compensated_residual_evaluations: int = 0
    backtrack_evaluations: int = 0
    changed_departures_with_unchanged_high_word: int = 0


@dataclass(frozen=True)
class SplitRadauPoint:
    time_s: float
    inventory: SplitInventory
    inventory_encoding_error_upper: Vector
    u: Vector
    rhs: tuple[Ball, ...]
    scaled_jacobian: Vector | None
    algebraic_residual: Vector
    storage_residual: Vector
    scaled_reconstruction_residual: float

    def bound_point(self) -> InventoryPoint:
        """High-word view only; the bound must receive its explicit low words."""
        return InventoryPoint(
            self.time_s,
            self.inventory.high,
            self.u,
            _frozen([float(value.mid()) for value in self.rhs]),
            self.algebraic_residual,
            self.storage_residual,
            self.scaled_reconstruction_residual,
        )


class SplitRadauModel:
    """Static assembled checks plus the independent inventory reconstruction."""

    def __init__(self, static: InventoryDAE, enclosure: NonlinearInventoryEnclosure):
        if enclosure.size != len(static.differential_rows) or not np.array_equal(
            enclosure.scale, static.inventory_scale
        ):
            raise ValueError("radau_split_enclosure_partition_or_scale_mismatch")
        self.static, self.enclosure = static, enclosure
        self.scale = static.inventory_scale

    def point(
        self,
        time_s: float,
        inventory: tuple[Ball, ...],
        controls: RadauControls,
        counters: SplitCounters,
        *,
        derivatives: bool = False,
    ) -> SplitRadauPoint:
        with _flint().ctx.workprec(PRECISION):
            counters.reconstructions += 1
            enclosed = self.enclosure.evaluate(
                Fraction(time_s), Fraction(time_s), inventory, derivatives=derivatives
            )
            u = _frozen([float(value.mid()) for value in enclosed.coordinates])
            static = self.static.assembled(time_s, u, counters)
            assert static.S is not None
            d, a = list(self.static.differential_rows), list(self.static.algebraic_rows)
            storage_error = tuple(
                _q(value) - z for value, z in zip(static.S[d], inventory)
            )
            errors = tuple(
                error.abs_upper() / _q(scale)
                for error, scale in zip(storage_error, self.scale)
            )
            errors += tuple(
                _q(value).abs_upper() / _q(scale)
                for value, scale in zip(static.F[a], self.static.algebraic_scale)
            )
            norm = _maximum(errors)
            if norm > controls.scaled_newton_tolerance:
                raise EnclosureBlocked(
                    "radau_split_original_static_checks_failed", str(norm)
                )
            jacobian = (
                _frozen(
                    [[float(x.mid()) for x in row] for row in enclosed.scaled_jacobian]
                )
                if derivatives
                else None
            )
            encoded, error = SplitInventory.from_values(inventory)
            return SplitRadauPoint(
                time_s,
                encoded,
                error,
                u,
                enclosed.rhs,
                jacobian,
                _frozen(static.F[a]),
                _frozen([float(x.mid()) for x in storage_error]),
                norm,
            )


@dataclass(frozen=True)
class SplitRadauStep:
    left_s: float
    right_s: float
    coefficients_high: Vector
    coefficients_low: Vector
    coefficient_encoding_error_upper: Vector
    stages: tuple[SplitRadauPoint, ...]
    stage_departures: Vector
    scaled_stage_residual: Vector
    scaled_stage_residual_upper: float
    endpoint: SplitRadauPoint

    def polynomial(self, time_s: float) -> tuple[Ball, ...]:
        if not math.isfinite(time_s) or not self.left_s <= time_s <= self.right_s:
            raise ValueError("radau_split_output_outside_step")
        with _flint().ctx.workprec(PRECISION):
            theta = _q(
                (Fraction(time_s) - Fraction(self.left_s))
                / (Fraction(self.right_s) - Fraction(self.left_s))
            )
            output = []
            for i in range(self.coefficients_high.shape[1]):
                value = _flint().arb(0)
                for k in (3, 2, 1, 0):
                    value = (
                        value * theta
                        + _q(self.coefficients_high[k, i])
                        + _q(self.coefficients_low[k, i])
                    )
                output.append(value)
            return tuple(output)


@dataclass(frozen=True)
class SplitRadauTrajectory:
    model: SplitRadauModel
    initial: SplitRadauPoint
    steps: tuple[SplitRadauStep, ...]
    controls: RadauControls
    counters: SplitCounters
    qualified: bool = field(default=False, init=False)

    def sample(self, time_s: float) -> SplitRadauPoint:
        self.counters.output_reconstructions += 1
        if time_s == self.initial.time_s:
            return self.model.point(
                time_s, self.initial.inventory.values(), self.controls, self.counters
            )
        for step in self.steps:
            if step.left_s <= time_s <= step.right_s:
                return self.model.point(
                    time_s, step.polynomial(time_s), self.controls, self.counters
                )
        raise ValueError("radau_split_output_outside_trajectory")

    def bound(
        self, *, initial_error: float, tube_radius: float, subdivisions: int = 2
    ) -> NonlinearTubeResult:
        steps = tuple(
            RadauStep(
                step.left_s,
                step.right_s,
                step.coefficients_high,
                tuple(point.bound_point() for point in step.stages),
                _frozen(step.scaled_stage_residual * self.model.scale),
                step.endpoint.bound_point(),
            )
            for step in self.steps
        )
        view = RadauTrajectory(
            self.model.static,
            self.initial.bound_point(),
            steps,
            self.controls,
            self.counters,
        )
        return nonlinear_tube_bound(
            self.model.enclosure,
            view,
            initial_error=initial_error,
            tube_radius=tube_radius,
            subdivisions=subdivisions,
            coefficient_corrections=tuple(step.coefficients_low for step in self.steps),
            initial_inventory_correction=self.initial.inventory.low,
        )


class SplitRadauFailure(RuntimeError):
    """A failed departure and exact reference, with the accepted prefix intact."""

    def __init__(
        self,
        code: str,
        start: SplitRadauPoint,
        departure: Vector | None,
        residual_upper: float | None = None,
    ):
        super().__init__(code)
        self.code, self.time_s = code, start.time_s
        self.step_start = start.inventory
        self.trial_departures = None if departure is None else _frozen(departure)
        self.scaled_residual_upper = residual_upper
        self.completed_steps: tuple[SplitRadauStep, ...] = ()
        self.failed_step: int | None = None
        self.counters: SplitCounters | None = None


def _residual(
    departures: Vector,
    points: tuple[SplitRadauPoint, ...],
    h: Ball,
    scale: Vector,
    counters: SplitCounters,
) -> tuple[Vector, float]:
    counters.compensated_residual_evaluations += 1
    rows = []
    bounds: list[Ball] = []
    for i in range(3):
        values = tuple(
            _q(departures[i, k])
            - h
            * sum(
                (_q(_A[i, j]) * points[j].rhs[k] / _q(scale[k]) for j in range(3)),
                _flint().arb(0),
            )
            for k in range(scale.size)
        )
        rows.append([float(value.mid()) for value in values])
        bounds.extend(value.abs_upper() for value in values)
    return _frozen(rows), _maximum(tuple(bounds))


def _step(
    model: SplitRadauModel,
    start: SplitRadauPoint,
    right_s: float,
    controls: RadauControls,
    counters: SplitCounters,
) -> SplitRadauStep:
    with _flint().ctx.workprec(PRECISION):
        h = _q(Fraction(right_s) - Fraction(start.time_s))
        times = np.array([float((_q(start.time_s) + h * _q(c)).mid()) for c in _C])
        times[-1] = right_s
        v = np.zeros((3, model.scale.size))
        if times[0] <= start.time_s or np.any(np.diff(times) <= 0):
            raise SplitRadauFailure("radau_split_stage_clock_resolution", start, v)

        def evaluate(
            values: Vector, derivatives: bool
        ) -> tuple[Vector, float, tuple[SplitRadauPoint, ...]]:
            points = tuple(
                model.point(
                    float(t),
                    start.inventory.shifted(row, model.scale),
                    controls,
                    counters,
                    derivatives=derivatives,
                )
                for t, row in zip(times, values)
            )
            residual, norm = _residual(values, points, h, model.scale, counters)
            return residual, norm, points

        norm: float | None = None
        for iteration in range(controls.max_newton_iterations + 1):
            try:
                residual, norm, points = evaluate(v, True)
            except (EnclosureBlocked, FloatingPointError) as error:
                raise SplitRadauFailure(
                    "radau_split_static_evaluation_failed", start, v, norm
                ) from error
            if norm <= controls.scaled_newton_tolerance:
                high, low = [start.inventory.high], [start.inventory.low]
                encoding = [np.zeros(model.scale.size)]
                for power in (1, 2, 3):
                    values = tuple(
                        h
                        / power
                        * sum(
                            (
                                _q(_BASIS[power - 1, j]) * points[j].rhs[i]
                                for j in range(3)
                            ),
                            _flint().arb(0),
                        )
                        for i in range(model.scale.size)
                    )
                    words, discarded = SplitInventory.from_values(values)
                    high.append(words.high)
                    low.append(words.low)
                    encoding.append(discarded)
                values = tuple(
                    sum((_q(hi[i]) + _q(lo[i]) for hi, lo in zip(high, low)), _flint().arb(0))
                    for i in range(model.scale.size)
                )
                endpoint_inventory, _ = SplitInventory.from_values(values)
                endpoint = model.point(
                    right_s, endpoint_inventory.values(), controls, counters
                )
                return SplitRadauStep(
                    start.time_s,
                    right_s,
                    _frozen(high),
                    _frozen(low),
                    _frozen(encoding),
                    points,
                    _frozen(v),
                    residual,
                    norm,
                    endpoint,
                )
            if iteration == controls.max_newton_iterations:
                break
            size = model.scale.size
            matrix = np.empty((3 * size, 3 * size))
            for i in range(3):
                for j, point in enumerate(points):
                    assert point.scaled_jacobian is not None
                    matrix[i * size : (i + 1) * size, j * size : (j + 1) * size] = (
                        -float((h * _q(_A[i, j])).mid()) * point.scaled_jacobian
                    )
                matrix[i * size : (i + 1) * size, i * size : (i + 1) * size] += np.eye(
                    size
                )
            counters.stage_updates += 1
            counters.linear_solves += 1
            try:
                correction = np.linalg.solve(matrix, -residual.ravel()).reshape(v.shape)
            except np.linalg.LinAlgError as error:
                raise SplitRadauFailure(
                    "radau_split_singular_stage_system", start, v, norm
                ) from error
            if not np.all(np.isfinite(correction)):
                raise SplitRadauFailure(
                    "radau_split_nonfinite_stage_correction", start, v, norm
                )
            for backtrack in range(controls.max_backtracks):
                trial = v + (0.5**backtrack) * correction
                counters.backtrack_evaluations += 1
                try:
                    _, trial_norm, trial_points = evaluate(trial, False)
                except (EnclosureBlocked, FloatingPointError, OverflowError):
                    continue
                if trial_norm < norm:
                    for row, old, new, before in zip(trial, points, trial_points, v):
                        counters.changed_departures_with_unchanged_high_word += int(
                            np.count_nonzero(
                                (row != before)
                                & (old.inventory.high == new.inventory.high)
                            )
                        )
                    v = trial
                    break
            else:
                raise SplitRadauFailure("radau_split_stage_line_search", start, v, norm)
        raise SplitRadauFailure("radau_split_stage_iteration_cap", start, v, norm)


def integrate_split_radau(
    model: SplitRadauModel,
    initial_u: ArrayLike,
    times_s: ArrayLike,
    *,
    controls: RadauControls = RadauControls(),
    initial_inventory: SplitInventory | None = None,
) -> SplitRadauTrajectory:
    """One explicit grid, unchanged controls, no adaptive retries or admission."""
    times = _frozen(times_s)
    if (
        times.ndim != 1
        or times.size < 2
        or np.any(np.diff(times) <= 0)
        or not np.all(np.isfinite(np.diff(times)))
    ):
        raise ValueError("radau_split_requires_increasing_grid")
    if times.size - 1 > controls.max_steps:
        raise ValueError("radau_split_step_cap")
    counters = SplitCounters()
    raw = model.static.assembled(float(times[0]), _frozen(initial_u), counters)
    assert raw.S is not None
    inventory = (
        SplitInventory(
            raw.S[list(model.static.differential_rows)], np.zeros(model.scale.size)
        )
        if initial_inventory is None
        else initial_inventory
    )
    if inventory.high.shape != model.scale.shape:
        raise ValueError("radau_split_initial_inventory_shape")
    initial = model.point(float(times[0]), inventory.values(), controls, counters)
    completed: list[SplitRadauStep] = []
    start = initial
    for index, right in enumerate(times[1:]):
        try:
            step = _step(model, start, float(right), controls, counters)
        except SplitRadauFailure as failure:
            failure.completed_steps, failure.failed_step = tuple(completed), index
            failure.counters = counters
            raise
        except (EnclosureBlocked, FloatingPointError, TimeoutError) as error:
            wrapped = SplitRadauFailure(
                "radau_split_budget_exhausted"
                if isinstance(error, TimeoutError)
                else "radau_split_endpoint_evaluation_failed",
                start,
                None,
            )
            wrapped.completed_steps, wrapped.failed_step = tuple(completed), index
            wrapped.counters = counters
            raise wrapped from error
        completed.append(step)
        start = step.endpoint
    return SplitRadauTrajectory(model, initial, tuple(completed), controls, counters)
