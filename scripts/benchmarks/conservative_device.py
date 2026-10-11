"""One sparse conservative BE step over public, physical-SI affine contracts.

This is a nonlinear equation consumer, not a trajectory or error controller.
The returned history contains finite storage differences/secants, not native
derivatives or a continuous physical tangent. Real-device gates are separate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Mapping

import numpy as np
from scipy.sparse.linalg import splu

from scripts.benchmarks.contract_prototype import (
    SECOND, ContractError, FloatArithmetic, FloatArray, PhysicalScaling,
    Point, StateIncrement, Unit, ValidatedProblem, frozen_array,
)
from scripts.benchmarks.precision_prototype import (
    DoubleArithmetic, DoubleArray, FrameInputExpansion, MappedAuthority,
    PrimitiveExpansion, RelativeCoordinates,
)


@dataclass(frozen=True)
class EquationBudget:
    """Absolute residual allowances in this equation's conservative units.

    Differential rows use amounts (Q); algebraic rows retain their own units.
    These are nonlinear solve budgets, not integration weights/physical gates.
    """

    unit: Unit
    absolute: np.ndarray

    def __post_init__(self):
        value = frozen_array(self.absolute)
        if not isinstance(self.unit, Unit) or value.ndim != 1 or np.any(value <= 0):
            raise ContractError("invalid_conservative_equation_budget")
        object.__setattr__(self, "absolute", value)


@dataclass(frozen=True)
class NewtonRecord:
    iteration: int
    rhs_projection_error: tuple[Fraction, ...]
    linear_residual_inf: float
    damping: float
    merit_before: Fraction | None = None
    merit_after: Fraction | None = None
    point_identity_before: str | None = None
    point_identity_after: str | None = None


@dataclass(frozen=True)
class RejectedSearch:
    damping: float
    point_identity: str | None
    feasible: bool | None
    merit: Fraction | None


@dataclass(frozen=True)
class ConservativeFailureDiagnostics:
    """Bounded data attached as ``error.conservative_diagnostics`` on failure.

    Only the latest requested trial retains a full Point/residual. A residual
    is present only after its public call returns for that exact Point; trial
    construction, feasibility and residual-call failures leave it absent.
    The last accepted identity denotes the Newton working iterate (initially
    the feasible initial guess), never an accepted integration step. Pending
    linear data belongs to ``linear_point_identity``, not a rejected trial.
    Records are bounded by max_iterations and the final max_line_search;
    residual_evaluations counts completed public calls, as on success.
    """

    phase: str
    local_coordinates: PrimitiveExpansion | None
    point: Point | None
    residual: object | None
    residual_point_identity: str | None
    feasible: bool | None
    merit: Fraction | None
    last_accepted_point_identity: str | None
    last_accepted_merit: Fraction | None
    iterations: tuple[NewtonRecord, ...]
    linear_iteration: int | None
    linear_point_identity: str | None
    rhs_projection_error: tuple[Fraction, ...] | None
    linear_residual_inf: float | None
    rejected_searches: tuple[RejectedSearch, ...]
    residual_evaluations: int
    line_search_evaluations: int


@dataclass(frozen=True)
class ConservativeStep:
    left: Point
    right: Point
    increment: StateIncrement
    local_coordinates: PrimitiveExpansion
    storage_delta: object
    storage_secant: object
    residual: object
    residual_units: tuple[Unit, ...]
    residual_allowances: tuple[Fraction, ...]
    residual_ratios: tuple[Fraction, ...]
    scaling: PhysicalScaling
    source_identity: str
    iterations: tuple[NewtonRecord, ...]
    residual_evaluations: int
    line_search_evaluations: int
    method: str = field(default="conservative-backward-euler", init=False)
    history_kind: str = field(default="finite-storage-difference-and-secant", init=False)
    scientific_qualified: bool = field(default=False, init=False)


def _exact_words(value, size):
    """Inspect all public result words; never round a residual for acceptance."""
    if isinstance(value, np.ndarray):
        parts = (value,)
    elif type(value) is FloatArray:
        parts = (value.values,)
    elif isinstance(value, (DoubleArray, PrimitiveExpansion)):
        parts = value.words
    else:
        raise ContractError("unsupported_conservative_value_representation")
    if any(w.shape != (size,) or w.dtype != np.float64 or not np.isfinite(w).all()
           for w in parts):
        raise ContractError("invalid_conservative_value_words")
    return tuple(sum((Fraction(float(w[i])) for w in parts), Fraction())
                 for i in range(size))


def _budgets(problem, budgets):
    if set(budgets) != {e.id for e in problem.layout.equations}:
        raise ContractError("incomplete_conservative_equation_budgets")
    units, limits = [], []
    for equation in problem.layout.equations:
        expected = equation.unit * SECOND if equation.role == "storage" else equation.unit
        budget = budgets[equation.id]
        count = int(np.prod(equation.shape))
        if (not isinstance(budget, EquationBudget) or budget.unit != expected
                or budget.absolute.shape != (count,)):
            raise ContractError("conservative_budget_unit_or_shape_mismatch")
        units.extend([expected] * count)
        limits.extend(Fraction(float(x)) for x in budget.absolute)
    return tuple(units), tuple(limits)


def _trial(coordinates, left, local, time, inputs):
    """Keep the accepted left endpoint and original reference; no rebase."""
    authority = left.state.authority
    if not isinstance(authority, MappedAuthority):
        raise ContractError("conservative_relative_authority_required")
    if authority.coordinate_kind == "local":
        return coordinates.advance(left, local, time, inputs)
    # Existing fixed-reference Points can include all twelve physical words.
    # Reuse their original zero-primitive reference and public paired trial.
    reference = authority.fixed_reference
    if (authority.coordinate_kind != "fixed-reference"
            or not isinstance(reference.state.authority, MappedAuthority)
            or any(not x.is_zero() for x in reference.state.authority.primitives.values())):
        raise ContractError("conservative_zero_primitive_reference_required")
    values = [authority.primitives[v.id] for v in coordinates.layout.variables]
    kind = FrameInputExpansion if authority.frame_inputs else PrimitiveExpansion
    count = 12 if kind is FrameInputExpansion else 4
    base = kind(tuple(np.concatenate([v.words[i].ravel() for v in values]) for i in range(count)))
    return coordinates.trial(reference, base.add(local), time, inputs, predecessor=left)


def _correction(solution, columns, damping):
    # A binary64 sparse solve proposes the correction. Preserve its exact
    # column-scale product, including its second word, in the trial primitive.
    arithmetic = DoubleArithmetic()
    value = arithmetic.weighted(arithmetic.array(solution), columns)
    value = arithmetic.weighted(value, np.full(solution.shape, damping))
    result = PrimitiveExpansion.from_value(arithmetic.freeze(value))
    expected = tuple(Fraction(float(s))*Fraction(float(c))*Fraction(damping)
                     for s, c in zip(solution, columns, strict=True))
    if _exact_words(result, solution.size) != expected:
        raise ContractError("conservative_correction_product_not_representable")
    return result


def conservative_be_step(
    problem: ValidatedProblem, coordinates: RelativeCoordinates, left: Point, h: float, *,
    scaling: PhysicalScaling, budgets: Mapping[str, EquationBudget], inputs=None,
    initial_increment: PrimitiveExpansion | DoubleArray | None = None,
    feasible: Callable[[Point], bool] = lambda point: True,
    max_iterations: int = 12, max_line_search: int = 8,
) -> ConservativeStep:
    """Solve DeltaQ-h*R(right)=0 and the unchanged algebraic equations.

    Only linear physical RelativeCoordinates are admitted. Each trial is a
    cumulative full-word physical increment from left; no absolute-state
    subtraction occurs. Time must advance exactly in the Point's binary64
    time representation. Runtime failures propagate unchanged, with bounded
    ``conservative_diagnostics`` attached; pre-entry validation is unchanged.
    """
    if not isinstance(problem, ValidatedProblem) or type(coordinates) is not RelativeCoordinates:
        raise ContractError("conservative_public_problem_and_coordinates_required")
    if coordinates.layout.identity != problem.layout.identity or any(m != "linear" for m in coordinates.modes.values()):
        raise ContractError("conservative_linear_physical_coordinates_required")
    n = problem.layout.size
    if (n == 0 or type(max_iterations) is not int or max_iterations < 0
            or type(max_line_search) is not int or not 1 <= max_line_search <= 64):
        raise ContractError("invalid_conservative_iteration_limits")
    if not isinstance(scaling, PhysicalScaling) or scaling.rows.shape != (n,) or scaling.columns.shape != (n,):
        raise ContractError("conservative_scale_shape_mismatch")
    if not np.isfinite(h) or h <= 0:
        raise ContractError("nonpositive_conservative_step")
    time = left.time + float(h)
    if (not np.isfinite(time) or time <= left.time
            or Fraction(time)-Fraction(left.time) != Fraction(float(h))):
        raise ContractError("conservative_time_increment_not_representable")
    inputs = left.inputs if inputs is None else frozen_array(inputs)
    if inputs.shape != (problem.input_count,):
        raise ContractError("conservative_input_shape_mismatch")
    units, limits = _budgets(problem, budgets)
    local = PrimitiveExpansion.from_value(np.zeros(n) if initial_increment is None else initial_increment)
    if local.shape != (n,):
        raise ContractError("conservative_increment_shape_mismatch")
    arithmetic = problem.arithmetic if problem.arithmetic is not None else FloatArithmetic()
    records, evaluations, searches = [], 0, 0
    current, phase = None, "initial_trial"
    requested = requested_point = requested_residual = requested_merit = requested_feasible = None
    linear_iteration = linear_point_identity = projection = linear_error = None
    rejected = []

    def clear_trial(candidate=None):
        nonlocal requested, requested_point, requested_residual, requested_merit, requested_feasible
        requested = candidate
        requested_point = requested_residual = requested_merit = requested_feasible = None

    def evaluate(candidate):
        nonlocal evaluations, phase, requested_point, requested_residual, requested_merit, requested_feasible
        clear_trial(candidate)
        phase = "trial"
        point, increment = _trial(coordinates, left, candidate, time, inputs)
        requested_point = point
        phase = "increment_validation"
        increment.validate(left, point)
        phase = "feasibility"
        if not feasible(point):
            requested_feasible = False
            return None
        requested_feasible = True
        phase = "residual"
        residual = problem.conservative_residual(left, point, increment)
        evaluations += 1
        requested_residual = residual
        phase = "residual_merit"
        exact = _exact_words(residual, n)
        ratios = tuple(abs(x)/a for x, a in zip(exact, limits, strict=True))
        requested_merit = max(ratios)
        return point, increment, residual, exact, ratios, requested_merit

    def rejected_trial(damping):
        return RejectedSearch(damping, None if requested_point is None else requested_point.identity,
                              requested_feasible, requested_merit)

    try:
        current = evaluate(local)
        if current is None:
            raise ContractError("infeasible_conservative_initial_trial")
        for iteration in range(max_iterations + 1):
            right, increment, residual, exact, ratios, merit = current
            if all(x <= 1 for x in ratios):
                phase = "accepted_storage"
                delta = problem.storage.delta(left, right, increment)
                _exact_words(delta, problem.storage_count)
                secant = arithmetic.freeze(arithmetic.divide(arithmetic.array(delta),
                                             arithmetic.array(np.full(problem.storage_count, h))))
                _exact_words(secant, problem.storage_count)
                return ConservativeStep(left, right, increment, local, delta, secant, residual,
                                        units, limits, ratios, scaling, problem.source_identity,
                                        tuple(records), evaluations, searches)
            if iteration == max_iterations:
                phase = "iteration_limit"
                raise ContractError("conservative_iteration_budget_exhausted")
            linear_iteration, linear_point_identity = iteration, right.identity
            phase = "jacobian"
            matrix = problem.conservative_jacobian(left, right)
            try:
                phase = "linear_solve"
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    scaled = scaling.jacobian(matrix)
                exact_rhs = tuple(-r/Fraction(float(s)) for r, s in zip(exact, scaling.rows, strict=True))
                rhs = np.array([float(x) for x in exact_rhs])
                if not np.isfinite(scaled.data).all() or not np.isfinite(rhs).all():
                    raise ArithmeticError("nonfinite scaled system")
                solution = splu(scaled).solve(rhs)
                if not np.isfinite(solution).all():
                    raise ArithmeticError("nonfinite sparse correction")
            except (RuntimeError, ArithmeticError) as error:
                raise ContractError("conservative_sparse_solve_failed") from error
            phase = "linear_residual"
            projection = tuple(x-Fraction(float(y)) for x, y in zip(exact_rhs, rhs, strict=True))
            linear_error = float(np.max(np.abs(scaled @ solution-rhs), initial=0.0))
            if not np.isfinite(linear_error):
                raise ContractError("nonfinite_conservative_linear_residual")
            for slot in range(max_line_search):
                searches += 1
                damping = 2.0**(-slot)
                clear_trial()
                phase = "correction"
                try:
                    proposed = local.add(_correction(solution, scaling.columns, damping))
                    trial = evaluate(proposed)
                except Exception:
                    rejected.append(rejected_trial(damping))
                    raise
                if trial is not None and trial[-1] < merit:
                    local, current = proposed, trial
                    records.append(NewtonRecord(iteration, projection, linear_error, damping,
                                                merit, trial[-1], right.identity, trial[0].identity))
                    linear_iteration = linear_point_identity = projection = linear_error = None
                    rejected.clear()
                    break
                rejected.append(rejected_trial(damping))
            else:
                phase = "line_search_limit"
                raise ContractError("conservative_line_search_budget_exhausted")
        raise AssertionError("unreachable")
    except Exception as error:
        error.conservative_diagnostics = ConservativeFailureDiagnostics(
            phase=phase, local_coordinates=requested, point=requested_point,
            residual=requested_residual,
            residual_point_identity=None if requested_residual is None else requested_point.identity,
            feasible=requested_feasible, merit=requested_merit,
            last_accepted_point_identity=None if current is None else current[0].identity,
            last_accepted_merit=None if current is None else current[-1], iterations=tuple(records),
            linear_iteration=linear_iteration, linear_point_identity=linear_point_identity,
            rhs_projection_error=projection, linear_residual_inf=linear_error,
            rejected_searches=tuple(rejected), residual_evaluations=evaluations,
            line_search_evaluations=searches)
        raise
