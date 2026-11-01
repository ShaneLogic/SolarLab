"""Manufactured public storage contracts; no coupled device or native solver."""
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction
from hashlib import sha256

import numpy as np
import pytest
from scipy.sparse import csc_matrix

import scripts.benchmarks.conservative_device as conservative
from scripts.benchmarks.contract_prototype import (
    PARTICLE, SECOND, VOLT, AcceptedStep, ContractError, EquationSpec, Layout,
    PhysicalScaling, PhysicalStorage, SparseStructure, StateView, Support,
    ValidatedProblem, VariableSpec,
)
from scripts.benchmarks.conservative_device import EquationBudget, NewtonRecord, conservative_be_step
from scripts.benchmarks.precision_prototype import (
    DoubleArithmetic, DoubleArray, FrameInputExpansion,
    RelativeCoordinates, encode_point,
)


def exact(value):
    return tuple(sum((Fraction(float(w[i])) for w in value.words), Fraction())
                 for i in range(value.shape[0]))


def manufactured(kind="linear", *, force=(2.**-80, -2.**-80), time=0.):
    """Reuse the two-cell RelativeCoordinates/PhysicalStorage fixture pattern."""
    initial = [2.] if kind == "input_storage" else [3., 3.] if kind == "linear" else [1., 2.]
    count = len(initial)
    algebraic = kind == "exchange"
    supports = [Support("cells", "cell", (count,))]
    variables = [VariableSpec("n", "manufactured", "cells", (count,), PARTICLE, lower=0)]
    equations = [EquationSpec("balance", "manufactured", "cells", (count,), PARTICLE/SECOND)]
    if algebraic:
        supports.append(Support("port", "global", (1,)))
        variables.append(VariableSpec("phi", "manufactured", "port", (1,), VOLT, role="constraint"))
        equations.append(EquationSpec("constraint", "manufactured", "port", (1,), VOLT, role="constraint"))
    layout = Layout(tuple(supports), tuple(variables), tuple(equations))
    coordinates = RelativeCoordinates(layout, {v.id: "linear" for v in variables})
    fields = [("n", DoubleArray(initial))]
    if algebraic:
        fields.append(("phi", DoubleArray([sum(initial)])))
    left = coordinates.initial(StateView(layout, fields), time, [0.] if kind == "input_storage" else [])
    source = sha256(repr((kind, initial, force, time)).encode()).hexdigest()
    arithmetic = DoubleArithmetic()
    size = layout.size
    # The tiny fixture declares every slot, including zeros. The consumer may
    # neither infer a sparsity pattern nor fall back to a dense global solve.
    structure = SparseStructure((size, size), np.tile(np.arange(size), size),
                                np.arange(size+1)*size, source, layout.identity)
    calls = {"delta": 0, "conservative": 0, "jacobian": 0, "instantaneous": 0}

    def value(point):
        n = point.state.field("n").as_dd()
        q = n if kind == "linear" else n*n  # quadratic coefficient: 1/PARTICLE
        if kind == "input_storage":
            q = q + point.inputs[0] + 2*point.time
        return DoubleArray.from_dd(q)

    def delta(a, b, increment):
        calls["delta"] += 1
        dn = increment.field("n").as_dd()
        q = dn if kind == "linear" else dn*(2*a.state.field("n").as_dd()+dn)
        if kind == "input_storage":
            q = q + (b.inputs[0]-a.inputs[0]) + 2*(b.time-a.time)
        return DoubleArray.from_dd(q)

    storage = PhysicalStorage(value, delta, (PARTICLE,)*count, source, arithmetic)

    def residual(a, b, increment):
        calls["conservative"] += 1
        n = b.state.field("n").as_dd()
        if kind == "exchange":
            rate = arithmetic.concatenate((arithmetic.reshape(n[1]-n[0], (1,)),
                                           arithmetic.reshape(n[0]-n[1], (1,))))
        else:
            rate = arithmetic.array(force if kind == "linear" else [0.])
        rows = storage.delta(a, b, increment).as_dd() - (b.time-a.time)*rate
        if algebraic:
            g = b.state.field("phi").as_dd() - (n[0]+n[1])
            rows = arithmetic.concatenate((rows, g))
        return arithmetic.freeze(rows)

    def jacobian(a, b):
        calls["jacobian"] += 1
        values = np.zeros((size, size))
        # CSC coefficients are explicitly binary64; full-word residuals decide
        # convergence, including the small corrections this proposal omits.
        values[np.arange(count), np.arange(count)] = 1 if kind == "linear" else 2*b.state.field("n").high
        if algebraic:
            h = b.time-a.time
            values[:2, :2] += h*np.array([[1., -1.], [-1., 1.]])
            values[2] = [-1., -1., 1.]
        return structure.filled(values.T.ravel())

    def instantaneous(*args):
        calls["instantaneous"] += 1
        raise AssertionError("The conservative consumer must not ask for tangent F/J")

    problem = ValidatedProblem(layout, left.inputs.size, count, storage, structure, source,
                               instantaneous, instantaneous, residual, jacobian, arithmetic=arithmetic)
    budgets = {"balance": EquationBudget(PARTICLE, np.full(count, 1e-24))}
    if algebraic:
        budgets["constraint"] = EquationBudget(VOLT, [1e-25])
    scaling = PhysicalScaling(np.ones(size), np.ones(size))
    return problem, coordinates, left, scaling, budgets, calls


def run_fixture(fixture, h=.5, **options):
    problem, coordinates, left, scaling, budgets, _ = fixture
    options.setdefault("scaling", scaling)
    options.setdefault("budgets", budgets)
    return conservative_be_step(problem, coordinates, left, h, **options)


@pytest.mark.parametrize("globalization", ["residual", "correction"])
def test_sub_ulp_increment_and_discrete_history_are_retained(globalization):
    fixture = manufactured()
    problem, _, left, _, _, calls = fixture
    before = encode_point(left)
    budgets = {"balance": EquationBudget(PARTICLE, [2.**-180, 2.**-180])}
    step = run_fixture(fixture, budgets=budgets, globalization=globalization)
    expected = (Fraction(1, 2**81), -Fraction(1, 2**81))
    assert exact(step.storage_delta) == exact(step.local_coordinates) == expected
    assert exact(step.increment.field("n")) == expected
    np.testing.assert_array_equal(step.right.state.field("n").high, left.state.field("n").high)
    assert step.right.identity != left.identity and encode_point(left) == before
    assert exact(step.storage_secant) == tuple(2*x for x in expected)
    assert sum(exact(step.storage_delta)) == 0
    assert step.source_identity == problem.source_identity
    assert step.history_kind == "finite-storage-difference-and-secant"
    assert not isinstance(step, AcceptedStep) and not step.scientific_qualified
    assert calls == {"delta": 3, "conservative": 2, "jacobian": 1, "instantaneous": 0}
    assert step.residual_evaluations == 2 and step.line_search_evaluations == 1
    assert step.globalization == globalization
    assert step.factorization_calls == step.factorization_returns == 1
    assert step.linear_solve_calls == step.linear_solve_returns == (2 if globalization == "correction" else 1)
    record, = step.iterations
    assert (record.merit_before, record.merit_after, record.damping) == (2**99, 0, 1.)
    assert record.point_identity_after == step.right.identity
    assert record.point_identity_before not in (left.identity, step.right.identity)
    # Four-argument construction remains compatible with existing consumers.
    assert NewtonRecord(0, (), 0., 1.).merit_before is None
    with pytest.raises((TypeError, ValueError), match="init=False"):
        replace(step, history_kind="native-continuous-tangent")


@pytest.mark.parametrize("globalization", ["residual", "correction"])
def test_nonbinary_solver_scales_do_not_become_physical_acceptance(globalization):
    fixture = manufactured()
    step = run_fixture(fixture, scaling=PhysicalScaling([.7, 1.3], [.1, .3]),
                       budgets={"balance": EquationBudget(PARTICLE, [2.**-175]*2)}, globalization=globalization)
    wanted = (Fraction(1, 2**81), -Fraction(1, 2**81))
    assert all(abs(x-y) <= Fraction(1, 2**175)
               for x, y in zip(exact(step.storage_delta), wanted, strict=True))
    assert any(x != 0 for row in step.iterations for x in row.rhs_projection_error)
    assert all(x <= 1 for x in step.residual_ratios)


@pytest.mark.parametrize("globalization", ["residual", "correction"])
def test_fixed_reference_twelve_words_survive_a_step(globalization):
    problem, coordinates, reference, scaling, budgets, calls = manufactured()
    words = [np.full(2, 2.**(-20-60*i)) for i in range(12)]
    left, _ = coordinates.trial(reference, FrameInputExpansion(words), 1.)
    before = encode_point(left)
    step = conservative_be_step(problem, coordinates, left, .5, scaling=scaling,
        budgets={"balance": EquationBudget(PARTICLE, [2.**-180]*2)}, globalization=globalization)
    expected = (Fraction(1, 2**81), -Fraction(1, 2**81))
    assert exact(step.storage_delta) == expected
    assert step.right.state.authority.fixed_reference is reference
    retained = step.right.state.authority.primitives["n"].words
    assert len(retained) == 12
    np.testing.assert_array_equal(retained[-1], words[-1])
    assert encode_point(left) == before and calls["instantaneous"] == 0


def exchange_reference(h):
    """Independent storage conservation plus scalar monotone bisection."""
    with localcontext() as ctx:
        ctx.prec = 80
        lo, hi = Decimal(1), (Decimal(5)/2).sqrt()
        dt = Decimal.from_float(h)
        for _ in range(260):
            x = (lo+hi)/2
            y = (5-x*x).sqrt()
            if x*x-1-dt*(y-x) > 0:
                hi = x
            else:
                lo = x
        x = (lo+hi)/2
        return x, (5-x*x).sqrt()


@pytest.mark.parametrize("globalization", ["residual", "correction"])
def test_vector_nonlinear_storage_algebraic_rows_and_sparse_only(monkeypatch, globalization):
    fixture = manufactured("exchange")
    def forbidden(*args, **kwargs):
        raise AssertionError("dense fallback")
    monkeypatch.setattr(np.linalg, "solve", forbidden)
    monkeypatch.setattr(np.linalg, "lstsq", forbidden)
    monkeypatch.setattr(csc_matrix, "toarray", forbidden)
    reference = exchange_reference(.25)
    steps = [run_fixture(fixture, .25, globalization=globalization), run_fixture(fixture, .25,
        scaling=PhysicalScaling([.5, 4., 8.], [2., .25, 4.]), globalization=globalization)]
    for step in steps:
        n = exact(step.right.state.field("n"))
        with localcontext() as ctx:
            ctx.prec = 80
            for actual, wanted in zip(n, reference, strict=True):
                assert abs(Decimal(actual.numerator)/Decimal(actual.denominator)-wanted) < Decimal('2e-24')
        assert abs(n[0]*n[0]+n[1]*n[1]-5) < Fraction(1, 10**23)
        assert abs(exact(step.right.state.field("phi"))[0]-sum(n)) < Fraction(1, 10**24)
        assert step.residual_units == (PARTICLE, PARTICLE, VOLT)
        assert all(r <= 1 for r in step.residual_ratios)
        assert step.increment.left_identity == fixture[2].identity
    assert fixture[-1]["instantaneous"] == 0


@pytest.mark.parametrize("globalization", ["residual", "correction"])
def test_finite_input_time_storage_and_jacobian_not_tangent_history(globalization):
    fixture = manufactured("input_storage")
    problem, coordinates, left, _, _, calls = fixture
    step = run_fixture(fixture, .25, inputs=[.5], globalization=globalization)
    with localcontext() as ctx:
        ctx.prec = 80
        x = exact(step.right.state.field("n"))[0]
        assert abs(Decimal(x.numerator)/Decimal(x.denominator)-Decimal(3).sqrt()) < Decimal('1e-24')
    # Q=n^2+u+2t is constant, although the state secant has a nonzero
    # continuous chain-rule storage rate at the right endpoint.
    assert abs(exact(step.storage_delta)[0]) < Fraction(1, 10**24)
    secant_state = (x-2)/Fraction(1, 4)
    assert abs(2*x*secant_state+4) > Fraction(1, 10)
    jac = problem.conservative_jacobian(left, step.right)
    for epsilon in (2.**-12, 2.**-16):
        plus, ip = coordinates.advance(left, step.local_coordinates.add([epsilon]), .25, [.5])
        minus, im = coordinates.advance(left, step.local_coordinates.add([-epsilon]), .25, [.5])
        rp = exact(problem.conservative_residual(left, plus, ip))[0]
        rm = exact(problem.conservative_residual(left, minus, im))[0]
        fd = (rp-rm)/(2*Fraction(epsilon))
        assert abs(fd-Fraction(float(jac[0, 0]))) < Fraction(1, 10**12)
    assert calls["instantaneous"] == 0


def test_low_residual_word_controls_acceptance():
    fixture = manufactured()
    problem = replace(fixture[0], conservative_values=lambda *args: DoubleArray([1., 0.], [2.**-80, 0.]))
    with pytest.raises(ContractError, match="iteration_budget"):
        conservative_be_step(problem, fixture[1], fixture[2], .5, scaling=fixture[3],
                             budgets={"balance": EquationBudget(PARTICLE, [1., 1.])}, max_iterations=0)


@pytest.mark.parametrize("error", ["missing", "unit", "shape"])
def test_equation_budgets_cannot_mix_units_or_broadcast(error):
    fixture = manufactured("exchange")
    budgets = dict(fixture[4])
    if error == "missing":
        del budgets["constraint"]
    elif error == "unit":
        budgets["balance"] = EquationBudget(PARTICLE/SECOND, [1e-24]*2)
    else:
        budgets["balance"] = EquationBudget(PARTICLE, [1e-24])
    with pytest.raises(ContractError, match="budget"):
        run_fixture(fixture, budgets=budgets)
    assert fixture[-1]["conservative"] == 0


@pytest.mark.parametrize("h", [0., -1., np.inf, np.nan, 2.**-80])
def test_invalid_or_unrepresentable_time_step_rejects(h):
    fixture = manufactured(time=1.)
    with pytest.raises(ContractError, match="step|time_increment"):
        run_fixture(fixture, h)
    assert fixture[-1]["conservative"] == 0


@pytest.mark.parametrize("failure", ["singular", "line_search", "iterations", "structure", "infeasible"])
@pytest.mark.parametrize("globalization", ["residual", "correction"])
def test_failures_do_not_rebase_clip_or_invoke_dense_fallback(failure, globalization):
    fixture = manufactured(force=(1., -1.))
    problem, coordinates, left, scaling, budgets, _ = fixture
    before = encode_point(left)
    options = {"globalization": globalization}
    if failure == "singular":
        problem = replace(problem, conservative_derivative=lambda *args: problem.structure.filled(np.zeros(4)))
    elif failure == "line_search":
        original = problem.conservative_derivative
        problem = replace(problem, conservative_derivative=lambda *args: -original(*args))
        options['max_line_search'] = 2
    elif failure == "iterations":
        options['max_iterations'] = 0
    elif failure == "structure":
        problem = replace(problem, conservative_derivative=lambda *args: csc_matrix(np.eye(2)))
    else:
        options['feasible'] = lambda point: False
    with pytest.raises(ContractError) as caught:
        conservative_be_step(problem, coordinates, left, .5, scaling=scaling, budgets=budgets, **options)
    assert encode_point(left) == before
    diagnostic = caught.value.conservative_diagnostics
    if failure == "singular":
        assert isinstance(caught.value.__cause__, RuntimeError)
        assert diagnostic.phase == "linear_solve" and diagnostic.linear_iteration == 0
        assert diagnostic.rhs_projection_error is diagnostic.linear_residual_inf is None
        assert diagnostic.residual_point_identity == diagnostic.point.identity
    elif failure == "infeasible":
        assert diagnostic.feasible is False and diagnostic.point is not None
        assert diagnostic.last_accepted_point_identity is diagnostic.residual is None
        assert diagnostic.residual_point_identity is None


def test_nonphysical_coordinate_chain_is_not_assumed():
    problem, coordinates, left, scaling, budgets, calls = manufactured()
    logarithmic = RelativeCoordinates(coordinates.layout, {"n": "log"})
    with pytest.raises(ContractError, match="linear_physical"):
        conservative_be_step(problem, logarithmic, left, .5, scaling=scaling, budgets=budgets)
    assert calls["conservative"] == 0


def test_iteration_exhaustion_retains_scalar_history_and_final_full_word_trial():
    fixture = manufactured(force=(1., -1.))
    problem, _, left, _, budgets, calls = fixture
    before = encode_point(left)
    derivative, values = problem.conservative_derivative, problem.conservative_values
    observed = []

    def observe(a, b, increment):
        result = values(a, b, increment)
        observed.append((b.identity, exact(result)))
        return result

    problem = replace(problem, conservative_values=observe,
                      conservative_derivative=lambda *args: 2*derivative(*args))
    with pytest.raises(ContractError, match="conservative_iteration_budget_exhausted") as caught:
        run_fixture((problem, *fixture[1:]), max_iterations=2)
    error = caught.value
    diagnostic = error.conservative_diagnostics
    assert error.reason == str(error) == "conservative_iteration_budget_exhausted"
    assert diagnostic.phase == "iteration_limit"
    assert calls == {"delta": 3, "conservative": 3, "jacobian": 2, "instantaneous": 0}
    assert (diagnostic.residual_evaluations, diagnostic.line_search_evaluations) == (3, 2)
    assert exact(diagnostic.local_coordinates) == (Fraction(3, 8), -Fraction(3, 8))
    assert exact(diagnostic.point.state.field("n")) == (Fraction(27, 8), Fraction(21, 8))
    assert exact(diagnostic.residual) == observed[-1][1] == (-Fraction(1, 8), Fraction(1, 8))
    assert diagnostic.point.identity == diagnostic.residual_point_identity == observed[-1][0]
    assert diagnostic.last_accepted_point_identity == diagnostic.point.identity
    assert diagnostic.last_accepted_merit == diagnostic.merit
    assert diagnostic.linear_iteration is diagnostic.linear_point_identity is None
    assert diagnostic.rhs_projection_error is diagnostic.linear_residual_inf is None
    assert diagnostic.rejected_searches == ()
    allowance = Fraction(float(budgets["balance"].absolute[0]))
    assert len(diagnostic.iterations) == 2
    for i, record in enumerate(diagnostic.iterations):
        assert record.iteration == i and record.damping == 1.
        assert record.rhs_projection_error == (0, 0) and record.linear_residual_inf == 0.
        assert record.merit_before == Fraction(1, 2**(i+1))/allowance
        assert record.merit_after == Fraction(1, 2**(i+2))/allowance
        assert (record.point_identity_before, record.point_identity_after) == (observed[i][0], observed[i+1][0])
    assert encode_point(left) == before


@pytest.mark.parametrize("infeasible", [False, True])
@pytest.mark.parametrize("globalization", ["residual", "correction"])
def test_failed_search_distinguishes_rejected_trial_from_newton_iterate(infeasible, globalization):
    fixture = manufactured(force=(1., -1.))
    problem, _, left, _, _, calls = fixture
    derivative = problem.conservative_derivative
    if not infeasible:
        problem = replace(problem, conservative_derivative=lambda *args: -derivative(*args))
    visited = []

    def feasible(point):
        visited.append(point.identity)
        return not infeasible or len(visited) == 1

    with pytest.raises(ContractError, match="conservative_line_search_budget_exhausted") as caught:
        run_fixture((problem, *fixture[1:]), feasible=feasible, max_line_search=2, globalization=globalization)
    diagnostic = caught.value.conservative_diagnostics
    assert diagnostic.phase == "line_search_limit"
    assert diagnostic.point.identity == visited[-1] != visited[0]
    assert diagnostic.last_accepted_point_identity == diagnostic.linear_point_identity == visited[0]
    assert diagnostic.linear_iteration == 0 and diagnostic.iterations == ()
    assert diagnostic.rhs_projection_error == (0, 0) and diagnostic.linear_residual_inf == 0.
    assert diagnostic.line_search_evaluations == 2
    assert [s.damping for s in diagnostic.rejected_searches] == [1., .5]
    assert [s.point_identity for s in diagnostic.rejected_searches] == visited[1:]
    if globalization == "correction":
        assert diagnostic.correction_base.point_identity == visited[0]
        assert diagnostic.linear_solve_calls == diagnostic.linear_solve_returns == (1 if infeasible else 3)
        assert [s.solve.point_identity for s in diagnostic.correction_searches] == ([] if infeasible else visited[1:])
        assert all(s.accepted is False for s in diagnostic.correction_searches)
    if infeasible:
        assert exact(diagnostic.local_coordinates) == (Fraction(1, 4), -Fraction(1, 4))
        assert diagnostic.residual is diagnostic.residual_point_identity is diagnostic.merit is None
        assert diagnostic.feasible is False
        assert all(s.feasible is False and s.merit is None for s in diagnostic.rejected_searches)
        assert calls == {"delta": 1, "conservative": 1, "jacobian": 1, "instantaneous": 0}
        assert diagnostic.residual_evaluations == 1
    else:
        assert exact(diagnostic.local_coordinates) == (-Fraction(1, 4), Fraction(1, 4))
        assert exact(diagnostic.residual) == (-Fraction(3, 4), Fraction(3, 4))
        assert diagnostic.residual_point_identity == diagnostic.point.identity
        assert diagnostic.feasible is True
        assert [s.merit/diagnostic.last_accepted_merit for s in diagnostic.rejected_searches] == [2, Fraction(3, 2)]
        assert calls == {"delta": 3, "conservative": 3, "jacobian": 1, "instantaneous": 0}
        assert diagnostic.residual_evaluations == 3
    assert exact(left.state.field("n")) == (3, 3)


@pytest.mark.parametrize("failure_phase", ["trial", "feasibility", "residual"])
@pytest.mark.parametrize("globalization", ["residual", "correction"])
def test_callback_failure_never_reuses_the_previous_trial_residual(monkeypatch, failure_phase, globalization):
    fixture = manufactured(force=(1., -1.))
    problem, _, left, _, _, calls = fixture
    original_trial, values = conservative._trial, problem.conservative_values
    failure, cause = ContractError("manufactured_trial_failure"), RuntimeError("manufactured_cause")
    points, feasibility_calls, residual_attempts = [], [], []
    trial_attempts = 0

    def trial(*args):
        nonlocal trial_attempts
        trial_attempts += 1
        if failure_phase == "trial" and trial_attempts == 2:
            raise failure from cause
        result = original_trial(*args)
        points.append(result[0])
        return result

    def feasible(point):
        feasibility_calls.append(point.identity)
        if failure_phase == "feasibility" and len(feasibility_calls) == 2:
            raise failure from cause
        return True

    def residual(a, b, increment):
        residual_attempts.append(b.identity)
        if failure_phase == "residual" and len(residual_attempts) == 2:
            raise failure from cause
        return values(a, b, increment)

    monkeypatch.setattr(conservative, "_trial", trial)
    problem = replace(problem, conservative_values=residual)
    before = encode_point(left)
    with pytest.raises(ContractError, match="manufactured_trial_failure") as caught:
        run_fixture((problem, *fixture[1:]), feasible=feasible, globalization=globalization)
    assert caught.value is failure and failure.__cause__ is cause
    diagnostic = failure.conservative_diagnostics
    assert diagnostic.phase == failure_phase
    assert diagnostic.residual is diagnostic.residual_point_identity is diagnostic.merit is None
    assert exact(diagnostic.local_coordinates) == (Fraction(1, 2), -Fraction(1, 2))
    assert diagnostic.last_accepted_point_identity == diagnostic.linear_point_identity == points[0].identity
    if failure_phase == "trial":
        assert diagnostic.point is None
    else:
        assert diagnostic.point is points[-1] and diagnostic.point.identity != points[0].identity
    assert diagnostic.feasible is (True if failure_phase == "residual" else None)
    search, = diagnostic.rejected_searches
    assert search.point_identity == (None if diagnostic.point is None else diagnostic.point.identity)
    assert search.merit is None and search.damping == 1.
    assert diagnostic.iterations == () and diagnostic.linear_iteration == 0
    assert diagnostic.rhs_projection_error == (0, 0) and diagnostic.linear_residual_inf == 0.
    assert (diagnostic.residual_evaluations, diagnostic.line_search_evaluations) == (1, 1)
    assert trial_attempts == 2 and len(residual_attempts) == (2 if failure_phase == "residual" else 1)
    assert calls == {"delta": 1, "conservative": 1, "jacobian": 1, "instantaneous": 0}
    assert encode_point(left) == before


def curved_fixture():
    """F=(x-1, y+2*x^2) at h=1; exact physical root n=(4,1).

    x,y are finite particle increments from (3,3). This triangular curve
    has a large residual remainder but a small column-scaled correction.
    """
    problem, coordinates, left, _, budgets, calls = manufactured(force=(1., 0.))
    arithmetic = problem.arithmetic

    def values(a, b, increment):
        calls["conservative"] += 1
        dn = problem.storage.delta(a, b, increment).as_dd()
        h = b.time-a.time
        return arithmetic.freeze(arithmetic.concatenate((
            arithmetic.reshape(dn[0]-h, (1,)),
            arithmetic.reshape(dn[1]+2*h*dn[0]*dn[0], (1,)))))

    def derivative(a, b):
        calls["jacobian"] += 1
        x = (b.state.field("n").as_dd()-a.state.field("n").as_dd())[0]
        return problem.structure.filled([1., 4*(b.time-a.time)*float(x.hi), 0., 1.])

    problem = replace(problem, conservative_values=values, conservative_derivative=derivative)
    return problem, coordinates, left, PhysicalScaling([1., 1.], [1., 16.]), budgets, calls


def test_correction_globalization_accepts_curvature_but_keeps_physical_gate():
    fixture = curved_fixture()
    with pytest.raises(ContractError, match="iteration_budget_exhausted") as caught:
        run_fixture(fixture, 1., max_iterations=2, max_line_search=6)
    old = caught.value.conservative_diagnostics
    assert old.globalization == "residual" and old.iterations[0].damping == .5
    assert old.merit > 1 and old.linear_solve_calls == 2

    fixture = curved_fixture()
    step = run_fixture(fixture, 1., max_iterations=2, max_line_search=6, globalization="correction")
    assert exact(step.right.state.field("n")) == (4, 1)
    assert exact(step.storage_delta) == (1, -2)
    assert all(r <= 1 for r in step.residual_ratios) and exact(step.residual) == (0, 0)
    assert step.factorization_calls == step.factorization_returns == 2
    assert step.linear_solve_calls == step.linear_solve_returns == 4
    assert (step.residual_evaluations, step.line_search_evaluations) == (3, 2)
    first, second = step.iterations
    assert first.merit_after == 2*first.merit_before
    assert first.damping == second.damping == 1.
    assert second.merit_after == 0
    base, = first.correction_base.solution[:1]
    assert base == 1.
    trial, = first.correction_searches
    assert first.correction_base.norm_squared == 1
    assert trial.solve.norm_squared == Fraction(1, 64)
    assert trial.threshold_squared == Fraction(9, 16) and trial.accepted
    assert trial.solve.rhs == (0., -2.) and trial.solve.solution == (0., -.125)
    assert trial.base_point_identity == first.point_identity_before
    assert trial.solve.point_identity == first.point_identity_after == second.point_identity_before
    assert trial.solve.rhs_projection_error == (0, 0) and trial.solve.linear_residual_inf == 0.
    assert step.source_identity == fixture[0].source_identity and not step.scientific_qualified
    assert fixture[-1] == {"delta": 4, "conservative": 3, "jacobian": 2, "instantaneous": 0}


def test_tiny_correction_with_failing_residual_is_not_success():
    fixture = manufactured(force=(1., -1.))
    original = fixture[0].conservative_derivative
    problem = replace(fixture[0], conservative_derivative=lambda *args: 2.**500*original(*args))
    with pytest.raises(ContractError, match="line_search_budget_exhausted") as caught:
        run_fixture((problem, *fixture[1:]), globalization="correction", max_line_search=2)
    diagnostic = caught.value.conservative_diagnostics
    assert 0 < diagnostic.correction_base.norm_squared < Fraction(1, 10**200)
    assert diagnostic.merit > 1 and diagnostic.iterations == ()
    assert all(s.solve.norm_squared == diagnostic.correction_base.norm_squared
               and not s.accepted for s in diagnostic.correction_searches)
    assert diagnostic.source_identity == problem.source_identity and diagnostic.scaling is fixture[3]


def test_zero_returned_correction_cannot_pass_a_failing_residual(monkeypatch):
    class ZeroFactor:
        def solve(self, rhs):
            return np.zeros_like(rhs)
    monkeypatch.setattr(conservative, "splu", lambda matrix: ZeroFactor())
    with pytest.raises(ContractError, match="zero_correction_with_failing_residual") as caught:
        run_fixture(manufactured(force=(1., -1.)), globalization="correction")
    diagnostic = caught.value.conservative_diagnostics
    assert diagnostic.correction_base.norm_squared == 0 and diagnostic.merit > 1
    assert diagnostic.correction_searches == () and diagnostic.iterations == ()
    assert diagnostic.linear_solve_calls == diagnostic.linear_solve_returns == 1
    assert diagnostic.line_search_evaluations == 0


@pytest.mark.parametrize("failed_return", ["raise", "nonfinite"])
def test_failed_simplified_solve_owns_its_trial_rhs_and_residual(monkeypatch, failed_return):
    original = conservative.splu
    calls = {"factor": 0, "solve": 0}
    cause = RuntimeError("manufactured_simplified_solve_failure")

    def factor(matrix):
        calls["factor"] += 1
        underlying = original(matrix)
        class Proxy:
            def solve(self, rhs):
                calls["solve"] += 1
                if calls["solve"] == 2:
                    if failed_return == "raise":
                        raise cause
                    return np.full(rhs.shape, np.nan)
                return underlying.solve(rhs)
        return Proxy()

    monkeypatch.setattr(conservative, "splu", factor)
    with pytest.raises(ContractError, match="conservative_sparse_solve_failed") as caught:
        run_fixture(curved_fixture(), 1., globalization="correction")
    diagnostic = caught.value.conservative_diagnostics
    assert calls == {"factor": 1, "solve": 2} and diagnostic.phase == "trial_linear_solve"
    if failed_return == "raise":
        assert caught.value.__cause__ is cause
    assert diagnostic.linear_solve_returns == (1 if failed_return == "raise" else 2)
    assert diagnostic.correction_base.rhs == (1., 0.)
    search, = diagnostic.correction_searches
    assert search.solve.rhs == (0., -2.) and search.solve.rhs_projection_error == (0, 0)
    assert search.solve.solution is search.solve.norm_squared is search.solve.linear_residual_inf is None
    assert search.accepted is None
    assert search.solve.point_identity == diagnostic.point.identity == diagnostic.residual_point_identity
    assert exact(diagnostic.residual) == (0, 2)
    assert diagnostic.last_accepted_point_identity == search.base_point_identity != diagnostic.point.identity
    assert diagnostic.rejected_searches[0].point_identity == diagnostic.point.identity


def test_correction_squared_norm_preserves_extreme_represented_values():
    assert conservative._norm_squared(np.array([2.**900, 2.**-900])) == Fraction(2)**1800+Fraction(2)**-1800
    assert conservative._norm_squared(np.array([2.**-600])) == Fraction(2)**-1200


@pytest.mark.parametrize("mode", ["L2", None])
def test_unrequested_globalization_rejected_before_public_calls(mode):
    fixture = manufactured()
    with pytest.raises(ContractError, match="unsupported_conservative_globalization"):
        run_fixture(fixture, globalization=mode)
    assert fixture[-1] == {"delta": 0, "conservative": 0, "jacobian": 0, "instantaneous": 0}
