"""Manufactured public storage contracts; no coupled device or native solver."""
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction
from hashlib import sha256

import numpy as np
import pytest
from scipy.sparse import csc_matrix

from scripts.benchmarks.contract_prototype import (
    PARTICLE, SECOND, VOLT, AcceptedStep, ContractError, EquationSpec, Layout,
    PhysicalScaling, PhysicalStorage, SparseStructure, StateView, Support,
    ValidatedProblem, VariableSpec,
)
from scripts.benchmarks.conservative_device import EquationBudget, conservative_be_step
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


def test_sub_ulp_increment_and_discrete_history_are_retained():
    fixture = manufactured()
    problem, _, left, _, _, calls = fixture
    before = encode_point(left)
    budgets = {"balance": EquationBudget(PARTICLE, [2.**-180, 2.**-180])}
    step = run_fixture(fixture, budgets=budgets)
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
    assert calls["instantaneous"] == 0 and calls["jacobian"] == 1
    with pytest.raises((TypeError, ValueError), match="init=False"):
        replace(step, history_kind="native-continuous-tangent")


def test_nonbinary_solver_scales_do_not_become_physical_acceptance():
    fixture = manufactured()
    step = run_fixture(fixture, scaling=PhysicalScaling([.7, 1.3], [.1, .3]),
                       budgets={"balance": EquationBudget(PARTICLE, [2.**-175]*2)})
    wanted = (Fraction(1, 2**81), -Fraction(1, 2**81))
    assert all(abs(x-y) <= Fraction(1, 2**175)
               for x, y in zip(exact(step.storage_delta), wanted, strict=True))
    assert any(x != 0 for row in step.iterations for x in row.rhs_projection_error)
    assert all(x <= 1 for x in step.residual_ratios)


def test_fixed_reference_twelve_words_survive_a_step():
    problem, coordinates, reference, scaling, budgets, calls = manufactured()
    words = [np.full(2, 2.**(-20-60*i)) for i in range(12)]
    left, _ = coordinates.trial(reference, FrameInputExpansion(words), 1.)
    before = encode_point(left)
    step = conservative_be_step(problem, coordinates, left, .5, scaling=scaling,
        budgets={"balance": EquationBudget(PARTICLE, [2.**-180]*2)})
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


def test_vector_nonlinear_storage_algebraic_rows_and_sparse_only(monkeypatch):
    fixture = manufactured("exchange")
    def forbidden(*args, **kwargs):
        raise AssertionError("dense fallback")
    monkeypatch.setattr(np.linalg, "solve", forbidden)
    monkeypatch.setattr(np.linalg, "lstsq", forbidden)
    monkeypatch.setattr(csc_matrix, "toarray", forbidden)
    reference = exchange_reference(.25)
    steps = [run_fixture(fixture, .25), run_fixture(fixture, .25,
        scaling=PhysicalScaling([.5, 4., 8.], [2., .25, 4.]))]
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


def test_finite_input_time_storage_and_jacobian_not_tangent_history():
    fixture = manufactured("input_storage")
    problem, coordinates, left, _, _, calls = fixture
    step = run_fixture(fixture, .25, inputs=[.5])
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
def test_failures_do_not_rebase_clip_or_invoke_dense_fallback(failure):
    fixture = manufactured(force=(1., -1.))
    problem, coordinates, left, scaling, budgets, _ = fixture
    before = encode_point(left)
    options = {}
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
    with pytest.raises(ContractError):
        conservative_be_step(problem, coordinates, left, .5, scaling=scaling, budgets=budgets, **options)
    assert encode_point(left) == before


def test_nonphysical_coordinate_chain_is_not_assumed():
    problem, coordinates, left, scaling, budgets, calls = manufactured()
    logarithmic = RelativeCoordinates(coordinates.layout, {"n": "log"})
    with pytest.raises(ContractError, match="linear_physical"):
        conservative_be_step(problem, logarithmic, left, .5, scaling=scaling, budgets=budgets)
    assert calls["conservative"] == 0
