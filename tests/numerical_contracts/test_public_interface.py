"""Independent small-polynomial and exact-word public interface controls.

No native solver is used. The polynomial reference is separate from callback
assembly, and the DD checks compare exact represented binary rational inputs.
"""

from dataclasses import replace
from decimal import Decimal, localcontext
from hashlib import sha256

import numpy as np
import pytest

from scripts.benchmarks.contract_prototype import (
    AREA, COULOMB, ONE, SECOND, BalanceTerms, ContractError, EquationSpec,
    ImplicitSystem, Layout, LinearCoordinates, PhysicalScaling, PhysicalStorage,
    SparseLinearization, SparseStructure, StateView, StoragePartials, Support,
    TerminalPort, ValidatedProblem, VariableSpec,
)
from scripts.benchmarks.precision_prototype import (
    DD, DoubleArithmetic, DoubleArray, RelativeCoordinates,
)
from scripts.benchmarks.sparse_prototype import SparseGraph, TermSupport


def polynomial_problem():
    """Q=x²+a*x+t², R=3*x+2*a+t, g=z-a*x-t, in named SI units."""
    layout = Layout(
        (Support("node", "global", (1,)),),
        (VariableSpec("x", "polynomial", "node", (1,), ONE),
         VariableSpec("z", "constraint", "node", (1,), ONE, role="constraint")),
        (EquationSpec("balance", "polynomial", "node", (1,), ONE / SECOND,
                      derivative_support=("x", "z")),
         EquationSpec("constraint", "constraint", "node", (1,), ONE, role="constraint",
                      derivative_support=("x", "z"))),
    )
    terms = tuple(TermSupport(f"{eq.id}:{var.id}", eq.owner, eq.id, var.id,
                              np.array([[0, 0]]), unit=eq.unit / var.unit)
                  for eq in layout.equations for var in layout.variables)
    graph = SparseGraph(layout, terms, topology_identity="full-polynomial-union-v1")
    structure = SparseStructure.from_graph(graph)
    source = sha256(b"independent-polynomial-callbacks-v1").hexdigest()
    coordinates = LinearCoordinates(layout)
    calls = {name: 0 for name in ("values", "linearize", "storage", "delta", "conservative")}

    def storage_value(point):
        calls["storage"] += 1
        x, a, t = point.y[0], point.inputs[0], point.time
        return np.array([x*x+a*x+t*t])

    def storage_delta(left, right, increment):
        calls["delta"] += 1
        dx = increment.field("x").values[0]
        da, dt = right.inputs[0]-left.inputs[0], right.time-left.time
        return np.array([(right.y[0]+left.y[0]+right.inputs[0])*dx
                         + left.y[0]*da + (right.time+left.time)*dt])

    def residual(point, ydot, adot):
        calls["values"] += 1
        x, z = point.y
        a, t, dx, da = point.inputs[0], point.time, ydot[0], adot[0]
        return np.array([(2*x+a)*dx+x*da+2*t-(3*x+2*a+t), z-a*x-t])

    def linearize(point, ydot, adot):
        calls["linearize"] += 1
        x, a, dx, da = point.y[0], point.inputs[0], ydot[0], adot[0]
        return SparseLinearization(structure.filled([2*dx+da-3, -a, 0, 1]),
                                   structure.filled([2*x+a, 0, 0, 0]),
                                   np.array([[dx-2], [-x]]), np.array([[x], [0]]),
                                   np.array([1, -1]), structure, source)

    def conservative(left, right, increment):
        calls["conservative"] += 1
        x, z = right.y
        a, t, h = right.inputs[0], right.time, right.time-left.time
        return np.r_[storage_delta(left, right, increment)-h*(3*x+2*a+t), z-a*x-t]

    def conservative_jacobian(left, right):
        x, a, h = right.y[0], right.inputs[0], right.time-left.time
        return structure.filled([2*x+a-3*h, -a, 0, 1])

    storage = PhysicalStorage(storage_value, storage_delta, (ONE,), source)
    problem = ValidatedProblem(layout, 1, 1, storage, structure, source, residual,
                               linearize, conservative, conservative_jacobian)
    return coordinates, problem, calls


def test_sparse_residual_requests_no_storage_derivatives_or_linearization():
    coordinates, problem, calls = polynomial_problem()
    point = coordinates.point([2, 5], 0.25, [0.5])
    for _ in range(4):
        np.testing.assert_array_equal(problem.residual(point, np.array([0.5, -1]), np.array([2.])),
                                      [-0.5, 3.75])
    assert calls == {"values": 4, "linearize": 0, "storage": 0, "delta": 0, "conservative": 0}


@pytest.mark.parametrize("cj", [0, -1, 0.25, 100])
def test_sparse_complete_chain_rule_against_independent_decimal_polynomial(cj):
    coordinates, problem, _ = polynomial_problem()
    point = coordinates.point([2, 5], 0.25, [0.5])
    lin = problem.linearize(point, np.array([0.5, -1]), np.array([2.]))
    np.testing.assert_array_equal(lin.y.toarray(), [[0, 0], [-0.5, 1]])
    np.testing.assert_array_equal(lin.ydot.toarray(), [[4.5, 0], [0, 0]])
    np.testing.assert_array_equal(lin.inputs, [[-1.5], [-2]])
    np.testing.assert_array_equal(lin.input_rate, [[2], [0]])
    np.testing.assert_array_equal(lin.time, [1, -1])
    # Expand the polynomial at exact rational inputs and joint perturbations.
    with localcontext() as context:
        context.prec = 80
        h = Decimal("0.0001")
        def reference(e):
            x, z, a, t = 2+e, 5-2*e, Decimal("0.5")+3*e, Decimal("0.25")+4*e
            dx, da = Decimal("0.5")+5*e, 2+6*e
            return [(2*x+a)*dx+x*da+t-3*x-2*a, z-a*x-t]
        finite = [(p-m)/(2*h) for p, m in zip(reference(h), reference(-h))]
    assembled = (lin.y @ [1, -2] + lin.ydot @ [5, 0] + lin.inputs[:, 0]*3
                 + lin.input_rate[:, 0]*6 + lin.time*4)
    np.testing.assert_array_equal(assembled, [float(v) for v in finite])
    matrix = lin.ida_matrix(cj)
    assert matrix.nnz == 4  # Retain all named slots, including structural zeros.
    np.testing.assert_array_equal(matrix.toarray(), [[4.5*cj, 0], [-0.5, 1]])


def test_sparse_finite_storage_and_unmultiplied_constraint_rows():
    coordinates, problem, _ = polynomial_problem()
    left = coordinates.point([2, 5], 0.25, [0.5])
    right, increment = coordinates.advance(left, [0.5, -1], 0.75, [1])
    np.testing.assert_array_equal(problem.storage.value(right), [9.3125])
    np.testing.assert_array_equal(problem.storage.delta(left, right, increment), [4.25])
    np.testing.assert_array_equal(problem.conservative_residual(left, right, increment), [-0.875, 0.75])
    np.testing.assert_array_equal(problem.conservative_jacobian(left, right).toarray(), [[4.5, 0], [-1, 1]])


def test_sparse_sources_shapes_finite_values_and_structural_slots_fail_closed():
    coordinates, problem, _ = polynomial_problem()
    point = coordinates.point([2, 5], 0.25, [0.5])
    rate, adot = np.zeros(2), np.zeros(1)
    for value in (np.zeros(1), np.zeros((2, 1)), np.array([np.nan, 0])):
        with pytest.raises(ContractError):
            replace(problem, residual_values=lambda *args: value).residual(point, rate, adot)
    with pytest.raises(ContractError, match="problem_source_mismatch"):
        replace(problem, source_identity="wrong")
    with pytest.raises(ContractError, match="storage_unit_mismatch"):
        replace(problem, storage=replace(problem.storage, units=(COULOMB,)))
    with pytest.raises(ContractError, match="problem_equation_order_mismatch"):
        replace(problem, storage_count=2)
    with pytest.raises(ContractError, match="rate_shape_mismatch"):
        problem.residual(point, np.zeros(1), adot)
    lin = problem.linearize(point, rate, adot)
    with pytest.raises(ContractError, match="linearization_source_or_structure_mismatch"):
        replace(problem, analytic_linearization=lambda *args: replace(lin, source_identity="other")).linearize(point, rate, adot)
    pruned = lin.ydot
    pruned.eliminate_zeros()
    with pytest.raises(ContractError, match="structural_recompile_required"):
        replace(lin, ydot=pruned)
    with pytest.raises(ContractError, match="sparse_matrix_shape_or_format"):
        replace(lin, y=np.zeros((2, 2)))
    with pytest.raises(ContractError, match="equation_derivative_shape_mismatch"):
        replace(lin, input_rate=np.zeros((2, 2)))
    with pytest.raises(ContractError, match="nonfinite_cj"):
        lin.ida_matrix(np.nan)


def test_sparse_metadata_and_values_are_owned_and_rectangular_storage_is_supported():
    coordinates, problem, _ = polynomial_problem()
    point = coordinates.point([1, 2], inputs=[0])
    lin = problem.linearize(point, np.zeros(2), np.zeros(1))
    matrix = lin.y
    matrix.data[:] = 999
    np.testing.assert_array_equal(lin.y.toarray(), [[-3, 0], [0, 1]])
    indices = problem.structure.indices
    indices.shape = (2, 2)
    assert problem.structure.indices.shape == (4,)
    rectangular = SparseStructure((1, 2), [0], [0, 1, 1], "Qy-declared", problem.layout.identity)
    assert rectangular.filled([2]).shape == (1, 2)
    for indices, indptr in (([0.5], [0, 1]), ([0, 0], [0, 2]), ([2], [0, 1])):
        with pytest.raises(ContractError):
            SparseStructure((2, 1), indices, indptr, "invalid", problem.layout.identity)


def exact_words(value):
    return [Decimal.from_float(float(h))+Decimal.from_float(float(l))
            for h, l in zip(value.high.ravel(), value.low.ravel())]


def dd_system():
    layout = Layout((Support("cell", "global", (1,)),),
                    (VariableSpec("n", "carrier", "cell", (1,), ONE, lower=0),),
                    (EquationSpec("balance", "carrier", "cell", (1,), ONE / SECOND,
                                  derivative_support=("n",)),))
    coordinates = RelativeCoordinates(layout, {"n": "log"})
    point = coordinates.initial(StateView(layout, [("n", DoubleArray([1e22], [1e5]))]))
    a = DoubleArithmetic()

    class Storage:
        def value(self, p):
            return p.state.field("n")

        def delta(self, left, right, increment):
            increment.validate(left, right)
            return increment.field("n")

        def first(self, p):
            q = self.value(p).as_dd()
            return StoragePartials(DoubleArray.from_dd(q.reshape(1, 1)), DoubleArray(np.zeros((1, 0))),
                                   DoubleArray([0]))

        def rate_jvp(self, p, ydot, adot, dy, da, dt):
            return DoubleArray.from_dd(self.value(p).as_dd() * ydot[0] * dy[0])

    def terms(p):
        rate = DD([1e22]) * (p.state.field("n").as_dd()/DD([1e22], [1e5]))
        return BalanceTerms(DoubleArray.from_dd(rate), DoubleArray([]),
                            DoubleArray.from_dd(rate.reshape(1, 1)), DoubleArray(np.zeros((1, 0))),
                            DoubleArray([0]), DoubleArray(np.zeros((0, 1))),
                            DoubleArray(np.zeros((0, 0))), DoubleArray([]))

    return coordinates, point, ImplicitSystem(Storage(), terms, a), a


def test_dd_residual_jacobian_ports_and_scaling_retain_low_words_before_cancellation():
    coordinates, point, system, a = dd_system()
    with localcontext() as context:
        context.prec = 80
        residual = system.residual(point, np.ones(1), np.empty(0))
        assert exact_words(residual) == [Decimal(100000)]
        linearization = system.linearize(point, np.ones(1), np.empty(0))
        assert exact_words(linearization.y) == [Decimal(100000)]
        assert exact_words(linearization.ida_matrix(1)) == [Decimal(10**22+200000)]
        port = TerminalPort("left", -1, 2)
        assert exact_words(port.current(system.storage.value(point), DoubleArray([-1e22]), arithmetic=a)) == [Decimal(200000)]
        assert exact_words(port.charge(DoubleArray([1], [1e-17]), arithmetic=a)) == [Decimal(2)+2*Decimal.from_float(1e-17)]
        scaling = PhysicalScaling([2], [4])
        assert exact_words(scaling.residual(system.storage.value(point), arithmetic=a)) == [Decimal(5*10**21+50000)]
        assert exact_words(scaling.jacobian(linearization.ida_matrix(1), arithmetic=a)) == [Decimal(2*10**22+400000)]
        right, increment = coordinates.advance(point, [1e-17], 1)
        qdelta = system.storage.delta(point, right, increment)
        expected = (Decimal(10**22+100000) * Decimal.from_float(1e-17).exp()
                    - Decimal(10**22+100000))
        assert abs(exact_words(qdelta)[0]-expected) < Decimal("1e-20")
        result = system.conservative_residual(point, right, increment)
        assert isinstance(result, DoubleArray)
        assert exact_words(result)[0] == exact_words(DoubleArray.from_dd(qdelta.as_dd()-system.terms(right).rate.as_dd()))[0]
    for value in (residual, linearization.y, qdelta, result):
        with pytest.raises(TypeError, match="implicit precision rounding"):
            np.asarray(value)


def test_dd_paths_are_explicit_and_port_unit_shape_failures_remain_failures():
    _, point, system, a = dd_system()
    with pytest.raises(TypeError, match="implicit precision rounding"):
        replace(system, arithmetic=None).residual(point, np.ones(1), np.empty(0))
    port = TerminalPort("left", -1, 1)
    with pytest.raises(TypeError, match="implicit precision rounding"):
        port.current(DoubleArray([1]), DoubleArray([0]))
    with pytest.raises(ContractError, match="port_shape_mismatch"):
        port.current(DoubleArray([1]), DoubleArray([0, 0]), arithmetic=a)
    with pytest.raises(ContractError, match="port_unit_mismatch"):
        port.charge(DoubleArray([1]), arithmetic=a, unit=COULOMB)
    with pytest.raises(ContractError, match="port_unit_mismatch"):
        port.current([1], [1], displacement_rate_unit=COULOMB / AREA)
    with pytest.raises(ContractError, match="invalid_solver_scale"):
        PhysicalScaling([0], [1])
    with pytest.raises(ContractError, match="scale_shape_mismatch"):
        PhysicalScaling([1], [1]).jacobian(DoubleArray([1]), arithmetic=a)


def test_sparse_scaling_keeps_zero_slots_and_valid_float_port_behavior():
    _, problem, _ = polynomial_problem()
    matrix = problem.structure.filled([2, 0, 0, 4])
    result = PhysicalScaling([2, 4], [3, 5]).jacobian(matrix)
    assert result.nnz == 4
    np.testing.assert_array_equal(result.toarray(), [[3, 0], [0, 5]])
    np.testing.assert_array_equal(TerminalPort("right", 1, 2).current([3, 4], [1, 2]), [-8, -12])
