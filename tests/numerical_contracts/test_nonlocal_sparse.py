"""Synthetic source-preparation checks; no physical fixed point is evaluated.

The independent Decimal oracle is built from input words and the historical
trapezoid expression, never from a candidate graph/matrix/factor. WKB checks
here exercise declared path topology only; physical coefficient/derivative
comparisons live in the separately admitted NonlocalGraphV1 driver.
"""

from decimal import Decimal, localcontext
import ast
import inspect
import json

import numpy as np
import pytest

from scripts.benchmarks.contract_prototype import ContractError
from scripts.benchmarks.nonlocal_sparse import (
    NonlocalPlan, auxiliary_matrix, disconnected_inventory, recycling_factors,
    recycling_plan, wkb_plan,
)
from scripts.benchmarks.sparse_prototype import sparse_solve


def decimal_solve(a, b):
    with localcontext() as context:
        context.prec = 70
        rows = [[Decimal(str(v)) for v in row] + [Decimal(str(rhs))]
                for row, rhs in zip(a, b)]
        count = len(rows)
        for column in range(count):
            pivot = max(range(column, count), key=lambda i: abs(rows[i][column]))
            rows[pivot], rows[column] = rows[column], rows[pivot]
            divisor = rows[column][column]
            assert divisor != 0
            rows[column] = [v / divisor for v in rows[column]]
            for row in range(count):
                if row != column:
                    multiple = rows[row][column]
                    rows[row] = [a - multiple * b for a, b in zip(rows[row], rows[column])]
        return np.array([float(row[-1]) for row in rows])


def recycling_input():
    # Deliberately nonuniform x, a noncontiguous mask and thickness != span.
    # These dimensionless words test the historical expression, not a material.
    return dict(x=[0, 1, 3, 4], n=[2, 3, 4, 5], p=[5, 4, 3, 2],
                b_rad=[1, 2, 0, 1], ni_sq=[1, 1, 1, 1],
                masks=[(0, 1, 3), (1, 2, 3)], escape=[.25, .5], thickness=[8, 7])


def decimal_recycling(data):
    """Independent edge-by-edge dense expression, not U/V multiplication."""
    d = lambda v: Decimal(str(v))
    count = len(data["x"])
    with localcontext() as context:
        context.prec = 70
        rate = [d(0) for _ in range(count)]
        jacobian = [[d(0) for _ in range(3 * count)] for _ in range(3 * count)]
        for k, selected in enumerate(data["masks"]):
            factor = (d(1) - d(data["escape"][k])) / d(data["thickness"][k])
            for left, right in zip(selected[:-1], selected[1:]):
                half_width = (d(data["x"][right]) - d(data["x"][left])) / d(2)
                for sender in (left, right):
                    b = d(data["b_rad"][sender])
                    n, p = d(data["n"][sender]), d(data["p"][sender])
                    emission = b * (n * p - d(data["ni_sq"][sender]))
                    for receiver in selected:
                        rate[receiver] += factor * half_width * emission
                        for row in (receiver, count + receiver):
                            jacobian[row][sender] += factor * half_width * b * p
                            jacobian[row][count + sender] += factor * half_width * b * n
        return rate, jacobian


def test_recycling_dense_residual_jacobian_graph_auxiliary_and_independent_solve():
    data = recycling_input()
    rate_ref, jac_ref = decimal_recycling(data)
    generation, u, v, branches = recycling_factors(**data)
    plan = recycling_plan(4, data["masks"])
    graph = plan.compile()
    matrices = {c.id: np.outer(u[:, k], v[:, k]) for k, c in enumerate(plan.couplings)}
    matrix = plan.assemble(graph, matrices, diagonal=4)
    reference = np.array(jac_ref, dtype=float)
    np.testing.assert_allclose(generation, np.array(rate_ref, dtype=float), rtol=2e-15)
    np.testing.assert_allclose(u @ v.T, reference, rtol=2e-15, atol=1e-15)
    np.testing.assert_allclose(matrix.toarray(), reference + 4 * np.eye(12), rtol=2e-15)
    assert branches == ("active_positive_integral",) * 2
    assert matrix[0, 7] != 0  # remotely coupled p at node 3 -> n at node 0
    assert (0, 2) not in {tuple(pair) for pair in graph.edges()}
    # B[2]==0 is not grounds for deleting a declared sender slot.
    assert matrix[1, 2] == 0 and (1, 2) in {tuple(pair) for pair in graph.edges()}
    rhs = [str((i % 4) - 1) for i in range(12)]
    dense = [[value + (Decimal(4) if i == j else Decimal(0))
              for j, value in enumerate(row)] for i, row in enumerate(jac_ref)]
    expected = decimal_solve(dense, rhs)
    solution = sparse_solve(matrix, np.array(rhs, dtype=float))
    auxiliary_graph, auxiliary = auxiliary_matrix(plan, u, v, diagonal=4)
    auxiliary_solution = sparse_solve(auxiliary, np.r_[np.array(rhs, dtype=float), [0, 0]])
    np.testing.assert_allclose(solution, expected, rtol=2e-14, atol=2e-15)
    np.testing.assert_allclose(auxiliary_solution[:12], expected, rtol=2e-14, atol=2e-15)
    np.testing.assert_allclose(auxiliary_solution[12:], v.T @ expected, rtol=2e-14, atol=2e-15)
    assert auxiliary_graph.nnz < graph.nnz + 4  # bordered rather than dense rank expansion


def test_zero_coefficient_and_overlapping_occurrences_preserve_structure():
    data = recycling_input()
    data["masks"] = [(0, 1, 3), (0, 1, 3)]
    data["escape"] = [1, 1]
    rate, u, v, branches = recycling_factors(**data)
    plan = recycling_plan(4, data["masks"])
    graph = plan.compile()
    matrix = plan.assemble(graph, {c.id: np.outer(u[:, k], v[:, k])
                                   for k, c in enumerate(plan.couplings)})
    assert np.all(rate == 0) and matrix.count_nonzero() == 12
    assert matrix.nnz > matrix.count_nonzero()
    assert len(plan.couplings) == 2 and branches == ("inactive_parameter_or_mask",) * 2
    data["escape"] = [.25, .5]
    _, new_u, new_v, _ = recycling_factors(**data)
    changed = plan.assemble(graph, {c.id: np.outer(new_u[:, k], new_v[:, k])
                                    for k, c in enumerate(plan.couplings)})
    assert changed.nnz == matrix.nnz and graph.identity == plan.compile().identity


@pytest.mark.parametrize("change,error", [
    ({"n": [1, 1, 1, 1], "p": [1, 1, 1, 1]}, "nonregular_recycling_threshold"),
    ({"x": [0, 1, 1, 4]}, "nonincreasing_recycling_grid"),
    ({"b_rad": [1, float("nan"), 1, 1]}, "finiteness"),
    ({"escape": [1]}, "shape"),
    ({"masks": [(1, 0), (1, 2, 3)]}, "grid_order"),
])
def test_recycling_branch_and_input_rejections(change, error):
    data = recycling_input()
    data.update(change)
    with pytest.raises(ContractError, match=error):
        recycling_factors(**data)


def test_negative_net_emission_and_nonpositive_thickness_keep_historical_inactivity():
    data = recycling_input()
    data["ni_sq"] = [100] * 4
    data["thickness"] = [8, 0]
    rate, u, v, branches = recycling_factors(**data)
    assert np.all(rate == 0) and np.all(u @ v.T == 0)
    assert branches == ("inactive_negative_integral", "inactive_parameter_or_mask")


def synthetic_wkb_plan(bounds=None):
    return wkb_plan(10, (0, 3, 7, 10),
                    [(1, 7), (2, 6), (2, 5), (3, 5)] if bounds is None else bounds,
                    (1, 9, 4, 0))


def test_wkb_full_region_potential_remote_levels_and_zero_flux_tangent():
    plan = synthetic_wkb_plan()
    graph = plan.compile()
    coupling = plan.couplings[0]
    assert set(range(20, 30)) <= set(coupling.columns)
    assert not set(range(10, 20)) & set(coupling.columns)  # no supported hole channel
    # A purely synthetic conservative transfer with independent endpoint and
    # distant-window derivatives. This checks storage, not physical WKB values.
    transfer = np.zeros(30)
    transfer[1], transfer[8] = 1, -1
    gradient = np.zeros(30)
    gradient[1], gradient[8], gradient[29] = 2, -2, 3
    dense = np.outer(transfer, gradient)
    equilibrium = np.zeros(30)
    assert np.all(dense @ equilibrium == 0) and dense[1, 8] != 0
    matrix = plan.assemble(graph, {coupling.id: dense}, diagonal=2)
    np.testing.assert_array_equal(matrix.toarray(), 2 * np.eye(30) + dense)
    rhs = np.arange(30, dtype=float) / 10
    reference = decimal_solve(2 * np.eye(30) + dense, rhs)
    np.testing.assert_allclose(sparse_solve(matrix, rhs), reference, rtol=2e-14, atol=1e-14)
    assert np.sum(dense, axis=0).max() == 0


def test_moving_energy_paths_invalidate_even_if_aggregate_edges_are_identical():
    original = synthetic_wkb_plan()
    moved = synthetic_wkb_plan([(1, 7), (2, 6), (3, 6), (3, 5)])
    np.testing.assert_array_equal(original.compile().edges(), moved.compile().edges())
    assert original.identity != moved.identity
    with pytest.raises(ContractError, match="structural_recompile_required"):
        moved.assemble(original.compile(), {"wkb.electron": np.zeros((30, 30))})


def test_extremum_selection_invalidates_even_without_a_turning_edge_change():
    original = synthetic_wkb_plan()
    moved = wkb_plan(10, (0, 3, 7, 10), [(1, 7), (2, 6), (2, 5), (3, 5)], (2, 8, 4, 1))
    assert moved.compile().nnz == original.compile().nnz
    assert moved.identity != original.identity


def test_undeclared_nonzero_derivative_is_rejected_without_a_tail_threshold():
    plan = synthetic_wkb_plan()
    jacobian = np.zeros((30, 30))
    jacobian[1, 15] = np.nextafter(0.0, 1.0)
    with pytest.raises(ContractError, match="derivative_outside_declared_support"):
        plan.assemble(plan.compile(), {"wkb.electron": jacobian})


@pytest.mark.parametrize("bounds", [[(0, 9)] * 4, [(2, 3)] * 4, [(1, 7)] * 3,
                                   [(1.1, 7)] * 4, [(2, 6), (1, 7), (2, 5), (3, 5)]])
def test_wkb_unsupported_or_ambiguous_path_declarations_reject(bounds):
    with pytest.raises(ContractError):
        synthetic_wkb_plan(bounds)


def test_json_roundtrip_identity_and_reject_wrong_coordinate_meaning():
    plan = synthetic_wkb_plan()
    document = json.loads(json.dumps(plan.to_document()))
    restored = NonlocalPlan.from_document(document)
    assert restored.identity == plan.identity
    assert restored.compile().identity == plan.compile().identity
    document["coordinate_order"] = ["phi", "n", "p"]
    with pytest.raises(ContractError, match="invalid_nonlocal_document"):
        NonlocalPlan.from_document(document)


def test_exact_zero_diffusion_retains_separate_inventories_and_species_slots():
    faces = [(0, 1), (1, 2), (2, 3)]
    domains, graph, matrix = disconnected_inventory(4, faces, [1, 0, 1], [1, 2, 3, 4])
    assert domains.domains == ((0, 1), (2, 3)) and matrix.shape == (6, 6)
    initial = np.array([2, 0, 4, 0])  # c0 words independent of the D connectivity
    rhs = np.r_[initial, [2, 12]]
    expected = np.block([[np.eye(4), np.array([[1, 0], [2, 0], [0, 3], [0, 4]])],
                         [np.array([[1, 2, 0, 0], [0, 0, 3, 4]]), np.zeros((2, 2))]])
    np.testing.assert_array_equal(matrix.toarray(), expected)
    solution = sparse_solve(matrix, rhs)
    np.testing.assert_allclose(solution, decimal_solve(expected, rhs), atol=1e-14)
    np.testing.assert_allclose(expected[4:, :4] @ solution[:4], [2, 12], atol=1e-14)
    frozen, frozen_graph, frozen_matrix = disconnected_inventory(4, faces, [0, -0., 0], [1, 2, 3, 4])
    assert frozen.domains == ((0,), (1,), (2,), (3,)) and frozen_matrix.shape == (8, 8)
    assert frozen_graph.identity != graph.identity
    mobile, _, _ = disconnected_inventory(4, faces, [1, np.nextafter(0., 1.), 1], [1, 2, 3, 4])
    assert mobile.domains == ((0, 1, 2, 3),)


def test_source_preparation_imports_no_legacy_engine_and_bounds_allocation():
    import scripts.benchmarks.nonlocal_sparse as module
    imports = [node.module for node in ast.walk(ast.parse(inspect.getsource(module)))
               if isinstance(node, ast.ImportFrom)]
    assert not any(name.startswith("perovskite_sim") for name in imports)
    with pytest.raises(ContractError, match="size_limit"):
        recycling_plan(129, [(0, 128)])


def test_support_document_owns_indices_and_rejects_reserved_identity():
    document = synthetic_wkb_plan().to_document()
    restored = NonlocalPlan.from_document(document)
    identity = restored.identity
    document["couplings"][0]["rows"][0] = 9
    assert restored.identity == identity
    document["couplings"][0]["id"] = "identity"
    with pytest.raises(ContractError, match="invalid_nonlocal_document"):
        NonlocalPlan.from_document(document)
