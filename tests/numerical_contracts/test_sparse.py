"""Independent frozen NC07 oracles and negative structural-support checks."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from decimal import Decimal, localcontext
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.sparse import csc_matrix

from scripts.benchmarks.contract_prototype import (
    LENGTH,
    ONE,
    ContractError,
    EquationSpec,
    Layout,
    Support,
    VariableSpec,
)
from scripts.benchmarks.sparse_prototype import (
    SparseGraph,
    TermSupport,
    factorize,
    ion_domains,
    require_rank_one_invertible,
    sparse_solve,
)

GATE_PATH = Path(__file__).with_name("AnalyticGatesV1.json")
GATE_SHA256 = "43c9ba4ab1b6a95b3e278efcd8eed9a67dc4d5e258295c8e6961c53f7ee57201"
assert sha256(GATE_PATH.read_bytes()).hexdigest() == GATE_SHA256
GATE = next(
    item for item in json.loads(GATE_PATH.read_text())["gates"] if item["id"] == "NC07"
)
INPUT = GATE["fixed_inputs"]
OBSERVATIONS: list[dict[str, Any]] = []


@pytest.fixture(scope="module", autouse=True)
def frozen_attempt(request: pytest.FixtureRequest):
    manifest_path = os.environ.get("SOLARLAB_SPARSE_MANIFEST")
    if manifest_path:
        manifest = json.loads(Path(manifest_path).read_text())
        for path, expected in manifest["source_sha256"].items():
            assert sha256(Path(path).read_bytes()).hexdigest() == expected
        assert manifest["gate_sha256"] == GATE_SHA256
    yield
    output = os.environ.get("SOLARLAB_SPARSE_METRICS")
    if output:
        Path(output).write_text(
            json.dumps(
                {
                    "gate_id": "NC07",
                    "gate_sha256": GATE_SHA256,
                    "observations": OBSERVATIONS,
                    "pytest_failures": request.session.testsfailed,
                },
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )


def D(value: Any) -> Decimal:
    return Decimal(str(value))


def decimal_dense_solve(
    matrix: list[list[Decimal]], rhs: list[Decimal]
) -> list[Decimal]:
    """Independent Decimal80 Gaussian elimination, with no candidate matrices."""
    with localcontext() as context:
        context.prec = 80
        work = [list(row) + [value] for row, value in zip(matrix, rhs)]
        n = len(rhs)
        for column in range(n):
            pivot = max(range(column, n), key=lambda row: abs(work[row][column]))
            if work[pivot][column] == 0:
                raise ValueError("singular decimal reference")
            work[column], work[pivot] = work[pivot], work[column]
            for row in range(column + 1, n):
                multiplier = work[row][column] / work[column][column]
                for entry in range(column, n + 1):
                    work[row][entry] -= multiplier * work[column][entry]
        solution = [D(0)] * n
        for row in range(n - 1, -1, -1):
            solution[row] = (
                work[row][n]
                - sum(
                    work[row][column] * solution[column] for column in range(row + 1, n)
                )
            ) / work[row][row]
        return solution


def decimal_reference(border: str | None = None):
    with localcontext() as context:
        context.prec = 80
        alpha = D(INPUT["alpha"])
        u, v, rhs = ([D(value) for value in INPUT[name]] for name in ("u", "v", "b"))
        n = len(rhs)
        matrix = [
            [(alpha if i == j else D(0)) - u[i] * v[j] for j in range(n)]
            for i in range(n)
        ]
        dot_vu, dot_vb = (
            sum(a * b for a, b in zip(v, u)),
            sum(a * b for a, b in zip(v, rhs)),
        )
        sm = [
            rhs[i] / alpha + u[i] * dot_vb / (alpha * (alpha - dot_vu))
            for i in range(n)
        ]
        direct = decimal_dense_solve(matrix, rhs)
        assert max(abs(a - b) for a, b in zip(sm, direct)) < D("1e-70")
        if border == "auxiliary":
            matrix = [
                [alpha if i == j else D(0) for j in range(n)] + [-u[i]]
                for i in range(n)
            ]
            matrix.append([-value for value in v] + [D(1)])
            rhs = rhs + [D(0)]
        elif border == "inventory":
            matrix = [row + [D(1)] for row in matrix] + [[D(1)] * n + [D(0)]]
            rhs = rhs + [D(4)]
        elif border == "port":
            p, q = [D(1), D(0), D(0), D(-1)], [D("0.5"), D(0), D(0), D("-0.5")]
            matrix = [row + [p[i]] for i, row in enumerate(matrix)] + [q + [D(-2)]]
            rhs = rhs + [D("0.125")]
        result = decimal_dense_solve(matrix, rhs)
        if border == "auxiliary":
            assert max(abs(a - b) for a, b in zip(sm, result[:n])) < D("1e-70")
        return matrix, rhs, result


def assert_gate(candidate: np.ndarray, matrix, rhs, reference) -> tuple[float, float]:
    assert candidate.shape == (len(reference),) and np.isfinite(candidate).all()
    with localcontext() as context:
        context.prec = 80
        values = [Decimal.from_float(float(value)) for value in candidate]
        errors = [abs(value - expected) for value, expected in zip(values, reference)]
        assert all(
            error <= D(GATE["atol"]) + D(GATE["rtol"]) * abs(expected)
            for error, expected in zip(errors, reference)
        )
        residual = [
            abs(sum(a * x for a, x in zip(row, values)) - b)
            for row, b in zip(matrix, rhs)
        ]
        assert max(residual) <= D(GATE["atol"])
        return float(max(errors)), float(max(residual))


def small_layout(border: str | None = None, *, tensor: bool = False) -> Layout:
    shape = (2, 2) if tensor else (4,)
    variables = [VariableSpec("00_state", "bulk", "cells", shape, ONE)]
    dependencies = ("00_state",) if border is None else ("00_state", "10_" + border)
    equations = [
        EquationSpec(
            "00_balance", "bulk", "cells", shape, ONE, derivative_support=dependencies
        )
    ]
    supports = [Support("cells", "cell", shape)]
    if border:
        supports.append(Support("global", "global", (1,)))
        variables.append(
            VariableSpec("10_" + border, border, "global", (1,), ONE, role="constraint")
        )
        derivatives = ("00_state",) if border == "inventory" else dependencies
        equations.append(
            EquationSpec(
                "10_" + border,
                border,
                "global",
                (1,),
                ONE,
                role="constraint",
                derivative_support=derivatives,
            )
        )
    return Layout(tuple(supports), tuple(variables), tuple(equations))


def small_case(border: str | None = None, *, tensor: bool = False):
    alpha, u, v = float(INPUT["alpha"]), np.asarray(INPUT["u"]), np.asarray(INPUT["v"])
    require_rank_one_invertible(alpha, u, v)
    nodes = np.arange(4)
    layout = small_layout(border, tensor=tensor)
    terms = [
        TermSupport(
            "diagonal",
            "bulk",
            "00_balance",
            "00_state",
            np.column_stack((nodes, nodes)),
        )
    ]
    values = {"diagonal": np.full(4, alpha)}
    if border != "auxiliary":
        term = TermSupport.cartesian(
            "nonlocal",
            "bulk",
            "00_balance",
            "00_state",
            INPUT["declared_u_support"],
            INPUT["declared_v_support"],
        )
        terms.append(term)
        values[term.id] = -u[term.pairs[:, 0]] * v[term.pairs[:, 1]]
    if border:
        name = "10_" + border
        columns = (
            INPUT["declared_u_support"]
            if border == "auxiliary"
            else (nodes if border == "inventory" else [0, 3])
        )
        rows = INPUT["declared_v_support"] if border == "auxiliary" else columns
        kind = "nonlocal" if border == "auxiliary" else border
        terms.extend(
            (
                TermSupport.cartesian(
                    "border.column", "bulk", "00_balance", name, columns, [0], kind=kind
                ),
                TermSupport.cartesian(
                    "border.row", border, name, "00_state", [0], rows, kind=kind
                ),
            )
        )
        values["border.column"] = (
            -u
            if border == "auxiliary"
            else (np.ones(4) if border == "inventory" else np.array([1.0, -1.0]))
        )
        values["border.row"] = (
            -v
            if border == "auxiliary"
            else (np.ones(4) if border == "inventory" else np.array([0.5, -0.5]))
        )
        if border != "inventory":
            terms.append(
                TermSupport("border.scalar", border, name, name, [[0, 0]], kind)
            )
            values["border.scalar"] = np.array([1.0 if border == "auxiliary" else -2.0])
    terms = tuple(terms)
    return SparseGraph(layout, terms), terms, values


@pytest.mark.parametrize("border", [None, "auxiliary", "inventory", "port"])
def test_nc07_sparse_and_dense_against_independent_decimal80(border):
    graph, terms, values = small_case(border)
    matrix = graph.assemble(values, supports=terms)
    reference_matrix, reference_rhs, reference = decimal_reference(border)
    rhs = np.array([float(value) for value in reference_rhs])
    expected = {(i, j) for i in range(4) for j in range(4)}
    if border == "auxiliary":
        expected = (
            {(i, i) for i in range(4)}
            | {(i, 4) for i in range(4)}
            | {(4, j) for j in range(4)}
            | {(4, 4)}
        )
    elif border == "inventory":
        expected |= {(i, 4) for i in range(4)} | {(4, j) for j in range(4)}
    elif border == "port":
        expected |= {(0, 4), (3, 4), (4, 0), (4, 3), (4, 4)}
    assert {tuple(pair) for pair in graph.edges()} == expected
    assert matrix.nnz == graph.nnz == len(expected)
    for method, solution in (
        ("sparse", sparse_solve(matrix, rhs)),
        ("dense_candidate", np.linalg.solve(matrix.toarray(), rhs)),
    ):
        state_error, residual = assert_gate(
            solution, reference_matrix, reference_rhs, reference
        )
        OBSERVATIONS.append(
            {
                "case": border or "rank_one",
                "method": method,
                "structural_nnz": graph.nnz,
                "numeric_nnz": int(matrix.count_nonzero()),
                "max_state_error": state_error,
                "max_independent_residual": residual,
                "graph_identity": graph.identity,
            }
        )


def test_zero_current_coefficient_keeps_all_declared_remote_edges():
    graph, terms, values = small_case()
    matrix = graph.assemble(values, supports=terms)
    assert matrix.nnz == 16 and matrix.count_nonzero() == 13
    assert {(1, column) for column in range(4)} <= {
        tuple(pair) for pair in graph.edges()
    }
    changed = {name: value.copy() for name, value in values.items()}
    changed["nonlocal"][4:8] = -0.75 * np.asarray(INPUT["v"])
    next_matrix = graph.assemble(changed, supports=terms)
    assert next_matrix.nnz == next_matrix.count_nonzero() == 16
    assert graph.identity == SparseGraph(graph.layout, terms).identity


def test_numeric_nonzero_mask_cannot_replace_a_frozen_support_declaration():
    graph, terms, values = small_case()
    bad = tuple(
        replace(term, pairs=term.pairs[values[term.id] != 0])
        if term.id == "nonlocal"
        else term
        for term in terms
    )
    with pytest.raises(ContractError, match="structural_recompile_required"):
        graph.assemble(values, supports=bad)


def test_registration_and_pair_order_are_deterministic():
    graph, terms, values = small_case("auxiliary")
    layout = replace(
        graph.layout,
        supports=graph.layout.supports[::-1],
        variables=graph.layout.variables[::-1],
        equations=graph.layout.equations[::-1],
    )
    reordered = tuple(replace(term, pairs=term.pairs[::-1]) for term in terms[::-1])
    other = SparseGraph(layout, reordered)
    assert other.identity == graph.identity
    np.testing.assert_array_equal(other.indices, graph.indices)
    np.testing.assert_array_equal(other.indptr, graph.indptr)
    matrix = graph.assemble(
        {name: value[::-1] for name, value in values.items()}, supports=reordered
    )
    np.testing.assert_array_equal(
        matrix.toarray(), graph.assemble(values, supports=terms).toarray()
    )


def test_tensor_support_has_explicit_c_order_flat_indices():
    flat, terms, values = small_case()
    tensor, tensor_terms, _ = small_case(tensor=True)
    assert flat.identity != tensor.identity
    np.testing.assert_array_equal(
        flat.assemble(values, supports=terms).toarray(),
        tensor.assemble(values, supports=tensor_terms).toarray(),
    )


def test_moving_support_forces_recompile_even_with_same_aggregate_graph():
    graph, terms, values = small_case()
    first, second = INPUT["moving_support"]
    moving = TermSupport(
        "moving_wkb", "bulk", "00_balance", "00_state", first, "nonlocal"
    )
    original, updated = terms + (moving,), terms + (replace(moving, pairs=second),)
    old_graph, new_graph = (
        SparseGraph(graph.layout, original),
        SparseGraph(graph.layout, updated),
    )
    assert old_graph.identity != new_graph.identity
    np.testing.assert_array_equal(old_graph.edges(), new_graph.edges())
    with pytest.raises(ContractError, match="structural_recompile_required"):
        old_graph.assemble({**values, "moving_wkb": [0.0]}, supports=updated)
    assert (
        new_graph.assemble({**values, "moving_wkb": [0.0]}, supports=updated).nnz == 16
    )


@pytest.mark.parametrize("case", INPUT["domain_cases"])
def test_exact_blocking_splits_domains_but_tiny_positive_faces_connect(case):
    partition = ion_domains(4, [[0, 1], [1, 2], [2, 3]], case["D_faces"])
    assert partition.domains == tuple(tuple(group) for group in case["domains"])
    baseline = ion_domains(4, [[0, 1], [1, 2], [2, 3]], [1.0, 1.0, 1.0])
    assert (partition.identity == baseline.identity) == (case["D_faces"][1] != 0)
    graph, terms, values = small_case()
    compiled = SparseGraph(graph.layout, terms, topology_identity=baseline.identity)
    if partition.identity != baseline.identity:
        with pytest.raises(ContractError, match="structural_recompile_required"):
            compiled.assemble(
                values, supports=terms, topology_identity=partition.identity
            )
    OBSERVATIONS.append(
        {
            "case": "ion_domains",
            "D_faces": case["D_faces"],
            "domains": partition.domains,
            "topology_identity": partition.identity,
        }
    )


def test_zero_and_subnormal_diffusivity_are_not_collapsed_by_a_threshold():
    zero = ion_domains(2, [[0, 1]], [0.0])
    tiny = ion_domains(2, [[0, 1]], [np.nextafter(0.0, 1.0)])
    assert zero.domains == ((0,), (1,)) and tiny.domains == ((0, 1),)


@pytest.mark.parametrize("case", INPUT["domain_cases"])
def test_each_physical_domain_has_its_own_inventory_border(case):
    faces = [[0, 1], [1, 2], [2, 3]]
    partition = ion_domains(4, faces, case["D_faces"])
    expected_domains = case["domains"]
    globals = tuple("10_inventory_" + str(i) for i in range(len(partition.domains)))
    variables = [VariableSpec("00_state", "bulk", "cells", (4,), ONE)]
    equations = [
        EquationSpec(
            "00_balance",
            "bulk",
            "cells",
            (4,),
            ONE,
            derivative_support=("00_state",) + globals,
        )
    ]
    terms = [
        TermSupport(
            "mass",
            "bulk",
            "00_balance",
            "00_state",
            np.column_stack((np.arange(4), np.arange(4))),
        )
    ]
    values = {"mass": np.full(4, 2.0)}
    for name, members in zip(globals, partition.domains):
        variables.append(
            VariableSpec(name, name, "global", (1,), ONE, role="constraint")
        )
        equations.append(
            EquationSpec(
                name,
                name,
                "global",
                (1,),
                ONE,
                role="constraint",
                derivative_support=("00_state",),
            )
        )
        terms.extend(
            (
                TermSupport.cartesian(
                    name + ".column",
                    "bulk",
                    "00_balance",
                    name,
                    members,
                    [0],
                    kind="inventory",
                ),
                TermSupport.cartesian(
                    name + ".row",
                    name,
                    name,
                    "00_state",
                    [0],
                    members,
                    kind="inventory",
                ),
            )
        )
        values[name + ".column"] = np.ones(len(members))
        values[name + ".row"] = np.ones(len(members))
    by_face = {tuple(pair): value for pair, value in zip(faces, case["D_faces"])}
    for index, (left, right) in enumerate(partition.active_faces):
        name = "diffusion_" + str(index)
        terms.append(
            TermSupport(
                name,
                "bulk",
                "00_balance",
                "00_state",
                [[left, left], [left, right], [right, left], [right, right]],
            )
        )
        values[name] = by_face[(left, right)] * np.array([1.0, -1.0, -1.0, 1.0])
    layout = Layout(
        (Support("cells", "cell", (4,)), Support("global", "global", (1,))),
        tuple(variables),
        tuple(equations),
    )
    graph = SparseGraph(layout, terms, topology_identity=partition.identity)
    matrix = graph.assemble(
        values, supports=terms, topology_identity=partition.identity
    )
    with localcontext() as context:
        context.prec = 80
        size = 4 + len(expected_domains)
        reference_matrix = [[D(0) for _ in range(size)] for _ in range(size)]
        expected_edges = {(i, i) for i in range(4)}
        for i in range(4):
            reference_matrix[i][i] = D(2)
        for (left, right), diffusion in zip(faces, case["D_faces"]):
            if diffusion == 0:
                continue
            value = D(diffusion)
            reference_matrix[left][left] += value
            reference_matrix[right][right] += value
            reference_matrix[left][right] -= value
            reference_matrix[right][left] -= value
            expected_edges |= {(left, right), (right, left)}
        for column, members in enumerate(expected_domains, 4):
            for member in members:
                reference_matrix[column][member] = D(1)
                reference_matrix[member][column] = D(1)
                expected_edges |= {(column, member), (member, column)}
        reference_rhs = [D(value) for value in INPUT["b"]] + [
            D(len(members)) for members in expected_domains
        ]
        reference = decimal_dense_solve(reference_matrix, reference_rhs)
    assert {tuple(pair) for pair in graph.edges()} == expected_edges
    solution = sparse_solve(matrix, [float(value) for value in reference_rhs])
    error, residual = assert_gate(solution, reference_matrix, reference_rhs, reference)
    OBSERVATIONS.append(
        {
            "case": "domain_inventory_borders",
            "D_faces": case["D_faces"],
            "domains": partition.domains,
            "unknowns": len(solution),
            "structural_nnz": graph.nnz,
            "max_state_error": error,
            "max_independent_residual": residual,
        }
    )


@pytest.mark.parametrize("attribute", ["indices", "indptr", "keys"])
def test_graph_arrays_own_values_and_metadata(attribute):
    graph, _, _ = small_case()
    before = getattr(graph, attribute).copy()
    view = getattr(graph, attribute)
    with pytest.raises(ValueError):
        view.flat[0] = 99
    with pytest.raises(ValueError):
        view.setflags(write=True)
    view.shape = (1, view.size)
    view.dtype = np.uint8
    actual = getattr(graph, attribute)
    assert actual.shape == before.shape and actual.dtype == before.dtype
    np.testing.assert_array_equal(actual, before)


def test_term_domain_and_matrix_arrays_do_not_mutate_owned_graph_state():
    source = np.array([[0, 0], [1, 1]])
    term = TermSupport("term", "bulk", "00_balance", "00_state", source)
    source[:] = 3
    view = term.pairs
    view.shape = (4,)
    view.dtype = np.uint8
    np.testing.assert_array_equal(term.pairs, [[0, 0], [1, 1]])
    partition = ion_domains(4, [[0, 1], [2, 3]], [1, 1])
    labels = partition.labels
    labels.shape = (2, 2)
    labels.dtype = np.uint8
    np.testing.assert_array_equal(partition.labels, [0, 0, 1, 1])
    graph, terms, values = small_case()
    matrix = graph.assemble(values, supports=terms)
    before = graph.indices.copy()
    matrix.indices[:] = 0
    values["diagonal"][:] = 99
    np.testing.assert_array_equal(graph.indices, before)


@pytest.mark.parametrize(
    "pairs",
    [
        [[0.5, 0]],
        [[np.nan, 0]],
        [[True, False]],
        [[1j, 0]],
        [["0", "1"]],
        np.array([[np.iinfo(np.uint64).max, 0]], dtype=np.uint64),
        [[2**80, 0]],
    ],
)
def test_invalid_indices_reuse_core_rejection(pairs):
    with pytest.raises(ContractError, match="invalid_index"):
        TermSupport("bad", "bulk", "00_balance", "00_state", pairs)


@pytest.mark.parametrize("pairs", [[0, 1], [[[0, 1]]], np.empty((2, 0))])
def test_unsupported_pair_shapes_are_rejected(pairs):
    with pytest.raises(ContractError, match="support_pair_shape"):
        TermSupport("bad", "bulk", "00_balance", "00_state", pairs)


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"owner": "someone_else"}, "term_owner_mismatch"),
        ({"equation": "missing"}, "unknown_equation_or_variable"),
        ({"variable": "missing"}, "unknown_equation_or_variable"),
        ({"pairs": [[4, 0]]}, "index_outside_support"),
        ({"pairs": [[0, -1]]}, "index_outside_support"),
        ({"unit": LENGTH}, "jacobian_unit_mismatch"),
        ({"kind": "port"}, "global_border_support_required"),
    ],
)
def test_unsupported_named_support_or_owner_is_rejected(change, reason):
    graph, terms, _ = small_case()
    bad = replace(terms[0], **change)
    with pytest.raises(ContractError, match=reason):
        SparseGraph(graph.layout, (bad,) + terms[1:])


def test_undeclared_derivative_and_duplicate_terms_are_rejected():
    graph, terms, _ = small_case()
    layout = replace(
        graph.layout,
        equations=(replace(graph.layout.equations[0], derivative_support=()),),
    )
    with pytest.raises(ContractError, match="undeclared_derivative_support"):
        SparseGraph(layout, terms)
    with pytest.raises(ContractError, match="duplicate_term"):
        SparseGraph(graph.layout, terms + terms[:1])
    with pytest.raises(ContractError, match="duplicate_term_edge"):
        replace(terms[0], pairs=[[0, 0], [0, 0]])


@pytest.mark.parametrize(
    "coefficient,reason",
    [
        (1.0, "coefficient_shape_mismatch"),
        (np.ones((4, 1)), "coefficient_shape_mismatch"),
        ([1, 2, 3, np.inf], "nonfinite_array"),
        ([1, 2, 3, np.nan], "nonfinite_array"),
        ([1j] * 4, "unsupported_coefficient_dtype"),
    ],
)
def test_coefficient_shape_and_finiteness_fail_closed(coefficient, reason):
    graph, terms, values = small_case()
    with pytest.raises(ContractError, match=reason):
        graph.assemble({**values, "diagonal": coefficient}, supports=terms)


def test_missing_coefficient_non_square_layout_and_expansion_are_rejected():
    graph, terms, _ = small_case()
    with pytest.raises(ContractError, match="coefficient_terms_mismatch"):
        graph.assemble({}, supports=terms)
    layout = replace(graph.layout, equations=())
    with pytest.raises(ContractError, match="non_square_or_invalid_layout"):
        SparseGraph(layout, ())
    with pytest.raises(ContractError, match="structural_expansion_limit"):
        TermSupport.cartesian(
            "large", "bulk", "00_balance", "00_state", np.arange(1001), np.arange(1001)
        )


@pytest.mark.parametrize(
    "diffusion", [[-1.0, 1.0, 1.0], [np.nan, 1.0, 1.0], [[1.0], [1.0], [1.0]]]
)
def test_invalid_diffusion_input_is_rejected(diffusion):
    with pytest.raises(ContractError):
        ion_domains(4, [[0, 1], [1, 2], [2, 3]], diffusion)


@pytest.mark.parametrize("alpha,u,v", [(0.0, [1.0], [1.0]), (2.0, [1.0], [2.0])])
def test_nc07_invertibility_preconditions_fail_before_factorization(alpha, u, v):
    with pytest.raises(ContractError, match="singular_rank_one"):
        require_rank_one_invertible(alpha, u, v)


def test_native_factorization_failure_is_retained_without_fallback():
    with pytest.raises(RuntimeError, match="singular"):
        factorize(csc_matrix((4, 4)))


def test_rhs_shape_and_dense_factor_input_are_rejected():
    graph, terms, values = small_case()
    matrix = graph.assemble(values, supports=terms)
    with pytest.raises(ContractError, match="rhs_shape_mismatch"):
        sparse_solve(matrix, np.ones((4, 1)))
    with pytest.raises(ContractError, match="square_csc_required"):
        factorize(np.eye(4))
