"""P02-07 structural sparsity prototype; no production entry point uses it.

Declarations describe possible derivatives, independently of current values.
The small Decimal references and the manufactured scale oracle live in tests.
The scale operator is a dimensionless structure proxy, not device physics.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from math import prod
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csc_matrix, isspmatrix_csc
from scipy.sparse.linalg import SuperLU, splu

from scripts.benchmarks.contract_prototype import (
    ONE,
    ContractError,
    EquationSpec,
    ImmutableArrays,
    Layout,
    Support,
    Unit,
    VariableSpec,
    frozen_array,
    integer_indices,
)

Indices = NDArray[np.intp]
Vector = NDArray[np.float64]
MAX_DECLARED_ENTRIES = 1_000_000


def _owned_indices(value: ArrayLike) -> Indices:
    indices = integer_indices(value)
    return np.frombuffer(indices.tobytes(), dtype=np.intp).reshape(indices.shape)


@dataclass(frozen=True)
class TermSupport(ImmutableArrays):
    """Unique local (equation row, variable column) pairs, flattened in C order."""

    id: str
    owner: str
    equation: str
    variable: str
    pairs: Indices
    kind: str = "local"
    unit: Unit = ONE

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value
            for value in (self.id, self.owner, self.equation, self.variable)
        ):
            raise ContractError("missing_term_identity")
        if self.kind not in {"local", "nonlocal", "inventory", "port"}:
            raise ContractError("unsupported_support_kind")
        pairs = _owned_indices(self.pairs)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ContractError("support_pair_shape")
        if len(pairs) > MAX_DECLARED_ENTRIES:
            raise ContractError("structural_expansion_limit")
        if len(np.unique(pairs, axis=0)) != len(pairs):
            raise ContractError("duplicate_term_edge")
        object.__setattr__(self, "pairs", pairs)

    @classmethod
    def cartesian(
        cls,
        id: str,
        owner: str,
        equation: str,
        variable: str,
        rows: ArrayLike,
        columns: ArrayLike,
        *,
        kind: str = "nonlocal",
        unit: Unit = ONE,
    ) -> TermSupport:
        rows, columns = integer_indices(rows), integer_indices(columns)
        if rows.ndim != 1 or columns.ndim != 1:
            raise ContractError("support_axis_shape")
        # Check before allocating: the large nonlocal operator must use a border.
        if int(rows.size) * int(columns.size) > MAX_DECLARED_ENTRIES:
            raise ContractError("structural_expansion_limit")
        pairs = np.column_stack(
            (np.repeat(rows, len(columns)), np.tile(columns, len(rows)))
        )
        return cls(id, owner, equation, variable, pairs, kind, unit)

    def identity_bytes(self) -> bytes:
        pairs = self.pairs
        order = np.lexsort((pairs[:, 0], pairs[:, 1]))
        metadata = (
            self.id,
            self.owner,
            self.equation,
            self.variable,
            self.kind,
            self.unit.powers,
        )
        return (
            json.dumps(metadata, separators=(",", ":")).encode()
            + pairs[order].astype("<i8").tobytes()
        )


def _declarations(
    layout: Layout,
    supports: Iterable[TermSupport],
    topology_identity: str,
) -> tuple[tuple[TermSupport, ...], dict[str, slice], tuple[int, int], str]:
    supports = tuple(supports)
    if not all(isinstance(term, TermSupport) for term in supports):
        raise ContractError("untyped_support")
    supports = tuple(sorted(supports, key=lambda term: term.id))
    if len({term.id for term in supports}) != len(supports):
        raise ContractError("duplicate_term")
    if not isinstance(topology_identity, str):
        raise ContractError("invalid_topology_identity")
    variables = {spec.id: spec for spec in layout.variables}
    equations = {spec.id: spec for spec in layout.equations}
    locations = {support.id: support.location for support in layout.supports}
    offsets: dict[str, slice] = {}
    rows = 0
    for equation in layout.equations:
        count = prod(equation.shape)
        offsets[equation.id] = slice(rows, rows + count)
        rows += count
    columns = sum(prod(variable.shape) for variable in layout.variables)
    if columns != layout.size or rows <= 0 or rows != columns:
        raise ContractError("non_square_or_invalid_layout")
    if (
        rows > MAX_DECLARED_ENTRIES
        or sum(len(term.pairs) for term in supports) > MAX_DECLARED_ENTRIES
    ):
        raise ContractError("structural_expansion_limit")
    if rows * columns > np.iinfo(np.intp).max:
        raise ContractError("structural_index_overflow")
    for term in supports:
        if term.equation not in equations or term.variable not in variables:
            raise ContractError("unknown_equation_or_variable")
        equation, variable = equations[term.equation], variables[term.variable]
        if term.owner != equation.owner:
            raise ContractError("term_owner_mismatch")
        if term.variable not in equation.derivative_support:
            raise ContractError("undeclared_derivative_support")
        if term.unit != equation.unit / variable.unit:
            raise ContractError("jacobian_unit_mismatch")
        if term.kind in {"inventory", "port"} and "global" not in {
            locations[equation.support],
            locations[variable.support],
        }:
            raise ContractError("global_border_support_required")
        pairs = term.pairs
        if (
            np.any(pairs < 0)
            or np.any(pairs[:, 0] >= prod(equation.shape))
            or np.any(pairs[:, 1] >= prod(variable.shape))
        ):
            raise ContractError("index_outside_support")
    fingerprint = sha256(layout.identity.encode() + topology_identity.encode())
    for term in supports:
        fingerprint.update(sha256(term.identity_bytes()).digest())
    return supports, offsets, (rows, columns), fingerprint.hexdigest()


@dataclass(frozen=True, init=False)
class SparseGraph(ImmutableArrays):
    """Immutable graph storage with fresh public array headers, like core arrays."""

    layout: Layout
    supports: tuple[TermSupport, ...]
    row_offsets: Mapping[str, slice]
    shape: tuple[int, int]
    keys: Indices
    indices: Indices
    indptr: Indices
    identity: str
    topology_identity: str

    def __init__(
        self,
        layout: Layout,
        supports: Iterable[TermSupport],
        *,
        topology_identity: str = "",
    ) -> None:
        terms, offsets, shape, identity = _declarations(
            layout, supports, topology_identity
        )
        keys = [self._term_keys(layout, offsets, shape[0], term) for term in terms]
        unique = np.unique(np.concatenate(keys)) if keys else np.empty(0, dtype=np.intp)
        indices = unique % shape[0]
        counts = np.bincount(unique // shape[0], minlength=shape[1])
        indptr = np.concatenate(([0], np.cumsum(counts, dtype=np.intp)))
        for name, value in (
            ("layout", layout),
            ("supports", terms),
            ("row_offsets", MappingProxyType(offsets)),
            ("shape", shape),
            ("keys", _owned_indices(unique)),
            ("indices", _owned_indices(indices)),
            ("indptr", _owned_indices(indptr)),
            ("identity", identity),
            ("topology_identity", topology_identity),
        ):
            object.__setattr__(self, name, value)

    @staticmethod
    def _term_keys(
        layout: Layout, offsets: Mapping[str, slice], rows: int, term: TermSupport
    ) -> Indices:
        pairs = term.pairs
        row = pairs[:, 0] + offsets[term.equation].start
        column = pairs[:, 1] + layout.offsets[term.variable].start
        return column * rows + row

    @property
    def nnz(self) -> int:
        return len(self.indices)

    def edges(self) -> Indices:
        return _owned_indices(
            np.column_stack((self.keys % self.shape[0], self.keys // self.shape[0]))
        )

    def assemble(
        self,
        values: Mapping[str, ArrayLike],
        *,
        supports: Iterable[TermSupport],
        topology_identity: str = "",
    ) -> csc_matrix:
        terms, offsets, shape, identity = _declarations(
            self.layout, supports, topology_identity
        )
        if identity != self.identity:
            raise ContractError("structural_recompile_required")
        if set(values) != {term.id for term in terms}:
            raise ContractError("coefficient_terms_mismatch")
        data = np.zeros(self.nnz)
        for term in terms:
            raw = np.asarray(values[term.id])
            if raw.dtype.kind not in {"i", "u", "f"}:
                raise ContractError("unsupported_coefficient_dtype")
            coefficient = frozen_array(raw)
            if coefficient.shape != (len(term.pairs),):
                raise ContractError("coefficient_shape_mismatch")
            # Align with the current declaration order, not a cached value mask.
            positions = np.searchsorted(
                self.keys, self._term_keys(self.layout, offsets, shape[0], term)
            )
            np.add.at(data, positions, coefficient)
        return csc_matrix(
            (data, self.indices.copy(), self.indptr.copy()), shape=self.shape
        )


@dataclass(frozen=True)
class IonDomains(ImmutableArrays):
    labels: Indices
    active_faces: Indices
    domains: tuple[tuple[int, ...], ...]
    identity: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels", _owned_indices(self.labels))
        object.__setattr__(self, "active_faces", _owned_indices(self.active_faces))


def ion_domains(count: int, faces: ArrayLike, diffusivity: ArrayLike) -> IonDomains:
    count_array = integer_indices([count])
    if (
        count_array.shape != (1,)
        or count_array[0] <= 0
        or count_array[0] > MAX_DECLARED_ENTRIES
    ):
        raise ContractError("invalid_domain_size")
    count = int(count_array[0])
    pairs = integer_indices(faces)
    diffusion = frozen_array(diffusivity)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or diffusion.shape != (len(pairs),):
        raise ContractError("diffusion_face_shape")
    if (
        np.any(pairs < 0)
        or np.any(pairs >= count)
        or np.any(pairs[:, 0] == pairs[:, 1])
    ):
        raise ContractError("face_outside_support")
    if np.any(diffusion < 0):
        raise ContractError("negative_diffusivity")
    # Exact physical blocking is a topology decision, never an epsilon cutoff.
    active = pairs[diffusion != 0]
    parent = list(range(count))

    def root(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for left, right in active:
        a, b = root(int(left)), root(int(right))
        parent[max(a, b)] = min(a, b)
    groups: dict[int, list[int]] = {}
    for node in range(count):
        groups.setdefault(root(node), []).append(node)
    domains = tuple(tuple(nodes) for _, nodes in sorted(groups.items()))
    labels = np.empty(count, dtype=np.intp)
    for label, nodes in enumerate(domains):
        labels[list(nodes)] = label
    canonical = np.unique(np.sort(active, axis=1), axis=0)
    fingerprint = sha256(
        str(count).encode() + canonical.astype("<i8").tobytes()
    ).hexdigest()
    return IonDomains(labels, active, domains, fingerprint)


def require_rank_one_invertible(alpha: float, u: ArrayLike, v: ArrayLike) -> None:
    u, v = frozen_array(u), frozen_array(v)
    if u.ndim != 1 or u.shape != v.shape or not u.size or not np.isfinite(alpha):
        raise ContractError("rank_one_shape_or_coefficient")
    contraction = float(v @ u)
    if not np.isfinite(contraction):
        raise ContractError("nonfinite_rank_one_contraction")
    if alpha == 0 or alpha == contraction:
        raise ContractError("singular_rank_one")


def factorize(matrix: csc_matrix) -> SuperLU:
    if (
        not isspmatrix_csc(matrix)
        or matrix.shape[0] == 0
        or matrix.shape[0] != matrix.shape[1]
    ):
        raise ContractError("square_csc_required")
    if not np.isfinite(matrix.data).all():
        raise ContractError("nonfinite_matrix")
    # A native failure propagates; no alternate solver or dense fallback.
    return splu(
        matrix,
        permc_spec="COLAMD",
        diag_pivot_thresh=1.0,
        relax=1,
        panel_size=10,
        options={"Equil": False, "IterRefine": "DOUBLE"},
    )


def sparse_solve(matrix: csc_matrix, rhs: ArrayLike) -> Vector:
    rhs = frozen_array(rhs)
    if rhs.shape != (matrix.shape[0],):
        raise ContractError("rhs_shape_mismatch")
    result = factorize(matrix).solve(rhs)
    if not np.isfinite(result).all():
        raise ContractError("nonfinite_solution")
    return result


def grid_proxy(
    nx: int, ny: int
) -> tuple[SparseGraph, tuple[TermSupport, ...], dict[str, Vector]]:
    """Five-point field plus three global variables; never expand u*v^T."""
    dimensions = integer_indices([nx, ny])
    if dimensions.shape != (2,) or np.any(dimensions < 2):
        raise ContractError("unsupported_grid_shape")
    nx, ny = map(int, dimensions)
    count = nx * ny
    if 9 * count - 2 * nx + 2 * ny + 2 > MAX_DECLARED_ENTRIES:
        raise ContractError("structural_expansion_limit")
    field, auxiliary, inventory, port = (
        "00_field",
        "10_nonlocal",
        "20_inventory",
        "30_port",
    )
    balance, auxiliary_eq, inventory_eq, port_eq = (
        "00_balance",
        "10_nonlocal",
        "20_inventory",
        "30_port",
    )
    layout = Layout(
        (Support("cells", "cell", (ny, nx)), Support("global", "global", (1,))),
        (
            VariableSpec(field, "bulk", "cells", (ny, nx), ONE),
            VariableSpec(auxiliary, "nonlocal", "global", (1,), ONE, role="constraint"),
            VariableSpec(
                inventory, "inventory", "global", (1,), ONE, role="constraint"
            ),
            VariableSpec(port, "port", "global", (1,), ONE, role="constraint"),
        ),
        (
            EquationSpec(
                balance,
                "bulk",
                "cells",
                (ny, nx),
                ONE,
                derivative_support=(field, auxiliary, inventory, port),
            ),
            EquationSpec(
                auxiliary_eq,
                "nonlocal",
                "global",
                (1,),
                ONE,
                role="constraint",
                derivative_support=(field, auxiliary),
            ),
            EquationSpec(
                inventory_eq,
                "inventory",
                "global",
                (1,),
                ONE,
                role="constraint",
                derivative_support=(field,),
            ),
            EquationSpec(
                port_eq,
                "port",
                "global",
                (1,),
                ONE,
                role="constraint",
                derivative_support=(field, port),
            ),
        ),
    )
    nodes = np.arange(count, dtype=np.intp)
    i = np.tile(np.arange(1, nx + 1), ny)
    j = np.repeat(np.arange(1, ny + 1), nx)
    u = ((i + j) % 5) / 16.0
    v = 2.0 * i / ((nx + 1) * count)
    boundary = nodes[(i == 1) | (i == nx)]
    p = np.where(i[boundary] == 1, 1.0, -1.0)
    terms: list[TermSupport] = []
    values: dict[str, Vector] = {}

    def add(
        id: str,
        owner: str,
        equation: str,
        variable: str,
        pairs: ArrayLike,
        coefficients: ArrayLike,
        kind: str = "local",
    ) -> None:
        terms.append(
            TermSupport(id, owner, equation, variable, _owned_indices(pairs), kind)
        )
        values[id] = np.asarray(coefficients, dtype=float)

    add(
        "local.diagonal",
        "bulk",
        balance,
        field,
        np.column_stack((nodes, nodes)),
        np.full(count, 8.0),
    )
    for name, left, right, weight in (
        (
            "x",
            nodes.reshape(ny, nx)[:, :-1].ravel(),
            nodes.reshape(ny, nx)[:, 1:].ravel(),
            -1.0,
        ),
        ("y", nodes[:-nx], nodes[nx:], -2.0),
    ):
        pairs = np.concatenate(
            (np.column_stack((left, right)), np.column_stack((right, left)))
        )
        add("local." + name, "bulk", balance, field, pairs, np.full(len(pairs), weight))
    column = np.column_stack((nodes, np.zeros(count, dtype=np.intp)))
    row = column[:, ::-1]
    add("nonlocal.u", "bulk", balance, auxiliary, column, -u, "nonlocal")
    add("nonlocal.v", "nonlocal", auxiliary_eq, field, row, -v, "nonlocal")
    add("nonlocal.z", "nonlocal", auxiliary_eq, auxiliary, [[0, 0]], [1.0], "nonlocal")
    add(
        "inventory.column",
        "bulk",
        balance,
        inventory,
        column,
        np.ones(count),
        "inventory",
    )
    add(
        "inventory.row",
        "inventory",
        inventory_eq,
        field,
        row,
        np.full(count, 1.0 / count),
        "inventory",
    )
    border_column = np.column_stack((boundary, np.zeros(len(boundary), dtype=np.intp)))
    add("port.column", "bulk", balance, port, border_column, p, "port")
    add("port.row", "port", port_eq, field, border_column[:, ::-1], p / ny, "port")
    add("port.scalar", "port", port_eq, port, [[0, 0]], [-2.0], "port")
    supports = tuple(terms)
    return SparseGraph(layout, supports), supports, values
