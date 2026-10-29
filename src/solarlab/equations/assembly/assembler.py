"""One Assembler: sparse F, J = dF/du, S and S_u = dS/du from Terms on a mesh.

Sign and storage convention (milestone A interface decision): the
time-dependent system is ``dS(u)/dt = F(u)``. ``F`` is the steady residual
(net inflow plus sources on balance rows, the declared algebraic value on
constraint rows), ``J = dF/du``, ``S`` is the physical equation storage (zero
on constraint rows) and ``S_u = dS/du``; the implicit form is
``R(u, udot) = F(u) - S_u udot`` with ``M = dR/dudot = -S_u``.

Accumulation order is fixed, so one declaration always produces the same bits:

* A slot is one quantity of one kind on one block (a value, or the tangent
  for one dependency). Its unweighted parts are summed in Term order and
  multiplied by the measure once; each weighted expression is then its own
  piece, in Term order. Pieces are added sequentially.
* Per balance block, pieces are added to a zero row in the order face flux,
  cell source, port flux (ports in declaration order). The face divergence of
  a piece scatters ``-flux`` into every left cell and then ``+flux`` into
  every right cell (the ``TermSink`` order).
* Jacobian entries are summed sequentially in emission order: face pieces
  face by face (left row, then right row; dependencies in first-use order),
  then cell pieces, then port pieces.
* Constraint rows take the owning Term's value and tangents; every other
  contribution to those rows is dropped from ``F``, ``J``, ``S`` and ``S_u``.
  ``balance`` keeps it: on a constraint row it is the conservation defect the
  boundary exchange must carry (:meth:`Assembler.reaction_flux`).

Represented matrix entries are the nonzero sums. Exact zeros are dropped, as
the historical LIL assembly drops them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from typing import Mapping, Sequence, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import coo_matrix, csc_matrix

from solarlab.contracts.arrays import frozen_array
from solarlab.contracts.errors import ContractError
from solarlab.contracts.terms import Geometry
from solarlab.contracts.units import SECOND, Unit
from solarlab.equations.assembly.layout import ConstraintRows, SystemLayout
from solarlab.equations.assembly.terms import (
    CELL_SOURCE, CONSTRAINT, FACE_FLUX, PORT_FLUX, STORAGE, TERM_KINDS, Contribution,
    EvaluationState, FaceValue, FieldMap, LocalValue, OperatingInputs, StorageTerm, Term,
    TermValue,
)
from solarlab.equations.assembly.weighted import Weighted, evaluate as evaluate_weighted, measure_count

__all__ = ["AssembledSystem", "Assembler", "Mesh", "Port", "SupportGeometry"]

Vector: TypeAlias = NDArray[np.float64]
Index: TypeAlias = NDArray[np.intp]
Evaluated: TypeAlias = Sequence[tuple[Term, Mapping[str, TermValue]]]


@dataclass(frozen=True)
class Port:
    """Boundary faces beside ``nodes``: outward normal and face measures."""

    id: str
    nodes: tuple[int, ...]
    outward_normal: int
    measures: Vector
    measure_unit: Unit

    def __post_init__(self) -> None:
        nodes = tuple(int(i) for i in self.nodes)
        measures = frozen_array(self.measures)
        if (not self.id or not nodes or self.outward_normal not in (-1, 1)
                or measures.shape != (len(nodes),) or np.any(measures <= 0.0)
                or len(set(nodes)) != len(nodes)):
            raise ContractError("invalid_port")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "measures", measures)


@dataclass(frozen=True)
class SupportGeometry:
    """Control volumes, oriented faces and ports of one node set."""

    node_set: str
    geometry: Geometry
    ports: tuple[Port, ...] = ()

    def __post_init__(self) -> None:
        if len({port.id for port in self.ports}) != len(self.ports):
            raise ContractError("duplicate_port")
        count = self.geometry.volumes.size
        if any(node >= count for port in self.ports for node in port.nodes):
            raise ContractError("port_outside_support")

    def port(self, port_id: str) -> Port:
        for port in self.ports:
            if port.id == port_id:
                return port
        raise ContractError("unknown_port")


@dataclass(frozen=True)
class Mesh:
    """Geometry of every node set of a layout, with a content identity."""

    supports: tuple[SupportGeometry, ...]
    identity: str = field(init=False)

    def __post_init__(self) -> None:
        if len({item.node_set for item in self.supports}) != len(self.supports):
            raise ContractError("duplicate_support_geometry")
        digest = sha256(b"solarlab.assembly_mesh.v1")
        for item in self.supports:
            g = item.geometry
            digest.update(json.dumps([item.node_set, list(g.volume_unit.powers),
                                      list(g.face_unit.powers)]).encode())
            for array in (g.volumes, g.face_measures):
                digest.update(np.ascontiguousarray(array, dtype="<f8").tobytes())
            digest.update(np.ascontiguousarray(g.pairs, dtype="<i8").tobytes())
            for port in item.ports:
                digest.update(json.dumps([port.id, list(port.nodes), port.outward_normal,
                                          list(port.measure_unit.powers)]).encode())
                digest.update(np.ascontiguousarray(port.measures, dtype="<f8").tobytes())
        object.__setattr__(self, "identity", digest.hexdigest())

    def support(self, node_set: str) -> SupportGeometry:
        for item in self.supports:
            if item.node_set == node_set:
                return item
        raise ContractError("unknown_support_geometry")


@dataclass(frozen=True)
class AssembledSystem:
    """One evaluation in the shared convention; ``M = -S_u``."""

    layout: SystemLayout
    admissible: bool
    F: Vector
    balance: Vector
    J: csc_matrix | None = None
    S: Vector | None = None
    S_u: csc_matrix | None = None

    @property
    def M(self) -> csc_matrix | None:
        return None if self.S_u is None else cast(csc_matrix, -self.S_u)

    @property
    def block_ranges(self) -> dict[str, tuple[int, int]]:
        return self.layout.block_ranges()

    @property
    def boundary_rows(self) -> NDArray[np.bool_]:
        return self.layout.constraint_mask()


class _Triplets:
    """Matrix entries in emission order, summed sequentially per entry."""

    def __init__(self) -> None:
        self.rows: list[Index] = []
        self.cols: list[Index] = []
        self.vals: list[Vector] = []

    def add(self, rows: ArrayLike, cols: ArrayLike, vals: ArrayLike) -> None:
        r = np.asarray(rows, dtype=np.intp).ravel()
        c = np.asarray(cols, dtype=np.intp).ravel()
        v = np.broadcast_to(np.asarray(vals, dtype=float), np.shape(rows)).ravel()
        keep = c >= 0
        if not np.all(v[~keep] == 0.0):
            raise ContractError("tangent_outside_embedded_node_set")
        self.rows.append(r[keep])
        self.cols.append(c[keep])
        self.vals.append(v[keep])

    def without_rows(self, rows: NDArray[np.bool_]) -> _Triplets:
        result = _Triplets()
        for r, c, v in zip(self.rows, self.cols, self.vals):
            keep = ~rows[r]
            result.rows.append(r[keep])
            result.cols.append(c[keep])
            result.vals.append(v[keep])
        return result

    def extend(self, other: _Triplets) -> None:
        self.rows.extend(other.rows)
        self.cols.extend(other.cols)
        self.vals.extend(other.vals)

    def matrix(self, size: int) -> csc_matrix:
        rows = np.concatenate(self.rows) if self.rows else np.empty(0, dtype=np.intp)
        cols = np.concatenate(self.cols) if self.cols else np.empty(0, dtype=np.intp)
        vals = np.concatenate(self.vals) if self.vals else np.empty(0)
        keys = rows.astype(np.int64) * size + cols
        unique, inverse = np.unique(keys, return_inverse=True)
        data = np.zeros(unique.size)
        np.add.at(data, inverse, vals)  # sequential, in emission order
        nonzero = data != 0.0
        unique, data = unique[nonzero], data[nonzero]
        return coo_matrix((data, (unique // size, unique % size)), shape=(size, size)).tocsc()


def _pieces(parts: Sequence[Contribution], measure: Vector, *, weighted: bool = True) -> list[Vector]:
    """Measure-weighted pieces of one slot, in accumulation order.

    The unweighted parts are summed first and weighted once; each weighted
    expression must contain the measure exactly once (constraints: none).
    """
    plain = [np.asarray(part, dtype=float) for part in parts if not isinstance(part, Weighted)]
    result: list[Vector] = []
    if plain:
        accumulated = plain[0]
        for part in plain[1:]:
            accumulated = accumulated + part
        accumulated = np.broadcast_to(accumulated, measure.shape)
        result.append(accumulated * measure if weighted else np.array(accumulated, dtype=float))
    for part in parts:
        if not isinstance(part, Weighted):
            continue
        if not weighted or measure_count(part) != 1:
            raise ContractError("weighted_contribution_needs_exactly_one_measure")
        result.append(evaluate_weighted(part, measure))
    if not result:
        raise ContractError("empty_contribution")
    return result


def _sum(pieces: Sequence[Vector]) -> Vector:
    total = pieces[0]
    for piece in pieces[1:]:
        total = total + piece
    return total


class Assembler:
    """Validate a binding once; evaluate F, J, S and S_u for any state and drive."""

    def __init__(self, layout: SystemLayout, mesh: Mesh, fields: FieldMap, terms: Sequence[Term]):
        self.layout = layout
        self.mesh = mesh
        self.fields = fields
        self.terms = tuple(terms)
        ids = [term.id for term in self.terms]
        if len(set(ids)) != len(ids) or any(not item for item in ids):
            raise ContractError("duplicate_or_empty_term_id")
        blocks = {block.id: block for block in layout.blocks}
        for item in mesh.supports:
            if item.geometry.volumes.size != layout.node_set(item.node_set).size:
                raise ContractError("mesh_support_size_mismatch")
        owners = {rows.owner: rows for rows in layout.constraints}
        constrained = set()
        for term in self.terms:
            if term.kind not in TERM_KINDS:
                raise ContractError("unknown_term_kind")
            support = mesh.support(term.support)
            if not term.equations or len(set(term.equations)) != len(term.equations):
                raise ContractError("term_needs_distinct_equations")
            if term.kind in (PORT_FLUX, CONSTRAINT) and len(term.equations) != 1:
                raise ContractError("port_and_constraint_terms_own_one_block")
            if term.kind == PORT_FLUX:
                if term.port is None:
                    raise ContractError("port_term_needs_port")
                support.port(term.port)
            elif term.port is not None:
                raise ContractError("port_on_non_port_term")
            for equation in term.equations:
                block = blocks.get(equation)
                if block is None:
                    raise ContractError("unknown_equation_block")
                if block.node_set != term.support:
                    raise ContractError("term_support_differs_from_equation_rows")
                self._check_unit(term, support, block.balance_unit, owners)
            if len(set(term.dependencies)) != len(term.dependencies):
                raise ContractError("duplicate_dependency")
            for dependency in term.dependencies:
                if dependency not in blocks:
                    raise ContractError("unknown_dependency_block")
                layout.column_map(term.support, dependency)
            if term.kind == CONSTRAINT:
                if term.id in constrained:
                    raise ContractError("constraint_rows_owned_twice")
                constrained.add(term.id)
            if term.kind == STORAGE and not callable(getattr(term, "increment", None)):
                raise ContractError("storage_term_without_increment")
        if constrained != set(owners):
            raise ContractError("declared_constraint_rows_without_owner")
        self._constraint_mask = layout.constraint_mask()
        self._constraint_mask.setflags(write=False)
        digest = sha256(b"solarlab.assembler.v1")
        for part in (layout.identity, mesh.identity, fields.identity):
            digest.update(part.encode())
        for term in self.terms:
            digest.update(json.dumps([term.id, term.kind, term.identity]).encode())
        self.identity = digest.hexdigest()

    @staticmethod
    def _check_unit(term: Term, support: SupportGeometry, balance: Unit,
                    owners: Mapping[str, ConstraintRows]) -> None:
        g = support.geometry
        if term.kind == FACE_FLUX:
            ok = term.unit * g.face_unit == balance
        elif term.kind == CELL_SOURCE:
            ok = term.unit * g.volume_unit == balance
        elif term.kind == PORT_FLUX:
            assert term.port is not None
            ok = term.unit * support.port(term.port).measure_unit == balance
        elif term.kind == STORAGE:
            ok = term.unit * g.volume_unit == balance * SECOND
        else:
            rows = owners.get(term.id)
            if rows is None or term.equations != (rows.block,):
                raise ContractError("constraint_term_without_declared_rows")
            ok = rows.unit == term.unit
        if not ok:
            raise ContractError("unit_mismatch")

    # -- evaluation ---------------------------------------------------------------------

    def state(self, u: ArrayLike, inputs: OperatingInputs) -> EvaluationState | None:
        """Shared evaluation state, or None outside the field map's domain."""
        values = np.array(u, dtype=float, copy=True)
        if values.shape != (self.layout.size,):
            raise ContractError("coordinate_size_mismatch")
        values.setflags(write=False)
        fields = self.fields.evaluate(self.layout, values, inputs)
        if not fields.admissible:
            return None
        return EvaluationState(self.layout, values, inputs, fields.fields)

    def _evaluate(self, state: EvaluationState, derivatives: bool, kinds: tuple[str, ...]) -> Evaluated:
        results = []
        for term in self.terms:
            if term.kind not in kinds:
                continue
            values = term.evaluate(state, derivatives=derivatives)
            if set(values) != set(term.equations):
                raise ContractError("term_returned_undeclared_equations")
            for value in values.values():
                expected = FaceValue if term.kind == FACE_FLUX else LocalValue
                if not isinstance(value, expected):
                    raise ContractError("term_value_kind_mismatch")
                tangents = value.left if isinstance(value, FaceValue) else value.tangents
                if not set(tangents) <= set(term.dependencies):
                    raise ContractError("tangent_with_undeclared_dependency")
            results.append((term, values))
        return results

    @staticmethod
    def _parts(results: Evaluated, kind: str, block_id: str, port: str | None = None) -> list[TermValue]:
        return [values[block_id] for term, values in results
                if term.kind == kind and block_id in values and term.port == port]

    def _columns(self, support: str, dependency: str) -> Index:
        local = self.layout.column_map(support, dependency)
        result: Index = np.where(local >= 0, self.layout.offsets[dependency].start + local, -1)
        return result

    def _balance(self, block_id: str, results: Evaluated) -> Vector:
        node_set = self.layout.block(block_id).node_set
        support = self.mesh.support(node_set)
        g = support.geometry
        row = np.zeros(self.layout.node_set(node_set).size)
        faces = [cast(FaceValue, value).flux for value in self._parts(results, FACE_FLUX, block_id)]
        for flux in (_pieces(faces, g.face_measures) if faces else ()):
            divergence = np.zeros(row.shape)
            np.add.at(divergence, g.pairs[:, 0], -flux)
            np.add.at(divergence, g.pairs[:, 1], flux)
            row = row + divergence
        cells = [cast(LocalValue, value).value for value in self._parts(results, CELL_SOURCE, block_id)]
        for source in (_pieces(cells, g.volumes) if cells else ()):
            row = row + source
        for port in support.ports:
            parts = [cast(LocalValue, value).value
                     for value in self._parts(results, PORT_FLUX, block_id, port.id)]
            for exchange in (_pieces(parts, port.measures) if parts else ()):
                np.add.at(row, np.asarray(port.nodes, dtype=np.intp), -port.outward_normal * exchange)
        return row

    def _balance_jacobian(self, block_id: str, results: Evaluated, triplets: _Triplets) -> None:
        node_set = self.layout.block(block_id).node_set
        support = self.mesh.support(node_set)
        g = support.geometry
        rows = self.layout.rows(block_id)
        faces = [cast(FaceValue, value) for value in self._parts(results, FACE_FLUX, block_id)]
        dependencies = list(dict.fromkeys(dep for value in faces for dep in value.left))
        if dependencies:
            left, right = g.pairs[:, 0], g.pairs[:, 1]
            sides = [[_pieces([getattr(value, side)[dependency] for value in faces
                               if dependency in getattr(value, side)], g.face_measures)
                      for side in ("left", "right")] for dependency in dependencies]
            for piece in range(max(len(pieces) for pair in sides for pieces in pair)):
                # Each face emits its left row, then its right row, before the next face.
                entry_rows = np.empty((left.size, 2, 2 * len(dependencies)), dtype=np.intp)
                entry_rows[:, 0, :] = rows[left][:, None]
                entry_rows[:, 1, :] = rows[right][:, None]
                entry_cols = np.full(entry_rows.shape, -1, dtype=np.intp)
                entry_vals = np.zeros(entry_rows.shape)
                for k, dependency in enumerate(dependencies):
                    columns = self._columns(node_set, dependency)
                    for offset, nodes in ((0, left), (1, right)):
                        pieces = sides[k][offset]
                        if piece >= len(pieces):
                            continue
                        entry_cols[:, :, 2 * k + offset] = columns[nodes][:, None]
                        entry_vals[:, 0, 2 * k + offset] = -pieces[piece]
                        entry_vals[:, 1, 2 * k + offset] = pieces[piece]
                triplets.add(entry_rows, entry_cols, entry_vals)
        cells = [cast(LocalValue, value) for value in self._parts(results, CELL_SOURCE, block_id)]
        for dependency in dict.fromkeys(dep for value in cells for dep in value.tangents):
            for tangent in _pieces([value.tangents[dependency] for value in cells
                                    if dependency in value.tangents], g.volumes):
                triplets.add(rows, self._columns(node_set, dependency), tangent)
        for port in support.ports:
            values = [cast(LocalValue, value) for value in self._parts(results, PORT_FLUX, block_id, port.id)]
            nodes = np.asarray(port.nodes, dtype=np.intp)
            for dependency in dict.fromkeys(dep for value in values for dep in value.tangents):
                for tangent in _pieces([value.tangents[dependency] for value in values
                                        if dependency in value.tangents], port.measures):
                    triplets.add(rows[nodes], self._columns(node_set, dependency)[nodes],
                                 -port.outward_normal * tangent)

    def _replace_constraints(self, results: Evaluated, F: Vector, triplets: _Triplets | None) -> None:
        for term, values in results:
            rows = self.layout.constraint_rows(term.id)
            value = cast(LocalValue, values[rows.block])
            nodes: Index = np.asarray(rows.nodes, dtype=np.intp)
            shape = np.empty(nodes.size)
            global_rows = self.layout.rows(rows.block, nodes)
            F[global_rows] = _pieces([value.value], shape, weighted=False)[0]
            if triplets is not None:
                for dependency, tangent in value.tangents.items():
                    triplets.add(global_rows, self._columns(term.support, dependency)[nodes],
                                 _pieces([tangent], shape, weighted=False)[0])

    def evaluate(self, u: ArrayLike, inputs: OperatingInputs, *, derivatives: bool = True,
                 storage: bool = False) -> AssembledSystem:
        """F (with J when ``derivatives``), and S (with S_u) when ``storage``."""
        state = self.state(u, inputs)
        size = self.layout.size
        if state is None:
            if derivatives or storage:
                raise FloatingPointError("state outside the field map's admissible domain")
            nan = np.full(size, np.nan)
            return AssembledSystem(self.layout, False, nan, nan.copy())
        results = self._evaluate(state, derivatives, (FACE_FLUX, CELL_SOURCE, PORT_FLUX))
        constraints = self._evaluate(state, derivatives, (CONSTRAINT,))
        balance = np.zeros(size)
        for block in self.layout.blocks:
            balance[self.layout.offsets[block.id]] = self._balance(block.id, results)
        F = balance.copy()
        J: csc_matrix | None = None
        if derivatives:
            triplets = _Triplets()
            for block in self.layout.blocks:
                self._balance_jacobian(block.id, results, triplets)
            replaced = _Triplets()
            self._replace_constraints(constraints, F, replaced)
            kept = triplets.without_rows(self._constraint_mask)
            kept.extend(replaced)
            J = kept.matrix(size)
        else:
            self._replace_constraints(constraints, F, None)
        S: Vector | None = None
        S_u: csc_matrix | None = None
        if storage:
            S, S_u = self._storage(state, derivatives)
        balance.setflags(write=False)
        F.setflags(write=False)
        return AssembledSystem(self.layout, True, F, balance, J, S, S_u)

    def _storage(self, state: EvaluationState, derivatives: bool, *,
                 physical: bool = False) -> tuple[Vector, csc_matrix | None]:
        results = self._evaluate(state, derivatives, (STORAGE,))
        S = np.zeros(self.layout.size)
        triplets = _Triplets()
        for block in self.layout.blocks:
            g = self.mesh.support(block.node_set).geometry
            values = [cast(LocalValue, value) for value in self._parts(results, STORAGE, block.id)]
            if not values:
                continue
            S[self.layout.offsets[block.id]] = _sum(_pieces([value.value for value in values], g.volumes))
            if derivatives:
                rows = self.layout.rows(block.id)
                for dependency in dict.fromkeys(dep for value in values for dep in value.tangents):
                    for tangent in _pieces([value.tangents[dependency] for value in values
                                            if dependency in value.tangents], g.volumes):
                        triplets.add(rows, self._columns(block.node_set, dependency), tangent)
        if not physical:
            S[self._constraint_mask] = 0.0
            triplets = triplets.without_rows(self._constraint_mask)
        S.setflags(write=False)
        return S, (triplets.matrix(self.layout.size) if derivatives else None)

    def physical_storage(self, u: ArrayLike, inputs: OperatingInputs) -> Vector:
        """Storage including constraint rows (actual boundary-cell inventories)."""
        state = self.state(u, inputs)
        if state is None:
            raise FloatingPointError("state outside the field map's admissible domain")
        return self._storage(state, False, physical=True)[0]

    def storage_increment(self, u: ArrayLike, inputs: OperatingInputs, previous_u: ArrayLike,
                          previous_inputs: OperatingInputs, *, physical: bool = False) -> Vector:
        """Cancellation-free ``S(u) - S(previous_u)``; zero on constraint rows unless physical."""
        state = self.state(u, inputs)
        previous = self.state(previous_u, previous_inputs)
        if state is None or previous is None:
            raise FloatingPointError("storage increment requires two admissible states")
        increments = []
        for term in self.terms:
            if term.kind == STORAGE:
                values = cast(StorageTerm, term).increment(state, previous)
                if set(values) != set(term.equations):
                    raise ContractError("term_returned_undeclared_equations")
                increments.append(values)
        result = np.zeros(self.layout.size)
        for block in self.layout.blocks:
            parts = [item[block.id] for item in increments if block.id in item]
            if parts:
                g = self.mesh.support(block.node_set).geometry
                result[self.layout.offsets[block.id]] = _sum(_pieces(parts, g.volumes))
        if not physical:
            result[self._constraint_mask] = 0.0
        return result

    def reaction_flux(self, system: AssembledSystem, block_id: str, port_id: str,
                      storage_rate: Vector | None = None) -> Vector:
        """Port flux along the support axis that closes constrained boundary cells.

        Each port cell satisfies ``balance - rate + (-n_out) * measure * flux = 0``;
        ``storage_rate`` is the physical storage rate of the full layout (transient).
        """
        node_set = self.layout.block(block_id).node_set
        port = self.mesh.support(node_set).port(port_id)
        rows = self.layout.rows(block_id, port.nodes)
        if not np.all(self._constraint_mask[rows]):
            raise ContractError("reaction_flux_requires_constraint_rows")
        defect = system.balance[rows]
        if storage_rate is not None:
            defect = defect - np.asarray(storage_rate, dtype=float)[rows]
        result: Vector = defect / (port.outward_normal * port.measures)
        return result
