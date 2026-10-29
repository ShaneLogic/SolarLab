"""Named variable blocks on node sets, with stable global indexing.

A binding declares one :class:`SystemLayout` before any Term exists. Each
:class:`Block` is one named variable on one node set together with its
paired equation rows, so row ranges equal column ranges and the system is
square block by block. Global offsets follow the declared block order. That
order is the coordinate convention of saved states (``phi, z_n, z_p`` for the
classical electronic system), so it is declared explicitly and is part of the
identity; Terms never create, reorder or resize blocks.

:class:`ConstraintRows` mark the rows of one block that one named constraint
Term replaces (Dirichlet potentials, fixed carrier reservoirs, inventory
rows). The Assembler discards every other residual, Jacobian and storage
contribution on those rows, so ownership of a row is never shared.

A :class:`NodeSet` may be embedded in a parent node set (an ion domain inside
the electronic nodes); a Term on one set reads a block on the other through
the embedding indices (:meth:`SystemLayout.column_map`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from solarlab.contracts.arrays import integer_indices
from solarlab.contracts.errors import ContractError
from solarlab.contracts.units import Unit

__all__ = ["Block", "ConstraintRows", "NodeSet", "SystemLayout", "node_indices"]

LAYOUT_SCHEMA = "solarlab.system_layout.v1"


def node_indices(values: ArrayLike) -> tuple[int, ...]:
    """Nonnegative integral node indices (the contracts integer-index rule)."""
    indices = integer_indices(values)
    if indices.ndim != 1 or np.any(indices < 0):
        raise ContractError("invalid_index")
    return tuple(int(i) for i in indices)


@dataclass(frozen=True)
class NodeSet:
    """A node set; ``parent_indices`` embed it in ``parent`` when given."""

    id: str
    size: int
    parent: str | None = None
    parent_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not self.id or type(self.size) is not int or self.size < 1:
            raise ContractError("invalid_node_set")
        if (self.parent is None) != (self.parent_indices is None):
            raise ContractError("embedding_needs_parent_and_indices")
        if self.parent_indices is not None:
            indices = node_indices(self.parent_indices)
            if (len(indices) != self.size
                    or any(b <= a for a, b in zip(indices, indices[1:]))):
                raise ContractError("embedding_indices_must_increase")
            object.__setattr__(self, "parent_indices", indices)


@dataclass(frozen=True)
class Block:
    """One variable on one node set and its paired equation rows.

    ``balance_unit`` is the unit of the integrated (geometry-weighted)
    balance rows; constraint rows declare their own unit.
    """

    id: str
    node_set: str
    variable_unit: Unit
    balance_unit: Unit

    def __post_init__(self) -> None:
        if not self.id or not self.node_set:
            raise ContractError("invalid_block")


@dataclass(frozen=True)
class ConstraintRows:
    """Rows of ``block`` (node indices) replaced by the constraint Term ``owner``."""

    block: str
    nodes: tuple[int, ...]
    owner: str
    unit: Unit

    def __post_init__(self) -> None:
        nodes = node_indices(self.nodes)
        if (not self.block or not self.owner or not nodes
                or any(b <= a for a, b in zip(nodes, nodes[1:]))):
            raise ContractError("invalid_constraint_rows")
        object.__setattr__(self, "nodes", nodes)


def _unit(unit: Unit) -> list[int]:
    return list(unit.powers)


@dataclass(frozen=True)
class SystemLayout:
    """Declared blocks, half-open global ranges, constraint rows and identity."""

    node_sets: tuple[NodeSet, ...]
    blocks: tuple[Block, ...]
    constraints: tuple[ConstraintRows, ...] = ()
    offsets: Mapping[str, slice] = field(init=False)
    size: int = field(init=False)
    identity: str = field(init=False)

    def __post_init__(self) -> None:
        node_sets = {item.id: item for item in self.node_sets}
        blocks = {item.id: item for item in self.blocks}
        if len(node_sets) != len(self.node_sets) or len(blocks) != len(self.blocks):
            raise ContractError("duplicate_layout_id")
        if not self.blocks:
            raise ContractError("empty_layout")
        for item in self.node_sets:
            if item.parent is not None:
                parent = node_sets.get(item.parent)
                if parent is None or parent.parent is not None or item.parent == item.id:
                    raise ContractError("embedding_parent_must_be_a_root_node_set")
                assert item.parent_indices is not None
                if item.parent_indices[-1] >= parent.size:
                    raise ContractError("embedding_outside_parent")
        for block in self.blocks:
            if block.node_set not in node_sets:
                raise ContractError("unknown_node_set")
        owners: set[str] = set()
        claimed: set[tuple[str, int]] = set()
        for rows in self.constraints:
            if rows.block not in blocks:
                raise ContractError("unknown_constraint_block")
            if rows.owner in owners:
                raise ContractError("duplicate_constraint_owner")
            owners.add(rows.owner)
            if rows.nodes[-1] >= node_sets[blocks[rows.block].node_set].size:
                raise ContractError("constraint_outside_block")
            for node in rows.nodes:
                if (rows.block, node) in claimed:
                    raise ContractError("constraint_row_owned_twice")
                claimed.add((rows.block, node))
        offsets: dict[str, slice] = {}
        size = 0
        for block in self.blocks:
            end = size + node_sets[block.node_set].size
            offsets[block.id] = slice(size, end)
            size = end
        object.__setattr__(self, "node_sets", tuple(self.node_sets))
        object.__setattr__(self, "blocks", tuple(self.blocks))
        object.__setattr__(self, "constraints", tuple(self.constraints))
        object.__setattr__(self, "offsets", MappingProxyType(offsets))
        object.__setattr__(self, "size", size)
        declaration = {
            "schema": LAYOUT_SCHEMA,
            "node_sets": [[item.id, item.size, item.parent,
                           None if item.parent_indices is None else list(item.parent_indices)]
                          for item in self.node_sets],
            "blocks": [[block.id, block.node_set, _unit(block.variable_unit),
                        _unit(block.balance_unit)] for block in self.blocks],
            "constraints": sorted([rows.block, rows.owner, list(rows.nodes), _unit(rows.unit)]
                                  for rows in self.constraints),
        }
        encoded = json.dumps(declaration, separators=(",", ":"), sort_keys=True).encode()
        object.__setattr__(self, "identity", sha256(encoded).hexdigest())

    def block(self, block_id: str) -> Block:
        for block in self.blocks:
            if block.id == block_id:
                return block
        raise ContractError("unknown_block")

    def node_set(self, node_set_id: str) -> NodeSet:
        for item in self.node_sets:
            if item.id == node_set_id:
                return item
        raise ContractError("unknown_node_set")

    def block_ranges(self) -> dict[str, tuple[int, int]]:
        """Half-open global ``[start, stop)`` index range of every block."""
        return {name: (span.start, span.stop) for name, span in self.offsets.items()}

    def rows(self, block_id: str, nodes: ArrayLike | None = None) -> NDArray[np.intp]:
        """Global row (= column) indices of ``nodes`` of one block."""
        span = self.offsets.get(block_id)
        if span is None:
            raise ContractError("unknown_block")
        local = (np.arange(span.stop - span.start, dtype=np.intp) if nodes is None
                 else np.asarray(nodes, dtype=np.intp))
        if local.size and (np.min(local) < 0 or np.max(local) >= span.stop - span.start):
            raise ContractError("node_outside_block")
        result: NDArray[np.intp] = span.start + local
        return result

    def view(self, u: ArrayLike, block_id: str) -> NDArray[np.float64]:
        """Read-only view of one block of a global coordinate vector."""
        values = np.asarray(u, dtype=float)
        if values.shape != (self.size,):
            raise ContractError("coordinate_size_mismatch")
        span = self.offsets.get(block_id)
        if span is None:
            raise ContractError("unknown_block")
        result = values[span].view()
        result.setflags(write=False)
        return result

    def constraint_rows(self, owner: str) -> ConstraintRows:
        for rows in self.constraints:
            if rows.owner == owner:
                return rows
        raise ContractError("unknown_constraint_owner")

    def constraint_mask(self) -> NDArray[np.bool_]:
        """Global rows owned by constraint Terms."""
        mask: NDArray[np.bool_] = np.zeros(self.size, dtype=bool)
        for rows in self.constraints:
            mask[self.rows(rows.block, rows.nodes)] = True
        return mask

    def column_map(self, support: str, block_id: str) -> NDArray[np.intp]:
        """Block-local index read by each ``support`` node; -1 where absent.

        Identical node sets map identically; a child set reads its parent
        through its embedding; a parent reads a child block only on the
        embedded nodes.
        """
        source = self.node_set(support)
        target = self.node_set(self.block(block_id).node_set)
        if source.id == target.id:
            return np.arange(source.size, dtype=np.intp)
        if source.parent == target.id:
            assert source.parent_indices is not None
            return np.asarray(source.parent_indices, dtype=np.intp)
        if target.parent == source.id:
            assert target.parent_indices is not None
            result: NDArray[np.intp] = np.full(source.size, -1, dtype=np.intp)
            result[np.asarray(target.parent_indices, dtype=np.intp)] = np.arange(target.size)
            return result
        raise ContractError("unrelated_node_sets")
