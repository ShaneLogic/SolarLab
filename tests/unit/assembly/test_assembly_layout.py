"""SystemLayout indexing, identity and constraint ownership; MEASURE expressions."""

from __future__ import annotations

import numpy as np
import pytest

from solarlab.contracts import ONE, VOLT
from solarlab.contracts.errors import ContractError
from solarlab.equations.assembly import MEASURE, Block, ConstraintRows, NodeSet, SystemLayout
from solarlab.equations.assembly.weighted import evaluate, measure_count


def electronic(count: int = 5, *, reservoirs: bool = True) -> SystemLayout:
    constraints = [ConstraintRows("potential", (0, count - 1), "potential.contacts", VOLT)]
    if reservoirs:
        constraints.append(ConstraintRows("electron", (count - 1,), "electron.reservoirs", ONE))
    return SystemLayout((NodeSet("nodes", count),),
                        (Block("potential", "nodes", VOLT, ONE), Block("electron", "nodes", ONE, ONE),
                         Block("hole", "nodes", ONE, ONE)), tuple(constraints))


def test_declared_order_gives_half_open_ranges_and_global_rows() -> None:
    layout = electronic()
    assert layout.size == 15
    assert layout.block_ranges() == {"potential": (0, 5), "electron": (5, 10), "hole": (10, 15)}
    np.testing.assert_array_equal(layout.rows("hole", [0, 4]), [10, 14])
    np.testing.assert_array_equal(np.flatnonzero(layout.constraint_mask()), [0, 4, 9])
    u = np.arange(15.0)
    view = layout.view(u, "electron")
    np.testing.assert_array_equal(view, [5, 6, 7, 8, 9])
    assert not view.flags.writeable
    with pytest.raises(ContractError):
        layout.rows("hole", [5])


def test_identity_is_content_addressed_and_order_sensitive() -> None:
    assert electronic().identity == electronic().identity
    assert electronic().identity != electronic(reservoirs=False).identity
    assert electronic().identity != electronic(6).identity
    swapped = SystemLayout((NodeSet("nodes", 5),),
                           (Block("electron", "nodes", ONE, ONE), Block("potential", "nodes", VOLT, ONE),
                            Block("hole", "nodes", ONE, ONE)),
                           electronic().constraints)
    assert swapped.identity != electronic().identity
    assert swapped.block_ranges()["potential"] == (5, 10)
    # Constraint declaration order is not part of the meaning.
    reordered = SystemLayout(electronic().node_sets, electronic().blocks, tuple(reversed(electronic().constraints)))
    assert reordered.identity == electronic().identity


def test_constraint_rows_have_exactly_one_owner() -> None:
    nodes, blocks = electronic().node_sets, electronic().blocks
    with pytest.raises(ContractError, match="constraint_row_owned_twice"):
        SystemLayout(nodes, blocks, (ConstraintRows("potential", (0,), "a", VOLT),
                                     ConstraintRows("potential", (0, 4), "b", VOLT)))
    with pytest.raises(ContractError, match="duplicate_constraint_owner"):
        SystemLayout(nodes, blocks, (ConstraintRows("potential", (0,), "a", VOLT),
                                     ConstraintRows("hole", (0,), "a", ONE)))
    with pytest.raises(ContractError, match="constraint_outside_block"):
        SystemLayout(nodes, blocks, (ConstraintRows("potential", (5,), "a", VOLT),))
    with pytest.raises(ContractError, match="invalid_constraint_rows"):
        ConstraintRows("potential", (3, 1), "a", VOLT)
    with pytest.raises(ContractError, match="duplicate_layout_id"):
        SystemLayout(nodes, blocks + (Block("hole", "nodes", ONE, ONE),))


def test_embedded_node_sets_map_columns_both_ways() -> None:
    layout = SystemLayout((NodeSet("nodes", 6), NodeSet("ions", 3, "nodes", (2, 3, 4))),
                          (Block("potential", "nodes", VOLT, ONE), Block("ion", "ions", ONE, ONE)))
    assert layout.block_ranges() == {"potential": (0, 6), "ion": (6, 9)}
    np.testing.assert_array_equal(layout.column_map("ions", "potential"), [2, 3, 4])
    np.testing.assert_array_equal(layout.column_map("nodes", "ion"), [-1, -1, 0, 1, 2, -1])
    np.testing.assert_array_equal(layout.column_map("ions", "ion"), [0, 1, 2])
    with pytest.raises(ContractError, match="embedding_indices_must_increase"):
        NodeSet("ions", 3, "nodes", (4, 3, 2))
    with pytest.raises(ContractError, match="embedding_outside_parent"):
        SystemLayout((NodeSet("nodes", 3), NodeSet("ions", 2, "nodes", (2, 3))),
                     (Block("ion", "ions", ONE, ONE),))


def test_measure_expressions_keep_the_written_product_order() -> None:
    rng = np.random.default_rng(2)
    w, n, p = (np.exp(rng.uniform(-40, 40, 64)) for _ in range(3))
    q, vt = 1.602176634e-19, 0.025852
    expression = -q * MEASURE * (n + p) / vt
    assert measure_count(expression) == 1
    assert evaluate(expression, w).tobytes() == (-q * w * (n + p) / vt).tobytes()
    # The same value with the measure applied last generally differs in its last bits.
    assert evaluate(MEASURE * (-q * (n + p) / vt), w).tobytes() == (w * (-q * (n + p) / vt)).tobytes()
    assert evaluate(-(MEASURE * p) / vt, w).tobytes() == (-(w * p) / vt).tobytes()
    with pytest.raises(ContractError, match="weighted_sum_not_allowed"):
        _ = MEASURE * n + p
    with pytest.raises(ContractError, match="measure_in_denominator"):
        _ = n / MEASURE
    assert measure_count(MEASURE * MEASURE * n) == 2
