"""Every current factory is finite and deterministic under independent builds."""

from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pytest
from scipy.sparse import issparse

from support.assembly_parity import compare
from support.assembly_references import REFERENCE_FACTORIES, STATE_NAMES


@pytest.mark.parametrize("name", tuple(REFERENCE_FACTORIES))
def test_independent_reference_evaluations(name: str, request: pytest.FixtureRequest) -> None:
    started = time.perf_counter()
    first, second = REFERENCE_FACTORIES[name](), REFERENCE_FACTORIES[name]()
    assert first.device is not second.device
    assert tuple(state.name for state in first.states) == STATE_NAMES
    assert first.blocks == second.blocks
    assert first.device.node_count == (65 if first.family == "dc2d" else 13)
    for state, other in zip(first.states, second.states):
        assert state.u is not other.u and state.u.tobytes() == other.u.tobytes()
        before = state.u.tobytes(), state.previous_u.tobytes(), state.udot.tobytes()
        left, right = first.extract(state), second.extract(other)
        assert left.provenance == right.provenance
        assert left.provenance["target_class"].rsplit(".", 1)[-1] == first.target_class.split("[")[0]
        assert left.provenance["state_sha256"] == hashlib.sha256(state.u.tobytes()).hexdigest()
        for quantity in left.quantities().values():
            values = quantity.data if issparse(quantity) else quantity
            assert np.all(np.isfinite(values))
        compare(left, right).assert_equal()
        assert before == (state.u.tobytes(), state.previous_u.tobytes(), state.udot.tobytes())
        if left.M is not None:
            np.testing.assert_array_equal(left.S_u.toarray(), -left.M.toarray())
        if first.family.endswith("constrained"):
            assert left.S is left.M is left.dS_du is None
        if first.family in {"be", "ionic_be", "multiple_be"}:
            assert {"R_step", "J_step"} <= left.quantities().keys()
        if first.family in {"ionic_ac", "multiple_ac"}:
            assert {"A_ac", "R_ac", "J_ac", "M_ac", "rhs_ac", "rhs_rate_ac", "fixed_ac"} <= left.quantities().keys()
            assert "ac_fixed" in left.boundary_masks
    # Small JSON metadata only; no physical arrays are stored in the harness.
    request.node.user_properties.append(("assembly_factory", json.dumps({
        "factory": name, "target": first.target_class, "states": len(first.states),
        "layout": [[block.name, block.start, block.stop] for block in first.blocks],
        "runtime_s": time.perf_counter() - started, "nonlinear_solves": 0,
    }, sort_keys=True)))


def test_registry_covers_the_existing_model_families() -> None:
    assert {factory.family for factory in REFERENCE_FACTORIES.values()} == {
        "ionic", "ionic_constrained", "multiple", "multiple_constrained", "electronic", "dc2d",
        "ionic_ac", "multiple_ac", "be", "ionic_be", "multiple_be",
    }
    assert REFERENCE_FACTORIES["electronic:ct"].target_class == "DC1DProblem[field_transport]"
    assert not any(factory.family == "electronic" and factory.variant == "fixed"
                   for factory in REFERENCE_FACTORIES.values())
