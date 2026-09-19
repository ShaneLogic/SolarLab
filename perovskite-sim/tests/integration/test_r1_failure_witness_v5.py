"""A saved real Newton iterate cannot opt out of content reconstruction."""
from copy import deepcopy

import pytest

from perovskite_sim.experiments.one_dimensional_mechanism_r1_failure_reconstruction import rebuild_failure_witness
from perovskite_sim.experiments.one_dimensional_mechanism_r1_result_validation import failure_scope_report
from tests.integration.test_r1_failure_witness_v4 import failed

pytestmark = pytest.mark.slow


def test_saved_newton_state_cannot_disable_reconstruction_by_relabelling_it(failed):
    stack, binding, prepared, result = failed
    original = rebuild_failure_witness(stack, 16, binding, prepared, result)
    assert original["terminal_state_physics_recomputed"] is True
    altered = deepcopy(result)
    witness = altered["failure"]["numerical_evidence"]
    witness["terminal_state_available"] = False
    witness["diagnostics"]["scaled_nonlinear_residual"] = .001
    witness["witness_collection_error"] = {"type": "RuntimeError", "message": "terminal not collected"}
    with pytest.raises(ValueError, match="unavailable Newton witness contains terminal state"):
        rebuild_failure_witness(stack, 16, binding, prepared, altered)


def test_actual_terminal_reconstruction_is_reported_separately_from_prefix(failed):
    stack, binding, prepared, result = failed
    terminal = rebuild_failure_witness(stack, 16, binding, prepared, result)
    prefix = {"checked_row_count": len(result["accepted_steps"]), "physical_limits_satisfied": True}
    report = failure_scope_report(result, prefix, terminal_reconstruction=terminal)
    assert report["saved_last_row_physics_verified"] is True
    assert report["terminal_state_reconstruction_available"] is True
    assert report["terminal_state_physics_verified"] is True
    assert report["terminal_state_physics_passed"] is terminal["terminal_state_physics_passed"]
    assert report["scientifically_accepted"] is False
