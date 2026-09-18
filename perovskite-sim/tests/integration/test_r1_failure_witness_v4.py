"""A solver fault produces a real failed iterate, then an unmodified replay."""
from copy import deepcopy

import numpy as np
import pytest

from perovskite_sim.experiments import interface_defect_transient as transient
from perovskite_sim.experiments.one_dimensional_mechanism_r1_failure_reconstruction import rebuild_failure_witness
from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import prepare_common_state
from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import r1_policy, run_r1_step, R1RunError
from perovskite_sim.models.config_loader import load_device_from_yaml
from tests.fixtures.r1_reference import approved_r1_binding
from tests.integration.test_one_dimensional_mechanism_r1 import FIXTURE

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def failed():
    stack, binding, policy = load_device_from_yaml(FIXTURE), approved_r1_binding(), r1_policy()
    prepared = prepare_common_state(stack, 16, binding, policy=policy)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(transient, "spsolve", lambda matrix, rhs: np.zeros_like(rhs))
        with pytest.raises(R1RunError, match="line search stalled") as error:
            run_r1_step(stack, 16, binding, prepared, policy=policy, physics_evidence=True)
    return stack, binding, prepared, error.value.result


def test_actual_failed_newton_state_is_saved_and_recomputed(failed):
    stack, binding, prepared, result = failed
    witness = result["failure"]["numerical_evidence"]
    assert witness["terminal_state_available"] is True
    assert witness["time_s"] == 1e-9 and witness["previous_time_s"] == 0.
    assert witness["substeps"] == 1 and len(result["accepted_steps"]) == 1
    report = rebuild_failure_witness(stack, 16, binding, prepared, result)
    assert report["terminal_state_physics_recomputed"] is True
    assert report["scaled_nonlinear_residual"] > .05
    assert report["accepted"] is False


@pytest.mark.parametrize("field", ["coordinate", "storage_scale", "scaled_residual_vector"])
def test_failed_iterate_or_scale_tampering_is_rejected(failed, field):
    stack, binding, prepared, result = failed
    wrong = deepcopy(result)
    values = wrong["failure"]["numerical_evidence"][field]
    values[0] = values[0] * 1.01 + 1e-8
    with pytest.raises(ValueError, match="failed Newton witness mismatch"):
        rebuild_failure_witness(stack, 16, binding, prepared, wrong)


def test_missing_terminal_state_does_not_become_passed_physics(failed):
    stack, binding, prepared, result = failed
    missing = deepcopy(result)
    missing["failure"].pop("numerical_evidence")
    report = rebuild_failure_witness(stack, 16, binding, prepared, missing)
    assert report["available"] is False and report["terminal_state_physics_recomputed"] is None
