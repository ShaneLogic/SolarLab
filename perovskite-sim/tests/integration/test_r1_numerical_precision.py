"""R1 traces that previously stopped at absolute-log quantization or contacts."""
import json
from pathlib import Path

import pytest

from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import r1_policy, run_r1_step
from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import prepare_common_state
from perovskite_sim.experiments.one_dimensional_mechanism_r1_physics_validation import verify_r1_step_physics
from perovskite_sim.models.config_loader import load_device_from_yaml


@pytest.fixture(scope="module")
def inputs():
    root = Path(__file__).resolve().parents[2]
    stack = load_device_from_yaml(root / "tests/fixtures/configs/dynamic_interface_defect_ion_transient_absorber_only.yaml")
    reference = json.loads((root / "tests/fixtures/OneDimensionalMechanismR1ReferenceBindingV1.json").read_text())
    return stack, reference, {}


@pytest.mark.slow
@pytest.mark.parametrize("intervals,factor", [(64, .01), (32, .01), (64, 1.)])
def test_original_failure_representatives_complete_without_gate_changes(inputs, intervals, factor):
    stack, reference, cache = inputs
    if intervals not in cache:
        cache[intervals] = prepare_common_state(stack, intervals, reference, policy=r1_policy())
    policy = r1_policy(factor)
    result = run_r1_step(stack, intervals, reference, cache[intervals], control="D", policy=policy)
    assert result["certificate"]["certified"]
    assert len(result["accepted_steps"]) == 31
    assert all(row["physical_checks_passed"] for row in result["accepted_steps"])
    assert result["certificate"]["metrics"]["nonlinear_residual"] <= .05
    assert policy.maximum_all_face_current_spread_relative == 2e-6
    assert policy.maximum_charge_balance_relative_error == 1e-10
    assert (policy.maximum_newton_iterations, policy.maximum_line_search_steps,
            policy.maximum_near_acceptance_nonmonotone_steps) == (100, 40, 2)


@pytest.mark.slow
def test_strict_trace_roundtrips_exact_coordinates_and_physical_equations(inputs):
    stack, reference, cache = inputs
    if 64 not in cache:
        cache[64] = prepare_common_state(stack, 64, reference, policy=r1_policy())
    result = run_r1_step(stack, 64, reference, cache[64], control="D", policy=r1_policy(.01),
                         physics_evidence=True)
    checked = verify_r1_step_physics(stack, 64, reference, cache[64], result)
    assert checked["certified"] and checked["evidence_matches_equations"]
    assert checked["checked_row_count"] == checked["expected_row_count"] == 31
    assert checked["metrics"] == result["certificate"]["metrics"]
