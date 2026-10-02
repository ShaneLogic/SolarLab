"""Actual common states exercise independent zero and nonzero right limits."""
from copy import deepcopy

import numpy as np
import pytest

from perovskite_sim.experiments.one_dimensional_mechanism_r1_independent_regular import independent_regular_current
from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import prepare_common_state, restore_common_state
from perovskite_sim.experiments.one_dimensional_mechanism_r1_step import build_initial_step, regular_current_at_state
from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import r1_policy
from perovskite_sim.models.config_loader import load_device_from_yaml
from tests.fixtures.r1_reference import approved_r1_binding
from tests.integration.test_one_dimensional_mechanism_r1 import FIXTURE

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def initial():
    stack, binding, policy = load_device_from_yaml(FIXTURE), approved_r1_binding(), r1_policy()
    prepared = prepare_common_state(stack, 16, binding, policy=policy)
    system, before = restore_common_state(prepared, stack, 16, binding, policy=policy)
    return system, before, policy


@pytest.mark.parametrize("voltage", [0., .005])
def test_actual_initial_regular_reconstruction(initial, voltage):
    system, before, policy = initial
    built = build_initial_step(system, before, voltage, policy=policy)
    report = independent_regular_current(built.system, built.zero_plus, policy=policy,
        require_relative_closure=bool(voltage), reported=built.event["regular_current"])
    assert report["passed"], report["reasons"]
    assert report["relative_current_applicable"] is bool(voltage)
    assert report["metrics"]["charge_balance_normalized"] < 1e-10


def test_actual_published_displacement_or_missing_interface_is_rejected(initial):
    system, before, policy = initial
    built = build_initial_step(system, before, .005, policy=policy)
    published = deepcopy(built.event["regular_current"])
    published["contact_displacement_A_m2"][0] *= 1.001
    published["interface_maxwell_A_m2"] = []
    checked = independent_regular_current(built.system, built.zero_plus, policy=policy, reported=published)
    assert not checked["passed"]
    assert "contact_displacement_A_m2_reported_content_matches" in checked["reasons"]
    assert "interface_maxwell_A_m2_reported_content_matches" in checked["reasons"]


def test_actual_coarse_step_fault_is_rejected_by_computed_physics_not_injected_flags():
    """Omitting contacts from solver stopping recreates the old N64 defect."""
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_dynamics import ControlledPhysicalInterfaceIonSystem
    from perovskite_sim.experiments.one_dimensional_mechanism_r1 import PhysicalInterfaceIonSystem
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import run_r1_step, R1RunError
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_physics_validation import verify_r1_step_physics
    stack, binding = load_device_from_yaml(FIXTURE), approved_r1_binding()
    prepared = prepare_common_state(stack, 64, binding, policy=r1_policy())
    observed = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ControlledPhysicalInterfaceIonSystem, "solver_current_metrics",
                      PhysicalInterfaceIonSystem.transient_current_metrics)
        with pytest.raises(R1RunError, match="contact_internal_current_spread_relative_exceeds_limit") as error:
            run_r1_step(stack, 64, binding, prepared, policy=r1_policy(1., time_substeps=(4, 8, 16)),
                physics_evidence=True, accepted_step_observer=lambda row: observed.append(deepcopy(row)))
    failed = error.value.result
    row = failed["accepted_steps"][-1]
    independent = row["physics_reconstruction"]["independent_physics"]
    assert row["substeps"] == 4 and row["solver_accepted"] is True
    assert row["substeps"] < max(failed["policy"]["refinement_substeps"])
    assert independent["metrics"]["internal_face_current_spread_relative"] < 2e-6
    assert independent["metrics"]["contact_internal_current_spread_relative"] > 2e-6
    assert independent["checks"]["contact_internal_current_spread_relative"]["passed"] is False
    assert independent["assessment"]["content_consistent"] is True
    assert independent["assessment"]["conservation_compliant"] is False
    assert observed[-1]["physical_checks_passed"] is False
    audit = verify_r1_step_physics(stack, 64, binding, prepared, failed, allow_incomplete=True)
    assert audit["independent_physics_passed"] is False
    assert audit["physical_limits_satisfied"] is False
