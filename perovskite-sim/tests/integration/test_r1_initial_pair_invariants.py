"""Exercise pair-word guards through the actual formal 0-/0+ entry point."""

import copy
from pathlib import Path

import numpy as np
import pytest

from perovskite_sim.experiments import one_dimensional_mechanism_r1_step as step
from perovskite_sim.experiments.interface_defect_ion_transient import (
    InterfaceDefectIonTransientError,
)
from perovskite_sim.experiments.one_dimensional_mechanism_r1_dynamics import R1DynamicsControls
from perovskite_sim.experiments.one_dimensional_mechanism_r1_precision import CompensatedR1System
from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import r1_policy
from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import (
    prepare_common_state,
    restore_common_state,
)
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.physics.compensated import DD
from tests.fixtures.r1_reference import approved_r1_binding


FIXED_FIELDS = ("n_m3", "p_m3", "positive_m3", "occupancy", "sheet_charge_C_m2")
REBASE_FIELDS = (
    "phi_V", "dqfn_V", "dqfp_V", "n_m3", "p_m3", "positive_m3",
    "occupancy", "trace_potential_V", "trace_state_m3", "sheet_charge_C_m2",
)


@pytest.fixture(scope="module")
def pair_origin():
    fixture = Path(__file__).parents[1] / "fixtures/configs/dynamic_interface_defect_ion_transient_absorber_only.yaml"
    stack, binding, policy = load_device_from_yaml(fixture), approved_r1_binding(), r1_policy()
    prepared = prepare_common_state(stack, 16, binding, policy=policy, backend="pair")
    system, initial = restore_common_state(prepared, stack, 16, binding,
        controls=R1DynamicsControls.from_label("D"), policy=policy, backend="pair")
    return system, initial, policy


@pytest.fixture
def pair_case(pair_origin):
    system, initial, policy = pair_origin
    # Fault injection may replace the state's fine mapping, never the shared
    # prepared fixture or its immutable DD values.
    state = copy.copy(initial)
    state.fine = initial.fine.copy()
    return system, state, policy


def _perturb_low(value):
    high, low = value.hi.copy(), value.lo.copy()
    index = tuple(np.argwhere(high != 0.)[0])
    low[index] = 0. if low[index] != 0. else np.spacing(abs(high[index])) / 4.
    changed = DD(high, low)
    np.testing.assert_array_equal(changed.hi, value.hi)
    assert not np.array_equal(changed.lo, value.lo)
    return changed


def test_formal_pair_step_retains_fixed_words_and_exact_zero_coordinate_state(pair_case):
    system, initial, policy = pair_case
    before = {field: initial.fine[field].copy() for field in REBASE_FIELDS}
    result = step.build_initial_step(system, initial, .005, policy=policy, backend="pair")
    assert any(np.any(value.lo != 0.) for value in before.values())
    for field in FIXED_FIELDS:
        for word in ("hi", "lo"):
            np.testing.assert_array_equal(getattr(before[field], word),
                                          getattr(result.zero_plus.fine[field], word))
    restored = result.system.evaluate(np.zeros(result.system.dimension), .005)
    for field in REBASE_FIELDS:
        for word in ("hi", "lo"):
            np.testing.assert_array_equal(getattr(result.zero_plus.fine[field], word),
                                          getattr(restored.fine[field], word))
    assert result.event["certified"]
    assert result.event["schema"] == "R1IdealStepV1"
    assert result.event["impulse_charge_C_m2"] != 0.


def test_formal_pair_step_rejects_population_low_loss_before_physics_checks(pair_case, monkeypatch):
    system, initial, policy = pair_case
    original = step._solve_local_carriers

    def altered(*args, **kwargs):
        state, iterations, norm = original(*args, **kwargs)
        state.fine["n_m3"] = _perturb_low(state.fine["n_m3"])
        return state, iterations, norm

    monkeypatch.setattr(step, "_solve_local_carriers", altered)
    with pytest.raises(InterfaceDefectIonTransientError) as failure:
        step.build_initial_step(system, initial, .005, policy=policy, backend="pair")
    assert failure.value.result == {
        "stage": "fixed_population_voltage_jump", "field": "n_m3", "word": "lo",
        "requirement": "exact_primary_pair_word_identity",
    }


def test_formal_pair_step_detects_rebase_mutating_its_input(pair_case, monkeypatch):
    system, initial, policy = pair_case
    original = CompensatedR1System.rebase

    def altered(self, previous):
        working, local = original(self, previous)
        previous.fine = {**previous.fine, "phi_V": _perturb_low(previous.fine["phi_V"])}
        return working, local

    monkeypatch.setattr(CompensatedR1System, "rebase", altered)
    with pytest.raises(InterfaceDefectIonTransientError) as failure:
        step.build_initial_step(system, initial, .005, policy=policy, backend="pair")
    assert failure.value.result["stage"] == "zero_minus_rebase_input"
    assert failure.value.result["field"] == "phi_V"
    assert failure.value.result["word"] == "lo"


def test_formal_pair_step_rejects_low_loss_on_zero_coordinate_restore(pair_case, monkeypatch):
    system, initial, policy = pair_case
    original_rebase, original_evaluate = CompensatedR1System.rebase, CompensatedR1System.evaluate
    rebases = 0

    def tracked_rebase(self, previous):
        nonlocal rebases
        result = original_rebase(self, previous)
        rebases += 1
        return result

    def altered_evaluate(self, *args, **kwargs):
        state = original_evaluate(self, *args, **kwargs)
        if rebases == 2:
            state.fine["phi_V"] = _perturb_low(state.fine["phi_V"])
        return state

    monkeypatch.setattr(CompensatedR1System, "rebase", tracked_rebase)
    monkeypatch.setattr(CompensatedR1System, "evaluate", altered_evaluate)
    with pytest.raises(InterfaceDefectIonTransientError) as failure:
        step.build_initial_step(system, initial, .005, policy=policy, backend="pair")
    assert failure.value.result["stage"] == "zero_plus_zero_coordinate_restore"
    assert failure.value.result["field"] == "phi_V"
    assert failure.value.result["word"] == "lo"
