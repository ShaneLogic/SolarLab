"""Same controlled physical-volume DC/AC operator at the approved reference."""

import numpy as np
import pytest

from tests.fixtures.r1_reference import approved_r1_binding
from tests.integration.test_one_dimensional_mechanism_r1 import FIXTURE
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import prepare_common_state, snapshot
from perovskite_sim.experiments.one_dimensional_mechanism_r1_response import (
    solve_controlled_dc, dc_conductance_study, small_signal_response, compare_transient_tail,
)


@pytest.fixture(scope="module")
def reference():
    stack, binding = load_device_from_yaml(FIXTURE), approved_r1_binding()
    return stack, binding, prepare_common_state(stack, 16, binding)


@pytest.mark.parametrize("control", ["A", "B", "C", "D"])
@pytest.mark.parametrize("voltage", [-.005, .005])
def test_biased_dc_retains_frozen_species_and_inventory(reference, control, voltage):
    stack, binding, prepared = reference
    result = solve_controlled_dc(stack, 16, binding, prepared, control=control, voltage_V=voltage)
    assert result.evidence["certified"]
    assert result.evidence["checks"]["metrics"]["inventory_relative_error"] <= 1e-10
    common = prepared.to_dict()["state"]
    if control in "AC":
        np.testing.assert_array_equal(result.state.positive, common["positive_m3"])
    if control in "AB":
        np.testing.assert_array_equal(result.state.occupancy, common["occupancy"])
    assert result.evidence["terminal_current_A_m2"]*voltage > 0.
    assert len(result.evidence["history"]) <= 101
    assert "left_contact" in result.evidence["physical_face_labels"]
    assert "interface_0_left" in result.evidence["physical_face_labels"]
    assert "interface_0_right" in result.evidence["physical_face_labels"]
    assert "right_contact" in result.evidence["physical_face_labels"]


@pytest.fixture(scope="module")
def conductance_and_ac(reference):
    stack, binding, prepared = reference
    conductance = dc_conductance_study(stack, 16, binding, prepared)
    dc = solve_controlled_dc(stack, 16, binding, prepared)
    ac = small_signal_response(dc, [0., 1e-3, 1., 1e3, 1e6, 1e8])
    return conductance, dc, ac


def test_dc_three_steps_and_independent_ac_zero_frequency_agree(conductance_and_ac):
    conductance, _, ac = conductance_and_ac
    np.testing.assert_array_equal(conductance["half_width_V"], [1e-4, 5e-5, 2.5e-5])
    assert conductance["finest_pair_agrees"]
    g = conductance["conductance_S_m2"][-1]
    assert abs(ac["admittance_S_m2"][0]-g) <= 1e-8+.01*abs(g)
    assert conductance["absolute_error_bound_S_m2"] is None


def test_ac_full_physical_checks_and_impulse_are_separate(conductance_and_ac):
    _, dc, ac = conductance_and_ac
    assert all(np.all(value) for value in ac["checks"].values())
    assert np.all(ac["numerically_eligible_frequency_points"])
    assert not ac["frequency_window_complete"]
    assert not ac["double_domain_consistent"]
    assert "time_window_and_early_interval" in ac["missing_validation"]
    assert len(ac["derivative_levels"]) == 3
    latest = ac["derivative_levels"][-1]
    # Pure dielectric lift fixes populations: inverse sum of face resistance.
    expected = 1./np.sum(1./dc.system.material.poisson_factor.C)
    np.testing.assert_allclose(latest["impulse_capacitance_F_m2"], expected, rtol=1e-7)
    assert np.all(ac["equilibrium_dissipation_sign_observation"])
    assert latest["inventory_response_m2_per_V"].shape == (6, 1)
    assert len(latest["coordinate_actual_maximum_physical_perturbations"]) == dc.system.dimension


def test_frozen_ac_species_stay_fixed(reference):
    stack, binding, prepared = reference
    dc = solve_controlled_dc(stack, 16, binding, prepared, control="A")
    ac = small_signal_response(dc, [0., 1., 1e4])
    for level in ac["derivative_levels"]:
        np.testing.assert_array_equal(level["positive_density_response_m3_per_V"], 0.)
        np.testing.assert_array_equal(level["occupancy_response_per_V"], 0.)
    assert all(np.all(value) for value in ac["checks"].values())


def test_tail_comparison_keeps_unknown_infinite_tail_and_checks_identity(reference):
    stack, binding, prepared = reference
    dc = solve_controlled_dc(stack, 16, binding, prepared, voltage_V=.005)
    state = snapshot(dc.system, dc.state)
    state.update({key: dc.evidence[key] for key in ("prepared_sha256", "control", "voltage_V")})
    initial = prepared.to_dict()["state"]
    result = compare_transient_tail(dc, initial_state=initial, tail_state=state,
                                    tail_regular_current_A_m2=dc.evidence["terminal_current_A_m2"], time_s=1.)
    assert result["all_observables_agree"]
    assert result["infinite_tail_integral_bound_F_m2"] is None
    changed = {**state, "positive_m3": (dc.state.positive*1.1).tolist()}
    assert not compare_transient_tail(dc, initial_state=initial, tail_state=changed,
                                      tail_regular_current_A_m2=dc.evidence["terminal_current_A_m2"],
                                      time_s=1.)["all_observables_agree"]
    with pytest.raises(ValueError, match="identity"):
        compare_transient_tail(dc, initial_state=initial, tail_state={**state, "control": "A"},
                               tail_regular_current_A_m2=0., time_s=1.)


@pytest.mark.parametrize("field", ["trace_state_m3", "trace_potential_V"])
def test_interface_only_tail_mismatch_is_rejected(reference, field):
    stack, binding, prepared = reference
    dc = solve_controlled_dc(stack, 16, binding, prepared, voltage_V=.005)
    state = snapshot(dc.system, dc.state)
    state.update({key: dc.evidence[key] for key in ("prepared_sha256", "control", "voltage_V")})
    altered = np.asarray(state[field])
    state[field] = (altered*1.1 if field == "trace_state_m3" else altered+.001).tolist()
    result = compare_transient_tail(dc, initial_state=prepared.to_dict()["state"], tail_state=state,
                                    tail_regular_current_A_m2=dc.evidence["terminal_current_A_m2"], time_s=1.)
    assert not result["all_observables_agree"]
    assert not result["comparisons"][field]["agrees"]
    assert result["comparisons"]["current_A_m2"]["agrees"]
