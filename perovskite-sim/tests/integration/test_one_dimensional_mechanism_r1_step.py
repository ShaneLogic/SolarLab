"""Independent ideal-step electrostatic and right-limit current checks."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import spsolve

from perovskite_sim.constants import EPS_0, Q
from perovskite_sim.experiments.interface_defect_ion_transient import InterfaceDefectIonTransientPolicy
from perovskite_sim.experiments.one_dimensional_mechanism_r1_dynamics import R1DynamicsControls
from perovskite_sim.experiments.one_dimensional_mechanism_r1_step import (
    build_initial_step, regular_current_at_state,
)
from perovskite_sim.models.config_loader import load_device_from_yaml
from tests.fixtures.r1_reference import approved_r1_binding


FIXTURE = Path(__file__).parents[1] / "fixtures/configs/dynamic_interface_defect_ion_transient_absorber_only.yaml"


@pytest.fixture(scope="module")
def common_state():
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import prepare_common_state

    stack = load_device_from_yaml(FIXTURE)
    binding = approved_r1_binding()
    policy = InterfaceDefectIonTransientPolicy(
        maximum_newton_iterations=100, maximum_line_search_steps=40,
        maximum_near_acceptance_nonmonotone_steps=2,
        maximum_ion_inventory_relative_drift=1e-10,
    )
    return stack, binding, policy, prepare_common_state(stack, 16, binding, policy=policy)


def _restore(common_state, control="D"):
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import restore_common_state

    stack, binding, policy, prepared = common_state
    system, initial = restore_common_state(
        prepared, stack, 16, binding, controls=R1DynamicsControls.from_label(control), policy=policy,
    )
    return system, initial, policy


@pytest.mark.slow
@pytest.mark.parametrize("voltage", [-.005, .0025, .005])
def test_fixed_population_step_matches_dielectric_oracle_and_charge_sign(common_state, voltage):
    system, initial, policy = _restore(common_state)
    step = build_initial_step(system, initial, voltage, policy=policy)
    expected_c = EPS_0 * 10. / (2e-7)
    assert step.event["capacitance_infinity_F_m2"] == pytest.approx(expected_c, rel=2e-15)
    assert step.event["impulse_charge_C_m2"] == pytest.approx(expected_c * voltage, rel=2e-13)
    np.testing.assert_allclose(
        step.event["electrode_charge_jump_C_m2"],
        system.polarity * expected_c * voltage * np.array([1., -1.]), rtol=2e-11, atol=1e-18,
    )
    # Both layers have identical dielectric constants, so the exact voltage
    # increment is linear even though equilibrium free charge is nonuniform.
    np.testing.assert_allclose(
        step.zero_plus.phi - initial.phi,
        -system.polarity * voltage * system.grid / system.grid[-1], rtol=2e-12, atol=2e-17,
    )
    assert initial.sheet_charge[0] != 0.
    for field in ("n", "p", "positive", "occupancy", "sheet_charge"):
        np.testing.assert_array_equal(getattr(step.zero_plus, field), getattr(initial, field))
    assert step.event["full_charge_jump_C_m2"] == 0.
    # A fresh evaluate of the returned system must not apply the lift twice.
    re_evaluated = step.system.evaluate(step.system.initial_coordinate(), voltage)
    np.testing.assert_array_equal(re_evaluated.phi, step.zero_plus.phi)


@pytest.mark.slow
@pytest.mark.parametrize("control", ["A", "B", "C", "D"])
def test_initial_regular_current_closes_and_retains_fixed_charge(common_state, control):
    system, initial, policy = _restore(common_state, control)
    step = build_initial_step(system, initial, .005, policy=policy)
    record = step.event["regular_current"]
    assert record["contact_internal_current_spread_relative"] <= 2e-6
    assert record["interface_current_spread_relative"] <= 2e-6
    assert record["charge_balance_normalized"] <= 1e-10
    assert record["differentiated_poisson_normalized"] <= 1e-10
    assert step.event["certified"]
    np.testing.assert_array_equal(step.zero_plus.sheet_charge, initial.sheet_charge)
    assert abs(step.event["impulse_charge_C_m2"]) > 0.
    if control in ("A", "C"):
        np.testing.assert_array_equal(step.zero_plus.positive_rate, 0.)
    if control in ("A", "B"):
        np.testing.assert_array_equal(
            step.zero_plus.rate[2*system.interior_count:2*system.interior_count+system.interface_count], 0.,
        )


@pytest.mark.slow
def test_regular_displacement_matches_full_dae_tangent(common_state):
    system, initial, policy = _restore(common_state)
    step = build_initial_step(system, initial, .005, policy=policy)
    state = step.zero_plus
    matrix = sparse.vstack((state.storage_jacobian, state.poisson_jacobian, state.local_jacobian), format="csr")
    rhs = np.r_[state.rate, np.zeros(state.poisson_jacobian.shape[0] + state.local_jacobian.shape[0])]
    scale = np.asarray(abs(matrix).max(axis=1).toarray()).ravel()
    derivative = spsolve(sparse.diags(1./scale) @ matrix, rhs/scale)
    phi_dot = np.zeros(system.node_count)
    phi_dot[1:-1] = system.thermal_voltage * derivative[system.potential_slice]
    current = regular_current_at_state(step.system, state, policy=policy)
    np.testing.assert_allclose(current.evidence["potential_derivative_V_s"], phi_dot, rtol=2e-9, atol=1e-6)
    residual = matrix @ derivative - rhs
    assert np.max(np.abs(residual/scale)) <= 2e-9 * max(np.max(np.abs(rhs/scale)), 1.)


@pytest.mark.slow
def test_zero_voltage_has_no_impulse(common_state):
    system, initial, policy = _restore(common_state)
    voltage_before = (system.material.V_bi_bc - initial.phi[-1]) / system.polarity
    step = build_initial_step(system, initial, voltage_before, policy=policy)
    assert step.event["impulse_charge_C_m2"] == 0.
    # A sub-roundoff consistency correction is saved separately from the
    # exactly zero physical voltage impulse, under the existing Gauss bound.
    assert max(abs(np.asarray(step.event["contact_displacement_jump_C_m2"]))) <= Q * 1e15 * 1e-10
    assert not step.event["regular_current"]["relative_current_certified"]
    assert step.event["regular_current"]["relative_current_status"] == "relative_current_not_certified_zero_excitation"


@pytest.mark.slow
def test_initial_algebraic_defect_is_solved_at_fixed_populations(common_state):
    system, initial, policy = _restore(common_state)
    step = build_initial_step(system, initial, .005, policy=policy)
    record = step.event["fixed_population_algebraic_correction"]
    before = np.asarray(record["poisson_residual_before_C_m2"])
    correction = np.asarray(record["potential_correction_V"])
    capacitance = system.material.poisson_factor.C
    # Independent dense linear solve, rather than the implementation's cached
    # Poisson factors. The inherited defect must be cancelled algebraically.
    matrix = np.diag(-(capacitance[:-1] + capacitance[1:]))
    matrix += np.diag(capacitance[1:-1], 1) + np.diag(capacitance[1:-1], -1)
    expected = np.linalg.solve(matrix, -before)
    np.testing.assert_allclose(correction[1:-1], expected, rtol=2e-14, atol=1e-32)
    assert 0. < max(abs(correction)) < 1e-12
    assert max(abs(np.asarray(record["poisson_residual_after_C_m2"]))) < max(abs(before)) * 1e-10
    assert max(abs(np.asarray(record["linear_poisson_residual_C_m2"]))) < max(abs(before)) * 1e-10
    local_before = np.asarray(record["local_electrostatic_residual_before"])
    local_after = np.asarray(record["local_electrostatic_residual_after"])
    assert np.max(abs(local_after[:, 1])) < max(np.max(abs(local_before[:, 1])), 1e-30) * 1e-10
    assert record["direct_poisson_residual_after_C_m2"] is not None
    for field in ("n", "p", "positive", "occupancy", "sheet_charge"):
        np.testing.assert_array_equal(getattr(step.zero_plus, field), getattr(initial, field))


@pytest.mark.slow
@pytest.mark.parametrize("control", ["A", "B", "C", "D"])
def test_first_nanosecond_and_short_trace_close_after_separate_impulse(common_state, control):
    from perovskite_sim.experiments.interface_defect_transient import _integrate_trace
    from perovskite_sim.experiments.one_dimensional_mechanism_r1 import physical_step_record

    system, initial, policy = _restore(common_state, control)
    fields = (
        "storage_relative_tolerance", "carrier_storage_atol_m3", "interface_storage_atol_m2",
        "ion_storage_atol_m3", "poisson_relative_tolerance", "poisson_atol_C_m2",
        "interface_algebraic_relative_tolerance", "interface_potential_atol_V",
        "interface_gauss_atol_C_m2", "interface_flux_atol_m2_s",
    )
    policy = replace(policy, **{key: getattr(policy, key) * .1 for key in fields})
    step = build_initial_step(system, initial, .005, policy=policy)
    times = np.array([0., 1e-9, 1e-8, 1e-6, 1e-4])
    records = []

    def observe(working, state, previous, dt, time, substeps, residual):
        if previous is not None:
            record = physical_step_record(working, state, previous, dt)
            assert record["contact_internal_current_spread_relative"] <= 2e-6
            assert record["charge_balance_normalized"] <= 1e-10
            assert record["gauss_normalized"] <= 1e-10
            records.append((time, dt))

    trace = _integrate_trace(
        step.system, times, np.full(times.size, .005), 4, policy,
        initial_state=step.zero_plus, initial_current_metrics=step.initial_current_metrics,
        accepted_step_observer=observe,
    )
    assert len(records) == 16
    assert records[0] == (2.5e-10, 2.5e-10)
    assert trace.maximum_face_spread <= 2e-6
    assert trace.maximum_interface_current_error <= 2e-6
    assert trace.maximum_charge_balance_error <= 1e-10
