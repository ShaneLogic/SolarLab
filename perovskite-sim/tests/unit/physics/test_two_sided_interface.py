"""Constitutive gates for the two-sided zero-volume interface element."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from perovskite_sim.constants import EPS_0, Q
from perovskite_sim.physics.fermi_dirac import fermi_dirac_half
from perovskite_sim.physics.two_sided_interface import (
    EquilibriumReferencedSheetCharge,
    TwoSidedBulkState,
    TwoSidedInterfaceGeometry,
    TwoSidedInterfacePhysics,
    build_two_sided_interface_stencils,
    carrier_balance_and_jacobian,
    electrostatic_trace_residual_and_jacobian,
    equilibrium_referenced_electrostatic_trace_balance,
    equilibrium_referenced_two_sided_balance,
    remove_shared_interface_nodes,
    solve_electrostatic_traces,
    solve_equilibrium_referenced_two_sided_interface,
    shared_trap_occupancy,
    shared_trap_occupancy_and_log_jacobian,
    solve_two_sided_interface,
)


def _geometry(**updates) -> TwoSidedInterfaceGeometry:
    return replace(
        TwoSidedInterfaceGeometry(
            left_distance_m=3.0e-9,
            right_distance_m=7.0e-9,
            eps_r_left=12.0,
            eps_r_right=24.0,
        ),
        **updates,
    )


def _physics(**updates) -> TwoSidedInterfacePhysics:
    return replace(
        TwoSidedInterfacePhysics(
            thermal_voltage_V=0.025852,
            temperature_K=300.0,
            D_n_left_m2_s=1.0e-4,
            D_n_right_m2_s=2.0e-4,
            D_p_left_m2_s=8.0e-5,
            D_p_right_m2_s=1.5e-4,
            N_C_left_m3=1.0e25,
            N_C_right_m3=2.0e25,
            N_V_left_m3=1.5e25,
            N_V_right_m3=3.0e25,
            richardson_n_A_m2_K2=1.0e6,
            richardson_p_A_m2_K2=8.0e5,
            transmission=1.0e-4,
        ),
        **updates,
    )


def _bulk(**updates) -> TwoSidedBulkState:
    return replace(
        TwoSidedBulkState(
            phi_left_V=-0.02,
            phi_right_V=0.06,
            n_left_m3=2.0e20,
            p_left_m3=8.0e19,
            n_right_m3=6.0e19,
            p_right_m3=3.0e20,
        ),
        **updates,
    )


def _bulk_coordinates(bulk: TwoSidedBulkState) -> np.ndarray:
    return np.array(
        [
            bulk.phi_left_V,
            bulk.phi_right_V,
            np.log(bulk.n_left_m3),
            np.log(bulk.p_left_m3),
            np.log(bulk.n_right_m3),
            np.log(bulk.p_right_m3),
        ]
    )


def _bulk_from_coordinates(values: np.ndarray) -> TwoSidedBulkState:
    return TwoSidedBulkState(
        phi_left_V=float(values[0]),
        phi_right_V=float(values[1]),
        n_left_m3=float(np.exp(values[2])),
        p_left_m3=float(np.exp(values[3])),
        n_right_m3=float(np.exp(values[4])),
        p_right_m3=float(np.exp(values[5])),
    )


def test_electrostatic_traces_obey_jump_and_gauss_law():
    geometry = _geometry(
        potential_jump_right_minus_left_V=0.015,
        fixed_sheet_charge_C_m2=2.0e-5,
    )
    bulk = _bulk()
    traces = solve_electrostatic_traces(geometry, bulk)
    values = np.array([traces.phi_left_V, traces.phi_right_V])
    residual, jacobian = electrostatic_trace_residual_and_jacobian(
        values, geometry, bulk
    )

    np.testing.assert_allclose(residual, 0.0, rtol=0.0, atol=2.0e-18)
    step = 1.0e-7
    finite_difference = np.empty((2, 2))
    for column in range(2):
        plus = values.copy()
        minus = values.copy()
        plus[column] += step
        minus[column] -= step
        finite_difference[:, column] = (
            electrostatic_trace_residual_and_jacobian(plus, geometry, bulk)[0]
            - electrostatic_trace_residual_and_jacobian(minus, geometry, bulk)[0]
        ) / (2.0 * step)
    np.testing.assert_allclose(jacobian, finite_difference, rtol=2.0e-10)


def test_zero_charge_no_dipole_reduces_to_series_dielectric_interpolation():
    geometry = _geometry()
    bulk = _bulk()
    traces = solve_electrostatic_traces(geometry, bulk)
    capacitance_left = EPS_0 * geometry.eps_r_left / geometry.left_distance_m
    capacitance_right = EPS_0 * geometry.eps_r_right / geometry.right_distance_m
    expected = (
        capacitance_left * bulk.phi_left_V
        + capacitance_right * bulk.phi_right_V
    ) / (capacitance_left + capacitance_right)

    assert traces.phi_left_V == pytest.approx(expected, rel=1.0e-14)
    assert traces.phi_right_V == pytest.approx(expected, rel=1.0e-14)


def test_deduplicated_boundary_node_is_not_used_as_a_bulk_reservoir():
    grid = np.array([0.0, 2.0e-9, 5.0e-9, 6.0e-9, 9.0e-9])
    (stencil,) = build_two_sided_interface_stencils(grid, [5.0e-9])

    assert stencil.shared_boundary_node == 2
    assert stencil.left_bulk_node == 1
    assert stencil.right_bulk_node == 3
    assert stencil.left_distance_m == pytest.approx(3.0e-9)
    assert stencil.right_distance_m == pytest.approx(1.0e-9)


def test_interface_without_a_shared_node_uses_strict_bracketing_nodes():
    grid = np.array([0.0, 2.0e-9, 6.0e-9, 9.0e-9])
    (stencil,) = build_two_sided_interface_stencils(grid, [5.0e-9])

    assert stencil.shared_boundary_node is None
    assert stencil.left_bulk_node == 1
    assert stencil.right_bulk_node == 2
    assert stencil.left_distance_m == pytest.approx(3.0e-9)
    assert stencil.right_distance_m == pytest.approx(1.0e-9)


def test_shared_interface_node_removal_preserves_all_other_grid_points():
    grid = np.array([0.0, 2.0e-9, 5.0e-9, 6.0e-9, 9.0e-9])
    reduced = remove_shared_interface_nodes(grid, [5.0e-9])

    np.testing.assert_array_equal(
        reduced,
        np.array([0.0, 2.0e-9, 6.0e-9, 9.0e-9]),
    )
    (stencil,) = build_two_sided_interface_stencils(reduced, [5.0e-9])
    assert stencil.shared_boundary_node is None


def test_heterojunction_common_quasi_fermi_state_has_zero_flux():
    geometry = _geometry(
        left_distance_m=5.0e-9,
        right_distance_m=5.0e-9,
        eps_r_left=20.0,
        eps_r_right=20.0,
    )
    electron_step = 0.18
    hole_step = -0.08
    physics = _physics(
        conduction_band_step_eV=electron_step,
        hole_transport_step_eV=hole_step,
    )
    eta_n_left = -2.0
    eta_p_left = -2.5
    eta_n_right = eta_n_left - electron_step / physics.thermal_voltage_V
    eta_p_right = eta_p_left - hole_step / physics.thermal_voltage_V
    state = np.array(
        [
            physics.N_C_left_m3 * fermi_dirac_half(eta_n_left),
            physics.N_V_left_m3 * fermi_dirac_half(eta_p_left),
            physics.N_C_right_m3 * fermi_dirac_half(eta_n_right),
            physics.N_V_right_m3 * fermi_dirac_half(eta_p_right),
        ]
    )
    bulk = TwoSidedBulkState(
        phi_left_V=0.0,
        phi_right_V=0.0,
        n_left_m3=state[0],
        p_left_m3=state[1],
        n_right_m3=state[2],
        p_right_m3=state[3],
    )
    balance = carrier_balance_and_jacobian(
        np.log(state), geometry, physics, bulk
    )
    bulk_scale = max(
        physics.D_n_left_m2_s / geometry.left_distance_m * state[0],
        physics.D_p_left_m2_s / geometry.left_distance_m * state[1],
        physics.D_n_right_m2_s / geometry.right_distance_m * state[2],
        physics.D_p_right_m2_s / geometry.right_distance_m * state[3],
    )
    scale = max(
        float(np.max(balance.one_way_cross_scale_m2_s)),
        bulk_scale,
        1.0,
    )

    assert np.max(np.abs(balance.bulk_flux_m2_s)) / bulk_scale < 2.0e-13
    assert np.max(np.abs(balance.residual_m2_s)) / scale < 2.0e-12


def test_carrier_log_jacobian_matches_central_finite_difference():
    geometry = _geometry(
        potential_jump_right_minus_left_V=0.01,
        fixed_sheet_charge_C_m2=1.0e-5,
    )
    physics = _physics(
        conduction_band_step_eV=0.12,
        hole_transport_step_eV=-0.07,
        surface_recombination_velocity_n_m_s=0.03,
        surface_recombination_velocity_p_m_s=0.05,
        n1_left_m3=2.0e17,
        n1_right_m3=5.0e17,
        p1_left_m3=8.0e16,
        p1_right_m3=3.0e17,
    )
    bulk = _bulk()
    log_state = np.log(np.array([3.0e20, 7.0e19, 5.0e19, 4.0e20]))
    analytic = carrier_balance_and_jacobian(
        log_state, geometry, physics, bulk
    ).jacobian_log_state_m2_s
    step = 1.0e-6
    finite_difference = np.empty_like(analytic)
    for column in range(4):
        plus = log_state.copy()
        minus = log_state.copy()
        plus[column] += step
        minus[column] -= step
        finite_difference[:, column] = (
            carrier_balance_and_jacobian(
                plus, geometry, physics, bulk
            ).residual_m2_s
            - carrier_balance_and_jacobian(
                minus, geometry, physics, bulk
            ).residual_m2_s
        ) / (2.0 * step)

    jacobian_scale = max(float(np.max(np.abs(analytic))), 1.0)
    np.testing.assert_allclose(
        analytic / jacobian_scale,
        finite_difference / jacobian_scale,
        rtol=3.0e-5,
        atol=2.0e-10,
    )


def test_exact_left_and_right_distances_control_only_their_half_fluxes():
    bulk = _bulk(phi_left_V=0.0, phi_right_V=0.0)
    physics = _physics(
        conduction_band_step_eV=0.0,
        hole_transport_step_eV=0.0,
    )
    state = np.array([1.5e20, 6.0e19, 8.0e19, 2.0e20])
    first = carrier_balance_and_jacobian(
        np.log(state), _geometry(), physics, bulk
    )
    second = carrier_balance_and_jacobian(
        np.log(state), _geometry(left_distance_m=6.0e-9), physics, bulk
    )

    np.testing.assert_allclose(
        second.bulk_flux_m2_s[:2],
        0.5 * first.bulk_flux_m2_s[:2],
        rtol=1.0e-13,
    )
    np.testing.assert_allclose(
        second.bulk_flux_m2_s[2:],
        first.bulk_flux_m2_s[2:],
        rtol=1.0e-13,
    )


def test_local_qss_certifies_conservation_with_shared_trap_capture():
    geometry = _geometry(
        left_distance_m=5.0e-9,
        right_distance_m=5.0e-9,
        eps_r_left=20.0,
        eps_r_right=20.0,
    )
    physics = _physics(
        D_n_left_m2_s=1.0e-4,
        D_n_right_m2_s=1.0e-4,
        D_p_left_m2_s=1.0e-4,
        D_p_right_m2_s=1.0e-4,
        N_C_right_m3=1.0e25,
        N_V_left_m3=2.0e25,
        N_V_right_m3=2.0e25,
        conduction_band_step_eV=0.0,
        hole_transport_step_eV=0.0,
        surface_recombination_velocity_n_m_s=0.02,
        surface_recombination_velocity_p_m_s=0.03,
        n1_left_m3=1.0e16,
        n1_right_m3=1.0e16,
        p1_left_m3=1.0e16,
        p1_right_m3=1.0e16,
    )
    bulk = TwoSidedBulkState(
        phi_left_V=0.0,
        phi_right_V=0.0,
        n_left_m3=1.0e22,
        p_left_m3=2.0e18,
        n_right_m3=3.0e18,
        p_right_m3=8.0e21,
    )
    result = solve_two_sided_interface(geometry, physics, bulk)

    assert result.converged
    assert result.normalized_residual <= 1.0e-9
    assert result.cross_flux_m2_s[[0, 2]].sum() == pytest.approx(0.0, abs=1.0)
    assert result.cross_flux_m2_s[[1, 3]].sum() == pytest.approx(0.0, abs=1.0)
    assert result.capture_flux_m2_s[[0, 2]].sum() == pytest.approx(
        result.capture_flux_m2_s[[1, 3]].sum(),
        rel=2.0e-12,
        abs=1.0,
    )
    np.testing.assert_allclose(
        result.bulk_flux_m2_s + result.cross_flux_m2_s,
        result.capture_flux_m2_s,
        rtol=0.0,
        atol=max(1.0, float(np.max(np.abs(result.bulk_flux_m2_s))) * 2.0e-9),
    )

    occupancy = shared_trap_occupancy(result.state_m3, physics)
    assert 0.0 <= occupancy <= 1.0


def test_shared_trap_occupancy_is_invariant_to_common_velocity_scaling():
    state = np.array([3.0e20, 7.0e19, 5.0e19, 4.0e20])
    physics = _physics(
        surface_recombination_velocity_n_m_s=0.03,
        surface_recombination_velocity_p_m_s=0.05,
        n1_left_m3=2.0e17,
        n1_right_m3=5.0e17,
        p1_left_m3=8.0e16,
        p1_right_m3=3.0e17,
    )
    scaled = replace(
        physics,
        surface_recombination_velocity_n_m_s=3.0e4,
        surface_recombination_velocity_p_m_s=5.0e4,
    )

    assert shared_trap_occupancy(state, scaled) == pytest.approx(
        shared_trap_occupancy(state, physics),
        rel=2.0e-16,
    )


def test_shared_trap_occupancy_log_jacobian_matches_central_difference():
    state = np.array([3.0e20, 7.0e19, 5.0e19, 4.0e20])
    physics = _physics(
        surface_recombination_velocity_n_m_s=0.03,
        surface_recombination_velocity_p_m_s=0.05,
        n1_left_m3=2.0e17,
        n1_right_m3=5.0e17,
        p1_left_m3=8.0e16,
        p1_right_m3=3.0e17,
    )
    occupancy, analytic = shared_trap_occupancy_and_log_jacobian(state, physics)
    log_state = np.log(state)
    step = 1.0e-6
    finite_difference = np.empty(4)
    for column in range(4):
        plus = log_state.copy()
        minus = log_state.copy()
        plus[column] += step
        minus[column] -= step
        finite_difference[column] = (
            shared_trap_occupancy(np.exp(plus), physics)
            - shared_trap_occupancy(np.exp(minus), physics)
        ) / (2.0 * step)

    assert 0.0 <= occupancy <= 1.0
    np.testing.assert_allclose(analytic, finite_difference, rtol=2.0e-7, atol=1e-12)


def test_equilibrium_referenced_gauss_tangent_matches_central_difference():
    geometry = _geometry(
        potential_jump_right_minus_left_V=0.011,
        fixed_sheet_charge_C_m2=3.0e-6,
    )
    physics = _physics(
        surface_recombination_velocity_n_m_s=0.03,
        surface_recombination_velocity_p_m_s=0.05,
        n1_left_m3=2.0e17,
        n1_right_m3=5.0e17,
        p1_left_m3=8.0e16,
        p1_right_m3=3.0e17,
    )
    state = np.array([3.0e20, 7.0e19, 5.0e19, 4.0e20])
    reference = shared_trap_occupancy(
        state * np.array([0.8, 1.1, 1.2, 0.9]), physics
    )
    charge = EquilibriumReferencedSheetCharge(reference, 2.5e16)
    coordinates = np.concatenate(([0.013, 0.024], np.log(state)))
    balance = equilibrium_referenced_electrostatic_trace_balance(
        coordinates, geometry, physics, _bulk(), charge
    )
    finite_difference = np.empty((2, 6))
    for column in range(6):
        step = 1.0e-7 if column < 2 else 1.0e-6
        plus = coordinates.copy()
        minus = coordinates.copy()
        plus[column] += step
        minus[column] -= step
        finite_difference[:, column] = (
            equilibrium_referenced_electrostatic_trace_balance(
                plus, geometry, physics, _bulk(), charge
            ).residual
            - equilibrium_referenced_electrostatic_trace_balance(
                minus, geometry, physics, _bulk(), charge
            ).residual
        ) / (2.0 * step)

    row_scale = np.maximum(
        np.max(np.abs(balance.jacobian_trace_and_log_state), axis=1),
        1.0e-30,
    )
    np.testing.assert_allclose(
        balance.jacobian_trace_and_log_state / row_scale[:, None],
        finite_difference / row_scale[:, None],
        rtol=2.0e-7,
        atol=2.0e-10,
    )


def test_equilibrium_reference_is_exactly_charge_off_and_sign_is_physical():
    geometry = _geometry()
    physics = _physics(
        surface_recombination_velocity_n_m_s=0.03,
        surface_recombination_velocity_p_m_s=0.05,
        n1_left_m3=2.0e17,
        n1_right_m3=5.0e17,
        p1_left_m3=8.0e16,
        p1_right_m3=3.0e17,
    )
    state = np.array([3.0e20, 7.0e19, 5.0e19, 4.0e20])
    traces = solve_electrostatic_traces(geometry, _bulk())
    coordinates = np.concatenate(
        ([traces.phi_left_V, traces.phi_right_V], np.log(state))
    )
    occupancy = shared_trap_occupancy(np.exp(coordinates[2:]), physics)
    off_residual = electrostatic_trace_residual_and_jacobian(
        coordinates[:2], geometry, _bulk()
    )[0]
    reference = equilibrium_referenced_electrostatic_trace_balance(
        coordinates,
        geometry,
        physics,
        _bulk(),
        EquilibriumReferencedSheetCharge(occupancy, 2.5e16),
    )

    np.testing.assert_array_equal(reference.residual, off_residual)
    assert reference.incremental_sheet_charge_C_m2 == 0.0

    more_electrons = coordinates.copy()
    more_electrons[[2, 4]] += 0.2
    charged = equilibrium_referenced_electrostatic_trace_balance(
        more_electrons,
        geometry,
        physics,
        _bulk(),
        EquilibriumReferencedSheetCharge(occupancy, 2.5e16),
    )
    assert charged.occupancy > occupancy
    assert charged.incremental_sheet_charge_C_m2 < 0.0
    assert abs(charged.incremental_sheet_charge_C_m2) <= Q * 2.5e16


@pytest.mark.parametrize(
    ("log_state_shift", "expected_sign"),
    [
        (np.array([0.3, 0.0, 0.3, 0.0]), -1.0),
        (np.array([0.0, 0.3, 0.0, 0.3]), 1.0),
    ],
)
def test_charged_gauss_law_closes_across_dielectric_jump(
    log_state_shift,
    expected_sign,
):
    geometry = _geometry(
        eps_r_left=8.0,
        eps_r_right=35.0,
        potential_jump_right_minus_left_V=0.017,
        fixed_sheet_charge_C_m2=4.0e-6,
    )
    physics = _physics(
        surface_recombination_velocity_n_m_s=0.03,
        surface_recombination_velocity_p_m_s=0.05,
        n1_left_m3=2.0e17,
        n1_right_m3=5.0e17,
        p1_left_m3=8.0e16,
        p1_right_m3=3.0e17,
    )
    bulk = _bulk()
    reference_log_state = np.log(np.array([3.0e20, 7.0e19, 5.0e19, 4.0e20]))
    reference = shared_trap_occupancy(np.exp(reference_log_state), physics)
    charge = EquilibriumReferencedSheetCharge(reference, 6.0e16)
    log_state = reference_log_state + log_state_shift
    probe = equilibrium_referenced_electrostatic_trace_balance(
        np.concatenate(([0.0, 0.017], log_state)),
        geometry,
        physics,
        bulk,
        charge,
    )
    effective_geometry = replace(
        geometry,
        fixed_sheet_charge_C_m2=(
            geometry.fixed_sheet_charge_C_m2
            + probe.incremental_sheet_charge_C_m2
        ),
    )
    traces = solve_electrostatic_traces(effective_geometry, bulk)
    balance = equilibrium_referenced_electrostatic_trace_balance(
        np.concatenate(([traces.phi_left_V, traces.phi_right_V], log_state)),
        geometry,
        physics,
        bulk,
        charge,
    )
    capacitance_left = EPS_0 * geometry.eps_r_left / geometry.left_distance_m
    capacitance_right = EPS_0 * geometry.eps_r_right / geometry.right_distance_m
    total_sheet_charge = (
        geometry.fixed_sheet_charge_C_m2
        + balance.incremental_sheet_charge_C_m2
    )
    displacement_jump = capacitance_left * (
        traces.phi_left_V - bulk.phi_left_V
    ) + capacitance_right * (traces.phi_right_V - bulk.phi_right_V)
    scale = np.array(
        [physics.thermal_voltage_V, max(abs(total_sheet_charge), 1.0e-30)]
    )

    assert np.sign(balance.incremental_sheet_charge_C_m2) == expected_sign
    assert displacement_jump == pytest.approx(total_sheet_charge, abs=2.0e-18)
    assert np.max(np.abs(balance.residual / scale)) < 1.0e-10


def test_coupled_charged_local_and_bulk_jacobians_match_central_difference():
    geometry = _geometry(
        potential_jump_right_minus_left_V=0.01,
        fixed_sheet_charge_C_m2=2.0e-6,
    )
    physics = _physics(
        conduction_band_step_eV=0.12,
        hole_transport_step_eV=-0.07,
        surface_recombination_velocity_n_m_s=0.03,
        surface_recombination_velocity_p_m_s=0.05,
        n1_left_m3=2.0e17,
        n1_right_m3=5.0e17,
        p1_left_m3=8.0e16,
        p1_right_m3=3.0e17,
    )
    bulk = _bulk()
    state = np.array([3.0e20, 7.0e19, 5.0e19, 4.0e20])
    charge = EquilibriumReferencedSheetCharge(
        shared_trap_occupancy(state * np.array([0.9, 1.1, 1.2, 0.8]), physics),
        3.0e16,
    )
    local = np.concatenate(([0.013, 0.023], np.log(state)))
    balance = equilibrium_referenced_two_sided_balance(
        local, geometry, physics, bulk, charge
    )

    local_fd = np.empty((6, 6))
    for column in range(6):
        step = 1.0e-7 if column < 2 else 1.0e-6
        plus = local.copy()
        minus = local.copy()
        plus[column] += step
        minus[column] -= step
        local_fd[:, column] = (
            equilibrium_referenced_two_sided_balance(
                plus, geometry, physics, bulk, charge
            ).residual
            - equilibrium_referenced_two_sided_balance(
                minus, geometry, physics, bulk, charge
            ).residual
        ) / (2.0 * step)

    bulk_coordinates = _bulk_coordinates(bulk)
    bulk_fd = np.empty((6, 6))
    for column in range(6):
        step = 1.0e-7 if column < 2 else 1.0e-6
        plus = bulk_coordinates.copy()
        minus = bulk_coordinates.copy()
        plus[column] += step
        minus[column] -= step
        bulk_fd[:, column] = (
            equilibrium_referenced_two_sided_balance(
                local, geometry, physics, _bulk_from_coordinates(plus), charge
            ).residual
            - equilibrium_referenced_two_sided_balance(
                local, geometry, physics, _bulk_from_coordinates(minus), charge
            ).residual
        ) / (2.0 * step)

    row_scale = np.maximum(
        np.max(
            np.abs(
                np.concatenate(
                    (balance.jacobian_local, balance.jacobian_bulk), axis=1
                )
            ),
            axis=1,
        ),
        1.0e-30,
    )
    np.testing.assert_allclose(
        balance.jacobian_local / row_scale[:, None],
        local_fd / row_scale[:, None],
        rtol=3.0e-5,
        atol=3.0e-10,
    )
    np.testing.assert_allclose(
        balance.jacobian_bulk / row_scale[:, None],
        bulk_fd / row_scale[:, None],
        rtol=3.0e-5,
        atol=3.0e-10,
    )


def test_charged_local_ift_sheet_charge_tangent_matches_resolved_bulk_fd():
    geometry = _geometry(
        eps_r_left=9.0,
        eps_r_right=31.0,
        potential_jump_right_minus_left_V=0.008,
    )
    physics = _physics(
        conduction_band_step_eV=0.11,
        hole_transport_step_eV=-0.06,
        surface_recombination_velocity_n_m_s=0.03,
        surface_recombination_velocity_p_m_s=0.05,
        n1_left_m3=2.0e17,
        n1_right_m3=5.0e17,
        p1_left_m3=8.0e16,
        p1_right_m3=3.0e17,
    )
    reference_bulk = _bulk()
    reference_off = solve_two_sided_interface(geometry, physics, reference_bulk)
    reference_occupancy = shared_trap_occupancy(
        np.exp(np.log(reference_off.state_m3)), physics
    )
    charge = EquilibriumReferencedSheetCharge(reference_occupancy, 2.0e16)
    biased_bulk = _bulk(
        phi_left_V=-0.01,
        phi_right_V=0.08,
        n_left_m3=2.4e20,
        p_right_m3=3.6e20,
    )
    result = solve_equilibrium_referenced_two_sided_interface(
        geometry,
        physics,
        biased_bulk,
        charge,
        residual_tolerance=1.0e-9,
    )

    bulk_coordinates = _bulk_coordinates(biased_bulk)
    finite_difference = np.empty(6)
    for column in range(6):
        step = 2.0e-6 if column < 2 else 2.0e-5
        plus = bulk_coordinates.copy()
        minus = bulk_coordinates.copy()
        plus[column] += step
        minus[column] -= step
        plus_result = solve_equilibrium_referenced_two_sided_interface(
            geometry,
            physics,
            _bulk_from_coordinates(plus),
            charge,
            initial_state_m3=result.balance.carrier.state_m3,
            residual_tolerance=1.0e-9,
        )
        minus_result = solve_equilibrium_referenced_two_sided_interface(
            geometry,
            physics,
            _bulk_from_coordinates(minus),
            charge,
            initial_state_m3=result.balance.carrier.state_m3,
            residual_tolerance=1.0e-9,
        )
        finite_difference[column] = (
            plus_result.balance.electrostatic.incremental_sheet_charge_C_m2
            - minus_result.balance.electrostatic.incremental_sheet_charge_C_m2
        ) / (2.0 * step)

    scale = max(
        float(np.max(np.abs(result.sheet_charge_jacobian_bulk))),
        float(np.max(np.abs(finite_difference))),
        1.0e-30,
    )
    assert result.converged
    assert result.normalized_electrostatic_residual <= 1.0e-9
    assert result.normalized_carrier_residual <= 1.0e-9
    assert abs(result.balance.electrostatic.incremental_sheet_charge_C_m2) <= (
        Q * charge.trap_density_m2
    )
    np.testing.assert_allclose(
        result.sheet_charge_jacobian_bulk / scale,
        finite_difference / scale,
        rtol=1.0e-4,
        atol=2.0e-7,
    )


def test_shared_trap_occupancy_rejects_undefined_or_negative_state():
    with pytest.raises(ValueError, match="undefined"):
        shared_trap_occupancy(np.ones(4), _physics())
    with pytest.raises(ValueError, match="non-negative"):
        shared_trap_occupancy(
            np.array([1.0, -1.0, 1.0, 1.0]),
            _physics(surface_recombination_velocity_n_m_s=1.0),
        )


def test_homojunction_mirror_reverses_cross_flux_and_trace_state():
    geometry = _geometry(
        left_distance_m=5.0e-9,
        right_distance_m=5.0e-9,
        eps_r_left=20.0,
        eps_r_right=20.0,
    )
    physics = _physics(
        D_n_left_m2_s=1.0e-4,
        D_n_right_m2_s=1.0e-4,
        D_p_left_m2_s=1.0e-4,
        D_p_right_m2_s=1.0e-4,
        N_C_right_m3=1.0e25,
        N_V_left_m3=2.0e25,
        N_V_right_m3=2.0e25,
    )
    forward_bulk = TwoSidedBulkState(
        phi_left_V=0.0,
        phi_right_V=0.0,
        n_left_m3=4.0e20,
        p_left_m3=2.0e20,
        n_right_m3=1.0e20,
        p_right_m3=6.0e20,
    )
    reverse_bulk = TwoSidedBulkState(
        phi_left_V=0.0,
        phi_right_V=0.0,
        n_left_m3=forward_bulk.n_right_m3,
        p_left_m3=forward_bulk.p_right_m3,
        n_right_m3=forward_bulk.n_left_m3,
        p_right_m3=forward_bulk.p_left_m3,
    )
    forward = solve_two_sided_interface(geometry, physics, forward_bulk)
    reverse = solve_two_sided_interface(geometry, physics, reverse_bulk)

    np.testing.assert_allclose(
        reverse.state_m3,
        forward.state_m3[[2, 3, 0, 1]],
        rtol=2.0e-10,
    )
    assert reverse.cross_flux_m2_s[2] == pytest.approx(
        -forward.cross_flux_m2_s[2], rel=2.0e-10
    )
    assert reverse.cross_flux_m2_s[3] == pytest.approx(
        -forward.cross_flux_m2_s[3], rel=2.0e-10
    )


def test_invalid_zero_trace_distance_fails_closed():
    with pytest.raises(ValueError, match="left_distance_m"):
        solve_two_sided_interface(
            _geometry(left_distance_m=0.0),
            _physics(),
            _bulk(),
        )
