"""Contracts for the opt-in physical interface-plane response."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from perovskite_sim.models.device import DeviceStack, LayerSpec
from perovskite_sim.physics.interface_plane import (
    _fermi_dirac_richardson_cross_fluxes,
    _project_density_to_plane,
    _project_density_to_plane_fermi_dirac,
    _reciprocal_fermi_edge_cross_fluxes,
    _scaps_boltzmann_cross_fluxes,
    _scaps_thermal_velocity_cross_fluxes,
    compute_interface_srh_occupancy_on_state,
    solve_interface_states_live_qss,
)
from perovskite_sim.physics.fermi_dirac import fermi_dirac_half


def _two_layer_stack(*, interface=(0.0, 0.0)) -> DeviceStack:
    return DeviceStack(
        layers=(
            LayerSpec("left", 1.0e-7, None, role="absorber"),
            LayerSpec("right", 1.0e-7, None, role="ETL"),
        ),
        interfaces=(interface,),
    )


def _physical_mat(**updates):
    values = dict(
        interface_V_partition_2=(0.5,),
        interface_nodes=(1,),
        interface_chi_step=(-0.2,),
        interface_Eg_step=(0.2,),
        interface_calibration_factor=(1.0,),
        interface_n1_L=(1.0e10,),
        interface_n1_R=(1.0e10,),
        interface_p1_L=(1.0e10,),
        interface_p1_R=(1.0e10,),
        interface_n1=(1.0e10,),
        interface_p1=(1.0e10,),
        iface_state_physical_offsets=True,
        iface_state_partition=False,
        V_bi_eff=0.0,
        V_T_device=0.025852,
        T_device=300.0,
        N_C_physical=np.array([2.0e25, 8.0e25]),
        N_V_physical=np.array([3.0e25, 6.0e25]),
        A_star_n=np.array([2.0e5, 8.0e5]),
        A_star_p=np.array([1.0e5, 4.0e5]),
        v_th_physical=np.array([3.0e4, 1.0e4]),
    )
    values.update(updates)
    return SimpleNamespace(**values)


def test_plane_projection_recovers_boltzmann_limit_and_never_exceeds_dos():
    density = 1.0e18
    density_of_states = 1.0e25
    projected = _project_density_to_plane(
        density,
        density_of_states,
        np.log(0.2),
    )
    assert projected == pytest.approx(0.2 * density, rel=3.0e-7)

    saturated = _project_density_to_plane(1.0e40, density_of_states, 20.0)
    assert 0.0 < saturated <= density_of_states


def test_fermi_dirac_projection_recovers_density_without_dos_cap():
    density_of_states = 1.0e25
    density = 3.0 * density_of_states
    projected = _project_density_to_plane_fermi_dirac(
        density,
        density_of_states,
        0.0,
    )
    shifted = _project_density_to_plane_fermi_dirac(
        density,
        density_of_states,
        -1.0,
    )

    assert projected == pytest.approx(density, rel=2.0e-6)
    assert 0.0 < shifted < projected


def test_cross_flux_obeys_detailed_balance_with_dos_contrast_and_barrier():
    mat = _physical_mat()
    thermal_voltage = mat.V_T_device
    electron_left_activity = 1.0e-3
    electron_right_activity = electron_left_activity * np.exp(
        -0.2 / thermal_voltage
    )
    hole_activity = 2.0e-4
    state = np.array(
        [
            mat.N_C_physical[1] * electron_right_activity,
            mat.N_V_physical[1] * hole_activity,
            mat.N_C_physical[0] * electron_left_activity,
            mat.N_V_physical[0] * hole_activity,
        ]
    )

    electron_flux, hole_flux = _reciprocal_fermi_edge_cross_fluxes(
        mat,
        0,
        state,
        thermal_voltage,
        1.0,
    )

    assert electron_flux == 0.0
    assert hole_flux == 0.0


def test_fermi_dirac_cross_flux_obeys_detailed_balance():
    mat = _physical_mat()
    thermal_voltage = mat.V_T_device
    eta_n_left = -1.0
    eta_n_right = eta_n_left - 0.2 / thermal_voltage
    eta_p = -2.0
    state = np.array(
        [
            mat.N_C_physical[1] * fermi_dirac_half(eta_n_right),
            mat.N_V_physical[1] * fermi_dirac_half(eta_p),
            mat.N_C_physical[0] * fermi_dirac_half(eta_n_left),
            mat.N_V_physical[0] * fermi_dirac_half(eta_p),
        ]
    )

    electron_flux, hole_flux = _fermi_dirac_richardson_cross_fluxes(
        mat,
        0,
        state,
        thermal_voltage,
        1.0,
    )

    supply_scale = 1.0e25
    assert abs(electron_flux) <= 1.0e-12 * supply_scale
    assert abs(hole_flux) <= 1.0e-12 * supply_scale


def test_fermi_dirac_cross_flux_recovers_dilute_scaps_limit():
    mat = _physical_mat(
        interface_chi_step=(0.0,),
        interface_Eg_step=(0.0,),
        N_C_physical=np.array([1.0e25, 1.0e25]),
        N_V_physical=np.array([2.0e25, 2.0e25]),
    )
    state = np.array([2.0e18, 3.0e18, 8.0e18, 9.0e18])

    fermi_dirac = _fermi_dirac_richardson_cross_fluxes(
        mat, 0, state, mat.V_T_device, 1.0
    )
    boltzmann = _scaps_boltzmann_cross_fluxes(
        mat, 0, state, mat.V_T_device, 1.0
    )

    np.testing.assert_allclose(fermi_dirac, boltzmann, rtol=2.0e-6, atol=0.0)


def test_scaps_cross_flux_uses_smaller_velocity_and_detailed_balance():
    mat = _physical_mat()
    thermal_voltage = mat.V_T_device
    electron_left_activity = 1.0e-3
    electron_right_activity = electron_left_activity * np.exp(
        -0.2 / thermal_voltage
    )
    hole_activity = 2.0e-4
    equilibrium = np.array(
        [
            mat.N_C_physical[1] * electron_right_activity,
            mat.N_V_physical[1] * hole_activity,
            mat.N_C_physical[0] * electron_left_activity,
            mat.N_V_physical[0] * hole_activity,
        ]
    )

    electron_flux, hole_flux = _scaps_thermal_velocity_cross_fluxes(
        mat,
        0,
        equilibrium,
        thermal_voltage,
        1.0,
    )
    # The one-way supplies are O(1e26); the residual below is at floating-point
    # round-off relative to those supplies.
    roundoff_scale = 1.0e4 * float(np.max(equilibrium))
    assert abs(electron_flux) <= 1.0e-15 * roundoff_scale
    assert abs(hole_flux) <= 1.0e-15 * roundoff_scale

    driven = equilibrium.copy()
    driven[2] *= 2.0
    full = _scaps_thermal_velocity_cross_fluxes(
        mat, 0, driven, thermal_voltage, 1.0
    )[0]
    half = _scaps_thermal_velocity_cross_fluxes(
        mat, 0, driven, thermal_voltage, 0.5
    )[0]
    assert full > 0.0
    assert half == pytest.approx(0.5 * full)


def test_scaps_boltzmann_law_preserves_balance_without_fermi_capping():
    mat = _physical_mat(
        interface_chi_step=(0.0,),
        interface_Eg_step=(0.0,),
        N_C_physical=np.array([1.0e25, 1.0e25]),
        N_V_physical=np.array([2.0e25, 2.0e25]),
    )
    balanced = np.array([4.0e24, 6.0e24, 4.0e24, 6.0e24])
    electron_flux, hole_flux = _scaps_boltzmann_cross_fluxes(
        mat, 0, balanced, mat.V_T_device, 1.0
    )
    assert electron_flux == 0.0
    assert hole_flux == 0.0

    driven = balanced.copy()
    driven[2] = 9.0e24
    driven[0] = 1.0e24
    boltzmann = _scaps_boltzmann_cross_fluxes(
        mat, 0, driven, mat.V_T_device, 1.0
    )[0]
    bounded = _reciprocal_fermi_edge_cross_fluxes(
        mat, 0, driven, mat.V_T_device, 1.0
    )[0]
    assert boltzmann > bounded > 0.0


def test_shared_trap_occupancy_conserves_electron_and_hole_capture():
    mat = _physical_mat()
    state = np.array([8.0e20, 2.0e17, 3.0e20, 5.0e18])
    sink = compute_interface_srh_occupancy_on_state(
        state,
        _two_layer_stack(interface=(2.0e-2, 3.0e-2)),
        mat,
    )

    assert sink[[0, 2]].sum() == pytest.approx(
        sink[[1, 3]].sum(),
        rel=1.0e-12,
        abs=1.0,
    )


def test_local_qss_state_is_bounded_and_constitutively_certified():
    mat = _physical_mat(
        interface_chi_step=(0.0,),
        interface_Eg_step=(0.0,),
        N_C_physical=np.array([1.0e25, 1.0e25]),
        N_V_physical=np.array([2.0e25, 2.0e25]),
    )
    result = solve_interface_states_live_qss(
        mat,
        _two_layer_stack(),
        n=np.array([1.0e20, 1.0e20]),
        p=np.array([2.0e20, 2.0e20]),
        phi=np.zeros(2),
    )
    capacity = np.array([1.0e25, 2.0e25, 1.0e25, 2.0e25])

    assert result.normalized_residual <= 1.0e-7
    assert result.transport_model == "fermi_richardson"
    assert np.all(result.state_m3 > 0.0)
    assert np.all(result.state_m3 <= capacity)
    np.testing.assert_allclose(
        result.bulk_flux_m2_s + result.cross_flux_m2_s,
        0.0,
        rtol=0.0,
        atol=1.0e-6,
    )


def test_local_qss_supports_scaps_thermionic_transport():
    mat = _physical_mat(
        interface_chi_step=(0.0,),
        interface_Eg_step=(0.0,),
        N_C_physical=np.array([1.0e25, 1.0e25]),
        N_V_physical=np.array([2.0e25, 2.0e25]),
    )
    result = solve_interface_states_live_qss(
        mat,
        _two_layer_stack(),
        n=np.array([1.0e20, 1.0e20]),
        p=np.array([2.0e20, 2.0e20]),
        phi=np.zeros(2),
        interface_transport_model="scaps_thermionic",
    )

    assert result.transport_model == "scaps_thermionic"
    assert result.normalized_residual <= 1.0e-7


def test_local_qss_supports_unbounded_fermi_dirac_transport():
    mat = _physical_mat(
        interface_chi_step=(0.0,),
        interface_Eg_step=(0.0,),
        N_C_physical=np.array([1.0e25, 1.0e25]),
        N_V_physical=np.array([2.0e25, 2.0e25]),
    )
    result = solve_interface_states_live_qss(
        mat,
        _two_layer_stack(),
        n=np.array([3.0e25, 3.0e25]),
        p=np.array([6.0e25, 6.0e25]),
        phi=np.zeros(2),
        interface_transport_model="fermi_dirac_richardson",
    )
    capacity = np.array([1.0e25, 2.0e25, 1.0e25, 2.0e25])

    assert result.transport_model == "fermi_dirac_richardson"
    assert result.normalized_residual <= 1.0e-7
    assert np.all(result.state_m3 > capacity)
