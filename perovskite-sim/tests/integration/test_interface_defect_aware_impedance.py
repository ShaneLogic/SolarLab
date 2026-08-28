"""Device-level closure for dynamic two-sided interface defects."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from perovskite_sim.constants import Q
from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.experiments.interface_defect_aware_impedance import (
    INTERFACE_DEFECT_DEVICE_AC_SCOPE,
    InterfaceDefectDeviceACCertificationError,
    InterfaceDefectDeviceACError,
    run_interface_defect_device_impedance,
)
from perovskite_sim.experiments.quasi_fermi_steady_state import (
    build_equilibrium_referenced_interface_charge_dark_reference,
    build_two_sided_trace_grid,
    solve_equilibrium_referenced_interface_charge_steady_state,
)
from perovskite_sim.models.device import DeviceStack, InterfaceDefect, LayerSpec
from perovskite_sim.models.interface_defects import InterfaceDefectDocument
from perovskite_sim.models.parameters import MaterialParams


def _research_interface_stack() -> DeviceStack:
    left = MaterialParams(
        eps_r=10.0,
        mu_n=1.0e-3,
        mu_p=1.0e-3,
        D_ion=0.0,
        P_lim=1.0e24,
        P0=0.0,
        ni=1.0e12,
        tau_n=1.0e-6,
        tau_p=1.0e-6,
        n1=1.0e12,
        p1=1.0e12,
        B_rad=0.0,
        C_n=0.0,
        C_p=0.0,
        alpha=0.0,
        N_A=0.0,
        N_D=0.0,
        chi=4.0,
        Eg=1.5,
        Nc300=1.0e25,
        Nv300=1.0e25,
    )
    right = replace(left, chi=4.1)
    document = InterfaceDefectDocument.from_scaps_cgs(
        sigma_n_cm2=6.0e-18,
        sigma_p_cm2=1.0e-17,
        thermal_velocity_cm_s=1.0e7,
        total_density_cm2=5.0e10,
        trap_depth_eV_below_cb=0.55,
    )
    return DeviceStack(
        layers=(
            LayerSpec("left", 1.0e-7, left, role="absorber"),
            LayerSpec("right", 1.0e-7, right, role="ETL"),
        ),
        interfaces=(document.capture_velocities_m_s,),
        interface_defects=(
            InterfaceDefect(
                E_t_eV=0.55,
                N_t_cm2=5.0e10,
                microscopic_document=document,
            ),
        ),
        interface_charge_closure="equilibrium_referenced",
        interface_charge_rebaseline_acknowledged=True,
        V_bi=0.0,
        Phi=0.0,
        mode="full",
    )


def _grid(stack: DeviceStack) -> np.ndarray:
    shared = multilayer_grid(
        [Layer(layer.thickness, 4) for layer in stack.layers],
        alpha=tuple(2.0 for _layer in stack.layers),
    )
    return build_two_sided_trace_grid(shared, stack)


def _research_two_interface_stack() -> DeviceStack:
    base = _research_interface_stack()
    left, right = base.layers
    middle = LayerSpec(
        "middle",
        1.0e-7,
        replace(left.params, chi=4.05),
        role="absorber",
    )
    document = base.interface_defects[0].microscopic_document
    defect = InterfaceDefect(
        E_t_eV=0.55,
        N_t_cm2=5.0e10,
        microscopic_document=document,
    )
    return replace(
        base,
        layers=(left, middle, right),
        interfaces=(
            document.capture_velocities_m_s,
            document.capture_velocities_m_s,
        ),
        interface_defects=(defect, defect),
    )


def test_real_dynamic_interface_device_ac_certifies_four_capture_legs():
    stack = _research_interface_stack()
    grid = _grid(stack)
    frequencies = np.logspace(-8.0, 14.0, 45)
    result = run_interface_defect_device_impedance(
        grid,
        stack,
        frequencies,
        illuminated=False,
    )

    assert result.scope == INTERFACE_DEFECT_DEVICE_AC_SCOPE
    assert result.certificate.certified
    assert result.certificate.frequency_window.certified
    assert result.admittance_faces_S_m2.shape == (frequencies.size, grid.size - 1)
    assert result.interface_trace_state_response_m3_V.shape == (
        frequencies.size,
        1,
        4,
    )
    assert result.electron_capture_response_m2_s_V.shape == (
        frequencies.size,
        1,
        2,
    )
    assert result.hole_capture_response_m2_s_V.shape == (
        frequencies.size,
        1,
        2,
    )
    np.testing.assert_allclose(
        result.interface_sheet_charge_storage_response_F_m2,
        -Q * result.interface_occupied_population_response_m2_V,
        rtol=0.0,
        atol=0.0,
    )
    assert np.max(np.abs(result.interface_occupancy_response_per_V)) > 0.0
    assert result.certificate.maximum_local_trap_balance_relative_error < 1.0e-4
    assert result.certificate.maximum_all_face_admittance_spread < 5.0e-4


def test_two_physical_interfaces_have_two_independent_shared_occupancies():
    stack = _research_two_interface_stack()
    grid = _grid(stack)
    frequencies = np.logspace(-8.0, 14.0, 45)
    result = run_interface_defect_device_impedance(grid, stack, frequencies)

    assert result.certificate.certified
    assert result.interface_occupancy_response_per_V.shape == (frequencies.size, 2)
    assert result.interface_trace_state_response_m3_V.shape == (
        frequencies.size,
        2,
        4,
    )
    assert result.electron_capture_response_m2_s_V.shape == (
        frequencies.size,
        2,
        2,
    )
    assert result.hole_capture_response_m2_s_V.shape == (
        frequencies.size,
        2,
        2,
    )
    assert result.certificate.maximum_local_trap_balance_relative_error < 1.0e-4


def test_interface_device_ac_rejects_tampered_dark_reference_before_linearization():
    stack = _research_interface_stack()
    grid = _grid(stack)
    reference = build_equilibrium_referenced_interface_charge_dark_reference(
        grid, stack
    )
    tampered = replace(reference, grid_sha256="0" * 64)

    with pytest.raises(InterfaceDefectDeviceACError, match="provenance"):
        run_interface_defect_device_impedance(
            grid,
            stack,
            np.logspace(-2.0, 2.0, 3),
            dark_reference=tampered,
        )


def test_interface_device_ac_recertifies_supplied_dc_state_on_live_operator():
    stack = _research_interface_stack()
    grid = _grid(stack)
    reference = build_equilibrium_referenced_interface_charge_dark_reference(
        grid, stack
    )
    dc_state = solve_equilibrium_referenced_interface_charge_steady_state(
        grid,
        stack,
        0.0,
        dark_reference=reference,
        illuminated=False,
    )
    tampered_increment = np.asarray(
        dc_state.electron_quasi_fermi_increment_V,
        dtype=float,
    ).copy()
    tampered_increment[1] += 1.0e-3
    tampered = replace(
        dc_state,
        electron_quasi_fermi_increment_V=tampered_increment,
    )

    with pytest.raises(InterfaceDefectDeviceACError, match="DC state is not certified"):
        run_interface_defect_device_impedance(
            grid,
            stack,
            np.logspace(-2.0, 2.0, 3),
            dark_reference=reference,
            dc_state=tampered,
        )


def test_interface_device_ac_frequency_window_is_fail_closed():
    stack = _research_interface_stack()
    grid = _grid(stack)
    frequencies = np.logspace(1.0, 2.0, 3)
    partial = run_interface_defect_device_impedance(
        grid,
        stack,
        frequencies,
        require_certificate=False,
    )

    assert not partial.certificate.certified
    assert not partial.certificate.frequency_window.certified
    assert "trap_frequency_window_incomplete" in partial.certificate.reasons
    with pytest.raises(InterfaceDefectDeviceACCertificationError) as exc_info:
        run_interface_defect_device_impedance(
            grid,
            stack,
            frequencies,
        )
    assert not exc_info.value.result.certificate.certified
