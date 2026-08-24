from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from perovskite_sim.constants import EPS_0, Q
from perovskite_sim.discretization.grid import Layer
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.experiments.jv_sweep import (
    compute_current_components,
    extract_spatial_snapshot,
)
from perovskite_sim.solver.mol import StateVec, build_material_arrays
from perovskite_sim.twod.grid_2d import build_grid_2d
from perovskite_sim.twod.continuity_2d import apply_thermionic_caps_y
from perovskite_sim.twod.field_mobility_2d import recompute_d_eff_2d
from perovskite_sim.twod.flux_2d import sg_fluxes_2d_n, sg_fluxes_2d_p
from perovskite_sim.twod.ion_migration_2d import positive_ion_fluxes_2d
from perovskite_sim.twod.microstructure import Microstructure
from perovskite_sim.twod.mobile_ion_current_2d import (
    evaluate_mobile_ion_current_components_2d,
)
from perovskite_sim.twod.solver_2d import (
    assemble_rhs_2d,
    build_material_arrays_2d,
    compute_mobile_ion_current_components_2d,
    extract_snapshot_2d,
)


def _problem():
    base = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    layers = list(base.layers)
    absorber = layers[1]
    layers[1] = replace(
        absorber,
        params=replace(absorber.params, D_ion=1.0e-12),
    )
    stack = replace(base, layers=tuple(layers))
    grid = build_grid_2d(
        [Layer(layer.thickness, 3) for layer in electrical_layers(stack)],
        lateral_length=120.0e-9,
        Nx=3,
        alpha_y=1.0,
        lateral_uniform=True,
    )
    material = build_material_arrays_2d(
        grid,
        stack,
        Microstructure(),
        lateral_bc="neumann",
        ion_dynamics="single_mobile",
    )
    yy, xx = np.meshgrid(
        grid.y / grid.y[-1],
        grid.x / grid.x[-1],
        indexing="ij",
    )
    n = np.maximum(material.ni, 1.0e15) * (1.0 + 0.05 * xx + 0.03 * yy)
    p = np.maximum(material.ni, 1.0e15) * (1.0 + 0.02 * xx + 0.04 * yy)
    n[0, :] = material.n_eq_left
    n[-1, :] = material.n_eq_right
    p[0, :] = material.p_eq_left
    p[-1, :] = material.p_eq_right
    P = material.P_ion0_2d * (1.0 + 0.04 * yy - 0.02 * xx)
    state = np.concatenate([n.ravel(), p.ravel(), P.ravel()])
    return stack, grid, material, state


def test_complete_current_uses_same_positive_ion_flux_source():
    _stack_value, _grid_value, material, state = _problem()
    snapshot = extract_snapshot_2d(state, material, V_app=0.03)
    derivative = assemble_rhs_2d(0.0, state, material, V_app=0.03)
    report = evaluate_mobile_ion_current_components_2d(
        snapshot,
        derivative,
        material,
    )
    expected = Q * positive_ion_fluxes_2d(
        snapshot.x,
        snapshot.y,
        snapshot.phi,
        snapshot.P_ion,
        material.D_ion_2d,
        material.V_T,
        material.P_lim_2d,
        steric_diffusion_only=material.ion_steric_diffusion_only,
    ).y

    np.testing.assert_array_equal(report.positive_ion_y_A_m2, expected)


def test_lateral_uniform_conduction_and_ion_channels_match_1d_source():
    stack, grid, material, _state = _problem()
    n_y = np.maximum(material.ni[:, 0], 1.0e16) * np.linspace(1.0, 1.3, grid.Ny)
    p_y = np.maximum(material.ni[:, 0], 1.0e16) * np.linspace(1.2, 0.9, grid.Ny)
    P_y = material.P_ion0_2d[:, 0] * np.linspace(0.99, 1.01, grid.Ny)
    n_y[[0, -1]] = [material.n_eq_left[0], material.n_eq_right[0]]
    p_y[[0, -1]] = [material.p_eq_left[0], material.p_eq_right[0]]
    state_2d = np.concatenate(
        [
            np.broadcast_to(n_y[:, None], (grid.Ny, grid.Nx)).ravel(),
            np.broadcast_to(p_y[:, None], (grid.Ny, grid.Nx)).ravel(),
            np.broadcast_to(P_y[:, None], (grid.Ny, grid.Nx)).ravel(),
        ]
    )
    report = compute_mobile_ion_current_components_2d(
        state_2d,
        material,
        0.02,
    )
    material_1d = build_material_arrays(grid.y, stack)
    state_1d = StateVec.pack(n_y, p_y, P_y)
    expected = compute_current_components(
        grid.y,
        state_1d,
        stack,
        0.02,
        mat=material_1d,
    )
    to_solar = -float(material.junction_polarity)

    np.testing.assert_allclose(
        to_solar * report.lateral_average_electron_A_m2,
        expected.J_n,
        rtol=5.0e-13,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        to_solar * report.lateral_average_hole_A_m2,
        expected.J_p,
        rtol=5.0e-13,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        to_solar * report.lateral_average_positive_ion_A_m2,
        expected.J_ion,
        rtol=5.0e-13,
        atol=1.0e-12,
    )
    derivative_2d = assemble_rhs_2d(0.0, state_2d, material, V_app=0.02)
    blocks = derivative_2d.reshape(3, grid.Ny, grid.Nx)
    derivative_1d = np.concatenate([block[:, 0] for block in blocks])
    step = 1.0e-12
    snapshot_1d = extract_spatial_snapshot(
        grid.y,
        state_1d,
        stack,
        0.02,
        mat=material_1d,
    )
    perturbed_1d = extract_spatial_snapshot(
        grid.y,
        state_1d + step * derivative_1d,
        stack,
        0.02,
        mat=material_1d,
    )
    eps_face = 2.0 * material_1d.eps_r[:-1] * material_1d.eps_r[1:] / (
        material_1d.eps_r[:-1] + material_1d.eps_r[1:]
    )
    displacement_1d = EPS_0 * eps_face * (
        perturbed_1d.E - snapshot_1d.E
    ) / step
    np.testing.assert_allclose(
        to_solar * report.lateral_average_displacement_A_m2,
        to_solar * displacement_1d,
        rtol=2.0e-7,
        atol=2.0e-5,
    )


def test_displacement_is_exact_poisson_directional_derivative():
    _stack_value, _grid_value, material, state = _problem()
    voltage = 0.03
    voltage_rate = 2.5
    snapshot = extract_snapshot_2d(state, material, V_app=voltage)
    derivative = assemble_rhs_2d(0.0, state, material, V_app=voltage)
    report = evaluate_mobile_ion_current_components_2d(
        snapshot,
        derivative,
        material,
        applied_voltage_rate_V_s=voltage_rate,
    )
    step = 1.0e-12
    perturbed = extract_snapshot_2d(
        state + step * derivative,
        material,
        V_app=voltage + step * voltage_rate,
    )
    dy = np.diff(snapshot.y)[:, None]
    field = -(snapshot.phi[1:] - snapshot.phi[:-1]) / dy
    field_perturbed = -(perturbed.phi[1:] - perturbed.phi[:-1]) / dy
    eps_face = 2.0 * material.eps_r[:-1] * material.eps_r[1:] / (
        material.eps_r[:-1] + material.eps_r[1:]
    )
    expected = EPS_0 * eps_face * (field_perturbed - field) / step

    np.testing.assert_allclose(
        report.displacement_y_A_m2,
        expected,
        rtol=2.0e-7,
        atol=2.0e-5,
    )


def test_snapshot_current_reuses_field_mobility_face_coefficients():
    stack, grid, _material, state = _problem()
    mobile_layers = tuple(
        replace(
            layer,
            params=replace(layer.params, v_sat_n=1.0e2, v_sat_p=2.0e2),
        )
        for layer in stack.layers
    )
    material = build_material_arrays_2d(
        grid,
        replace(stack, layers=mobile_layers),
        Microstructure(),
        lateral_bc="neumann",
        ion_dynamics="single_mobile",
    )
    snapshot = extract_snapshot_2d(state, material, V_app=0.03)
    effective = recompute_d_eff_2d(
        phi=snapshot.phi,
        x=grid.x,
        y=grid.y,
        D_n=material.D_n,
        D_p=material.D_p,
        V_T=material.V_T,
        v_sat_n_x_face=material.v_sat_n_x_face,
        v_sat_n_y_face=material.v_sat_n_y_face,
        ct_beta_n_x_face=material.ct_beta_n_x_face,
        ct_beta_n_y_face=material.ct_beta_n_y_face,
        pf_gamma_n_x_face=material.pf_gamma_n_x_face,
        pf_gamma_n_y_face=material.pf_gamma_n_y_face,
        v_sat_p_x_face=material.v_sat_p_x_face,
        v_sat_p_y_face=material.v_sat_p_y_face,
        ct_beta_p_x_face=material.ct_beta_p_x_face,
        ct_beta_p_y_face=material.ct_beta_p_y_face,
        pf_gamma_p_x_face=material.pf_gamma_p_x_face,
        pf_gamma_p_y_face=material.pf_gamma_p_y_face,
        lateral_bc="neumann",
        v_sat_n_wrap=material.v_sat_n_wrap,
        v_sat_p_wrap=material.v_sat_p_wrap,
        ct_beta_n_wrap=material.ct_beta_n_wrap,
        ct_beta_p_wrap=material.ct_beta_p_wrap,
        pf_gamma_n_wrap=material.pf_gamma_n_wrap,
        pf_gamma_p_wrap=material.pf_gamma_p_wrap,
    )
    phi_n = snapshot.phi + material.chi
    phi_p = snapshot.phi + material.chi + material.Eg
    _jx_n, expected_n = sg_fluxes_2d_n(
        phi_n,
        snapshot.n,
        grid.x,
        grid.y,
        material.D_n,
        material.V_T,
        D_n_x_face=effective.D_n_x,
        D_n_y_face=effective.D_n_y,
    )
    _jx_p, expected_p = sg_fluxes_2d_p(
        phi_p,
        snapshot.p,
        grid.x,
        grid.y,
        material.D_p,
        material.V_T,
        D_p_x_face=effective.D_p_x,
        D_p_y_face=effective.D_p_y,
    )
    expected_n, expected_p = apply_thermionic_caps_y(
        expected_n,
        expected_p,
        snapshot.n,
        snapshot.p,
        material.chi,
        material.Eg,
        material.V_T,
        interface_y_faces=material.interface_y_faces,
        A_star_n=material.A_star_n,
        A_star_p=material.A_star_p,
        T=material.T_device,
    )
    _raw_x, raw_n = sg_fluxes_2d_n(
        phi_n,
        snapshot.n,
        grid.x,
        grid.y,
        material.D_n,
        material.V_T,
    )

    np.testing.assert_array_equal(snapshot.Jy_n, expected_n)
    np.testing.assert_array_equal(snapshot.Jy_p, expected_p)
    assert not np.array_equal(snapshot.Jy_n, raw_n)


def test_thermionic_cap_changes_only_declared_interface_face():
    raw_n = np.full((2, 2), 3.0)
    raw_p = np.full((2, 2), -4.0)
    n = np.array([[2.0, 3.0], [1.0, 1.5], [1.0, 1.0]])
    p = np.array([[1.0, 1.5], [2.0, 3.0], [1.0, 1.0]])
    chi = np.array([[3.8, 3.8], [4.2, 4.2], [4.2, 4.2]])
    gap = np.array([[2.0, 2.0], [1.5, 1.5], [1.5, 1.5]])
    richardson = np.full((3, 2), 1.0e-20)

    capped_n, capped_p = apply_thermionic_caps_y(
        raw_n,
        raw_p,
        n,
        p,
        chi,
        gap,
        0.025,
        interface_y_faces=(0,),
        A_star_n=richardson,
        A_star_p=richardson,
        T=300.0,
    )

    assert not np.array_equal(capped_n[0], raw_n[0])
    assert not np.array_equal(capped_p[0], raw_p[0])
    np.testing.assert_array_equal(capped_n[1], raw_n[1])
    np.testing.assert_array_equal(capped_p[1], raw_p[1])


def test_snapshot_current_routes_through_shared_thermionic_cap(monkeypatch):
    _stack_value, _grid_value, material, state = _problem()
    captured: dict[str, np.ndarray] = {}

    def sentinel_cap(Jy_n, Jy_p, *_args, **_kwargs):
        captured["raw_n"] = Jy_n.copy()
        captured["raw_p"] = Jy_p.copy()
        return 0.25 * Jy_n, 0.5 * Jy_p

    monkeypatch.setattr(
        "perovskite_sim.twod.solver_2d.apply_thermionic_caps_y",
        sentinel_cap,
    )
    snapshot = extract_snapshot_2d(state, material, V_app=0.03)

    np.testing.assert_array_equal(snapshot.Jy_n, 0.25 * captured["raw_n"])
    np.testing.assert_array_equal(snapshot.Jy_p, 0.5 * captured["raw_p"])


def test_terminal_total_equals_all_four_components_and_arrays_are_immutable():
    _stack_value, _grid_value, material, state = _problem()
    report = compute_mobile_ion_current_components_2d(
        state,
        material,
        0.03,
    )

    assert report.terminal_total_A_m2 == pytest.approx(
        report.terminal_electron_A_m2
        + report.terminal_hole_A_m2
        + report.terminal_positive_ion_A_m2
        + report.terminal_displacement_A_m2,
        rel=2.0e-15,
    )
    assert not report.total_y_A_m2.flags.writeable
    assert not report.lateral_average_total_A_m2.flags.writeable


def test_lateral_uniform_rhs_has_uniform_maxwell_current_across_y_faces():
    _stack_value, grid, material, _state = _problem()
    n_y = np.maximum(material.ni[:, 0], 1.0e16) * np.linspace(1.0, 1.4, grid.Ny)
    p_y = np.maximum(material.ni[:, 0], 1.0e16) * np.linspace(1.3, 0.9, grid.Ny)
    P_y = material.P_ion0_2d[:, 0] * np.linspace(0.98, 1.02, grid.Ny)
    n_y[[0, -1]] = [material.n_eq_left[0], material.n_eq_right[0]]
    p_y[[0, -1]] = [material.p_eq_left[0], material.p_eq_right[0]]
    state = np.concatenate(
        [
            np.broadcast_to(n_y[:, None], (grid.Ny, grid.Nx)).ravel(),
            np.broadcast_to(p_y[:, None], (grid.Ny, grid.Nx)).ravel(),
            np.broadcast_to(P_y[:, None], (grid.Ny, grid.Nx)).ravel(),
        ]
    )
    report = compute_mobile_ion_current_components_2d(
        state,
        material,
        0.02,
    )

    assert report.max_relative_face_spread < 2.0e-11


def test_complete_current_fails_closed_on_wrong_topology_and_derivative():
    stack, grid, material, state = _problem()
    snapshot = extract_snapshot_2d(state, material, V_app=0.03)
    derivative = assemble_rhs_2d(0.0, state, material, V_app=0.03)
    with pytest.raises(ValueError, match="three 2D blocks"):
        evaluate_mobile_ion_current_components_2d(
            snapshot,
            derivative[:-1],
            material,
        )

    frozen = build_material_arrays_2d(
        grid,
        stack,
        Microstructure(),
        lateral_bc="neumann",
    )
    with pytest.raises(ValueError, match="active 2D ion state"):
        evaluate_mobile_ion_current_components_2d(
            snapshot,
            derivative,
            frozen,
        )

    incompatible_contact = replace(material, has_selective_contacts=True)
    with pytest.raises(ValueError, match="Robin contacts"):
        evaluate_mobile_ion_current_components_2d(
            snapshot,
            derivative,
            incompatible_contact,
        )

    mismatched = replace(snapshot, n=snapshot.n.copy())
    mismatched.n[0, 0] *= 1.01
    with pytest.raises(ValueError, match="ohmic left electron"):
        evaluate_mobile_ion_current_components_2d(
            mismatched,
            derivative,
            material,
        )


def test_existing_carrier_only_terminal_function_still_rejects_mobile_snapshot():
    from perovskite_sim.twod.solver_2d import compute_terminal_current_2d

    _stack_value, _grid_value, material, state = _problem()
    snapshot = extract_snapshot_2d(state, material, V_app=0.03)
    with pytest.raises(ValueError, match="not mobile-ion complete"):
        compute_terminal_current_2d(snapshot)
