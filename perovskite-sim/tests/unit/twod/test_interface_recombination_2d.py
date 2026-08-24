from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from perovskite_sim._compat.numpy_compat import trapezoid
from perovskite_sim.discretization.grid import Layer
from perovskite_sim.models.device import DeviceStack, InterfaceDefect, LayerSpec
from perovskite_sim.models.parameters import MaterialParams
from perovskite_sim.solver.mol import (
    _apply_interface_recombination,
    build_material_arrays,
)
from perovskite_sim.twod.grid_2d import build_grid_2d
from perovskite_sim.twod.interface_recombination_2d import (
    build_two_sided_interface_srh_couplings_2d,
    evaluate_two_sided_interface_srh_2d,
)
from perovskite_sim.twod.ion_migration_2d import control_volume_areas_2d
from perovskite_sim.twod.microstructure import Microstructure
from perovskite_sim.twod.solver_2d import (
    assemble_rhs_2d,
    build_material_arrays_2d,
    compute_mobile_ion_current_components_2d,
)


def _material(*, mobile_ions: bool = False) -> MaterialParams:
    return MaterialParams(
        eps_r=10.0,
        mu_n=1.0e-3,
        mu_p=1.0e-3,
        D_ion=1.0e-16 if mobile_ions else 0.0,
        P_lim=1.0e24,
        P0=1.0e22 if mobile_ions else 0.0,
        ni=1.0e12,
        tau_n=1.0e30,
        tau_p=1.0e30,
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


def _stack(
    *,
    two_sided: bool = True,
    defect: bool = True,
    projection: bool = False,
    mobile_ions: bool = False,
) -> DeviceStack:
    material = _material(mobile_ions=mobile_ions)
    return DeviceStack(
        layers=(
            LayerSpec("left", 1.0e-7, material, role="absorber"),
            LayerSpec("right", 1.0e-7, material, role="ETL"),
        ),
        interfaces=((0.03, 0.05),),
        interface_defects=(InterfaceDefect(E_t_eV=0.5),) if defect else (),
        interface_two_sided=two_sided,
        interface_plane_projection=projection,
        V_bi=0.0,
        Phi=0.0,
        mode="full",
    )


def _grid(stack: DeviceStack, *, intervals: int = 4):
    return build_grid_2d(
        [Layer(layer.thickness, intervals) for layer in stack.layers],
        lateral_length=1.0e-7,
        Nx=3,
        alpha_y=1.0,
        lateral_uniform=True,
    )


def _enabled_material(*, mobile_ions: bool = False):
    stack = _stack(mobile_ions=mobile_ions)
    grid = _grid(stack)
    material = build_material_arrays_2d(
        grid,
        stack,
        Microstructure(),
        lateral_bc="neumann",
        ion_dynamics="single_mobile" if mobile_ions else "frozen",
        interface_srh="two_sided_cross_node",
    )
    return stack, grid, material


def _high_injection(grid):
    n_y = np.linspace(2.0e20, 5.0e20, grid.Ny)
    p_y = np.linspace(4.0e20, 3.0e20, grid.Ny)
    n = np.broadcast_to(n_y[:, None], (grid.Ny, grid.Nx)).copy()
    p = np.broadcast_to(p_y[:, None], (grid.Ny, grid.Nx)).copy()
    return n_y, p_y, n, p


def test_builder_resolves_exact_cross_node_two_sided_sheet():
    _stack_value, _grid_value, material = _enabled_material()
    coupling = material.interface_srh_couplings[0]

    assert len(material.interface_srh_couplings) == 1
    assert coupling.left_sample_row == coupling.interface_row - 1
    assert coupling.right_sample_row == coupling.interface_row + 1
    assert coupling.electron_capture_velocity_m_s == pytest.approx(0.03)
    assert coupling.hole_capture_velocity_m_s == pytest.approx(0.05)


def test_lateral_uniform_surface_sink_matches_1d_production_path_exactly():
    stack, grid, material = _enabled_material()
    n_y, p_y, n, p = _high_injection(grid)
    report = evaluate_two_sided_interface_srh_2d(
        n,
        p,
        grid.y,
        material.interface_srh_couplings,
    )
    material_1d = build_material_arrays(grid.y, stack)
    dn_1d = np.zeros(grid.Ny)
    dp_1d = np.zeros(grid.Ny)
    _apply_interface_recombination(
        dn_1d,
        dp_1d,
        n_y,
        p_y,
        stack,
        material_1d,
        None,
    )

    np.testing.assert_array_equal(
        -report.volumetric_sink_m3_s[:, 0],
        dn_1d,
    )
    np.testing.assert_array_equal(dn_1d, dp_1d)


def test_surface_to_volume_mapping_is_area_conservative_on_nonuniform_grid():
    _stack_value, grid, material = _enabled_material()
    _n_y, _p_y, n, p = _high_injection(grid)
    n *= np.linspace(0.8, 1.2, grid.Nx)[None, :]
    p *= np.linspace(1.1, 0.9, grid.Nx)[None, :]
    report = evaluate_two_sided_interface_srh_2d(
        n,
        p,
        grid.y,
        material.interface_srh_couplings,
    )
    areas = control_volume_areas_2d(grid.x, grid.y)
    volume_integral = float(np.sum(report.volumetric_sink_m3_s * areas))
    surface_integral = sum(
        float(trapezoid(row, grid.x))
        for row in report.total_surface_rate_m2_s
    )

    assert volume_integral == pytest.approx(surface_integral, rel=2.0e-15)


def test_high_injection_slice_is_clamp_inactive_and_read_only():
    _stack_value, grid, material = _enabled_material()
    _n_y, _p_y, n, p = _high_injection(grid)
    report = evaluate_two_sided_interface_srh_2d(
        n,
        p,
        grid.y,
        material.interface_srh_couplings,
    )

    assert np.all(report.pair_a_surface_rate_m2_s > 0.0)
    assert np.all(report.pair_b_surface_rate_m2_s > 0.0)
    assert not np.any(report.pair_a_clamped)
    assert not np.any(report.pair_b_clamped)
    assert not report.total_surface_rate_m2_s.flags.writeable
    assert not report.volumetric_sink_m3_s.flags.writeable


def test_generation_side_of_each_channel_is_independently_clamped():
    _stack_value, grid, material = _enabled_material()
    n = np.zeros((grid.Ny, grid.Nx))
    p = np.zeros_like(n)
    report = evaluate_two_sided_interface_srh_2d(
        n,
        p,
        grid.y,
        material.interface_srh_couplings,
    )

    np.testing.assert_array_equal(report.pair_a_surface_rate_m2_s, 0.0)
    np.testing.assert_array_equal(report.pair_b_surface_rate_m2_s, 0.0)
    assert np.all(report.pair_a_clamped)
    assert np.all(report.pair_b_clamped)


def test_evaluator_rejects_malformed_direct_coupling():
    _stack_value, grid, material = _enabled_material()
    _n_y, _p_y, n, p = _high_injection(grid)
    malformed = replace(
        material.interface_srh_couplings[0],
        left_sample_row=-1,
    )

    with pytest.raises(ValueError, match=r"idx-1/idx\+1"):
        evaluate_two_sided_interface_srh_2d(
            n,
            p,
            grid.y,
            (malformed,),
        )


def test_default_and_explicit_off_rhs_are_bit_identical():
    stack = _stack()
    grid = _grid(stack)
    default = build_material_arrays_2d(
        grid,
        stack,
        Microstructure(),
        lateral_bc="neumann",
    )
    explicit = build_material_arrays_2d(
        grid,
        stack,
        Microstructure(),
        lateral_bc="neumann",
        interface_srh="off",
    )
    _n_y, _p_y, n, p = _high_injection(grid)
    state = np.concatenate([n.ravel(), p.ravel()])

    assert default.interface_srh_couplings == ()
    assert explicit.interface_srh_couplings == ()
    np.testing.assert_array_equal(
        assemble_rhs_2d(0.0, state, default, V_app=0.0),
        assemble_rhs_2d(0.0, state, explicit, V_app=0.0),
    )


def test_enabled_rhs_subtracts_equal_electron_and_hole_sheet_sink():
    stack, grid, enabled = _enabled_material()
    disabled = build_material_arrays_2d(
        grid,
        stack,
        Microstructure(),
        lateral_bc="neumann",
    )
    _n_y, _p_y, n, p = _high_injection(grid)
    state = np.concatenate([n.ravel(), p.ravel()])
    report = evaluate_two_sided_interface_srh_2d(
        n,
        p,
        grid.y,
        enabled.interface_srh_couplings,
    )
    rhs_off = assemble_rhs_2d(0.0, state, disabled, V_app=0.0)
    rhs_on = assemble_rhs_2d(0.0, state, enabled, V_app=0.0)
    block = grid.n_nodes
    difference_n = (rhs_off[:block] - rhs_on[:block]).reshape(n.shape)
    difference_p = (rhs_off[block:] - rhs_on[block:]).reshape(n.shape)

    np.testing.assert_allclose(
        difference_n,
        report.volumetric_sink_m3_s,
        rtol=1.0e-11,
        atol=0.0,
    )
    np.testing.assert_allclose(
        difference_n,
        difference_p,
        rtol=1.0e-14,
        atol=0.0,
    )


def test_interface_sheet_composes_with_conservative_mobile_ion_block():
    _stack_value, grid, material = _enabled_material(mobile_ions=True)
    _n_y, _p_y, n, p = _high_injection(grid)
    state = np.concatenate(
        [n.ravel(), p.ravel(), material.P_ion0_2d.ravel()]
    )
    derivative = assemble_rhs_2d(0.0, state, material, V_app=0.02)
    dP = derivative[2 * grid.n_nodes :].reshape(n.shape)
    weighted = dP * control_volume_areas_2d(grid.x, grid.y)
    cancellation_scale = max(float(np.sum(np.abs(weighted))), 1.0)

    assert derivative.shape == state.shape
    assert np.all(np.isfinite(derivative))
    assert abs(float(np.sum(weighted))) / cancellation_scale < 5.0e-14


def test_interface_sheet_and_mobile_ion_have_complete_maxwell_current():
    _stack_value, grid, material = _enabled_material(mobile_ions=True)
    _n_y, _p_y, n, p = _high_injection(grid)
    n[[0, -1], :] = np.stack([material.n_eq_left, material.n_eq_right])
    p[[0, -1], :] = np.stack([material.p_eq_left, material.p_eq_right])
    state = np.concatenate(
        [n.ravel(), p.ravel(), material.P_ion0_2d.ravel()]
    )

    report = compute_mobile_ion_current_components_2d(
        state,
        material,
        V_app=0.02,
    )

    assert report.max_relative_face_spread < 5.0e-12


@pytest.mark.parametrize(
    ("stack", "message"),
    [
        (_stack(two_sided=False), "interface_two_sided"),
        (_stack(defect=False), "InterfaceDefect"),
        (_stack(projection=True), "interface-plane projection"),
    ],
)
def test_builder_rejects_uncertified_interface_modes(stack, message):
    grid = _grid(stack)
    material_1d = build_material_arrays(grid.y, stack)
    with pytest.raises(ValueError, match=message):
        build_two_sided_interface_srh_couplings_2d(stack, material_1d)


def test_public_builder_rejects_unknown_mode_and_periodic_area_topology():
    stack = _stack()
    grid = _grid(stack)
    with pytest.raises(ValueError, match="interface_srh must"):
        build_material_arrays_2d(
            grid,
            stack,
            Microstructure(),
            interface_srh="unknown",
        )
    with pytest.raises(ValueError, match="periodic-x sheet area"):
        build_material_arrays_2d(
            grid,
            stack,
            Microstructure(),
            lateral_bc="periodic",
            interface_srh="two_sided_cross_node",
        )
