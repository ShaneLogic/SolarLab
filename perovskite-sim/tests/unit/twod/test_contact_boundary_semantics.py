"""The same declared carrier reservoir must survive dimensional extrusion."""

from dataclasses import replace

import numpy as np
import pytest

from perovskite_sim.constants import Q
from perovskite_sim.discretization.grid import Layer
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.solver.mol import build_material_arrays
from perovskite_sim.twod.grid_2d import build_grid_2d
from perovskite_sim.twod.microstructure import Microstructure, lateral_dual_cell_widths
from perovskite_sim.twod.solver_2d import (
    assemble_rhs_2d,
    build_material_arrays_2d,
    extract_snapshot_2d,
    run_transient_2d,
)


_CONTACTS = (
    ("S_n_left", "S_n_top", "S_n_L", 0, 0),
    ("S_p_left", "S_p_top", "S_p_L", 1, 0),
    ("S_n_right", "S_n_bot", "S_n_R", 0, -1),
    ("S_p_right", "S_p_bot", "S_p_R", 1, -1),
)


def _problem(**options):
    base = load_device_from_yaml("tests/fixtures/configs/nip_MAPbI3.yaml")
    stack = replace(base, **options)
    grid = build_grid_2d(
        [Layer(layer.thickness, 4) for layer in electrical_layers(stack)],
        300e-9,
        3,
        lateral_uniform=True,
    )
    one = build_material_arrays(grid.y, stack)
    two = build_material_arrays_2d(grid, stack, Microstructure(), lateral_bc="neumann")
    return stack, grid, one, two


@pytest.mark.parametrize("mode", ["legacy", "fast", "full"])
@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("field", [item[0] for item in _CONTACTS])
@pytest.mark.parametrize("velocity", [None, 0.0, 100.0])
def test_every_resolved_contact_matches_one_dimension(mode, flat, field, velocity):
    _, _, one, two = _problem(mode=mode, flat_band_contacts=flat, **{field: velocity})
    assert two.has_selective_contacts == one.has_selective_contacts
    for _, two_name, one_name, _, _ in _CONTACTS:
        assert getattr(two, two_name) == getattr(one, one_name)


def _state(material):
    n = np.maximum(material.ni, 1e14).copy()
    p = n.copy()
    n[0], n[-1] = material.n_eq_left, material.n_eq_right
    p[0], p[-1] = material.p_eq_left, material.p_eq_right
    n[1:-1] *= 1.2
    p[1:-1] *= 0.8
    return np.concatenate([n.ravel(), p.ravel()])


@pytest.mark.parametrize("field,two_name,one_name,block,row", _CONTACTS)
def test_only_the_selected_contact_evolves(field, two_name, one_name, block, row):
    _, grid, _, material = _problem(mode="full", **{field: 0.0})
    state = _state(material)
    rate = assemble_rhs_2d(0.0, state, material, V_app=0.1).reshape(2, grid.Ny, grid.Nx)
    for current_field, _, _, current_block, current_row in _CONTACTS:
        if current_field != field:
            np.testing.assert_array_equal(rate[current_block, current_row], 0.0)
    assert np.any(rate[block, row] != 0.0)
    assert getattr(material, two_name) == 0.0


def test_rhs_and_snapshot_use_fixed_reservoir_values_without_mutating_input():
    _, grid, _, material = _problem(mode="full", S_n_left=0.0, S_p_right=0.0)
    state = _state(material)
    altered = state.copy().reshape(2, grid.Ny, grid.Nx)
    altered[0, -1] *= 3.0
    altered[1, 0] *= 2.0
    altered = altered.ravel()
    before = altered.copy()
    expected = assemble_rhs_2d(0.0, state, material, V_app=0.1)
    actual = assemble_rhs_2d(0.0, altered, material, V_app=0.1)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(altered, before)
    snapshot = extract_snapshot_2d(altered, material, V_app=0.1)
    np.testing.assert_array_equal(snapshot.n[-1], material.n_eq_right)
    np.testing.assert_array_equal(snapshot.p[0], material.p_eq_left)
    assert np.any(snapshot.Jy_p[0] != 0.0)
    assert np.any(snapshot.Jy_n[-1] != 0.0)


def test_all_blocking_contacts_conserve_carriers_apart_from_generation_and_recombination():
    from perovskite_sim.twod.solver_2d import recombination_rate_2d

    _, grid, _, material = _problem(mode="full", **{item[0]: 0.0 for item in _CONTACTS})
    state = _state(material)
    n, p = state.reshape(2, grid.Ny, grid.Nx)
    rate = assemble_rhs_2d(0.0, state, material, V_app=0.1).reshape(2, grid.Ny, grid.Nx)
    source = material.G_optical - recombination_rate_2d(n, p, material)
    wx = lateral_dual_cell_widths(grid.x)
    wy = lateral_dual_cell_widths(grid.y)
    weights = wy[:, None] * wx[None, :]
    for component in rate:
        residual = Q * np.sum((component - source) * weights)
        scale = Q * np.sum((np.abs(component) + np.abs(source)) * weights)
        assert abs(residual) < 1e-12 * scale


@pytest.mark.parametrize("mode", ["legacy", "fast", "full"])
def test_execution_protocol_includes_explicit_flat_band_contact_kinetics(mode):
    from perovskite_sim.twod.experiments.jv_sweep_2d import (
        build_jv_2d_execution_protocol,
    )

    stack, _, _, _ = _problem(mode=mode, flat_band_contacts=True)
    protocol = build_jv_2d_execution_protocol(
        stack,
        lateral_length=300e-9,
        Nx=3,
        Ny_per_layer=4,
        V_max=0.1,
        V_step=0.05,
        lateral_bc="neumann",
    )
    assert protocol.carrier_boundary_condition == "selective_robin"


def test_reversing_contact_orientation_preserves_the_signed_poisson_boundary():
    base, _, _, _ = _problem(mode="full", S_n_left=0.0, S_p_right=0.0)
    stack = replace(
        base,
        layers=tuple(reversed(base.layers)),
        S_n_left=None,
        S_p_left=0.0,
        S_n_right=0.0,
        S_p_right=None,
        built_in_potential_mode="metal_work_function",
        work_function_left_eV=4.0,
        work_function_right_eV=5.0,
    )
    grid = build_grid_2d(
        [Layer(layer.thickness, 4) for layer in electrical_layers(stack)],
        300e-9,
        3,
        lateral_uniform=True,
    )
    one = build_material_arrays(grid.y, stack)
    material = build_material_arrays_2d(
        grid, stack, Microstructure(), lateral_bc="neumann"
    )
    assert material.junction_polarity == one.junction_polarity == -1.0
    snapshot = extract_snapshot_2d(_state(material), material, V_app=0.1)
    np.testing.assert_array_equal(snapshot.phi[0], 0.0)
    np.testing.assert_allclose(snapshot.phi[-1], one.V_bi_bc + 0.1, atol=1e-14)
    assert material.S_p_top == material.S_n_bot == 0.0
    assert material.S_n_top is material.S_p_bot is None


def _homogeneous_stack():
    base, _, _, _ = _problem(mode="full")
    layer = base.layers[1]
    params = replace(
        layer.params,
        D_ion=0.0,
        P0=0.0,
        N_A=0.0,
        N_D=0.0,
        ni=1e16,
        Nc300=None,
        Nv300=None,
        chi=0.0,
        Eg=0.0,
        mu_n=1e-4,
        mu_p=1e-4,
        B_rad=0.0,
        C_n=0.0,
        C_p=0.0,
        tau_n=1.0,
        tau_p=1.0,
        n1=1e16,
        p1=1e16,
        alpha=0.0,
    )
    return replace(
        base,
        layers=(replace(layer, params=params, thickness=1e-6),),
        interfaces=(),
        interface_defects=(),
        Phi=0.0,
        V_bi=0.0,
        built_in_potential_mode="legacy_manual",
    )


def test_increasing_exchange_velocity_approaches_fixed_reservoirs():
    stack = _homogeneous_stack()
    grid = build_grid_2d([Layer(1e-6, 8)], 3e-7, 2, lateral_uniform=True)
    pinned = build_material_arrays_2d(
        grid, stack, Microstructure(), lateral_bc="neumann"
    )
    density = 1e16 * (1.0 + 0.1 * np.sin(np.pi * grid.y / grid.y[-1]))
    state = np.tile(np.broadcast_to(density[:, None], (grid.Ny, grid.Nx)).ravel(), 2)
    reference = run_transient_2d(
        state, pinned, V_app=0.0, t_end=1e-7, rtol=1e-8, atol=1e3
    )
    differences = []
    for velocity in (10.0, 100.0, 1000.0):
        robin = replace(stack, **{item[0]: velocity for item in _CONTACTS})
        material = build_material_arrays_2d(
            grid, robin, Microstructure(), lateral_bc="neumann"
        )
        result = run_transient_2d(
            state, material, V_app=0.0, t_end=1e-7, rtol=1e-8, atol=1e3
        )
        differences.append(np.max(np.abs(result - reference)) / 1e16)
    assert differences[2] < differences[1] < differences[0]
    assert differences[2] < 1e-3


@pytest.mark.slow
def test_mixed_contact_transient_approaches_the_same_one_dimensional_limit():
    from perovskite_sim.solver.mol import StateVec, run_transient

    stack = replace(_homogeneous_stack(), S_n_left=0.0, S_p_right=0.0)
    errors = []
    for intervals in (8, 16, 32):
        grid = build_grid_2d(
            [Layer(1e-6, intervals)],
            3e-7,
            2,
            alpha_y=2.0,
            lateral_uniform=True,
        )
        one = build_material_arrays(grid.y, stack)
        two = build_material_arrays_2d(
            grid, stack, Microstructure(), lateral_bc="neumann"
        )
        density = 1e16 * (1.0 + 0.1 * np.sin(np.pi * grid.y / grid.y[-1]))
        initial = StateVec.pack(density, density, one.P_ion0)
        solved = run_transient(
            grid.y,
            initial,
            (0.0, 1e-7),
            np.array([1e-7]),
            stack,
            illuminated=False,
            V_app=0.01,
            rtol=1e-8,
            atol=1e3,
            max_step=1e-8,
            mat=one,
        )
        assert solved.success
        extruded = np.tile(
            np.broadcast_to(density[:, None], (grid.Ny, grid.Nx)).ravel(),
            2,
        )
        result = run_transient_2d(
            extruded,
            two,
            V_app=0.01,
            t_end=1e-7,
            rtol=1e-8,
            atol=1e3,
            max_step=1e-8,
        ).reshape(2, grid.Ny, grid.Nx)
        expected = solved.y[: 2 * grid.Ny, -1].reshape(2, grid.Ny)
        errors.append(float(np.max(np.abs(result[:, :, 0] - expected)) / 1e16))
        np.testing.assert_array_equal(result[0, -1], two.n_eq_right)
        np.testing.assert_array_equal(result[1, 0], two.p_eq_left)
        assert np.max(np.abs(result[0, 0] - density[0])) > 1e12
        np.testing.assert_allclose(
            result, np.broadcast_to(result[:, :, :1], result.shape), rtol=1e-9
        )
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 5e-3
