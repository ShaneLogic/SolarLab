from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.solver.dae import (
    DAECapabilityError,
    build_consistent_initial_condition,
    build_no_ion_no_interface_dae,
    finite_difference_derivative_jacobian,
    finite_difference_state_jacobian,
    project_algebraic_state,
)
from perovskite_sim.solver.mol import StateVec, assemble_rhs
from perovskite_sim.solver.newton import solve_equilibrium


@pytest.fixture
def homogeneous_problem():
    source = load_device_from_yaml("configs/csi_vannijen2025_pn_cv.yaml")
    layer = source.layers[1]
    stack = replace(
        source,
        layers=(layer,),
        V_bi=0.0,
        built_in_potential_mode="legacy_manual",
        interfaces=(),
        interface_defects=(),
        grid_interval_weights=(),
        grid_alphas=(),
    )
    grid = multilayer_grid([Layer(layer.thickness, 8)], alpha=1.0)
    reference = solve_equilibrium(grid, stack)
    model = build_no_ion_no_interface_dae(grid, stack, reference)
    return grid, stack, reference, model


def test_layout_classifies_differential_and_algebraic_rows(homogeneous_problem):
    grid, _stack, reference, model = homogeneous_problem
    layout = model.layout

    assert layout.size == 3 * grid.size
    assert np.count_nonzero(layout.differential_mask) == 2 * (grid.size - 2)
    assert np.count_nonzero(layout.algebraic_mask) == grid.size + 4
    assert not np.any(layout.differential_mask & layout.algebraic_mask)
    assert np.all(layout.differential_mask | layout.algebraic_mask)
    assert not layout.differential_mask.flags.writeable
    assert not layout.algebraic_mask.flags.writeable
    np.testing.assert_array_equal(
        StateVec.unpack(reference, grid.size).P,
        np.zeros(grid.size),
    )


def test_consistent_initial_condition_is_reproducible_and_certified(
    homogeneous_problem,
):
    _grid, _stack, _reference, model = homogeneous_problem

    first = build_consistent_initial_condition(model)
    second = build_consistent_initial_condition(model)

    assert first.certified
    assert first.report.max_normalized_algebraic_residual < 1.0e-13
    assert first.report.max_normalized_differential_residual < 1.0e-13
    assert first.state_sha256 == second.state_sha256
    assert len(first.state_sha256) == 64
    np.testing.assert_array_equal(first.coordinate, second.coordinate)
    np.testing.assert_array_equal(first.derivative, second.derivative)
    np.testing.assert_array_equal(first.physical_state, second.physical_state)
    np.testing.assert_array_equal(first.potential_V, second.potential_V)
    for value in (
        first.coordinate,
        first.derivative,
        first.physical_state,
        first.potential_V,
        first.report.normalized_residual,
    ):
        assert not value.flags.writeable


def test_projection_matches_existing_eliminated_poisson_rhs(homogeneous_problem):
    _grid, _stack, _reference, model = homogeneous_problem
    count = model.layout.node_count
    coordinate = np.zeros(model.layout.size)
    coordinate[1 : count - 1] = np.linspace(-0.15, 0.2, count - 2)
    coordinate[count + 1 : 2 * count - 1] = np.linspace(0.12, -0.1, count - 2)
    coordinate[model.layout.potential_slice] = 10.0

    projected = project_algebraic_state(model, coordinate)
    packed = model.packed_physical_state(projected)
    _n, _p, phi = model.physical_fields(projected)
    frozen = assemble_rhs(
        0.0,
        packed,
        model.grid_m,
        model.stack,
        model.material,
        illuminated=False,
        V_app=model.V_app_V,
        phi_frozen=phi,
    )
    eliminated = assemble_rhs(
        0.0,
        packed,
        model.grid_m,
        model.stack,
        model.material,
        illuminated=False,
        V_app=model.V_app_V,
    )
    report = model.residual_report(projected, np.zeros(model.layout.size))

    np.testing.assert_allclose(frozen, eliminated, rtol=2.0e-14, atol=0.0)
    assert report.max_normalized_algebraic_residual < 1.0e-13
    np.testing.assert_array_equal(
        projected[[0, count - 1, count, 2 * count - 1]],
        np.zeros(4),
    )


def test_exact_algebraic_state_jacobian_matches_central_difference(
    homogeneous_problem,
):
    _grid, _stack, _reference, model = homogeneous_problem
    initial = build_consistent_initial_condition(model)
    coordinate = np.array(initial.coordinate, copy=True)
    count = model.layout.node_count
    coordinate[1 : count - 1] += np.linspace(-0.08, 0.1, count - 2)
    coordinate[count + 1 : 2 * count - 1] += np.linspace(0.06, -0.04, count - 2)
    coordinate = project_algebraic_state(model, coordinate)

    exact = model.algebraic_state_jacobian(coordinate)
    central = finite_difference_state_jacobian(
        model,
        coordinate,
        initial.derivative,
    )

    np.testing.assert_allclose(
        exact[model.layout.algebraic_mask],
        central[model.layout.algebraic_mask],
        rtol=2.0e-7,
        atol=2.0e-9,
    )
    np.testing.assert_array_equal(
        exact[model.layout.differential_mask],
        np.zeros((2 * (count - 2), model.layout.size)),
    )


def test_exact_derivative_jacobian_matches_central_difference(
    homogeneous_problem,
):
    _grid, _stack, _reference, model = homogeneous_problem
    initial = build_consistent_initial_condition(model)

    exact = model.derivative_jacobian(initial.coordinate)
    central = finite_difference_derivative_jacobian(
        model,
        initial.coordinate,
        initial.derivative,
    )

    np.testing.assert_allclose(exact, central, rtol=2.0e-10, atol=2.0e-14)
    np.testing.assert_array_equal(
        exact[model.layout.algebraic_mask],
        np.zeros((model.layout.node_count + 4, model.layout.size)),
    )


def test_report_separates_differential_and_algebraic_failures(
    homogeneous_problem,
):
    _grid, _stack, _reference, model = homogeneous_problem
    initial = build_consistent_initial_condition(model)
    derivative = np.array(initial.derivative, copy=True)
    derivative[1] += 2.0e6
    differential = model.residual_report(initial.coordinate, derivative)

    coordinate = np.array(initial.coordinate, copy=True)
    coordinate[model.layout.potential_slice.start] += 0.01
    algebraic = model.residual_report(coordinate, initial.derivative)

    assert differential.max_normalized_differential_residual > 1.0
    assert differential.max_normalized_algebraic_residual < 1.0e-13
    assert algebraic.max_normalized_algebraic_residual > 0.1


def test_capability_rejects_physical_interfaces():
    stack = load_device_from_yaml("configs/csi_vannijen2025_pn_cv.yaml")
    grid = multilayer_grid(
        [Layer(layer.thickness, 4) for layer in stack.layers],
        alpha=1.0,
    )
    reference = solve_equilibrium(grid, stack)

    with pytest.raises(DAECapabilityError, match="physical interfaces"):
        build_no_ion_no_interface_dae(grid, stack, reference)


def test_capability_rejects_mobile_ions():
    source = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    layer = source.layers[1]
    stack = replace(
        source,
        layers=(layer,),
        V_bi=0.0,
        built_in_potential_mode="legacy_manual",
        interfaces=(),
        interface_defects=(),
        grid_interval_weights=(),
        grid_alphas=(),
    )
    grid = multilayer_grid([Layer(layer.thickness, 4)], alpha=1.0)
    reference = solve_equilibrium(grid, stack)

    with pytest.raises(DAECapabilityError, match="mobile ions"):
        build_no_ion_no_interface_dae(grid, stack, reference)


def test_capability_rejects_selective_contacts(homogeneous_problem):
    grid, stack, _reference, _model = homogeneous_problem
    selective = replace(stack, S_n_left=0.0)
    reference = solve_equilibrium(grid, selective)

    with pytest.raises(DAECapabilityError, match="selective contacts"):
        build_no_ion_no_interface_dae(grid, selective, reference)


def test_capability_rejects_nonzero_structural_ion_block(homogeneous_problem):
    grid, stack, reference, _model = homogeneous_problem
    invalid = reference.copy()
    invalid[2 * grid.size + 1] = 1.0

    with pytest.raises(DAECapabilityError, match="structural ion block"):
        build_no_ion_no_interface_dae(grid, stack, invalid)


def test_builder_and_coordinate_validation_fail_closed(homogeneous_problem):
    grid, stack, reference, model = homogeneous_problem
    nonpositive = reference.copy()
    nonpositive[1] = 0.0

    with pytest.raises(ValueError, match="strictly positive"):
        build_no_ion_no_interface_dae(grid, stack, nonpositive)
    with pytest.raises(ValueError, match="strictly increasing"):
        build_no_ion_no_interface_dae(grid[::-1], stack, reference)
    with pytest.raises(ValueError, match="reference_time_s"):
        build_no_ion_no_interface_dae(
            grid,
            stack,
            reference,
            reference_time_s=0.0,
        )
    overflow = np.zeros(model.layout.size)
    overflow[1] = 1.0e4
    with pytest.raises(ValueError, match="overflowed"):
        model.physical_fields(overflow)
    with pytest.raises(ValueError, match="finite layout-sized"):
        project_algebraic_state(model, np.zeros(model.layout.size - 1))


@pytest.mark.parametrize("value", [0.0, -1.0, np.inf, np.nan])
def test_consistency_tolerance_must_be_positive(homogeneous_problem, value):
    _grid, _stack, _reference, model = homogeneous_problem
    with pytest.raises(ValueError, match="residual_tolerance"):
        build_consistent_initial_condition(model, residual_tolerance=value)
