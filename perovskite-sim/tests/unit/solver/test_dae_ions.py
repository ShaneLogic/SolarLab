from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.physics.generation import dual_cell_integral
from perovskite_sim.solver.dae import (
    DAECapabilityError,
    build_no_ion_no_interface_dae,
    project_algebraic_state,
)
from perovskite_sim.solver.dae_ions import (
    build_single_ion_consistent_initial_condition,
    build_single_positive_ion_dae,
    finite_difference_single_ion_derivative_jacobian,
    finite_difference_single_ion_state_jacobian,
    project_single_ion_algebraic_state,
)
from perovskite_sim.solver.mol import StateVec
from perovskite_sim.solver.newton import solve_equilibrium


def _single_ion_problem(*, intervals: int = 6, illuminated: bool = False):
    source = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    layer = source.layers[1]
    stack = replace(
        source,
        layers=(layer,),
        V_bi=0.0,
        built_in_potential_mode="legacy_manual",
        Phi=1.0e17,
        interfaces=(),
        interface_defects=(),
        grid_interval_weights=(),
        grid_alphas=(),
    )
    grid = multilayer_grid([Layer(layer.thickness, intervals)], alpha=1.0)
    reference = solve_equilibrium(grid, stack)
    model = build_single_positive_ion_dae(
        grid,
        stack,
        reference,
        illuminated=illuminated,
        carrier_reference_time_s=1.0e-9,
        ion_reference_time_s=1.0,
    )
    return grid, stack, reference, model


def test_single_ion_layout_classifies_carrier_ion_and_poisson_rows():
    grid, _stack, _reference, model = _single_ion_problem()
    count = grid.size
    layout = model.layout

    assert layout.size == 4 * count
    assert np.count_nonzero(layout.differential_mask) == 3 * count - 4
    assert np.count_nonzero(layout.algebraic_mask) == count + 4
    assert np.all(layout.differential_mask[layout.positive_ion_slice])
    assert np.all(layout.algebraic_mask[layout.potential_slice])
    assert not layout.differential_mask.flags.writeable


def test_reference_coordinate_recovers_strictly_admissible_ion_density():
    _grid, _stack, reference, model = _single_ion_problem()
    n, p, positive_ion, _phi = model.physical_fields(
        np.zeros(model.layout.size)
    )
    state = StateVec.unpack(reference, model.layout.node_count)

    np.testing.assert_allclose(n, state.n, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(p, state.p, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(positive_ion, state.P, rtol=2.0e-15, atol=0.0)
    assert np.all(positive_ion > 0.0)
    assert np.all(positive_ion < model.material.P_lim_node)


def test_logit_coordinate_fails_closed_at_floating_point_saturation():
    _grid, _stack, _reference, model = _single_ion_problem()
    coordinate = np.zeros(model.layout.size)
    coordinate[model.layout.positive_ion_slice] = 1.0e3
    with pytest.raises(ValueError, match="logit coordinate saturated"):
        model.physical_fields(coordinate)


def test_logit_mapping_is_continuous_at_reference_coordinate():
    _grid, _stack, _reference, model = _single_ion_problem()
    reference = model.physical_fields(np.zeros(model.layout.size))[2]
    coordinate = np.zeros(model.layout.size)
    coordinate[model.layout.positive_ion_slice] = np.finfo(float).eps**2

    perturbed = model.physical_fields(coordinate)[2]

    np.testing.assert_array_equal(perturbed, reference)


def test_consistent_initial_condition_closes_all_rows_and_blocking_inventory():
    grid, _stack, _reference, model = _single_ion_problem(illuminated=True)
    initial = build_single_ion_consistent_initial_condition(model)

    assert initial.certified
    assert initial.report.max_normalized_residual < 1.0e-12
    assert initial.report.max_normalized_positive_ion_residual == 0.0
    assert abs(initial.report.positive_ion_inventory_residual_m2_s) == 0.0
    assert abs(initial.report.positive_ion_rhs_inventory_rate_m2_s) < 1.0e-6
    assert len(initial.state_sha256) == 64
    assert not initial.coordinate.flags.writeable
    np.testing.assert_allclose(
        model.residual(initial.coordinate, initial.derivative),
        0.0,
        rtol=0.0,
        atol=1.0e-12,
    )
    state = StateVec.unpack(initial.physical_state, grid.size)
    np.testing.assert_allclose(state.P, model.material.P_ion0, rtol=2.0e-15)


def test_projected_perturbation_is_rate_compatible_and_conservative():
    grid, _stack, _reference, model = _single_ion_problem()
    coordinate = np.zeros(model.layout.size)
    coordinate[1 : grid.size - 1] = 0.02 * np.sin(
        np.linspace(0.0, np.pi, grid.size - 2)
    )
    coordinate[model.layout.node_count + 1 : 2 * model.layout.node_count - 1] = (
        -0.01 * np.sin(np.linspace(0.0, np.pi, grid.size - 2))
    )
    coordinate[model.layout.positive_ion_slice] = np.linspace(-0.15, 0.15, grid.size)
    coordinate = project_single_ion_algebraic_state(model, coordinate)
    derivative = model.compatible_derivative(coordinate)
    report = model.residual_report(coordinate, derivative)

    assert report.max_normalized_differential_residual < 1.0e-13
    assert report.max_normalized_algebraic_residual < 1.0e-13
    assert abs(report.positive_ion_inventory_residual_m2_s) < 1.0e-12
    inventory_scale = dual_cell_integral(grid, model.material.P_ion0)
    assert abs(report.positive_ion_rhs_inventory_rate_m2_s) / inventory_scale < 1.0e-15


def test_exact_derivative_jacobian_matches_independent_central_difference():
    _grid, _stack, _reference, model = _single_ion_problem()
    initial = build_single_ion_consistent_initial_condition(model)
    analytic = model.derivative_jacobian(initial.coordinate)
    finite_difference = finite_difference_single_ion_derivative_jacobian(
        model,
        initial.coordinate,
        initial.derivative,
    )

    np.testing.assert_allclose(finite_difference, analytic, rtol=2.0e-10, atol=1.0e-12)
    diagonal = np.diag(analytic)
    assert np.all(diagonal[model.layout.positive_ion_slice] > 0.0)
    assert np.all(diagonal[model.layout.potential_slice] == 0.0)


def test_exact_algebraic_state_rows_match_independent_central_difference():
    grid, _stack, _reference, model = _single_ion_problem()
    coordinate = np.zeros(model.layout.size)
    coordinate[model.layout.positive_ion_slice] = np.linspace(-0.05, 0.05, grid.size)
    coordinate = project_single_ion_algebraic_state(model, coordinate)
    derivative = model.compatible_derivative(coordinate)
    analytic = model.algebraic_state_jacobian(coordinate)
    finite_difference = finite_difference_single_ion_state_jacobian(
        model,
        coordinate,
        derivative,
        relative_step=2.0e-6,
    )

    np.testing.assert_allclose(
        finite_difference[model.layout.algebraic_mask],
        analytic[model.layout.algebraic_mask],
        rtol=2.0e-7,
        atol=2.0e-10,
    )
    poisson_interior = slice(3 * grid.size + 1, 4 * grid.size - 1)
    assert np.any(analytic[poisson_interior, 2 * grid.size : 3 * grid.size] != 0.0)


def test_zero_net_ion_charge_recovers_no_ion_carrier_poisson_limit():
    grid, stack, reference, ion_model = _single_ion_problem()
    assert stack.layers[0].params is not None
    no_ion_layer = replace(
        stack.layers[0],
        params=replace(stack.layers[0].params, D_ion=0.0, P0=0.0),
    )
    no_ion_stack = replace(stack, layers=(no_ion_layer,))
    no_ion_reference = solve_equilibrium(grid, no_ion_stack)
    no_ion_model = build_no_ion_no_interface_dae(
        grid,
        no_ion_stack,
        no_ion_reference,
        reference_time_s=1.0e-9,
    )

    carrier_coordinate = np.zeros(2 * grid.size)
    carrier_coordinate[1 : grid.size - 1] = 0.01
    carrier_coordinate[grid.size + 1 : 2 * grid.size - 1] = 0.01
    ion_coordinate = np.zeros(ion_model.layout.size)
    ion_coordinate[: 2 * grid.size] = carrier_coordinate
    no_ion_coordinate = np.zeros(no_ion_model.layout.size)
    no_ion_coordinate[: 2 * grid.size] = carrier_coordinate
    ion_coordinate = project_single_ion_algebraic_state(ion_model, ion_coordinate)
    no_ion_coordinate = project_algebraic_state(no_ion_model, no_ion_coordinate)

    ion_n, ion_p, ion_density, ion_phi = ion_model.physical_fields(ion_coordinate)
    no_ion_n, no_ion_p, no_ion_phi = no_ion_model.physical_fields(no_ion_coordinate)
    np.testing.assert_allclose(ion_n, no_ion_n, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(ion_p, no_ion_p, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(ion_density, ion_model.material.P_ion0, rtol=2.0e-15)
    np.testing.assert_allclose(ion_phi, no_ion_phi, rtol=0.0, atol=2.0e-18)


def test_capability_rejects_no_mobile_ion_and_dual_ion():
    grid, stack, reference, _model = _single_ion_problem()
    assert stack.layers[0].params is not None
    no_ion_layer = replace(
        stack.layers[0],
        params=replace(stack.layers[0].params, D_ion=0.0, P0=0.0),
    )
    no_ion_stack = replace(stack, layers=(no_ion_layer,))
    with pytest.raises(DAECapabilityError, match="positive mobile ion"):
        build_single_positive_ion_dae(grid, no_ion_stack, solve_equilibrium(grid, no_ion_stack))

    dual_layer = replace(
        stack.layers[0],
        params=replace(
            stack.layers[0].params,
            D_ion_neg=1.0e-17,
            P0_neg=1.0e24,
            P_lim_neg=1.0e27,
        ),
    )
    dual_stack = replace(stack, layers=(dual_layer,))
    with pytest.raises(DAECapabilityError, match="dual mobile ions"):
        build_single_positive_ion_dae(grid, dual_stack, solve_equilibrium(grid, dual_stack))


def test_capability_rejects_selective_contacts():
    grid, stack, _reference, _model = _single_ion_problem()
    selective = replace(stack, S_n_left=1.0)
    with pytest.raises(DAECapabilityError, match="selective contacts"):
        build_single_positive_ion_dae(
            grid,
            selective,
            solve_equilibrium(grid, selective),
        )
