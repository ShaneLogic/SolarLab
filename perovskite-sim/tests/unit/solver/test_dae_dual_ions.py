from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.physics.generation import dual_cell_integral
from perovskite_sim.solver.dae import DAECapabilityError
from perovskite_sim.solver.dae_dual_ions import (
    build_dual_ion_consistent_initial_condition,
    build_dual_ion_dae,
    finite_difference_dual_ion_derivative_jacobian,
    finite_difference_dual_ion_state_jacobian,
    project_dual_ion_algebraic_state,
)
from perovskite_sim.solver.mol import StateVec
from perovskite_sim.solver.newton import solve_equilibrium


def _dual_ion_stack(
    *,
    shared_site: bool = True,
    negative_limit_m3: float | None = None,
):
    source = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    layer = source.layers[1]
    assert layer.params is not None
    limit = layer.params.P_lim if negative_limit_m3 is None else negative_limit_m3
    params = replace(
        layer.params,
        D_ion_neg=3.2e-18,
        P0_neg=layer.params.P0,
        P_lim_neg=limit,
    )
    return replace(
        source,
        layers=(replace(layer, params=params),),
        V_bi=0.0,
        built_in_potential_mode="legacy_manual",
        Phi=1.0e17,
        interfaces=(),
        interface_defects=(),
        grid_interval_weights=(),
        grid_alphas=(),
        ion_steric_diffusion_only=True,
        ion_steric_shared_site=shared_site,
        mode="full",
    )


def _dual_ion_problem(
    *,
    intervals: int = 6,
    illuminated: bool = False,
    shared_site: bool = True,
):
    stack = _dual_ion_stack(shared_site=shared_site)
    layer = stack.layers[0]
    grid = multilayer_grid([Layer(layer.thickness, intervals)], alpha=1.0)
    reference = solve_equilibrium(grid, stack)
    model = build_dual_ion_dae(
        grid,
        stack,
        reference,
        illuminated=illuminated,
        carrier_reference_time_s=1.0e-9,
        ion_reference_time_s=1.0,
    )
    return grid, stack, reference, model


def test_dual_ion_layout_classifies_both_ion_species_as_differential():
    grid, _stack, _reference, model = _dual_ion_problem()
    count = grid.size
    layout = model.layout

    assert layout.size == 5 * count
    assert layout.shared_site
    assert np.count_nonzero(layout.differential_mask) == 4 * count - 4
    assert np.count_nonzero(layout.algebraic_mask) == count + 4
    assert np.all(layout.differential_mask[layout.positive_ion_slice])
    assert np.all(layout.differential_mask[layout.negative_ion_slice])
    assert np.all(layout.algebraic_mask[layout.potential_slice])
    assert not layout.differential_mask.flags.writeable


def test_reference_coordinate_recovers_shared_site_species_and_vacancy():
    _grid, _stack, reference, model = _dual_ion_problem()
    n, p, positive_ion, negative_ion, _phi = model.physical_fields(
        np.zeros(model.layout.size)
    )
    state = StateVec.unpack(reference, model.layout.node_count)
    assert state.P_neg is not None

    np.testing.assert_allclose(n, state.n, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(p, state.p, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(positive_ion, state.P, rtol=2.0e-15, atol=0.0)
    np.testing.assert_allclose(negative_ion, state.P_neg, rtol=2.0e-15, atol=0.0)
    assert np.all(positive_ion > 0.0)
    assert np.all(negative_ion > 0.0)
    assert np.all(
        positive_ion + negative_ion < model.layout.positive_ion_site_limit_m3
    )


def test_shared_site_coordinate_couples_species_and_preserves_total_bound():
    _grid, _stack, _reference, model = _dual_ion_problem()
    base = model.physical_fields(np.zeros(model.layout.size))
    coordinate = np.zeros(model.layout.size)
    coordinate[model.layout.negative_ion_slice] = 0.4
    perturbed = model.physical_fields(coordinate)

    assert np.all(perturbed[2] < base[2])
    assert np.all(perturbed[3] > base[3])
    assert np.all(
        perturbed[2] + perturbed[3]
        < model.layout.positive_ion_site_limit_m3
    )
    mass = model.ion_coordinate_jacobian_m3(coordinate)
    assert np.all(mass[:, 0, 1] < 0.0)
    assert np.all(mass[:, 1, 0] < 0.0)


def test_distinct_sublattice_coordinates_are_independent():
    _grid, _stack, _reference, model = _dual_ion_problem(shared_site=False)
    assert not model.layout.shared_site
    base = model.physical_fields(np.zeros(model.layout.size))
    coordinate = np.zeros(model.layout.size)
    coordinate[model.layout.negative_ion_slice] = 0.4
    perturbed = model.physical_fields(coordinate)

    np.testing.assert_array_equal(perturbed[2], base[2])
    assert np.all(perturbed[3] > base[3])
    mass = model.ion_coordinate_jacobian_m3(coordinate)
    np.testing.assert_array_equal(mass[:, 0, 1], 0.0)
    np.testing.assert_array_equal(mass[:, 1, 0], 0.0)


def test_shared_site_coordinate_fails_closed_at_floating_point_saturation():
    _grid, _stack, _reference, model = _dual_ion_problem()
    coordinate = np.zeros(model.layout.size)
    coordinate[model.layout.positive_ion_slice] = 1.0e3
    with pytest.raises(ValueError, match="shared-site coordinate saturated"):
        model.physical_fields(coordinate)


def test_consistent_initial_condition_closes_all_rows_and_both_inventories():
    grid, _stack, _reference, model = _dual_ion_problem(illuminated=True)
    initial = build_dual_ion_consistent_initial_condition(model)

    assert initial.certified
    assert initial.report.max_normalized_residual < 1.0e-12
    assert initial.report.max_normalized_positive_ion_residual < 1.0e-15
    assert initial.report.max_normalized_negative_ion_residual < 1.0e-15
    positive_scale = dual_cell_integral(grid, model.material.P_ion0)
    assert model.material.P_ion0_neg is not None
    negative_scale = dual_cell_integral(grid, model.material.P_ion0_neg)
    assert (
        abs(initial.report.positive_ion_rhs_inventory_rate_m2_s)
        / positive_scale
        < 1.0e-15
    )
    assert (
        abs(initial.report.negative_ion_rhs_inventory_rate_m2_s)
        / negative_scale
        < 1.0e-15
    )
    assert len(initial.state_sha256) == 64
    assert not initial.coordinate.flags.writeable
    np.testing.assert_allclose(
        model.residual(initial.coordinate, initial.derivative),
        0.0,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_projected_perturbation_is_rate_compatible_and_conservative():
    grid, _stack, _reference, model = _dual_ion_problem()
    count = grid.size
    coordinate = np.zeros(model.layout.size)
    coordinate[1 : count - 1] = 0.02 * np.sin(
        np.linspace(0.0, np.pi, count - 2)
    )
    coordinate[count + 1 : 2 * count - 1] = -0.01 * np.sin(
        np.linspace(0.0, np.pi, count - 2)
    )
    coordinate[model.layout.positive_ion_slice] = np.linspace(-0.1, 0.1, count)
    coordinate[model.layout.negative_ion_slice] = np.linspace(0.08, -0.08, count)
    coordinate = project_dual_ion_algebraic_state(model, coordinate)
    derivative = model.compatible_derivative(coordinate)
    report = model.residual_report(coordinate, derivative)

    assert report.max_normalized_differential_residual < 2.0e-12
    assert report.max_normalized_algebraic_residual < 1.0e-13
    positive_scale = dual_cell_integral(grid, model.material.P_ion0)
    assert model.material.P_ion0_neg is not None
    negative_scale = dual_cell_integral(grid, model.material.P_ion0_neg)
    assert abs(report.positive_ion_rhs_inventory_rate_m2_s) / positive_scale < 1e-15
    assert abs(report.negative_ion_rhs_inventory_rate_m2_s) / negative_scale < 1e-15


def test_exact_derivative_jacobian_matches_independent_central_difference():
    _grid, _stack, _reference, model = _dual_ion_problem()
    initial = build_dual_ion_consistent_initial_condition(model)
    analytic = model.derivative_jacobian(initial.coordinate)
    finite_difference = finite_difference_dual_ion_derivative_jacobian(
        model,
        initial.coordinate,
        initial.derivative,
    )

    np.testing.assert_allclose(finite_difference, analytic, rtol=3.0e-10, atol=1e-12)
    count = model.layout.node_count
    assert np.any(analytic[2 * count : 3 * count, 3 * count : 4 * count] < 0.0)
    assert np.all(analytic[model.layout.potential_slice] == 0.0)


def test_exact_algebraic_rows_match_independent_central_difference():
    grid, _stack, _reference, model = _dual_ion_problem()
    coordinate = np.zeros(model.layout.size)
    coordinate[model.layout.positive_ion_slice] = np.linspace(-0.05, 0.05, grid.size)
    coordinate[model.layout.negative_ion_slice] = np.linspace(0.04, -0.04, grid.size)
    coordinate = project_dual_ion_algebraic_state(model, coordinate)
    derivative = model.compatible_derivative(coordinate)
    analytic = model.algebraic_state_jacobian(coordinate)
    finite_difference = finite_difference_dual_ion_state_jacobian(
        model,
        coordinate,
        derivative,
        relative_step=2.0e-6,
    )

    np.testing.assert_allclose(
        finite_difference[model.layout.algebraic_mask],
        analytic[model.layout.algebraic_mask],
        rtol=3.0e-7,
        atol=3.0e-10,
    )
    count = grid.size
    poisson_rows = slice(4 * count + 1, 5 * count - 1)
    assert np.any(analytic[poisson_rows, 2 * count : 3 * count] > 0.0)
    assert np.any(analytic[poisson_rows, 3 * count : 4 * count] < 0.0)


def test_symmetric_species_motion_cancels_net_ionic_poisson_charge():
    grid, _stack, _reference, model = _dual_ion_problem()
    base = project_dual_ion_algebraic_state(
        model,
        np.zeros(model.layout.size),
    )
    coordinate = np.zeros(model.layout.size)
    profile = 0.2 * np.sin(np.linspace(0.0, np.pi, grid.size))
    coordinate[model.layout.positive_ion_slice] = profile
    coordinate[model.layout.negative_ion_slice] = profile
    coordinate = project_dual_ion_algebraic_state(model, coordinate)
    _n, _p, positive_ion, negative_ion, phi = model.physical_fields(coordinate)

    np.testing.assert_allclose(positive_ion, negative_ion, rtol=2.0e-15, atol=0.0)
    np.testing.assert_allclose(
        phi,
        model.physical_fields(base)[4],
        rtol=0.0,
        atol=2.0e-18,
    )


def test_positive_and_negative_coordinate_charge_responses_are_antisymmetric():
    grid, _stack, _reference, model = _dual_ion_problem()
    base = project_dual_ion_algebraic_state(model, np.zeros(model.layout.size))
    profile = 0.1 * np.sin(np.linspace(0.0, np.pi, grid.size))
    positive = np.zeros(model.layout.size)
    negative = np.zeros(model.layout.size)
    positive[model.layout.positive_ion_slice] = profile
    negative[model.layout.negative_ion_slice] = profile
    positive = project_dual_ion_algebraic_state(model, positive)
    negative = project_dual_ion_algebraic_state(model, negative)
    base_phi = model.physical_fields(base)[4]

    np.testing.assert_allclose(
        model.physical_fields(positive)[4] - base_phi,
        -(model.physical_fields(negative)[4] - base_phi),
        rtol=2.0e-12,
        atol=2.0e-18,
    )


def test_capability_rejects_missing_negative_species():
    source = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
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
    grid = multilayer_grid([Layer(layer.thickness, 5)], alpha=1.0)
    with pytest.raises(DAECapabilityError, match="positive and negative"):
        build_dual_ion_dae(grid, stack, solve_equilibrium(grid, stack))


def test_capability_rejects_mismatched_shared_site_limits():
    stack = _dual_ion_stack(negative_limit_m3=1.5e27)
    layer = stack.layers[0]
    grid = multilayer_grid([Layer(layer.thickness, 5)], alpha=1.0)
    with pytest.raises(DAECapabilityError, match="common site limit"):
        build_dual_ion_dae(grid, stack, solve_equilibrium(grid, stack))


def test_capability_rejects_overoccupied_shared_reference():
    grid, stack, reference, model = _dual_ion_problem()
    state = StateVec.unpack(reference, grid.size)
    assert state.P_neg is not None
    limit = model.material.P_lim_node
    overoccupied = StateVec.pack(
        state.n,
        state.p,
        0.6 * limit,
        0.5 * limit,
    )
    with pytest.raises(ValueError, match="total occupancy"):
        build_dual_ion_dae(grid, stack, overoccupied)


def test_consistent_initial_condition_is_deterministic_and_read_only():
    _grid, _stack, _reference, model = _dual_ion_problem()
    first = build_dual_ion_consistent_initial_condition(model)
    second = build_dual_ion_consistent_initial_condition(model)

    assert first.state_sha256 == second.state_sha256
    np.testing.assert_array_equal(first.coordinate, second.coordinate)
    np.testing.assert_array_equal(first.derivative, second.derivative)
    assert not first.physical_state.flags.writeable
    assert not first.report.normalized_residual.flags.writeable
