from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.solver.dae_dual_ion_integrator import (
    dual_ion_backward_euler_derivative,
    dual_ion_density_difference_m3,
    finite_difference_dual_ion_backward_euler_jacobian,
    run_dual_ion_backward_euler_reference,
)
from perovskite_sim.solver.dae_dual_ions import (
    build_dual_ion_consistent_initial_condition,
    build_dual_ion_dae,
    project_dual_ion_algebraic_state,
)
from perovskite_sim.solver.newton import solve_equilibrium


def _problem(*, intervals: int = 5, shared_site: bool = True, voltage_V: float = 0.01):
    source = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    layer = source.layers[1]
    assert layer.params is not None
    dual_layer = replace(
        layer,
        params=replace(
            layer.params,
            D_ion_neg=3.2e-18,
            P0_neg=layer.params.P0,
            P_lim_neg=layer.params.P_lim,
        ),
    )
    stack = replace(
        source,
        layers=(dual_layer,),
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
    grid = multilayer_grid([Layer(dual_layer.thickness, intervals)], alpha=1.0)
    reference = solve_equilibrium(grid, stack)
    model = build_dual_ion_dae(
        grid,
        stack,
        reference,
        V_app_V=voltage_V,
        carrier_reference_time_s=1.0e-9,
        ion_reference_time_s=1.0,
    )
    initial = build_dual_ion_consistent_initial_condition(model)
    return grid, stack, model, initial


@pytest.mark.parametrize("shared_site", [True, False], ids=["shared", "distinct"])
def test_stable_density_difference_matches_direct_moderate_change(shared_site):
    _grid, _stack, model, initial = _problem(shared_site=shared_site)
    previous = initial.coordinate
    coordinate = np.array(previous, copy=True)
    coordinate[model.layout.positive_ion_slice] += np.linspace(-0.03, 0.04, model.layout.node_count)
    coordinate[model.layout.negative_ion_slice] += np.linspace(0.02, -0.05, model.layout.node_count)

    positive_difference, negative_difference = dual_ion_density_difference_m3(
        model,
        coordinate,
        previous,
    )
    new_fields = model.physical_fields(coordinate)
    old_fields = model.physical_fields(previous)

    np.testing.assert_allclose(
        positive_difference,
        new_fields[2] - old_fields[2],
        rtol=2.0e-13,
        atol=2.0e9,
    )
    np.testing.assert_allclose(
        negative_difference,
        new_fields[3] - old_fields[3],
        rtol=2.0e-13,
        atol=2.0e9,
    )


def test_shared_site_small_difference_retains_coupled_first_order_change():
    _grid, _stack, model, initial = _problem()
    previous = initial.coordinate
    coordinate = np.array(previous, copy=True)
    positive_delta = np.linspace(-1.0e-12, 1.0e-12, model.layout.node_count)
    negative_delta = np.linspace(0.5e-12, -0.5e-12, model.layout.node_count)
    coordinate[model.layout.positive_ion_slice] += positive_delta
    coordinate[model.layout.negative_ion_slice] += negative_delta

    positive_difference, negative_difference = dual_ion_density_difference_m3(
        model,
        coordinate,
        previous,
    )
    expected = np.einsum(
        "nij,nj->ni",
        model.ion_coordinate_jacobian_m3(previous),
        np.stack((positive_delta, negative_delta), axis=1),
    )

    assert np.any(positive_difference != 0.0)
    assert np.any(negative_difference != 0.0)
    np.testing.assert_allclose(positive_difference, expected[:, 0], rtol=2e-12)
    np.testing.assert_allclose(negative_difference, expected[:, 1], rtol=2e-12)


def test_backward_euler_coordinate_rate_recovers_physical_storage():
    _grid, _stack, model, initial = _problem()
    previous = initial.coordinate
    coordinate = np.array(previous, copy=True)
    coordinate[1 : model.layout.node_count - 1] += 0.01
    coordinate[model.layout.node_count + 1 : 2 * model.layout.node_count - 1] -= 0.02
    coordinate[model.layout.positive_ion_slice] += 0.03
    coordinate[model.layout.negative_ion_slice] -= 0.01
    coordinate = project_dual_ion_algebraic_state(model, coordinate)
    dt_s = 2.0e-3

    derivative = dual_ion_backward_euler_derivative(
        model,
        coordinate,
        previous,
        dt_s,
    )
    positive_difference, negative_difference = dual_ion_density_difference_m3(
        model,
        coordinate,
        previous,
    )
    ion_rate = np.stack(
        (
            derivative[model.layout.positive_ion_slice],
            derivative[model.layout.negative_ion_slice],
        ),
        axis=1,
    )
    recovered = np.einsum(
        "nij,nj->ni",
        model.ion_coordinate_jacobian_m3(coordinate),
        ion_rate,
    )

    np.testing.assert_allclose(recovered[:, 0], positive_difference / dt_s, rtol=3e-15)
    np.testing.assert_allclose(recovered[:, 1], negative_difference / dt_s, rtol=3e-15)


def test_complete_be_jacobian_has_exact_algebraic_rows():
    _grid, _stack, model, initial = _problem()
    previous = initial.coordinate
    coordinate = np.array(previous, copy=True)
    coordinate[model.layout.positive_ion_slice] += 0.01
    coordinate[model.layout.negative_ion_slice] -= 0.01
    coordinate = project_dual_ion_algebraic_state(model, coordinate)
    derivative = dual_ion_backward_euler_derivative(model, coordinate, previous, 1e-3)
    finite_difference = finite_difference_dual_ion_backward_euler_jacobian(
        model,
        coordinate,
        previous,
        1e-3,
        relative_step=2e-6,
    )
    analytic_algebraic = model.algebraic_state_jacobian(coordinate)

    np.testing.assert_allclose(
        finite_difference[model.layout.algebraic_mask],
        analytic_algebraic[model.layout.algebraic_mask],
        rtol=4e-7,
        atol=4e-10,
    )
    assert np.all(np.isfinite(derivative))


@pytest.mark.parametrize("shared_site", [True, False], ids=["shared", "distinct"])
def test_dense_reference_advances_both_topologies_and_preserves_inventories(
    shared_site,
):
    _grid, _stack, model, initial = _problem(shared_site=shared_site)
    result = run_dual_ion_backward_euler_reference(
        model,
        np.linspace(0.0, 2.0e-3, 3),
        initial=initial,
    )

    assert result.success
    assert result.jacobian_mode == "dense_central"
    assert result.coordinates.shape == (3, model.layout.size)
    assert result.physical_states.shape == (3, 4 * model.layout.node_count)
    assert len(result.step_reports) == 2
    assert result.total_nonlinear_iterations > 0
    assert result.max_normalized_differential_residual <= 1.0e-9
    assert result.max_normalized_algebraic_residual <= 1.0e-9
    assert result.max_relative_positive_ion_inventory_drift < 1.0e-12
    assert result.max_relative_negative_ion_inventory_drift < 1.0e-12
    assert result.minimum_site_vacancy_fraction > 0.0
    assert len(result.trajectory_sha256) == 64
    assert not result.coordinates.flags.writeable


def test_dense_reference_is_deterministic():
    _grid, _stack, model, initial = _problem()
    time = np.array([0.0, 1.0e-3])

    first = run_dual_ion_backward_euler_reference(model, time, initial=initial)
    second = run_dual_ion_backward_euler_reference(model, time, initial=initial)

    assert first.trajectory_sha256 == second.trajectory_sha256
    np.testing.assert_array_equal(first.coordinates, second.coordinates)
    np.testing.assert_array_equal(first.physical_states, second.physical_states)


@pytest.mark.parametrize("shared_site", [True, False], ids=["shared", "distinct"])
def test_structured_reference_matches_dense_trajectory(shared_site):
    _grid, _stack, model, initial = _problem(shared_site=shared_site)
    time = np.linspace(0.0, 2.0e-3, 3)
    dense = run_dual_ion_backward_euler_reference(
        model,
        time,
        initial=initial,
    )
    structured = run_dual_ion_backward_euler_reference(
        model,
        time,
        initial=initial,
        jacobian_mode="structured_analytic",
    )

    assert structured.jacobian_mode == "structured_analytic"
    np.testing.assert_allclose(
        structured.coordinates,
        dense.coordinates,
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        structured.physical_states,
        dense.physical_states,
        rtol=3.0e-14,
        atol=0.0,
    )
    np.testing.assert_allclose(
        structured.potentials_V,
        dense.potentials_V,
        rtol=0.0,
        atol=3.0e-15,
    )
    assert structured.total_residual_evaluations < dense.total_residual_evaluations


@pytest.mark.parametrize(
    ("time_s", "kwargs", "match"),
    [
        (np.array([0.0]), {}, "time_s"),
        (np.array([0.0, 0.0]), {}, "time_s"),
        (np.array([0.0, 1.0e-3]), {"residual_tolerance": 0.0}, "residual_tolerance"),
        (np.array([0.0, 1.0e-3]), {"max_newton_iterations": 0}, "max_newton_iterations"),
        (np.array([0.0, 1.0e-3]), {"jacobian_mode": "unknown"}, "jacobian_mode"),
    ],
)
def test_dense_reference_rejects_invalid_controls(time_s, kwargs, match):
    _grid, _stack, model, initial = _problem()
    with pytest.raises(ValueError, match=match):
        run_dual_ion_backward_euler_reference(
            model,
            time_s,
            initial=initial,
            **kwargs,
        )
