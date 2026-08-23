from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.solver.dae_integrator import DAEIntegrationError
from perovskite_sim.solver.dae_ion_integrator import (
    finite_difference_single_ion_backward_euler_jacobian,
    run_single_ion_backward_euler_reference,
    single_ion_backward_euler_derivative,
    single_ion_backward_euler_derivative_chain,
)
from perovskite_sim.solver.dae_ions import (
    build_single_ion_consistent_initial_condition,
    build_single_positive_ion_dae,
    finite_difference_single_ion_state_jacobian,
    project_single_ion_algebraic_state,
)
from perovskite_sim.solver.newton import solve_equilibrium


def _single_ion_model(*, illuminated: bool = False, V_app_V: float = 0.0):
    source = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    layer = source.layers[1]
    stack = replace(
        source,
        layers=(layer,),
        V_bi=0.0,
        built_in_potential_mode="legacy_manual",
        Phi=1.0e17 if illuminated else 0.0,
        interfaces=(),
        interface_defects=(),
        grid_interval_weights=(),
        grid_alphas=(),
    )
    grid = multilayer_grid([Layer(layer.thickness, 6)], alpha=1.0)
    reference = solve_equilibrium(grid, stack)
    model = build_single_positive_ion_dae(
        grid,
        stack,
        reference,
        V_app_V=V_app_V,
        illuminated=illuminated,
        carrier_reference_time_s=1.0e-9,
        ion_reference_time_s=1.0,
    )
    return grid, model


def test_exact_be_derivative_chain_matches_central_difference():
    _grid, model = _single_ion_model()
    previous = np.zeros(model.layout.size)
    coordinate = previous.copy()
    count = model.layout.node_count
    coordinate[1 : count - 1] = np.linspace(-0.02, 0.03, count - 2)
    coordinate[count + 1 : 2 * count - 1] = np.linspace(0.01, -0.02, count - 2)
    coordinate[model.layout.positive_ion_slice] = np.linspace(-0.1, 0.1, count)
    dt_s = 2.0e-3
    analytic = single_ion_backward_euler_derivative_chain(
        model, coordinate, previous, dt_s
    )

    finite_difference = np.zeros(model.layout.size)
    for index in np.flatnonzero(model.layout.differential_mask):
        step = 1.0e-6
        plus = coordinate.copy()
        minus = coordinate.copy()
        plus[index] += step
        minus[index] -= step
        finite_difference[index] = (
            single_ion_backward_euler_derivative(model, plus, previous, dt_s)[index]
            - single_ion_backward_euler_derivative(model, minus, previous, dt_s)[index]
        ) / (2.0 * step)
    np.testing.assert_allclose(
        analytic[model.layout.differential_mask],
        finite_difference[model.layout.differential_mask],
        rtol=2.0e-9,
        atol=2.0e-7,
    )
    assert np.all(analytic[model.layout.algebraic_mask] == 0.0)


def test_hybrid_be_jacobian_matches_independent_complete_difference():
    _grid, model = _single_ion_model(illuminated=True)
    initial = build_single_ion_consistent_initial_condition(model)
    previous = initial.coordinate
    coordinate = np.array(previous, copy=True)
    count = model.layout.node_count
    coordinate[1 : count - 1] = np.linspace(-0.01, 0.02, count - 2)
    coordinate[count + 1 : 2 * count - 1] = np.linspace(0.02, -0.01, count - 2)
    coordinate[model.layout.positive_ion_slice] = np.linspace(-0.05, 0.05, count)
    coordinate = project_single_ion_algebraic_state(model, coordinate)
    dt_s = 1.0e-3
    derivative = single_ion_backward_euler_derivative(
        model, coordinate, previous, dt_s
    )
    hybrid = finite_difference_single_ion_state_jacobian(
        model, coordinate, derivative, relative_step=2.0e-6
    )
    exact_algebraic = model.algebraic_state_jacobian(coordinate)
    hybrid[model.layout.algebraic_mask] = exact_algebraic[
        model.layout.algebraic_mask
    ]
    hybrid += np.diag(
        model.derivative_jacobian_diagonal(coordinate)
        * single_ion_backward_euler_derivative_chain(
            model, coordinate, previous, dt_s
        )
    )
    complete = finite_difference_single_ion_backward_euler_jacobian(
        model,
        coordinate,
        previous,
        dt_s,
        relative_step=2.0e-6,
    )

    np.testing.assert_allclose(complete, hybrid, rtol=3.0e-6, atol=3.0e-8)


def test_dark_equilibrium_is_stationary_conservative_and_reproducible():
    _grid, model = _single_ion_model()
    time = np.array([0.0, 1.0e-3, 2.0e-3])

    first = run_single_ion_backward_euler_reference(model, time)
    second = run_single_ion_backward_euler_reference(model, time)

    assert first.success
    assert first.trajectory_sha256 == second.trajectory_sha256
    np.testing.assert_array_equal(first.coordinates, second.coordinates)
    np.testing.assert_array_equal(first.physical_states, second.physical_states)
    np.testing.assert_array_equal(first.potentials_V, second.potentials_V)
    assert first.max_normalized_differential_residual <= 1.0e-9
    assert first.max_normalized_algebraic_residual <= 1.0e-9
    assert first.max_relative_positive_ion_inventory_drift < 1.0e-15
    assert first.max_positive_ion_balance_defect_m2_s < 1.0e-6
    assert not first.time_s.flags.writeable
    assert not first.coordinates.flags.writeable
    assert not first.physical_states.flags.writeable
    assert not first.potentials_V.flags.writeable


def test_illuminated_step_closes_carrier_ion_and_algebraic_rows():
    _grid, model = _single_ion_model(illuminated=True)
    result = run_single_ion_backward_euler_reference(
        model,
        np.linspace(0.0, 1.0e-9, 3),
        residual_tolerance=1.0e-10,
    )

    assert result.success
    assert result.total_nonlinear_iterations > 0
    assert result.total_jacobian_evaluations > 0
    assert result.max_normalized_carrier_residual <= 1.0e-10
    assert result.max_normalized_positive_ion_residual <= 1.0e-10
    assert result.max_normalized_algebraic_residual <= 1.0e-10
    assert result.max_electron_balance_defect_A_m2 < 1.0e-7
    assert result.max_hole_balance_defect_A_m2 < 1.0e-7
    assert result.max_relative_positive_ion_inventory_drift < 1.0e-14
    assert result.max_positive_ion_balance_defect_m2_s < 1.0e9
    count = model.layout.node_count
    ion = result.physical_states[:, 2 * count : 3 * count]
    assert np.all(ion > 0.0)
    assert np.all(ion < model.material.P_lim_node)


def test_voltage_driven_ion_motion_preserves_blocking_inventory():
    grid, model = _single_ion_model(V_app_V=0.01)
    time = np.array([0.0, 5.0e-3, 1.0e-2])

    result = run_single_ion_backward_euler_reference(
        model,
        time,
        residual_tolerance=1.0e-9,
        max_newton_iterations=24,
    )

    count = model.layout.node_count
    initial_ion = result.physical_states[0, 2 * count : 3 * count]
    terminal_ion = result.physical_states[-1, 2 * count : 3 * count]
    relative_motion = np.max(np.abs(terminal_ion - initial_ion) / initial_ion)
    inventory_rate_scale = result.initial_positive_ion_inventory_m2 / np.min(
        np.diff(time)
    )
    assert relative_motion > 1.0e-6
    assert result.max_relative_positive_ion_inventory_drift < 2.0e-15
    assert (
        result.max_positive_ion_balance_defect_m2_s / inventory_rate_scale
        < 1.0e-18
    )
    assert result.max_positive_ion_rhs_inventory_rate_m2_s == 0.0
    assert all(
        abs(report.positive_ion_inventory_change_m2)
        / result.initial_positive_ion_inventory_m2
        < 2.0e-15
        for report in result.step_reports
    )
    assert grid.size == count


@pytest.mark.parametrize(
    ("keyword", "value", "match"),
    [
        ("residual_tolerance", 0.0, "residual_tolerance"),
        ("max_newton_iterations", 0, "max_newton_iterations"),
        ("max_line_search_backtracks", -1, "max_line_search_backtracks"),
        ("max_log_density_update", np.inf, "max_log_density_update"),
        ("max_ion_logit_update", 0.0, "max_ion_logit_update"),
        ("finite_difference_relative_step", np.nan, "finite_difference"),
    ],
)
def test_solver_control_validation_fails_closed(keyword, value, match):
    _grid, model = _single_ion_model()
    with pytest.raises(ValueError, match=match):
        run_single_ion_backward_euler_reference(
            model,
            np.array([0.0, 1.0e-9]),
            **{keyword: value},
        )


def test_uncertified_initial_condition_is_rejected():
    _grid, model = _single_ion_model()
    initial = build_single_ion_consistent_initial_condition(model)
    invalid = replace(initial, certified=False)
    with pytest.raises(ValueError, match="not certified"):
        run_single_ion_backward_euler_reference(
            model,
            np.array([0.0, 1.0e-9]),
            initial=invalid,
        )


def test_newton_exhaustion_reports_step_and_time():
    _grid, model = _single_ion_model(illuminated=True)
    with pytest.raises(DAEIntegrationError) as caught:
        run_single_ion_backward_euler_reference(
            model,
            np.array([0.0, 1.0e-5]),
            residual_tolerance=1.0e-12,
            max_newton_iterations=1,
            max_line_search_backtracks=0,
        )
    assert caught.value.step_index == 1
    assert caught.value.time_s == pytest.approx(1.0e-5)
    assert np.isfinite(caught.value.residual_norm)
