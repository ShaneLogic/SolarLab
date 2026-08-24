from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from perovskite_sim.physics.generation import dual_cell_integral
from perovskite_sim.solver.dae_ion_interface_integrator import (
    IonInterfaceDAEIntegrationError,
    finite_difference_ion_interface_backward_euler_jacobian,
    ion_interface_backward_euler_derivative,
    ion_interface_backward_euler_derivative_chain,
    run_ion_interface_backward_euler_reference,
)
from perovskite_sim.solver.dae_ion_interface_states import (
    build_single_ion_algebraic_interface_consistent_initial_condition,
    finite_difference_state_jacobian,
    project_single_ion_algebraic_interface_state,
)
from perovskite_sim.solver.mol import StateVec
from tests.unit.solver.test_dae_ion_interface_states import _problem


def test_exact_be_derivative_chain_matches_independent_central_stencil():
    _grid, _stack, _reference, model = _problem()
    initial = build_single_ion_algebraic_interface_consistent_initial_condition(model)
    previous = initial.coordinate
    coordinate = np.array(previous, copy=True)
    count = model.layout.node_count
    coordinate[1 : count - 1] += np.linspace(-0.02, 0.03, count - 2)
    coordinate[count + 1 : 2 * count - 1] += np.linspace(
        0.01,
        -0.02,
        count - 2,
    )
    coordinate[model.layout.positive_ion_slice] += np.linspace(-0.1, 0.1, count)
    coordinate = project_single_ion_algebraic_interface_state(model, coordinate)
    dt_s = 2.0e-3
    analytic = ion_interface_backward_euler_derivative_chain(
        model,
        coordinate,
        previous,
        dt_s,
    )

    reference = np.zeros(model.layout.size)
    for index in np.flatnonzero(model.layout.differential_mask):
        step = 1.0e-6
        plus = coordinate.copy()
        minus = coordinate.copy()
        plus[index] += step
        minus[index] -= step
        reference[index] = (
            ion_interface_backward_euler_derivative(
                model,
                plus,
                previous,
                dt_s,
            )[index]
            - ion_interface_backward_euler_derivative(
                model,
                minus,
                previous,
                dt_s,
            )[index]
        ) / (2.0 * step)

    np.testing.assert_allclose(
        analytic[model.layout.differential_mask],
        reference[model.layout.differential_mask],
        rtol=2.0e-9,
        atol=2.0e-7,
    )
    assert np.all(analytic[model.layout.algebraic_mask] == 0.0)


def test_hybrid_storage_jacobian_matches_complete_be_stencil():
    _grid, _stack, _reference, model = _problem()
    initial = build_single_ion_algebraic_interface_consistent_initial_condition(model)
    previous = initial.coordinate
    coordinate = np.array(previous, copy=True)
    count = model.layout.node_count
    coordinate[1 : count - 1] += np.linspace(-0.01, 0.02, count - 2)
    coordinate[count + 1 : 2 * count - 1] += np.linspace(
        0.02,
        -0.01,
        count - 2,
    )
    coordinate[model.layout.positive_ion_slice] += np.linspace(-0.05, 0.05, count)
    coordinate = project_single_ion_algebraic_interface_state(model, coordinate)
    dt_s = 1.0e-3
    derivative = ion_interface_backward_euler_derivative(
        model,
        coordinate,
        previous,
        dt_s,
    )
    hybrid = finite_difference_state_jacobian(
        model,
        coordinate,
        derivative,
        relative_step=2.0e-6,
    )
    exact_rows = model.boundary_poisson_state_jacobian(coordinate)
    exact_mask = np.zeros(model.layout.size, dtype=bool)
    exact_mask[[0, count - 1, count, 2 * count - 1]] = True
    exact_mask[model.layout.potential_slice] = True
    hybrid[exact_mask] = exact_rows[exact_mask]
    hybrid += np.diag(
        model.derivative_jacobian_diagonal(coordinate)
        * ion_interface_backward_euler_derivative_chain(
            model,
            coordinate,
            previous,
            dt_s,
        )
    )
    complete = finite_difference_ion_interface_backward_euler_jacobian(
        model,
        coordinate,
        previous,
        dt_s,
        relative_step=2.0e-6,
    )
    complete[exact_mask] = exact_rows[exact_mask]

    np.testing.assert_allclose(complete, hybrid, rtol=4.0e-6, atol=5.0e-8)


def test_dense_reference_is_reproducible_and_residual_certified():
    _grid, _stack, _reference, model = _problem()
    time = np.linspace(0.0, 1.0e-9, 3)

    first = run_ion_interface_backward_euler_reference(model, time)
    second = run_ion_interface_backward_euler_reference(model, time)

    assert first.success
    assert first.jacobian_mode == "dense_central"
    assert first.trajectory_sha256 == second.trajectory_sha256
    np.testing.assert_array_equal(first.coordinates, second.coordinates)
    np.testing.assert_array_equal(first.physical_states, second.physical_states)
    np.testing.assert_array_equal(
        first.interface_states_m3,
        second.interface_states_m3,
    )
    np.testing.assert_array_equal(first.potentials_V, second.potentials_V)
    assert first.max_normalized_carrier_residual <= 1.0e-9
    assert first.max_normalized_positive_ion_residual <= 1.0e-9
    assert first.max_normalized_interface_residual <= 1.0e-9
    assert first.max_normalized_algebraic_residual <= 1.0e-9
    assert first.total_nonlinear_iterations > 0
    assert first.total_jacobian_evaluations > 0
    assert first.total_residual_evaluations > first.total_jacobian_evaluations
    for value in (
        first.time_s,
        first.coordinates,
        first.physical_states,
        first.interface_states_m3,
        first.potentials_V,
    ):
        assert not value.flags.writeable


def test_physical_carrier_and_ion_storage_are_exact_backward_euler_differences():
    _grid, _stack, _reference, model = _problem()
    result = run_ion_interface_backward_euler_reference(
        model,
        np.linspace(0.0, 1.0e-9, 3),
    )
    count = model.layout.node_count

    for index, report in enumerate(result.step_reports, start=1):
        current_coordinate = result.coordinates[index]
        previous_coordinate = result.coordinates[index - 1]
        derivative = ion_interface_backward_euler_derivative(
            model,
            current_coordinate,
            previous_coordinate,
            report.dt_s,
        )
        current = StateVec.unpack(result.physical_states[index], count)
        previous = StateVec.unpack(result.physical_states[index - 1], count)
        np.testing.assert_allclose(
            current.n[1:-1] * derivative[1 : count - 1],
            (current.n[1:-1] - previous.n[1:-1]) / report.dt_s,
            rtol=3.0e-14,
            atol=0.0,
        )
        np.testing.assert_allclose(
            current.p[1:-1] * derivative[count + 1 : 2 * count - 1],
            (current.p[1:-1] - previous.p[1:-1]) / report.dt_s,
            rtol=3.0e-14,
            atol=0.0,
        )
        ion_slope = model.positive_ion_coordinate_derivative_m3(current_coordinate)
        ion_coordinate = np.asarray(
            current_coordinate[model.layout.positive_ion_slice],
            dtype=np.longdouble,
        )
        old_ion_coordinate = np.asarray(
            previous_coordinate[model.layout.positive_ion_slice],
            dtype=np.longdouble,
        )
        logit_reference = np.asarray(
            model.layout.positive_ion_logit_reference,
            dtype=np.longdouble,
        )
        site_limit = np.asarray(
            model.layout.positive_ion_site_limit_m3,
            dtype=np.longdouble,
        )
        occupation = 1.0 / (1.0 + np.exp(-(logit_reference + ion_coordinate)))
        old_occupation = 1.0 / (1.0 + np.exp(-(logit_reference + old_ion_coordinate)))
        delta = ion_coordinate - old_ion_coordinate
        positive_difference = occupation * (1.0 - old_occupation) * (-np.expm1(-delta))
        negative_difference = -old_occupation * (1.0 - occupation) * (-np.expm1(delta))
        stable_ion_rate = np.asarray(
            site_limit
            * np.where(delta >= 0.0, positive_difference, negative_difference)
            / np.longdouble(report.dt_s),
            dtype=float,
        )
        np.testing.assert_allclose(
            ion_slope * derivative[model.layout.positive_ion_slice],
            stable_ion_rate,
            rtol=2.0e-14,
            atol=0.0,
        )
        assert np.all(derivative[model.layout.interface_slice] == 0.0)
        assert np.all(derivative[model.layout.potential_slice] == 0.0)


def test_voltage_driven_ion_and_interface_response_preserve_inventory():
    grid, _stack, _reference, model = _problem(V_app_V=0.01)
    time = np.array([0.0, 5.0e-3, 1.0e-2])
    result = run_ion_interface_backward_euler_reference(
        model,
        time,
        max_newton_iterations=24,
    )
    count = model.layout.node_count
    ion = result.physical_states[:, 2 * count : 3 * count]
    relative_ion_motion = np.max(np.abs(ion[-1] / ion[0] - 1.0))
    relative_interface_motion = np.max(
        np.abs(result.interface_states_m3[-1] / result.interface_states_m3[0] - 1.0)
    )
    inventory_rate_scale = result.initial_positive_ion_inventory_m2 / np.min(
        np.diff(time)
    )

    assert relative_ion_motion > 1.0e-6
    assert relative_interface_motion > 1.0e-3
    assert result.max_relative_positive_ion_inventory_drift < 2.0e-15
    assert result.max_positive_ion_balance_defect_m2_s / inventory_rate_scale < 1.0e-18
    assert result.max_positive_ion_rhs_inventory_rate_m2_s == 0.0
    assert result.max_interface_state_balance_m2_s < 1.0e3
    assert all(
        abs(report.positive_ion_inventory_change_m2)
        / result.initial_positive_ion_inventory_m2
        < 2.0e-15
        for report in result.step_reports
    )
    np.testing.assert_allclose(
        [dual_cell_integral(grid, state) for state in ion],
        result.initial_positive_ion_inventory_m2,
        rtol=2.0e-15,
        atol=0.0,
    )


def test_every_accepted_bounded_state_stays_inside_its_capacity():
    _grid, _stack, _reference, model = _problem(V_app_V=0.01)
    result = run_ion_interface_backward_euler_reference(
        model,
        np.linspace(0.0, 2.0e-3, 3),
    )
    count = model.layout.node_count
    ion = result.physical_states[:, 2 * count : 3 * count]

    assert np.all(ion > 0.0)
    assert np.all(ion < model.layout.positive_ion_site_limit_m3[np.newaxis, :])
    assert np.all(result.interface_states_m3 > 0.0)
    assert np.all(
        result.interface_states_m3 < model.layout.interface_capacity_m3[np.newaxis, :]
    )
    assert all(
        np.isfinite(report.max_scaled_jacobian_condition)
        for report in result.step_reports
    )


def test_structured_newton_matches_dense_trajectory_with_far_fewer_residuals():
    _grid, _stack, _reference, model = _problem(V_app_V=0.01)
    time = np.array([0.0, 5.0e-3, 1.0e-2])
    dense = run_ion_interface_backward_euler_reference(
        model,
        time,
        residual_tolerance=1.0e-8,
        max_newton_iterations=24,
    )
    structured = run_ion_interface_backward_euler_reference(
        model,
        time,
        residual_tolerance=1.0e-8,
        max_newton_iterations=24,
        jacobian_mode="structured_analytic",
    )

    assert dense.success and structured.success
    assert structured.jacobian_mode == "structured_analytic"
    assert structured.method.endswith("sparse-analytic-newton-v1")
    np.testing.assert_allclose(
        structured.coordinates,
        dense.coordinates,
        rtol=0.0,
        atol=2.0e-10,
    )
    np.testing.assert_allclose(
        structured.physical_states,
        dense.physical_states,
        rtol=2.0e-10,
        atol=0.0,
    )
    np.testing.assert_allclose(
        structured.interface_states_m3,
        dense.interface_states_m3,
        rtol=2.0e-10,
        atol=0.0,
    )
    np.testing.assert_allclose(
        structured.potentials_V,
        dense.potentials_V,
        rtol=0.0,
        atol=2.0e-12,
    )
    assert structured.total_residual_evaluations < (
        dense.total_residual_evaluations / 20
    )
    assert structured.max_normalized_carrier_residual <= 1.0e-8
    assert structured.max_normalized_positive_ion_residual <= 1.0e-8
    assert structured.max_normalized_interface_residual <= 1.0e-8
    assert structured.max_normalized_algebraic_residual <= 1.0e-8
    assert structured.max_relative_positive_ion_inventory_drift < 2.0e-15


def test_structured_work_is_grid_stable_while_dense_work_grows():
    dense_evaluations = []
    structured_evaluations = []
    for intervals in (2, 4, 8):
        _grid, _stack, _reference, model = _problem(intervals=intervals)
        time = np.array([0.0, 1.0e-9])
        dense = run_ion_interface_backward_euler_reference(
            model,
            time,
            residual_tolerance=1.0e-8,
        )
        structured = run_ion_interface_backward_euler_reference(
            model,
            time,
            residual_tolerance=1.0e-8,
            jacobian_mode="structured_analytic",
        )
        np.testing.assert_allclose(
            structured.coordinates,
            dense.coordinates,
            rtol=0.0,
            atol=2.0e-10,
        )
        dense_evaluations.append(dense.total_residual_evaluations)
        structured_evaluations.append(structured.total_residual_evaluations)

    assert np.all(np.diff(dense_evaluations) > 0)
    assert max(structured_evaluations) - min(structured_evaluations) <= 2
    assert structured_evaluations[-1] < dense_evaluations[-1] / 20


def test_unknown_jacobian_mode_fails_before_integration():
    _grid, _stack, _reference, model = _problem()
    with pytest.raises(ValueError, match="jacobian_mode"):
        run_ion_interface_backward_euler_reference(
            model,
            np.array([0.0, 1.0e-9]),
            jacobian_mode="automatic",
        )


@pytest.mark.parametrize(
    "time",
    [
        np.array([0.0]),
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([-1.0, 0.0]),
        np.array([0.0, np.nan]),
    ],
)
def test_time_grid_validation_fails_closed(time):
    _grid, _stack, _reference, model = _problem()
    with pytest.raises(ValueError, match="time_s"):
        run_ion_interface_backward_euler_reference(model, time)


@pytest.mark.parametrize(
    ("keyword", "value", "match"),
    [
        ("residual_tolerance", 0.0, "residual_tolerance"),
        ("max_newton_iterations", 0, "max_newton_iterations"),
        ("max_line_search_backtracks", -1, "max_line_search_backtracks"),
        ("max_log_density_update", np.inf, "max_log_density_update"),
        ("max_ion_logit_update", 0.0, "max_ion_logit_update"),
        ("max_interface_logit_update", 0.0, "max_interface_logit_update"),
        ("finite_difference_relative_step", np.nan, "finite_difference"),
    ],
)
def test_solver_control_validation_fails_closed(keyword, value, match):
    _grid, _stack, _reference, model = _problem()
    with pytest.raises(ValueError, match=match):
        run_ion_interface_backward_euler_reference(
            model,
            np.array([0.0, 1.0e-9]),
            **{keyword: value},
        )


def test_uncertified_or_foreign_initial_condition_is_rejected():
    _grid, _stack, _reference, model = _problem()
    initial = build_single_ion_algebraic_interface_consistent_initial_condition(model)
    with pytest.raises(ValueError, match="not certified"):
        run_ion_interface_backward_euler_reference(
            model,
            np.array([0.0, 1.0e-9]),
            initial=replace(initial, certified=False),
        )
    changed = np.array(initial.interface_state_m3, copy=True)
    changed[0] *= 1.001
    with pytest.raises(ValueError, match="does not belong"):
        run_ion_interface_backward_euler_reference(
            model,
            np.array([0.0, 1.0e-9]),
            initial=replace(initial, interface_state_m3=changed),
        )


def test_newton_exhaustion_reports_step_time_and_residual():
    _grid, _stack, _reference, model = _problem(V_app_V=0.01)
    with pytest.raises(IonInterfaceDAEIntegrationError) as caught:
        run_ion_interface_backward_euler_reference(
            model,
            np.array([0.0, 1.0e-1]),
            residual_tolerance=1.0e-12,
            max_newton_iterations=1,
            max_line_search_backtracks=0,
        )

    assert caught.value.step_index == 1
    assert caught.value.time_s == pytest.approx(1.0e-1)
    assert np.isfinite(caught.value.residual_norm)
