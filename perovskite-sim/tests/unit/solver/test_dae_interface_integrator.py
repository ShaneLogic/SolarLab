from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.models.device import DeviceStack, LayerSpec
from perovskite_sim.models.parameters import MaterialParams
from perovskite_sim.solver.dae_interface_integrator import (
    AlgebraicInterfaceDAEIntegrationError,
    _backward_euler_derivative,
    run_algebraic_interface_backward_euler_reference,
)
from perovskite_sim.solver.dae_interface_jacobian import (
    build_algebraic_interface_structured_backward_euler_jacobian,
)
from perovskite_sim.solver.dae_interface_states import (
    build_algebraic_interface_consistent_initial_condition,
    build_algebraic_interface_state_dae,
)
from perovskite_sim.solver.mol import StateVec
from perovskite_sim.solver.newton import solve_equilibrium


def _model(*, intervals: int = 4, V_app_V: float = 0.01):
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
    stack = DeviceStack(
        layers=(
            LayerSpec("left", 1.0e-7, left, role="absorber"),
            LayerSpec("right", 1.0e-7, replace(left, chi=4.1), role="ETL"),
        ),
        interfaces=((0.03, 0.05),),
        V_bi=0.0,
        Phi=0.0,
        mode="full",
    )
    grid = multilayer_grid(
        [Layer(layer.thickness, intervals) for layer in stack.layers],
        alpha=1.0,
    )
    return build_algebraic_interface_state_dae(
        grid,
        stack,
        solve_equilibrium(grid, stack),
        V_app_V=V_app_V,
        carrier_reference_time_s=1.0e-7,
    )


def test_dense_backward_euler_is_reproducible_and_residual_certified():
    model = _model()
    time = np.linspace(0.0, 1.0e-9, 3)

    first = run_algebraic_interface_backward_euler_reference(model, time)
    second = run_algebraic_interface_backward_euler_reference(model, time)

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


def test_physical_density_storage_is_exact_backward_euler_difference():
    model = _model()
    time = np.linspace(0.0, 1.0e-9, 3)
    result = run_algebraic_interface_backward_euler_reference(model, time)
    count = model.layout.node_count

    for index, report in enumerate(result.step_reports, start=1):
        current = result.coordinates[index]
        previous = result.coordinates[index - 1]
        derivative = _backward_euler_derivative(
            model,
            current,
            previous,
            report.dt_s,
        )
        current_state = StateVec.unpack(result.physical_states[index], count)
        previous_state = StateVec.unpack(
            result.physical_states[index - 1],
            count,
        )
        np.testing.assert_allclose(
            current_state.n[1:-1] * derivative[1 : count - 1],
            (current_state.n[1:-1] - previous_state.n[1:-1]) / report.dt_s,
            rtol=3.0e-14,
            atol=0.0,
        )
        np.testing.assert_allclose(
            current_state.p[1:-1]
            * derivative[count + 1 : 2 * count - 1],
            (current_state.p[1:-1] - previous_state.p[1:-1]) / report.dt_s,
            rtol=3.0e-14,
            atol=0.0,
        )
        assert np.all(derivative[model.layout.interface_slice] == 0.0)
        assert np.all(derivative[model.layout.potential_slice] == 0.0)


def test_every_accepted_interface_state_stays_on_bounded_algebraic_manifold():
    model = _model()
    result = run_algebraic_interface_backward_euler_reference(
        model,
        np.linspace(0.0, 2.0e-9, 5),
    )

    assert np.all(result.interface_states_m3 > 0.0)
    assert np.all(
        result.interface_states_m3
        < model.layout.interface_capacity_m3[np.newaxis, :]
    )
    assert result.max_interface_state_balance_m2_s > 0.0
    assert result.max_interface_state_balance_m2_s < 1.0e3
    assert all(
        report.residual_report.max_normalized_interface_residual <= 1.0e-9
        for report in result.step_reports
    )
    assert all(
        np.isfinite(report.max_scaled_jacobian_condition)
        for report in result.step_reports
    )


def test_trajectory_contains_nontrivial_carrier_and_interface_response():
    model = _model()
    result = run_algebraic_interface_backward_euler_reference(
        model,
        np.linspace(0.0, 1.0e-9, 3),
    )

    assert np.max(
        np.abs(
            np.log(
                result.physical_states[-1, : 2 * model.layout.node_count]
                / result.physical_states[0, : 2 * model.layout.node_count]
            )
        )
    ) > 1.0e-5
    assert np.max(
        np.abs(
            result.interface_states_m3[-1] / result.interface_states_m3[0]
            - 1.0
        )
    ) > 1.0e-5


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
    with pytest.raises(ValueError, match="time_s"):
        run_algebraic_interface_backward_euler_reference(_model(), time)


@pytest.mark.parametrize(
    ("keyword", "value", "match"),
    [
        ("residual_tolerance", 0.0, "residual_tolerance"),
        ("max_newton_iterations", 0, "max_newton_iterations"),
        ("max_line_search_backtracks", -1, "max_line_search_backtracks"),
        ("max_log_density_update", np.inf, "max_log_density_update"),
        ("max_interface_logit_update", 0.0, "max_interface_logit_update"),
        ("finite_difference_relative_step", np.nan, "finite_difference"),
        ("jacobian_mode", "automatic", "jacobian_mode"),
    ],
)
def test_solver_control_validation_fails_closed(keyword, value, match):
    with pytest.raises(ValueError, match=match):
        run_algebraic_interface_backward_euler_reference(
            _model(),
            np.array([0.0, 1.0e-9]),
            **{keyword: value},
        )


def test_uncertified_or_foreign_initial_condition_is_rejected():
    model = _model()
    initial = build_algebraic_interface_consistent_initial_condition(model)
    with pytest.raises(ValueError, match="not certified"):
        run_algebraic_interface_backward_euler_reference(
            model,
            np.array([0.0, 1.0e-9]),
            initial=replace(initial, certified=False),
        )
    changed = np.array(initial.interface_state_m3, copy=True)
    changed[0] *= 1.001
    with pytest.raises(ValueError, match="does not belong"):
        run_algebraic_interface_backward_euler_reference(
            model,
            np.array([0.0, 1.0e-9]),
            initial=replace(initial, interface_state_m3=changed),
        )


def test_newton_exhaustion_reports_step_time_and_residual():
    model = _model()
    with pytest.raises(AlgebraicInterfaceDAEIntegrationError) as caught:
        run_algebraic_interface_backward_euler_reference(
            model,
            np.array([0.0, 1.0e-7]),
            max_newton_iterations=1,
            max_line_search_backtracks=0,
        )

    assert caught.value.step_index == 1
    assert caught.value.time_s == pytest.approx(1.0e-7)
    assert np.isfinite(caught.value.residual_norm)


def test_structured_newton_matches_dense_trajectory_with_far_fewer_residuals():
    model = _model()
    time = np.linspace(0.0, 2.0e-9, 5)
    dense = run_algebraic_interface_backward_euler_reference(
        model,
        time,
        jacobian_mode="dense_central",
    )
    structured = run_algebraic_interface_backward_euler_reference(
        model,
        time,
        jacobian_mode="structured_analytic",
    )

    assert dense.success and structured.success
    assert structured.jacobian_mode == "structured_analytic"
    assert structured.method.endswith("sparse-analytic-newton-v1")
    np.testing.assert_allclose(
        structured.coordinates,
        dense.coordinates,
        rtol=0.0,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        structured.physical_states,
        dense.physical_states,
        rtol=3.0e-13,
        atol=0.0,
    )
    np.testing.assert_allclose(
        structured.interface_states_m3,
        dense.interface_states_m3,
        rtol=3.0e-13,
        atol=0.0,
    )
    np.testing.assert_allclose(
        structured.potentials_V,
        dense.potentials_V,
        rtol=0.0,
        atol=2.0e-16,
    )
    assert structured.total_residual_evaluations < (
        dense.total_residual_evaluations / 20
    )
    assert structured.max_normalized_carrier_residual <= 1.0e-9
    assert structured.max_normalized_interface_residual <= 1.0e-9
    assert all(
        np.isfinite(report.max_scaled_jacobian_condition)
        for report in structured.step_reports
    )


def test_structured_work_is_linear_and_residual_evaluations_are_grid_stable():
    node_counts = []
    layout_sizes = []
    nonzero_counts = []
    structured_evaluations = []
    dense_evaluations = []
    for intervals in (4, 8, 16):
        model = _model(intervals=intervals)
        initial = build_algebraic_interface_consistent_initial_condition(
            model,
            residual_tolerance=1.0e-8,
        )
        tangent = build_algebraic_interface_structured_backward_euler_jacobian(
            model,
            initial.coordinate,
            1.0e-9,
        )
        time = np.array([0.0, 1.0e-10])
        structured = run_algebraic_interface_backward_euler_reference(
            model,
            time,
            initial=initial,
            residual_tolerance=1.0e-8,
            jacobian_mode="structured_analytic",
        )
        dense = run_algebraic_interface_backward_euler_reference(
            model,
            time,
            initial=initial,
            residual_tolerance=1.0e-8,
            jacobian_mode="dense_central",
        )
        node_counts.append(model.layout.node_count)
        layout_sizes.append(model.layout.size)
        nonzero_counts.append(tangent.nonzero_count)
        structured_evaluations.append(structured.total_residual_evaluations)
        dense_evaluations.append(dense.total_residual_evaluations)

    assert node_counts == [9, 17, 33]
    assert nonzero_counts == [157, 309, 613]
    np.testing.assert_array_equal(
        np.diff(nonzero_counts),
        19 * np.diff(node_counts),
    )
    assert all(
        nonzero_count < 6 * layout_size
        for nonzero_count, layout_size in zip(nonzero_counts, layout_sizes)
    )
    assert max(structured_evaluations) - min(structured_evaluations) <= 1
    assert dense_evaluations[0] < dense_evaluations[1] < dense_evaluations[2]
    assert structured_evaluations[-1] < dense_evaluations[-1] / 100
