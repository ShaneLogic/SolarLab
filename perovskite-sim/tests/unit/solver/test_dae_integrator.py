from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.solver.dae import (
    build_consistent_initial_condition,
    build_no_ion_no_interface_dae,
)
from perovskite_sim.solver.dae_integrator import (
    DAEIntegrationError,
    run_backward_euler_reference,
)
from perovskite_sim.solver.newton import solve_equilibrium


def _homogeneous_model(*, illuminated: bool = False):
    source = load_device_from_yaml("configs/csi_vannijen2025_pn_cv.yaml")
    source_layer = source.layers[1]
    params = source_layer.params
    if illuminated:
        assert params is not None
        params = replace(params, alpha=2.0e4)
    layer = replace(source_layer, params=params)
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
    model = build_no_ion_no_interface_dae(
        grid,
        stack,
        reference,
        illuminated=illuminated,
        reference_time_s=1.0e-9,
    )
    return grid, stack, reference, model


def test_dark_equilibrium_remains_stationary_and_reproducible():
    _grid, _stack, _reference, model = _homogeneous_model()
    time = np.linspace(0.0, 1.0e-8, 3)

    first = run_backward_euler_reference(model, time)
    second = run_backward_euler_reference(model, time)

    assert first.success
    assert first.trajectory_sha256 == second.trajectory_sha256
    np.testing.assert_array_equal(first.coordinates, second.coordinates)
    np.testing.assert_array_equal(first.physical_states, second.physical_states)
    np.testing.assert_array_equal(first.potentials_V, second.potentials_V)
    np.testing.assert_allclose(
        first.physical_states,
        np.broadcast_to(first.physical_states[0], first.physical_states.shape),
        rtol=2.0e-14,
        atol=0.0,
    )
    assert not first.time_s.flags.writeable
    assert not first.coordinates.flags.writeable
    assert not first.physical_states.flags.writeable
    assert not first.potentials_V.flags.writeable


def test_illuminated_steps_are_residual_and_balance_certified():
    _grid, _stack, _reference, model = _homogeneous_model(illuminated=True)
    time = np.linspace(0.0, 1.0e-9, 5)

    result = run_backward_euler_reference(
        model,
        time,
        residual_tolerance=1.0e-11,
    )

    assert result.success
    assert len(result.step_reports) == 4
    assert result.total_nonlinear_iterations > 0
    assert result.total_jacobian_evaluations > 0
    assert result.total_residual_evaluations > result.total_jacobian_evaluations
    assert result.max_normalized_differential_residual <= 1.0e-11
    assert result.max_normalized_algebraic_residual <= 1.0e-11
    # The electron balance reaches its absolute floating-point floor when
    # subtracting O(n/dt) terms; the independently reported current defect
    # remains below the declared reference gate.
    assert result.max_electron_balance_defect_A_m2 < 1.0e-8
    assert result.max_hole_balance_defect_A_m2 < 1.0e-8
    assert all(report.dt_s == pytest.approx(2.5e-10) for report in result.step_reports)
    assert all(
        np.isfinite(report.max_scaled_jacobian_condition)
        for report in result.step_reports
    )
    count = model.layout.node_count
    assert np.all(result.physical_states[:, : 2 * count] > 0.0)
    assert np.any(result.physical_states[-1] != result.physical_states[0])


def test_structured_newton_matches_dense_reference_with_less_rhs_work():
    _grid, _stack, _reference, model = _homogeneous_model(illuminated=True)
    time = np.linspace(0.0, 1.0e-9, 5)

    dense = run_backward_euler_reference(model, time)
    structured = run_backward_euler_reference(
        model,
        time,
        jacobian_mode="structured_analytic",
    )

    assert dense.jacobian_mode == "dense_central"
    assert structured.jacobian_mode == "structured_analytic"
    assert "sparse-analytic" in structured.method
    np.testing.assert_allclose(
        structured.physical_states,
        dense.physical_states,
        rtol=2.0e-11,
        atol=0.0,
    )
    np.testing.assert_allclose(
        structured.potentials_V,
        dense.potentials_V,
        rtol=0.0,
        atol=2.0e-14,
    )
    assert structured.total_residual_evaluations < (
        dense.total_residual_evaluations / 20
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
    _grid, _stack, _reference, model = _homogeneous_model()
    with pytest.raises(ValueError, match="time_s"):
        run_backward_euler_reference(model, time)


@pytest.mark.parametrize(
    ("keyword", "value", "match"),
    [
        ("residual_tolerance", 0.0, "residual_tolerance"),
        ("max_newton_iterations", 0, "max_newton_iterations"),
        ("max_line_search_backtracks", -1, "max_line_search_backtracks"),
        ("max_log_density_update", np.inf, "max_log_density_update"),
        ("finite_difference_relative_step", np.nan, "finite_difference"),
        ("jacobian_mode", "unsupported", "jacobian_mode"),
    ],
)
def test_solver_control_validation_fails_closed(keyword, value, match):
    _grid, _stack, _reference, model = _homogeneous_model()
    with pytest.raises(ValueError, match=match):
        run_backward_euler_reference(
            model,
            np.array([0.0, 1.0e-9]),
            **{keyword: value},
        )


def test_uncertified_initial_condition_is_rejected():
    _grid, _stack, _reference, model = _homogeneous_model()
    initial = build_consistent_initial_condition(model)
    invalid = replace(initial, certified=False)

    with pytest.raises(ValueError, match="not certified"):
        run_backward_euler_reference(
            model,
            np.array([0.0, 1.0e-9]),
            initial=invalid,
        )


def test_newton_exhaustion_reports_step_and_time():
    _grid, _stack, _reference, model = _homogeneous_model(illuminated=True)

    with pytest.raises(DAEIntegrationError) as caught:
        run_backward_euler_reference(
            model,
            np.array([0.0, 1.0e-6]),
            residual_tolerance=1.0e-14,
            max_newton_iterations=1,
            max_line_search_backtracks=0,
        )

    assert caught.value.step_index == 1
    assert caught.value.time_s == pytest.approx(1.0e-6)
    assert np.isfinite(caught.value.residual_norm)
