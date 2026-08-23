"""Deterministic work-scaling gates for the first structured DAE slice."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.solver.dae import (
    build_consistent_initial_condition,
    build_no_ion_no_interface_dae,
)
from perovskite_sim.solver.dae_integrator import run_backward_euler_reference
from perovskite_sim.solver.dae_jacobian import build_structured_state_jacobian
from perovskite_sim.solver.newton import solve_equilibrium


def _stack():
    source = load_device_from_yaml("configs/csi_vannijen2025_pn_cv.yaml")
    source_layer = source.layers[1]
    assert source_layer.params is not None
    layer = replace(
        source_layer,
        params=replace(source_layer.params, alpha=2.0e4),
    )
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
    return layer, stack


def test_structured_lane_preserves_trajectory_with_linear_matrix_storage():
    layer, stack = _stack()
    rows = []
    for intervals in (8, 16, 32):
        grid = multilayer_grid([Layer(layer.thickness, intervals)], alpha=1.0)
        reference = solve_equilibrium(grid, stack)
        model = build_no_ion_no_interface_dae(
            grid,
            stack,
            reference,
            illuminated=True,
            reference_time_s=1.0e-9,
        )
        initial = build_consistent_initial_condition(model)
        time = np.array([0.0, 2.5e-10])
        dense = run_backward_euler_reference(model, time, initial=initial)
        structured = run_backward_euler_reference(
            model,
            time,
            initial=initial,
            jacobian_mode="structured_analytic",
        )
        tangent = build_structured_state_jacobian(
            model,
            initial.coordinate,
            initial.derivative,
        )

        np.testing.assert_allclose(
            structured.physical_states,
            dense.physical_states,
            rtol=1.0e-14,
            atol=0.0,
        )
        np.testing.assert_allclose(
            structured.potentials_V,
            dense.potentials_V,
            rtol=0.0,
            atol=3.0e-15,
        )
        assert tangent.nonzero_count < 20 * grid.size
        rows.append(
            (
                grid.size,
                tangent.nonzero_count,
                dense.total_residual_evaluations,
                structured.total_residual_evaluations,
            )
        )

    nodes = np.asarray([row[0] for row in rows])
    nonzeros = np.asarray([row[1] for row in rows])
    dense_work = np.asarray([row[2] for row in rows])
    structured_work = np.asarray([row[3] for row in rows])
    assert np.all(np.diff(nonzeros) > 0)
    np.testing.assert_array_equal(nonzeros, 19 * nodes - 36)
    assert structured_work[-1] <= 6
    assert dense_work[-1] / structured_work[-1] > 150.0
    assert dense_work[-1] / dense_work[0] > 2.0
    assert structured_work[-1] / structured_work[0] < 1.5
