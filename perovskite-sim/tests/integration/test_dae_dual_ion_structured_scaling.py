"""Work-scaling gates for the structured dual-mobile-ion DAE slice."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.solver.dae_dual_ion_integrator import (
    run_dual_ion_backward_euler_reference,
)
from perovskite_sim.solver.dae_dual_ion_jacobian import (
    build_dual_ion_structured_backward_euler_jacobian,
)
from perovskite_sim.solver.dae_dual_ions import build_dual_ion_dae
from perovskite_sim.solver.newton import solve_equilibrium


def _stack():
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
        Phi=0.0,
        interfaces=(),
        interface_defects=(),
        grid_interval_weights=(),
        grid_alphas=(),
        ion_steric_diffusion_only=True,
        ion_steric_shared_site=True,
        mode="full",
    )
    return dual_layer, stack


def test_structured_dual_ion_lane_has_linear_storage_and_constant_rhs_work():
    layer, stack = _stack()
    rows = []
    for intervals in (8, 16, 32):
        grid = multilayer_grid([Layer(layer.thickness, intervals)], alpha=1.0)
        model = build_dual_ion_dae(
            grid,
            stack,
            solve_equilibrium(grid, stack),
            V_app_V=0.01,
            carrier_reference_time_s=1.0e-9,
            ion_reference_time_s=1.0,
        )
        time = np.array([0.0, 5.0e-3])
        dense = run_dual_ion_backward_euler_reference(
            model,
            time,
            max_newton_iterations=24,
        )
        structured = run_dual_ion_backward_euler_reference(
            model,
            time,
            max_newton_iterations=24,
            jacobian_mode="structured_analytic",
        )
        tangent = build_dual_ion_structured_backward_euler_jacobian(
            model,
            structured.coordinates[0],
            time[-1] - time[0],
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
    np.testing.assert_array_equal(nonzeros, 39 * nodes - 52)
    np.testing.assert_array_equal(dense_work, 20 * nodes + 3)
    np.testing.assert_array_equal(structured_work, 3)
    assert dense_work[-1] / structured_work[-1] > 200.0
    assert dense_work[-1] / dense_work[0] > 3.5
    assert structured_work[-1] / structured_work[0] == 1.0
