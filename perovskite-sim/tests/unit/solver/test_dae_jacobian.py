from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.solver.dae import (
    build_consistent_initial_condition,
    build_no_ion_no_interface_dae,
    finite_difference_state_jacobian,
    project_algebraic_state,
)
from perovskite_sim.solver.dae_jacobian import (
    DAEStructuredJacobianCapabilityError,
    build_structured_state_jacobian,
)
from perovskite_sim.solver.newton import solve_equilibrium


def _model(*, field_mobility: bool = False):
    source = load_device_from_yaml("configs/csi_vannijen2025_pn_cv.yaml")
    source_layer = source.layers[1]
    params = source_layer.params
    assert params is not None
    if field_mobility:
        params = replace(
            params,
            v_sat_n=1.0e5,
            v_sat_p=8.0e4,
            ct_beta_n=2.0,
            ct_beta_p=2.0,
        )
    layer = replace(source_layer, params=params)
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
    model = build_no_ion_no_interface_dae(
        grid,
        stack,
        reference,
        V_app_V=0.02,
        reference_time_s=1.0e-8,
    )
    initial = build_consistent_initial_condition(model)
    coordinate = np.array(initial.coordinate, copy=True)
    count = grid.size
    coordinate[1 : count - 1] += np.linspace(-0.08, 0.12, count - 2)
    coordinate[count + 1 : 2 * count - 1] += np.linspace(0.09, -0.05, count - 2)
    coordinate = project_algebraic_state(model, coordinate)
    return model, initial, coordinate


@pytest.mark.parametrize("field_mobility", [False, True])
def test_structured_state_jacobian_matches_central_reference(field_mobility):
    model, initial, coordinate = _model(field_mobility=field_mobility)
    structured = build_structured_state_jacobian(
        model,
        coordinate,
        initial.derivative,
    )
    central = finite_difference_state_jacobian(
        model,
        coordinate,
        initial.derivative,
        relative_step=2.0e-6,
    )
    analytic = structured.matrix.toarray()
    difference = np.abs(analytic - central)
    column_scale = np.maximum(np.max(np.abs(central), axis=0), 1.0e-12)
    normalized = np.max(difference, axis=0) / column_scale

    assert structured.field_mobility_active is field_mobility
    assert structured.nonzero_count < 16 * model.layout.node_count
    assert structured.minimum_bulk_srh_denominator_s_m3 > 0.0
    assert float(np.max(normalized)) < 1.2e-5
    np.testing.assert_allclose(analytic, central, rtol=2.0e-5, atol=2.0e-8)


def test_boundary_carrier_columns_do_not_enter_differential_rows():
    model, initial, coordinate = _model()
    matrix = build_structured_state_jacobian(
        model,
        coordinate,
        initial.derivative,
    ).matrix.toarray()
    count = model.layout.node_count
    boundary_columns = (0, count - 1, count, 2 * count - 1)

    for column in boundary_columns:
        rows = np.flatnonzero(model.layout.differential_mask)
        np.testing.assert_array_equal(matrix[rows, column], np.zeros(rows.size))


def test_nonlocal_photon_recycling_is_rejected():
    model, initial, coordinate = _model()
    material = replace(model.material, has_radiative_reabsorption=True)
    unsupported = replace(model, material=material)

    with pytest.raises(
        DAEStructuredJacobianCapabilityError,
        match="radiative reabsorption",
    ):
        build_structured_state_jacobian(
            unsupported,
            coordinate,
            initial.derivative,
        )


def test_zero_field_pf_cusp_is_rejected():
    source = load_device_from_yaml("configs/csi_vannijen2025_pn_cv.yaml")
    source_layer = source.layers[1]
    assert source_layer.params is not None
    layer = replace(
        source_layer,
        params=replace(source_layer.params, pf_gamma_n=3.0e-4),
    )
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
    model = build_no_ion_no_interface_dae(grid, stack, reference)
    initial = build_consistent_initial_condition(model)

    with pytest.raises(
        DAEStructuredJacobianCapabilityError,
        match="non-differentiable",
    ):
        build_structured_state_jacobian(
            model,
            initial.coordinate,
            initial.derivative,
        )
