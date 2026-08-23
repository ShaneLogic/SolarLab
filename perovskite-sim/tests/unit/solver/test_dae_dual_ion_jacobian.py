from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import perovskite_sim.solver.dae_dual_ion_jacobian as dual_jacobian
from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.solver.dae_dual_ion_integrator import (
    dual_ion_backward_euler_derivative,
    finite_difference_dual_ion_backward_euler_jacobian,
)
from perovskite_sim.solver.dae_dual_ion_jacobian import (
    build_dual_ion_structured_backward_euler_jacobian,
    build_dual_ion_structured_state_jacobian,
)
from perovskite_sim.solver.dae_dual_ions import (
    build_dual_ion_dae,
    finite_difference_dual_ion_state_jacobian,
    project_dual_ion_algebraic_state,
)
from perovskite_sim.solver.dae_jacobian import (
    DAEStructuredJacobianCapabilityError,
)
from perovskite_sim.solver.newton import solve_equilibrium


def _model(
    *,
    intervals: int = 6,
    diffusion_only: bool = True,
    shared_site: bool = True,
):
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
        ion_steric_diffusion_only=diffusion_only,
        ion_steric_shared_site=shared_site,
        mode="full",
    )
    grid = multilayer_grid([Layer(dual_layer.thickness, intervals)], alpha=1.0)
    model = build_dual_ion_dae(
        grid,
        stack,
        solve_equilibrium(grid, stack),
        V_app_V=0.01,
        illuminated=True,
        carrier_reference_time_s=1.0e-9,
        ion_reference_time_s=1.0,
    )
    coordinate = np.zeros(model.layout.size)
    count = grid.size
    coordinate[1 : count - 1] = np.linspace(-0.02, 0.03, count - 2)
    coordinate[count + 1 : 2 * count - 1] = np.linspace(0.01, -0.02, count - 2)
    coordinate[model.layout.positive_ion_slice] = np.linspace(-0.08, 0.1, count)
    coordinate[model.layout.negative_ion_slice] = np.linspace(0.06, -0.09, count)
    coordinate = project_dual_ion_algebraic_state(model, coordinate)
    return grid, model, coordinate


def test_shared_site_coordinate_hessian_matches_mass_matrix_stencil():
    _grid, model, coordinate = _model()
    analytic = model.ion_coordinate_hessian_m3(coordinate)
    count = model.layout.node_count
    offsets = (2 * count, 3 * count)
    step = 2.0e-6
    finite_difference = np.zeros_like(analytic)
    for node in range(count):
        for coordinate_species, offset in enumerate(offsets):
            plus = np.array(coordinate, copy=True)
            minus = np.array(coordinate, copy=True)
            plus[offset + node] += step
            minus[offset + node] -= step
            finite_difference[node, :, :, coordinate_species] = (
                model.ion_coordinate_jacobian_m3(plus)[node]
                - model.ion_coordinate_jacobian_m3(minus)[node]
            ) / (2.0 * step)

    np.testing.assert_allclose(
        analytic,
        finite_difference,
        rtol=6.0e-9,
        atol=2.0e9,
    )
    np.testing.assert_allclose(
        analytic,
        np.swapaxes(analytic, 2, 3),
        rtol=0.0,
        atol=2.0e9,
    )


@pytest.mark.parametrize(
    ("diffusion_only", "shared_site"),
    [(True, True), (True, False), (False, True)],
    ids=["shared-pnp", "distinct-pnp", "legacy-whole-flux"],
)
def test_structured_dual_ion_state_jacobian_matches_central_difference(
    diffusion_only,
    shared_site,
):
    grid, model, coordinate = _model(
        diffusion_only=diffusion_only,
        shared_site=shared_site,
    )
    derivative = model.compatible_derivative(coordinate)
    structured = build_dual_ion_structured_state_jacobian(
        model,
        coordinate,
        derivative,
    )
    central = finite_difference_dual_ion_state_jacobian(
        model,
        coordinate,
        derivative,
        relative_step=3.0e-4,
    )
    analytic = structured.matrix.toarray()
    difference = np.abs(analytic - central)
    column_scale = np.maximum(np.max(np.abs(central), axis=0), 1.0e-12)
    normalized = np.max(difference, axis=0) / column_scale

    assert structured.storage_mode == "dae_state"
    assert structured.shared_site is (diffusion_only and shared_site)
    assert structured.nonzero_count < 45 * grid.size
    assert structured.minimum_bulk_srh_denominator_s_m3 > 0.0
    assert structured.positive_ion_particle_flux_faces_m2_s.shape == (grid.size - 1,)
    assert structured.negative_ion_particle_flux_faces_m2_s.shape == (grid.size - 1,)
    assert float(np.max(normalized)) < 1.5e-5
    np.testing.assert_allclose(analytic, central, rtol=2.5e-5, atol=4.0e-8)


@pytest.mark.parametrize("shared_site", [True, False], ids=["shared", "distinct"])
def test_structured_backward_euler_jacobian_matches_complete_stencil(shared_site):
    _grid, model, previous = _model(shared_site=shared_site)
    coordinate = np.array(previous, copy=True)
    coordinate[model.layout.positive_ion_slice] += 0.015
    coordinate[model.layout.negative_ion_slice] -= 0.012
    coordinate = project_dual_ion_algebraic_state(model, coordinate)
    dt_s = 2.0e-3
    structured = build_dual_ion_structured_backward_euler_jacobian(
        model,
        coordinate,
        dt_s,
    )
    central = finite_difference_dual_ion_backward_euler_jacobian(
        model,
        coordinate,
        previous,
        dt_s,
        relative_step=3.0e-4,
    )
    analytic = structured.matrix.toarray()
    difference = np.abs(analytic - central)
    column_scale = np.maximum(np.max(np.abs(central), axis=0), 1.0e-12)
    normalized = np.max(difference, axis=0) / column_scale

    assert structured.storage_mode == "physical_density_backward_euler"
    assert float(np.max(normalized)) < 1.5e-5
    np.testing.assert_allclose(analytic, central, rtol=2.5e-5, atol=4.0e-8)
    assert np.all(
        np.isfinite(
            dual_ion_backward_euler_derivative(
                model,
                coordinate,
                previous,
                dt_s,
            )
        )
    )


def test_shared_site_ion_rows_include_partner_coordinate_coupling():
    grid, model, coordinate = _model()
    tangent = build_dual_ion_structured_state_jacobian(
        model,
        coordinate,
        model.compatible_derivative(coordinate),
    ).matrix.toarray()
    count = grid.size
    positive_rows = tangent[2 * count : 3 * count]
    negative_rows = tangent[3 * count : 4 * count]

    assert np.any(positive_rows[:, 3 * count : 4 * count] != 0.0)
    assert np.any(negative_rows[:, 2 * count : 3 * count] != 0.0)
    assert np.all(
        np.count_nonzero(positive_rows[:, 2 * count : 4 * count], axis=1) >= 4
    )
    assert np.all(
        np.count_nonzero(negative_rows[:, 2 * count : 4 * count], axis=1) >= 4
    )


def test_distinct_sublattice_ion_continuity_rows_have_no_partner_columns():
    grid, model, coordinate = _model(shared_site=False)
    tangent = build_dual_ion_structured_state_jacobian(
        model,
        coordinate,
        model.compatible_derivative(coordinate),
    ).matrix.toarray()
    count = grid.size

    np.testing.assert_array_equal(
        tangent[2 * count : 3 * count, 3 * count : 4 * count],
        0.0,
    )
    np.testing.assert_array_equal(
        tangent[3 * count : 4 * count, 2 * count : 3 * count],
        0.0,
    )


def test_either_species_steric_clipping_kink_fails_closed(monkeypatch):
    _grid, model, coordinate = _model()
    physical_tangent = dual_jacobian.ion_face_flux_jacobian
    calls = 0

    def second_species_nondifferentiable(*args, **kwargs):
        nonlocal calls
        calls += 1
        tangent = physical_tangent(*args, **kwargs)
        if calls == 2:
            return replace(
                tangent,
                differentiable_faces=np.zeros_like(tangent.differentiable_faces),
            )
        return tangent

    monkeypatch.setattr(
        dual_jacobian,
        "ion_face_flux_jacobian",
        second_species_nondifferentiable,
    )

    with pytest.raises(
        DAEStructuredJacobianCapabilityError,
        match="negative-ion.*non-differentiable",
    ):
        build_dual_ion_structured_state_jacobian(
            model,
            coordinate,
            model.compatible_derivative(coordinate),
        )
