from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import perovskite_sim.solver.dae_ion_jacobian as dae_ion_jacobian
from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.solver.dae_ion_jacobian import (
    build_single_ion_structured_state_jacobian,
)
from perovskite_sim.solver.dae_ions import (
    build_single_positive_ion_dae,
    finite_difference_single_ion_state_jacobian,
    project_single_ion_algebraic_state,
)
from perovskite_sim.solver.dae_jacobian import (
    DAEStructuredJacobianCapabilityError,
)
from perovskite_sim.solver.newton import solve_equilibrium


def _model(*, intervals: int = 6, diffusion_only: bool = True):
    source = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    layer = source.layers[1]
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
        ion_steric_diffusion_only=diffusion_only,
    )
    grid = multilayer_grid([Layer(layer.thickness, intervals)], alpha=1.0)
    model = build_single_positive_ion_dae(
        grid,
        stack,
        solve_equilibrium(grid, stack),
        illuminated=True,
        carrier_reference_time_s=1.0e-9,
        ion_reference_time_s=1.0,
    )
    return grid, model


@pytest.mark.parametrize("diffusion_only", [False, True])
def test_structured_single_ion_state_jacobian_matches_central_difference(
    diffusion_only,
):
    grid, model = _model(diffusion_only=diffusion_only)
    coordinate = np.zeros(model.layout.size)
    count = grid.size
    coordinate[1 : count - 1] = np.linspace(-0.02, 0.03, count - 2)
    coordinate[count + 1 : 2 * count - 1] = np.linspace(0.01, -0.02, count - 2)
    coordinate[model.layout.positive_ion_slice] = np.linspace(-0.1, 0.1, count)
    coordinate = project_single_ion_algebraic_state(model, coordinate)
    derivative = model.compatible_derivative(coordinate)

    structured = build_single_ion_structured_state_jacobian(
        model,
        coordinate,
        derivative,
    )
    finite_difference = finite_difference_single_ion_state_jacobian(
        model,
        coordinate,
        derivative,
        relative_step=3.0e-4,
    )

    np.testing.assert_allclose(
        structured.matrix.toarray(),
        finite_difference,
        rtol=4.0e-6,
        atol=3.0e-8,
    )
    assert structured.nonzero_count < 28 * count
    assert structured.ion_steric_diffusion_only is diffusion_only
    assert structured.minimum_bulk_srh_denominator_s_m3 > 0.0
    assert structured.positive_ion_particle_flux_faces_m2_s.shape == (count - 1,)


def test_ion_rows_couple_neighbor_logit_and_potential_coordinates():
    grid, model = _model()
    coordinate = np.zeros(model.layout.size)
    coordinate[model.layout.positive_ion_slice] = np.linspace(-0.05, 0.05, grid.size)
    coordinate = project_single_ion_algebraic_state(model, coordinate)
    tangent = build_single_ion_structured_state_jacobian(
        model,
        coordinate,
        model.compatible_derivative(coordinate),
    ).matrix.toarray()
    count = grid.size
    ion_rows = tangent[2 * count : 3 * count]

    assert np.all(np.count_nonzero(ion_rows[:, 2 * count : 3 * count], axis=1) >= 2)
    assert np.all(np.count_nonzero(ion_rows[:, 3 * count : 4 * count], axis=1) >= 2)
    assert np.any(tangent[3 * count + 1 : 4 * count - 1, 2 * count : 3 * count])


def test_steric_clipping_kink_fails_closed(monkeypatch):
    grid, model = _model(diffusion_only=True)
    coordinate = np.zeros(model.layout.size)
    coordinate = project_single_ion_algebraic_state(model, coordinate)
    physical_tangent = dae_ion_jacobian.ion_face_flux_jacobian

    def nondifferentiable_tangent(*args, **kwargs):
        tangent = physical_tangent(*args, **kwargs)
        return replace(
            tangent,
            differentiable_faces=np.zeros_like(tangent.differentiable_faces),
        )

    monkeypatch.setattr(
        dae_ion_jacobian,
        "ion_face_flux_jacobian",
        nondifferentiable_tangent,
    )

    with pytest.raises(
        DAEStructuredJacobianCapabilityError,
        match="non-differentiable",
    ):
        build_single_ion_structured_state_jacobian(
            model,
            coordinate,
            model.compatible_derivative(coordinate),
        )
    assert grid.size == model.layout.node_count
