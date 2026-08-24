from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import perovskite_sim.solver.dae_ion_interface_jacobian as combined_jacobian
from perovskite_sim.solver.dae_ion_interface_integrator import (
    finite_difference_ion_interface_backward_euler_jacobian,
)
from perovskite_sim.solver.dae_ion_interface_jacobian import (
    IonInterfaceJacobianCapabilityError,
    build_ion_interface_structured_backward_euler_jacobian,
    build_ion_interface_structured_state_jacobian,
)
from perovskite_sim.solver.dae_ion_interface_states import (
    build_single_ion_algebraic_interface_consistent_initial_condition,
    finite_difference_state_jacobian,
    project_single_ion_algebraic_interface_state,
)
from tests.unit.solver.test_dae_ion_interface_states import _problem


def _perturbed_operating_point(*, intervals: int = 4, diffusion_only: bool = True):
    grid, _stack, _reference, model = _problem(intervals=intervals)
    if not diffusion_only:
        model = replace(
            model,
            stack=replace(model.stack, ion_steric_diffusion_only=False),
            material=replace(model.material, ion_steric_diffusion_only=False),
        )
    coordinate = np.array(
        build_single_ion_algebraic_interface_consistent_initial_condition(
            model
        ).coordinate,
        copy=True,
    )
    count = model.layout.node_count
    interface = int(model.material.interface_nodes[0])
    coordinate[
        [interface, count + interface, interface - 1, count + interface - 1]
    ] += np.array([0.02, -0.03, 0.04, -0.01])
    coordinate[model.layout.positive_ion_slice] += np.linspace(-0.08, 0.08, count)
    coordinate = project_single_ion_algebraic_interface_state(model, coordinate)
    coordinate[model.layout.interface_slice] += np.array([0.03, -0.02, 0.01, -0.04])
    coordinate[model.layout.potential_slice.start + interface] += 2.0e-4
    return grid, model, coordinate


@pytest.mark.parametrize(
    ("intervals", "diffusion_only"),
    [(2, False), (2, True), (4, False), (4, True), (8, False), (8, True)],
)
def test_combined_structured_state_jacobian_matches_full_central_reference(
    intervals,
    diffusion_only,
):
    grid, model, coordinate = _perturbed_operating_point(
        intervals=intervals,
        diffusion_only=diffusion_only,
    )
    derivative = model.compatible_derivative(coordinate)
    structured = build_ion_interface_structured_state_jacobian(
        model,
        coordinate,
        derivative,
    )
    central = finite_difference_state_jacobian(
        model,
        coordinate,
        derivative,
        relative_step=1.0e-4,
    )
    analytic = structured.matrix.toarray()
    scaled_error = np.abs(analytic - central) / np.maximum(1.0, np.abs(analytic))

    assert isinstance(structured.matrix, csr_matrix)
    assert structured.nonzero_count == structured.matrix.nnz
    assert structured.nonzero_count < 30 * grid.size
    assert structured.ion_steric_diffusion_only is diffusion_only
    assert np.max(scaled_error) < 5.0e-7
    np.testing.assert_array_equal(central[analytic == 0.0], 0.0)


def test_complete_backward_euler_tangent_matches_full_central_reference():
    _grid, model, coordinate = _perturbed_operating_point()
    previous = np.array(coordinate, copy=True)
    count = model.layout.node_count
    previous[1 : count - 1] -= 1.0e-4
    previous[count + 1 : 2 * count - 1] += 2.0e-4
    previous[model.layout.positive_ion_slice] += np.linspace(-2.0e-4, 3.0e-4, count)
    dt_s = 1.0e-3
    structured = build_ion_interface_structured_backward_euler_jacobian(
        model,
        coordinate,
        dt_s,
    )
    central = finite_difference_ion_interface_backward_euler_jacobian(
        model,
        coordinate,
        previous,
        dt_s,
        relative_step=3.0e-5,
    )
    analytic = structured.matrix.toarray()
    scaled_error = np.abs(analytic - central) / np.maximum(1.0, np.abs(analytic))

    assert np.max(scaled_error) < 5.0e-7
    np.testing.assert_array_equal(central[analytic == 0.0], 0.0)


def test_combined_tangent_reports_interface_and_ion_constitutive_evidence():
    grid, model, coordinate = _perturbed_operating_point()
    tangent = build_ion_interface_structured_state_jacobian(
        model,
        coordinate,
        model.compatible_derivative(coordinate),
    )
    interface_face = tangent.local_interface.interface_node - 1

    assert tangent.minimum_bulk_srh_denominator_s_m3 > 0.0
    assert tangent.minimum_positive_ion_occupation_margin > 0.0
    assert tangent.local_interface.minimum_projection_occupation_margin > 0.0
    assert tangent.local_interface.minimum_cross_occupation_margin > 0.0
    assert tangent.local_interface.minimum_srh_occupancy_margin > 0.0
    assert tangent.electron_current_faces_A_m2[interface_face] == 0.0
    assert tangent.hole_current_faces_A_m2[interface_face] == 0.0
    assert tangent.positive_ion_particle_flux_faces_m2_s.shape == (grid.size - 1,)
    assert tangent.ion_steric_diffusion_only is True
    for value in (
        tangent.electron_current_faces_A_m2,
        tangent.hole_current_faces_A_m2,
        tangent.positive_ion_particle_flux_faces_m2_s,
    ):
        assert not value.flags.writeable


def test_ion_rows_couple_neighbor_logits_and_potentials_and_poisson_couples_ion():
    grid, model, coordinate = _perturbed_operating_point()
    tangent = build_ion_interface_structured_state_jacobian(
        model,
        coordinate,
        model.compatible_derivative(coordinate),
    ).matrix.toarray()
    count = grid.size
    ion_rows = tangent[model.layout.positive_ion_slice]

    assert np.all(
        np.count_nonzero(ion_rows[:, model.layout.positive_ion_slice], axis=1) >= 2
    )
    assert np.all(
        np.count_nonzero(ion_rows[:, model.layout.potential_slice], axis=1) >= 2
    )
    assert np.any(
        tangent[
            model.layout.potential_slice.start + 1 : model.layout.potential_slice.stop
            - 1,
            model.layout.positive_ion_slice,
        ]
    )
    assert count == model.layout.node_count


def test_interface_rows_do_not_acquire_direct_ion_columns():
    _grid, model, coordinate = _perturbed_operating_point()
    tangent = build_ion_interface_structured_state_jacobian(
        model,
        coordinate,
        model.compatible_derivative(coordinate),
    ).matrix.toarray()

    np.testing.assert_array_equal(
        tangent[model.layout.interface_slice, model.layout.positive_ion_slice],
        0.0,
    )


def test_structured_builder_does_not_mutate_coordinate_or_derivative():
    _grid, model, coordinate = _perturbed_operating_point()
    derivative = model.compatible_derivative(coordinate)
    coordinate_before = coordinate.copy()
    derivative_before = derivative.copy()

    build_ion_interface_structured_state_jacobian(model, coordinate, derivative)

    np.testing.assert_array_equal(coordinate, coordinate_before)
    np.testing.assert_array_equal(derivative, derivative_before)


def test_steric_clipping_kink_fails_closed(monkeypatch):
    _grid, model, coordinate = _perturbed_operating_point()
    physical_tangent = combined_jacobian.ion_face_flux_jacobian

    def nondifferentiable_tangent(*args, **kwargs):
        tangent = physical_tangent(*args, **kwargs)
        return replace(
            tangent,
            differentiable_faces=np.zeros_like(tangent.differentiable_faces),
        )

    monkeypatch.setattr(
        combined_jacobian,
        "ion_face_flux_jacobian",
        nondifferentiable_tangent,
    )

    with pytest.raises(
        IonInterfaceJacobianCapabilityError,
        match="non-differentiable",
    ):
        build_ion_interface_structured_state_jacobian(
            model,
            coordinate,
            model.compatible_derivative(coordinate),
        )


def test_interface_projection_clamp_fails_closed_through_combined_contract():
    _grid, model, coordinate = _perturbed_operating_point()
    interface = int(model.material.interface_nodes[0])
    potential_offset = model.layout.potential_slice.start
    coordinate[potential_offset + interface] = (
        coordinate[potential_offset + interface - 1] + 30.0 * model.material.V_T_device
    )

    with pytest.raises(
        IonInterfaceJacobianCapabilityError,
        match="projection exponent clamp",
    ):
        build_ion_interface_structured_state_jacobian(
            model,
            coordinate,
            model.compatible_derivative(coordinate),
        )


@pytest.mark.parametrize("dt_s", [0.0, -1.0, np.inf, np.nan])
def test_backward_euler_tangent_rejects_invalid_time_step(dt_s):
    _grid, model, coordinate = _perturbed_operating_point()

    with pytest.raises(ValueError, match="backward_euler_dt_s"):
        build_ion_interface_structured_backward_euler_jacobian(
            model,
            coordinate,
            dt_s,
        )


@pytest.mark.parametrize(
    "derivative",
    [np.zeros(3), np.array([np.nan])],
)
def test_state_tangent_rejects_invalid_derivative(derivative):
    _grid, model, coordinate = _perturbed_operating_point()

    with pytest.raises(ValueError, match="derivative"):
        build_ion_interface_structured_state_jacobian(
            model,
            coordinate,
            derivative,
        )
