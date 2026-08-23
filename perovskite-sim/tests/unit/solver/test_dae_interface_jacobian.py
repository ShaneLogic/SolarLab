from __future__ import annotations

import numpy as np
import pytest

from perovskite_sim.physics.interface_plane import (
    compute_interface_srh_occupancy_on_state,
)
from perovskite_sim.solver.dae_interface_jacobian import (
    AlgebraicInterfaceJacobianCapabilityError,
    linearize_algebraic_interface_response,
)
from perovskite_sim.solver.dae_interface_states import (
    build_algebraic_interface_consistent_initial_condition,
)
from tests.unit.solver.test_dae_interface_integrator import _model


def _perturbed_operating_point():
    model = _model()
    coordinate = np.array(
        build_algebraic_interface_consistent_initial_condition(model).coordinate,
        copy=True,
    )
    count = model.layout.node_count
    interface = int(model.material.interface_nodes[0])
    coordinate[
        [interface, count + interface, interface - 1, count + interface - 1]
    ] += np.array([0.02, -0.03, 0.04, -0.01])
    coordinate[model.layout.interface_slice] += np.array(
        [0.03, -0.02, 0.01, -0.04]
    )
    coordinate[model.layout.potential_slice.start + interface] += 2.0e-4
    return model, coordinate


def _central_flux_tangent(model, coordinate, columns, step):
    bulk_columns = []
    cross_columns = []
    srh_columns = []
    state_columns = []
    for column in columns:
        plus = np.array(coordinate, copy=True)
        minus = np.array(coordinate, copy=True)
        plus[column] += step
        minus[column] -= step
        plus_fields = model.physical_fields(plus)
        minus_fields = model.physical_fields(minus)
        plus_response = model.interface_response(*plus_fields)
        minus_response = model.interface_response(*minus_fields)
        plus_srh = compute_interface_srh_occupancy_on_state(
            plus_fields[2], model.stack, model.material
        )
        minus_srh = compute_interface_srh_occupancy_on_state(
            minus_fields[2], model.stack, model.material
        )
        bulk_columns.append(
            (plus_response.bulk_flux_m2_s - minus_response.bulk_flux_m2_s)
            / (2.0 * step)
        )
        cross_columns.append(
            (plus_response.cross_flux_m2_s - minus_response.cross_flux_m2_s)
            / (2.0 * step)
        )
        srh_columns.append((plus_srh - minus_srh) / (2.0 * step))
        state_columns.append(
            (plus_response.state_flux_m2_s - minus_response.state_flux_m2_s)
            / (2.0 * step)
        )
    return tuple(
        np.stack(columns_for_flux, axis=1)
        for columns_for_flux in (
            bulk_columns,
            cross_columns,
            srh_columns,
            state_columns,
        )
    )


def test_local_linearization_matches_production_flux_decomposition():
    model, coordinate = _perturbed_operating_point()
    tangent = linearize_algebraic_interface_response(model, coordinate)
    fields = model.physical_fields(coordinate)
    response = model.interface_response(*fields)

    np.testing.assert_array_equal(tangent.bulk_flux_m2_s, response.bulk_flux_m2_s)
    np.testing.assert_array_equal(tangent.cross_flux_m2_s, response.cross_flux_m2_s)
    np.testing.assert_allclose(
        tangent.state_flux_m2_s,
        tangent.bulk_flux_m2_s + tangent.cross_flux_m2_s + tangent.srh_flux_m2_s,
        rtol=0.0,
        atol=8.0,
    )
    np.testing.assert_allclose(
        tangent.cross_interface_coordinate_jacobian_m2_s[0],
        -tangent.cross_interface_coordinate_jacobian_m2_s[2],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        tangent.cross_interface_coordinate_jacobian_m2_s[1],
        -tangent.cross_interface_coordinate_jacobian_m2_s[3],
        rtol=0.0,
        atol=0.0,
    )
    assert 0.0 < tangent.srh_occupancy < 1.0
    assert tangent.srh_denominator_m2_s > 0.0
    assert tangent.minimum_projection_occupation_margin > 0.0
    assert tangent.minimum_cross_occupation_margin > 0.0
    assert tangent.minimum_srh_occupancy_margin > 0.0
    assert tangent.minimum_interface_density_margin_m3 > 0.0
    assert tangent.minimum_interface_dos_margin_m3 > 0.0
    for value in (
        tangent.bulk_flux_m2_s,
        tangent.cross_flux_m2_s,
        tangent.srh_flux_m2_s,
        tangent.state_flux_m2_s,
        tangent.bulk_bulk_log_jacobian_m2_s,
        tangent.bulk_potential_jacobian_m2_s_V,
        tangent.bulk_interface_coordinate_jacobian_m2_s,
        tangent.cross_interface_coordinate_jacobian_m2_s,
        tangent.srh_interface_coordinate_jacobian_m2_s,
        tangent.state_interface_coordinate_jacobian_m2_s,
    ):
        assert not value.flags.writeable


def test_bulk_to_plane_log_density_and_potential_tangents_are_analytic():
    model, coordinate = _perturbed_operating_point()
    tangent = linearize_algebraic_interface_response(model, coordinate)
    count = model.layout.node_count
    interface = tangent.interface_node
    bulk_columns = (
        interface,
        count + interface,
        interface - 1,
        count + interface - 1,
    )
    bulk, cross, srh, state = _central_flux_tangent(
        model, coordinate, bulk_columns, 1.0e-6
    )
    np.testing.assert_allclose(
        bulk,
        tangent.bulk_bulk_log_jacobian_m2_s,
        rtol=2.0e-8,
        atol=1.0e3,
    )
    np.testing.assert_array_equal(cross, np.zeros((4, 4)))
    np.testing.assert_array_equal(srh, np.zeros((4, 4)))
    np.testing.assert_allclose(
        state,
        tangent.state_bulk_log_jacobian_m2_s,
        rtol=2.0e-8,
        atol=1.0e3,
    )

    potential_offset = model.layout.potential_slice.start
    potential_columns = (potential_offset + interface, potential_offset + interface - 1)
    bulk, cross, srh, state = _central_flux_tangent(
        model, coordinate, potential_columns, 1.0e-8
    )
    np.testing.assert_allclose(
        bulk,
        tangent.bulk_potential_jacobian_m2_s_V,
        rtol=2.0e-8,
        atol=1.0e9,
    )
    np.testing.assert_array_equal(cross, np.zeros((4, 2)))
    np.testing.assert_array_equal(srh, np.zeros((4, 2)))
    np.testing.assert_allclose(
        state,
        tangent.state_potential_jacobian_m2_s_V,
        rtol=2.0e-8,
        atol=1.0e9,
    )


def test_cross_node_interface_state_and_srh_tangents_match_central_reference():
    model, coordinate = _perturbed_operating_point()
    tangent = linearize_algebraic_interface_response(model, coordinate)
    interface_columns = range(
        model.layout.interface_slice.start,
        model.layout.interface_slice.stop,
    )
    bulk, cross, srh, state = _central_flux_tangent(
        model, coordinate, interface_columns, 3.0e-4
    )
    np.testing.assert_allclose(
        bulk,
        tangent.bulk_interface_coordinate_jacobian_m2_s,
        rtol=3.0e-7,
        atol=2.0e3,
    )
    np.testing.assert_allclose(
        cross,
        tangent.cross_interface_coordinate_jacobian_m2_s,
        rtol=3.0e-7,
        atol=2.0e3,
    )
    np.testing.assert_allclose(
        srh,
        tangent.srh_interface_coordinate_jacobian_m2_s,
        rtol=3.0e-7,
        atol=2.0e3,
    )
    np.testing.assert_allclose(
        state,
        tangent.state_interface_coordinate_jacobian_m2_s,
        rtol=2.0e-4,
        atol=2.0e4,
    )


def test_projection_exponent_clamp_is_explicitly_outside_capability():
    model, coordinate = _perturbed_operating_point()
    interface = int(model.material.interface_nodes[0])
    potential_offset = model.layout.potential_slice.start
    coordinate[potential_offset + interface] = (
        coordinate[potential_offset + interface - 1]
        + 30.0 * model.material.V_T_device
    )

    with pytest.raises(
        AlgebraicInterfaceJacobianCapabilityError,
        match="projection exponent clamp",
    ):
        linearize_algebraic_interface_response(model, coordinate)


def test_positive_density_clamp_is_explicitly_outside_capability():
    model, coordinate = _perturbed_operating_point()
    interface = int(model.material.interface_nodes[0])
    reference = model.layout.electron_reference_m3[interface]
    coordinate[interface] = np.log(1.0e-301 / reference)

    with pytest.raises(
        AlgebraicInterfaceJacobianCapabilityError,
        match="positive-density clamp",
    ):
        linearize_algebraic_interface_response(model, coordinate)
