from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import perovskite_sim.experiments.ion_aware_analytic_reaction as analytic_reaction
import perovskite_sim.experiments.ion_aware_analytic_transport as analytic_transport
import perovskite_sim.experiments.ion_aware_structured_jacobian as structured
from perovskite_sim.experiments.ion_aware_dc import (
    build_ion_aware_dc_protocol,
    solve_ion_aware_dc,
)
from perovskite_sim.experiments.ion_aware_impedance import (
    _physical_state,
    _state_coordinate_layout,
    build_ion_aware_impedance_protocol,
)
from perovskite_sim.experiments.jv_sweep import (
    build_electrical_grid,
    compute_current_components,
    extract_spatial_snapshot,
)
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import InterfaceDefect
from perovskite_sim.solver.mol import build_material_arrays


def _stack_with_cross_node_defect(stack):
    defects = list(stack.interface_defects)
    defects.extend([None] * (len(stack.interfaces) - len(defects)))
    defects[-1] = InterfaceDefect(
        E_t_eV=0.8,
        calibration_factor=1.0e-10,
    )
    return replace(stack, interface_defects=tuple(defects))


@pytest.fixture(scope="module")
def comparison_fixture():
    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    x = build_electrical_grid(stack, 12)
    mat = build_material_arrays(x, stack)
    dc_state = solve_ion_aware_dc(
        x,
        stack,
        build_ion_aware_dc_protocol(
            stack,
            V_dc=0.9,
            illuminated=True,
        ),
        mat=mat,
    )
    impedance_protocol = build_ion_aware_impedance_protocol(
        dc_state,
        np.array([1.0e-3, 1.0, 1.0e3]),
    )
    comparison_protocol = (
        structured.build_ion_aware_structured_jacobian_protocol(
            impedance_protocol
        )
    )
    result = structured.run_ion_aware_structured_jacobian_comparison(
        x,
        stack,
        impedance_protocol,
        comparison_protocol,
        dc_state=dc_state,
        mat=mat,
    )
    return (
        stack,
        x,
        mat,
        dc_state,
        impedance_protocol,
        comparison_protocol,
        result,
    )


def test_protocol_round_trip_binds_the_reference_level(comparison_fixture):
    *_, impedance_protocol, protocol, _result = comparison_fixture

    rebuilt = structured.IonAwareStructuredJacobianProtocol.from_json(
        protocol.canonical_json()
    )

    assert rebuilt == protocol
    assert rebuilt.protocol_hash == protocol.protocol_hash
    assert rebuilt.impedance_protocol_sha256 == impedance_protocol.protocol_hash
    assert rebuilt.minimum_state_step == (
        impedance_protocol.state_step
        * impedance_protocol.refinement_factors[-1]
    )
    assert rebuilt.maximum_state_step == 1.0e-3
    assert rebuilt.target_potential_step_V == 1.0e-9
    assert rebuilt.voltage_step == impedance_protocol.voltage_step
    assert rebuilt.max_nonsmooth_field_stencil_fraction == 0.1
    assert rebuilt.transport_linearization == (
        "analytic_sg_field_mobility_transport"
    )
    assert rebuilt.reaction_linearization == (
        "analytic_bulk_local_cross_node_projected_interface_selective_contact"
    )
    assert rebuilt.interface_clamp_linearization == (
        "positive_branch_stencil_certified"
    )
    assert rebuilt.interface_projection_linearization == (
        "smooth_unclipped_boltzmann"
    )
    assert rebuilt.schema_version.endswith("-v8")


def test_protocol_schema_and_numeric_fields_fail_closed(comparison_fixture):
    *_, protocol, _result = comparison_fixture
    payload = protocol.to_dict()
    payload["claim"] = "analytic_certified"
    with pytest.raises(ValueError, match="extra"):
        structured.IonAwareStructuredJacobianProtocol.from_dict(payload)

    payload.pop("claim")
    payload.pop("impedance_protocol_sha256")
    with pytest.raises(ValueError, match="missing"):
        structured.IonAwareStructuredJacobianProtocol.from_dict(payload)

    with pytest.raises(ValueError, match="cannot exceed one"):
        replace(protocol, column_relevance_floor_relative=1.01)
    with pytest.raises(ValueError, match="positive"):
        replace(protocol, max_rate_jacobian_column_relative_error=0.0)
    with pytest.raises(ValueError, match="interface clamp"):
        replace(protocol, interface_clamp_linearization="allow_generation")
    with pytest.raises(ValueError, match="interface projection"):
        replace(protocol, interface_projection_linearization="hard_cap")
    with pytest.raises(ValueError, match="positive"):
        replace(
            protocol,
            max_analytic_interface_reaction_jacobian_column_relative_error=0.0,
        )
    with pytest.raises(ValueError, match="positive"):
        replace(
            protocol,
            max_analytic_contact_jacobian_column_relative_error=0.0,
        )
    with pytest.raises(ValueError, match="positive"):
        replace(
            protocol,
            max_analytic_field_mobility_derivative_relative_error=0.0,
        )
    with pytest.raises(ValueError, match="cannot exceed"):
        replace(protocol, max_nonsmooth_field_stencil_fraction=0.5)
    with pytest.raises(ValueError, match="unsupported reaction"):
        replace(protocol, reaction_linearization="finite_difference_bulk")
    with pytest.raises(ValueError, match="must not be below"):
        replace(
            protocol,
            maximum_state_step=protocol.minimum_state_step * 0.5,
        )
    with pytest.raises(ValueError, match="cannot exceed"):
        replace(protocol, maximum_state_step=2.0e-2)


def test_poisson_field_stencils_fail_closed_before_crossing_pf_zero_surface(
    comparison_fixture,
):
    stack, x, mat, dc_state, _impedance, protocol, result = comparison_fixture
    face_count = x.size - 1
    zeros = np.zeros(face_count)
    field_mat = replace(
        mat,
        has_field_mobility=True,
        v_sat_n_face=zeros,
        v_sat_p_face=zeros,
        ct_beta_n_face=np.full(face_count, 2.0),
        ct_beta_p_face=np.full(face_count, 2.0),
        pf_gamma_n_face=zeros,
        pf_gamma_p_face=np.full(face_count, 3.0e-4),
    )
    layout = result.reference.coordinate_layout

    with pytest.raises(
        structured.IonAwareStructuredJacobianCapabilityError,
        match="minimum state step crosses",
    ):
        structured._build_poisson_implicit_sensitivity(
            x,
            stack,
            dc_state,
            field_mat,
            layout,
            replace(
                protocol,
                minimum_state_step=1.0e-3,
                maximum_state_step=1.0e-3,
            ),
            progress=None,
        )

    with pytest.raises(
        structured.IonAwareStructuredJacobianCapabilityError,
        match="voltage step crosses",
    ):
        structured._build_poisson_implicit_sensitivity(
            x,
            stack,
            dc_state,
            field_mat,
            layout,
            replace(
                protocol,
                minimum_state_step=1.0e-12,
                voltage_step=1.0e-2,
            ),
            progress=None,
        )


def test_species_grouped_comparison_bounds_only_declared_weak_columns():
    reference = np.array(
        [
            [1.0, 1.0e-8, 0.0, 0.0],
            [0.0, 0.0, 2.0, 1.0e-9],
        ]
    )
    candidate = reference.copy()
    candidate[:, 1] *= 2.0
    candidate[:, 3] *= 2.0

    comparison = structured._matrix_column_comparison(
        "synthetic",
        reference,
        candidate,
        limit=1.0e-4,
        group_normalized_limit=1.0e-6,
        column_groups=(slice(0, 2), slice(2, 4)),
        column_relevance_floor_relative=1.0e-6,
    )

    assert comparison.passed
    assert comparison.bounded_weak_columns == (1, 3)
    assert comparison.absolute_bounded_columns == (1, 3)
    assert not comparison.failed_columns
    np.testing.assert_array_equal(
        comparison.resolved_by_column,
        np.array([True, False, True, False]),
    )

    candidate[0, 0] = 1.01
    failed = structured._matrix_column_comparison(
        "synthetic",
        reference,
        candidate,
        limit=1.0e-4,
        group_normalized_limit=1.0e-6,
        column_groups=(slice(0, 2), slice(2, 4)),
        column_relevance_floor_relative=1.0e-6,
    )
    assert not failed.passed
    assert failed.worst_column == 0
    assert failed.failed_columns == (0,)


def test_discrete_poisson_implicit_sensitivity_matches_resolved_stencils(
    comparison_fixture,
):
    stack, x, mat, dc_state, _ip, _sp, result = comparison_fixture
    layout = result.reference.coordinate_layout
    column = layout.coordinate_slice("positive_ion").start
    step = 1.0e-4
    plus = np.zeros(layout.size)
    minus = np.zeros(layout.size)
    plus[column] = step
    minus[column] = -step
    phi_plus = extract_spatial_snapshot(
        x,
        _physical_state(plus, dc_state.y, layout),
        stack,
        dc_state.protocol.V_dc,
        mat=mat,
    ).phi
    phi_minus = extract_spatial_snapshot(
        x,
        _physical_state(minus, dc_state.y, layout),
        stack,
        dc_state.protocol.V_dc,
        mat=mat,
    ).phi
    finite_difference = (phi_plus - phi_minus) / (2.0 * step)
    exact = result.poisson_sensitivity.potential_state_jacobian_V[:, column]
    scale = max(float(np.max(np.abs(exact))), np.finfo(float).tiny)

    assert float(np.max(np.abs(finite_difference - exact)) / scale) < 1.0e-7
    assert exact[0] == 0.0
    assert exact[-1] == 0.0
    assert result.poisson_sensitivity.max_componentwise_backward_error < 1.0e-12

    voltage_step = 1.0e-5
    phi_v_plus = extract_spatial_snapshot(
        x,
        dc_state.y,
        stack,
        dc_state.protocol.V_dc + voltage_step,
        mat=mat,
    ).phi
    phi_v_minus = extract_spatial_snapshot(
        x,
        dc_state.y,
        stack,
        dc_state.protocol.V_dc - voltage_step,
        mat=mat,
    ).phi
    voltage_fd = (phi_v_plus - phi_v_minus) / (2.0 * voltage_step)
    np.testing.assert_allclose(
        result.poisson_sensitivity.potential_voltage_derivative,
        voltage_fd,
        rtol=2.0e-10,
        atol=2.0e-11,
    )


def test_phi_frozen_current_path_is_opt_in_and_matches_default(comparison_fixture):
    stack, x, mat, dc_state, _ip, _sp, result = comparison_fixture
    phi = result.poisson_sensitivity.potential_at_operating_point_V

    default_snapshot = extract_spatial_snapshot(
        x,
        dc_state.y,
        stack,
        dc_state.protocol.V_dc,
        mat=mat,
    )
    frozen_snapshot = extract_spatial_snapshot(
        x,
        dc_state.y,
        stack,
        dc_state.protocol.V_dc,
        mat=mat,
        phi_frozen=phi,
    )
    default_current = compute_current_components(
        x,
        dc_state.y,
        stack,
        dc_state.protocol.V_dc,
        mat=mat,
    )
    frozen_current = compute_current_components(
        x,
        dc_state.y,
        stack,
        dc_state.protocol.V_dc,
        mat=mat,
        phi_frozen=phi,
    )

    np.testing.assert_array_equal(frozen_snapshot.phi, default_snapshot.phi)
    np.testing.assert_allclose(frozen_current.J_total, default_current.J_total)
    with pytest.raises(ValueError, match="match the spatial grid"):
        compute_current_components(
            x,
            dc_state.y,
            stack,
            dc_state.protocol.V_dc,
            mat=mat,
            phi_frozen=np.zeros(x.size - 1),
        )


def test_real_comparison_retains_every_reference_level_and_passes(
    comparison_fixture,
):
    *_prefix, impedance_protocol, _structured_protocol, result = comparison_fixture
    certificate = result.certificate

    assert certificate.numerically_certified
    assert not certificate.thermodynamically_certified
    assert not certificate.certified
    assert not certificate.reasons
    assert len(result.reference.reference_linearizations) == len(
        impedance_protocol.refinement_factors
    )
    assert result.reference.reference_linearizations[0].state_step == (
        impedance_protocol.state_step
    )
    assert result.operator_reference.state_step == 1.0
    assert result.structured.state_step == 1.0
    assert np.min(result.poisson_sensitivity.state_steps) >= (
        _structured_protocol.minimum_state_step
    )
    assert np.max(result.poisson_sensitivity.state_steps) <= (
        _structured_protocol.maximum_state_step
    )
    assert np.ptp(result.poisson_sensitivity.state_steps) > 0.0
    assert certificate.mass_matrix.passed
    assert certificate.storage_voltage_derivative.passed
    assert certificate.rate_jacobian.passed
    assert certificate.conduction_jacobian.passed
    assert certificate.displacement_jacobian.passed
    assert certificate.analytic_transport_conduction_jacobian.passed
    assert certificate.analytic_bulk_reaction_rate_jacobian.passed
    assert certificate.analytic_bulk_reaction_rate_voltage_derivative.passed
    assert certificate.analytic_interface_reaction_rate_jacobian.passed
    assert certificate.analytic_interface_reaction_rate_voltage_derivative.passed
    assert certificate.analytic_contact_rate_jacobian.passed
    assert certificate.analytic_contact_rate_voltage_derivative.passed
    assert certificate.analytic_electron_field_mobility_derivative.passed
    assert certificate.analytic_hole_field_mobility_derivative.passed
    assert not result.analytic_transport.field_mobility.active
    assert not result.analytic_contact.active_channels
    assert (
        certificate.analytic_transport_conduction_voltage_derivative.passed
    )
    assert all(
        item.jacobian.passed
        and item.voltage_derivative.passed
        for item in certificate.analytic_transport_components
    )
    assert (
        certificate.analytic_transport_conduction_jacobian.max_relative_error
        < 1.0e-6
    )
    assert certificate.max_poisson_backward_error < (
        _structured_protocol.max_poisson_backward_error
    )
    assert all(item.jacobian.passed for item in certificate.current_components)
    assert all(
        not comparison.resolved_by_column[index]
        for comparison in (
            certificate.conduction_jacobian,
            certificate.displacement_jacobian,
        )
        for index in comparison.bounded_weak_columns
    )
    assert certificate.max_impedance_magnitude_relative_error < 1.0e-6
    assert certificate.max_impedance_phase_error_deg < 1.0e-5
    assert not np.array_equal(
        result.structured.rate_jacobian,
        result.frozen_phi_finite_difference.rate_jacobian,
    )
    assert not np.array_equal(
        result.structured.conduction_current_jacobian,
        result.frozen_phi_finite_difference.conduction_current_jacobian,
    )


def test_analytic_bulk_reaction_is_local_and_matches_its_independent_stencil(
    comparison_fixture,
):
    *_prefix, result = comparison_fixture
    layout = result.reference.coordinate_layout
    reaction = result.analytic_bulk_reaction
    comparison = result.certificate.analytic_bulk_reaction_rate_jacobian

    assert comparison.passed
    assert comparison.max_group_normalized_error < 1.0e-6
    for species in ("positive_ion", "negative_ion"):
        coordinate_slice = layout.coordinate_slice(species)
        np.testing.assert_array_equal(
            reaction.rate_jacobian[:, coordinate_slice],
            0.0,
        )
        np.testing.assert_array_equal(
            reaction.finite_difference_rate_jacobian[:, coordinate_slice],
            0.0,
        )

    row_by_state_index = {
        state_index: row for row, state_index in enumerate(layout.state_indices)
    }
    nonzero_carrier_columns = 0
    for species in ("electron", "hole"):
        columns = np.arange(layout.size)[layout.coordinate_slice(species)]
        nodes = layout.node_indices(species)
        for column, node in zip(columns, nodes, strict=True):
            expected_rows = {
                row_by_state_index[state_index]
                for state_index in (int(node), layout.n_nodes + int(node))
                if state_index in row_by_state_index
            }
            actual_rows = set(np.flatnonzero(reaction.rate_jacobian[:, column]))
            assert actual_rows <= expected_rows
            nonzero_carrier_columns += bool(actual_rows)
    assert nonzero_carrier_columns > 0


def test_analytic_interface_reaction_is_local_and_matches_independent_stencil(
    comparison_fixture,
):
    *_prefix, result = comparison_fixture
    layout = result.reference.coordinate_layout
    reaction = result.analytic_interface_reaction
    comparison = result.certificate.analytic_interface_reaction_rate_jacobian

    assert reaction.interface_nodes
    assert reaction.surface_recombination_rate_m2_s.shape == (
        len(reaction.interface_nodes),
    )
    assert reaction.electron_evaluation_nodes == reaction.interface_nodes
    assert reaction.hole_evaluation_nodes == reaction.interface_nodes
    assert not reaction.cross_node_interface_indices
    assert not reaction.projected_interface_indices
    assert reaction.minimum_cross_node_clamp_margin_m2_s is None
    assert reaction.minimum_projection_exponent_cap_margin is None
    np.testing.assert_array_equal(
        reaction.finite_difference_rate_voltage_derivative,
        0.0,
    )
    np.testing.assert_array_equal(
        reaction.complex_step_rate_voltage_derivative,
        0.0,
    )
    assert comparison.passed
    assert comparison.max_group_normalized_error < 1.0e-6
    for species in ("positive_ion", "negative_ion"):
        coordinate_slice = layout.coordinate_slice(species)
        np.testing.assert_array_equal(
            reaction.rate_jacobian[:, coordinate_slice],
            0.0,
        )
        np.testing.assert_array_equal(
            reaction.finite_difference_rate_jacobian[:, coordinate_slice],
            0.0,
        )
        np.testing.assert_array_equal(
            reaction.complex_step_rate_jacobian[:, coordinate_slice],
            0.0,
        )

    row_by_state_index = {
        state_index: row for row, state_index in enumerate(layout.state_indices)
    }
    interface_nodes = set(reaction.interface_nodes)
    nonzero_carrier_columns = 0
    for species in ("electron", "hole"):
        columns = np.arange(layout.size)[layout.coordinate_slice(species)]
        nodes = layout.node_indices(species)
        for column, node in zip(columns, nodes, strict=True):
            actual_rows = set(np.flatnonzero(reaction.rate_jacobian[:, column]))
            if int(node) not in interface_nodes:
                assert not actual_rows
                continue
            expected_rows = {
                row_by_state_index[state_index]
                for state_index in (int(node), layout.n_nodes + int(node))
                if state_index in row_by_state_index
            }
            assert actual_rows <= expected_rows
            nonzero_carrier_columns += bool(actual_rows)
    assert nonzero_carrier_columns > 0


def test_analytic_interface_reaction_uses_cross_node_columns_and_sink_rows(
    comparison_fixture,
):
    stack, x, _mat, dc_state, impedance_protocol, _protocol, result = (
        comparison_fixture
    )
    defect_stack = _stack_with_cross_node_defect(stack)
    defect_mat = build_material_arrays(x, defect_stack)
    reaction = (
        analytic_reaction
        .build_ion_aware_analytic_interface_reaction_linearization(
            x,
            defect_stack,
            dc_state.y,
            impedance_protocol.V_dc,
            defect_mat,
            result.reference.coordinate_layout,
            potential_at_operating_point_V=(
                result.poisson_sensitivity.potential_at_operating_point_V
            ),
            potential_state_jacobian_V=(
                result.poisson_sensitivity.potential_state_jacobian_V
            ),
            potential_voltage_derivative=(
                result.poisson_sensitivity.potential_voltage_derivative
            ),
            state_steps=result.poisson_sensitivity.state_steps,
            voltage_step=result.protocol.voltage_step,
        )
    )
    layout = result.reference.coordinate_layout
    interface_index = len(reaction.interface_nodes) - 1
    sink_node = reaction.interface_nodes[interface_index]
    electron_node = reaction.electron_evaluation_nodes[interface_index]
    hole_node = reaction.hole_evaluation_nodes[interface_index]

    assert reaction.cross_node_interface_indices == (interface_index,)
    assert electron_node == sink_node + 1
    assert hole_node == sink_node - 1
    assert reaction.minimum_cross_node_clamp_margin_m2_s > 0.0
    np.testing.assert_allclose(
        reaction.rate_jacobian,
        reaction.complex_step_rate_jacobian,
        rtol=2.0e-10,
        atol=0.0,
    )

    coordinate_by_state_index = {
        state_index: column
        for column, state_index in enumerate(layout.state_indices)
    }
    row_by_state_index = {
        state_index: row
        for row, state_index in enumerate(layout.state_indices)
    }
    target_rows = {
        row_by_state_index[state_index]
        for state_index in (sink_node, layout.n_nodes + sink_node)
    }
    for state_index in (
        electron_node,
        layout.n_nodes + hole_node,
    ):
        column = coordinate_by_state_index[state_index]
        assert set(np.flatnonzero(reaction.rate_jacobian[:, column])) == (
            target_rows
        )


def test_analytic_interface_reaction_closes_smooth_projected_chain_rule(
    comparison_fixture,
):
    stack, x, _mat, dc_state, impedance_protocol, _protocol, result = (
        comparison_fixture
    )
    projected_stack = replace(
        _stack_with_cross_node_defect(stack),
        interface_plane_projection=True,
    )
    projected_mat = build_material_arrays(x, projected_stack)
    reaction = (
        analytic_reaction
        .build_ion_aware_analytic_interface_reaction_linearization(
            x,
            projected_stack,
            dc_state.y,
            impedance_protocol.V_dc,
            projected_mat,
            result.reference.coordinate_layout,
            potential_at_operating_point_V=(
                result.poisson_sensitivity.potential_at_operating_point_V
            ),
            potential_state_jacobian_V=(
                result.poisson_sensitivity.potential_state_jacobian_V
            ),
            potential_voltage_derivative=(
                result.poisson_sensitivity.potential_voltage_derivative
            ),
            state_steps=result.poisson_sensitivity.state_steps,
            voltage_step=result.protocol.voltage_step,
        )
    )
    interface_index = len(reaction.interface_nodes) - 1

    assert reaction.projected_interface_indices == (interface_index,)
    assert reaction.minimum_projection_exponent_cap_margin > 0.0
    assert reaction.minimum_cross_node_clamp_margin_m2_s > 0.0
    np.testing.assert_allclose(
        reaction.rate_jacobian,
        reaction.complex_step_rate_jacobian,
        rtol=5.0e-10,
        atol=0.0,
    )
    np.testing.assert_allclose(
        reaction.rate_voltage_derivative,
        reaction.complex_step_rate_voltage_derivative,
        rtol=5.0e-10,
        atol=0.0,
    )
    state_scale = max(
        float(np.max(np.abs(reaction.rate_jacobian))),
        np.finfo(float).tiny,
    )
    voltage_scale = max(
        float(np.max(np.abs(reaction.rate_voltage_derivative))),
        np.finfo(float).tiny,
    )
    assert (
        float(
            np.max(
                np.abs(
                    reaction.rate_jacobian
                    - reaction.finite_difference_rate_jacobian
                )
            )
        )
        / state_scale
        < 5.0e-6
    )
    assert (
        float(
            np.max(
                np.abs(
                    reaction.rate_voltage_derivative
                    - reaction.finite_difference_rate_voltage_derivative
                )
            )
        )
        / voltage_scale
        < 5.0e-6
    )

    layout = result.reference.coordinate_layout
    ion_columns = np.concatenate(
        (
            np.arange(layout.size)[layout.coordinate_slice("positive_ion")],
            np.arange(layout.size)[layout.coordinate_slice("negative_ion")],
        )
    )
    assert np.any(reaction.rate_jacobian[:, ion_columns] != 0.0)


def test_analytic_selective_contact_block_covers_all_four_sign_conventions(
    comparison_fixture,
):
    stack, x, _mat, dc_state, impedance_protocol, _protocol, result = (
        comparison_fixture
    )
    contact_stack = replace(
        stack,
        S_n_left=2.0e-3,
        S_n_right=3.0e2,
        S_p_left=4.0e2,
        S_p_right=5.0e-3,
    )
    contact_mat = build_material_arrays(x, contact_stack)
    layout = _state_coordinate_layout(contact_mat, x.size)
    state_steps = np.full(layout.size, 1.0e-4)
    contact = analytic_reaction.build_ion_aware_analytic_contact_linearization(
        x,
        contact_stack,
        dc_state.y,
        impedance_protocol.V_dc,
        contact_mat,
        layout,
        potential_at_operating_point_V=(
            result.poisson_sensitivity.potential_at_operating_point_V
        ),
        state_steps=state_steps,
    )

    assert contact.active_channels == (
        "electron_left",
        "electron_right",
        "hole_left",
        "hole_right",
    )
    assert contact.boundary_state_indices == (
        0,
        x.size - 1,
        x.size,
        2 * x.size - 1,
    )
    assert contact.relaxation_rate_s1.shape == (4,)
    assert contact.rate_at_operating_point_m3_s.shape == (4,)
    np.testing.assert_array_equal(contact.rate_voltage_derivative, 0.0)

    row_by_state_index = {
        state_index: row
        for row, state_index in enumerate(layout.state_indices)
    }
    boundary_rows = tuple(
        row_by_state_index[state_index]
        for state_index in contact.boundary_state_indices
    )
    expected_nonzero = {
        *boundary_rows,
    }
    actual_nonzero = set(np.flatnonzero(contact.rate_jacobian))
    assert actual_nonzero == {
        row * layout.size + row for row in expected_nonzero
    }
    diagonal = np.diag(contact.rate_jacobian)
    assert np.all(diagonal[list(expected_nonzero)] < 0.0)
    expected_diagonal = (
        -contact.relaxation_rate_s1
        * np.asarray(dc_state.y)[list(contact.boundary_state_indices)]
        * state_steps[list(boundary_rows)]
    )
    np.testing.assert_allclose(
        diagonal[list(boundary_rows)],
        expected_diagonal,
        rtol=2.0e-15,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        np.delete(diagonal, sorted(expected_nonzero)),
        0.0,
    )
    finite_diagonal = np.diag(contact.finite_difference_rate_jacobian)
    np.testing.assert_allclose(
        finite_diagonal[list(expected_nonzero)]
        / diagonal[list(expected_nonzero)],
        np.sinh(1.0e-4) / 1.0e-4,
        rtol=2.0e-9,
        atol=2.0e-9,
    )
    for species in ("positive_ion", "negative_ion"):
        coordinate_slice = layout.coordinate_slice(species)
        np.testing.assert_array_equal(
            contact.rate_jacobian[:, coordinate_slice],
            0.0,
        )


def test_analytic_selective_contact_block_handles_partial_zero_and_invalid_inputs(
    comparison_fixture,
):
    stack, x, _mat, dc_state, impedance_protocol, _protocol, result = (
        comparison_fixture
    )
    partial_stack = replace(
        stack,
        S_n_left=0.0,
        S_n_right=None,
        S_p_left=None,
        S_p_right=None,
    )
    partial_mat = build_material_arrays(x, partial_stack)
    layout = _state_coordinate_layout(partial_mat, x.size)
    kwargs = {
        "potential_at_operating_point_V": (
            result.poisson_sensitivity.potential_at_operating_point_V
        ),
        "state_steps": np.full(layout.size, 1.0e-4),
    }
    contact = analytic_reaction.build_ion_aware_analytic_contact_linearization(
        x,
        partial_stack,
        dc_state.y,
        impedance_protocol.V_dc,
        partial_mat,
        layout,
        **kwargs,
    )

    assert contact.active_channels == ("electron_left",)
    np.testing.assert_array_equal(contact.relaxation_rate_s1, 0.0)
    np.testing.assert_array_equal(contact.rate_jacobian, 0.0)
    np.testing.assert_array_equal(contact.finite_difference_rate_jacobian, 0.0)

    with pytest.raises(
        analytic_reaction.IonAwareAnalyticReactionCapabilityError,
        match="physically admissible",
    ):
        analytic_reaction.build_ion_aware_analytic_contact_linearization(
            x,
            partial_stack,
            dc_state.y,
            impedance_protocol.V_dc,
            replace(partial_mat, S_n_L=-1.0),
            layout,
            **kwargs,
        )

    dirichlet_layout = result.reference.coordinate_layout
    with pytest.raises(
        analytic_reaction.IonAwareAnalyticReactionCapabilityError,
        match="absent from the state layout",
    ):
        analytic_reaction.build_ion_aware_analytic_contact_linearization(
            x,
            partial_stack,
            dc_state.y,
            impedance_protocol.V_dc,
            partial_mat,
            dirichlet_layout,
            potential_at_operating_point_V=(
                result.poisson_sensitivity.potential_at_operating_point_V
            ),
            state_steps=np.full(dirichlet_layout.size, 1.0e-4),
        )


def test_analytic_component_base_currents_match_the_nonlinear_evaluator(
    comparison_fixture,
):
    stack, x, mat, dc_state, impedance_protocol, _protocol, result = (
        comparison_fixture
    )
    evaluator = structured._structured_evaluator(
        x,
        stack,
        dc_state,
        impedance_protocol,
        mat,
        result.reference.coordinate_layout,
        result.poisson_sensitivity,
    )
    base = evaluator(
        np.zeros(result.reference.coordinate_layout.size),
        impedance_protocol.V_dc,
    )
    expected = {
        component.name: component.current_faces
        for component in base.current_components
    }

    assert set(expected) == {
        component.name for component in result.analytic_transport.current_components
    }
    for component in result.analytic_transport.current_components:
        np.testing.assert_allclose(
            component.current_faces,
            expected[component.name],
            rtol=2.0e-14,
            atol=max(float(np.max(np.abs(expected[component.name]))), 1.0)
            * 2.0e-14,
        )


def test_analytic_field_mobility_chain_matches_local_and_current_stencils(
    comparison_fixture,
):
    stack, x, mat, dc_state, impedance_protocol, _protocol, result = (
        comparison_fixture
    )
    face_count = x.size - 1
    zeros = np.zeros(face_count)
    v_sat_n = zeros.copy()
    v_sat_p = zeros.copy()
    gamma_p = zeros.copy()
    v_sat_n[4:8] = 1.0e5
    v_sat_p[4:8] = 2.0e5
    gamma_p[:4] = 1.0e-5
    electron_diffusivity = mat.D_n_face.copy()
    electron_diffusivity[4] = 0.0
    active_mat = replace(
        mat,
        D_n_face=electron_diffusivity,
        has_field_mobility=True,
        v_sat_n_face=v_sat_n,
        v_sat_p_face=v_sat_p,
        ct_beta_n_face=np.full(face_count, 2.0),
        ct_beta_p_face=np.full(face_count, 1.0),
        pf_gamma_n_face=zeros,
        pf_gamma_p_face=gamma_p,
    )
    analytic = (
        analytic_transport.build_ion_aware_analytic_transport_linearization(
            x,
            stack,
            dc_state.y,
            impedance_protocol.V_dc,
            active_mat,
            result.reference.coordinate_layout,
            potential_at_operating_point_V=(
                result.poisson_sensitivity.potential_at_operating_point_V
            ),
            potential_state_jacobian_V=(
                result.poisson_sensitivity.potential_state_jacobian_V
            ),
            potential_voltage_derivative=(
                result.poisson_sensitivity.potential_voltage_derivative
            ),
            state_steps=result.poisson_sensitivity.state_steps,
        )
    )
    mobility = analytic.field_mobility

    assert mobility.active
    assert mobility.electron_mobility_m2_V_s[4] == 0.0
    assert mobility.electron_field_derivative_m3_V2_s[4] == 0.0
    assert np.any(mobility.electron_field_derivative_m3_V2_s != 0.0)
    assert np.any(mobility.hole_field_derivative_m3_V2_s != 0.0)
    for expected, actual in (
        (
            mobility.electron_finite_difference_derivative_m3_V2_s,
            mobility.electron_field_derivative_m3_V2_s,
        ),
        (
            mobility.hole_finite_difference_derivative_m3_V2_s,
            mobility.hole_field_derivative_m3_V2_s,
        ),
    ):
        scale = max(float(np.max(np.abs(expected))), np.finfo(float).tiny)
        assert float(np.max(np.abs(actual - expected)) / scale) < 5.0e-6

    evaluator = structured._structured_evaluator(
        x,
        stack,
        dc_state,
        impedance_protocol,
        active_mat,
        result.reference.coordinate_layout,
        result.poisson_sensitivity,
    )
    base = evaluator(
        np.zeros(result.reference.coordinate_layout.size),
        impedance_protocol.V_dc,
    )
    expected_currents = {
        component.name: component.current_faces
        for component in base.current_components
    }
    for component in analytic.current_components:
        np.testing.assert_allclose(
            component.current_faces,
            expected_currents[component.name],
            rtol=2.0e-14,
            atol=max(
                float(np.max(np.abs(expected_currents[component.name]))),
                1.0,
            )
            * 2.0e-14,
        )


def test_analytic_transport_capability_gates_nonanalytic_active_branches(
    comparison_fixture,
):
    stack, x, mat, dc_state, impedance_protocol, _protocol, result = (
        comparison_fixture
    )
    kwargs = {
        "potential_at_operating_point_V": (
            result.poisson_sensitivity.potential_at_operating_point_V
        ),
        "potential_state_jacobian_V": (
            result.poisson_sensitivity.potential_state_jacobian_V
        ),
        "potential_voltage_derivative": (
            result.poisson_sensitivity.potential_voltage_derivative
        ),
        "state_steps": result.poisson_sensitivity.state_steps,
    }
    with pytest.raises(
        analytic_transport.IonAwareAnalyticTransportCapabilityError,
        match="arrays are incomplete",
    ):
        analytic_transport.build_ion_aware_analytic_transport_linearization(
            x,
            stack,
            dc_state.y,
            impedance_protocol.V_dc,
            replace(mat, has_field_mobility=True),
            result.reference.coordinate_layout,
            **kwargs,
        )

    face_count = x.size - 1
    zeros = np.zeros(face_count)
    cusp_mat = replace(
        mat,
        has_field_mobility=True,
        v_sat_n_face=zeros,
        v_sat_p_face=zeros,
        ct_beta_n_face=np.full(face_count, 2.0),
        ct_beta_p_face=np.full(face_count, 2.0),
        pf_gamma_n_face=np.full(face_count, 3.0e-4),
        pf_gamma_p_face=zeros,
    )
    with pytest.raises(
        analytic_transport.IonAwareAnalyticTransportCapabilityError,
        match="non-differentiable face",
    ):
        analytic_transport.build_ion_aware_analytic_transport_linearization(
            x,
            stack,
            dc_state.y,
            impedance_protocol.V_dc,
            cusp_mat,
            result.reference.coordinate_layout,
            **{
                **kwargs,
                "potential_at_operating_point_V": np.zeros_like(x),
            },
        )

    with pytest.raises(
        analytic_transport.IonAwareAnalyticTransportCapabilityError,
        match="thermionic cap is active",
    ):
        analytic_transport.build_ion_aware_analytic_transport_linearization(
            x,
            stack,
            dc_state.y,
            impedance_protocol.V_dc,
            replace(mat, A_star_n=np.zeros_like(mat.A_star_n)),
            result.reference.coordinate_layout,
            **kwargs,
        )


def test_analytic_bulk_reaction_gates_unimplemented_nonlocal_branches(
    comparison_fixture,
):
    stack, x, mat, dc_state, impedance_protocol, _protocol, result = (
        comparison_fixture
    )
    kwargs = {
        "potential_at_operating_point_V": (
            result.poisson_sensitivity.potential_at_operating_point_V
        ),
        "state_steps": result.poisson_sensitivity.state_steps,
    }
    build = analytic_reaction.build_ion_aware_analytic_bulk_reaction_linearization

    with pytest.raises(
        analytic_reaction.IonAwareAnalyticReactionCapabilityError,
        match="radiative reabsorption",
    ):
        build(
            x,
            stack,
            dc_state.y,
            impedance_protocol.V_dc,
            replace(mat, has_radiative_reabsorption=True),
            result.reference.coordinate_layout,
            **kwargs,
        )

    with pytest.raises(
        analytic_reaction.IonAwareAnalyticReactionCapabilityError,
        match="recombination de-spike",
    ):
        build(
            x,
            stack,
            dc_state.y,
            impedance_protocol.V_dc,
            replace(
                mat,
                het_recomb_despike=0.5,
                het_recomb_nodes=(x.size // 2,),
            ),
            result.reference.coordinate_layout,
            **kwargs,
        )

    with pytest.raises(
        analytic_reaction.IonAwareAnalyticReactionCapabilityError,
        match="denominator",
    ):
        build(
            x,
            stack,
            dc_state.y,
            impedance_protocol.V_dc,
            replace(
                mat,
                tau_n=np.zeros_like(mat.tau_n),
                tau_p=np.zeros_like(mat.tau_p),
            ),
            result.reference.coordinate_layout,
            **kwargs,
        )


def test_analytic_interface_reaction_gates_unimplemented_topologies(
    comparison_fixture,
    monkeypatch,
):
    stack, x, mat, dc_state, impedance_protocol, _protocol, result = (
        comparison_fixture
    )
    kwargs = {
        "potential_at_operating_point_V": (
            result.poisson_sensitivity.potential_at_operating_point_V
        ),
        "potential_state_jacobian_V": (
            result.poisson_sensitivity.potential_state_jacobian_V
        ),
        "potential_voltage_derivative": (
            result.poisson_sensitivity.potential_voltage_derivative
        ),
        "state_steps": result.poisson_sensitivity.state_steps,
        "voltage_step": result.protocol.voltage_step,
    }
    build = (
        analytic_reaction
        .build_ion_aware_analytic_interface_reaction_linearization
    )

    for changed, message in (
        ({"iface_shared_occ": True}, "shared-occupancy"),
        ({"iface_two_sided": True}, "two-sided"),
        ({"iface_plane_closure": True}, "interface-plane closure"),
        ({"N_iface_state": 1}, "dynamic interface-plane states"),
        (
            {"interface_n1": mat.interface_n1[:-1]},
            "parameter arrays are not topology aligned",
        ),
        (
            {"interface_n1": (-1.0, *mat.interface_n1[1:])},
            "physically admissible",
        ),
        (
            {
                "interface_eval_node_n": (
                    mat.interface_nodes[0] + 1,
                    *mat.interface_eval_node_n[1:],
                )
            },
            "requires a declared InterfaceDefect",
        ),
    ):
        with pytest.raises(
            analytic_reaction.IonAwareAnalyticReactionCapabilityError,
            match=message,
        ):
            build(
                x,
                stack,
                dc_state.y,
                impedance_protocol.V_dc,
                replace(mat, **changed),
                result.reference.coordinate_layout,
                **kwargs,
            )

    defect_stack = replace(
        stack,
        interface_defects=(InterfaceDefect(E_t_eV=0.5),),
    )
    with pytest.raises(
        analytic_reaction.IonAwareAnalyticReactionCapabilityError,
        match="not represented by cross-node material sampling",
    ):
        build(
            x,
            defect_stack,
            dc_state.y,
            impedance_protocol.V_dc,
            mat,
            result.reference.coordinate_layout,
            **kwargs,
        )

    defect_stack = _stack_with_cross_node_defect(stack)
    defect_mat = build_material_arrays(x, defect_stack)
    monkeypatch.setenv("SOLARLAB_IFACE_ALLOW_GEN", "1")
    with pytest.raises(
        analytic_reaction.IonAwareAnalyticReactionCapabilityError,
        match="SOLARLAB_IFACE_ALLOW_GEN",
    ):
        build(
            x,
            defect_stack,
            dc_state.y,
            impedance_protocol.V_dc,
            defect_mat,
            result.reference.coordinate_layout,
            **kwargs,
        )
    monkeypatch.delenv("SOLARLAB_IFACE_ALLOW_GEN")

    cross_index = len(defect_mat.interface_nodes) - 1
    electron_node = defect_mat.interface_eval_node_n[cross_index]
    hole_node = defect_mat.interface_eval_node_p[cross_index]
    negative_state = np.asarray(dc_state.y, dtype=float).copy()
    negative_state[electron_node] = 1.0
    negative_state[x.size + hole_node] = 1.0
    with pytest.raises(
        analytic_reaction.IonAwareAnalyticReactionCapabilityError,
        match="clamp must be inactive",
    ):
        build(
            x,
            defect_stack,
            negative_state,
            impedance_protocol.V_dc,
            defect_mat,
            result.reference.coordinate_layout,
            **kwargs,
        )

    zero_state = np.asarray(dc_state.y, dtype=float).copy()
    zero_state[electron_node] = 1.0
    zero_state[x.size + hole_node] = (
        defect_mat.interface_ni_sq_eff[cross_index]
    )
    with pytest.raises(
        analytic_reaction.IonAwareAnalyticReactionCapabilityError,
        match="clamp must be inactive",
    ):
        build(
            x,
            defect_stack,
            zero_state,
            impedance_protocol.V_dc,
            defect_mat,
            result.reference.coordinate_layout,
            **kwargs,
        )

    coordinate_by_state_index = {
        state_index: column
        for column, state_index in enumerate(
            result.reference.coordinate_layout.state_indices
        )
    }
    electron_step = result.poisson_sensitivity.state_steps[
        coordinate_by_state_index[electron_node]
    ]
    crossing_state = np.asarray(dc_state.y, dtype=float).copy()
    crossing_state[electron_node] = 1.0
    crossing_state[x.size + hole_node] = (
        defect_mat.interface_ni_sq_eff[cross_index]
        * float(np.exp(0.5 * electron_step))
    )
    with pytest.raises(
        analytic_reaction.IonAwareAnalyticReactionCapabilityError,
        match="central stencil crosses",
    ):
        build(
            x,
            defect_stack,
            crossing_state,
            impedance_protocol.V_dc,
            defect_mat,
            result.reference.coordinate_layout,
            **kwargs,
        )

    monkeypatch.setenv("SOLARLAB_IFACE_QSS", "1")
    with pytest.raises(
        analytic_reaction.IonAwareAnalyticReactionCapabilityError,
        match="QSS interface-plane root solve",
    ):
        build(
            x,
            stack,
            dc_state.y,
            impedance_protocol.V_dc,
            mat,
            result.reference.coordinate_layout,
            **kwargs,
        )


def test_projected_interface_reaction_rejects_exponent_cap_surfaces(
    comparison_fixture,
):
    stack, x, _mat, dc_state, impedance_protocol, _protocol, result = (
        comparison_fixture
    )
    projected_stack = replace(
        _stack_with_cross_node_defect(stack),
        interface_plane_projection=True,
    )
    projected_mat = build_material_arrays(x, projected_stack)
    layout = result.reference.coordinate_layout
    potential = np.asarray(
        result.poisson_sensitivity.potential_at_operating_point_V,
        dtype=float,
    )
    potential_jacobian = np.asarray(
        result.poisson_sensitivity.potential_state_jacobian_V,
        dtype=float,
    )
    potential_voltage = np.asarray(
        result.poisson_sensitivity.potential_voltage_derivative,
        dtype=float,
    )
    state_steps = result.poisson_sensitivity.state_steps
    voltage_step = result.protocol.voltage_step
    interface_index = len(projected_mat.interface_nodes) - 1
    node = projected_mat.interface_nodes[interface_index]
    electron_node = projected_mat.interface_eval_node_n[interface_index]
    hole_node = projected_mat.interface_eval_node_p[interface_index]
    thermal_voltage = projected_mat.V_T_device

    def build(**overrides):
        arguments = {
            "potential_at_operating_point_V": potential,
            "potential_state_jacobian_V": potential_jacobian,
            "potential_voltage_derivative": potential_voltage,
            "state_steps": state_steps,
            "voltage_step": voltage_step,
        }
        arguments.update(overrides)
        return (
            analytic_reaction
            .build_ion_aware_analytic_interface_reaction_linearization(
                x,
                projected_stack,
                dc_state.y,
                impedance_protocol.V_dc,
                projected_mat,
                layout,
                **arguments,
            )
        )

    capped_potential = potential.copy()
    capped_potential[electron_node] = (
        capped_potential[node]
        - analytic_reaction._IFACE_PROJ_EXP_CAP * thermal_voltage
    )
    with pytest.raises(
        analytic_reaction.IonAwareAnalyticReactionCapabilityError,
        match="cap is active",
    ):
        build(potential_at_operating_point_V=capped_potential)

    log_n = (potential[node] - potential[electron_node]) / thermal_voltage
    excess_log_step = (
        analytic_reaction._IFACE_PROJ_EXP_CAP - abs(log_n) + 1.0
    )
    capped_state_jacobian = np.zeros_like(potential_jacobian)
    capped_state_jacobian[node, 0] = (
        excess_log_step * thermal_voltage / state_steps[0]
    )
    capped_state_jacobian[hole_node, 0] = capped_state_jacobian[node, 0]
    with pytest.raises(
        analytic_reaction.IonAwareAnalyticReactionCapabilityError,
        match="state stencil reaches",
    ):
        build(potential_state_jacobian_V=capped_state_jacobian)

    capped_voltage = np.zeros_like(potential_voltage)
    capped_voltage[node] = (
        excess_log_step * thermal_voltage / voltage_step
    )
    capped_voltage[hole_node] = capped_voltage[node]
    with pytest.raises(
        analytic_reaction.IonAwareAnalyticReactionCapabilityError,
        match="voltage stencil reaches",
    ):
        build(potential_voltage_derivative=capped_voltage)


def test_mismatched_hash_is_rejected_before_comparison(
    comparison_fixture,
):
    stack, x, mat, dc_state, impedance_protocol, protocol, _result = (
        comparison_fixture
    )
    with pytest.raises(
        structured.IonAwareStructuredJacobianCapabilityError,
        match="protocol hash",
    ):
        structured.run_ion_aware_structured_jacobian_comparison(
            x,
            stack,
            impedance_protocol,
            replace(protocol, impedance_protocol_sha256="0" * 64),
            dc_state=dc_state,
            mat=mat,
        )


def test_failed_comparison_gate_preserves_diagnostic_evidence(comparison_fixture):
    stack, x, mat, dc_state, impedance_protocol, protocol, _result = (
        comparison_fixture
    )
    strict = replace(
        protocol,
        max_mass_matrix_column_relative_error=1.0e-14,
        max_group_normalized_column_error=1.0e-14,
    )

    with pytest.raises(
        structured.IonAwareStructuredJacobianCertificationError,
        match="mass_matrix_exceeds_limit",
    ) as captured:
        structured.run_ion_aware_structured_jacobian_comparison(
            x,
            stack,
            impedance_protocol,
            strict,
            dc_state=dc_state,
            mat=mat,
        )
    assert not captured.value.result.certificate.numerically_certified

    diagnostic = structured.run_ion_aware_structured_jacobian_comparison(
        x,
        stack,
        impedance_protocol,
        strict,
        dc_state=dc_state,
        mat=mat,
        require_numerical_certificate=False,
    )
    assert "mass_matrix_exceeds_limit" in diagnostic.certificate.reasons
