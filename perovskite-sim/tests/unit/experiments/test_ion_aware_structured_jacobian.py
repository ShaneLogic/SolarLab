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
    build_ion_aware_impedance_protocol,
)
from perovskite_sim.experiments.jv_sweep import (
    build_electrical_grid,
    compute_current_components,
    extract_spatial_snapshot,
)
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.solver.mol import build_material_arrays


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
    assert rebuilt.transport_linearization == "analytic_sg_transport"
    assert rebuilt.reaction_linearization == (
        "analytic_bulk_central_difference_interface_contact"
    )
    assert rebuilt.schema_version.endswith("-v3")


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
    with pytest.raises(ValueError, match="unsupported reaction"):
        replace(protocol, reaction_linearization="finite_difference_bulk")
    with pytest.raises(ValueError, match="must not be below"):
        replace(
            protocol,
            maximum_state_step=protocol.minimum_state_step * 0.5,
        )
    with pytest.raises(ValueError, match="cannot exceed"):
        replace(protocol, maximum_state_step=2.0e-2)


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
        match="field-dependent mobility",
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
