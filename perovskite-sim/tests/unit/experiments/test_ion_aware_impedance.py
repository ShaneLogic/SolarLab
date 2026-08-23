from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import perovskite_sim.experiments.ion_aware_impedance as impedance
from perovskite_sim.experiments.ion_aware_dc import (
    build_ion_aware_dc_protocol,
    solve_ion_aware_dc,
)
from perovskite_sim.experiments.jv_sweep import build_electrical_grid
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.solver.mol import build_material_arrays


@pytest.fixture(scope="module")
def dc_fixture():
    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    x = build_electrical_grid(stack, 12)
    mat = build_material_arrays(x, stack)
    protocol = build_ion_aware_dc_protocol(
        stack,
        V_dc=0.9,
        illuminated=True,
    )
    result = solve_ion_aware_dc(x, stack, protocol, mat=mat)
    return stack, x, mat, result


def test_protocol_round_trip_binds_dc_protocol_and_state(dc_fixture):
    _stack, _x, _mat, dc_state = dc_fixture
    protocol = impedance.build_ion_aware_impedance_protocol(
        dc_state,
        np.array([1.0e-3, 1.0, 1.0e3]),
    )

    rebuilt = impedance.IonAwareImpedanceProtocol.from_json(
        protocol.canonical_json()
    )

    assert rebuilt == protocol
    assert rebuilt.protocol_hash == protocol.protocol_hash
    assert rebuilt.dc_protocol_sha256 == dc_state.protocol_hash
    assert len(rebuilt.dc_state_sha256) == 64
    assert rebuilt.frequency_branch_margin_decades == 1.0
    assert rebuilt.max_frequency_sampling_gap_decades == 0.5
    assert rebuilt.schema_version.endswith("-v2")
    assert replace(
        rebuilt,
        frequency_branch_margin_decades=2.0,
    ).protocol_hash != rebuilt.protocol_hash


def test_protocol_rejects_unknown_and_missing_fields(dc_fixture):
    _stack, _x, _mat, dc_state = dc_fixture
    payload = impedance.build_ion_aware_impedance_protocol(
        dc_state,
        np.array([1.0]),
    ).to_dict()
    payload["claim"] = "externally_validated"
    with pytest.raises(ValueError, match="extra"):
        impedance.IonAwareImpedanceProtocol.from_dict(payload)

    payload.pop("claim")
    payload.pop("dc_state_sha256")
    with pytest.raises(ValueError, match="missing"):
        impedance.IonAwareImpedanceProtocol.from_dict(payload)


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"frequencies_Hz": (1.0, 1.0)}, "increasing"),
        ({"delta_V": 0.02}, "20 mV"),
        ({"refinement_factors": (1.0, 0.5)}, "at least three"),
        ({"refinement_factors": (1.0, 0.5, 0.75)}, "strictly decreasing"),
        ({"frequency_branch_margin_decades": 0.0}, "positive"),
        ({"max_frequency_sampling_gap_decades": 0.0}, "positive"),
    ],
)
def test_protocol_rejects_ambiguous_or_nonlinear_requests(
    updates, message, dc_fixture
):
    _stack, _x, _mat, dc_state = dc_fixture
    protocol = impedance.build_ion_aware_impedance_protocol(
        dc_state,
        np.array([1.0, 10.0]),
    )

    with pytest.raises(ValueError, match=message):
        replace(protocol, **updates)


def test_coordinate_layout_excludes_fixed_contacts_and_structural_ion_zeros(
    dc_fixture,
):
    _stack, x, mat, _dc_state = dc_fixture

    layout = impedance._state_coordinate_layout(mat, len(x))

    assert 0 not in layout.electron_state_indices
    assert len(x) - 1 not in layout.electron_state_indices
    assert len(x) not in layout.hole_state_indices
    assert 2 * len(x) - 1 not in layout.hole_state_indices
    expected_positive = np.flatnonzero(
        (np.asarray(mat.P_ion0) > 0.0) & (np.asarray(mat.D_ion_node) > 0.0)
    )
    np.testing.assert_array_equal(
        layout.node_indices("positive_ion"), expected_positive
    )
    assert not layout.negative_ion_state_indices


def test_reference_lane_returns_mass_current_and_storage_decomposition(dc_fixture):
    stack, x, mat, dc_state = dc_fixture
    protocol = impedance.build_ion_aware_impedance_protocol(
        dc_state,
        np.array([1.0e-2, 1.0e2]),
    )

    result = impedance.run_ion_aware_impedance(
        x,
        stack,
        protocol,
        dc_state=dc_state,
        mat=mat,
    )

    assert result.certificate.numerically_certified
    assert not result.certificate.thermodynamically_certified
    assert not result.certificate.certified
    assert not result.certificate.frequency_window_certified
    assert result.frequency_window.has_mobile_ions
    assert not result.frequency_window.characteristic_frequency_bracketed
    assert not result.frequency_window.full_timescale_envelope_bracketed
    assert not result.frequency_window.ionic_branch_covered
    assert result.frequency_window.recommended_f_min_Hz < (
        np.min(protocol.frequencies_Hz)
    )
    np.testing.assert_array_equal(
        result.frequencies,
        np.asarray(protocol.frequencies_Hz),
    )
    assert result.negative_ion_admittance_faces_S_m2 is None
    assert result.negative_ion_storage_response_F_m2 is None
    np.testing.assert_allclose(
        result.Y_faces,
        result.conduction_admittance_faces_S_m2
        + result.displacement_admittance_faces_S_m2,
    )
    np.testing.assert_allclose(
        result.conduction_admittance_faces_S_m2,
        result.electron_admittance_faces_S_m2
        + result.hole_admittance_faces_S_m2
        + result.positive_ion_admittance_faces_S_m2,
        rtol=protocol.max_current_decomposition_relative_error,
        atol=1.0e-12,
    )
    assert result.certificate.max_mass_diagonal_relative_error < 1.0e-8
    assert result.certificate.max_mass_off_diagonal_relative == 0.0
    assert result.certificate.max_ion_inventory_response_relative < 1.0e-8
    assert result.certificate.max_current_decomposition_relative_error < (
        protocol.max_current_decomposition_relative_error
    )
    assert all(item.passed for item in result.certificate.perturbation_assessments)
    point_certificates = result.certificate.frequency_point_certificates
    assert len(point_certificates) == len(result.frequencies)
    assert all(item.numerically_certified for item in point_certificates)
    np.testing.assert_allclose(
        [item.max_relative_face_spread for item in point_certificates],
        result.reference_linearization.max_relative_face_spread,
    )
    np.testing.assert_allclose(
        [item.backward_error for item in point_certificates],
        result.reference_linearization.backward_error,
    )
    np.testing.assert_allclose(
        [item.reciprocal_condition for item in point_certificates],
        result.reference_linearization.reciprocal_condition,
    )
    assert all(
        len(item.perturbation_assessments)
        == len(protocol.refinement_factors) - 1
        for item in point_certificates
    )
    np.testing.assert_allclose(
        [item.net_charge_storage_response_F_m2 for item in point_certificates],
        result.net_charge_storage_response_F_m2,
    )
    assert np.all(np.isfinite(result.Z))


def test_perturbation_assessment_retains_a_fail_closed_result_per_frequency():
    frequencies = np.array([1.0, 10.0])
    coarse = SimpleNamespace(
        frequencies=frequencies,
        impedance=np.array([1.0 + 0.0j, 1.0 + 0.0j]),
    )
    fine = SimpleNamespace(
        frequencies=frequencies,
        impedance=np.array([1.001 + 0.0j, 0.0 + 2.0j]),
    )
    protocol = SimpleNamespace(
        max_impedance_magnitude_relative_change=0.01,
        max_impedance_phase_change_deg=0.5,
    )

    assessment = impedance._perturbation_assessment(
        1.0,
        0.5,
        coarse,
        fine,
        protocol,
    )

    assert assessment.frequency_assessments[0].passed
    assert not assessment.frequency_assessments[1].passed
    assert not assessment.passed
    assert assessment.max_impedance_magnitude_relative_change == pytest.approx(
        0.5
    )
    assert assessment.max_impedance_phase_change_deg == pytest.approx(90.0)


def test_dense_full_timescale_window_is_separately_certified(dc_fixture):
    stack, x, mat, dc_state = dc_fixture
    seed = impedance.assess_impedance_frequency_window(
        x,
        mat,
        np.array([1.0]),
    )
    low = seed.recommended_f_min_Hz
    high = seed.recommended_f_max_Hz
    assert low is not None and high is not None
    count = int(np.ceil(np.log10(high / low) / 0.25)) + 3
    frequencies = np.logspace(
        np.log10(low) - 0.01,
        np.log10(high) + 0.01,
        count,
    )
    protocol = impedance.build_ion_aware_impedance_protocol(
        dc_state,
        frequencies,
    )

    result = impedance.run_ion_aware_impedance(
        x,
        stack,
        protocol,
        dc_state=dc_state,
        mat=mat,
    )

    assert result.certificate.numerically_certified
    assert result.certificate.frequency_window_certified
    assert not result.certificate.thermodynamically_certified
    assert not result.certificate.certified
    assert result.frequency_window.characteristic_frequency_bracketed
    assert result.frequency_window.full_timescale_envelope_bracketed
    assert result.frequency_window.ionic_branch_covered
    assert result.frequency_window.warnings == ()
    assert result.frequency_window.max_observed_sampling_gap_decades <= (
        protocol.max_frequency_sampling_gap_decades
    )


def test_frequency_window_strict_gate_retains_diagnostic_result(dc_fixture):
    stack, x, mat, dc_state = dc_fixture
    protocol = impedance.build_ion_aware_impedance_protocol(
        dc_state,
        np.array([10.0, 100.0]),
    )

    with pytest.raises(
        impedance.IonAwareImpedanceCertificationError,
        match="frequency-window certificate failed",
    ) as captured:
        impedance.run_ion_aware_impedance(
            x,
            stack,
            protocol,
            dc_state=dc_state,
            mat=mat,
            require_frequency_window_certificate=True,
        )

    assert captured.value.result.certificate.numerically_certified
    assert not captured.value.result.certificate.frequency_window_certified


def test_reference_lane_rejects_stale_state_hash_before_linearization(dc_fixture):
    stack, x, mat, dc_state = dc_fixture
    protocol = impedance.build_ion_aware_impedance_protocol(
        dc_state,
        np.array([1.0]),
    )
    stale = replace(dc_state, y=dc_state.y.copy())
    stale.y[1] = np.nextafter(stale.y[1], np.inf)

    with pytest.raises(
        impedance.IonAwareImpedanceCapabilityError,
        match="packed DC state hash",
    ):
        impedance.run_ion_aware_impedance(
            x,
            stack,
            protocol,
            dc_state=stale,
            mat=mat,
        )


def test_reference_lane_rejects_a_different_device_stack(dc_fixture):
    stack, x, mat, dc_state = dc_fixture
    protocol = impedance.build_ion_aware_impedance_protocol(
        dc_state,
        np.array([1.0]),
    )
    changed_stack = replace(stack, Phi=stack.Phi * 1.01)

    with pytest.raises(
        impedance.IonAwareImpedanceCapabilityError,
        match="stack does not match",
    ):
        impedance.run_ion_aware_impedance(
            x,
            changed_stack,
            protocol,
            dc_state=dc_state,
            mat=mat,
        )


def test_reference_lane_rejects_declared_uncertified_dc_state(dc_fixture):
    stack, x, mat, dc_state = dc_fixture
    uncertified = replace(dc_state, numerically_certified=False)
    protocol = impedance.build_ion_aware_impedance_protocol(
        uncertified,
        np.array([1.0]),
    )

    with pytest.raises(
        impedance.IonAwareImpedanceCapabilityError,
        match="numerical DC certificate",
    ):
        impedance.run_ion_aware_impedance(
            x,
            stack,
            protocol,
            dc_state=uncertified,
            mat=mat,
        )


def test_reference_lane_keeps_contact_thermodynamics_as_a_strict_axis(dc_fixture):
    stack, x, mat, dc_state = dc_fixture
    protocol = impedance.build_ion_aware_impedance_protocol(
        dc_state,
        np.array([1.0]),
    )

    with pytest.raises(
        impedance.IonAwareImpedanceCapabilityError,
        match="contact thermodynamics",
    ):
        impedance.run_ion_aware_impedance(
            x,
            stack,
            protocol,
            dc_state=dc_state,
            mat=mat,
            require_contact_certificate=True,
        )


def test_reference_lane_retains_failed_gate_evidence_in_diagnostic_mode(dc_fixture):
    stack, x, mat, dc_state = dc_fixture
    baseline = impedance.build_ion_aware_impedance_protocol(
        dc_state,
        np.array([1.0]),
    )
    protocol = replace(baseline, max_mass_matrix_relative_error=1.0e-30)

    with pytest.raises(
        impedance.IonAwareImpedanceCertificationError,
        match="mass_matrix_log_coordinate_identity_failed",
    ) as captured:
        impedance.run_ion_aware_impedance(
            x,
            stack,
            protocol,
            dc_state=dc_state,
            mat=mat,
        )
    assert not captured.value.result.certificate.numerically_certified

    diagnostic = impedance.run_ion_aware_impedance(
        x,
        stack,
        protocol,
        dc_state=dc_state,
        mat=mat,
        require_numerical_certificate=False,
    )
    assert "mass_matrix_log_coordinate_identity_failed" in (
        diagnostic.certificate.reasons
    )
    assert all(
        not item.numerically_certified
        and "mass_matrix_log_coordinate_identity_failed" in item.reasons
        for item in diagnostic.certificate.frequency_point_certificates
    )
