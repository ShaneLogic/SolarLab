from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from perovskite_sim.constants import Q
from perovskite_sim.models.defects import (
    ACCEPTOR,
    INTEGRATED_TOTAL,
    NEUTRAL,
    NEUTRAL_WHEN_EMPTY,
    SINGLE_LEVEL,
    DONOR,
    BulkDefectDistribution,
    BulkDefectKinetics,
    BulkDefectSpecies,
)
from perovskite_sim.physics.defect_closure import (
    evaluate_monovalent_defect_closure,
)
from perovskite_sim.physics.trap_kinetics import (
    TrapKineticsCertificationError,
    TrapKineticsError,
    TrapReservoirKinetics,
    TrapReservoirState,
    evaluate_trap_dc_operating_point,
    fixed_quasi_fermi_density_response_per_potential,
    linearize_trap_kinetics,
    solve_trap_frequency_response,
)


def _bulk_kinetics() -> TrapReservoirKinetics:
    return TrapReservoirKinetics(
        identifier="bulk/single",
        electron_capture_coefficients_m3_s=np.array([2.0e-14]),
        hole_capture_coefficients_m3_s=np.array([5.0e-15]),
        electron_reference_densities_m3=np.array([1.0e14]),
        hole_reference_densities_m3=np.array([2.0e15]),
    )


def _bulk_state() -> TrapReservoirState:
    return TrapReservoirState(
        electron_densities_m3=np.array([3.0e20]),
        hole_densities_m3=np.array([4.0e16]),
    )


@pytest.mark.parametrize("kind", ["kinetics", "state"])
def test_reservoir_contracts_have_strict_canonical_round_trip(kind):
    original = _bulk_kinetics() if kind == "kinetics" else _bulk_state()
    restored = type(original).from_json(original.canonical_json())
    assert restored.to_dict() == original.to_dict()
    assert restored.sha256 == original.sha256
    unknown = {**original.to_dict(), "unhashed_claim": "certified"}
    with pytest.raises(TrapKineticsError, match="keys do not match"):
        type(original).from_dict(unknown)
    missing = original.to_dict()
    missing.pop("schema_version")
    with pytest.raises(TrapKineticsError, match="keys do not match"):
        type(original).from_dict(missing)


def test_dc_operating_point_matches_closed_form_and_capture_balance():
    kinetics = _bulk_kinetics()
    state = _bulk_state()
    point = evaluate_trap_dc_operating_point(kinetics, state)

    c_n = kinetics.electron_capture_coefficients_m3_s[0]
    c_p = kinetics.hole_capture_coefficients_m3_s[0]
    n1 = kinetics.electron_reference_densities_m3[0]
    p1 = kinetics.hole_reference_densities_m3[0]
    n = state.electron_densities_m3[0]
    p = state.hole_densities_m3[0]
    filled = c_n * n + c_p * p1
    empty = c_n * n1 + c_p * p

    assert point.filled_rate_s1 == pytest.approx(filled)
    assert point.empty_rate_s1 == pytest.approx(empty)
    assert point.relaxation_rate_s1 == pytest.approx(filled + empty)
    assert point.occupancy == pytest.approx(filled / (filled + empty))
    assert point.electron_capture_rates_s1[0] == pytest.approx(
        point.hole_capture_rates_s1[0], rel=1.0e-12
    )
    assert point.normalized_residual < 1.0e-15
    assert point.certified


def test_local_kinetics_matches_existing_single_level_dc_defect_closure():
    species = BulkDefectSpecies(
        name="acceptor",
        distribution=BulkDefectDistribution(
            kind=SINGLE_LEVEL,
            normalization=INTEGRATED_TOTAL,
            total_density_m3=3.0e21,
            center_eV_above_vb=0.62,
        ),
        charge_transition=ACCEPTOR,
        neutral_reference=NEUTRAL_WHEN_EMPTY,
        kinetics=BulkDefectKinetics(
            sigma_n_m2=2.0e-19,
            sigma_p_m2=7.0e-20,
            thermal_velocity_n_m_s=1.3e5,
            thermal_velocity_p_m_s=8.0e4,
        ),
    )
    n = 4.0e19
    p = 7.0e18
    closure = evaluate_monovalent_defect_closure(
        n,
        p,
        (species,),
        band_gap_eV=1.5,
        effective_conduction_dos_m3=2.4e25,
        effective_valence_dos_m3=1.1e25,
        temperature_K=300.0,
    )
    kinetics = TrapReservoirKinetics(
        identifier="bulk/acceptor",
        electron_capture_coefficients_m3_s=closure.capture_n_m3_s,
        hole_capture_coefficients_m3_s=closure.capture_p_m3_s,
        electron_reference_densities_m3=closure.n1_m3,
        hole_reference_densities_m3=closure.p1_m3,
    )
    point = evaluate_trap_dc_operating_point(
        kinetics,
        TrapReservoirState(np.array([n]), np.array([p])),
    )

    assert point.occupancy == pytest.approx(closure.occupancy[0].item())
    assert point.relaxation_rate_s1 == pytest.approx(
        closure.kinetic_denominator_s1[0].item()
    )


def test_two_sided_reservoirs_share_one_occupancy_and_sum_capture():
    kinetics = TrapReservoirKinetics(
        identifier="interface/shared",
        electron_capture_coefficients_m3_s=np.array([3.0e-15, 3.0e-15]),
        hole_capture_coefficients_m3_s=np.array([7.0e-16, 7.0e-16]),
        electron_reference_densities_m3=np.array([2.0e12, 8.0e13]),
        hole_reference_densities_m3=np.array([4.0e17, 5.0e15]),
    )
    state = TrapReservoirState(
        electron_densities_m3=np.array([6.0e19, 2.0e18]),
        hole_densities_m3=np.array([3.0e16, 9.0e17]),
    )
    point = evaluate_trap_dc_operating_point(kinetics, state)

    filled = np.dot(
        kinetics.electron_capture_coefficients_m3_s,
        state.electron_densities_m3,
    ) + np.dot(
        kinetics.hole_capture_coefficients_m3_s,
        kinetics.hole_reference_densities_m3,
    )
    empty = np.dot(
        kinetics.electron_capture_coefficients_m3_s,
        kinetics.electron_reference_densities_m3,
    ) + np.dot(
        kinetics.hole_capture_coefficients_m3_s,
        state.hole_densities_m3,
    )
    assert point.occupancy == pytest.approx(filled / (filled + empty))
    assert np.sum(point.electron_capture_rates_s1) == pytest.approx(
        np.sum(point.hole_capture_rates_s1), rel=1.0e-12
    )


def test_supplied_nonstationary_occupancy_fails_closed_but_can_be_diagnosed():
    kinetics = _bulk_kinetics()
    state = _bulk_state()
    steady = evaluate_trap_dc_operating_point(kinetics, state)

    with pytest.raises(TrapKineticsCertificationError, match="did not certify"):
        evaluate_trap_dc_operating_point(
            kinetics,
            state,
            occupancy=steady.occupancy - 0.1,
        )
    diagnostic = evaluate_trap_dc_operating_point(
        kinetics,
        state,
        occupancy=steady.occupancy - 0.1,
        require_certified=False,
    )
    assert not diagnostic.certified
    assert diagnostic.normalized_residual == pytest.approx(0.1)


def test_forged_certified_operating_point_is_recomputed_before_linearization():
    kinetics = _bulk_kinetics()
    state = _bulk_state()
    point = evaluate_trap_dc_operating_point(kinetics, state)
    forged = dataclasses.replace(
        point,
        filled_rate_s1=point.filled_rate_s1 * 2.0,
        relaxation_rate_s1=(point.filled_rate_s1 * 2.0 + point.empty_rate_s1),
        normalized_residual=(
            abs(point.occupancy_rate_residual_s1)
            / (point.filled_rate_s1 * 2.0 + point.empty_rate_s1)
        ),
    )
    with pytest.raises(TrapKineticsError, match="content does not match"):
        linearize_trap_kinetics(kinetics, state, forged)


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (
            TrapReservoirState(np.array([1.0e20]), np.array([0.0])),
            1.0,
        ),
        (
            TrapReservoirState(np.array([0.0]), np.array([1.0e20])),
            0.0,
        ),
    ],
)
def test_exact_empty_and_filled_endpoints_do_not_require_clipping(state, expected):
    kinetics = TrapReservoirKinetics(
        identifier="endpoint",
        electron_capture_coefficients_m3_s=np.array([1.0e-14]),
        hole_capture_coefficients_m3_s=np.array([1.0e-14]),
        electron_reference_densities_m3=np.array([0.0]),
        hole_reference_densities_m3=np.array([0.0]),
    )
    point = evaluate_trap_dc_operating_point(kinetics, state)
    assert point.occupancy == expected
    assert point.certified


def test_zero_relaxation_and_misaligned_reservoirs_are_rejected():
    kinetics = TrapReservoirKinetics(
        identifier="zero-state",
        electron_capture_coefficients_m3_s=np.array([1.0]),
        hole_capture_coefficients_m3_s=np.array([0.0]),
        electron_reference_densities_m3=np.array([0.0]),
        hole_reference_densities_m3=np.array([0.0]),
    )
    with pytest.raises(TrapKineticsError, match="relaxation rate"):
        evaluate_trap_dc_operating_point(
            kinetics,
            TrapReservoirState(np.array([0.0]), np.array([0.0])),
        )
    with pytest.raises(TrapKineticsError, match="electron state"):
        evaluate_trap_dc_operating_point(
            _bulk_kinetics(),
            TrapReservoirState(np.ones(2), np.ones(1)),
        )


def test_quasistatic_occupancy_derivatives_match_resolved_centered_difference():
    kinetics = _bulk_kinetics()
    state = _bulk_state()
    point = evaluate_trap_dc_operating_point(kinetics, state)
    tangent = linearize_trap_kinetics(kinetics, state, point)
    expected_n = tangent.electron_density_forcing_m3_s[0] / (point.relaxation_rate_s1)
    expected_p = tangent.hole_density_forcing_m3_s[0] / point.relaxation_rate_s1

    def occupancy(n: float, p: float) -> float:
        varied = TrapReservoirState(np.array([n]), np.array([p]))
        return evaluate_trap_dc_operating_point(kinetics, varied).occupancy

    n0 = state.electron_densities_m3[0]
    p0 = state.hole_densities_m3[0]
    dn = n0 * 1.0e-5
    dp = p0 * 1.0e-5
    numerical_n = (occupancy(n0 + dn, p0) - occupancy(n0 - dn, p0)) / (2.0 * dn)
    numerical_p = (occupancy(n0, p0 + dp) - occupancy(n0, p0 - dp)) / (2.0 * dp)
    assert numerical_n == pytest.approx(expected_n, rel=1.0e-8)
    assert numerical_p == pytest.approx(expected_p, rel=1.0e-8)


def test_fixed_qf_potential_derivative_uses_density_response_only():
    state = _bulk_state()
    dn_dphi, dp_dphi = fixed_quasi_fermi_density_response_per_potential(state, 0.025)
    np.testing.assert_array_equal(dn_dphi, state.electron_densities_m3 / 0.025)
    np.testing.assert_array_equal(dp_dphi, -state.hole_densities_m3 / 0.025)


def test_frequency_response_is_exact_single_pole_debye_relaxation():
    kinetics = _bulk_kinetics()
    state = _bulk_state()
    point = evaluate_trap_dc_operating_point(kinetics, state)
    corner = point.relaxation_rate_s1 / (2.0 * np.pi)
    frequencies = corner * np.array([1.0e-4, 1.0, 1.0e4])
    response = solve_trap_frequency_response(
        kinetics,
        state,
        point,
        frequencies,
        np.array([1.0e18]),
        np.array([0.0]),
        charge_transition=ACCEPTOR,
    )
    expected = response.quasistatic_occupancy_response_per_V / (
        1.0 + 1j * frequencies / corner
    )
    np.testing.assert_allclose(
        response.occupancy_response_per_V, expected, rtol=1.0e-15
    )
    assert abs(response.occupancy_response_per_V[0] / expected[0] - 1.0) < 1.0e-15
    assert abs(response.occupancy_response_per_V[-1]) < (
        2.0e-4 * abs(response.quasistatic_occupancy_response_per_V[-1])
    )
    assert np.max(response.linear_solve_backward_error) < 1.0e-15


def test_positive_and_negative_frequencies_are_conjugates_for_real_forcing():
    kinetics = _bulk_kinetics()
    state = _bulk_state()
    point = evaluate_trap_dc_operating_point(kinetics, state)
    frequencies = np.array([1.0e2, 1.0e5, 1.0e8])
    positive = solve_trap_frequency_response(
        kinetics,
        state,
        point,
        frequencies,
        np.array([2.0e17]),
        np.array([-3.0e15]),
        charge_transition=DONOR,
    )
    negative = solve_trap_frequency_response(
        kinetics,
        state,
        point,
        -frequencies,
        np.array([2.0e17]),
        np.array([-3.0e15]),
        charge_transition=DONOR,
    )
    np.testing.assert_array_equal(
        positive.occupancy_response_per_V,
        np.conjugate(negative.occupancy_response_per_V),
    )
    np.testing.assert_array_equal(
        positive.electron_capture_response_s1_per_V,
        np.conjugate(negative.electron_capture_response_s1_per_V),
    )
    np.testing.assert_array_equal(
        positive.hole_capture_response_s1_per_V,
        np.conjugate(negative.hole_capture_response_s1_per_V),
    )


def test_acceptor_and_donor_have_same_dynamic_charge_sign_while_neutral_is_zero():
    kinetics = _bulk_kinetics()
    state = _bulk_state()
    point = evaluate_trap_dc_operating_point(kinetics, state)
    arguments = (
        kinetics,
        state,
        point,
        np.array([1.0e6]),
        np.array([1.0e18]),
        np.array([0.0]),
    )
    acceptor = solve_trap_frequency_response(*arguments, charge_transition=ACCEPTOR)
    donor = solve_trap_frequency_response(*arguments, charge_transition=DONOR)
    neutral = solve_trap_frequency_response(*arguments, charge_transition=NEUTRAL)

    np.testing.assert_array_equal(
        acceptor.charge_per_trap_response_C_per_V,
        donor.charge_per_trap_response_C_per_V,
    )
    np.testing.assert_allclose(
        acceptor.charge_per_trap_response_C_per_V,
        -Q * acceptor.occupancy_response_per_V,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        neutral.charge_per_trap_response_C_per_V,
        np.zeros(1, dtype=complex),
    )


def test_capture_and_trap_storage_close_local_charge_conservation():
    kinetics = TrapReservoirKinetics(
        identifier="interface/shared",
        electron_capture_coefficients_m3_s=np.array([2.0e-15, 4.0e-15]),
        hole_capture_coefficients_m3_s=np.array([3.0e-15, 1.0e-15]),
        electron_reference_densities_m3=np.array([1.0e12, 2.0e13]),
        hole_reference_densities_m3=np.array([3.0e16, 4.0e15]),
    )
    state = TrapReservoirState(
        electron_densities_m3=np.array([1.0e19, 5.0e18]),
        hole_densities_m3=np.array([2.0e17, 8.0e16]),
    )
    point = evaluate_trap_dc_operating_point(kinetics, state)
    frequencies = np.geomspace(1.0, 1.0e10, 17)
    response = solve_trap_frequency_response(
        kinetics,
        state,
        point,
        frequencies,
        np.array([1.0e17, -3.0e16]),
        np.array([2.0e15, 7.0e15]),
        charge_transition=ACCEPTOR,
    )
    scale = (
        np.sum(np.abs(response.electron_capture_response_s1_per_V), axis=1)
        + np.sum(np.abs(response.hole_capture_response_s1_per_V), axis=1)
        + np.abs(2j * np.pi * frequencies * response.occupancy_response_per_V)
    )
    relative = np.abs(response.occupancy_balance_residual_s1_per_V) / scale
    assert np.max(relative) < 5.0e-16


def test_frequency_response_arrays_are_immutable():
    kinetics = _bulk_kinetics()
    state = _bulk_state()
    point = evaluate_trap_dc_operating_point(kinetics, state)
    response = solve_trap_frequency_response(
        kinetics,
        state,
        point,
        np.array([1.0]),
        np.array([1.0]),
        np.array([0.0]),
        charge_transition=NEUTRAL,
    )
    with pytest.raises(ValueError, match="read-only"):
        response.occupancy_response_per_V[0] = 0.0
