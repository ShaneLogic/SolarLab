from __future__ import annotations

import numpy as np
import pytest

from perovskite_sim.constants import Q
from perovskite_sim.models.defects import ACCEPTOR, DONOR, NEUTRAL
from perovskite_sim.physics.trap_kinetics import (
    TrapReservoirKinetics,
    TrapReservoirState,
    evaluate_trap_dc_operating_point,
    linearize_trap_kinetics,
)
from perovskite_sim.physics.trap_transient import (
    TrapTransientError,
    constant_reservoir_trap_trace,
    evaluate_trap_transient,
    linearize_trap_transient,
    occupancy_from_logit,
    occupancy_logit,
)


def _kinetics(scale: float = 1.0) -> TrapReservoirKinetics:
    return TrapReservoirKinetics(
        identifier="local/transient",
        electron_capture_coefficients_m3_s=scale * np.array([2.0e-14]),
        hole_capture_coefficients_m3_s=scale * np.array([5.0e-15]),
        electron_reference_densities_m3=np.array([1.0e14]),
        hole_reference_densities_m3=np.array([2.0e15]),
    )


def _state() -> TrapReservoirState:
    return TrapReservoirState(
        electron_densities_m3=np.array([3.0e20]),
        hole_densities_m3=np.array([4.0e16]),
    )


@pytest.mark.parametrize("occupancy", [1.0e-12, 0.2, 0.8, 1.0 - 1.0e-12])
def test_logit_coordinate_round_trip_is_strictly_bounded(occupancy):
    restored = occupancy_from_logit(occupancy_logit(occupancy))
    assert restored == pytest.approx(occupancy, rel=1.0e-15)
    assert 0.0 < restored < 1.0


@pytest.mark.parametrize("occupancy", [0.0, 1.0, -0.1, 1.1, np.nan])
def test_logit_coordinate_rejects_endpoints_and_nonfinite_values(occupancy):
    with pytest.raises(TrapTransientError, match="strictly inside"):
        occupancy_logit(occupancy)
    with pytest.raises(TrapTransientError):
        evaluate_trap_transient(
            _kinetics(),
            _state(),
            occupancy,
            charge_transition=ACCEPTOR,
        )


def test_logit_coordinate_fails_closed_when_float_reconstruction_saturates():
    with pytest.raises(TrapTransientError, match="saturated"):
        occupancy_from_logit(1.0e3)
    with pytest.raises(TrapTransientError, match="saturated"):
        occupancy_from_logit(-1.0e3)


def test_acceptor_and_donor_share_charge_rate_and_close_free_carrier_charge():
    arguments = (_kinetics(), _state(), 0.31)
    acceptor = evaluate_trap_transient(
        *arguments,
        charge_transition=ACCEPTOR,
    )
    donor = evaluate_trap_transient(
        *arguments,
        charge_transition=DONOR,
    )

    assert acceptor.trap_charge_C == pytest.approx(-Q * acceptor.occupancy)
    assert donor.trap_charge_C == pytest.approx(Q * (1.0 - donor.occupancy))
    assert acceptor.trap_charge_rate_C_s == donor.trap_charge_rate_C_s
    assert acceptor.carrier_charge_rate_C_s == donor.carrier_charge_rate_C_s
    assert acceptor.trap_charge_rate_C_s == pytest.approx(
        -acceptor.carrier_charge_rate_C_s,
        abs=0.0,
    )
    assert acceptor.charge_balance_residual_C_s == 0.0
    assert acceptor.charge_balance_relative_error == 0.0


def test_two_sided_reservoir_capture_difference_is_the_occupancy_rate():
    kinetics = TrapReservoirKinetics(
        identifier="interface/shared/transient",
        electron_capture_coefficients_m3_s=np.array([2.0e-15, 4.0e-15]),
        hole_capture_coefficients_m3_s=np.array([3.0e-15, 1.0e-15]),
        electron_reference_densities_m3=np.array([1.0e12, 2.0e13]),
        hole_reference_densities_m3=np.array([3.0e16, 4.0e15]),
    )
    state = TrapReservoirState(
        electron_densities_m3=np.array([1.0e19, 5.0e18]),
        hole_densities_m3=np.array([2.0e17, 8.0e16]),
    )
    evaluation = evaluate_trap_transient(
        kinetics,
        state,
        0.43,
        charge_transition=ACCEPTOR,
    )

    assert evaluation.electron_capture_rates_s1.shape == (2,)
    assert evaluation.hole_capture_rates_s1.shape == (2,)
    assert evaluation.occupancy_rate_s1 == pytest.approx(
        np.sum(evaluation.electron_capture_rates_s1)
        - np.sum(evaluation.hole_capture_rates_s1),
        rel=0.0,
        abs=0.0,
    )


def test_quasi_steady_occupancy_recovers_detailed_capture_balance():
    kinetics = _kinetics()
    state = _state()
    steady = evaluate_trap_dc_operating_point(kinetics, state)
    evaluation = evaluate_trap_transient(
        kinetics,
        state,
        steady.occupancy,
        charge_transition=DONOR,
    )

    assert abs(evaluation.occupancy_rate_s1) <= (
        2.0e-15 * evaluation.relaxation_rate_s1
    )
    assert np.sum(evaluation.electron_capture_rates_s1) == pytest.approx(
        np.sum(evaluation.hole_capture_rates_s1),
        rel=2.0e-12,
    )


def test_qss_transient_tangent_is_identical_to_frequency_domain_tangent():
    kinetics = _kinetics()
    state = _state()
    steady = evaluate_trap_dc_operating_point(kinetics, state)
    transient = linearize_trap_transient(
        kinetics,
        state,
        steady.occupancy,
        charge_transition=ACCEPTOR,
    )
    frequency = linearize_trap_kinetics(kinetics, state, steady)

    np.testing.assert_array_equal(
        transient.occupancy_rate_electron_density_derivative_m3_s,
        frequency.electron_density_forcing_m3_s,
    )
    np.testing.assert_array_equal(
        transient.occupancy_rate_hole_density_derivative_m3_s,
        frequency.hole_density_forcing_m3_s,
    )
    np.testing.assert_array_equal(
        transient.electron_capture_occupancy_derivative_s1,
        frequency.electron_capture_occupancy_derivative_s1,
    )
    np.testing.assert_array_equal(
        transient.hole_capture_occupancy_derivative_s1,
        frequency.hole_capture_occupancy_derivative_s1,
    )


def test_analytic_logit_and_density_tangents_match_centered_differences():
    kinetics = _kinetics()
    state = _state()
    occupancy = 0.37
    coordinate = occupancy_logit(occupancy)
    tangent = linearize_trap_transient(
        kinetics,
        state,
        occupancy,
        charge_transition=ACCEPTOR,
    )

    coordinate_step = 1.0e-6
    numerical_coordinate = (
        evaluate_trap_transient(
            kinetics,
            state,
            occupancy_from_logit(coordinate + coordinate_step),
            charge_transition=ACCEPTOR,
        ).logit_rate_s1
        - evaluate_trap_transient(
            kinetics,
            state,
            occupancy_from_logit(coordinate - coordinate_step),
            charge_transition=ACCEPTOR,
        ).logit_rate_s1
    ) / (2.0 * coordinate_step)
    assert numerical_coordinate == pytest.approx(
        tangent.logit_rate_logit_derivative_s1,
        rel=2.0e-10,
    )

    n0 = state.electron_densities_m3[0]
    density_step = 1.0e16

    def logit_rate(n: float) -> float:
        varied = TrapReservoirState(np.array([n]), state.hole_densities_m3)
        return evaluate_trap_transient(
            kinetics,
            varied,
            occupancy,
            charge_transition=ACCEPTOR,
        ).logit_rate_s1

    numerical_density = (
        logit_rate(n0 + density_step) - logit_rate(n0 - density_step)
    ) / (2.0 * density_step)
    assert numerical_density == pytest.approx(
        tangent.logit_rate_electron_density_derivative_m3_s[0],
        rel=2.0e-10,
    )


def test_capture_occupancy_tangents_match_centered_differences():
    kinetics = _kinetics()
    state = _state()
    occupancy = 0.41
    tangent = linearize_trap_transient(
        kinetics,
        state,
        occupancy,
        charge_transition=DONOR,
    )
    step = 1.0e-6
    low = evaluate_trap_transient(
        kinetics,
        state,
        occupancy - step,
        charge_transition=DONOR,
    )
    high = evaluate_trap_transient(
        kinetics,
        state,
        occupancy + step,
        charge_transition=DONOR,
    )
    electron = (high.electron_capture_rates_s1 - low.electron_capture_rates_s1) / (
        2.0 * step
    )
    hole = (high.hole_capture_rates_s1 - low.hole_capture_rates_s1) / (
        2.0 * step
    )
    np.testing.assert_allclose(
        electron,
        tangent.electron_capture_occupancy_derivative_s1,
        rtol=2.0e-10,
    )
    np.testing.assert_allclose(
        hole,
        tangent.hole_capture_occupancy_derivative_s1,
        rtol=2.0e-10,
    )


def test_constant_reservoir_trace_matches_closed_form_and_is_immutable():
    kinetics = _kinetics()
    state = _state()
    initial = 0.17
    times = np.linspace(0.0, 2.0e-6, 31)
    trace = constant_reservoir_trap_trace(
        kinetics,
        state,
        initial,
        times,
        charge_transition=ACCEPTOR,
    )
    expected = trace.quasi_steady_occupancy + (
        initial - trace.quasi_steady_occupancy
    ) * np.exp(-trace.relaxation_rate_s1 * times)
    expected[0] = initial

    np.testing.assert_allclose(trace.occupancy, expected, rtol=0.0, atol=0.0)
    assert trace.occupancy[0] == initial
    assert np.all(np.diff(trace.occupancy) > 0.0)
    assert trace.maximum_charge_balance_relative_error == 0.0
    with pytest.raises(ValueError):
        trace.occupancy[0] = 0.5


def test_dynamic_transient_rejects_neutral_transition_and_bad_time_grid():
    with pytest.raises(TrapTransientError, match="acceptor or donor"):
        evaluate_trap_transient(
            _kinetics(),
            _state(),
            0.5,
            charge_transition=NEUTRAL,
        )
    with pytest.raises(TrapTransientError, match="strictly increasing"):
        constant_reservoir_trap_trace(
            _kinetics(),
            _state(),
            0.5,
            np.array([0.0, 1.0, 1.0]),
            charge_transition=ACCEPTOR,
        )
