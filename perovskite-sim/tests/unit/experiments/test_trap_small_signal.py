from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from perovskite_sim.experiments.trap_small_signal import (
    TrapSmallSignalCertificationError,
    TrapSmallSignalPerturbation,
    TrapSmallSignalProtocol,
    TrapSmallSignalProtocolError,
    assess_trap_frequency_window,
    build_trap_small_signal_protocol,
    run_local_trap_small_signal,
)
from perovskite_sim.models.defects import ACCEPTOR, DONOR, NEUTRAL
from perovskite_sim.physics.trap_kinetics import (
    TrapKineticsCertificationError,
    TrapReservoirKinetics,
    TrapReservoirState,
    evaluate_trap_dc_operating_point,
)


def _case():
    kinetics = TrapReservoirKinetics(
        identifier="bulk/test",
        electron_capture_coefficients_m3_s=np.array([2.0e-14]),
        hole_capture_coefficients_m3_s=np.array([5.0e-15]),
        electron_reference_densities_m3=np.array([1.0e14]),
        hole_reference_densities_m3=np.array([2.0e15]),
    )
    state = TrapReservoirState(
        electron_densities_m3=np.array([3.0e20]),
        hole_densities_m3=np.array([4.0e16]),
    )
    point = evaluate_trap_dc_operating_point(kinetics, state)
    perturbation = TrapSmallSignalPerturbation(
        electron_density_amplitude_m3_per_V=(1.0e18,),
        hole_density_amplitude_m3_per_V=(0.0,),
    )
    corner = point.relaxation_rate_s1 / (2.0 * np.pi)
    frequencies = np.geomspace(corner * 1.0e-2, corner * 1.0e2, 9)
    protocol = build_trap_small_signal_protocol(
        kinetics,
        state,
        point,
        perturbation,
        frequencies,
        charge_transition=ACCEPTOR,
    )
    return kinetics, state, point, perturbation, protocol


def test_perturbation_and_protocol_strict_round_trip_and_hash():
    _, _, _, perturbation, protocol = _case()
    assert (
        TrapSmallSignalPerturbation.from_json(perturbation.canonical_json())
        == perturbation
    )
    assert TrapSmallSignalProtocol.from_json(protocol.canonical_json()) == protocol
    assert len(perturbation.sha256) == 64
    assert len(protocol.protocol_hash) == 64

    changed = dataclasses.replace(protocol, charge_transition=DONOR)
    assert changed.protocol_hash != protocol.protocol_hash
    changed_perturbation = dataclasses.replace(
        perturbation,
        electron_density_amplitude_m3_per_V=(2.0e18,),
    )
    assert changed_perturbation.sha256 != perturbation.sha256


@pytest.mark.parametrize("target", ["protocol", "perturbation"])
def test_unknown_and_missing_schema_keys_fail_closed(target):
    _, _, _, perturbation, protocol = _case()
    value = protocol.to_dict() if target == "protocol" else perturbation.to_dict()
    parser = (
        TrapSmallSignalProtocol.from_dict
        if target == "protocol"
        else TrapSmallSignalPerturbation.from_dict
    )
    missing = dict(value)
    missing.pop(next(iter(missing)))
    with pytest.raises(TrapSmallSignalProtocolError, match="keys do not match"):
        parser(missing)
    unknown = {**value, "unhashed_claim": "certified"}
    with pytest.raises(TrapSmallSignalProtocolError, match="keys do not match"):
        parser(unknown)


def test_protocol_rejects_unsorted_sparse_and_nonfinite_frequency_requests():
    _, _, _, _, protocol = _case()
    with pytest.raises(TrapSmallSignalProtocolError, match="three increasing"):
        dataclasses.replace(protocol, frequencies_Hz=(1.0, 3.0, 2.0))
    with pytest.raises(TrapSmallSignalProtocolError, match="three increasing"):
        dataclasses.replace(protocol, frequencies_Hz=(1.0, 2.0))
    with pytest.raises(TrapSmallSignalProtocolError, match="finite"):
        dataclasses.replace(protocol, frequencies_Hz=(1.0, 2.0, float("nan")))


def test_protocol_rejects_recombination_only_neutral_transition():
    _, _, _, _, protocol = _case()
    with pytest.raises(
        TrapSmallSignalProtocolError,
        match="requires an acceptor or donor",
    ):
        dataclasses.replace(protocol, charge_transition=NEUTRAL)


def test_builder_rejects_reservoir_count_and_operating_identity_mismatch():
    kinetics, state, point, perturbation, protocol = _case()
    wrong_perturbation = TrapSmallSignalPerturbation(
        electron_density_amplitude_m3_per_V=(1.0, 2.0),
        hole_density_amplitude_m3_per_V=(1.0,),
    )
    with pytest.raises(TrapSmallSignalProtocolError, match="reservoir counts"):
        build_trap_small_signal_protocol(
            kinetics,
            state,
            point,
            wrong_perturbation,
            protocol.frequencies_Hz,
            charge_transition=ACCEPTOR,
        )
    changed_state = TrapReservoirState(
        electron_densities_m3=state.electron_densities_m3 * 2.0,
        hole_densities_m3=state.hole_densities_m3,
    )
    with pytest.raises(TrapSmallSignalProtocolError, match="does not match trap state"):
        build_trap_small_signal_protocol(
            kinetics,
            changed_state,
            point,
            perturbation,
            protocol.frequencies_Hz,
            charge_transition=ACCEPTOR,
        )


def test_local_protocol_certifies_debye_limits_and_frequency_window():
    kinetics, state, point, perturbation, protocol = _case()
    result = run_local_trap_small_signal(
        kinetics,
        state,
        point,
        perturbation,
        protocol,
    )
    certificate = result.certificate
    assert certificate.certified
    assert certificate.operating_point_certified
    assert certificate.numerical_response_certified
    assert certificate.low_frequency_limit_certified
    assert certificate.high_frequency_limit_certified
    assert certificate.frequency_window_certified
    assert certificate.low_frequency_relative_error < 1.01e-2
    assert certificate.high_frequency_frozen_ratio < 1.01e-2
    assert certificate.max_linear_solve_backward_error < 1.0e-15
    assert certificate.max_local_charge_conservation_relative_error < 1.0e-15
    assert certificate.max_conjugate_symmetry_relative_error == 0.0
    assert certificate.reasons == ()


def test_frequency_window_uses_actual_relaxation_rate_not_ten_hz_default():
    relaxation_rate = 2.0 * np.pi * 1.0e-5
    covered = assess_trap_frequency_window(
        np.geomspace(1.0e-7, 1.0e-3, 9),
        relaxation_rate,
        branch_margin_decades=2.0,
        max_sampling_gap_decades=0.5,
    )
    assert covered.relaxation_frequency_Hz == pytest.approx(1.0e-5)
    assert covered.certified

    omitted = assess_trap_frequency_window(
        np.geomspace(10.0, 1.0e5, 9),
        relaxation_rate,
        branch_margin_decades=2.0,
        max_sampling_gap_decades=0.5,
    )
    assert not omitted.certified
    assert not omitted.low_frequency_limit_covered
    assert not omitted.relaxation_bracketed


def test_incomplete_frequency_window_returns_partial_or_raises_as_requested():
    kinetics, state, point, perturbation, protocol = _case()
    corner = point.relaxation_rate_s1 / (2.0 * np.pi)
    incomplete = dataclasses.replace(
        protocol,
        frequencies_Hz=tuple(np.geomspace(corner, corner * 1.0e2, 5)),
    )
    diagnostic = run_local_trap_small_signal(
        kinetics,
        state,
        point,
        perturbation,
        incomplete,
        require_certificate=False,
    )
    assert not diagnostic.certificate.certified
    assert "low_frequency_limit_not_covered" in diagnostic.certificate.reasons
    assert "trap_relaxation_not_bracketed" in diagnostic.certificate.reasons
    with pytest.raises(TrapSmallSignalCertificationError) as exc_info:
        run_local_trap_small_signal(
            kinetics,
            state,
            point,
            perturbation,
            incomplete,
        )
    assert exc_info.value.result.certificate == diagnostic.certificate


def test_run_rejects_protocol_identity_changes_before_computation():
    kinetics, state, point, perturbation, protocol = _case()
    mismatched = dataclasses.replace(protocol, perturbation_sha256="0" * 64)
    with pytest.raises(TrapSmallSignalProtocolError, match="perturbation"):
        run_local_trap_small_signal(
            kinetics,
            state,
            point,
            perturbation,
            mismatched,
        )


def test_uncertified_dc_occupancy_cannot_enter_frequency_response():
    kinetics, state, point, perturbation, protocol = _case()
    diagnostic = evaluate_trap_dc_operating_point(
        kinetics,
        state,
        occupancy=point.occupancy - 0.1,
        require_certified=False,
    )
    bound = dataclasses.replace(
        protocol,
        operating_point_sha256=diagnostic.sha256,
    )
    with pytest.raises(TrapKineticsCertificationError, match="certified"):
        run_local_trap_small_signal(
            kinetics,
            state,
            diagnostic,
            perturbation,
            bound,
        )


def test_nonzero_reservoir_perturbation_with_zero_net_trap_forcing_is_rejected():
    kinetics, state, point, _, protocol = _case()
    c_n = kinetics.electron_capture_coefficients_m3_s[0]
    c_p = kinetics.hole_capture_coefficients_m3_s[0]
    occupancy = point.occupancy
    electron = 1.0e18
    hole = c_n * (1.0 - occupancy) * electron / (c_p * occupancy)
    cancellation = TrapSmallSignalPerturbation(
        electron_density_amplitude_m3_per_V=(electron,),
        hole_density_amplitude_m3_per_V=(hole,),
    )
    cancellation_protocol = build_trap_small_signal_protocol(
        kinetics,
        state,
        point,
        cancellation,
        protocol.frequencies_Hz,
        charge_transition=ACCEPTOR,
    )
    with pytest.raises(TrapSmallSignalProtocolError, match="zero net"):
        run_local_trap_small_signal(
            kinetics,
            state,
            point,
            cancellation,
            cancellation_protocol,
        )
