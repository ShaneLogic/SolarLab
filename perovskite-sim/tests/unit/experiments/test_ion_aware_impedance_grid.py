from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import perovskite_sim.experiments.ion_aware_impedance_grid as grid_module
from perovskite_sim.experiments.ion_aware_impedance import (
    IonAwareImpedanceProtocol,
)


def _impedance_protocols(
    frequencies: tuple[float, ...] = (1.0, 10.0),
) -> tuple[IonAwareImpedanceProtocol, ...]:
    return tuple(
        IonAwareImpedanceProtocol(
            V_dc=0.9,
            illuminated=True,
            temperature_K=300.0,
            dc_protocol_sha256="a" * 64,
            dc_state_sha256=character * 64,
            frequencies_Hz=frequencies,
        )
        for character in ("1", "2", "3")
    )


def _grid_hashes() -> tuple[str, ...]:
    return tuple(
        grid_module.ion_aware_impedance_grid_sha256(
            np.linspace(0.0, 1.0, count)
        )
        for count in (10, 20, 30)
    )


def _protocol(
    *,
    max_magnitude: float = 0.02,
    max_phase: float = 1.0,
) -> grid_module.IonAwareImpedanceGridProtocol:
    return grid_module.IonAwareImpedanceGridProtocol(
        grid_node_counts=(10, 20, 30),
        grid_sha256s=_grid_hashes(),
        impedance_protocols=_impedance_protocols(),
        max_grid_impedance_magnitude_relative_change=max_magnitude,
        max_grid_impedance_phase_change_deg=max_phase,
    )


def _result(
    impedance: tuple[complex, ...],
    *,
    point_passes: tuple[bool, ...] = (True, True),
):
    point_certificates = tuple(
        SimpleNamespace(numerically_certified=passed)
        for passed in point_passes
    )
    return SimpleNamespace(
        frequencies=np.array([1.0, 10.0]),
        Z=np.asarray(impedance, dtype=complex),
        certificate=SimpleNamespace(
            numerically_certified=all(point_passes),
            thermodynamically_certified=True,
            frequency_window_certified=True,
            frequency_point_certificates=point_certificates,
        ),
    )


def test_grid_hash_binds_exact_coordinates_and_normalizes_negative_zero():
    baseline = np.array([-0.0, 0.5, 1.0])
    positive_zero = np.array([0.0, 0.5, 1.0])
    changed = np.array([0.0, 0.5000000000000001, 1.0])

    assert grid_module.ion_aware_impedance_grid_sha256(
        baseline
    ) == grid_module.ion_aware_impedance_grid_sha256(positive_zero)
    assert grid_module.ion_aware_impedance_grid_sha256(
        baseline
    ) != grid_module.ion_aware_impedance_grid_sha256(changed)

    with pytest.raises(ValueError, match="strictly increasing"):
        grid_module.ion_aware_impedance_grid_sha256(np.array([0.0, 0.0, 1.0]))


def test_grid_protocol_round_trip_and_hash_bind_every_acceptance_field():
    protocol = _protocol()

    rebuilt = grid_module.IonAwareImpedanceGridProtocol.from_json(
        protocol.canonical_json()
    )

    assert rebuilt == protocol
    assert rebuilt.protocol_hash == protocol.protocol_hash
    assert rebuilt.frequencies_Hz == (1.0, 10.0)
    assert replace(
        rebuilt,
        max_grid_impedance_phase_change_deg=2.0,
    ).protocol_hash != protocol.protocol_hash


def test_grid_protocol_schema_and_comparability_fail_closed():
    payload = _protocol().to_dict()
    payload["claim"] = "externally_validated"
    with pytest.raises(ValueError, match="extra"):
        grid_module.IonAwareImpedanceGridProtocol.from_dict(payload)

    payload.pop("claim")
    payload.pop("grid_sha256s")
    with pytest.raises(ValueError, match="missing"):
        grid_module.IonAwareImpedanceGridProtocol.from_dict(payload)

    with pytest.raises(ValueError, match="at least three"):
        replace(_protocol(), grid_node_counts=(10, 20))
    with pytest.raises(ValueError, match="strictly increasing"):
        replace(_protocol(), grid_node_counts=(10, 30, 20))
    with pytest.raises(ValueError, match="share one DC history"):
        replace(
            _protocol(),
            impedance_protocols=(
                *_impedance_protocols()[:2],
                replace(
                    _impedance_protocols()[2],
                    frequencies_Hz=(1.0, 100.0),
                ),
            ),
        )


def test_grid_pair_assessment_is_independent_for_every_frequency():
    assessments = grid_module._grid_pair_frequency_assessments(
        _result((1.0 + 0.0j, 1.0 + 0.0j)),
        _result((1.01 + 0.0j, 0.0 + 2.0j)),
        20,
        30,
        _protocol(),
    )

    assert assessments[0].passed
    assert assessments[0].impedance_magnitude_relative_change == pytest.approx(
        1.0 - 1.0 / 1.01
    )
    assert not assessments[1].passed
    assert assessments[1].impedance_magnitude_relative_change == pytest.approx(
        0.5
    )
    assert assessments[1].impedance_phase_change_deg == pytest.approx(90.0)


def test_grid_certificate_retains_all_pairs_but_gates_the_finest_pair():
    certificate = grid_module._grid_certificate(
        _protocol(),
        (
            _result((10.0 + 0.0j, 10.0 + 0.0j)),
            _result((1.0 + 0.0j, 1.0 + 0.0j)),
            _result((1.01 + 0.0j, 1.005 + 0.0j)),
        ),
    )

    assert certificate.numerically_certified
    assert certificate.certified
    assert not certificate.frequency_point_certificates[
        0
    ].grid_refinement_assessments[0].passed
    assert certificate.frequency_point_certificates[
        0
    ].grid_refinement_assessments[-1].passed
    assert certificate.finest_pair_max_impedance_magnitude_relative_change < 0.02


def test_one_unconverged_finest_frequency_fails_only_that_point_and_global_gate():
    certificate = grid_module._grid_certificate(
        _protocol(),
        (
            _result((1.0 + 0.0j, 1.0 + 0.0j)),
            _result((1.0 + 0.0j, 1.0 + 0.0j)),
            _result((1.01 + 0.0j, 0.0 + 2.0j)),
        ),
    )

    assert certificate.frequency_point_certificates[0].numerically_certified
    assert not certificate.frequency_point_certificates[1].numerically_certified
    assert not certificate.numerically_certified
    assert certificate.numerical_reasons == ("grid_refinement_not_converged",)


def test_member_point_failure_is_not_misreported_as_grid_nonconvergence():
    certificate = grid_module._grid_certificate(
        _protocol(),
        (
            _result((1.0 + 0.0j, 1.0 + 0.0j)),
            _result((1.0 + 0.0j, 1.0 + 0.0j)),
            _result(
                (1.01 + 0.0j, 1.005 + 0.0j),
                point_passes=(True, False),
            ),
        ),
    )

    assert not certificate.numerically_certified
    assert certificate.numerical_reasons == (
        "grid_member_numerical_certificate_failed",
    )
    assert certificate.frequency_point_certificates[1].reasons == (
        "grid_member_frequency_certificate_failed",
    )
