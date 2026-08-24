from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from perovskite_sim.experiments import identifiability
from perovskite_sim.experiments.identifiability import (
    PARAMETER_NAMES,
    IdentifiabilityError,
    InterfaceSRHIdentifiabilityProtocol,
    build_interface_srh_identifiability_protocol,
    run_interface_srh_identifiability,
)


def _run(
    family="recombination_plus_charge",
    estimated=PARAMETER_NAMES,
    **kwargs,
):
    protocol = build_interface_srh_identifiability_protocol(
        observable_family=family,
        estimated_parameters=estimated,
        carrier_condition_count=5,
        **kwargs,
    )
    return protocol, run_interface_srh_identifiability(protocol)


def test_protocol_round_trip_is_canonical_and_strict():
    protocol = build_interface_srh_identifiability_protocol(carrier_condition_count=5)
    restored = InterfaceSRHIdentifiabilityProtocol.from_dict(protocol.to_dict())

    assert restored == protocol
    assert restored.sha256 == protocol.sha256
    assert restored.canonical_json() == protocol.canonical_json()

    payload = protocol.to_dict()
    payload["unknown"] = True
    with pytest.raises(IdentifiabilityError, match=r"extra=\['unknown'\]"):
        InterfaceSRHIdentifiabilityProtocol.from_dict(payload)


def test_nested_parameter_schema_rejects_unknown_keys():
    payload = build_interface_srh_identifiability_protocol(
        carrier_condition_count=5
    ).to_dict()
    payload["parameters"][0]["claim"] = "identified"
    with pytest.raises(IdentifiabilityError, match=r"extra=\['claim'\]"):
        InterfaceSRHIdentifiabilityProtocol.from_dict(payload)


@pytest.mark.parametrize(
    ("estimated", "family", "expected_rank"),
    (
        (PARAMETER_NAMES, "recombination_only", 1),
        (PARAMETER_NAMES, "recombination_plus_charge", 2),
        (
            ("trap_density_cm2", "calibration_factor"),
            "recombination_plus_charge",
            2,
        ),
        (
            ("capture_cross_section_scale", "calibration_factor"),
            "recombination_plus_charge",
            1,
        ),
    ),
)
def test_builder_declares_structural_rank(estimated, family, expected_rank):
    protocol = build_interface_srh_identifiability_protocol(
        observable_family=family,
        estimated_parameters=estimated,
        carrier_condition_count=5,
    )
    assert protocol.expected_rank == expected_rank


def test_recombination_only_exposes_two_dimensional_nullspace():
    protocol, result = _run("recombination_only")

    assert result.analysis_certified
    assert result.rank_expectation_met
    assert result.numerical_rank == protocol.expected_rank == 1
    assert not result.parameters_identifiable
    assert not result.truth_recovered
    assert result.condition_number is None
    assert len(result.nullspace_vectors) == 2
    assert result.forward_failure_count == 0
    assert result.best_chi_square == pytest.approx(0.0, abs=1.0e-20)


def test_charge_observable_identifies_nt_but_not_capture_vs_calibration():
    protocol, result = _run("recombination_plus_charge")

    assert result.analysis_certified
    assert result.numerical_rank == protocol.expected_rank == 2
    assert not result.parameters_identifiable
    assert len(result.nullspace_vectors) == 1
    trap, capture, calibration = result.nullspace_vectors[0]
    assert abs(trap) < 1.0e-10
    assert capture == pytest.approx(-calibration, rel=1.0e-10, abs=1.0e-10)
    assert abs(capture) == pytest.approx(2.0**-0.5, rel=1.0e-10)


def test_known_capture_scale_gives_full_rank_multistart_truth_recovery():
    estimated = ("trap_density_cm2", "calibration_factor")
    protocol, result = _run(
        "recombination_plus_charge",
        estimated=estimated,
    )

    assert result.analysis_certified
    assert result.parameters_identifiable
    assert result.truth_recovered
    assert result.numerical_rank == protocol.expected_rank == 2
    assert result.condition_number == pytest.approx(2.61803398875, rel=2.0e-6)
    assert result.nullspace_vectors == ()
    expected_log = tuple(
        parameter.truth_log10 for parameter in protocol.estimated_parameters
    )
    assert result.best_fit_log10 == pytest.approx(expected_log, abs=1.0e-10)
    assert all(attempt.success for attempt in result.fit_attempts)


def test_production_kinetic_scaling_and_charge_identity_are_exact():
    protocol = build_interface_srh_identifiability_protocol(carrier_condition_count=5)
    truth = np.asarray(
        [parameter.truth_log10 for parameter in protocol.estimated_parameters]
    )
    labels, units, baseline = identifiability._predict_interface_srh_observables(
        protocol, truth
    )
    calibration_decade = truth.copy()
    calibration_decade[2] += 1.0
    _, _, scaled = identifiability._predict_interface_srh_observables(
        protocol, calibration_decade
    )
    count = len(protocol.carrier_conditions)

    assert labels[:count] == tuple(
        f"interface_current_condition_{index}_A_m2" for index in range(count)
    )
    assert units[:count] == ("A m-2",) * count
    assert scaled[:count] == pytest.approx(10.0 * baseline[:count], rel=2.0e-14)
    assert scaled[count:] == pytest.approx(
        baseline[count:], rel=2.0e-15, abs=1.0e-30
    )


def test_capture_scale_and_calibration_are_observationally_equivalent():
    protocol = build_interface_srh_identifiability_protocol(carrier_condition_count=5)
    truth = np.asarray(
        [parameter.truth_log10 for parameter in protocol.estimated_parameters]
    )
    capture_shift = truth.copy()
    capture_shift[1] += 0.4
    calibration_shift = truth.copy()
    calibration_shift[2] += 0.4

    capture_values = identifiability._predict_interface_srh_observables(
        protocol, capture_shift
    )[2]
    calibration_values = identifiability._predict_interface_srh_observables(
        protocol, calibration_shift
    )[2]
    assert capture_values == pytest.approx(calibration_values, rel=2.0e-14, abs=1.0e-30)


def test_profiles_are_complete_and_dimension_matched():
    protocol, result = _run("recombination_plus_charge")

    assert result.profiles_completed
    assert tuple(profile.parameter_name for profile in result.profiles) == tuple(
        parameter.name for parameter in protocol.estimated_parameters
    )
    assert all(
        len(profile.parameter_values_log10) == protocol.profile_grid_count
        and len(profile.chi_square) == protocol.profile_grid_count
        and all(profile.successful)
        for profile in result.profiles
    )


def test_synthetic_noise_is_reproducible_and_content_addressed():
    kwargs = {"synthetic_noise_sigma_multiplier": 0.25, "noise_seed": 91}
    protocol_a, result_a = _run("recombination_plus_charge", **kwargs)
    protocol_b, result_b = _run("recombination_plus_charge", **kwargs)
    protocol_c, result_c = _run(
        "recombination_plus_charge",
        synthetic_noise_sigma_multiplier=0.25,
        noise_seed=92,
    )

    assert protocol_a == protocol_b
    assert result_a.observed_values == result_b.observed_values
    assert result_a.mapping_sha256 == result_b.mapping_sha256
    assert protocol_a.sha256 != protocol_c.sha256
    assert result_a.observed_values != result_c.observed_values
    assert result_a.mapping_sha256 != result_c.mapping_sha256


def test_noisy_full_rank_recovery_fails_the_strict_truth_gate():
    protocol, result = _run(
        "recombination_plus_charge",
        estimated=("trap_density_cm2", "calibration_factor"),
        synthetic_noise_sigma_multiplier=0.5,
        noise_seed=7,
    )

    assert result.numerical_rank == protocol.expected_rank
    assert result.parameters_identifiable
    assert not result.truth_recovered
    assert not result.analysis_certified


def test_forward_failures_are_penalized_reported_and_invalidate(monkeypatch):
    protocol = build_interface_srh_identifiability_protocol(carrier_condition_count=5)
    original = identifiability._predict_interface_srh_observables

    def guarded(candidate_protocol, coordinates):
        if float(coordinates[0]) > 12.5:
            raise FloatingPointError("synthetic forward wall")
        return original(candidate_protocol, coordinates)

    monkeypatch.setattr(
        identifiability,
        "_predict_interface_srh_observables",
        guarded,
    )
    result = run_interface_srh_identifiability(protocol)

    assert result.forward_failure_count > 0
    assert not result.analysis_certified


def test_result_rejects_flag_and_content_hash_tampering():
    _protocol, result = _run(
        "recombination_plus_charge",
        estimated=("trap_density_cm2", "calibration_factor"),
    )

    with pytest.raises(ValueError, match="parameters_identifiable"):
        dataclasses.replace(result, parameters_identifiable=False)
    with pytest.raises(ValueError, match="mapping_sha256"):
        dataclasses.replace(result, best_chi_square=result.best_chi_square + 1.0)


def test_result_document_is_finite_json_and_immutable():
    _protocol, result = _run("recombination_plus_charge")
    document = result.to_dict()
    encoded = json.dumps(document, allow_nan=False, sort_keys=True)

    assert "NaN" not in encoded
    assert document["condition_number"] is None
    assert isinstance(result.observed_values, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.analysis_certified = False


@pytest.mark.parametrize(
    "kwargs",
    (
        {"estimated_parameters": ()},
        {"estimated_parameters": ("not_a_parameter",)},
        {"carrier_condition_count": 2},
        {"synthetic_noise_sigma_multiplier": -1.0},
    ),
)
def test_builder_rejects_invalid_contracts(kwargs):
    with pytest.raises((TypeError, ValueError)):
        build_interface_srh_identifiability_protocol(**kwargs)
