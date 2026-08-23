from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import perovskite_sim.experiments.ion_aware_impedance_lockin as lockin
from perovskite_sim.experiments.ion_aware_dc import (
    build_ion_aware_dc_protocol,
    solve_ion_aware_dc,
)
from perovskite_sim.experiments.ion_aware_impedance import (
    build_ion_aware_impedance_protocol,
    run_ion_aware_impedance,
)
from perovskite_sim.experiments.jv_sweep import build_electrical_grid
from perovskite_sim.models.config_loader import load_device_from_yaml


@pytest.fixture(scope="module")
def frequency_domain_fixture():
    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    x = build_electrical_grid(stack, 12)
    dc_state = solve_ion_aware_dc(
        x,
        stack,
        build_ion_aware_dc_protocol(
            stack,
            V_dc=0.9,
            illuminated=True,
        ),
    )
    protocol = build_ion_aware_impedance_protocol(
        dc_state,
        np.array([1.0e-2, 1.0]),
    )
    result = run_ion_aware_impedance(
        x,
        stack,
        protocol,
        dc_state=dc_state,
    )
    return dc_state, result


def test_lockin_protocol_round_trip_binds_exact_reference(
    frequency_domain_fixture,
):
    _dc_state, frequency_domain = frequency_domain_fixture
    protocol = lockin.build_ion_aware_transient_lockin_protocol(
        frequency_domain,
        np.array([1.0e-2, 1.0]),
    )

    rebuilt = lockin.IonAwareTransientLockInProtocol.from_json(
        protocol.canonical_json()
    )

    assert rebuilt == protocol
    assert rebuilt.protocol_hash == protocol.protocol_hash
    assert rebuilt.frequency_domain_protocol_sha256 == (
        frequency_domain.protocol_hash
    )
    assert rebuilt.dc_state_sha256 == frequency_domain.protocol.dc_state_sha256
    assert replace(rebuilt, cycles=7).protocol_hash != rebuilt.protocol_hash


def test_lockin_protocol_schema_and_frequency_subset_fail_closed(
    frequency_domain_fixture,
):
    _dc_state, frequency_domain = frequency_domain_fixture
    protocol = lockin.build_ion_aware_transient_lockin_protocol(
        frequency_domain,
        np.array([1.0e-2]),
    )
    payload = protocol.to_dict()
    payload["claim"] = "externally_validated"
    with pytest.raises(ValueError, match="extra"):
        lockin.IonAwareTransientLockInProtocol.from_dict(payload)

    payload.pop("claim")
    payload.pop("dc_state_sha256")
    with pytest.raises(ValueError, match="missing"):
        lockin.IonAwareTransientLockInProtocol.from_dict(payload)
    with pytest.raises(lockin.IonAwareTransientLockInCapabilityError):
        lockin.build_ion_aware_transient_lockin_protocol(
            frequency_domain,
            np.array([2.0e-2]),
        )
    with pytest.raises(ValueError, match="20 mV"):
        replace(protocol, delta_V=0.02)


def _install_synthetic_transients(monkeypatch, frequency_domain):
    calls = []
    reports = {
        float(frequency): frequency_domain.dc_state.steps[-1].numerical_diagnostics
        for frequency in frequency_domain.frequencies
    }
    impedance = {
        float(frequency): complex(value)
        for frequency, value in zip(
            frequency_domain.frequencies,
            frequency_domain.Z,
            strict=True,
        )
    }

    def fake_run(
        _dc_state,
        _protocol,
        frequency,
        points_per_cycle,
        _material,
    ):
        calls.append((frequency, points_per_cycle))
        relative_offset = {40: 4.0e-3, 80: 1.0e-3, 160: 5.0e-4}[
            points_per_cycle
        ]
        return lockin.TransientLockInFrequencyEvidence(
            frequency_Hz=frequency,
            points_per_cycle=points_per_cycle,
            impedance_ohm_m2=impedance[frequency] * (1.0 + relative_offset),
            cycle_current_relative_change=1.0e-4,
            state_periodicity_relative_change=1.0e-5,
            max_ion_inventory_relative_drift=1.0e-12,
            accepted_method="Radau",
            numerical_diagnostics=reports[frequency],
        )

    monkeypatch.setattr(lockin, "_run_frequency", fake_run)
    return calls


def test_lockin_crosscheck_certifies_each_numerical_axis(
    monkeypatch,
    frequency_domain_fixture,
):
    dc_state, frequency_domain = frequency_domain_fixture
    protocol = lockin.build_ion_aware_transient_lockin_protocol(
        frequency_domain,
        frequency_domain.frequencies,
    )
    calls = _install_synthetic_transients(monkeypatch, frequency_domain)

    result = lockin.run_ion_aware_transient_lockin_crosscheck(
        frequency_domain,
        protocol,
        dc_state=dc_state,
    )

    assert calls == [
        (float(frequency), points_per_cycle)
        for points_per_cycle in (40, 80, 160)
        for frequency in frequency_domain.frequencies
    ]
    certificate = result.certificate
    assert certificate.numerically_certified
    assert certificate.frequency_domain_agreement_certified
    assert certificate.time_resolution_certified
    assert certificate.periodicity_certified
    assert certificate.inventory_certified
    assert not certificate.thermodynamically_certified
    assert not certificate.frequency_window_certified
    assert not certificate.certified
    assert certificate.numerical_reasons == ()
    assert certificate.reasons == (
        "contact_thermodynamics_not_certified",
        "frequency_window_not_certified",
    )
    assert all(
        item.numerically_certified
        for item in certificate.frequency_certificates
    )


def test_lockin_crosscheck_rejects_replaced_dc_state_before_transient(
    monkeypatch,
    frequency_domain_fixture,
):
    dc_state, frequency_domain = frequency_domain_fixture
    protocol = lockin.build_ion_aware_transient_lockin_protocol(
        frequency_domain,
        np.array([1.0e-2]),
    )
    changed = dc_state.y.copy()
    changed[1] = np.nextafter(changed[1], np.inf)
    stale = replace(dc_state, y=changed)
    monkeypatch.setattr(
        lockin,
        "_run_frequency",
        lambda *_args, **_kwargs: pytest.fail("stale state reached transient"),
    )

    with pytest.raises(
        lockin.IonAwareTransientLockInCapabilityError,
        match="DC state does not match",
    ):
        lockin.run_ion_aware_transient_lockin_crosscheck(
            frequency_domain,
            protocol,
            dc_state=stale,
        )


def test_tighter_preregistered_agreement_gate_returns_failed_evidence(
    monkeypatch,
    frequency_domain_fixture,
):
    dc_state, frequency_domain = frequency_domain_fixture
    protocol = lockin.build_ion_aware_transient_lockin_protocol(
        frequency_domain,
        np.array([1.0e-2]),
        max_frequency_domain_magnitude_relative_difference=1.0e-5,
    )
    _install_synthetic_transients(monkeypatch, frequency_domain)

    result = lockin.run_ion_aware_transient_lockin_crosscheck(
        frequency_domain,
        protocol,
        dc_state=dc_state,
        require_numerical_certificate=False,
    )

    assert not result.certificate.numerically_certified
    assert not result.certificate.frequency_domain_agreement_certified
    assert result.certificate.numerical_reasons == (
        "frequency_domain_agreement_failed",
    )
    with pytest.raises(
        lockin.IonAwareTransientLockInCertificationError,
        match="frequency_domain_agreement_failed",
    ):
        lockin.run_ion_aware_transient_lockin_crosscheck(
            frequency_domain,
            protocol,
            dc_state=dc_state,
        )
