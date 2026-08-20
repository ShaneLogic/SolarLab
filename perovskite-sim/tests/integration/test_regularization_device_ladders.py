"""Real-device evidence for the three Phase-1 RHS width ladders."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from perovskite_sim.validation.regularization_certificate import (
    REGULARIZATION_LADDER_FACTORS,
    RegularizationCertificate,
)
from perovskite_sim.validation.regularization_executors import (
    DEVICE_REGULARIZATION_STUDY_IDS,
    run_device_regularization_study,
)


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def device_certificates():
    return {
        study_id: run_device_regularization_study(study_id, project_root=ROOT)
        for study_id in DEVICE_REGULARIZATION_STUDY_IDS
    }


@pytest.mark.slow
@pytest.mark.parametrize("study_id", DEVICE_REGULARIZATION_STUDY_IDS)
def test_real_device_width_ladder_is_certified(
    device_certificates,
    study_id,
):
    certificate = device_certificates[study_id]

    assert certificate.status == "certified"
    assert all(check.passed for check in certificate.checks)
    assert tuple(rung.factor for rung in certificate.rungs) == (
        REGULARIZATION_LADDER_FACTORS
    )
    assert not certificate.rungs[-1].policy.active
    assert all(rung.outcome == "completed" for rung in certificate.rungs)
    assert all(rung.measurement.solver_accepted for rung in certificate.rungs)
    assert all(rung.measurement.work.nfev > 0 for rung in certificate.rungs)
    assert all(
        rung.measurement.nonfinite_event_count == 0 for rung in certificate.rungs
    )
    assert all(
        metric.scalar > 0.0
        for rung in certificate.rungs
        for metric in rung.measurement.terminal_minimum_state_m3
    )


@pytest.mark.slow
def test_each_real_study_exercises_its_declared_constitutive_path(
    device_certificates,
):
    pf = device_certificates["poole-frenkel-device"]
    te = device_certificates["thermionic-cap-device"]
    interface = device_certificates["interface-density-device"]

    def observable(certificate, factor_index, name):
        metrics = {
            metric.name: metric
            for metric in certificate.rungs[factor_index].measurement.observables
        }
        return metrics[name]

    pf_coarse = observable(pf, 0, "terminal_current_A_m2").scalar
    pf_zero = observable(pf, 3, "terminal_current_A_m2").scalar
    assert abs(pf_coarse - pf_zero) > 1.0

    te_coarse = observable(te, 0, "terminal_current_A_m2").scalar
    te_zero = observable(te, 3, "terminal_current_A_m2").scalar
    assert abs(te_coarse - te_zero) > 1.0e-3

    interface_coarse = observable(interface, 0, "interface_state_m3")
    interface_zero = observable(interface, 3, "interface_state_m3")
    assert interface_coarse.shape == interface_zero.shape == (8,)
    assert interface_coarse.values != interface_zero.values


@pytest.mark.slow
def test_real_device_certificates_round_trip_and_bind_source(
    device_certificates,
):
    for certificate in device_certificates.values():
        restored = RegularizationCertificate.from_dict(
            json.loads(certificate.canonical_json())
        )
        assert restored == certificate
        config = certificate.study.config.value
        assert config["config_sha256"]
        assert config["source_sha256"]
        assert "perovskite_sim/physics/regularization.py" in (config["source_sha256"])
