from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

import backend.main as backend
from backend.progress import ProgressReporter
from perovskite_sim.experiments.impedance import ImpedanceResult
from perovskite_sim.experiments.jv_sweep import build_electrical_grid
from perovskite_sim.experiments.dynamic_defect_impedance import (
    build_dynamic_defect_impedance_protocol,
)
from tests.integration.test_charged_explicit_defects_qf import _stack as _bulk_stack


def test_complex_diagnostic_serialization_preserves_array_shape():
    values = np.array([
        [1.0 + 2.0j, 3.0 + 4.0j],
        [5.0 + 6.0j, 7.0 + 8.0j],
    ])

    serialized = backend.to_serializable(values)

    assert len(serialized) == 2
    assert all(len(row) == 2 for row in serialized)
    assert serialized[1][0] == {"real": 5.0, "imag": 6.0}


def test_impedance_api_forwards_certification_protocol(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(backend, "build_stack", lambda *args: object())

    def fake_run(stack, frequencies, **kwargs):
        captured.update(kwargs)
        return ImpedanceResult(
            frequencies=np.asarray(frequencies),
            Z=np.full(len(frequencies), 2.0 - 3.0j),
        )

    monkeypatch.setattr(backend.impedance, "run_impedance", fake_run)
    response = backend.run_impedance_api(backend.ISRequest(
        device={"device": {}, "layers": []},
        N_grid=42,
        V_dc=0.75,
        n_freq=3,
        f_min=1e-3,
        f_max=1e4,
        delta_V=5e-3,
        n_cycles=7,
        n_extract=3,
        points_per_cycle=80,
        dc_settle_time=2.0,
        illuminated=False,
        method="ion_aware_frequency_certified",
        require_operating_point_certificate=True,
        require_frequency_window_certificate=True,
    ))

    assert captured == {
        "V_dc": 0.75,
        "delta_V": 5e-3,
        "N_grid": 42,
        "n_cycles": 7,
        "n_extract": 3,
        "points_per_cycle": 80,
        "illuminated": False,
        "method": "ion_aware_frequency_certified",
        "dc_settle_time": 2.0,
        "require_operating_point_certificate": True,
        "require_frequency_window_certificate": True,
        "experiment_protocol": None,
        "protocol_mode": "compatibility",
    }
    result = response["result"]
    assert result["frequencies"] == pytest.approx([1e-3, np.sqrt(10.0), 1e4])
    assert result["Z_real"] == [2.0, 2.0, 2.0]
    assert result["Z_imag"] == [-3.0, -3.0, -3.0]


def test_impedance_api_maps_certification_failure_to_http_422(monkeypatch):
    monkeypatch.setattr(backend, "build_stack", lambda *args: object())

    def reject_uncertified_operating_point(*args, **kwargs):
        raise backend.impedance.ImpedanceCertificationError(
            "DC operating point is uncertified"
        )

    monkeypatch.setattr(
        backend.impedance,
        "run_impedance",
        reject_uncertified_operating_point,
    )

    with TestClient(backend.app) as client:
        response = client.post(
            "/api/impedance",
            json={
                "device": {"device": {}, "layers": []},
                "require_operating_point_certificate": True,
            },
        )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "DC operating point is uncertified",
    }


def test_impedance_api_maps_capability_failure_to_http_422(monkeypatch):
    monkeypatch.setattr(backend, "build_stack", lambda *args: object())
    monkeypatch.setattr(
        backend.impedance,
        "run_impedance",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            backend.impedance.ImpedanceCapabilityError(
                "dynamic interface-state impedance is unsupported"
            )
        ),
    )

    with TestClient(backend.app) as client:
        response = client.post(
            "/api/impedance",
            json={"device": {"device": {}, "layers": []}},
        )

    assert response.status_code == 422
    assert "interface-state" in response.json()["detail"]


def test_impedance_job_forwards_full_protocol(monkeypatch):
    submitted = {}
    captured = {}

    class CaptureRegistry:
        def submit(self, fn):
            submitted["fn"] = fn
            return "impedance-job"

    monkeypatch.setattr(backend, "_JOB_REGISTRY", CaptureRegistry())
    monkeypatch.setattr(backend, "build_stack", lambda *args: object())

    def fake_run(stack, frequencies, **kwargs):
        captured["frequencies"] = np.asarray(frequencies)
        captured.update(kwargs)
        return ImpedanceResult(
            frequencies=np.asarray(frequencies),
            Z=np.ones(len(frequencies), dtype=complex),
        )

    monkeypatch.setattr(backend.impedance, "run_impedance", fake_run)
    response = backend.start_job(backend.JobRequest(
        kind="impedance",
        device={"device": {}, "layers": []},
        params={
            "V_dc": 0.8,
            "delta_V": 0.004,
            "N_grid": 44,
            "n_freq": 4,
            "f_min": 1.0e-2,
            "f_max": 1.0e2,
            "n_cycles": 9,
            "n_extract": 4,
            "points_per_cycle": 160,
            "illuminated": "false",
            "method": "transient_ion_aware",
            "dc_settle_time": 3.0,
            "require_operating_point_certificate": "true",
            "require_frequency_window_certificate": "true",
        },
    ))

    assert response == {"status": "ok", "job_id": "impedance-job"}
    result = submitted["fn"](ProgressReporter())
    assert result["Z_real"] == [1.0] * 4
    np.testing.assert_allclose(captured.pop("frequencies"), np.logspace(-2, 2, 4))
    captured.pop("progress")
    assert captured == {
        "V_dc": 0.8,
        "delta_V": 0.004,
        "N_grid": 44,
        "n_cycles": 9,
        "n_extract": 4,
        "points_per_cycle": 160,
        "illuminated": False,
        "method": "transient_ion_aware",
        "dc_settle_time": 3.0,
        "require_operating_point_certificate": True,
        "require_frequency_window_certificate": True,
        "experiment_protocol": None,
        "protocol_mode": "compatibility",
    }


def test_dynamic_defect_api_builds_and_forwards_exact_protocol(monkeypatch):
    captured: dict[str, object] = {}
    stack = _bulk_stack()
    monkeypatch.setattr(backend, "build_stack", lambda *args: stack)

    def fake_run(_stack, frequencies, **kwargs):
        captured.update(kwargs)
        return ImpedanceResult(
            frequencies=np.asarray(frequencies),
            Z=np.ones(len(frequencies), dtype=complex),
        )

    monkeypatch.setattr(backend.impedance, "run_impedance", fake_run)
    response = backend.run_impedance_api(
        backend.ISRequest(
            device={"device": {}, "layers": []},
            N_grid=4,
            V_dc=0.0,
            n_freq=3,
            f_min=1.0e-4,
            f_max=1.0e12,
            illuminated=False,
            method="dynamic_defect_frequency_certified",
            defect_energy_quadrature_order=16,
            dynamic_defect_state_step=2.0e-5,
            dynamic_defect_voltage_step=3.0e-5,
        )
    )

    protocol = captured["dynamic_defect_protocol"]
    assert isinstance(protocol, backend.impedance.DynamicDefectImpedanceProtocol)
    assert protocol.capability == "bulk_dynamic_defect"
    assert protocol.requested_grid_intervals == 4
    assert protocol.defect_energy_quadrature_order == 16
    assert protocol.state_step == 2.0e-5
    assert protocol.voltage_step == 3.0e-5
    assert captured["defect_energy_quadrature_order"] == 16
    assert response["result"]["Z_real"] == [1.0, 1.0, 1.0]


def test_dynamic_defect_job_rejects_mismatched_protocol_before_submit(monkeypatch):
    stack = _bulk_stack()
    frequencies = np.logspace(-4.0, 12.0, 3)
    protocol = build_dynamic_defect_impedance_protocol(
        stack,
        build_electrical_grid(stack, 4),
        frequencies,
        requested_grid_intervals=4,
        V_dc=0.0,
        delta_V=0.01,
        illuminated=False,
    )
    payload = protocol.to_dict()
    payload["frequencies_Hz"] = [1.0e-4, 1.0, 1.0e10]

    class RejectSubmitRegistry:
        def submit(self, _fn):
            raise AssertionError("mismatched job must fail before submit")

    monkeypatch.setattr(backend, "_JOB_REGISTRY", RejectSubmitRegistry())
    monkeypatch.setattr(backend, "build_stack", lambda *args: stack)

    with pytest.raises(backend.HTTPException) as exc_info:
        backend.start_job(
            backend.JobRequest(
                kind="impedance",
                device={"device": {}, "layers": []},
                params={
                    "N_grid": 4,
                    "V_dc": 0.0,
                    "n_freq": 3,
                    "f_min": 1.0e-4,
                    "f_max": 1.0e12,
                    "illuminated": False,
                    "method": "dynamic_defect_frequency_certified",
                    "dynamic_defect_protocol": payload,
                },
            )
        )

    assert exc_info.value.status_code == 422
    assert "does not match" in str(exc_info.value.detail)


def test_dynamic_defect_job_forwards_preflighted_protocol_to_worker(monkeypatch):
    submitted: dict[str, object] = {}
    captured: dict[str, object] = {}
    stack = _bulk_stack()

    class CaptureRegistry:
        def submit(self, fn):
            submitted["fn"] = fn
            return "dynamic-defect-job"

    def fake_run(_stack, frequencies, **kwargs):
        captured.update(kwargs)
        return ImpedanceResult(
            frequencies=np.asarray(frequencies),
            Z=np.ones(len(frequencies), dtype=complex),
        )

    monkeypatch.setattr(backend, "_JOB_REGISTRY", CaptureRegistry())
    monkeypatch.setattr(backend, "build_stack", lambda *args: stack)
    monkeypatch.setattr(backend.impedance, "run_impedance", fake_run)
    response = backend.start_job(
        backend.JobRequest(
            kind="impedance",
            device={"device": {}, "layers": []},
            params={
                "N_grid": 4,
                "V_dc": 0.0,
                "n_freq": 3,
                "f_min": 1.0e-4,
                "f_max": 1.0e12,
                "illuminated": False,
                "method": "dynamic_defect_frequency_certified",
                "defect_energy_quadrature_order": 24,
            },
        )
    )

    assert response == {"status": "ok", "job_id": "dynamic-defect-job"}
    submitted["fn"](ProgressReporter())
    protocol = captured["dynamic_defect_protocol"]
    assert isinstance(protocol, backend.impedance.DynamicDefectImpedanceProtocol)
    assert protocol.defect_energy_quadrature_order == 24
    assert captured["defect_energy_quadrature_order"] == 24


def test_dynamic_defect_protocol_is_rejected_for_legacy_method(monkeypatch):
    monkeypatch.setattr(backend, "build_stack", lambda *args: _bulk_stack())

    with TestClient(backend.app) as client:
        response = client.post(
            "/api/impedance",
            json={
                "device": {"device": {}, "layers": []},
                "method": "transient_ion_aware",
                "dynamic_defect_protocol": {"claim": "certified"},
            },
        )

    assert response.status_code == 422
    assert "valid only" in response.json()["detail"]


def test_impedance_request_rejects_unknown_fields():
    with TestClient(backend.app) as client:
        response = client.post(
            "/api/impedance",
            json={
                "device": {"device": {}, "layers": []},
                "dynamic_defect_energy_order": 32,
            },
        )

    assert response.status_code == 422
