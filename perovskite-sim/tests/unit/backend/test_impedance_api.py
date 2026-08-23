from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

import backend.main as backend
from backend.progress import ProgressReporter
from perovskite_sim.experiments.impedance import ImpedanceResult


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
