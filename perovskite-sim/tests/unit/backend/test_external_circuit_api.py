from __future__ import annotations

import numpy as np
import pytest
from fastapi import HTTPException

import backend.main as backend
from perovskite_sim.experiments.external_circuit import ExternalCircuitProtocol
from perovskite_sim.experiments.jv_sweep import JVMetrics, JVPointStatus, JVResult


def _intrinsic_result(*, certified: bool = True) -> JVResult:
    voltage_fwd = np.array([0.0, 0.4, 0.8, 1.0, 1.1])
    current_fwd = np.array([20.0, 19.0, 14.0, 0.0, -30.0])
    voltage_rev = voltage_fwd[::-1].copy()
    current_rev = (current_fwd + 0.1)[::-1].copy()

    def statuses(branch: str, voltage: np.ndarray):
        return tuple(
            JVPointStatus(
                branch=branch,
                index=index,
                voltage=float(value),
                valid=certified,
            )
            for index, value in enumerate(voltage)
        )

    metrics = JVMetrics(V_oc=1.0, J_sc=20.0, FF=0.7, PCE=0.014)
    return JVResult(
        V_fwd=voltage_fwd,
        J_fwd=current_fwd,
        V_rev=voltage_rev,
        J_rev=current_rev,
        metrics_fwd=metrics,
        metrics_rev=metrics,
        hysteresis_index=0.0,
        status_fwd=statuses("jv_forward", voltage_fwd),
        status_rev=statuses("jv_reverse", voltage_rev),
    )


def _request(protocol: ExternalCircuitProtocol) -> backend.ExternalCircuitJVRequest:
    return backend.ExternalCircuitJVRequest(
        device={"device": {}, "layers": []},
        N_grid=12,
        n_points=5,
        V_max=1.1,
        external_circuit_protocol=protocol.to_dict(),
    )


def test_external_circuit_endpoint_returns_separate_terminal_evidence(monkeypatch):
    monkeypatch.setattr(backend, "build_stack", lambda *_args: object())
    monkeypatch.setattr(
        backend,
        "_run_jv_dispatch",
        lambda *_args, **_kwargs: _intrinsic_result(),
    )
    protocol = ExternalCircuitProtocol(
        series_resistance_ohm_m2=1.0e-3,
        shunt_resistance_ohm_m2=0.5,
    )

    response = backend.run_external_circuit_jv(_request(protocol))
    result = response["result"]

    assert response["status"] == "ok"
    assert result["certified"] is True
    assert result["source_certified"] is True
    assert result["mapping_certified"] is True
    assert result["circuit_protocol_sha256"] == protocol.sha256
    assert len(result["source_result_sha256"]) == 64
    assert len(result["mapping_sha256"]) == 64
    assert result["forward"]["junction_voltage_V"] == [0.0, 0.4, 0.8, 1.0, 1.1]
    assert result["forward"]["terminal_voltage_V"] != result["forward"][
        "junction_voltage_V"
    ]


def test_invalid_circuit_schema_is_rejected_before_solver(monkeypatch):
    called = False

    def fail_if_called(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("solver must not run")

    monkeypatch.setattr(backend, "_run_jv_dispatch", fail_if_called)
    payload = ExternalCircuitProtocol().to_dict()
    payload["unregistered_claim"] = True
    request = _request(ExternalCircuitProtocol())
    request = request.model_copy(update={"external_circuit_protocol": payload})

    with pytest.raises(HTTPException) as exc:
        backend.run_external_circuit_jv(request)
    assert exc.value.status_code == 422
    assert "unregistered_claim" in str(exc.value.detail)
    assert called is False


def test_uncertified_intrinsic_result_is_rejected(monkeypatch):
    monkeypatch.setattr(backend, "build_stack", lambda *_args: object())
    monkeypatch.setattr(
        backend,
        "_run_jv_dispatch",
        lambda *_args, **_kwargs: _intrinsic_result(certified=False),
    )

    with pytest.raises(HTTPException) as exc:
        backend.run_external_circuit_jv(_request(ExternalCircuitProtocol()))
    assert exc.value.status_code == 422
    assert "certified intrinsic JVResult" in str(exc.value.detail)


def test_invalid_incident_power_is_rejected(monkeypatch):
    monkeypatch.setattr(backend, "build_stack", lambda *_args: object())
    monkeypatch.setattr(
        backend,
        "_run_jv_dispatch",
        lambda *_args, **_kwargs: _intrinsic_result(),
    )
    request = _request(ExternalCircuitProtocol())
    request = request.model_copy(update={"incident_power_W_m2": 0.0})

    with pytest.raises(HTTPException) as exc:
        backend.run_external_circuit_jv(request)
    assert exc.value.status_code == 422
    assert "incident_power_W_m2 must be positive" in str(exc.value.detail)


def test_unexpected_solver_error_remains_server_error(monkeypatch):
    monkeypatch.setattr(backend, "build_stack", lambda *_args: object())

    def broken_solver(*_args, **_kwargs):
        raise RuntimeError("internal solver failure")

    monkeypatch.setattr(backend, "_run_jv_dispatch", broken_solver)

    with pytest.raises(HTTPException) as exc:
        backend.run_external_circuit_jv(_request(ExternalCircuitProtocol()))
    assert exc.value.status_code == 500
    assert "internal solver failure" in str(exc.value.detail)
