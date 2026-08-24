from __future__ import annotations

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

import backend.main as backend
from perovskite_sim.experiments.electrothermal import (
    ElectrothermalConvergenceError,
    ElectrothermalJVProtocol,
    ElectrothermalOperatingPointProtocol,
)
from perovskite_sim.experiments.external_circuit import ExternalCircuitProtocol
from perovskite_sim.experiments.thermal_balance import LumpedThermalProtocol


def _thermal() -> LumpedThermalProtocol:
    return LumpedThermalProtocol(
        absorbed_optical_power_W_m2=800.0,
        ambient_temperature_K=300.0,
        areal_heat_capacity_J_m2_K=2000.0,
        heat_transfer_coefficient_W_m2_K=20.0,
        emissivity=0.85,
        maximum_temperature_K=500.0,
    )


def _electrical() -> ElectrothermalJVProtocol:
    return ElectrothermalJVProtocol(
        grid_points_per_electrical_layer=20,
        voltage_points_per_branch=12,
        scan_rate_V_s=20.0,
        voltage_max_V=1.2,
    )


def _request() -> backend.ElectrothermalOperatingPointRequest:
    return backend.ElectrothermalOperatingPointRequest(
        device={"device": {}, "layers": []},
        thermal_protocol=_thermal().to_dict(),
        external_circuit_protocol=ExternalCircuitProtocol().to_dict(),
        electrical_protocol=_electrical().to_dict(),
        operating_protocol=ElectrothermalOperatingPointProtocol().to_dict(),
    )


def test_endpoint_parses_protocols_and_returns_separate_result(monkeypatch):
    sentinel_stack = object()
    captured = {}
    monkeypatch.setattr(backend, "build_stack", lambda *_args: sentinel_stack)

    def fake_solve(stack, thermal, circuit, electrical, operating):
        captured.update(
            {
                "stack": stack,
                "thermal": thermal,
                "circuit": circuit,
                "electrical": electrical,
                "operating": operating,
            }
        )
        return {"certified": True, "operating_temperature_K": 331.0}

    monkeypatch.setattr(
        backend.electrothermal_exp,
        "solve_electrothermal_operating_point",
        fake_solve,
    )

    response = backend.run_electrothermal_operating_point(_request())

    assert response == {
        "status": "ok",
        "result": {"certified": True, "operating_temperature_K": 331.0},
    }
    assert captured["stack"] is sentinel_stack
    assert captured["thermal"] == _thermal()
    assert captured["circuit"] == ExternalCircuitProtocol()
    assert captured["electrical"] == _electrical()
    assert captured["operating"] == ElectrothermalOperatingPointProtocol()


def test_invalid_protocol_is_rejected_before_stack_build(monkeypatch):
    called = False

    def fail_if_called(*_args):
        nonlocal called
        called = True
        raise AssertionError("stack must not be built")

    monkeypatch.setattr(backend, "build_stack", fail_if_called)
    request = _request()
    payload = dict(request.thermal_protocol)
    payload["unregistered_claim"] = True
    request = request.model_copy(update={"thermal_protocol": payload})

    with pytest.raises(HTTPException) as exc:
        backend.run_electrothermal_operating_point(request)
    assert exc.value.status_code == 422
    assert "unregistered_claim" in str(exc.value.detail)
    assert called is False


def test_expected_coupling_failure_returns_422(monkeypatch):
    monkeypatch.setattr(backend, "build_stack", lambda *_args: object())

    def no_root(*_args):
        raise ElectrothermalConvergenceError("no bounded root")

    monkeypatch.setattr(
        backend.electrothermal_exp,
        "solve_electrothermal_operating_point",
        no_root,
    )

    with pytest.raises(HTTPException) as exc:
        backend.run_electrothermal_operating_point(_request())
    assert exc.value.status_code == 422
    assert exc.value.detail == "no bounded root"


def test_unexpected_solver_failure_remains_500(monkeypatch):
    monkeypatch.setattr(backend, "build_stack", lambda *_args: object())

    def broken(*_args):
        raise RuntimeError("unexpected electrical failure")

    monkeypatch.setattr(
        backend.electrothermal_exp,
        "solve_electrothermal_operating_point",
        broken,
    )

    with pytest.raises(HTTPException) as exc:
        backend.run_electrothermal_operating_point(_request())
    assert exc.value.status_code == 500
    assert exc.value.detail == "unexpected electrical failure"


def test_request_rejects_unknown_top_level_fields():
    payload = _request().model_dump()
    payload["solver"] = "unregistered"
    with pytest.raises(ValidationError, match="solver"):
        backend.ElectrothermalOperatingPointRequest(**payload)
