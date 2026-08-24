from __future__ import annotations

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

import backend.main as backend
from perovskite_sim.experiments.identifiability import (
    IdentifiabilityForwardError,
    build_interface_srh_identifiability_protocol,
)


def _request() -> backend.InterfaceSRHIdentifiabilityRequest:
    protocol = build_interface_srh_identifiability_protocol(carrier_condition_count=5)
    return backend.InterfaceSRHIdentifiabilityRequest(protocol=protocol.to_dict())


def test_endpoint_returns_rank_deficiency_without_identification_claim():
    response = backend.run_interface_srh_identifiability(_request())
    result = response["result"]

    assert response["status"] == "ok"
    assert result["analysis_certified"] is True
    assert result["parameters_identifiable"] is False
    assert result["numerical_rank"] == 2
    assert len(result["nullspace_vectors"]) == 1
    assert len(result["protocol_sha256"]) == 64
    assert len(result["mapping_sha256"]) == 64


def test_unknown_protocol_field_is_rejected_before_analysis(monkeypatch):
    called = False

    def fail_if_called(*_args):
        nonlocal called
        called = True
        raise AssertionError("analysis must not run")

    monkeypatch.setattr(
        backend.identifiability_exp,
        "run_interface_srh_identifiability",
        fail_if_called,
    )
    request = _request()
    payload = dict(request.protocol)
    payload["external_validation"] = True
    request = request.model_copy(update={"protocol": payload})

    with pytest.raises(HTTPException) as exc:
        backend.run_interface_srh_identifiability(request)
    assert exc.value.status_code == 422
    assert "external_validation" in str(exc.value.detail)
    assert called is False


def test_expected_analysis_error_returns_422(monkeypatch):
    def fail(*_args):
        raise IdentifiabilityForwardError("invalid synthetic forward state")

    monkeypatch.setattr(
        backend.identifiability_exp,
        "run_interface_srh_identifiability",
        fail,
    )
    with pytest.raises(HTTPException) as exc:
        backend.run_interface_srh_identifiability(_request())
    assert exc.value.status_code == 422
    assert exc.value.detail == "invalid synthetic forward state"


def test_unexpected_analysis_error_returns_500(monkeypatch):
    def fail(*_args):
        raise RuntimeError("unexpected optimizer failure")

    monkeypatch.setattr(
        backend.identifiability_exp,
        "run_interface_srh_identifiability",
        fail,
    )
    with pytest.raises(HTTPException) as exc:
        backend.run_interface_srh_identifiability(_request())
    assert exc.value.status_code == 500
    assert exc.value.detail == "unexpected optimizer failure"


def test_request_rejects_unknown_top_level_fields():
    payload = _request().model_dump()
    payload["material_parameters"] = {"N_t": 1.0e12}
    with pytest.raises(ValidationError, match="material_parameters"):
        backend.InterfaceSRHIdentifiabilityRequest(**payload)
