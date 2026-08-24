from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException

import backend.main as backend
from perovskite_sim.twod.experiments.jv_sweep_2d import JV2DResult
from perovskite_sim.twod.experiments.jv_sweep_2d import (
    build_jv_2d_execution_protocol,
)
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.solver.tolerances import ComponentwiseAtol
from perovskite_sim.twod.microstructure import GrainBoundary, Microstructure


class _CaptureRegistry:
    def __init__(self) -> None:
        self.fn = None

    def submit(self, fn):
        self.fn = fn
        return "twod-job"


def _result(lateral_bc: str, protocol=None) -> JV2DResult:
    return JV2DResult(
        V=np.array([0.0]),
        J=np.array([0.0]),
        snapshots=(),
        grid_x=np.array([0.0, 1.0]),
        grid_y=np.array([0.0, 1.0]),
        lateral_bc=lateral_bc,
        protocol=protocol,
    )


def _install_fakes(monkeypatch, stack):
    registry = _CaptureRegistry()
    captured: dict[str, object] = {}
    monkeypatch.setattr(backend, "_JOB_REGISTRY", registry)
    monkeypatch.setattr(backend, "build_stack", lambda *_args: stack)
    monkeypatch.setattr(
        backend,
        "_describe_active_physics",
        lambda _stack: "Active physics: test",
    )

    import perovskite_sim.twod.experiments.jv_sweep_2d as jv_2d

    def fake_run(*, stack, **kwargs):
        del stack
        captured.update(kwargs)
        return _result(
            str(kwargs["lateral_bc"]),
            protocol=kwargs.get("jv_2d_protocol"),
        )

    monkeypatch.setattr(jv_2d, "run_jv_sweep_2d", fake_run)
    return registry, captured


def _run_captured(registry: _CaptureRegistry):
    assert registry.fn is not None
    return registry.fn(SimpleNamespace(report=lambda *_args: None))


def test_jv_2d_empty_microstructure_defaults_to_periodic(monkeypatch):
    stack = SimpleNamespace(microstructure=Microstructure())
    registry, captured = _install_fakes(monkeypatch, stack)
    response = backend.start_job(backend.JobRequest(kind="jv_2d", params={}))
    assert response == {"status": "ok", "job_id": "twod-job"}
    _run_captured(registry)
    assert captured["lateral_bc"] == "periodic"


def test_jv_2d_inline_grain_boundary_defaults_to_neumann(monkeypatch):
    stack = SimpleNamespace(microstructure=Microstructure())
    registry, captured = _install_fakes(monkeypatch, stack)
    response = backend.start_job(
        backend.JobRequest(
            kind="jv_2d",
            params={
                "microstructure": {
                    "grain_boundaries": [
                        {
                            "x_position": 250e-9,
                            "width": 5e-9,
                            "tau_n": 5e-8,
                            "tau_p": 5e-8,
                        }
                    ]
                }
            },
        )
    )
    assert response["job_id"] == "twod-job"
    _run_captured(registry)
    assert captured["lateral_bc"] == "neumann"


def test_jv_2d_config_grain_boundary_defaults_to_neumann(monkeypatch):
    stack = SimpleNamespace(
        microstructure=Microstructure(
            (GrainBoundary(250e-9, 5e-9, 5e-8, 5e-8),)
        )
    )
    registry, captured = _install_fakes(monkeypatch, stack)
    backend.start_job(backend.JobRequest(kind="jv_2d", params={}))
    _run_captured(registry)
    assert captured["lateral_bc"] == "neumann"


def test_jv_2d_periodic_grain_boundary_rejected_before_submit(monkeypatch):
    stack = SimpleNamespace(
        microstructure=Microstructure(
            (GrainBoundary(250e-9, 5e-9, 5e-8, 5e-8),)
        )
    )
    registry, _captured = _install_fakes(monkeypatch, stack)
    with pytest.raises(HTTPException) as error:
        backend.start_job(
            backend.JobRequest(
                kind="jv_2d",
                params={"lateral_bc": "periodic"},
            )
        )
    assert error.value.status_code == 422
    assert "not area-certified" in error.value.detail
    assert registry.fn is None


def _mobile_protocol(stack):
    return build_jv_2d_execution_protocol(
        stack,
        Microstructure(),
        lateral_length=500e-9,
        Nx=10,
        V_max=1.2,
        V_step=0.05,
        illuminated=True,
        lateral_bc="neumann",
        Ny_per_layer=20,
        settle_t=1.0e-7,
        save_snapshots=True,
        ion_dynamics="single_mobile",
        atol=ComponentwiseAtol(),
        max_nfev_per_solve=200_000,
        max_bisect=6,
        ion_inventory_rtol=1.0e-9,
        initial_state_settle_s=1.0e-3,
    )


def test_jv_2d_mobile_requires_strict_protocol_before_submit(monkeypatch):
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    registry, _captured = _install_fakes(monkeypatch, stack)

    with pytest.raises(HTTPException) as error:
        backend.start_job(
            backend.JobRequest(
                kind="jv_2d",
                params={
                    "lateral_bc": "neumann",
                    "ion_dynamics": "single_mobile",
                },
            )
        )

    assert error.value.status_code == 422
    assert "research_strict" in error.value.detail
    assert registry.fn is None


def test_jv_2d_mobile_protocol_mismatch_rejected_before_submit(monkeypatch):
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    registry, _captured = _install_fakes(monkeypatch, stack)
    protocol = replace(
        _mobile_protocol(stack),
        dwell_time_per_voltage_s=2.0e-7,
    )

    with pytest.raises(HTTPException) as error:
        backend.start_job(
            backend.JobRequest(
                kind="jv_2d",
                params={
                    "lateral_bc": "neumann",
                    "ion_dynamics": "single_mobile",
                    "protocol_mode": "research_strict",
                    "jv_2d_protocol": protocol.to_dict(),
                },
            )
        )

    assert error.value.status_code == 422
    assert "dwell_time_per_voltage_s" in error.value.detail
    assert registry.fn is None


def test_jv_2d_matching_mobile_protocol_is_forwarded_and_serialized(monkeypatch):
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    registry, captured = _install_fakes(monkeypatch, stack)
    protocol = _mobile_protocol(stack)

    response = backend.start_job(
        backend.JobRequest(
            kind="jv_2d",
            params={
                "lateral_bc": "neumann",
                "ion_dynamics": "single_mobile",
                "protocol_mode": "research_strict",
                "jv_2d_protocol": protocol.to_dict(),
            },
        )
    )
    output = _run_captured(registry)

    assert response["job_id"] == "twod-job"
    assert captured["jv_2d_protocol"] == protocol
    assert captured["protocol_mode"] == "research_strict"
    assert isinstance(captured["atol"], ComponentwiseAtol)
    assert output["protocol"] == protocol.to_dict()
    assert output["protocol_hash"] == protocol.protocol_hash


def test_jv_2d_rejects_ambiguous_absolute_tolerance_before_submit(monkeypatch):
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    registry, _captured = _install_fakes(monkeypatch, stack)

    with pytest.raises(HTTPException) as error:
        backend.start_job(
            backend.JobRequest(
                kind="jv_2d",
                params={
                    "atol": 1.0e-8,
                    "componentwise_atol": asdict(ComponentwiseAtol()),
                },
            )
        )

    assert error.value.status_code == 422
    assert "either atol or componentwise_atol" in error.value.detail
    assert registry.fn is None
