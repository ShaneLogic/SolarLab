from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException

import backend.main as backend
from perovskite_sim.twod.experiments.jv_sweep_2d import JV2DResult
from perovskite_sim.twod.microstructure import GrainBoundary, Microstructure


class _CaptureRegistry:
    def __init__(self) -> None:
        self.fn = None

    def submit(self, fn):
        self.fn = fn
        return "twod-job"


def _result(lateral_bc: str) -> JV2DResult:
    return JV2DResult(
        V=np.array([0.0]),
        J=np.array([0.0]),
        snapshots=(),
        grid_x=np.array([0.0, 1.0]),
        grid_y=np.array([0.0, 1.0]),
        lateral_bc=lateral_bc,
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
        return _result(str(kwargs["lateral_bc"]))

    monkeypatch.setattr(jv_2d, "run_jv_sweep_2d", fake_run)
    return registry, captured


def _run_captured(registry: _CaptureRegistry) -> None:
    assert registry.fn is not None
    registry.fn(SimpleNamespace(report=lambda *_args: None))


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
