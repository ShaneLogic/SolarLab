from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException

import backend.main as backend
from backend.progress import ProgressReporter
from perovskite_sim.experiments import waveform_jv as wf
from perovskite_sim.models.config_loader import load_device_from_yaml


@pytest.fixture
def stack(monkeypatch):
    stack = load_device_from_yaml("configs/calado2016_fig1f.yaml")
    monkeypatch.setattr(backend, "build_stack", lambda *_args: stack)
    monkeypatch.setattr(backend, "build_jv_stack", lambda *_args: stack)
    return stack


def parameters():
    return {"N_grid": 15, "n_points": 3, "V_max": 1.2, "v_rate": 0.04,
            "waveform": asdict(wf.JVWaveform()), "waveform_controls": {"rtol": 1e-4, "atol_m3": 1.0}}


def test_research_default_matches_driver_and_explicit_control_is_preserved(stack):
    request = parameters()
    request.pop("waveform_controls")
    _, controls = backend._parse_jv_waveform(stack, request)
    assert controls == {"rtol": 1e-4, "atol_m3": wf.DEFAULT_DENSITY_ATOL_M3}
    assert controls["atol_m3"] == 100.0
    _, explicit = backend._parse_jv_waveform(stack, parameters())
    assert explicit == {"rtol": 1e-4, "atol_m3": 1.0}


@pytest.mark.parametrize("change", [
    {"solver": "steady_state"}, {"solver": "quasi_fermi"},
    {"waveform_controls": {"atol_m3": True, "rtol": 1e-4}},
    {"waveform_controls": {"atol_m3": 0, "rtol": 1e-4}},
    {"V_max": None}, {"waveform": {"start_voltage_V": -1}},
])
def test_job_rejects_invalid_waveform_before_submission(stack, monkeypatch, change):
    def forbidden(_fn):
        pytest.fail("invalid waveform reached the worker")
    monkeypatch.setattr(backend._JOB_REGISTRY, "submit", forbidden)
    with pytest.raises(HTTPException) as error:
        backend.start_job(backend.JobRequest(kind="jv", device={}, params={**parameters(), **change}))
    assert error.value.status_code == 422


@pytest.mark.parametrize("kind", ["jv", "current_decomp", "spatial"])
def test_same_waveform_and_materials_reach_all_jv_views(stack, monkeypatch, kind):
    captured = {}
    registry = SimpleNamespace(fn=None)
    def submit(fn):
        registry.fn = fn
        return "waveform-job"
    monkeypatch.setattr(backend._JOB_REGISTRY, "submit", submit)
    def run(device, waveform, **kwargs):
        captured.update(kwargs, device=device, waveform=waveform)
        voltage = np.array([-1, 0, 1.2])
        current = np.array([160, 160, -10])
        metrics = wf.jv.compute_metrics(voltage, current)
        components = wf.jv.JVCurrentDecomp(*(current.copy() for _ in range(5)))
        return wf.WaveformJVResult(
            V_fwd=voltage, J_fwd=current, V_rev=voltage[::-1], J_rev=current[::-1],
            metrics_fwd=metrics, metrics_rev=metrics, hysteresis_index=0,
            waveform=waveform, hysteresis_index_paper=0,
            decomp_fwd=components, decomp_rev=components,
            snapshots_fwd=(), snapshots_rev=(),
            numerical_controls={"atol_m3": kwargs["atol"], "rtol": kwargs["rtol"]},
            inventory_relative_drift=(0,), generation_budget_A_m2=160,
            waveform_protocol_sha256="0" * 64,
        )
    monkeypatch.setattr(wf, "run_waveform_jv", run)
    before = asdict(stack)
    backend.start_job(backend.JobRequest(kind=kind, device={}, params=parameters()))
    out = registry.fn(ProgressReporter())
    assert captured["device"] is stack
    assert asdict(stack) == before
    assert captured["waveform"] == wf.JVWaveform()
    assert captured["atol"] == 1
    assert captured["save_snapshots"] == (kind == "spatial")
    assert captured["decompose_currents"] == (kind == "current_decomp")
    assert out["waveform"] == asdict(wf.JVWaveform())
    assert out["hysteresis_index_paper"] == 0
    assert out["numerical_scope"] == "finite_time_diagnostic"
    assert out["jacobian_evaluator"] == "finite_difference"
    assert out["jacobian_fallback_reason"] is None
