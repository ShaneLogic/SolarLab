from __future__ import annotations

from pathlib import Path

import pytest

import backend.main as backend
from backend.progress import ProgressReporter
from perovskite_sim.experiments.dynamic_defect_transient import (
    build_dynamic_defect_transient_protocol,
)
from perovskite_sim.experiments.jv_sweep import build_electrical_grid
from perovskite_sim.experiments.quasi_fermi_steady_state import (
    build_two_sided_trace_grid,
)
from perovskite_sim.models.config_loader import load_device_from_yaml


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs/dynamic_interface_defect_ion_transient_absorber_only.yaml"
TIMES_S = (0.0, 1.0e-8, 1.0e-6, 1.0e-4)
VOLTAGE_V = (0.0, 0.05, 0.05, 0.05)


def _stack():
    return load_device_from_yaml(CONFIG)


def _protocol(stack):
    grid = build_two_sided_trace_grid(build_electrical_grid(stack, 4), stack)
    return build_dynamic_defect_transient_protocol(
        stack,
        grid,
        TIMES_S,
        VOLTAGE_V,
        requested_grid_intervals=4,
    )


def test_direct_api_builds_and_forwards_exact_protocol(monkeypatch):
    stack = _stack()
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        backend,
        "build_dynamic_defect_transient_stack",
        lambda *_args: stack,
    )

    def fake_run(grid, run_stack, protocol, **kwargs):
        captured.update(
            grid=grid,
            stack=run_stack,
            protocol=protocol,
            kwargs=kwargs,
        )
        return {"protocol": protocol, "evidence": {"certified": True}}

    monkeypatch.setattr(
        backend.dynamic_defect_transient_exp,
        "run_dynamic_defect_transient",
        fake_run,
    )
    response = backend.run_dynamic_defect_transient_api(
        backend.DynamicDefectTransientRequest(
            device={"device": {}, "layers": []},
            N_grid=4,
            times_s=TIMES_S,
            voltage_V=VOLTAGE_V,
        )
    )

    protocol = captured["protocol"]
    assert protocol.capability == "interface_defect_plus_positive_ions"
    assert protocol.requested_grid_intervals == 4
    assert protocol.times_s == TIMES_S
    assert protocol.voltage_V == VOLTAGE_V
    assert protocol.time_step_refinement_factor == 1.0
    assert protocol.solver_policy.refinement_substeps == (1, 2, 4)
    assert captured["stack"] is stack
    assert captured["kwargs"] == {}
    assert response["result"]["protocol"]["method"] == (
        "dynamic_defect_transient_certified"
    )
    assert response["result"]["protocol"]["time_step_refinement_factor"] == 1.0
    assert response["result"]["evidence"] == {"certified": True}


def test_direct_api_rejects_mismatched_protocol(monkeypatch):
    stack = _stack()
    payload = _protocol(stack).to_dict()
    payload["voltage_V"] = [0.0, 0.04, 0.04, 0.04]
    monkeypatch.setattr(
        backend,
        "build_dynamic_defect_transient_stack",
        lambda *_args: stack,
    )

    with pytest.raises(backend.HTTPException) as exc_info:
        backend.run_dynamic_defect_transient_api(
            backend.DynamicDefectTransientRequest(
                device={"device": {}, "layers": []},
                N_grid=4,
                times_s=TIMES_S,
                voltage_V=VOLTAGE_V,
                dynamic_defect_transient_protocol=payload,
            )
        )

    assert exc_info.value.status_code == 422
    assert "does not match" in str(exc_info.value.detail)


def test_job_preflights_and_forwards_resolved_protocol(monkeypatch):
    stack = _stack()
    submitted: dict[str, object] = {}
    captured: dict[str, object] = {}

    class CaptureRegistry:
        def submit(self, fn):
            submitted["fn"] = fn
            return "dynamic-transient-job"

    monkeypatch.setattr(backend, "_JOB_REGISTRY", CaptureRegistry())
    monkeypatch.setattr(
        backend,
        "build_dynamic_defect_transient_stack",
        lambda *_args: stack,
    )

    def fake_run(grid, run_stack, protocol, **kwargs):
        captured.update(
            grid=grid,
            stack=run_stack,
            protocol=protocol,
            kwargs=kwargs,
        )
        return {"protocol": protocol, "evidence": {"certified": True}}

    monkeypatch.setattr(
        backend.dynamic_defect_transient_exp,
        "run_dynamic_defect_transient",
        fake_run,
    )
    response = backend.start_job(
        backend.JobRequest(
            kind="dynamic_defect_transient",
            device={"device": {}, "layers": []},
            params={
                "N_grid": 4,
                "times_s": list(TIMES_S),
                "voltage_V": list(VOLTAGE_V),
                "illuminated": False,
                "method": "dynamic_defect_transient_certified",
            },
        )
    )

    assert response == {"status": "ok", "job_id": "dynamic-transient-job"}
    result = submitted["fn"](ProgressReporter())
    assert result["evidence"] == {"certified": True}
    assert captured["protocol"].times_s == TIMES_S
    assert callable(captured["kwargs"]["progress"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("protocol", "does not match"),
        ("extra", "extra"),
        ("bool", "illuminated must be boolean"),
    ],
)
def test_job_rejects_invalid_contract_before_submit(monkeypatch, mutation, message):
    stack = _stack()
    params: dict[str, object] = {
        "N_grid": 4,
        "times_s": list(TIMES_S),
        "voltage_V": list(VOLTAGE_V),
        "illuminated": False,
        "method": "dynamic_defect_transient_certified",
    }
    if mutation == "protocol":
        payload = _protocol(stack).to_dict()
        payload["grid_sha256"] = "0" * 64
        params["dynamic_defect_transient_protocol"] = payload
    elif mutation == "extra":
        params["unregistered_control"] = 1
    else:
        params["illuminated"] = "false"

    class RejectSubmitRegistry:
        def submit(self, _fn):
            raise AssertionError("invalid job must fail before submit")

    monkeypatch.setattr(backend, "_JOB_REGISTRY", RejectSubmitRegistry())
    monkeypatch.setattr(
        backend,
        "build_dynamic_defect_transient_stack",
        lambda *_args: stack,
    )

    with pytest.raises(backend.HTTPException) as exc_info:
        backend.start_job(
            backend.JobRequest(
                kind="dynamic_defect_transient",
                device={"device": {}, "layers": []},
                params=params,
            )
        )

    assert exc_info.value.status_code == 422
    assert message in str(exc_info.value.detail)
