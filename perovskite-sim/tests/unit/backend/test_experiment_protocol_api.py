from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import backend.main as backend
from backend.progress import ProgressReporter
from perovskite_sim.experiments import eqe as eqe_exp
from perovskite_sim.experiments import impedance as impedance_exp
from perovskite_sim.experiments import jv_sweep
from perovskite_sim.experiments import suns_voc as suns_voc_exp
from perovskite_sim.experiments import tpv as tpv_exp
from perovskite_sim.experiments.protocol import (
    ExperimentProtocol,
    ExperimentProtocolError,
)


def _minimal_stack():
    return backend.stack_from_dict(
        {
            "device": {"mode": "full", "temperature": 300.0},
            "layers": [
                {
                    "name": "ABS",
                    "role": "absorber",
                    "thickness": 5e-7,
                    "eps_r": 10.0,
                    "mu_n": 1e-4,
                    "mu_p": 1e-4,
                    "D_ion": 0.0,
                    "P_lim": 1e26,
                    "P0": 1e24,
                    "ni": 1e15,
                    "tau_n": 1e-9,
                    "tau_p": 1e-9,
                    "n1": 1e15,
                    "p1": 1e15,
                    "B_rad": 0.0,
                    "C_n": 0.0,
                    "C_p": 0.0,
                    "alpha": 0.0,
                    "N_A": 0.0,
                    "N_D": 0.0,
                }
            ],
        }
    )


@dataclass(frozen=True)
class _DummyResult:
    value: float = 1.0


class _CaptureRegistry:
    def __init__(self) -> None:
        self.fn = None

    def submit(self, fn):
        self.fn = fn
        return "protocol-job"


def test_direct_jv_default_and_explicit_protocol_preserve_solver_arguments(
    monkeypatch,
):
    stack = _minimal_stack()
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(backend, "build_stack", lambda *_args: stack)

    def fake_run(_stack, **kwargs):
        calls.append(dict(kwargs))
        return _DummyResult()

    monkeypatch.setattr(backend.jv_sweep, "run_jv_sweep", fake_run)
    common = {
        "device": {"device": {}, "layers": []},
        "N_grid": 12,
        "n_points": 3,
        "v_rate": 2.0,
        "V_max": 0.4,
    }
    explicit = jv_sweep.build_jv_experiment_protocol(
        stack,
        n_points=3,
        v_rate=2.0,
        V_max=0.4,
    )

    backend.run_jv(backend.JVRequest(**common))
    backend.run_jv(
        backend.JVRequest(
            **common,
            protocol_mode="research_strict",
            experiment_protocol=explicit.to_dict(),
        )
    )

    default_call, explicit_call = calls
    assert default_call.pop("experiment_protocol") is None
    assert default_call.pop("protocol_mode") == "compatibility"
    assert explicit_call.pop("experiment_protocol") == explicit
    assert explicit_call.pop("protocol_mode") == "research_strict"
    assert default_call == explicit_call


@pytest.mark.parametrize("solver", ["steady_state", "quasi_fermi"])
def test_nontransient_jv_dispatch_rejects_explicit_protocol(solver):
    protocol = jv_sweep.build_jv_experiment_protocol(
        _minimal_stack(),
        n_points=3,
        V_max=0.4,
    )

    with pytest.raises(ExperimentProtocolError, match="only by solver='transient'"):
        backend._run_jv_dispatch(
            None,
            N_grid=12,
            n_points=3,
            v_rate=0.1,
            V_max=0.4,
            illuminated=True,
            solver=solver,
            experiment_protocol=protocol,
        )


def test_direct_jv_invalid_protocol_payload_maps_to_http_422(monkeypatch):
    monkeypatch.setattr(backend, "build_stack", lambda *_args: object())

    with pytest.raises(backend.HTTPException) as exc_info:
        backend.run_jv(
            backend.JVRequest(
                device={"device": {}, "layers": []},
                experiment_protocol={"unknown": "field"},
            )
        )

    assert exc_info.value.status_code == 422
    assert "keys do not match schema" in str(exc_info.value.detail)


@pytest.mark.parametrize("use_implicit_payload", [False, True])
def test_direct_jv_research_strict_rejects_implicit_history(
    monkeypatch,
    use_implicit_payload,
):
    stack = _minimal_stack()
    monkeypatch.setattr(backend, "build_stack", lambda *_args: stack)
    payload = None
    if use_implicit_payload:
        payload = jv_sweep.build_jv_experiment_protocol(
            stack,
            n_points=3,
            V_max=0.4,
            implicit_legacy_protocol=True,
        ).to_dict()

    with pytest.raises(backend.HTTPException) as exc_info:
        backend.run_jv(
            backend.JVRequest(
                device={"device": {}, "layers": []},
                n_points=3,
                V_max=0.4,
                protocol_mode="research_strict",
                experiment_protocol=payload,
            )
        )

    assert exc_info.value.status_code == 422
    assert "requires an explicit experiment history" in str(exc_info.value.detail)


def test_direct_jv_protocol_mismatch_maps_to_http_422(monkeypatch):
    stack = _minimal_stack()
    monkeypatch.setattr(backend, "build_stack", lambda *_args: stack)
    mismatched = jv_sweep.build_jv_experiment_protocol(
        stack,
        n_points=3,
        v_rate=1.0,
        V_max=0.4,
    )

    with pytest.raises(backend.HTTPException) as exc_info:
        backend.run_jv(
            backend.JVRequest(
                device={"device": {}, "layers": []},
                N_grid=3,
                n_points=3,
                v_rate=2.0,
                V_max=0.4,
                experiment_protocol=mismatched.to_dict(),
            )
        )

    assert exc_info.value.status_code == 422
    assert "does not match the requested execution" in str(exc_info.value.detail)


def test_direct_impedance_forwards_parsed_protocol(monkeypatch):
    stack = _minimal_stack()
    frequencies = np.logspace(0.0, 2.0, 3)
    protocol = impedance_exp.build_impedance_experiment_protocol(
        stack,
        frequencies,
        V_dc=0.2,
        delta_V=0.005,
        n_cycles=3,
        n_extract=1,
        points_per_cycle=20,
        dc_settle_time=0.01,
        method="transient_ion_aware",
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(backend, "build_stack", lambda *_args: stack)

    def fake_run(_stack, _frequencies, **kwargs):
        captured.update(kwargs)
        return impedance_exp.ImpedanceResult(
            frequencies=frequencies,
            Z=np.ones(3, dtype=complex),
        )

    monkeypatch.setattr(backend.impedance, "run_impedance", fake_run)
    backend.run_impedance_api(
        backend.ISRequest(
            device={"device": {}, "layers": []},
            n_freq=3,
            f_min=1.0,
            f_max=100.0,
            V_dc=0.2,
            delta_V=0.005,
            n_cycles=3,
            n_extract=1,
            points_per_cycle=20,
            dc_settle_time=0.01,
            protocol_mode="research_strict",
            experiment_protocol=protocol.to_dict(),
        )
    )

    assert captured["experiment_protocol"] == protocol
    assert captured["protocol_mode"] == "research_strict"


@pytest.mark.parametrize(
    "kind",
    ["jv", "impedance", "tpv", "suns_voc", "eqe"],
)
def test_jobs_forward_parsed_protocol_and_mode(monkeypatch, kind):
    stack = _minimal_stack()
    registry = _CaptureRegistry()
    captured: dict[str, object] = {}
    monkeypatch.setattr(backend, "_JOB_REGISTRY", registry)
    monkeypatch.setattr(backend, "build_stack", lambda *_args: stack)

    if kind == "jv":
        params = {"n_points": 3, "v_rate": 2.0, "V_max": 0.4}
        protocol = jv_sweep.build_jv_experiment_protocol(
            stack, n_points=3, v_rate=2.0, V_max=0.4
        )

        def fake_run(_stack, **kwargs):
            captured.update(kwargs)
            return _DummyResult()

        monkeypatch.setattr(backend, "_run_jv_dispatch", fake_run)
    elif kind == "impedance":
        frequencies = np.logspace(0.0, 2.0, 3)
        params = {"n_freq": 3, "f_min": 1.0, "f_max": 100.0}
        protocol = impedance_exp.build_impedance_experiment_protocol(
            stack, frequencies
        )

        def fake_run(_stack, frequencies, **kwargs):
            captured.update(kwargs)
            return _DummyResult()

        monkeypatch.setattr(backend.impedance, "run_impedance", fake_run)
    elif kind == "tpv":
        params = {"n_points": 20, "t_pulse": 1e-6, "t_decay": 50e-6}
        protocol = tpv_exp.build_tpv_experiment_protocol(
            stack, n_points=20, t_pulse=1e-6, t_decay=50e-6
        )

        def fake_run(_stack, **kwargs):
            captured.update(kwargs)
            return _DummyResult()

        monkeypatch.setattr(tpv_exp, "run_tpv", fake_run)
    elif kind == "suns_voc":
        params = {"suns_levels": [0.1, 1.0], "t_settle": 0.02}
        protocol = suns_voc_exp.build_suns_voc_experiment_protocol(
            stack, (0.1, 1.0), t_settle=0.02
        )

        def fake_run(_stack, **kwargs):
            captured.update(kwargs)
            return _DummyResult()

        monkeypatch.setattr(backend.suns_voc_exp, "run_suns_voc", fake_run)
    else:
        wavelengths = np.linspace(400.0, 600.0, 3)
        params = {
            "lambda_min_nm": 400.0,
            "lambda_max_nm": 600.0,
            "n_lambda": 3,
            "Phi_incident": 1e20,
            "t_settle": 0.02,
        }
        protocol = eqe_exp.build_eqe_experiment_protocol(
            stack,
            wavelengths,
            Phi_incident=1e20,
            t_settle=0.02,
        )

        def fake_run(_stack, **kwargs):
            captured.update(kwargs)
            return _DummyResult()

        monkeypatch.setattr(backend.eqe_exp, "compute_eqe", fake_run)

    params.update(
        protocol_mode="research_strict",
        experiment_protocol=protocol.to_dict(),
    )
    response = backend.start_job(
        backend.JobRequest(
            kind=kind,
            device={"device": {}, "layers": []},
            params=params,
        )
    )
    assert response == {"status": "ok", "job_id": "protocol-job"}
    assert registry.fn is not None
    registry.fn(ProgressReporter())

    assert captured["experiment_protocol"] == ExperimentProtocol.from_dict(
        protocol.to_dict()
    )
    assert captured["protocol_mode"] == "research_strict"


@pytest.mark.parametrize(
    ("kind", "params", "protocol_builder"),
    [
        (
            "jv",
            {"n_points": 3, "v_rate": 2.0, "V_max": 0.4},
            lambda stack: jv_sweep.build_jv_experiment_protocol(
                stack, n_points=3, v_rate=1.0, V_max=0.4
            ),
        ),
        (
            "impedance",
            {
                "n_freq": 3,
                "f_min": 1.0,
                "f_max": 100.0,
                "delta_V": 0.02,
            },
            lambda stack: impedance_exp.build_impedance_experiment_protocol(
                stack,
                np.logspace(0.0, 2.0, 3),
                delta_V=0.01,
                method="transient_ion_aware",
            ),
        ),
        (
            "tpv",
            {"n_points": 20, "t_pulse": 2e-6, "t_decay": 50e-6},
            lambda stack: tpv_exp.build_tpv_experiment_protocol(
                stack, n_points=20, t_pulse=1e-6, t_decay=50e-6
            ),
        ),
        (
            "suns_voc",
            {"suns_levels": [0.1, 1.0], "t_settle": 0.02},
            lambda stack: suns_voc_exp.build_suns_voc_experiment_protocol(
                stack, (0.1, 1.0), t_settle=0.01
            ),
        ),
        (
            "eqe",
            {
                "lambda_min_nm": 400.0,
                "lambda_max_nm": 600.0,
                "n_lambda": 3,
                "Phi_incident": 2e20,
                "t_settle": 0.02,
            },
            lambda stack: eqe_exp.build_eqe_experiment_protocol(
                stack,
                np.linspace(400.0, 600.0, 3),
                Phi_incident=1e20,
                t_settle=0.02,
            ),
        ),
    ],
)
def test_job_protocol_mismatch_rejected_before_submit(
    monkeypatch,
    kind,
    params,
    protocol_builder,
):
    stack = _minimal_stack()
    registry = _CaptureRegistry()
    monkeypatch.setattr(backend, "_JOB_REGISTRY", registry)
    monkeypatch.setattr(backend, "build_stack", lambda *_args: stack)
    params = {
        **params,
        "experiment_protocol": protocol_builder(stack).to_dict(),
    }

    with pytest.raises(backend.HTTPException) as exc_info:
        backend.start_job(
            backend.JobRequest(
                kind=kind,
                device={"device": {}, "layers": []},
                params=params,
            )
        )

    assert exc_info.value.status_code == 422
    assert "does not match the requested execution" in str(exc_info.value.detail)
    assert registry.fn is None


def test_job_research_strict_without_explicit_protocol_returns_http_422():
    with pytest.raises(backend.HTTPException) as exc_info:
        backend.start_job(
            backend.JobRequest(
                kind="tpv",
                device={"device": {}, "layers": []},
                params={"protocol_mode": "research_strict"},
            )
        )

    assert exc_info.value.status_code == 422
    assert "requires an explicit experiment history" in str(exc_info.value.detail)


def test_job_unknown_protocol_payload_returns_http_422():
    with pytest.raises(backend.HTTPException) as exc_info:
        backend.start_job(
            backend.JobRequest(
                kind="eqe",
                device={"device": {}, "layers": []},
                params={"experiment_protocol": {"schema_version": 1}},
            )
        )

    assert exc_info.value.status_code == 422
    assert "keys do not match schema" in str(exc_info.value.detail)
