from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

import backend.main as backend
from backend.progress import ProgressReporter
from perovskite_sim.experiments import interface_charge_jv as charged_jv
from perovskite_sim.experiments import jv_sweep
from perovskite_sim.models.config_loader import load_device_from_yaml


def _charged_stack():
    return load_device_from_yaml("configs/interface_charge_jv_research.yaml")


def _charged_params(**overrides):
    return {
        "N_grid": 30,
        "n_points": 5,
        "v_rate": 0.0,
        "V_max": 0.1,
        "illuminated": True,
        "solver": "quasi_fermi",
        "iface_states": False,
        "interface_boundary": True,
        "interface_transport_model": "fermi_dirac_richardson",
        **overrides,
    }


class _CaptureRegistry:
    def __init__(self) -> None:
        self.fn = None

    def submit(self, fn):
        self.fn = fn
        return "charged-jv-job"


def _resolve(stack, **overrides):
    params = _charged_params(**overrides)
    return backend._resolve_interface_charge_jv_protocol(
        stack,
        N_grid=params["N_grid"],
        n_points=params["n_points"],
        v_rate=params["v_rate"],
        V_max=params["V_max"],
        illuminated=params["illuminated"],
        solver=params["solver"],
        iface_states=params["iface_states"],
        interface_boundary=params["interface_boundary"],
        interface_transport_model=params["interface_transport_model"],
        experiment_protocol=None,
        protocol_mode="compatibility",
        supplied_protocol=params.get("interface_charge_jv_protocol"),
    )


def test_charged_protocol_resolver_builds_the_canonical_zero_scan_contract():
    stack = _charged_stack()
    protocol = _resolve(stack)

    assert isinstance(protocol, charged_jv.InterfaceChargeJVProtocol)
    assert protocol.voltages_V == pytest.approx((0.0, 0.025, 0.05, 0.075, 0.1))
    assert protocol.temperature_K == pytest.approx(stack.T)
    assert protocol.branch_semantics == "ascending_zero_scan_rate"
    assert protocol.interface_transport_model == "fermi_dirac_richardson"
    assert len(protocol.protocol_sha256) == 64


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("solver", "transient", "solver='quasi_fermi'"),
        ("v_rate", 0.1, "v_rate=0"),
        ("illuminated", False, "illuminated=true"),
        ("iface_states", True, "iface_states=false"),
        ("interface_boundary", False, "interface_boundary=true"),
        (
            "interface_transport_model",
            "fermi_richardson",
            "fermi_dirac_richardson",
        ),
        ("N_grid", 2, "N_grid must be an integer >= 3"),
        ("n_points", 1, "n_points must be an integer >= 2"),
        ("V_max", 0.0, "finite V_max > 0"),
    ],
)
def test_charged_protocol_resolver_rejects_execution_mismatch(
    field,
    value,
    message,
):
    with pytest.raises(charged_jv.InterfaceChargeJVProtocolError, match=message):
        _resolve(_charged_stack(), **{field: value})


def test_charged_protocol_resolver_rejects_mobile_ions_and_bulk_defects():
    stack = _charged_stack()
    first = stack.layers[0]
    ionic = replace(
        stack,
        layers=(
            replace(first, params=replace(first.params, D_ion=1.0e-16)),
            *stack.layers[1:],
        ),
    )
    with pytest.raises(charged_jv.InterfaceChargeJVProtocolError, match="ion-free"):
        _resolve(ionic)

    explicit_params = SimpleNamespace(
        **{
            **first.params.__dict__,
            "defect_model": "explicit_quasi_steady",
        },
    )
    bulk = replace(
        stack,
        layers=(replace(first, params=explicit_params), *stack.layers[1:]),
    )
    with pytest.raises(
        charged_jv.InterfaceChargeJVProtocolError,
        match="excludes explicit bulk-defect composition",
    ):
        _resolve(bulk)


def test_charged_protocol_resolver_rejects_mismatched_or_charge_off_payload():
    stack = _charged_stack()
    expected = _resolve(stack)
    mismatched = replace(expected, voltages_V=(0.0, 0.04, 0.08, 0.12))
    with pytest.raises(
        charged_jv.InterfaceChargeJVProtocolError,
        match="does not match",
    ):
        _resolve(stack, interface_charge_jv_protocol=mismatched.to_dict())

    charge_off = replace(stack, interface_charge_closure="off")
    with pytest.raises(
        charged_jv.InterfaceChargeJVProtocolError,
        match="requires interface_charge_closure",
    ):
        backend._resolve_interface_charge_jv_protocol(
            charge_off,
            N_grid=30,
            n_points=5,
            v_rate=0.0,
            V_max=0.1,
            illuminated=True,
            solver="quasi_fermi",
            iface_states=False,
            interface_boundary=True,
            interface_transport_model="fermi_dirac_richardson",
            experiment_protocol=None,
            protocol_mode="compatibility",
            supplied_protocol=expected.to_dict(),
        )


def test_sync_charged_jv_returns_jv_compatible_curve_and_full_evidence(monkeypatch):
    stack = _charged_stack()
    captured: dict[str, object] = {}
    monkeypatch.setattr(backend, "build_jv_stack", lambda *_args: stack)

    def fake_dispatch(_stack, **kwargs):
        protocol = kwargs["interface_charge_jv_protocol"]
        captured.update(kwargs)
        metrics = jv_sweep.JVMetrics(
            V_oc=0.078,
            J_sc=0.013,
            FF=0.6,
            PCE=6.0e-7,
            voc_bracketed=True,
        )
        return jv_sweep.JVResult(
            V_fwd=np.asarray([0.0, 0.05, 0.1]),
            J_fwd=np.asarray([0.013, 0.004, -0.01]),
            V_rev=np.asarray([0.0, 0.05, 0.1]),
            J_rev=np.asarray([0.013, 0.004, -0.01]),
            metrics_fwd=metrics,
            metrics_rev=metrics,
            hysteresis_index=0.0,
            interface_charge_evidence={
                "model": "interface-charge-jv-evidence-v1",
                "protocol": protocol.to_dict(),
                "protocol_sha256": protocol.protocol_sha256,
                "dark_state_sha256": "d" * 64,
            },
        )

    monkeypatch.setattr(backend, "_run_jv_dispatch", fake_dispatch)
    with TestClient(backend.app) as client:
        response = client.post(
            "/api/jv",
            json={
                "config_path": "interface_charge_research.yaml",
                **{k: v for k, v in _charged_params().items() if k != "illuminated"},
            },
        )

    assert response.status_code == 200, response.text
    result = response.json()["result"]
    evidence = result["interface_charge_evidence"]
    assert result["V_fwd"] == result["V_rev"]
    assert result["J_fwd"] == result["J_rev"]
    assert result["hysteresis_index"] == 0.0
    assert evidence["model"] == "interface-charge-jv-evidence-v1"
    assert evidence["protocol_sha256"] == (
        captured["interface_charge_jv_protocol"].protocol_sha256
    )


def test_dispatch_maps_certified_charged_execution_without_duplicate_physics(
    monkeypatch,
):
    stack = _charged_stack()
    protocol = _resolve(stack, n_points=2)
    shared = np.asarray([0.0, 0.5, 1.0])
    two_sided = np.asarray([0.0, 0.4, 0.6, 1.0])
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        backend.jv_sweep,
        "build_electrical_grid",
        lambda *_args: shared,
    )

    def fake_grid_guard(x, _stack, **kwargs):
        captured["guard_grid"] = np.asarray(x)
        captured["guard_controls"] = kwargs

    monkeypatch.setattr(
        backend,
        "require_thick_layer_interface_resolution",
        fake_grid_guard,
    )
    monkeypatch.setattr(
        backend,
        "build_two_sided_trace_grid",
        lambda x, _stack: two_sided if np.array_equal(x, shared) else None,
    )
    points = tuple(
        SimpleNamespace(
            V_app=voltage,
            certified=True,
            current_A_m2=current,
            max_normalized_cell_residual=1.0e-8,
            electron_continuity_bound_A_m2=2.0e-8,
            hole_continuity_bound_A_m2=3.0e-8,
            face_current_spread_A_m2=4.0e-8,
            poisson_residual=5.0e-10,
        )
        for voltage, current in ((0.0, 0.01), (0.1, -0.01))
    )
    metrics = jv_sweep.JVMetrics(0.05, 0.01, 0.5, 2.5e-7, True)
    evidence = {"model": "interface-charge-jv-evidence-v1"}

    def fake_solve(x, _stack, resolved, **kwargs):
        captured["solve_grid"] = np.asarray(x)
        captured["protocol"] = resolved
        captured["progress"] = kwargs["progress"]
        return SimpleNamespace(
            sweep=SimpleNamespace(
                voltages_V=np.asarray([0.0, 0.1]),
                currents_A_m2=np.asarray([0.01, -0.01]),
                points=points,
                metrics=metrics,
            ),
            evidence=evidence,
        )

    monkeypatch.setattr(backend.interface_charge_jv_exp, "solve_interface_charge_jv", fake_solve)
    result = backend._run_jv_dispatch(
        stack,
        N_grid=30,
        n_points=2,
        v_rate=0.0,
        V_max=0.1,
        illuminated=True,
        solver="quasi_fermi",
        interface_boundary=True,
        interface_transport_model="fermi_dirac_richardson",
        interface_charge_jv_protocol=protocol,
    )

    assert np.array_equal(captured["guard_grid"], shared)
    assert np.array_equal(captured["solve_grid"], two_sided)
    assert captured["protocol"] is protocol
    assert result.interface_charge_evidence is evidence
    assert result.status_fwd[0].reason_code == "certified_interface_charge_qf"
    assert result.certified


def test_async_charged_jv_preflights_before_submit_and_binds_protocol(monkeypatch):
    stack = _charged_stack()
    registry = _CaptureRegistry()
    captured: dict[str, object] = {}
    monkeypatch.setattr(backend, "_JOB_REGISTRY", registry)
    monkeypatch.setattr(backend, "build_jv_stack", lambda *_args: stack)

    def fake_dispatch(_stack, **kwargs):
        captured.update(kwargs)
        metrics = jv_sweep.JVMetrics(0.08, 0.01, 0.5, 4.0e-7, True)
        return jv_sweep.JVResult(
            V_fwd=np.asarray([0.0, 0.1]),
            J_fwd=np.asarray([0.01, -0.01]),
            V_rev=np.asarray([0.0, 0.1]),
            J_rev=np.asarray([0.01, -0.01]),
            metrics_fwd=metrics,
            metrics_rev=metrics,
            hysteresis_index=0.0,
        )

    monkeypatch.setattr(backend, "_run_jv_dispatch", fake_dispatch)
    response = backend.start_job(
        backend.JobRequest(
            kind="jv",
            config_path="interface_charge_research.yaml",
            params=_charged_params(),
        )
    )
    assert response == {"status": "ok", "job_id": "charged-jv-job"}
    assert registry.fn is not None
    registry.fn(ProgressReporter())
    protocol = captured["interface_charge_jv_protocol"]
    assert isinstance(protocol, charged_jv.InterfaceChargeJVProtocol)
    assert protocol.voltages_V[-1] == pytest.approx(0.1)


@pytest.mark.parametrize(
    "bad_params",
    [
        {"v_rate": 0.01},
        {"solver": "steady_state"},
        {"interface_boundary": False},
        {"interface_transport_model": "scaps_thermionic"},
        {"unknown_control": 1},
    ],
)
def test_async_charged_jv_mismatch_is_422_before_submit(
    monkeypatch,
    bad_params,
):
    registry = _CaptureRegistry()
    monkeypatch.setattr(backend, "_JOB_REGISTRY", registry)
    monkeypatch.setattr(backend, "build_jv_stack", lambda *_args: _charged_stack())

    with pytest.raises(backend.HTTPException) as exc_info:
        backend.start_job(
            backend.JobRequest(
                kind="jv",
                config_path="interface_charge_research.yaml",
                params=_charged_params(**bad_params),
            )
        )

    assert exc_info.value.status_code == 422
    assert registry.fn is None


@pytest.mark.parametrize("kind", ["impedance", "tpv", "eqe", "degradation"])
def test_non_jv_job_routes_keep_charged_interface_closure_parked(kind):
    with pytest.raises(backend.HTTPException) as exc_info:
        backend.start_job(
            backend.JobRequest(
                kind=kind,
                config_path="interface_charge_research.yaml",
                params={},
            )
        )

    assert exc_info.value.status_code == 422
    assert "PARKED" in str(exc_info.value.detail)


@pytest.mark.slow
def test_real_sync_charged_jv_api_returns_certified_point_evidence():
    with TestClient(backend.app) as client:
        response = client.post(
            "/api/jv",
            json={
                "config_path": "interface_charge_jv_research.yaml",
                **{k: v for k, v in _charged_params().items() if k != "illuminated"},
            },
        )

    assert response.status_code == 200, response.text
    result = response.json()["result"]
    evidence = result["interface_charge_evidence"]
    assert result["V_fwd"] == result["V_rev"]
    assert result["J_fwd"] == result["J_rev"]
    assert result["hysteresis_index"] == 0.0
    assert result["metrics_fwd"]["voc_bracketed"] is True
    assert evidence["model"] == "interface-charge-jv-evidence-v1"
    assert evidence["dark_charge_off_bit_identity_verified"] is True
    assert len(evidence["points"]) == len(result["V_fwd"])
    assert all(point["certified"] for point in evidence["points"])


@pytest.mark.slow
def test_real_async_charged_jv_worker_preserves_protocol_and_evidence(monkeypatch):
    registry = _CaptureRegistry()
    monkeypatch.setattr(backend, "_JOB_REGISTRY", registry)
    response = backend.start_job(
        backend.JobRequest(
            kind="jv",
            config_path="interface_charge_jv_research.yaml",
            params=_charged_params(n_points=2),
        )
    )

    assert response == {"status": "ok", "job_id": "charged-jv-job"}
    assert registry.fn is not None
    result = registry.fn(ProgressReporter())
    evidence = result["interface_charge_evidence"]
    assert evidence["protocol"]["voltages_V"] == [0.0, 0.1]
    assert evidence["protocol_sha256"]
    assert evidence["continuation_bridge_count"] >= 0
    assert result["active_physics"].endswith(
        "equilibrium-referenced interface charge"
    )
