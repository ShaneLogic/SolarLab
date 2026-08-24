from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from perovskite_sim.experiments import electrothermal
from perovskite_sim.experiments.electrothermal import (
    ElectrothermalCapabilityError,
    ElectrothermalConvergenceError,
    ElectrothermalError,
    ElectrothermalJVProtocol,
    ElectrothermalOperatingPointProtocol,
    ElectrothermalSourceError,
    solve_electrothermal_operating_point,
)
from perovskite_sim.experiments.external_circuit import ExternalCircuitProtocol
from perovskite_sim.experiments.jv_sweep import (
    JVPointStatus,
    JVResult,
    compute_metrics,
)
from perovskite_sim.experiments.thermal_balance import (
    LumpedThermalProtocol,
    ThermalEnergySourceError,
)
from perovskite_sim.models.config_loader import load_device_from_yaml


def _thermal(**overrides) -> LumpedThermalProtocol:
    values = {
        "absorbed_optical_power_W_m2": 800.0,
        "ambient_temperature_K": 300.0,
        "areal_heat_capacity_J_m2_K": 2000.0,
        "heat_transfer_coefficient_W_m2_K": 20.0,
        "emissivity": 0.0,
        "maximum_temperature_K": 400.0,
        "steady_power_residual_tolerance_W_m2": 1.0e-7,
    }
    values.update(overrides)
    return LumpedThermalProtocol(**values)


def _electrical(**overrides) -> ElectrothermalJVProtocol:
    values = {
        "grid_points_per_electrical_layer": 10,
        "voltage_points_per_branch": 3,
        "scan_rate_V_s": 20.0,
        "voltage_max_V": 1.0,
    }
    values.update(overrides)
    return ElectrothermalJVProtocol(**values)


def _status(branch: str, voltage: np.ndarray, *, valid: bool = True):
    return tuple(
        JVPointStatus(
            branch=branch,
            index=index,
            voltage=float(value),
            valid=valid,
        )
        for index, value in enumerate(voltage)
    )


def _temperature_jv(
    temperature_K: float,
    protocol,
    *,
    valid: bool = True,
    retain_protocol: bool = True,
) -> JVResult:
    maximum_power = 200.0 - 0.5 * (temperature_K - 300.0)
    voltage = np.asarray([0.0, 0.5, 1.0])
    current = np.asarray([2.0 * maximum_power, 2.0 * maximum_power, 0.0])
    reverse_voltage = voltage[::-1].copy()
    reverse_current = current[::-1].copy()
    metrics = compute_metrics(voltage, current)
    return JVResult(
        V_fwd=voltage,
        J_fwd=current,
        V_rev=reverse_voltage,
        J_rev=reverse_current,
        metrics_fwd=metrics,
        metrics_rev=metrics,
        hysteresis_index=0.0,
        status_fwd=_status("jv_forward", voltage, valid=valid),
        status_rev=_status("jv_reverse", reverse_voltage, valid=valid),
        protocol=protocol if retain_protocol else None,
    )


def _install_temperature_solver(monkeypatch, *, valid=True, retain_protocol=True):
    calls = []

    def fake_run(stack, **kwargs):
        calls.append((stack, kwargs))
        return _temperature_jv(
            stack.T,
            kwargs["experiment_protocol"],
            valid=valid,
            retain_protocol=retain_protocol,
        )

    monkeypatch.setattr(electrothermal, "run_jv_sweep", fake_run)
    return calls


def test_protocols_round_trip_hash_and_reject_unknown_keys():
    electrical = _electrical(atol_refinement_factor=0.1)
    operating = ElectrothermalOperatingPointProtocol(operating_branch="reverse")

    assert ElectrothermalJVProtocol.from_dict(electrical.to_dict()) == electrical
    assert (
        ElectrothermalOperatingPointProtocol.from_dict(operating.to_dict())
        == operating
    )
    assert ElectrothermalJVProtocol.from_dict(electrical.to_dict()).sha256 == (
        electrical.sha256
    )
    assert electrical.absolute_tolerance.refinement_factor == 0.1

    payload = operating.to_dict()
    payload["claim"] = "validated"
    with pytest.raises(ElectrothermalError, match="extra=.*claim"):
        ElectrothermalOperatingPointProtocol.from_dict(payload)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"grid_points_per_electrical_layer": 2}, "integer >= 3"),
        ({"voltage_points_per_branch": 2}, "integer >= 3"),
        ({"scan_rate_V_s": 0.0}, "positive"),
        ({"relative_tolerance": 1.0}, "less than 1"),
        ({"incident_power_W_m2": np.inf}, "finite"),
    ],
)
def test_electrical_protocol_rejects_invalid_controls(kwargs, message):
    with pytest.raises((TypeError, ValueError), match=message):
        _electrical(**kwargs)


def test_linear_temperature_dependent_mpp_has_analytic_coupled_root(monkeypatch):
    calls = _install_temperature_solver(monkeypatch)
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    result = solve_electrothermal_operating_point(
        stack,
        _thermal(),
        ExternalCircuitProtocol(),
        _electrical(),
        ElectrothermalOperatingPointProtocol(
            temperature_absolute_tolerance_K=1.0e-10,
        ),
    )

    expected_rise = 600.0 / 19.5
    assert result.certified
    assert result.operating_temperature_K == pytest.approx(
        300.0 + expected_rise,
        abs=1.0e-9,
    )
    assert result.terminal_power_W_m2 == pytest.approx(
        200.0 - 0.5 * expected_rise,
        abs=1.0e-9,
    )
    assert abs(result.power_balance_residual_W_m2) <= 1.0e-7
    assert result.electrical_evaluations == len(calls)
    assert result.electrical_evaluations == len(result.temperature_evaluations)
    assert all(call_stack is not stack for call_stack, _kwargs in calls)
    assert all(call_stack.T == evaluation.temperature_K for (call_stack, _), evaluation in zip(calls, result.temperature_evaluations))
    assert all(kwargs["protocol_mode"] == "research_strict" for _, kwargs in calls)
    assert all(kwargs["collect_numerical_diagnostics"] for _, kwargs in calls)
    assert stack.T == 300.0
    assert len(result.mapping_sha256) == 64


def test_series_loss_reduces_export_and_increases_temperature(monkeypatch):
    _install_temperature_solver(monkeypatch)
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    arguments = (
        stack,
        _thermal(),
    )
    electrical = _electrical()
    operating = ElectrothermalOperatingPointProtocol(
        temperature_absolute_tolerance_K=1.0e-10,
    )
    ideal = solve_electrothermal_operating_point(
        *arguments,
        ExternalCircuitProtocol(),
        electrical,
        operating,
    )
    lossy = solve_electrothermal_operating_point(
        *arguments,
        ExternalCircuitProtocol(series_resistance_ohm_m2=5.0e-4),
        electrical,
        operating,
    )

    assert lossy.terminal_power_W_m2 < ideal.terminal_power_W_m2
    assert lossy.operating_temperature_K > ideal.operating_temperature_K
    assert lossy.thermal_ledger.net_heating_W_m2 == pytest.approx(0.0, abs=1.0e-7)


def test_legacy_mode_is_rejected_before_an_electrical_solve(monkeypatch):
    calls = _install_temperature_solver(monkeypatch)
    stack = dataclasses.replace(
        load_device_from_yaml("configs/nip_MAPbI3.yaml"),
        mode="legacy",
    )

    with pytest.raises(ElectrothermalCapabilityError, match="temperature-scaling"):
        solve_electrothermal_operating_point(
            stack,
            _thermal(),
            ExternalCircuitProtocol(),
            _electrical(),
            ElectrothermalOperatingPointProtocol(),
        )
    assert calls == []


@pytest.mark.parametrize(
    "valid,retain_protocol,message",
    [
        (False, True, "uncertified"),
        (True, False, "different experiment protocol"),
    ],
)
def test_invalid_electrical_source_fails_closed(
    monkeypatch,
    valid,
    retain_protocol,
    message,
):
    _install_temperature_solver(
        monkeypatch,
        valid=valid,
        retain_protocol=retain_protocol,
    )
    with pytest.raises(ElectrothermalSourceError, match=message):
        solve_electrothermal_operating_point(
            load_device_from_yaml("configs/nip_MAPbI3.yaml"),
            _thermal(),
            ExternalCircuitProtocol(),
            _electrical(),
            ElectrothermalOperatingPointProtocol(),
        )


def test_impossible_temperature_envelope_fails_closed(monkeypatch):
    _install_temperature_solver(monkeypatch)
    with pytest.raises(ElectrothermalConvergenceError, match="no electrothermal root"):
        solve_electrothermal_operating_point(
            load_device_from_yaml("configs/nip_MAPbI3.yaml"),
            _thermal(
                heat_transfer_coefficient_W_m2_K=1.0,
                maximum_temperature_K=310.0,
            ),
            ExternalCircuitProtocol(),
            _electrical(),
            ElectrothermalOperatingPointProtocol(),
        )


def test_export_above_declared_absorption_fails_closed(monkeypatch):
    _install_temperature_solver(monkeypatch)
    with pytest.raises(ThermalEnergySourceError, match="exceeds"):
        solve_electrothermal_operating_point(
            load_device_from_yaml("configs/nip_MAPbI3.yaml"),
            _thermal(absorbed_optical_power_W_m2=100.0),
            ExternalCircuitProtocol(),
            _electrical(),
            ElectrothermalOperatingPointProtocol(),
        )


def test_result_recomputes_evaluation_and_certification_evidence(monkeypatch):
    _install_temperature_solver(monkeypatch)
    result = solve_electrothermal_operating_point(
        load_device_from_yaml("configs/nip_MAPbI3.yaml"),
        _thermal(),
        ExternalCircuitProtocol(),
        _electrical(),
        ElectrothermalOperatingPointProtocol(
            temperature_absolute_tolerance_K=1.0e-10,
        ),
    )

    altered = dataclasses.replace(
        result.temperature_evaluations[0],
        power_balance_residual_W_m2=(
            result.temperature_evaluations[0].power_balance_residual_W_m2 + 1.0
        ),
    )
    with pytest.raises(ValueError, match="residual"):
        dataclasses.replace(
            result,
            temperature_evaluations=(
                altered,
                *result.temperature_evaluations[1:],
            ),
        )
    with pytest.raises(ValueError, match="certified flag"):
        dataclasses.replace(result, certified=False)
