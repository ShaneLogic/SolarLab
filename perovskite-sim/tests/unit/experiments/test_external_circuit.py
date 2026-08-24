from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from perovskite_sim.experiments.external_circuit import (
    ExternalCircuitError,
    ExternalCircuitProtocol,
    ExternalCircuitSourceError,
    ExternalCircuitTopologyError,
    apply_external_circuit,
    map_external_circuit_branch,
)
from perovskite_sim.experiments.jv_sweep import (
    JVPointStatus,
    JVResult,
    build_jv_experiment_protocol,
    compute_metrics,
)
from perovskite_sim.models.config_loader import load_device_from_yaml


def _status(branch: str, voltage: np.ndarray) -> tuple[JVPointStatus, ...]:
    return tuple(
        JVPointStatus(branch=branch, index=index, voltage=float(value))
        for index, value in enumerate(voltage)
    )


def _certified_result() -> JVResult:
    voltage_fwd = np.linspace(0.0, 1.2, 25)
    current_fwd = 20.0 * (1.0 - np.exp((voltage_fwd - 1.0) / 0.08))
    voltage_rev = voltage_fwd[::-1].copy()
    current_rev = (current_fwd + 0.2)[::-1].copy()
    metrics_fwd = compute_metrics(voltage_fwd, current_fwd)
    metrics_rev = compute_metrics(voltage_rev, current_rev)
    return JVResult(
        V_fwd=voltage_fwd,
        J_fwd=current_fwd,
        V_rev=voltage_rev,
        J_rev=current_rev,
        metrics_fwd=metrics_fwd,
        metrics_rev=metrics_rev,
        hysteresis_index=(
            (metrics_rev.PCE - metrics_fwd.PCE) / metrics_rev.PCE
        ),
        status_fwd=_status("jv_forward", voltage_fwd),
        status_rev=_status("jv_reverse", voltage_rev),
    )


def test_protocol_round_trip_hash_and_unknown_key_fail_closed():
    protocol = ExternalCircuitProtocol(
        series_resistance_ohm_m2=2.0e-4,
        shunt_resistance_ohm_m2=0.1,
    )

    restored = ExternalCircuitProtocol.from_dict(protocol.to_dict())
    assert restored == protocol
    assert restored.canonical_json() == protocol.canonical_json()
    assert restored.sha256 == protocol.sha256

    payload = protocol.to_dict()
    payload["claim"] = "validated"
    with pytest.raises(ExternalCircuitError, match="extra=.*claim"):
        ExternalCircuitProtocol.from_dict(payload)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"series_resistance_ohm_m2": -1.0}, "non-negative"),
        ({"series_resistance_ohm_m2": np.inf}, "finite"),
        ({"shunt_resistance_ohm_m2": 0.0}, "positive"),
        ({"shunt_resistance_ohm_m2": np.nan}, "finite"),
    ],
)
def test_protocol_rejects_unphysical_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ExternalCircuitProtocol(**kwargs)


def test_zero_coupling_is_value_bit_identical_and_read_only():
    voltage = np.array([0.0, 0.5, 1.0])
    current = np.array([20.0, 10.0, 0.0])
    branch = map_external_circuit_branch(
        voltage,
        current,
        ExternalCircuitProtocol(),
    )

    assert np.array_equal(branch.terminal_voltage_V, voltage)
    assert np.array_equal(branch.terminal_current_A_m2, current)
    assert np.array_equal(branch.shunt_current_A_m2, np.zeros_like(voltage))
    assert np.array_equal(branch.series_voltage_drop_V, np.zeros_like(voltage))
    assert branch.max_current_balance_error_A_m2 == 0.0
    assert branch.max_voltage_balance_error_V == 0.0
    with pytest.raises(ValueError, match="read-only"):
        branch.terminal_voltage_V[0] = 1.0


def test_series_and_junction_shunt_laws_are_exact():
    voltage = np.array([0.0, 0.4, 0.8])
    current = np.array([20.0, 12.0, 4.0])
    protocol = ExternalCircuitProtocol(
        series_resistance_ohm_m2=0.01,
        shunt_resistance_ohm_m2=0.2,
    )

    branch = map_external_circuit_branch(voltage, current, protocol)
    expected_shunt = voltage / 0.2
    expected_terminal_current = current - expected_shunt
    expected_terminal_voltage = voltage - 0.01 * expected_terminal_current

    assert np.array_equal(branch.shunt_current_A_m2, expected_shunt)
    assert np.array_equal(
        branch.terminal_current_A_m2,
        expected_terminal_current,
    )
    assert np.array_equal(branch.terminal_voltage_V, expected_terminal_voltage)
    assert np.array_equal(
        branch.terminal_power_W_m2,
        expected_terminal_voltage * expected_terminal_current,
    )


def test_reverse_branch_preserves_descending_orientation():
    branch = map_external_circuit_branch(
        [1.0, 0.5, 0.0],
        [0.0, 10.0, 20.0],
        ExternalCircuitProtocol(series_resistance_ohm_m2=1.0e-3),
    )
    assert branch.orientation == "descending"
    assert np.all(np.diff(branch.terminal_voltage_V) < 0.0)


def test_nonmonotonic_source_and_folded_terminal_curve_fail_closed():
    protocol = ExternalCircuitProtocol()
    with pytest.raises(ExternalCircuitTopologyError, match="strictly monotonic"):
        map_external_circuit_branch([0.0, 0.5, 0.4], [1.0, 0.5, 0.0], protocol)

    folded = ExternalCircuitProtocol(series_resistance_ohm_m2=1.0)
    with pytest.raises(ExternalCircuitTopologyError, match="strictly monotonic"):
        map_external_circuit_branch([0.0, 0.5, 1.0], [0.0, -1.0, 1.0], folded)


def test_apply_requires_certified_source_by_default():
    result = dataclasses.replace(_certified_result(), status_fwd=None)
    with pytest.raises(ExternalCircuitSourceError, match="certified"):
        apply_external_circuit(result, ExternalCircuitProtocol())


def test_apply_retains_source_hash_and_reports_certified_mapping():
    result = _certified_result()
    circuit = ExternalCircuitProtocol(
        series_resistance_ohm_m2=1.0e-3,
        shunt_resistance_ohm_m2=1.0,
    )

    mapped = apply_external_circuit(result, circuit)
    repeated = apply_external_circuit(result, circuit)

    assert mapped.source_certified
    assert mapped.mapping_certified
    assert mapped.certified
    assert mapped.circuit_protocol_sha256 == circuit.sha256
    assert mapped.source_result_sha256 == repeated.source_result_sha256
    assert len(mapped.source_result_sha256) == 64
    assert mapped.forward.orientation == "ascending"
    assert mapped.reverse.orientation == "descending"


def test_source_experiment_protocol_property_is_bound_into_result():
    source_protocol = build_jv_experiment_protocol(
        load_device_from_yaml("configs/nip_MAPbI3.yaml"),
        n_points=25,
        V_max=1.2,
        v_rate=1.0,
    )
    result = dataclasses.replace(_certified_result(), protocol=source_protocol)

    mapped = apply_external_circuit(result, ExternalCircuitProtocol())

    assert mapped.source_experiment_protocol_sha256 == source_protocol.sha256
    assert mapped.source_result_sha256 != apply_external_circuit(
        dataclasses.replace(result, protocol=None),
        ExternalCircuitProtocol(),
    ).source_result_sha256


def test_default_zero_coupling_preserves_source_metrics_exactly():
    result = _certified_result()
    mapped = apply_external_circuit(result, ExternalCircuitProtocol())

    assert mapped.metrics_fwd is result.metrics_fwd
    assert mapped.metrics_rev is result.metrics_rev
    assert mapped.hysteresis_index == result.hysteresis_index


def test_mapping_hash_binds_incident_power_without_changing_terminal_curve():
    result = _certified_result()
    first = apply_external_circuit(
        result,
        ExternalCircuitProtocol(series_resistance_ohm_m2=1.0e-3),
        incident_power_W_m2=1000.0,
    )
    second = apply_external_circuit(
        result,
        ExternalCircuitProtocol(series_resistance_ohm_m2=1.0e-3),
        incident_power_W_m2=500.0,
    )

    assert np.array_equal(
        first.forward.terminal_voltage_V,
        second.forward.terminal_voltage_V,
    )
    assert np.array_equal(
        first.forward.terminal_current_A_m2,
        second.forward.terminal_current_A_m2,
    )
    assert first.metrics_fwd.PCE * 2.0 == pytest.approx(second.metrics_fwd.PCE)
    assert first.mapping_sha256 != second.mapping_sha256


def test_shunt_reduces_voc_and_series_resistance_reduces_fill_factor():
    result = _certified_result()
    ideal = apply_external_circuit(result, ExternalCircuitProtocol())
    shunted = apply_external_circuit(
        result,
        ExternalCircuitProtocol(shunt_resistance_ohm_m2=0.1),
    )
    series = apply_external_circuit(
        result,
        ExternalCircuitProtocol(series_resistance_ohm_m2=5.0e-3),
    )

    assert shunted.metrics_fwd.V_oc < ideal.metrics_fwd.V_oc
    assert series.metrics_fwd.FF < ideal.metrics_fwd.FF
    assert series.metrics_fwd.PCE < ideal.metrics_fwd.PCE


def test_source_hash_changes_with_curve_or_source_protocol():
    result = _certified_result()
    base = apply_external_circuit(result, ExternalCircuitProtocol())
    changed_current = dataclasses.replace(
        result,
        J_fwd=np.asarray(result.J_fwd) + np.linspace(0.0, 1.0, len(result.J_fwd)),
    )
    changed = apply_external_circuit(changed_current, ExternalCircuitProtocol())
    assert changed.source_result_sha256 != base.source_result_sha256


def test_source_hash_binds_metrics_and_branch_evidence_rejects_tampering():
    result = _certified_result()
    mapped = apply_external_circuit(result, ExternalCircuitProtocol())
    altered_metrics = dataclasses.replace(
        result,
        metrics_fwd=dataclasses.replace(result.metrics_fwd, FF=0.1),
    )
    altered = apply_external_circuit(altered_metrics, ExternalCircuitProtocol())
    assert altered.source_result_sha256 != mapped.source_result_sha256

    altered_current = mapped.forward.terminal_current_A_m2 + 1.0
    with pytest.raises(ValueError, match="current_balance_error"):
        dataclasses.replace(
            mapped.forward,
            terminal_current_A_m2=altered_current,
            terminal_power_W_m2=(
                mapped.forward.terminal_voltage_V * altered_current
            ),
        )


def test_input_shape_and_incident_power_fail_closed():
    with pytest.raises(ExternalCircuitSourceError, match="same shape"):
        map_external_circuit_branch([0.0, 1.0], [1.0, 0.5, 0.0], ExternalCircuitProtocol())
    with pytest.raises(ValueError, match="positive"):
        apply_external_circuit(
            _certified_result(),
            ExternalCircuitProtocol(),
            incident_power_W_m2=0.0,
        )
