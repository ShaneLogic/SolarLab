from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from perovskite_sim.experiments.jv_sweep import (
    JVPointStatus,
    JVResult,
    compute_metrics,
)
from perovskite_sim.validation import external_circuit_refinement as refinement
from perovskite_sim.validation.numerical_certificate import (
    MatrixPoint,
    content_sha256,
    load_refinement_registry,
)


ROOT = Path(__file__).resolve().parents[3]


def _lane():
    return load_refinement_registry(
        ROOT / "reproducibility/numerical_refinement_registry.yaml",
        project_root=ROOT,
    ).lane("external-series-shunt-dc-v1")


def _statuses(branch: str, voltage: np.ndarray) -> tuple[JVPointStatus, ...]:
    return tuple(
        JVPointStatus(branch=branch, index=index, voltage=float(value))
        for index, value in enumerate(voltage)
    )


def _source_result(protocol) -> JVResult:
    voltage = np.linspace(0.0, 1.2, 12)
    current = 210.0 * (1.0 - np.exp((voltage - 1.03) / 0.075))
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
        status_fwd=_statuses("jv_forward", voltage),
        status_rev=_statuses("jv_reverse", reverse_voltage),
        protocol=protocol,
    )


def _metrics(measurement, *, quality: bool = False):
    values = measurement.quality if quality else measurement.observables
    return {item.name: item for item in values}


def test_executor_matches_registry_and_binds_explicit_protocol(monkeypatch):
    lane = _lane()
    captured = {}

    def run(*_args, **kwargs):
        captured.update(kwargs)
        return _source_result(kwargs["experiment_protocol"])

    monkeypatch.setattr(refinement, "run_jv_sweep", run)
    measurement = refinement.run_external_series_shunt_dc_refinement(
        lane,
        MatrixPoint(30, 0.1),
        ROOT,
    )

    observables = _metrics(measurement)
    quality = _metrics(measurement, quality=True)
    assert set(observables) == {gate.metric for gate in lane.observables}
    assert set(quality) == {gate.metric for gate in lane.quality_gates}
    assert all(
        observables[gate.metric].units == gate.units for gate in lane.observables
    )
    assert all(quality[gate.metric].units == gate.units for gate in lane.quality_gates)
    assert captured["N_grid"] == 30
    assert captured["atol"].refinement_factor == pytest.approx(0.1)
    assert captured["certification_mode"] == "strict"
    assert captured["protocol_mode"] == "research_strict"
    assert captured["collect_numerical_diagnostics"] is True
    assert quality["intrinsic_jv_certified"].values == (1.0,)
    assert quality["external_circuit_certified"].values == (1.0,)
    assert quality["zero_coupling_exact"].values == (1.0,)
    assert quality["min_pce_loss_fraction"].values[0] > 0.01
    assert quality["max_current_balance_error_A_m2"].values == (0.0,)
    assert quality["max_voltage_balance_error_V"].values == (0.0,)

    metadata = json.loads(measurement.metadata_json)
    assert metadata["protocol"]["schema_version"] == metadata["protocol_schema"]
    assert content_sha256(metadata["protocol"]) == metadata["protocol_hash"]
    assert metadata["protocol"]["circuit"]["protocol"] == {
        "application": "dc_postprocess",
        "current_convention": "photovoltaic_output_positive",
        "schema_version": 1,
        "series_resistance_ohm_m2": 2.0e-4,
        "shunt_resistance_ohm_m2": 0.2,
        "shunt_voltage_reference": "junction",
    }


def test_study_protocol_is_identical_across_matrix_cells(monkeypatch):
    lane = _lane()

    def run(*_args, **kwargs):
        return _source_result(kwargs["experiment_protocol"])

    monkeypatch.setattr(refinement, "run_jv_sweep", run)
    first = refinement.run_external_series_shunt_dc_refinement(
        lane,
        MatrixPoint(20, 1.0),
        ROOT,
    )
    second = refinement.run_external_series_shunt_dc_refinement(
        lane,
        MatrixPoint(40, 0.01),
        ROOT,
    )

    first_metadata = json.loads(first.metadata_json)
    second_metadata = json.loads(second.metadata_json)
    assert first_metadata["protocol_hash"] == second_metadata["protocol_hash"]
    assert first_metadata["protocol"] == second_metadata["protocol"]
    assert first_metadata["actual"] != second_metadata["actual"]


def test_executor_fails_closed_on_source_protocol_mismatch(monkeypatch):
    lane = _lane()
    stack = refinement.load_device_from_yaml(ROOT / lane.config_path)
    mismatch = refinement.build_jv_experiment_protocol(
        stack,
        v_rate=20.0,
        n_points=12,
        V_max=1.1,
        illuminated=True,
        implicit_legacy_protocol=False,
    )
    monkeypatch.setattr(
        refinement,
        "run_jv_sweep",
        lambda *_args, **_kwargs: _source_result(mismatch),
    )

    with pytest.raises(RuntimeError, match="different experiment protocol"):
        refinement.run_external_series_shunt_dc_refinement(
            lane,
            MatrixPoint(20, 1.0),
            ROOT,
        )
