from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from perovskite_sim.experiments import electrothermal
from perovskite_sim.experiments.jv_sweep import (
    JVPointStatus,
    JVResult,
    compute_metrics,
)
from perovskite_sim.validation import electrothermal_refinement as refinement
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
    ).lane("electrothermal-terminal-mpp-v1")


def _status(branch: str, voltage: np.ndarray):
    return tuple(
        JVPointStatus(branch=branch, index=index, voltage=float(value))
        for index, value in enumerate(voltage)
    )


def _source(stack, kwargs) -> JVResult:
    voltage = np.linspace(0.0, kwargs["V_max"], kwargs["n_points"])
    temperature_factor = 1.0 - 1.0e-3 * (float(stack.T) - 300.0)
    current = 400.0 * temperature_factor * (1.0 - voltage)
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
        status_fwd=_status("jv_forward", voltage),
        status_rev=_status("jv_reverse", reverse_voltage),
        protocol=kwargs["experiment_protocol"],
    )


def _metrics(measurement, *, quality=False):
    values = measurement.quality if quality else measurement.observables
    return {item.name: item for item in values}


def test_executor_matches_registry_and_binds_actual_matrix_controls(monkeypatch):
    calls = []

    def fake_run(stack, **kwargs):
        calls.append((stack, kwargs))
        return _source(stack, kwargs)

    monkeypatch.setattr(electrothermal, "run_jv_sweep", fake_run)
    lane = _lane()
    measurement = refinement.run_electrothermal_terminal_mpp_refinement(
        lane,
        MatrixPoint(15, 0.1),
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
    assert len(calls) >= 3
    assert all(kwargs["N_grid"] == 15 for _stack, kwargs in calls)
    assert all(
        kwargs["atol"].refinement_factor == pytest.approx(0.1)
        for _stack, kwargs in calls
    )
    assert all(kwargs["protocol_mode"] == "research_strict" for _, kwargs in calls)
    assert quality["electrothermal_certified"].values == (1.0,)
    assert quality["first_law_reconstruction_error_W_m2"].values == (0.0,)
    assert quality["temperature_protocols_aligned"].values == (1.0,)
    assert quality["temperature_response_active_W_m2"].values[0] > 1.0

    metadata = json.loads(measurement.metadata_json)
    assert metadata["protocol_schema"] == (
        "electrothermal-terminal-mpp-refinement-protocol-v1"
    )
    assert content_sha256(metadata["protocol"]) == metadata["protocol_hash"]
    actual = metadata["actual"]["electrical_protocol"]
    assert actual["grid_points_per_electrical_layer"] == 15
    assert actual["atol_refinement_factor"] == pytest.approx(0.1)
    assert len(metadata["actual"]["evaluation_temperatures_K"]) == len(calls)


def test_study_protocol_is_identical_across_matrix_cells(monkeypatch):
    monkeypatch.setattr(
        electrothermal,
        "run_jv_sweep",
        lambda stack, **kwargs: _source(stack, kwargs),
    )
    lane = _lane()
    coarse = refinement.run_electrothermal_terminal_mpp_refinement(
        lane,
        MatrixPoint(10, 1.0),
        ROOT,
    )
    fine = refinement.run_electrothermal_terminal_mpp_refinement(
        lane,
        MatrixPoint(20, 0.01),
        ROOT,
    )

    coarse_metadata = json.loads(coarse.metadata_json)
    fine_metadata = json.loads(fine.metadata_json)
    assert coarse_metadata["protocol_hash"] == fine_metadata["protocol_hash"]
    assert coarse_metadata["protocol"] == fine_metadata["protocol"]
    assert coarse_metadata["actual"]["electrical_protocol"] != (
        fine_metadata["actual"]["electrical_protocol"]
    )
