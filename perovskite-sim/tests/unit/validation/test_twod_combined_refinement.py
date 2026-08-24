from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from perovskite_sim.twod.microstructure import GrainBoundary, Microstructure
from perovskite_sim.validation import refinement_executors as executors
from perovskite_sim.validation.numerical_certificate import (
    MatrixPoint,
    content_sha256,
    load_refinement_registry,
)


ROOT = Path(__file__).resolve().parents[3]


class _Protocol:
    voltage_values_V = (0.0, 0.05, 0.1)
    implicit_legacy_protocol = False
    protocol_hash = "a" * 64

    def to_dict(self):
        return {
            "implicit_legacy_protocol": False,
            "schema_version": "jv-2d-execution-protocol-v1",
            "voltage_values_V": list(self.voltage_values_V),
        }


def _registered_lane():
    return load_refinement_registry(
        ROOT / "reproducibility/numerical_refinement_registry.yaml",
        project_root=ROOT,
    ).lane("twod-mobile-ion-interface-srh-v1")


def _install_problem(
    monkeypatch,
    *,
    clamp_active: bool = False,
    malformed_clamp: bool = False,
    missing: bool = False,
):
    microstructure = Microstructure(
        (
            GrainBoundary(
                x_position=0.5,
                width=0.2,
                tau_n=1.0e-9,
                tau_p=1.0e-9,
            ),
        )
    )
    stack = SimpleNamespace(microstructure=microstructure)
    layers = (SimpleNamespace(thickness=0.5), SimpleNamespace(thickness=0.5))
    grid = SimpleNamespace(
        x=np.array([0.0, 0.5, 1.0]),
        y=np.linspace(0.0, 1.0, 5),
    )
    grid.Nx = grid.x.size
    grid.Ny = grid.y.size
    shape = (grid.Ny, grid.Nx)
    active = np.zeros(shape, dtype=bool)
    active[:3, :] = True
    region = SimpleNamespace(
        x_overlap_fraction=np.array([0.0, 0.4, 0.0]),
        physical_width=0.2,
    )
    material = SimpleNamespace(
        grid=grid,
        P_lim_2d=np.full(shape, 100.0),
        D_ion_2d=active.astype(float),
        grain_boundary_regions=(region,),
        has_mobile_ions=True,
        interface_srh_couplings=(object(),),
    )
    protocol = _Protocol()
    captured = {}

    monkeypatch.setattr(executors, "_load_stack", lambda *_args: stack)
    monkeypatch.setattr(executors, "electrical_layers", lambda *_args: layers)
    monkeypatch.setattr(executors, "build_grid_2d", lambda *_args, **_kwargs: grid)
    monkeypatch.setattr(
        executors,
        "build_material_arrays_2d",
        lambda *_args, **_kwargs: material,
    )

    def build_protocol(*_args, **kwargs):
        captured["build"] = kwargs
        return protocol

    monkeypatch.setattr(executors, "build_jv_2d_execution_protocol", build_protocol)

    snapshots = []
    for index in range(3):
        n = np.ones(shape)
        p = np.ones(shape)
        n[:, 1] += 2.0e-3
        p[:, 1] += 1.0e-3
        ions = np.zeros(shape)
        ions[active] = 1.0 + index * 1.0e-3
        snapshots.append(SimpleNamespace(n=n, p=p, P_ion=ions))
    currents = tuple(
        SimpleNamespace(
            terminal_electron_A_m2=0.04,
            terminal_hole_A_m2=0.05,
            terminal_positive_ion_A_m2=0.001,
            terminal_displacement_A_m2=target - 0.091,
            terminal_total_A_m2=target,
            max_face_spread_A_m2=1.0e-13,
        )
        for target in (0.1, 0.2, 0.3)
    )
    ions = tuple(
        SimpleNamespace(
            terminal_min_electron_density_m3=1.0,
            terminal_min_hole_density_m3=1.0,
            relative_inventory_drift=1.0e-12,
            passed=True,
        )
        for _ in range(3)
    )
    interfaces = tuple(
        SimpleNamespace(
            total_surface_rate_m2_s=np.full((1, grid.Nx), 2.0),
            pair_a_clamped=(
                np.array([], dtype=bool)
                if malformed_clamp
                else np.array([[clamp_active, False, False]])
            ),
            pair_b_clamped=np.zeros((1, grid.Nx), dtype=bool),
        )
        for _ in range(3)
    )
    result = SimpleNamespace(
        V=np.array(protocol.voltage_values_V),
        J=np.array([0.1, 0.2, 0.3]),
        snapshots=tuple(snapshots),
        current_components=(() if missing else currents),
        ion_diagnostics=ions,
        interface_srh_diagnostics=interfaces,
        protocol=protocol,
        grid_x=grid.x.copy(),
        grid_y=grid.y.copy(),
    )

    def run(*_args, **kwargs):
        captured["run"] = kwargs
        return result

    monkeypatch.setattr(executors, "run_jv_sweep_2d", run)
    return captured


def _metrics(measurement, *, quality=False):
    values = measurement.quality if quality else measurement.observables
    return {item.name: item for item in values}


def test_combined_twod_executor_matches_registry_and_explicit_protocol(monkeypatch):
    captured = _install_problem(monkeypatch)
    lane = _registered_lane()

    measurement = executors.run_twod_mobile_ion_interface_srh(
        lane,
        MatrixPoint(6, 0.1),
        ROOT,
    )

    observables = _metrics(measurement)
    quality = _metrics(measurement, quality=True)
    assert set(observables) == {gate.metric for gate in lane.observables}
    assert set(quality) == {gate.metric for gate in lane.quality_gates}
    assert captured["build"]["Nx"] == 6
    assert captured["build"]["Ny_per_layer"] == 6
    assert captured["build"]["atol"].refinement_factor == pytest.approx(0.1)
    assert captured["run"]["protocol_mode"] == "research_strict"
    assert quality["clamp_inactive_slice_verified"].values == (1.0,)
    assert quality["minimum_mobile_ion_relative_redistribution"].values[0] > 0.0
    metadata = json.loads(measurement.metadata_json)
    assert metadata["execution_protocol_hash"] == "a" * 64
    assert metadata["protocol"]["schema_version"] == metadata["protocol_schema"]
    assert content_sha256(metadata["protocol"]) == metadata["protocol_hash"]


def test_combined_twod_study_protocol_is_matrix_stable():
    lane = _registered_lane()

    first = executors._twod_mobile_interface_refinement_protocol(lane)
    second = executors._twod_mobile_interface_refinement_protocol(lane)

    assert content_sha256(first) == content_sha256(second)
    assert first["matrix"] == {
        "grid_parameter": "matched_xy_intervals_per_layer",
        "grid_values": [4, 6, 8],
        "tolerance_factors": [1.0, 0.1, 0.01],
        "tolerance_parameter": "componentwise_atol_refinement_factor",
    }


def test_combined_twod_executor_fails_closed_on_missing_point_evidence(monkeypatch):
    _install_problem(monkeypatch, missing=True)

    with pytest.raises(RuntimeError, match="evidence count mismatch"):
        executors.run_twod_mobile_ion_interface_srh(
            _registered_lane(),
            MatrixPoint(6, 0.1),
            ROOT,
        )


def test_combined_twod_clamp_activity_is_a_failing_quality_value(monkeypatch):
    _install_problem(monkeypatch, clamp_active=True)

    measurement = executors.run_twod_mobile_ion_interface_srh(
        _registered_lane(),
        MatrixPoint(6, 0.1),
        ROOT,
    )

    assert _metrics(measurement, quality=True)[
        "clamp_inactive_slice_verified"
    ].values == (0.0,)


def test_combined_twod_executor_rejects_malformed_clamp_evidence(monkeypatch):
    _install_problem(monkeypatch, malformed_clamp=True)

    with pytest.raises(RuntimeError, match="malformed clamp masks"):
        executors.run_twod_mobile_ion_interface_srh(
            _registered_lane(),
            MatrixPoint(6, 0.1),
            ROOT,
        )
