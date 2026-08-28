"""D3-E3 distributed-defect device refinement contract tests."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.validation.distributed_defect_refinement import (
    DISTRIBUTED_DEFECT_DEVICE_REFINEMENT_VERSION,
    _energy_orders,
    _execution_protocol,
    _safe_config_path,
    _source_species,
    run_distributed_defect_qf_dc_refinement,
)
from perovskite_sim.validation.numerical_certificate import (
    MatrixPoint,
    content_sha256,
    load_refinement_registry,
)


ROOT = Path(__file__).resolve().parents[3]
LANE_ID = "distributed-explicit-defect-qf-dc-v1"


def _lane():
    return load_refinement_registry(
        ROOT / "reproducibility/numerical_refinement_registry.yaml",
        project_root=ROOT,
    ).lane(LANE_ID)


def _protocol(lane):
    stack = load_device_from_yaml(_safe_config_path(lane, ROOT))
    species = _source_species(stack)
    options = lane.options
    return _execution_protocol(
        lane,
        species=species,
        energy_orders=_energy_orders(options),
        grid_alpha=options["grid_alpha"],
        profile_points=options["profile_points"],
        voltage_grid_V=tuple(options["voltage_grid_V"]),
        illumination_steps=tuple(options["illumination_steps"]),
        base_newton_residual_tolerance=(
            options["base_newton_residual_tolerance"]
        ),
        base_poisson_tolerance_V=options["base_poisson_tolerance_V"],
        base_finite_difference_step=options["base_finite_difference_step"],
        continuity_tolerance_A_m2=options["continuity_tolerance_A_m2"],
        current_spread_tolerance_A_m2=(
            options["current_spread_tolerance_A_m2"]
        ),
    )


def test_protocol_binds_independent_energy_space_and_tolerance_axes():
    lane = _lane()
    protocol = _protocol(lane)

    assert protocol["schema_version"] == (
        DISTRIBUTED_DEFECT_DEVICE_REFINEMENT_VERSION
    )
    assert protocol["matrix"] == {
        "energy_parameter": "defect_energy_quadrature_order",
        "energy_values": [16, 32, 64],
        "grid_parameter": "intervals_per_electrical_layer",
        "grid_values": [16, 32, 64],
        "tolerance_factors": [1.0, 0.1, 0.01],
        "tolerance_parameter": "qf_dc_residual_tolerance_factor",
    }
    assert protocol["constitutive_closure"]["energy_distributions"] == [
        "valence_band_tail",
        "uniform",
        "gaussian",
        "conduction_band_tail",
    ]
    assert protocol["topology"]["spatial_grading"] == "excluded"
    assert len(content_sha256(protocol)) == 64


def test_protocol_hash_changes_when_any_refinement_axis_changes():
    lane = _lane()
    baseline = _protocol(lane)

    changed_energy = json.loads(json.dumps(baseline))
    changed_energy["matrix"]["energy_values"] = [32, 64, 128]
    changed_grid = json.loads(json.dumps(baseline))
    changed_grid["matrix"]["grid_values"] = [32, 64, 128]
    changed_tolerance = json.loads(json.dumps(baseline))
    changed_tolerance["matrix"]["tolerance_factors"] = [1.0, 0.2, 0.04]

    hashes = {
        content_sha256(baseline),
        content_sha256(changed_energy),
        content_sha256(changed_grid),
        content_sha256(changed_tolerance),
    }
    assert len(hashes) == 4


@pytest.mark.parametrize(
    "orders",
    ([16, 32], [16, 48, 96], [16, 32, 32], [16.0, 32, 64]),
)
def test_energy_order_ladder_is_strict(orders):
    with pytest.raises(ValueError, match="energy_quadrature_orders"):
        _energy_orders({"energy_quadrature_orders": orders})


def test_config_hash_drift_fails_before_solver_execution(tmp_path):
    lane = _lane()
    source = ROOT / lane.config_path
    target = tmp_path / "drifted.yaml"
    target.write_bytes(source.read_bytes() + b"\n# drift\n")
    copied_lane = replace(lane, config_path=target.name)

    with pytest.raises(ValueError, match="config hash drift"):
        _safe_config_path(copied_lane, tmp_path)


def test_registered_executor_real_cell_matches_exact_contract():
    lane = _lane()
    measurement = run_distributed_defect_qf_dc_refinement(
        lane,
        MatrixPoint(16, 1.0),
        ROOT,
    )
    observables = {item.name: item for item in measurement.observables}
    quality = {item.name: item for item in measurement.quality}

    assert set(observables) == {gate.metric for gate in lane.observables}
    assert set(quality) == {gate.metric for gate in lane.quality_gates}
    assert all(
        observables[gate.metric].units == gate.units for gate in lane.observables
    )
    assert all(
        quality[gate.metric].units == gate.units for gate in lane.quality_gates
    )
    assert all(
        gate.passes(quality[gate.metric].values[0])
        for gate in lane.quality_gates
    )
    assert quality["energy_orders_completed"].values == (3.0,)
    assert quality["source_species_count"].values == (4.0,)
    assert quality["distributed_energy_metadata_verified"].values == (1.0,)
    assert quality["default_distributed_path_rejected"].values == (1.0,)

    metadata = json.loads(measurement.metadata_json)
    assert metadata["protocol_schema"] == (
        DISTRIBUTED_DEFECT_DEVICE_REFINEMENT_VERSION
    )
    assert content_sha256(metadata["protocol"]) == metadata["protocol_hash"]
    assert metadata["actual"]["energy_orders"] == [16, 32, 64]
    assert metadata["actual"]["grid_nodes"] == 33
    assert set(metadata["actual"]["model_identity_sha256"]) == {
        "16",
        "32",
        "64",
    }
    assert len(set(metadata["actual"]["model_identity_sha256"].values())) == 3


def test_registered_config_hash_is_literal_and_current():
    lane = _lane()
    path = ROOT / lane.config_path

    assert hashlib.sha256(path.read_bytes()).hexdigest() == lane.config_sha256
