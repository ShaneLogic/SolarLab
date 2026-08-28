"""D3-E4c spatial-defect device refinement contract tests."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.validation.distributed_defect_refinement import (
    _energy_orders,
)
from perovskite_sim.validation.numerical_certificate import (
    MatrixPoint,
    content_sha256,
    load_refinement_registry,
)
from perovskite_sim.validation.spatial_defect_refinement import (
    SPATIAL_DEFECT_DEVICE_REFINEMENT_VERSION,
    _execution_protocol,
    _safe_config_path,
    _source_species,
    run_spatial_defect_qf_dc_refinement,
)


ROOT = Path(__file__).resolve().parents[3]
LANE_ID = "spatially-graded-explicit-defect-qf-dc-v1"


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


def test_protocol_binds_energy_space_tolerance_and_spatial_contract():
    lane = _lane()
    protocol = _protocol(lane)

    assert protocol["schema_version"] == (
        SPATIAL_DEFECT_DEVICE_REFINEMENT_VERSION
    )
    assert protocol["matrix"] == {
        "energy_parameter": "defect_energy_quadrature_order",
        "energy_values": [16, 32, 64],
        "grid_parameter": "intervals_per_electrical_layer",
        "grid_values": [16, 32, 64],
        "tolerance_factors": [1.0, 0.1, 0.01],
        "tolerance_parameter": "qf_dc_residual_tolerance_factor",
    }
    assert protocol["spatial_closure"]["profile_presence"] == [
        True,
        False,
        True,
        True,
    ]
    assert sum(
        value is not None
        for value in protocol["spatial_closure"]["profile_sha256s"]
    ) == 3
    assert protocol["topology"]["device"] == (
        "continuous_band_graded_two_layer_pn_notch"
    )
    assert len(content_sha256(protocol)) == 64


def test_protocol_hash_changes_for_each_axis_and_spatial_profile():
    lane = _lane()
    baseline = _protocol(lane)

    changed_energy = json.loads(json.dumps(baseline))
    changed_energy["matrix"]["energy_values"] = [32, 64, 128]
    changed_grid = json.loads(json.dumps(baseline))
    changed_grid["matrix"]["grid_values"] = [32, 64, 128]
    changed_tolerance = json.loads(json.dumps(baseline))
    changed_tolerance["matrix"]["tolerance_factors"] = [1.0, 0.2, 0.04]
    changed_profile = json.loads(json.dumps(baseline))
    changed_profile["source"]["source_species"][0]["spatial_profile"][
        "knots"
    ][0]["density_multiplier"] = 0.7

    hashes = {
        content_sha256(baseline),
        content_sha256(changed_energy),
        content_sha256(changed_grid),
        content_sha256(changed_tolerance),
        content_sha256(changed_profile),
    }
    assert len(hashes) == 5


def test_registered_config_is_continuous_graded_v3_with_three_profiles():
    lane = _lane()
    stack = load_device_from_yaml(_safe_config_path(lane, ROOT))
    species = _source_species(stack)

    assert stack.band_grading
    assert stack.layers[0].params.Eg_back == stack.layers[1].params.Eg
    assert stack.layers[0].params.chi_back == stack.layers[1].params.chi
    assert tuple(item.name for item in species) == (
        "p_vb_tail_donor",
        "p_uniform_neutral",
        "n_gaussian_acceptor",
        "n_cb_tail_neutral",
    )
    assert sum(item.spatial_profile is not None for item in species) == 3
    for item in species:
        profile = item.spatial_profile
        if profile is None:
            continue
        positions = np.asarray(
            [knot.position_fraction for knot in profile.knots]
        )
        multipliers = np.asarray(
            [knot.density_multiplier for knot in profile.knots]
        )
        assert np.trapezoid(multipliers, positions) == pytest.approx(
            1.0,
            abs=1.0e-15,
        )


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
    measurement = run_spatial_defect_qf_dc_refinement(
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
    for name in (
        "all_states_certified",
        "contact_endpoints_verified",
        "contact_thermodynamics_certified",
        "default_spatial_path_rejected",
        "graded_energy_metadata_verified",
        "graded_model_hashes_verified",
        "graded_profiles_compiled_verified",
        "graded_topology_verified",
        "occupancy_bounded_without_clipping",
        "terminal_densities_positive",
    ):
        assert quality[name].values == (1.0,)
    assert quality["energy_orders_completed"].values == (3.0,)
    assert quality["profiled_species_count"].values == (3.0,)
    assert quality["source_species_count"].values == (4.0,)
    assert quality["voltage_points_completed"].values == (3.0,)

    metadata = json.loads(measurement.metadata_json)
    assert metadata["protocol_schema"] == (
        SPATIAL_DEFECT_DEVICE_REFINEMENT_VERSION
    )
    assert content_sha256(metadata["protocol"]) == metadata["protocol_hash"]
    assert metadata["actual"]["energy_orders"] == [16, 32, 64]
    assert metadata["actual"]["grid_nodes"] == 33
    assert len(
        [
            value
            for value in metadata["actual"]["profile_sha256s"]
            if value is not None
        ]
    ) == 3
    assert len(set(metadata["actual"]["model_identity_sha256"].values())) == 3


def test_registered_config_hash_is_literal_and_current():
    lane = _lane()
    path = ROOT / lane.config_path

    assert hashlib.sha256(path.read_bytes()).hexdigest() == lane.config_sha256
