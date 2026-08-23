from __future__ import annotations

import json
from pathlib import Path

import pytest

from perovskite_sim.validation.dae_ion_refinement import (
    _execution_protocol,
    _time_step_count,
    run_single_ion_dae_transient,
)
from perovskite_sim.validation.numerical_certificate import (
    MatrixPoint,
    content_sha256,
    load_refinement_registry,
)


ROOT = Path(__file__).resolve().parents[3]
LANE_ID = "single-positive-ion-dae-transient-v1"


def _lane():
    return load_refinement_registry(
        ROOT / "reproducibility/numerical_refinement_registry.yaml",
        project_root=ROOT,
    ).lane(LANE_ID)


def _metrics(measurement, *, quality: bool = False):
    values = measurement.quality if quality else measurement.observables
    return {item.name: item for item in values}


def test_time_step_count_requires_exact_positive_integer():
    assert _time_step_count(
        2, 1.0, grid_intervals=8, reference_grid_intervals=8
    ) == 2
    assert _time_step_count(
        2, 0.5, grid_intervals=16, reference_grid_intervals=8
    ) == 16
    assert _time_step_count(
        2, 0.25, grid_intervals=32, reference_grid_intervals=8
    ) == 128
    with pytest.raises(ValueError, match="positive integer"):
        _time_step_count(
            2,
            0.3,
            grid_intervals=8,
            reference_grid_intervals=8,
        )


def test_protocol_is_lane_stable_and_binds_single_ion_topology():
    lane = _lane()
    options = lane.options
    protocol = _execution_protocol(
        lane,
        source_layer_index=options["source_layer_index"],
        source_layer_name=options["source_layer_name"],
        applied_voltage_V=options["applied_voltage_V"],
        final_time_s=options["final_time_s"],
        carrier_reference_time_s=options["carrier_reference_time_s"],
        ion_reference_time_s=options["ion_reference_time_s"],
        base_time_steps=options["base_time_steps"],
        reference_grid_intervals=options["reference_grid_intervals"],
        residual_tolerance=options["newton_residual_tolerance"],
        max_newton_iterations=options["max_newton_iterations"],
        max_line_search_backtracks=options["max_line_search_backtracks"],
        max_log_density_update=options["max_log_density_update"],
        max_ion_logit_update=options["max_ion_logit_update"],
        finite_difference_relative_step=options[
            "finite_difference_relative_step"
        ],
        mol_rtol=options["mol_rtol"],
        mol_atol_m3=options["mol_atol_m3"],
        mol_max_step_divisor=options["mol_max_step_divisor"],
    )

    assert protocol["matrix"] == {
        "grid_parameter": "single_layer_intervals",
        "grid_values": [8, 16, 32],
        "tolerance_factors": [1.0, 0.5, 0.25],
        "tolerance_parameter": "backward_euler_time_step_factor",
    }
    assert protocol["topology"]["mobile_ions"] == "single_positive"
    assert protocol["topology"]["ion_boundary"] == "blocking_zero_flux"
    assert protocol["topology"]["poisson_potential"] == "algebraic"
    assert len(content_sha256(protocol)) == 64


def test_registered_executor_real_cell_matches_exact_contract():
    lane = _lane()
    measurement = run_single_ion_dae_transient(
        lane,
        MatrixPoint(8, 1.0),
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
    assert quality["single_ion_topology_verified"].values == (1.0,)
    assert quality["site_occupancy_admissible"].values == (1.0,)
    assert quality["structured_analytic_success"].values == (1.0,)
    assert quality["minimum_positive_ion_relative_motion"].values[0] > 1.0e-6

    metadata = json.loads(measurement.metadata_json)
    assert metadata["protocol"]["schema_version"] == metadata["protocol_schema"]
    assert content_sha256(metadata["protocol"]) == metadata["protocol_hash"]
    assert metadata["actual"]["grid_nodes"] == 9
    assert metadata["actual"]["time_steps"] == 2
