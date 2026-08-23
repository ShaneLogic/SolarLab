from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from perovskite_sim.validation.dae_dual_ion_refinement import (
    _execution_protocol,
    run_dual_ion_dae_transient,
)
from perovskite_sim.validation.numerical_certificate import (
    MatrixPoint,
    content_sha256,
    load_refinement_registry,
)


ROOT = Path(__file__).resolve().parents[3]
LANE_ID = "dual-mobile-ion-dae-transient-v1"


def _lane():
    return load_refinement_registry(
        ROOT / "reproducibility/numerical_refinement_registry.yaml",
        project_root=ROOT,
    ).lane(LANE_ID)


def _metrics(measurement, *, quality: bool = False):
    values = measurement.quality if quality else measurement.observables
    return {item.name: item for item in values}


def test_protocol_is_lane_stable_and_binds_dual_shared_site_topology():
    lane = _lane()
    options = lane.options
    protocol = _execution_protocol(
        lane,
        source_layer_index=options["source_layer_index"],
        source_layer_name=options["source_layer_name"],
        applied_voltage_V=float(options["applied_voltage_V"]),
        final_time_s=float(options["final_time_s"]),
        carrier_reference_time_s=float(options["carrier_reference_time_s"]),
        ion_reference_time_s=float(options["ion_reference_time_s"]),
        negative_ion_diffusion_m2_s=float(options["negative_ion_diffusion_m2_s"]),
        negative_ion_density_m3=float(options["negative_ion_density_m3"]),
        negative_ion_site_limit_m3=float(options["negative_ion_site_limit_m3"]),
        ion_steric_diffusion_only=options["ion_steric_diffusion_only"],
        ion_steric_shared_site=options["ion_steric_shared_site"],
        base_time_steps=options["base_time_steps"],
        reference_grid_intervals=options["reference_grid_intervals"],
        residual_tolerance=float(options["newton_residual_tolerance"]),
        max_newton_iterations=options["max_newton_iterations"],
        max_line_search_backtracks=options["max_line_search_backtracks"],
        max_log_density_update=float(options["max_log_density_update"]),
        max_ion_coordinate_update=float(options["max_ion_coordinate_update"]),
        finite_difference_relative_step=float(
            options["finite_difference_relative_step"]
        ),
        mol_rtol=float(options["mol_rtol"]),
        mol_atol_m3=float(options["mol_atol_m3"]),
        mol_max_step_divisor=options["mol_max_step_divisor"],
    )

    assert protocol["matrix"] == {
        "grid_parameter": "single_layer_intervals",
        "grid_values": [8, 16, 32],
        "tolerance_factors": [1.0, 0.5, 0.25],
        "tolerance_parameter": "backward_euler_time_step_factor",
    }
    assert protocol["topology"]["mobile_ions"] == (
        "positive_and_negative_unit_charge"
    )
    assert protocol["topology"]["ion_coordinates"] == (
        "shared_site_three_state_softmax"
    )
    assert protocol["topology"]["ion_boundary"] == (
        "blocking_zero_flux_each_species"
    )
    assert protocol["derived_negative_ion"] == {
        "diffusion_m2_s": 3.2e-18,
        "equilibrium_density_m3": 1.6e25,
        "site_limit_m3": 1.6e27,
    }
    assert len(content_sha256(protocol)) == 64


@pytest.mark.parametrize(
    "option_name",
    ["ion_steric_diffusion_only", "ion_steric_shared_site"],
)
def test_registered_executor_rejects_nonshared_or_nondiffusion_steric_slice(
    option_name,
):
    lane = _lane()
    incompatible = replace(
        lane,
        options_json=json.dumps({**lane.options, option_name: False}),
    )

    with pytest.raises(ValueError, match="diffusion-only shared-site"):
        run_dual_ion_dae_transient(
            incompatible,
            MatrixPoint(8, 1.0),
            ROOT,
        )


def test_registered_executor_real_cell_matches_exact_contract():
    lane = _lane()
    measurement = run_dual_ion_dae_transient(
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
    assert quality["dual_ion_topology_verified"].values == (1.0,)
    assert quality["site_occupancy_admissible"].values == (1.0,)
    assert quality["structured_analytic_success"].values == (1.0,)
    assert quality["minimum_positive_ion_relative_motion"].values[0] > 1.0e-6
    assert quality["minimum_negative_ion_relative_motion"].values[0] > 1.0e-7
    assert quality["minimum_shared_site_vacancy_fraction"].values[0] > 0.95

    metadata = json.loads(measurement.metadata_json)
    assert metadata["protocol"]["schema_version"] == metadata["protocol_schema"]
    assert content_sha256(metadata["protocol"]) == metadata["protocol_hash"]
    assert metadata["actual"]["grid_nodes"] == 9
    assert metadata["actual"]["time_steps"] == 2
    assert metadata["actual"]["initial_positive_ion_inventory_m2"] > 0.0
    assert metadata["actual"]["initial_negative_ion_inventory_m2"] > 0.0
