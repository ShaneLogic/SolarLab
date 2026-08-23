from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from perovskite_sim.validation.dae_interface_refinement import (
    _execution_protocol,
    run_algebraic_interface_dae_transient,
)
from perovskite_sim.validation.numerical_certificate import (
    MatrixPoint,
    content_sha256,
    load_refinement_registry,
)


ROOT = Path(__file__).resolve().parents[3]
LANE_ID = "algebraic-interface-state-dae-transient-v1"


def _lane():
    return load_refinement_registry(
        ROOT / "reproducibility/numerical_refinement_registry.yaml",
        project_root=ROOT,
    ).lane(LANE_ID)


def _metrics(measurement, *, quality: bool = False):
    values = measurement.quality if quality else measurement.observables
    return {item.name: item for item in values}


def test_protocol_is_lane_stable_and_binds_explicit_interface_exclusions():
    lane = _lane()
    options = lane.options
    protocol = _execution_protocol(
        lane,
        applied_voltage_V=float(options["applied_voltage_V"]),
        final_time_s=float(options["final_time_s"]),
        carrier_reference_time_s=float(options["carrier_reference_time_s"]),
        interface_velocity_m_s=float(options["interface_velocity_m_s"]),
        cross_transmission=float(options["cross_transmission"]),
        base_time_steps=options["base_time_steps"],
        reference_grid_intervals_per_layer=options[
            "reference_grid_intervals_per_layer"
        ],
        residual_tolerance=float(options["newton_residual_tolerance"]),
        max_newton_iterations=options["max_newton_iterations"],
        max_line_search_backtracks=options["max_line_search_backtracks"],
        max_log_density_update=float(options["max_log_density_update"]),
        max_interface_logit_update=float(
            options["max_interface_logit_update"]
        ),
        finite_difference_relative_step=float(
            options["finite_difference_relative_step"]
        ),
        mol_rtol=float(options["mol_rtol"]),
        mol_atol_m3=float(options["mol_atol_m3"]),
        mol_max_step_divisor=options["mol_max_step_divisor"],
    )

    assert protocol["matrix"] == {
        "grid_parameter": "intervals_per_electrical_layer",
        "grid_values": [4, 8, 16],
        "tolerance_factors": [1.0, 0.5, 0.25],
        "tolerance_parameter": "backward_euler_time_step_factor",
    }
    assert protocol["topology"]["interface_states"] == (
        "four_algebraic_fermi_richardson_states"
    )
    assert protocol["topology"]["interface_defect"] == "excluded"
    assert protocol["topology"]["cross_node_carrier_sampling"] == "excluded"
    assert protocol["topology"]["interface_charge"] == "off"
    assert protocol["interface_transport"]["clamp_contract"] == (
        "fail_closed_if_any_clamp_is_active"
    )
    assert protocol["backward_euler"]["jacobian_modes"] == [
        "dense_central",
        "structured_analytic",
    ]
    assert len(content_sha256(protocol)) == 64


def test_registered_executor_rejects_out_of_range_cross_transmission():
    lane = _lane()
    incompatible = replace(
        lane,
        options_json=json.dumps({**lane.options, "cross_transmission": 1.01}),
    )

    with pytest.raises(ValueError, match="cross_transmission"):
        run_algebraic_interface_dae_transient(
            incompatible,
            MatrixPoint(4, 1.0),
            ROOT,
        )


def test_registered_executor_real_cell_matches_exact_contract():
    lane = _lane()
    measurement = run_algebraic_interface_dae_transient(
        lane,
        MatrixPoint(4, 1.0),
        ROOT,
    )
    observables = _metrics(measurement)
    quality = _metrics(measurement, quality=True)

    assert set(observables) == {gate.metric for gate in lane.observables}
    assert set(quality) == {gate.metric for gate in lane.quality_gates}
    assert all(
        observables[gate.metric].units == gate.units for gate in lane.observables
    )
    assert all(
        quality[gate.metric].units == gate.units for gate in lane.quality_gates
    )
    assert observables["terminal_interface_occupation"].shape == (4,)
    assert quality["algebraic_interface_topology_verified"].values == (1.0,)
    assert quality["clamp_inactive_slice_verified"].values == (1.0,)
    assert quality["structured_analytic_success"].values == (1.0,)
    assert quality["mol_numerical_health_passed"].values == (1.0,)
    assert quality["structured_rhs_work_fraction"].values[0] < 0.1
    assert quality["max_terminal_log_density_error"].values[0] < 0.005
    assert (
        quality["max_terminal_interface_state_relative_error"].values[0]
        < 0.005
    )

    metadata = json.loads(measurement.metadata_json)
    assert metadata["protocol"]["schema_version"] == metadata["protocol_schema"]
    assert content_sha256(metadata["protocol"]) == metadata["protocol_hash"]
    assert metadata["actual"]["grid_nodes"] == 9
    assert metadata["actual"]["time_steps"] == 2
    assert metadata["actual"]["minimum_projection_occupation_margin"] > 0.0
    assert metadata["actual"]["minimum_cross_occupation_margin"] > 0.0
    assert metadata["actual"]["minimum_srh_occupancy_margin"] > 0.0
