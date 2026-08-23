"""Registered energy-distributed charged bulk-trap refinement contract."""

from __future__ import annotations

import json
from pathlib import Path

from perovskite_sim.validation.bulk_trap_refinement import (
    _execution_protocol,
    run_bulk_trap_equilibrium_refinement,
)
from perovskite_sim.validation.numerical_certificate import (
    MatrixPoint,
    content_sha256,
    load_refinement_registry,
)


ROOT = Path(__file__).resolve().parents[3]
LANE_ID = "bulk-energy-distributed-trap-equilibrium-v1"


def _lane():
    return load_refinement_registry(
        ROOT / "reproducibility/numerical_refinement_registry.yaml",
        project_root=ROOT,
    ).lane(LANE_ID)


def _metrics(measurement, *, quality: bool = False):
    values = measurement.quality if quality else measurement.observables
    return {item.name: item for item in values}


def _protocol(lane):
    options = lane.options
    return _execution_protocol(
        lane,
        quadrature_orders=tuple(options["energy_quadrature_orders"]),
        base_poisson_tolerance=options["base_poisson_tolerance"],
        max_newton_iterations=options["max_newton_iterations"],
        max_potential_step_V=options["max_potential_step_V"],
        max_line_search_backtracks=options["max_line_search_backtracks"],
        probe_electron_density_m3=float(options["probe_electron_density_m3"]),
        probe_hole_density_m3=float(options["probe_hole_density_m3"]),
        left_layer_name=options["left_layer_name"],
        right_layer_name=options["right_layer_name"],
    )


def test_protocol_binds_energy_quadrature_charge_reference_and_full_matrix():
    lane = _lane()
    protocol = _protocol(lane)

    assert protocol["matrix"] == {
        "grid_parameter": "intervals_per_electrical_layer",
        "grid_values": [40, 80, 160],
        "tolerance_factors": [1.0, 0.1, 0.01],
        "tolerance_parameter": "poisson_residual_tolerance_factor",
    }
    assert protocol["energy_quadrature"] == {
        "coordinate": "truncated_normal_probability",
        "orders": [16, 32, 64],
        "rule": "gauss_legendre",
        "recombination_probe": {
            "electron_density_m3": 2.0e20,
            "hole_density_m3": 2.0e20,
        },
    }
    assert protocol["constitutive_closure"]["trap_charge"] == (
        "absolute_not_equilibrium_referenced"
    )
    assert protocol["topology"]["production_mol"] == "fail_closed"
    assert len(content_sha256(protocol)) == 64


def test_each_energy_protocol_control_changes_content_hash():
    lane = _lane()
    baseline = _protocol(lane)

    changed_orders = dict(baseline)
    changed_orders["energy_quadrature"] = dict(baseline["energy_quadrature"])
    changed_orders["energy_quadrature"]["orders"] = [8, 16, 32]
    assert content_sha256(changed_orders) != content_sha256(baseline)

    changed_probe = dict(baseline)
    changed_probe["energy_quadrature"] = dict(baseline["energy_quadrature"])
    changed_probe["energy_quadrature"]["recombination_probe"] = {
        "electron_density_m3": 3.0e20,
        "hole_density_m3": 2.0e20,
    }
    assert content_sha256(changed_probe) != content_sha256(baseline)


def test_registered_executor_real_cell_matches_exact_contract():
    lane = _lane()
    measurement = run_bulk_trap_equilibrium_refinement(
        lane,
        MatrixPoint(40, 1.0),
        ROOT,
    )
    observables = _metrics(measurement)
    quality = _metrics(measurement, quality=True)

    assert set(observables) == {gate.metric for gate in lane.observables}
    assert set(quality) == {gate.metric for gate in lane.quality_gates}
    assert all(
        observables[gate.metric].units == gate.units
        for gate in lane.observables
    )
    assert all(
        quality[gate.metric].units == gate.units
        for gate in lane.quality_gates
    )
    assert quality["bulk_trap_topology_verified"].values == (1.0,)
    assert quality["contact_thermodynamics_certified"].values == (1.0,)
    assert quality["default_production_path_rejected"].values == (1.0,)
    assert quality["max_energy_charge_relative_change"].values[0] < 0.005
    assert quality["max_energy_recombination_relative_change"].values[0] < 0.005
    assert quality["max_gauss_law_relative_error"].values[0] < 1.0e-7

    metadata = json.loads(measurement.metadata_json)
    assert metadata["protocol"]["schema_version"] == metadata["protocol_schema"]
    assert content_sha256(metadata["protocol"]) == metadata["protocol_hash"]
    assert metadata["actual"]["grid_nodes"] == 81
    assert metadata["actual"]["energy_quadrature_orders"] == [16, 32, 64]
    assert len(metadata["actual"]["probe_recombination_rates_m3_s"]) == 3
