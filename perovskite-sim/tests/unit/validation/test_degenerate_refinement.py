"""Registered high-doping Fermi-Dirac equilibrium refinement contract."""

from __future__ import annotations

import json
from pathlib import Path

from perovskite_sim.validation.degenerate_refinement import (
    _execution_protocol,
    run_degenerate_pn_equilibrium_refinement,
)
from perovskite_sim.validation.numerical_certificate import (
    MatrixPoint,
    content_sha256,
    load_refinement_registry,
)


ROOT = Path(__file__).resolve().parents[3]
LANE_ID = "degenerate-pn-equilibrium-v1"


def _lane():
    return load_refinement_registry(
        ROOT / "reproducibility/numerical_refinement_registry.yaml",
        project_root=ROOT,
    ).lane(LANE_ID)


def _metrics(measurement, *, quality: bool = False):
    values = measurement.quality if quality else measurement.observables
    return {item.name: item for item in values}


def test_protocol_is_lane_stable_and_binds_full_matrix():
    lane = _lane()
    options = lane.options
    protocol = _execution_protocol(
        lane,
        base_poisson_tolerance=options["base_poisson_tolerance"],
        max_newton_iterations=options["max_newton_iterations"],
        max_potential_step_V=options["max_potential_step_V"],
        max_line_search_backtracks=options["max_line_search_backtracks"],
        left_layer_name=options["left_layer_name"],
        right_layer_name=options["right_layer_name"],
    )

    assert protocol["matrix"] == {
        "grid_parameter": "intervals_per_electrical_layer",
        "grid_values": [40, 80, 160],
        "tolerance_factors": [1.0, 0.1, 0.01],
        "tolerance_parameter": "poisson_residual_tolerance_factor",
    }
    assert protocol["topology"]["bulk_recombination"] == "disabled"
    assert protocol["carrier_statistics"]["ionization"] == "fully_ionized"
    assert len(content_sha256(protocol)) == 64


def test_registered_executor_real_cell_matches_exact_contract():
    lane = _lane()
    measurement = run_degenerate_pn_equilibrium_refinement(
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
    assert quality["contact_thermodynamics_certified"].values == (1.0,)
    assert quality["fd_recombination_off_topology_verified"].values == (1.0,)
    assert quality["max_relative_face_current"].values[0] < 1.0e-12

    metadata = json.loads(measurement.metadata_json)
    assert metadata["protocol"]["schema_version"] == metadata["protocol_schema"]
    assert content_sha256(metadata["protocol"]) == metadata["protocol_hash"]
    assert metadata["actual"]["grid_nodes"] == 81
