"""Registered composition-graded CIGS optical refinement contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from perovskite_sim.validation.cigs_optics_refinement import (
    _execution_protocol,
    _kk_quadrature_order,
    run_cigs_graded_optics_refinement,
)
from perovskite_sim.validation.numerical_certificate import (
    MatrixPoint,
    content_sha256,
    load_refinement_registry,
)


ROOT = Path(__file__).resolve().parents[3]
LANE_ID = "cigs-graded-optics-v1"


def _lane():
    return load_refinement_registry(
        ROOT / "reproducibility/numerical_refinement_registry.yaml",
        project_root=ROOT,
    ).lane(LANE_ID)


def _protocol(lane):
    options = lane.options
    return _execution_protocol(
        lane,
        absorber_layer_name=options["absorber_layer_name"],
        fixed_electrical_intervals_per_layer=(
            options["fixed_electrical_intervals_per_layer"]
        ),
        wavelength_points=options["wavelength_points"],
        wavelength_min_nm=options["wavelength_min_nm"],
        wavelength_max_nm=options["wavelength_max_nm"],
        base_kk_quadrature_order=options["base_kk_quadrature_order"],
        carron_energy_points_per_composition=(
            options["carron_energy_points_per_composition"]
        ),
        carron_minimum_excess_above_gap_eV=(
            options["carron_minimum_excess_above_gap_eV"]
        ),
    )


def _metrics(measurement, *, quality: bool = False):
    values = measurement.quality if quality else measurement.observables
    return {item.name: item for item in values}


def test_protocol_binds_both_refinement_axes_and_external_sources():
    lane = _lane()
    protocol = _protocol(lane)

    assert protocol["matrix"] == {
        "grid_parameter": "cigs_optical_slices",
        "grid_values": [8, 16, 32],
        "tolerance_factors": [1.0, 0.5, 0.25],
        "tolerance_parameter": "inverse_kk_quadrature_order_factor",
    }
    assert protocol["source"]["minoura_doi"] == "10.1063/1.4921300"
    assert protocol["source"]["carron_doi"] == (
        "10.1080/14686996.2018.1458579"
    )
    assert protocol["constitutive_closure"]["composition_coordinate"] == (
        "shared_with_electrical_Eg_chi_grade"
    )
    assert len(content_sha256(protocol)) == 64


def test_each_numerical_control_changes_protocol_hash():
    lane = _lane()
    baseline = _protocol(lane)

    changed = dict(baseline)
    changed["numerical_resolution"] = dict(baseline["numerical_resolution"])
    changed["numerical_resolution"]["base_kk_quadrature_order"] = 128
    assert content_sha256(changed) != content_sha256(baseline)

    changed_benchmark = dict(baseline)
    changed_benchmark["independent_benchmark"] = dict(
        baseline["independent_benchmark"]
    )
    changed_benchmark["independent_benchmark"]["minimum_excess_eV"] = 0.2
    assert content_sha256(changed_benchmark) != content_sha256(baseline)


def test_inverse_tolerance_factor_maps_to_exact_kk_orders():
    assert [_kk_quadrature_order(96, factor) for factor in (1.0, 0.5, 0.25)] == [
        96,
        192,
        384,
    ]
    with pytest.raises(ValueError, match="must be an integer"):
        _kk_quadrature_order(96, 0.7)


def test_registered_executor_real_cell_matches_exact_contract():
    lane = _lane()
    measurement = run_cigs_graded_optics_refinement(
        lane,
        MatrixPoint(8, 1.0),
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
    assert quality["causal_nk_verified"].values == (1.0,)
    assert quality["cigs_optical_topology_verified"].values == (1.0,)
    assert quality["default_gate_off_inert"].values == (1.0,)
    assert quality["independent_carron_energy_points_completed"].values == (
        453.0,
    )
    assert quality["max_photon_budget_excess_fraction"].values == (0.0,)
    assert quality["max_uniform_composition_reflectance_difference"].values[0] < (
        1.0e-12
    )

    metadata = json.loads(measurement.metadata_json)
    assert metadata["protocol"]["schema_version"] == metadata["protocol_schema"]
    assert content_sha256(metadata["protocol"]) == metadata["protocol_hash"]
    assert metadata["actual"]["optical_slices"] == 8
    assert metadata["actual"]["kk_quadrature_order"] == 96
    assert metadata["actual"]["electrical_grid_nodes"] == 97
