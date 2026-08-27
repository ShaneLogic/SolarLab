"""DEF-4 charged explicit-defect numerical-refinement contract tests."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from perovskite_sim.validation.charged_defect_refinement import (
    _execution_protocol,
    _load_suite,
    run_charged_defect_qf_dc_refinement,
)
from perovskite_sim.validation.numerical_certificate import (
    MatrixPoint,
    content_sha256,
    load_refinement_registry,
)


ROOT = Path(__file__).resolve().parents[3]
LANE_ID = "charged-explicit-defect-qf-dc-v1"


def _lane():
    return load_refinement_registry(
        ROOT / "reproducibility/numerical_refinement_registry.yaml",
        project_root=ROOT,
    ).lane(LANE_ID)


def _metrics(measurement, *, quality: bool = False):
    values = measurement.quality if quality else measurement.observables
    return {item.name: item for item in values}


def _protocol(lane):
    suite, scenarios = _load_suite(lane, ROOT)
    options = lane.options
    return _execution_protocol(
        lane,
        suite=suite,
        scenarios=scenarios,
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


def test_protocol_binds_s0_s2_physics_and_full_matrix():
    lane = _lane()
    protocol = _protocol(lane)

    assert protocol["matrix"] == {
        "grid_parameter": "intervals_per_electrical_layer",
        "grid_values": [32, 64, 128],
        "tolerance_factors": [1.0, 0.1, 0.01],
        "tolerance_parameter": "qf_dc_residual_tolerance_factor",
    }
    assert [item["id"] for item in protocol["scenarios"]] == ["S0", "S1", "S2"]
    assert [item["charge_transition"] for item in protocol["scenarios"]] == [
        "neutral",
        "acceptor",
        "donor",
    ]
    assert protocol["constitutive_closure"] == {
        "carrier_statistics": "maxwell_boltzmann",
        "charge_states": ["neutral", "acceptor", "donor"],
        "dopant_ionization": "fully_ionized",
        "occupancy": "local_quasi_steady_single_level",
        "occupancy_clipping": "none",
        "poisson_tangent": "analytic_fixed_qf",
        "recombination": "exact_per_species_srh",
    }
    assert protocol["source"]["external_reference_contract"][
        "independent_export_attestation_required"
    ] is True
    assert len(content_sha256(protocol)) == 64


def test_each_solver_protocol_control_changes_content_hash():
    baseline = _protocol(_lane())

    changed_grid = json.loads(json.dumps(baseline))
    changed_grid["solver"]["grid_alpha"] = 3.0
    assert content_sha256(changed_grid) != content_sha256(baseline)

    changed_voltage = json.loads(json.dumps(baseline))
    changed_voltage["operating_points"]["derived_pn"]["voltage_grid_V"][-1] = 0.35
    assert content_sha256(changed_voltage) != content_sha256(baseline)

    changed_illumination = json.loads(json.dumps(baseline))
    changed_illumination["operating_points"]["derived_pn"][
        "illumination_steps"
    ][1] = 1.0e-5
    assert content_sha256(changed_illumination) != content_sha256(baseline)


def test_suite_fails_closed_when_a_frozen_scenario_config_drifts(tmp_path):
    lane = _lane()
    suite_source = ROOT / lane.options["suite_manifest"]
    suite = json.loads(suite_source.read_text(encoding="utf-8"))
    for scenario in suite["scenarios"]:
        source = ROOT / scenario["config_path"]
        destination = tmp_path / scenario["config_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(json.dumps(suite), encoding="utf-8")
    options = dict(lane.options)
    options["suite_manifest"] = "suite.json"
    options["suite_manifest_sha256"] = hashlib.sha256(
        suite_path.read_bytes()
    ).hexdigest()
    copied_lane = replace(
        lane,
        options_json=json.dumps(options),
    )

    drifting = tmp_path / suite["scenarios"][1]["config_path"]
    drifting.write_text(drifting.read_text() + "\n# drift\n", encoding="utf-8")

    with pytest.raises(ValueError, match="config hash drift"):
        _load_suite(copied_lane, tmp_path)


def test_suite_manifest_hash_drift_fails_closed(tmp_path):
    lane = _lane()
    suite_source = ROOT / lane.options["suite_manifest"]
    suite_path = tmp_path / "suite.json"
    suite_path.write_bytes(suite_source.read_bytes() + b"\n")
    options = dict(lane.options)
    options["suite_manifest"] = "suite.json"
    copied_lane = replace(lane, options_json=json.dumps(options))

    with pytest.raises(ValueError, match="suite manifest hash drift"):
        _load_suite(copied_lane, tmp_path)


def test_registered_executor_real_cell_matches_exact_contract():
    lane = _lane()
    measurement = run_charged_defect_qf_dc_refinement(
        lane,
        MatrixPoint(32, 1.0),
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
    assert quality["all_states_certified"].values == (1.0,)
    assert quality["charged_model_hashes_verified"].values == (1.0,)
    assert quality["charge_signs_verified"].values == (1.0,)
    assert quality["contact_thermodynamics_certified"].values == (1.0,)
    assert quality["default_charged_path_rejected"].values == (1.0,)
    assert quality["monovalent_topology_verified"].values == (1.0,)
    assert quality["neutral_lifetime_bit_identical"].values == (1.0,)
    assert quality["occupancy_bounded_without_clipping"].values == (1.0,)
    assert quality["terminal_densities_positive"].values == (1.0,)
    assert quality["voc_bracketed"].values == (1.0,)
    assert quality["scenario_count"].values == (3.0,)
    assert quality["voltage_points_completed"].values == (7.0,)

    metadata = json.loads(measurement.metadata_json)
    assert metadata["protocol"]["schema_version"] == metadata["protocol_schema"]
    assert content_sha256(metadata["protocol"]) == metadata["protocol_hash"]
    assert metadata["actual"]["grid_nodes"] == {
        "S0": 33,
        "S1": 33,
        "S2": 33,
        "derived_pn": 65,
    }
    assert set(metadata["actual"]["scenario_document_sha256"]) == {
        "S0",
        "S1",
        "S2",
    }
