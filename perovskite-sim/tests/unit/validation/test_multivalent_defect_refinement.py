"""D7-E2 multivalent explicit-defect numerical-refinement contract tests."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from perovskite_sim.validation.multivalent_defect_refinement import (
    _execution_protocol,
    _load_suite,
    _single_transition_pair,
    run_multivalent_defect_qf_dc_refinement,
)
from perovskite_sim.validation.numerical_certificate import (
    MatrixPoint,
    content_sha256,
    load_refinement_registry,
)


ROOT = Path(__file__).resolve().parents[3]
LANE_ID = "multivalent-explicit-defect-qf-dc-v1"


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
        base_newton_residual_tolerance=options["base_newton_residual_tolerance"],
        base_poisson_tolerance_V=options["base_poisson_tolerance_V"],
        base_finite_difference_step=options["base_finite_difference_step"],
        continuity_tolerance_A_m2=options["continuity_tolerance_A_m2"],
        current_spread_tolerance_A_m2=options["current_spread_tolerance_A_m2"],
    )


def test_protocol_binds_the_three_families_and_the_full_matrix():
    lane = _lane()
    protocol = _protocol(lane)

    assert protocol["matrix"] == {
        "grid_parameter": "intervals_per_electrical_layer",
        "grid_values": [24, 48, 96],
        "tolerance_factors": [1.0, 0.1, 0.01],
        "tolerance_parameter": "qf_dc_residual_tolerance_factor",
    }
    assert [item["id"] for item in protocol["scenarios"]] == ["M1", "M2", "M3"]
    assert [item["family"] for item in protocol["scenarios"]] == [
        "double_donor",
        "double_acceptor",
        "amphoteric",
    ]
    assert protocol["constitutive_closure"]["occupancy"] == (
        "local_stationary_master_equation"
    )
    assert protocol["constitutive_closure"]["state_normalization"] == (
        "one_shared_density_per_physical_defect"
    )
    assert protocol["constitutive_closure"]["occupancy_clipping"] == "none"
    # The lane must state that everything outside the stationary bulk QF/DC
    # route is excluded rather than silently unexercised.
    assert protocol["topology"]["multivalent_execution"] == "qf_dc_only"
    for excluded in (
        "dynamic_occupancy",
        "energy_distributions",
        "metastable_configurations",
        "mobile_ions",
        "spatial_grading",
    ):
        assert protocol["topology"][excluded] == "excluded"
    assert len(content_sha256(protocol)) == 64


def test_external_reference_contract_is_declared_unsupplied():
    """The SCAPS comparison is contracted but NOT performed by this lane."""
    lane = _lane()
    suite, _scenarios = _load_suite(lane, ROOT)
    contract = suite["external_reference_contract"]

    assert contract["status"] == "not_supplied"
    assert contract["required_solver"] == "SCAPS-1D"
    assert contract["independent_export_attestation_required"] is True
    assert contract["interpolation_allowed"] is False
    assert (
        "charge_state_occupation_fraction_per_state" in contract["raw_profile_columns"]
    )
    assert lane.claim_level == "internal-numerical-candidate"
    assert any("no SCAPS" in item for item in lane.limitations)


def test_each_solver_protocol_control_changes_content_hash():
    baseline = _protocol(_lane())

    changed_grid = json.loads(json.dumps(baseline))
    changed_grid["solver"]["grid_alpha"] = 3.0
    assert content_sha256(changed_grid) != content_sha256(baseline)

    changed_voltage = json.loads(json.dumps(baseline))
    changed_voltage["operating_points"]["derived_pn"]["voltage_grid_V"][-1] = 0.35
    assert content_sha256(changed_voltage) != content_sha256(baseline)

    changed_illumination = json.loads(json.dumps(baseline))
    changed_illumination["operating_points"]["derived_pn"]["illumination_steps"][1] = (
        1.0e-5
    )
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
    copied_lane = replace(lane, options_json=json.dumps(options))

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


def test_suite_rejects_a_family_or_polarity_swap(tmp_path):
    lane = _lane()
    suite_source = ROOT / lane.options["suite_manifest"]
    suite = json.loads(suite_source.read_text(encoding="utf-8"))
    for scenario in suite["scenarios"]:
        source = ROOT / scenario["config_path"]
        destination = tmp_path / scenario["config_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    suite["scenarios"][0]["family"] = "amphoteric"
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(json.dumps(suite), encoding="utf-8")
    options = dict(lane.options)
    options["suite_manifest"] = "suite.json"
    options["suite_manifest_sha256"] = hashlib.sha256(
        suite_path.read_bytes()
    ).hexdigest()
    copied_lane = replace(lane, options_json=json.dumps(options))

    with pytest.raises(ValueError, match="wrong family/polarity"):
        _load_suite(copied_lane, tmp_path)


def test_equivalence_probe_pairs_one_physical_defect_two_ways():
    lane = _lane()
    _suite, scenarios = _load_suite(lane, ROOT)
    multivalent_stack, monovalent_stack = _single_transition_pair(scenarios[0])

    multivalent_species = multivalent_stack.layers[0].params.bulk_defects[0]
    monovalent_species = monovalent_stack.layers[0].params.bulk_defects[0]
    assert multivalent_species.configuration.family == "single_donor"
    assert multivalent_species.configuration.charge_states_e == (1, 0)
    assert multivalent_species.total_density_m3 == (
        monovalent_species.distribution.total_density_m3
    )
    assert (
        multivalent_species.configuration.energy_levels.first_transition_eV_above_vb
        == monovalent_species.distribution.center_eV_above_vb
    )
    assert (
        multivalent_species.configuration.transition_kinetics[0]
        == monovalent_species.kinetics
    )
    # Same doping and band structure on both sides; only the defect document
    # differs, which is what makes the measured difference meaningful.
    assert multivalent_stack.layers[0].params.N_A == (
        monovalent_stack.layers[0].params.N_A
    )
    assert multivalent_stack.layers[0].params.Eg == (
        monovalent_stack.layers[0].params.Eg
    )


@pytest.mark.slow
def test_registered_executor_real_cell_matches_exact_contract():
    lane = _lane()
    measurement = run_multivalent_defect_qf_dc_refinement(
        lane,
        MatrixPoint(24, 1.0),
        ROOT,
    )
    observables = _metrics(measurement)
    quality = _metrics(measurement, quality=True)

    assert set(observables) == {gate.metric for gate in lane.observables}
    assert {gate.metric for gate in lane.quality_gates} == set(quality)
    assert all(
        observables[gate.metric].units == gate.units for gate in lane.observables
    )
    assert all(quality[gate.metric].units == gate.units for gate in lane.quality_gates)
    for flag in (
        "all_states_certified",
        "charge_signs_verified",
        "contact_thermodynamics_certified",
        "default_multivalent_path_rejected",
        "multivalent_model_hashes_verified",
        "multivalent_topology_verified",
        "probability_bounded_without_clipping",
        "terminal_densities_positive",
        "voc_bracketed",
    ):
        assert quality[flag].values == (1.0,), flag
    assert quality["scenario_count"].values == (3.0,)
    assert quality["voltage_points_completed"].values == (7.0,)
    # The device-level D2 reduction is the load-bearing measurement: it fails
    # if any QF/DC consumer stops sourcing the shared master-equation closure.
    assert quality["single_transition_d2_state_relative_error"].values[0] <= 1.0e-8
    assert quality["single_transition_d2_potential_error_V"].values[0] <= 1.0e-9
    assert quality["max_owned_probability_sum_error"].values[0] <= 1.0e-12
    assert quality["min_transition_rate_s1"].values[0] > 0.0

    metadata = json.loads(measurement.metadata_json)
    assert metadata["protocol"]["schema_version"] == metadata["protocol_schema"]
    assert content_sha256(metadata["protocol"]) == metadata["protocol_hash"]
    assert metadata["actual"]["grid_nodes"] == {
        "M1": 25,
        "M2": 25,
        "M3": 25,
        "derived_pn": 49,
        "equivalence_probe": 25,
    }
    assert set(metadata["actual"]["scenario_document_sha256"]) == {"M1", "M2", "M3"}
    assert metadata["actual"]["state_counts"] == {
        "M1": [3],
        "M2": [3],
        "M3": [3],
    }
