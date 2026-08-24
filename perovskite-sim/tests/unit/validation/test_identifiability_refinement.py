"""Registered synthetic interface-SRH identifiability refinement contract."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest
import yaml

from perovskite_sim.validation.identifiability_refinement import (
    _load_interface_anchor,
    run_interface_srh_identifiability_refinement,
)
from perovskite_sim.validation.numerical_certificate import (
    MatrixPoint,
    content_sha256,
    load_refinement_registry,
)


ROOT = Path(__file__).resolve().parents[3]
LANE_ID = "interface-srh-identifiability-synthetic-v1"


def _lane():
    return load_refinement_registry(
        ROOT / "reproducibility/numerical_refinement_registry.yaml",
        project_root=ROOT,
    ).lane(LANE_ID)


def _metrics(measurement, *, quality: bool = False):
    values = measurement.quality if quality else measurement.observables
    return {item.name: item for item in values}


def test_registered_real_cell_matches_exact_contract_and_anchor():
    lane = _lane()
    measurement = run_interface_srh_identifiability_refinement(
        lane,
        MatrixPoint(7, 0.5),
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
    assert observables["full_rank_best_fit_log10"].shape == (2,)
    assert observables["rank_deficient_absolute_nullspace_vector"].shape == (3,)
    assert quality["rank_deficient_parameter_claim_absent"].values == (1.0,)
    assert quality["full_rank_parameters_identifiable"].values == (1.0,)
    assert quality["full_rank_truth_recovered"].values == (1.0,)

    metadata = json.loads(measurement.metadata_json)
    assert metadata["protocol_schema"] == (
        "interface-srh-identifiability-refinement-protocol-v1"
    )
    assert content_sha256(metadata["protocol"]) == metadata["protocol_hash"]
    assert metadata["protocol"]["anchor"] == {
        "calibration_factor": 0.1,
        "calibration_field": "iface_state_calibration_factor",
        "config_path": "configs/scaps_mirror_v2.yaml",
        "role": "formula_input_anchor_not_material_truth",
        "sigma_n_cm2": 1.0e-19,
        "sigma_p_cm2": 1.0e-19,
        "target": "PVK/ETL",
        "thermal_velocity_cm_s": 1.0e7,
        "trap_density_cm2": 1.0e12,
    }
    assert metadata["actual"]["carrier_condition_count"] == 7
    assert metadata["actual"]["finite_difference_step_log10"] == pytest.approx(
        5.0e-4
    )
    assert len(metadata["actual"]["full_rank_protocol_sha256"]) == 64
    assert len(metadata["actual"]["rank_deficient_mapping_sha256"]) == 64


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_executor_rejects_option_schema_drift(mutation):
    lane = _lane()
    options = dict(lane.options)
    if mutation == "missing":
        options.pop("observable_family")
    else:
        options["unregistered_control"] = True
    changed = replace(lane, options_json=json.dumps(options))

    with pytest.raises(ValueError, match="options do not match"):
        run_interface_srh_identifiability_refinement(
            changed,
            MatrixPoint(7, 0.5),
            ROOT,
        )


def test_raw_anchor_loader_fails_closed_on_missing_calibration(tmp_path):
    config = tmp_path / "anchor.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "interfaces": [
                    {
                        "target": "PVK/ETL",
                        "sigma_n_cm2": 1.0e-19,
                        "sigma_p_cm2": 1.0e-19,
                        "N_t_cm2": 1.0e12,
                        "v_th_cm_s": 1.0e7,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing fields"):
        _load_interface_anchor(
            config,
            target="PVK/ETL",
            calibration_field="iface_state_calibration_factor",
        )
