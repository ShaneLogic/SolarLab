from __future__ import annotations

import json
from pathlib import Path

from perovskite_sim.validation.numerical_certificate import (
    MatrixPoint,
    load_refinement_registry,
)
from perovskite_sim.validation.refinement_executors import (
    run_twod_mobile_ion_interface_srh,
)


ROOT = Path(__file__).resolve().parents[2]


def test_real_registered_combined_twod_cell_returns_all_certificate_axes():
    lane = load_refinement_registry(
        ROOT / "reproducibility/numerical_refinement_registry.yaml",
        project_root=ROOT,
    ).lane("twod-mobile-ion-interface-srh-v1")

    measurement = run_twod_mobile_ion_interface_srh(
        lane,
        MatrixPoint(4, 0.1),
        ROOT,
    )

    observables = {item.name: item for item in measurement.observables}
    quality = {item.name: item.values[0] for item in measurement.quality}
    assert set(observables) == {gate.metric for gate in lane.observables}
    assert set(quality) == {gate.metric for gate in lane.quality_gates}
    assert quality["combined_gb_ion_interface_topology_verified"] == 1.0
    assert quality["clamp_inactive_slice_verified"] == 1.0
    assert quality["minimum_mobile_ion_relative_redistribution"] > 1.0e-4
    assert quality["minimum_lateral_carrier_variation_relative"] > 1.0e-4
    metadata = json.loads(measurement.metadata_json)
    assert metadata["execution_protocol"]["implicit_legacy_protocol"] is False
    assert metadata["execution_protocol"]["current_composition"] == (
        "electron_hole_positive_ion_displacement"
    )
