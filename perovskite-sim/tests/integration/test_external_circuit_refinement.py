from __future__ import annotations

import json
from pathlib import Path

from perovskite_sim.validation.external_circuit_refinement import (
    run_external_series_shunt_dc_refinement,
)
from perovskite_sim.validation.numerical_certificate import (
    MatrixPoint,
    load_refinement_registry,
)


ROOT = Path(__file__).resolve().parents[2]


def test_real_registered_external_circuit_cell_returns_all_certificate_axes():
    lane = load_refinement_registry(
        ROOT / "reproducibility/numerical_refinement_registry.yaml",
        project_root=ROOT,
    ).lane("external-series-shunt-dc-v1")

    measurement = run_external_series_shunt_dc_refinement(
        lane,
        MatrixPoint(20, 1.0),
        ROOT,
    )

    observables = {item.name: item for item in measurement.observables}
    quality = {item.name: item.values[0] for item in measurement.quality}
    assert set(observables) == {gate.metric for gate in lane.observables}
    assert set(quality) == {gate.metric for gate in lane.quality_gates}
    assert quality["intrinsic_jv_certified"] == 1.0
    assert quality["external_circuit_certified"] == 1.0
    assert quality["zero_coupling_exact"] == 1.0
    assert quality["min_pce_loss_fraction"] >= 0.01
    assert quality["max_pce_loss_fraction"] <= 0.5
    metadata = json.loads(measurement.metadata_json)
    assert metadata["protocol"]["schema_version"] == (
        "external-series-shunt-dc-refinement-protocol-v1"
    )
    assert metadata["actual"]["grid"] == 20
    assert metadata["actual"]["tolerance_factor"] == 1.0
