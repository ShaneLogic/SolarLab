from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from perovskite_sim.experiments.impedance import run_impedance
from perovskite_sim.validation.numerical_certificate import (
    MatrixPoint,
    load_refinement_registry,
)
from perovskite_sim.validation.refinement_executors import (
    run_dynamic_defect_impedance_production,
)
from tests.integration.test_charged_explicit_defects_qf import _stack as _bulk_stack
from tests.integration.test_defect_ion_combined_impedance import (
    _bulk_interface_ion_stack,
    _contact_consistent_interface_stack,
)


def _production_grid_contract(stack):
    interval_count = len(stack.layers)
    return replace(
        stack,
        grid_interval_weights=(1.0,) * interval_count,
        grid_alphas=(2.0,) * interval_count,
    )


def _assert_public_dynamic_result(result, capability: str, n_frequency: int) -> None:
    evidence = result.dynamic_defect_evidence
    assert evidence is not None
    assert evidence.certified
    assert evidence.reasons == ()
    assert evidence.capability == capability
    assert evidence.protocol_sha256 == evidence.protocol.protocol_hash
    assert result.protocol is not None
    assert result.protocol.dynamic_defect_protocol == evidence.protocol
    assert result.Z.shape == (n_frequency,)
    assert np.all(np.isfinite(result.Z))
    assert result.diagnostics is not None
    assert result.diagnostics.admittance_faces_S_m2 is not None


def test_public_bulk_dynamic_defect_impedance_is_protocol_bound_and_certified():
    frequencies = np.logspace(-4.0, 12.0, 33)
    result = run_impedance(
        _bulk_stack(),
        frequencies,
        V_dc=0.0,
        N_grid=4,
        illuminated=False,
        method="dynamic_defect_frequency_certified",
    )

    _assert_public_dynamic_result(result, "bulk_dynamic_defect", frequencies.size)
    assert result.diagnostics.bulk_trap_charge_storage_response_F_m2 is not None
    assert result.diagnostics.interface_sheet_charge_storage_response_F_m2 is None


def test_public_interface_dynamic_defect_impedance_uses_two_sided_faces():
    stack = _production_grid_contract(_contact_consistent_interface_stack())
    frequencies = np.logspace(-8.0, 14.0, 45)
    result = run_impedance(
        stack,
        frequencies,
        V_dc=0.0,
        N_grid=8,
        illuminated=False,
        method="dynamic_defect_frequency_certified",
    )

    _assert_public_dynamic_result(
        result,
        "interface_dynamic_defect",
        frequencies.size,
    )
    assert result.dynamic_defect_evidence is not None
    assert result.dynamic_defect_evidence.interface_current_observation == (
        "symmetric_adjacent_physical_faces"
    )
    assert result.diagnostics.interface_sheet_charge_storage_response_F_m2 is not None


def test_public_triple_coupled_defect_ion_impedance_is_certified():
    stack = _production_grid_contract(_bulk_interface_ion_stack())
    frequencies = np.logspace(-3.0, 6.0, 19)
    result = run_impedance(
        stack,
        frequencies,
        V_dc=0.0,
        N_grid=8,
        illuminated=False,
        method="dynamic_defect_frequency_certified",
    )

    _assert_public_dynamic_result(
        result,
        "bulk_interface_defect_plus_ions",
        frequencies.size,
    )
    assert result.diagnostics.bulk_trap_charge_storage_response_F_m2 is not None
    assert result.diagnostics.interface_sheet_charge_storage_response_F_m2 is not None
    assert result.diagnostics.positive_ion_storage_response_F_m2 is not None


def test_registered_production_executor_emits_exact_metrics_and_protocol():
    root = Path(__file__).resolve().parents[2]
    lane = load_refinement_registry(
        root / "reproducibility/numerical_refinement_registry.yaml",
        project_root=root,
    ).lane("dynamic-defect-ion-impedance-production-v1")

    measurement = run_dynamic_defect_impedance_production(
        lane,
        MatrixPoint(8, 1.0),
        root,
    )

    assert {item.name for item in measurement.observables} == {
        gate.metric for gate in lane.observables
    }
    assert {item.name for item in measurement.quality} == {
        gate.metric for gate in lane.quality_gates
    }
    metadata = json.loads(measurement.metadata_json)
    assert metadata["protocol"]["schema_version"] == (
        "dynamic-defect-impedance-refinement-execution-protocol-v1"
    )
    assert metadata["protocol"]["frequency_sampling"]["fine_count"] == 145
    assert metadata["dynamic_defect_evidence"]["certified"] is True
