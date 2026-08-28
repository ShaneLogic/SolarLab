from __future__ import annotations

import json
from dataclasses import replace

import pytest

from perovskite_sim.models.device import InterfaceDefect
from perovskite_sim.models.interface_defects import (
    ENERGY_BELOW_REFERENCE_CONDUCTION_BAND,
    EQUILIBRIUM_REFERENCED_ELECTRON_OCCUPANCY,
    EXPLICIT_INTERFACE_DEFECT_SCHEMA_VERSION,
    ExplicitInterfaceDefectSchemaError,
    INTEGRATED_AREAL_TOTAL,
    InterfaceDefectDocument,
    InterfaceDefectKinetics,
    REFERENCE_ABSORBER_ELSE_LOWER_GAP,
)
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.reproducibility import semantic_sha256
from perovskite_sim.scaps_compat.loader import load_scaps_yaml


def _document() -> InterfaceDefectDocument:
    return InterfaceDefectDocument.from_scaps_cgs(
        sigma_n_cm2=3.0e-20,
        sigma_p_cm2=5.0e-20,
        thermal_velocity_cm_s=2.0e7,
        total_density_cm2=4.0e12,
        trap_depth_eV_below_cb=0.55,
    )


def test_scaps_adapter_preserves_microscopic_units_and_capture_velocity():
    document = _document()

    assert document.schema_version == EXPLICIT_INTERFACE_DEFECT_SCHEMA_VERSION
    assert document.energy_reference == ENERGY_BELOW_REFERENCE_CONDUCTION_BAND
    assert document.reference_selection == REFERENCE_ABSORBER_ELSE_LOWER_GAP
    assert document.density_normalization == INTEGRATED_AREAL_TOTAL
    assert document.charge_convention == EQUILIBRIUM_REFERENCED_ELECTRON_OCCUPANCY
    assert document.total_density_m2 == pytest.approx(4.0e16)
    assert document.kinetics.sigma_n_m2 == pytest.approx(3.0e-24)
    assert document.kinetics.sigma_p_m2 == pytest.approx(5.0e-24)
    assert document.kinetics.thermal_velocity_n_m_s == pytest.approx(2.0e5)
    assert document.capture_velocities_m_s == pytest.approx((0.024, 0.04))
    assert document.to_scaps_cgs_fields() == pytest.approx(
        {
            "sigma_n_cm2": 3.0e-20,
            "sigma_p_cm2": 5.0e-20,
            "v_th_cm_s": 2.0e7,
            "N_t_cm2": 4.0e12,
            "E_t_eV_below_cb": 0.55,
        }
    )


def test_canonical_round_trip_and_hash_are_stable():
    document = _document()
    reordered = json.loads(document.canonical_json())

    rebuilt = InterfaceDefectDocument.from_dict(reordered)

    assert rebuilt == document
    assert rebuilt.canonical_json() == document.canonical_json()
    assert rebuilt.sha256 == document.sha256
    assert len(document.sha256) == 64


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"unknown": 1}, "unknown keys"),
        ({"drop": "charge_convention"}, "missing keys"),
    ],
)
def test_document_schema_rejects_unknown_and_missing_fields(mutation, message):
    payload = _document().to_dict()
    if "drop" in mutation:
        payload.pop(mutation["drop"])
    else:
        payload.update(mutation)

    with pytest.raises(ExplicitInterfaceDefectSchemaError, match=message):
        InterfaceDefectDocument.from_dict(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("total_density_m2", 0.0),
        ("trap_depth_eV", -0.1),
        ("degeneracy", float("nan")),
    ],
)
def test_document_rejects_nonphysical_scalar_fields(field, value):
    payload = _document().to_dict()
    payload[field] = value

    with pytest.raises(ExplicitInterfaceDefectSchemaError):
        InterfaceDefectDocument.from_dict(payload)


def test_zero_cross_section_is_a_valid_capture_off_limit():
    kinetics = InterfaceDefectKinetics(
        sigma_n_m2=0.0,
        sigma_p_m2=1.0e-19,
        thermal_velocity_n_m_s=1.0e5,
        thermal_velocity_p_m_s=2.0e5,
    )
    assert kinetics.capture_coefficient_n_m3_s == 0.0
    assert kinetics.capture_coefficient_p_m3_s == pytest.approx(2.0e-14)


def test_legacy_interface_defect_remains_valid_without_microscopic_document():
    defect = InterfaceDefect(E_t_eV=0.55, N_t_cm2=4.0e12)

    assert defect.microscopic_document is None
    with pytest.raises(ValueError, match="no microscopic kinetics"):
        _ = defect.microscopic_capture_velocities_m_s


def test_interface_defect_rejects_duplicate_density_or_energy_drift():
    document = _document()

    with pytest.raises(ValueError, match="E_t_eV must match"):
        InterfaceDefect(
            E_t_eV=0.50,
            N_t_cm2=4.0e12,
            microscopic_document=document,
        )
    with pytest.raises(ValueError, match="N_t_cm2 must match"):
        InterfaceDefect(
            E_t_eV=0.55,
            N_t_cm2=5.0e12,
            microscopic_document=document,
        )


def test_charge_on_stack_semantic_identity_includes_microscopic_document():
    stack = load_device_from_yaml("configs/interface_charge_research.yaml")
    defect = stack.interface_defects[0]
    assert defect is not None and defect.microscopic_document is not None
    changed_document = InterfaceDefectDocument.from_dict(
        {
            **defect.microscopic_document.to_dict(),
            "kinetics": {
                **defect.microscopic_document.kinetics.to_dict(),
                "sigma_n_m2": 2.0
                * defect.microscopic_document.kinetics.sigma_n_m2,
            },
        }
    )
    changed = replace(
        stack,
        interface_defects=(
            replace(
                defect,
                microscopic_document=changed_document,
            ),
        ),
    )

    assert semantic_sha256(changed) != semantic_sha256(stack)


def test_scaps_loader_promotes_only_resolved_single_level_interface_species():
    stack = load_scaps_yaml("configs/interface_charge_reference.yaml")

    assert stack.interface_defects[1].microscopic_document is not None
    assert stack.interface_defects[2].microscopic_document is None
