"""R1-0 gates; no mechanism or real-material validation claim."""

import copy
import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from perovskite_sim.experiments.one_dimensional_mechanism_r1 import (
    _digest, build_r1_material, validate_binding, validate_physical_material,
)
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.physics.physical_control_volume import PHYSICAL_GEOMETRY_V1
from perovskite_sim.experiments.quasi_fermi_steady_state import _research_charge_off_stack
from perovskite_sim.experiments.interface_defect_ion_transient import (
    InterfaceDefectIonTransientPolicy, run_interface_defect_ion_device_transient,
)


FIXTURE = Path(__file__).parents[1] / "fixtures/configs/dynamic_interface_defect_ion_transient_absorber_only.yaml"


def binding(stack):
    f = 7.928011033749389e-5
    return seal({
        "schema": "ReferenceBindingV1", "geometry": PHYSICAL_GEOMETRY_V1,
        "f_ref": [f], "reference_intervals": 512,
        "rungs": [
            {"intervals": n, "f_ref": [v], "difference": d}
            for n, v, d in [
                (32, 7.954622426871699e-5, None),
                (64, 7.934721077963447e-5, 1.9901348908252074e-7),
                (128, 7.929619841679834e-5, 5.1012362836127157e-8),
                (256, 7.928333653224332e-5, 1.2861884555017203e-8),
                (512, f, 3.2261947494337554e-9),
            ]
        ],
        "microscopic_documents": list(_research_charge_off_stack(stack)[1].document_sha256),
        "stack_sha256": hashlib.sha256(repr(stack).encode("utf-8")).hexdigest(),
    })


def seal(payload):
    payload["sha256"] = _digest({k: v for k, v in payload.items() if k != "sha256"})
    return payload


@pytest.mark.parametrize("field,value", [
    ("f_ref", [.1]), ("geometry", "legacy"), ("stack_sha256", "0"*64),
    ("reference_intervals", 128),
])
def test_binding_rejects_changed_identity(field, value):
    stack = load_device_from_yaml(FIXTURE)
    record = binding(stack)
    validate_binding(record, stack)
    record[field] = value
    with pytest.raises(ValueError):
        validate_binding(record, stack)
    with pytest.raises(ValueError):
        validate_binding(seal(record), stack)


def test_rejects_mismatched_volumes_and_unbound_research_geometry():
    stack = load_device_from_yaml(FIXTURE)
    x, mat = build_r1_material(stack, 4)
    validate_physical_material(x, stack, mat)
    with pytest.raises(ValueError):
        validate_physical_material(x, stack, replace(mat, dx_cell=mat.dx_cell*2))
    with pytest.raises(RuntimeError, match="fixed reference"):
        run_interface_defect_ion_device_transient(x, stack, [0, 1e-8], [0, .005], mat=mat)


@pytest.mark.slow
@pytest.mark.parametrize("intervals", [4, 16])
def test_physical_coupled_steps_keep_fixed_reference_and_close_charge(intervals):
    from perovskite_sim.experiments.one_dimensional_mechanism_r1 import physical_step_record
    stack = load_device_from_yaml(FIXTURE)
    x, mat = build_r1_material(stack, intervals)
    reference = binding(stack)
    records = []
    def observe(system, state, previous, dt, time, substeps, residual):
        r = physical_step_record(system, state, previous, dt)
        assert r["gauss_normalized"] <= 1e-10
        assert r["inventory_relative_drift"] <= 1e-10
        if dt:
            assert r["charge_balance_normalized"] <= 1e-10
            assert r["contact_internal_current_spread_relative"] <= 2e-6
        records.append(r)
    result = run_interface_defect_ion_device_transient(
        x, stack, [0, 1e-8, 1e-6, 1e-4], [0, .005, .005, .005],
        mat=mat, research_binding=reference, accepted_step_observer=observe,
        policy=InterfaceDefectIonTransientPolicy(maximum_ion_inventory_relative_drift=1e-10),
    )
    assert len(records) == 3 + 3*(1+2+4)
    assert result.certificate.certified
    assert result.version == "r1-0-physical-volumes-v1"
    assert result.interface_sheet_charge_C_m2[0, 0] != 0
    np.testing.assert_array_equal(result.dark_reference.equilibrium_occupancy, reference["f_ref"])
    np.testing.assert_array_equal(result.positive_ion_density_m3[:, x > 1e-7], 1e22)


@pytest.mark.slow
@pytest.mark.parametrize("intervals", [4, 16])
def test_physical_ac_zero_inventory_and_blocking_contacts(intervals):
    from perovskite_sim.experiments.defect_ion_combined_impedance import run_defect_ion_combined_impedance
    from perovskite_sim.constants import Q
    stack = load_device_from_yaml(FIXTURE)
    x, mat = build_r1_material(stack, intervals)
    result = run_defect_ion_combined_impedance(
        x, stack, np.logspace(-3, 8, 45), research_binding=binding(stack),
    )
    assert result.certificate.certified
    assert result.admittance_faces_S_m2.shape == (45, x.size+2)
    assert max(mat.V_T_device*abs(result.positive_ion_storage_response_F_m2/Q)/1e15) <= 1e-10
    np.testing.assert_array_equal(result.positive_ion_admittance_faces_S_m2[:, [0, -1]], 0)
