from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from perovskite_sim.constants import Q
from perovskite_sim.physics.contacts import ContactThermodynamicCertificate
from perovskite_sim.validation import interface_charge_stress as stress
from perovskite_sim.validation.numerical_certificate import (
    MatrixPoint,
    content_sha256,
    load_refinement_registry,
)


ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class _FakeStack:
    point: object | None = None
    interface_charge_closure: str = "equilibrium_referenced"
    interface_charge_rebaseline_acknowledged: bool = True
    het_recomb_despike: float = 0.0
    flat_band_contacts: bool = False
    flat_band_metal_contacts: bool = False
    contact_phi_B_eV: float = 0.0
    autoloop_generated_lever: bool = False


def _result(
    *,
    equilibrium: float,
    density: float,
    closure: str,
    occupancy: float,
    current: float,
):
    charge = -Q * density * (occupancy - equilibrium)
    shift = np.sign(charge) * abs(charge) * 0.1
    arrays = {
        "y": np.ones(12),
        "phi": np.zeros(4),
        "electron_quasi_fermi_potential_V": np.zeros(4),
        "hole_quasi_fermi_potential_V": np.zeros(4),
        "electron_face_current_A_m2": np.zeros(3),
        "hole_face_current_A_m2": np.zeros(3),
        "total_face_current_A_m2": np.zeros(3),
        "electron_rate_per_s": np.zeros(4),
        "hole_rate_per_s": np.zeros(4),
    }
    return SimpleNamespace(
        **arrays,
        certified=True,
        current_A_m2=current,
        electron_continuity_bound_A_m2=1.0e-9,
        hole_continuity_bound_A_m2=2.0e-9,
        face_current_spread_A_m2=3.0e-9,
        interface_local_residual=4.0e-11,
        max_normalized_cell_residual=5.0e-10,
        poisson_residual=6.0e-12,
        interface_topology=stress.TWO_SIDED_TRACE,
        interface_charge_closure=closure,
        interface_equilibrium_occupancy=(equilibrium,),
        interface_occupancy=(occupancy,),
        interface_incremental_sheet_charge_C_m2=(charge,),
        interface_trace_potential_shift_V=((shift, shift),),
        interface_normalized_gauss_residual=(7.0e-12,),
        interface_scaled_local_jacobian_condition=(100.0,),
    )


def test_interface_charge_stress_executor_smoke(monkeypatch):
    registry = load_refinement_registry(
        ROOT / "reproducibility/numerical_refinement_registry.yaml",
        project_root=ROOT,
    )
    lane = registry.lane("interface-charge-device-stress-v1")
    baseline = _FakeStack()
    calls = {"references": 0, "targets": 0}

    monkeypatch.setattr(stress, "_load_stack", lambda *_args: baseline)
    monkeypatch.setattr(
        stress,
        "apply_sweep_point",
        lambda stack, point, **_kwargs: replace(stack, point=point),
    )

    def defect_for(stack):
        updates = {} if stack.point is None else dict(stack.point.updates)
        return SimpleNamespace(
            E_t_eV=float(updates.get("interface_defect_E_t_eV", 0.55)),
            N_t_cm2=float(updates.get("interface_defect_N_t_cm2", 1.0e13)),
            calibration_factor=1.0,
            iface_state_calibration_factor=1.0,
        )

    monkeypatch.setattr(
        stress,
        "electrical_interface_defects",
        lambda stack: (defect_for(stack),),
    )
    monkeypatch.setattr(
        stress,
        "build_electrical_grid",
        lambda *_args: np.linspace(0.0, 1.0, 4),
    )
    monkeypatch.setattr(stress, "build_two_sided_trace_grid", lambda x, _s: x)
    monkeypatch.setattr(
        stress,
        "build_material_arrays",
        lambda *_args: SimpleNamespace(iface_state_charge=0.0),
    )
    contact = ContactThermodynamicCertificate(
        status="certified",
        built_in_potential_mode="semiconductor_work_function",
        tolerance_eV=5.0e-3,
        fermi_level_span_eV=0.0,
        potential_mismatch_V=0.0,
        metal_work_function_mismatch_eV=None,
        contact_quasi_fermi_levels_eV=(0.0, 0.0, 0.0, 0.0),
        message="test",
    )
    monkeypatch.setattr(
        stress,
        "require_contact_thermodynamic_certificate",
        lambda *_args: contact,
    )

    def dark_reference(_grid, stack, **_kwargs):
        calls["references"] += 1
        defect = defect_for(stack)
        density = defect.N_t_cm2 * 1.0e4
        dark = _result(
            equilibrium=0.25,
            density=density,
            closure="off",
            occupancy=0.25,
            current=0.0,
        )
        token = "baseline" if stack.point is None else stack.point.point_id
        return SimpleNamespace(
            dark_state=dark,
            equilibrium_occupancy=(0.25,),
            trap_density_m2=(density,),
            interface_transmission=1.0,
            grid_sha256="1" * 64,
            stack_sha256=hashlib.sha256(token.encode()).hexdigest(),
            dark_state_sha256=hashlib.sha256((token + "dark").encode()).hexdigest(),
        )

    monkeypatch.setattr(
        stress,
        "build_equilibrium_referenced_interface_charge_dark_reference",
        dark_reference,
    )

    def solve(_grid, _stack, voltage, *, dark_reference, illuminated, **_kwargs):
        calls["targets"] += 1
        equilibrium = dark_reference.equilibrium_occupancy[0]
        density = dark_reference.trap_density_m2[0]
        if voltage == 0.0 and not illuminated:
            return _result(
                equilibrium=equilibrium,
                density=density,
                closure="equilibrium_referenced",
                occupancy=equilibrium,
                current=0.0,
            )
        occupancy = 0.30 if illuminated else 0.20
        return _result(
            equilibrium=equilibrium,
            density=density,
            closure="equilibrium_referenced",
            occupancy=occupancy,
            current=-2.0 if illuminated else 1.0,
        )

    monkeypatch.setattr(
        stress,
        "solve_equilibrium_referenced_interface_charge_steady_state",
        solve,
    )

    def describe(stack):
        updates = {} if stack.point is None else dict(stack.point.updates)
        return {
            "etl_delta_ec_eV": float(updates.get("etl_delta_ec_eV", -0.1)),
            "etl_N_D_cm3": float(updates.get("etl_doping_cm3", 0.0)),
        }

    monkeypatch.setattr(stress, "describe_stack", describe)

    measurement = stress.run_equilibrium_referenced_interface_charge_stress(
        lane,
        MatrixPoint(30, 0.5),
        ROOT,
    )

    observables = {item.name: item for item in measurement.observables}
    quality = {item.name: item for item in measurement.quality}
    assert set(observables) == {gate.metric for gate in lane.observables}
    assert set(quality) == {gate.metric for gate in lane.quality_gates}
    assert observables["stress_current_density_A_m2"].shape == (9, 2)
    assert observables["stress_interface_occupancy"].shape == (9, 2, 1)
    assert observables["stress_trace_potential_shift_V"].shape == (9, 2, 1, 2)
    assert all(item.values == (1.0,) for name, item in quality.items() if name in {
        "all_points_certified",
        "barrier_shift_charge_sign_consistent",
        "charge_law_consistent",
        "parameter_values_applied",
        "variant_stack_identities_unique",
    })
    assert quality["stress_point_count"].values == (9.0,)
    assert calls == {"references": 9, "targets": 27}

    charge = np.asarray(
        observables["stress_sheet_charge_C_m2"].values
    ).reshape(observables["stress_sheet_charge_C_m2"].shape)
    assert abs(charge[7, 0, 0] / charge[0, 0, 0]) == pytest.approx(1.0e-5)
    assert abs(charge[8, 0, 0] / charge[0, 0, 0]) == pytest.approx(1.0e-3)
    metadata = json.loads(measurement.metadata_json)
    assert content_sha256(metadata["protocol"]) == metadata["protocol_hash"]
    assert metadata["protocol_schema"] == (
        "interface-charge-device-stress-protocol-v1"
    )
    assert [record["point"]["point_id"] for record in metadata["stress_records"]] == [
        point["point_id"] for point in lane.options["stress_points"]
    ]


def test_stress_protocol_rejects_duplicate_point_ids():
    registry = load_refinement_registry(
        ROOT / "reproducibility/numerical_refinement_registry.yaml",
        project_root=ROOT,
    )
    raw = list(registry.lane("interface-charge-device-stress-v1").options["stress_points"])
    raw[1] = {**raw[1], "point_id": raw[0]["point_id"]}

    with pytest.raises(ValueError, match="point IDs must be unique"):
        stress._stress_points({"stress_points": raw})
