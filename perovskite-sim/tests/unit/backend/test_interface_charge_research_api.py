from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

import backend.main as backend
from perovskite_sim.constants import Q
from perovskite_sim.physics.contacts import ContactThermodynamicCertificate


@dataclass(frozen=True)
class _FakeStack:
    interface_charge_closure: str = "equilibrium_referenced"


def _install_fake_research_lane(monkeypatch, *, corrupt_charge: bool = False):
    captured: dict[str, object] = {}
    grid = np.asarray([0.0, 0.4e-6, 0.6e-6, 1.0e-6])
    equilibrium = (0.4,)
    occupancy = (0.45,)
    trap_density = (1.0e17,)
    sheet_charge = -Q * trap_density[0] * (
        occupancy[0] - equilibrium[0]
    )
    if corrupt_charge:
        sheet_charge *= -1.0
    y_dark = np.arange(12, dtype=float) + 1.0
    phi_dark = np.linspace(0.0, 0.3, 4)
    qfn_dark = np.linspace(0.1, 0.2, 4)
    qfp_dark = np.linspace(-0.2, -0.1, 4)
    electron_faces_dark = np.zeros(3)
    hole_faces_dark = np.zeros(3)
    total_faces_dark = np.zeros(3)
    electron_rate_dark = np.zeros(4)
    hole_rate_dark = np.zeros(4)

    contact = ContactThermodynamicCertificate(
        status="certified",
        built_in_potential_mode="semiconductor_work_function",
        tolerance_eV=5.0e-3,
        fermi_level_span_eV=0.0,
        potential_mismatch_V=0.0,
        metal_work_function_mismatch_eV=None,
        contact_quasi_fermi_levels_eV=(0.0, 0.0, 0.0, 0.0),
        message="certified test contact",
    )
    dark_state = SimpleNamespace(
        certified=True,
        interface_occupancy=equilibrium,
        y=y_dark,
        phi=phi_dark,
        electron_quasi_fermi_potential_V=qfn_dark,
        hole_quasi_fermi_potential_V=qfp_dark,
        electron_face_current_A_m2=electron_faces_dark,
        hole_face_current_A_m2=hole_faces_dark,
        total_face_current_A_m2=total_faces_dark,
        electron_rate_per_s=electron_rate_dark,
        hole_rate_per_s=hole_rate_dark,
    )
    dark = SimpleNamespace(
        dark_state=dark_state,
        equilibrium_occupancy=equilibrium,
        trap_density_m2=trap_density,
        grid_sha256="a" * 64,
        stack_sha256="b" * 64,
        dark_state_sha256="c" * 64,
    )
    charged_dark = SimpleNamespace(
        certified=True,
        interface_charge_closure="equilibrium_referenced",
        interface_equilibrium_occupancy=equilibrium,
        interface_occupancy=equilibrium,
        interface_incremental_sheet_charge_C_m2=(0.0,),
        interface_trace_potential_shift_V=((0.0, 0.0),),
        y=y_dark,
        phi=phi_dark,
        electron_quasi_fermi_potential_V=qfn_dark,
        hole_quasi_fermi_potential_V=qfp_dark,
        electron_face_current_A_m2=electron_faces_dark,
        hole_face_current_A_m2=hole_faces_dark,
        total_face_current_A_m2=total_faces_dark,
        electron_rate_per_s=electron_rate_dark,
        hole_rate_per_s=hole_rate_dark,
    )
    result = SimpleNamespace(
        certified=True,
        interface_charge_closure="equilibrium_referenced",
        interface_topology="two_sided_trace",
        interface_boundary=True,
        interface_equilibrium_occupancy=equilibrium,
        interface_occupancy=occupancy,
        interface_incremental_sheet_charge_C_m2=(sheet_charge,),
        interface_trace_potential_shift_V=((1.0e-4, -2.0e-4),),
        interface_normalized_gauss_residual=(2.0e-12,),
        interface_scaled_local_jacobian_condition=(125.0,),
        current_A_m2=-21.0,
        electron_continuity_bound_A_m2=2.0e-8,
        hole_continuity_bound_A_m2=3.0e-8,
        face_current_spread_A_m2=4.0e-8,
        max_normalized_cell_residual=5.0e-8,
        poisson_residual=6.0e-12,
        poisson_residual_C_m2=7.0e-18,
        interface_local_residual=8.0e-12,
        numerical_residual_limit=4.0e-7,
        newton_iterations=9,
        residual_evaluations=41,
        V_app=0.05,
        illuminated=True,
        illumination_steps=(0.0, 1.0),
        y=np.arange(12, dtype=float) + 2.0,
        phi=np.linspace(0.05, 0.35, 4),
        electron_quasi_fermi_potential_V=np.linspace(0.15, 0.25, 4),
        hole_quasi_fermi_potential_V=np.linspace(-0.15, -0.05, 4),
    )

    monkeypatch.setattr(
        backend,
        "build_interface_charge_research_stack",
        lambda *_args: _FakeStack(),
    )
    monkeypatch.setattr(
        backend.jv_sweep,
        "build_electrical_grid",
        lambda *_args: np.asarray([0.0, 0.5e-6, 1.0e-6]),
    )
    monkeypatch.setattr(
        backend,
        "build_two_sided_trace_grid",
        lambda *_args: grid,
    )
    monkeypatch.setattr(
        backend,
        "build_material_arrays",
        lambda *_args: SimpleNamespace(
            iface_state_charge=0.0,
            has_dual_ions=False,
            N_iface_state=0,
        ),
    )
    monkeypatch.setattr(
        backend,
        "require_contact_thermodynamic_certificate",
        lambda *_args: contact,
    )

    def fake_dark_reference(x, stack, **kwargs):
        captured["dark_grid"] = np.asarray(x)
        captured["dark_stack"] = stack
        captured["dark_controls"] = kwargs
        return dark

    def fake_charged_solve(x, stack, voltage, **kwargs):
        if voltage == 0.0 and kwargs["illuminated"] is False:
            captured["dark_validation_controls"] = kwargs
            return charged_dark
        captured["target_grid"] = np.asarray(x)
        captured["target_stack"] = stack
        captured["target_voltage"] = voltage
        captured["target_controls"] = kwargs
        return result

    monkeypatch.setattr(
        backend,
        "build_equilibrium_referenced_interface_charge_dark_reference",
        fake_dark_reference,
    )
    monkeypatch.setattr(
        backend,
        "solve_equilibrium_referenced_interface_charge_steady_state",
        fake_charged_solve,
    )
    return captured


def test_research_endpoint_requires_explicit_acknowledgement(monkeypatch):
    monkeypatch.setattr(
        backend,
        "build_interface_charge_research_stack",
        lambda *_args: pytest.fail("stack must not load before acknowledgement"),
    )

    with TestClient(backend.app) as client:
        response = client.post(
            "/api/research/interface-charge/steady-state",
            json={"config_path": "interface_charge_research.yaml"},
        )

    assert response.status_code == 422
    assert "research_acknowledged=true" in response.json()["detail"]


def test_research_endpoint_returns_aligned_evidence(monkeypatch):
    captured = _install_fake_research_lane(monkeypatch)

    with TestClient(backend.app) as client:
        response = client.post(
            "/api/research/interface-charge/steady-state",
            json={
                "config_path": "interface_charge_research.yaml",
                "N_grid": 30,
                "V_app": 0.05,
                "illuminated": True,
                "research_acknowledged": True,
            },
        )

    assert response.status_code == 200, response.text
    payload = response.json()["result"]
    assert payload["evidence_status"] == "internal_numerical_research"
    assert payload["production_unlocked"] is False
    assert payload["provenance"] == {
        "requested_grid_intervals": 30,
        "actual_grid_nodes": 4,
        "interface_count": 1,
        "interface_topology": "two_sided_trace",
        "interface_charge_closure": "equilibrium_referenced",
        "research_acknowledged": True,
    }
    assert payload["dark_reference"]["dark_state_sha256"] == "c" * 64
    assert payload["dark_reference"]["charge_on_off_bit_identical"] is True
    assert payload["dark_reference"]["incremental_sheet_charge_C_m2"] == [0.0]
    assert payload["operating_point"]["certified"] is True
    assert len(payload["operating_point"]["operating_state_sha256"]) == 64
    interface = payload["interfaces"][0]
    assert interface["equilibrium_occupancy"] == pytest.approx(0.4)
    assert interface["occupancy"] == pytest.approx(0.45)
    assert interface["incremental_sheet_charge_C_m2"] == pytest.approx(
        -Q * 1.0e17 * 0.05
    )
    assert captured["target_voltage"] == 0.05
    target_controls = captured["target_controls"]
    assert target_controls["dark_reference"] is not None
    assert target_controls["illuminated"] is True
    assert target_controls["newton_residual_tolerance"] == 4.0e-7
    assert captured["dark_controls"]["interface_transmission"] == 1.0
    assert captured["dark_validation_controls"]["illuminated"] is False


def test_research_endpoint_rejects_forged_charge_evidence(monkeypatch):
    _install_fake_research_lane(monkeypatch, corrupt_charge=True)

    with TestClient(backend.app) as client:
        response = client.post(
            "/api/research/interface-charge/steady-state",
            json={
                "config_path": "interface_charge_research.yaml",
                "N_grid": 30,
                "V_app": 0.05,
                "illuminated": True,
                "research_acknowledged": True,
            },
        )

    assert response.status_code == 422
    assert "-q*Nt*(f-f_eq)" in response.json()["detail"]


def test_production_jv_route_keeps_interface_charge_parked():
    with TestClient(backend.app) as client:
        response = client.post(
            "/api/jv",
            json={
                "config_path": "interface_charge_research.yaml",
                "N_grid": 30,
                "n_points": 2,
                "solver": "quasi_fermi",
            },
        )

    assert response.status_code == 422
    assert "PARKED" in response.json()["detail"]


def test_research_endpoint_forbids_unknown_request_fields():
    with TestClient(backend.app) as client:
        response = client.post(
            "/api/research/interface-charge/steady-state",
            json={
                "config_path": "interface_charge_research.yaml",
                "research_acknowledged": True,
                "continuity_tolerance_A_m2": 1.0,
            },
        )

    assert response.status_code == 422


@pytest.mark.slow
def test_research_endpoint_real_n30_dark_bias():
    with TestClient(backend.app) as client:
        response = client.post(
            "/api/research/interface-charge/steady-state",
            json={
                "config_path": "interface_charge_research.yaml",
                "N_grid": 30,
                "V_app": 0.05,
                "illuminated": False,
                "research_acknowledged": True,
            },
        )

    assert response.status_code == 200, response.text
    payload = response.json()["result"]
    assert payload["operating_point"]["certified"] is True
    assert payload["provenance"]["interface_count"] == 1
    interface = payload["interfaces"][0]
    expected = -Q * interface["trap_density_m2"] * (
        interface["occupancy"] - interface["equilibrium_occupancy"]
    )
    assert interface["incremental_sheet_charge_C_m2"] == pytest.approx(
        expected,
        rel=1.0e-12,
        abs=1.0e-24,
    )
    assert abs(interface["normalized_gauss_residual"]) <= 1.0e-7
