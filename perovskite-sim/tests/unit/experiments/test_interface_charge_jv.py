from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from perovskite_sim.constants import Q
from perovskite_sim.experiments import interface_charge_jv as charged_jv
from perovskite_sim.experiments import quasi_fermi_steady_state as qf_module
from perovskite_sim.experiments.jv_sweep import build_electrical_grid
from perovskite_sim.experiments.quasi_fermi_steady_state import (
    EquilibriumReferencedInterfaceChargeDarkReference,
    QuasiFermiSteadyStateError,
    QuasiFermiSteadyStateResult,
    build_equilibrium_referenced_interface_charge_dark_reference,
    build_two_sided_trace_grid,
    solve_equilibrium_referenced_interface_charge_steady_state,
)
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.physics.interface_plane import FERMI_DIRAC_RICHARDSON
from perovskite_sim.physics.two_sided_interface import TWO_SIDED_TRACE
from perovskite_sim.sweeps.device_parameter_sweep import SweepPoint, apply_sweep_point


def _protocol() -> charged_jv.InterfaceChargeJVProtocol:
    return charged_jv.InterfaceChargeJVProtocol(
        voltages_V=(0.0, 0.6, 1.2),
        temperature_K=300.0,
    )


def _state(
    voltage: float,
    current: float,
    *,
    illuminated: bool,
    closure: str,
    occupancy: float,
    gauss: float = 1.0e-12,
    contact_span_eV: float = 0.0,
) -> QuasiFermiSteadyStateResult:
    nodes = 4
    sheet_charge = -Q * 5.0e14 * (occupancy - 0.5)
    return QuasiFermiSteadyStateResult(
        y=np.full(2 * nodes, 1.0),
        phi=np.linspace(0.0, voltage, nodes),
        electron_quasi_fermi_potential_V=np.full(nodes, voltage),
        hole_quasi_fermi_potential_V=np.zeros(nodes),
        electron_face_current_A_m2=np.full(nodes - 1, current),
        hole_face_current_A_m2=np.zeros(nodes - 1),
        total_face_current_A_m2=np.full(nodes - 1, current),
        electron_rate_per_s=np.zeros(nodes),
        hole_rate_per_s=np.zeros(nodes),
        current_A_m2=current,
        face_current_spread_A_m2=1.0e-9,
        electron_continuity_bound_A_m2=2.0e-9,
        hole_continuity_bound_A_m2=3.0e-9,
        max_normalized_cell_residual=4.0e-9,
        poisson_residual=5.0e-10,
        poisson_residual_C_m2=1.0e-15,
        illumination_steps=(0.0, 1.0) if illuminated else (0.0,),
        newton_iterations=3,
        residual_evaluations=5,
        V_app=voltage,
        illuminated=illuminated,
        certified=True,
        interface_boundary=True,
        interface_transport_model=FERMI_DIRAC_RICHARDSON,
        interface_topology=TWO_SIDED_TRACE,
        interface_faces=(1,),
        interface_local_residual=6.0e-10,
        interface_charge_closure=closure,
        interface_equilibrium_occupancy=((0.5,) if closure != "off" else ()),
        interface_occupancy=(occupancy,),
        interface_incremental_sheet_charge_C_m2=(
            (sheet_charge,) if closure != "off" else ()
        ),
        interface_trace_potential_shift_V=(
            ((1.0e-6, -1.0e-6),) if closure != "off" else ()
        ),
        interface_normalized_gauss_residual=((gauss,) if closure != "off" else ()),
        interface_scaled_local_jacobian_condition=(
            (2.0e4,) if closure != "off" else ()
        ),
        interface_charge_reference_grid_sha256=(
            "b" * 64 if closure != "off" else None
        ),
        interface_charge_reference_stack_sha256=(
            "c" * 64 if closure != "off" else None
        ),
        interface_charge_reference_dark_state_sha256=(
            "d" * 64 if closure != "off" else None
        ),
        contact_thermodynamic_status="certified",
        contact_fermi_level_span_eV=contact_span_eV,
    )


def _reference() -> EquilibriumReferencedInterfaceChargeDarkReference:
    dark = _state(
        0.0,
        0.0,
        illuminated=False,
        closure="off",
        occupancy=0.5,
    )
    return EquilibriumReferencedInterfaceChargeDarkReference(
        dark_state=dark,
        equilibrium_occupancy=(0.5,),
        trap_density_m2=(5.0e14,),
        interface_defect_document_sha256=("a" * 64,),
        capture_velocities_m_s=((0.03, 0.05),),
        interface_transmission=1.0,
        grid_sha256="b" * 64,
        stack_sha256="c" * 64,
        dark_state_sha256="d" * 64,
    )


def _charged_dark(
    reference: EquilibriumReferencedInterfaceChargeDarkReference,
) -> QuasiFermiSteadyStateResult:
    return replace(
        reference.dark_state,
        interface_charge_closure="equilibrium_referenced",
        interface_equilibrium_occupancy=reference.equilibrium_occupancy,
        interface_occupancy=reference.equilibrium_occupancy,
        interface_incremental_sheet_charge_C_m2=(0.0,),
        interface_trace_potential_shift_V=((0.0, 0.0),),
        interface_normalized_gauss_residual=(0.0,),
        interface_scaled_local_jacobian_condition=(0.0,),
        interface_charge_reference_grid_sha256=reference.grid_sha256,
        interface_charge_reference_stack_sha256=reference.stack_sha256,
        interface_charge_reference_dark_state_sha256=reference.dark_state_sha256,
    )


class _Stack:
    T = 300.0
    layers = ()


def _real_etl_stack():
    stack = load_device_from_yaml(Path("configs/interface_charge_research.yaml"))
    return apply_sweep_point(
        stack,
        SweepPoint(
            point_id="etl_nd_high",
            axis="etl_doping",
            label="ETL Nd=2e15 cm-3",
            updates={"etl_doping_cm3": 2.0e15},
        ),
    )


def _install_fake_solver(
    monkeypatch,
    *,
    bad_gauss: bool = False,
    bad_contact: bool = False,
    bracket: bool = True,
):
    reference = _reference()
    seeds: list[QuasiFermiSteadyStateResult | None] = []
    contact_requirements: list[bool] = []

    def build(*args, **kwargs):
        contact_requirements.append(bool(kwargs["require_contact_certificate"]))
        return reference

    monkeypatch.setattr(
        charged_jv,
        "build_equilibrium_referenced_interface_charge_dark_reference",
        build,
    )

    def solve(*args, **kwargs):
        contact_requirements.append(bool(kwargs["require_contact_certificate"]))
        voltage = float(args[2])
        illuminated = bool(kwargs["illuminated"])
        if not illuminated:
            return _charged_dark(reference)
        seeds.append(kwargs.get("initial_state"))
        current = 20.0 - (40.0 if bracket else 5.0) * voltage
        return _state(
            voltage,
            current,
            illuminated=True,
            closure="equilibrium_referenced",
            occupancy=0.55 + 0.01 * voltage,
            gauss=2.0e-10 if bad_gauss else 1.0e-12,
            contact_span_eV=6.0e-3 if bad_contact else 0.0,
        )

    monkeypatch.setattr(
        charged_jv,
        "solve_equilibrium_referenced_interface_charge_steady_state",
        solve,
    )
    return reference, seeds, contact_requirements


def test_protocol_round_trip_and_hash_bind_every_field():
    protocol = _protocol()
    restored = charged_jv.InterfaceChargeJVProtocol.from_json(
        protocol.canonical_json()
    )

    assert restored == protocol
    assert restored.protocol_sha256 == protocol.protocol_sha256
    assert "rate_V_s" not in protocol.canonical_json()
    assert replace(protocol, voltages_V=(0.0, 0.5, 1.0)).protocol_sha256 != (
        protocol.protocol_sha256
    )
    assert replace(protocol, P_in_W_m2=900.0).protocol_sha256 != (
        protocol.protocol_sha256
    )


@pytest.mark.parametrize(
    "voltages",
    [(), (0.0,), (0.1, 0.2), (0.0, 0.0), (0.0, float("nan"))],
)
def test_protocol_rejects_invalid_voltage_sampling(voltages):
    with pytest.raises((TypeError, ValueError)):
        charged_jv.InterfaceChargeJVProtocol(
            voltages_V=voltages,
            temperature_K=300.0,
        )


def test_protocol_rejects_unknown_fields_and_unstable_fd_probe():
    payload = _protocol().to_dict()
    payload["unknown"] = True
    with pytest.raises(charged_jv.InterfaceChargeJVProtocolError):
        charged_jv.InterfaceChargeJVProtocol.from_dict(payload)

    with pytest.raises(ValueError, match="fixes finite_difference_step"):
        charged_jv.InterfaceChargeJVSolverControls(
            finite_difference_step=5.0e-6
        )
    with pytest.raises(ValueError, match="may tighten but not relax"):
        charged_jv.InterfaceChargeJVSolverControls().refined(2.0)
    with pytest.raises(ValueError, match="cannot relax"):
        charged_jv.InterfaceChargeJVAcceptance(
            max_contact_fermi_level_span_eV=1.0e-2
        )
    duplicate = _protocol().canonical_json().replace(
        '"temperature_K":300.0',
        '"temperature_K":300.0,"temperature_K":300.0',
    )
    with pytest.raises(
        charged_jv.InterfaceChargeJVProtocolError,
        match="duplicate",
    ):
        charged_jv.InterfaceChargeJVProtocol.from_json(duplicate)


def test_fake_sweep_reuses_charged_points_and_retains_evidence(monkeypatch):
    reference, seeds, contact_requirements = _install_fake_solver(monkeypatch)
    progress = []
    execution = charged_jv.solve_interface_charge_jv(
        np.linspace(0.0, 1.0, 4),
        _Stack(),
        _protocol(),
        progress=lambda *args: progress.append(args),
    )

    assert execution.dark_reference is reference
    assert execution.sweep.certified
    assert execution.sweep.metrics.voc_bracketed
    assert execution.sweep.voltages_V.tolist() == [0.0, 0.6]
    assert seeds[0] is None
    assert seeds[1] is execution.sweep.points[0]
    assert contact_requirements == [True, True, True, True]
    assert len(progress) == 2
    evidence = execution.evidence
    assert evidence.protocol_sha256 == execution.protocol.protocol_sha256
    assert evidence.dark_charge_off_bit_identity_verified
    assert evidence.interface_defect_document_sha256 == ("a" * 64,)
    assert evidence.maximum_normalized_gauss_residual == pytest.approx(1.0e-12)
    assert evidence.dark_contact_thermodynamic_status == "certified"
    assert evidence.maximum_contact_fermi_level_span_eV == 0.0
    assert evidence.points[1].incremental_sheet_charge_C_m2[0] == pytest.approx(
        -Q * 5.0e14 * (evidence.points[1].occupancy[0] - 0.5),
        rel=1.0e-12,
    )


def test_sweep_fails_closed_on_gauss_gate(monkeypatch):
    _install_fake_solver(monkeypatch, bad_gauss=True)
    with pytest.raises(
        charged_jv.InterfaceChargeJVCertificationError,
        match="Gauss residual",
    ):
        charged_jv.solve_interface_charge_jv(
            np.linspace(0.0, 1.0, 4),
            _Stack(),
            _protocol(),
        )


def test_sweep_fails_closed_when_voc_is_not_bracketed(monkeypatch):
    _install_fake_solver(monkeypatch, bracket=False)
    with pytest.raises(
        charged_jv.InterfaceChargeJVCertificationError,
        match="did not bracket open circuit",
    ):
        charged_jv.solve_interface_charge_jv(
            np.linspace(0.0, 1.0, 4),
            _Stack(),
            _protocol(),
        )


def test_sweep_fails_closed_on_contact_span(monkeypatch):
    _install_fake_solver(monkeypatch, bad_contact=True)
    with pytest.raises(
        charged_jv.InterfaceChargeJVCertificationError,
        match="contact certification",
    ):
        charged_jv.solve_interface_charge_jv(
            np.linspace(0.0, 1.0, 4),
            _Stack(),
            _protocol(),
        )


def test_voltage_bridge_is_audited_but_not_retained(monkeypatch):
    reference = _reference()
    attempted: list[float] = []
    monkeypatch.setattr(
        charged_jv,
        "build_equilibrium_referenced_interface_charge_dark_reference",
        lambda *args, **kwargs: reference,
    )

    def solve(*args, **kwargs):
        voltage = float(args[2])
        if not kwargs["illuminated"]:
            return _charged_dark(reference)
        seed = kwargs.get("initial_state")
        attempted.append(voltage)
        if voltage == 0.6 and seed is not None and seed.V_app == 0.0:
            raise QuasiFermiSteadyStateError("outside direct basin")
        return _state(
            voltage,
            20.0 - 40.0 * voltage,
            illuminated=True,
            closure="equilibrium_referenced",
            occupancy=0.55,
        )

    monkeypatch.setattr(
        charged_jv,
        "solve_equilibrium_referenced_interface_charge_steady_state",
        solve,
    )
    execution = charged_jv.solve_interface_charge_jv(
        np.linspace(0.0, 1.0, 4),
        _Stack(),
        _protocol(),
    )

    assert attempted[:4] == [0.0, 0.6, 0.3, 0.6]
    assert execution.sweep.voltages_V.tolist() == [0.0, 0.6]
    assert execution.sweep.continuation_bridge_count == 1
    assert execution.evidence.continuation_bridge_count == 1
    assert [point.voltage_V for point in execution.evidence.continuation_bridges] == [
        0.3
    ]


def test_contact_opt_in_is_bit_identical_and_seed_tampering_fails_closed(
    monkeypatch,
):
    stack = _real_etl_stack()
    shared_grid = build_electrical_grid(stack, 8)
    grid = build_two_sided_trace_grid(shared_grid, stack)
    default_reference = build_equilibrium_referenced_interface_charge_dark_reference(
        grid,
        stack,
    )
    required_reference = build_equilibrium_referenced_interface_charge_dark_reference(
        grid,
        stack,
        require_contact_certificate=True,
    )

    assert default_reference.dark_state.contact_thermodynamic_status is None
    assert required_reference.dark_state.contact_thermodynamic_status == "certified"
    np.testing.assert_array_equal(
        default_reference.dark_state.y,
        required_reference.dark_state.y,
    )
    np.testing.assert_array_equal(
        default_reference.dark_state.phi,
        required_reference.dark_state.phi,
    )
    assert default_reference.dark_state_sha256 == required_reference.dark_state_sha256

    seed = solve_equilibrium_referenced_interface_charge_steady_state(
        grid,
        stack,
        0.0,
        dark_reference=required_reference,
        illuminated=False,
        require_contact_certificate=True,
    )
    assert seed.interface_charge_reference_grid_sha256 == (
        required_reference.grid_sha256
    )
    assert seed.interface_charge_reference_stack_sha256 == (
        required_reference.stack_sha256
    )
    assert seed.interface_charge_reference_dark_state_sha256 == (
        required_reference.dark_state_sha256
    )

    monkeypatch.setattr(
        qf_module,
        "solve_quasi_fermi_steady_state",
        lambda *args, **kwargs: pytest.fail("invalid seed reached the nonlinear solve"),
    )
    nonfinite_phi = seed.phi.copy()
    nonfinite_phi[0] = np.nan
    invalid = (
        (replace(seed, certified=False), "must remain certified"),
        (
            replace(seed, interface_charge_reference_stack_sha256="0" * 64),
            "provenance does not match",
        ),
        (
            replace(seed, interface_equilibrium_occupancy=(0.25,)),
            "does not share the dark reference",
        ),
        (
            replace(seed, interface_incremental_sheet_charge_C_m2=(1.0,)),
            "violates the interface charge law",
        ),
        (
            replace(seed, contact_thermodynamic_status=None),
            "lacks contact certification",
        ),
        (
            replace(seed, phi=nonfinite_phi),
            "non-finite or grid-incompatible",
        ),
    )
    for invalid_seed, message in invalid:
        with pytest.raises(ValueError, match=message):
            solve_equilibrium_referenced_interface_charge_steady_state(
                grid,
                stack,
                0.01,
                dark_reference=required_reference,
                illuminated=True,
                initial_state=invalid_seed,
                require_contact_certificate=True,
            )


@pytest.mark.slow
def test_real_small_grid_charged_jv_has_pointwise_evidence():
    stack = _real_etl_stack()
    shared_grid = build_electrical_grid(stack, 30)
    grid = build_two_sided_trace_grid(shared_grid, stack)
    protocol = charged_jv.build_interface_charge_jv_protocol(
        stack,
        np.linspace(0.0, 0.20, 9),
    )

    execution = charged_jv.solve_interface_charge_jv(grid, stack, protocol)

    assert execution.sweep.certified
    assert execution.sweep.metrics.voc_bracketed
    assert len(execution.evidence.points) == len(execution.sweep.voltages_V)
    assert execution.evidence.maximum_normalized_gauss_residual <= (
        protocol.acceptance.max_normalized_gauss_residual
    )
