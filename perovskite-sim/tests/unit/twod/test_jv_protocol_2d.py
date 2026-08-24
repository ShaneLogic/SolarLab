from __future__ import annotations

from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from perovskite_sim.discretization.grid import Layer
from perovskite_sim.experiments.protocol import (
    ImplicitProtocolError,
    ProtocolMismatchError,
)
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.solver.tolerances import ComponentwiseAtol
from perovskite_sim.twod.experiments.jv_protocol_2d import (
    JV2DAtolProtocol,
    JV2DProtocol,
    build_jv_2d_protocol,
    resolve_jv_2d_protocol,
)
from perovskite_sim.twod.experiments.jv_sweep_2d import (
    build_jv_2d_execution_protocol,
    run_jv_sweep_2d,
)
from perovskite_sim.twod.grid_2d import build_grid_2d
from perovskite_sim.twod.microstructure import GrainBoundary, Microstructure


def _protocol(*, implicit: bool = True) -> JV2DProtocol:
    grid = build_grid_2d(
        [Layer(1.0e-7, 2), Layer(1.0e-7, 2)],
        lateral_length=1.0e-7,
        Nx=3,
        alpha_y=1.0,
        lateral_uniform=True,
    )
    microstructure = Microstructure(
        (
            GrainBoundary(
                x_position=5.0e-8,
                width=1.0e-8,
                tau_n=1.0e-8,
                tau_p=2.0e-8,
            ),
        )
    )
    return build_jv_2d_protocol(
        temperature_K=300.0,
        illuminated=True,
        grid=grid,
        microstructure=microstructure,
        voltages_V=np.array([0.0, 0.1, 0.2]),
        dwell_time_per_voltage_s=1.0e-6,
        ion_dynamics="single_mobile",
        carrier_boundary_condition="ohmic",
        interface_srh="two_sided_cross_node",
        lateral_bc="neumann",
        solver_rtol=1.0e-6,
        solver_atol=ComponentwiseAtol(),
        max_nfev_per_solve=20_000,
        max_bisect=4,
        ion_inventory_rtol=1.0e-10,
        save_snapshots=True,
        implicit_legacy_protocol=implicit,
    )


def test_protocol_has_canonical_strict_round_trip_and_hash():
    protocol = _protocol().as_explicit()
    rebuilt = JV2DProtocol.from_json(protocol.canonical_json())

    assert rebuilt == protocol
    assert rebuilt.protocol_hash == protocol.protocol_hash
    assert rebuilt.sha256 == protocol.protocol_hash
    assert len(protocol.protocol_hash) == 64
    assert rebuilt.solver_atol.to_absolute_tolerance() == ComponentwiseAtol()


def test_protocol_rejects_unknown_top_level_and_nested_keys():
    payload = json.loads(_protocol().canonical_json())
    payload["claim"] = "certified"
    with pytest.raises(ValueError, match="extra=.*claim"):
        JV2DProtocol.from_dict(payload)

    payload = json.loads(_protocol().canonical_json())
    payload["solver_atol"]["unknown"] = 1.0
    with pytest.raises(ValueError, match="extra=.*unknown"):
        JV2DProtocol.from_dict(payload)


@pytest.mark.parametrize(
    "changed",
    [
        lambda value: replace(value, dwell_time_per_voltage_s=2.0e-6),
        lambda value: replace(value, voltage_values_V=(0.0, 0.05, 0.2)),
        lambda value: replace(value, solver_rtol=5.0e-7),
        lambda value: replace(value, max_bisect=5),
        lambda value: replace(
            value,
            solver_atol=replace(value.solver_atol, refinement_factor=0.1),
        ),
        lambda value: replace(value, save_snapshots=False),
    ],
)
def test_each_execution_field_change_mints_a_new_hash(changed):
    protocol = _protocol().as_explicit()

    assert changed(protocol).protocol_hash != protocol.protocol_hash


def test_resolver_marks_compatibility_and_fails_closed_in_strict_mode():
    expected = _protocol(implicit=True)

    assert (
        resolve_jv_2d_protocol(None, expected, mode="compatibility")
        is expected
    )
    with pytest.raises(ImplicitProtocolError, match="explicit execution"):
        resolve_jv_2d_protocol(None, expected, mode="research_strict")

    explicit = expected.as_explicit()
    assert (
        resolve_jv_2d_protocol(explicit, expected, mode="research_strict")
        == explicit
    )


def test_resolver_rejects_execution_mismatch():
    expected = _protocol(implicit=True)
    supplied = replace(
        expected.as_explicit(),
        dwell_time_per_voltage_s=2.0e-6,
    )

    with pytest.raises(ProtocolMismatchError, match="dwell_time_per_voltage_s"):
        resolve_jv_2d_protocol(supplied, expected, mode="research_strict")


def test_topology_and_current_composition_must_agree():
    protocol = _protocol().as_explicit()

    with pytest.raises(ValueError, match="current_composition"):
        replace(protocol, current_composition="electron_hole_conduction")
    with pytest.raises(ValueError, match="Neumann-x"):
        replace(protocol, lateral_bc="periodic")


def test_scalar_atol_round_trip_is_exact():
    policy = JV2DAtolProtocol.from_absolute_tolerance(1.0e-8)

    assert JV2DAtolProtocol.from_dict(asdict(policy)) == policy
    assert policy.to_absolute_tolerance() == 1.0e-8


def test_mobile_runner_requires_explicit_research_protocol():
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")

    with pytest.raises(ImplicitProtocolError, match="research_strict"):
        run_jv_sweep_2d(
            stack,
            Microstructure(),
            lateral_length=1.0e-7,
            Nx=2,
            V_max=0.0,
            V_step=0.1,
            illuminated=False,
            lateral_bc="neumann",
            Ny_per_layer=2,
            settle_t=1.0e-10,
            save_snapshots=False,
            ion_dynamics="single_mobile",
            atol=ComponentwiseAtol(),
        )


def test_strict_mobile_runner_returns_protocol_current_and_inventory(monkeypatch):
    import perovskite_sim.twod.experiments.jv_sweep_2d as runner

    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    common = {
        "lateral_length": 1.0e-7,
        "Nx": 2,
        "V_max": 0.0,
        "V_step": 0.1,
        "illuminated": False,
        "lateral_bc": "neumann",
        "Ny_per_layer": 2,
        "settle_t": 1.0e-10,
        "save_snapshots": False,
        "ion_dynamics": "single_mobile",
        "atol": ComponentwiseAtol(),
        "max_nfev_per_solve": 2_000,
        "max_bisect": 1,
        "ion_inventory_rtol": 1.0e-10,
    }
    protocol = build_jv_2d_execution_protocol(
        stack,
        Microstructure(),
        **common,
    )
    monkeypatch.setattr(
        runner,
        "_integrate_step_2d",
        lambda state, *_args, **_kwargs: state.copy(),
    )

    result = run_jv_sweep_2d(
        stack,
        Microstructure(),
        **common,
        jv_2d_protocol=protocol,
        protocol_mode="research_strict",
    )

    assert result.protocol == protocol
    assert result.protocol.implicit_legacy_protocol is False
    assert result.snapshots == ()
    assert len(result.current_components) == 1
    assert len(result.ion_diagnostics) == 1
    assert result.J[0] == result.current_components[0].terminal_total_A_m2
    assert result.ion_diagnostics[0].relative_inventory_drift == 0.0
    assert result.ion_diagnostics[0].passed is True


def test_default_frozen_runner_returns_visible_implicit_protocol(monkeypatch):
    import perovskite_sim.twod.experiments.jv_sweep_2d as runner

    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    monkeypatch.setattr(
        runner,
        "_integrate_step_2d",
        lambda state, *_args, **_kwargs: state.copy(),
    )

    result = run_jv_sweep_2d(
        stack,
        Microstructure(),
        lateral_length=1.0e-7,
        Nx=2,
        V_max=0.0,
        V_step=0.1,
        illuminated=False,
        lateral_bc="periodic",
        Ny_per_layer=2,
        settle_t=1.0e-10,
        save_snapshots=False,
    )

    assert result.protocol is not None
    assert result.protocol.implicit_legacy_protocol is True
    assert result.protocol.state_topology == "frozen_ion_background"
    assert result.current_components == ()
    assert result.ion_diagnostics == ()
