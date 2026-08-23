from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from fastapi import HTTPException

from backend import main as backend_main
from perovskite_sim.discretization.grid import Layer, multilayer_grid
from perovskite_sim.models.config_loader import (
    interface_charge_fields_from_device_dict,
    load_device_from_yaml,
)
from perovskite_sim.models.device import InterfaceChargeClosureParkedError
from perovskite_sim.solver.mol import build_material_arrays


def _grid(stack):
    return multilayer_grid(
        [Layer(layer.thickness, 6) for layer in stack.layers]
    )


def test_interface_charge_defaults_to_explicit_off():
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")

    assert stack.interface_charge_closure == "off"
    assert stack.interface_charge_rebaseline_acknowledged is False


def test_parser_recognizes_research_intent_and_acknowledgement():
    fields = interface_charge_fields_from_device_dict(
        {
            "interface_charge_closure": "equilibrium_referenced",
            "interface_charge_rebaseline_acknowledged": True,
        }
    )

    assert fields == {
        "interface_charge_closure": "equilibrium_referenced",
        "interface_charge_rebaseline_acknowledged": True,
    }


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"interface_charge_closure": "absolute"}, "must be one of"),
        (
            {"interface_charge_closure": "equilibrium_referenced"},
            "rebaseline_acknowledged=true",
        ),
        (
            {"interface_charge_rebaseline_acknowledged": True},
            "only valid",
        ),
    ],
)
def test_device_stack_rejects_ambiguous_charge_contracts(updates, message):
    base = load_device_from_yaml("configs/nip_MAPbI3.yaml")

    with pytest.raises(ValueError, match=message):
        dataclasses.replace(base, **updates)


def test_invalid_acknowledgement_is_not_truthiness_coerced():
    with pytest.raises(ValueError, match="must be boolean"):
        interface_charge_fields_from_device_dict(
            {"interface_charge_rebaseline_acknowledged": 1}
        )


def test_research_intent_roundtrips_but_is_not_solver_capability():
    base = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    research = dataclasses.replace(
        base,
        interface_charge_closure="equilibrium_referenced",
        interface_charge_rebaseline_acknowledged=True,
    )
    config = backend_main._stack_to_config_dict(research)
    rebuilt = backend_main.stack_from_dict(config)

    assert rebuilt.interface_charge_closure == "equilibrium_referenced"
    assert rebuilt.interface_charge_rebaseline_acknowledged is True
    with pytest.raises(InterfaceChargeClosureParkedError, match="PARKED"):
        build_material_arrays(_grid(rebuilt), rebuilt)


def test_backend_experiment_entry_rejects_parked_research_intent_as_422():
    base = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    config = backend_main._stack_to_config_dict(base)
    config["device"].update(
        interface_charge_closure="equilibrium_referenced",
        interface_charge_rebaseline_acknowledged=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        backend_main.build_stack(None, config)

    assert exc_info.value.status_code == 422
    assert "PARKED" in str(exc_info.value.detail)


def test_explicit_off_material_arrays_are_bit_identical_to_default():
    default = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    explicit_off = dataclasses.replace(
        default,
        interface_charge_closure="off",
        interface_charge_rebaseline_acknowledged=False,
    )
    x = _grid(default)
    before = build_material_arrays(x, default)
    after = build_material_arrays(x, explicit_off)

    for name in (
        "eps_r",
        "D_ion_node",
        "P_ion0",
        "N_A",
        "N_D",
        "chi",
        "Eg",
        "ni_sq",
        "D_n_face",
        "D_p_face",
        "dx_cell",
    ):
        np.testing.assert_array_equal(getattr(after, name), getattr(before, name))
    assert after.iface_state_charge == before.iface_state_charge == 0.0
    assert after.V_bi_bc == before.V_bi_bc
