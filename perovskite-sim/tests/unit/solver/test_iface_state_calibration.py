"""SS interface-plane-state calibration (``iface_state_calibration_factor``).

The per-interface calibration attenuates ONLY the steady-state
interface-plane recombination channel: ``build_material_arrays`` caches it on
``MaterialArrays.iface_state_calibration`` and ``_enable_iface_states`` folds
it into ``interface_calibration_factor`` on the SS mat (the field that the
state-SRH rate already reads in ``compute_interface_srh_on_state``). Default
1.0 = bit-identical; the transient bulk-node interface path never reads it.

Calibrated values on ``scaps_mirror_v2`` (HTL/PVK 0.02, PVK/ETL 0.10) bring
the SS interface channel from over-strong (base -61 mV, Nd_ETL ~2x over,
HTL/PVK N_t ~12x over) to base -0.1 mV / Nd_ETL 0.84x / HTL/PVK N_t 1.0x.
"""
from __future__ import annotations

import dataclasses

import pytest

from perovskite_sim.discretization.grid import multilayer_grid, Layer
from perovskite_sim.models.device import InterfaceDefect, electrical_layers
from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.solver.mol import build_material_arrays
from perovskite_sim.experiments.steady_state import _enable_iface_states

_V2 = "configs/scaps_mirror_v2.yaml"


def _build(stack):
    elec = electrical_layers(stack)
    x = multilayer_grid(
        [Layer(thickness=L.thickness, N=30 // len(elec)) for L in elec]
    )
    return build_material_arrays(x, stack)


def _set_sscal(stack, vals):
    new, k = [], 0
    for d in stack.interface_defects:
        if d is None:
            new.append(None)
        else:
            new.append(
                dataclasses.replace(d, iface_state_calibration_factor=vals[k])
            )
            k += 1
    return dataclasses.replace(stack, interface_defects=tuple(new))


def test_default_factor_is_one():
    assert InterfaceDefect(E_t_eV=0.6).iface_state_calibration_factor == 1.0


def test_loader_parses_iface_state_calibration():
    stack = load_scaps_yaml(_V2)
    cals = [
        None if d is None else d.iface_state_calibration_factor
        for d in stack.interface_defects
    ]
    # substrate-prefixed v2 -> first entry None; HTL/PVK 0.02, PVK/ETL 0.10
    assert 0.02 in cals and 0.10 in cals


def test_build_caches_iface_state_calibration():
    mat = _build(load_scaps_yaml(_V2))
    assert mat.iface_state_calibration == (0.02, 0.10)


def test_enable_iface_states_folds_into_calibration_factor():
    mat = _build(load_scaps_yaml(_V2))
    # cf is unset in the YAML -> 1.0 on both interfaces
    assert mat.interface_calibration_factor == (1.0, 1.0)
    ss = _enable_iface_states(mat)
    assert ss.interface_calibration_factor == pytest.approx((0.02, 0.10))
    assert ss.N_iface_state == 2


def test_enable_iface_states_noop_when_all_unity():
    # Default field (1.0) must leave interface_calibration_factor unchanged
    # so the SS path is bit-identical to pre-calibration behaviour.
    mat = _build(_set_sscal(load_scaps_yaml(_V2), [1.0, 1.0]))
    assert mat.iface_state_calibration == (1.0, 1.0)
    ss = _enable_iface_states(mat)
    assert ss.interface_calibration_factor == mat.interface_calibration_factor


def test_calibration_does_not_touch_iface_state_calibration_field():
    # The fold writes interface_calibration_factor; the source field is left
    # intact (so re-enabling is idempotent / inspectable).
    mat = _build(load_scaps_yaml(_V2))
    ss = _enable_iface_states(mat)
    assert ss.iface_state_calibration == (0.02, 0.10)
