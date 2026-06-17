import dataclasses
import importlib
from pathlib import Path

import numpy as np

from perovskite_sim.discretization.grid import multilayer_grid, Layer
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.solver.mol import build_material_arrays
from perovskite_sim.autoloop.codegen import LEVER_TEMPLATE, splice_lever_body
from perovskite_sim.autoloop.generated.lever import adjust_material_arrays
from perovskite_sim.autoloop.generated._ctx import _LeverContext

_V2 = "configs/scaps_mirror_v2.yaml"
_LEVER_PATH = Path(adjust_material_arrays.__code__.co_filename)


def _build(stack):
    elec = electrical_layers(stack)
    x = multilayer_grid([Layer(thickness=L.thickness, N=30 // len(elec)) for L in elec])
    return x, build_material_arrays(x, stack)


def test_identity_default_returns_same_object():
    sentinel = object()
    assert adjust_material_arrays(sentinel, _LeverContext(x=None, stack=None)) is sentinel


def test_flag_off_bit_identical(monkeypatch):
    monkeypatch.delenv("SOLARLAB_AUTOLOOP_GEN", raising=False)
    base = load_scaps_yaml(_V2)
    _, a_off = _build(dataclasses.replace(base, autoloop_generated_lever=False))
    _, a_on = _build(dataclasses.replace(base, autoloop_generated_lever=True))  # identity body
    assert np.array_equal(a_off.chi, a_on.chi)
    assert np.array_equal(a_off.Eg, a_on.Eg)
    assert np.array_equal(a_off.ni_sq, a_on.ni_sq)


def test_flag_on_spliced_body_shifts_chi(monkeypatch):
    """A non-identity body supplied as BODY-ONLY statements (no def, no imports),
    spliced into the real lever.py and exercised through the flag-ON hook which
    imports adjust_material_arrays (lever) + _LeverContext (_ctx)."""
    monkeypatch.delenv("SOLARLAB_AUTOLOOP_GEN", raising=False)
    base = load_scaps_yaml(_V2)
    _, a_off = _build(dataclasses.replace(base, autoloop_generated_lever=False))

    original = _LEVER_PATH.read_text(encoding="utf-8")
    spliced = splice_lever_body(LEVER_TEMPLATE,
                                "return dataclasses.replace(arrays, chi=arrays.chi + 0.1)")
    import perovskite_sim.autoloop.generated.lever as lev
    try:
        _LEVER_PATH.write_text(spliced, encoding="utf-8")
        importlib.reload(lev)                         # pick up the spliced body
        _, a_on = _build(dataclasses.replace(base, autoloop_generated_lever=True))
        assert np.allclose(a_on.chi, a_off.chi + 0.1)
    finally:
        _LEVER_PATH.write_text(original, encoding="utf-8")
        importlib.reload(lev)                         # restore identity


def test_env_flag_triggers_hook(monkeypatch):
    base = load_scaps_yaml(_V2)
    _, a_off = _build(dataclasses.replace(base, autoloop_generated_lever=False))

    original = _LEVER_PATH.read_text(encoding="utf-8")
    spliced = splice_lever_body(LEVER_TEMPLATE,
                                "return dataclasses.replace(arrays, Eg=arrays.Eg + 0.05)")
    import perovskite_sim.autoloop.generated.lever as lev
    try:
        _LEVER_PATH.write_text(spliced, encoding="utf-8")
        importlib.reload(lev)
        monkeypatch.setenv("SOLARLAB_AUTOLOOP_GEN", "1")
        _, a_on = _build(dataclasses.replace(base, autoloop_generated_lever=False))  # off on stack, on via env
        assert np.allclose(a_on.Eg, a_off.Eg + 0.05)
    finally:
        _LEVER_PATH.write_text(original, encoding="utf-8")
        importlib.reload(lev)
