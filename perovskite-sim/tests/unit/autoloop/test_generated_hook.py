import dataclasses
import numpy as np

from perovskite_sim.discretization.grid import multilayer_grid, Layer
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.solver.mol import build_material_arrays
from perovskite_sim.autoloop.generated.lever import adjust_material_arrays, _LeverContext

_V2 = "configs/scaps_mirror_v2.yaml"


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


def test_flag_on_nonidentity_body_shifts_chi(monkeypatch):
    monkeypatch.delenv("SOLARLAB_AUTOLOOP_GEN", raising=False)
    base = load_scaps_yaml(_V2)
    _, a_off = _build(dataclasses.replace(base, autoloop_generated_lever=False))
    import perovskite_sim.autoloop.generated.lever as lev

    def _shift(arrays, ctx):
        return dataclasses.replace(arrays, chi=arrays.chi + 0.1)

    monkeypatch.setattr(lev, "adjust_material_arrays", _shift)
    _, a_on = _build(dataclasses.replace(base, autoloop_generated_lever=True))
    assert np.allclose(a_on.chi, a_off.chi + 0.1)


def test_env_flag_triggers_hook(monkeypatch):
    base = load_scaps_yaml(_V2)
    _, a_off = _build(dataclasses.replace(base, autoloop_generated_lever=False))
    import perovskite_sim.autoloop.generated.lever as lev
    monkeypatch.setattr(lev, "adjust_material_arrays",
                        lambda arrays, ctx: dataclasses.replace(arrays, Eg=arrays.Eg + 0.05))
    monkeypatch.setenv("SOLARLAB_AUTOLOOP_GEN", "1")
    _, a_on = _build(dataclasses.replace(base, autoloop_generated_lever=False))  # off on stack, on via env
    assert np.allclose(a_on.Eg, a_off.Eg + 0.05)
