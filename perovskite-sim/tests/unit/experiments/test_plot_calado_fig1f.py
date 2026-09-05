"""Unit contracts for the pure helpers in scripts/plot_calado_fig1f.py.

The solver run itself (two ~2 min paper-protocol scans) is not exercised here;
these tests pin the metric extraction, the hysteresis-index definitions, the
control-stack derivation and the uniform-generation profile the script uses.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from perovskite_sim.models.config_loader import load_device_from_yaml

Q_E = 1.602176634e-19


def _load_module():
    path = Path("scripts/plot_calado_fig1f.py")
    spec = importlib.util.spec_from_file_location(
        "plot_calado_fig1f_test_module", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve the module by name
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load_module()


@pytest.fixture(scope="module")
def stack():
    return load_device_from_yaml("configs/calado2016_fig1f.yaml")


def _diode(V: np.ndarray, J_sc: float = 160.0, V_oc: float = 0.7) -> np.ndarray:
    """Exponential diode in the active-cell sign convention (J > 0 at V = 0)."""
    return J_sc - J_sc * np.expm1(V / 0.05) / np.expm1(V_oc / 0.05)


def test_branch_metrics_recover_jsc_voc_pmax(mod):
    V = np.linspace(-1.0, 1.2, 221)
    J = _diode(V)
    m = mod.branch_metrics(V, J)
    assert m.J_sc == pytest.approx(160.0, abs=1e-9)
    assert m.V_oc == pytest.approx(0.7, abs=0.011)  # linear interp on a 10 mV grid
    P = V * J
    assert m.P_max == pytest.approx(P[V > 0].max())
    assert 0.0 < m.V_mp < 0.7
    assert m.FF == pytest.approx(m.P_max / (m.J_sc * m.V_oc))


def test_branch_metrics_is_order_independent(mod):
    V = np.linspace(1.2, -1.0, 221)
    assert mod.branch_metrics(V, _diode(V)) == mod.branch_metrics(
        V[::-1], _diode(V[::-1])
    )


def test_branch_metrics_skips_failed_points(mod):
    V = np.linspace(-1.0, 1.2, 221)
    J = _diode(V)
    J[[5, 150]] = np.nan
    m = mod.branch_metrics(V, J)
    assert np.isfinite(m.P_max) and np.isfinite(m.V_oc) and np.isfinite(m.J_sc)


def test_hysteresis_index_definitions(mod):
    assert mod.hysteresis_index_paper(p_fwd=50.0, p_rev=100.0) == pytest.approx(1.0)
    assert mod.hysteresis_index_solarlab(p_fwd=50.0, p_rev=100.0) == pytest.approx(0.5)


def test_control_stack_removes_contact_srh_only(mod, stack):
    control = mod.control_stack(stack)
    for layer, ctrl in zip(stack.layers, control.layers):
        if layer.role == "absorber":
            assert ctrl.params == layer.params
        else:
            assert ctrl.params.tau_n == mod.CONTROL_TAU_S
            assert ctrl.params.tau_p == mod.CONTROL_TAU_S
            assert ctrl.params.n1 == layer.params.n1
            assert ctrl.params.p1 == layer.params.p1
    assert stack.layers[0].params.tau_n == 2e-15  # source stack untouched
    assert control.V_bi == stack.V_bi


def test_uniform_generation_fills_absorber_only(mod, stack):
    x = np.linspace(0.0, 800e-9, 801)
    G = mod.uniform_generation(x, stack)
    inside = (x >= 200e-9) & (x <= 600e-9)
    assert np.all(G[inside] == mod.G_UNIFORM)
    assert np.all(G[~inside] == 0.0)
    # SI Table 1: 2.5e21 cm^-3 s^-1 over 400 nm -> ~16 mA/cm^2 (160 A/m^2)
    assert Q_E * mod.G_UNIFORM * 400e-9 == pytest.approx(160.0, rel=0.01)
