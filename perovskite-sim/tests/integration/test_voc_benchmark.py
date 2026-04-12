"""
Integration tests: V_oc, J_sc, FF validation against IonMonger benchmark.

Our model includes thermionic emission (TE) capping at heterointerfaces,
which IonMonger does not. This physically reduces minority carrier injection
across band offsets, lowering interface recombination and raising V_oc by
~0.1 V relative to IonMonger's ~1.07 V. The bounds below accommodate both
the TE-corrected physics (our model) and leave room for future tuning.

Run with: pytest -m slow
"""
import pytest
import numpy as np
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.experiments.jv_sweep import run_jv_sweep

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def ionmonger_result():
    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    return run_jv_sweep(stack, N_grid=40, n_points=20, v_rate=5.0)


def test_voc_ionmonger_benchmark(ionmonger_result):
    """V_oc should be in [0.95, 1.25] V.

    IonMonger predicts ~1.07 V; our TE-corrected model gives ~1.18 V.
    Both are within experimental MAPbI3 range (0.9-1.2 V).
    """
    V_oc = ionmonger_result.metrics_rev.V_oc
    assert 0.95 < V_oc < 1.25, f"V_oc = {V_oc:.4f} V outside expected range"


def test_jsc_ionmonger_benchmark(ionmonger_result):
    """J_sc should be in [150, 300] A/m² (15-30 mA/cm²).

    IonMonger predicts ~220 A/m²; our model should be close since J_sc
    is mainly determined by generation and collection, not V_oc physics.
    """
    J_sc = ionmonger_result.metrics_rev.J_sc
    assert 150.0 < J_sc < 300.0, f"J_sc = {J_sc:.1f} A/m² outside expected range"


def test_ff_ionmonger_benchmark(ionmonger_result):
    """FF should exceed 0.55 for a functional perovskite device."""
    FF = ionmonger_result.metrics_rev.FF
    assert FF > 0.55, f"FF = {FF:.3f} below minimum threshold"


def test_legacy_config_no_regression():
    """nip_MAPbI3 (no chi/Eg) should still produce reasonable V_oc.

    Legacy configs have chi=Eg=0 so TE capping does not activate and
    compute_V_bi falls back to the manual V_bi field. This test guards
    against regressions in the legacy code path.
    """
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    result = run_jv_sweep(stack, N_grid=30, n_points=8, v_rate=5.0)
    m = result.metrics_fwd
    assert 0.7 < m.V_oc < 1.2, f"legacy V_oc = {m.V_oc:.4f} V"
    assert 100.0 < m.J_sc < 800.0, f"legacy J_sc = {m.J_sc:.1f} A/m²"
    assert m.FF > 0.4, f"legacy FF = {m.FF:.3f}"
