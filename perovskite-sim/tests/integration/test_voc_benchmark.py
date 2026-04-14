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


# ── Reference metrics (TE-corrected, N_grid=40, n_points=20, v_rate=5.0) ────
# Captured from a clean run against configs/ionmonger_benchmark.yaml with the
# band-offset V_bi and thermionic-emission capping landed in Phase 1. These
# tight bounds (±2% V_oc, ±3% J_sc/FF, ±5% PCE) are the deeper-benchmark gate:
# if the physics drifts under a future change, these tests fail immediately
# rather than waiting for a loose guardrail to flag it. Update the reference
# ONLY when a physics change is deliberate and documented in the commit body.
_REF_V_OC = 1.1932
_REF_J_SC = 231.70
_REF_FF = 0.7774
_REF_PCE = 0.2149


def _assert_within(value: float, target: float, rel: float, name: str) -> None:
    lo = target * (1.0 - rel)
    hi = target * (1.0 + rel)
    assert lo <= value <= hi, (
        f"{name} = {value:.4f} outside [{lo:.4f}, {hi:.4f}] "
        f"(target {target:.4f} ± {rel * 100:.1f}%)"
    )


def test_voc_ionmonger_benchmark(ionmonger_result):
    """V_oc (reverse scan) must be within ±2% of the Phase 1 reference.

    Reference: 1.1932 V at N_grid=40, n_points=20, v_rate=5.0 V/s with
    band-offset V_bi + thermionic-emission capping. IonMonger's own value
    is ~1.07 V; our TE-corrected ~1.19 V sits inside the experimental
    MAPbI3 window (0.9-1.2 V).
    """
    _assert_within(ionmonger_result.metrics_rev.V_oc, _REF_V_OC, 0.02, "V_oc_rev")


def test_jsc_ionmonger_benchmark(ionmonger_result):
    """J_sc must be within ±3% of the Phase 1 reference (~231.7 A/m²)."""
    _assert_within(ionmonger_result.metrics_rev.J_sc, _REF_J_SC, 0.03, "J_sc_rev")


def test_ff_ionmonger_benchmark(ionmonger_result):
    """FF must be within ±3% of the Phase 1 reference (~0.777)."""
    _assert_within(ionmonger_result.metrics_rev.FF, _REF_FF, 0.03, "FF_rev")


def test_pce_ionmonger_benchmark(ionmonger_result):
    """PCE must be within ±5% of the Phase 1 reference (~0.2149, i.e. 21.5%).

    PCE is the most sensitive metric because it compounds V_oc·J_sc·FF
    drift — a 2% V_oc slip and a 3% J_sc slip alone can move PCE ~5%.
    """
    _assert_within(ionmonger_result.metrics_rev.PCE, _REF_PCE, 0.05, "PCE_rev")


def test_forward_reverse_consistency(ionmonger_result):
    """Forward and reverse V_oc must agree within 5 mV — the ionmonger stack
    has weak ionic hysteresis, so a larger split means the solver took a
    wrong branch on one of the sweeps.
    """
    delta = abs(
        ionmonger_result.metrics_fwd.V_oc - ionmonger_result.metrics_rev.V_oc
    )
    assert delta < 5e-3, f"|V_oc_fwd - V_oc_rev| = {delta * 1e3:.2f} mV (>5 mV)"


def test_hysteresis_index_bounded(ionmonger_result):
    """|HI| must be under 0.05 — the ionmonger benchmark is a weak-hysteresis
    case; anything larger indicates unphysical current spikes in the sweep.
    """
    assert abs(ionmonger_result.hysteresis_index) < 0.05, (
        f"|HI| = {abs(ionmonger_result.hysteresis_index):.4f} (>0.05)"
    )


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
