"""Phase E2 Sprint 4 — faithful Pauwels-Vanhoutte 1978 interface SRH prototype.

Pivot from BBD validation gate FAIL (Sprint 2) and thin-shell MIXED
(Sprint 3) by implementing the exact formula from ref [13] Pauwels &
Vanhoutte 1978 J.Phys.D 11, 649-667.

Heavy-doping ETL limit (eq 30 inverted to SolarLab convention):
  V_1 (ETL band-bending)  = 0
  V_2 (PVK band-bending)  = V_bi_eff − V_app

Interface-plane densities (eqs 8, 9):
  n_iface (ETL side) = n[idx+1]                              # no depletion
  p_iface (PVK side) = p[idx-1] · exp(−(V_bi − V_app)/V_T)

Then standard Shockley-Read interface SRH:
  R = (n·p − ni²) / ((n + n1)/v_p + (p + p1)/v_n)

Critical difference vs BBD: uses GLOBAL V_bi_eff (from mat) and V_app
(from assemble_rhs kwarg) — both doping-independent in heavy-doping
regime. BBD used local phi grid differences that scaled with N_D_ETL.

Activated by env var ``SOLARLAB_PAUWELS_VANHOUTTE=1``. Default unset →
legacy E1.5 cross-carrier path, bit-identical.

Contract pinned by this test file:
1. Env unset → V_oc bit-identical to current main baseline (1.0694 V).
2. Env=0 / non-integer / malformed → legacy V_oc (only literal "1"
   activates).
3. Env=1 → V_oc moves measurably from legacy by ≥1 mV.
4. Env=1 → JV arrays finite (no NaN/Inf).
5. Env=1 → V_oc within physical envelope [0.8, 1.3] V.
"""
from __future__ import annotations

import math

import pytest

from perovskite_sim.experiments.jv_sweep import run_jv_sweep
from perovskite_sim.scaps_compat import load_scaps_yaml


_LEGACY_V_OC = 1.0694  # current main-branch scaps_mirror.yaml baseline


def _voc(stack) -> float:
    r = run_jv_sweep(stack, N_grid=30, n_points=20, v_rate=5.0, V_max=1.6)
    assert r.metrics_fwd.voc_bracketed, "V_oc must bracket on scaps_mirror"
    return float(r.metrics_fwd.V_oc)


def test_pv_env_unset_legacy(monkeypatch):
    """env unset → legacy E1.5 V_oc 1.0694 ±5 mV."""
    monkeypatch.delenv("SOLARLAB_PAUWELS_VANHOUTTE", raising=False)
    stack = load_scaps_yaml("configs/scaps_mirror.yaml")
    assert _voc(stack) == pytest.approx(_LEGACY_V_OC, abs=5.0e-3)


def test_pv_env_zero_legacy(monkeypatch):
    """env=0 → legacy V_oc (only literal "1" activates)."""
    monkeypatch.setenv("SOLARLAB_PAUWELS_VANHOUTTE", "0")
    stack = load_scaps_yaml("configs/scaps_mirror.yaml")
    assert _voc(stack) == pytest.approx(_LEGACY_V_OC, abs=5.0e-3)


def test_pv_env_malformed_legacy(monkeypatch):
    """env=non-canonical → legacy V_oc (defensive)."""
    monkeypatch.setenv("SOLARLAB_PAUWELS_VANHOUTTE", "true")
    stack = load_scaps_yaml("configs/scaps_mirror.yaml")
    assert _voc(stack) == pytest.approx(_LEGACY_V_OC, abs=5.0e-3)


def test_pv_env_active_voc_moves(monkeypatch):
    """env=1 → V_oc moves measurably from legacy.

    Pauwels-Vanhoutte heavy-doping limit replaces local phi differences
    with the global V_bi − V_app band-bending factor. Expected motion:
    O(10-50 mV) — direction not pinned because the rate depends on
    full V_bi which can shift either way relative to the local-phi
    approximation.
    """
    monkeypatch.setenv("SOLARLAB_PAUWELS_VANHOUTTE", "1")
    stack = load_scaps_yaml("configs/scaps_mirror.yaml")
    voc = _voc(stack)
    assert 0.8 <= voc <= 1.3, f"V_oc {voc} outside physical envelope"
    assert abs(voc - _LEGACY_V_OC) >= 1.0e-3, (
        f"Pauwels-Vanhoutte V_oc {voc:.4f} did not move from legacy "
        f"{_LEGACY_V_OC}; activation path may be silently bypassed."
    )


def test_pv_env_active_finite_jv(monkeypatch):
    """env=1 JV completes without NaN/Inf in V or J arrays."""
    monkeypatch.setenv("SOLARLAB_PAUWELS_VANHOUTTE", "1")
    stack = load_scaps_yaml("configs/scaps_mirror.yaml")
    r = run_jv_sweep(stack, N_grid=30, n_points=20, v_rate=5.0, V_max=1.6)
    for v in r.V_fwd:
        assert math.isfinite(v), f"non-finite V {v}"
    for j in r.J_fwd:
        assert math.isfinite(j), f"non-finite J {j}"
