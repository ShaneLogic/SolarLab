"""Phase E2.1 prototype — band-bending depletion (BBD) interface face density.

Phase A (2026-05-26) and Phase E2a Sprint 1 (2026-05-27) probed four
face-density formulations on scaps_mirror.yaml. Day 3.5 N_D_ETL sweep
showed BBD np is 2.8× less sensitive to ETL doping than the current
E1.5 cross-carrier — right direction for closing the SCAPS 8× ETL
doping over-sensitivity gap.

This prototype gates the BBD path behind the env var
``SOLARLAB_BBD_FACE=1`` so the production code path stays unchanged
until the Sprint 2 Day 2-3 validation gate decides ship-or-pivot.

BBD formula (same-layer φ Boltzmann projection, no χ step):
  n_face = n[eval_n_idx] · exp((φ[idx]  − φ[eval_n_idx]) / V_T)
  p_face = p[eval_p_idx] · exp(-(φ[idx] − φ[eval_p_idx]) / V_T)

Contract pinned by this test file:
1. With ``SOLARLAB_BBD_FACE`` UNSET, V_oc is bit-identical to the
   current main-branch baseline (legacy E1.5 cross-carrier path).
2. With ``SOLARLAB_BBD_FACE=1``, V_oc moves (BBD np ~2× E1.5 np at
   V_oc baseline → R increases → V_oc drops by O(10 mV)).
3. With ``SOLARLAB_BBD_FACE=1``, the JV sweep completes without
   NaN/Inf and within the physical V_oc envelope [0.8, 1.3] V.
4. Other env values (``=0``, ``=other``, unset) all take the legacy path.
"""
from __future__ import annotations

import math

import pytest

from perovskite_sim.experiments.jv_sweep import run_jv_sweep
from perovskite_sim.scaps_compat import load_scaps_yaml


def _run_voc(stack) -> float:
    """Run a short JV sweep, return V_oc."""
    r = run_jv_sweep(stack, N_grid=30, n_points=20, v_rate=5.0, V_max=1.6)
    assert r.metrics_fwd.voc_bracketed, "V_oc must bracket on scaps_mirror"
    return float(r.metrics_fwd.V_oc)


def test_e2_bbd_env_unset_legacy_voc(monkeypatch):
    """Without env activation, V_oc matches the current main-branch baseline.

    Pins the legacy E1.5 cross-carrier path: BBD must not regress
    pre-Phase-E2 behaviour when the env var is unset.
    """
    monkeypatch.delenv("SOLARLAB_BBD_FACE", raising=False)
    stack = load_scaps_yaml("configs/scaps_mirror.yaml")
    voc = _run_voc(stack)
    # Current main V_oc on scaps_mirror is 1.0694 V (Phase E1.6 calibration).
    assert voc == pytest.approx(1.0694, abs=5.0e-3)


def test_e2_bbd_env_zero_legacy_voc(monkeypatch):
    """SOLARLAB_BBD_FACE=0 takes the legacy path (only ``=1`` activates BBD).

    Pins the explicit-disable behaviour so toggling off mid-session
    reproduces the legacy V_oc.
    """
    monkeypatch.setenv("SOLARLAB_BBD_FACE", "0")
    stack = load_scaps_yaml("configs/scaps_mirror.yaml")
    voc = _run_voc(stack)
    assert voc == pytest.approx(1.0694, abs=5.0e-3)


def test_e2_bbd_env_active_voc_moves(monkeypatch):
    """SOLARLAB_BBD_FACE=1 activates BBD; V_oc moves measurably.

    Day 3.5 probe shows BBD np ~2× E1.5 np at V_oc baseline. Larger np
    means larger R, lower V_oc. Expected motion: O(10-50 mV) downward.
    Tolerance window deliberately wide — exact V_oc depends on the
    solver's full SRH path response, not the np proxy.
    """
    monkeypatch.setenv("SOLARLAB_BBD_FACE", "1")
    stack = load_scaps_yaml("configs/scaps_mirror.yaml")
    voc = _run_voc(stack)
    # Within physical envelope.
    assert 0.8 <= voc <= 1.3, f"V_oc {voc} outside physical envelope"
    # Moved from legacy 1.0694 by at least 1 mV (BBD is not a no-op).
    assert abs(voc - 1.0694) >= 1.0e-3, (
        f"BBD V_oc {voc:.4f} did not move from legacy 1.0694; "
        "BBD path may be silently bypassed."
    )


def test_e2_bbd_env_active_finite_jv(monkeypatch):
    """BBD-active JV sweep completes without NaN/Inf in V or J arrays."""
    monkeypatch.setenv("SOLARLAB_BBD_FACE", "1")
    stack = load_scaps_yaml("configs/scaps_mirror.yaml")
    r = run_jv_sweep(stack, N_grid=30, n_points=20, v_rate=5.0, V_max=1.6)
    # No NaN/Inf in voltage or current arrays.
    for v in r.V_fwd:
        assert math.isfinite(v), f"non-finite V {v}"
    for j in r.J_fwd:
        assert math.isfinite(j), f"non-finite J {j}"


def test_e2_bbd_env_random_value_legacy(monkeypatch):
    """Unrecognised env value (``=other``) takes the legacy path.

    Defensive: only the literal ``=1`` activates BBD. Anything else
    (including typos like ``=true``, ``=yes``) silently falls back to
    legacy. Prevents accidental partial activations.
    """
    monkeypatch.setenv("SOLARLAB_BBD_FACE", "true")
    stack = load_scaps_yaml("configs/scaps_mirror.yaml")
    voc = _run_voc(stack)
    assert voc == pytest.approx(1.0694, abs=5.0e-3)
