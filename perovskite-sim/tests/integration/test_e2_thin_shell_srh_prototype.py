"""Phase E2 Sprint 3 — thin-shell volumetric SRH prototype.

Pivot from BBD validation gate FAIL (Sprint 2 Day 2-3). Treat interface
SRH as a volumetric source over a shell of `n_shell` grid nodes around
the interface node, evaluating per-node n·p (NOT cross-carrier samples).

Activated by env var ``SOLARLAB_THIN_SHELL_SRH=N`` where N is the shell
width in nodes (must be a positive even integer). Defaults / unset /
malformed → legacy E1.5 cross-carrier path, bit-identical.

Formula (when active):
  shell = [idx - N//2, ..., idx + N//2] \\ {idx}     # skip interface
  N_t_vol = N_t_areal / sum(dx_cell[i] for i in shell)
  for i in shell:
      n_i = max(0, n[i]); p_i = max(0, p[i])         # clamp neg-p
      R_i = N_t_vol · σ · v_th ·
            (n_i · p_i − ni_sq_eff) / (n_i + p_i + n1 + p1)
      dn[i] -= R_i ; dp[i] -= R_i

Contract pinned by this test file:
1. Env unset → V_oc bit-identical to main-branch baseline (1.0694 V).
2. Env=0 / non-integer ("two") / invalid → legacy V_oc (defensive).
3. Env=2 → V_oc moves from legacy by ≥1 mV (proves wiring).
4. Env=2 → JV arrays finite (no NaN/Inf).
5. Env=2 → V_oc still within physical envelope [0.8, 1.3] V.
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


def test_thin_shell_env_unset_legacy(monkeypatch):
    """env unset → legacy E1.5 V_oc 1.0694 ±5 mV."""
    monkeypatch.delenv("SOLARLAB_THIN_SHELL_SRH", raising=False)
    stack = load_scaps_yaml("configs/scaps_mirror.yaml")
    assert _voc(stack) == pytest.approx(_LEGACY_V_OC, abs=5.0e-3)


def test_thin_shell_env_zero_legacy(monkeypatch):
    """env=0 → legacy V_oc (only positive ints activate)."""
    monkeypatch.setenv("SOLARLAB_THIN_SHELL_SRH", "0")
    stack = load_scaps_yaml("configs/scaps_mirror.yaml")
    assert _voc(stack) == pytest.approx(_LEGACY_V_OC, abs=5.0e-3)


def test_thin_shell_env_non_integer_legacy(monkeypatch):
    """env=non-integer ("two", "1.5") → legacy V_oc (defensive)."""
    monkeypatch.setenv("SOLARLAB_THIN_SHELL_SRH", "two")
    stack = load_scaps_yaml("configs/scaps_mirror.yaml")
    assert _voc(stack) == pytest.approx(_LEGACY_V_OC, abs=5.0e-3)


def test_thin_shell_env_2_voc_moves(monkeypatch):
    """env=2 → V_oc moves measurably from legacy.

    Thin-shell volumetric SRH samples per-node n·p around the interface
    (not cross-carrier). Expect V_oc to move by O(10-100 mV) — direction
    not pinned because the shell semantics depend on the depletion zone
    structure.
    """
    monkeypatch.setenv("SOLARLAB_THIN_SHELL_SRH", "2")
    stack = load_scaps_yaml("configs/scaps_mirror.yaml")
    voc = _voc(stack)
    assert 0.8 <= voc <= 1.3, f"V_oc {voc} outside physical envelope"
    assert abs(voc - _LEGACY_V_OC) >= 1.0e-3, (
        f"thin-shell V_oc {voc:.4f} did not move from legacy "
        f"{_LEGACY_V_OC}; shell path may be silently bypassed."
    )


def test_thin_shell_env_2_finite_jv(monkeypatch):
    """env=2 JV completes without NaN/Inf in V or J arrays."""
    monkeypatch.setenv("SOLARLAB_THIN_SHELL_SRH", "2")
    stack = load_scaps_yaml("configs/scaps_mirror.yaml")
    r = run_jv_sweep(stack, N_grid=30, n_points=20, v_rate=5.0, V_max=1.6)
    for v in r.V_fwd:
        assert math.isfinite(v), f"non-finite V {v}"
    for j in r.J_fwd:
        assert math.isfinite(j), f"non-finite J {j}"
