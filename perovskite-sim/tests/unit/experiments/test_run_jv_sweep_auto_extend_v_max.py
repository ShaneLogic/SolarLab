"""Phase E1.9 — ``run_jv_sweep`` auto-extends V_max when bracket fails.

Sprint 1a (2026-05-25) revealed that with Robin contacts enabled and low
ETL donor doping (N_D ≤ 1e12 cm⁻³), V_oc can climb above the default
V_max=1.6 V and the sweep reports ``voc_bracketed=False`` with sentinel
zero V_oc/FF/PCE. The user has no signal beyond the boolean flag, and
no automatic recovery.

E1.9 adds an opt-in ``v_max_max_attempts`` kwarg: when the first sweep
fails to bracket, retry with V_max bumped by 0.5 V per attempt (capped
at V_initial + 2.0 V). Conservative cost (1-2 extra sweeps),
predictable upper bound. Default value 1 preserves the legacy
no-retry behaviour bit-identically.

Contract:
1. ``v_max_max_attempts=1`` (default) → no retry; failed bracket
   produces sentinel zeros (current behaviour).
2. ``v_max_max_attempts=N`` (N > 1) → on failed bracket, retry up to
   N-1 times with V_max += 0.5 per attempt.
3. Successful first attempt → no retry regardless of attempts setting.
4. Exhausted attempts → still returns ``voc_bracketed=False`` (no
   exception); caller can inspect.
5. Legacy bit-identity: callers that pass default attempts get identical
   numerical results to pre-E1.9.
"""
from __future__ import annotations

import dataclasses

import pytest

from perovskite_sim.experiments.jv_sweep import run_jv_sweep
from perovskite_sim.scaps_compat import load_scaps_yaml


def _scaps_mirror_robin_low_etl():
    """scaps_mirror.yaml with Robin S_n_right=1e5, low ETL doping → V_oc
    climbs above V_max=1.6 V (per Sprint 1a probe: V_oc ≈ 1.64 V at
    N_D_ETL=1e12 cm⁻³)."""
    base = load_scaps_yaml("configs/scaps_mirror.yaml")
    robin = dataclasses.replace(
        base, mode="full",
        S_n_right=1.0e5, S_p_right=1.0e-4,
        S_p_left=1.0e5, S_n_left=1.0e-4,
    )
    # Replace ETL params with N_D=1e12 cm⁻³ = 1e18 m⁻³.
    layers = list(robin.layers)
    etl = layers[-1]
    etl_params = dataclasses.replace(etl.params, N_D=1.0e18)
    layers[-1] = dataclasses.replace(etl, params=etl_params)
    return dataclasses.replace(robin, layers=tuple(layers))


def test_default_attempts_preserves_legacy_no_retry_behaviour():
    """``v_max_max_attempts=1`` (default) on a stack that fails to bracket
    at V_max=1.6 returns sentinel zeros — bit-identical to pre-E1.9."""
    stack = _scaps_mirror_robin_low_etl()
    r = run_jv_sweep(stack, N_grid=30, n_points=20, v_rate=5.0, V_max=1.6)
    assert r.metrics_fwd.voc_bracketed is False
    assert r.metrics_fwd.V_oc == 0.0  # sentinel


def test_auto_extend_v_max_succeeds_on_second_attempt():
    """``v_max_max_attempts=2`` retries once with V_max=2.1 → succeeds.

    Per Sprint 1a probe, V_oc at this stack is ~1.64 V. V_max=1.6 fails;
    V_max=1.6 + 0.5 = 2.1 succeeds."""
    stack = _scaps_mirror_robin_low_etl()
    r = run_jv_sweep(
        stack, N_grid=30, n_points=20, v_rate=5.0,
        V_max=1.6, v_max_max_attempts=2,
    )
    assert r.metrics_fwd.voc_bracketed is True
    assert 1.5 < r.metrics_fwd.V_oc < 2.1, (
        f"V_oc={r.metrics_fwd.V_oc:.4f} V outside expected [1.5, 2.1] range "
        "for Robin scaps_mirror at N_D_ETL=1e12 cm⁻³"
    )


def test_already_bracketed_first_attempt_does_not_retry():
    """When the first sweep already brackets V_oc, ``v_max_max_attempts``
    has no effect — same result whether attempts=1 or attempts=5."""
    stack = load_scaps_yaml("configs/scaps_mirror.yaml")
    r1 = run_jv_sweep(stack, N_grid=30, n_points=20, v_rate=5.0, V_max=1.6)
    r2 = run_jv_sweep(
        stack, N_grid=30, n_points=20, v_rate=5.0,
        V_max=1.6, v_max_max_attempts=5,
    )
    assert r1.metrics_fwd.voc_bracketed is True
    assert r2.metrics_fwd.voc_bracketed is True
    # Identical V_oc — no retry happened so the second call's sweep is
    # bit-identical to the first.
    assert r1.metrics_fwd.V_oc == pytest.approx(r2.metrics_fwd.V_oc)


def test_exhausted_attempts_returns_unbracketed_no_exception():
    """When all attempts exhaust without bracketing (e.g. V_oc above
    V_initial + 2.0 V cap), return ``voc_bracketed=False`` without
    raising — caller inspects the flag, never an exception."""
    stack = _scaps_mirror_robin_low_etl()
    # Force a too-low V_max + too-few attempts. Per probe, this stack's
    # V_oc is ~1.64 V. Start at V_max=1.0 with 1 retry → tries 1.0 and
    # 1.5, both below V_oc → unbracketed result expected (no exception).
    r = run_jv_sweep(
        stack, N_grid=30, n_points=20, v_rate=5.0,
        V_max=1.0, v_max_max_attempts=2,
    )
    assert r.metrics_fwd.voc_bracketed is False
    assert r.metrics_fwd.V_oc == 0.0
