"""Tests for `experiments/dark_jv.py` — diode-region ideality + J_0 fit.

The existing test file `test_dark_jv.py` covers the dark-mode switch in
`run_jv_sweep(illuminated=False)` — J_sc ≈ 0, injection at forward bias,
etc. This file covers the new `run_dark_jv` wrapper that adds the
post-processing diode fit (ideality factor, saturation current).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from perovskite_sim.constants import V_T
from perovskite_sim.experiments.dark_jv import (
    DarkJVResult,
    _fit_ideality_and_J0,
    _select_diode_window,
    run_dark_jv,
)
from perovskite_sim.models.config_loader import load_device_from_yaml


@pytest.fixture(scope="module")
def nip_stack():
    return load_device_from_yaml("configs/nip_MAPbI3.yaml")


# ---------------------------------------------------------------------------
# Pure-math unit tests on the fit helpers — fast, no solver.
# ---------------------------------------------------------------------------

def test_fit_recovers_known_ideality_and_J0():
    """Synthetic diode (no -1 term) must round-trip through the fit.

    Above the diode knee (V >> n·V_T) the Shockley current is
    J(V) ≈ J_0 · exp(V/(n·V_T)) — a pure exponential, straight in
    log-space. The fitter must recover n and log10(J_0) on a curve of
    this shape, since this is exactly the regime the auto-window
    selector is supposed to find in real sweeps.
    """
    n_true = 1.6
    J_0_true = 1e-4
    V = np.linspace(0.3, 1.0, 50)  # diode regime only
    absJ = J_0_true * np.exp(V / (n_true * V_T))
    J = -absJ  # injection-sign convention matches run_jv_sweep

    n_fit, J_0_fit, V_lo, V_hi = _fit_ideality_and_J0(V, J)
    assert abs(n_fit - n_true) / n_true < 0.02, (
        f"ideality fit {n_fit:.4f} off from {n_true}"
    )
    assert abs(np.log10(J_0_fit) - np.log10(J_0_true)) < 0.05, (
        f"log10(J_0) fit {np.log10(J_0_fit):.3f} off from {np.log10(J_0_true):.3f}"
    )
    assert 0.3 <= V_lo < V_hi <= 1.0
    # On a straight-line log|J|, the window selector should land on the
    # full input range (all points are equally "straight"). Any window
    # narrower than half the input signals that the growth threshold is
    # too tight.
    assert V_hi - V_lo > 0.5, (
        f"Fit window too narrow on a pure exponential: [{V_lo:.3f}, {V_hi:.3f}]"
    )


def test_fit_window_picks_linear_region_over_rolloff():
    """Fit window should skip a series-resistance roll-off tail.

    Construct a piecewise curve: linear-in-log for V in [0.3, 0.8] (ideal
    diode), then a resistive kink for V > 0.8 V where log|J| bends over.
    The window selector must pick the [0.3, 0.8] straight-line region,
    not the kinked tail.
    """
    V = np.linspace(0.0, 1.1, 45)
    n_true = 1.3
    J_0 = 1e-1
    logJ_ideal = np.log(J_0) + V / (n_true * V_T)
    # Kink at V = 0.8: above that, slope drops by 3x (series resistance)
    slope_ideal = 1.0 / (n_true * V_T)
    slope_kink = slope_ideal / 3.0
    V_knee = 0.8
    mask = V > V_knee
    logJ = logJ_ideal.copy()
    logJ[mask] = logJ_ideal[V <= V_knee][-1] + slope_kink * (V[mask] - V_knee)
    absJ = np.exp(logJ)
    absJ[V < 0.1] = J_0 * 1e-4  # push below J_floor
    J = -absJ  # injection sign convention

    i_lo, i_hi = _select_diode_window(V, J)
    # The selected window should land in the straight-line region,
    # not the post-knee section.
    V_lo_fit = V[i_lo]
    V_hi_fit = V[i_hi]
    assert V_lo_fit >= 0.1, f"window started too low: V_lo={V_lo_fit:.3f}"
    assert V_hi_fit <= V_knee + 0.05, (
        f"window ran into resistive tail: V_hi={V_hi_fit:.3f}, knee at {V_knee}"
    )


def test_fit_handles_negative_J_convention():
    """The fitter must take |J|; passing J<0 (run_jv_sweep sign) must work."""
    V = np.linspace(0.0, 1.0, 41)
    n_true = 1.8
    J_0 = 1e-5
    J_negative_convention = -J_0 * (np.exp(V / (n_true * V_T)) - 1.0)
    n_fit, J_0_fit, _, _ = _fit_ideality_and_J0(V, J_negative_convention)
    assert abs(n_fit - n_true) / n_true < 0.02


# ---------------------------------------------------------------------------
# Integration: actually run a dark sweep on a small grid.
# ---------------------------------------------------------------------------

def test_run_dark_jv_returns_populated_dataclass(nip_stack):
    """Smoke test: dataclass fields are populated, shapes consistent."""
    r = run_dark_jv(nip_stack, V_max=1.0, N_grid=30, n_points=20, v_rate=2.0)
    assert isinstance(r, DarkJVResult)
    assert r.V.shape == r.J.shape == (20,)
    assert np.all(np.isfinite(r.V))
    assert np.all(np.isfinite(r.J))
    assert np.isfinite(r.n_ideality) and r.n_ideality > 0.0
    assert np.isfinite(r.J_0) and r.J_0 > 0.0
    assert 0.0 <= r.V_fit_lo < r.V_fit_hi <= 1.0


def test_run_dark_jv_ideality_in_physical_range(nip_stack):
    """nip_MAPbI3 is SRH-dominated; ideality should be in [1.0, 2.5]."""
    r = run_dark_jv(nip_stack, V_max=1.0, N_grid=30, n_points=25, v_rate=2.0)
    assert 1.0 <= r.n_ideality <= 2.5, (
        f"nip_MAPbI3 dark ideality {r.n_ideality:.3f} outside physical [1.0, 2.5]"
    )


def test_run_dark_jv_fit_matches_shockley_in_window(nip_stack):
    """Predicted J from (n, J_0) must match |J| in the fit window to ~30%."""
    r = run_dark_jv(nip_stack, V_max=1.0, N_grid=30, n_points=30, v_rate=2.0)
    mask = (r.V >= r.V_fit_lo) & (r.V <= r.V_fit_hi)
    V_win = r.V[mask]
    absJ_win = np.abs(r.J[mask])
    # Shockley prediction from the extracted parameters
    J_pred = r.J_0 * (np.exp(V_win / (r.n_ideality * V_T)) - 1.0)
    # Relative error averaged in log-space to avoid the V→0 corner
    # dominating the norm.
    rel_err = np.abs(np.log(absJ_win + 1e-30) - np.log(J_pred + 1e-30))
    # Within the fit window, mean residual in log-space < 0.3 → factor ~1.35.
    assert float(np.mean(rel_err)) < 0.3, (
        f"dark fit does not track |J| in the diode window: mean log-resid="
        f"{np.mean(rel_err):.3f}"
    )
