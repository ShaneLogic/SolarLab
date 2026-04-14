"""Tests for perovskite_sim.experiments.tandem_jv.

Step 2/7 — written before the implementation to drive TDD.

The series_match_jv tests are pure unit tests (no physics, fast).
The run_tandem_jv smoke test uses real sub-cell sweeps but keeps
N_grid=30 and n_points=12 to minimise wall time.
"""
from __future__ import annotations

import numpy as np
import pytest

from perovskite_sim.experiments.tandem_jv import series_match_jv


# ---------------------------------------------------------------------------
# series_match_jv — pure unit tests
# ---------------------------------------------------------------------------

def test_series_match_sums_voltages_at_matched_current():
    # Sub-cell 1: V = 1.0 - 0.01 * J (V_oc=1.0 at J=0, slope -0.01)
    # Sub-cell 2: V = 0.8 - 0.005 * J (V_oc=0.8 at J=0, slope -0.005)
    J_top = np.linspace(-50, 0, 51)
    V_top = 1.0 - 0.01 * J_top
    J_bot = np.linspace(-50, 0, 51)
    V_bot = 0.8 - 0.005 * J_bot

    J_common, V_top_m, V_bot_m, V_tandem = series_match_jv(
        J_top, V_top, J_bot, V_bot, V_junction=0.0,
    )

    idx_zero = np.argmin(np.abs(J_common))
    assert V_tandem[idx_zero] == pytest.approx(1.8, abs=1e-6)
    # At J=-50 (most negative): V_top=1.5, V_bot=1.05 → V_tandem=2.55
    assert V_tandem[0] == pytest.approx(2.55, abs=1e-6)
    # J-V curve must be monotonically non-increasing
    assert np.all(np.diff(V_tandem) <= 1e-9)


def test_series_match_is_limited_by_smaller_jsc():
    J_top = np.linspace(-20, 0, 41)
    V_top = 1.0 + 0.02 * J_top
    J_bot = np.linspace(-30, 0, 61)
    V_bot = 0.8 + 0.015 * J_bot

    J_common, _, _, V_tandem = series_match_jv(J_top, V_top, J_bot, V_bot)

    # Overlap limited to [-20, 0], so minimum of J_common is -20
    assert J_common[0] == pytest.approx(-20.0, abs=1e-9)
    idx_zero = np.argmin(np.abs(J_common))
    assert V_tandem[idx_zero] == pytest.approx(1.8, abs=1e-6)


def test_series_match_v_junction_offset_applied():
    J = np.linspace(-10, 0, 11)
    V = np.ones_like(J)          # flat V(J) = 1.0 for both sub-cells

    _, _, _, V_tandem = series_match_jv(J, V, J, V, V_junction=0.05)

    assert np.allclose(V_tandem, 2.05)


def test_series_match_non_overlapping_raises():
    J_top = np.linspace(-20, -5, 16)
    V_top = np.ones_like(J_top)
    J_bot = np.linspace(0, 30, 31)
    V_bot = np.ones_like(J_bot)

    with pytest.raises(ValueError, match="overlap"):
        series_match_jv(J_top, V_top, J_bot, V_bot)


# ---------------------------------------------------------------------------
# run_tandem_jv — smoke test with mocked load_nk (real sub-cell sweeps)
# ---------------------------------------------------------------------------

def test_run_tandem_jv_smoke(monkeypatch):
    """Smoke test: two identical sub-cells, mocked n,k data.

    Uses a very coarse grid (N_grid=30, n_points=12) so the two sub-cell
    sweeps complete in a few seconds. The absorber optical material is
    monkeypatched to a constant n/k so load_nk doesn't need the CSV file.

    Assertions:
    - Returns a TandemJVResult.
    - metrics has the expected fields with physically plausible values.
    - V_tandem length == J_common length.
    - Tandem V_oc ≈ top.V_oc + bot.V_oc (series addition).
    """
    import perovskite_sim.physics.tandem_optics as tandem_optics_mod
    from perovskite_sim.experiments.tandem_jv import TandemJVResult, run_tandem_jv
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.models.tandem_config import JunctionLayer, TandemConfig

    # Stub load_nk with constant n=2.5, k=0.05 (moderate absorption).
    def _fake_load_nk(material, wavelengths_nm):
        n = np.full_like(wavelengths_nm, 2.5, dtype=float)
        k = np.full_like(wavelengths_nm, 0.05, dtype=float)
        return wavelengths_nm, n, k

    monkeypatch.setattr(tandem_optics_mod, "load_nk", _fake_load_nk)

    top_cell = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    bot_cell = load_device_from_yaml("configs/nip_MAPbI3.yaml")

    cfg = TandemConfig(
        top_cell=top_cell,
        bottom_cell=bot_cell,
        junction_stack=(),           # no recombination layer → simpler grid
        junction_model="ideal_ohmic",
        light_direction="top_first",
        benchmark=None,
    )

    wavelengths_nm = np.linspace(300.0, 800.0, 30)
    wavelengths_m = wavelengths_nm * 1e-9
    spectral_flux = np.full_like(wavelengths_m, 1e21)

    result = run_tandem_jv(
        cfg,
        wavelengths_m=wavelengths_m,
        spectral_flux=spectral_flux,
        wavelengths_nm=wavelengths_nm,
        N_grid=30,
        n_points=12,
    )

    # Type check
    assert isinstance(result, TandemJVResult)

    # Shape consistency
    assert len(result.V) == len(result.J)
    assert len(result.V_top) == len(result.J)
    assert len(result.V_bot) == len(result.J)

    # Metrics fields exist and are plausible
    m = result.metrics
    assert hasattr(m, "V_oc")
    assert hasattr(m, "J_sc")
    assert hasattr(m, "FF")
    assert hasattr(m, "PCE")
    assert m.V_oc >= 0.0
    assert m.J_sc >= 0.0

    # Tandem V_oc should be approximately top.V_oc + bot.V_oc
    top_voc = result.top_result.metrics_fwd.V_oc
    bot_voc = result.bot_result.metrics_fwd.V_oc
    expected_voc = top_voc + bot_voc
    # Allow ±30 % tolerance: series-matching shifts the operating point so
    # the matched V_oc is a lower bound; exact equality would be too strict.
    if expected_voc > 0:
        assert m.V_oc == pytest.approx(expected_voc, rel=0.3)
