"""Tests for V_oc(T) activation-energy experiment."""
import numpy as np
import pytest

from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.voc_t import VocTResult
from perovskite_sim.experiments.voc_t import run_voc_t, _linear_fit


@pytest.fixture
def nip_stack():
    return load_device_from_yaml("configs/nip_MAPbI3.yaml")


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_rejects_small_n_points(nip_stack):
    with pytest.raises(ValueError, match="n_points"):
        run_voc_t(nip_stack, n_points=1)


def test_rejects_nonpositive_T_min(nip_stack):
    with pytest.raises(ValueError, match="T_min"):
        run_voc_t(nip_stack, T_min=0.0)


def test_rejects_T_max_le_T_min(nip_stack):
    with pytest.raises(ValueError, match="T_max"):
        run_voc_t(nip_stack, T_min=300.0, T_max=300.0)


def test_rejects_small_N_grid(nip_stack):
    with pytest.raises(ValueError, match="N_grid"):
        run_voc_t(nip_stack, N_grid=2)


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_voc_t_runs(nip_stack):
    """run_voc_t should complete and return a VocTResult."""
    result = run_voc_t(nip_stack, T_min=280.0, T_max=320.0, n_points=3,
                       N_grid=30, jv_n_points=12)
    assert isinstance(result, VocTResult)
    assert len(result.T_arr) == 3
    assert len(result.V_oc_arr) == 3
    assert len(result.J_sc_arr) == 3


@pytest.mark.slow
def test_voc_t_temperature_sweep_matches_input(nip_stack):
    """T_arr should span [T_min, T_max] inclusive in n_points linear steps."""
    result = run_voc_t(nip_stack, T_min=280.0, T_max=320.0, n_points=3,
                       N_grid=30, jv_n_points=12)
    assert result.T_arr[0] == pytest.approx(280.0)
    assert result.T_arr[-1] == pytest.approx(320.0)
    assert np.all(np.diff(result.T_arr) > 0)


@pytest.mark.slow
def test_voc_t_voc_physical(nip_stack):
    """V_oc at each T should be in a physically reasonable range."""
    result = run_voc_t(nip_stack, T_min=280.0, T_max=320.0, n_points=3,
                       N_grid=30, jv_n_points=12)
    assert np.all(result.V_oc_arr > 0.5)
    assert np.all(result.V_oc_arr < 1.5)


@pytest.mark.slow
def test_voc_t_slope_negative(nip_stack):
    """dV_oc/dT should be negative (heating narrows V_oc for any
    non-degenerate semiconductor — the kT·ln(J_00/J_sc) term wins over
    the weak J_sc(T) dependence in the reasonable 280-320 K range)."""
    result = run_voc_t(nip_stack, T_min=280.0, T_max=320.0, n_points=3,
                       N_grid=30, jv_n_points=12)
    assert result.slope < 0, f"slope = {result.slope:.4e} V/K"


@pytest.mark.slow
def test_voc_t_activation_energy_below_bandgap(nip_stack):
    """The extrapolated T=0 intercept (proxy for E_A) should not exceed
    the absorber bandgap by more than a small margin — recombination can
    never pump V_oc above Eg/q. MAPbI3 Eg ≈ 1.55 eV."""
    result = run_voc_t(nip_stack, T_min=280.0, T_max=320.0, n_points=3,
                       N_grid=30, jv_n_points=12)
    assert result.E_A_eV < 2.0, f"E_A = {result.E_A_eV:.3f} eV"
    assert result.E_A_eV > 0.5, f"E_A = {result.E_A_eV:.3f} eV"


@pytest.mark.slow
def test_voc_t_all_finite(nip_stack):
    """All arrays and scalars in the result should be finite."""
    result = run_voc_t(nip_stack, T_min=280.0, T_max=320.0, n_points=3,
                       N_grid=30, jv_n_points=12)
    assert np.all(np.isfinite(result.T_arr))
    assert np.all(np.isfinite(result.V_oc_arr))
    assert np.all(np.isfinite(result.J_sc_arr))
    assert np.isfinite(result.slope)
    assert np.isfinite(result.intercept_0K)
    assert np.isfinite(result.E_A_eV)
    assert np.isfinite(result.R_squared)


@pytest.mark.slow
def test_voc_t_result_frozen(nip_stack):
    """VocTResult should be immutable."""
    result = run_voc_t(nip_stack, T_min=280.0, T_max=320.0, n_points=3,
                       N_grid=30, jv_n_points=12)
    with pytest.raises(AttributeError):
        result.slope = 0.0


@pytest.mark.slow
def test_voc_t_progress_callback(nip_stack):
    """Progress callback should be called during the sweep."""
    calls = []

    def cb(stage, current, total, msg):
        calls.append((stage, current, total))

    run_voc_t(nip_stack, T_min=280.0, T_max=320.0, n_points=3,
              N_grid=30, jv_n_points=12, progress=cb)
    assert len(calls) > 0
    assert all(c[0] == "voc_t" for c in calls)


# ---------------------------------------------------------------------------
# Linear-fit unit tests
# ---------------------------------------------------------------------------

def test_linear_fit_recovers_slope_and_intercept():
    """_linear_fit should recover the true parameters from a clean line."""
    T = np.linspace(260.0, 340.0, 10)
    slope_true = -1.5e-3   # -1.5 mV/K
    intercept_true = 1.6   # 1.6 V at T=0
    V = slope_true * T + intercept_true
    slope, intercept, r2 = _linear_fit(T, V)
    assert slope == pytest.approx(slope_true, rel=1e-6)
    assert intercept == pytest.approx(intercept_true, rel=1e-6)
    assert r2 == pytest.approx(1.0, abs=1e-9)


def test_linear_fit_r_squared_degenerate():
    """_linear_fit should return R²=0 when all V_oc are identical (no
    variance to explain)."""
    T = np.linspace(260.0, 340.0, 10)
    V = np.full_like(T, 1.0)
    slope, intercept, r2 = _linear_fit(T, V)
    assert slope == pytest.approx(0.0, abs=1e-9)
    assert intercept == pytest.approx(1.0)
    assert r2 == 0.0


def test_linear_fit_rejects_single_point():
    """A single-point input has no slope — the helper must return a sentinel
    rather than exploding inside polyfit."""
    slope, intercept, r2 = _linear_fit(np.array([300.0]), np.array([1.0]))
    assert slope == 0.0
    assert intercept == 1.0
    assert r2 == 0.0


# ---------------------------------------------------------------------------
# voc_bracketed handling (2026-07 audit)
#
# compute_metrics zeroes V_oc / FF / PCE when the sweep contains no
# zero-current crossing. run_voc_t read metrics.V_oc unconditionally, so one
# such point put a 0 V sample into the Arrhenius fit. That does not shift the
# answer, it INVERTS it: E_A is this experiment's only scientific output.
# ---------------------------------------------------------------------------

class _FakeSweep:
    """Minimal stand-in for JVResult -- run_voc_t only reads V_fwd / J_fwd."""

    def __init__(self):
        self.V_fwd = np.linspace(0.0, 1.2, 5)
        self.J_fwd = np.linspace(200.0, -50.0, 5)


class _FakeMetrics:
    def __init__(self, V_oc, bracketed):
        self.V_oc = V_oc
        self.J_sc = 220.0
        self.FF = 0.78
        self.PCE = 0.20
        self.voc_bracketed = bracketed


def _patch_sweep(monkeypatch, unbracketed_at=()):
    """Synthetic V_oc(T) with a known true slope of -2.20 mV/K at 1.10 V/300 K."""
    from perovskite_sim.experiments import voc_t as mod

    state = {"i": 0}
    T_all = np.linspace(250.0, 350.0, 6)

    def fake_run_jv_sweep(stack, **kwargs):
        return _FakeSweep()

    def fake_compute_metrics(V, J, **kwargs):
        k = state["i"]
        state["i"] += 1
        V_true = 1.10 + (-2.20e-3) * (T_all[k] - 300.0)
        if k in unbracketed_at:
            # exactly what compute_metrics returns when nothing brackets
            return _FakeMetrics(0.0, False)
        return _FakeMetrics(V_true, True)

    monkeypatch.setattr(mod, "run_jv_sweep", fake_run_jv_sweep)
    monkeypatch.setattr(mod, "compute_metrics", fake_compute_metrics)
    return T_all


def test_clean_sweep_recovers_the_true_slope(nip_stack, monkeypatch):
    _patch_sweep(monkeypatch)
    r = run_voc_t(nip_stack, T_min=250.0, T_max=350.0, n_points=6)
    assert r.slope == pytest.approx(-2.20e-3, rel=1e-6)
    assert np.all(r.voc_bracketed_arr)
    assert np.all(np.isfinite(r.V_oc_arr))


def test_unbracketed_point_is_excluded_and_slope_stays_negative(
    nip_stack, monkeypatch
):
    _patch_sweep(monkeypatch, unbracketed_at=(0,))
    with pytest.warns(RuntimeWarning, match="did not bracket"):
        r = run_voc_t(nip_stack, T_min=250.0, T_max=350.0, n_points=6)

    # The surviving points still recover the true physics exactly.
    assert r.slope == pytest.approx(-2.20e-3, rel=1e-6)
    # dV_oc/dT < 0 is required for a non-degenerate semiconductor.
    assert r.slope < 0.0
    # The bad sample is reported as NaN, never as a physical 0 V.
    assert np.isnan(r.V_oc_arr[0])
    assert r.V_oc_arr[0] != 0.0
    assert not r.voc_bracketed_arr[0]
    assert np.all(r.voc_bracketed_arr[1:])


def test_the_sentinel_would_have_inverted_the_slope(nip_stack, monkeypatch):
    """Pins that the exclusion is load-bearing, not cosmetic.

    Fitting the SAME data with the 0 V sentinel left in -- the pre-fix
    behaviour -- reverses the sign of dV_oc/dT, reporting V_oc rising with
    temperature.
    """
    T_all = np.linspace(250.0, 350.0, 6)
    V_true = 1.10 + (-2.20e-3) * (T_all - 300.0)

    slope_clean, _, _ = _linear_fit(T_all, V_true)
    V_with_sentinel = V_true.copy()
    V_with_sentinel[0] = 0.0                     # what compute_metrics returns
    slope_poisoned, _, _ = _linear_fit(T_all, V_with_sentinel)

    assert slope_clean < 0.0
    assert slope_poisoned > 0.0                  # physically impossible
    # and the excluded-point fit agrees with the clean one
    slope_excluded, _, _ = _linear_fit(T_all[1:], V_true[1:])
    assert slope_excluded == pytest.approx(slope_clean, rel=1e-9)


def test_raises_when_fewer_than_two_temperatures_bracket(nip_stack, monkeypatch):
    _patch_sweep(monkeypatch, unbracketed_at=(0, 1, 2, 3, 4))
    with pytest.raises(ValueError, match="at least 2 temperatures"):
        run_voc_t(nip_stack, T_min=250.0, T_max=350.0, n_points=6)
