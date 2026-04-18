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

def test_voc_t_runs(nip_stack):
    """run_voc_t should complete and return a VocTResult."""
    result = run_voc_t(nip_stack, T_min=280.0, T_max=320.0, n_points=3,
                       N_grid=30, jv_n_points=12)
    assert isinstance(result, VocTResult)
    assert len(result.T_arr) == 3
    assert len(result.V_oc_arr) == 3
    assert len(result.J_sc_arr) == 3


def test_voc_t_temperature_sweep_matches_input(nip_stack):
    """T_arr should span [T_min, T_max] inclusive in n_points linear steps."""
    result = run_voc_t(nip_stack, T_min=280.0, T_max=320.0, n_points=3,
                       N_grid=30, jv_n_points=12)
    assert result.T_arr[0] == pytest.approx(280.0)
    assert result.T_arr[-1] == pytest.approx(320.0)
    assert np.all(np.diff(result.T_arr) > 0)


def test_voc_t_voc_physical(nip_stack):
    """V_oc at each T should be in a physically reasonable range."""
    result = run_voc_t(nip_stack, T_min=280.0, T_max=320.0, n_points=3,
                       N_grid=30, jv_n_points=12)
    assert np.all(result.V_oc_arr > 0.5)
    assert np.all(result.V_oc_arr < 1.5)


def test_voc_t_slope_negative(nip_stack):
    """dV_oc/dT should be negative (heating narrows V_oc for any
    non-degenerate semiconductor — the kT·ln(J_00/J_sc) term wins over
    the weak J_sc(T) dependence in the reasonable 280-320 K range)."""
    result = run_voc_t(nip_stack, T_min=280.0, T_max=320.0, n_points=3,
                       N_grid=30, jv_n_points=12)
    assert result.slope < 0, f"slope = {result.slope:.4e} V/K"


def test_voc_t_activation_energy_below_bandgap(nip_stack):
    """The extrapolated T=0 intercept (proxy for E_A) should not exceed
    the absorber bandgap by more than a small margin — recombination can
    never pump V_oc above Eg/q. MAPbI3 Eg ≈ 1.55 eV."""
    result = run_voc_t(nip_stack, T_min=280.0, T_max=320.0, n_points=3,
                       N_grid=30, jv_n_points=12)
    assert result.E_A_eV < 2.0, f"E_A = {result.E_A_eV:.3f} eV"
    assert result.E_A_eV > 0.5, f"E_A = {result.E_A_eV:.3f} eV"


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


def test_voc_t_result_frozen(nip_stack):
    """VocTResult should be immutable."""
    result = run_voc_t(nip_stack, T_min=280.0, T_max=320.0, n_points=3,
                       N_grid=30, jv_n_points=12)
    with pytest.raises(AttributeError):
        result.slope = 0.0


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
