"""Tests for transient photovoltage (TPV) experiment."""
import numpy as np
import pytest

from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.tpv import TPVResult
from perovskite_sim.experiments.tpv import run_tpv, _fit_decay_tau


@pytest.fixture
def nip_stack():
    return load_device_from_yaml("tests/fixtures/configs/nip_MAPbI3.yaml")


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_rejects_small_n_grid(nip_stack):
    with pytest.raises(ValueError, match="N_grid"):
        run_tpv(nip_stack, N_grid=2)


def test_rejects_invalid_delta_G_frac(nip_stack):
    with pytest.raises(ValueError, match="delta_G_frac"):
        run_tpv(nip_stack, delta_G_frac=0.0)
    with pytest.raises(ValueError, match="delta_G_frac"):
        run_tpv(nip_stack, delta_G_frac=1.0)


def test_rejects_nonpositive_t_pulse(nip_stack):
    with pytest.raises(ValueError, match="t_pulse"):
        run_tpv(nip_stack, t_pulse=0.0)


def test_rejects_t_decay_le_t_pulse(nip_stack):
    with pytest.raises(ValueError, match="t_decay"):
        run_tpv(nip_stack, t_pulse=1e-5, t_decay=1e-5)


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_tpv_runs(nip_stack):
    """TPV should complete and return a TPVResult."""
    result = run_tpv(nip_stack, N_grid=40, n_points=30,
                     t_pulse=1e-6, t_decay=20e-6)
    assert isinstance(result, TPVResult)
    assert len(result.t) == len(result.V)
    assert len(result.t) == len(result.J)
    assert result.protocol is not None
    assert result.protocol.implicit_legacy_protocol
    assert result.reference_protocol.illumination_history[2].condition == "baseline"
    assert result.reference_protocol.illumination_history[2].relative_generation_change is None
    assert result.numerical_settings.rtol == 1e-4
    assert result.numerical_settings.max_step_s == pytest.approx(2e-7)
    np.testing.assert_array_equal(result.delta_V, result.V - result.V_reference)
    assert np.all(result.valid)


@pytest.mark.slow
def test_tpv_voc_physical(nip_stack):
    """V_oc from TPV should be in a physically reasonable range."""
    result = run_tpv(nip_stack, N_grid=40, n_points=30,
                     t_pulse=1e-6, t_decay=20e-6)
    assert 0.5 < result.V_oc < 1.5, f"V_oc = {result.V_oc:.4f} V"


@pytest.mark.slow
def test_tpv_tau_positive(nip_stack):
    """Fitted decay time tau should be positive."""
    result = run_tpv(nip_stack, N_grid=40, n_points=30,
                     t_pulse=1e-6, t_decay=20e-6)
    assert result.tau > 0


@pytest.mark.slow
def test_tpv_all_finite(nip_stack):
    """All arrays in the result should be finite."""
    result = run_tpv(nip_stack, N_grid=40, n_points=30,
                     t_pulse=1e-6, t_decay=20e-6)
    assert np.all(np.isfinite(result.t))
    assert np.all(np.isfinite(result.V))
    assert np.all(np.isfinite(result.J))
    assert np.isfinite(result.V_oc)
    assert np.isfinite(result.tau)
    assert np.isfinite(result.delta_V0)


@pytest.mark.slow
def test_tpv_time_monotonic(nip_stack):
    """Time array should be monotonically increasing."""
    result = run_tpv(nip_stack, N_grid=40, n_points=30,
                     t_pulse=1e-6, t_decay=20e-6)
    assert np.all(np.diff(result.t) > 0)


@pytest.mark.slow
def test_tpv_result_frozen(nip_stack):
    """TPVResult should be immutable."""
    result = run_tpv(nip_stack, N_grid=40, n_points=30,
                     t_pulse=1e-6, t_decay=20e-6)
    with pytest.raises(AttributeError):
        result.V_oc = 0.5


@pytest.mark.slow
def test_tpv_progress_callback(nip_stack):
    """Progress callback should be called during TPV."""
    calls = []
    def cb(stage, current, total, msg):
        calls.append((stage, current, total))

    run_tpv(nip_stack, N_grid=40, n_points=30,
            t_pulse=1e-6, t_decay=20e-6, progress=cb)
    assert len(calls) > 0
    assert all(c[0] == "tpv" for c in calls)


# ---------------------------------------------------------------------------
# Decay fitting unit tests
# ---------------------------------------------------------------------------

def test_fit_decay_tau_exponential():
    """_fit_decay_tau should recover tau from a clean exponential."""
    tau_true = 5e-6
    t = np.linspace(0, 50e-6, 500)
    V_oc = 1.0
    delta_V0 = 0.01
    V = V_oc + delta_V0 * np.exp(-t / tau_true)
    tau_fit, dV0_fit = _fit_decay_tau(t, V, V_oc)
    assert abs(tau_fit - tau_true) / tau_true < 0.1, (
        f"tau_fit={tau_fit:.3e}, expected {tau_true:.3e}"
    )
    assert abs(dV0_fit - delta_V0) / delta_V0 < 0.1


def test_fit_decay_tau_no_perturbation():
    """_fit_decay_tau should handle zero perturbation gracefully."""
    t = np.linspace(0, 50e-6, 100)
    V_oc = 1.0
    V = np.full_like(t, V_oc)
    tau, dV0 = _fit_decay_tau(t, V, V_oc)
    assert tau is None
    assert abs(dV0) < 1e-6


def test_fit_decay_tau_does_not_invent_an_unobserved_lifetime():
    t = np.linspace(0.0, 1e-7, 100)
    voltage = 1.0 + 0.01 * np.exp(-t / 1e-3)
    tau, _ = _fit_decay_tau(t, voltage, 1.0)
    assert tau is None


def test_fit_decay_tau_rejects_distinct_fast_and_slow_populations():
    t = np.linspace(0.0, 100e-6, 500)
    voltage = 1.0 + 0.004 * np.exp(-t / 2e-6) + 0.006 * np.exp(-t / 20e-6)
    tau, _ = _fit_decay_tau(t, voltage, 1.0)
    assert tau is None


def test_open_circuit_search_rejects_a_missing_zero_crossing(nip_stack, monkeypatch):
    from types import SimpleNamespace
    from perovskite_sim.experiments import tpv

    monkeypatch.setattr(tpv, "_integrate_step", lambda _x, y, *_args: y.copy())
    monkeypatch.setattr(tpv, "OpenCircuitSystem", lambda *_args: SimpleNamespace(
        capacitance_A=1.0,
        validate_state=lambda *_args: None,
        rates=lambda *_args: (np.zeros(9), 1.0, np.zeros(2)),
    ))
    with pytest.raises(RuntimeError, match="bracket|cross"):
        tpv._find_voc(np.arange(3), np.ones(9), nip_stack, object(), V_guess=1.0)


@pytest.mark.parametrize("count", [10, 11, 30, 200])
def test_sampling_always_includes_pulse_switch_and_observation_end(count):
    from perovskite_sim.experiments.tpv import _tpv_sampling_times

    times = _tpv_sampling_times(1e-6, 20e-6, count)
    assert times[0] == 0.0
    assert times[-1] == 20e-6
    assert 1e-6 in times
    assert len(times) == count + 1
    assert np.all(np.diff(times) > 0.0)
