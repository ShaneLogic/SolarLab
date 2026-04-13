import numpy as np

from perovskite_sim.experiments.jv_sweep import run_jv_sweep
from perovskite_sim.models.config_loader import load_device_from_yaml


def test_end_to_end_jv_response_is_finite_and_reasonable():
    """Low-resolution device run should produce physically reasonable J-V output."""
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    result = run_jv_sweep(stack, N_grid=30, n_points=8, v_rate=5.0)

    assert np.all(np.isfinite(result.V_fwd))
    assert np.all(np.isfinite(result.J_fwd))
    assert np.all(np.isfinite(result.V_rev))
    assert np.all(np.isfinite(result.J_rev))

    assert np.all(np.diff(result.V_fwd) > 0.0)
    assert np.all(np.diff(result.V_rev) < 0.0)
    assert result.J_fwd[0] > 0.0
    assert result.J_fwd[-1] < 0.0
    assert result.J_rev[0] < 0.0
    assert result.J_rev[-1] > 0.0

    metrics = result.metrics_fwd
    assert 100.0 < metrics.J_sc < 800.0
    assert 0.7 < metrics.V_oc < 1.1
    assert 0.5 < metrics.FF < 0.95
    assert 0.1 < metrics.PCE < 0.4
    assert abs(result.hysteresis_index) < 0.2
