import numpy as np
import pytest
from perovskite_sim.experiments.jv_sweep import JVResult, compute_metrics


def test_compute_metrics_mpp():
    """MPP power should be between 0 and Voc*Jsc."""
    V = np.linspace(0, 1.1, 50)
    J_sc = 200.0  # A/m²
    J = J_sc * (1 - np.exp((V - 1.1) / 0.05))
    result = compute_metrics(V, J)
    assert 0.0 < result.PCE < 1.0
    assert 0.0 < result.FF < 1.0
    assert result.V_oc > 0.0
    assert result.J_sc > 0.0


def test_jv_sweep_rejects_small_n_grid():
    from perovskite_sim.experiments.jv_sweep import run_jv_sweep
    from perovskite_sim.models.config_loader import load_device_from_yaml
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    with pytest.raises(ValueError, match="N_grid"):
        run_jv_sweep(stack, N_grid=2)


def test_jv_sweep_rejects_small_n_points():
    from perovskite_sim.experiments.jv_sweep import run_jv_sweep
    from perovskite_sim.models.config_loader import load_device_from_yaml
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    with pytest.raises(ValueError, match="n_points"):
        run_jv_sweep(stack, n_points=1)


def test_jv_sweep_rejects_nonpositive_v_rate():
    from perovskite_sim.experiments.jv_sweep import run_jv_sweep
    from perovskite_sim.models.config_loader import load_device_from_yaml
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    with pytest.raises(ValueError, match="v_rate"):
        run_jv_sweep(stack, v_rate=0.0)


def test_hysteresis_index_zero_for_symmetric():
    """HI = 0 when forward and reverse J-V are identical."""
    from perovskite_sim.experiments.jv_sweep import hysteresis_index
    V = np.linspace(0, 1.0, 50)
    J = np.linspace(200, 0, 50)
    hi = hysteresis_index(V, J, V, J)
    assert abs(hi) < 1e-6
