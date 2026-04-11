import numpy as np
import pytest
from perovskite_sim.experiments.degradation import DegradationResult


def test_result_dataclass():
    t = np.linspace(0, 1e4, 10)
    pce = np.linspace(0.18, 0.15, 10)
    result = DegradationResult(t=t, PCE=pce, V_oc=np.ones(10),
                               J_sc=np.ones(10)*200.0,
                               ion_profiles=None)
    assert result.t[0] == 0.0
    assert result.PCE[-1] < result.PCE[0]


def test_j_sc_constant_and_positive():
    """Degradation snapshots should report finite, physically consistent metrics."""
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.experiments.degradation import run_degradation
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    result = run_degradation(stack, t_end=5.0, n_snapshots=3,
                             V_bias=0.9, N_grid=20, dt_max=1.0,
                             metric_n_points=6, metric_settle_time=1e-3)
    assert result.J_sc[0] > 0, f"J_sc={result.J_sc[0]:.3f} should be positive"
    assert np.all(result.V_oc > 0)
    assert np.all(result.PCE > 0)
    assert np.all(np.isfinite(result.PCE))
    assert np.all(np.isfinite(result.V_oc))
    assert np.all(np.isfinite(result.J_sc))
    # True PCE should include FF losses, so it must stay below J_sc * V_oc / Pin.
    pce_upper_bound = result.J_sc * result.V_oc / 1000.0
    assert np.all(result.PCE <= pce_upper_bound + 1e-12)
    assert np.any(result.PCE < 0.98 * pce_upper_bound)


def test_degradation_metrics_decline_under_bias_stress():
    """Irreversible damage should overcome light-soaking and reduce performance."""
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.experiments.degradation import run_degradation

    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    result = run_degradation(
        stack,
        t_end=30.0,
        n_snapshots=4,
        V_bias=0.9,
        N_grid=20,
        dt_max=1.0,
        metric_n_points=6,
        metric_settle_time=1e-3,
    )

    assert result.PCE[-1] < result.PCE[0], (
        f"PCE should decline under sustained bias stress: {result.PCE}"
    )
    assert result.V_oc[-1] < result.V_oc[0], (
        f"V_oc should decline under sustained bias stress: {result.V_oc}"
    )


def test_degradation_rejects_small_metric_grid():
    from perovskite_sim.experiments.degradation import run_degradation
    from perovskite_sim.models.config_loader import load_device_from_yaml
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    with pytest.raises(ValueError, match="metric_n_points"):
        run_degradation(stack, metric_n_points=2)


def test_degradation_rejects_nonpositive_metric_settle_time():
    from perovskite_sim.experiments.degradation import run_degradation
    from perovskite_sim.models.config_loader import load_device_from_yaml
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    with pytest.raises(ValueError, match="metric_settle_time"):
        run_degradation(stack, metric_settle_time=0.0)


def test_degradation_rejects_negative_damage_motion_gain():
    from perovskite_sim.experiments.degradation import run_degradation
    from perovskite_sim.models.config_loader import load_device_from_yaml
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    with pytest.raises(ValueError, match="damage_motion_gain"):
        run_degradation(stack, damage_motion_gain=-1.0)


def test_degradation_rejects_nonpositive_t_end():
    from perovskite_sim.experiments.degradation import run_degradation
    from perovskite_sim.models.config_loader import load_device_from_yaml
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    with pytest.raises(ValueError, match="t_end"):
        run_degradation(stack, t_end=0.0)


def test_degradation_rejects_small_n_grid():
    from perovskite_sim.experiments.degradation import run_degradation
    from perovskite_sim.models.config_loader import load_device_from_yaml
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    with pytest.raises(ValueError, match="N_grid"):
        run_degradation(stack, N_grid=2)


def test_degradation_rejects_small_n_snapshots():
    from perovskite_sim.experiments.degradation import run_degradation
    from perovskite_sim.models.config_loader import load_device_from_yaml
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    with pytest.raises(ValueError, match="n_snapshots"):
        run_degradation(stack, n_snapshots=0)


def test_dt_max_caps_internal_step():
    """dt_max must be accepted and must bound each internal Radau/split-step
    interval so that large snapshot gaps don't explode into O(dt/0.05) sub-steps.
    Uses tiny grid (N=20) and short run (t_end=5 s, 2 snapshots) for speed."""
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.experiments.degradation import run_degradation
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    result = run_degradation(stack, t_end=5.0, n_snapshots=2,
                             V_bias=0.9, N_grid=20, dt_max=1.0,
                             metric_n_points=6, metric_settle_time=1e-3)
    assert result.PCE.shape == (2,)
    assert np.all(np.isfinite(result.PCE))
    assert np.all(np.isfinite(result.V_oc))
    assert np.all(np.isfinite(result.J_sc))
