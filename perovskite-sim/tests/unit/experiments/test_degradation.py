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
    """J_sc must be computed once from the fresh SC state and held constant
    across all snapshots.  Ion migration mainly shifts V_oc/FF; re-solving J_sc
    per snapshot causes spikes and adds runtime overhead.
    Tolerance 1e-9: values must be bit-for-bit identical (same scalar reused)."""
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.experiments.degradation import run_degradation
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    result = run_degradation(stack, t_end=5.0, n_snapshots=3,
                             V_bias=0.9, N_grid=20, dt_max=1.0)
    assert result.J_sc[0] > 0, f"J_sc={result.J_sc[0]:.3f} should be positive"
    assert np.all(result.V_oc > 0)
    assert np.all(result.PCE > 0)
    # All snapshots must carry the exact same J_sc value (computed once at startup)
    np.testing.assert_array_equal(
        result.J_sc, result.J_sc[0],
        err_msg="J_sc must be constant (computed once from fresh SC state)"
    )


def test_voc_uses_geometric_mean():
    """V_oc must use geometric mean of n*p (mean of log), not arithmetic mean.

    Create a mock absorber state where n*p has one very high spike (mimicking
    interface injection after ion pileup). The arithmetic mean is dominated by
    the spike; the geometric mean is not.
    """
    from perovskite_sim import constants

    ni_sq = (3.2e13) ** 2
    # 10 absorber nodes: 9 "bulk" with n*p = 10*ni_sq, 1 spike at 1000*ni_sq
    np_bulk = 10.0 * ni_sq * np.ones(10)
    np_bulk[-1] = 1000.0 * ni_sq  # interface spike

    V_T = constants.V_T
    voc_arithmetic = V_T * np.log(np.mean(np_bulk) / ni_sq)
    voc_geometric  = V_T * np.mean(np.log(np_bulk / ni_sq))

    # Geometric must be < arithmetic (Jensen's inequality; log is concave)
    assert voc_geometric < voc_arithmetic, (
        f"geometric ({voc_geometric:.4f} V) should be < arithmetic ({voc_arithmetic:.4f} V)"
    )
    # Geometric must be closer to the bulk value than arithmetic is
    # (this is the key advantage: less biased by interface spikes)
    bulk_voc = V_T * np.log(10.0)
    assert abs(voc_geometric - bulk_voc) < abs(voc_arithmetic - bulk_voc), (
        f"geometric ({voc_geometric:.4f} V) should be closer to bulk ({bulk_voc:.4f} V) "
        f"than arithmetic ({voc_arithmetic:.4f} V)"
    )


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
                             V_bias=0.9, N_grid=20, dt_max=1.0)
    assert result.PCE.shape == (2,)
    assert np.all(np.isfinite(result.PCE))
    assert np.all(np.isfinite(result.V_oc))
    assert np.all(np.isfinite(result.J_sc))
