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
    # Layer 1 of Phase 6 acceptance follow-up: bracket flag set when J(V)
    # crosses zero inside the sampled range.
    assert result.voc_bracketed is True


def test_compute_metrics_voc_interpolation_between_samples():
    """V_oc should be linearly interpolated between the two adjacent
    voltage points that bracket the zero crossing — not snapped to a
    grid point."""
    # Synthetic J(V): J = +20 at V=0.95, J = -20 at V=1.05, zero at V=1.00
    V = np.array([0.0, 0.95, 1.05, 1.10])
    J = np.array([100.0, 20.0, -20.0, -100.0])
    result = compute_metrics(V, J)
    assert result.voc_bracketed is True
    assert abs(result.V_oc - 1.0) < 1e-12, (
        f"expected V_oc=1.0 (linear interp midpoint), got {result.V_oc}"
    )


def test_compute_metrics_voc_not_bracketed():
    """When V_max stops short of V_oc the result must flag ``voc_bracketed=False``
    and return sentinel zeros for V_oc / FF / PCE. J_sc is still
    interpolated at V=0 and remains meaningful."""
    # Stays positive across the whole sweep — no zero crossing.
    V = np.linspace(0.0, 0.6, 30)
    J = np.full_like(V, 200.0) - 50.0 * V  # 200 → 170 A/m², still > 0
    result = compute_metrics(V, J)
    assert result.voc_bracketed is False
    assert result.V_oc == 0.0
    assert result.FF == 0.0
    assert result.PCE == 0.0
    assert result.J_sc > 0.0  # J_sc still meaningful


def test_compute_metrics_assume_jsc_negative_flips_internally():
    """``assume_jsc_positive=False`` must be equivalent to negating J
    before calling ``compute_metrics`` with the default convention.
    This is the path the 2D solver takes (J(V=0) < 0 in 2D)."""
    V = np.linspace(0, 1.1, 50)
    J_pos = 200.0 * (1.0 - np.exp((V - 1.0) / 0.05))   # 1D-style: J_sc > 0
    J_neg = -J_pos                                      # 2D-style: J_sc < 0
    m_pos = compute_metrics(V, J_pos, assume_jsc_positive=True)
    m_flip = compute_metrics(V, J_neg, assume_jsc_positive=False)
    # Both should yield IDENTICAL metrics in the J_sc-positive convention.
    assert m_pos.voc_bracketed and m_flip.voc_bracketed
    assert abs(m_pos.V_oc - m_flip.V_oc) < 1e-12
    assert abs(m_pos.J_sc - m_flip.J_sc) < 1e-12
    assert abs(m_pos.FF - m_flip.FF) < 1e-12
    assert abs(m_pos.PCE - m_flip.PCE) < 1e-12


def test_compute_metrics_default_behaviour_bit_identical():
    """Default keyword args must keep pre-Layer-1 1D behaviour exactly.

    Same input → same V_oc / J_sc / FF / PCE values; only the new
    ``voc_bracketed`` field is added."""
    V = np.linspace(0, 1.1, 100)
    J = 250.0 * (1.0 - np.exp((V - 0.95) / 0.025))
    m = compute_metrics(V, J)
    # Reproduce the pre-Layer-1 algorithm inline and compare.
    order = np.argsort(V)
    V_s = V[order]
    J_s = J[order]
    J_sc_ref = float(np.interp(0.0, V_s, J_s))
    signs = np.sign(J_s)
    crossings = np.where((signs[:-1] > 0) & (signs[1:] <= 0))[0]
    idx = int(crossings[0])
    dV = V_s[idx + 1] - V_s[idx]
    dJ = J_s[idx + 1] - J_s[idx]
    V_oc_ref = float(V_s[idx] - J_s[idx] * dV / dJ)
    mask = (V_s >= 0.0) & (V_s <= V_oc_ref)
    P_ref = float(np.max(V_s[mask] * J_s[mask]))
    FF_ref = P_ref / (V_oc_ref * J_sc_ref)
    PCE_ref = P_ref / 1000.0
    assert abs(m.V_oc - V_oc_ref) < 1e-12
    assert abs(m.J_sc - J_sc_ref) < 1e-12
    assert abs(m.FF - FF_ref) < 1e-12
    assert abs(m.PCE - PCE_ref) < 1e-12
    assert m.voc_bracketed is True


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


# ---------------------------------------------------------------------------
# fixed_generation kwarg tests
# ---------------------------------------------------------------------------

def _make_stack_and_N(n_grid: int = 60):
    """Return (stack, N) for nip_MAPbI3 at the given n_grid.

    Replicates the grid construction from run_jv_sweep so the caller can
    build a fixed_generation array of exactly the right shape.
    """
    from perovskite_sim.models.config_loader import load_device_from_yaml
    from perovskite_sim.models.device import electrical_layers
    from perovskite_sim.discretization.grid import multilayer_grid, Layer

    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    elec = electrical_layers(stack)
    layers_grid = [Layer(l.thickness, n_grid // len(elec)) for l in elec]
    x = multilayer_grid(layers_grid)
    return stack, len(x)


def test_fixed_generation_override_is_honored():
    """Zero generation profile should drive J_sc to ~0."""
    from perovskite_sim.experiments.jv_sweep import run_jv_sweep
    from perovskite_sim.models.config_loader import load_device_from_yaml

    N_grid = 60
    stack, N = _make_stack_and_N(N_grid)
    G_zero = np.zeros(N)
    result = run_jv_sweep(
        stack, N_grid=N_grid, n_points=20, fixed_generation=G_zero,
    )
    assert abs(result.metrics_fwd.J_sc) < 1.0  # A/m²; effectively zero


def test_fixed_generation_wrong_shape_raises():
    """Passing an array with wrong shape should raise ValueError."""
    from perovskite_sim.experiments.jv_sweep import run_jv_sweep
    from perovskite_sim.models.config_loader import load_device_from_yaml

    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    with pytest.raises(ValueError, match="fixed_generation"):
        run_jv_sweep(stack, N_grid=60, n_points=20,
                     fixed_generation=np.zeros(30))
