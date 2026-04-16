"""Tests for dark J-V mode (illuminated=False)."""
import numpy as np
import pytest

from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.experiments.jv_sweep import run_jv_sweep


@pytest.fixture
def nip_stack():
    return load_device_from_yaml("configs/nip_MAPbI3.yaml")


def test_dark_jv_runs(nip_stack):
    """Dark J-V should complete without error."""
    result = run_jv_sweep(
        nip_stack, N_grid=40, n_points=15, v_rate=1.0, illuminated=False,
    )
    assert result.V_fwd is not None
    assert result.J_fwd is not None
    assert len(result.V_fwd) == 15
    assert len(result.J_fwd) == 15


def test_dark_jv_no_photocurrent(nip_stack):
    """In the dark, J_sc (current at V=0) should be ≈ 0."""
    result = run_jv_sweep(
        nip_stack, N_grid=40, n_points=20, v_rate=1.0, illuminated=False,
    )
    # J at V≈0 should be negligible (< 1 A/m² = 0.1 mA/cm²)
    J_at_zero = float(np.interp(0.0, result.V_fwd, result.J_fwd))
    assert abs(J_at_zero) < 1.0, f"Dark J(V=0) = {J_at_zero:.3f} A/m², expected ~0"


def test_dark_jv_diode_shape(nip_stack):
    """Dark J-V should show diode-like behaviour: J < 0 at forward bias."""
    result = run_jv_sweep(
        nip_stack, N_grid=40, n_points=20, v_rate=1.0, illuminated=False,
    )
    # At high forward bias, injection current should be negative (passive sign)
    V_high_mask = result.V_fwd > 0.8
    if V_high_mask.any():
        J_high = result.J_fwd[V_high_mask]
        # At least some points should have negative J (injection)
        assert np.any(J_high < 0), "Expected injection current at high forward bias"


def test_dark_jv_rejects_fixed_generation(nip_stack):
    """Cannot combine illuminated=False with fixed_generation."""
    G = np.ones(100)
    with pytest.raises(ValueError, match="Cannot combine"):
        run_jv_sweep(
            nip_stack, N_grid=40, n_points=10, illuminated=False,
            fixed_generation=G,
        )


def test_dark_vs_light_jsc_differs(nip_stack):
    """Light J-V should have significantly higher J_sc than dark."""
    result_dark = run_jv_sweep(
        nip_stack, N_grid=40, n_points=20, v_rate=1.0, illuminated=False,
    )
    result_light = run_jv_sweep(
        nip_stack, N_grid=40, n_points=20, v_rate=1.0, illuminated=True,
    )
    J_dark_0 = float(np.interp(0.0, result_dark.V_fwd, result_dark.J_fwd))
    J_light_0 = float(np.interp(0.0, result_light.V_fwd, result_light.J_fwd))
    assert J_light_0 > J_dark_0 + 50, (
        f"Light J_sc ({J_light_0:.1f}) should be much larger than dark ({J_dark_0:.1f})"
    )
