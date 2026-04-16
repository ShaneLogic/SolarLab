"""Tests for current decomposition (compute_current_components)."""
import numpy as np
import pytest

from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.models.current import CurrentComponents
from perovskite_sim.discretization.grid import multilayer_grid, Layer
from perovskite_sim.solver.illuminated_ss import solve_illuminated_ss
from perovskite_sim.solver.mol import build_material_arrays, run_transient
from perovskite_sim.experiments.jv_sweep import (
    compute_current_components,
    _compute_current,
    _integrate_step,
)


@pytest.fixture
def setup():
    """Build grid, mat, and illuminated SS for nip_MAPbI3."""
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    N_grid = 40
    elec = electrical_layers(stack)
    layers_grid = [Layer(l.thickness, N_grid // len(elec)) for l in elec]
    x = multilayer_grid(layers_grid)
    mat = build_material_arrays(x, stack)
    y_ss = solve_illuminated_ss(x, stack, V_app=0.0)
    return stack, x, mat, y_ss


def test_returns_current_components(setup):
    """compute_current_components should return a CurrentComponents instance."""
    stack, x, mat, y_ss = setup
    cc = compute_current_components(x, y_ss, stack, V_app=0.0, mat=mat)
    assert isinstance(cc, CurrentComponents)
    N = len(x)
    assert cc.J_n.shape == (N - 1,)
    assert cc.J_p.shape == (N - 1,)
    assert cc.J_ion.shape == (N - 1,)
    assert cc.J_disp.shape == (N - 1,)
    assert cc.J_total.shape == (N - 1,)


def test_total_equals_sum_of_components(setup):
    """J_total should equal J_n + J_p + J_ion + J_disp."""
    stack, x, mat, y_ss = setup
    cc = compute_current_components(x, y_ss, stack, V_app=0.0, mat=mat)
    J_sum = cc.J_n + cc.J_p + cc.J_ion + cc.J_disp
    np.testing.assert_allclose(cc.J_total, J_sum, rtol=1e-10)


def test_displacement_zero_without_prev(setup):
    """Without y_prev, displacement current should be zero."""
    stack, x, mat, y_ss = setup
    cc = compute_current_components(x, y_ss, stack, V_app=0.0, mat=mat)
    np.testing.assert_array_equal(cc.J_disp, 0.0)


def test_displacement_nonzero_with_prev(setup):
    """With y_prev and dt, displacement current should be non-zero."""
    stack, x, mat, y_ss = setup
    # Step forward in time to get a different state
    dt = 1e-5
    y_next = _integrate_step(x, y_ss, stack, mat, V_app=0.5, t_lo=0.0,
                              t_hi=dt, rtol=1e-4, atol=1e-6)
    cc = compute_current_components(
        x, y_next, stack, V_app=0.5, y_prev=y_ss, dt=dt, mat=mat,
        V_app_prev=0.0,
    )
    assert np.any(cc.J_disp != 0.0), "Expected non-zero displacement current"


def test_consistent_with_compute_current(setup):
    """J_total[0] should match _compute_current (terminal current)."""
    stack, x, mat, y_ss = setup
    cc = compute_current_components(x, y_ss, stack, V_app=0.0, mat=mat)
    J_terminal = _compute_current(x, y_ss, stack, V_app=0.0, mat=mat)
    np.testing.assert_allclose(cc.J_total[0], J_terminal, rtol=1e-10)


def test_electron_hole_dominate_at_sc(setup):
    """At short circuit, conduction currents should dominate over ion current."""
    stack, x, mat, y_ss = setup
    cc = compute_current_components(x, y_ss, stack, V_app=0.0, mat=mat)
    J_cond = np.abs(cc.J_n) + np.abs(cc.J_p)
    J_ion_abs = np.abs(cc.J_ion)
    # Conduction current should be much larger than ionic at short circuit
    assert np.mean(J_cond) > 10 * np.mean(J_ion_abs), (
        "Conduction current should dominate at short circuit"
    )


def test_all_finite(setup):
    """All current components should be finite."""
    stack, x, mat, y_ss = setup
    cc = compute_current_components(x, y_ss, stack, V_app=0.0, mat=mat)
    assert np.all(np.isfinite(cc.J_n))
    assert np.all(np.isfinite(cc.J_p))
    assert np.all(np.isfinite(cc.J_ion))
    assert np.all(np.isfinite(cc.J_disp))
    assert np.all(np.isfinite(cc.J_total))
