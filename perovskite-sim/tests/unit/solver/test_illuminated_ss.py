import numpy as np
import pytest
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.discretization.grid import multilayer_grid, Layer
from perovskite_sim.solver.illuminated_ss import solve_illuminated_ss
from perovskite_sim.solver.newton import solve_equilibrium
from perovskite_sim.solver.mol import StateVec


@pytest.fixture(scope="module")
def grid_and_stack():
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    layers_grid = [Layer(l.thickness, 10) for l in stack.layers]
    x = multilayer_grid(layers_grid)
    return x, stack


def test_shape(grid_and_stack):
    x, stack = grid_and_stack
    y = solve_illuminated_ss(x, stack, V_app=0.0)
    assert y.shape == (3 * len(x),)


def test_carriers_finite(grid_and_stack):
    """All carrier values must be finite after illuminated settling."""
    x, stack = grid_and_stack
    N = len(x)
    y = solve_illuminated_ss(x, stack, V_app=0.0)
    n, p = y[:N], y[N:2*N]
    assert np.all(np.isfinite(n))
    assert np.all(np.isfinite(p))


def test_absorber_np_product_larger_under_illumination(grid_and_stack):
    """Under illumination n·p > ni² in absorber (quasi-Fermi level splitting)."""
    x, stack = grid_and_stack
    N = len(x)
    offset = stack.layers[0].thickness
    abs_mask = (x > offset) & (x < offset + stack.layers[1].thickness)
    absorber = next(l for l in stack.layers if l.role == "absorber")
    ni_sq = absorber.params.ni_sq
    y_dark = solve_equilibrium(x, stack)
    y_light = solve_illuminated_ss(x, stack, V_app=0.0)
    n_dark, p_dark = y_dark[:N][abs_mask], y_dark[N:2*N][abs_mask]
    n_light, p_light = y_light[:N][abs_mask], y_light[N:2*N][abs_mask]
    # Dark: n·p ≈ ni²  (thermal equilibrium)
    # Illuminated: n·p >> ni²  (photogeneration splits quasi-Fermi levels)
    assert np.mean(n_light * p_light) > np.mean(n_dark * p_dark) * 10


def test_ions_in_absorber_unchanged(grid_and_stack):
    """Ion density in absorber must be essentially unchanged after 1 ms."""
    x, stack = grid_and_stack
    N = len(x)
    offset = stack.layers[0].thickness
    abs_mask = (x > offset) & (x < offset + stack.layers[1].thickness)
    y_dark = solve_equilibrium(x, stack)
    y_light = solve_illuminated_ss(x, stack, V_app=0.0, t_settle=1e-3)
    P_dark = y_dark[2*N:][abs_mask]
    P_light = y_light[2*N:][abs_mask]
    # Ion displacement in 1 ms ~ 0.3 nm, negligible vs absorber thickness
    np.testing.assert_allclose(P_light, P_dark, rtol=0.05)


def test_v_app_changes_carriers(grid_and_stack):
    """Different V_app values must produce different carrier distributions."""
    x, stack = grid_and_stack
    N = len(x)
    y_sc = solve_illuminated_ss(x, stack, V_app=0.0)
    y_oc = solve_illuminated_ss(x, stack, V_app=0.9)
    n_sc, p_sc = y_sc[:N], y_sc[N:2*N]
    n_oc, p_oc = y_oc[:N], y_oc[N:2*N]
    # Near-OC bias injects more carriers into absorber
    assert np.mean(n_oc) > np.mean(n_sc)
    assert np.mean(p_oc) > np.mean(p_sc)
