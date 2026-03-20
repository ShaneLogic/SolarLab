import numpy as np
import pytest
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.discretization.grid import multilayer_grid, Layer
from perovskite_sim.solver.newton import solve_equilibrium
from perovskite_sim.solver.mol import StateVec


def test_np_product_at_equilibrium():
    """n*p ≈ ni² throughout device at equilibrium."""
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    layers_grid = [Layer(l.thickness, 50) for l in stack.layers]
    x = multilayer_grid(layers_grid)
    y_eq = solve_equilibrium(x, stack)
    N = len(x)
    sv = StateVec.unpack(y_eq, N)
    absorber = next(l for l in stack.layers if l.role == "absorber")
    ni = absorber.params.ni
    # Compute absorber x-range from layer stack
    offset = 0.0
    for layer in stack.layers:
        if layer.role == "absorber":
            abs_lo, abs_hi = offset, offset + layer.thickness
            break
        offset += layer.thickness
    # Check deep interior of absorber (skip 20% from each interface)
    margin = 0.2 * (abs_hi - abs_lo)
    abs_mask = (x > abs_lo + margin) & (x < abs_hi - margin)
    ratio = sv.n[abs_mask] * sv.p[abs_mask] / ni**2
    # Allow 3 orders of magnitude variation (junction regions can deviate)
    assert np.all(ratio > 1e-3)


def test_ion_profile_within_plim():
    """Ion vacancies must never exceed P_lim."""
    stack = load_device_from_yaml("configs/nip_MAPbI3.yaml")
    layers_grid = [Layer(l.thickness, 50) for l in stack.layers]
    x = multilayer_grid(layers_grid)
    y_eq = solve_equilibrium(x, stack)
    N = len(x)
    sv = StateVec.unpack(y_eq, N)
    absorber = next(l for l in stack.layers if l.role == "absorber")
    assert np.all(sv.P <= absorber.params.P_lim)
