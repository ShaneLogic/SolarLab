import numpy as np
import pytest
from perovskite_sim.discretization.grid import tanh_grid, multilayer_grid, Layer


def test_tanh_grid_endpoints():
    x = tanh_grid(100, L=400e-9, alpha=3.0)
    assert x[0] == pytest.approx(0.0)
    assert x[-1] == pytest.approx(400e-9)


def test_tanh_grid_length():
    x = tanh_grid(100, L=400e-9, alpha=3.0)
    assert len(x) == 101  # N+1 points


def test_tanh_grid_monotone():
    x = tanh_grid(100, L=400e-9, alpha=3.0)
    assert np.all(np.diff(x) > 0)


def test_tanh_grid_boundary_concentration():
    x_tanh = tanh_grid(100, L=400e-9, alpha=5.0)
    x_uni = np.linspace(0, 400e-9, 101)
    # Tanh grid should have smaller first spacing than uniform
    assert x_tanh[1] - x_tanh[0] < x_uni[1] - x_uni[0]


def test_multilayer_grid_continuity():
    layers = [
        Layer(thickness=100e-9, N=50),
        Layer(thickness=400e-9, N=100),
        Layer(thickness=200e-9, N=50),
    ]
    x = multilayer_grid(layers, alpha=3.0)
    assert x[0] == pytest.approx(0.0)
    assert x[-1] == pytest.approx(700e-9)
    assert np.all(np.diff(x) > 0)
