from __future__ import annotations
import numpy as np
import pytest

from perovskite_sim.twod.grid_2d import Grid2D, build_grid_2d
from perovskite_sim.discretization.grid import Layer


def test_grid_2d_dimensions():
    layers = [Layer(thickness=50e-9, N=10),
              Layer(thickness=300e-9, N=20),
              Layer(thickness=50e-9, N=10)]
    g = build_grid_2d(layers, lateral_length=500e-9, Nx=20)
    assert g.Nx == 21               # Nx is number of POINTS, not intervals
    assert g.Ny == 41               # 10 + 20 + 10 = 40 intervals → 41 points
    assert g.n_nodes == 21 * 41


def test_grid_2d_endpoints():
    layers = [Layer(thickness=400e-9, N=20)]
    g = build_grid_2d(layers, lateral_length=500e-9, Nx=10)
    assert g.x[0] == pytest.approx(0.0)
    assert g.x[-1] == pytest.approx(500e-9)
    assert g.y[0] == pytest.approx(0.0)
    assert g.y[-1] == pytest.approx(400e-9)


def test_grid_2d_lateral_uniform_grid_when_periodic():
    """Periodic lateral BC needs uniform spacing for the simple Poisson stencil."""
    layers = [Layer(thickness=400e-9, N=20)]
    g = build_grid_2d(layers, lateral_length=500e-9, Nx=10, lateral_uniform=True)
    dx = np.diff(g.x)
    assert np.allclose(dx, dx[0], rtol=1e-12)


def test_grid_2d_clusters_y_at_layer_interfaces():
    """tanh clustering in y compresses spacing near layer boundaries."""
    layers = [Layer(thickness=100e-9, N=10), Layer(thickness=100e-9, N=10)]
    g = build_grid_2d(layers, lateral_length=200e-9, Nx=10)
    dy = np.diff(g.y)
    # Spacing near the inter-layer boundary should be smaller than mid-layer.
    assert dy[9] < dy[5]
    assert dy[10] < dy[5]
