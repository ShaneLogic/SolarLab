from __future__ import annotations
import numpy as np

from perovskite_sim.twod.grid_2d import build_grid_2d
from perovskite_sim.twod.flux_2d import sg_fluxes_2d_n, sg_fluxes_2d_p
from perovskite_sim.discretization.grid import Layer
from perovskite_sim.discretization.fe_operators import sg_fluxes_n, sg_fluxes_p


def test_2d_sg_zero_field_zero_concentration_gradient():
    """phi=const, n=const → flux = 0 on every edge."""
    layers = [Layer(thickness=400e-9, N=20)]
    g = build_grid_2d(layers, lateral_length=500e-9, Nx=10, lateral_uniform=True)
    phi = np.zeros((g.Ny, g.Nx))
    n = np.full((g.Ny, g.Nx), 1e21)
    D_n = np.full((g.Ny, g.Nx), 1e-4)
    V_T = 0.0259
    Jx, Jy = sg_fluxes_2d_n(phi, n, g.x, g.y, D_n, V_T)
    assert np.allclose(Jx, 0.0, atol=1e-6)
    assert np.allclose(Jy, 0.0, atol=1e-6)


def test_2d_sg_recovers_1d_when_y_uniform():
    """When phi(x,y) and n(x,y) depend only on y, J_x must be 0 and J_y must equal
    the 1D SG flux on a slice."""
    layers = [Layer(thickness=400e-9, N=20)]
    g = build_grid_2d(layers, lateral_length=500e-9, Nx=10, lateral_uniform=True)
    phi_1d = np.linspace(0.0, 1.0, g.Ny)
    n_1d = np.linspace(1e21, 1e22, g.Ny)
    phi = np.broadcast_to(phi_1d[:, None], (g.Ny, g.Nx)).copy()
    n = np.broadcast_to(n_1d[:, None], (g.Ny, g.Nx)).copy()
    D_n = np.full((g.Ny, g.Nx), 1e-4)
    V_T = 0.0259
    Jx, Jy = sg_fluxes_2d_n(phi, n, g.x, g.y, D_n, V_T)
    assert np.allclose(Jx, 0.0, atol=1e-12)
    ref = sg_fluxes_n(phi_1d, n_1d, np.diff(g.y), 1e-4, V_T)   # (Ny-1,)
    for i in range(g.Nx):
        np.testing.assert_allclose(Jy[:, i], ref, rtol=1e-10, atol=1e-12)


def test_2d_sg_p_recovers_1d_when_y_uniform():
    """Same as above but for holes."""
    layers = [Layer(thickness=400e-9, N=20)]
    g = build_grid_2d(layers, lateral_length=500e-9, Nx=10, lateral_uniform=True)
    phi_1d = np.linspace(0.0, 1.0, g.Ny)
    p_1d = np.linspace(1e22, 1e21, g.Ny)
    phi = np.broadcast_to(phi_1d[:, None], (g.Ny, g.Nx)).copy()
    p = np.broadcast_to(p_1d[:, None], (g.Ny, g.Nx)).copy()
    D_p = np.full((g.Ny, g.Nx), 1e-4)
    V_T = 0.0259
    Jx, Jy = sg_fluxes_2d_p(phi, p, g.x, g.y, D_p, V_T)
    assert np.allclose(Jx, 0.0, atol=1e-12)
    ref = sg_fluxes_p(phi_1d, p_1d, np.diff(g.y), 1e-4, V_T)
    for i in range(g.Nx):
        np.testing.assert_allclose(Jy[:, i], ref, rtol=1e-10, atol=1e-12)
