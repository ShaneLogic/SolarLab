from __future__ import annotations
import numpy as np
import pytest

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


# ---------------------------------------------------------------------------
# Stage B(c.2) Task 2 — per-face D override smoke tests.
# ---------------------------------------------------------------------------


def _setup():
    Ny, Nx = 6, 5
    x = np.linspace(0.0, 1e-6, Nx)
    y = np.linspace(0.0, 1e-6, Ny)
    phi = np.linspace(0.0, 0.5, Ny)[:, None] * np.ones((Ny, Nx))
    n = np.full((Ny, Nx), 1e16)
    p = np.full((Ny, Nx), 1e16)
    D_n = np.full((Ny, Nx), 1.5e-3)
    D_p = np.full((Ny, Nx), 7.0e-4)
    V_T = 0.025852
    return x, y, phi, n, p, D_n, D_p, V_T, Ny, Nx


def test_sg_fluxes_2d_n_no_override_unchanged():
    """Calling without overrides reproduces the existing constant-D output."""
    x, y, phi, n, p, D_n, _, V_T, _, _ = _setup()
    Jx, Jy = sg_fluxes_2d_n(phi, n, x, y, D_n, V_T)
    Jx_alt, Jy_alt = sg_fluxes_2d_n(
        phi, n, x, y, D_n, V_T,
        D_n_x_face=None, D_n_y_face=None,
    )
    np.testing.assert_array_equal(Jx, Jx_alt)
    np.testing.assert_array_equal(Jy, Jy_alt)


def test_sg_fluxes_2d_p_no_override_unchanged():
    x, y, phi, n, p, _, D_p, V_T, _, _ = _setup()
    Jx, Jy = sg_fluxes_2d_p(phi, p, x, y, D_p, V_T)
    Jx_alt, Jy_alt = sg_fluxes_2d_p(
        phi, p, x, y, D_p, V_T,
        D_p_x_face=None, D_p_y_face=None,
    )
    np.testing.assert_array_equal(Jx, Jx_alt)
    np.testing.assert_array_equal(Jy, Jy_alt)


def test_sg_fluxes_2d_n_override_with_harmonic_mean_matches_no_override():
    """Passing the explicit harmonic-mean of D_n as an override must produce
    the same output as letting the function compute it internally."""
    x, y, phi, n, p, D_n, _, V_T, Ny, Nx = _setup()
    _eps = 1e-300
    D_n_x_harm = 2.0 * D_n[:, :-1] * D_n[:, 1:] / (D_n[:, :-1] + D_n[:, 1:] + _eps)
    D_n_y_harm = 2.0 * D_n[:-1, :] * D_n[1:, :] / (D_n[:-1, :] + D_n[1:, :] + _eps)
    Jx_internal, Jy_internal = sg_fluxes_2d_n(phi, n, x, y, D_n, V_T)
    Jx_override, Jy_override = sg_fluxes_2d_n(
        phi, n, x, y, D_n, V_T,
        D_n_x_face=D_n_x_harm, D_n_y_face=D_n_y_harm,
    )
    np.testing.assert_allclose(Jx_internal, Jx_override, rtol=1e-15)
    np.testing.assert_allclose(Jy_internal, Jy_override, rtol=1e-15)


def test_sg_fluxes_2d_n_override_actually_overrides():
    """Passing 2x harmonic-mean D as override must produce 2x the J flux."""
    x, y, phi, n, p, D_n, _, V_T, Ny, Nx = _setup()
    _eps = 1e-300
    D_n_x_harm = 2.0 * D_n[:, :-1] * D_n[:, 1:] / (D_n[:, :-1] + D_n[:, 1:] + _eps)
    D_n_y_harm = 2.0 * D_n[:-1, :] * D_n[1:, :] / (D_n[:-1, :] + D_n[1:, :] + _eps)
    Jx_base, Jy_base = sg_fluxes_2d_n(phi, n, x, y, D_n, V_T)
    Jx_2x, Jy_2x = sg_fluxes_2d_n(
        phi, n, x, y, D_n, V_T,
        D_n_x_face=2.0 * D_n_x_harm, D_n_y_face=2.0 * D_n_y_harm,
    )
    np.testing.assert_allclose(Jx_2x, 2.0 * Jx_base, rtol=1e-13)
    np.testing.assert_allclose(Jy_2x, 2.0 * Jy_base, rtol=1e-13)


def test_sg_fluxes_2d_p_override_actually_overrides():
    x, y, phi, n, p, _, D_p, V_T, Ny, Nx = _setup()
    _eps = 1e-300
    D_p_x_harm = 2.0 * D_p[:, :-1] * D_p[:, 1:] / (D_p[:, :-1] + D_p[:, 1:] + _eps)
    D_p_y_harm = 2.0 * D_p[:-1, :] * D_p[1:, :] / (D_p[:-1, :] + D_p[1:, :] + _eps)
    Jx_base, Jy_base = sg_fluxes_2d_p(phi, p, x, y, D_p, V_T)
    Jx_2x, Jy_2x = sg_fluxes_2d_p(
        phi, p, x, y, D_p, V_T,
        D_p_x_face=2.0 * D_p_x_harm, D_p_y_face=2.0 * D_p_y_harm,
    )
    np.testing.assert_allclose(Jx_2x, 2.0 * Jx_base, rtol=1e-13)
    np.testing.assert_allclose(Jy_2x, 2.0 * Jy_base, rtol=1e-13)
