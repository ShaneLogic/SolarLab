from __future__ import annotations
import numpy as np
import pytest

from perovskite_sim.twod.grid_2d import build_grid_2d
from perovskite_sim.twod.poisson_2d import (
    build_poisson_2d_factor, solve_poisson_2d,
)
from perovskite_sim.discretization.grid import Layer
from perovskite_sim.constants import EPS_0


def _uniform_grid_factor(Lx=500e-9, Ly=400e-9, Nx=20, Ny=20, eps_r=10.0,
                         lateral_bc="periodic"):
    layers = [Layer(thickness=Ly, N=Ny)]
    g = build_grid_2d(layers, lateral_length=Lx, Nx=Nx, lateral_uniform=True)
    eps_field = np.full((g.Ny, g.Nx), eps_r, dtype=float)
    fac = build_poisson_2d_factor(g, eps_field, lateral_bc=lateral_bc)
    return g, fac


def test_poisson_2d_solves_zero_charge_with_dirichlet_y():
    """Zero charge, phi=0 at y=0 and phi=V at y=Ly → linear ramp in y."""
    g, fac = _uniform_grid_factor()
    rho = np.zeros((g.Ny, g.Nx), dtype=float)
    V = 1.0
    phi = solve_poisson_2d(fac, rho, phi_bottom=0.0, phi_top=V)
    Ly = g.y[-1]
    ramp = V * g.y / Ly
    for j in range(g.Ny):
        np.testing.assert_allclose(phi[j, :], ramp[j], atol=1e-10)


def test_poisson_2d_periodic_x_consistency():
    """With periodic x and Dirichlet y, phi at i=0 must equal phi at i=Nx-1."""
    g, fac = _uniform_grid_factor()
    rho = np.zeros((g.Ny, g.Nx), dtype=float)
    phi = solve_poisson_2d(fac, rho, phi_bottom=0.5, phi_top=1.5)
    np.testing.assert_allclose(phi[:, 0], phi[:, -1], atol=1e-10)


def test_poisson_2d_recovers_1d_parabolic_profile():
    """Uniform space charge with phi(y=0)=phi(y=Ly)=0 → parabolic profile in y,
    independent of x. Verifies the FV stencil agrees with the 1D analytic limit."""
    g, fac = _uniform_grid_factor()
    rho_val = 1e6
    rho = rho_val * np.ones((g.Ny, g.Nx), dtype=float)
    phi = solve_poisson_2d(fac, rho, phi_bottom=0.0, phi_top=0.0)
    eps_r = 10.0
    Ly = g.y[-1]
    # ∇·(ε₀ε_r ∇φ) = -ρ → φ(y) = +ρ/(2 ε₀ ε_r) y (Ly - y)  [positive hump]
    # Sign confirmed against 1D solve_poisson: positive rho → positive interior phi.
    analytic = rho_val / (2.0 * EPS_0 * eps_r) * g.y * (Ly - g.y)
    for j in range(g.Ny):
        np.testing.assert_allclose(phi[j, :], analytic[j], rtol=1e-3, atol=1e-6)
