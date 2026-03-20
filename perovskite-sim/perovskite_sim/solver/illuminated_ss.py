"""Illuminated steady-state initialiser.

Provides an initial carrier distribution that has equilibrated under
illumination at a given applied voltage, while keeping the ion profile
close to its initial uniform distribution.

Strategy: integrate the full MOL system for t_settle seconds starting
from dark equilibrium.  Carrier dynamics equilibrate on a sub-microsecond
timescale, so t_settle = 1e-3 s is more than sufficient.  Ion migration
is negligible over this interval (D_ion * t_settle ≈ 0.3 nm displacement).

This pre-conditioned state eliminates the dark → light transient artefact
that otherwise contaminates J-V sweeps, impedance spectra, and degradation
curves when they start from the dark equilibrium.
"""
from __future__ import annotations

import numpy as np

from perovskite_sim.models.device import DeviceStack
from perovskite_sim.solver.newton import solve_equilibrium
from perovskite_sim.solver.mol import run_transient


def solve_illuminated_ss(
    x: np.ndarray,
    stack: DeviceStack,
    V_app: float = 0.0,
    t_settle: float = 1e-3,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> np.ndarray:
    """Return illuminated steady-state carrier distribution at V_app.

    Integrates the MOL system for t_settle seconds under illumination,
    starting from dark equilibrium.  The ion profile remains essentially
    unchanged (displacement << grid spacing), while carriers reach their
    light-soaked quasi-steady state.

    Parameters
    ----------
    x        : grid node positions [m]
    stack    : device stack
    V_app    : applied voltage [V] (0 = short circuit)
    t_settle : settling time [s]; default 1e-3 s
    rtol, atol : ODE solver tolerances

    Returns
    -------
    y : packed state vector [n, p, P] of shape (3N,).
        Falls back to dark equilibrium if the transient solve fails.
    """
    y_dark = solve_equilibrium(x, stack)
    sol = run_transient(
        x, y_dark,
        (0.0, t_settle), np.array([t_settle]),
        stack, illuminated=True, V_app=V_app,
        rtol=rtol, atol=atol,
    )
    if not sol.success:
        return y_dark
    return sol.y[:, -1]
