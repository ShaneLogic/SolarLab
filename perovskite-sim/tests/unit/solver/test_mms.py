"""Method of Manufactured Solutions (MMS) convergence tests.

Verifies O(h²) spatial convergence of the Poisson finite-volume operator
on smooth manufactured solutions, both with uniform permittivity and with
a step-discontinuous eps_r profile (to exercise the harmonic-mean face
treatment in ``physics/poisson.py``).

Rationale
---------
A Method-of-Manufactured-Solutions pass is the sharpest way to catch a
silently broken discretisation. We pick an analytic φ(x), derive ρ(x)
by applying the *continuous* Poisson operator, feed ρ(x) into the discrete
solver, and measure the L2 error against the analytic φ. If the spatial
operator is second-order accurate, the error on a uniform grid must halve
by a factor of ~4 each time we double N. Anything less is a bug.

Related review items: §Poisson solver, §Harmonic-mean face permittivity.
"""
from __future__ import annotations
import math

import numpy as np
import pytest

from perovskite_sim.constants import EPS_0
from perovskite_sim.physics.poisson import solve_poisson


def _l2_norm(u: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal L2 norm of u on grid x."""
    return float(math.sqrt(np.trapz(u**2, x)))


def _observed_order(h: np.ndarray, errs: np.ndarray) -> float:
    """Slope of log(err) vs log(h) — empirical convergence order."""
    logs_h = np.log(h)
    logs_e = np.log(errs)
    # Best-fit line; robust to small noise near machine precision.
    slope, _ = np.polyfit(logs_h, logs_e, 1)
    return float(slope)


# ── MMS 1: uniform eps_r, sinusoidal solution ───────────────────────────
def test_poisson_mms_uniform_eps_second_order():
    """φ_ex(x) = sin(π x / L), uniform ε — error ~ h²."""
    L = 1.0e-6
    eps_r_val = 10.0
    Ns = [25, 50, 100, 200]
    errs = []
    hs = []
    for N in Ns:
        x = np.linspace(0.0, L, N + 1)
        phi_ex = np.sin(np.pi * x / L)
        # ρ = -ε₀·ε_r · φ'' = ε₀·ε_r·(π/L)² · sin(πx/L)
        rho = EPS_0 * eps_r_val * (np.pi / L) ** 2 * np.sin(np.pi * x / L)
        eps_r = eps_r_val * np.ones(N + 1)
        phi_num = solve_poisson(x, eps_r, rho, phi_left=0.0, phi_right=0.0)
        errs.append(_l2_norm(phi_num - phi_ex, x))
        hs.append(L / N)
    order = _observed_order(np.array(hs), np.array(errs))
    # Allow a generous floor: second-order FV on uniform grid should give
    # close to 2.0; we only fail if the scheme silently degrades to first
    # order or below.
    assert order >= 1.8, (
        f"Poisson MMS (uniform ε) order {order:.2f} below 1.8 target; "
        f"errs = {errs}"
    )


# ── MMS 2: step-discontinuous eps_r, continuous displacement field ──────
def test_poisson_mms_step_eps_harmonic_face():
    """Two-region ε with analytic continuous D = -ε·dφ/dx.

    Pick φ linear on each side, slopes chosen so that D = εₗ·Eₗ = εᵣ·Eᵣ
    is continuous at the interface. A correctly implemented harmonic-mean
    face permittivity reproduces this analytic solution to machine
    precision with zero source (ρ = 0).

    This is the cleanest check that the sharp-interface treatment is right.
    If the face permittivity used an arithmetic mean instead of harmonic,
    the interface condition D_L = D_R would be violated and the solver
    would pile extra voltage drop into the high-ε layer.
    """
    L_total = 2.0e-6
    L_half = L_total / 2.0
    eps_L = 5.0
    eps_R = 20.0
    phi_L_bc = 0.0
    phi_R_bc = 1.0
    # Continuous-displacement solution: let the common flux be D_common.
    # Then φ_L(x) = (D_common / (ε₀·ε_L))·x
    #      φ_R(x) = φ_L(L_half) + (D_common / (ε₀·ε_R))·(x - L_half)
    # Demanding φ_R(L_total) = phi_R_bc fixes D_common:
    #   D_common · L_half · (1/(ε₀·ε_L) + 1/(ε₀·ε_R)) = phi_R_bc - phi_L_bc
    D_common = (phi_R_bc - phi_L_bc) * EPS_0 / (
        L_half * (1.0 / eps_L + 1.0 / eps_R)
    )

    def phi_analytic(x_arr: np.ndarray) -> np.ndarray:
        left = x_arr <= L_half + 1e-18
        right = ~left
        out = np.empty_like(x_arr)
        out[left] = phi_L_bc + (D_common / (EPS_0 * eps_L)) * x_arr[left]
        phi_mid = phi_L_bc + (D_common / (EPS_0 * eps_L)) * L_half
        out[right] = phi_mid + (D_common / (EPS_0 * eps_R)) * (x_arr[right] - L_half)
        return out

    # Grid choice: uniform with an ODD number of intervals so the interface
    # at L/2 sits exactly at a *face midpoint* (not on a node). This is the
    # regime where the harmonic-mean face permittivity is known to be exact
    # for the series-capacitor boundary condition — that's the property under
    # test. With an even N, the interface would coincide with a node and the
    # surrounding faces would be pure-single-material, making the harmonic
    # mean irrelevant (and the test would verify something weaker).
    N = 201
    x = np.linspace(0.0, L_total, N + 1)
    # Nodes 0..N//2 lie to the left of the interface, N//2+1..N to the right.
    mid_idx = N // 2           # = 100
    eps_r = np.empty_like(x)
    eps_r[: mid_idx + 1] = eps_L
    eps_r[mid_idx + 1 :] = eps_R
    rho = np.zeros_like(x)
    phi_num = solve_poisson(x, eps_r, rho, phi_left=phi_L_bc, phi_right=phi_R_bc)
    phi_ex = phi_analytic(x)

    err = _l2_norm(phi_num - phi_ex, x) / _l2_norm(phi_ex, x)
    assert err < 1.0e-10, (
        f"harmonic-mean face permittivity failed exact-solution check: "
        f"relative L2 err = {err:.3e}"
    )


# ── MMS 3: smooth eps_r(x), higher-frequency solution ───────────────────
def test_poisson_mms_smooth_eps_convergence():
    """φ_ex(x) = sin(2πx/L)·(1 − x/L), smooth eps_r(x) = 5 + 10·(x/L).

    Exercises the general finite-volume + harmonic-mean path with a
    non-constant ε and a non-trivial source. Analytic ρ is computed from
    ρ = -d/dx (ε₀·ε_r · dφ/dx). Order should approach 2.
    """
    L = 1.0e-6

    def phi_ex(x_arr):
        return np.sin(2 * np.pi * x_arr / L) * (1.0 - x_arr / L)

    def dphi_ex(x_arr):
        s = np.sin(2 * np.pi * x_arr / L)
        c = np.cos(2 * np.pi * x_arr / L)
        return (2 * np.pi / L) * c * (1.0 - x_arr / L) + s * (-1.0 / L)

    def d2phi_ex(x_arr):
        s = np.sin(2 * np.pi * x_arr / L)
        c = np.cos(2 * np.pi * x_arr / L)
        k = 2 * np.pi / L
        # d²/dx² [sin(kx)(1 - x/L)] = -k² sin(kx)(1 - x/L) - 2(k/L) cos(kx)
        return -(k ** 2) * s * (1.0 - x_arr / L) - 2.0 * (k / L) * c

    def eps_r_ex(x_arr):
        return 5.0 + 10.0 * (x_arr / L)

    def deps_dx(x_arr):
        return 10.0 / L * np.ones_like(x_arr)

    # ρ = -ε₀ · d/dx [ε_r · φ'] = -ε₀ · (ε_r·φ'' + ε_r'·φ')
    def rho_ex(x_arr):
        return -EPS_0 * (
            eps_r_ex(x_arr) * d2phi_ex(x_arr) + deps_dx(x_arr) * dphi_ex(x_arr)
        )

    Ns = [25, 50, 100, 200]
    hs, errs = [], []
    for N in Ns:
        x = np.linspace(0.0, L, N + 1)
        phi_true = phi_ex(x)
        rho = rho_ex(x)
        eps_r = eps_r_ex(x)
        phi_num = solve_poisson(x, eps_r, rho, phi_left=0.0, phi_right=0.0)
        errs.append(_l2_norm(phi_num - phi_true, x))
        hs.append(L / N)
    order = _observed_order(np.array(hs), np.array(errs))
    assert order >= 1.8, (
        f"smooth-ε MMS order {order:.2f} below 1.8 target; errs = {errs}"
    )
