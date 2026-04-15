"""Physical conservation regression checks.

These tests catch silent state corruption that does not crash the solver
but violates a fundamental invariant. They are the canary for the failure
mode that made the substrate-stack bug a 12-minute hang: Radau accepted
bad RHS values because nothing ever asserted they were finite or that
conserved quantities stayed bounded.

Run with: pytest -m slow tests/regression/test_conservation.py
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.solver.mol import (
    _charge_density, build_material_arrays, run_transient, StateVec,
)
from perovskite_sim.solver.newton import solve_equilibrium
from perovskite_sim.discretization.grid import multilayer_grid, Layer
from perovskite_sim.models.device import electrical_layers

pytestmark = pytest.mark.slow


def _build_state(stack, N_grid: int = 60):
    elec = electrical_layers(stack)
    layers_grid = [Layer(l.thickness, N_grid // len(elec)) for l in elec]
    x = multilayer_grid(layers_grid)
    mat = build_material_arrays(x, stack)
    y = solve_equilibrium(x, stack)
    return x, y, mat


def _integrate_charge(rho: np.ndarray, x: np.ndarray) -> float:
    return float(np.trapezoid(rho, x))


def _total_ion(y: np.ndarray, N: int, x: np.ndarray) -> float:
    sv = StateVec.unpack(y, N)
    return float(np.trapezoid(sv.P, x))


def _total_charge_from_y(
    y: np.ndarray, mat, x: np.ndarray,
) -> float:
    N = len(x)
    sv = StateVec.unpack(y, N)
    n = sv.n.copy(); n[0] = mat.n_L; n[-1] = mat.n_R
    p = sv.p.copy(); p[0] = mat.p_L; p[-1] = mat.p_R
    rho = _charge_density(p, n, sv.P, mat.P_ion0, mat.N_A, mat.N_D)
    return _integrate_charge(rho, x)


@pytest.fixture(scope="module")
def _finite_check_on():
    """Turn on the assemble_rhs finite-check for the duration of this module.

    The guard is read from the environment at module load time in mol.py, so
    we poke the module-level flag directly. On teardown we restore the
    original value so other tests aren't affected.
    """
    import perovskite_sim.solver.mol as _mol
    prev = _mol._RHS_FINITE_CHECK
    _mol._RHS_FINITE_CHECK = True
    try:
        yield
    finally:
        _mol._RHS_FINITE_CHECK = prev


def test_dark_equilibrium_charge_neutrality():
    """At equilibrium, ∫ρ dx must be ~0 within numerical tolerance.

    Equilibrium is constructed by Newton's method on the coupled Poisson +
    continuity system, so charge neutrality is a direct consistency check
    on the initial condition — a nonzero integrated charge implies the
    equilibrium solver accepted an inconsistent state.
    """
    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    x, y, mat = _build_state(stack)
    N = len(x)
    sv = StateVec.unpack(y, N)
    n = sv.n.copy(); n[0] = mat.n_L; n[-1] = mat.n_R
    p = sv.p.copy(); p[0] = mat.p_L; p[-1] = mat.p_R
    rho = _charge_density(p, n, sv.P, mat.P_ion0, mat.N_A, mat.N_D)
    total_rho = _integrate_charge(rho, x)
    # Reference charge: the total *dopant* charge that would sit in the
    # device if every doped layer had its full N_D - N_A density. Integrated
    # ρ must be many orders of magnitude below this — at equilibrium the
    # coupled Newton solve should cancel carrier + ionic + dopant charges
    # to well under 1 ppm of this scale.
    from perovskite_sim.constants import Q
    L = float(x[-1] - x[0])
    dopant_scale = Q * float(np.max(np.abs(mat.N_D - mat.N_A))) * L
    assert abs(total_rho) < 1e-6 * dopant_scale, (
        f"|∫ρ dx|={abs(total_rho):.3e} exceeds 1 ppm of dopant scale {dopant_scale:.3e}"
    )


def test_dark_transient_ion_and_charge_conservation(_finite_check_on):
    """Dark transient at V=0 must conserve ion mass and stay neutral.

    Two invariants are checked:

    1. ∫P_ion dx is conserved to <0.1% — the ion flux is zero at both
       contacts by construction, so the total ionic count is a strict
       conservation law of the continuous equations; discretization
       preserves it up to Radau's integration error.

    2. |∫ρ dx| stays well below a dopant-scale reference charge — quasi
       neutrality should hold in the bulk throughout the transient.
       Carriers recombine and re-inject from the contacts, so ∫n dx and
       ∫p dx individually drift and are NOT conserved; but the net charge
       balance between carriers, ions, and dopants must stay tight.

    The RHS finite-check is active so any NaN/Inf in the transient aborts
    the solve instead of silently contaminating the trajectory.
    """
    from perovskite_sim.constants import Q

    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    x, y0, mat = _build_state(stack)
    N = len(x)
    t_settle = 1e-3
    sol = run_transient(
        x, y0, (0.0, t_settle), np.array([t_settle]),
        stack, illuminated=False, V_app=0.0, rtol=1e-5, atol=1e-7, mat=mat,
    )
    assert sol.success, f"dark transient failed: {getattr(sol, 'message', '?')}"
    y1 = sol.y[:, -1]

    # (1) Ion mass conservation.
    P0 = _total_ion(y0, N, x)
    P1 = _total_ion(y1, N, x)
    assert abs(P0) > 0.0, "ion total is zero — fixture did not load ionic layer"
    rel_P = abs(P1 - P0) / abs(P0)
    assert rel_P < 1e-3, (
        f"dark transient violated ion conservation: ∫P drifted by {rel_P:.2%} "
        f"({P0:.3e} → {P1:.3e})"
    )

    # (2) Global charge stays negligible vs dopant scale.
    L = float(x[-1] - x[0])
    dopant_scale = Q * float(np.max(np.abs(mat.N_D - mat.N_A))) * L
    total_rho_1 = _total_charge_from_y(y1, mat, x)
    assert abs(total_rho_1) < 1e-4 * dopant_scale, (
        f"|∫ρ dx|={abs(total_rho_1):.3e} drifted above 1e-4 × dopant scale "
        f"{dopant_scale:.3e} during dark transient"
    )


def test_rhs_finite_guard_catches_nan_state(_finite_check_on):
    """assemble_rhs must raise _RhsNonFinite when its output contains NaN/Inf.

    Injects NaN into the state vector and calls assemble_rhs directly. The
    guard is the canary that would have surfaced the substrate-stack Radau
    hang in seconds instead of 12 minutes — any future change that lets
    NaN leak into the RHS output must fail this test.
    """
    from perovskite_sim.solver.mol import assemble_rhs, _RhsNonFinite

    stack = load_device_from_yaml("configs/ionmonger_benchmark.yaml")
    x, y0, mat = _build_state(stack)
    y_bad = y0.copy()
    y_bad[len(x) // 2] = np.nan
    with pytest.raises(_RhsNonFinite):
        assemble_rhs(0.0, y_bad, x, stack, mat, illuminated=False, V_app=0.0)
