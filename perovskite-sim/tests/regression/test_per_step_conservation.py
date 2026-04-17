"""Per-step conservation invariants for the MOL transient (fast suite).

Complements the slow ``test_conservation.py`` by checking the same invariants
over a much shorter transient (1 µs vs 1 ms) on a smaller grid (30 vs 60 nodes)
so the checks run in the default `-m 'not slow'` lane. The fast path catches
regressions earlier than a slow-suite-only guard.

Invariants
----------
(1) Integral ion count ∫P(x) dx conserved across the step — ion_continuity_rhs
    is divergence-form with zero-flux BCs, so this is a strict continuous law.
(2) Dual-species: ∫P_neg dx is independently conserved.
(3) Discrete Gauss's law at t=0: ∫ρ dx + ε·(E_right − E_left) ≈ 0.

A violation in (1) or (2) indicates a ion-flux sign bug or a boundary-
treatment change. A violation in (3) indicates the post-processor has
drifted out of sync with the finite-volume Poisson operator.
"""
from __future__ import annotations

import numpy as np
import pytest

from perovskite_sim.constants import EPS_0
from perovskite_sim.discretization.grid import Layer as GridLayer, multilayer_grid
from perovskite_sim.models.config_loader import load_device_from_yaml
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.physics.poisson import solve_poisson_prefactored
from perovskite_sim.solver.mol import (
    StateVec,
    _charge_density,
    build_material_arrays,
    run_transient,
)
from perovskite_sim.solver.newton import solve_equilibrium


_CONFIG = "configs/nip_MAPbI3.yaml"
_N_PER_LAYER = 10
_T_TRANSIENT = 1.0e-6     # 1 µs


def _setup():
    stack = load_device_from_yaml(_CONFIG)
    elec = electrical_layers(stack)
    layers_grid = [GridLayer(l.thickness, _N_PER_LAYER) for l in elec]
    x = multilayer_grid(layers_grid)
    y0 = solve_equilibrium(x, stack)
    mat = build_material_arrays(x, stack)
    return x, stack, y0, mat


def _integrate(field: np.ndarray, dx_cell: np.ndarray) -> float:
    return float(np.sum(field * dx_cell))


def test_total_ion_count_conserved_across_transient():
    """∫ P dx invariant across a 1 µs transient to <0.01% drift."""
    x, stack, y0, mat = _setup()
    if not np.any(mat.D_ion_node > 0):
        pytest.skip("config has no mobile ions; invariant trivially holds")

    sv0 = StateVec.unpack(y0, len(x))
    P0 = _integrate(sv0.P, mat.dx_cell)
    assert P0 > 0, "no ions in config; fixture mismatch"

    sol = run_transient(
        x, y0,
        (0.0, _T_TRANSIENT), np.array([_T_TRANSIENT]),
        stack, illuminated=True, V_app=0.0,
        rtol=1e-6, atol=1e-4,
        mat=mat,
    )
    assert sol.success, f"transient failed: {getattr(sol, 'message', '?')}"
    sv1 = StateVec.unpack(sol.y[:, -1], len(x))
    P1 = _integrate(sv1.P, mat.dx_cell)
    drift = abs(P1 - P0) / P0
    assert drift < 1e-4, (
        f"mobile ion count drifted {drift*100:.4f}% over 1 µs "
        f"(P0={P0:.6e}, P1={P1:.6e})"
    )


def test_dual_species_ion_count_conserved():
    """Both ∫P and ∫P_neg separately conserved in dual-species mode."""
    x, stack, y0, mat = _setup()
    if not mat.has_dual_ions:
        pytest.skip("config is single-species; dual-ion invariant N/A")

    sv0 = StateVec.unpack(y0, len(x))
    P0 = _integrate(sv0.P, mat.dx_cell)
    P_neg0 = _integrate(sv0.P_neg, mat.dx_cell)

    sol = run_transient(
        x, y0,
        (0.0, _T_TRANSIENT), np.array([_T_TRANSIENT]),
        stack, illuminated=True, V_app=0.0,
        rtol=1e-6, atol=1e-4,
        mat=mat,
    )
    assert sol.success
    sv1 = StateVec.unpack(sol.y[:, -1], len(x))
    P1 = _integrate(sv1.P, mat.dx_cell)
    P_neg1 = _integrate(sv1.P_neg, mat.dx_cell)

    assert abs(P1 - P0) / P0 < 1e-4, (
        f"positive-ion drift {abs(P1-P0)/P0*100:.4f}%"
    )
    assert abs(P_neg1 - P_neg0) / P_neg0 < 1e-4, (
        f"negative-ion drift {abs(P_neg1-P_neg0)/P_neg0*100:.4f}%"
    )


def test_discrete_gauss_law_at_initial_state():
    """ε_face·(E_right − E_left) + ∫ρ dx ≈ 0 at t=0.

    Verifies the post-processor (solve_poisson_prefactored) agrees with the
    divergence theorem over the full device using the SAME dual-grid widths
    as the FV operator. A mismatch would not show up as a failed Radau
    step — it would quietly corrupt every downstream field-derived quantity
    (contact currents, Mott-Schottky capacitance, etc.).
    """
    x, stack, y0, mat = _setup()
    sv = StateVec.unpack(y0, len(x))
    n = sv.n.copy(); n[0] = mat.n_L; n[-1] = mat.n_R
    p = sv.p.copy(); p[0] = mat.p_L; p[-1] = mat.p_R
    rho = _charge_density(
        p, n, sv.P, mat.P_ion0, mat.N_A, mat.N_D,
        P_neg=sv.P_neg, P_neg0=mat.P_ion0_neg,
    )
    phi = solve_poisson_prefactored(
        mat.poisson_factor, rho,
        phi_left=0.0, phi_right=stack.V_bi,
    )

    # ∫ρ dx using the FV operator's dual-grid widths.
    charge_integral = _integrate(rho, mat.dx_cell)

    # Face permittivities and field on the two outermost faces — same
    # harmonic-mean treatment the solver uses.
    h = np.diff(x)
    eps_face_L = 2.0 * mat.eps_r[0] * mat.eps_r[1] / (mat.eps_r[0] + mat.eps_r[1])
    eps_face_R = 2.0 * mat.eps_r[-2] * mat.eps_r[-1] / (mat.eps_r[-2] + mat.eps_r[-1])
    E_L = -(phi[1] - phi[0]) / h[0]
    E_R = -(phi[-1] - phi[-2]) / h[-1]
    flux_diff = EPS_0 * (eps_face_R * E_R - eps_face_L * E_L)

    # ∫ρ dx - (∫_0^L ρ dx from Gauss) = 0 ⇒ ∫ρ dx + ε·(E_L − E_R) = 0
    # Equivalently ∫ρ dx = -flux_diff (sign convention: D = εE, div D = ρ).
    #
    # Floor for the near-zero case: at equilibrium the device is globally
    # charge-neutral (∫ρ dx == 0 to machine precision) and the contact
    # fields can also be ≈0 when depletion does not reach the ohmic contacts.
    # A naive relative error is then noise/noise ≈ 1. Scale the tolerance
    # to a physically meaningful floor ε·(V_bi)·(1/L) ≈ peak-field-like
    # quantity so that "both sides are tiny" counts as passing.
    L_total = float(x[-1] - x[0])
    field_floor = EPS_0 * np.mean(mat.eps_r) * abs(stack.V_bi) / L_total
    scale = max(abs(charge_integral), abs(flux_diff), field_floor)
    residual = abs(charge_integral + flux_diff) / scale
    assert residual < 1e-3, (
        "discrete Gauss-law violation at t=0: "
        f"∫ρ dx={charge_integral:.4e}, ε·(E_R−E_L)={flux_diff:.4e}, "
        f"relative residual={residual:.2e}"
    )
