"""
Equilibrium initialiser for the perovskite drift-diffusion system.

Uses a quasi-neutral initialisation with a neutral ionic background:
  1. For each grid node, set n and p so that:
       n · p = ni²(layer)      (mass-action law)
       n – p = N_D – N_A       (local charge neutrality with neutral ions)
     This gives an analytic, overflow-free closed-form solution.

Physical rationale
------------------
- The configured vacancy density P₀ is treated as a neutral ionic
  background, so it should not appear as net space charge in the
  initial carrier balance.
- The resulting state is not a full ion-relaxed equilibrium, but it is a
  fast, physically consistent seed for the transient solver and avoids the
  enormous artificial carrier imbalance produced when P₀ is treated as net
  positive space charge.
"""
from __future__ import annotations
import numpy as np

from perovskite_sim.solver.mol import (
    StateVec,
    _compute_iface_state_dark_eq,
    build_material_arrays,
)
from perovskite_sim.models.device import DeviceStack


def solve_equilibrium(
    x: np.ndarray,
    stack: DeviceStack,
) -> np.ndarray:
    """
    Return a quasi-neutral dark initial condition with fixed ionic background.
    """
    mat = build_material_arrays(x, stack)
    P_ion0 = mat.P_ion0
    N_A = mat.N_A
    N_D = mat.N_D

    # Reuse the solver's temperature- and grading-aware intrinsic density.
    # Reconstructing it from the raw YAML scalar here made the initial seed
    # inconsistent with the very first RHS evaluation away from ungraded
    # 300 K operation.
    ni_sq = mat.ni_sq

    # ── quasi-neutral carrier densities ─────────────────────────────────────
    # Solve:  n - p = net,  n * p = ni²
    # → n = 0.5 * (net + sqrt(net² + 4·ni²))   for net >= 0 (n-type / ion-dominated)
    # → p = 0.5 * (|net| + sqrt(net² + 4·ni²)) for net < 0  (p-type)
    # Using numerically stable two-branch formula to avoid cancellation.
    net  = N_D - N_A
    disc = np.sqrt(net ** 2 + 4.0 * ni_sq)

    # Compute both branches; np.where evaluates both before masking.
    # Suppress divide-by-zero: half_pos→0 on p-type nodes (masked by np.where).
    half_pos = 0.5 * (net + disc)          # > 0 everywhere (disc > |net|)
    half_neg = 0.5 * (-net + disc)         # > 0 everywhere

    with np.errstate(divide="ignore", invalid="ignore"):
        n = np.where(net >= 0, half_pos, ni_sq / half_neg)
        p = np.where(net >= 0, ni_sq / half_pos, half_neg)

    n = np.clip(n, 1.0, 1e36)
    p = np.clip(p, 1.0, 1e36)

    # Enforce Dirichlet contact values (ohmic contacts)
    n[0] = mat.n_L
    n[-1] = mat.n_R
    p[0] = mat.p_L
    p[-1] = mat.p_R

    # Phase E3 — pack interface-plane state block when active.
    iface_state = (
        _compute_iface_state_dark_eq(mat) if mat.N_iface_state > 0 else None
    )
    if mat.has_dual_ions:
        return StateVec.pack(
            n, p, P_ion0.copy(), mat.P_ion0_neg.copy(),
            iface_state=iface_state,
        )
    return StateVec.pack(n, p, P_ion0.copy(), iface_state=iface_state)
