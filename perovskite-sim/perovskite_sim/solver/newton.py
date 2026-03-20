"""
Equilibrium initialiser for the perovskite drift-diffusion system.

Uses a quasi-neutral (charge-neutrality) initialisation:
  1. For each grid node, set n and p so that:
       n · p = ni²(layer)      (mass-action law)
       n – p = N_D – N_A + P   (local charge neutrality)
     This gives an analytic, overflow-free closed-form solution.
  2. Perform ONE linear Poisson solve to obtain a self-consistent
     electrostatic potential φ.

Physical rationale
------------------
- Boltzmann-based Gummel iterations assume a single intrinsic density
  ni throughout the device.  For multi-layer PSC stacks (where ni
  spans ni=1 m⁻³ in transport layers vs ni~3×10¹³ m⁻³ in the
  perovskite absorber) the Boltzmann formula produces exp-overflow and
  never converges.
- The quasi-neutral initial condition gives np = ni²(layer) at every
  node, so the SRH recombination term vanishes identically.  Carrier
  drift currents in each layer are individually uniform (∂J/∂x ≈ 0)
  for a slowly-varying potential, keeping the interior-node residual
  near zero.
- The returned state is an excellent seed for the time-domain (Radau)
  integrator, which will quickly relax any residual transients.
"""
from __future__ import annotations
import numpy as np
from perovskite_sim.physics.poisson import solve_poisson
from perovskite_sim.solver.mol import StateVec, _build_layerwise_arrays, _equilibrium_bc
from perovskite_sim.models.device import DeviceStack

Q   = 1.602176634e-19


def solve_equilibrium(
    x: np.ndarray,
    stack: DeviceStack,
) -> np.ndarray:
    """
    Return a quasi-neutral equilibrium initial condition.

    At every node the carriers satisfy:
        n · p = ni²(layer),     n – p = N_D – N_A + P
    followed by one Poisson solve for the potential.

    Returns the packed state vector y = [n, p, P] of shape (3N,).
    """
    N = len(x)
    eps_r, _, _, N_A, N_D, _ = _build_layerwise_arrays(x, stack)

    # ── per-node ion profile and intrinsic density ───────────────────────────
    P     = np.zeros(N)
    ni_arr = np.ones(N)
    offset = 0.0
    for layer in stack.layers:
        lo = offset - 1e-15
        hi = offset + layer.thickness + 1e-15
        mask = (x >= lo) & (x <= hi)
        ni_arr[mask] = layer.params.ni
        if layer.role == "absorber":
            P[mask] = layer.params.P0
        offset += layer.thickness

    ni_sq = ni_arr ** 2

    # ── quasi-neutral carrier densities ─────────────────────────────────────
    # Solve:  n - p = net,  n * p = ni²
    # → n = 0.5 * (net + sqrt(net² + 4·ni²))   for net >= 0 (n-type / ion-dominated)
    # → p = 0.5 * (|net| + sqrt(net² + 4·ni²)) for net < 0  (p-type)
    # Using numerically stable two-branch formula to avoid cancellation.
    net  = N_D - N_A + P
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
    n_L, p_L, n_R, p_R = _equilibrium_bc(stack, x)
    n[0] = n_L;  n[-1] = n_R
    p[0] = p_L;  p[-1] = p_R

    # ── single Poisson solve for self-consistent φ ───────────────────────────
    # With quasi-neutral ICs, rho ≈ 0 everywhere → φ ≈ linear.
    rho = Q * (p - n + P - N_A + N_D)
    phi = solve_poisson(x, eps_r, rho,
                        phi_left=0.0, phi_right=stack.V_bi)

    return StateVec.pack(n, p, P)
