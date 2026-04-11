from __future__ import annotations
import numpy as np
from perovskite_sim.discretization.fe_operators import sg_fluxes_n, sg_fluxes_p
from perovskite_sim.physics.recombination import total_recombination

Q = 1.602176634e-19


def carrier_continuity_rhs(
    x: np.ndarray,
    phi: np.ndarray,
    n: np.ndarray,
    p: np.ndarray,
    G: np.ndarray,
    params: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized dn/dt and dp/dt via Scharfetter-Gummel fluxes.
    Boundary nodes are held fixed (returned as 0).
    """
    D_n = params["D_n"]; D_p = params["D_p"]; V_T = params["V_T"]
    dx = np.diff(x)                                # (N-1,)

    # Band-corrected potentials for heterojunctions:
    #   phi_n = phi + chi          (conduction band — drives electrons)
    #   phi_p = phi + chi + Eg     (valence band    — drives holes)
    # When chi = Eg = 0 everywhere, these reduce to phi and the SG fluxes
    # are unchanged (backward compatible with homojunction configs).
    chi = params.get("chi")
    Eg  = params.get("Eg")
    if chi is None:
        phi_n = phi
        phi_p = phi
    else:
        phi_n = phi + chi
        phi_p = phi + chi + Eg

    J_n = sg_fluxes_n(phi_n, n, dx, D_n, V_T)     # (N-1,)
    J_p = sg_fluxes_p(phi_p, p, dx, D_p, V_T)     # (N-1,)

    R = total_recombination(
        n, p, params["ni_sq"], params["tau_n"], params["tau_p"],
        params["n1"], params["p1"], params["B_rad"], params["C_n"], params["C_p"]
    )

    # Dual-grid cell widths
    dx_cell = np.empty(len(x))
    dx_cell[0]    = dx[0]
    dx_cell[-1]   = dx[-1]
    dx_cell[1:-1] = 0.5 * (dx[:-1] + dx[1:])

    # Pad fluxes with zero-flux BCs at both boundaries
    J_n_full = np.concatenate([[0.0], J_n, [0.0]])
    J_p_full = np.concatenate([[0.0], J_p, [0.0]])

    dn =  (J_n_full[1:] - J_n_full[:-1]) / (Q * dx_cell) - R + G
    dp = -(J_p_full[1:] - J_p_full[:-1]) / (Q * dx_cell) - R + G

    dn[0] = dn[-1] = 0.0
    dp[0] = dp[-1] = 0.0
    return dn, dp
