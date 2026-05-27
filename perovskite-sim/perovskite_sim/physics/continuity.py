from __future__ import annotations
import numpy as np
from perovskite_sim.discretization.fe_operators import (
    bernoulli, sg_fluxes_n, sg_fluxes_p,
)
from perovskite_sim.physics.recombination import total_recombination

Q = 1.602176634e-19


def split_interface_flux(
    *,
    n_idx: float, n_idx_plus_1: float,
    n_L_iface: float, n_R_iface: float,
    phi_idx: float, phi_idx_plus_1: float, phi_iface: float,
    chi_L: float, chi_R: float,
    D_L: float, D_R: float,
    dx_face: float,
    V_T: float,
) -> tuple[float, float]:
    """Phase E4 — per-layer half-flux at a heterointerface face.

    Splits the legacy chi/Eg-aware SG flux into TWO half-fluxes coupled
    to the interface-plane state densities (n_L_iface, n_R_iface):

      J_L_half (idx → iface plane on L side):
        uses chi_L on both ends; dx_half = dx_face / 2
        xi_L = (phi_iface − phi_idx) / V_T    (chi cancels: both sides chi_L)
        J_L = q · D_L / dx_half · (B(xi_L) · n_L_iface − B(−xi_L) · n_idx)

      J_R_half (iface plane on R side → idx+1):
        uses chi_R on both ends; dx_half = dx_face / 2
        xi_R = (phi_idx_plus_1 − phi_iface) / V_T
        J_R = q · D_R / dx_half · (B(xi_R) · n_idx_plus_1 − B(−xi_R) · n_R_iface)

    The χ step lives OUTSIDE the half-fluxes — Sprint 7 Day 4-6
    cross-interface TE flux handles n_L_iface ↔ n_R_iface coupling via
    exp(−ΔE_c/V_T) factor.

    Sign convention matches the legacy sg_fluxes_n: positive flux flows
    from lower-index node toward higher-index node (idx → idx+1).

    Returns (J_L_half, J_R_half) in A/m².
    """
    dx_half = dx_face / 2.0
    # L-half: chi_L is constant on both ends → SG xi uses just phi diff.
    xi_L = (phi_iface - phi_idx) / V_T
    B_pos_L = float(bernoulli(np.array([xi_L]))[0])
    B_neg_L = float(bernoulli(np.array([-xi_L]))[0])
    J_L = float(
        Q * D_L / dx_half * (B_pos_L * n_L_iface - B_neg_L * n_idx)
    )
    # R-half: chi_R constant on both ends.
    xi_R = (phi_idx_plus_1 - phi_iface) / V_T
    B_pos_R = float(bernoulli(np.array([xi_R]))[0])
    B_neg_R = float(bernoulli(np.array([-xi_R]))[0])
    J_R = float(
        Q * D_R / dx_half * (B_pos_R * n_idx_plus_1 - B_neg_R * n_R_iface)
    )
    return J_L, J_R


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

    By default both boundary nodes are Dirichlet-pinned (dn/dp returned as 0
    at indices 0 and -1). Selective / Schottky contacts (Phase 3.3 — Apr
    2026) replace that pin with a Robin-type flux on a per-carrier,
    per-side basis. Callers opt in by stashing any of the four optional
    keys ``J_n_L`` / ``J_n_R`` / ``J_p_L`` / ``J_p_R`` in ``params``. When
    a key is present, its value is used as the boundary flux in place of
    the zero-flux pad and the matching dn/dp entry is allowed to evolve.
    Missing or ``None`` keys keep the pre-3.3 Dirichlet pin. The flux
    signs follow the convention used by
    :func:`perovskite_sim.physics.contacts.selective_contact_flux` so that
    ``S → ∞`` recovers the ohmic limit and ``S = 0`` the blocking limit.
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

    # Thermionic emission capping at heterointerfaces
    interface_faces = params.get("interface_faces")
    if interface_faces:
        from perovskite_sim.discretization.fe_operators import thermionic_emission_flux
        A_star_n_arr = params["A_star_n"]
        A_star_p_arr = params["A_star_p"]
        T_val = params["T"]
        # Ensure flux arrays are writable (they may be views)
        J_n = J_n.copy()
        J_p = J_p.copy()
        for f_idx in interface_faces:
            # Electron CB offset. Convention: E_c = E_vac - chi, so the
            # energy step from left to right is chi_left - chi_right (not
            # chi_right - chi_left). A *negative* delta_Ec means the CB goes
            # DOWN left->right, i.e. electrons flow downhill with no barrier
            # — exactly what spiro -> MAPbI3 looks like, where the SG flux
            # must pass through unchanged. The previous sign inversion was
            # turning every downhill step into a "barrier" and capping the
            # diode injection current at Richardson * exp(-|DeltaE|/kT) ~ 0.
            delta_Ec = chi[f_idx] - chi[f_idx + 1]
            if abs(delta_Ec) > 0.05:
                J_te_n = thermionic_emission_flux(
                    float(n[f_idx]), float(n[f_idx + 1]), float(delta_Ec), T_val,
                    float(A_star_n_arr[f_idx]),
                )
                if abs(J_n[f_idx]) > abs(J_te_n):
                    J_n[f_idx] = J_te_n
            # Hole VB offset. E_v = E_vac - chi - Eg, so
            # E_v_right - E_v_left = (chi_left + Eg_left) - (chi_right + Eg_right),
            # which is what's written below — the VB sign was already correct.
            delta_Ev = (chi[f_idx] + Eg[f_idx]) - (chi[f_idx + 1] + Eg[f_idx + 1])
            if abs(delta_Ev) > 0.05:
                J_te_p = thermionic_emission_flux(
                    float(p[f_idx]), float(p[f_idx + 1]), float(delta_Ev), T_val,
                    float(A_star_p_arr[f_idx]),
                )
                if abs(J_p[f_idx]) > abs(J_te_p):
                    J_p[f_idx] = J_te_p

    R = total_recombination(
        n, p, params["ni_sq"], params["tau_n"], params["tau_p"],
        params["n1"], params["p1"], params["B_rad"], params["C_n"], params["C_p"]
    )

    # Dual-grid cell widths
    dx_cell = np.empty(len(x))
    dx_cell[0]    = dx[0]
    dx_cell[-1]   = dx[-1]
    dx_cell[1:-1] = 0.5 * (dx[:-1] + dx[1:])

    # Boundary flux treatment. By default pad with zero (Dirichlet pin
    # overrides dn/dp anyway). Selective / Schottky contacts supply a
    # Robin flux per carrier / per side; any key left as None keeps the
    # original pin.
    J_n_L_val = params.get("J_n_L")
    J_n_R_val = params.get("J_n_R")
    J_p_L_val = params.get("J_p_L")
    J_p_R_val = params.get("J_p_R")

    J_n_full = np.concatenate(
        [[0.0 if J_n_L_val is None else float(J_n_L_val)], J_n,
         [0.0 if J_n_R_val is None else float(J_n_R_val)]]
    )
    J_p_full = np.concatenate(
        [[0.0 if J_p_L_val is None else float(J_p_L_val)], J_p,
         [0.0 if J_p_R_val is None else float(J_p_R_val)]]
    )

    dn =  (J_n_full[1:] - J_n_full[:-1]) / (Q * dx_cell) - R + G
    dp = -(J_p_full[1:] - J_p_full[:-1]) / (Q * dx_cell) - R + G

    # Dirichlet pins for any boundary/carrier without a Robin flux set.
    if J_n_L_val is None:
        dn[0] = 0.0
    if J_n_R_val is None:
        dn[-1] = 0.0
    if J_p_L_val is None:
        dp[0] = 0.0
    if J_p_R_val is None:
        dp[-1] = 0.0
    return dn, dp
