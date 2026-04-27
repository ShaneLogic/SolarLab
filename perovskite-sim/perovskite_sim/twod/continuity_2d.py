from __future__ import annotations
import numpy as np

from perovskite_sim.constants import Q
from perovskite_sim.twod.flux_2d import sg_fluxes_2d_n, sg_fluxes_2d_p


def continuity_rhs_2d(
    x: np.ndarray, y: np.ndarray,
    phi: np.ndarray, n: np.ndarray, p: np.ndarray,
    G: np.ndarray, R: np.ndarray,
    D_n: np.ndarray, D_p: np.ndarray,
    V_T: float,
    *,
    chi: np.ndarray | None = None,
    Eg: np.ndarray | None = None,
    lateral_bc: str = "periodic",
    interface_y_faces: tuple[int, ...] = (),
    A_star_n: np.ndarray | None = None,
    A_star_p: np.ndarray | None = None,
    T: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return dn/dt, dp/dt on shape (Ny, Nx).

    Computes -(div J) / Q + G - R per interior node. Boundary nodes in y are
    NOT pinned here (caller applies Dirichlet). Lateral BC controls how the
    x-boundary nodes (i=0 and i=Nx-1) get their flux contributions:
      - "periodic" — adds the wrap-around face flux between (i=Nx-1) and (i=0)
      - "neumann"  — zero-flux at i=0 and i=Nx-1 (no wrap face)

    Sign convention matches the 1D carrier_continuity_rhs in physics/continuity.py.

    Electron flux convention (sg_fluxes_2d_n):
      Jx_n[j, i] = flux at face between nodes i and i+1; positive = +x direction.
    Hole flux convention (sg_fluxes_2d_p):
      Jx_p[j, i] = flux at face between nodes i and i+1; positive = flux from
      i+1 toward i (opposite to electron convention, matching 1D SG for holes).

    Divergence for electrons at interior node i:
      div_x_n[:, i] = (Jx_n[:, i] - Jx_n[:, i-1]) / hx_cell[i]
    Divergence for holes at interior node i:
      div_x_p[:, i] = (Jx_p[:, i-1] - Jx_p[:, i]) / hx_cell[i]
      (sign flipped because Jx_p direction is reversed)

    Wait — we use a unified divergence formula. Let's be precise:
    The continuity equation is dn/dt = -(1/q) * div(J_n) + G - R
    where J_n is the electron current density (A/m^2) flowing in the +x, +y
    directions. The SG flux Jx_n[j, i] IS the current crossing the face between
    node i and node i+1, with positive = rightward. So:
      (1/q) * div_x_n at node i = (Jx_n[i] - Jx_n[i-1]) / (q * hx_cell[i])
    and dn/dt = -(1/q) * div_x_n + ... = -(Jx_n[i] - Jx_n[i-1]) / (Q * hx_cell[i])

    For holes, J_p convention in the 1D SG is that Jx_p[i] is positive when
    holes flow from node i+1 toward node i (i.e., in the -x direction). This
    matches the perovskite-sim convention: holes drift/diffuse opposite to E.
    The hole continuity is dp/dt = -(1/q) * div(J_p) + G - R but J_p is
    defined with the reversed sign, so:
      (1/q) * div_x_p at node i = (Jx_p[i-1] - Jx_p[i]) / (q * hx_cell[i])
    Equivalently this equals -(Jx_p[i] - Jx_p[i-1]) / (q * hx_cell[i]).

    We unify by defining divergence as (J_right - J_left) / hx for both
    carriers but with the SIGNED flux including the direction convention.
    Since sg_fluxes_2d_p already encodes the reversed direction in its sign,
    we can write a single formula:
      div_x[i] = (Jx[i] - Jx[i-1]) / hx_cell[i]
    and then dn/dt = -div_x_n / Q + G - R, dp/dt = -div_x_p / Q + G - R.

    Verification: for holes at interior i with no fields (pure diffusion, uniform
    p), Jx_p[i] = Jx_p[i-1], so div_x_p = 0. Correct.
    """
    if chi is None:
        phi_n = phi
        phi_p = phi
    else:
        phi_n = phi + chi
        phi_p = phi + chi + Eg

    Jx_n, Jy_n = sg_fluxes_2d_n(phi_n, n, x, y, D_n, V_T)   # Jx (Ny, Nx-1), Jy (Ny-1, Nx)
    Jx_p, Jy_p = sg_fluxes_2d_p(phi_p, p, x, y, D_p, V_T)

    # ---- Thermionic-emission capping at heterointerface y-faces -------------
    # Mirrors physics/continuity.py:carrier_continuity_rhs. When chi/Eg vary
    # across a y-face, the SG flux can be wildly larger than the Richardson-
    # Dushman thermionic-emission limit because the band offset compresses
    # into a single grid spacing. Without this cap the 2D RHS at a 1D-
    # converged steady state evaluates to ~1e31 instead of ~0.
    if interface_y_faces and chi is not None and Eg is not None and T is not None:
        Jy_n = Jy_n.copy()
        Jy_p = Jy_p.copy()
        T_sq = T * T
        for f in interface_y_faces:
            # chi varies in y only (Stage A); take column 0 for the band-offset
            # scalar but apply the cap vectorised across all i columns.
            dEc = float(chi[f, 0] - chi[f + 1, 0])
            if abs(dEc) > 0.05:
                left_term = n[f, :] * np.exp(-max(dEc, 0.0) / V_T)
                right_term = n[f + 1, :] * np.exp(-max(-dEc, 0.0) / V_T)
                # A_star_n is (Ny, Nx) extruded; left side is row f.
                J_te_n = A_star_n[f, :] * T_sq * (left_term - right_term)
                # SG sign convention for J_n is +y direction = positive; TE
                # also returns +y so we cap on |·|.
                mask = np.abs(Jy_n[f, :]) > np.abs(J_te_n)
                Jy_n[f, mask] = J_te_n[mask]
            dEv = float(
                (chi[f, 0] + Eg[f, 0]) - (chi[f + 1, 0] + Eg[f + 1, 0])
            )
            if abs(dEv) > 0.05:
                left_term = p[f, :] * np.exp(-max(dEv, 0.0) / V_T)
                right_term = p[f + 1, :] * np.exp(-max(-dEv, 0.0) / V_T)
                J_te_p = A_star_p[f, :] * T_sq * (left_term - right_term)
                mask = np.abs(Jy_p[f, :]) > np.abs(J_te_p)
                Jy_p[f, mask] = J_te_p[mask]

    Ny, Nx = phi.shape
    dx = np.diff(x)
    dy = np.diff(y)

    # Dual-cell widths (per node; consistent with poisson_2d)
    hx_cell = np.empty(Nx)
    hy_cell = np.empty(Ny)
    if lateral_bc == "periodic":
        # For periodic BC, boundary cells share the wrap face on both sides.
        # The wrap face spacing is (dx[-1] + dx[0]) / 2, so the dual-cell
        # width at i=0 and i=Nx-1 each spans half that face plus half the
        # adjacent interior face.
        hx_cell[0] = 0.5 * (dx[0] + dx[-1])
        hx_cell[-1] = 0.5 * (dx[0] + dx[-1])
    else:
        hx_cell[0] = dx[0] / 2.0
        hx_cell[-1] = dx[-1] / 2.0
    if Nx > 2:
        hx_cell[1:-1] = 0.5 * (dx[:-1] + dx[1:])

    hy_cell[0] = dy[0] / 2.0
    hy_cell[-1] = dy[-1] / 2.0
    if Ny > 2:
        hy_cell[1:-1] = 0.5 * (dy[:-1] + dy[1:])

    # -----------------------------------------------------------------------
    # x-divergence for interior nodes 1 <= i <= Nx-2:
    #   div_x[j, i] = (Jx[j, i] - Jx[j, i-1]) / hx_cell[i]
    # Boundary nodes handled separately per lateral_bc.
    # -----------------------------------------------------------------------
    div_x_n = np.zeros_like(phi)
    div_x_p = np.zeros_like(phi)
    if Nx > 2:
        div_x_n[:, 1:-1] = (Jx_n[:, 1:] - Jx_n[:, :-1]) / hx_cell[None, 1:-1]
        div_x_p[:, 1:-1] = (Jx_p[:, 1:] - Jx_p[:, :-1]) / hx_cell[None, 1:-1]

    if lateral_bc == "periodic":
        # Wrap-around face between node i=Nx-1 (left end of wrap) and i=0 (right end).
        # Convention: Jx_wrap_n > 0 means flux flows from Nx-1 toward 0 (wrapping +x).
        # This is the same SG formula as Jx_n[j, i] but with nodes Nx-1 and 0.
        from perovskite_sim.discretization.fe_operators import bernoulli as _B

        dx_wrap = 0.5 * (dx[0] + dx[-1])

        # Electrons: B(xi)*n[right] - B(-xi)*n[left], right=0, left=Nx-1
        # Harmonic-mean face avg matches the interior sg_fluxes_2d_n averaging.
        _eps_face = 1e-300
        D_face_wrap_n = 2.0 * D_n[:, -1] * D_n[:, 0] / (D_n[:, -1] + D_n[:, 0] + _eps_face)
        xi_wrap_n = (phi_n[:, 0] - phi_n[:, -1]) / V_T
        Jx_wrap_n = (Q * D_face_wrap_n / dx_wrap) * (
            _B(xi_wrap_n) * n[:, 0] - _B(-xi_wrap_n) * n[:, -1]
        )

        # Holes: same phi gradient, but Bernoulli terms swap n[left/right]:
        # Jx_p convention: positive = flux from i+1 toward i (rightward→leftward).
        # For the wrap face from Nx-1 to 0: positive means from 0 toward Nx-1.
        # This matches the interior sg_fluxes_2d_p convention:
        #   Jx_p[j, i] = Q*D/dx * (B(xi)*p[i] - B(-xi)*p[i+1])
        # For wrap (left=Nx-1, right=0):
        #   Jx_wrap_p = Q*D/dx * (B(xi)*p[Nx-1] - B(-xi)*p[0])
        # where xi = (phi_p[0] - phi_p[Nx-1]) / V_T (same as for electrons)
        D_face_wrap_p = 2.0 * D_p[:, -1] * D_p[:, 0] / (D_p[:, -1] + D_p[:, 0] + _eps_face)
        xi_wrap_p = (phi_p[:, 0] - phi_p[:, -1]) / V_T
        Jx_wrap_p = (Q * D_face_wrap_p / dx_wrap) * (
            _B(xi_wrap_p) * p[:, -1] - _B(-xi_wrap_p) * p[:, 0]
        )

        # Node i=0:
        #   div_x_n[0] = (Jx_n[0] - Jx_wrap_n) / hx_cell[0]
        #     Jx_n[0] = outgoing rightward flux from node 0
        #     Jx_wrap_n = incoming leftward flux (wrap from Nx-1 to 0)
        #   div_x_p[0] = (Jx_p[0] - Jx_wrap_p) / hx_cell[0]
        #     Jx_p[0] = flux at face between 0 and 1 (leftward convention)
        #     Jx_wrap_p = flux at wrap face (leftward convention, from 0 to Nx-1)
        div_x_n[:, 0] = (Jx_n[:, 0] - Jx_wrap_n) / hx_cell[0]
        div_x_p[:, 0] = (Jx_p[:, 0] - Jx_wrap_p) / hx_cell[0]

        # Node i=Nx-1:
        #   div_x_n[-1] = (Jx_wrap_n - Jx_n[-1]) / hx_cell[-1]
        #     Jx_wrap_n = outgoing from Nx-1 to 0 (wrap)
        #     Jx_n[-1] = Jx_n[:, Nx-2] = incoming from Nx-2 to Nx-1
        #   div_x_p[-1] = (Jx_wrap_p - Jx_p[-1]) / hx_cell[-1]
        #     Jx_wrap_p = flux at wrap face (leftward, from 0 toward Nx-1)
        #     Jx_p[-1] = flux at face Nx-2→Nx-1 (leftward convention)
        div_x_n[:, -1] = (Jx_wrap_n - Jx_n[:, -1]) / hx_cell[-1]
        div_x_p[:, -1] = (Jx_wrap_p - Jx_p[:, -1]) / hx_cell[-1]
    else:
        # Neumann: zero flux outside the domain boundaries.
        # Node i=0: only rightward face Jx[:,0] contributes; left flux = 0.
        # Node i=Nx-1: only incoming from Jx[:,-1]; right flux = 0.
        div_x_n[:, 0] = Jx_n[:, 0] / hx_cell[0]
        div_x_p[:, 0] = Jx_p[:, 0] / hx_cell[0]
        div_x_n[:, -1] = -Jx_n[:, -1] / hx_cell[-1]
        div_x_p[:, -1] = -Jx_p[:, -1] / hx_cell[-1]

    # -----------------------------------------------------------------------
    # y-divergence on interior rows (j=1..Ny-2). j=0 and j=Ny-1 are Dirichlet
    # contacts — caller pins them to zero so we skip them here.
    # -----------------------------------------------------------------------
    div_y_n = np.zeros_like(phi)
    div_y_p = np.zeros_like(phi)
    if Ny > 2:
        div_y_n[1:-1, :] = (Jy_n[1:, :] - Jy_n[:-1, :]) / hy_cell[1:-1, None]
        div_y_p[1:-1, :] = (Jy_p[1:, :] - Jy_p[:-1, :]) / hy_cell[1:-1, None]

    # Sign convention matches 1D physics/continuity.py:carrier_continuity_rhs.
    # For electrons (charge −q), J_n is the electric current density and the
    # continuity is dn/dt = +∇·J_n / q + G − R: a positive divergence means
    # current flows OUT, which is electron particles flowing IN, raising n.
    # For holes (charge +q), dp/dt = −∇·J_p / q + G − R.
    dn =  (div_x_n + div_y_n) / Q + G - R
    dp = -(div_x_p + div_y_p) / Q + G - R
    return dn, dp
