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


def split_interface_flux_p(
    *,
    p_idx: float, p_idx_plus_1: float,
    p_L_iface: float, p_R_iface: float,
    phi_idx: float, phi_idx_plus_1: float, phi_iface: float,
    D_L: float, D_R: float,
    dx_face: float,
    V_T: float,
) -> tuple[float, float]:
    """Phase E4 — per-layer hole half-flux at a heterointerface face.

    Mirror of ``split_interface_flux`` for holes. Sign convention matches
    ``sg_fluxes_p``:

      J_L_p = q · D_L / dx_half · (B(xi_L) · p_idx − B(−xi_L) · p_L_iface)
      J_R_p = q · D_R / dx_half · (B(xi_R) · p_R_iface − B(−xi_R) · p_idx_plus_1)

    Within a single layer chi + Eg is constant, so xi simplifies to
    (phi_iface − phi_idx) / V_T on each side (same as electron half-flux).

    Returns (J_L_p_half, J_R_p_half) in A/m².
    """
    dx_half = dx_face / 2.0
    xi_L = (phi_iface - phi_idx) / V_T
    B_pos_L = float(bernoulli(np.array([xi_L]))[0])
    B_neg_L = float(bernoulli(np.array([-xi_L]))[0])
    J_L = float(
        Q * D_L / dx_half * (B_pos_L * p_idx - B_neg_L * p_L_iface)
    )
    xi_R = (phi_idx_plus_1 - phi_iface) / V_T
    B_pos_R = float(bernoulli(np.array([xi_R]))[0])
    B_neg_R = float(bernoulli(np.array([-xi_R]))[0])
    J_R = float(
        Q * D_R / dx_half * (B_pos_R * p_R_iface - B_neg_R * p_idx_plus_1)
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
    D_n = params["D_n"]
    D_p = params["D_p"]
    V_T = params["V_T"]
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

    carrier_statistics = params.get(
        "carrier_statistics", "maxwell_boltzmann"
    )
    if carrier_statistics == "fermi_dirac":
        from perovskite_sim.physics.degenerate_transport import (
            generalized_sg_fluxes_n,
            generalized_sg_fluxes_p,
        )

        if params.get("degenerate_recombination_model") != "off":
            raise ValueError(
                "Fermi-Dirac bulk transport requires an explicit compatible "
                "recombination closure"
            )
        chi_statistics = params.get("chi_statistics", chi)
        Eg_statistics = params.get("Eg_statistics", Eg)
        phi_n_statistics = (
            phi
            if chi_statistics is None
            else phi + chi_statistics
        )
        phi_p_statistics = (
            phi
            if chi_statistics is None
            else phi + chi_statistics + Eg_statistics
        )
        J_n = generalized_sg_fluxes_n(
            phi_n_statistics,
            n,
            dx,
            np.asarray(D_n) / V_T,
            V_T,
            params["N_C"],
            statistics=carrier_statistics,
        )
        J_p = generalized_sg_fluxes_p(
            phi_p_statistics,
            p,
            dx,
            np.asarray(D_p) / V_T,
            V_T,
            params["N_V"],
            statistics=carrier_statistics,
        )
    else:
        J_n = sg_fluxes_n(phi_n, n, dx, D_n, V_T)     # (N-1,)
        J_p = sg_fluxes_p(phi_p, p, dx, D_p, V_T)     # (N-1,)

    # Thermionic emission capping at heterointerfaces
    interface_faces = params.get("interface_faces")
    if interface_faces:
        from perovskite_sim.discretization.fe_operators import thermionic_emission_flux
        A_star_n_arr = params["A_star_n"]
        A_star_p_arr = params["A_star_p"]
        T_val = params["T"]
        # Physical TE normalization (review F02). When on, divide the TE flux
        # at each capped face by the band-edge DOS (geometric mean of the two
        # face nodes — a single divisor scales both legs equally, preserving
        # equilibrium J=0). A face whose DOS is NaN (a layer without
        # Nc300/Nv300) falls back to the legacy density-weighted form, so
        # mixed / DOS-free configs stay bit-identical.
        _te_phys = params.get("te_physical_norm", False)
        _N_C = params.get("N_C_node")
        _N_V = params.get("N_V_node")

        def _face_dos(dos_arr, i):
            if dos_arr is None:
                return None
            a, b = dos_arr[i], dos_arr[i + 1]
            if not (np.isfinite(a) and np.isfinite(b) and a > 0.0 and b > 0.0):
                return None
            return float(np.sqrt(a * b))
        # The steady-state driver historically uses a logistic magnitude
        # blend at te_softness=0.02. Explicit Phase-1 regularization selects
        # the compact-support mode; keeping the modes distinct preserves the
        # legacy default while making a width-to-zero ladder interpretable.
        te_soft = params.get("te_softness", 0.0)
        te_regularization_mode = params.get(
            "te_regularization_mode", "legacy_logistic"
        )

        def _cap(J_sg, J_te):
            """Limit the MAGNITUDE of the SG flux; never change its direction.

            The thermionic bound says how much current the interface can
            emit, not which way it flows — that is set by the drift-diffusion
            solution. Returning ``J_te`` wholesale (as this did until
            2026-07-28) imports the sign of a separately-computed quantity,
            so whenever the two disagreed the "cap" REVERSED the flux instead
            of limiting it.

            Measured on scaps_mirror_v2 with ``te_physical_norm`` on, holes at
            the HTL/PVK face, at every bias probed (V = 0, 0.9, 1.2):

                J_sg = +1.30e7 ... +2.02e7      J_te = -2.33e2

            i.e. the dominant hole-extraction current was replaced by a
            reversed one five orders smaller. Carriers piled up behind the
            interface and the steady-state V_oc rose to 1.39 V, above the
            1.2535 V detailed-balance ceiling for this absorber gap — which
            is how the defect was found.

            Latent under the legacy density-weighted bound because that makes
            |J_te| ~ 1e28-1e35, so ``|J_sg| > |J_te|`` is essentially never
            true and the branch never ran: measured 0 binds in 80 checks
            across five shipped presets, which is why this fix is
            bit-identical on every current default.
            """
            a_sg, a_te = abs(J_sg), abs(J_te)
            if te_soft <= 0.0:
                return float(np.copysign(min(a_sg, a_te), J_sg))
            if te_regularization_mode == "compact_support":
                from perovskite_sim.physics.regularization import (
                    direction_preserving_magnitude_min,
                )

                return direction_preserving_magnitude_min(
                    J_sg,
                    J_te,
                    relative_width=te_soft,
                )
            if te_regularization_mode != "legacy_logistic":
                raise ValueError(
                    "te_regularization_mode must be 'legacy_logistic' or "
                    "'compact_support'"
                )
            t_arg = (a_sg - a_te) / (
                te_soft * (a_sg + a_te) + 1e-300
            )
            if t_arg > 40.0:
                weight = 1.0
            elif t_arg < -40.0:
                weight = 0.0
            else:
                weight = 1.0 / (1.0 + np.exp(-t_arg))
            magnitude = weight * a_te + (1.0 - weight) * a_sg
            return float(np.copysign(magnitude, J_sg))
        # Band edges for the THERMIONIC barrier. These are the physical ones
        # when the caller supplies them: with ``dos_band_potentials`` active,
        # ``chi``/``Eg`` above are transport potentials carrying
        # V_T·ln(N_C/N_C_ref), which makes the Scharfetter-Gummel flux correct
        # under Boltzmann statistics but is not a real energy shift. Thermionic
        # emission crosses the real band step. Measured on scaps_mirror_v2, the
        # folded valence step at the HTL/PVK face is +0.097 eV against a
        # physical +0.180 eV — nearly a factor two in the Boltzmann exponent.
        # Falls back to the folded arrays so hand-built params dicts still work.
        chi_te = params.get("chi_te")
        Eg_te = params.get("Eg_te")
        if chi_te is None:
            chi_te = chi
        if Eg_te is None:
            Eg_te = Eg
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
            delta_Ec = chi_te[f_idx] - chi_te[f_idx + 1]
            if abs(delta_Ec) > 0.05:
                J_te_n = thermionic_emission_flux(
                    float(n[f_idx]), float(n[f_idx + 1]), float(delta_Ec), T_val,
                    float(A_star_n_arr[f_idx]),
                    N_dos=(_face_dos(_N_C, f_idx) if _te_phys else None),
                )
                J_n[f_idx] = _cap(float(J_n[f_idx]), float(J_te_n))
            # Hole VB offset. E_v = E_vac - chi - Eg, so
            # E_v_right - E_v_left = (chi_left + Eg_left) - (chi_right + Eg_right),
            # which is what's written below — the VB sign was already correct.
            delta_Ev = ((chi_te[f_idx] + Eg_te[f_idx])
                        - (chi_te[f_idx + 1] + Eg_te[f_idx + 1]))
            if abs(delta_Ev) > 0.05:
                J_te_p = thermionic_emission_flux(
                    float(p[f_idx]), float(p[f_idx + 1]), float(delta_Ev), T_val,
                    float(A_star_p_arr[f_idx]),
                    N_dos=(_face_dos(_N_V, f_idx) if _te_phys else None),
                )
                J_p[f_idx] = _cap(float(J_p[f_idx]), float(J_te_p))

    # Opt-in algebraic interface-plane path. The four plane densities carry
    # the bulk-to-plane supply, cross-interface exchange, and surface SRH in
    # assemble_rhs. Leaving this continuous-material SG face active creates a
    # parallel bypass around that boundary and delays the CBO response. Zero
    # only the declared faces; terminal and intra-layer SG transport remain.
    exclusive_faces = params.get("exclusive_interface_faces")
    if exclusive_faces:
        J_n = J_n.copy()
        J_p = J_p.copy()
        for face in exclusive_faces:
            if 0 <= int(face) < len(J_n):
                J_n[int(face)] = 0.0
                J_p[int(face)] = 0.0

    # Heterointerface bulk-recombination de-spike (SCAPS-emulation, off by
    # default). The band offset produces a Boltzmann carrier spike at the
    # junction node that double-counts against the interface SRH channel;
    # blend those nodes' recomb density toward the geometric mean of the
    # neighbours by the de-spike fraction. Transport (n,p) is untouched —
    # only the density fed to the BULK recombination rate is corrected.
    _despike = params.get("het_recomb_despike", 0.0)
    if _despike > 0.0:
        n_rec = np.array(n, dtype=float, copy=True)
        p_rec = np.array(p, dtype=float, copy=True)
        for _i in params.get("het_recomb_nodes", ()):
            if 0 < _i < len(n_rec) - 1:
                nb_n = np.sqrt(max(n[_i - 1], 1.0) * max(n[_i + 1], 1.0))
                nb_p = np.sqrt(max(p[_i - 1], 1.0) * max(p[_i + 1], 1.0))
                n_rec[_i] = nb_n + (n[_i] - nb_n) * (1.0 - _despike)
                p_rec[_i] = nb_p + (p[_i] - nb_p) * (1.0 - _despike)
    else:
        n_rec, p_rec = n, p
    if params.get("degenerate_recombination_model") == "off":
        R = np.zeros_like(n_rec)
    else:
        R = total_recombination(
            n_rec, p_rec, params["ni_sq"], params["tau_n"], params["tau_p"],
            params["n1"], params["p1"], params["B_rad"], params["C_n"], params["C_p"],
            neutral_bulk_defects=params.get("neutral_bulk_defects"),
            monovalent_bulk_defects=params.get("monovalent_bulk_defects"),
            multivalent_bulk_defects=params.get("multivalent_bulk_defects"),
            frozen_metastable_defects=params.get("frozen_metastable_defects"),
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

    # Phase E4 — split-interface-flux divergence override at heterointerface
    # faces. When ``interface_split_data`` is supplied, the legacy
    # harmonic-mean SG flux at each heterointerface face is replaced with
    # two single-layer half-fluxes coupled to interface-plane state
    # densities (n_L_iface, n_R_iface, p_L_iface, p_R_iface). This is the
    # missing-physics piece for SCAPS parity (paper eq 14a TE BC).
    iface_split = params.get("interface_split_data")
    if iface_split is not None:
        iface_face_list = iface_split["iface_face_list"]
        iface_state = iface_split["iface_state"]
        D_n_node = iface_split["D_n_node"]
        D_p_node = iface_split["D_p_node"]
        for k, f in enumerate(iface_face_list):
            if f < 0 or f >= len(J_n):
                continue
            base = 4 * k
            # iface_state block layout (Sprint 6): n_1s, p_1s, n_2s, p_2s.
            # In our convention 1s = R side (ETL, idx+1); 2s = L side
            # (PVK, idx).
            n_R_iface = float(iface_state[base + 0])
            p_R_iface = float(iface_state[base + 1])
            n_L_iface = float(iface_state[base + 2])
            p_L_iface = float(iface_state[base + 3])
            # Interface plane phi: midpoint of bulk nodes (first-order
            # approximation; refine to charge-balance-derived value later).
            phi_iface = 0.5 * (phi[f] + phi[f + 1])
            chi_L = float(chi[f]) if chi is not None else 0.0
            chi_R = float(chi[f + 1]) if chi is not None else 0.0
            # Electron half-fluxes.
            J_L_n, J_R_n = split_interface_flux(
                n_idx=float(n[f]), n_idx_plus_1=float(n[f + 1]),
                n_L_iface=n_L_iface, n_R_iface=n_R_iface,
                phi_idx=float(phi[f]), phi_idx_plus_1=float(phi[f + 1]),
                phi_iface=phi_iface,
                chi_L=chi_L, chi_R=chi_R,
                D_L=float(D_n_node[f]), D_R=float(D_n_node[f + 1]),
                dx_face=float(dx[f]), V_T=V_T,
            )
            # Hole half-fluxes.
            J_L_p, J_R_p = split_interface_flux_p(
                p_idx=float(p[f]), p_idx_plus_1=float(p[f + 1]),
                p_L_iface=p_L_iface, p_R_iface=p_R_iface,
                phi_idx=float(phi[f]), phi_idx_plus_1=float(phi[f + 1]),
                phi_iface=phi_iface,
                D_L=float(D_p_node[f]), D_R=float(D_p_node[f + 1]),
                dx_face=float(dx[f]), V_T=V_T,
            )
            # Apply divergence-correction at idx (left bulk node):
            # legacy used J_n[f] as right-face flux; replace with J_L_n.
            J_n_legacy_f = float(J_n[f])
            J_p_legacy_f = float(J_p[f])
            dn[f] += (J_L_n - J_n_legacy_f) / (Q * dx_cell[f])
            dp[f] += -(J_L_p - J_p_legacy_f) / (Q * dx_cell[f])
            # At idx+1 (right bulk node): legacy used J_n[f] as left-face
            # flux; replace with J_R_n. Sign: dn = (right - left)/dx, so
            # replacing left-flux means subtracting delta.
            dn[f + 1] -= (J_R_n - J_n_legacy_f) / (Q * dx_cell[f + 1])
            dp[f + 1] -= -(J_R_p - J_p_legacy_f) / (Q * dx_cell[f + 1])
    return dn, dp
