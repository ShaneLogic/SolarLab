"""Shared one-dimensional thermionic magnitude cap for carrier face currents."""

from __future__ import annotations

import numpy as np


def thermionic_normalization_status(params: dict) -> str:
    """Label physical normalization separately from historical DOS fallback."""
    faces = tuple(params.get("interface_faces", ()))
    if not faces:
        return "disabled"
    if not params.get("te_physical_norm", False):
        return "legacy_density_weighted"
    physical = []
    for name in ("N_C_node", "N_V_node"):
        values = params.get(name)
        if values is None:
            physical.append(False)
            continue
        selected = np.asarray(values)[np.asarray([(face, face + 1) for face in faces])]
        physical.extend((np.isfinite(selected) & (selected > 0.0)).ravel())
    if all(physical):
        return "physical_dos_normalized"
    return "legacy_fallback_missing_dos"


def apply_thermionic_caps(
    J_n: np.ndarray,
    J_p: np.ndarray,
    n: np.ndarray,
    p: np.ndarray,
    params: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the declared cap without changing the input current arrays."""
    chi = params.get("chi")
    Eg = params.get("Eg")
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
        te_regularization_mode = params.get("te_regularization_mode", "legacy_logistic")

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
            t_arg = (a_sg - a_te) / (te_soft * (a_sg + a_te) + 1e-300)
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
                    float(n[f_idx]),
                    float(n[f_idx + 1]),
                    float(delta_Ec),
                    T_val,
                    float(A_star_n_arr[f_idx]),
                    N_dos=(_face_dos(_N_C, f_idx) if _te_phys else None),
                )
                J_n[f_idx] = _cap(float(J_n[f_idx]), float(J_te_n))
            # Hole VB offset. E_v = E_vac - chi - Eg, so
            # E_v_right - E_v_left = (chi_left + Eg_left) - (chi_right + Eg_right),
            # which is what's written below — the VB sign was already correct.
            delta_Ev = (chi_te[f_idx] + Eg_te[f_idx]) - (
                chi_te[f_idx + 1] + Eg_te[f_idx + 1]
            )
            if abs(delta_Ev) > 0.05:
                J_te_p = thermionic_emission_flux(
                    float(p[f_idx]),
                    float(p[f_idx + 1]),
                    float(delta_Ev),
                    T_val,
                    float(A_star_p_arr[f_idx]),
                    N_dos=(_face_dos(_N_V, f_idx) if _te_phys else None),
                )
                J_p[f_idx] = _cap(float(J_p[f_idx]), float(J_te_p))
    return J_n, J_p
