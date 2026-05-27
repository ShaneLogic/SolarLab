"""Phase E3 — thermionic-emission flux primitive for interface-plane state.

Implements Pauwels-Vanhoutte 1978 (J.Phys.D 11, 649-667) eq 14a/b
translated to SolarLab's n+p heterojunction convention.

For each heterointerface k, four state-vec densities live on the
interface plane: (n_1s, p_1s, n_2s, p_2s). They couple to the bulk
nodes on each side via a thermionic-emission (TE) flux:

  J_TE = v_th_eff * (density_bulk_projected - density_state)

where ``density_bulk_projected`` is the equilibrium bulk density on the
appropriate side, Boltzmann-projected to the interface plane via the
local band-bending V_1 or V_2 partition of (V_bi - V_app).

Sign convention: positive flux means carriers flow INTO the interface-
plane state (state density grows). The Sprint 6 Day 8-10 work then
applies these fluxes as source terms on the dy/dt vector for the
iface_state block.

Units: m^-3 * m/s = m^-2 s^-1 (surface flux).

The default ``v_th_eff = 1.0e5`` m/s matches SCAPS Manual sec 3.8
typical thermal velocity (~ 10^7 cm/s). Per-layer override via the
``v_th_eff`` kwarg; future Sprint 8 will plumb this through the YAML
loader.
"""
from __future__ import annotations

import math

import numpy as np

from perovskite_sim.constants import V_T as _V_T_300
from perovskite_sim.physics.recombination import interface_recombination
from perovskite_sim.models.device import electrical_interfaces


_DEFAULT_V_TH_MS = 1.0e-2  # m/s — matches SCAPS-mirror surface recombination
                           # velocity scale to keep iface_state ODE timescale
                           # comparable to the existing carrier transient
                           # (tau ~ dx_iface / v_th ~ 1e-9/1e-2 = 1e-7 s).
                           # Full thermal velocity (1e5 m/s) makes Newton
                           # diverge at the diode knee; Sprint 7 Day 4+
                           # will revisit with QSS algebraic reduction.
_EXP_CAP = 30.0           # cap exponent to avoid overflow at V_bi/V_T ~ 42


def te_flux(
    density_bulk_projected: float,
    density_state: float,
    v_th: float,
) -> float:
    """Pure-function TE flux primitive (paper eq 14 reduced).

    Returns surface flux INTO the interface-plane state [m^-2 s^-1].
    Positive when bulk-projected > state (state fills); negative when
    state > bulk-projected (state drains).
    """
    return float(v_th) * (float(density_bulk_projected) - float(density_state))


def compute_interface_te_fluxes(
    mat,
    iface_state: np.ndarray,
    V_app: float = 0.0,
    *,
    v_th_eff: float = _DEFAULT_V_TH_MS,
    v_cross_eff: float = 0.0,
    V_T: float | None = None,
) -> np.ndarray:
    """Compute all 4*N_iface TE fluxes for one RHS evaluation.

    Per interface k, the block layout is (n_1s, p_1s, n_2s, p_2s)
    matching ``StateVec.iface_state`` and ``_compute_iface_state_dark_eq``.

    Bulk-projected densities use cached equilibrium values plus the
    V_app-dependent band-bending partition:

      V_total = V_bi_eff - V_app  (clamped to >= 0)
      V_2     = partition_left[k] * V_total       # PVK (light) side
      V_1     = (1 - partition_left[k]) * V_total # ETL (heavy) side

      n_1s_bulk_proj = n_R_eq * exp(-V_1 / V_T)
      p_1s_bulk_proj = p_R_eq * exp(+V_1 / V_T)
      n_2s_bulk_proj = n_L_eq * exp(+V_2 / V_T)
      p_2s_bulk_proj = p_L_eq * exp(-V_2 / V_T)

    At dark equilibrium (V_app = 0, iface_state = dark-eq init) the
    bulk-projected equals the state on every entry -> all fluxes = 0.

    Returns array shape (4 * N_iface,).
    """
    n_iface = len(mat.interface_V_partition_2)
    if n_iface == 0:
        return np.zeros(0, dtype=float)
    V_T_local = V_T if V_T is not None else (
        mat.V_T_device if hasattr(mat, "V_T_device") else _V_T_300
    )
    V_total = max(0.0, float(mat.V_bi_eff) - float(V_app))
    out = np.zeros(4 * n_iface, dtype=float)
    for k in range(n_iface):
        partition_left = float(mat.interface_V_partition_2[k])
        V_2 = partition_left * V_total
        V_1 = (1.0 - partition_left) * V_total
        v1_norm = max(-_EXP_CAP, min(_EXP_CAP, V_1 / V_T_local))
        v2_norm = max(-_EXP_CAP, min(_EXP_CAP, V_2 / V_T_local))
        n_R = float(mat.interface_n_R_eq[k])
        p_R = float(mat.interface_p_R_eq[k])
        n_L = float(mat.interface_n_L_eq[k])
        p_L = float(mat.interface_p_L_eq[k])
        n_1s_proj = n_R * math.exp(-v1_norm)
        p_1s_proj = p_R * math.exp(+v1_norm)
        # Phase E3 Day 4-6 — χ-step-consistent 2s projection. Without
        # this, single-side Boltzmann from L bulk gives an unphysical
        # 2s_proj that ignores cross-interface equilibrium (n_2s_eq =
        # n_1s_eq · exp(-ΔE_c/V_T) at flat E_F). Falls back to legacy
        # single-side when chi_step cache is empty.
        if mat.interface_chi_step and len(mat.interface_chi_step) > k:
            dE_c = float(mat.interface_chi_step[k])
            dE_g = float(mat.interface_Eg_step[k])
            dE_v_local = dE_c - dE_g
            ec_n = max(-_EXP_CAP, min(_EXP_CAP, dE_c / V_T_local))
            ev_n = max(-_EXP_CAP, min(_EXP_CAP, dE_v_local / V_T_local))
            n_2s_proj = n_1s_proj * math.exp(-ec_n)
            p_2s_proj = p_1s_proj * math.exp(-ev_n)
        else:
            n_2s_proj = n_L * math.exp(+v2_norm)
            p_2s_proj = p_L * math.exp(-v2_norm)
        base = 4 * k
        n_1s = float(iface_state[base + 0])
        p_1s = float(iface_state[base + 1])
        n_2s = float(iface_state[base + 2])
        p_2s = float(iface_state[base + 3])
        out[base + 0] = te_flux(n_1s_proj, n_1s, v_th_eff)
        out[base + 1] = te_flux(p_1s_proj, p_1s, v_th_eff)
        out[base + 2] = te_flux(n_2s_proj, n_2s, v_th_eff)
        out[base + 3] = te_flux(p_2s_proj, p_2s, v_th_eff)
        # Phase E3 Day 4-6 — cross-interface χ-step TE flux (paper eq 15).
        # Electrons: equilibrium ratio n_1s/n_2s = exp(ΔE_c/V_T) where
        # ΔE_c = chi_R − chi_L. Cross-flux drives n_1s, n_2s toward this
        # ratio across the χ-step barrier.
        # Holes: equilibrium ratio p_1s/p_2s = exp(ΔE_v/V_T) where
        # ΔE_v = (E_v_R − E_v_L) = ΔE_c − (Eg_R − Eg_L).
        if (
            v_cross_eff > 0.0
            and mat.interface_chi_step
            and len(mat.interface_chi_step) > k
        ):
            dE_c = float(mat.interface_chi_step[k])  # eV
            dE_g = float(mat.interface_Eg_step[k])   # eV
            dE_v = dE_c - dE_g
            # Exponent caps for numerical stability.
            ec_norm = max(-_EXP_CAP, min(_EXP_CAP, dE_c / V_T_local))
            ev_norm = max(-_EXP_CAP, min(_EXP_CAP, dE_v / V_T_local))
            # Cross-flux: positive value flows from 2s side to 1s side.
            J_cross_n = v_cross_eff * (
                n_2s * math.exp(ec_norm) - n_1s
            )
            J_cross_p = v_cross_eff * (
                p_2s * math.exp(ev_norm) - p_1s
            )
            # Mass-conserving redistribution: +J on 1s side, -J on 2s side.
            out[base + 0] += J_cross_n      # n_1s gains
            out[base + 2] -= J_cross_n      # n_2s loses
            out[base + 1] += J_cross_p      # p_1s gains
            out[base + 3] -= J_cross_p      # p_2s loses
    return out


def compute_interface_srh_on_state(
    iface_state: np.ndarray,
    stack,
    mat,
) -> np.ndarray:
    """Two-sided Shockley-Read SRH evaluated on interface-plane state.

    Paper eq 12, 13 translated to SolarLab n+p convention.

    Per interface k, four state densities (n_1s, p_1s, n_2s, p_2s) pair
    up into two SRH paths:
      R_s1 = SRH(n_1s, p_2s, ...)  # ETL electron + PVK hole
      R_s2 = SRH(n_2s, p_1s, ...)  # PVK electron + ETL hole

    Sinks (negative contributions to dy/dt) on each state density:
      d(n_1s)/dt -= R_s1
      d(p_2s)/dt -= R_s1   # paired with n_1s
      d(n_2s)/dt -= R_s2
      d(p_1s)/dt -= R_s2   # paired with n_2s

    Returns array shape (4 * N_iface,) of NEGATIVE sink magnitudes (each
    entry = -R_s for the carrier consumed in its pair). Caller adds this
    array to the diface_state block to apply the loss.

    Uses cached per-interface (n_1, p_1, ni_eff_sq, calibration_factor)
    from MaterialArrays. Surface velocities (v_n, v_p) read from
    stack.interfaces, scaled by calibration_factor (Phase E1.6 pattern).
    """
    ifaces = electrical_interfaces(stack)
    n_iface = len(mat.interface_V_partition_2)
    out = np.zeros(4 * n_iface, dtype=float)
    if n_iface == 0 or not ifaces:
        return out
    for k in range(n_iface):
        if k >= len(ifaces):
            break
        v_n, v_p = ifaces[k]
        if mat.interface_calibration_factor:
            cf = float(mat.interface_calibration_factor[k])
            v_n = v_n * cf
            v_p = v_p * cf
        if v_n == 0.0 and v_p == 0.0:
            continue
        n1_k = float(mat.interface_n1[k])
        p1_k = float(mat.interface_p1[k])
        # χ-step-consistent ni_eff² for iface-plane SRH path. At equilibrium
        # n_1s_eq · p_2s_eq = n_R_eq · p_R_eq · exp(-ΔE_v/V_T) (detailed
        # balance across the χ step). Falls back to legacy cached value
        # when chi_step cache is empty.
        if (
            mat.interface_chi_step
            and len(mat.interface_chi_step) > k
            and mat.interface_n_R_eq
        ):
            dE_c_k = float(mat.interface_chi_step[k])
            dE_g_k = float(mat.interface_Eg_step[k])
            dE_v_k = dE_c_k - dE_g_k
            V_T_iface = mat.V_T_device if hasattr(mat, "V_T_device") else _V_T_300
            ev_n_k = max(-_EXP_CAP, min(_EXP_CAP, dE_v_k / V_T_iface))
            n_R_eq = float(mat.interface_n_R_eq[k])
            p_R_eq = float(mat.interface_p_R_eq[k])
            ni_eff_sq = n_R_eq * p_R_eq * math.exp(-ev_n_k)
        else:
            ni_eff_sq = (
                float(mat.interface_ni_sq_eff[k])
                if mat.interface_ni_sq_eff else 0.0
            )
        base = 4 * k
        n_1s = float(iface_state[base + 0])
        p_1s = float(iface_state[base + 1])
        n_2s = float(iface_state[base + 2])
        p_2s = float(iface_state[base + 3])
        # Clamp negatives (defensive against transient overshoots).
        n_1s = max(0.0, n_1s); p_1s = max(0.0, p_1s)
        n_2s = max(0.0, n_2s); p_2s = max(0.0, p_2s)
        # R_s1: ETL-side electron capture paired with PVK-side hole.
        R_s1 = interface_recombination(
            n_1s, p_2s, ni_eff_sq, n1_k, p1_k, v_n, v_p,
        )
        # R_s2: PVK-side electron capture paired with ETL-side hole.
        R_s2 = interface_recombination(
            n_2s, p_1s, ni_eff_sq, n1_k, p1_k, v_n, v_p,
        )
        # Sinks (carrier consumed -> negative dy/dt contribution).
        out[base + 0] = -R_s1   # n_1s
        out[base + 1] = -R_s2   # p_1s
        out[base + 2] = -R_s2   # n_2s
        out[base + 3] = -R_s1   # p_2s
    return out
