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
            ec_norm = dE_c / V_T_local
            ev_norm = dE_v / V_T_local
            # Cross-flux (positive flows 2s → 1s) toward the detailed-balance
            # ratio n_1s/n_2s = exp(ΔE_c/V_T). Phase E8.4 — bounded form: the
            # legacy ``n_2s·exp(ec_norm) − n_1s`` carried exp(+|ΔE|/V_T), which
            # at a large offset (HTL/PVK ΔE_c=1.54 eV → exp(59), capped at
            # exp(30)≈1e13) produced the ~1e36 cross-flux that both hung the
            # Sprint-9 wire-through and flattened the CBO response under the
            # cap. Factor out the LARGER exponential so the surviving exp arg
            # is always ≤ 0 (bounded ≤ 1). Same zero-point, physical magnitude
            # (cross-barrier emission is Boltzmann-suppressed, not enhanced).
            if ec_norm >= 0.0:
                J_cross_n = v_cross_eff * (n_2s - n_1s * math.exp(-ec_norm))
            else:
                J_cross_n = v_cross_eff * (n_2s * math.exp(ec_norm) - n_1s)
            if ev_norm >= 0.0:
                J_cross_p = v_cross_eff * (p_2s - p_1s * math.exp(-ev_norm))
            else:
                J_cross_p = v_cross_eff * (p_2s * math.exp(ev_norm) - p_1s)
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


# =====================================================================
# QSS algebraic interface-plane closure (2026-06) — the "Sprint 7 Day 4+
# QSS algebraic reduction" anticipated above, finally built.
#
# Instead of carrying the four plane densities as ODE state (whose explicit
# TE coupling made Newton diverge at full thermal velocity and whose bounded
# cross-flux flattened the CBO response — see module header), the plane
# densities (n_s, p_s) are solved as a LOCAL steady-state flux balance per
# RHS call:
#
#     S*(nb - 2*n_s) = R(n_s, p_s)        nb = bn_L*n_L + bn_R*n_R
#     S*(pb - 2*p_s) = R(n_s, p_s)        pb = bp_L*p_L + bp_R*p_R
#
# with the P-V single-level rate on the plane densities. Plane band edges
# are the FAVOURABLE edges (electrons: lower E_C of the two sides; holes:
# higher E_V), so the plane gap Eg_s = E_C,s - E_V,s is the reduced
# interface gap — a conduction-band cliff shrinks Eg_s and raises ni_s^2
# exponentially (the SCAPS cliff mechanism). The b factors are the
# Boltzmann barrier penalties from each side's node (already projected to
# the plane potential by the caller) onto the favourable edge.
#
# Detailed balance is analytic: at dark equilibrium the SG zero-flux
# condition makes node ratios exactly Boltzmann, both sides' projected
# supplies agree, and n_s*p_s = ni_s^2 exactly -> R = 0 with no
# cross-product reference needed.
#
# Structural properties the bulk-node formulations lacked:
#   * supply limitation: R <= S*(nb + pb)/2 by construction (the measured
#     HTL/PVK six-decade SRV insensitivity emerges as saturation);
#   * trap-level visibility: plane minority densities are small enough
#     that n1_s/p1_s compete in the denominator (E_t responds);
#   * locally implicit: no sign-flipping explicit rates (the historical
#     Radau wall).
# =====================================================================

from dataclasses import dataclass


# Effusion supply velocity [m/s]: v_th/4 with the SCAPS default thermal
# velocity 1e7 cm/s = 1e5 m/s. Supply is huge except where a band offset
# blocks it, so results are weakly sensitive to the exact prefactor.
S_SUPPLY = 2.5e4

_QSS_MAX_NEWTON = 40
_QSS_TOL = 1.0e-12
_QSS_FLOOR = 1.0e-30


@dataclass(frozen=True)
class PlaneInterfaceParams:
    """Build-time constants of one defect interface's plane closure.

    Built from the POST-DOS-fold chi/Eg node arrays (the closure is part of
    the parity configuration and activates only under
    ``dos_band_potentials`` + reference-layer DOS data). Densities m^-3.
    """
    bn_L: float      # exp(-(chi_s - chi_L)/V_T) <= 1
    bn_R: float
    bp_L: float      # exp(-((chi+Eg)_L - (chi+Eg)_s)/V_T) <= 1
    bp_R: float
    ni_s_sq: float   # N_C,ref * N_V,ref * exp(-Eg_s/V_T)
    n1_s: float      # trap level vs plane edges; n1_s*p1_s == ni_s_sq
    p1_s: float


def build_plane_params(
    chi_L: float, chi_R: float, ceg_L: float, ceg_R: float,
    Nc_ref: float, Nv_ref: float, chi_ref: float, E_t_eV: float,
    V_T: float,
) -> PlaneInterfaceParams:
    """Derive the closure constants from folded band quantities.

    ``chi_i`` are the folded chi values at the two interface-adjacent nodes;
    ``ceg_i = chi_i + Eg_i`` (folded). ``chi_ref`` is the RAW reference-layer
    chi (the fold shift is zero on the reference layer, so the trap energy
    E_t below the reference CB needs no fold correction).
    """
    chi_s = max(chi_L, chi_R)            # electron edge: lower E_C
    ceg_s = min(ceg_L, ceg_R)            # hole edge: higher E_V
    eg_s = ceg_s - chi_s                 # reduced interface gap [eV]
    def _b(d):                           # Boltzmann penalty, d >= 0
        return math.exp(-min(max(d, 0.0), 60.0 * V_T) / V_T)
    depth_n = E_t_eV - (chi_s - chi_ref)  # trap depth below plane E_C [eV]
    # Clamp the level to the (reduced) plane gap: a trap pushed energetically
    # outside the gap by a band offset acts as a band-edge state — emission
    # cannot outrun the band-edge DOS. Without this, a deep cliff puts the
    # trap above the plane E_C and n1_s = N_C*exp(+|depth|/V_T) explodes,
    # suppressing the cliff recombination SCAPS resolves (measured: R 8
    # orders low at dE_C = -1.0, V_oc arm collapsed 734 -> 284 mV).
    depth_n = min(max(depth_n, 0.0), max(eg_s, 0.0))
    # exponentials guarded: |args| <= ~60 in practice (Eg/V_T ~ 60 worst)
    def _e(a):
        return math.exp(max(-120.0, min(120.0, a)))
    return PlaneInterfaceParams(
        bn_L=_b(chi_s - chi_L),
        bn_R=_b(chi_s - chi_R),
        bp_L=_b(ceg_L - ceg_s),
        bp_R=_b(ceg_R - ceg_s),
        ni_s_sq=Nc_ref * Nv_ref * _e(-eg_s / V_T),
        n1_s=Nc_ref * _e(-depth_n / V_T),
        p1_s=Nv_ref * _e(-(eg_s - depth_n) / V_T),
    )


def plane_rate(n_s: float, p_s: float, prm: PlaneInterfaceParams,
               v_n: float, v_p: float) -> float:
    """P-V single-level rate on plane densities [m^-2 s^-1], NOGEN-clamped."""
    num = n_s * p_s - prm.ni_s_sq
    if num <= 0.0:
        return 0.0
    den = (n_s + prm.n1_s) / v_p + (p_s + prm.p1_s) / v_n
    return num / den


def solve_plane_densities(
    n_L: float, n_R: float, p_L: float, p_R: float,
    prm: PlaneInterfaceParams, v_n: float, v_p: float,
    s_supply: float = S_SUPPLY,
) -> tuple[float, float, float]:
    """Solve the 2x2 plane closure; return ``(n_s, p_s, R)``.

    Inputs are the adjacent node densities already Boltzmann-projected to
    the plane POTENTIAL (caller applies the phi factors); the band-offset
    penalties live in ``prm``. Newton in (ln n_s, ln p_s) — positivity by
    construction, analytic Jacobian, damped steps. On the rare
    non-convergence the last iterate is used with the NOGEN-clamped rate:
    fail-bounded (R <= delivery flux), never fail-wild.
    """
    nb = prm.bn_L * max(n_L, 0.0) + prm.bn_R * max(n_R, 0.0)
    pb = prm.bp_L * max(p_L, 0.0) + prm.bp_R * max(p_R, 0.0)
    s2 = 2.0 * s_supply
    ln_n = math.log(max(nb / 2.0, _QSS_FLOOR))
    ln_p = math.log(max(pb / 2.0, _QSS_FLOOR))
    for _ in range(_QSS_MAX_NEWTON):
        n_s, p_s = math.exp(ln_n), math.exp(ln_p)
        num = n_s * p_s - prm.ni_s_sq
        den = (n_s + prm.n1_s) / v_p + (p_s + prm.p1_s) / v_n
        R = num / den if num > 0.0 else 0.0
        F1 = s_supply * nb - s2 * n_s - R
        F2 = s_supply * pb - s2 * p_s - R
        scale = s_supply * (nb + pb) + abs(R) + _QSS_FLOOR
        if abs(F1) + abs(F2) < _QSS_TOL * scale:
            break
        if num > 0.0:
            dR_dn = (p_s * den - num / v_p) / (den * den)
            dR_dp = (n_s * den - num / v_n) / (den * den)
        else:
            dR_dn = dR_dp = 0.0
        J11 = (-s2 - dR_dn) * n_s
        J12 = (-dR_dp) * p_s
        J21 = (-dR_dn) * n_s
        J22 = (-s2 - dR_dp) * p_s
        det = J11 * J22 - J12 * J21
        if det == 0.0 or not math.isfinite(det):
            break
        d_ln_n = (-F1 * J22 + F2 * J12) / det
        d_ln_p = (-F2 * J11 + F1 * J21) / det
        d_ln_n = max(-30.0, min(30.0, d_ln_n))
        d_ln_p = max(-30.0, min(30.0, d_ln_p))
        ln_n += d_ln_n
        ln_p += d_ln_p
    n_s, p_s = math.exp(ln_n), math.exp(ln_p)
    return n_s, p_s, plane_rate(n_s, p_s, prm, v_n, v_p)
