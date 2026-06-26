"""Intra-band thermionic-field-emission (TFE) tunnelling at heterointerfaces.

SolarLab's interface treatment caps the Scharfetter-Gummel flux at the
Richardson-Dushman *thermionic* limit (pure over-the-barrier emission, no
transmission factor). SCAPS additionally exposes an intra-band tunnelling
channel through a conduction-/valence-band spike. This module supplies the
STATIC, detailed-balance-preserving enhancement factor that models that
channel as an effective barrier-lowering (Padovani-Stratton / Crowell-Rideout
thermionic-field-emission), to be folded into the per-face Richardson constant
A* by the solver build:

    A*_eff = Gamma · A*,   Gamma >= 1

Because the SAME Gamma multiplies BOTH legs of ``thermionic_emission_flux``,
the equilibrium current stays exactly zero (the two-leg bracket is unchanged
under a common scale); and because the solver's TE cap blends J_TE toward the
SG flux, the SG flux remains the structural ceiling (Gamma can only lift the
thermionic limit toward SG, never above it). The factor is built once from
geometry/doping — no live field, no per-RHS state — so it cannot perturb the
Newton/Radau path (unlike a per-RHS field-dependent term).

Physics (Padovani-Stratton):
    E_00 = (q·ħ/2)·sqrt(N_iface / (m*_eff · eps_s))        characteristic energy
    E_0  = E_00 · coth(E_00 / kT)                          effective TFE energy (>= kT)
    delta_tun = |delta_E| · (1 - V_T / E_0),  clamped to [0, |delta_E|)
    Gamma = exp(delta_tun / V_T)

Limits: E_00 -> 0 (low doping / intrinsic side) ⇒ E_0 -> kT ⇒ delta_tun -> 0
⇒ Gamma -> 1 (pure thermionic, bit-identical). This means an *intrinsic*
absorber interface (the depletion-side doping ~0) gets little enhancement — a
documented physics boundary of the static doping form, not a bug.
"""
from __future__ import annotations

import math

from perovskite_sim.constants import Q, EPS_0

# Reduced Planck constant [J·s] and free-electron mass [kg]. Not in
# constants.py (which carries only the drift-diffusion essentials), so they
# are defined here where the tunnelling model is the only consumer.
_HBAR = 1.054571817e-34
_M_E = 9.1093837015e-31


def _coth(z: float) -> float:
    """Numerically safe hyperbolic cotangent for z > 0."""
    if z > 40.0:
        return 1.0          # coth saturates to 1; exp(2z) would overflow
    if z < 1e-8:
        return 1.0 / z      # small-argument series leading term (avoids 0/0)
    return 1.0 / math.tanh(z)


def tfe_gamma(
    delta_E_eV: float,
    N_iface: float,
    m_eff_rel: float,
    eps_r: float,
    V_T: float,
) -> float:
    """Static Padovani-Stratton TFE enhancement Gamma >= 1 for one carrier.

    Parameters
    ----------
    delta_E_eV
        Band offset magnitude across the interface [eV] (|delta_Ec| for
        electrons, |delta_Ev| for holes).
    N_iface
        Effective interface doping [m^-3] — use the lighter-doped adjacent
        layer (depletion sits there). <= 0 (e.g. intrinsic side) ⇒ Gamma = 1.
    m_eff_rel
        Tunnelling effective mass relative to the free-electron mass.
    eps_r
        Relative permittivity at the interface.
    V_T
        Thermal voltage kT/q [V] (numerically = kT in eV).

    Returns
    -------
    float
        Gamma in [1, exp(|delta_E|/V_T)). Returns 1.0 (no enhancement) when
        N_iface <= 0, m_eff_rel <= 0, eps_r <= 0, or |delta_E| ~ 0.
    """
    dE = abs(delta_E_eV)
    if N_iface <= 0.0 or m_eff_rel <= 0.0 or eps_r <= 0.0 or dE <= 0.0 or V_T <= 0.0:
        return 1.0
    eps_s = eps_r * EPS_0
    # E_00 in joules → eV.
    E_00_J = (Q * _HBAR / 2.0) * math.sqrt(N_iface / (m_eff_rel * _M_E * eps_s))
    E_00 = E_00_J / Q                      # [eV]
    E_0 = E_00 * _coth(E_00 / V_T)         # [eV], >= V_T
    delta_tun = dE * (1.0 - V_T / E_0)
    # Clamp 0 <= delta_tun < |delta_E| so the enhanced over-barrier factor
    # exp(-(|delta_E|-delta_tun)/V_T) stays < 1 (physical) and Gamma is bounded.
    if delta_tun <= 0.0:
        return 1.0
    if delta_tun >= dE:
        delta_tun = dE * (1.0 - 1e-9)
    return math.exp(delta_tun / V_T)
