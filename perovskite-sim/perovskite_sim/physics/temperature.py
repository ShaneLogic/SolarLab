"""Temperature-dependent parameter scaling for drift-diffusion simulations.

All scaling functions take a reference value at T_ref=300 K and return the
value at an arbitrary temperature T. When T=300 K, every function returns
the input unchanged (identity).

Usage:
    ni_T = ni_at_T(ni_300, Eg, T, Nc300, Nv300)
    mu_T = mu_at_T(mu_300, T, gamma)
    D_ion_T = D_ion_at_T(D_ion_300, T, E_a)
    B_T = B_rad_at_T(B_300, T, gamma)
    Eg_T = eg_at_T(Eg_300, T, alpha, beta)
    V_T_at = thermal_voltage(T)
"""
from __future__ import annotations

import numpy as np

from perovskite_sim.constants import K_B, Q

T_REF = 300.0


def thermal_voltage(T: float) -> float:
    """Thermal voltage kT/q [V]."""
    return K_B * T / Q


def ni_at_T(
    ni_300: float,
    Eg: float,
    T: float,
    Nc300: float | None = None,
    Nv300: float | None = None,
) -> float:
    """Intrinsic carrier density at temperature T.

    ni(T) = sqrt(Nc(T) * Nv(T)) * exp(-Eg / 2kT)

    where Nc(T) = Nc300 * (T/300)^(3/2), Nv(T) = Nv300 * (T/300)^(3/2).

    When Nc300 or Nv300 are not provided, uses the simpler scaling:
        ni(T) = ni_300 * (T/300)^(3/2) * exp(-Eg/(2k) * (1/T - 1/300))
    """
    if T == T_REF:
        return ni_300
    if Nc300 is not None and Nv300 is not None:
        ratio = T / T_REF
        Nc_T = Nc300 * ratio ** 1.5
        Nv_T = Nv300 * ratio ** 1.5
        return float(np.sqrt(Nc_T * Nv_T) * np.exp(-Eg * Q / (2.0 * K_B * T)))
    # Simplified scaling when DOS not provided
    ratio = T / T_REF
    return float(
        ni_300 * ratio ** 1.5
        * np.exp(-Eg * Q / (2.0 * K_B) * (1.0 / T - 1.0 / T_REF))
    )


def mu_at_T(mu_300: float, T: float, gamma: float = -1.5) -> float:
    """Mobility at temperature T with power-law phonon scattering.

    mu(T) = mu_300 * (T/300)^gamma

    Default gamma=-1.5 (acoustic phonon scattering).
    """
    if T == T_REF:
        return mu_300
    return float(mu_300 * (T / T_REF) ** gamma)


def D_ion_at_T(D_ion_300: float, T: float, E_a: float) -> float:
    """Ion diffusion coefficient at temperature T (Arrhenius).

    D_ion(T) = D_ion_300 * exp(-E_a/k * (1/T - 1/300))

    E_a is the activation energy in eV.
    """
    if T == T_REF or D_ion_300 == 0.0:
        return D_ion_300
    return float(
        D_ion_300 * np.exp(-E_a * Q / K_B * (1.0 / T - 1.0 / T_REF))
    )


def B_rad_at_T(B_300: float, T: float, gamma: float = 0.0) -> float:
    """Band-to-band radiative coefficient at temperature T (Phase 4b).

    ``B(T) = B_300 · (T/300)^gamma``

    The default ``gamma = 0`` is the opt-out sentinel so pre-Phase-4b
    configs are bit-identical. A typical literature value for MAPbI3
    under the Varshni-free detailed-balance scaling is ``gamma ≈ -1.5``:
    under ``n·p = ni²``, keeping the radiative J_0 proportional to
    ``ni² · sqrt(D/tau)`` requires ``B ∝ T^-1.5``. Literature reports
    values in the ``[-1.5, -1]`` window for MAPbI3.

    At ``T = T_REF`` or ``B_300 = 0`` this is the identity so configs
    that did not set a custom ``B_rad_T_gamma`` behave exactly like
    they did before Phase 4b.
    """
    if T == T_REF or B_300 == 0.0 or gamma == 0.0:
        return B_300
    return float(B_300 * (T / T_REF) ** gamma)


def eg_at_T(
    Eg_300: float,
    T: float,
    alpha: float = 0.0,
    beta: float = 0.0,
) -> float:
    """Varshni bandgap shift relative to 300 K (Phase 4b).

    The classical Varshni equation is referenced to 0 K:
        ``Eg(T) = Eg(0) − α·T² / (T + β)``

    This helper instead returns the bandgap at ``T`` given the value at
    the reference temperature ``T_REF = 300 K``, which is what every
    ``MaterialParams.Eg`` already represents:

        ``Eg(T) = Eg_300 − α · [T² / (T + β) − T_REF² / (T_REF + β)]``

    Parameters
    ----------
    Eg_300
        Bandgap at 300 K (eV). Returned unchanged when ``alpha = 0``.
    T
        Target temperature (K).
    alpha, beta
        Varshni coefficients (``alpha`` in eV/K, ``beta`` in K). The
        defaults ``alpha = beta = 0`` disable the shift — Eg stays at
        its 300 K value, which preserves pre-Phase-4b behaviour for
        every config that has not opted in.

    Notes
    -----
    ``alpha = 0`` is the opt-out sentinel rather than a literal
    "Varshni with alpha exactly zero" (which would also return
    ``Eg_300`` for any T). A user that genuinely wants Eg to be T-
    independent should leave alpha at its default; a user that wants
    the Varshni shift should set both alpha and beta from the
    literature. For silicon the standard values are α ≈ 4.73e-4 eV/K,
    β ≈ 636 K (Eg narrows with T). MAPbI3 is opposite — the bandgap
    *increases* with T, which the Varshni form reproduces with
    α ≈ −3e-4 eV/K and β > 0 (reports vary; ~+200 K is a
    representative value).
    """
    if T == T_REF or alpha == 0.0:
        return Eg_300
    denom_T = T + beta
    denom_ref = T_REF + beta
    if denom_T == 0.0 or denom_ref == 0.0:
        # Degenerate Varshni — return the reference value rather than
        # dividing by zero.
        return Eg_300
    shift = alpha * (T * T / denom_T - T_REF * T_REF / denom_ref)
    return float(Eg_300 - shift)
