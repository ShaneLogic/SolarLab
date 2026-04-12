"""Temperature-dependent parameter scaling for drift-diffusion simulations.

All scaling functions take a reference value at T_ref=300 K and return the
value at an arbitrary temperature T. When T=300 K, every function returns
the input unchanged (identity).

Usage:
    ni_T = ni_at_T(ni_300, Eg, T, Nc300, Nv300)
    mu_T = mu_at_T(mu_300, T, gamma)
    D_ion_T = D_ion_at_T(D_ion_300, T, E_a)
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
