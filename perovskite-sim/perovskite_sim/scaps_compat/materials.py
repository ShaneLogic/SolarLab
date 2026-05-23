"""SCAPS material parameter conversions.

SCAPS layers carry the effective conduction- and valence-band density of
states ``N_C`` and ``N_V`` plus the band gap ``E_g``. SolarLab's
``MaterialParams`` carries the intrinsic carrier density ``ni`` instead, so
this helper closes the gap with the standard non-degenerate semiconductor
relation ``ni^2 = N_C * N_V * exp(-E_g / kT)``.
"""
from __future__ import annotations

import math

from perovskite_sim.constants import K_B, Q


def ni_from_dos(
    N_C_m3: float, N_V_m3: float, E_g_eV: float, T: float = 300.0
) -> float:
    """Return the intrinsic carrier density in m^-3.

    Inputs are SI: ``N_C``, ``N_V`` in m^-3, ``E_g`` in eV, ``T`` in K.
    """
    if N_C_m3 <= 0.0:
        raise ValueError(f"N_C_m3 must be positive, got {N_C_m3}")
    if N_V_m3 <= 0.0:
        raise ValueError(f"N_V_m3 must be positive, got {N_V_m3}")
    if E_g_eV <= 0.0:
        raise ValueError(f"E_g_eV must be positive, got {E_g_eV}")
    if T <= 0.0:
        raise ValueError(f"T must be positive, got {T}")
    kT_eV = K_B * T / Q
    return math.sqrt(N_C_m3 * N_V_m3 * math.exp(-E_g_eV / kT_eV))
