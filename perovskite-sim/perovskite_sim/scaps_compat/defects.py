"""Microscopic SCAPS defect parameters to SolarLab lumped quantities.

SCAPS specifies each SRH-active site with a microscopic triplet
``(sigma, v_th, N_t)``. SolarLab consumes a bulk SRH lifetime ``tau`` and an
interface surface recombination velocity ``v_eff``. Both follow from the
identical kinetic identity ``rate = sigma * v_th * N_t * carrier_density``.
"""
from __future__ import annotations

import math


def srh_lifetime(sigma_m2: float, v_th_m_s: float, N_t_m3: float) -> float:
    """Return the SRH carrier lifetime in seconds.

    ``tau = 1 / (sigma * v_th * N_t)``. ``N_t = 0`` returns ``+inf`` to
    signal an undamped (defect-free) layer.
    """
    if sigma_m2 < 0.0 or v_th_m_s < 0.0 or N_t_m3 < 0.0:
        raise ValueError("sigma, v_th, N_t must be non-negative")
    product = sigma_m2 * v_th_m_s * N_t_m3
    if product == 0.0:
        return math.inf
    return 1.0 / product


def interface_surface_velocity(
    sigma_m2: float, v_th_m_s: float, N_t_m3: float
) -> float:
    """Return the interface surface recombination velocity in m/s.

    Uses the same kinetic identity as bulk SRH, dropping the inverse: the
    interface trap density ``N_t`` is interpreted as the projected density
    at the heterojunction face (m^-3 here; callers using areal density
    m^-2 should pre-multiply by an effective trap width).
    """
    if sigma_m2 < 0.0 or v_th_m_s < 0.0 or N_t_m3 < 0.0:
        raise ValueError("sigma, v_th, N_t must be non-negative")
    return sigma_m2 * v_th_m_s * N_t_m3
