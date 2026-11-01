"""Historical operating-point preparation reproduced bit for bit (M class).

* :func:`legacy_quasi_neutral_seed` -- the dark seed of
  ``solver.newton.solve_equilibrium``: two-branch ``n - p = N_D - N_A``,
  ``n p = n_i^2`` per node, clipped to ``[1, 1e36]``, with the four contact
  reservoirs written on the boundary nodes. It is a seed, not an equilibrium.
* :func:`legacy_illumination_stages` and the two historical stage tables of
  the quasi-Fermi steady-state preparation (dark is the single stage 0).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "LEGACY_DEFAULT_ILLUMINATION_STAGES", "LEGACY_INTERFACE_FALLBACK_ILLUMINATION_STAGES",
    "legacy_illumination_stages", "legacy_quasi_neutral_seed",
]

LEGACY_DEFAULT_ILLUMINATION_STAGES = (
    0.0, 1.0e-14, 1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2,
    1.0e-1, 1.0,
)
LEGACY_INTERFACE_FALLBACK_ILLUMINATION_STAGES = (
    0.0, 1.0e-14, 1.0e-13, 1.0e-12, 1.0e-11, 1.0e-10, 1.0e-9, 1.0e-8, 1.0e-7, 1.0e-6,
    1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0,
)


def legacy_quasi_neutral_seed(
    donor_density_m3: ArrayLike,
    acceptor_density_m3: ArrayLike,
    intrinsic_density_squared_m6: ArrayLike,
    *,
    n_left: float,
    n_right: float,
    p_left: float,
    p_right: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Quasi-neutral ``(n, p)`` seed with the contact reservoirs pinned."""
    net = np.asarray(donor_density_m3) - np.asarray(acceptor_density_m3)
    ni_sq = np.asarray(intrinsic_density_squared_m6)
    disc = np.sqrt(net ** 2 + 4.0 * ni_sq)
    half_pos = 0.5 * (net + disc)
    half_neg = 0.5 * (-net + disc)
    with np.errstate(divide="ignore", invalid="ignore"):
        n = np.where(net >= 0, half_pos, ni_sq / half_neg)
        p = np.where(net >= 0, ni_sq / half_pos, half_neg)
    n = np.clip(n, 1.0, 1e36)
    p = np.clip(p, 1.0, 1e36)
    n[0] = n_left
    n[-1] = n_right
    p[0] = p_left
    p[-1] = p_right
    return n, p


def legacy_illumination_stages(illuminated: bool,
                               values: tuple[float, ...]) -> tuple[float, ...]:
    """Validated illumination continuation: strictly increasing in [0, 1], ending at 1."""
    if not illuminated:
        return (0.0,)
    stages = tuple(float(value) for value in values)
    if not stages or stages[-1] != 1.0:
        raise ValueError("illuminated continuation must end at 1.0")
    if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in stages):
        raise ValueError("illumination continuation values must lie in [0, 1]")
    if any(right <= left for left, right in zip(stages[:-1], stages[1:])):
        raise ValueError("illumination continuation values must strictly increase")
    return stages
