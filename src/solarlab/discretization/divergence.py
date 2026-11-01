"""Control-volume divergence of oriented face fluxes."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["flux_divergence"]


def flux_divergence(
    flux: ArrayLike,
    widths: ArrayLike,
    *,
    left_flux: float = 0.0,
    right_flux: float = 0.0,
) -> NDArray[np.float64]:
    """Particle rate per volume: ``w*dP/dt = F_left - F_right`` with boundary fluxes.

    ``flux`` holds the N-1 interior face fluxes oriented along +x; the contact
    fluxes enter through ``left_flux``/``right_flux`` (also along +x).
    """
    w = np.asarray(widths, dtype=float)
    f = np.asarray(flux)
    if (
        w.shape != (f.size + 1,)
        or f.ndim != 1
        or not np.all(np.isfinite(w))
        or np.any(w <= 0)
    ):
        raise ValueError("flux and positive finite volumes must align")
    rate: NDArray[np.float64] = -np.diff(np.r_[left_flux, f, right_flux]) / w
    return rate
