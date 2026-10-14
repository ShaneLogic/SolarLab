"""Historical Scharfetter-Gummel electrical currents (M class).

The ``legacy_sg_current_*`` functions reproduce the historical currents
(A/m^2, historical direction convention) and their analytic face Jacobians
bit-for-bit. The unified directed particle flux is
:mod:`solarlab.discretization.particle_flux` (N class); the conversion is
``F_n = -J_n / q`` and ``F_p = +J_p / q``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from solarlab.contracts.units import ELEMENTARY_CHARGE_C
from solarlab.discretization.bernoulli import bernoulli, bernoulli_derivative

__all__ = [
    "LegacyFaceJacobian", "legacy_sg_current_n", "legacy_sg_current_n_jacobian",
    "legacy_sg_current_p", "legacy_sg_current_p_jacobian",
]

Q = ELEMENTARY_CHARGE_C


def legacy_sg_current_n(
    phi: ArrayLike,
    n: ArrayLike,
    dx: ArrayLike,
    D_n: ArrayLike,
    V_T: float,
) -> NDArray[np.float64]:
    """Historical vectorized SG electron current at all N-1 faces [A/m^2]."""
    phi = np.asarray(phi, dtype=float)
    n = np.asarray(n, dtype=float)
    q = 1.602176634e-19
    xi = (phi[1:] - phi[:-1]) / V_T
    current: NDArray[np.float64] = q * D_n / dx * (bernoulli(xi) * n[1:] - bernoulli(-xi) * n[:-1])
    return current


def legacy_sg_current_p(
    phi: ArrayLike,
    p: ArrayLike,
    dx: ArrayLike,
    D_p: ArrayLike,
    V_T: float,
) -> NDArray[np.float64]:
    """Historical vectorized SG hole current at all N-1 faces [A/m^2]."""
    phi = np.asarray(phi, dtype=float)
    p = np.asarray(p, dtype=float)
    q = 1.602176634e-19
    xi = (phi[1:] - phi[:-1]) / V_T
    current: NDArray[np.float64] = q * D_p / dx * (bernoulli(xi) * p[:-1] - bernoulli(-xi) * p[1:])
    return current


@dataclass(frozen=True, slots=True)
class LegacyFaceJacobian:
    """Local derivatives of one historical carrier current on every grid face."""

    flux: NDArray[np.float64]
    density_left_derivative: NDArray[np.float64]
    density_right_derivative: NDArray[np.float64]
    potential_left_derivative: NDArray[np.float64]
    potential_right_derivative: NDArray[np.float64]


def _validate_legacy_jacobian_inputs(
    phi: ArrayLike,
    density: ArrayLike,
    dx: ArrayLike,
    diffusivity: ArrayLike,
    V_T: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    potential = np.asarray(phi, dtype=float)
    carrier_density = np.asarray(density, dtype=float)
    spacing = np.asarray(dx, dtype=float)
    if (
        potential.ndim != 1
        or potential.size < 2
        or carrier_density.shape != potential.shape
        or spacing.shape != (potential.size - 1,)
        or not np.all(np.isfinite(potential))
        or not np.all(np.isfinite(carrier_density))
        or not np.all(np.isfinite(spacing))
        or np.any(spacing <= 0.0)
        or not np.isfinite(V_T)
        or V_T <= 0.0
    ):
        raise ValueError(
            "SG Jacobian inputs must be finite, shape matched, and use "
            "positive spacing and thermal voltage"
        )
    diffusion = np.broadcast_to(np.asarray(diffusivity, dtype=float), spacing.shape)
    if not np.all(np.isfinite(diffusion)) or np.any(diffusion < 0.0):
        raise ValueError("SG diffusivity must be finite and non-negative")
    return potential, carrier_density, spacing, diffusion


def legacy_sg_current_n_jacobian(
    phi: ArrayLike,
    n: ArrayLike,
    dx: ArrayLike,
    D_n: ArrayLike,
    V_T: float,
) -> LegacyFaceJacobian:
    """Historical analytic electron-current derivatives for fixed face diffusivity."""
    potential, density, spacing, diffusion = _validate_legacy_jacobian_inputs(
        phi, n, dx, D_n, V_T
    )
    xi = (potential[1:] - potential[:-1]) / V_T
    B_pos = bernoulli(xi)
    B_neg = bernoulli(-xi)
    dB_pos = bernoulli_derivative(xi)
    dB_neg = bernoulli_derivative(-xi)
    prefactor = Q * diffusion / spacing
    xi_derivative = prefactor * (dB_pos * density[1:] + dB_neg * density[:-1])
    return LegacyFaceJacobian(
        flux=legacy_sg_current_n(potential, density, spacing, diffusion, V_T),
        density_left_derivative=-prefactor * B_neg,
        density_right_derivative=prefactor * B_pos,
        potential_left_derivative=-xi_derivative / V_T,
        potential_right_derivative=xi_derivative / V_T,
    )


def legacy_sg_current_p_jacobian(
    phi: ArrayLike,
    p: ArrayLike,
    dx: ArrayLike,
    D_p: ArrayLike,
    V_T: float,
) -> LegacyFaceJacobian:
    """Historical analytic hole-current derivatives for fixed face diffusivity."""
    potential, density, spacing, diffusion = _validate_legacy_jacobian_inputs(
        phi, p, dx, D_p, V_T
    )
    xi = (potential[1:] - potential[:-1]) / V_T
    B_pos = bernoulli(xi)
    B_neg = bernoulli(-xi)
    dB_pos = bernoulli_derivative(xi)
    dB_neg = bernoulli_derivative(-xi)
    prefactor = Q * diffusion / spacing
    xi_derivative = prefactor * (dB_pos * density[:-1] + dB_neg * density[1:])
    return LegacyFaceJacobian(
        flux=legacy_sg_current_p(potential, density, spacing, diffusion, V_T),
        density_left_derivative=prefactor * B_pos,
        density_right_derivative=-prefactor * B_neg,
        potential_left_derivative=-xi_derivative / V_T,
        potential_right_derivative=xi_derivative / V_T,
    )
