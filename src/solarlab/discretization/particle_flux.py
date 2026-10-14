"""Unified Scharfetter-Gummel particle flux on oriented faces (N class).

    F = k * (B(D) * c_left - B(-D) * c_right),

where ``k = D_face / h`` [m/s] and ``D`` is the dimensionless potential-energy
rise from the left to the right node (``-dphi/V_T`` for electrons, ``+dphi/V_T``
for holes and other positive species). ``F`` is positive along the face
orientation, in particles per area per time. It is evaluated with
:func:`~solarlab.discretization.stable_bernoulli.bernoulli_pair` and returns
analytic derivatives. Reversing the face (swap densities, negate ``D``)
reverses ``F`` exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from solarlab.discretization.stable_bernoulli import bernoulli_pair

__all__ = ["ParticleFlux", "sg_particle_flux"]


@dataclass(frozen=True)
class ParticleFlux:
    """Directed particle flux and its local partial derivatives on each face."""

    flux: NDArray[np.float64]
    left_density_derivative: NDArray[np.float64]
    right_density_derivative: NDArray[np.float64]
    energy_step_derivative: NDArray[np.float64]
    coefficient_derivative: NDArray[np.float64]


def sg_particle_flux(
    c_left: ArrayLike,
    c_right: ArrayLike,
    energy_step: ArrayLike,
    coefficient: ArrayLike,
) -> ParticleFlux:
    """Unified SG particle flux with derivatives w.r.t. ``c_L``, ``c_R``, ``D`` and ``k``."""
    left = np.asarray(c_left, dtype=float)
    right = np.asarray(c_right, dtype=float)
    step = np.asarray(energy_step, dtype=float)
    k = np.asarray(coefficient, dtype=float)
    shape = np.broadcast_shapes(left.shape, right.shape, step.shape, k.shape)
    if (left.shape != shape or right.shape != shape or step.shape != shape
            or k.shape not in {(), shape}):
        raise ValueError("SG densities and energy steps must share one face shape")
    if not (np.all(np.isfinite(left)) and np.all(np.isfinite(right))
            and np.all(np.isfinite(k))):
        raise ValueError("SG inputs must be finite")
    if np.any(k < 0.0):
        raise ValueError("SG transfer coefficient must be non-negative")
    pair = bernoulli_pair(step)
    unscaled = pair.forward * left - pair.backward * right
    k_faces = np.broadcast_to(k, shape)
    return ParticleFlux(
        flux=k_faces * unscaled,
        left_density_derivative=k_faces * pair.forward,
        right_density_derivative=-k_faces * pair.backward,
        energy_step_derivative=k_faces * (
            pair.forward_derivative * left + pair.backward_derivative * right),
        coefficient_derivative=unscaled,
    )
