"""Historical Maxwell-Boltzmann drift-diffusion face quantities (M class).

The electron and hole currents are the historical Scharfetter-Gummel currents
(A/m^2, historical direction convention) driven by the DOS-folded transport
potentials ``phi + chi_t`` and ``phi + chi_t + Eg_t`` of a
:class:`~solarlab.physics.statistics.BandStructure`. They exclude the
thermionic-emission cap and exclusive interface faces, which belong to the
interface-transport slice. The unified directed particle flux and its stable
electrochemical evaluation are in
:mod:`solarlab.physics.transport.electrochemical_flux` (N class).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from solarlab.discretization.face_average import harmonic_face_mean
from solarlab.discretization.scharfetter_gummel import legacy_sg_current_n, legacy_sg_current_p
from solarlab.physics.statistics.band_potentials import BandStructure

__all__ = ["legacy_carrier_face_currents", "legacy_face_diffusivity"]


def legacy_face_diffusivity(node_diffusivity: ArrayLike) -> NDArray[np.float64]:
    """Historical carrier face diffusivity ``2 D_i D_{i+1} / (D_i + D_{i+1})``.

    Kept exactly, including the 0/0 = NaN face between two zero-diffusivity
    nodes (LD-03); a guarded mean would be a separate, declared change.
    """
    return harmonic_face_mean(node_diffusivity)


def legacy_carrier_face_currents(
    x: ArrayLike,
    phi: ArrayLike,
    n: ArrayLike,
    p: ArrayLike,
    D_n_face: ArrayLike,
    D_p_face: ArrayLike,
    thermal_voltage_V: float,
    bands: BandStructure,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Historical MB face currents ``(J_n, J_p)`` [A/m^2] on the N-1 faces."""
    spacing = np.diff(np.asarray(x, dtype=float))
    electron_potential, hole_potential = bands.transport_potentials(phi)
    J_n = legacy_sg_current_n(electron_potential, n, spacing, D_n_face, thermal_voltage_V)
    J_p = legacy_sg_current_p(hole_potential, p, spacing, D_p_face, thermal_voltage_V)
    return J_n, J_p
