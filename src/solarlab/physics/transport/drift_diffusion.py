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

from solarlab.contracts.units import ELEMENTARY_CHARGE_C
from solarlab.discretization.bernoulli import bernoulli, bernoulli_derivative
from solarlab.discretization.face_average import harmonic_face_mean
from solarlab.discretization.scharfetter_gummel import (
    LegacyFaceJacobian,
    legacy_sg_current_n,
    legacy_sg_current_n_jacobian,
    legacy_sg_current_p,
    legacy_sg_current_p_jacobian,
)
from solarlab.physics.statistics.band_potentials import BandStructure
from solarlab.physics.transport.electrochemical_flux import (
    BAND_STEP_MODES,
    HOLE_SEPARATE_INCREMENT_V1,
    LEGACY_SUMMED_POTENTIAL_V1,
    carrier_transport_potential_steps,
)

__all__ = [
    "BAND_STEP_MODES",
    "HOLE_SEPARATE_INCREMENT_V1",
    "LEGACY_SUMMED_POTENTIAL_V1",
    "carrier_transport_potential_steps",
    "legacy_carrier_face_currents",
    "legacy_carrier_face_jacobians",
    "legacy_face_diffusivity",
]

Q = ELEMENTARY_CHARGE_C


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
    *,
    band_step_mode: str = LEGACY_SUMMED_POTENTIAL_V1,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Historical MB face currents ``(J_n, J_p)`` [A/m^2] on the N-1 faces."""
    if band_step_mode not in BAND_STEP_MODES:
        raise ValueError(
            f"unknown band_step_mode {band_step_mode!r}; expected one of {BAND_STEP_MODES}"
        )
    spacing = np.diff(np.asarray(x, dtype=float))
    if band_step_mode == LEGACY_SUMMED_POTENTIAL_V1:
        electron_potential, hole_potential = bands.transport_potentials(phi)
        J_n = legacy_sg_current_n(electron_potential, n, spacing, D_n_face, thermal_voltage_V)
        J_p = legacy_sg_current_p(hole_potential, p, spacing, D_p_face, thermal_voltage_V)
        return J_n, J_p

    electrons = np.asarray(n, dtype=float)
    holes = np.asarray(p, dtype=float)
    dpsi_n, dpsi_p = carrier_transport_potential_steps(
        phi, bands, band_step_mode=band_step_mode
    )
    q = 1.602176634e-19
    xi_n = dpsi_n / thermal_voltage_V
    xi_p = dpsi_p / thermal_voltage_V
    j_n: NDArray[np.float64] = (
        q * np.asarray(D_n_face, dtype=float) / spacing * (bernoulli(xi_n) * electrons[1:] - bernoulli(-xi_n) * electrons[:-1])
    )
    j_p: NDArray[np.float64] = (
        q * np.asarray(D_p_face, dtype=float) / spacing * (bernoulli(xi_p) * holes[:-1] - bernoulli(-xi_p) * holes[1:])
    )
    return j_n, j_p


def legacy_carrier_face_jacobians(
    x: ArrayLike,
    phi: ArrayLike,
    n: ArrayLike,
    p: ArrayLike,
    D_n_face: ArrayLike,
    D_p_face: ArrayLike,
    thermal_voltage_V: float,
    bands: BandStructure,
    *,
    band_step_mode: str = LEGACY_SUMMED_POTENTIAL_V1,
) -> tuple[LegacyFaceJacobian, LegacyFaceJacobian]:
    """Analytic ``(J_n, J_p)`` face Jacobians wrt ``(phi_L, phi_R, c_L, c_R)``."""
    if band_step_mode not in BAND_STEP_MODES:
        raise ValueError(
            f"unknown band_step_mode {band_step_mode!r}; expected one of {BAND_STEP_MODES}"
        )
    grid = np.asarray(x, dtype=float)
    spacing = np.diff(grid)
    if band_step_mode == LEGACY_SUMMED_POTENTIAL_V1:
        electron_potential, hole_potential = bands.transport_potentials(phi)
        return (
            legacy_sg_current_n_jacobian(electron_potential, n, spacing, D_n_face, thermal_voltage_V),
            legacy_sg_current_p_jacobian(hole_potential, p, spacing, D_p_face, thermal_voltage_V),
        )

    potential = np.asarray(phi, dtype=float)
    electrons = np.asarray(n, dtype=float)
    holes = np.asarray(p, dtype=float)
    thermal = float(thermal_voltage_V)
    if (
        grid.ndim != 1
        or grid.size < 2
        or potential.shape != grid.shape
        or electrons.shape != grid.shape
        or holes.shape != grid.shape
        or np.any(spacing <= 0.0)
        or not (
            np.all(np.isfinite(grid))
            and np.all(np.isfinite(potential))
            and np.all(np.isfinite(electrons))
            and np.all(np.isfinite(holes))
        )
        or not np.isfinite(thermal)
        or thermal <= 0.0
    ):
        raise ValueError(
            "SG Jacobian inputs must be finite, shape matched, and use "
            "positive spacing and thermal voltage"
        )
    dn = np.broadcast_to(np.asarray(D_n_face, dtype=float), spacing.shape)
    dp = np.broadcast_to(np.asarray(D_p_face, dtype=float), spacing.shape)
    if not (np.all(np.isfinite(dn)) and np.all(np.isfinite(dp))) or np.any(dn < 0.0) or np.any(dp < 0.0):
        raise ValueError("SG diffusivity must be finite and non-negative")

    dpsi_n, dpsi_p = carrier_transport_potential_steps(
        potential, bands, band_step_mode=band_step_mode
    )
    j_n, j_p = legacy_carrier_face_currents(
        grid, potential, electrons, holes, dn, dp, thermal, bands, band_step_mode=band_step_mode
    )
    xi_n = dpsi_n / thermal
    xi_p = dpsi_p / thermal
    pref_n = Q * dn / spacing
    pref_p = Q * dp / spacing
    d_xi_n = pref_n * (bernoulli_derivative(xi_n) * electrons[1:] + bernoulli_derivative(-xi_n) * electrons[:-1])
    d_xi_p = pref_p * (bernoulli_derivative(xi_p) * holes[:-1] + bernoulli_derivative(-xi_p) * holes[1:])
    return (
        LegacyFaceJacobian(
            flux=j_n,
            density_left_derivative=-pref_n * bernoulli(-xi_n),
            density_right_derivative=pref_n * bernoulli(xi_n),
            potential_left_derivative=-d_xi_n / thermal,
            potential_right_derivative=d_xi_n / thermal,
        ),
        LegacyFaceJacobian(
            flux=j_p,
            density_left_derivative=pref_p * bernoulli(xi_p),
            density_right_derivative=-pref_p * bernoulli(-xi_p),
            potential_left_derivative=-d_xi_p / thermal,
            potential_right_derivative=d_xi_p / thermal,
        ),
    )
