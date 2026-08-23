"""Thermodynamically consistent generalized Scharfetter-Gummel fluxes."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from perovskite_sim.constants import Q
from perovskite_sim.discretization.fe_operators import bernoulli
from perovskite_sim.physics.statistics import (
    CarrierStatistics,
    MAXWELL_BOLTZMANN,
    generalized_einstein_factor,
    normalize_carrier_statistics,
    reduced_fermi_level_from_density,
)


@dataclass(frozen=True, slots=True)
class GeneralizedCarrierFaceStatistics:
    """Node chemical potentials and face diffusion-enhancement factors."""

    statistics: CarrierStatistics
    reduced_fermi_level: np.ndarray
    log_occupation: np.ndarray
    diffusion_enhancement: np.ndarray


def generalized_carrier_face_statistics(
    density_m3: np.ndarray,
    effective_density_of_states_m3: float,
    *,
    statistics: CarrierStatistics | str,
) -> GeneralizedCarrierFaceStatistics:
    """Return the logarithmic-secant generalized Einstein factor per face.

    For a statistics function ``F(eta)`` the diffusion-enhancement factor is

    ``g_KL = (eta_L - eta_K) / (log(F_L) - log(F_K))``.

    The removable equal-state singularity is evaluated from the local
    logarithmic compressibility. Maxwell-Boltzmann statistics return exact
    ones rather than a numerically reconstructed identity.
    """
    density = np.asarray(density_m3, dtype=float)
    dos = float(effective_density_of_states_m3)
    model = normalize_carrier_statistics(statistics)
    if (
        density.ndim != 1
        or density.size < 2
        or not np.all(np.isfinite(density))
        or np.any(density <= 0.0)
    ):
        raise ValueError("carrier density must be a finite positive 1D array")
    if not math.isfinite(dos) or dos <= 0.0:
        raise ValueError("effective density of states must be finite and positive")

    log_occupation = np.log(density / dos)
    if model == MAXWELL_BOLTZMANN:
        eta = log_occupation.copy()
        enhancement = np.ones(density.size - 1, dtype=float)
    else:
        eta = np.asarray(
            [
                reduced_fermi_level_from_density(
                    value,
                    dos,
                    statistics=model,
                )
                for value in density
            ],
            dtype=float,
        )
        delta_eta = np.diff(eta)
        delta_log = np.diff(log_occupation)
        scale = np.maximum(
            1.0,
            np.maximum(
                np.abs(log_occupation[:-1]),
                np.abs(log_occupation[1:]),
            ),
        )
        regular = np.abs(delta_log) > 64.0 * np.finfo(float).eps * scale
        enhancement = np.empty_like(delta_log)
        enhancement[regular] = delta_eta[regular] / delta_log[regular]
        for index in np.flatnonzero(~regular):
            enhancement[index] = generalized_einstein_factor(
                0.5 * (eta[index] + eta[index + 1]),
                statistics=model,
            )
    if (
        not np.all(np.isfinite(eta))
        or not np.all(np.isfinite(enhancement))
        or np.any(enhancement <= 0.0)
    ):
        raise FloatingPointError(
            "generalized carrier face statistics are non-finite or non-positive"
        )
    return GeneralizedCarrierFaceStatistics(
        statistics=model,
        reduced_fermi_level=eta,
        log_occupation=log_occupation,
        diffusion_enhancement=enhancement,
    )


def _validated_flux_inputs(
    band_potential_V: np.ndarray,
    density_m3: np.ndarray,
    spacing_m: np.ndarray,
    mobility_m2_V_s: np.ndarray | float,
    thermal_voltage_V: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    potential = np.asarray(band_potential_V, dtype=float)
    density = np.asarray(density_m3, dtype=float)
    spacing = np.asarray(spacing_m, dtype=float)
    thermal = float(thermal_voltage_V)
    mobility = np.broadcast_to(
        np.asarray(mobility_m2_V_s, dtype=float),
        spacing.shape,
    )
    if (
        potential.ndim != 1
        or density.shape != potential.shape
        or potential.size < 2
        or spacing.shape != (potential.size - 1,)
        or not np.all(np.isfinite(potential))
        or not np.all(np.isfinite(spacing))
        or np.any(spacing <= 0.0)
        or not np.all(np.isfinite(mobility))
        or np.any(mobility < 0.0)
        or not math.isfinite(thermal)
        or thermal <= 0.0
    ):
        raise ValueError(
            "generalized SG inputs must be finite, shape matched, and use "
            "positive spacing/thermal voltage with non-negative mobility"
        )
    return potential, density, spacing, mobility, thermal


def generalized_sg_fluxes_n(
    band_potential_V: np.ndarray,
    electron_density_m3: np.ndarray,
    spacing_m: np.ndarray,
    mobility_m2_V_s: np.ndarray | float,
    thermal_voltage_V: float,
    effective_conduction_dos_m3: float,
    *,
    statistics: CarrierStatistics | str,
) -> np.ndarray:
    """Return generalized electron current on all faces [A m-2]."""
    potential, density, spacing, mobility, thermal = _validated_flux_inputs(
        band_potential_V,
        electron_density_m3,
        spacing_m,
        mobility_m2_V_s,
        thermal_voltage_V,
    )
    face = generalized_carrier_face_statistics(
        density,
        effective_conduction_dos_m3,
        statistics=statistics,
    )
    enhancement = face.diffusion_enhancement
    argument = np.diff(potential) / (thermal * enhancement)
    return (
        Q
        * mobility
        * thermal
        * enhancement
        / spacing
        * (
            bernoulli(argument) * density[1:]
            - bernoulli(-argument) * density[:-1]
        )
    )


def generalized_sg_fluxes_p(
    band_potential_V: np.ndarray,
    hole_density_m3: np.ndarray,
    spacing_m: np.ndarray,
    mobility_m2_V_s: np.ndarray | float,
    thermal_voltage_V: float,
    effective_valence_dos_m3: float,
    *,
    statistics: CarrierStatistics | str,
) -> np.ndarray:
    """Return generalized hole current on all faces [A m-2]."""
    potential, density, spacing, mobility, thermal = _validated_flux_inputs(
        band_potential_V,
        hole_density_m3,
        spacing_m,
        mobility_m2_V_s,
        thermal_voltage_V,
    )
    face = generalized_carrier_face_statistics(
        density,
        effective_valence_dos_m3,
        statistics=statistics,
    )
    enhancement = face.diffusion_enhancement
    argument = np.diff(potential) / (thermal * enhancement)
    return (
        Q
        * mobility
        * thermal
        * enhancement
        / spacing
        * (
            bernoulli(argument) * density[:-1]
            - bernoulli(-argument) * density[1:]
        )
    )


__all__ = [
    "GeneralizedCarrierFaceStatistics",
    "generalized_carrier_face_statistics",
    "generalized_sg_fluxes_n",
    "generalized_sg_fluxes_p",
]
