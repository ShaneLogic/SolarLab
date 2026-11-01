"""Restricted Fermi-Dirac transport: generalized SG and its capability gate (M class).

For statistics ``F(eta)`` the face diffusion enhancement is the logarithmic
secant ``g = (eta_R - eta_L) / (log F_R - log F_L)``, the removable
equal-state singularity is the local generalized Einstein factor, and the
current is the SG current with ``D -> D g`` and ``dpsi -> dpsi / g``. Fermi-Dirac
transport reads the physical band edges and one homogeneous DOS; it is
admitted only in the existing qualified research combination. Currents are
bit-identical to the legacy engine. The Blakemore approximation is not
implemented and is rejected by name.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from solarlab.contracts.units import ELEMENTARY_CHARGE_C
from solarlab.discretization.bernoulli import bernoulli
from solarlab.physics.statistics.band_potentials import BandStructure
from solarlab.physics.statistics.carriers import (
    FERMI_DIRAC, MAXWELL_BOLTZMANN, CarrierStatistics, generalized_einstein_factor,
    normalize_carrier_statistics, reduced_fermi_level_from_density,
)
from solarlab.physics.statistics.dopants import FULLY_IONIZED, normalize_dopant_ionization_model

__all__ = [
    "BLAKEMORE", "GeneralizedCarrierFaceStatistics", "RECOMBINATION_OFF",
    "TransportCapabilityError", "TransportCapabilityRequest",
    "generalized_carrier_face_statistics", "legacy_fermi_dirac_face_currents",
    "legacy_generalized_sg_current_n", "legacy_generalized_sg_current_p",
    "require_carrier_transport",
]

Q = ELEMENTARY_CHARGE_C
BLAKEMORE = "blakemore"
RECOMBINATION_OFF = "off"
_SEMICONDUCTOR_WORK_FUNCTION = "semiconductor_work_function"
_BAND_GAP_NARROWING_OFF = "off"
_BAND_GAP_NARROWING_MODELS = frozenset({_BAND_GAP_NARROWING_OFF, "slotboom"})


class TransportCapabilityError(ValueError):
    """The requested statistics/closure combination is not qualified."""


@dataclass(frozen=True, slots=True)
class GeneralizedCarrierFaceStatistics:
    """Node chemical potentials and face diffusion-enhancement factors."""

    statistics: CarrierStatistics
    reduced_fermi_level: NDArray[np.float64]
    log_occupation: NDArray[np.float64]
    diffusion_enhancement: NDArray[np.float64]


def generalized_carrier_face_statistics(
    density_m3: ArrayLike,
    effective_density_of_states_m3: float,
    *,
    statistics: CarrierStatistics | str,
) -> GeneralizedCarrierFaceStatistics:
    """Return the logarithmic-secant generalized Einstein factor per face.

    Maxwell-Boltzmann statistics return exact ones rather than a numerically
    reconstructed identity.
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
                reduced_fermi_level_from_density(value, dos, statistics=model)
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
    band_potential_V: ArrayLike,
    density_m3: ArrayLike,
    spacing_m: ArrayLike,
    mobility_m2_V_s: ArrayLike,
    thermal_voltage_V: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64],
           NDArray[np.float64], float]:
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


def legacy_generalized_sg_current_n(
    band_potential_V: ArrayLike,
    electron_density_m3: ArrayLike,
    spacing_m: ArrayLike,
    mobility_m2_V_s: ArrayLike,
    thermal_voltage_V: float,
    effective_conduction_dos_m3: float,
    *,
    statistics: CarrierStatistics | str,
) -> NDArray[np.float64]:
    """Historical generalized electron current on all faces [A m-2]."""
    potential, density, spacing, mobility, thermal = _validated_flux_inputs(
        band_potential_V, electron_density_m3, spacing_m, mobility_m2_V_s,
        thermal_voltage_V,
    )
    face = generalized_carrier_face_statistics(
        density, effective_conduction_dos_m3, statistics=statistics,
    )
    enhancement = face.diffusion_enhancement
    argument = np.diff(potential) / (thermal * enhancement)
    current: NDArray[np.float64] = (
        Q * mobility * thermal * enhancement / spacing
        * (bernoulli(argument) * density[1:] - bernoulli(-argument) * density[:-1])
    )
    return current


def legacy_generalized_sg_current_p(
    band_potential_V: ArrayLike,
    hole_density_m3: ArrayLike,
    spacing_m: ArrayLike,
    mobility_m2_V_s: ArrayLike,
    thermal_voltage_V: float,
    effective_valence_dos_m3: float,
    *,
    statistics: CarrierStatistics | str,
) -> NDArray[np.float64]:
    """Historical generalized hole current on all faces [A m-2]."""
    potential, density, spacing, mobility, thermal = _validated_flux_inputs(
        band_potential_V, hole_density_m3, spacing_m, mobility_m2_V_s,
        thermal_voltage_V,
    )
    face = generalized_carrier_face_statistics(
        density, effective_valence_dos_m3, statistics=statistics,
    )
    enhancement = face.diffusion_enhancement
    argument = np.diff(potential) / (thermal * enhancement)
    current: NDArray[np.float64] = (
        Q * mobility * thermal * enhancement / spacing
        * (bernoulli(argument) * density[:-1] - bernoulli(-argument) * density[1:])
    )
    return current


def legacy_fermi_dirac_face_currents(
    x: ArrayLike,
    phi: ArrayLike,
    n: ArrayLike,
    p: ArrayLike,
    D_n_face: ArrayLike,
    D_p_face: ArrayLike,
    thermal_voltage_V: float,
    bands: BandStructure,
    *,
    conduction_dos_m3: float,
    valence_dos_m3: float,
    recombination_closure: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Historical FD face currents on the physical (unfolded) band edges.

    The mobility handed to the generalized SG is ``D / V_T``; the
    recombination closure must be named explicitly and only ``"off"`` exists.
    """
    if recombination_closure != RECOMBINATION_OFF:
        raise TransportCapabilityError(
            "Fermi-Dirac bulk transport requires an explicit compatible "
            "recombination closure"
        )
    potential = np.asarray(phi, dtype=float)
    spacing = np.diff(np.asarray(x, dtype=float))
    chi = bands.physical_electron_affinity_eV
    gap = bands.physical_band_gap_eV
    J_n = legacy_generalized_sg_current_n(
        potential + chi, n, spacing, np.asarray(D_n_face) / thermal_voltage_V,
        thermal_voltage_V, conduction_dos_m3, statistics=FERMI_DIRAC,
    )
    J_p = legacy_generalized_sg_current_p(
        potential + chi + gap, p, spacing, np.asarray(D_p_face) / thermal_voltage_V,
        thermal_voltage_V, valence_dos_m3, statistics=FERMI_DIRAC,
    )
    return J_n, J_p


@dataclass(frozen=True, slots=True)
class TransportCapabilityRequest:
    """One requested bulk-transport combination, as a device adapter declares it.

    ``carrier_statistics``, ``dopant_ionization`` and ``band_gap_narrowing``
    hold one law per electrical layer. ``inhomogeneous_fields`` names the
    material fields (eps_r, chi, Eg, Nc300, Nv300, BGN parameters) that differ
    between layers. ``recombination_closure`` is the bulk recombination law
    paired with the transport: ``"maxwell_boltzmann"`` (default) or ``"off"``,
    the only one qualified for FD.
    """

    carrier_statistics: tuple[str, ...]
    dopant_ionization: tuple[str, ...]
    band_gap_narrowing: tuple[str, ...]
    recombination_closure: str
    built_in_potential_mode: str | None
    inhomogeneous_fields: tuple[str, ...] = ()
    interface_recombination: bool = False
    optical_generation: bool = False
    band_grading: bool = False
    doping_profiles: bool = False
    contact_floors: bool = False
    selective_contacts: bool = False
    mobile_ions: bool = False
    field_dependent_mobility: bool = False
    advanced_interface_closure: bool = False
    environment_overrides: tuple[str, ...] = ()


def require_carrier_transport(request: TransportCapabilityRequest) -> CarrierStatistics:
    """Admit only qualified statistics/closure combinations; return the law.

    The default path is MB statistics, fully ionized dopants, no BGN and the
    MB recombination closure. Naming the recombination closure ``"off"`` is
    the historical research selector; it admits FD, discrete-level and BGN
    layers only in the qualified topology: one law per field on every layer,
    a homojunction, dark, no interface or contact extras, no ions, no field
    mobility, the semiconductor work-function contact potential and no
    ``SOLARLAB_*`` overrides. FD is never auto-paired with an MB
    recombination law.
    """
    requested = [str(value).strip().lower() for value in request.carrier_statistics]
    layer_count = len(requested)
    if not layer_count:
        raise TransportCapabilityError("at least one electrical layer is required")
    if (len(request.dopant_ionization) != layer_count
            or len(request.band_gap_narrowing) != layer_count):
        raise TransportCapabilityError(
            "research degenerate transport requires material parameters "
            "and one carrier-statistics/dopant-ionization/BGN law on "
            "every electrical layer"
        )
    if BLAKEMORE in requested:
        raise TransportCapabilityError(
            "Blakemore carrier statistics are not implemented; only "
            "'maxwell_boltzmann' and the restricted 'fermi_dirac' law exist"
        )
    models = {normalize_carrier_statistics(value) for value in requested}
    ionization = {
        normalize_dopant_ionization_model(value) for value in request.dopant_ionization
    }
    narrowing = {str(value).strip().lower() for value in request.band_gap_narrowing}
    if not narrowing <= _BAND_GAP_NARROWING_MODELS:
        raise TransportCapabilityError("band-gap narrowing model must be 'off' or 'slotboom'")
    closure = str(request.recombination_closure).strip().lower()
    if closure not in {MAXWELL_BOLTZMANN, RECOMBINATION_OFF}:
        raise TransportCapabilityError(
            f"unknown bulk recombination closure {request.recombination_closure!r}"
        )
    needs_research = (
        FERMI_DIRAC in models
        or ionization != {FULLY_IONIZED}
        or narrowing != {_BAND_GAP_NARROWING_OFF}
    )
    if closure != RECOMBINATION_OFF:
        if needs_research:
            raise TransportCapabilityError(
                "Fermi-Dirac, discrete-level and BGN bulk transport are not "
                "enabled on the default path; they require the explicit "
                "recombination closure 'off' and are never paired with the MB "
                "recombination law automatically"
            )
        return MAXWELL_BOLTZMANN
    if len(models) != 1 or len(ionization) != 1 or len(narrowing) != 1:
        raise TransportCapabilityError(
            "research degenerate transport requires material parameters "
            "and one carrier-statistics/dopant-ionization/BGN law on "
            "every electrical layer"
        )
    disallowed = []
    if request.inhomogeneous_fields:
        disallowed.append(
            "non-homojunction fields=" + ",".join(request.inhomogeneous_fields)
        )
    flags = (
        (request.interface_recombination, "interface recombination"),
        (request.optical_generation, "optical generation"),
        (request.band_grading, "band grading"),
        (request.doping_profiles, "spatial doping profiles"),
        (request.contact_floors, "calibrated contact floors"),
        (request.selective_contacts, "selective contacts"),
        (request.mobile_ions, "mobile ions"),
        (request.field_dependent_mobility, "field-dependent mobility"),
        (request.advanced_interface_closure, "advanced interface closure"),
        (request.built_in_potential_mode != _SEMICONDUCTOR_WORK_FUNCTION,
         "non-thermodynamic contact potential"),
    )
    disallowed.extend(label for active, label in flags if active)
    if request.environment_overrides:
        disallowed.append(
            "environment overrides=" + ",".join(sorted(request.environment_overrides))
        )
    if disallowed:
        raise TransportCapabilityError(
            "research degenerate transport topology rejected: "
            + "; ".join(disallowed)
        )
    return next(iter(models))
