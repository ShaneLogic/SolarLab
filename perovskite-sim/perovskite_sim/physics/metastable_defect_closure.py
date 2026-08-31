"""Local stationary closure for metastable defect configurations.

Follows Decock, Zabierowski, and Burgelman, J. Appl. Phys. 111 (2012) 043703,
doi:10.1063/1.3686651. One physical defect owns a single total density that is
split between a donor-like and an acceptor-like *configuration*; the two
configurations differ by two elementary charges and interconvert through a
double-carrier process with its own activation barriers.

Rate model
----------
The configuration change moves two electrons, so at equilibrium detailed
balance requires

    k_forward / k_backward = exp(2 (F - E_t) / V_T),

with ``F`` the Fermi level and ``E_t`` the declared transition energy. Writing
each pathway as ``prefactor * activity * exp(-barrier / V_T)`` and its reverse
as ``prefactor * exp(-reverse_barrier / V_T)``, that requirement fixes the
carrier activity of every pathway uniquely:

===================================  ==========================
pathway                              activity
===================================  ==========================
``double_electron_capture``          ``(n / N_C)**2``
``electron_capture_plus_hole_emission``  ``(n / N_C) * (N_V / p)``
``double_hole_capture``              ``(p / N_V)**2``
``hole_capture_plus_electron_emission``  ``(p / N_V) * (N_C / n)``
===================================  ==========================

Those are precisely the factors for which the barrier relations already
validated by :meth:`MetastableConversionKinetics.validate_detailed_balance`
are detailed balance, so the schema and this closure agree by construction
rather than by convention. The equivalence is pinned by a test that solves the
stationary fraction from each pathway alone at thermal equilibrium and
requires the two answers to agree.

The donor configuration is the more positive one (the schema requires the two
selected conversion states to differ by exactly ``+2``), so capturing
electrons drives donor -> acceptor and capturing holes drives the reverse.

Nothing here is clipped. Non-physical inputs — a rate above the declared
phonon attempt frequency, a non-finite activity, a non-positive carrier
density — raise instead of being clamped into range.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

from perovskite_sim.models.multivalent_defects import (
    DOUBLE_ELECTRON_CAPTURE,
    DOUBLE_HOLE_CAPTURE,
    MetastableDefectDefinition,
)
from perovskite_sim.physics.temperature import thermal_voltage


METASTABLE_CONFIGURATION_CLOSURE_VERSION = "metastable-local-configuration-v1"


class MetastableConfigurationClosureError(RuntimeError):
    """The local metastable configuration system was invalid or non-finite."""


def _positive(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be a real number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be a real number") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be finite and positive")
    return result


def _readonly(value: object) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    array.setflags(write=False)
    return array


def _identity(
    definition: MetastableDefectDefinition,
    *,
    band_gap_eV: float,
    effective_conduction_dos_m3: float,
    effective_valence_dos_m3: float,
    temperature_K: float,
) -> str:
    payload = {
        "closure": METASTABLE_CONFIGURATION_CLOSURE_VERSION,
        "source_model": {
            "doi": "10.1063/1.3686651",
            "activity_derivation": "two_electron_detailed_balance",
        },
        "band_gap_eV": band_gap_eV,
        "effective_conduction_dos_m3": effective_conduction_dos_m3,
        "effective_valence_dos_m3": effective_valence_dos_m3,
        "temperature_K": temperature_K,
        "definition": definition.to_dict(),
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class MetastableConfigurationClosureResult:
    """Stationary donor/acceptor configuration split and its local tangents."""

    closure_identity_sha256: str
    defect_name: str
    temperature_K: float
    thermal_voltage_V: float
    band_gap_eV: float
    total_density_m3: float
    electron_capture_path: str
    hole_capture_path: str
    electron_activity: np.ndarray
    hole_activity: np.ndarray
    donor_to_acceptor_rate_s1: np.ndarray
    acceptor_to_donor_rate_s1: np.ndarray
    donor_fraction: np.ndarray
    acceptor_fraction: np.ndarray
    donor_fraction_derivative_n_m3: np.ndarray
    donor_fraction_derivative_p_m3: np.ndarray
    donor_density_m3: np.ndarray
    acceptor_density_m3: np.ndarray
    minimum_donor_fraction: float
    maximum_donor_fraction: float
    maximum_stationary_residual_s1: float
    maximum_rate_s1: float
    phonon_frequency_Hz: float

    def __post_init__(self) -> None:
        digest = str(self.closure_identity_sha256).lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("closure_identity_sha256 must be a SHA-256 hex")
        for field in (
            "temperature_K",
            "thermal_voltage_V",
            "band_gap_eV",
            "total_density_m3",
            "phonon_frequency_Hz",
        ):
            object.__setattr__(self, field, _positive(getattr(self, field), field))
        shape = np.asarray(self.donor_fraction).shape
        for field in (
            "electron_activity",
            "hole_activity",
            "donor_to_acceptor_rate_s1",
            "acceptor_to_donor_rate_s1",
            "donor_fraction",
            "acceptor_fraction",
            "donor_fraction_derivative_n_m3",
            "donor_fraction_derivative_p_m3",
            "donor_density_m3",
            "acceptor_density_m3",
        ):
            values = np.asarray(getattr(self, field), dtype=float)
            if values.shape != shape:
                raise ValueError(f"{field} must match the carrier shape")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{field} must be finite")
            object.__setattr__(self, field, _readonly(values))
        minimum = float(self.minimum_donor_fraction)
        maximum = float(self.maximum_donor_fraction)
        residual = float(self.maximum_stationary_residual_s1)
        peak_rate = float(self.maximum_rate_s1)
        if (
            not all(
                math.isfinite(value)
                for value in (minimum, maximum, residual, peak_rate)
            )
            or minimum < 0.0
            or maximum > 1.0
            or residual < 0.0
            or peak_rate <= 0.0
        ):
            raise ValueError("metastable closure diagnostics are inconsistent")
        object.__setattr__(self, "minimum_donor_fraction", minimum)
        object.__setattr__(self, "maximum_donor_fraction", maximum)
        object.__setattr__(self, "maximum_stationary_residual_s1", residual)
        object.__setattr__(self, "maximum_rate_s1", peak_rate)
        object.__setattr__(self, "closure_identity_sha256", digest)


def evaluate_metastable_configuration_closure(
    electron_density_m3: np.ndarray | float,
    hole_density_m3: np.ndarray | float,
    definition: MetastableDefectDefinition,
    *,
    band_gap_eV: float,
    effective_conduction_dos_m3: float,
    effective_valence_dos_m3: float,
    temperature_K: float,
) -> MetastableConfigurationClosureResult:
    """Solve the stationary donor/acceptor configuration split at (n, p)."""

    if not isinstance(definition, MetastableDefectDefinition):
        raise TypeError("definition must be a MetastableDefectDefinition")
    n, p = np.broadcast_arrays(
        np.asarray(electron_density_m3, dtype=float),
        np.asarray(hole_density_m3, dtype=float),
    )
    if (
        not np.all(np.isfinite(n))
        or not np.all(np.isfinite(p))
        or np.any(n <= 0.0)
        or np.any(p <= 0.0)
    ):
        raise ValueError(
            "metastable closure carrier densities must be finite and positive"
        )
    gap = _positive(band_gap_eV, "band_gap_eV")
    conduction_dos = _positive(
        effective_conduction_dos_m3, "effective_conduction_dos_m3"
    )
    valence_dos = _positive(effective_valence_dos_m3, "effective_valence_dos_m3")
    temperature = _positive(temperature_K, "temperature_K")
    definition.validate_band_gap(gap)

    thermal = thermal_voltage(temperature)
    kinetics = definition.conversion_kinetics
    reduced_n = n / conduction_dos
    reduced_p = p / valence_dos

    if kinetics.electron_capture_path == DOUBLE_ELECTRON_CAPTURE:
        electron_activity = reduced_n**2
    else:
        electron_activity = reduced_n / reduced_p
    if kinetics.hole_capture_path == DOUBLE_HOLE_CAPTURE:
        hole_activity = reduced_p**2
    else:
        hole_activity = reduced_p / reduced_n

    # Each pathway keeps one prefactor across its capture and emission legs, so
    # the barrier relations frozen in the schema remain exact detailed balance.
    electron_prefactor = kinetics.capture_n_m3_s * conduction_dos
    hole_prefactor = kinetics.capture_p_m3_s * valence_dos
    electron_capture = (
        electron_prefactor
        * electron_activity
        * math.exp(-kinetics.electron_capture_activation_eV / thermal)
    )
    electron_emission = electron_prefactor * math.exp(
        -kinetics.electron_emission_activation_eV / thermal
    )
    hole_capture = (
        hole_prefactor
        * hole_activity
        * math.exp(-kinetics.hole_capture_activation_eV / thermal)
    )
    hole_emission = hole_prefactor * math.exp(
        -kinetics.hole_emission_activation_eV / thermal
    )

    # Donor is the more positive configuration: capturing electrons converts it
    # to the acceptor configuration, emitting them converts it back.
    donor_to_acceptor = electron_capture + hole_emission
    acceptor_to_donor = hole_capture + electron_emission
    total_rate = donor_to_acceptor + acceptor_to_donor
    if (
        not np.all(np.isfinite(total_rate))
        or np.any(donor_to_acceptor <= 0.0)
        or np.any(acceptor_to_donor <= 0.0)
    ):
        raise MetastableConfigurationClosureError(
            "metastable conversion rates must be finite and positive"
        )
    peak_rate = float(np.max(total_rate))
    if peak_rate > kinetics.phonon_frequency_Hz:
        raise MetastableConfigurationClosureError(
            "metastable conversion rate exceeds the declared phonon attempt "
            f"frequency: {peak_rate:.6g} > {kinetics.phonon_frequency_Hz:.6g} Hz"
        )

    donor_fraction = acceptor_to_donor / total_rate
    acceptor_fraction = donor_to_acceptor / total_rate
    residual = np.abs(
        acceptor_to_donor * acceptor_fraction - donor_to_acceptor * donor_fraction
    )

    # d(donor_fraction)/dz with the quotient rule on the two rate sums.
    if kinetics.electron_capture_path == DOUBLE_ELECTRON_CAPTURE:
        d_electron_capture_dn = 2.0 * electron_capture / n
        d_electron_capture_dp = np.zeros_like(n)
    else:
        d_electron_capture_dn = electron_capture / n
        d_electron_capture_dp = -electron_capture / p
    if kinetics.hole_capture_path == DOUBLE_HOLE_CAPTURE:
        d_hole_capture_dp = 2.0 * hole_capture / p
        d_hole_capture_dn = np.zeros_like(n)
    else:
        d_hole_capture_dp = hole_capture / p
        d_hole_capture_dn = -hole_capture / n
    d_forward_dn = d_electron_capture_dn
    d_forward_dp = d_electron_capture_dp
    d_backward_dn = d_hole_capture_dn
    d_backward_dp = d_hole_capture_dp
    donor_derivative_n = (
        d_backward_dn * total_rate - acceptor_to_donor * (d_forward_dn + d_backward_dn)
    ) / total_rate**2
    donor_derivative_p = (
        d_backward_dp * total_rate - acceptor_to_donor * (d_forward_dp + d_backward_dp)
    ) / total_rate**2

    density = definition.total_density_m3
    outputs = (
        donor_fraction,
        acceptor_fraction,
        donor_derivative_n,
        donor_derivative_p,
    )
    if not all(np.all(np.isfinite(value)) for value in outputs):
        raise MetastableConfigurationClosureError(
            "metastable stationary closure produced non-finite output"
        )
    minimum = float(np.min(donor_fraction))
    maximum = float(np.max(donor_fraction))
    if minimum < 0.0 or maximum > 1.0:
        raise MetastableConfigurationClosureError(
            "metastable configuration fraction left [0, 1] without clipping"
        )

    return MetastableConfigurationClosureResult(
        closure_identity_sha256=_identity(
            definition,
            band_gap_eV=gap,
            effective_conduction_dos_m3=conduction_dos,
            effective_valence_dos_m3=valence_dos,
            temperature_K=temperature,
        ),
        defect_name=definition.name,
        temperature_K=temperature,
        thermal_voltage_V=thermal,
        band_gap_eV=gap,
        total_density_m3=density,
        electron_capture_path=kinetics.electron_capture_path,
        hole_capture_path=kinetics.hole_capture_path,
        electron_activity=electron_activity,
        hole_activity=hole_activity,
        donor_to_acceptor_rate_s1=donor_to_acceptor,
        acceptor_to_donor_rate_s1=acceptor_to_donor,
        donor_fraction=donor_fraction,
        acceptor_fraction=acceptor_fraction,
        donor_fraction_derivative_n_m3=donor_derivative_n,
        donor_fraction_derivative_p_m3=donor_derivative_p,
        donor_density_m3=density * donor_fraction,
        acceptor_density_m3=density * acceptor_fraction,
        minimum_donor_fraction=minimum,
        maximum_donor_fraction=maximum,
        maximum_stationary_residual_s1=float(np.max(residual)),
        maximum_rate_s1=peak_rate,
        phonon_frequency_Hz=kinetics.phonon_frequency_Hz,
    )


__all__ = [
    "METASTABLE_CONFIGURATION_CLOSURE_VERSION",
    "MetastableConfigurationClosureError",
    "MetastableConfigurationClosureResult",
    "evaluate_metastable_configuration_closure",
]
