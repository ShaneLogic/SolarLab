"""Bulk carrier-statistics constitutive primitives.

The reduced Fermi levels follow the semiconductor convention

``eta_n = (E_F - E_C) / kT`` and ``eta_p = (E_V - E_F) / kT``.

This module is intentionally independent of the production drift-diffusion
assembly.  It provides the thermodynamic closure needed to introduce bulk
Fermi-Dirac statistics without changing the repository's default
Maxwell-Boltzmann path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

from perovskite_sim.physics.fermi_dirac import (
    fermi_dirac_half,
    fermi_dirac_half_log_derivative,
    inverse_fermi_dirac_half,
)
from perovskite_sim.physics.temperature import thermal_voltage


MAXWELL_BOLTZMANN = "maxwell_boltzmann"
FERMI_DIRAC = "fermi_dirac"
CHARGE_NEUTRALITY_RELATIVE_TOLERANCE = 1.0e-12
CarrierStatistics = Literal["maxwell_boltzmann", "fermi_dirac"]


@dataclass(frozen=True, slots=True)
class BulkChargeNeutralityState:
    """One fully-ionized, spatially uniform equilibrium state."""

    statistics: CarrierStatistics
    temperature_K: float
    thermal_voltage_V: float
    band_gap_eV: float
    effective_conduction_dos_m3: float
    effective_valence_dos_m3: float
    acceptor_density_m3: float
    donor_density_m3: float
    reduced_electron_fermi_level: float
    reduced_hole_fermi_level: float
    electron_density_m3: float
    hole_density_m3: float
    normalized_charge_residual: float

    @property
    def fermi_level_above_conduction_eV(self) -> float:
        return self.thermal_voltage_V * self.reduced_electron_fermi_level

    @property
    def electron_degeneracy_ratio(self) -> float:
        return self.electron_density_m3 / self.effective_conduction_dos_m3

    @property
    def hole_degeneracy_ratio(self) -> float:
        return self.hole_density_m3 / self.effective_valence_dos_m3


def normalize_carrier_statistics(value: object) -> CarrierStatistics:
    """Return one supported carrier-statistics identifier or fail closed."""
    if not isinstance(value, str):
        raise ValueError("carrier statistics must be a string")
    normalized = value.strip().lower()
    if normalized not in {MAXWELL_BOLTZMANN, FERMI_DIRAC}:
        raise ValueError(
            "carrier statistics must be 'maxwell_boltzmann' or 'fermi_dirac'"
        )
    return normalized  # type: ignore[return-value]


def _finite_positive(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _finite_nonnegative(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _boltzmann_activity(eta: float) -> float:
    try:
        return math.exp(eta)
    except OverflowError:
        return math.inf


def carrier_occupation(
    reduced_fermi_level: float,
    *,
    statistics: CarrierStatistics | str = MAXWELL_BOLTZMANN,
) -> float:
    """Return ``n/N_C`` or ``p/N_V`` for one reduced Fermi level."""
    eta = float(reduced_fermi_level)
    if math.isnan(eta):
        raise ValueError("reduced_fermi_level must not be NaN")
    model = normalize_carrier_statistics(statistics)
    if model == MAXWELL_BOLTZMANN:
        return _boltzmann_activity(eta)
    return fermi_dirac_half(eta)


def carrier_density_from_reduced_fermi_level(
    reduced_fermi_level: float,
    effective_density_of_states_m3: float,
    *,
    statistics: CarrierStatistics | str = MAXWELL_BOLTZMANN,
) -> float:
    """Return a bulk carrier density from its reduced Fermi level."""
    density_of_states = _finite_positive(
        effective_density_of_states_m3,
        "effective_density_of_states_m3",
    )
    occupation = carrier_occupation(
        reduced_fermi_level,
        statistics=statistics,
    )
    density = density_of_states * occupation
    return float(density)


def reduced_fermi_level_from_density(
    carrier_density_m3: float,
    effective_density_of_states_m3: float,
    *,
    statistics: CarrierStatistics | str = MAXWELL_BOLTZMANN,
) -> float:
    """Invert the bulk density law for ``eta_n`` or ``eta_p``."""
    density = _finite_nonnegative(carrier_density_m3, "carrier_density_m3")
    density_of_states = _finite_positive(
        effective_density_of_states_m3,
        "effective_density_of_states_m3",
    )
    if density == 0.0:
        return -math.inf
    ratio = density / density_of_states
    model = normalize_carrier_statistics(statistics)
    if model == MAXWELL_BOLTZMANN:
        return math.log(ratio)
    return inverse_fermi_dirac_half(ratio)


def carrier_logarithmic_compressibility(
    reduced_fermi_level: float,
    *,
    statistics: CarrierStatistics | str = MAXWELL_BOLTZMANN,
) -> float:
    """Return ``d(log n)/d eta`` (or the equivalent hole derivative)."""
    eta = float(reduced_fermi_level)
    if not math.isfinite(eta):
        raise ValueError("reduced_fermi_level must be finite")
    model = normalize_carrier_statistics(statistics)
    if model == MAXWELL_BOLTZMANN:
        return 1.0
    derivative = float(fermi_dirac_half_log_derivative(eta))
    if not math.isfinite(derivative) or derivative <= 0.0:
        raise FloatingPointError("Fermi-Dirac compressibility must be positive")
    return derivative


def generalized_einstein_factor(
    reduced_fermi_level: float,
    *,
    statistics: CarrierStatistics | str = MAXWELL_BOLTZMANN,
) -> float:
    """Return ``D/(mu*V_T) = 1 / d(log n)/d eta``."""
    return 1.0 / carrier_logarithmic_compressibility(
        reduced_fermi_level,
        statistics=statistics,
    )


def carrier_density_derivative_reduced_fermi_level(
    reduced_fermi_level: float,
    effective_density_of_states_m3: float,
    *,
    statistics: CarrierStatistics | str = MAXWELL_BOLTZMANN,
) -> float:
    """Return ``dn/deta`` or ``dp/deta`` for the chosen statistics."""
    density = carrier_density_from_reduced_fermi_level(
        reduced_fermi_level,
        effective_density_of_states_m3,
        statistics=statistics,
    )
    return density * carrier_logarithmic_compressibility(
        reduced_fermi_level,
        statistics=statistics,
    )


def solve_fully_ionized_charge_neutrality(
    *,
    temperature_K: float,
    band_gap_eV: float,
    effective_conduction_dos_m3: float,
    effective_valence_dos_m3: float,
    acceptor_density_m3: float = 0.0,
    donor_density_m3: float = 0.0,
    statistics: CarrierStatistics | str = MAXWELL_BOLTZMANN,
) -> BulkChargeNeutralityState:
    """Solve ``p - n + N_D - N_A = 0`` for a common Fermi level.

    Dopants are fully ionized in this first closure. Incomplete ionization is a
    separate constitutive layer because it introduces donor/acceptor energy and
    degeneracy parameters into the same nonlinear neutrality equation.
    """
    temperature = _finite_positive(temperature_K, "temperature_K")
    band_gap = _finite_nonnegative(band_gap_eV, "band_gap_eV")
    conduction_dos = _finite_positive(
        effective_conduction_dos_m3,
        "effective_conduction_dos_m3",
    )
    valence_dos = _finite_positive(
        effective_valence_dos_m3,
        "effective_valence_dos_m3",
    )
    acceptors = _finite_nonnegative(acceptor_density_m3, "acceptor_density_m3")
    donors = _finite_nonnegative(donor_density_m3, "donor_density_m3")
    model = normalize_carrier_statistics(statistics)
    thermal = thermal_voltage(temperature)
    reduced_gap = band_gap / thermal
    net_donor_density = donors - acceptors

    def evaluate(eta_n: float) -> tuple[float, float, float, float]:
        eta_p = -reduced_gap - eta_n
        electron = carrier_density_from_reduced_fermi_level(
            eta_n,
            conduction_dos,
            statistics=model,
        )
        hole = carrier_density_from_reduced_fermi_level(
            eta_p,
            valence_dos,
            statistics=model,
        )
        residual = hole - electron + net_donor_density
        return residual, electron, hole, eta_p

    intrinsic_center = 0.5 * (
        -reduced_gap + math.log(valence_dos / conduction_dos)
    )
    half_width = max(32.0, 0.5 * reduced_gap + 8.0)
    lower = intrinsic_center - half_width
    upper = intrinsic_center + half_width
    lower_residual = evaluate(lower)[0]
    upper_residual = evaluate(upper)[0]
    for _ in range(32):
        if lower_residual >= 0.0 and upper_residual <= 0.0:
            break
        half_width *= 2.0
        lower = intrinsic_center - half_width
        upper = intrinsic_center + half_width
        lower_residual = evaluate(lower)[0]
        upper_residual = evaluate(upper)[0]
    else:
        raise RuntimeError("could not bracket the charge-neutral Fermi level")

    for _ in range(200):
        midpoint = 0.5 * (lower + upper)
        residual = evaluate(midpoint)[0]
        if residual > 0.0:
            lower = midpoint
        else:
            upper = midpoint
        coordinate_tolerance = max(2.0e-14, 4.0 * math.ulp(midpoint))
        if upper - lower <= coordinate_tolerance:
            break

    eta_n = 0.5 * (lower + upper)
    residual, electron, hole, eta_p = evaluate(eta_n)
    if not all(
        math.isfinite(value) and value >= 0.0
        for value in (electron, hole)
    ):
        raise FloatingPointError("charge-neutral carrier densities are non-finite")
    charge_scale = max(electron, hole, donors, acceptors, 1.0)
    normalized_residual = abs(residual) / charge_scale
    if not math.isfinite(normalized_residual):
        raise FloatingPointError("charge-neutrality residual is non-finite")
    if normalized_residual > CHARGE_NEUTRALITY_RELATIVE_TOLERANCE:
        raise RuntimeError(
            "charge-neutrality solve exceeded the fixed relative residual gate"
        )

    return BulkChargeNeutralityState(
        statistics=model,
        temperature_K=temperature,
        thermal_voltage_V=thermal,
        band_gap_eV=band_gap,
        effective_conduction_dos_m3=conduction_dos,
        effective_valence_dos_m3=valence_dos,
        acceptor_density_m3=acceptors,
        donor_density_m3=donors,
        reduced_electron_fermi_level=eta_n,
        reduced_hole_fermi_level=eta_p,
        electron_density_m3=electron,
        hole_density_m3=hole,
        normalized_charge_residual=normalized_residual,
    )


__all__ = [
    "BulkChargeNeutralityState",
    "CHARGE_NEUTRALITY_RELATIVE_TOLERANCE",
    "CarrierStatistics",
    "FERMI_DIRAC",
    "MAXWELL_BOLTZMANN",
    "carrier_density_derivative_reduced_fermi_level",
    "carrier_density_from_reduced_fermi_level",
    "carrier_logarithmic_compressibility",
    "carrier_occupation",
    "generalized_einstein_factor",
    "normalize_carrier_statistics",
    "reduced_fermi_level_from_density",
    "solve_fully_ionized_charge_neutrality",
]
