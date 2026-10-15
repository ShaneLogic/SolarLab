"""One carrier-statistics definition for MB and restricted FD (M class).

Reduced Fermi levels follow the semiconductor convention
``eta_n = (E_F - E_C) / kT`` and ``eta_p = (E_V - E_F) / kT``. The forward
map, its inverse, the logarithmic compressibility ``d log n / d eta``, the
generalized Einstein factor and ``dn/d eta`` all dispatch through this module,
so a neutrality solve, a contact reservoir and a transport face cannot use
different statistics. Values are bit-identical to the legacy engine.
"""

from __future__ import annotations

import math
from typing import Final, Literal

from solarlab.physics.statistics._validation import finite_nonnegative, finite_positive
from solarlab.physics.statistics.fermi_dirac import (
    fermi_dirac_half, fermi_dirac_half_log_derivative, inverse_fermi_dirac_half,
)

__all__ = [
    "CARRIER_STATISTICS", "CarrierStatistics", "FERMI_DIRAC", "MAXWELL_BOLTZMANN",
    "boltzmann_activity", "carrier_density_derivative_reduced_fermi_level",
    "carrier_density_from_reduced_fermi_level", "carrier_logarithmic_compressibility",
    "carrier_occupation", "generalized_einstein_factor", "normalize_carrier_statistics",
    "reduced_fermi_level_from_density",
]

MAXWELL_BOLTZMANN: Final = "maxwell_boltzmann"
FERMI_DIRAC: Final = "fermi_dirac"
CARRIER_STATISTICS = (MAXWELL_BOLTZMANN, FERMI_DIRAC)
CarrierStatistics = Literal["maxwell_boltzmann", "fermi_dirac"]


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


def boltzmann_activity(eta: float) -> float:
    """``exp(eta)`` with overflow mapped to ``+inf``."""
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
        return boltzmann_activity(eta)
    return fermi_dirac_half(eta)


def carrier_density_from_reduced_fermi_level(
    reduced_fermi_level: float,
    effective_density_of_states_m3: float,
    *,
    statistics: CarrierStatistics | str = MAXWELL_BOLTZMANN,
) -> float:
    """Return a bulk carrier density from its reduced Fermi level."""
    density_of_states = finite_positive(
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
    density = finite_nonnegative(carrier_density_m3, "carrier_density_m3")
    density_of_states = finite_positive(
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
