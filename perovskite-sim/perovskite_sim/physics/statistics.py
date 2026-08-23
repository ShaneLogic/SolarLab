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
FULLY_IONIZED = "fully_ionized"
DISCRETE_LEVEL = "discrete_level"
CHARGE_NEUTRALITY_RELATIVE_TOLERANCE = 1.0e-12
CarrierStatistics = Literal["maxwell_boltzmann", "fermi_dirac"]
DopantIonizationModel = Literal["fully_ionized", "discrete_level"]


@dataclass(frozen=True, slots=True)
class DopantChargeState:
    """Ionized donor/acceptor charge and reduced-level derivatives."""

    model: DopantIonizationModel
    ionized_donor_density_m3: float
    ionized_acceptor_density_m3: float
    donor_ionized_fraction: float
    acceptor_ionized_fraction: float
    donor_density_derivative_eta_n_m3: float
    acceptor_density_derivative_eta_p_m3: float


@dataclass(frozen=True, slots=True)
class BulkChargeNeutralityState:
    """One spatially uniform equilibrium carrier/dopant state."""

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
    dopant_ionization_model: DopantIonizationModel
    ionized_acceptor_density_m3: float
    ionized_donor_density_m3: float
    acceptor_ionized_fraction: float
    donor_ionized_fraction: float

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


def normalize_dopant_ionization_model(value: object) -> DopantIonizationModel:
    """Return one supported dopant-ionization identifier or fail closed."""
    if not isinstance(value, str):
        raise ValueError("dopant ionization model must be a string")
    normalized = value.strip().lower()
    if normalized not in {FULLY_IONIZED, DISCRETE_LEVEL}:
        raise ValueError(
            "dopant ionization model must be 'fully_ionized' or "
            "'discrete_level'"
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


def _inverse_one_plus_exp(log_term: float) -> float:
    """Return ``1 / (1 + exp(log_term))`` without overflow."""
    if math.isnan(log_term):
        raise ValueError("dopant occupation exponent must not be NaN")
    if log_term >= 0.0:
        tail = math.exp(-log_term)
        return tail / (1.0 + tail)
    return 1.0 / (1.0 + math.exp(log_term))


def ionized_donor_fraction(
    reduced_electron_fermi_level: float,
    *,
    binding_energy_eV: float,
    thermal_voltage_V: float,
    degeneracy: float = 2.0,
) -> float:
    """Return ``N_D+ / N_D`` for a discrete donor level below ``E_C``."""
    binding = _finite_nonnegative(binding_energy_eV, "binding_energy_eV")
    thermal = _finite_positive(thermal_voltage_V, "thermal_voltage_V")
    factor = _finite_positive(degeneracy, "degeneracy")
    eta_n = float(reduced_electron_fermi_level)
    if math.isnan(eta_n):
        raise ValueError("reduced_electron_fermi_level must not be NaN")
    return _inverse_one_plus_exp(
        math.log(factor) + eta_n + binding / thermal
    )


def ionized_acceptor_fraction(
    reduced_hole_fermi_level: float,
    *,
    binding_energy_eV: float,
    thermal_voltage_V: float,
    degeneracy: float = 4.0,
) -> float:
    """Return ``N_A- / N_A`` for a discrete acceptor level above ``E_V``."""
    binding = _finite_nonnegative(binding_energy_eV, "binding_energy_eV")
    thermal = _finite_positive(thermal_voltage_V, "thermal_voltage_V")
    factor = _finite_positive(degeneracy, "degeneracy")
    eta_p = float(reduced_hole_fermi_level)
    if math.isnan(eta_p):
        raise ValueError("reduced_hole_fermi_level must not be NaN")
    return _inverse_one_plus_exp(
        math.log(factor) + eta_p + binding / thermal
    )


def dopant_charge_state(
    *,
    reduced_electron_fermi_level: float,
    reduced_hole_fermi_level: float,
    donor_density_m3: float,
    acceptor_density_m3: float,
    thermal_voltage_V: float,
    model: DopantIonizationModel | str = FULLY_IONIZED,
    donor_binding_energy_eV: float | None = None,
    acceptor_binding_energy_eV: float | None = None,
    donor_degeneracy: float = 2.0,
    acceptor_degeneracy: float = 4.0,
) -> DopantChargeState:
    """Evaluate fixed or discrete-level dopant charge and exact derivatives."""
    donors = _finite_nonnegative(donor_density_m3, "donor_density_m3")
    acceptors = _finite_nonnegative(acceptor_density_m3, "acceptor_density_m3")
    ionization = normalize_dopant_ionization_model(model)
    if ionization == FULLY_IONIZED:
        if (
            donor_binding_energy_eV is not None
            or acceptor_binding_energy_eV is not None
        ):
            raise ValueError(
                "binding energies require dopant ionization model 'discrete_level'"
            )
        return DopantChargeState(
            model=ionization,
            ionized_donor_density_m3=donors,
            ionized_acceptor_density_m3=acceptors,
            donor_ionized_fraction=1.0,
            acceptor_ionized_fraction=1.0,
            donor_density_derivative_eta_n_m3=0.0,
            acceptor_density_derivative_eta_p_m3=0.0,
        )

    donor_factor = _finite_positive(donor_degeneracy, "donor_degeneracy")
    acceptor_factor = _finite_positive(
        acceptor_degeneracy,
        "acceptor_degeneracy",
    )
    if donors > 0.0 and donor_binding_energy_eV is None:
        raise ValueError("active discrete donors require donor_binding_energy_eV")
    if acceptors > 0.0 and acceptor_binding_energy_eV is None:
        raise ValueError(
            "active discrete acceptors require acceptor_binding_energy_eV"
        )
    donor_binding = _finite_nonnegative(
        0.0 if donor_binding_energy_eV is None else donor_binding_energy_eV,
        "donor_binding_energy_eV",
    )
    acceptor_binding = _finite_nonnegative(
        0.0 if acceptor_binding_energy_eV is None else acceptor_binding_energy_eV,
        "acceptor_binding_energy_eV",
    )
    donor_fraction = ionized_donor_fraction(
        reduced_electron_fermi_level,
        binding_energy_eV=donor_binding,
        thermal_voltage_V=thermal_voltage_V,
        degeneracy=donor_factor,
    )
    acceptor_fraction = ionized_acceptor_fraction(
        reduced_hole_fermi_level,
        binding_energy_eV=acceptor_binding,
        thermal_voltage_V=thermal_voltage_V,
        degeneracy=acceptor_factor,
    )
    ionized_donors = donors * donor_fraction
    ionized_acceptors = acceptors * acceptor_fraction
    return DopantChargeState(
        model=ionization,
        ionized_donor_density_m3=ionized_donors,
        ionized_acceptor_density_m3=ionized_acceptors,
        donor_ionized_fraction=donor_fraction,
        acceptor_ionized_fraction=acceptor_fraction,
        donor_density_derivative_eta_n_m3=(
            -ionized_donors * (1.0 - donor_fraction)
        ),
        acceptor_density_derivative_eta_p_m3=(
            -ionized_acceptors * (1.0 - acceptor_fraction)
        ),
    )


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
        dopant_ionization_model=FULLY_IONIZED,
        ionized_acceptor_density_m3=acceptors,
        ionized_donor_density_m3=donors,
        acceptor_ionized_fraction=1.0,
        donor_ionized_fraction=1.0,
    )


def solve_discrete_level_charge_neutrality(
    *,
    temperature_K: float,
    band_gap_eV: float,
    effective_conduction_dos_m3: float,
    effective_valence_dos_m3: float,
    acceptor_density_m3: float = 0.0,
    donor_density_m3: float = 0.0,
    acceptor_binding_energy_eV: float | None = None,
    donor_binding_energy_eV: float | None = None,
    acceptor_degeneracy: float = 4.0,
    donor_degeneracy: float = 2.0,
    statistics: CarrierStatistics | str = MAXWELL_BOLTZMANN,
) -> BulkChargeNeutralityState:
    """Solve equilibrium neutrality with discrete donor/acceptor levels."""
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
    carrier_model = normalize_carrier_statistics(statistics)
    thermal = thermal_voltage(temperature)
    reduced_gap = band_gap / thermal

    def evaluate(
        eta_n: float,
    ) -> tuple[float, float, float, float, DopantChargeState]:
        eta_p = -reduced_gap - eta_n
        electron = carrier_density_from_reduced_fermi_level(
            eta_n,
            conduction_dos,
            statistics=carrier_model,
        )
        hole = carrier_density_from_reduced_fermi_level(
            eta_p,
            valence_dos,
            statistics=carrier_model,
        )
        dopants = dopant_charge_state(
            reduced_electron_fermi_level=eta_n,
            reduced_hole_fermi_level=eta_p,
            donor_density_m3=donors,
            acceptor_density_m3=acceptors,
            thermal_voltage_V=thermal,
            model=DISCRETE_LEVEL,
            donor_binding_energy_eV=donor_binding_energy_eV,
            acceptor_binding_energy_eV=acceptor_binding_energy_eV,
            donor_degeneracy=donor_degeneracy,
            acceptor_degeneracy=acceptor_degeneracy,
        )
        residual = (
            hole
            - electron
            + dopants.ionized_donor_density_m3
            - dopants.ionized_acceptor_density_m3
        )
        return residual, electron, hole, eta_p, dopants

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
    residual, electron, hole, eta_p, dopants = evaluate(eta_n)
    if not all(
        math.isfinite(value) and value >= 0.0
        for value in (
            electron,
            hole,
            dopants.ionized_donor_density_m3,
            dopants.ionized_acceptor_density_m3,
        )
    ):
        raise FloatingPointError("charge-neutral densities are non-finite")
    charge_scale = max(electron, hole, donors, acceptors, 1.0)
    normalized_residual = abs(residual) / charge_scale
    if not math.isfinite(normalized_residual):
        raise FloatingPointError("charge-neutrality residual is non-finite")
    if normalized_residual > CHARGE_NEUTRALITY_RELATIVE_TOLERANCE:
        raise RuntimeError(
            "charge-neutrality solve exceeded the fixed relative residual gate"
        )
    return BulkChargeNeutralityState(
        statistics=carrier_model,
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
        dopant_ionization_model=DISCRETE_LEVEL,
        ionized_acceptor_density_m3=dopants.ionized_acceptor_density_m3,
        ionized_donor_density_m3=dopants.ionized_donor_density_m3,
        acceptor_ionized_fraction=dopants.acceptor_ionized_fraction,
        donor_ionized_fraction=dopants.donor_ionized_fraction,
    )


def solve_charge_neutrality(
    *,
    dopant_ionization_model: DopantIonizationModel | str = FULLY_IONIZED,
    acceptor_binding_energy_eV: float | None = None,
    donor_binding_energy_eV: float | None = None,
    acceptor_degeneracy: float = 4.0,
    donor_degeneracy: float = 2.0,
    **kwargs: float | str,
) -> BulkChargeNeutralityState:
    """Dispatch to the fixed-charge or discrete-level neutrality closure."""
    model = normalize_dopant_ionization_model(dopant_ionization_model)
    if model == FULLY_IONIZED:
        if (
            acceptor_binding_energy_eV is not None
            or donor_binding_energy_eV is not None
            or acceptor_degeneracy != 4.0
            or donor_degeneracy != 2.0
        ):
            raise ValueError(
                "dopant level parameters require 'discrete_level' ionization"
            )
        return solve_fully_ionized_charge_neutrality(**kwargs)
    return solve_discrete_level_charge_neutrality(
        acceptor_binding_energy_eV=acceptor_binding_energy_eV,
        donor_binding_energy_eV=donor_binding_energy_eV,
        acceptor_degeneracy=acceptor_degeneracy,
        donor_degeneracy=donor_degeneracy,
        **kwargs,
    )


__all__ = [
    "BulkChargeNeutralityState",
    "CHARGE_NEUTRALITY_RELATIVE_TOLERANCE",
    "CarrierStatistics",
    "DISCRETE_LEVEL",
    "DopantChargeState",
    "DopantIonizationModel",
    "FERMI_DIRAC",
    "FULLY_IONIZED",
    "MAXWELL_BOLTZMANN",
    "carrier_density_derivative_reduced_fermi_level",
    "carrier_density_from_reduced_fermi_level",
    "carrier_logarithmic_compressibility",
    "carrier_occupation",
    "dopant_charge_state",
    "generalized_einstein_factor",
    "ionized_acceptor_fraction",
    "ionized_donor_fraction",
    "normalize_carrier_statistics",
    "normalize_dopant_ionization_model",
    "reduced_fermi_level_from_density",
    "solve_charge_neutrality",
    "solve_discrete_level_charge_neutrality",
    "solve_fully_ionized_charge_neutrality",
]
