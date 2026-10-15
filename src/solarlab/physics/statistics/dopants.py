"""Dopant ionization: fully ionized or discrete donor/acceptor levels (M class).

A discrete donor level sits ``binding_energy_eV`` below ``E_C`` and a discrete
acceptor level the same distance above ``E_V``; the ionized fractions are
``1 / (1 + g exp(eta + E_b/kT))`` evaluated without overflow, with exact
derivatives with respect to the reduced Fermi level. Bit-identical to the
legacy engine.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final, Literal

from solarlab.physics.statistics._validation import finite_nonnegative, finite_positive

__all__ = [
    "DISCRETE_LEVEL", "DOPANT_IONIZATION_MODELS", "DopantChargeState",
    "DopantIonizationModel", "FULLY_IONIZED", "dopant_charge_state",
    "ionized_acceptor_fraction", "ionized_donor_fraction",
    "normalize_dopant_ionization_model",
]

FULLY_IONIZED: Final = "fully_ionized"
DISCRETE_LEVEL: Final = "discrete_level"
DOPANT_IONIZATION_MODELS = (FULLY_IONIZED, DISCRETE_LEVEL)
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
    binding = finite_nonnegative(binding_energy_eV, "binding_energy_eV")
    thermal = finite_positive(thermal_voltage_V, "thermal_voltage_V")
    factor = finite_positive(degeneracy, "degeneracy")
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
    binding = finite_nonnegative(binding_energy_eV, "binding_energy_eV")
    thermal = finite_positive(thermal_voltage_V, "thermal_voltage_V")
    factor = finite_positive(degeneracy, "degeneracy")
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
    donors = finite_nonnegative(donor_density_m3, "donor_density_m3")
    acceptors = finite_nonnegative(acceptor_density_m3, "acceptor_density_m3")
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

    donor_factor = finite_positive(donor_degeneracy, "donor_degeneracy")
    acceptor_factor = finite_positive(
        acceptor_degeneracy,
        "acceptor_degeneracy",
    )
    if donors > 0.0 and donor_binding_energy_eV is None:
        raise ValueError("active discrete donors require donor_binding_energy_eV")
    if acceptors > 0.0 and acceptor_binding_energy_eV is None:
        raise ValueError(
            "active discrete acceptors require acceptor_binding_energy_eV"
        )
    donor_binding = finite_nonnegative(
        0.0 if donor_binding_energy_eV is None else donor_binding_energy_eV,
        "donor_binding_energy_eV",
    )
    acceptor_binding = finite_nonnegative(
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
