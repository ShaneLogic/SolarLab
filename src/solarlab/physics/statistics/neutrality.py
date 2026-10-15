"""Bulk charge neutrality and the doping mass-action closure (M class).

``solve_charge_neutrality`` finds the common Fermi level of
``p - n + N_D+ - N_A- = 0`` by bracketing around the intrinsic level and
bisecting on ``eta_n``, with a fixed ``1e-12`` relative residual gate; both
dopant closures share one bracket and bisection sequence and one carrier
statistics definition. ``mass_action_carrier_densities`` is the two-branch
Maxwell-Boltzmann closed form behind the historical contact reservoirs. All
results are bit-identical to the legacy engine.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

import numpy as np

from solarlab.physics.statistics._validation import finite_nonnegative, finite_positive
from solarlab.physics.statistics.carriers import (
    CarrierStatistics, MAXWELL_BOLTZMANN, carrier_density_from_reduced_fermi_level,
    normalize_carrier_statistics,
)
from solarlab.physics.statistics.dopants import (
    DISCRETE_LEVEL, FULLY_IONIZED, DopantChargeState, DopantIonizationModel,
    dopant_charge_state, normalize_dopant_ionization_model,
)
from solarlab.physics.statistics.thermal import thermal_voltage

__all__ = [
    "BulkChargeNeutralityState", "CHARGE_NEUTRALITY_RELATIVE_TOLERANCE",
    "mass_action_carrier_densities", "solve_charge_neutrality",
    "solve_discrete_level_charge_neutrality", "solve_fully_ionized_charge_neutrality",
]

CHARGE_NEUTRALITY_RELATIVE_TOLERANCE = 1.0e-12


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


def mass_action_carrier_densities(
    donor_density_m3: float,
    acceptor_density_m3: float,
    intrinsic_density_m3: float,
) -> tuple[float, float]:
    """Two-branch MB equilibrium ``(n, p)`` from net doping and ``n_i``.

    ``net = (N_D - N_A)/2`` and ``disc = sqrt(net^2 + n_i^2)``; the majority
    carrier is ``|net| + disc`` and the minority carrier ``n_i^2`` divided by
    it, so ``n p = n_i^2`` without cancellation. The arithmetic runs on
    binary64 NumPy scalars in the historical order.
    """
    donors = np.float64(finite_nonnegative(donor_density_m3, "donor_density_m3"))
    acceptors = np.float64(finite_nonnegative(acceptor_density_m3, "acceptor_density_m3"))
    intrinsic = np.float64(finite_nonnegative(intrinsic_density_m3, "intrinsic_density_m3"))
    if donors == acceptors and intrinsic == 0.0:
        raise ValueError("an exactly compensated contact needs a positive intrinsic density")
    net = 0.5 * (donors - acceptors)
    disc = np.sqrt(net ** 2 + intrinsic ** 2)
    if net >= 0:
        electrons = net + disc
        holes = intrinsic ** 2 / electrons
    else:
        holes = -net + disc
        electrons = intrinsic ** 2 / holes
    return float(electrons), float(holes)


def _neutral_reduced_electron_level(
    residual: Callable[[float], float],
    *,
    reduced_gap: float,
    conduction_dos: float,
    valence_dos: float,
) -> float:
    """Bracket around the intrinsic level, then bisect on ``eta_n``."""
    intrinsic_center = 0.5 * (
        -reduced_gap + math.log(valence_dos / conduction_dos)
    )
    half_width = max(32.0, 0.5 * reduced_gap + 8.0)
    lower = intrinsic_center - half_width
    upper = intrinsic_center + half_width
    lower_residual = residual(lower)
    upper_residual = residual(upper)
    for _ in range(32):
        if lower_residual >= 0.0 and upper_residual <= 0.0:
            break
        half_width *= 2.0
        lower = intrinsic_center - half_width
        upper = intrinsic_center + half_width
        lower_residual = residual(lower)
        upper_residual = residual(upper)
    else:
        raise RuntimeError("could not bracket the charge-neutral Fermi level")

    for _ in range(200):
        midpoint = 0.5 * (lower + upper)
        if residual(midpoint) > 0.0:
            lower = midpoint
        else:
            upper = midpoint
        coordinate_tolerance = max(2.0e-14, 4.0 * math.ulp(midpoint))
        if upper - lower <= coordinate_tolerance:
            break
    return 0.5 * (lower + upper)


def _normalized_residual(residual: float, electron: float, hole: float,
                         donors: float, acceptors: float) -> float:
    charge_scale = max(electron, hole, donors, acceptors, 1.0)
    normalized = abs(residual) / charge_scale
    if not math.isfinite(normalized):
        raise FloatingPointError("charge-neutrality residual is non-finite")
    if normalized > CHARGE_NEUTRALITY_RELATIVE_TOLERANCE:
        raise RuntimeError(
            "charge-neutrality solve exceeded the fixed relative residual gate"
        )
    return normalized


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
    """Solve ``p - n + N_D - N_A = 0`` for a common Fermi level."""
    temperature = finite_positive(temperature_K, "temperature_K")
    band_gap = finite_nonnegative(band_gap_eV, "band_gap_eV")
    conduction_dos = finite_positive(
        effective_conduction_dos_m3,
        "effective_conduction_dos_m3",
    )
    valence_dos = finite_positive(
        effective_valence_dos_m3,
        "effective_valence_dos_m3",
    )
    acceptors = finite_nonnegative(acceptor_density_m3, "acceptor_density_m3")
    donors = finite_nonnegative(donor_density_m3, "donor_density_m3")
    model = normalize_carrier_statistics(statistics)
    thermal = thermal_voltage(temperature)
    reduced_gap = band_gap / thermal
    net_donor_density = donors - acceptors

    def evaluate(eta_n: float) -> tuple[float, float, float, float]:
        eta_p = -reduced_gap - eta_n
        electron = carrier_density_from_reduced_fermi_level(
            eta_n, conduction_dos, statistics=model,
        )
        hole = carrier_density_from_reduced_fermi_level(
            eta_p, valence_dos, statistics=model,
        )
        return hole - electron + net_donor_density, electron, hole, eta_p

    eta_n = _neutral_reduced_electron_level(
        lambda value: evaluate(value)[0],
        reduced_gap=reduced_gap,
        conduction_dos=conduction_dos,
        valence_dos=valence_dos,
    )
    residual, electron, hole, eta_p = evaluate(eta_n)
    if not all(
        math.isfinite(value) and value >= 0.0
        for value in (electron, hole)
    ):
        raise FloatingPointError("charge-neutral carrier densities are non-finite")
    normalized_residual = _normalized_residual(residual, electron, hole, donors, acceptors)
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
    temperature = finite_positive(temperature_K, "temperature_K")
    band_gap = finite_nonnegative(band_gap_eV, "band_gap_eV")
    conduction_dos = finite_positive(
        effective_conduction_dos_m3,
        "effective_conduction_dos_m3",
    )
    valence_dos = finite_positive(
        effective_valence_dos_m3,
        "effective_valence_dos_m3",
    )
    acceptors = finite_nonnegative(acceptor_density_m3, "acceptor_density_m3")
    donors = finite_nonnegative(donor_density_m3, "donor_density_m3")
    carrier_model = normalize_carrier_statistics(statistics)
    thermal = thermal_voltage(temperature)
    reduced_gap = band_gap / thermal

    def evaluate(
        eta_n: float,
    ) -> tuple[float, float, float, float, DopantChargeState]:
        eta_p = -reduced_gap - eta_n
        electron = carrier_density_from_reduced_fermi_level(
            eta_n, conduction_dos, statistics=carrier_model,
        )
        hole = carrier_density_from_reduced_fermi_level(
            eta_p, valence_dos, statistics=carrier_model,
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

    eta_n = _neutral_reduced_electron_level(
        lambda value: evaluate(value)[0],
        reduced_gap=reduced_gap,
        conduction_dos=conduction_dos,
        valence_dos=valence_dos,
    )
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
    normalized_residual = _normalized_residual(residual, electron, hole, donors, acceptors)
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
    temperature_K: float,
    band_gap_eV: float,
    effective_conduction_dos_m3: float,
    effective_valence_dos_m3: float,
    acceptor_density_m3: float = 0.0,
    donor_density_m3: float = 0.0,
    statistics: CarrierStatistics | str = MAXWELL_BOLTZMANN,
) -> BulkChargeNeutralityState:
    """Dispatch to the fixed-charge or discrete-level neutrality closure."""
    model = normalize_dopant_ionization_model(dopant_ionization_model)
    common = dict(
        temperature_K=temperature_K,
        band_gap_eV=band_gap_eV,
        effective_conduction_dos_m3=effective_conduction_dos_m3,
        effective_valence_dos_m3=effective_valence_dos_m3,
        acceptor_density_m3=acceptor_density_m3,
        donor_density_m3=donor_density_m3,
    )
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
        return solve_fully_ionized_charge_neutrality(**common, statistics=statistics)
    return solve_discrete_level_charge_neutrality(
        **common,
        acceptor_binding_energy_eV=acceptor_binding_energy_eV,
        donor_binding_energy_eV=donor_binding_energy_eV,
        acceptor_degeneracy=acceptor_degeneracy,
        donor_degeneracy=donor_degeneracy,
        statistics=statistics,
    )
