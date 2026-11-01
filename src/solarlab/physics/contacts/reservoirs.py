"""Equilibrium contact reservoirs, Schottky and metal-floor densities (M class).

A contact reservoir is the carrier density the contact holds in equilibrium.
Its source is declared separately from the exchange kinetics
(:mod:`solarlab.physics.contacts.exchange`) and from the Poisson boundary
(:mod:`solarlab.physics.contacts.potential`):

* ``doping_equilibrium_densities`` -- the historical default, the two-branch
  MB mass-action closure of the contact node's net doping and ``n_i^2``;
* ``semiconductor_contact_state`` -- one statistics/ionization closure (MB or
  restricted FD, fully ionized or discrete levels) for both the reservoir
  densities and the semiconductor work function;
* ``schottky_equilibrium_n/p`` -- ``N exp(-phi_B/V_T)`` behind a barrier;
* ``metal_reservoir_floor`` -- the calibrated flat-band metal floor that
  raises a weakly doped contact's majority reservoir to ``N exp(-phi_B/V_T)``.

Band-gap narrowing, bulk traps and explicit defects at a contact belong to
later slices and are rejected rather than silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from solarlab.physics.statistics.carriers import (
    FERMI_DIRAC, MAXWELL_BOLTZMANN, CarrierStatistics, normalize_carrier_statistics,
)
from solarlab.physics.statistics.dopants import (
    DISCRETE_LEVEL, FULLY_IONIZED, normalize_dopant_ionization_model,
)
from solarlab.physics.statistics.neutrality import (
    BulkChargeNeutralityState, mass_action_carrier_densities, solve_charge_neutrality,
)
from solarlab.physics.statistics.thermal import thermal_voltage

__all__ = [
    "ContactClosureCapabilityError", "ContactMaterial", "SemiconductorContactState",
    "doping_equilibrium_densities", "metal_reservoir_floor", "schottky_equilibrium_n",
    "schottky_equilibrium_p", "semiconductor_contact_state", "semiconductor_work_function",
]

_BAND_GAP_NARROWING_OFF = "off"


class ContactClosureCapabilityError(ValueError):
    """A contact closure needs physics that this slice does not provide."""


@dataclass(frozen=True, slots=True)
class ContactMaterial:
    """Equilibrium inputs of one contact layer at the contact temperature.

    ``band_gap_eV`` and the effective DOS are already resolved at
    ``temperature_K`` (Varshni shift and ``(T/300)^1.5`` DOS scaling are applied
    by the caller when its temperature model is active), exactly as the
    historical contact closures receive them. ``explicit_defects`` means
    explicit defect species that carry charge in the contact neutrality
    (charged, energy-distributed or multivalent); neutral single-level species
    do not enter it and must not set the flag.
    """

    temperature_K: float
    electron_affinity_eV: float
    band_gap_eV: float
    conduction_dos_m3: float
    valence_dos_m3: float
    donor_density_m3: float
    acceptor_density_m3: float
    carrier_statistics: str = MAXWELL_BOLTZMANN
    dopant_ionization_model: str = FULLY_IONIZED
    donor_binding_energy_eV: float | None = None
    acceptor_binding_energy_eV: float | None = None
    donor_degeneracy: float = 2.0
    acceptor_degeneracy: float = 4.0
    band_gap_narrowing_model: str = _BAND_GAP_NARROWING_OFF
    bulk_trap_distribution: bool = False
    explicit_defects: bool = False

    def requires_statistics_closure(self) -> bool:
        """Historical dispatch: anything beyond MB/fully ionized uses the closure."""
        return (
            normalize_carrier_statistics(self.carrier_statistics) == FERMI_DIRAC
            or normalize_dopant_ionization_model(self.dopant_ionization_model) == DISCRETE_LEVEL
            or self.band_gap_narrowing_model != _BAND_GAP_NARROWING_OFF
            or self.bulk_trap_distribution
            or self.explicit_defects
        )


@dataclass(frozen=True, slots=True)
class SemiconductorContactState:
    """Thermodynamically closed equilibrium state at one contact face."""

    statistics: CarrierStatistics
    electron_affinity_eV: float
    band_gap_eV: float
    work_function_eV: float
    neutrality: BulkChargeNeutralityState
    band_gap_narrowing_eV: float = 0.0
    conduction_band_shift_eV: float = 0.0
    valence_band_shift_eV: float = 0.0

    @property
    def electron_density_m3(self) -> float:
        return self.neutrality.electron_density_m3

    @property
    def hole_density_m3(self) -> float:
        return self.neutrality.hole_density_m3

    @property
    def reduced_electron_fermi_level(self) -> float:
        return self.neutrality.reduced_electron_fermi_level

    @property
    def reduced_hole_fermi_level(self) -> float:
        return self.neutrality.reduced_hole_fermi_level


def _require_positive_band_inputs(material: ContactMaterial, message: str) -> None:
    required = {
        "chi": material.electron_affinity_eV,
        "Eg": material.band_gap_eV,
        "Nc300": material.conduction_dos_m3,
        "Nv300": material.valence_dos_m3,
    }
    invalid = [
        name for name, value in required.items()
        if value is None or not math.isfinite(float(value)) or float(value) <= 0.0
    ]
    if invalid:
        raise ValueError(message + ", ".join(invalid))


def _reject_unowned_closures(material: ContactMaterial) -> None:
    if material.band_gap_narrowing_model != _BAND_GAP_NARROWING_OFF:
        raise ContactClosureCapabilityError(
            "band-gap narrowing at a contact is owned by the parameter-model "
            "slice (P04-11) and is not available here"
        )
    if material.bulk_trap_distribution or material.explicit_defects:
        raise ContactClosureCapabilityError(
            "trap- and defect-charged contact closures are owned by the "
            "dynamic-defect slice (P04-10) and are not available here"
        )


def semiconductor_contact_state(material: ContactMaterial) -> SemiconductorContactState:
    """One statistics/ionization closure for densities and work function.

    The reservoir densities and the work function come from the same common
    Fermi level, so an FD Poisson boundary cannot be paired with MB
    reservoirs constructed independently.
    """
    _require_positive_band_inputs(
        material, "semiconductor contact state requires finite positive ")
    temperature = float(material.temperature_K)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("contact temperature must be finite and positive")
    _reject_unowned_closures(material)
    # BGN off: the effective edges are the physical ones (+0.0/-0.0 as before).
    affinity = float(material.electron_affinity_eV) + 0.0
    band_gap = float(material.band_gap_eV) - 0.0
    neutrality = solve_charge_neutrality(
        temperature_K=temperature,
        band_gap_eV=band_gap,
        effective_conduction_dos_m3=float(material.conduction_dos_m3),
        effective_valence_dos_m3=float(material.valence_dos_m3),
        acceptor_density_m3=float(material.acceptor_density_m3),
        donor_density_m3=float(material.donor_density_m3),
        statistics=material.carrier_statistics,
        dopant_ionization_model=material.dopant_ionization_model,
        donor_binding_energy_eV=material.donor_binding_energy_eV,
        acceptor_binding_energy_eV=material.acceptor_binding_energy_eV,
        donor_degeneracy=float(material.donor_degeneracy),
        acceptor_degeneracy=float(material.acceptor_degeneracy),
    )
    work_function = affinity - (
        neutrality.thermal_voltage_V * neutrality.reduced_electron_fermi_level
    )
    if not np.isfinite(work_function):
        raise FloatingPointError("semiconductor work function is non-finite")
    return SemiconductorContactState(
        statistics=neutrality.statistics,
        electron_affinity_eV=affinity,
        band_gap_eV=band_gap,
        work_function_eV=float(work_function),
        neutrality=neutrality,
    )


def semiconductor_work_function(material: ContactMaterial) -> float:
    """Semiconductor work function below vacuum [eV] at the contact face.

    MB statistics with fully ionized dopants use the historical closed form;
    every other qualified law goes through :func:`semiconductor_contact_state`.
    """
    if material.requires_statistics_closure():
        return semiconductor_contact_state(material).work_function_eV
    _require_positive_band_inputs(
        material, "semiconductor_work_function requires positive contact-layer ")
    if not all(
        math.isfinite(float(value)) and float(value) >= 0.0
        for value in (material.acceptor_density_m3, material.donor_density_m3)
    ):
        raise ValueError(
            "semiconductor_work_function requires finite, non-negative contact "
            "doping densities"
        )
    chi = material.electron_affinity_eV
    V_T_contact = thermal_voltage(material.temperature_K)
    Eg = float(material.band_gap_eV)
    Nc = float(material.conduction_dos_m3)
    Nv = float(material.valence_dos_m3)
    net_doping = float(material.donor_density_m3 - material.acceptor_density_m3)

    if net_doping == 0.0:
        band_distance = 0.5 * Eg + 0.5 * V_T_contact * math.log(Nc / Nv)
        return float(chi + band_distance)

    log_ni = 0.5 * (math.log(Nc) + math.log(Nv)) - Eg / (2.0 * V_T_contact)
    ni = math.exp(log_ni) if log_ni > -745.0 else 0.0
    half_net = 0.5 * net_doping
    disc = math.hypot(half_net, ni)
    if net_doping > 0.0:
        n_majority = half_net + disc
        return float(chi + V_T_contact * math.log(Nc / n_majority))
    p_majority = -half_net + disc
    return float(chi + Eg - V_T_contact * math.log(Nv / p_majority))


def doping_equilibrium_densities(donor_density_m3: float, acceptor_density_m3: float,
                                 intrinsic_density_squared_m6: float) -> tuple[float, float]:
    """Historical ohmic reservoir ``(n, p)`` from the contact node's doping.

    ``n_i`` is ``sqrt(n_i^2)`` of the node's own (temperature- and
    grading-resolved) mass-action product.
    """
    intrinsic = np.sqrt(float(intrinsic_density_squared_m6))
    return mass_action_carrier_densities(donor_density_m3, acceptor_density_m3, float(intrinsic))


def schottky_equilibrium_n(N_c: float, phi_B: float, V_T: float) -> float:
    """Equilibrium electron density behind a Schottky barrier, ``N_c exp(-phi_B/V_T)``."""
    return float(N_c) * float(np.exp(-float(phi_B) / float(V_T)))


def schottky_equilibrium_p(N_v: float, phi_B: float, V_T: float) -> float:
    """Equilibrium hole density behind a Schottky barrier, ``N_v exp(-phi_B/V_T)``."""
    return float(N_v) * float(np.exp(-float(phi_B) / float(V_T)))


def metal_reservoir_floor(
    electron_density_m3: float,
    hole_density_m3: float,
    *,
    intrinsic_density_squared_m6: float,
    donor_density_m3: float,
    acceptor_density_m3: float,
    conduction_dos_m3: float | None,
    valence_dos_m3: float | None,
    barrier_eV: float,
    thermal_voltage_V: float,
) -> tuple[float, float]:
    """Raise the majority reservoir to the metal density ``N exp(-phi_B/V_T)``.

    The doping sign picks the carrier (``N_D >= N_A`` electrons, else holes);
    the floor is dormant (inputs returned unchanged) when the doping
    equilibrium already exceeds it or the layer lacks that DOS. ``n p = n_i^2``
    is kept.
    """
    gate = math.exp(-float(barrier_eV) / thermal_voltage_V)
    if donor_density_m3 >= acceptor_density_m3:
        if conduction_dos_m3:
            floor = float(conduction_dos_m3) * gate
            if floor > electron_density_m3:
                return floor, intrinsic_density_squared_m6 / floor
    else:
        if valence_dos_m3:
            floor = float(valence_dos_m3) * gate
            if floor > hole_density_m3:
                return intrinsic_density_squared_m6 / floor, floor
    return electron_density_m3, hole_density_m3
