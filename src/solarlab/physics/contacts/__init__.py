"""Outer contacts: Poisson potential sources, reservoirs, exchange channels, thermodynamics.

The namespace exports the M-class (legacy bit-identical) contact physics. The
unified boundary particle flux (N class) is
:mod:`solarlab.physics.contacts.boundary_flux`.
"""

from solarlab.physics.contacts.exchange import (
    BLOCKING, CONTACT_SIDES, DOPING_EQUILIBRIUM, EXCHANGE_KINDS, FIXED_RESERVOIR,
    FLAT_BAND_CONTACT_VELOCITY_M_S, METAL_FLOOR, RESERVOIR_SOURCES, ROBIN,
    SCHOTTKY_BARRIER, STATISTICS_CLOSURE, CarrierExchangeChannel, exchange_kind,
    legacy_selective_contact_current, resolve_exchange_velocities,
)
from solarlab.physics.contacts.potential import (
    BUILT_IN_POTENTIAL_MODES, LEGACY_MANUAL, METAL_WORK_FUNCTION,
    SEMICONDUCTOR_WORK_FUNCTION, BuiltInPotentialSources, ContactPotentialBoundary,
    LegacyBandContact, junction_polarity, legacy_band_contact_potential,
    legacy_band_fermi_level, resolve_built_in_potential_mode,
)
from solarlab.physics.contacts.reservoirs import (
    ContactClosureCapabilityError, ContactMaterial, SemiconductorContactState,
    doping_equilibrium_densities, metal_reservoir_floor, schottky_equilibrium_n,
    schottky_equilibrium_p, semiconductor_contact_state, semiconductor_work_function,
)
from solarlab.physics.contacts.thermodynamics import (
    CONTACT_THERMODYNAMIC_TOLERANCE_EV, LEGACY_MANUAL_CONTACT_OFFSET, ContactEndpoint,
    ContactThermodynamicAssessment, ContactThermodynamicError,
    assess_contact_thermodynamics, contact_quasi_fermi_levels,
    require_contact_thermodynamic_certificate,
)

__all__ = [
    "BLOCKING", "BUILT_IN_POTENTIAL_MODES", "BuiltInPotentialSources",
    "CONTACT_SIDES", "CONTACT_THERMODYNAMIC_TOLERANCE_EV", "CarrierExchangeChannel",
    "ContactClosureCapabilityError", "ContactEndpoint", "ContactMaterial",
    "ContactPotentialBoundary", "ContactThermodynamicAssessment",
    "ContactThermodynamicError", "DOPING_EQUILIBRIUM", "EXCHANGE_KINDS",
    "FIXED_RESERVOIR", "FLAT_BAND_CONTACT_VELOCITY_M_S", "LEGACY_MANUAL",
    "LEGACY_MANUAL_CONTACT_OFFSET", "LegacyBandContact", "METAL_FLOOR",
    "METAL_WORK_FUNCTION", "RESERVOIR_SOURCES", "ROBIN", "SCHOTTKY_BARRIER",
    "SEMICONDUCTOR_WORK_FUNCTION", "STATISTICS_CLOSURE", "SemiconductorContactState",
    "assess_contact_thermodynamics", "contact_quasi_fermi_levels",
    "doping_equilibrium_densities", "exchange_kind", "junction_polarity",
    "legacy_band_contact_potential", "legacy_band_fermi_level",
    "legacy_selective_contact_current", "metal_reservoir_floor",
    "require_contact_thermodynamic_certificate", "resolve_built_in_potential_mode",
    "resolve_exchange_velocities", "schottky_equilibrium_n", "schottky_equilibrium_p",
    "semiconductor_contact_state", "semiconductor_work_function",
]
