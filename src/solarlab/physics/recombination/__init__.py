"""Bulk recombination constitutive laws (P04-07, values, tangents and DC).

Maxwell-Boltzmann lifetime SRH, radiative and Auger recombination; the DEF-1
explicit neutral SRH inventory and its node ownership; position-dependent
trap profiles; and the energy-resolved bulk-trap closure. Interface SRH
belongs to the interface-transport slice and dynamic, multivalent and
metastable defect closures to the defect slice; neither is provided here.
"""

from solarlab.physics.recombination.bulk import (
    RecombinationRate,
    auger_recombination,
    lifetime_srh_denominator,
    lifetime_srh_from_denominator,
    lifetime_srh_recombination,
    radiative_recombination,
    sum_recombination_rates,
)
from solarlab.physics.recombination.energy_resolved_traps import (
    BulkTrapDistribution,
    BulkTrapQuadrature,
    BulkTrapState,
    TrapAwareNeutrality,
    build_bulk_trap_quadrature,
    bulk_trap_distribution_from_mapping,
    evaluate_bulk_trap_state,
    solve_bulk_trap_charge_neutrality,
)
from solarlab.physics.recombination.explicit_neutral import (
    NeutralDefectSpecies,
    NeutralExplicitDefectEvaluation,
    NeutralExplicitDefectModel,
    compile_neutral_defect_species,
    evaluate_neutral_explicit_defects,
)
from solarlab.physics.recombination.reaction import (
    BulkRecombinationCoefficients,
    BulkRecombinationEvaluation,
    BulkRecombinationReaction,
    CarrierSources,
    RecombinationDomainError,
)
from solarlab.physics.recombination.trap_profile import (
    TrapProfile,
    TrapProfileBinding,
    exponential_edge_profile,
    gaussian_edge_profile,
    tau_from_trap_density,
)

__all__ = [
    "BulkRecombinationCoefficients",
    "BulkRecombinationEvaluation",
    "BulkRecombinationReaction",
    "BulkTrapDistribution",
    "BulkTrapQuadrature",
    "BulkTrapState",
    "CarrierSources",
    "NeutralDefectSpecies",
    "NeutralExplicitDefectEvaluation",
    "NeutralExplicitDefectModel",
    "RecombinationDomainError",
    "RecombinationRate",
    "TrapAwareNeutrality",
    "TrapProfile",
    "TrapProfileBinding",
    "auger_recombination",
    "build_bulk_trap_quadrature",
    "bulk_trap_distribution_from_mapping",
    "compile_neutral_defect_species",
    "evaluate_bulk_trap_state",
    "evaluate_neutral_explicit_defects",
    "exponential_edge_profile",
    "gaussian_edge_profile",
    "lifetime_srh_denominator",
    "lifetime_srh_from_denominator",
    "lifetime_srh_recombination",
    "radiative_recombination",
    "solve_bulk_trap_charge_neutrality",
    "sum_recombination_rates",
    "tau_from_trap_density",
]
