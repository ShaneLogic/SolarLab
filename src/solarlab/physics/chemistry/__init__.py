"""Constitutive iodine reaction-diffusion physics kernels (SI units)."""

from __future__ import annotations

from solarlab.physics.chemistry.iodine_kinetics import (
    IODINE_KINETICS_SCHEMA,
    IodineKineticsParameters,
    IodineReactionEvaluation,
    IodineThermodynamicMode,
    ReactionEOrder,
    equilibrium_iodine_populations_m3,
    evaluate_iodine_kinetics,
    smoluchowski_rate_m3_s,
)
from solarlab.physics.chemistry.iodine_transport import (
    BoundedSpeciesMap,
    IODINE_TRANSPORT_SCHEMA,
    PositiveSpeciesMap,
    SoftplusSpeciesMap,
    SpeciesCoordinateMap,
    SpeciesFaceFluxEvaluation,
    harmonic_face_diffusivity_m2_s,
    species_sg_face_flux,
)
from solarlab.physics.chemistry.stoichiometry import (
    CANONICAL_IODINE_NETWORK,
    IODINE_REACTIONS,
    IODINE_SPECIES,
    STOICHIOMETRY_SCHEMA,
    StoichiometricNetwork,
    canonical_iodine_network,
)

__all__ = [
    "BoundedSpeciesMap",
    "CANONICAL_IODINE_NETWORK",
    "IODINE_KINETICS_SCHEMA",
    "IODINE_REACTIONS",
    "IODINE_SPECIES",
    "IODINE_TRANSPORT_SCHEMA",
    "IodineKineticsParameters",
    "IodineReactionEvaluation",
    "IodineThermodynamicMode",
    "PositiveSpeciesMap",
    "ReactionEOrder",
    "STOICHIOMETRY_SCHEMA",
    "SoftplusSpeciesMap",
    "SpeciesCoordinateMap",
    "SpeciesFaceFluxEvaluation",
    "StoichiometricNetwork",
    "canonical_iodine_network",
    "equilibrium_iodine_populations_m3",
    "evaluate_iodine_kinetics",
    "harmonic_face_diffusivity_m2_s",
    "smoluchowski_rate_m3_s",
    "species_sg_face_flux",
]
