"""Carrier statistics: one MB/restricted-FD definition, dopants, neutrality, bands.

Every name exported here is M class (bit-identical to the legacy engine).
"""

from solarlab.physics.statistics.band_potentials import (
    ABSORBER_ROLE, BandStructure, LayerBandInput, dos_fold_reference_layer,
    fold_dos_into_transport_bands, layered_band_structure,
)
from solarlab.physics.statistics.carriers import (
    CARRIER_STATISTICS, FERMI_DIRAC, MAXWELL_BOLTZMANN, CarrierStatistics,
    boltzmann_activity, carrier_density_derivative_reduced_fermi_level,
    carrier_density_from_reduced_fermi_level, carrier_logarithmic_compressibility,
    carrier_occupation, generalized_einstein_factor, normalize_carrier_statistics,
    reduced_fermi_level_from_density,
)
from solarlab.physics.statistics.dopants import (
    DISCRETE_LEVEL, DOPANT_IONIZATION_MODELS, FULLY_IONIZED, DopantChargeState,
    DopantIonizationModel, dopant_charge_state, ionized_acceptor_fraction,
    ionized_donor_fraction, normalize_dopant_ionization_model,
)
from solarlab.physics.statistics.fermi_dirac import (
    fermi_dirac_half, fermi_dirac_half_log_derivative, fermi_dirac_half_table,
    inverse_fermi_dirac_half,
)
from solarlab.physics.statistics.neutrality import (
    CHARGE_NEUTRALITY_RELATIVE_TOLERANCE, BulkChargeNeutralityState,
    mass_action_carrier_densities, solve_charge_neutrality,
    solve_discrete_level_charge_neutrality, solve_fully_ionized_charge_neutrality,
)
from solarlab.physics.statistics.thermal import (
    BOLTZMANN_CONSTANT_J_K, REFERENCE_TEMPERATURE_K, REFERENCE_THERMAL_VOLTAGE_V,
    thermal_voltage,
)

__all__ = [
    "ABSORBER_ROLE", "BOLTZMANN_CONSTANT_J_K", "BandStructure", "BulkChargeNeutralityState",
    "CARRIER_STATISTICS", "CHARGE_NEUTRALITY_RELATIVE_TOLERANCE", "CarrierStatistics",
    "DISCRETE_LEVEL", "DOPANT_IONIZATION_MODELS", "DopantChargeState",
    "DopantIonizationModel", "FERMI_DIRAC", "FULLY_IONIZED", "LayerBandInput",
    "MAXWELL_BOLTZMANN", "REFERENCE_TEMPERATURE_K", "REFERENCE_THERMAL_VOLTAGE_V",
    "boltzmann_activity", "carrier_density_derivative_reduced_fermi_level",
    "carrier_density_from_reduced_fermi_level", "carrier_logarithmic_compressibility",
    "carrier_occupation", "dopant_charge_state", "dos_fold_reference_layer",
    "fermi_dirac_half", "fermi_dirac_half_log_derivative", "fermi_dirac_half_table",
    "fold_dos_into_transport_bands", "generalized_einstein_factor",
    "inverse_fermi_dirac_half", "ionized_acceptor_fraction", "ionized_donor_fraction",
    "layered_band_structure", "mass_action_carrier_densities",
    "normalize_carrier_statistics", "normalize_dopant_ionization_model",
    "reduced_fermi_level_from_density", "solve_charge_neutrality",
    "solve_discrete_level_charge_neutrality", "solve_fully_ionized_charge_neutrality",
    "thermal_voltage",
]
