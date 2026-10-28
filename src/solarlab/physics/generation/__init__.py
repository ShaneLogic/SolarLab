"""Optical generation constitutive layer (P04-07, values and DC).

Monochromatic Beer-Lambert, coherent transfer-matrix optics with an optional
incoherent front substrate, explicitly supplied fixed generation arrays and
tabulated spectra/optical constants. Generation is integrated per control
volume (exact absorbed photons per cell) and divided by the caller's cell
measures; optical bindings carry a content identity so a profile is never
reused after its inputs change. Spectral Beer-Lambert is not provided (P08-04).
"""

from solarlab.physics.generation.beer_lambert import (
    BeerLambertBinding,
    beer_lambert_absorbed_photon_flux,
    beer_lambert_generation,
    beer_lambert_optical_depth,
    face_optical_depth,
)
from solarlab.physics.generation.fixed import (
    BASELINE_GENERATION_REFERENCE,
    FixedGenerationProfile,
    fixed_generation_reference,
)
from solarlab.physics.generation.optical_binding import (
    INCIDENCE_SIDES,
    OpticalLayerSpec,
    OpticalStackBinding,
    PhotonBudget,
    constant_index_optical_constants,
    graded_slice_centres,
)
from solarlab.physics.generation.profile import (
    GENERATION_SOURCE_UNIT,
    ControlVolumes,
    GenerationProfile,
    StaleGenerationProfileError,
    control_volume_digest,
)
from solarlab.physics.generation.spectrum import (
    SPECTRAL_PHOTON_FLUX_UNIT,
    SpectralPhotonFlux,
    sample_nk_table,
    sample_tabulated,
    wavelength_grid_m,
)
from solarlab.physics.generation.transfer_matrix import (
    TMMLayer,
    dropped_multipass_fraction,
    tmm_absorbed_photon_flux_per_cell,
    tmm_absorption_profile,
    tmm_cumulative_absorptance,
    tmm_point_generation,
    tmm_reflectance,
    tmm_transmittance,
)

__all__ = [
    "BASELINE_GENERATION_REFERENCE",
    "BeerLambertBinding",
    "ControlVolumes",
    "FixedGenerationProfile",
    "GENERATION_SOURCE_UNIT",
    "GenerationProfile",
    "INCIDENCE_SIDES",
    "OpticalLayerSpec",
    "OpticalStackBinding",
    "PhotonBudget",
    "SPECTRAL_PHOTON_FLUX_UNIT",
    "SpectralPhotonFlux",
    "StaleGenerationProfileError",
    "TMMLayer",
    "beer_lambert_absorbed_photon_flux",
    "beer_lambert_generation",
    "beer_lambert_optical_depth",
    "constant_index_optical_constants",
    "control_volume_digest",
    "dropped_multipass_fraction",
    "face_optical_depth",
    "fixed_generation_reference",
    "graded_slice_centres",
    "sample_nk_table",
    "sample_tabulated",
    "tmm_absorbed_photon_flux_per_cell",
    "tmm_absorption_profile",
    "tmm_cumulative_absorptance",
    "tmm_point_generation",
    "tmm_reflectance",
    "tmm_transmittance",
    "wavelength_grid_m",
]
