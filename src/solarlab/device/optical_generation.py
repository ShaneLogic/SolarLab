"""Bind explicit coherent spectral optics to the physical 1D control volumes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from solarlab.device.resolved import PreparedDevice
from solarlab.equations.dc_1d import CapabilityError
from solarlab.physics.generation.optical_binding import (
    OpticalLayerSpec, OpticalStackBinding, PhotonBudget, constant_index_optical_constants,
)
from solarlab.physics.generation.profile import ControlVolumes, GenerationProfile
from solarlab.physics.generation.spectrum import SpectralPhotonFlux, sample_nk_table

OPTICAL_BINDING_1D_V1 = "prepared_coherent_spectral_tmm_1d_v1"
PHOTON_BUDGET_RTOL = 1e-12


@dataclass(frozen=True)
class OpticalGeneration1D:
    binding: OpticalStackBinding
    profile: GenerationProfile
    budget: PhotonBudget


def bind_optical_generation_1d(
    prepared: PreparedDevice, control_volumes: ControlVolumes,
) -> OpticalGeneration1D:
    """Use the complete supplied spectrum; no inferred illumination or scaling.

    The first binding is coherent, normal/front incidence between unit-index
    media. Each layer supplies a named n-k resource or explicit constant n and
    alpha. Optical-only front layers retain absorption but generate no carriers.
    Resource wavelengths are stored in metres; interpolation retains the
    existing kernel's nanometre axis and metre wavelength quadrature.
    """
    data = prepared.to_mapping()
    name = data["declaration"].get("spectrum")
    if name is None:
        raise CapabilityError("optical_generation_1d requires an explicit spectrum resource")
    if data["declaration"].get("fixed_generation") is not None:
        raise CapabilityError("optical_generation_1d: spectrum and fixed_generation are mutually exclusive")
    if data["settings"]["graded_optics"]:
        raise CapabilityError("optical_generation_1d: graded optics is not bound")
    table = prepared.resources.get(name, "spectrum")
    if table.units != ("m", "m^-3/s"):
        raise CapabilityError("optical_generation_1d requires spectral photon flux per metre, not irradiance")
    wavelengths_nm = table.array[:, 0] * 1e9
    spectrum = SpectralPhotonFlux(wavelengths_nm, table.array[:, 1], table.source.id)
    dependencies = {
        "binding_convention": OPTICAL_BINDING_1D_V1,
        "prepared_device_sha256": prepared.content_sha256,
        "spectrum_resource_sha256": table.content_sha256,
        "temperature_K": str(data["settings"]["T"]),
        "monochromatic_Phi_policy": "not_used_absolute_spectrum",
    }
    specs = []
    for layer in prepared.layers:
        parameters = dict(layer.parameters)
        if parameters["incoherent"]:
            raise CapabilityError(f"optical_generation_1d: {layer.id} is incoherent; this slice is coherent")
        if layer.cigs_graded_optics_json is not None:
            raise CapabilityError(f"optical_generation_1d: {layer.id} has unsupported graded optics")
        optical = layer.optical
        if optical is not None:
            if parameters["n_optical"] is not None:
                raise CapabilityError(f"optical_generation_1d: {layer.id} must select a table or constant index, not both")
            native = optical.array
            try:
                n, k = sample_nk_table(native[:, 0] * 1e9, native[:, 1], native[:, 2],
                                       wavelengths_nm, material=optical.name)
            except ValueError as error:
                raise CapabilityError(f"optical_generation_1d: {layer.id}: {error}") from error
            dependencies["nk_resource:" + layer.id] = optical.content_sha256
        else:
            if parameters["n_optical"] is None:
                raise CapabilityError(f"optical_generation_1d: {layer.id} needs explicit n_optical or an n-k resource")
            n, k = constant_index_optical_constants(
                wavelengths_nm, absorption_coefficient_m1=cast(float, parameters["alpha"]),
                refractive_index=cast(float, parameters["n_optical"]),
            )
        specs.append(OpticalLayerSpec(layer.id, layer.thickness, n, k,
                                     incoherent=False, electrical=layer.role != "substrate"))
    binding = OpticalStackBinding(tuple(specs), spectrum, ambient_index=1.0, exit_index=1.0,
                                  incidence="front", dependencies=tuple(dependencies.items()))
    try:
        profile = binding.generation_profile(control_volumes)
        budget = binding.photon_budget()
    except ValueError as error:
        raise CapabilityError("optical_generation_1d: " + str(error)) from error
    for value in (profile.source_m3_s, profile.absorbed_m2_s, budget.reflectance,
                  budget.transmittance, budget.layer_absorptance, budget.dropped_multipass):
        if not np.all(np.isfinite(value)) or np.any(value < 0.0):
            raise CapabilityError("optical_generation_1d: generation and photon fractions must be finite and nonnegative")
    if np.max(np.abs(budget.spectral_closure_residual())) > PHOTON_BUDGET_RTOL:
        raise CapabilityError("optical_generation_1d: spectral photon accounting does not close")
    tolerance = PHOTON_BUDGET_RTOL * budget.incident_m2_s
    weighted_source = float(np.sum(profile.source_m3_s * control_volumes.cell_measures_m))
    if (abs(profile.areal_generation_m2_s - budget.electrical_absorbed_m2_s) > tolerance
            or abs(weighted_source - budget.electrical_absorbed_m2_s) > tolerance):
        raise CapabilityError("optical_generation_1d: physical control volumes do not close the electrical photon budget")
    return OpticalGeneration1D(binding, profile, budget)
