"""Compile the bounded single-positive-ion residual from real PreparedDevice inputs.

The ion-free compilation below is an explicit electronic *component* projection,
not a returned ionic device or an ion-stripping fallback. All original ionic
inputs and their domain/inventory remain bound in the final manifest/identity.
"""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
import math
from typing import cast

import numpy as np

from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_1d import compile_device_1d
from solarlab.device.inputs import DeviceInput
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.dc_1d import CapabilityError
from solarlab.equations.ionic_1d import CompiledIonicDevice1D
from solarlab.mesh.geometry import half_endpoint_geometry
from solarlab.physics.band_structure.temperature import ion_diffusivity_at_temperature
from solarlab.physics.ions import (
    DIFFUSION_ONLY_STERIC_V1, HISTORICAL_OCCUPANCY_CLIP,
    HISTORICAL_STERIC_FLOOR, WHOLE_FLUX_STERIC_V1,
)

__all__ = ["CompiledIonicDevice1D", "compile_ionic_device_1d"]
_ION_INPUTS = ("P0", "P0_neg", "D_ion", "D_ion_neg", "P_lim", "P_lim_neg", "E_a_ion")


def compile_ionic_device_1d(
    prepared: PreparedDevice, *, intervals_per_layer: int | Sequence[int] = 24,
) -> CompiledIonicDevice1D:
    """Compile one layer-wide positive population with finite sites and blocking contacts.

    Zero D retains the configured ion population/background/storage. Negative
    populations, shared sites, electrical interfaces and absent/saturated
    positive references explicitly require a subsequent reviewed binding.
    """
    if not isinstance(prepared, PreparedDevice):
        raise TypeError("compile_ionic_device_1d requires an existing PreparedDevice")
    data = prepared.to_mapping()
    layers = tuple(layer for layer in prepared.layers if layer.role != "substrate")
    if len(layers) != 1 or any(interface.electrical for interface in prepared.interfaces):
        raise CapabilityError("first_ionic_1d: one electrical layer without ion interfaces is supported")
    if data["settings"]["ion_steric_shared_site"]:
        raise CapabilityError("first_ionic_1d: shared-site ions are not assembled")
    declarations = [{"layer": layer.id, **{name: dict(layer.parameters)[name] for name in _ION_INPUTS}}
                    for layer in prepared.layers]
    for layer in prepared.layers:
        parameters = dict(layer.parameters)
        if parameters["P0_neg"] != 0.0 or parameters["D_ion_neg"] != 0.0:
            raise CapabilityError("first_ionic_1d: negative/dual ions are not assembled")
        if layer.role == "substrate" and any(parameters[name] != 0.0 for name in ("P0", "D_ion")):
            raise CapabilityError("first_ionic_1d: substrate ion domains are not assembled")
    parameters = dict(layers[0].parameters)
    reference, capacity = float(cast(float, parameters["P0"])), float(cast(float, parameters["P_lim"]))
    diffusion, activation = float(cast(float, parameters["D_ion"])), float(cast(float, parameters["E_a_ion"]))
    if not all(math.isfinite(value) for value in (reference, capacity, diffusion, activation)):
        raise CapabilityError("first_ionic_1d: finite ionic parameters are required")
    if not 0.0 < reference < capacity:
        raise CapabilityError("first_ionic_1d: a strictly positive, unsaturated finite-site reference is required")
    if diffusion < 0.0 or activation < 0.0:
        raise CapabilityError("first_ionic_1d: diffusivity/activation energy must be non-negative")

    # Explicit component projection; only ionic populations/transport switches
    # are excluded from this component, and every change is recorded below.
    projection = prepared.to_input().editing_data()
    changes: list[dict[str, object]] = []
    for layer in projection["layers"]:
        for name in ("P0", "P0_neg", "D_ion", "D_ion_neg"):
            layer.setdefault("parameters", {})[name] = 0.0
            changes.append({"path": "layers." + layer["id"] + "." + name, "component_value": 0.0})
    for name in ("ion_steric_diffusion_only", "ion_steric_shared_site"):
        projection.setdefault("settings", {})[name] = False
        changes.append({"path": "settings." + name, "component_value": False})
    component_prepared = resolve_device(
        DeviceInput.model_validate(projection), prepared.defaults, prepared.resources, sources=prepared.sources,
    )
    component = compile_device_1d(component_prepared, intervals_per_layer=intervals_per_layer)
    ion_geometry = half_endpoint_geometry(component.nodes_m)
    count = component.node_count
    scaled_diffusion = ion_diffusivity_at_temperature(diffusion, component.parameters.temperature_K, activation)
    if not math.isfinite(scaled_diffusion) or scaled_diffusion < 0.0:
        raise CapabilityError("first_ionic_1d: temperature-scaled ion diffusivity is outside the finite envelope")
    occupancy = reference / capacity
    logit = math.log(occupancy) - math.log1p(-occupancy)
    steric_model = DIFFUSION_ONLY_STERIC_V1 if data["settings"]["ion_steric_diffusion_only"] else WHOLE_FLUX_STERIC_V1
    reference_array, limit_array = np.full(count, reference), np.full(count, capacity)
    # One uniform ionic domain: retain the ion kernel's scalar-broadcast law,
    # including exact zero diffusivity, rather than a carrier-only harmonic mean.
    faces = np.full(count - 1, scaled_diffusion)
    manifest = {
        "binding": "single_positive_ion_classical_1d_residual_v1",
        "original_prepared_sha256": prepared.content_sha256,
        "original_ionic_inputs": declarations,
        "original_ionic_settings": {name: data["settings"][name] for name in ("ion_steric_diffusion_only", "ion_steric_shared_site")},
        "electronic_component": {"prepared_sha256": component_prepared.content_sha256, "compiled_sha256": component.identity,
                                 "projection": changes, "geometry": component.geometry.name},
        "ion_domain": {"layer": layers[0].id, "bounds_m": [float(component.nodes_m[0]), float(component.nodes_m[-1])],
                       "nodes_m": component.nodes_m.tolist(), "cell_measures_m": ion_geometry.cell_measures_m.tolist(),
                       "geometry": ion_geometry.name, "contacts": ["blocking", "blocking"],
                       "reference_inventory_m2": math.fsum((ion_geometry.cell_measures_m * reference_array).tolist())},
        "ion_model": {"steric_model": steric_model, "occupancy_clip": HISTORICAL_OCCUPANCY_CLIP,
                      "steric_floor": HISTORICAL_STERIC_FLOOR, "activation_energy_eV": activation,
                      "temperature_K": component.parameters.temperature_K, "face_diffusivity_m2_s": faces.tolist()},
        "ion_charge_deposition": {"target": "electronic_component_cell_volumes",
                                  "physical_density_ratio": (ion_geometry.cell_measures_m / component.geometry.cell_measures_m).tolist(),
                                  "rule": "P_charge=(W_P/W_e)*P; identical projection for fixed P0 background"},
        "scope": "assembled residual and analytic derivatives; ionic steady/transient qualification remains open",
    }
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    identity = hashlib.sha256(encoded.encode()).hexdigest()
    return CompiledIonicDevice1D(
        data["id"], identity, prepared.content_sha256, component, ion_geometry, layers[0].id,
        reference_array, limit_array, np.full(count, logit), faces, steric_model, encoded,
    )
