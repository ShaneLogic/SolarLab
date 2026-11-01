"""Additive two-domain compiler; the accepted one-domain API is unchanged."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
import hashlib
import json

from solarlab.config.resolve_device import resolve_device
from solarlab.device.compile_ionic_1d import ION_INPUTS, compile_ionic_device_1d
from solarlab.device.inputs import DeviceInput
from solarlab.device.resolved import PreparedDevice
from solarlab.equations.dc_1d import CapabilityError
from solarlab.equations.ionic_1d import CompiledIonicDevice1D
from solarlab.equations.multiple_ionic_1d import CompiledMultipleIonicDevice1D

__all__ = ["CompiledMultipleIonicDevice1D", "compile_multiple_ionic_device_1d"]


def compile_multiple_ionic_device_1d(
    prepared: PreparedDevice, *, intervals_per_layer: int | Sequence[int] = 24,
    ion_domains: Sequence[tuple[str, tuple[str, str]]],
) -> CompiledMultipleIonicDevice1D:
    """Exactly two layer-wide blocking domains separated by inactive material.

    Isolated ionic components are explicit projections for reusing the original
    constitutive/coordinate binding. The returned identity binds the complete
    original document, both projections and every original population.
    """
    if not isinstance(prepared, PreparedDevice):
        raise TypeError("an actual PreparedDevice is required")
    if isinstance(intervals_per_layer, bool) or (
        not isinstance(intervals_per_layer, int)
        and (not isinstance(intervals_per_layer, Sequence)
             or any(type(value) is not int for value in intervals_per_layer))
    ):
        raise ValueError("intervals_per_layer must be an integer or a sequence of integers, not bool")
    if not isinstance(ion_domains, Sequence) or isinstance(ion_domains, (str, bytes)):
        raise CapabilityError("ion_domains must contain exactly two explicit domain declarations")
    declarations = tuple(ion_domains)
    if (len(declarations) != 2
            or any(not isinstance(item, tuple) or len(item) != 2
                   or not isinstance(item[0], str) or item[1] != ("blocking", "blocking")
                   for item in declarations)):
        raise CapabilityError("two explicit (layer, ('blocking', 'blocking')) declarations are required; exchange is unsupported")
    names = tuple(item[0] for item in declarations)
    layers = tuple(layer for layer in prepared.layers if layer.role != "substrate")
    layer_ids = tuple(layer.id for layer in layers)
    if len(set(names)) != 2 or any(name not in layer_ids for name in names):
        raise CapabilityError("domains must name two distinct electrical layers")
    left, right = (layer_ids.index(name) for name in names)
    if right - left < 2:
        raise CapabilityError("domains must be left-to-right and separated by a configured inactive layer; touching domains are unsupported")
    originals = [{"layer": layer.id, **{key: dict(layer.parameters)[key] for key in ION_INPUTS}}
                 for layer in prepared.layers]
    for entry in originals:
        if entry["layer"] not in names and any(entry[key] != 0.0 for key in ("P0", "P0_neg", "D_ion", "D_ion_neg")):
            raise CapabilityError("all undeclared layers must have configured zero ionic populations and mobility")
    components: list[CompiledIonicDevice1D] = []
    records = []
    for name, endpoints in declarations:
        editing = prepared.to_input().editing_data()
        changes = []
        for layer in editing["layers"]:
            if layer["id"] != name:
                original = next(entry for entry in originals if entry["layer"] == layer["id"])
                for key in ("P0", "P0_neg", "D_ion", "D_ion_neg"):
                    layer.setdefault("parameters", {})[key] = 0.0
                    changes.append({"path": "layers." + layer["id"] + "." + key,
                                    "original_value": original[key], "component_value": 0.0})
        isolated = resolve_device(DeviceInput.model_validate(editing), prepared.defaults, prepared.resources,
                                  sources=prepared.sources)
        component = compile_ionic_device_1d(isolated, intervals_per_layer=intervals_per_layer,
                                           ion_domain_layer=name, ion_domain_endpoints=endpoints)
        if components:
            if component.electronic_component.identity != components[0].electronic_component.identity:
                raise ValueError("isolated domains produced different electronic components")
            component = replace(component, electronic_component=components[0].electronic_component)
        components.append(component)
        records.append({"layer": name, "endpoints": list(endpoints), "source": "compile_arguments",
            "isolated_prepared_sha256": isolated.content_sha256, "component_projection": changes,
            "component_sha256": component.identity, "component_manifest": component.binding_manifest,
            "reference_inventories_m2": {species: component.reference_inventory_m2 if species == "positive"
                                       else component.negative_reference_inventory_m2
                                       for species in component.ion_species_names}})
    size = 3 * components[0].node_count
    layout = []
    for component in components:
        for species in component.ion_species_names:
            layout.append({"domain": component.domain_layer_id, "species": species,
                           "start": size, "stop": size + component.ion_node_count,
                           "coordinate": "original_finite_site_coordinate"})
            size += component.ion_node_count
    settings = prepared.to_mapping()["settings"]
    manifest = {
        "binding": "two_separated_blocking_ion_domains_classical_1d_v1",
        "original_prepared_sha256": prepared.content_sha256, "original_ionic_inputs": originals,
        "original_ionic_settings": {key: settings[key] for key in ("ion_steric_diffusion_only", "ion_steric_shared_site")},
        "electronic_component_sha256": components[0].electronic_component.identity,
        "domains": records, "state_size": size, "ionic_state_blocks": layout,
        "outside_domain_policy": "configured_zero_population_and_mobility",
        "between_domain_policy": "configured_inactive_layer_and_declared_blocking_endpoints",
        "charge_deposition": "sum_domain q*(WP/We)*[(P_positive-P0)-(P_negative-P0_neg)]",
        "storage": "original_physical_populations_per_domain; all_local_rows_retained",
        "scope": "residual/J/storage assembly only; no steady/time/AC driver or scientific qualification",
    }
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return CompiledMultipleIonicDevice1D(prepared.to_mapping()["id"], hashlib.sha256(encoded.encode()).hexdigest(),
        prepared.content_sha256, components[0].electronic_component, tuple(components), encoded)
