"""Immutable resolved declaration data, with no solver or file-loader dependency.

The authoritative constructor inputs are a validated editing document, an
injected data catalog and supplied resources. Derived children cannot be passed
to or replaced on PreparedDevice: editing the input reconstructs them all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any, Mapping

from solarlab.device.defaults import DefaultCatalog
from solarlab.device.inputs import (
    BulkDefectInput, DeviceInput, FullLayerInput, InterfaceInput, SpatialProfileInput,
    TandemInput,
)
from solarlab.device.settings import DeviceSettingsInput
from solarlab.materials.full_parameters import FullParameterInput, LAYER_OWNED_PARAMETERS, full_parameter_items
from solarlab.materials.parameters import Scalar
from solarlab.materials.resources import ResourceLibrary, ResourceTable
from solarlab.materials.source import SourceDocument
from solarlab.units import UNIT_SCHEMA_VERSION, normalize_quantity

__all__ = ["PreparedMaterial", "PreparedLayer", "PreparedDefect", "PreparedInterface", "PreparedDevice", "PreparedTandem"]


def _encode(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _decode(value: bytes) -> Any:
    if type(value) is not bytes or not value or len(value) > 2**23:
        raise ValueError("declaration requires bounded immutable JSON bytes")
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        if len({name for name, _ in items}) != len(items):
            raise ValueError("duplicate JSON declaration key")
        return dict(items)
    return json.loads(value, object_pairs_hook=pairs)


def _readonly(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({name: _readonly(item) for name, item in value.items()})
    if isinstance(value, list):
        return tuple(_readonly(item) for item in value)
    return value


def _identifier(value: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.:-]*", value) is None:
        raise ValueError("invalid stable declaration ID")


@dataclass(frozen=True, slots=True)
class PreparedDefect:
    input_json: bytes
    band_gap_eV: float
    _values_json: bytes = field(init=False, repr=False)

    def __post_init__(self) -> None:
        gap = normalize_quantity(self.band_gap_eV, "eV")
        if gap <= 0:
            raise ValueError("explicit defect requires a positive band gap")
        model = BulkDefectInput.model_validate(_decode(self.input_json))
        data = model.normalized_data()
        distribution = data["distribution"]
        energy = data.get("energy_level")
        if energy is not None:
            value = energy["value_eV"]
            center = float(Decimal(str(gap)) - Decimal(str(value))) if energy["reference"] == "below_conduction_band" else value
            distribution["center_eV_above_vb"] = center
        center = distribution["center_eV_above_vb"]
        if center is None or not 0 <= center <= gap:
            raise ValueError(f"defect {data['id']}: energy lies outside the layer gap")
        width, multiplier = distribution.get("width_eV"), distribution.get("support_width_multiplier")
        support = None
        if distribution["kind"] == "uniform":
            support = (center - 0.5 * width, center + 0.5 * width)
        elif distribution["kind"] == "gaussian" and width is not None and multiplier is not None:
            half_width = 0.5 * multiplier * width
            support = (center - half_width, center + half_width)
        elif distribution["kind"] == "conduction_band_tail":
            support = (center - multiplier * width, center)
        elif distribution["kind"] == "valence_band_tail":
            support = (center, center + multiplier * width)
        if support is not None:
            if not all(math.isfinite(value) for value in support):
                raise ValueError(f"defect {data['id']}: nonfinite distribution support")
            roundoff = 16.0 * math.ulp(max(gap, abs(support[0]), abs(support[1]), 1.0))
            if support[0] < -roundoff or support[1] > gap + roundoff:
                raise ValueError(f"defect {data['id']}: distribution support lies outside the layer gap")
        profile = data.get("spatial_profile")
        if profile is not None:
            SpatialProfileInput.model_validate(profile)
            knots = profile["knots"]
            if len(knots) < 2 or knots[0]["position_fraction"] != 0 or knots[-1]["position_fraction"] != 1:
                raise ValueError("spatial defect knots must span normalized [0,1]")
            segments = []
            for left, right in zip(knots, knots[1:]):
                width = right["position_fraction"] - left["position_fraction"]
                if width <= 0:
                    raise ValueError("spatial defect knots must be strictly increasing")
                segments.append(0.5 * (left["density_multiplier"] + right["density_multiplier"]) * width)
            integral = math.fsum(segments)
            if not math.isclose(integral, 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("spatial defect density must have layer-average unity")
        object.__setattr__(self, "band_gap_eV", gap)
        object.__setattr__(self, "input_json", _encode(model.editing_data()))
        object.__setattr__(self, "_values_json", _encode(data))

    @property
    def values(self) -> Mapping[str, Any]:
        return _readonly(_decode(self._values_json))

    def to_mapping(self) -> dict[str, Any]:
        return _decode(self._values_json)


@dataclass(frozen=True, slots=True)
class PreparedMaterial:
    id: str
    parameters: tuple[tuple[str, Scalar], ...]
    optical: ResourceTable | None

    def __post_init__(self) -> None:
        _identifier(self.id)
        parameters = full_parameter_items(tuple(self.parameters))
        if {name for name, _ in parameters} != set(FullParameterInput.model_fields) - LAYER_OWNED_PARAMETERS:
            raise ValueError("material parameters must exclude layer doping and defect/lifetime fields")
        optical_name = dict(parameters)["optical_material"]
        if optical_name is None:
            if self.optical is not None:
                raise ValueError("unselected optical resource cannot be bound to material")
        elif not isinstance(self.optical, ResourceTable) or self.optical.name != optical_name or self.optical.kind != "nk":
            raise ValueError("material optical name and supplied n-k resource do not match")
        object.__setattr__(self, "parameters", parameters)

    @property
    def values(self) -> Mapping[str, Scalar]:
        return MappingProxyType(dict(self.parameters))

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(_encode((self.id, self.parameters, self.optical.content_sha256 if self.optical else None))).hexdigest()


@dataclass(frozen=True, slots=True)
class PreparedLayer:
    id: str
    name: str
    role: str
    thickness: float
    parameters: tuple[tuple[str, Scalar], ...]
    field_origins: tuple[tuple[str, str], ...]
    defects: tuple[PreparedDefect, ...]
    defect_model: str
    defect_schema_version: str | None
    material_name: str | None
    optical: ResourceTable | None
    material: PreparedMaterial = field(init=False)

    def __post_init__(self) -> None:
        _identifier(self.id)
        if any(not isinstance(value, str) or not value for value in (self.name, self.role)):
            raise ValueError("layer name and role must be nonempty")
        thickness = normalize_quantity(self.thickness, "m")
        if thickness <= 0:
            raise ValueError("layer thickness must be positive")
        parameters = full_parameter_items(tuple(self.parameters), complete=True)
        origins = tuple(self.field_origins)
        if len(origins) != len(parameters) or {name for name, _ in origins} != {name for name, _ in parameters} or any(not isinstance(origin, str) or not origin for _, origin in origins):
            raise ValueError("every resolved parameter requires an explicit origin")
        defects = tuple(self.defects)
        if any(not isinstance(defect, PreparedDefect) for defect in defects) or len({defect.values["id"] for defect in defects}) != len(defects):
            raise ValueError("layer defects require unique validated IDs")
        if self.defect_model not in {"effective_lifetime", "explicit_quasi_steady"}:
            raise ValueError("unsupported declared defect model")
        _validate_defect_version(self.id, self.defect_schema_version, defects, self.defect_model)
        values = dict(parameters)
        if self.defect_model == "explicit_quasi_steady" and (values["carrier_statistics"] != "maxwell_boltzmann" or values["dopant_ionization_model"] != "fully_ionized" or values["band_gap_narrowing_model"] != "off"):
            raise ValueError("unavailable explicit-defect/statistics/ionization/BGN combination")
        if any(defect.band_gap_eV != dict(parameters)["Eg"] for defect in defects):
            raise ValueError("defect energy binding does not match its layer gap")
        object.__setattr__(self, "thickness", thickness)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "field_origins", tuple(sorted(origins)))
        object.__setattr__(self, "defects", defects)
        object.__setattr__(self, "material", PreparedMaterial(self.material_name or self.id + ":material",
                          tuple((name, value) for name, value in parameters if name not in LAYER_OWNED_PARAMETERS), self.optical))

    @property
    def local_parameters(self) -> Mapping[str, Scalar]:
        return MappingProxyType({name: value for name, value in self.parameters if name in LAYER_OWNED_PARAMETERS})

    def to_mapping(self) -> dict[str, Any]:
        values = dict(self.parameters)
        return {"id": self.id, "name": self.name, "role": self.role, "thickness_m": self.thickness,
                "material_id": self.material.id, "material_sha256": self.material.content_sha256,
                "material_parameters": dict(self.material.parameters), "layer_parameters": dict(self.local_parameters),
                "defect_model": self.defect_model, "defect_schema_version": self.defect_schema_version,
                "bulk_defects": [defect.to_mapping() for defect in self.defects],
                "provenance": [{"parameter": name, "effective_value": values[name], "origin": origin} for name, origin in self.field_origins]}


@dataclass(frozen=True, slots=True)
class PreparedInterface:
    input_json: bytes
    adjacent_gaps_eV: tuple[float, float]
    electrical: bool
    _values_json: bytes = field(init=False, repr=False)

    def __post_init__(self) -> None:
        model = InterfaceInput.model_validate(_decode(self.input_json))
        gaps = tuple(normalize_quantity(value, "eV") for value in self.adjacent_gaps_eV)
        if len(gaps) != 2 or any(value < 0 for value in gaps) or type(self.electrical) is not bool:
            raise ValueError("interface requires two adjacent band gaps and explicit ownership")
        data = model.normalized_data()
        if (data.get("v_n") is None) != (data.get("v_p") is None):
            raise ValueError("bare interface velocities must be declared or cleared together")
        defect = data.get("defect")
        if defect is not None:
            if not self.electrical:
                raise ValueError("microscopic electrical defects cannot belong to an optical-only interface")
            gap = min(gaps)
            depth = defect["trap_depth_eV"]
            if not 0 <= depth <= gap:
                raise ValueError("interface defect energy lies outside the adjacent reference gap")
            kinetics = defect["kinetics"]
            capture = tuple(kinetics[f"sigma_{carrier}_m2"] * kinetics[f"thermal_velocity_{carrier}_m_s"] * defect["total_density_m2"] for carrier in ("n", "p"))
            if any(not math.isfinite(value) for value in capture):
                raise ValueError("microscopic interface capture is not finite")
            data["microscopic_capture_velocities_m_s"] = capture
            data["reference_gap_eV"] = gap
        object.__setattr__(self, "adjacent_gaps_eV", gaps)
        object.__setattr__(self, "input_json", _encode(model.editing_data()))
        object.__setattr__(self, "_values_json", _encode(data))

    @property
    def values(self) -> Mapping[str, Any]:
        return _readonly(_decode(self._values_json))

    def to_mapping(self) -> dict[str, Any]:
        return {**_decode(self._values_json), "electrical": self.electrical}


def _scaps_derived(values: dict[str, Any], defects: tuple[PreparedDefect, ...], defaults: DefaultCatalog) -> dict[str, Any]:
    """Data conversion at the supplied reference temperature; no equilibrium solve."""
    constants: dict[str, Any] = dict(defaults.constants)
    voltage = constants["K_B"] * constants["T"] / constants["Q"]
    nc, nv, gap = (values[name] for name in ("Nc300", "Nv300", "Eg"))
    if any(value is None or value <= 0 for value in (nc, nv, gap)):
        raise ValueError("SCAPS material data require positive N_C, N_V and E_g")
    ni = math.sqrt(nc * nv * math.exp(-gap / voltage))
    if not math.isfinite(ni) or ni <= 0:
        raise ValueError("SCAPS DOS conversion is outside finite positive float representation")
    result: dict[str, Any] = {"ni": ni, "n1": ni, "p1": ni}
    if not defects:
        return result
    inverse_n = inverse_p = weighted_n = weighted_p = density = 0.0
    for defect in defects:
        data = defect.values
        kinetics, distribution = data["kinetics"], data["distribution"]
        population = distribution["total_density_m3"]
        energy = data.get("energy_level")
        if energy is None:
            raise ValueError("SCAPS defect conversion requires its referenced energy")
        # Preserve the historical scalar evaluation from the declared energy;
        # the rounded absolute center is separate distribution metadata.
        delta = (gap / 2 - energy["value_eV"] if energy["reference"] == "below_conduction_band"
                 else energy["value_eV"] - gap / 2)
        ratio = math.exp(delta / voltage)
        n1, p1 = ni * ratio, ni / ratio
        rate_n = kinetics["sigma_n_m2"] * kinetics["thermal_velocity_n_m_s"] * population
        rate_p = kinetics["sigma_p_m2"] * kinetics["thermal_velocity_p_m_s"] * population
        inverse_n += rate_n
        inverse_p += rate_p
        weighted_n += n1 * rate_n
        weighted_p += p1 * rate_p
        density += population
    fallback = dict(defaults.scaps)
    result.update(tau_n=1 / inverse_n if inverse_n > 0 else fallback["tau_n"],
                  tau_p=1 / inverse_p if inverse_p > 0 else fallback["tau_p"],
                  n1=weighted_n / inverse_n if inverse_n > 0 else 0.0,
                  p1=weighted_p / inverse_p if inverse_p > 0 else 0.0,
                  trap_N_t_bulk=density if density > 0 else None)
    return result


def _validate_defect_version(layer_id: str, version: str | None, defects: tuple[PreparedDefect, ...], mode: str) -> None:
    if version is None:
        if mode != "effective_lifetime" or defects:
            raise ValueError(f"layers.{layer_id}: microscopic inventory requires a schema version")
        return
    if version not in {"solarlab-explicit-bulk-defects-v1", "solarlab-explicit-bulk-defects-v2", "solarlab-explicit-bulk-defects-v3"}:
        raise ValueError(f"layers.{layer_id}.defect_schema_version: typed preparation is unavailable for this version")
    if mode == "explicit_quasi_steady" and not defects:
        raise ValueError(f"layers.{layer_id}: explicit_quasi_steady requires a nonempty defect inventory")
    has_profile = False
    for defect in defects:
        data = defect.values
        distribution = data["distribution"]
        profile = data.get("spatial_profile")
        has_profile |= profile is not None
        v1_ready = (distribution["kind"] in {"single_level", "gaussian"}
                    and distribution.get("energy_reference") is None
                    and distribution.get("support_width_multiplier") is None and profile is None)
        v2_ready = (distribution.get("energy_reference") == "above_valence_band"
                    and (distribution["kind"] != "gaussian" or
                         (distribution.get("width_convention") != "unresolved"
                          and distribution.get("width_eV") is not None
                          and distribution.get("support_width_multiplier") is not None)))
        named_charge = data.get("name") is not None and data["charge_transition"] != "unresolved"
        where = f"layers.{layer_id}.bulk_defects.{data['id']}"
        if version.endswith("-v1"):
            if not v1_ready:
                raise ValueError(f"{where}: v1 forbids v2 support/reference and spatial fields")
            if mode == "explicit_quasi_steady" and (not named_charge or distribution["kind"] != "single_level"):
                raise ValueError(f"{where}: v1 explicit mode requires named charge-resolved single levels")
        elif version.endswith("-v2"):
            if not v2_ready or profile is not None:
                raise ValueError(f"{where}: v2 requires complete valence-referenced energy support without spatial fields")
            if mode == "explicit_quasi_steady" and not named_charge:
                raise ValueError(f"{where}: v2 explicit mode requires named charge-resolved species")
        elif not v2_ready or not named_charge:
            raise ValueError(f"{where}: v3 requires named charge-resolved species and complete energy support")
    if version.endswith("-v3") and not has_profile:
        raise ValueError(f"layers.{layer_id}: v3 requires at least one normalized spatial profile")


def _resolve_layer(layer: FullLayerInput, material: dict[str, Any], defaults: DefaultCatalog, resources: ResourceLibrary) -> PreparedLayer:
    default_values = dict(defaults.material)
    origins = {name: "catalog:" + defaults.content_sha256 for name in default_values}
    if layer.parameterization == "scaps":
        default_values.update(defaults.scaps)
        origins.update((name, "scaps_catalog:" + defaults.content_sha256) for name, _ in defaults.scaps)
    default_values.update(material)
    origins.update((name, "material:" + str(layer.material)) for name in material)
    explicit = dict(layer.parameters.normalized_items())
    if layer.parameterization == "scaps" and set(explicit) & {"ni", "tau_n", "tau_p", "n1", "p1", "trap_N_t_bulk"}:
        raise ValueError(f"layers.{layer.id}: SCAPS-derived values must be edited through DOS/defect inputs")
    values: dict[str, Any] = {**default_values, **explicit}
    origins.update((name, "layer_input:" + layer.id) for name in explicit)
    defect_data = [item.editing_data() for item in layer.bulk_defects]
    for item in defect_data:
        if item["distribution"]["kind"] == "single_level":
            item["distribution"].setdefault("width_convention", defaults.distribution_width_convention)
    defects = tuple(PreparedDefect(_encode(item), values.get("Eg", 0)) for item in defect_data)
    metadata_ids = [item.defect_id for item in layer.scaps_defect_metadata]
    if len(set(metadata_ids)) != len(metadata_ids) or set(metadata_ids) - {item.values["id"] for item in defects}:
        raise ValueError(f"layers.{layer.id}.scaps_defect_metadata: duplicate or unknown defect ID")
    if metadata_ids and layer.parameterization != "scaps":
        raise ValueError(f"layers.{layer.id}.scaps_defect_metadata: requires SCAPS parameterization")
    if layer.parameterization == "scaps":
        derived = _scaps_derived(values, defects, defaults)
        values.update(derived)
        origins.update((name, "derived:scaps_dos_and_parallel_inventory") for name in derived)
        if values.get("trap_N_t_interface") is not None and values.get("trap_N_t_bulk") is None:
            values["trap_N_t_bulk"] = values["trap_N_t_interface"]
            origins["trap_N_t_bulk"] = "derived:source_interface_profile_fallback"
    mode = layer.defect_model or defaults.defect_model
    if (layer.defect_model is None) != (layer.defect_schema_version is None) or (defects and layer.defect_schema_version is None):
        raise ValueError(f"layers.{layer.id}: microscopic inventory/schema/model is incomplete")
    optical = None if values.get("optical_material") is None else resources.get(values["optical_material"], "nk")
    return PreparedLayer(layer.id, layer.name, layer.role, normalize_quantity(layer.thickness, "m"), tuple(values.items()),
                         tuple(origins.items()), defects, mode, layer.defect_schema_version, layer.material, optical)


@dataclass(frozen=True, slots=True)
class PreparedDevice:
    input_json: bytes
    defaults: DefaultCatalog
    resources: ResourceLibrary
    sources: tuple[SourceDocument, ...] = ()
    layers: tuple[PreparedLayer, ...] = field(init=False)
    interfaces: tuple[PreparedInterface, ...] = field(init=False)
    _resolved_json: bytes = field(init=False, repr=False)
    can_execute: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.defaults, DefaultCatalog) or not isinstance(self.resources, ResourceLibrary):
            raise ValueError("resolution requires an injected data catalog and resource library")
        model = DeviceInput.model_validate(_decode(self.input_json))
        sources = tuple(self.sources)
        if any(not isinstance(source, SourceDocument) for source in sources) or len({source.id for source in sources}) != len(sources):
            raise ValueError("input sources require unique content-bound records")
        ids = [layer.id for layer in model.layers]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("device requires nonempty unique layer IDs")
        materials = {}
        for material in model.materials:
            if material.id in materials:
                raise ValueError("duplicate material ID")
            values = dict(material.parameters.normalized_items())
            if set(values) & LAYER_OWNED_PARAMETERS:
                raise ValueError(f"materials.{material.id}: doping and defect/lifetime parameters belong to layers")
            materials[material.id] = values
        layers = []
        for layer in model.layers:
            if layer.material is not None and layer.material not in materials:
                raise ValueError(f"layers.{layer.id}.material: unknown named material")
            material_values = materials.get(layer.material, {}) if layer.material is not None else {}
            layers.append(_resolve_layer(layer, material_values, self.defaults, self.resources))
        by_id = {layer.id: layer for layer in layers}
        electrical = [layer for layer in layers if layer.role != "substrate"]
        if not electrical:
            raise ValueError("device requires at least one electrical layer")
        if any(layer.role == "substrate" for layer in layers[len(layers) - len(electrical):]):
            raise ValueError("optical substrates must form a front prefix")
        adjacency = set(zip(ids, ids[1:]))
        interfaces = []
        seen_interfaces, endpoints = set(), set()
        interface_inputs = list(model.interfaces)
        supplied_pairs = {(interface.left, interface.right) for interface in interface_inputs}
        for left_id, right_id in zip(ids, ids[1:]):
            if (left_id, right_id) not in supplied_pairs:
                stable = "interface_" + hashlib.sha256(_encode((left_id, right_id))).hexdigest()[:16]
                interface_inputs.append(InterfaceInput(id=stable, left=left_id, right=right_id))
        for interface in interface_inputs:
            pair = (interface.left, interface.right)
            if pair not in adjacency or pair in endpoints or interface.id in seen_interfaces:
                raise ValueError(f"interfaces.{interface.id}: non-adjacent/reversed/duplicate interface")
            endpoints.add(pair)
            seen_interfaces.add(interface.id)
            left, right = (by_id[name] for name in pair)
            interfaces.append(PreparedInterface(_encode(interface.editing_data()),
                              (normalize_quantity(dict(left.parameters)["Eg"], "eV"), normalize_quantity(dict(right.parameters)["Eg"], "eV")),
                              left.role != "substrate" and right.role != "substrate"))
        settings = DeviceSettingsInput.model_validate({**dict(self.defaults.device), **dict(model.settings.normalized_items())})
        setting_values = dict(settings.normalized_items())
        if setting_values["built_in_potential_mode"] == "metal_work_function" and any(setting_values[name] is None for name in ("work_function_left_eV", "work_function_right_eV")):
            raise ValueError("metal_work_function requires both work functions")
        if setting_values["interface_charge_closure"] == "equilibrium_referenced" and not setting_values["interface_charge_rebaseline_acknowledged"]:
            raise ValueError("equilibrium-referenced interface charge requires explicit rebaseline acknowledgement")
        contacts = []
        contact_defaults = dict(self.defaults.contacts)
        for side, endpoint_layer in (("left", electrical[0]), ("right", electrical[-1])):
            matches = [contact for contact in model.contacts if contact.side == side]
            if len(matches) > 1 or (matches and matches[0].layer != endpoint_layer.id):
                raise ValueError(f"contacts.{side}: require exactly one outer electrical-layer binding")
            contact = matches[0].normalized_data() if matches else {"id": "contact_" + side, "side": side, "layer": endpoint_layer.id}
            contacts.append({**contact, **{name: contact[name] if name in contact else contact_defaults[f"{name}_{side}"] for name in ("S_n", "S_p")}})
        if len({contact["id"] for contact in contacts}) != len(contacts):
            raise ValueError("duplicate contact IDs")
        grains = [grain.normalized_data() for grain in model.grain_boundaries]
        if len({grain["id"] for grain in grains}) != len(grains):
            raise ValueError("duplicate grain-boundary IDs")
        for grain in grains:
            if not grain["layer_ids"] or len(set(grain["layer_ids"])) != len(grain["layer_ids"]) or any(name not in by_id or by_id[name].role == "substrate" for name in grain["layer_ids"]):
                raise ValueError("grain boundaries require existing electrical layer IDs")
            if grain["x_position"] - grain["width"] / 2 < 0:
                raise ValueError("grain boundary extends before the lateral origin")
        for index, left_grain in enumerate(grains):
            for right_grain in grains[index + 1:]:
                if not set(left_grain["layer_ids"]) & set(right_grain["layer_ids"]):
                    continue
                left_edge = max(grain["x_position"] - grain["width"] / 2 for grain in (left_grain, right_grain))
                right_edge = min(grain["x_position"] + grain["width"] / 2 for grain in (left_grain, right_grain))
                if right_edge > left_edge:
                    raise ValueError("overlapping grain-boundary bands in the same layer are unsupported")
        grid = [entry.normalized_data() for entry in model.electrical_grid]
        if grid and (len(grid) != len(electrical) or {item["layer"] for item in grid} != {layer.id for layer in electrical}):
            raise ValueError("electrical grid must cover each electrical layer exactly once")
        selected_resources = {layer.optical.name: layer.optical for layer in layers if layer.optical}
        for name, kind in ((model.spectrum, "spectrum"), (model.fixed_generation, "fixed_generation")):
            if name is not None:
                selected_resources[name] = self.resources.get(name, kind)
        diagnostics = []
        if model.source_format == "scaps":
            old_default = dict(self.defaults.material)["incoherent"]
            loader = [pair for pair in self.defaults.evidence if pair[0].endswith("scaps_compat/loader.py")]
            for layer in model.layers:
                if "incoherent" in layer.parameters.model_fields_set and layer.parameters.incoherent != old_default:
                    diagnostics.append({"code": "legacy_scaps_ignored_incoherent", "path": f"layers.{layer.id}.parameters.incoherent",
                        "declared": layer.parameters.incoherent, "historical_effective": old_default,
                        "source_bindings": [{"id": source.id, "sha256": source.sha256} for source in sources],
                        "loader_bindings": [list(pair) for pair in loader],
                        "required_action": "P03-04 preserve historical effect; activation is a separate optical P correction"})
        gaps = ["G2_pending_no_executor", "P03_04_effective_behavior_context_pending"]
        if model.source_format == "scaps":
            gaps.append("P03_04_legacy_unit_conversion_bytes_pending")
        if grains:
            gaps.append("microstructure_requires_lateral_geometry_and_neumann_boundary")
        if any(layer.defects for layer in layers):
            gaps.append("microscopic_defect_execution_requires_qualified_P04_binding")
        if any(dict(layer.parameters)["carrier_statistics"] == "fermi_dirac" for layer in layers):
            gaps.append("fermi_dirac_narrow_scope_not_admitted_by_configuration")
        if setting_values["built_in_potential_mode"] != "legacy_manual":
            gaps.append("contact_potential_requires_behavior_or_physical_model_evaluation")
        data = {"schema": "solarlab.resolved-device-preparation.v1", "unit_schema": UNIT_SCHEMA_VERSION,
                "id": model.id, "settings": setting_values, "layers": [layer.to_mapping() for layer in layers],
                "declaration": model.normalized_data(),
                "interfaces": [interface.to_mapping() for interface in interfaces], "contacts": contacts,
                "grain_boundaries": grains, "electrical_grid": grid,
                "resource_bindings": [{"name": name, "kind": resource.kind, "source_id": resource.source.id,
                                       "source_sha256": resource.source.sha256, "content_sha256": resource.content_sha256,
                                       "shape": resource.shape, "columns": resource.columns, "units": resource.units}
                                      for name, resource in sorted(selected_resources.items())],
                "diagnostics": diagnostics, "capability_gaps": sorted(gaps), "can_execute": False,
                "default_catalog": self.defaults.to_mapping(),
                "input_sources": [{"id": source.id, "sha256": source.sha256} for source in sources]}
        object.__setattr__(self, "input_json", _encode(model.editing_data()))
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "layers", tuple(layers))
        object.__setattr__(self, "interfaces", tuple(interfaces))
        object.__setattr__(self, "_resolved_json", _encode(data))

    def to_input(self) -> DeviceInput:
        return DeviceInput.model_validate(_decode(self.input_json))

    def to_mapping(self) -> dict[str, Any]:
        return _decode(self._resolved_json)

    @property
    def values(self) -> Mapping[str, Any]:
        return _readonly(self.to_mapping())

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.input_json + self._resolved_json).hexdigest()


@dataclass(frozen=True, slots=True)
class PreparedTandem:
    input_json: bytes
    defaults: DefaultCatalog
    resources: ResourceLibrary
    sources: tuple[SourceDocument, ...] = ()
    top_cell: PreparedDevice = field(init=False)
    bottom_cell: PreparedDevice = field(init=False)
    _resolved_json: bytes = field(init=False, repr=False)
    can_execute: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        model = TandemInput.model_validate(_decode(self.input_json))
        if model.top_cell.id == model.bottom_cell.id:
            raise ValueError("tandem cells require distinct stable IDs")
        top = PreparedDevice(_encode(model.top_cell.editing_data()), self.defaults, self.resources, self.sources)
        bottom = PreparedDevice(_encode(model.bottom_cell.editing_data()), self.defaults, self.resources, self.sources)
        optics = []
        ids = set()
        for layer in (*model.junction_stack, *((model.back_reflector,) if model.back_reflector is not None else ())):
            if layer.id in ids:
                raise ValueError("duplicate tandem optical-layer ID")
            ids.add(layer.id)
            resource = self.resources.get(layer.optical_material, "nk")
            optics.append({**layer.normalized_data(), "resource_sha256": resource.content_sha256})
        data = {"schema": "solarlab.resolved-tandem-preparation.v1", "id": model.id,
                "declaration": model.normalized_data(),
                "top_cell": top.to_mapping(), "bottom_cell": bottom.to_mapping(), "optical_layers": optics,
                "junction_model": model.junction_model, "light_direction": model.light_direction,
                "benchmark": model.benchmark.normalized_data() if model.benchmark else None,
                "can_execute": False, "capability_gaps": ["G2_pending_no_executor", "tandem_optical_and_series_coupling_not_qualified"]}
        object.__setattr__(self, "input_json", _encode(model.editing_data()))
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "top_cell", top)
        object.__setattr__(self, "bottom_cell", bottom)
        object.__setattr__(self, "_resolved_json", _encode(data))

    def to_input(self) -> TandemInput:
        return TandemInput.model_validate(_decode(self.input_json))

    def to_mapping(self) -> dict[str, Any]:
        return _decode(self._resolved_json)

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.input_json + self._resolved_json).hexdigest()
