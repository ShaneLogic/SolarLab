"""Bind existing closed device DTOs to registry instances without physics.

The prepared device, supplied default catalog and supplied resource bytes are
the authority. An edit replaces one complete DTO and resolves that device again;
this module has no second field/default table and no plugin discovery. Input
presence, normalized parameters and the resolver's derived projection remain
separate. The normal path never reads old Python source or imports its engine.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import re
from typing import Any

from solarlab.config.resolve_device import resolve_device
from solarlab.device.defects import (
    LegacyBulkTrapInput, MetastableDocumentInput, MetastablePreparationInput,
    MultivalentDefectInput,
)
from solarlab.device.inputs import BulkDefectInput, ContactInput, DeviceInput, InterfaceInput
from solarlab.device.resolved import PreparedDevice
from solarlab.device.tunnelling import TunnellingInput
from solarlab.materials.full_parameters import StructuredInput
from solarlab.materials.optics import CigsOpticsInput
from solarlab.physics.registry import (
    EvidenceRef, Scope, StructuredData, StructuredParameters, StructuredSchema,
    metadata_digest,
)
from solarlab.physics.selection import ParameterSource, Topology
from solarlab.units import UNIT_SCHEMA_VERSION

__all__ = ["structured_parameter_source", "structured_parameter_sources",
           "update_structured_source", "resolve_structured_edit", "device_topology"]

# These are actual field/DTO bindings, not another schema or default table.
_LAYER_DTOS: dict[str, type[StructuredInput]] = {
    "bulk_trap_distribution": LegacyBulkTrapInput,
    "cigs_graded_optics": CigsOpticsInput,
    "metastable_document": MetastableDocumentInput,
    "metastable_preparation": MetastablePreparationInput,
}
_ABSENT = object()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _normalized_schema(value: Any) -> Any:
    """Derive quantity output types from the DTO's own unit annotations."""
    if isinstance(value, list):
        return [_normalized_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: _normalized_schema(item) for key, item in value.items()}
    if "unit" in result:
        number: dict[str, Any] = {"type": "number"}
        if result.get("positive"):
            number["exclusiveMinimum"] = 0
        elif result.get("nonnegative"):
            number["minimum"] = 0
        result.pop("anyOf", None)
        result.pop("type", None)
        if result.get("nullable"):
            result["anyOf"] = [number, {"type": "null"}]
        else:
            result.update(number)
    if "item_unit" in result:
        result["items"] = {"type": "number", "unit": result["item_unit"]}
        if result.get("item_positive"):
            result["items"]["exclusiveMinimum"] = 0
    return result


def _schema(dto: type[StructuredInput], family: str, scope: str, version: str) -> StructuredSchema:
    raw = dto.model_json_schema()
    normalized = _normalized_schema(raw)
    normalized["title"] += "NormalizedParameters"
    normalized["x-solarlab-representation"] = "normalized_parameters"
    normalized["x-solarlab-unit-schema"] = UNIT_SCHEMA_VERSION
    match = re.search(r"-v([1-9][0-9]*)$", version)
    number = int(match.group(1)) if match else 1
    return StructuredSchema(
        f"solarlab.parameters.{family}.v{number}", number, family, scope, version,
        _json(raw), _json(normalized),
        EvidenceRef("dto-json-schema:" + dto.__module__ + ":" + dto.__qualname__, metadata_digest(raw), "source"),
    )


@dataclass(frozen=True)
class _Location:
    dto: type[StructuredInput]
    version: str
    scope: Scope
    orientation: tuple[str, ...]
    declared: bool
    input: StructuredInput | None
    normalized: dict[str, Any] | None
    projection: dict[str, Any] | None
    path_by_id: tuple[str, ...]
    context: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _DeviceParameterAuthority:
    prepared: PreparedDevice
    family: str
    owner_id: str
    local_id: str

    def _locate(self) -> _Location:
        if not isinstance(self.prepared, PreparedDevice):
            raise ValueError("structured parameters require a validated prepared device")
        model, resolved = self.prepared.to_input(), self.prepared.to_mapping()
        dto: type[StructuredInput]
        if self.family == "bulk_defect" or self.family in _LAYER_DTOS:
            layer_matches = [(raw, layer) for raw, layer in zip(model.layers, self.prepared.layers) if raw.id == self.owner_id]
            if len(layer_matches) != 1:
                raise ValueError("structured parameters name an unknown layer ID")
            raw_layer, layer = layer_matches[0]
            scope = Scope("layer", (layer.id,))
            context = {"band_gap_eV": dict(layer.parameters)["Eg"],
                       "material_id": layer.material.id, "material_sha256": layer.material.content_sha256,
                       "defect_model": layer.defect_model, "defect_schema_version": layer.defect_schema_version}
            if self.family == "bulk_defect":
                found = [(raw, item) for raw, item in zip(raw_layer.bulk_defects, layer.defects) if raw.id == self.local_id]
                if len(found) != 1:
                    raise ValueError("bulk defect parameters require an existing stable species ID")
                raw, item = found[0]
                dto = MultivalentDefectInput if isinstance(raw, MultivalentDefectInput) else BulkDefectInput
                normalized = dto.model_validate_json(item.input_json).normalized_data()
                return _Location(dto, str(layer.defect_schema_version), scope, scope.ids, True, raw,
                                 normalized, item.to_mapping(), ("layers", layer.id, "bulk_defects", self.local_id), context)
            if self.local_id != self.family:
                raise ValueError("a singleton layer declaration has one canonical local ID")
            dto = _LAYER_DTOS[self.family]
            source_json = getattr(layer, self.family + "_json")
            effective = dto.model_validate_json(source_json) if source_json is not None else None
            version = dto.model_json_schema()["properties"].get("schema_version", {}).get(
                "const", model.schema_version + "#" + self.family)
            return _Location(dto, version, scope, scope.ids, self.family in raw_layer.model_fields_set,
                             getattr(raw_layer, self.family), effective.normalized_data() if effective is not None else None,
                             layer.to_mapping().get(self.family), ("layers", layer.id, self.family), context)
        if self.local_id != self.family:
            raise ValueError("a singleton declaration has one canonical local ID")
        if self.family == "tunnelling_channels":
            if self.owner_id != model.id:
                raise ValueError("tunnelling channel document belongs to the explicit device ID")
            effective_data = resolved.get(self.family)
            effective = TunnellingInput.model_validate(effective_data) if effective_data is not None else None
            # The version is a DTO constant; no physical defaults are supplied here.
            version = TunnellingInput.model_fields["schema_version"].default
            return _Location(TunnellingInput, version, Scope("device"), (), self.family in model.model_fields_set,
                             model.tunnelling_channels, effective.normalized_data() if effective else None,
                             effective_data, (self.family,), {})
        if self.family == "interface":
            interfaces = [item for item in self.prepared.interfaces if item.values["id"] == self.owner_id]
            if len(interfaces) != 1:
                raise ValueError("unknown interface ID")
            interface = interfaces[0]
            if not interface.electrical:
                raise ValueError("optical-only interfaces do not belong to electrical model selection")
            effective = InterfaceInput.model_validate_json(interface.input_json)
            interface_inputs = [item for item in model.interfaces if item.id == self.owner_id]
            scope = Scope("interface", (effective.left, effective.right))
            return _Location(InterfaceInput, model.schema_version + "#interface", scope, scope.ids, bool(interface_inputs),
                             interface_inputs[0] if interface_inputs else None, effective.normalized_data(), interface.to_mapping(),
                             ("interfaces", self.owner_id), {"adjacent_gaps_eV": list(interface.adjacent_gaps_eV)})
        if self.family == "contact":
            contacts = [item for item in resolved["contacts"] if item["id"] == self.owner_id]
            if len(contacts) != 1:
                raise ValueError("unknown contact ID")
            contact = contacts[0]
            contact_inputs = [item for item in model.contacts if item.id == self.owner_id]
            return _Location(ContactInput, model.schema_version + "#contact", Scope("contact", (self.owner_id,)),
                             (contact["side"], contact["layer"]), bool(contact_inputs), contact_inputs[0] if contact_inputs else None,
                             ContactInput.model_validate(contact).normalized_data(), contact,
                             ("contacts", self.owner_id), {})
        raise ValueError(f"unavailable structured parameter family: {self.family!r}")

    def snapshot(self) -> StructuredData:
        location = self._locate()
        resolved = self.prepared.to_mapping()
        schema = _schema(location.dto, self.family, location.scope.kind, location.version)
        raw = location.input.editing_data() if location.input is not None else None
        presence = "absent" if not location.declared else ("null" if raw is None else "value")
        reference = {"schema": resolved["schema"], "resolver": "solarlab.device.resolved:PreparedDevice",
                     "device_sha256": self.prepared.content_sha256, "path_by_id": list(location.path_by_id),
                     "projection": location.projection, "context": location.context,
                     "default_catalog_sha256": metadata_digest(resolved["default_catalog"]),
                     "resource_bindings": resolved["resource_bindings"], "input_sources": resolved["input_sources"]}
        return StructuredData(schema, location.scope, resolved["id"], self.owner_id, self.local_id,
                              location.orientation, presence, _json(raw), _json(location.normalized), _json(reference),
                              EvidenceRef("prepared.device:" + resolved["id"], self.prepared.content_sha256, "source"),
                              tuple(resolved["capability_gaps"]))

    def replace(self, document: object) -> _DeviceParameterAuthority:
        location = self._locate()
        if document is _ABSENT or document is None:
            if self.family not in _LAYER_DTOS and self.family != "tunnelling_channels":
                raise ValueError("required instance/owner identities cannot be cleared or inherited")
            value = document
        else:
            if not isinstance(document, (Mapping, StructuredInput)):
                raise ValueError("a complete typed structured document is required")
            value = location.dto.model_validate(document).editing_data()
        data = self.prepared.to_input().editing_data()
        if self.family == "bulk_defect" or self.family in _LAYER_DTOS:
            layer = next(item for item in data["layers"] if item["id"] == self.owner_id)
            if self.family == "bulk_defect":
                if not isinstance(value, dict):
                    raise ValueError("a complete species document is required")
                if value["id"] != self.local_id:
                    raise ValueError("an edit cannot rename the selected species identity")
                layer["bulk_defects"] = [value if item["id"] == self.local_id else item for item in layer["bulk_defects"]]
            elif value is _ABSENT:
                layer.pop(self.family, None)
            else:
                layer[self.family] = value
        elif self.family in {"interface", "contact"}:
            if not isinstance(value, dict):
                raise ValueError("a complete owner document is required")
            if value["id"] != self.owner_id:
                raise ValueError("an edit cannot rename its owner identity")
            collection = self.family + "s"
            entries = data.get(collection, [])
            data[collection] = ([value if item["id"] == self.owner_id else item for item in entries]
                                if any(item["id"] == self.owner_id for item in entries) else [*entries, value])
        elif value is _ABSENT:
            data.pop(self.family, None)
        else:
            data[self.family] = value
        prepared = resolve_device(DeviceInput.model_validate(data), self.prepared.defaults,
                                  self.prepared.resources, sources=self.prepared.sources)
        result = _DeviceParameterAuthority(prepared, self.family, self.owner_id, self.local_id)
        updated = result._locate()
        if updated.scope != location.scope or updated.orientation != location.orientation:
            raise ValueError("an edit cannot change selected scope or orientation")
        return result


def structured_parameter_source(
    prepared: PreparedDevice, family: str, owner_id: str, *, instance_id: str | None = None,
) -> ParameterSource:
    """Bind an actual species ID, or the singleton field's canonical local ID."""
    if family == "bulk_defect" and instance_id is None:
        raise ValueError("bulk_defect requires its existing stable species instance_id")
    local_id = family if instance_id is None else instance_id
    parameters = StructuredParameters(_DeviceParameterAuthority(prepared, family, owner_id, local_id))
    source_id = "structured:" + metadata_digest([parameters.data.device_id, family, owner_id, local_id])
    return ParameterSource(source_id, (),
                           parameters.data.source, parameters.data.schema.id, parameters)


def update_structured_source(source: ParameterSource, document: object = _ABSENT) -> ParameterSource:
    """Replace a whole validated DTO; omission inherits, explicit None clears.

    Nested edits use the existing DTO validated_update methods before this call.
    Parent/device constraints are then rechecked with the same supplied catalog
    and resources. Required species and owner identities cannot be removed.
    """
    if not isinstance(source, ParameterSource) or source.structured is None:
        raise ValueError("an existing structured parameter source is required")
    authority = source.structured._authority
    if not isinstance(authority, _DeviceParameterAuthority):
        raise ValueError("this edit requires the closed prepared-device authority")
    parameters = StructuredParameters(authority.replace(document))
    if parameters.data.schema != source.structured.data.schema:
        raise ValueError("schema/version changes require an explicit new model binding")
    return ParameterSource(source.id, (), parameters.data.source, parameters.data.schema.id, parameters)


def resolve_structured_edit(source: ParameterSource, document: object = _ABSENT) -> PreparedDevice:
    """Return the edited device so all selected instances can be rebound together.

    A selection cannot mix structured records from distinct device snapshots.
    Re-enumerate structured_parameter_sources on this result after an edit.
    """
    updated = update_structured_source(source, document)
    assert updated.structured is not None
    authority = updated.structured._authority
    assert isinstance(authority, _DeviceParameterAuthority)
    return authority.prepared


def device_topology(prepared: PreparedDevice) -> Topology:
    """Preserve actual electrical layer, interface and contact identities."""
    if not isinstance(prepared, PreparedDevice):
        raise ValueError("a prepared device is required")
    resolved = prepared.to_mapping()
    return Topology(tuple(layer.id for layer in prepared.layers if layer.role != "substrate"),
                    tuple((item["id"], item["layer"]) for item in resolved["contacts"]),
                    tuple((item.values["id"], item.values["left"], item.values["right"])
                          for item in prepared.interfaces if item.electrical),
                    tuple((item["id"], item["side"]) for item in resolved["contacts"]), resolved["id"])


def structured_parameter_sources(prepared: PreparedDevice) -> tuple[ParameterSource, ...]:
    """Canonical ordered actual declarations, including defaults/explicit clears."""
    topology = device_topology(prepared)
    model = prepared.to_input()
    sources: list[ParameterSource] = []
    for raw, layer in zip(model.layers, prepared.layers):
        if layer.id not in topology.electrical_layer_ids:
            if layer.defects or any(family in raw.model_fields_set or getattr(layer, family + "_json") is not None for family in _LAYER_DTOS):
                raise ValueError("structured declarations on optical-only layers are unavailable in electrical model selection")
            continue
        sources.extend(structured_parameter_source(prepared, "bulk_defect", layer.id, instance_id=item.values["id"])
                       for item in layer.defects)
        for family in _LAYER_DTOS:
            if family in raw.model_fields_set or getattr(layer, family + "_json") is not None:
                sources.append(structured_parameter_source(prepared, family, layer.id))
    sources.extend(structured_parameter_source(prepared, "interface", identifier) for identifier, _, _ in topology.interface_ids)
    sources.extend(structured_parameter_source(prepared, "contact", identifier) for identifier, _ in topology.contacts)
    if "tunnelling_channels" in model.model_fields_set:
        sources.append(structured_parameter_source(prepared, "tunnelling_channels", model.id))
    def identifier(source: ParameterSource) -> str:
        assert source.structured is not None  # Every entry above uses the closed factory.
        return source.structured.data.instance_id
    return tuple(sorted(sources, key=identifier))
