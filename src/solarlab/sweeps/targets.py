"""Closed bindings to actual DTO fields; stable IDs never mean list positions."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from pydantic import TypeAdapter
from solarlab.device.inputs import DeviceInput, DistributionInput, EnergyLevelInput, InterfaceDefectInput
from solarlab.device.settings import DeviceSettingsInput
from solarlab.experiments.inputs import invalid
from solarlab.experiments.jv.inputs import DarkJVInput, JVExperimentInput, JVInput
from solarlab.materials.full_parameters import FullParameterInput, StructuredInput
from solarlab.sweeps.inputs import SweepCoordinateInput, SweepTargetInput
from solarlab.units import normalize_quantity


def _one(rows: list[dict[str, Any]], id: str | None, path: tuple[str | int, ...]) -> tuple[int, dict[str, Any]]:
    found = [(i, row) for i, row in enumerate(rows) if row.get("id") == id]
    if len(found) != 1:
        invalid("SweepTarget", path, "stable target ID is missing or ambiguous; no positional fallback", id)
    return found[0]


@dataclass(frozen=True)
class TargetBinding:
    target: SweepTargetInput
    input_path: tuple[str | int, ...]
    stable_path: tuple[str, ...]
    dto: type[StructuredInput]
    field: str
    reference_layer: str | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        return self.dto.model_json_schema()["properties"][self.field]

    @property
    def required(self) -> bool:
        return self.dto.model_fields[self.field].is_required() or self.target.family == "cbo"

    def coordinate(self, value: SweepCoordinateInput) -> dict[str, Any]:
        if value.kind == "omit":
            if self.required:
                raise ValueError("required target has no inheritance/omission mode")
            return {"kind": "omit"}
        raw, metadata = value.value, self.metadata
        if raw is None:
            if not self.dto._allows_null(self.field):
                raise ValueError("target does not permit explicit null")
            return {"kind": "value", "value": None}
        if "unit" in metadata:
            number = normalize_quantity(raw, metadata["unit"], input_unit=metadata.get("input_unit"))
            if metadata.get("positive") and number <= 0:
                raise ValueError("target value must be positive")
            if metadata.get("nonnegative", True) and number < 0 and self.target.family != "cbo":
                raise ValueError("target value must be nonnegative")
            return {"kind": "value", "value": number}
        checked = TypeAdapter(self.dto.model_fields[self.field].rebuild_annotation()).validate_python(raw, strict=True)
        return {"kind": "value", "value": checked}


def bind_target(base: DeviceInput | JVExperimentInput, target: SweepTargetInput) -> TargetBinding:
    device = base.device if isinstance(base, JVExperimentInput) else base
    raw = device.editing_data()
    prefix: tuple[str | int, ...] = ("device",) if isinstance(base, JVExperimentInput) else ()
    family, owner, local = target.family, target.owner_id, target.local_id
    parameter = target.parameter
    stable = (family, *(x for x in (owner, local, target.reference_id) if x is not None), *parameter)
    dto: type[StructuredInput]
    reference = None
    if family == "setting":
        dto, path = DeviceSettingsInput, (*prefix, "settings")
    elif family == "jv":
        if not isinstance(base, JVExperimentInput):
            raise ValueError("a J-V target requires a supplied J-V experiment base")
        dto, path = (JVInput if isinstance(base.experiment, JVInput) else DarkJVInput), ("experiment",)
    elif family == "material_parameter":
        index, _ = _one(raw.get("materials", []), owner, ("materials", str(owner)))
        dto, path = FullParameterInput, (*prefix, "materials", index, "parameters")
    elif family in {"layer_parameter", "cbo", "bulk_defect"}:
        index, layer = _one(raw["layers"], owner, ("layers", str(owner)))
        path = (*prefix, "layers", index)
        if family == "bulk_defect":
            defect_index, defect = _one(layer.get("bulk_defects", []), local, ("layers", str(owner), "bulk_defects", str(local)))
            if len(parameter) != 2 or parameter[0] not in {"distribution", "energy_level"}:
                raise ValueError("bulk sweep fields belong to a supplied distribution or referenced energy_level")
            if not isinstance(defect.get(parameter[0]), dict):
                raise ValueError("create the required defect declaration before selecting this sweep target")
            dto = DistributionInput if parameter[0] == "distribution" else EnergyLevelInput
            path = (*path, "bulk_defects", defect_index, parameter[0])
        else:
            dto, path = FullParameterInput, (*path, "parameters")
        if family == "cbo":
            _, anchor = _one(raw["layers"], target.reference_id, ("layers", str(target.reference_id)))
            if parameter != ("chi",) or owner == target.reference_id or layer["role"] != "ETL" or anchor["role"] != "absorber":
                raise ValueError("CBO requires distinct supplied ETL/absorber IDs and the chi field")
            reference = target.reference_id
    elif family == "interface_defect":
        index, interface = _one(raw.get("interfaces", []), owner, ("interfaces", str(owner)))
        if not isinstance(interface.get("defect"), dict) or interface["defect"].get("id") != local:
            raise ValueError("interface target requires its supplied defect ID")
        dto, path = InterfaceDefectInput, (*prefix, "interfaces", index, "defect")
    else:
        raise ValueError("unsupported sweep family")
    field = parameter[-1]
    if len(parameter) != (2 if family == "bulk_defect" else 1) or field not in dto.model_fields:
        raise ValueError("target parameter is not a field in its authoritative DTO")
    metadata = dto.model_json_schema()["properties"][field]
    variants = metadata.get("anyOf", [metadata])
    if field in {"id", "kind", "request_api", "schema_version"} or any("$ref" in value or value.get("type") in {"array", "object"} for value in variants):
        raise ValueError("only existing scalar model parameters are sweep targets; identities and complex declarations are retained")
    return TargetBinding(target, (*path, field), stable, dto, field, reference)


def write_coordinate(data: dict[str, Any], binding: TargetBinding, coordinate: SweepCoordinateInput,
                     reference_affinity: float | None = None) -> None:
    owner = data
    for key in binding.input_path[:-1]:
        if isinstance(owner, dict) and key not in owner:
            if coordinate.kind == "omit":
                return
            if key not in {"parameters", "settings"}:
                raise ValueError("sweep edits cannot initialize a missing complex declaration")
            owner[key] = {}
        owner = owner[key]
    key = binding.input_path[-1]
    assert isinstance(key, str)
    if coordinate.kind == "omit":
        owner.pop(key, None)
    elif binding.reference_layer is not None:
        assert reference_affinity is not None
        number = normalize_quantity(coordinate.value, "eV")
        owner[key] = f"{Decimal(str(reference_affinity)) - Decimal(str(number))} eV"
    else:
        owner[key] = coordinate.value


def effective_target(binding: TargetBinding, resolved: dict[str, Any]) -> dict[str, Any]:
    device = resolved.get("device", resolved)
    family = binding.target.family
    if family == "jv":
        return {"value": resolved["experiment"]["controls"].get(binding.field), "origin": resolved["experiment"]["field_origins"].get(binding.field)}
    if family == "setting":
        return {"value": device["settings"].get(binding.field), "origin": "prepared_device_settings"}
    if family == "material_parameter":
        layers = [layer for layer in device["layers"] if layer["material_id"] == binding.target.owner_id]
        return {"consumers": [{"layer_id": layer["id"], **next(row for row in layer["provenance"] if row["parameter"] == binding.field)} for layer in layers], "origin": "named_material_with_layer_overrides_retained"}
    if family == "interface_defect":
        _, interface = _one(device["interfaces"], binding.target.owner_id, ("interfaces", str(binding.target.owner_id)))
        # The prepared interface retains the authoritative normalized declaration.
        return {"value": interface["defect"][binding.field], "origin": "supplied_interface_defect", "energy_reference": interface["defect"]["energy_reference"]}
    _, layer = _one(device["layers"], binding.target.owner_id, ("layers", str(binding.target.owner_id)))
    if family == "bulk_defect":
        _, defect = _one(layer["bulk_defects"], binding.target.local_id, ("bulk_defects", str(binding.target.local_id)))
        value = defect
        for part in binding.target.parameter:
            value = value[part]
        return {"value": value, "origin": "supplied_bulk_defect", "energy_reference": defect.get("energy_level", {}).get("reference")}
    provenance = next(row for row in layer["provenance"] if row["parameter"] == binding.field)
    result = {"value": provenance["effective_value"], "origin": provenance["origin"]}
    if family == "cbo":
        _, anchor = _one(device["layers"], binding.reference_layer, ("layers", str(binding.reference_layer)))
        affinity = anchor["material_parameters"]["chi"]
        result.update(reference_affinity_eV=affinity, offset_eV=float(Decimal(str(affinity)) - Decimal(str(result["value"]))))
    return result
