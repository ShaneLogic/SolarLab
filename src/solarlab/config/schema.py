"""Export prepared configuration metadata without activating a model or solver.

Every nested ``schema`` is a complete, independent JSON Schema document. Resolve
its local references against that document, never against the enclosing bundle.
Unit annotations and presence rules come from the DTOs; backend validators remain
authoritative for semantic constraints that JSON Schema cannot express.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from solarlab.config.inputs import LayerInput
from solarlab.device.defects import (
    LegacyBulkTrapInput, MetastableDocumentInput, MetastablePreparationInput,
    MultivalentDefectInput,
)
from solarlab.device.inputs import (
    BulkDefectInput, ContactInput, DeviceInput, FullLayerInput, InterfaceInput,
    NamedMaterialInput, TandemInput,
)
from solarlab.device.settings import DeviceSettingsInput
from solarlab.device.tunnelling import TunnellingInput
from solarlab.materials.full_parameters import FullParameterInput
from solarlab.materials.optics import CigsOpticsInput
from solarlab.materials.parameter_schema import parameter_schema
from solarlab.materials.parameters import LayerValuesInput
from solarlab.units import UNIT_SCHEMA_VERSION, supported_units

if TYPE_CHECKING:
    from solarlab.physics.registry import StructuredSchema

__all__ = ["export_configuration_schema"]

# Explicit coverage of existing editing entry points, not a model catalogue.
# Their nested DTOs remain in each entry point's own $defs namespace.
_INPUT_DTOS = (
    LayerInput, LayerValuesInput, FullParameterInput, DeviceSettingsInput,
    DeviceInput, TandemInput, FullLayerInput, NamedMaterialInput,
    BulkDefectInput, MultivalentDefectInput, LegacyBulkTrapInput,
    MetastableDocumentInput, MetastablePreparationInput, InterfaceInput,
    ContactInput, TunnellingInput, CigsOpticsInput,
)


def _declared_units(value: Any) -> set[str]:
    if isinstance(value, list):
        return set().union(*(_declared_units(item) for item in value))
    if not isinstance(value, dict):
        return set()
    own = {value[key] for key in ("unit", "item_unit")
           if isinstance(value.get(key), str)}
    return own.union(*(_declared_units(item) for item in value.values()))


def export_configuration_schema(
    *, structured_schemas: Iterable[StructuredSchema] = (),
) -> dict[str, Any]:
    """Return detached schema DATA, with optional existing structured metadata.

    Callers may supply actual ``StructuredSchema`` records obtained from their
    prepared configuration. Their public exports preserve editable/normalized
    schema scope and source identity. No default catalogue or prepared device is
    created here, and normalized parameters are not a resolved configuration.
    Conflicting definitions of the same structured ID/version are rejected.
    """
    inputs = {
        dto.__name__: {
            "python_type": f"{dto.__module__}:{dto.__qualname__}",
            "representation": "editable_input",
            "schema": dto.model_json_schema(mode="validation"),
        }
        for dto in sorted(_INPUT_DTOS, key=lambda dto: dto.__name__)
    }
    parameters = {name: parameter_schema(name)
                  for name in ("scalar_layer", "full_layer")}
    structured: dict[str, Any] = {}
    for item in structured_schemas:
        key = f"{item.id}@{item.version}"
        document = item.export()
        if key in structured and structured[key] != document:
            raise ValueError(f"conflicting structured schema: {key}")
        structured[key] = document
    structured = dict(sorted(structured.items()))
    units = _declared_units([inputs, parameters, structured])
    return {
        "schema_version": "solarlab.configuration-schema-export.v1",
        "status": "prepared_pending_dependencies",
        "scope": "prepared_configuration_metadata_only",
        "schema_reference_scope": "each complete schema document",
        "backend_semantic_validation_required": True,
        "unit_schema_version": UNIT_SCHEMA_VERSION,
        "unit_spellings": {unit: list(supported_units(unit))
                           for unit in sorted(units)},
        "dto_schemas": inputs,
        "parameter_schemas": parameters,
        "structured_parameter_schemas": structured,
        "pending": [
            "input_resolved_result_wire_types",
            "resolved_physical_and_numerical_configuration_schema",
            "production_model_catalogue_and_executors",
            "complex_editors_and_frontend_model_service_integration",
            "service_browser_identity_handshake",
            "P03-07_and_P06-08_integration_and_G3_G6_qualification",
        ],
    }
