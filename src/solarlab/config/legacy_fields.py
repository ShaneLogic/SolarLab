"""Retained standard-loader input evidence, never additional physics settings.

Version 1 describes the reviewed loader's ignored temperature and full raw-layer
interface indexing. A physical correction must use a distinct behavior version.
Original source bytes remain in the PreparedDevice's SourceDocuments.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Any, Literal

from pydantic import Field, field_serializer, field_validator

from solarlab.config.yaml import load_yaml_mapping
from solarlab.materials.full_parameters import Nonempty, StableId, StructuredInput
from solarlab.materials.parameters import QuantityInput
from solarlab.materials.source import SourceDocument

LegacyVelocityPair = Annotated[tuple[QuantityInput, ...], Field(min_length=2, max_length=2)]


class LegacyDeviceFieldsInput(StructuredInput):
    schema_version: Literal["solarlab.standard-loader-fields.v1"]
    source_id: Nonempty
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    layer_ids: tuple[StableId, ...]
    # Deliberately no temperature unit validator: the old reader ignores this
    # value even when it is zero, false, null or a nonnumeric string.
    temperature: QuantityInput | bool | None = None
    interfaces: tuple[LegacyVelocityPair, ...] | None = None
    interface_defect_count: int | None = Field(default=None, ge=0)

    @field_validator("interfaces", mode="before")
    @classmethod
    def _pairs(cls, value: object) -> object:
        if value is None:
            return value
        if not isinstance(value, (tuple, list)) or any(
            not isinstance(pair, (tuple, list)) or len(pair) != 2 for pair in value
        ):
            raise ValueError("legacy interfaces require ordered [v_n,v_p] pairs")
        return tuple(tuple(pair) for pair in value)

    @field_serializer("temperature", "interfaces", when_used="json")
    def _raw_numbers(self, value: object) -> object:
        def numeric(item: object) -> object:
            if isinstance(item, Decimal):
                return float(item)
            if isinstance(item, (list, tuple)):
                return [numeric(child) for child in item]
            return item
        return numeric(value)


def retained_legacy_fields(source: SourceDocument) -> LegacyDeviceFieldsInput | None:
    """Read only original supplied evidence; no defaults, file access or models."""
    original = load_yaml_mapping(source.content)
    raw, layers = original["device"], original["layers"]
    retained: dict[str, Any] = {"temperature": raw["temperature"]} if "temperature" in raw else {}
    short = any(key in raw and len(raw[key] or ()) < len(layers) - 1
                for key in ("interfaces", "interface_defects"))
    if not retained and not short:
        return None
    if "interfaces" in raw:
        retained["interfaces"] = raw["interfaces"]
    if "interface_defects" in raw:
        retained["interface_defect_count"] = len(raw["interface_defects"] or ())
    return LegacyDeviceFieldsInput.model_validate({
        "schema_version": "solarlab.standard-loader-fields.v1",
        "source_id": source.id, "source_sha256": source.sha256,
        "layer_ids": [layer.get("id", f"layer_{i}") for i, layer in enumerate(layers)],
        **retained,
    })


# Audited source identities, not a second physical default catalogue. Resolution
# performs no repository lookup and does not import these legacy modules.
STANDARD_LOADER_BINDING = {
    "path": "perovskite-sim/perovskite_sim/models/config_loader.py",
    "sha256": "f25dd749fec6e2e755cf18d221986efdc323b6f2c22e92ec7297577117442bb2",
}
INLINE_LOADER_BINDING = {
    "path": "perovskite-sim/backend/main.py",
    "sha256": "7dd1292f6f34f0f065a50686802c20a8484edcc9e4a7d98287799c076f9da01f",
}
JV_HINT_CONSUMER_BINDINGS = (
    {"path": "web/src/workstation/panes/jv-pane.ts",
     "sha256": "2b50f1819655ea89f2bd83baef4656e64ac9a2e680a5bf0ebfa4f8b74c125a06"},
    {"path": "web/src/jv-waveform-controls.ts",
     "sha256": "9d9fdf0b6d9245d9707dac60f09177343e4e7ad6adb3391a016bad5d9a64d4e2"},
)
