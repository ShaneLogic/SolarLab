"""P05-01a draft data contracts; no state transfer or experiment execution.

WS2/WS3 co-signing is still required before registry/service activation. A
content reference locates evidence; it does not certify a working point.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Any

from solarlab.io.artifacts import _json_document, _parse_document

__all__ = ["Protocol", "WorkingPoint", "protocol_schema", "working_point_schema", "SCHEMA_STATUS"]

SCHEMA_STATUS = "draft_pending_WS2_WS3_cosign"
_PRESERVED = ("state", "inventories", "coordinate_reference", "materials", "damage", "grid", "history")
_POINT_REFS = ("state", "inventories", "coordinate_reference", "materials", "damage", "grid", "history", "qualification")


def _text(value: Any) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError("an explicit nonempty string is required")


def _sha(value: Any) -> None:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("an explicit SHA-256 content identity is required")


def _keys(value: Any, keys: set[str]) -> None:
    if type(value) is not dict or set(value) != keys:
        raise ValueError("missing or unknown versioned document fields")


def _reference(value: Any) -> None:
    _keys(value, {"id", "sha256"})
    _text(value["id"])
    _sha(value["sha256"])


def _strings(value: Any, *, nonempty: bool = False) -> None:
    if type(value) is not list or nonempty and not value:
        raise ValueError("an explicit string list is required")
    for item in value:
        _text(item)
    if len(set(value)) != len(value):
        raise ValueError("duplicate declaration")


def _ref_schema() -> dict[str, Any]:
    return {"type": "object", "additionalProperties": False, "required": ["id", "sha256"],
            "properties": {"id": {"type": "string", "minLength": 1},
                           "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"}}}


@dataclass(frozen=True, slots=True)
class Protocol:
    """Complete versioned declaration; experiment-owned sections stay intact.

    Preparation, waveform, switching and sampling each name their own schema.
    Transfers declare preserved quantities without performing a reset/rebind.
    All mutable objects returned to callers are detached from canonical bytes.
    """

    canonical_json: str

    def __post_init__(self) -> None:
        value = _parse_document(self.canonical_json)
        _keys(value, {"schema", "id", "version", "experiment_id", "preparation", "waveform", "switches",
                      "sampling", "control_mode", "phases", "transfers"})
        if value["schema"] != "solarlab.protocol.v1" or type(value["version"]) is not int or value["version"] < 1:
            raise ValueError("unsupported protocol schema/version")
        for name in ("id", "experiment_id", "control_mode"):
            _text(value[name])
        for name in ("preparation", "waveform", "switches", "sampling"):
            if type(value[name]) is not dict or "schema" not in value[name]:
                raise ValueError("each experiment-owned protocol section must name its schema")
            _text(value[name]["schema"])
        _strings(value["phases"], nonempty=True)
        if type(value["transfers"]) is not list:
            raise ValueError("transfers require explicit declarations, including an empty list")
        for transfer in value["transfers"]:
            _keys(transfer, {"from", "to", "working_point", "preserve"})
            if (transfer["from"] not in value["phases"] or transfer["to"] not in value["phases"]
                    or transfer["from"] == transfer["to"]):
                raise ValueError("transfer endpoints must be distinct declared phases")
            _reference(transfer["working_point"])
            _strings(transfer["preserve"], nonempty=True)
            if set(transfer["preserve"]) - set(_PRESERVED):
                raise ValueError("unknown preserved working-point quantity")
        object.__setattr__(self, "canonical_json", _json_document(value))

    @classmethod
    def from_document(cls, value: dict[str, Any]) -> Protocol:
        return cls(_json_document(value))

    def export(self) -> dict[str, Any]:
        return _parse_document(self.canonical_json)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class WorkingPoint:
    """State/reference/material/grid/history identity, not a live state object.

    Physical identity and encoded state are separate supplied identities. Their
    equivalence and phase-left/right compatibility need P05-01b validation.
    Damage and qualification records are explicit even when they declare none.
    """

    canonical_json: str

    def __post_init__(self) -> None:
        value = _parse_document(self.canonical_json)
        _keys(value, {"schema", "id", "engine", "owner_epoch", "physical_identity", "state_encoding", *_POINT_REFS})
        if value["schema"] != "solarlab.working-point.v1" or value["engine"] not in {"legacy", "solarlab"}:
            raise ValueError("unsupported working-point schema/engine")
        for key in ("id", "owner_epoch", "state_encoding"):
            _text(value[key])
        _sha(value["physical_identity"])
        for key in _POINT_REFS:
            _reference(value[key])
        object.__setattr__(self, "canonical_json", _json_document(value))

    @classmethod
    def from_document(cls, value: dict[str, Any]) -> WorkingPoint:
        return cls(_json_document(value))

    def export(self) -> dict[str, Any]:
        return _parse_document(self.canonical_json)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json.encode()).hexdigest()


def protocol_schema() -> dict[str, Any]:
    properties: dict[str, Any] = {"schema": {"const": "solarlab.protocol.v1"},
        "version": {"type": "integer", "minimum": 1}}
    properties.update({key: {"type": "string", "minLength": 1} for key in ("id", "experiment_id", "control_mode")})
    properties.update({key: {"type": "object", "required": ["schema"], "properties": {"schema": {"type": "string", "minLength": 1}}}
                       for key in ("preparation", "waveform", "switches", "sampling")})
    properties["phases"] = {"type": "array", "minItems": 1, "uniqueItems": True, "items": {"type": "string", "minLength": 1}}
    properties["transfers"] = {"type": "array", "items": {"type": "object", "additionalProperties": False,
        "required": ["from", "to", "working_point", "preserve"], "properties": {
            "from": {"type": "string"}, "to": {"type": "string"}, "working_point": _ref_schema(),
            "preserve": {"type": "array", "minItems": 1, "uniqueItems": True, "items": {"enum": list(_PRESERVED)}}}}}
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object",
            "additionalProperties": False, "required": list(properties), "properties": properties}


def working_point_schema() -> dict[str, Any]:
    properties: dict[str, Any] = {"schema": {"const": "solarlab.working-point.v1"}, "engine": {"enum": ["legacy", "solarlab"]},
                                "physical_identity": {"type": "string", "pattern": "^[0-9a-f]{64}$"}}
    properties.update({key: {"type": "string", "minLength": 1} for key in ("id", "owner_epoch", "state_encoding")})
    properties.update({key: _ref_schema() for key in _POINT_REFS})
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object",
            "additionalProperties": False, "required": list(properties), "properties": properties}
