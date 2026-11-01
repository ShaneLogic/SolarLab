"""Immutable records derived from the one supplied layer-template catalog."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import hashlib
import json
import re
from types import MappingProxyType

from solarlab.materials.parameters import LayerValuesInput, Scalar
from solarlab.units import UNIT_SCHEMA_VERSION

__all__ = ["LayerTemplate", "MaterialLibrary", "validated_parameters"]


def validated_parameters(
    items: Iterable[tuple[str, Scalar]], *, complete: bool = False,
) -> tuple[tuple[str, Scalar], ...]:
    """Own the values and reject duplicate/unknown fields before freezing."""
    pairs = tuple(items)
    if any(not isinstance(pair, tuple) or len(pair) != 2 for pair in pairs):
        raise ValueError("parameters must be (name, value) pairs")
    names = [pair[0] for pair in pairs]
    if any(not isinstance(name, str) for name in names) or len(set(names)) != len(names):
        raise ValueError("parameter names must be unique strings")
    return LayerValuesInput.model_validate(dict(pairs)).normalized_items(complete=complete)


@dataclass(frozen=True, slots=True)
class LayerTemplate:
    id: str
    role: str
    description: str
    citation: str
    parameters: tuple[tuple[str, Scalar], ...]

    def __post_init__(self) -> None:
        for name in ("id", "role", "description", "citation"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"template.{name} must be a nonempty string")
        object.__setattr__(self, "parameters", validated_parameters(self.parameters, complete=True))

    @property
    def values(self) -> Mapping[str, Scalar]:
        return MappingProxyType(dict(self.parameters))


@dataclass(frozen=True, slots=True)
class MaterialLibrary:
    """Validated scalar templates, with source bytes and effective content IDs.

    A template includes layer thickness and doping. A resolved material later
    excludes these, matching the material/layer separation in the plan.
    """

    source_id: str
    source_sha256: str
    templates: tuple[LayerTemplate, ...]
    unit_schema: str = field(default=UNIT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("source_id must be a nonempty string")
        if not isinstance(self.source_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", self.source_sha256) is None:
            raise ValueError("source_sha256 must be a SHA-256 digest")
        templates = tuple(self.templates)
        if not templates or any(not isinstance(item, LayerTemplate) for item in templates):
            raise ValueError("library requires validated layer templates")
        if len({item.id for item in templates}) != len(templates):
            raise ValueError("duplicate layer-template ID")
        object.__setattr__(self, "templates", tuple(sorted(templates, key=lambda item: item.id)))

    def get(self, template_id: str) -> LayerTemplate:
        for template in self.templates:
            if template.id == template_id:
                return template
        raise ValueError(f"template: unknown template ID {template_id!r}")

    @property
    def content_sha256(self) -> str:
        payload = {
            "schema": "solarlab.scalar-template-library.v1", "units": self.unit_schema,
            "source_id": self.source_id, "source_sha256": self.source_sha256,
            "templates": [
                {"id": item.id, "role": item.role, "description": item.description,
                 "citation": item.citation, "parameters": dict(item.parameters)}
                for item in self.templates
            ],
        }
        return hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest()
