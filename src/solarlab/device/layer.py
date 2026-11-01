"""Immutable resolved data for the explicitly supported scalar-template slice."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
import re
from types import MappingProxyType

from solarlab.materials.library import validated_parameters
from solarlab.materials.parameters import LayerValuesInput, Scalar, parameter_units
from solarlab.units import UNIT_SCHEMA_VERSION

__all__ = ["ResolvedLayer", "ResolvedMaterial", "ValueOrigin"]


@dataclass(frozen=True, slots=True)
class ValueOrigin:
    parameter: str
    kind: str
    source_id: str

    def __post_init__(self) -> None:
        if self.kind not in {"template", "override"}:
            raise ValueError("origin kind must be template or override")
        if any(not isinstance(value, str) or not value.strip() for value in (self.parameter, self.source_id)):
            raise ValueError("origin requires parameter and source ID")


@dataclass(frozen=True, slots=True)
class ResolvedMaterial:
    template_id: str
    parameters: tuple[tuple[str, Scalar], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.template_id, str) or not self.template_id:
            raise ValueError("material template_id must be nonempty")
        items = validated_parameters(self.parameters)
        if {name for name, _ in items} & {"thickness", "N_A", "N_D"}:
            raise ValueError("thickness and doping belong to the layer, not the material")
        expected = set(LayerValuesInput.model_fields) - {"thickness", "N_A", "N_D"}
        if {name for name, _ in items} != expected:
            raise ValueError("material scalar parameters are incomplete")
        object.__setattr__(self, "parameters", items)

    @property
    def values(self) -> Mapping[str, Scalar]:
        return MappingProxyType(dict(self.parameters))


@dataclass(frozen=True, slots=True)
class ResolvedLayer:
    id: str
    name: str
    role: str
    thickness: float
    N_A: float
    N_D: float
    material: ResolvedMaterial
    source_id: str
    source_sha256: str
    library_content_sha256: str
    origins: tuple[ValueOrigin, ...]
    units: tuple[tuple[str, str], ...] = field(default_factory=parameter_units, init=False)
    unit_schema: str = field(default=UNIT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.:-]*", self.id) is None:
            raise ValueError("layer requires a stable ID")
        for name in ("name", "role", "source_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"layer.{name} must be nonempty")
        for digest in (self.source_sha256, self.library_content_sha256):
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError("layer source identities must be SHA-256 digests")
        if not isinstance(self.material, ResolvedMaterial):
            raise ValueError("layer requires a resolved material")
        values = dict(LayerValuesInput.model_validate({
            "thickness": self.thickness, "N_A": self.N_A, "N_D": self.N_D,
        }).normalized_items())
        for name, value in values.items():
            object.__setattr__(self, name, value)
        origins = tuple(self.origins)
        if any(not isinstance(origin, ValueOrigin) for origin in origins):
            raise ValueError("layer requires validated value origins")
        names = [origin.parameter for origin in origins]
        if len(set(names)) != len(names) or set(names) != set(LayerValuesInput.model_fields):
            raise ValueError("each resolved field requires exactly one origin")
        object.__setattr__(self, "origins", tuple(sorted(origins, key=lambda item: item.parameter)))

    def to_mapping(self) -> dict[str, object]:
        """Return a detached representation; never expose a mutable backing map."""
        return {
            "schema": "solarlab.resolved-scalar-layer.v1", "unit_schema": self.unit_schema,
            "id": self.id, "name": self.name, "role": self.role,
            "thickness": self.thickness, "N_A": self.N_A, "N_D": self.N_D,
            "material": {"template_id": self.material.template_id, "parameters": dict(self.material.parameters)},
            "units": dict(self.units), "source_id": self.source_id,
            "source_sha256": self.source_sha256, "library_content_sha256": self.library_content_sha256,
            "origins": [{"parameter": item.parameter, "kind": item.kind, "source_id": item.source_id} for item in self.origins],
        }

    @property
    def content_sha256(self) -> str:
        """Resolved document identity, not the later versioned physics hash."""
        return hashlib.sha256(json.dumps(
            self.to_mapping(), sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest()
