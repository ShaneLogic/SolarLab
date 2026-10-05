"""Versioned, immutable model declarations for configuration preflight.

Declarations do not compile a numerical graph or certify an executor. Consumed
scalar parameters use the existing input/unit definition, with supplied defaults
bound separately by selection. No built-in physics/default table is duplicated.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, fields, is_dataclass
import hashlib
import json
import re
from typing import Any

from solarlab.materials.parameter_schema import parameter_names, parameter_schema
from solarlab.units import normalize_quantity, supported_units

__all__ = [
    "CapabilityContext", "CapabilityRule", "EvidenceRef", "ModelDefinition",
    "ModelRef", "ModelRegistry", "OwnedVariable", "Requirement", "Scope",
    "metadata_digest", "metadata_value",
]

_ID = re.compile(r"[A-Za-z_][A-Za-z0-9_.:-]*\Z")
_KINDS = {"device", "layer", "interface", "contact"}
_SUPPORTS = {"node", "face", "interface_left", "interface_right", "port", "global", "nonlocal"}


def _token(value: str, label: str) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise ValueError(f"{label}: invalid stable identifier {value!r}")
    return value


def _strings(values: Iterable[str], label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label}: expected a sequence, not one string")
    result = tuple(values)
    for value in result:
        _token(value, label)
    if len(set(result)) != len(result):
        raise ValueError(f"{label}: duplicate entries")
    return tuple(sorted(result))


def metadata_value(value: Any) -> Any:
    """Detached JSON data for these immutable declarations, in canonical order."""
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: metadata_value(getattr(value, item.name))
                for item in fields(value) if not item.name.startswith("_")}
    if isinstance(value, tuple):
        return [metadata_value(item) for item in value]
    if value is None or type(value) in {str, int, float, bool}:
        return value
    raise ValueError(f"unsupported metadata value: {type(value).__name__}")


def metadata_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


@dataclass(frozen=True, slots=True, order=True)
class ModelRef:
    id: str
    version: int

    def __post_init__(self) -> None:
        _token(self.id, "model.id")
        if type(self.version) is not int or self.version < 1:
            raise ValueError("model.version must be a positive integer")


@dataclass(frozen=True, slots=True, order=True)
class EvidenceRef:
    id: str
    sha256: str
    kind: str = "scope_reference"

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("evidence requires an explicit source ID")
        if not isinstance(self.sha256, str) or re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None:
            raise ValueError("evidence requires a SHA-256 digest")
        if self.kind not in {"source", "scope_reference", "qualification"}:
            raise ValueError("unknown evidence kind")


@dataclass(frozen=True, slots=True, order=True)
class Scope:
    kind: str
    ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in _KINDS or isinstance(self.ids, (str, bytes)):
            raise ValueError("invalid scope kind or endpoint shape")
        ids = tuple(self.ids)
        if len(ids) != {"device": 0, "layer": 1, "interface": 2, "contact": 1}[self.kind]:
            raise ValueError(f"{self.kind}: incorrect endpoint count")
        if len(set(ids)) != len(ids):
            raise ValueError("scope endpoints must be distinct")
        for value in ids:
            _token(value, "scope endpoint")
        object.__setattr__(self, "ids", ids)  # Orientation is meaningful.

    @property
    def key(self) -> str:
        return "/".join((self.kind, *self.ids))


@dataclass(frozen=True, slots=True)
class OwnedVariable:
    name: str
    unit: str
    support: str
    shape_axes: tuple[str, ...]

    def __post_init__(self) -> None:
        _token(self.name, "owned variable")
        supported_units(self.unit)
        if self.support not in _SUPPORTS:
            raise ValueError("unknown structural support")
        axes = tuple(self.shape_axes)
        if isinstance(self.shape_axes, (str, bytes)):
            raise ValueError("shape_axes requires explicit named axes")
        _strings(axes, "shape axis")
        if self.support in {"node", "face", "nonlocal"} and not axes:
            raise ValueError("distributed support requires explicit shape axes")
        object.__setattr__(self, "shape_axes", axes)


@dataclass(frozen=True, slots=True, order=True)
class Requirement:
    model: ModelRef
    relation: str = "same"

    def __post_init__(self) -> None:
        if not isinstance(self.model, ModelRef) or self.relation not in {"same", "device", "left", "right"}:
            raise ValueError("invalid model requirement or scope relation")


@dataclass(frozen=True, slots=True)
class CapabilityContext:
    dimension: int
    mode: str
    backend: str
    statistics: str
    facts: tuple[str, ...] = ()
    resources: tuple[str, ...] = ()
    available_variables: tuple[str, ...] = ()
    temperature_K: float | None = None

    def __post_init__(self) -> None:
        if type(self.dimension) is not int or self.dimension not in {1, 2}:
            raise ValueError("only explicit 1D/2D metadata contexts are supported")
        if self.mode not in {"equilibrium", "dc", "transient", "ac"}:
            raise ValueError("unknown execution mode")
        _token(self.backend, "backend")
        if self.statistics not in {"maxwell_boltzmann", "fermi_dirac"}:
            raise ValueError("unsupported carrier statistics")
        for name in ("facts", "resources", "available_variables"):
            object.__setattr__(self, name, _strings(getattr(self, name), name))
        if self.temperature_K is not None:
            value = normalize_quantity(self.temperature_K, "K", path="temperature_K")
            if value <= 0:
                raise ValueError("temperature_K must be positive")
            object.__setattr__(self, "temperature_K", value)


@dataclass(frozen=True, slots=True)
class CapabilityRule:
    """Necessary declared constraints, not evidence that an executor exists."""

    id: str
    dimensions: tuple[int, ...]
    modes: tuple[str, ...]
    evidence: tuple[EvidenceRef, ...]
    backends: tuple[str, ...] = ()
    statistics: tuple[str, ...] = ()
    required_facts: tuple[str, ...] = ()
    forbidden_facts: tuple[str, ...] = ()
    required_resources: tuple[str, ...] = ()
    forbidden_models: tuple[ModelRef, ...] = ()
    temperature_K: float | None = None

    def __post_init__(self) -> None:
        _token(self.id, "capability rule")
        dimensions = tuple(self.dimensions)
        if not dimensions or any(type(v) is not int or v not in {1, 2} for v in dimensions) or len(set(dimensions)) != len(dimensions):
            raise ValueError("invalid capability dimensions")
        object.__setattr__(self, "dimensions", tuple(sorted(dimensions)))
        for name in ("modes", "backends", "statistics", "required_facts", "forbidden_facts", "required_resources"):
            object.__setattr__(self, name, _strings(getattr(self, name), name))
        if not self.modes or set(self.modes) - {"equilibrium", "dc", "transient", "ac"}:
            raise ValueError("capability requires known modes")
        if set(self.required_facts) & set(self.forbidden_facts):
            raise ValueError("a capability cannot require and forbid the same fact")
        if set(self.statistics) - {"maxwell_boltzmann", "fermi_dirac"}:
            raise ValueError("unknown capability statistics")
        evidence = tuple(self.evidence)
        if not evidence or any(not isinstance(item, EvidenceRef) for item in evidence) or len(set(evidence)) != len(evidence):
            raise ValueError("capability requires source evidence")
        object.__setattr__(self, "evidence", tuple(sorted(evidence)))
        models = tuple(self.forbidden_models)
        if any(not isinstance(item, ModelRef) for item in models) or len(set(models)) != len(models):
            raise ValueError("forbidden_models requires explicit model versions")
        object.__setattr__(self, "forbidden_models", tuple(sorted(models)))
        if self.temperature_K is not None:
            temperature = normalize_quantity(self.temperature_K, "K")
            if temperature <= 0:
                raise ValueError("capability temperature must be positive")
            object.__setattr__(self, "temperature_K", temperature)

    def problems(self, context: CapabilityContext, models: tuple[ModelRef, ...]) -> tuple[str, ...]:
        issues = []
        if context.dimension not in self.dimensions:
            issues.append("dimension")
        if context.mode not in self.modes:
            issues.append("mode")
        if self.backends and context.backend not in self.backends:
            issues.append("backend")
        if self.statistics and context.statistics not in self.statistics:
            issues.append("statistics")
        issues.extend("missing_fact:" + name for name in self.required_facts if name not in context.facts)
        issues.extend("forbidden_fact:" + name for name in self.forbidden_facts if name in context.facts)
        issues.extend("missing_resource:" + name for name in self.required_resources if name not in context.resources)
        issues.extend("forbidden_model:" + ref.id for ref in self.forbidden_models if ref in models)
        if self.temperature_K is not None and context.temperature_K != self.temperature_K:
            issues.append("temperature")
        return tuple(issues)


@dataclass(frozen=True, slots=True)
class ModelDefinition:
    ref: ModelRef
    label: str
    family: str
    scopes: tuple[str, ...]
    support: str
    parameters: tuple[str, ...]
    reads: tuple[str, ...]
    owns: tuple[OwnedVariable, ...]
    capability: CapabilityRule
    composition: str = "exclusive"
    requires: tuple[Requirement, ...] = ()
    conflicts: tuple[ModelRef, ...] = ()
    derivative_support: str = "not_registered"
    parameter_schema: str = "scalar_layer"

    def __post_init__(self) -> None:
        if not isinstance(self.ref, ModelRef) or not isinstance(self.capability, CapabilityRule):
            raise ValueError("model requires a versioned ref and declared capability")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("model label must be nonempty")
        _token(self.family, "model family")
        for name in ("scopes", "parameters", "reads"):
            object.__setattr__(self, name, _strings(getattr(self, name), name))
        if not self.scopes or set(self.scopes) - _KINDS or self.support not in _SUPPORTS:
            raise ValueError("invalid model scope/support declaration")
        if set(self.parameters) - parameter_names(self.parameter_schema):
            raise ValueError("parameter schema is not available in this scalar preparation slice")
        if self.composition not in {"exclusive", "additive"}:
            raise ValueError("composition must be exclusive or additive")
        if self.derivative_support not in {"not_registered", "declared_analytic", "declared_directional"}:
            raise ValueError("unknown derivative declaration")
        for name, expected in (("owns", OwnedVariable), ("requires", Requirement), ("conflicts", ModelRef)):
            values = tuple(getattr(self, name))
            if any(not isinstance(value, expected) for value in values):
                raise ValueError(f"invalid {name} declaration")
            object.__setattr__(self, name, values)
        if len({value.name for value in self.owns}) != len(self.owns):
            raise ValueError("duplicate owned variable declaration")
        if len(set(self.requires)) != len(self.requires) or len(set(self.conflicts)) != len(self.conflicts):
            raise ValueError("duplicate dependency/conflict declaration")
        object.__setattr__(self, "owns", tuple(sorted(self.owns, key=lambda item: item.name)))
        object.__setattr__(self, "requires", tuple(sorted(self.requires)))
        object.__setattr__(self, "conflicts", tuple(sorted(self.conflicts)))


@dataclass(frozen=True, slots=True)
class ModelRegistry:
    definitions: tuple[ModelDefinition, ...]
    _parameter_schema: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        definitions = tuple(self.definitions)
        if any(not isinstance(item, ModelDefinition) for item in definitions):
            raise ValueError("registry requires model definitions")
        refs = {item.ref for item in definitions}
        if len(refs) != len(definitions):
            raise ValueError("duplicate model ID/version")
        for item in definitions:
            if any(requirement.model not in refs for requirement in item.requires):
                raise ValueError(f"{item.ref.id}: unknown dependency")
            if any(conflict not in refs for conflict in item.conflicts):
                raise ValueError(f"{item.ref.id}: unknown conflict")
        object.__setattr__(self, "definitions", tuple(sorted(definitions, key=lambda item: item.ref)))
        object.__setattr__(self, "_parameter_schema", json.dumps(self._schemas(), sort_keys=True, allow_nan=False))

    def _schemas(self) -> dict[str, Any]:
        return {name: parameter_schema(name) for name in sorted({"scalar_layer", *(item.parameter_schema for item in self.definitions)})}

    def get(self, ref: ModelRef) -> ModelDefinition:
        for item in self.definitions:
            if item.ref == ref:
                return item
        raise ValueError(f"unknown model ID/version: {ref!r}")

    def validate_parameter_schema(self) -> None:
        if json.dumps(self._schemas(), sort_keys=True, allow_nan=False) != self._parameter_schema:
            raise ValueError("parameter schema changed after registry construction; rebuild explicitly")

    def export(self) -> dict[str, Any]:
        return {"schema": "solarlab.model-registry.v1", "models": metadata_value(self.definitions),
                "parameter_schema": json.loads(self._parameter_schema)["scalar_layer"],
                "parameter_schemas": json.loads(self._parameter_schema), "numerical_graph_compiled": False}

    @property
    def content_sha256(self) -> str:
        return metadata_digest(self.export())
