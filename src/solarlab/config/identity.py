"""Three identities of actual prepared configuration, model and source records.

This is a preparation domain, not a cache or a qualified execution manifest.
Original model overrides are required because a Selection cannot reconstruct
whether its models were defaulted or explicitly requested. Execution requires
every upstream content role; this module cannot certify the missing P05 RunSpec
or P07 release-coverage contracts merely by hashing supplied documents.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from typing import Any

from solarlab.config.model_parameters import (
    device_topology, layer_parameter_source, structured_parameter_source,
)
from solarlab.device.resolved import PreparedDevice
from solarlab.materials.resources import ResourceTable
from solarlab.materials.source import SourceDocument
from solarlab.physics.registry import ModelRegistry, Scope, metadata_value
from solarlab.physics.selection import ModelChoice, Selection, select_models, validate_selection
from solarlab.physics.semantic_projection import SemanticProjection
from solarlab.units import UNIT_SCHEMA_VERSION

__all__ = [
    "ModelInputs", "IdentityInputs", "Identity", "BuildSource", "RuntimeEnvironment",
    "ExecutionDescription", "input_identity", "physics_identity", "execution_identity",
    "CACHE_LAYERS",
]

CACHE_LAYERS = ("topology_plan", "numeric_parameter_binding", "working_point_initial_state", "numeric_factorization")
_ZERO_POLICY = "normalized_configuration_plus_zero; source_bytes_exact"
_RESOLVED_KEYS = {
    "schema", "unit_schema", "id", "settings", "layers", "declaration", "interfaces", "contacts",
    "grain_boundaries", "electrical_grid", "resource_bindings", "diagnostics", "capability_gaps",
    "can_execute", "default_catalog", "input_sources",
}
_RESOLVED_OPTIONAL = {"tunnelling_channels", "metastable_measurement_protocol_bindings"}


def _name(value: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError("an explicit nonempty identity label is required")
    return value


def _json_value(value: Any, depth: int = 0) -> Any:
    if depth > 40:
        raise ValueError("identity document nesting exceeds the preparation domain")
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float and math.isfinite(value):
        return 0.0 if value == 0 else value
    if isinstance(value, Mapping) and all(type(key) is str for key in value):
        return {key: _json_value(item, depth + 1) for key, item in value.items()}
    if type(value) in {list, tuple}:
        return [_json_value(item, depth + 1) for item in value]
    raise ValueError(f"unsupported/nonfinite identity value: {type(value).__name__}")


def _json_bytes(value: Any) -> bytes:
    result = json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(result) > 2**23:
        raise ValueError("identity document exceeds 8 MiB")
    return result


def _source(source: SourceDocument) -> dict[str, Any]:
    if type(source) is not SourceDocument:
        raise ValueError("actual immutable SourceDocument bytes are required")
    return {"id": source.id, "sha256": source.sha256, "bytes": len(source.content)}


def _choices(values: Sequence[ModelChoice]) -> tuple[ModelChoice, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("model choices must be an explicit sequence")
    result = tuple(values)
    if any(type(item) is not ModelChoice for item in result) or len({item.id for item in result}) != len(result):
        raise ValueError("model inputs require unique actual ModelChoice records")
    return tuple(sorted(result, key=lambda item: item.id))


def _choice_document(choice: ModelChoice) -> dict[str, Any]:
    result = {"id": choice.id, "model": metadata_value(choice.model),
              "parameters": dict(choice.parameters), "parameter_schema": choice.parameter_schema,
              "field_bindings": dict(choice.field_bindings)}
    if choice.structured is not None:
        data = choice.structured.data
        result["structured"] = {
            "schema": data.schema.id, "document_version": data.schema.document_version,
            "owner_id": data.owner_id, "local_id": data.local_id, "scope": metadata_value(data.scope),
            "presence": data.presence, "normalized_supplied": json.loads(data.normalized_json),
        }
    return result


@dataclass(frozen=True, slots=True)
class ModelInputs:
    """Original selector arguments; an empty override family is an explicit clear.

    Defaults are software policy, excluded from H_input but bound by execution.
    Both arguments are required, including an explicitly empty overrides tuple.
    No presence is inferred from the already resolved Selection.
    """

    defaults: tuple[tuple[str, tuple[ModelChoice, ...]], ...]
    overrides: tuple[tuple[Scope, tuple[tuple[str, tuple[ModelChoice, ...]], ...]], ...]

    def __post_init__(self) -> None:
        defaults = tuple((kind, _choices(choices)) for kind, choices in self.defaults)
        if len({kind for kind, _ in defaults}) != len(defaults) or any(kind not in {"device", "layer", "interface", "contact"} for kind, _ in defaults):
            raise ValueError("invalid/duplicate model default scope kind")
        overrides = []
        for scope, values in self.overrides:
            if not isinstance(scope, Scope):
                raise ValueError("model overrides require actual Scope records")
            families = tuple((_name(name), _choices(choices)) for name, choices in values)
            if len({name for name, _ in families}) != len(families):
                raise ValueError("duplicate model override family")
            overrides.append((scope, tuple(sorted(families))))
        if len({scope for scope, _ in overrides}) != len(overrides):
            raise ValueError("duplicate model override scope")
        object.__setattr__(self, "defaults", tuple(sorted(defaults)))
        object.__setattr__(self, "overrides", tuple(sorted(overrides, key=lambda item: item[0].key)))

    @classmethod
    def from_mappings(
        cls, *, defaults: Mapping[str, Sequence[ModelChoice]],
        overrides: Mapping[Scope, Mapping[str, Sequence[ModelChoice]]],
    ) -> ModelInputs:
        if not isinstance(defaults, Mapping) or not isinstance(overrides, Mapping):
            raise ValueError("original selector default and override mappings are required")
        if any(not isinstance(values, Mapping) for values in overrides.values()):
            raise ValueError("null is not an inherited or cleared model family")
        return cls(tuple((kind, tuple(values)) for kind, values in defaults.items()),
                   tuple((scope, tuple((name, tuple(values)) for name, values in families.items()))
                         for scope, families in overrides.items()))

    def default_mapping(self) -> dict[str, tuple[ModelChoice, ...]]:
        return dict(self.defaults)

    def override_mapping(self) -> dict[Scope, dict[str, tuple[ModelChoice, ...]]]:
        return {scope: dict(values) for scope, values in self.overrides}

    def supplied_document(self) -> list[dict[str, Any]]:
        return [{"scope": metadata_value(scope), "families": {
            family: [_choice_document(choice) for choice in values] for family, values in families}}
            for scope, families in self.overrides]

    def defaults_document(self) -> dict[str, Any]:
        return {kind: [_choice_document(choice) for choice in choices] for kind, choices in self.defaults}


@dataclass(frozen=True, slots=True)
class IdentityInputs:
    """Bind real prepared/selected data to the original selector inputs.

    Validation uses existing public factories and selection, without modifying
    resolution or executing a physical model. Canonical snapshots are private
    immutable record data, not any of the four reusable numerical cache layers.
    """

    prepared: PreparedDevice
    selection: Selection
    registry: ModelRegistry
    model_inputs: ModelInputs
    projections: tuple[SemanticProjection, ...]
    _input_document: bytes = field(init=False, repr=False)
    _physical_document: bytes = field(init=False, repr=False)
    _execution_binding: bytes = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if (type(self.prepared) is not PreparedDevice or type(self.selection) is not Selection
                or type(self.registry) is not ModelRegistry or type(self.model_inputs) is not ModelInputs):
            raise ValueError("identity requires actual prepared device, selection, registry and original model inputs")
        # Rebuild only declaration data, not equations or a numerical system.
        prepared = PreparedDevice(self.prepared.input_json, self.prepared.defaults,
                                  self.prepared.resources, self.prepared.sources)
        if (prepared.to_mapping() != self.prepared.to_mapping()
                or [layer.to_mapping() for layer in prepared.layers] != [layer.to_mapping() for layer in self.prepared.layers]):
            raise ValueError("prepared declaration no longer matches its authoritative input")
        topology = device_topology(prepared)
        if self.selection.topology != topology:
            raise ValueError("selection topology/device identity mismatch")
        if validate_selection(self.selection, self.registry):
            raise ValueError("selection has unresolved declared capability requirements")
        sources = {Scope("layer", (layer.id,)): layer_parameter_source(layer)
                   for layer in prepared.layers if layer.id in topology.electrical_layer_ids}
        structured = {}
        for instance in self.selection.instances:
            if instance.structured is None:
                if instance.parameter_schema != "full_layer" or instance.scope.kind != "layer":
                    raise ValueError("scalar identity currently requires actual full_layer PreparedDevice bindings")
            else:
                structured_data = instance.structured.data
                source = structured_parameter_source(prepared, structured_data.schema.family,
                                                     structured_data.owner_id, instance_id=structured_data.local_id)
                if source.structured is None or source.structured.content_sha256 != instance.structured.content_sha256:
                    raise ValueError("structured selection belongs to a different prepared declaration")
                structured[instance.id] = source
        replay = select_models(self.registry, topology, self.selection.context,
                               self.model_inputs.default_mapping(), sources, self.model_inputs.override_mapping(),
                               instance_parameter_sources=structured)
        if replay != self.selection:
            raise ValueError("selection does not match the supplied device and original model inputs")
        projections = tuple(self.projections)
        if any(type(item) is not SemanticProjection for item in projections):
            raise ValueError("model-owned semantic projections are required")
        if (len({item.model for item in projections}) != len(projections)
                or {item.model for item in projections} != {item.model for item in self.selection.instances}):
            raise ValueError("semantic projections must cover exactly the selected model versions")
        object.__setattr__(self, "projections", tuple(sorted(projections, key=lambda item: item.model)))
        data = prepared.to_mapping()
        if data["schema"] != "solarlab.resolved-device-preparation.v1" or set(data) - _RESOLVED_OPTIONAL != _RESOLVED_KEYS:
            raise ValueError("unaccounted resolved-device schema/fields require a new identity projection")
        if self.selection.context.dimension != 1 or data["grain_boundaries"]:
            raise ValueError("full lateral geometry identity is not available in the 1D prepared-device domain")
        if data["diagnostics"]:
            raise ValueError("historical effective-behavior diagnostics require a supplied behavior context")
        temperature = self.selection.context.temperature_K
        if temperature is not None and temperature != data["settings"]["T"]:
            raise ValueError("selection temperature differs from the prepared device")
        projected: list[dict[str, Any]] = []
        claimed: dict[str, set[str]] = {layer.id: set() for layer in prepared.layers}
        by_model = {item.model: item for item in projections}
        raw_models = []
        for instance in sorted(self.selection.instances, key=lambda item: item.id):
            rule = by_model[instance.model]
            document, names = rule.project(self.registry.get(instance.model), instance)
            projected.append(document)
            if names:
                claimed[instance.scope.ids[0]].update(names)
            raw_models.append({"instance_id": instance.id, "parameters": dict(instance.parameters),
                               "structured_effective": document["effective_values"] if instance.structured is not None else None})
        physical = self._device_physics(prepared, data, claimed)
        context = metadata_value(self.selection.context)
        context.pop("backend")  # Numerical implementation belongs to execution.
        physical.update(models=projected, model_context=context,
                        topology=metadata_value(topology), evaluation_order=list(self.selection.evaluation_order))
        supplied = {"device": prepared.to_input().normalized_data(),
                    "model_overrides": self.model_inputs.supplied_document(), "unit_schema": UNIT_SCHEMA_VERSION}
        binding = {
            "registry_sha256": self.registry.content_sha256,
            "software_model_defaults": self.model_inputs.defaults_document(),
            "selection_backend": self.selection.context.backend, "models": raw_models,
            "all_effective_layer_parameters": [{"id": layer.id, "parameters": dict(layer.parameters)} for layer in prepared.layers],
            "electrical_grid_recipe": data["electrical_grid"],
            "jv_solver_policy": data["settings"]["jv_solver_policy"],
            "metastable_preparation": [{"layer_id": layer.id, "preparation": (
                layer.to_mapping().get("metastable_preparation"))} for layer in prepared.layers],
        }
        object.__setattr__(self, "_input_document", _json_bytes(supplied))
        object.__setattr__(self, "_physical_document", _json_bytes(physical))
        object.__setattr__(self, "_execution_binding", _json_bytes(binding))

    def _device_physics(
        self, prepared: PreparedDevice, data: dict[str, Any], claimed: dict[str, set[str]],
    ) -> dict[str, Any]:
        layers = []
        for layer in prepared.layers:
            complete = dict(layer.parameters)
            if not claimed[layer.id] <= set(complete):
                raise ValueError("model projection claims a parameter absent from its actual layer")
            layers.append({
                "id": layer.id, "role": layer.role, "thickness_m": layer.thickness,
                "material_id": layer.material.id,
                "remaining_effective_parameters": {name: value for name, value in complete.items() if name not in claimed[layer.id]},
                "defect_model": layer.defect_model, "defect_schema_version": layer.defect_schema_version,
                "bulk_defects": [item.to_mapping() for item in layer.defects],
                "bulk_trap_distribution": layer.bulk_trap_distribution,
                "cigs_graded_optics": layer.material.cigs_graded_optics,
                "cigs_model_source": layer.cigs_source_binding,
                "metastable_document": layer.metastable_document,
            })
        used = {(item["name"], item["kind"]) for item in data["resource_bindings"]}
        all_resources = {item.name: item for item in prepared.resources.tables}
        for name in self.selection.context.resources:
            if name not in all_resources:
                raise ValueError("declared model resource has no actual supplied table: " + name)
            used.add((name, all_resources[name].kind))
        resources = []
        for name, kind in sorted(used):
            table = prepared.resources.get(name, kind)
            actual = ResourceTable(table.name, table.kind, table.source)
            if actual.content_sha256 != table.content_sha256:
                raise ValueError("resource values no longer match their authoritative source bytes")
            resources.append({"name": name, "kind": kind, "source": _source(table.source),
                              "content_sha256": table.content_sha256, "shape": table.shape,
                              "columns": table.columns, "units": table.units})
        settings = dict(data["settings"])
        settings.pop("jv_solver_policy")
        return {"domain": "prepared-device-1d.v1", "unit_schema": UNIT_SCHEMA_VERSION,
                "device_id": data["id"], "settings": settings, "layers": layers,
                "interfaces": data["interfaces"], "contacts": data["contacts"],
                "tunnelling_channels": data.get("tunnelling_channels"),
                "resolver_constants": dict(prepared.defaults.constants), "used_resources": resources}


@dataclass(frozen=True, slots=True)
class Identity:
    kind: str
    canonical_json: bytes
    current_repository_head: str | None = None
    sha256: str = field(init=False)
    status: str = field(default="prepared_pending_dependencies", init=False)
    cache_eligible: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.kind not in {"H_input", "H_physics", "H_execution"} or type(self.canonical_json) is not bytes:
            raise ValueError("invalid identity record")
        document = json.loads(self.canonical_json)
        if document.get("schema") != "solarlab." + self.kind + ".v1" or _json_bytes(document) != self.canonical_json:
            raise ValueError("identity document is not in its canonical versioned domain")
        if self.current_repository_head is not None:
            _commit(self.current_repository_head)
        object.__setattr__(self, "sha256", hashlib.sha256(self.canonical_json).hexdigest())

    def export(self) -> dict[str, Any]:
        return {"kind": self.kind, "sha256": self.sha256, "document": json.loads(self.canonical_json),
                "current_repository_head": self.current_repository_head,
                "status": self.status, "cache_eligible": self.cache_eligible}


def _commit(value: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value) is None:
        raise ValueError("an immutable explicit build-source commit is required, not a branch or guessed HEAD")
    return value


@dataclass(frozen=True, slots=True)
class BuildSource:
    """Supplied immutable implementation content; not a published ReleaseBundle.

    Commit/content association and complete release coverage need the upstream
    build authority. Current working-tree HEAD is never substituted here.
    """

    build_source_commit: str
    files: tuple[SourceDocument, ...]

    def __post_init__(self) -> None:
        _commit(self.build_source_commit)
        files = tuple(self.files)
        if not files or any(type(item) is not SourceDocument for item in files) or len({item.id for item in files}) != len(files):
            raise ValueError("implementation requires unique actual source-content records")
        object.__setattr__(self, "files", tuple(sorted(files, key=lambda item: item.id)))


def _version_pairs(values: tuple[tuple[str, str], ...]) -> tuple[tuple[str, str], ...]:
    pairs = tuple(tuple(pair) for pair in values)
    if any(len(pair) != 2 for pair in pairs) or len({pair[0] for pair in pairs}) != len(pairs):
        raise ValueError("environment entries must be unique named pairs")
    return tuple(sorted((_name(name), _name(value)) for name, value in pairs))


@dataclass(frozen=True, slots=True)
class RuntimeEnvironment:
    python_implementation: str
    python_version: str
    system: str
    machine: str
    packages: tuple[tuple[str, str], ...]
    settings: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        for value in (self.python_implementation, self.python_version, self.system, self.machine):
            _name(value)
        packages = _version_pairs(self.packages)
        if not packages:
            raise ValueError("an explicit dependency environment is required")
        object.__setattr__(self, "packages", packages)
        object.__setattr__(self, "settings", _version_pairs(self.settings))


@dataclass(frozen=True, slots=True)
class ExecutionDescription:
    """Required complete upstream content roles, with no invented defaults.

    These SourceDocuments retain exact bytes (including raw initial-state
    words). P05 must validate protocol/state completeness and P07 must attest
    release coverage; this preparation does not pretend either adapter exists.
    The schema/serialization of each upstream document is its producer's job.
    """

    protocol: SourceDocument
    preparation: SourceDocument
    initial_state: SourceDocument
    grid: SourceDocument
    numeric_controls: SourceDocument
    implementation: BuildSource
    environment: RuntimeEnvironment
    current_repository_head: str

    def __post_init__(self) -> None:
        for name in ("protocol", "preparation", "initial_state", "grid", "numeric_controls"):
            _source(getattr(self, name))
        if type(self.implementation) is not BuildSource or type(self.environment) is not RuntimeEnvironment:
            raise ValueError("actual implementation and environment records are required")
        _commit(self.current_repository_head)


def _identity(kind: str, payload: Any, head: str | None = None) -> Identity:
    return Identity(kind, _json_bytes({"schema": "solarlab." + kind + ".v1", "zero_policy": _ZERO_POLICY, "payload": payload}), head)


def input_identity(inputs: IdentityInputs) -> Identity:
    if type(inputs) is not IdentityInputs:
        raise ValueError("input identity requires the actual prepared identity inputs")
    return _identity("H_input", json.loads(inputs._input_document))


def physics_identity(inputs: IdentityInputs) -> Identity:
    if type(inputs) is not IdentityInputs:
        raise ValueError("physics identity requires the actual prepared identity inputs")
    return _identity("H_physics", json.loads(inputs._physical_document))


def execution_identity(inputs: IdentityInputs, execution: ExecutionDescription) -> Identity:
    if type(inputs) is not IdentityInputs or type(execution) is not ExecutionDescription:
        raise ValueError("all actual identity inputs and explicit execution content roles are required")
    return _identity("H_execution", {
        "H_input": input_identity(inputs).sha256, "H_physics": physics_identity(inputs).sha256,
        "effective_binding": json.loads(inputs._execution_binding),
        "documents": {name: _source(getattr(execution, name)) for name in
                      ("protocol", "preparation", "initial_state", "grid", "numeric_controls")},
        "implementation": {"build_source_commit": execution.implementation.build_source_commit,
                           "files": [_source(item) for item in execution.implementation.files]},
        "environment": metadata_value(execution.environment),
    }, execution.current_repository_head)
