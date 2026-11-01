"""Three identities of actual prepared configuration, model and source records.

This is a preparation domain, not a cache or a qualified execution manifest.
Original model overrides are required because a Selection cannot reconstruct
whether its models were defaulted or explicitly requested. Execution requires
every upstream content role; this module cannot certify the missing P05 RunSpec
or P07 release-coverage contracts merely by hashing supplied documents.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
import re
from typing import Any

from solarlab.config.behavior import BehaviorContext, _material_switches
from solarlab.config.legacy_execution import R1ExecutionContext
from solarlab.config.model_parameters import (
    device_topology, layer_parameter_source, structured_parameter_source,
)
from solarlab.contracts.layout import Layout
from solarlab.device.resolved import PreparedDevice
from solarlab.materials.resources import ResourceTable
from solarlab.materials.source import SourceDocument
from solarlab.physics.registry import ModelRegistry, Scope, metadata_value
from solarlab.physics.selection import ModelChoice, Selection, select_models, validate_selection
from solarlab.physics.semantic_projection import SemanticProjection, _project_historical_physics, _project_r1_physics
from solarlab.units import UNIT_SCHEMA_VERSION

__all__ = [
    "ModelInputs", "IdentityInputs", "Identity", "BuildSource", "RuntimeEnvironment",
    "ExecutionDescription", "ExecutionEngine", "input_identity", "physics_identity", "execution_identity",
    "CACHE_LAYERS", "ExecutionManifest", "execution_manifest", "verify_execution_manifest",
    "CacheLayerIdentity", "ColdPointBinding", "cold_point_binding",
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


def _uniform_2d_geometry(source: SourceDocument | None, prepared: PreparedDevice) -> dict[str, Any]:
    """Validate an explicit no-grain geometry declaration without a compiler."""
    schema = "solarlab.uniform-no-grain-2d-geometry.v1"
    if type(source) is not SourceDocument or source.id != schema:
        raise ValueError("dimension2 identity requires an explicit uniform lateral geometry SourceDocument")
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        if len({key for key, _ in items}) != len(items):
            raise ValueError("duplicate 2D geometry identity field")
        return dict(items)
    data = json.loads(source.content, object_pairs_hook=pairs)
    fields = {"schema", "lateral_length_m", "Nx", "Ny_per_layer", "lateral_bc",
              "geometry_policy", "electrical_layer_ids"}
    if not isinstance(data, dict) or set(data) != fields or data["schema"] != schema:
        raise ValueError("unaccounted uniform2D geometry identity fields")
    if (type(data["lateral_length_m"]) not in (int, float) or not math.isfinite(data["lateral_length_m"])
            or data["lateral_length_m"] <= 0
            or any(type(data[key]) is not int or not 2 <= data[key] <= 4096 for key in ("Nx", "Ny_per_layer"))
            or data["lateral_bc"] != "neumann"
            or data["geometry_policy"] != "rectangular_half_endpoint_neumann_2d_v1"
            or data["electrical_layer_ids"] != [layer.id for layer in prepared.layers if layer.role != "substrate"]):
        raise ValueError("invalid uniform2D lateral geometry, interval counts or layer ownership")
    return {**data, "lateral_length_m": float(data["lateral_length_m"])}


@dataclass(frozen=True, slots=True)
class IdentityInputs:
    """Bind real prepared/selected data to the original selector inputs.

    Validation uses existing public factories and selection, without modifying
    resolution or executing a physical model. Canonical snapshots are private
    immutable record data, not any of the four reusable numerical cache layers.

    ``effective_behavior`` selects the versioned historical physical rule;
    its complete source/protocol/history record is bound only by execution.
    Without it the original prepared-device projection stays byte-identical.
    """

    prepared: PreparedDevice
    selection: Selection
    registry: ModelRegistry
    model_inputs: ModelInputs
    projections: tuple[SemanticProjection, ...]
    effective_behavior: BehaviorContext | R1ExecutionContext | None = None
    spatial_geometry: SourceDocument | None = None
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
        dimension = self.selection.context.dimension
        if data["grain_boundaries"] or dimension not in (1, 2):
            raise ValueError("full lateral geometry identity is not available in the 1D prepared-device domain")
        geometry = None
        if dimension == 2:
            geometry = _uniform_2d_geometry(self.spatial_geometry, prepared)
            if self.effective_behavior is not None:
                raise ValueError("uniform2D identity does not admit a historical1D effective-behavior context")
        elif self.spatial_geometry is not None:
            raise ValueError("dimension1 identity cannot carry a lateral geometry document")
        behavior = self.effective_behavior
        if behavior is not None:
            _match_behavior(behavior, prepared, self.selection)
        if data["diagnostics"] and (type(behavior) is not BehaviorContext or any(
            item["code"] != "legacy_scaps_ignored_incoherent" for item in data["diagnostics"]
        )):
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
        if geometry is not None:
            # Resolution/interval counts belong to input and execution; the
            # physical projection retains the actual domain and boundary law.
            supplied["spatial_geometry"] = geometry
            physical["spatial_geometry"] = {key: value for key, value in geometry.items()
                                            if key not in {"Nx", "Ny_per_layer"}}
            assert self.spatial_geometry is not None
            binding["spatial_geometry"] = {**_source(self.spatial_geometry),
                                           "content_hex": self.spatial_geometry.content.hex(), "declaration": geometry}
        if behavior is not None:
            document = behavior.export()
            if isinstance(behavior, BehaviorContext):
                physical = _project_historical_physics(physical, document["source_inspection"])
            else:
                ceilings, switches = _material_switches(data["settings"], dict(behavior.environment),
                                                       {item.id: item for item in behavior.sources})
                physical = _project_r1_physics(physical, document["material_behavior"], ceilings=ceilings,
                    switches=switches, controls=document["arguments"]["controls"], reference=json.loads(behavior.reference.content))
            binding["effective_behavior"] = {"sha256": behavior.content_sha256, "context": document}
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


@dataclass(frozen=True, slots=True)
class ExecutionEngine:
    """Explicit implementation choice for the high-level execution facade.

    This only binds identity. It neither imports an engine nor grants execution;
    historical contexts retain their existing preparation-only status.
    There is no default, automatic selection, or failure-fallback mode.
    """

    name: str
    behavior: BehaviorContext | R1ExecutionContext | None = None

    def __post_init__(self) -> None:
        if type(self.name) is not str or self.name not in {"legacy", "solarlab"}:
            raise ValueError("execution engine must be explicitly legacy or solarlab")
        if self.name == "legacy" and type(self.behavior) not in {BehaviorContext, R1ExecutionContext}:
            raise ValueError("legacy execution identity requires an explicit BehaviorContext or complete R1ExecutionContext")
        if self.name == "solarlab" and self.behavior is not None:
            raise ValueError("a legacy BehaviorContext cannot be relabelled as the solarlab engine")

    def export(self) -> dict[str, Any]:
        return {
            "schema": "solarlab.execution-engine.v1",
            "name": self.name,
            "fallback": "forbidden",
            "behavior_context_sha256": self.behavior.content_sha256 if self.behavior else None,
            "behavior_context": self.behavior.export() if self.behavior else None,
        }


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


def _match_behavior(behavior: BehaviorContext | R1ExecutionContext, device: PreparedDevice, selection: Selection) -> None:
    if type(behavior) not in {BehaviorContext, R1ExecutionContext}:
        raise ValueError("actual source-bound BehaviorContext or complete R1ExecutionContext required")
    if behavior.device.content_sha256 != device.content_sha256 or (
        isinstance(behavior, BehaviorContext) and behavior.selection.content_sha256 != selection.content_sha256
    ):
        raise ValueError("behavior context belongs to a different prepared device or selection")


def execution_identity(
    inputs: IdentityInputs, execution: ExecutionDescription, *, engine: ExecutionEngine | None = None,
) -> Identity:
    """Build preparation identity; facade callers must supply ``engine``.

    Omitting it preserves the existing non-executable preparation document.
    A legacy facade must reject such a document, not infer an implementation.
    """
    if type(inputs) is not IdentityInputs or type(execution) is not ExecutionDescription:
        raise ValueError("all actual identity inputs and explicit execution content roles are required")
    if engine is not None:
        if type(engine) is not ExecutionEngine:
            raise ValueError("a typed explicit execution engine is required")
        if inputs.effective_behavior is not None and (engine.behavior is None
                or engine.behavior.content_sha256 != inputs.effective_behavior.content_sha256):
            raise ValueError("execution engine differs from the effective physics behavior")
        if engine.behavior is not None:
            _match_behavior(engine.behavior, inputs.prepared, inputs.selection)
            if inputs.effective_behavior is None:
                inputs = replace(inputs, effective_behavior=engine.behavior)
    payload = {
        "H_input": input_identity(inputs).sha256, "H_physics": physics_identity(inputs).sha256,
        "effective_binding": json.loads(inputs._execution_binding),
        "documents": {name: _source(getattr(execution, name)) for name in
                      ("protocol", "preparation", "initial_state", "grid", "numeric_controls")},
        "implementation": {"build_source_commit": execution.implementation.build_source_commit,
                           "files": [_source(item) for item in execution.implementation.files]},
        "environment": metadata_value(execution.environment),
    }
    if engine is not None:
        payload["engine"] = engine.export()
    return _identity("H_execution", payload, execution.current_repository_head)


_EXECUTION_ROLES = ("protocol", "preparation", "initial_state", "grid", "numeric_controls")


def _manifest_bytes(value: Any) -> bytes:
    # Complete implementation content needs a larger transport envelope than
    # the original identity summaries. Their 8 MiB domain remains unchanged.
    result = json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(result) > 2**26:
        raise ValueError("execution manifest exceeds 64 MiB")
    return result


def _content_record(source: SourceDocument) -> dict[str, Any]:
    return {**_source(source), "content_hex": source.content.hex()}


def _manifest_source(value: Any) -> SourceDocument:
    if not isinstance(value, dict) or set(value) != {"id", "sha256", "bytes", "content_hex"}:
        raise ValueError("manifest requires complete source bytes and their identity")
    if type(value["content_hex"]) is not str:
        raise ValueError("manifest source content must be hexadecimal bytes")
    try:
        source = SourceDocument(value["id"], bytes.fromhex(value["content_hex"]))
    except (TypeError, ValueError) as error:
        raise ValueError("invalid manifest source content") from error
    if _content_record(source) != value or type(value["bytes"]) is not int:
        raise ValueError("manifest source bytes do not match their recorded identity")
    return source


@dataclass(frozen=True, slots=True)
class ExecutionManifest:
    """Portable complete content binding, with no execution/cache admission.

    All five upstream documents and implementation files retain their exact
    bytes, including state words and signed zero. Their domain completeness
    remains the upstream producer's contract. Consumers must call
    ``verify_execution_manifest`` with independently supplied current inputs,
    engine, content and environment; parsing alone is not verification.
    Maintenance HEAD is recorded separately from the immutable build commit.
    """

    canonical_json: bytes
    current_repository_head: str
    sha256: str = field(init=False)
    execution_sha256: str = field(init=False)
    can_execute: bool = field(default=False, init=False)
    cache_eligible: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        _commit(self.current_repository_head)
        if type(self.canonical_json) is not bytes or len(self.canonical_json) > 2**26:
            raise ValueError("manifest requires bounded immutable JSON bytes")
        value = json.loads(self.canonical_json)
        if (not isinstance(value, dict) or set(value) != {
            "schema", "execution_identity", "content_roles", "implementation_content",
        } or value["schema"] != "solarlab.execution-manifest.v1" or _manifest_bytes(value) != self.canonical_json):
            raise ValueError("manifest is not in its canonical versioned domain")
        document = value["execution_identity"]
        if not isinstance(document, dict) or set(document) != {"schema", "zero_policy", "payload"}:
            raise ValueError("manifest requires a complete execution identity document")
        identity = Identity("H_execution", _json_bytes(document))
        payload = document["payload"]
        if not isinstance(payload, dict) or set(payload) != {
            "H_input", "H_physics", "effective_binding", "documents", "implementation", "environment", "engine",
        }:
            raise ValueError("manifest requires the complete explicit-engine execution identity")
        if (not isinstance(payload["engine"], dict) or payload["engine"].get("name") not in {"legacy", "solarlab"}
                or payload["engine"].get("fallback") != "forbidden"):
            raise ValueError("manifest requires an explicit engine without fallback")
        if not isinstance(payload["implementation"], dict) or set(payload["implementation"]) != {"build_source_commit", "files"}:
            raise ValueError("manifest requires the explicit implementation source binding")
        roles = value["content_roles"]
        if not isinstance(roles, dict) or set(roles) != set(_EXECUTION_ROLES):
            raise ValueError("manifest is missing a complete upstream content role")
        documents = {name: _source(_manifest_source(record)) for name, record in roles.items()}
        implementation = value["implementation_content"]
        if not isinstance(implementation, list):
            raise ValueError("manifest requires implementation content")
        build = BuildSource(payload["implementation"]["build_source_commit"],
                            tuple(_manifest_source(record) for record in implementation))
        if (payload["documents"] != documents or payload["implementation"] != {
            "build_source_commit": build.build_source_commit, "files": [_source(source) for source in build.files],
        }):
            raise ValueError("manifest content differs from the bound execution identity")
        object.__setattr__(self, "sha256", hashlib.sha256(self.canonical_json).hexdigest())
        object.__setattr__(self, "execution_sha256", identity.sha256)

    def export(self) -> dict[str, Any]:
        return {"sha256": self.sha256, "document": json.loads(self.canonical_json),
                "current_repository_head": self.current_repository_head,
                "can_execute": self.can_execute, "cache_eligible": self.cache_eligible}


def execution_manifest(
    inputs: IdentityInputs, execution: ExecutionDescription, *, engine: ExecutionEngine,
) -> ExecutionManifest:
    """Bind every supplied role; never infer engine, initial state or protocol."""
    if type(engine) is not ExecutionEngine:
        raise ValueError("manifest requires a typed explicit execution engine")
    identity = execution_identity(inputs, execution, engine=engine)
    value = {"schema": "solarlab.execution-manifest.v1", "execution_identity": json.loads(identity.canonical_json),
             "content_roles": {name: _content_record(getattr(execution, name)) for name in _EXECUTION_ROLES},
             "implementation_content": [_content_record(source) for source in execution.implementation.files]}
    return ExecutionManifest(_manifest_bytes(value), execution.current_repository_head)


def verify_execution_manifest(
    manifest: ExecutionManifest, inputs: IdentityInputs, execution: ExecutionDescription, *,
    engine: ExecutionEngine, environment: RuntimeEnvironment, expected_sha256: str,
) -> Identity:
    """Check an externally pinned manifest at a real consumer boundary.

    The caller supplies its observed runtime environment, not one decoded from
    the manifest. Matching bytes do not establish P05 state qualification or
    P07 release coverage, and the returned identity remains cache-ineligible.
    """
    if type(manifest) is not ExecutionManifest or manifest.sha256 != expected_sha256:
        raise ValueError("manifest differs from the expected frozen identity")
    if type(environment) is not RuntimeEnvironment or execution.environment != environment:
        raise ValueError("consumer runtime environment differs from the manifest request")
    if type(engine) is not ExecutionEngine or manifest.export()["document"]["execution_identity"]["payload"]["engine"] != engine.export():
        raise ValueError("consumer engine/context differs from the manifest; fallback is forbidden")
    expected = execution_manifest(inputs, execution, engine=engine)
    if manifest.canonical_json != expected.canonical_json:
        raise ValueError("consumer inputs, protocol, state, grid, controls or implementation differ from the manifest")
    return execution_identity(inputs, execution, engine=engine)


@dataclass(frozen=True, slots=True)
class CacheLayerIdentity:
    """One distinct metadata domain; matching keys never grant reuse."""
    layer: str
    canonical_json: bytes | None
    sha256: str | None = field(init=False)
    cache_eligible: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.layer not in CACHE_LAYERS:
            raise ValueError("unknown cache identity layer")
        if self.canonical_json is None:
            if self.layer != "numeric_factorization":
                raise ValueError("only an unconstructed factorization may have no identity")
            object.__setattr__(self, "sha256", None)
            return
        if type(self.canonical_json) is not bytes:
            raise ValueError("cache identity requires immutable content")
        data = json.loads(self.canonical_json)
        if (not isinstance(data, dict) or set(data) != {"schema", "layer", "payload"}
                or data["schema"] != "solarlab.cache-layer-identity.v1" or data["layer"] != self.layer
                or not isinstance(data["payload"], dict) or _json_bytes(data) != self.canonical_json):
            raise ValueError("cache layer identity has the wrong canonical domain")
        object.__setattr__(self, "sha256", hashlib.sha256(self.canonical_json).hexdigest())

    def export(self) -> dict[str, Any]:
        return {"layer": self.layer, "sha256": self.sha256,
                "document": json.loads(self.canonical_json) if self.canonical_json is not None else None,
                "cache_eligible": self.cache_eligible}


@dataclass(frozen=True, slots=True)
class ColdPointBinding:
    """Fresh immutable per-point metadata, never a solver binding or LU."""
    point_id: str
    input_sha256: str
    physics_sha256: str
    execution_sha256: str
    layers: tuple[CacheLayerIdentity, ...]
    can_execute: bool = field(default=False, init=False)
    numerical_reuse: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        _name(self.point_id)
        for value in (self.input_sha256, self.physics_sha256, self.execution_sha256):
            if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("point binding requires actual complete identity digests")
        layers = tuple(self.layers)
        if any(type(item) is not CacheLayerIdentity for item in layers) or tuple(item.layer for item in layers) != CACHE_LAYERS:
            raise ValueError("point binding requires all four distinct layers in their declared order")
        object.__setattr__(self, "layers", layers)

    def layer(self, name: str) -> CacheLayerIdentity:
        for item in self.layers:
            if item.layer == name:
                return item
        raise ValueError("unknown cache identity layer")

    def export(self) -> dict[str, Any]:
        return {"schema": "solarlab.cold-point-binding.v1", "point_id": self.point_id,
                "H_input": self.input_sha256, "H_physics": self.physics_sha256, "H_execution": self.execution_sha256,
                "layers": [item.export() for item in self.layers], "can_execute": self.can_execute,
                "numerical_reuse": self.numerical_reuse}

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(_json_bytes(self.export())).hexdigest()


def cold_point_binding(
    inputs: IdentityInputs, execution: ExecutionDescription, manifest: ExecutionManifest, *,
    point_id: str, layout: Layout, topology_plan: SourceDocument, engine: ExecutionEngine,
    environment: RuntimeEnvironment, expected_sha256: str, numeric_matrix: SourceDocument | None = None,
) -> ColdPointBinding:
    """Separate structure, numeric parameters, state/history and factorization.

    WS3 supplies the actual Layout and complete compiled-plan document; neither
    is fabricated from a device hash. Layout's own constructor revalidates its
    identity. Only immutable snapshots are retained here. The first version
    always cold-builds: no key, even an exact match, enables rebind or LU reuse.
    A missing matrix has no factorization identity, rather than a made-up LU.
    """
    verified = verify_execution_manifest(manifest, inputs, execution, engine=engine,
                                         environment=environment, expected_sha256=expected_sha256)
    if type(layout) is not Layout:
        raise ValueError("point binding requires the actual public Layout contract")
    actual = Layout(layout.supports, layout.variables, layout.equations)
    if actual.identity != layout.identity or actual.size != layout.size or dict(actual.offsets) != dict(layout.offsets):
        raise ValueError("layout identity/offsets differ from their actual declarations")
    # This is exactly the existing Layout producer's declaration representation;
    # no new physical-coordinate, unit or layout identity algorithm is invented.
    declaration = repr((actual.supports, actual.variables, actual.equations))
    payload = verified.export()["document"]["payload"]

    def layer(name: str, value: dict[str, Any] | None) -> CacheLayerIdentity:
        return CacheLayerIdentity(name, _json_bytes({"schema": "solarlab.cache-layer-identity.v1",
                                  "layer": name, "payload": value}) if value is not None else None)

    models = [{"id": item.id, "scope": metadata_value(item.scope),
               "definition": metadata_value(inputs.registry.get(item.model)),
               "owned_variables": metadata_value(item.owned_variables), "reads": list(item.read_variables)}
              for item in inputs.selection.instances]
    topology = layer("topology_plan", {"layout_sha256": actual.identity, "layout_declarations": declaration,
        "topology_plan": _source(topology_plan), "device_topology": metadata_value(inputs.selection.topology),
        "models": models, "evaluation_order": list(inputs.selection.evaluation_order), "implementation": payload["implementation"]})
    parameters = layer("numeric_parameter_binding", {"topology_sha256": topology.sha256,
        "H_physics": payload["H_physics"], "effective_binding_sha256": hashlib.sha256(_json_bytes(payload["effective_binding"])).hexdigest(),
        "actual_grid": _source(execution.grid), "environment": payload["environment"],
        "engine_sha256": hashlib.sha256(_json_bytes(engine.export())).hexdigest()})
    state = layer("working_point_initial_state", {"parameters_sha256": parameters.sha256,
        "H_execution": verified.sha256, "protocol": _source(execution.protocol),
        "preparation": _source(execution.preparation), "initial_state": _source(execution.initial_state)})
    factorization = layer("numeric_factorization", None if numeric_matrix is None else {
        "working_point_sha256": state.sha256, "numeric_controls": _source(execution.numeric_controls),
        "actual_matrix": _source(numeric_matrix)})
    return ColdPointBinding(point_id, payload["H_input"], payload["H_physics"], verified.sha256,
                            (topology, parameters, state, factorization))
