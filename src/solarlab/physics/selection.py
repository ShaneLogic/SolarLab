"""Resolve explicit model selections without a solver or a Device dependency."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import re
from typing import Any

from solarlab.materials.parameter_schema import parameter_values
from solarlab.materials.parameters import Scalar
from solarlab.physics.registry import (
    CapabilityContext, EvidenceRef, ModelRef, ModelRegistry, OwnedVariable,
    Scope, StructuredParameters, _strings, _token, metadata_digest, metadata_value,
)

__all__ = [
    "ModelChoice", "ParameterSource", "SelectedModel", "Selection", "Topology",
    "select_models", "validate_selection",
]


@dataclass(frozen=True, slots=True)
class Topology:
    """Explicit electrical layer IDs; optical substrates are not inferred here."""

    electrical_layer_ids: tuple[str, ...]
    contacts: tuple[tuple[str, str], ...] = ()
    interface_ids: tuple[tuple[str, str, str], ...] = field(default=(), metadata={"omit_empty": True})
    contact_sides: tuple[tuple[str, str], ...] = field(default=(), metadata={"omit_empty": True})
    device_id: str | None = field(default=None, metadata={"omit_none": True})

    def __post_init__(self) -> None:
        ids = tuple(self.electrical_layer_ids)
        _strings(ids, "electrical layer IDs")
        if isinstance(self.electrical_layer_ids, (str, bytes)) or not ids:
            raise ValueError("topology requires explicit electrical layer IDs")
        if isinstance(self.contacts, (str, bytes)):
            raise ValueError("contacts require explicit endpoint pairs")
        entries = tuple(self.contacts)
        if any(isinstance(item, (str, bytes)) for item in entries):
            raise ValueError("contacts require explicit endpoint pairs")
        contacts = tuple(tuple(item) for item in entries)
        if any(len(item) != 2 for item in contacts):
            raise ValueError("contact binding must be (contact ID, endpoint layer ID)")
        if len({item[0] for item in contacts}) != len(contacts):
            raise ValueError("duplicate contact ID")
        for contact, layer in contacts:
            _token(contact, "contact ID")
            if layer not in {ids[0], ids[-1]}:
                raise ValueError("contact must bind an outer electrical layer")
        object.__setattr__(self, "electrical_layer_ids", ids)
        object.__setattr__(self, "contacts", tuple(sorted(contacts)))
        if self.device_id is not None:
            _token(self.device_id, "device ID")
        if isinstance(self.interface_ids, (str, bytes)) or isinstance(self.contact_sides, (str, bytes)):
            raise ValueError("topology IDs require explicit tuples")
        if any(isinstance(item, (str, bytes)) for item in (*self.interface_ids, *self.contact_sides)):
            raise ValueError("topology bindings require explicit endpoint tuples")
        interfaces = tuple(tuple(item) for item in self.interface_ids)
        if any(len(item) != 3 for item in interfaces) or len({item[0] for item in interfaces}) != len(interfaces):
            raise ValueError("interfaces require unique (ID, left, right) triples")
        pairs = set(zip(ids, ids[1:]))
        if len({item[1:] for item in interfaces}) != len(interfaces) or any(item[1:] not in pairs for item in interfaces):
            raise ValueError("interface IDs must bind directed adjacent electrical layers")
        for identifier, _, _ in interfaces:
            _token(identifier, "interface ID")
        sides = tuple(tuple(item) for item in self.contact_sides)
        if any(len(item) != 2 for item in sides) or len({item[0] for item in sides}) != len(sides):
            raise ValueError("contact sides require unique (ID, side) pairs")
        if len({item[1] for item in sides}) != len(sides):
            raise ValueError("only one contact may bind each side")
        contact_layers = {pair[0]: pair[1] for pair in contacts}
        for identifier, side in sides:
            if side not in {"left", "right"} or contact_layers.get(identifier) != ids[0 if side == "left" else -1]:
                raise ValueError("contact side and endpoint disagree")
        object.__setattr__(self, "interface_ids", tuple(sorted(interfaces)))
        object.__setattr__(self, "contact_sides", tuple(sorted(sides)))

    def scopes(self) -> tuple[Scope, ...]:
        return (Scope("device"), *(Scope("layer", (value,)) for value in self.electrical_layer_ids),
                *(Scope("interface", pair) for pair in zip(self.electrical_layer_ids, self.electrical_layer_ids[1:])),
                *(Scope("contact", (contact,)) for contact, _ in self.contacts))

    def validate(self, scope: Scope) -> None:
        if not isinstance(scope, Scope) or scope not in self.scopes():
            raise ValueError(f"scope is not in the directed electrical topology: {scope!r}")

    def validate_parameters(self, parameters: StructuredParameters) -> None:
        data = parameters.data
        self.validate(data.scope)
        if self.device_id != data.device_id:
            raise ValueError("structured parameters require their explicit device identity")
        if data.scope.kind == "layer":
            valid = data.scope.ids == (data.owner_id,) and data.orientation == data.scope.ids
        elif data.scope.kind == "interface":
            valid = (data.owner_id, *data.scope.ids) in self.interface_ids and data.orientation == data.scope.ids
        elif data.scope.kind == "contact":
            valid = (data.scope.ids == (data.owner_id,) and data.owner_id in dict(self.contacts)
                     and data.owner_id in dict(self.contact_sides)
                     and data.orientation == (dict(self.contact_sides)[data.owner_id], dict(self.contacts)[data.owner_id]))
        else:
            valid = data.owner_id == self.device_id and data.orientation == ()
        if not valid:
            raise ValueError("structured owner ID or orientation disagrees with topology")


@dataclass(frozen=True, slots=True)
class ParameterSource:
    id: str
    values: tuple[tuple[str, Scalar], ...]
    evidence: EvidenceRef
    parameter_schema: str = "scalar_layer"
    structured: StructuredParameters | None = field(default=None, metadata={"omit_none": True})

    def __post_init__(self) -> None:
        _token(self.id, "parameter source")
        if not isinstance(self.evidence, EvidenceRef):
            raise ValueError("parameter source needs evidence")
        if self.structured is None:
            object.__setattr__(self, "values", parameter_values(self.values, self.parameter_schema))
        else:
            _structured_binding(self.structured, self.parameter_schema, self.values)
            if self.evidence != self.structured.data.source:
                raise ValueError("structured parameter source evidence mismatch")
            object.__setattr__(self, "values", ())

    @property
    def content_sha256(self) -> str:
        return metadata_digest(metadata_value(self))


@dataclass(frozen=True, slots=True)
class ModelChoice:
    id: str
    model: ModelRef
    parameters: tuple[tuple[str, Scalar], ...] = ()
    field_bindings: tuple[tuple[str, str], ...] = ()
    parameter_schema: str = "scalar_layer"
    structured: StructuredParameters | None = field(default=None, metadata={"omit_none": True})

    def __post_init__(self) -> None:
        _token(self.id, "local model instance ID")
        if not isinstance(self.model, ModelRef):
            raise ValueError("model choice requires an explicit ID/version")
        if self.parameter_schema in {"scalar_layer", "full_layer"}:
            if self.structured is not None:
                raise ValueError("scalar choices cannot carry a structured document")
            object.__setattr__(self, "parameters", parameter_values(self.parameters, self.parameter_schema))
        else:
            _token(self.parameter_schema, "structured parameter schema")
            if tuple(self.parameters):
                raise ValueError("structured choices cannot also contain scalar overrides")
            if self.structured is not None:
                _structured_binding(self.structured, self.parameter_schema, ())
            object.__setattr__(self, "parameters", ())
        if isinstance(self.field_bindings, (str, bytes)):
            raise ValueError("field bindings require explicit pairs")
        entries = tuple(self.field_bindings)
        if any(isinstance(item, (str, bytes)) for item in entries):
            raise ValueError("field bindings require explicit pairs")
        bindings = tuple(tuple(item) for item in entries)
        if any(len(item) != 2 for item in bindings) or len({item[0] for item in bindings}) != len(bindings):
            raise ValueError("field bindings must have unique local names")
        for name, target in bindings:
            _token(name, "local field binding")
            if not isinstance(target, str) or not target.strip():
                raise ValueError("field binding requires an explicit target ID")
        object.__setattr__(self, "field_bindings", tuple(sorted(bindings)))


@dataclass(frozen=True, slots=True)
class SelectedModel:
    local_id: str
    model: ModelRef
    scope: Scope
    parameters: tuple[tuple[str, Scalar], ...]
    parameter_source_sha256: str | None
    overridden_parameters: tuple[str, ...]
    owned_variables: tuple[tuple[str, OwnedVariable], ...]
    read_variables: tuple[str, ...]
    parameter_schema: str = "scalar_layer"
    structured: StructuredParameters | None = field(default=None, metadata={"omit_none": True})

    def __post_init__(self) -> None:
        _token(self.local_id, "selected model instance")
        if not isinstance(self.model, ModelRef) or not isinstance(self.scope, Scope):
            raise ValueError("selected model requires explicit ref and scope")
        if self.structured is None:
            object.__setattr__(self, "parameters", parameter_values(self.parameters, self.parameter_schema))
            parameter_names = {name for name, _ in self.parameters}
        else:
            _structured_binding(self.structured, self.parameter_schema, self.parameters)
            if self.structured.data.scope != self.scope or self.structured.data.local_id != self.local_id:
                raise ValueError("structured instance identity does not match selection")
            if self.parameter_source_sha256 != self.structured.content_sha256:
                raise ValueError("structured selected source identity is stale")
            object.__setattr__(self, "parameters", ())
            parameter_names = set(self.structured.data.schema.parameter_names)
        if self.parameter_source_sha256 is not None and re.fullmatch(r"[0-9a-f]{64}", self.parameter_source_sha256) is None:
            raise ValueError("invalid parameter source identity")
        overridden = _strings(self.overridden_parameters, "overridden parameter")
        if set(overridden) - parameter_names:
            raise ValueError("override origin names an absent parameter")
        if self.structured is not None and set(overridden) not in (set(), parameter_names):
            raise ValueError("structured overrides replace one complete validated document")
        object.__setattr__(self, "overridden_parameters", overridden)
        owned = tuple(self.owned_variables)
        if any(not isinstance(item, tuple) or len(item) != 2 or not isinstance(item[0], str)
               or not item[0] or not isinstance(item[1], OwnedVariable) for item in owned):
            raise ValueError("invalid variable ownership binding")
        object.__setattr__(self, "owned_variables", owned)
        reads = tuple(self.read_variables)
        if isinstance(self.read_variables, (str, bytes)) or any(not isinstance(item, str) or not item for item in reads):
            raise ValueError("invalid physical variable references")
        object.__setattr__(self, "read_variables", reads)

    @property
    def id(self) -> str:
        return self.scope.key + "/" + self.local_id


@dataclass(frozen=True, slots=True)
class Selection:
    registry_sha256: str
    topology: Topology
    context: CapabilityContext
    instances: tuple[SelectedModel, ...]
    evaluation_order: tuple[str, ...]
    declared_problems: tuple[str, ...]

    def __post_init__(self) -> None:
        if re.fullmatch(r"[0-9a-f]{64}", self.registry_sha256) is None:
            raise ValueError("selection requires a registry identity")
        if not isinstance(self.topology, Topology) or not isinstance(self.context, CapabilityContext):
            raise ValueError("selection requires topology and context")
        instances = tuple(self.instances)
        if any(not isinstance(item, SelectedModel) for item in instances):
            raise ValueError("invalid selected model instance")
        ids = {item.id for item in instances}
        order = tuple(self.evaluation_order)
        if len(ids) != len(instances) or len(order) != len(ids) or set(order) != ids:
            raise ValueError("selection order must cover every unique model instance")
        for item in instances:
            self.topology.validate(item.scope)
            if item.structured is not None:
                self.topology.validate_parameters(item.structured)
        problems = tuple(self.declared_problems)
        if any(not isinstance(item, str) or not item for item in problems):
            raise ValueError("invalid declared compatibility problem")
        object.__setattr__(self, "instances", instances)
        object.__setattr__(self, "evaluation_order", order)
        object.__setattr__(self, "declared_problems", problems)

    @property
    def content_sha256(self) -> str:
        # Parameters are included: this is not permission to reuse numeric
        # caches when D=0, thickness or another physical input changes.
        return metadata_digest(metadata_value(self))

    def export(self) -> dict[str, Any]:
        """Ordered metadata for future input/UI consumers; never run permission."""
        schemas = {item.structured.data.schema.id: item.structured.data.schema.export()
                   for item in self.instances if item.structured is not None}
        return {"schema": "solarlab.model-selection.v1", **metadata_value(self),
                "parameter_schemas": {name: schemas[name] for name in sorted(schemas)},
                "status": "prepared_pending_G2", "can_execute": False}


def _structured_binding(parameters: StructuredParameters, schema: str, scalars: object) -> None:
    if not isinstance(parameters, StructuredParameters) or parameters.data.schema.id != schema:
        raise ValueError("structured parameter schema binding mismatch")
    if scalars != ():
        raise ValueError("structured parameters cannot be mixed with scalar parameters")


def _choices(values: Sequence[ModelChoice]) -> tuple[ModelChoice, ...]:
    if values is None or isinstance(values, (str, bytes)):
        raise ValueError("model choices require a sequence")
    result = tuple(values)
    if any(not isinstance(value, ModelChoice) for value in result):
        raise ValueError("invalid model choice")
    if len({value.id for value in result}) != len(result):
        raise ValueError("duplicate model instance ID within a scope")
    return result


def _bind(
    choice: ModelChoice, scope: Scope, registry: ModelRegistry,
    source: ParameterSource | None,
) -> SelectedModel:
    definition = registry.get(choice.model)
    if choice.parameter_schema != definition.parameter_schema or (source is not None and source.parameter_schema != definition.parameter_schema):
        raise ValueError(f"{choice.id}: parameter schema binding mismatch")
    if scope.kind not in definition.scopes:
        raise ValueError(f"{scope.key}/{choice.id}: model does not support this scope")
    structured = None
    source_sha = source.content_sha256 if source is not None else None
    if definition.structured_schema is not None:
        structured = choice.structured if choice.structured is not None else (source.structured if source is not None else None)
        if structured is None or structured.data.schema != definition.structured_schema:
            raise ValueError(f"{choice.id}: missing or mismatched closed structured parameters")
        parameters = ()
        explicit_names = definition.parameters if choice.structured is not None else ()
        source_sha = structured.content_sha256
    else:
        explicit = dict(choice.parameters)
        if set(explicit) - set(definition.parameters):
            raise ValueError(f"{choice.id}: unconsumed model parameters")
        inherited = dict(source.values) if source is not None else {}
        values = {name: explicit[name] if name in explicit else inherited.get(name)
                  for name in definition.parameters}
        missing = [name for name in definition.parameters if name not in explicit and name not in inherited]
        if missing:
            raise ValueError(f"{choice.id}: missing consumed parameters {missing}")
        parameters = parameter_values(tuple(values.items()), definition.parameter_schema)
        explicit_names = tuple(sorted(explicit))
    bindings = dict(choice.field_bindings)
    names = {item.name for item in definition.owns} | set(definition.reads)
    if set(bindings) - names:
        raise ValueError(f"{choice.id}: unused field binding")
    owned = tuple((bindings.get(item.name, f"{scope.key}/{choice.id}:{item.name}"), item)
                  for item in definition.owns)
    local_owned = {item.name: target for target, item in owned}
    reads = tuple(bindings.get(name, local_owned.get(name, name)) for name in definition.reads)
    return SelectedModel(choice.id, choice.model, scope, parameters,
                         source_sha, explicit_names, owned, reads, definition.parameter_schema, structured)


def _dependency_scope(scope: Scope, relation: str) -> Scope:
    if relation == "same":
        return scope
    if relation == "device":
        return Scope("device")
    if scope.kind != "interface":
        raise ValueError("left/right model dependencies require a directed interface scope")
    return Scope("layer", (scope.ids[0 if relation == "left" else 1],))


def _order(instances: tuple[SelectedModel, ...], registry: ModelRegistry) -> tuple[str, ...]:
    edges: dict[str, tuple[str, ...]] = {}
    for instance in instances:
        dependencies = []
        for requirement in registry.get(instance.model).requires:
            target = _dependency_scope(instance.scope, requirement.relation)
            matches = [item.id for item in instances if item.scope == target and item.model == requirement.model]
            if not matches:
                raise ValueError(f"{instance.id}: missing dependency {requirement.model} at {target.key}")
            dependencies.extend(matches)
        edges[instance.id] = tuple(sorted(set(dependencies)))
    done: set[str] = set()
    active: set[str] = set()
    ordered: list[str] = []

    def visit(name: str) -> None:
        if name in active:
            raise ValueError(f"cyclic model dependency at {name}")
        if name in done:
            return
        active.add(name)
        for dependency in edges[name]:
            visit(dependency)
        active.remove(name)
        done.add(name)
        ordered.append(name)

    for name in sorted(edges):
        visit(name)
    return tuple(ordered)


def _declared_problems(
    instances: tuple[SelectedModel, ...], registry: ModelRegistry,
    context: CapabilityContext,
) -> tuple[str, ...]:
    refs = tuple(sorted({instance.model for instance in instances}))
    return tuple(sorted(
        f"{instance.id}:{reason}" for instance in instances
        for reason in registry.get(instance.model).capability.problems(context, refs)
    ))


def validate_selection(selection: Selection, registry: ModelRegistry) -> tuple[str, ...]:
    """Revalidate resolved metadata, including ordinary dataclass copy/edits.

    Physical state reads need an available owner, but are not execution-DAG
    dependencies: simultaneously coupled variables may read one another.
    Only explicit model Requirements constrain the evaluation order.
    """
    if not isinstance(selection, Selection) or not isinstance(registry, ModelRegistry):
        raise ValueError("validation requires a selection and model registry")
    registry.validate_parameter_schema()
    if selection.registry_sha256 != registry.content_sha256:
        raise ValueError("selection/model registry identity mismatch")
    owners = {name: "external" for name in selection.context.available_variables}
    families: dict[tuple[Scope, str], list[SelectedModel]] = {}
    structured_sources: set[str] = set()
    for instance in selection.instances:
        selection.topology.validate(instance.scope)
        definition = registry.get(instance.model)
        if instance.scope.kind not in definition.scopes:
            raise ValueError(f"{instance.id}: model does not support this scope")
        if instance.parameter_schema != definition.parameter_schema:
            raise ValueError(f"{instance.id}: parameter schema binding mismatch")
        if definition.structured_schema is None:
            parameters = parameter_values(instance.parameters, definition.parameter_schema)
            if instance.structured is not None or tuple(name for name, _ in parameters) != definition.parameters:
                raise ValueError(f"{instance.id}: consumed parameter names do not match declaration")
            if instance.parameter_source_sha256 is None and set(definition.parameters) != set(instance.overridden_parameters):
                raise ValueError(f"{instance.id}: inherited parameters require a source identity")
        else:
            structured = instance.structured
            if structured is None or structured.data.schema != definition.structured_schema:
                raise ValueError(f"{instance.id}: structured schema identity mismatch")
            selection.topology.validate_parameters(structured)
            if structured.data.instance_id != instance.id or instance.parameter_source_sha256 != structured.content_sha256:
                raise ValueError(f"{instance.id}: structured parameter instance/source identity mismatch")
            structured_sources.add(structured.data.source.sha256)
        declared_owns = tuple(variable for _, variable in instance.owned_variables)
        if declared_owns != definition.owns:
            raise ValueError(f"{instance.id}: owned variable specs do not match declaration")
        if len(instance.read_variables) != len(definition.reads):
            raise ValueError(f"{instance.id}: read bindings do not match declaration")
        local_owners = {variable.name: target for target, variable in instance.owned_variables}
        for name, target in zip(definition.reads, instance.read_variables):
            if name in local_owners and local_owners[name] != target:
                raise ValueError(f"{instance.id}: local read and ownership binding disagree")
        for target, _ in instance.owned_variables:
            if target in owners:
                raise ValueError(f"duplicate ownership of physical variable {target}")
            owners[target] = instance.id
        families.setdefault((instance.scope, definition.family), []).append(instance)
    if len(structured_sources) > 1:
        raise ValueError("structured selections cannot mix distinct prepared-device snapshots; rebind all instances")
    for group in families.values():
        if len(group) > 1 and any(registry.get(item.model).composition == "exclusive" for item in group):
            raise ValueError(f"{group[0].scope.key}: exclusive model family has multiple instances")
    for instance in selection.instances:
        definition = registry.get(instance.model)
        if set(instance.read_variables) - set(owners):
            raise ValueError(f"{instance.id}: missing required physical variable")
        if any(other.scope == instance.scope and other.model in definition.conflicts for other in selection.instances):
            raise ValueError(f"{instance.id}: conflicting model selection")
    if selection.evaluation_order != _order(selection.instances, registry):
        raise ValueError("selection evaluation order does not match declared dependencies")
    problems = _declared_problems(selection.instances, registry, selection.context)
    if selection.declared_problems != problems:
        raise ValueError("selection compatibility diagnostics are stale; resolve again")
    return problems


def select_models(
    registry: ModelRegistry, topology: Topology, context: CapabilityContext,
    defaults: Mapping[str, Sequence[ModelChoice]],
    parameter_sources: Mapping[Scope, ParameterSource],
    overrides: Mapping[Scope, Mapping[str, Sequence[ModelChoice]]] | None = None,
    *, instance_parameter_sources: Mapping[str, ParameterSource] | None = None,
) -> Selection:
    """Inherit kind defaults, then replace explicitly overridden families.

    An empty family tuple explicitly clears it. Null is not an inheritance or
    disable shortcut. Interface endpoint orientation comes from Topology.
    """
    if not isinstance(registry, ModelRegistry) or not isinstance(topology, Topology) or not isinstance(context, CapabilityContext):
        raise ValueError("selection requires a registry, topology and context")
    registry.validate_parameter_schema()
    patches = {} if overrides is None else overrides
    if not all(isinstance(value, Mapping) for value in (defaults, parameter_sources, patches)):
        raise ValueError("selection defaults, parameter sources and overrides must be mappings")
    if any(not isinstance(value, Mapping) for value in patches.values()):
        raise ValueError("scope overrides must map families to choices")
    if set(defaults) - {"device", "layer", "interface", "contact"}:
        raise ValueError("unknown default scope kind")
    for scope in (*patches, *parameter_sources):
        topology.validate(scope)
    for source in parameter_sources.values():
        if not isinstance(source, ParameterSource):
            raise ValueError("parameter sources must be immutable validated records")
    per_instance = {} if instance_parameter_sources is None else instance_parameter_sources
    if not isinstance(per_instance, Mapping):
        raise ValueError("instance parameter sources must be a mapping")
    for identifier, source in per_instance.items():
        if not isinstance(source, ParameterSource) or source.structured is None or identifier != source.structured.data.instance_id:
            raise ValueError("structured sources must be keyed by their canonical instance IDs")
        topology.validate_parameters(source.structured)
    selected: list[SelectedModel] = []
    for scope in topology.scopes():
        families: dict[str, tuple[ModelChoice, ...]] = {}
        for choice in _choices(defaults.get(scope.kind, ())):
            family = registry.get(choice.model).family
            families[family] = (*families.get(family, ()), choice)
        for family, values in patches.get(scope, {}).items():
            _token(family, "override family")
            choices = _choices(values)
            if family not in {item.family for item in registry.definitions}:
                raise ValueError("unknown model family override")
            if any(registry.get(choice.model).family != family for choice in choices):
                raise ValueError("model choice does not belong to the overridden family")
            families[family] = choices
        choices = _choices(tuple(choice for family in sorted(families) for choice in families[family]))
        selected.extend(_bind(choice, scope, registry,
                              per_instance.get(scope.key + "/" + choice.id) if registry.get(choice.model).structured_schema is not None
                              else parameter_sources.get(scope)) for choice in choices)
    instances = tuple(sorted(selected, key=lambda item: item.id))
    selection = Selection(registry.content_sha256, topology, context, instances,
                          _order(instances, registry), _declared_problems(instances, registry, context))
    validate_selection(selection, registry)
    return selection
