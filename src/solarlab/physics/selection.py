"""Resolve explicit model selections without a solver or a Device dependency."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re

from solarlab.materials.library import validated_parameters
from solarlab.materials.parameters import Scalar
from solarlab.physics.registry import (
    CapabilityContext, EvidenceRef, ModelRef, ModelRegistry, OwnedVariable,
    Scope, _strings, _token, metadata_digest, metadata_value,
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

    def scopes(self) -> tuple[Scope, ...]:
        return (Scope("device"), *(Scope("layer", (value,)) for value in self.electrical_layer_ids),
                *(Scope("interface", pair) for pair in zip(self.electrical_layer_ids, self.electrical_layer_ids[1:])),
                *(Scope("contact", (contact,)) for contact, _ in self.contacts))

    def validate(self, scope: Scope) -> None:
        if not isinstance(scope, Scope) or scope not in self.scopes():
            raise ValueError(f"scope is not in the directed electrical topology: {scope!r}")


@dataclass(frozen=True, slots=True)
class ParameterSource:
    id: str
    values: tuple[tuple[str, Scalar], ...]
    evidence: EvidenceRef

    def __post_init__(self) -> None:
        _token(self.id, "parameter source")
        if not isinstance(self.evidence, EvidenceRef):
            raise ValueError("parameter source needs evidence")
        object.__setattr__(self, "values", validated_parameters(self.values))

    @property
    def content_sha256(self) -> str:
        return metadata_digest(metadata_value(self))


@dataclass(frozen=True, slots=True)
class ModelChoice:
    id: str
    model: ModelRef
    parameters: tuple[tuple[str, Scalar], ...] = ()
    field_bindings: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _token(self.id, "local model instance ID")
        if not isinstance(self.model, ModelRef):
            raise ValueError("model choice requires an explicit ID/version")
        object.__setattr__(self, "parameters", validated_parameters(self.parameters))
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

    def __post_init__(self) -> None:
        _token(self.local_id, "selected model instance")
        if not isinstance(self.model, ModelRef) or not isinstance(self.scope, Scope):
            raise ValueError("selected model requires explicit ref and scope")
        object.__setattr__(self, "parameters", validated_parameters(self.parameters))
        if self.parameter_source_sha256 is not None and re.fullmatch(r"[0-9a-f]{64}", self.parameter_source_sha256) is None:
            raise ValueError("invalid parameter source identity")
        overridden = _strings(self.overridden_parameters, "overridden parameter")
        if set(overridden) - {name for name, _ in self.parameters}:
            raise ValueError("override origin names an absent parameter")
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


def _choices(values: Sequence[ModelChoice]) -> tuple[ModelChoice, ...]:
    if isinstance(values, (str, bytes)):
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
    if scope.kind not in definition.scopes:
        raise ValueError(f"{scope.key}/{choice.id}: model does not support this scope")
    explicit = dict(choice.parameters)
    if set(explicit) - set(definition.parameters):
        raise ValueError(f"{choice.id}: unconsumed model parameters")
    inherited = dict(source.values) if source is not None else {}
    values = {name: explicit[name] if name in explicit else inherited.get(name)
              for name in definition.parameters}
    missing = [name for name in definition.parameters if name not in explicit and name not in inherited]
    if missing:
        raise ValueError(f"{choice.id}: missing consumed parameters {missing}")
    parameters = validated_parameters(tuple(values.items()))
    bindings = dict(choice.field_bindings)
    names = {item.name for item in definition.owns} | set(definition.reads)
    if set(bindings) - names:
        raise ValueError(f"{choice.id}: unused field binding")
    owned = tuple((bindings.get(item.name, f"{scope.key}/{choice.id}:{item.name}"), item)
                  for item in definition.owns)
    local_owned = {item.name: target for target, item in owned}
    reads = tuple(bindings.get(name, local_owned.get(name, name)) for name in definition.reads)
    return SelectedModel(choice.id, choice.model, scope, parameters,
                         source.content_sha256 if source is not None else None,
                         tuple(sorted(explicit)), owned, reads)


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
    for instance in selection.instances:
        selection.topology.validate(instance.scope)
        definition = registry.get(instance.model)
        if instance.scope.kind not in definition.scopes:
            raise ValueError(f"{instance.id}: model does not support this scope")
        parameters = validated_parameters(instance.parameters)
        if tuple(name for name, _ in parameters) != definition.parameters:
            raise ValueError(f"{instance.id}: consumed parameter names do not match declaration")
        if instance.parameter_source_sha256 is None and set(definition.parameters) != set(instance.overridden_parameters):
            raise ValueError(f"{instance.id}: inherited parameters require a source identity")
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
        selected.extend(_bind(choice, scope, registry, parameter_sources.get(scope)) for choice in choices)
    instances = tuple(sorted(selected, key=lambda item: item.id))
    selection = Selection(registry.content_sha256, topology, context, instances,
                          _order(instances, registry), _declared_problems(instances, registry, context))
    validate_selection(selection, registry)
    return selection
