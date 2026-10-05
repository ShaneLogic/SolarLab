"""Named starting selections, without the old mode-ceiling semantics.

Migration supplies effective choices and historical disables with their source
evidence. This module neither imports old drivers nor guesses effective physics
from mode flags. Unresolved driver/environment behavior remains explicit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from solarlab.physics.registry import (
    CapabilityContext, EvidenceRef, ModelRef, ModelRegistry, Scope,
    _strings, _token, metadata_digest, metadata_value,
)
from solarlab.physics.selection import (
    ModelChoice, ParameterSource, Selection, Topology, _choices, select_models,
)

__all__ = ["HistoricalDisable", "NamedPreset", "PresetSelection", "select_preset"]


@dataclass(frozen=True, slots=True)
class HistoricalDisable:
    """One source-observed suppression, with explicit model-version mapping."""

    id: str
    models: tuple[ModelRef, ...]
    evidence: EvidenceRef

    def __post_init__(self) -> None:
        _token(self.id, "historical disabled mechanism")
        models = tuple(self.models)
        if not models or any(not isinstance(model, ModelRef) for model in models):
            raise ValueError("historical disable requires explicit model versions")
        if len(set(models)) != len(models) or not isinstance(self.evidence, EvidenceRef):
            raise ValueError("invalid historical disable mapping/evidence")
        object.__setattr__(self, "models", tuple(sorted(models)))


@dataclass(frozen=True, slots=True)
class NamedPreset:
    id: str
    version: int
    label: str
    defaults: tuple[tuple[str, tuple[ModelChoice, ...]], ...]
    evidence: tuple[EvidenceRef, ...]
    historical_disables: tuple[HistoricalDisable, ...] = ()
    unresolved_behavior: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        ModelRef(self.id, self.version)
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("preset requires a display label")
        defaults = tuple(self.defaults)
        if any(not isinstance(item, tuple) or len(item) != 2 for item in defaults):
            raise ValueError("preset defaults require (scope kind, choices) pairs")
        if len({kind for kind, _ in defaults}) != len(defaults):
            raise ValueError("duplicate preset scope kind")
        if any(kind not in {"device", "layer", "interface", "contact"} for kind, _ in defaults):
            raise ValueError("unknown preset scope kind")
        object.__setattr__(self, "defaults", tuple(sorted(
            (kind, tuple(sorted(_choices(choices), key=lambda item: item.id)))
            for kind, choices in defaults
        )))
        evidence = tuple(self.evidence)
        if not evidence or any(not isinstance(item, EvidenceRef) for item in evidence):
            raise ValueError("preset requires evidence for its supplied defaults")
        object.__setattr__(self, "evidence", tuple(sorted(set(evidence))))
        disabled = tuple(self.historical_disables)
        if any(not isinstance(item, HistoricalDisable) for item in disabled):
            raise ValueError("invalid historical disable record")
        if len({item.id for item in disabled}) != len(disabled):
            raise ValueError("duplicate historical mechanism ID")
        active = {choice.model for _, choices in self.defaults for choice in choices}
        if any(active.intersection(item.models) for item in disabled):
            raise ValueError("starting preset enables a historically disabled mechanism")
        object.__setattr__(self, "historical_disables", tuple(sorted(disabled, key=lambda item: item.id)))
        object.__setattr__(self, "unresolved_behavior", _strings(self.unresolved_behavior, "unresolved legacy behavior"))

    @property
    def content_sha256(self) -> str:
        return metadata_digest(metadata_value(self))


@dataclass(frozen=True, slots=True)
class PresetSelection:
    preset: NamedPreset
    selection: Selection

    def __post_init__(self) -> None:
        if not isinstance(self.preset, NamedPreset) or not isinstance(self.selection, Selection):
            raise ValueError("preset selection requires validated metadata")

    @property
    def reenabled_historical_mechanisms(self) -> tuple[str, ...]:
        active = {item.model for item in self.selection.instances}
        return tuple(item.id for item in self.preset.historical_disables if active.intersection(item.models))

    @property
    def content_sha256(self) -> str:
        return metadata_digest(metadata_value(self))


def select_preset(
    preset: NamedPreset, registry: ModelRegistry, topology: Topology,
    context: CapabilityContext, parameter_sources: Mapping[Scope, ParameterSource],
    overrides: Mapping[Scope, Mapping[str, Sequence[ModelChoice]]] | None = None,
    *, instance_parameter_sources: Mapping[str, ParameterSource] | None = None,
) -> PresetSelection:
    """Explicit overrides may depart from history; that departure is retained."""
    if not isinstance(preset, NamedPreset):
        raise ValueError("a named preset is required")
    return PresetSelection(preset, select_models(
        registry, topology, context, dict(preset.defaults), parameter_sources, overrides,
        instance_parameter_sources=instance_parameter_sources,
    ))
