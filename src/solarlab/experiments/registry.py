"""Deterministic experiment declarations and non-executing preflight.

References identify actual source contracts awaiting migration. A source path,
schema or successful metadata check is not a registered callable or numerical
qualification. Executor binding belongs to the gated P05 runner; none is
installed by this preparation API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from solarlab.physics.registry import (
    CapabilityRule, EvidenceRef, ModelRef, ModelRegistry, _token,
    metadata_digest, metadata_value,
)
from solarlab.physics.selection import Selection, validate_selection

__all__ = ["ExperimentDefinition", "ExperimentPreflight", "ExperimentRegistry"]


@dataclass(frozen=True, slots=True)
class ExperimentDefinition:
    id: str
    version: int
    label: str
    parameter_schema: EvidenceRef
    result_schema: EvidenceRef
    protocol_builder: EvidenceRef
    executor_source: EvidenceRef
    renderer_id: str
    renderer_source: EvidenceRef
    capability: CapabilityRule
    required_models: tuple[ModelRef, ...] = ()

    def __post_init__(self) -> None:
        ModelRef(self.id, self.version)
        _token(self.renderer_id, "renderer ID")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("experiment requires a display label")
        for name in ("parameter_schema", "result_schema", "protocol_builder", "executor_source", "renderer_source"):
            if not isinstance(getattr(self, name), EvidenceRef):
                raise ValueError(f"experiment {name} requires a source reference")
        if not isinstance(self.capability, CapabilityRule):
            raise ValueError("experiment requires a declared capability")
        models = tuple(self.required_models)
        if any(not isinstance(item, ModelRef) for item in models) or len(set(models)) != len(models):
            raise ValueError("experiment requires unique versioned model references")
        object.__setattr__(self, "required_models", tuple(sorted(models)))


@dataclass(frozen=True, slots=True)
class ExperimentPreflight:
    experiment: ExperimentDefinition
    experiment_registry_sha256: str
    selection_sha256: str
    declared_problems: tuple[str, ...]
    # Non-configurable: declaration-only APIs cannot enable an executor.
    can_execute: bool = field(default=False, init=False)
    availability_problems: tuple[str, ...] = field(
        default=("qualified_executor_not_registered",), init=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.experiment, ExperimentDefinition):
            raise ValueError("preflight requires a declared experiment")
        for digest in (self.experiment_registry_sha256, self.selection_sha256):
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError("preflight requires registry and selection identities")
        if isinstance(self.declared_problems, (str, bytes)):
            raise ValueError("preflight problems require a sequence")
        problems = tuple(self.declared_problems)
        if any(not isinstance(item, str) or not item for item in problems):
            raise ValueError("invalid preflight problem")
        object.__setattr__(self, "declared_problems", tuple(sorted(set(problems))))

    @property
    def declared_compatible(self) -> bool:
        return not self.declared_problems


@dataclass(frozen=True, slots=True)
class ExperimentRegistry:
    definitions: tuple[ExperimentDefinition, ...]

    def __post_init__(self) -> None:
        definitions = tuple(self.definitions)
        if any(not isinstance(item, ExperimentDefinition) for item in definitions):
            raise ValueError("experiment registry requires declarations")
        if len({(item.id, item.version) for item in definitions}) != len(definitions):
            raise ValueError("duplicate experiment ID/version")
        object.__setattr__(self, "definitions", tuple(sorted(definitions, key=lambda item: (item.id, item.version))))

    def get(self, id: str, version: int) -> ExperimentDefinition:
        ModelRef(id, version)
        for item in self.definitions:
            if (item.id, item.version) == (id, version):
                return item
        raise ValueError(f"unknown experiment ID/version: {id}/{version}")

    def export(self) -> dict[str, Any]:
        return {"schema": "solarlab.experiment-registry.v1", "experiments": metadata_value(self.definitions),
                "registered_executors": [], "status": "prepared_pending_G2"}

    @property
    def content_sha256(self) -> str:
        return metadata_digest(self.export())

    def preflight(
        self, id: str, version: int, selection: Selection, models: ModelRegistry,
    ) -> ExperimentPreflight:
        problems = list(validate_selection(selection, models))
        experiment = self.get(id, version)
        refs = tuple(sorted({item.model for item in selection.instances}))
        problems.extend(f"experiment:{reason}" for reason in experiment.capability.problems(selection.context, refs))
        for ref in experiment.required_models:
            models.get(ref)
            if ref not in refs:
                problems.append(f"experiment:missing_model:{ref.id}/{ref.version}")
        return ExperimentPreflight(experiment, self.content_sha256, selection.content_sha256,
                                   tuple(sorted(set(problems))))
