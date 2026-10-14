"""Deterministic experiment declarations and non-executing preflight.

References identify actual source contracts awaiting migration. A source path,
schema or successful metadata check is not a registered callable or numerical
qualification. Executor binding belongs to the gated P05 runner; none is
installed by this preparation API.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
import ast
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import re
from typing import Any, TYPE_CHECKING

from solarlab.physics.registry import (
    CapabilityRule, EvidenceRef, ModelRef, ModelRegistry, _token, _verify_evidence_sources,
    metadata_digest, metadata_value,
)
from solarlab.physics.selection import Selection, validate_selection

if TYPE_CHECKING:
    from solarlab.device.defaults import DefaultCatalog
    from solarlab.experiments.jv.preparation import PreparedJVExperiment
    from solarlab.experiments.two_dimensional.preparation import PreparedSpatialExperiment
    from solarlab.materials.resources import ResourceLibrary
    from solarlab.materials.source import SourceDocument

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
                "registered_executors": [], "status": "prepared_pending_dependencies"}

    @property
    def content_sha256(self) -> str:
        return metadata_digest(self.export())

    def verify_startup(
        self, models: ModelRegistry, sources: Iterable[tuple[str, bytes]], *,
        expected_sha256: str, expected_model_sha256: str,
    ) -> None:
        """Verify the exact registry, schemas and source contracts at startup.

        Source/qualification records are checked as bytes, not promoted to
        executable capability. No source path is imported by this check.
        """
        if expected_sha256 != self.content_sha256:
            raise ValueError("experiment registry changed from the expected frozen identity")
        documents = tuple(sources)
        models.verify_sources(documents, expected_sha256=expected_model_sha256)
        references: list[EvidenceRef] = []
        for item in self.definitions:
            references.extend((item.parameter_schema, item.result_schema, item.protocol_builder,
                               item.executor_source, item.renderer_source, *item.capability.evidence))
            for ref in item.required_models:
                models.get(ref)
        _verify_evidence_sources(references, documents)
        by_id = dict(documents)
        for item in self.definitions:
            for reference in (item.parameter_schema, item.result_schema, item.protocol_builder,
                              item.executor_source, item.renderer_source):
                path, _, symbol = reference.id.partition("#")
                if not symbol:
                    continue
                if path.endswith(".py"):
                    body = ast.parse(by_id[path]).body
                    for part in symbol.split("."):
                        node = next((node for node in body if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                                     and node.name == part), None)
                        if node is None:
                            raise ValueError(f"declared entry symbol is absent: {reference.id}")
                        body = node.body
                elif reference is item.renderer_source and path.endswith((".ts", ".js")):
                    pattern = r"\bexport\s+(?:async\s+)?function\s+" + re.escape(symbol) + r"\s*\("
                    if re.search(pattern, by_id[path].decode("utf-8")) is None:
                        raise ValueError(f"declared renderer symbol is absent: {reference.id}")

    def prepare(
        self, id: str, version: int, input: Any,
        defaults: DefaultCatalog, resources: ResourceLibrary, *,
        models: ModelRegistry, selection: Selection,
        startup_sources: Iterable[tuple[str, bytes]],
        expected_sha256: str, expected_model_sha256: str,
        sources: tuple[SourceDocument, ...] = (),
    ) -> PreparedJVExperiment | PreparedSpatialExperiment:
        """Call an existing public data preparer after the complete startup check.

        Only the two actual preparation functions below are dispatchable. A
        legacy executor reference remains evidence, never a callable fallback.
        Input schema bytes bind the DTO; result_schema binds the actual source
        contract of the immutable prepared result, not a future P05 result DTO.
        """
        self.verify_startup(models, startup_sources, expected_sha256=expected_sha256,
                            expected_model_sha256=expected_model_sha256)
        preflight = self.preflight(id, version, selection, models)
        if not preflight.declared_compatible:
            raise ValueError("experiment preparation capability refused: " + "; ".join(preflight.declared_problems))
        definition = preflight.experiment
        from solarlab.config.model_parameters import device_topology
        from solarlab.config.resolve_device import resolve_device
        from solarlab.experiments.jv.inputs import JVExperimentInput
        from solarlab.experiments.jv.preparation import PreparedJVExperiment, prepare_jv_experiment
        from solarlab.experiments.two_dimensional.inputs import SpatialExperimentInput
        from solarlab.experiments.two_dimensional.preparation import PreparedSpatialExperiment, prepare_spatial_experiment

        symbol = definition.protocol_builder.id.partition("#")[2]
        result_type: type[PreparedJVExperiment | PreparedSpatialExperiment]
        factory: Callable[..., PreparedJVExperiment | PreparedSpatialExperiment]
        if symbol == "prepare_jv_experiment":
            dto, result_type, factory, dimension = JVExperimentInput, PreparedJVExperiment, prepare_jv_experiment, 1
        elif symbol == "prepare_spatial_experiment":
            dto, result_type, factory, dimension = SpatialExperimentInput, PreparedSpatialExperiment, prepare_spatial_experiment, 2
        else:
            raise ValueError("experiment is not bound to a supported public preparation entry")
        current_source = Path(factory.__code__.co_filename).read_bytes()
        digest = hashlib.sha256(current_source).hexdigest()
        if definition.protocol_builder.sha256 != digest:
            raise ValueError("loaded preparation source differs from its registry binding")
        if (definition.result_schema.id.partition("#")[2] != result_type.__name__
                or definition.result_schema.sha256 != digest):
            raise ValueError("prepared result source contract differs from its registry binding")
        if definition.parameter_schema.sha256 != metadata_digest(dto.model_json_schema(mode="validation")):
            raise ValueError("actual preparation input schema differs from its registry binding")
        checked = dto.model_validate(input)
        if checked.experiment.kind != id or selection.context.dimension != dimension:
            raise ValueError("experiment kind/dimension differs from the selected registry entry")
        device = resolve_device(checked.device, defaults, resources, sources=sources)
        if selection.topology != device_topology(device):
            raise ValueError("selected model topology differs from the actual prepared device")
        if selection.context.temperature_K is not None and selection.context.temperature_K != device.values["settings"]["T"]:
            raise ValueError("selected temperature context differs from the actual prepared device")
        statistics = {layer.material.values["carrier_statistics"] for layer in device.layers if layer.role != "substrate"}
        if statistics != {selection.context.statistics}:
            raise ValueError("selected statistics context differs from the actual prepared device")
        return factory(checked, defaults, resources, sources=sources)

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
