"""Complete pending V2 configuration/physics/protocol/entry correspondence.

Actual configurations use the strict W09 consumer. No model selection, driver,
protocol, state or execution identity is inferred from a legacy entry's name.
The current mapping domain deliberately refuses populated scientific bindings;
their genuine owner-produced contracts require a separately reviewed extension.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
import hashlib
import json
from typing import Any

from solarlab.config.migration import read_configuration_v2
from solarlab.materials.source import SourceDocument
from solarlab.reproducibility.configuration_migration import (
    ConfigurationMigrationPlan, prepare_registered_configuration_v2, read_selected_configuration_v2,
)
from solarlab.reproducibility.preparation import read_preparation_context
from solarlab.reproducibility.registry import MigrationRegistry, canonical, record_digest, validate_registry_pair
from solarlab.reproducibility.job_provenance import verify_loaded_source
from solarlab.reproducibility.sources import SourceSet, SymbolRef, relative_path

__all__ = ["RegistryMapping", "prepare_registry_mapping", "read_registry_mapping"]

_SCHEMA = "solarlab.preparation-mapping.v2"
_CONSUMERS = (
    ("src/solarlab/config/migration.py", "read_configuration_v2", read_configuration_v2),
    ("src/solarlab/reproducibility/configuration_migration.py", "prepare_registered_configuration_v2", prepare_registered_configuration_v2),
)
_MISSING = (
    ("model_registry", "WS2/WS3", "P03-02 actual model/source/schema records"),
    ("model_inputs", "WS2/WS4", "P03-02 original defaults and overrides for this entry"),
    ("selection", "WS2/WS3", "P03-02 actual model/topology/capability selection"),
    ("semantic_projections", "WS2/WS3", "W10 model-version and effective-physics source projections"),
    ("driver", "WS4", "W03 facade explicit effective driver for this entry"),
    ("effective_behavior", "WS4", "W03 facade with W08/W10 BehaviorContext or R1ExecutionContext"),
    ("protocol", "P05 executor owner", "W11 full actual protocol document"),
    ("preparation", "P05 executor owner", "W11 actual preparation/history document"),
    ("initial_state", "P05 executor owner", "W11 actual initial-state bytes and qualification"),
    ("grid", "WS3/P05 executor owner", "actual compiled mesh/layout binding"),
    ("numeric_controls", "P05 executor owner", "complete numerical policy for the actual entry"),
    ("engine", "WS4", "W03 explicit engine with no fallback"),
    ("runtime_environment", "WS4", "worker observed environment before engine import"),
    ("release_bundle", "Root/P07", "W13 reviewed build-source bundle and isolated installation"),
    ("execution_manifest", "WS4/P05 executor owner", "W11 complete externally pinned and consumed manifest"),
    ("new_executor_and_tests", "executor owner/Root", "W15 actual entry and independent applicable node evidence"),
)


def _base(registry: MigrationRegistry) -> dict[str, str]:
    data = registry.export()
    data.pop("preparation_mapping", None)
    return {"id": registry.source.id, "kind": registry.spec.kind, "canonical_sha256": record_digest(data)}


def _consumer_bindings(sources: SourceSet, *, admission: bool = True) -> list[dict[str, Any]]:
    required = {path for path, _, _ in _CONSUMERS}
    if {ref.path for ref in sources.references} != required:
        raise ValueError("mapping requires the exact current v2 consumer source set")
    result = []
    for path, symbol, consumer in _CONSUMERS:
        document = sources.at_path(path)
        sources.verify_symbol(SymbolRef(source_id=document.id, symbol=symbol))
        if admission:
            verify_loaded_source(document.content, consumer.__code__.co_filename, label="v2 consumer")
        result.append({"source": sources.reference(document.id).model_dump(mode="json"), "symbol": symbol,
                       "kind": "data_preparation_only", "can_execute": False})
    return result


@dataclass(frozen=True, slots=True)
class RegistryMapping:
    document: SourceDocument
    reconciliation: dict[str, Any] | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if type(self.document) is not SourceDocument:
            raise ValueError("mapping requires actual immutable source bytes")
        relative_path(self.document.id)
        data = json.loads(self.document.content)
        if (not isinstance(data, dict) or data.get("schema") != _SCHEMA
                or data.get("activation") != "pending_migration" or data.get("can_execute") is not False
                or data.get("scientific_gates_awarded") != [] or canonical(data) != self.document.content):
            raise ValueError("mapping requires canonical, non-activating source-only data")

    @property
    def sha256(self) -> str:
        return self.document.sha256

    def export(self) -> dict[str, Any]:
        return json.loads(self.document.content)

    def entry(self, id: str) -> dict[str, Any]:
        for entry in self.export()["entries"]:
            if entry["id"] == id:
                return entry
        raise ValueError("unknown mapped registry entry")

    def reference(self) -> dict[str, str]:
        return {"path": self.document.id, "sha256": self.sha256,
                "migration_plan_sha256": self.export()["migration_plan_sha256"]}


def _mapping(
    matrix: MigrationRegistry, refinement: MigrationRegistry, *, migration: ConfigurationMigrationPlan,
    expected_migration_sha256: str, consumer_sources: SourceSet,
    document_id: str = "reproducibility/PreparationMappingV2.json",
    reprepare: bool = True,
) -> RegistryMapping:
    """Map every actual entry and consume all proposed configuration bytes.

    The independently pinned W09 dry-run is an input. No data namespace,
    selector, model or numerical execution is generated by this function.
    """
    validate_registry_pair(matrix, refinement)
    relative_path(document_id)
    if type(migration) is not ConfigurationMigrationPlan or migration.sha256 != expected_migration_sha256 or not migration.supported:
        raise ValueError("mapping requires the exact complete configuration migration plan")
    consumers = _consumer_bindings(consumer_sources, admission=reprepare)
    report, selection = migration.export(), json.loads(migration.selection_json)
    config_ids = {entry.id for entry in matrix.spec.entries if entry.section == "configs"}
    if (set(report["selected_ids"]) != config_ids or len(report["selected_ids"]) != len(config_ids)
            or report["selection_sha256"] != hashlib.sha256(migration.selection_json).hexdigest()):
        raise ValueError("mapping requires the complete original configuration inventory and selection")
    rows = selection["configurations"]
    if len(rows) != len(config_ids) or {row["id"] for row in rows} != config_ids:
        raise ValueError("mapping configuration selection inventory is incomplete or duplicated")
    files = dict(migration.files)
    if (len(files) != len(migration.files) or len(files) != len(config_ids)
            or {(path, doc.id, doc.sha256) for path, doc in migration.files}
               != {(row["path"], row["document_id"], row["sha256"]) for row in rows}):
        raise ValueError("mapping file paths/content differ from the frozen migration selection")
    context = matrix.spec.preparation_context
    if (context is None or report["default_catalog_sha256"] != context.default_catalog_sha256
            or report["resource_library_sha256"] != context.resource_library_sha256):
        raise ValueError("mapping preparation context differs from the frozen migration plan")
    # The W09 selection is still prospective. Do not override any active source
    # selector; one would require an explicit new joint mapping/switch review.
    if matrix.spec.configuration_selection is not None:
        raise ValueError("this mapping domain requires the inactive W09 source selection")
    selection_source = SourceDocument("ConfigurationSourcesV2.json", migration.selection_json)
    configurations = {}
    defaults, resources = read_preparation_context(matrix) if reprepare else (None, None)
    for row in rows:
        relative_path(row["path"])
        document = files[row["path"]]
        raw = json.loads(document.content)
        original_document = matrix.sources.document(row["original_source"]["id"])
        preparation = matrix.entry(row["id"]).preparation
        if (row["original_source"]["sha256"] != original_document.sha256 or raw["source"] != row["original_source"]
                or raw["default_catalog_sha256"] != context.default_catalog_sha256
                or raw["resource_library_sha256"] != context.resource_library_sha256
                or preparation is None or row["prepared_content_sha256"] != preparation.content_sha256):
            raise ValueError("mapping recorded input/dependency identity mismatch")
        # This digest uses the original serialized typed input, not today's resolver.
        typed_sha = hashlib.sha256(canonical(raw["input"])).hexdigest()
        if reprepare:
            assert defaults is not None and resources is not None
            prepared = read_selected_configuration_v2(matrix, selection_source, row["id"],
                documents=(document,), defaults=defaults, resources=resources)
            typed_sha = hashlib.sha256(prepared.input_json).hexdigest()
        configurations[row["id"]] = {**row, "proposed_path": "reproducibility/ConfigurationsV2/" + row["path"],
            "typed_input_sha256": typed_sha,
            "input_identity_kind": "prepared_content_only; not H_input", "data_verified": True,
            "publication": "pending_WS4_editor_and_consumer_adaptation"}
    entries: list[dict[str, Any]] = []
    for registry in (matrix, refinement):
        for entry in registry.spec.entries:
            original = registry.original(entry.id)
            if any(getattr(entry.identities, name) is not None for name in ("H_input", "H_physics", "protocol_sha256", "H_execution")) or entry.new_executor is not None:
                raise ValueError("actual identity/executor binding needs a reviewed extension, not implicit promotion")
            executor = None
            if entry.executor:
                executor = {"source": registry.sources.reference(entry.executor.source_id).model_dump(mode="json"),
                            "symbol": entry.executor.symbol, "version": entry.executor_version}
            tests = [{"path": path, "source": registry.sources.reference(registry.sources.at_path(registry.spec.original_path_root + "/" + path).id).model_dump(mode="json")}
                     for path in entry.test_files]
            entries.append({"id": entry.id, "section": entry.section, "original_key": entry.original_key,
                "original_record_sha256": entry.original_record_sha256, "original_record": original,
                "legacy_evidence": {"status": entry.legacy_status, "claim_level": entry.legacy_claim_level,
                    "evidence_tier": entry.legacy_evidence_tier, "semantic_sha256": entry.legacy_semantic_sha256},
                "configurations": [configurations[id] for id in entry.configurations],
                "physical": {"status": "pending", "H_input": None, "H_physics": None, "contexts": []},
                "protocol": {"status": "pending", "sha256": None, "declared_legacy_entry": executor},
                "execution": {"status": "pending", "H_execution": None, "new_executor": None, "manifest": None},
                "old_test_files": tests, "old_test_nodes": list(entry.test_nodes), "new_scientific_test_nodes": [],
                "missing_fields": [{"field": field, "status": "pending", "owner": owner, "source_dependency": dependency}
                                   for field, owner, dependency in _MISSING],
                "preparation_consumers": consumers, "new_evidence": "source_verified_data_preparation_only",
                "can_execute": False})
    value = {"schema": _SCHEMA, "registries": [_base(matrix), _base(refinement)],
        "original_sources": [{"source": r.sources.reference(r.spec.original_registry).model_dump(mode="json"),
            "reader": r.spec.original_reader.model_dump(mode="json"), "verification": r.spec.historical_verification}
                             for r in (matrix, refinement)],
        "migration_plan_sha256": migration.sha256, "configuration_selection_sha256": report["selection_sha256"],
        "preparation_context": {"default_catalog_sha256": context.default_catalog_sha256, "resource_library_sha256": context.resource_library_sha256},
        "inventory_counts": dict(Counter(entry["section"] for entry in entries)),
        "entries": sorted(entries, key=lambda entry: entry["id"]),
        "activation": "pending_migration", "can_execute": False, "scientific_gates_awarded": []}
    return RegistryMapping(SourceDocument(document_id, canonical(value)))


def prepare_registry_mapping(
    matrix: MigrationRegistry, refinement: MigrationRegistry, *, migration: ConfigurationMigrationPlan,
    expected_migration_sha256: str, consumer_sources: SourceSet,
    document_id: str = "reproducibility/PreparationMappingV2.json",
) -> RegistryMapping:
    """Prepare new data; installed consumer consistency is an admission check."""
    return _mapping(matrix, refinement, migration=migration,
        expected_migration_sha256=expected_migration_sha256, consumer_sources=consumer_sources,
        document_id=document_id, reprepare=True)


def read_registry_mapping(
    matrix: MigrationRegistry, refinement: MigrationRegistry, source: SourceDocument, *,
    migration: ConfigurationMigrationPlan, expected_migration_sha256: str,
    expected_sha256: str, consumer_sources: SourceSet,
    reconcile_current: bool = False, strict_same_source: bool = False,
) -> RegistryMapping:
    """Check immutable input/entry bindings; current re-preparation is explicit."""
    validate_registry_pair(matrix, refinement)
    reference = matrix.spec.preparation_mapping
    if (reference is None or source.id != reference.path or source.sha256 != reference.sha256
            or source.sha256 != expected_sha256 or reference.migration_plan_sha256 != expected_migration_sha256):
        raise ValueError("mapping source differs from the two registries and the external expected pins")
    actual = RegistryMapping(source)
    expected = _mapping(matrix, refinement, migration=migration,
        expected_migration_sha256=expected_migration_sha256, consumer_sources=consumer_sources, document_id=source.id, reprepare=False)
    if actual.document.content != expected.document.content:
        raise ValueError("mapping content differs from actual configurations, original evidence, entries or pending context")
    if reconcile_current or strict_same_source:
        try:
            now = prepare_registry_mapping(matrix, refinement, migration=migration,
                expected_migration_sha256=expected_migration_sha256, consumer_sources=consumer_sources,
                document_id=source.id)
            comparison = {"status": "same" if now.sha256 == actual.sha256 else "different",
                          "producer_sha256": actual.sha256, "current_sha256": now.sha256}
        except ValueError as exc:
            comparison = {"status": "different", "producer_sha256": actual.sha256,
                          "reason": str(exc)}
        if strict_same_source and comparison["status"] != "same":
            raise ValueError("strict_same_source mapping reconciliation differs")
        return replace(actual, reconciliation=comparison)
    return actual
