"""Inactive V2 mappings over the exact, separately retained V1 registries.

This module validates source and location correspondence. It neither executes
registered callables nor interprets a research certificate as a new award.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator
import yaml

from solarlab.materials.source import SourceDocument
from solarlab.reproducibility.sources import Digest, Record, SourceRef, SourceSet, SymbolRef, relative_path


class _Loader(yaml.SafeLoader):
    """Retain the V1 readers' scalar rules, adding duplicate-key rejection."""


def _mapping(loader: _Loader, node: yaml.MappingNode) -> dict[str, Any]:
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if not isinstance(key, str) or key in result:
            raise ValueError("registry keys must be unique strings")
        result[key] = loader.construct_object(value_node, deep=True)
    return result


_Loader.add_constructor("tag:yaml.org,2002:map", _mapping)


def mapping(source: SourceDocument) -> dict[str, Any]:
    value = yaml.load(source.content, Loader=_Loader)
    if not isinstance(value, dict):
        raise ValueError("registry source must contain a mapping")
    canonical(value)  # Reject nonfinite values and non-JSON tags.
    return value


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def record_digest(value: Any) -> str:
    """V1 parsed record locator integrity; not its scientific semantic hash."""
    return hashlib.sha256(canonical(value)).hexdigest()


def config_id(path: str) -> str:
    return "config:" + relative_path(path)


class PreparationIssue(Record):
    path: tuple[str | int, ...]
    type: str
    message: str


class ConfigurationPreparation(Record):
    status: Literal["prepared_only", "blocked"]
    input_id: Annotated[str, Field(min_length=1)]
    loader: SymbolRef
    resolver: SymbolRef
    content_sha256: Digest | None
    issues: tuple[PreparationIssue, ...]

    @model_validator(mode="after")
    def _status(self) -> ConfigurationPreparation:
        if self.status == "prepared_only" and (self.content_sha256 is None or self.issues):
            raise ValueError("prepared mapping requires actual content identity and no input errors")
        if self.status == "blocked" and (self.content_sha256 is not None or not self.issues):
            raise ValueError("blocked mapping requires the original preparation errors and no invented identity")
        return self


class IdentityGaps(Record):
    # Actual registry configurations have no migrated model-selection,
    # protocol/state or executor contract. Prepared content is bound separately.
    H_input: None
    H_physics: None
    protocol_sha256: None
    H_execution: None
    dependencies: Annotated[tuple[str, ...], Field(min_length=1)]


class SourceNote(Record):
    source: SymbolRef
    note: Annotated[str, Field(min_length=1)]


class MigrationEntry(Record):
    id: Annotated[str, Field(min_length=1)]
    section: Literal["configs", "benchmarks", "lanes"]
    original_key: Annotated[str, Field(min_length=1)]
    original_record_sha256: Digest
    legacy_status: str | None
    legacy_claim_level: str | None
    legacy_evidence_tier: str | None
    legacy_semantic_sha256: Digest | None
    configurations: tuple[str, ...]
    test_files: tuple[str, ...]
    test_nodes: tuple[str, ...]
    executor: SymbolRef | None
    executor_version: str | None
    migration_status: Literal["prepared_only", "blocked_input", "legacy_only"]
    migration_evidence: Literal["source_verified_preparation_only"]
    new_executor: None
    new_test_nodes: tuple[str, ...]
    identities: IdentityGaps
    preparation: ConfigurationPreparation | None
    migration_notes: tuple[SourceNote, ...]


class PreparationContext(Record):
    default_catalog_sha256: Digest
    resource_library_sha256: Digest
    default_sources: tuple[str, ...]
    model_sources: tuple[str, ...]
    resource_sources: tuple[str, ...]


class RegistryDocument(Record):
    schema_version: Literal["solarlab.registry-migration.v2"]
    kind: Literal["configuration_matrix", "numerical_refinement"]
    activation: Literal["pending_migration"]
    original_registry: str
    companion_registry: str
    original_path_root: str
    original_reader: SymbolRef
    historical_verification: Literal["original_bytes_paths_and_original_reader; no_certificate_reinterpretation"]
    sources: Annotated[tuple[SourceRef, ...], Field(min_length=1)]
    preparation_context: PreparationContext | None
    entries: Annotated[tuple[MigrationEntry, ...], Field(min_length=1)]


def _rows(document: dict[str, Any], section: str) -> dict[str, dict[str, Any]]:
    raw = document[section]
    if section == "configs":
        if not isinstance(raw, list) or any(not isinstance(row, dict) or "path" not in row for row in raw):
            raise ValueError("V1 configuration inventory must be a list of path records")
        values = {row["path"]: row for row in raw}
        if len(values) != len(raw):
            raise ValueError("duplicate V1 configuration path")
        return values
    if not isinstance(raw, dict) or any(not isinstance(row, dict) for row in raw.values()):
        raise ValueError("V1 registry inventory must be a mapping of records")
    return raw


def _module_symbol(location: str, prefix: str) -> tuple[str, str]:
    parts = location.split(":")
    if len(parts) != 2:
        raise ValueError("executor requires its original module:symbol location")
    module, symbol = parts
    return relative_path(prefix + "/" + module.replace(".", "/") + ".py"), symbol


def _require_symbol(sources: SourceSet, ref: SymbolRef, path: str, symbol: str) -> None:
    if sources.reference(ref.source_id).path != path or ref.symbol != symbol:
        raise ValueError(f"test/executor/reader binding does not match the original declared path: {path}:{symbol}")
    sources.verify_symbol(ref)


def _test_node(sources: SourceSet, node: str, prefix: str) -> None:
    file, *parts = node.split("::")
    path = relative_path(prefix + "/" + file)
    source = sources.at_path(path)
    if not parts:
        raise ValueError("test node must name its actual test function")
    symbol = ".".join(part.split("[", 1)[0] for part in parts)
    sources.verify_symbol(SymbolRef(source_id=source.id, symbol=symbol))


@dataclass(frozen=True, slots=True)
class MigrationRegistry:
    source: SourceDocument
    spec: RegistryDocument
    sources: SourceSet

    @property
    def can_execute(self) -> bool:
        return False

    @property
    def content_sha256(self) -> str:
        return record_digest(self.spec.model_dump(mode="json"))

    def entry(self, id: str) -> MigrationEntry:
        for entry in self.spec.entries:
            if entry.id == id:
                return entry
        raise KeyError(id)

    def original(self, id: str) -> dict[str, Any]:
        entry = self.entry(id)
        return _rows(mapping(self.sources.document(self.spec.original_registry)), entry.section)[entry.original_key]

    def original_bytes(self) -> bytes:
        return self.sources.document(self.spec.original_registry).content

    def export(self) -> dict[str, Any]:
        return self.spec.model_dump(mode="json")


def load_registry(source: SourceDocument, documents: tuple[SourceDocument, ...]) -> MigrationRegistry:
    """Verify a complete supplied V2 registry and its immutable source bundle.

    No registered scientific executor is imported or called. Pending V2 data
    cannot activate a consumer; migration needs a later qualified contract.
    """
    spec = RegistryDocument.model_validate(mapping(source))
    sources = SourceSet(spec.sources, documents)
    root = relative_path(spec.original_path_root)
    original = mapping(sources.document(spec.original_registry))
    companion = mapping(sources.document(spec.companion_registry))
    is_matrix = spec.kind == "configuration_matrix"
    matrix, refinement = (original, companion) if is_matrix else (companion, original)
    if matrix.get("schema_version") != 1 or refinement.get("schema_version") != "numerical-refinement-registry-v1":
        raise ValueError("V2 source types must match the original V1 registries")
    matrix_source = sources.document(spec.original_registry if is_matrix else spec.companion_registry)
    refinement_source = sources.document(spec.companion_registry if is_matrix else spec.original_registry)
    if sources.reference(refinement_source.id).path != root + "/" + matrix["numerical_refinement_registry"]["path"] or refinement_source.sha256 != matrix["numerical_refinement_registry"]["sha256"]:
        raise ValueError("V1 companion registry path/hash mismatch")
    if sources.reference(matrix_source.id).path != root + "/reproducibility/ConfigBenchmarkMatrix.yaml":
        raise ValueError("V1 matrix path mismatch")
    schema_source = sources.at_path(root + "/" + matrix["schema_registry"])
    schemas = mapping(schema_source)["schemas"]
    configuration_rows = _rows(matrix, "configs")
    for key, row in configuration_rows.items():
        actual = sources.at_path(root + "/" + relative_path(key))
        if actual.sha256 != row["sha256"]:
            raise ValueError(f"original config SHA-256 mismatch: {key}")
        if row["schema"] not in schemas:
            raise ValueError("missing original input schema declaration")
    for row in matrix["resources"]:
        if sources.at_path(root + "/" + relative_path(row["path"])).sha256 != row["sha256"]:
            raise ValueError("original resource SHA-256 mismatch")
    reader = "perovskite_sim.reproducibility:validate_matrix" if is_matrix else "perovskite_sim.validation.numerical_certificate:load_refinement_registry"
    _require_symbol(sources, spec.original_reader, *_module_symbol(reader, root))
    sections = ("configs", "benchmarks") if is_matrix else ("lanes",)
    expected = {(section, key) for section in sections for key in _rows(original, section)}
    found = {(entry.section, entry.original_key) for entry in spec.entries}
    if len(found) != len(spec.entries) or found != expected or len({entry.id for entry in spec.entries}) != len(spec.entries):
        raise ValueError("missing, duplicate or unexpected V1-to-V2 inventory entry")
    if is_matrix != (spec.preparation_context is not None):
        raise ValueError("configuration preparation requires one explicit context")
    if spec.preparation_context:
        context = spec.preparation_context
        for id in (*context.default_sources, *context.model_sources, *context.resource_sources):
            sources.document(id)
    input_ids = set()
    for entry in spec.entries:
        row = _rows(original, entry.section)[entry.original_key]
        identifier = config_id(entry.original_key) if entry.section == "configs" else ("benchmark:" if is_matrix else "lane:") + entry.original_key
        if entry.id != identifier or entry.original_record_sha256 != record_digest(row):
            raise ValueError("old/new stable identity or original record digest mismatch")
        if (entry.legacy_status, entry.legacy_claim_level, entry.legacy_evidence_tier, entry.legacy_semantic_sha256) != (row.get("status"), row.get("claim_level"), row.get("evidence_tier"), row.get("semantic_sha256")):
            raise ValueError("original evidence/semantic identity must be preserved without reinterpretation")
        paths = [entry.original_key] if entry.section == "configs" else row.get("configs", [row.get("config")])
        if not paths or any(path not in configuration_rows for path in paths) or entry.configurations != tuple(config_id(path) for path in paths):
            raise ValueError("configuration link is missing or does not match its original record")
        if entry.section == "lanes" and row["config_sha256"] != configuration_rows[row["config"]]["sha256"]:
            raise ValueError("lane/config identity mismatch")
        if entry.test_files != tuple(row.get("tests", [])) or entry.test_nodes != tuple(row.get("node_ids", [])):
            raise ValueError("original test/node binding mismatch")
        for path in entry.test_files:
            sources.at_path(root + "/" + relative_path(path))
        for node in entry.test_nodes:
            if node.split("::", 1)[0] not in entry.test_files:
                raise ValueError("test node does not belong to the declared test files")
            _test_node(sources, node, root)
        if "executor" in row:
            if entry.executor is None or entry.executor_version != row["executor_version"]:
                raise ValueError("missing/mismatched original executor binding")
            _require_symbol(sources, entry.executor, *_module_symbol(row["executor"], root))
        elif entry.executor is not None or entry.executor_version is not None:
            raise ValueError("no executor was declared in the original record")
        if entry.new_test_nodes:
            raise ValueError("no migrated scientific test-node binding is established")
        for note in entry.migration_notes:
            sources.verify_symbol(note.source)
        prep = entry.preparation
        if entry.section == "configs":
            if prep is None or prep.input_id in input_ids or entry.migration_status != ("prepared_only" if prep.status == "prepared_only" else "blocked_input"):
                raise ValueError("configuration preparation status/identity mismatch")
            input_ids.add(prep.input_id)
            if prep.input_id != "registry." + PurePosixPath(entry.original_key).stem:
                raise ValueError("new configuration identity must retain its explicit original-name binding")
            loader, resolver = preparation_symbols(row["schema"])
            _require_symbol(sources, prep.loader, *loader)
            _require_symbol(sources, prep.resolver, *resolver)
        elif prep is not None or entry.migration_status != "legacy_only":
            raise ValueError("unmigrated benchmark/lane must remain legacy-only")
    return MigrationRegistry(source, spec, sources)


def preparation_symbols(schema: str) -> tuple[tuple[str, str], tuple[str, str]]:
    if schema == "standard-device-v1":
        return ("src/solarlab/config/device_import.py", "import_standard_device"), ("src/solarlab/config/resolve_device.py", "resolve_device")
    if schema == "scaps-device-v1":
        return ("src/solarlab/config/scaps_input.py", "import_scaps_device"), ("src/solarlab/config/resolve_device.py", "resolve_device")
    if schema == "tandem-v1":
        return ("src/solarlab/config/tandem_input.py", "import_tandem"), ("src/solarlab/config/resolve_device.py", "resolve_tandem")
    raise ValueError("no public preparation reader for the original schema")


def validate_registry_pair(matrix: MigrationRegistry, refinement: MigrationRegistry) -> None:
    if matrix.spec.kind != "configuration_matrix" or refinement.spec.kind != "numerical_refinement":
        raise ValueError("a configuration matrix and numerical registry are required")
    for id in (matrix.spec.original_registry, matrix.spec.companion_registry):
        if matrix.sources.reference(id) != refinement.sources.reference(id) or matrix.sources.document(id) != refinement.sources.document(id):
            raise ValueError("V2 registries refer to different original source snapshots")
    for entry in refinement.spec.entries:
        for id in entry.configurations:
            matrix.entry(id)
