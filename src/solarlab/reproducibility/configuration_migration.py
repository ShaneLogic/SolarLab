"""Explicit data-source switchover for the existing configuration inventory.

The original registry retains all old scientific claims and remains immutable.
A new selection manifest binds v2 data to a strict v2 consumer; it grants no
experiment or numerical qualification. All I/O is confined to the CLI/publisher.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import warnings
from pathlib import Path, PurePosixPath
import posixpath
from typing import Any

from solarlab.config.migration import migrate_configuration, read_configuration_v2
from solarlab.device.defaults import DefaultCatalog
from solarlab.device.resolved import PreparedDevice, PreparedTandem
from solarlab.materials.resources import ResourceLibrary
from solarlab.materials.source import SourceDocument
from solarlab.reproducibility.preparation import prepare_configuration, read_preparation_context
from solarlab.reproducibility.registry import MigrationRegistry, RegistryDocument, load_registry, mapping
from solarlab.reproducibility.sources import read_sources, relative_path

__all__ = ["ConfigurationMigrationPlan", "prepare_configuration_migration",
           "prepare_registered_configuration_v2", "read_selected_configuration_v2", "load_selected_configuration_v2",
           "publish_configuration_migration"]

_READER = "solarlab.reproducibility.configuration_migration:read_selected_configuration_v2"


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@dataclass(frozen=True, slots=True)
class ConfigurationMigrationPlan:
    files: tuple[tuple[str, SourceDocument], ...]
    report_json: bytes
    selection_json: bytes

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.report_json).hexdigest()

    @property
    def supported(self) -> bool:
        return bool(self.files) and not self.export()["blocked"]

    def export(self) -> dict[str, Any]:
        return json.loads(self.report_json)


def prepare_registered_configuration_v2(
    registry: MigrationRegistry, id: str, document: SourceDocument, *,
    defaults: DefaultCatalog, resources: ResourceLibrary,
    expected_document_sha256: str | None = None, strict_derived_metadata: bool = False,
    reconciliation: dict[str, Any] | None = None,
) -> PreparedDevice | PreparedTandem:
    """Consume v2 directly; old-source labels remain original evidence only."""
    entry = registry.entry(id)
    expected = entry.preparation
    if entry.section != "configs" or expected is None or expected.status != "prepared_only":
        raise ValueError("the selected original configuration has no accepted data preparation")
    if expected_document_sha256 is not None and document.sha256 != expected_document_sha256:
        raise ValueError("v2 selected configuration content binding mismatch")
    value = read_configuration_v2(document, defaults=defaults, resources=resources, sources=registry.sources.documents)
    if value.to_input().id != expected.input_id:
        raise ValueError("v2 data differs from the original registered preparation; retain the old path")
    comparison = {"recorded_prepared_sha256": expected.content_sha256,
                  "current_prepared_sha256": value.content_sha256,
                  "status": "same" if value.content_sha256 == expected.content_sha256 else "different"}
    if reconciliation is not None:
        reconciliation.update(comparison)
    if comparison["status"] == "different":
        # Without an independent original-document digest a changed declaration
        # cannot be distinguished from derived metadata drift.
        if expected_document_sha256 is None or strict_derived_metadata:
            raise ValueError("v2 data differs from the original registered preparation; retain the old path")
        warnings.warn("bound v2 input bytes unchanged; derived preparation metadata differs from the recorded identity",
                      RuntimeWarning, stacklevel=2)
    return value


def read_selected_configuration_v2(
    registry: MigrationRegistry, selection: SourceDocument, id: str, *, documents: tuple[SourceDocument, ...],
    defaults: DefaultCatalog, resources: ResourceLibrary,
    strict_derived_metadata: bool = False, reconciliation: dict[str, Any] | None = None,
) -> PreparedDevice | PreparedTandem:
    """The actual consumer of the published data-source selection manifest."""
    reference = registry.spec.configuration_selection
    if reference is not None and (selection.id != reference.path or selection.sha256 != reference.sha256):
        raise ValueError("v2 selection differs from the matrix data pin")
    data = mapping(selection)
    original_registry = registry.sources.document(registry.spec.original_registry)
    if (set(data) != {"schema_version", "reader", "original_registry", "configurations", "scope", "can_execute", "scientific_gates_awarded"}
            or data["schema_version"] != "solarlab.configuration-sources.v2" or data["reader"] != _READER
            or data["original_registry"] != {"id": original_registry.id, "sha256": original_registry.sha256}
            or data["scope"] != "data_source_selection_only" or data["can_execute"] is not False
            or data["scientific_gates_awarded"] != []):
        raise ValueError("v2 selection schema, source or scope binding mismatch")
    rows = data["configurations"]
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("v2 selection requires configuration rows")
    names = []
    for row in rows:
        if set(row) != {"id", "path", "document_id", "sha256", "prepared_content_sha256", "original_source"}:
            raise ValueError("v2 selection row schema mismatch")
        if any(type(row[key]) is not str for key in ("id", "path", "document_id", "sha256", "prepared_content_sha256")):
            raise ValueError("v2 selection identifiers and digests must be strings")
        names.append(relative_path(row["path"]))
    if len({row["id"] for row in rows}) != len(rows) or len(set(names)) != len(rows):
        raise ValueError("duplicate v2 configuration selection")
    row = next((row for row in rows if row["id"] == id), None)
    trusted = {document.id: document for document in documents if type(document) is SourceDocument}
    if row is None or len(trusted) != len(documents):
        raise ValueError("unknown v2 selection or duplicate supplied document ID")
    document = trusted.get(row["document_id"])
    original = registry.sources.at_path(registry.spec.original_path_root + "/" + registry.entry(id).original_key)
    if (document is None or document.sha256 != row["sha256"]
            or row["original_source"] != {"id": original.id, "sha256": original.sha256}):
        raise ValueError("v2 selected configuration content binding mismatch")
    expected = registry.entry(id).preparation
    if expected is None or row["prepared_content_sha256"] != expected.content_sha256:
        raise ValueError("v2 selected preparation content binding mismatch")
    value = prepare_registered_configuration_v2(registry, id, document, defaults=defaults, resources=resources,
        expected_document_sha256=row["sha256"], strict_derived_metadata=strict_derived_metadata,
        reconciliation=reconciliation)
    return value


def load_selected_configuration_v2(
    registry: MigrationRegistry, id: str, *, source_root: Path,
    defaults: DefaultCatalog, resources: ResourceLibrary,
) -> PreparedDevice | PreparedTandem:
    """Read the matrix's pinned v2 selection from an explicit data bundle."""
    reference = registry.spec.configuration_selection
    if reference is None or not Path(source_root).is_absolute():
        raise ValueError("the matrix must select v2 data with an explicit absolute source root")
    root = Path(source_root).resolve(strict=True)
    selection_path = (root / reference.path).resolve(strict=True)
    if not selection_path.is_relative_to(root):
        raise ValueError("v2 selection escapes the explicit source root")
    selection = SourceDocument(reference.path, selection_path.read_bytes())
    if selection.sha256 != reference.sha256:
        raise ValueError("v2 selection differs from the matrix data pin")
    rows = mapping(selection).get("configurations")
    if not isinstance(rows, list):
        raise ValueError("v2 selection requires configuration rows")
    row = next((row for row in rows if isinstance(row, dict) and row.get("id") == id), None)
    if row is None:
        raise ValueError("configuration is absent from the selected v2 dataset")
    path = (selection_path.parent / relative_path(row["path"])).resolve(strict=True)
    if not path.is_relative_to(selection_path.parent):
        raise ValueError("v2 configuration escapes the selected namespace")
    document = SourceDocument(row["document_id"], path.read_bytes())
    return read_selected_configuration_v2(registry, selection, id, documents=(document,), defaults=defaults, resources=resources)


def prepare_configuration_migration(
    registry: MigrationRegistry, *, ids: tuple[str, ...] | None = None,
) -> ConfigurationMigrationPlan:
    """Dry-run the exact registered inputs through the one forward migrator."""
    if registry.spec.kind != "configuration_matrix":
        raise ValueError("configuration migration requires the actual configuration matrix")
    defaults, resources = read_preparation_context(registry)
    all_ids = tuple(entry.id for entry in registry.spec.entries if entry.section == "configs")
    selected = all_ids if ids is None else tuple(ids)
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(all_ids):
        raise ValueError("select unique existing configuration entry IDs")
    files = []
    rows = []
    blocked = []
    bindings = []
    for id in selected:
        entry = registry.entry(id)
        prepared = prepare_configuration(registry, id, defaults=defaults, resources=resources).prepared
        if prepared is None:
            blocked.append({"id": id, "issues": entry.preparation.model_dump(mode="json") if entry.preparation else None})
            continue
        source = registry.sources.at_path(registry.spec.original_path_root + "/" + entry.original_key)
        references = {}
        if isinstance(prepared, PreparedTandem):
            parent = PurePosixPath(registry.sources.reference(source.id).path).parent
            declaration = prepared.to_input()
            for name in (declaration.top_cell_reference, declaration.bottom_cell_reference):
                references[name] = registry.sources.at_path(relative_path(posixpath.normpath(str(parent / name))))
        result = migrate_configuration(source, format=registry.original(id)["schema"], input_id=prepared.to_input().id,
            defaults=defaults, resources=resources, references=references)
        row = {"id": id, "original_key": entry.original_key, "legacy_status": entry.legacy_status,
               "legacy_claim_level": entry.legacy_claim_level, "legacy_evidence_tier": entry.legacy_evidence_tier,
               "legacy_semantic_sha256": entry.legacy_semantic_sha256, "migration": result.export()}
        rows.append(row)
        if result.document is None:
            blocked.append({"id": id, "issues": result.export()["issues"]})
            continue
        reopened = prepare_registered_configuration_v2(registry, id, result.document, defaults=defaults, resources=resources)
        if reopened.to_mapping() != prepared.to_mapping() or reopened.input_json != prepared.input_json:
            raise ValueError("registered v2 consumer changed the actual prepared payload")
        path = relative_path(str(PurePosixPath(entry.original_key).with_suffix(".json")))
        files.append((path, result.document))
        bindings.append({"id": id, "path": path, "document_id": result.document.id, "sha256": result.document.sha256,
                         "prepared_content_sha256": prepared.content_sha256,
                         "original_source": {"id": source.id, "sha256": source.sha256}})
    if len({path for path, _ in files}) != len(files):
        raise ValueError("v2 target configuration paths collide")
    source = registry.sources.document(registry.spec.original_registry)
    selection = {"schema_version": "solarlab.configuration-sources.v2", "reader": _READER,
        "original_registry": {"id": source.id, "sha256": source.sha256}, "configurations": bindings,
        "scope": "data_source_selection_only", "can_execute": False, "scientific_gates_awarded": []}
    report = {"schema": "solarlab.configuration-switch-plan.v1", "registry": selection["original_registry"],
        "migration_control_registry": {"id": registry.source.id, "sha256": registry.source.sha256},
        "default_catalog_sha256": defaults.content_sha256, "resource_library_sha256": resources.content_sha256,
        "selected_ids": list(selected), "rows": rows, "blocked": blocked,
        "selection_sha256": hashlib.sha256(_json(selection)).hexdigest(), "can_execute": False,
        "scientific_gates_awarded": [], "original_registry_activation": registry.spec.activation}
    return ConfigurationMigrationPlan(tuple(files), _json(report), _json(selection))


def publish_configuration_migration(
    plan: ConfigurationMigrationPlan, destination: Path, *, expected_plan_sha256: str,
) -> Path:
    """Publish an explicitly selected, previously inspected data-only plan.

    Existing namespaces are refused. The manifest is written last, so a write
    failure cannot make an incomplete namespace appear as a complete selection.
    """
    if type(plan) is not ConfigurationMigrationPlan or plan.sha256 != expected_plan_sha256 or not plan.supported:
        raise ValueError("a complete matching dry-run plan is required before data publication")
    destination = Path(destination)
    if not destination.is_absolute() or destination.exists():
        raise ValueError("publication requires a new explicit absolute destination")
    selection = json.loads(plan.selection_json)
    report = plan.export()
    names = tuple(relative_path(path) for path, _ in plan.files)
    if (hashlib.sha256(plan.selection_json).hexdigest() != report["selection_sha256"]
            or len(set(names)) != len(names) or "ConfigurationSourcesV2.json" in names
            or {(row["path"], row["sha256"]) for row in selection["configurations"]}
               != {(path, document.sha256) for path, document in plan.files}):
        raise ValueError("migration file/selection content binding mismatch")
    destination.mkdir()
    for name, document in plan.files:
        path = destination / relative_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(document.content)
        if hashlib.sha256(path.read_bytes()).hexdigest() != document.sha256:
            raise ValueError("published configuration bytes differ")
    manifest = destination / "ConfigurationSourcesV2.json"
    with manifest.open("xb") as stream:
        stream.write(plan.selection_json)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--entry", action="append")
    parser.add_argument("--report", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--apply", action="store_true")
    modes.add_argument("--check-selected", action="store_true")
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--expected-plan-sha256")
    args = parser.parse_args()
    registry_path = args.registry.resolve()
    registry_id = relative_path(registry_path.relative_to(args.source_root.resolve()).as_posix())
    original = SourceDocument(registry_id, registry_path.read_bytes())
    spec = RegistryDocument.model_validate(mapping(original))
    registry = load_registry(original, read_sources(args.source_root, spec.sources).documents)
    if args.check_selected:
        defaults, resources = read_preparation_context(registry)
        ids = args.entry or [entry.id for entry in registry.spec.entries if entry.section == "configs"]
        rows = [{"id": id, "prepared_content_sha256": load_selected_configuration_v2(registry, id,
                    source_root=args.source_root, defaults=defaults, resources=resources).content_sha256} for id in ids]
        result = {"schema": "solarlab.selected-configurations-check.v1", "passed": True,
                  "configurations": rows, "can_execute": False, "scientific_gates_awarded": []}
        args.report.write_bytes(_json(result))
        print(json.dumps({"passed": True, "configurations": len(rows), "report": str(args.report)}, indent=2))
        return
    plan = prepare_configuration_migration(registry, ids=tuple(args.entry) if args.entry else None)
    output = {"report": str(args.report), "plan_sha256": plan.sha256, "supported": plan.supported,
              "configurations": len(plan.files), "can_execute": False}
    if args.apply:
        if args.destination is None or args.expected_plan_sha256 is None:
            parser.error("--apply requires --destination and --expected-plan-sha256 from the inspected dry-run")
        output["manifest"] = str(publish_configuration_migration(plan, args.destination,
                                                                expected_plan_sha256=args.expected_plan_sha256))
    args.report.write_bytes(plan.report_json)
    print(json.dumps(output, indent=2))
    if not plan.supported:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
