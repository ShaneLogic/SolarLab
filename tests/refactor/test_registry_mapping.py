"""Complete real registry/data mapping, with every unprovided identity pending."""
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import subprocess

import pytest
import yaml

from solarlab.materials.source import SourceDocument
from solarlab.reproducibility.configuration_migration import prepare_configuration_migration
from solarlab.reproducibility.registry import canonical, load_registry, validate_registry_pair
from solarlab.reproducibility.registry_mapping import prepare_registry_mapping, read_registry_mapping
from solarlab.reproducibility.sources import SourceRef, SourceSet
from test_registry_migration import ROOT, registries as registries


@pytest.fixture(scope="module")
def consumer_sources():
    refs, documents = [], []
    for path in ("src/solarlab/config/migration.py", "src/solarlab/reproducibility/configuration_migration.py"):
        commit = subprocess.run(["git", "log", "-1", "--format=%H", "--", path], cwd=ROOT,
                                check=True, capture_output=True, text=True).stdout.strip()
        content = subprocess.run(["git", "show", commit+":"+path], cwd=ROOT, check=True, capture_output=True).stdout
        assert content == (ROOT/path).read_bytes()
        document = SourceDocument(path, content)
        refs.append(SourceRef(id=path, path=path, sha256=document.sha256, source_commit=commit))
        documents.append(document)
    return SourceSet(tuple(refs), tuple(documents))


@pytest.fixture(scope="module")
def migration(registries):
    return prepare_configuration_migration(registries[0])


@pytest.fixture(scope="module")
def prepared(registries, migration, consumer_sources):
    return prepare_registry_mapping(*registries, migration=migration,
        expected_migration_sha256=migration.sha256, consumer_sources=consumer_sources)


def bind(registries, source, plan_sha256):
    results = []
    for registry in registries:
        data = registry.export()
        data["preparation_mapping"] = {"path": source.id, "sha256": source.sha256,
                                       "migration_plan_sha256": plan_sha256}
        results.append(load_registry(SourceDocument(registry.source.id, yaml.safe_dump(data).encode()), registry.sources.documents))
    return tuple(results)


def read(registries, prepared, migration, consumer_sources, source=None):
    source = prepared.document if source is None else source
    selected = bind(registries, source, migration.sha256)
    return read_registry_mapping(*selected, source, migration=migration,
        expected_migration_sha256=migration.sha256, expected_sha256=source.sha256, consumer_sources=consumer_sources)


def test_complete_original_mapping_has_exact_data_and_honest_context_gaps(registries, migration, prepared):
    value = prepared.export()
    assert value["inventory_counts"] == {"configs": 55, "benchmarks": 44, "lanes": 47}
    assert value["migration_plan_sha256"] == migration.sha256
    assert value["configuration_selection_sha256"] == hashlib.sha256(migration.selection_json).hexdigest()
    assert value["activation"] == "pending_migration" and not value["can_execute"]
    assert value["scientific_gates_awarded"] == []
    original = {entry.id: (registry, entry) for registry in registries for entry in registry.spec.entries}
    for row in value["entries"]:
        registry, entry = original[row["id"]]
        assert row["original_record"] == registry.original(entry.id)
        assert row["original_record_sha256"] == entry.original_record_sha256
        assert row["legacy_evidence"]["semantic_sha256"] == entry.legacy_semantic_sha256
        assert row["old_test_nodes"] == list(entry.test_nodes)
        assert row["physical"] == dict(status="pending", H_input=None, H_physics=None, contexts=[])
        assert row["protocol"]["status"] == "pending" and row["protocol"]["sha256"] is None
        assert row["execution"] == dict(status="pending", H_execution=None, new_executor=None, manifest=None)
        assert not row["can_execute"] and row["new_scientific_test_nodes"] == []
        fields = {item["field"]: item for item in row["missing_fields"]}
        assert set(fields) == {"model_registry", "model_inputs", "selection", "semantic_projections", "driver",
            "effective_behavior", "protocol", "preparation", "initial_state", "grid", "numeric_controls",
            "engine", "runtime_environment", "release_bundle", "execution_manifest", "new_executor_and_tests"}
        assert all(item["status"] == "pending" and item["owner"] and item["source_dependency"] for item in fields.values())
        assert "W03" in fields["driver"]["source_dependency"] and fields["driver"]["owner"] == "WS4"
        assert "P05" in fields["initial_state"]["owner"]
        assert all(item["data_verified"] and item["publication"].startswith("pending") for item in row["configurations"])
        assert all(item["kind"] == "data_preparation_only" and not item["can_execute"] for item in row["preparation_consumers"])
    for registry in registries:
        assert registry.spec.preparation_mapping is None
        assert registry.spec.configuration_selection is None
        assert (ROOT/registry.source.id).read_bytes() == registry.source.content


def test_actual_bound_consumer_rechecks_all_real_inputs_without_activating_any_entry(registries, prepared, migration, consumer_sources, record_property):
    result = read(registries, prepared, migration, consumer_sources)
    assert result.document == prepared.document
    assert result.reference() == {"path": prepared.document.id, "sha256": prepared.sha256, "migration_plan_sha256": migration.sha256}
    with pytest.raises(FrozenInstanceError):
        result.document = SourceDocument("other.json", b"{}")
    row = result.entry("config:configs/scaps_mirror_v2.yaml")
    row["missing_fields"].clear()
    assert result.entry(row["id"])["missing_fields"]
    with pytest.raises(ValueError, match="unknown"):
        result.entry("absent")
    record_property("mapping_check", json.dumps(dict(sha256=result.sha256, rows=146, configurations=55,
        all_real_inputs_consumed=True, exact_mapping_bytes_equal=True, identities_pending=True, activated=False)))


@pytest.mark.parametrize("change", ["missing_entry", "evidence", "configuration", "physics", "protocol", "execution", "missing_field", "test_node"])
def test_relabelled_or_incomplete_mapping_refuses_even_after_metadata_repin(registries, prepared, migration, consumer_sources, change):
    value = prepared.export()
    row = value["entries"][0]
    if change == "missing_entry":
        value["entries"].pop()
    elif change == "evidence":
        row["legacy_evidence"]["claim_level"] = "qualified_new_engine"
    elif change == "configuration":
        row["configurations"][0]["sha256"] = "0" * 64
    elif change == "physics":
        row["physical"].update(status="verified", H_physics="1" * 64)
    elif change == "protocol":
        row["protocol"].update(status="verified", sha256="1" * 64)
    elif change == "execution":
        row["execution"]["new_executor"] = "guessed:run"
    elif change == "missing_field":
        row["missing_fields"].pop()
    else:
        row["old_test_nodes"] = ["invented_test.py::test_qualified"]
    source = SourceDocument(prepared.document.id, canonical(value))
    with pytest.raises(ValueError, match="actual configurations"):
        read(registries, prepared, migration, consumer_sources, source)


def test_two_registry_and_external_source_pins_are_required(registries, prepared, migration, consumer_sources):
    selected = bind(registries, prepared.document, migration.sha256)
    with pytest.raises(ValueError, match="same non-activating"):
        validate_registry_pair(selected[0], registries[1])
    with pytest.raises(ValueError, match="external expected"):
        read_registry_mapping(*selected, prepared.document, migration=migration,
            expected_migration_sha256=migration.sha256, expected_sha256="0"*64, consumer_sources=consumer_sources)
    with pytest.raises(ValueError, match="external expected"):
        read_registry_mapping(*registries, prepared.document, migration=migration,
            expected_migration_sha256=migration.sha256, expected_sha256=prepared.sha256, consumer_sources=consumer_sources)


@pytest.mark.parametrize("change", ["missing", "duplicate", "wrong_path", "changed_input"])
def test_actual_configuration_file_inventory_and_bytes_are_checked(registries, migration, consumer_sources, change):
    files = list(migration.files)
    if change == "missing":
        files.pop()
    elif change == "duplicate":
        files[-1] = files[0]
    elif change == "wrong_path":
        files[0] = ("retargeted.json", files[0][1])
    else:
        files[0] = (files[0][0], replace(files[0][1], content=files[0][1].content+b"\n"))
    wrong = replace(migration, files=tuple(files))
    with pytest.raises(ValueError, match="file paths/content"):
        prepare_registry_mapping(*registries, migration=wrong,
            expected_migration_sha256=migration.sha256, consumer_sources=consumer_sources)


def test_current_reader_source_is_verified_without_a_guessed_commit(registries, migration, consumer_sources):
    missing = SourceSet(consumer_sources.references[:1], consumer_sources.documents[:1])
    with pytest.raises(ValueError, match="exact current"):
        prepare_registry_mapping(*registries, migration=migration,
            expected_migration_sha256=migration.sha256, consumer_sources=missing)
    refs, documents = list(consumer_sources.references), list(consumer_sources.documents)
    documents[0] = replace(documents[0], content=documents[0].content+b"\n")
    refs[0] = refs[0].model_copy(update={"sha256": documents[0].sha256})
    altered = SourceSet(tuple(refs), tuple(documents))
    with pytest.raises(ValueError, match="loaded v2 consumer"):
        prepare_registry_mapping(*registries, migration=migration,
            expected_migration_sha256=migration.sha256, consumer_sources=altered)
