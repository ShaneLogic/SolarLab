"""Actual registered inputs through one forward migration and the strict v2 path."""
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import pytest

from solarlab.config.migration import (
    CONFIGURATION_VERSION, migrate_configuration, prepare_v2_legacy_wire, read_configuration_v2,
)
from solarlab.materials.source import SourceDocument
from solarlab.reproducibility.configuration_migration import (
    prepare_configuration_migration, prepare_registered_configuration_v2, publish_configuration_migration,
    read_selected_configuration_v2,
    load_selected_configuration_v2,
)
from solarlab.reproducibility.preparation import prepare_configuration, read_preparation_context
from solarlab.reproducibility.registry import RegistryDocument, load_registry, mapping
from solarlab.reproducibility.sources import read_sources

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def matrix():
    path = ROOT / "reproducibility/ConfigBenchmarkMatrixV2.yaml"
    source = SourceDocument(str(path.relative_to(ROOT)), path.read_bytes())
    spec = RegistryDocument.model_validate(mapping(source))
    return load_registry(source, read_sources(ROOT, spec.sources).documents)


@pytest.fixture(scope="module")
def context(matrix):
    return read_preparation_context(matrix)


@pytest.fixture(scope="module")
def plan(matrix):
    return prepare_configuration_migration(matrix)


def mutate(document, update):
    data = json.loads(document.content)
    update(data)
    return SourceDocument(document.id, json.dumps(data, allow_nan=False).encode())


def words(value):
    if type(value) is float:
        return [("binary64", struct.pack(">d", value).hex())]
    if isinstance(value, dict):
        return [("key", key) for key in sorted(value)] + [item for key in sorted(value) for item in words(value[key])]
    if isinstance(value, (tuple, list)):
        return [("length", len(value))] + [item for child in value for item in words(child)]
    return [(type(value).__name__, value)]


def test_all_55_real_configurations_retain_preparation_and_exact_reverse_bytes(matrix, context, plan, record_property):
    defaults, resources = context
    assert plan.supported and len(plan.files) == 55
    header = json.loads(plan.selection_json)
    assert header["reader"].endswith(":read_selected_configuration_v2")
    assert header["can_execute"] is False
    rows = {row["path"]: row for row in header["configurations"]}
    reports = {row["id"]: row for row in plan.export()["rows"]}
    total_words = 0
    observed = []
    environment = dict(os.environ)
    for name, document in plan.files:
        assert json.loads(document.content)["schema_version"] == CONFIGURATION_VERSION
        row = rows[name]
        original = prepare_configuration(matrix, row["id"], defaults=defaults, resources=resources).prepared
        assert original is not None
        actual = prepare_registered_configuration_v2(matrix, row["id"], document, defaults=defaults, resources=resources)
        assert words(actual.to_mapping()) == words(original.to_mapping())
        assert actual.input_json == original.input_json
        count = sum(kind == "binary64" for kind, _ in words(actual.to_mapping()))
        total_words += count
        inverse = prepare_v2_legacy_wire(document, defaults=defaults, resources=resources, sources=matrix.sources.documents)
        source = matrix.sources.document(row["original_source"]["id"])
        assert inverse.supported and inverse.document == source and not inverse.can_execute
        for _, reference in inverse.references:
            assert reference == matrix.sources.document(reference.id)
        entry = matrix.entry(row["id"])
        assert reports[entry.id]["legacy_claim_level"] == entry.legacy_claim_level
        assert reports[entry.id]["legacy_status"] == entry.legacy_status
        assert not actual.can_execute
        observed.append({"id": entry.id, "prepared_sha256": actual.content_sha256, "binary64_words": count,
                         "v2_sha256": document.sha256, "original_wire_sha256": source.sha256})
    assert dict(os.environ) == environment and total_words > 5000
    record_property("unchanged_binary64_words", total_words)
    target = os.environ.get("SOLARLAB_MIGRATION_EVIDENCE")
    if target:
        Path(target).write_text(json.dumps({"schema": "solarlab.configuration-migration-M.v1", "passed": True,
            "configurations": len(observed), "binary64_words": total_words, "rows": observed,
            "old_driver_calls": 0, "scientific_gates_awarded": []}, indent=2) + "\n")


def test_v2_entry_never_falls_back_to_a_v1_importer(matrix, context, plan, monkeypatch):
    import solarlab.config.migration as migration
    import solarlab.reproducibility.preparation as preparation
    def refused(*args, **kwargs):
        raise AssertionError("v1 importer was reached by the formal v2 entry")
    for name in ("import_standard_device", "import_scaps_device", "import_tandem"):
        monkeypatch.setattr(migration, name, refused)
    monkeypatch.setattr(preparation, "prepare_original", refused)
    defaults, resources = context
    row = json.loads(plan.selection_json)["configurations"][0]
    actual = prepare_registered_configuration_v2(matrix, row["id"], plan.files[0][1], defaults=defaults, resources=resources)
    assert actual.content_sha256 == row["prepared_content_sha256"]
    with pytest.raises(ValueError):
        read_configuration_v2(matrix.sources.document(row["original_source"]["id"]),
                              defaults=defaults, resources=resources, sources=matrix.sources.documents)


@pytest.mark.parametrize("mutation", ("schema", "extra", "source", "defaults", "resources", "missing_sources", "duplicate_source", "reference"))
def test_v2_refuses_changed_bindings_and_unknown_fields(matrix, context, plan, mutation):
    changes = {
        "schema": lambda d: d.update(schema_version="solarlab.configuration.v1"),
        "extra": lambda d: d.update(allow_fallback=True),
        "source": lambda d: d["source"].update(sha256="0" * 64),
        "defaults": lambda d: d.update(default_catalog_sha256="0" * 64),
        "resources": lambda d: d.update(resource_library_sha256="0" * 64),
        "missing_sources": lambda d: d.update(sources=[]),
        "duplicate_source": lambda d: d["sources"].append(d["source"]),
        "reference": lambda d: d["references"].update(unexpected=d["source"]),
    }
    document = mutate(plan.files[0][1], changes[mutation])
    with pytest.raises(ValueError):
        read_configuration_v2(document, defaults=context[0], resources=context[1], sources=matrix.sources.documents)


def test_duplicate_and_nonfinite_json_and_wrong_trusted_source_are_refused(matrix, context, plan):
    source = plan.files[0][1]
    for raw in (source.content[:-1] + b',"schema_version":"solarlab.configuration.v2"}',
                source.content[:-1] + b',"unexpected":NaN}'):
        with pytest.raises(ValueError):
            read_configuration_v2(SourceDocument(source.id, raw), defaults=context[0], resources=context[1], sources=matrix.sources.documents)
    documents = tuple(replace(item, content=item.content+b"\n") if item.id == json.loads(source.content)["source"]["id"] else item
                      for item in matrix.sources.documents)
    with pytest.raises(ValueError, match="source content"):
        read_configuration_v2(source, defaults=context[0], resources=context[1], sources=documents)


def test_changes_cannot_inherit_the_original_registered_materialization(matrix, context, plan):
    row = json.loads(plan.selection_json)["configurations"][0]
    changed = mutate(plan.files[0][1], lambda d: d["input"]["settings"].update(T="310 K"))
    actual = read_configuration_v2(changed, defaults=context[0], resources=context[1], sources=matrix.sources.documents)
    assert actual.content_sha256 != row["prepared_content_sha256"]
    with pytest.raises(ValueError, match="differs from the original"):
        prepare_registered_configuration_v2(matrix, row["id"], changed, defaults=context[0], resources=context[1])


def test_failed_dry_run_has_no_partial_v2_configuration(context):
    source = SourceDocument("invalid_original", b"device:\n  T: false\nlayers: []\n")
    result = migrate_configuration(source, format="standard-device-v1", input_id="invalid", defaults=context[0], resources=context[1])
    assert not result.supported and result.document is None and result.export()["issues"]
    assert not result.can_execute


def test_publication_is_explicit_exact_and_does_not_overwrite(matrix, context, plan, tmp_path):
    destination = tmp_path / "not_authorizing_scientific_execution"
    with pytest.raises(ValueError, match="matching dry-run"):
        publish_configuration_migration(plan, destination, expected_plan_sha256="0" * 64)
    assert not destination.exists()
    with pytest.raises(ValueError, match="content binding"):
        changed = replace(plan, files=((plan.files[0][0], replace(plan.files[0][1], content=b"{}")), *plan.files[1:]))
        publish_configuration_migration(changed, destination, expected_plan_sha256=changed.sha256)
    assert not destination.exists()
    selected_id = json.loads(plan.selection_json)["configurations"][0]["id"]
    single = prepare_configuration_migration(matrix, ids=(selected_id,))
    manifest = publish_configuration_migration(single, destination, expected_plan_sha256=single.sha256)
    assert manifest.read_bytes() == single.selection_json
    for path, document in single.files:
        assert (destination / path).read_bytes() == document.content
    selected = SourceDocument("actual_temporary_selection", manifest.read_bytes())
    documents = tuple(SourceDocument(document.id, (destination / path).read_bytes()) for path, document in single.files)
    actual = read_selected_configuration_v2(matrix, selected, selected_id, documents=documents,
                                            defaults=context[0], resources=context[1])
    assert actual.content_sha256 == json.loads(single.selection_json)["configurations"][0]["prepared_content_sha256"]
    broken = mutate(selected, lambda value: value["configurations"][0].update(sha256="0" * 64))
    with pytest.raises(ValueError, match="content binding"):
        read_selected_configuration_v2(matrix, broken, selected_id, documents=documents,
                                        defaults=context[0], resources=context[1])
    with pytest.raises(ValueError, match="new explicit"):
        publish_configuration_migration(single, destination, expected_plan_sha256=single.sha256)
    with pytest.raises(FrozenInstanceError):
        single.files = ()
    with pytest.raises(ValueError, match="unique existing"):
        prepare_configuration_migration(matrix, ids=(selected_id, selected_id))


def test_real_cli_dry_run_and_isolated_metadata_import(tmp_path):
    report = tmp_path / "dry-run.json"
    argv = [sys.executable, "-I", "-B", "-m", "solarlab.reproducibility.configuration_migration",
            "--registry", str(ROOT / "reproducibility/ConfigBenchmarkMatrixV2.yaml"),
            "--source-root", str(ROOT), "--dry-run", "--report", str(report)]
    result = subprocess.run(argv, cwd=tmp_path, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["supported"] and receipt["configurations"] == 55 and not receipt["can_execute"]
    assert hashlib.sha256(report.read_bytes()).hexdigest() == receipt["plan_sha256"]
    assert set(tmp_path.iterdir()) == {report}
    isolated = subprocess.run([sys.executable, "-I", "-B", "-c",
        "import sys; import solarlab.config.migration; import solarlab.reproducibility.configuration_migration; assert not set(sys.modules) & {'perovskite_sim','scipy','sksundae','flint','solarlab_server'}"],
        cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert isolated.returncode == 0, isolated.stdout + isolated.stderr


def test_actual_matrix_selection_uses_new_files_and_refuses_the_old_reader(matrix, context, plan, tmp_path):
    selected_id = json.loads(plan.selection_json)["configurations"][0]["id"]
    single = prepare_configuration_migration(matrix, ids=(selected_id,))
    destination = tmp_path / "ConfigurationsV2"
    manifest = publish_configuration_migration(single, destination, expected_plan_sha256=single.sha256)
    data = matrix.export()
    data["configuration_selection"] = {"path": manifest.relative_to(tmp_path).as_posix(),
        "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(), "migration_plan_sha256": single.sha256}
    candidate = load_registry(SourceDocument("non_authorizing_switch_fixture", json.dumps(data).encode()), matrix.sources.documents)
    actual = load_selected_configuration_v2(candidate, selected_id, source_root=tmp_path, defaults=context[0], resources=context[1])
    assert actual.content_sha256 == matrix.entry(selected_id).preparation.content_sha256
    with pytest.raises(ValueError, match="formal v2 data is selected"):
        prepare_configuration(candidate, selected_id, defaults=context[0], resources=context[1])
    path, _ = single.files[0]
    (destination / path).write_bytes(b"{}")
    with pytest.raises(ValueError, match="content binding"):
        load_selected_configuration_v2(candidate, selected_id, source_root=tmp_path, defaults=context[0], resources=context[1])


def test_exact_selected_input_reconciles_derived_metadata_without_relabelling(matrix, context, plan, monkeypatch):
    import solarlab.reproducibility.configuration_migration as module
    from types import SimpleNamespace
    row = json.loads(plan.selection_json)["configurations"][0]
    document = next(doc for _, doc in plan.files if doc.id == row["document_id"])
    original = document.content
    actual = module.read_configuration_v2(document, defaults=context[0], resources=context[1], sources=matrix.sources.documents)
    derived = SimpleNamespace(content_sha256="0" * 64, to_input=actual.to_input)
    monkeypatch.setattr(module, "read_configuration_v2", lambda *a, **k: derived)
    comparison = {}
    with pytest.warns(RuntimeWarning, match="derived preparation metadata"):
        reopened = module.read_selected_configuration_v2(matrix, SourceDocument("selection", plan.selection_json), row["id"],
            documents=(document,), defaults=context[0], resources=context[1], reconciliation=comparison)
    assert reopened is derived and comparison["status"] == "different"
    assert comparison["recorded_prepared_sha256"] == row["prepared_content_sha256"] and document.content == original
    with pytest.raises(ValueError, match="differs from the original"):
        module.read_selected_configuration_v2(matrix, SourceDocument("selection", plan.selection_json), row["id"],
            documents=(document,), defaults=context[0], resources=context[1], strict_derived_metadata=True)


def test_scaps_and_tandem_migration_surface_declared_vs_effective_optical_coherence_without_changing_words(
    matrix, context, plan,
):
    import yaml

    defaults, resources = context
    rows = plan.export()["rows"]
    scaps_rows = [
        row for row in rows
        if row["migration"]["format"] == "scaps-device-v1" and row["migration"]["diagnostics"]
    ]
    assert scaps_rows
    for row in scaps_rows:
        mig = row["migration"]
        diag = next(item for item in mig["diagnostics"] if item["code"] == "legacy_scaps_ignored_incoherent")
        assert diag["declared"] is True and diag["historical_effective"] is False
        glass = mig["optical_coherence"][0]
        assert glass["declared_present"] is True and glass["declared"] is True
        assert glass["normalized_configured"] is True and glass["effective"] is False
        assert glass["historical_loader_value"] is False and glass["ignored_by_legacy_loader"] is True
        assert glass["prospective_fix_owner"] == "P08-04"
        v2_doc = next(doc for _, doc in plan.files if doc.id == mig["proposed"]["id"])
        original = prepare_configuration(matrix, row["id"], defaults=defaults, resources=resources).prepared
        migrated = read_configuration_v2(v2_doc, defaults=defaults, resources=resources, sources=matrix.sources.documents)
        assert words(migrated.to_mapping()) == words(original.to_mapping())

    tandem_rows = [row for row in rows if row["migration"]["format"] == "tandem-v1"]
    assert tandem_rows
    for row in tandem_rows:
        assert set(row["migration"]["optical_coherence"]) == {"top_cell", "bottom_cell"}

    # Verify subcell diagnostic and optical_coherence propagation on tandem migration.
    from pathlib import PurePosixPath
    import posixpath

    tandem_entry = matrix.entry(tandem_rows[0]["id"])
    tandem_src = matrix.sources.at_path(matrix.spec.original_path_root + "/" + tandem_entry.original_key)
    tandem_prepared = prepare_configuration(matrix, tandem_entry.id, defaults=defaults, resources=resources).prepared
    assert tandem_prepared is not None
    parent = PurePosixPath(matrix.sources.reference(tandem_src.id).path).parent
    tandem_decl = tandem_prepared.to_input()
    tandem_refs = {
        name: matrix.sources.at_path(posixpath.normpath(str(parent / name)))
        for name in (tandem_decl.top_cell_reference, tandem_decl.bottom_cell_reference)
    }
    top_key = tandem_decl.top_cell_reference
    top_wire = yaml.safe_load(tandem_refs[top_key].content)
    top_wire["simulation_hints"] = {"jv_sweep": {"v_rate": 1.0}}
    top_wire["layers"][0]["incoherent"] = True
    synthetic_refs = {
        **tandem_refs,
        top_key: SourceDocument(top_key, yaml.safe_dump(top_wire, sort_keys=False).encode()),
    }
    tandem_mig = migrate_configuration(
        tandem_src, format="tandem-v1", input_id="tandem_subcell_diag",
        defaults=defaults, resources=resources, references=synthetic_refs,
    )
    assert tandem_mig.supported
    exported = tandem_mig.export()
    assert any(item.get("cell") == "top_cell" for item in exported["diagnostics"])
    top_coherence = exported["optical_coherence"]["top_cell"][0]
    assert top_coherence["declared_present"] is True and top_coherence["declared"] is True
    assert top_coherence["effective"] is True and top_coherence["ignored_by_legacy_loader"] is False


def test_scaps_bulk_defect_edits_multi_defect_promotion_and_conflict_guards_in_legacy_wire(context):
    from copy import deepcopy
    import yaml

    from solarlab.config.legacy_wire import prepare_legacy_wire
    from solarlab.config.scaps_input import import_scaps_device
    from solarlab.device.inputs import DeviceInput

    defaults, _ = context
    v2_path = ROOT / "perovskite-sim/configs/scaps_mirror_v2.yaml"
    scaps_src = SourceDocument(str(v2_path.relative_to(ROOT)), v2_path.read_bytes())
    base_input = import_scaps_device(scaps_src, id="scaps_edit", defaults=defaults)

    # 1. Single-field & Gaussian promotion edits on bulk_defects without manually editing scaps_defect_metadata.
    edited_data = deepcopy(base_input.editing_data())
    edited_data["layers"][2]["bulk_defects"][0]["energy_level"] = {
        "reference": "above_valence_band",
        "value_eV": 0.55,
    }
    edited_data["layers"][2]["bulk_defects"][0]["kinetics"]["sigma_n_m2"] = "2e-15 cm^2"
    edited_data["layers"][2]["bulk_defects"][1]["distribution"] = {
        "kind": "gaussian",
        "normalization": "integrated_total",
        "total_density_m3": "1e12 cm^-3",
        "width_eV": 0.08,
        "width_convention": "scaps_characteristic_energy",
    }
    wire_res = prepare_legacy_wire(DeviceInput.model_validate(edited_data), source=scaps_src, defaults=defaults)
    assert wire_res.supported and wire_res.document is not None
    reopened = import_scaps_device(wire_res.document, id="scaps_edit", defaults=defaults)
    meta = reopened.layers[2].scaps_defect_metadata
    assert meta[0].energy_reference == "above_valence_band" and meta[0].trap_depth_eV == 0.55
    assert meta[1].distribution == "gaussian" and meta[1].E_char_eV == 0.08
    assert wire_res.export()["optical_coherence"][0]["ignored_by_legacy_loader"] is True

    # 2. Singular `bulk_defect` -> plural `bulk_defects` multi-defect promotion and clearing all bulk defects.
    singular_yaml = yaml.safe_load(scaps_src.content)
    first_defect = singular_yaml["layers"][2].pop("bulk_defects")[0]
    singular_yaml["layers"][2]["bulk_defect"] = first_defect
    singular_src = SourceDocument("singular_scaps.yaml", yaml.safe_dump(singular_yaml, sort_keys=False).encode())
    singular_in = import_scaps_device(singular_src, id="singular", defaults=defaults)
    promoted = deepcopy(singular_in.editing_data())
    second_defect = deepcopy(promoted["layers"][2]["bulk_defects"][0])
    second_defect["id"] = "bulk_1"
    second_defect["name"] = "Perovskite-VB-2"
    second_defect["energy_level"] = {"reference": "above_valence_band", "value_eV": 0.25}
    promoted["layers"][2]["bulk_defects"] = (*promoted["layers"][2]["bulk_defects"], second_defect)
    promoted_wire = prepare_legacy_wire(DeviceInput.model_validate(promoted), source=singular_src, defaults=defaults)
    assert promoted_wire.supported and promoted_wire.document is not None
    promoted_parsed = yaml.safe_load(promoted_wire.document.content)
    assert "bulk_defect" not in promoted_parsed["layers"][2]
    assert len(promoted_parsed["layers"][2]["bulk_defects"]) == 2

    cleared = deepcopy(singular_in.editing_data())
    cleared["layers"][2].pop("bulk_defects", None)
    cleared["layers"][2].pop("scaps_defect_metadata", None)
    cleared["layers"][2].pop("defect_model", None)
    cleared["layers"][2].pop("defect_schema_version", None)
    cleared_wire = prepare_legacy_wire(DeviceInput.model_validate(cleared), source=singular_src, defaults=defaults)
    assert cleared_wire.supported and cleared_wire.document is not None
    cleared_parsed = yaml.safe_load(cleared_wire.document.content)
    assert "bulk_defect" not in cleared_parsed["layers"][2] and "bulk_defects" not in cleared_parsed["layers"][2]

    # 3. Independent scaps_defect_metadata N_peak_m3 edit patches N_peak_cm3 cleanly; conflicts fail closed.
    meta_edit = deepcopy(base_input.editing_data())
    meta_edit["layers"][2]["scaps_defect_metadata"][0]["N_peak_m3"] = "5e14 cm^-3"
    meta_wire = prepare_legacy_wire(DeviceInput.model_validate(meta_edit), source=scaps_src, defaults=defaults)
    assert meta_wire.supported and meta_wire.document is not None
    assert yaml.safe_load(meta_wire.document.content)["layers"][2]["bulk_defects"][0]["N_peak_cm3"] == 5e14

    conflict = deepcopy(base_input.editing_data())
    conflict["layers"][2]["bulk_defects"][0]["energy_level"]["value_eV"] = 0.45
    conflict["layers"][2]["scaps_defect_metadata"][0]["trap_depth_eV"] = 0.30
    conflict_wire = prepare_legacy_wire(DeviceInput.model_validate(conflict), source=scaps_src, defaults=defaults)
    assert not conflict_wire.supported and conflict_wire.document is None
    assert conflict_wire.export()["unsupported"][0]["code"] == "scaps_partner_metadata_conflict"

    bad_id = deepcopy(base_input.editing_data())
    bad_id["layers"][2]["bulk_defects"][0]["id"] = "bulk_9"
    bad_id_wire = prepare_legacy_wire(DeviceInput.model_validate(bad_id), source=scaps_src, defaults=defaults)
    assert not bad_id_wire.supported and bad_id_wire.export()["unsupported"][0]["code"] == "scaps_bulk_defect_id"

    bad_vth = deepcopy(base_input.editing_data())
    bad_vth["layers"][2]["parameters"]["v_th"] = "2e7 cm/s"
    bad_vth_wire = prepare_legacy_wire(DeviceInput.model_validate(bad_vth), source=scaps_src, defaults=defaults)
    assert not bad_vth_wire.supported and bad_vth_wire.export()["unsupported"][0]["code"] == "scaps_bulk_representation"


def test_duplicate_layer_display_names_edit_via_stable_ids_without_collateral_mutation(context):
    from copy import deepcopy
    import yaml

    from solarlab.config.legacy_wire import prepare_legacy_wire
    from solarlab.config.scaps_input import import_scaps_device
    from solarlab.device.inputs import DeviceInput

    defaults, _ = context
    v2_path = ROOT / "perovskite-sim/configs/scaps_mirror_v2.yaml"
    parsed = yaml.safe_load(v2_path.read_bytes())
    # Give two distinct layers the exact same display name while editing only one via stable layer_id / interface_id.
    parsed["layers"][1]["name"] = "shared_transport"
    parsed["layers"][3]["name"] = "shared_transport"
    dup_src = SourceDocument("duplicate_names_scaps.yaml", yaml.safe_dump(parsed, sort_keys=False).encode())
    imported = import_scaps_device(dup_src, id="dup_scaps", defaults=defaults)
    assert [layer.id for layer in imported.layers] == ["layer_0", "layer_1", "layer_2", "layer_3"]
    assert imported.layers[1].name == imported.layers[3].name == "shared_transport"

    edited = deepcopy(imported.editing_data())
    edited["layers"][3]["thickness"] = "42 nm"
    edited["interfaces"][2]["defect"]["total_density_m2"] = "3e12 cm^-2"
    wire_res = prepare_legacy_wire(DeviceInput.model_validate(edited), source=dup_src, defaults=defaults)
    assert wire_res.supported and wire_res.document is not None
    emitted = yaml.safe_load(wire_res.document.content)
    assert emitted["layers"][1]["thickness_nm"] == 20.0
    assert emitted["layers"][3]["thickness_nm"] == 42.0
    assert emitted["interfaces"][0]["N_t_cm2"] == 1.0e12
    assert emitted["interfaces"][1]["N_t_cm2"] == 3.0e12

