"""Configuration v1->v2 forward migration, reverse legacy wire, and strict v2 reader."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import struct
import subprocess
import sys

import pytest
import yaml

from solarlab.config.device_import import import_standard_device
from solarlab.config.legacy_wire import prepare_legacy_wire
from solarlab.config.migration import (
    CONFIGURATION_VERSION,
    migrate_configuration,
    prepare_v2_legacy_wire,
    read_configuration_v2,
)
from solarlab.config.resolve_device import resolve_device, resolve_tandem
from solarlab.config.scaps_input import import_scaps_device
from solarlab.config.tandem_input import import_tandem
from solarlab.device.inputs import DeviceInput
from solarlab.materials.packaged import (
    load_packaged_default_catalog,
    load_packaged_resource_library,
)
from solarlab.materials.source import SourceDocument

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures/configs"
PRESETS = ROOT / "src/solarlab/materials/data/presets"


@pytest.fixture(scope="module")
def context():
    return (
        load_packaged_default_catalog("matrix_v2"),
        load_packaged_resource_library(source_id_prefix="src/solarlab/materials/data"),
    )


def mutate(document, update):
    data = json.loads(document.content)
    update(data)
    return SourceDocument(document.id, json.dumps(data, allow_nan=False).encode())


def words(value):
    if type(value) is float:
        return [("binary64", struct.pack(">d", value).hex())]
    if isinstance(value, dict):
        return [("key", key) for key in sorted(value)] + [
            item for key in sorted(value) for item in words(value[key])
        ]
    if isinstance(value, (tuple, list)):
        return [("length", len(value))] + [item for child in value for item in words(child)]
    return [(type(value).__name__, value)]


def test_standard_scaps_and_tandem_forward_v2_and_exact_reverse_wire(context):
    defaults, resources = context
    std_path = FIXTURES / "nip_MAPbI3.yaml"
    std_src = SourceDocument("nip_MAPbI3.yaml", std_path.read_bytes())
    std_mig = migrate_configuration(
        std_src, format="standard-device-v1", input_id="nip_MAPbI3",
        defaults=defaults, resources=resources,
    )
    assert std_mig.supported and std_mig.document is not None and not std_mig.can_execute
    assert json.loads(std_mig.document.content)["schema_version"] == CONFIGURATION_VERSION
    std_orig = resolve_device(
        import_standard_device(std_src, id="nip_MAPbI3", defaults=defaults),
        defaults, resources, sources=(std_src,),
    )
    std_v2 = read_configuration_v2(
        std_mig.document, defaults=defaults, resources=resources, sources=(std_src,),
    )
    assert words(std_v2.to_mapping()) == words(std_orig.to_mapping())
    std_inv = prepare_v2_legacy_wire(
        std_mig.document, defaults=defaults, resources=resources, sources=(std_src,),
    )
    assert std_inv.supported and std_inv.document == std_src

    scaps_path = PRESETS / "scaps_mirror_v2.yaml"
    scaps_src = SourceDocument("scaps_mirror_v2.yaml", scaps_path.read_bytes())
    scaps_mig = migrate_configuration(
        scaps_src, format="scaps-device-v1", input_id="scaps_v2",
        defaults=defaults, resources=resources,
    )
    assert scaps_mig.supported and scaps_mig.document is not None
    glass = scaps_mig.export()["optical_coherence"][0]
    assert glass["declared"] is True and glass["ignored_by_legacy_loader"] is True
    scaps_orig = resolve_device(
        import_scaps_device(scaps_src, id="scaps_v2", defaults=defaults),
        defaults, resources, sources=(scaps_src,),
    )
    scaps_v2 = read_configuration_v2(
        scaps_mig.document, defaults=defaults, resources=resources, sources=(scaps_src,),
    )
    assert words(scaps_v2.to_mapping()) == words(scaps_orig.to_mapping())

    tandem_src = SourceDocument("tandem_lin2019.yaml", (FIXTURES / "tandem_lin2019.yaml").read_bytes())
    refs = {
        name: SourceDocument(name, (FIXTURES / name).read_bytes())
        for name in ("nip_wideGap_FACs_1p77.yaml", "nip_SnPb_1p22.yaml")
    }
    tandem_mig = migrate_configuration(
        tandem_src, format="tandem-v1", input_id="tandem_lin2019",
        defaults=defaults, resources=resources, references=refs,
    )
    assert tandem_mig.supported and tandem_mig.document is not None
    tandem_orig = resolve_tandem(
        import_tandem(tandem_src, id="tandem_lin2019", references=refs, defaults=defaults),
        defaults, resources, sources=(tandem_src, *refs.values()),
    )
    tandem_v2 = read_configuration_v2(
        tandem_mig.document, defaults=defaults, resources=resources,
        sources=(tandem_src, *refs.values()),
    )
    assert words(tandem_v2.to_mapping()) == words(tandem_orig.to_mapping())


@pytest.mark.parametrize(
    "mutation",
    ("schema", "extra", "source", "defaults", "resources", "missing_sources", "duplicate_source", "reference"),
)
def test_v2_refuses_changed_bindings_and_unknown_fields(context, mutation):
    defaults, resources = context
    src = SourceDocument("nip_MAPbI3.yaml", (FIXTURES / "nip_MAPbI3.yaml").read_bytes())
    v2_doc = migrate_configuration(
        src, format="standard-device-v1", input_id="nip_MAPbI3",
        defaults=defaults, resources=resources,
    ).document
    assert v2_doc is not None
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
    document = mutate(v2_doc, changes[mutation])
    with pytest.raises(ValueError):
        read_configuration_v2(document, defaults=defaults, resources=resources, sources=(src,))


def test_duplicate_and_nonfinite_json_and_wrong_trusted_source_are_refused(context):
    defaults, resources = context
    src = SourceDocument("nip_MAPbI3.yaml", (FIXTURES / "nip_MAPbI3.yaml").read_bytes())
    source = migrate_configuration(
        src, format="standard-device-v1", input_id="nip_MAPbI3",
        defaults=defaults, resources=resources,
    ).document
    assert source is not None
    for raw in (
        source.content[:-1] + b',"schema_version":"solarlab.configuration.v2"}',
        source.content[:-1] + b',"unexpected":NaN}',
    ):
        with pytest.raises(ValueError):
            read_configuration_v2(
                SourceDocument(source.id, raw), defaults=defaults, resources=resources, sources=(src,),
            )
    with pytest.raises(ValueError, match="source content"):
        read_configuration_v2(
            source, defaults=defaults, resources=resources,
            sources=(replace(src, content=src.content + b"\n"),),
        )


def test_failed_dry_run_has_no_partial_v2_configuration(context):
    source = SourceDocument("invalid_original", b"device:\n  T: false\nlayers: []\n")
    result = migrate_configuration(
        source, format="standard-device-v1", input_id="invalid",
        defaults=context[0], resources=context[1],
    )
    assert not result.supported and result.document is None and result.export()["issues"]
    assert not result.can_execute


def test_isolated_metadata_import(tmp_path):
    isolated = subprocess.run(
        [
            sys.executable, "-I", "-B", "-c",
            "import sys; import solarlab.config.migration; import solarlab.reproducibility.configuration_migration; "
            "assert not set(sys.modules) & {'perovskite_sim','scipy','sksundae','flint','solarlab_server'}",
        ],
        cwd=tmp_path, capture_output=True, text=True, timeout=10,
    )
    assert isolated.returncode == 0, isolated.stdout + isolated.stderr


def test_scaps_bulk_defect_edits_multi_defect_promotion_and_conflict_guards_in_legacy_wire(context):
    defaults, _ = context
    v2_path = PRESETS / "scaps_mirror_v2.yaml"
    scaps_src = SourceDocument(str(v2_path.relative_to(ROOT)), v2_path.read_bytes())
    base_input = import_scaps_device(scaps_src, id="scaps_edit", defaults=defaults)

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
    defaults, _ = context
    v2_path = PRESETS / "scaps_mirror_v2.yaml"
    parsed = yaml.safe_load(v2_path.read_bytes())
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
