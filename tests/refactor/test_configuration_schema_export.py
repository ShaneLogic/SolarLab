"""Source fidelity and metadata-only generation; no device or solver run."""

from dataclasses import replace
from hashlib import sha256
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from solarlab.config.schema import (
    configuration_schema_representation, export_configuration_schema,
)
from solarlab.config.yaml import load_yaml_mapping
from solarlab.device.settings import DeviceSettingsInput
from solarlab.device.tunnelling import TunnellingInput
from solarlab.materials.full_parameters import FullParameterInput
from solarlab.materials.optics import CigsOpticsInput
from solarlab.materials.parameter_schema import parameter_schema
from solarlab.units import UNIT_SCHEMA_VERSION, supported_units

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "perovskite-sim/tests/fixtures/configs"


def check_local_references(document):
    def resolve(reference):
        assert reference.startswith("#/"), reference
        value = document
        for part in reference[2:].split("/"):
            value = value[part.replace("~1", "/").replace("~0", "~")]
        assert isinstance(value, dict)

    def walk(value):
        if isinstance(value, dict):
            if "$ref" in value:
                resolve(value["$ref"])
            for reference in value.get("discriminator", {}).get("mapping", {}).values():
                resolve(reference)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(document)


def test_complete_schemas_equal_actual_dtos_with_independent_reference_scope():
    exported = export_configuration_schema()
    assert len(exported["dto_schemas"]) == 19
    for entry in exported["dto_schemas"].values():
        module, name = entry["python_type"].split(":")
        dto = getattr(importlib.import_module(module), name)
        assert entry["schema"] == dto.model_json_schema(mode="validation")
        assert entry["representation"] == "editable_input"
        check_local_references(entry["schema"])
    device = exported["dto_schemas"]["DeviceInput"]["schema"]
    tandem = exported["dto_schemas"]["TandemInput"]["schema"]
    assert device["properties"]["layers"]["items"]["$ref"] == "#/$defs/FullLayerInput"
    assert tandem["properties"]["top_cell"]["$ref"] == "#/$defs/DeviceInput"
    assert len(device["$defs"]["FullLayerInput"]["properties"]["bulk_defects"]["items"]["anyOf"]) == 2


def test_existing_parameter_metadata_units_and_detached_exports():
    exported = export_configuration_schema()
    for name in ("scalar_layer", "full_layer"):
        assert exported["parameter_schemas"][name] == parameter_schema(name)
        check_local_references(exported["parameter_schemas"][name])
    assert exported["unit_schema_version"] == UNIT_SCHEMA_VERSION
    for unit, spellings in exported["unit_spellings"].items():
        assert spellings == list(supported_units(unit))
    assert "cm^-3" in exported["unit_spellings"]["m^-3"]
    exported["dto_schemas"]["DeviceInput"]["schema"]["$defs"].clear()
    assert export_configuration_schema()["dto_schemas"]["DeviceInput"]["schema"]["$defs"]


def test_compact_representation_has_exact_identity_and_detached_payload():
    data, digest = configuration_schema_representation()
    decoded = json.loads(data)
    assert decoded == export_configuration_schema()
    assert isinstance(data, bytes)
    assert data == json.dumps(decoded, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=False, allow_nan=False).encode("utf-8")
    assert digest == sha256(data).hexdigest()
    decoded["dto_schemas"]["DeviceInput"]["schema"]["$defs"].clear()
    assert configuration_schema_representation() == (data, digest)


def test_omission_null_zero_false_and_signed_unit_semantics():
    fields = export_configuration_schema()["dto_schemas"]["FullParameterInput"]["schema"]["properties"]
    assert "default" not in fields["mu_n"]
    assert {"type": "null"} not in fields["mu_n"]["anyOf"]
    assert {"type": "null"} in fields["Nc300"]["anyOf"]
    assert fields["mu_n"]["unit"] == "m^2/(V s)"
    assert FullParameterInput().editing_data() == {}
    values = FullParameterInput.model_validate({"mu_n": 0, "incoherent": False, "Nc300": None})
    assert values.editing_data() == {"mu_n": 0, "incoherent": False, "Nc300": None}
    assert dict(values.normalized_items()) == {"mu_n": 0.0, "incoherent": False, "Nc300": None}
    with pytest.raises(ValueError, match="null"):
        FullParameterInput.model_validate({"mu_n": None})
    assert dict(DeviceSettingsInput(phi_left="-20 mV").normalized_items())["phi_left"] == -0.02


def test_real_structured_fixture_inputs_and_backend_constraints():
    cigs = load_yaml_mapping((FIXTURES / "cigs_graded_optics.yaml").read_bytes())
    original = cigs["layers"][2]["cigs_graded_optics"]
    model = CigsOpticsInput.model_validate(original)
    assert model.editing_data() == original
    assert model.slices == original["slices"]
    with pytest.raises(ValueError, match="cgi"):
        model.validated_update({"cgi": "0.5"})
    wkb = load_yaml_mapping((FIXTURES / "wkb_resolved_electron_barrier.yaml").read_bytes())
    channels = TunnellingInput.model_validate(wkb["device"]["tunnelling_channels"])
    assert channels.editing_data() == wkb["device"]["tunnelling_channels"]
    assert channels.intraband.enabled is True
    zero = channels.validated_update({"contact": {"enabled": False, "barrier_height_eV": 0}})
    assert zero.normalized_data()["contact"] == {"enabled": False, "barrier_height_eV": 0.0}
    with pytest.raises(ValueError, match="positive barrier"):
        channels.validated_update({"contact": {"enabled": True, "barrier_height_eV": 0}})


def test_supplied_structured_schema_exports_existing_normalization_and_identity():
    # Reuse the reviewed schema-only producer, without constructing a prepared
    # device or invoking its resolver. No second normalization/default table.
    from solarlab.config.structured_parameters import _schema

    schema = _schema(TunnellingInput, "tunnelling_channels", "device",
                     TunnellingInput.model_fields["schema_version"].default)
    expected = schema.export()
    result = export_configuration_schema(structured_schemas=(schema, schema))
    data, _ = configuration_schema_representation(structured_schemas=(schema, schema))
    assert json.loads(data) == result
    key = f"{schema.id}@{schema.version}"
    assert result["structured_parameter_schemas"] == {key: expected}
    for name in ("input_schema", "normalized_parameter_schema"):
        check_local_references(result["structured_parameter_schemas"][key][name])
    assert expected["input_schema"] != expected["normalized_parameter_schema"]
    assert expected["source"]["sha256"] == schema.source.sha256
    normalized = expected["normalized_parameter_schema"]["$defs"]["IntrabandInput"]
    assert normalized["properties"]["electron_effective_mass_rel"]["exclusiveMinimum"] == 0
    result["structured_parameter_schemas"][key]["input_schema"]["$defs"].clear()
    assert schema.export() == expected
    with pytest.raises(ValueError, match="conflicting structured schema"):
        export_configuration_schema(structured_schemas=(schema, replace(schema, document_version="conflict")))


def test_generation_is_deterministic_and_check_rejects_stale_or_missing(tmp_path, capsys):
    path = ROOT / "scripts/generate_configuration_schema.py"
    spec = importlib.util.spec_from_file_location("schema_generation", path)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    first = generator.render()
    assert first == generator.render() == generator.DEFAULT_OUTPUT.read_bytes()
    output = tmp_path / "configuration-schema.ts"
    assert generator.main(["--output", str(output)]) == 0
    assert output.read_bytes() == first
    assert generator.main(["--check", "--output", str(output)]) == 0
    output.write_bytes(first + b"// stale\n")
    assert generator.main(["--check", "--output", str(output)]) == 1
    assert output.read_bytes() == first + b"// stale\n"
    missing = tmp_path / "missing.ts"
    assert generator.main(["--check", "--output", str(missing)]) == 1
    assert not missing.exists()
    assert "stale or missing" in capsys.readouterr().err


def test_generation_imports_no_physics_or_numerical_engine():
    code = """
import importlib.abc, json, sys
class RejectEngine(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'scipy', 'sksundae', 'perovskite_sim', 'solarlab_server'} or fullname.startswith(('solarlab.physics', 'solarlab.numerics')):
            raise AssertionError('unexpected engine import: ' + fullname)
sys.meta_path.insert(0, RejectEngine())
from solarlab.config.schema import export_configuration_schema
value = export_configuration_schema()
assert value['structured_parameter_schemas'] == {}
print(json.dumps({'status': value['status'], 'modules': sorted(m for m in sys.modules if m.startswith('solarlab'))}))
"""
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT,
                            text=True, capture_output=True, timeout=10, check=True)
    exported = json.loads(result.stdout)
    assert exported["status"] == "prepared_pending_dependencies"
    assert not any(name.startswith("solarlab.physics") for name in exported["modules"])
