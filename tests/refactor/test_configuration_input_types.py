"""Structural TypeScript checks bound to actual prepared DTOs and YAML inputs."""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from solarlab.config.inputs import LayerInput
from solarlab.config.schema import export_configuration_schema
from solarlab.config.yaml import load_yaml_mapping
from solarlab.device.inputs import ContactInput, DeviceInput, FullLayerInput
from solarlab.device.settings import DeviceSettingsInput
from solarlab.device.tunnelling import TunnellingInput
from solarlab.materials.full_parameters import FullParameterInput
from solarlab.materials.optics import CigsOpticsInput
from solarlab.materials.parameters import LayerValuesInput

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "web"
EXPORTER = ROOT / "scripts/generate_configuration_schema.py"
GENERATOR = WEB / "scripts/generate-configuration-inputs.mjs"
TYPES = WEB / "src/generated/configuration-inputs.ts"
FIXTURES = ROOT / "perovskite-sim/tests/fixtures/configs"


def run(command, cwd, expected=0):
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True,
                            timeout=30, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    assert result.returncode == expected, result.stdout + result.stderr
    return result


def node():
    executable = os.environ.get("SOLARLAB_NODE") or shutil.which("node")
    assert executable, "the existing frontend Node runtime is required"
    return executable


def test_json_mode_matches_public_metadata_and_preserves_default(tmp_path):
    result = run([sys.executable, "-B", str(EXPORTER), "--json"], tmp_path)
    assert json.loads(result.stdout) == export_configuration_schema()
    target = tmp_path / "metadata.ts"
    run([sys.executable, "-B", str(EXPORTER), "--output", str(target)], tmp_path)
    assert target.read_bytes() == (WEB / "src/generated/configuration-schema.ts").read_bytes()
    run([sys.executable, "-B", str(EXPORTER), "--json", "--check"], tmp_path, expected=2)


def test_generation_determinism_check_and_isolated_scopes(tmp_path):
    command = [node(), str(GENERATOR), "--python", sys.executable]
    run([*command, "--check"], tmp_path)
    target = tmp_path / "inputs.ts"
    run([*command, "--output", str(target)], tmp_path)
    assert target.read_bytes() == TYPES.read_bytes()
    run([*command, "--output", str(target), "--check"], tmp_path)
    changed = target.read_bytes() + b"// stale\n"
    target.write_bytes(changed)
    run([*command, "--output", str(target), "--check"], tmp_path, expected=1)
    assert target.read_bytes() == changed
    target.unlink()
    run([*command, "--output", str(target), "--check"], tmp_path, expected=1)
    assert not target.exists()
    names = sorted(export_configuration_schema()["dto_schemas"])
    text = TYPES.read_text()
    assert re.findall(r"export declare namespace (\w+)Definitions \{", text) == names
    for name in names:
        assert f"export type {name} = {name}Definitions.{name};" in text
    # Device and tandem really have same-named local definitions; neither graph
    # is hoisted into the other document's namespace.
    assert text.count("export interface FullLayerInput") >= 3


def test_actual_inputs_and_discriminating_compile_time_negatives(tmp_path):
    # These are existing research input fragments, not model defaults or a
    # production catalogue. DTO JSON output preserves unit strings/explicit data.
    cigs_path = FIXTURES / "cigs_graded_optics.yaml"
    wkb_path = FIXTURES / "wkb_resolved_electron_barrier.yaml"
    raw_layer = load_yaml_mapping(cigs_path.read_bytes())["layers"][2]
    cigs = CigsOpticsInput.model_validate(raw_layer["cigs_graded_optics"])
    tunnelling = TunnellingInput.model_validate(
        load_yaml_mapping(wkb_path.read_bytes())["device"]["tunnelling_channels"])
    parameters = FullParameterInput.model_validate(
        {key: value for key, value in raw_layer.items() if key in FullParameterInput.model_fields})
    full_layer = FullLayerInput(id=raw_layer["name"], name=raw_layer["name"], role=raw_layer["role"],
                               thickness=raw_layer["thickness"], parameters=parameters, cigs_graded_optics=cigs)
    device = DeviceInput(schema_version="solarlab.device-preparation.v1", id="cigs_type_fixture",
                         source_format="standard", layers=(full_layer,))
    values = FullParameterInput(mu_n=0, incoherent=False, Nc300=None, optical_material=None)
    template_values = LayerValuesInput(mu_n=0, incoherent=False, optical_material=None)
    layer = LayerInput(id="input_fixture", name="Input fixture", template="supplied", overrides=template_values)
    samples = {
        "cigs": ("CigsOpticsInput", cigs), "tunnelling": ("TunnellingInput", tunnelling),
        "parameters": ("FullParameterInput", parameters), "zeroFalseNull": ("FullParameterInput", values),
        "emptyOverrides": ("FullParameterInput", FullParameterInput()),
        "templateOverrides": ("LayerValuesInput", template_values),
        "layer": ("LayerInput", layer), "fullLayer": ("FullLayerInput", full_layer),
        "device": ("DeviceInput", device),
        "contact": ("ContactInput", ContactInput(id="left", side="left", layer=full_layer.id, S_n=None)),
        "signedUnit": ("DeviceSettingsInput", DeviceSettingsInput(phi_left="-20 mV")),
    }
    types_path = TYPES.with_suffix("").as_posix()
    lines = [f"import type * as Input from {json.dumps(types_path)};",
             f"// Actual fixture sources: {cigs_path.relative_to(ROOT)}, {wkb_path.relative_to(ROOT)}",
             "type IsAny<T> = 0 extends (1 & T) ? true : false;",
             "type NotAny<T extends false> = T;"]
    names = sorted(export_configuration_schema()["dto_schemas"])
    lines.append("export type AllInputTypes = [" + ", ".join(f"Input.{name}" for name in names) + "];")
    lines.append("export type AllInputsConstrained = [" + ", ".join(f"NotAny<IsAny<Input.{name}>>" for name in names) + "];")
    for name, (type_name, value) in samples.items():
        lines.append(f"export const {name}: Input.{type_name} = {value.model_dump_json(exclude_unset=True)};")
    negatives = [
        ("unknownField", "FullParameterInput", "{ unknownParameter: 1 }"),
        ("booleanQuantity", "FullParameterInput", "{ mu_n: false }"),
        ("nullQuantity", "FullParameterInput", "{ mu_n: null }"),
        ("numericBoolean", "FullParameterInput", "{ incoherent: 0 }"),
        ("missingCgi", "CigsOpticsInput", "{ ggi_front: 0.225, ggi_back: 0.6 }"),
        ("nullOptional", "CigsOpticsInput", "{ ...cigs, slices: null }"),
        ("badVersion", "TunnellingInput", "{ schema_version: 'unsupported' }"),
        ("badCarrier", "TunnellingInput", "{ intraband: { carrier: 'ion' } }"),
        ("unknownNested", "TunnellingInput", "{ intraband: { missingChannelField: true } }"),
        ("badKind", "LayerInput", "{ ...layer, kind: 'material' }"),
        ("missingRequired", "LayerInput", "{ name: 'x', template: 'x' }"),
        ("wrongArray", "DeviceInput", "{ ...device, layers: fullLayer }"),
        ("wrongContactNull", "ContactInput", "{ ...contact, id: null }"),
    ]
    for name, type_name, expression in negatives:
        lines.extend([f"// @ts-expect-error {name}: Python schema forbids this structural shape",
                      f"export const {name}: Input.{type_name} = {expression};"])
    source = tmp_path / "actual-inputs.type-test.ts"
    source.write_text("\n".join(lines) + "\n")
    command = [node(), str(WEB / "node_modules/typescript/bin/tsc"), "--strict", "--noEmit",
               "--skipLibCheck", "--target", "es2023", "--module", "esnext",
               "--moduleResolution", "bundler", "--erasableSyntaxOnly", str(source)]
    run(command, tmp_path)
    # The successful run also proves that every @ts-expect-error was used.
    # No schema compiler or unsafe casts are substituted for the real toolchain.
