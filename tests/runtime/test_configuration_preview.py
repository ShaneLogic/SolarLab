"""Actual prepared configuration POSTs; RunStore/HTTP fixtures are infrastructure."""

from __future__ import annotations

import builtins
from copy import deepcopy
import json
from pathlib import Path
import subprocess

from fastapi.testclient import TestClient
import pytest

from solarlab.config.device_import import import_standard_device
from solarlab.config.legacy_defaults import read_legacy_default_catalog
from solarlab.config.resolve_device import resolve_device, resolve_tandem
from solarlab.config.scaps_input import import_scaps_device
from solarlab.config.tandem_input import import_tandem
from solarlab.device.inputs import DeviceInput, NamedMaterialInput, TandemInput
from solarlab.materials.full_parameters import FullParameterInput
from solarlab.materials.resources import ResourceLibrary, ResourceTable
from solarlab.materials.source import SourceDocument
from solarlab.io.run_store import RunStore
from solarlab_server.app import create_app
from solarlab_server.configuration import ConfigurationPreviewContext
from test_event_service import _HTTPServer

ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT / "perovskite-sim/perovskite_sim"
CONFIGS = ROOT / "perovskite-sim/tests/fixtures/configs"
DEFAULT_PATHS = ("models/parameters.py", "models/device.py", "models/defects.py",
                 "physics/statistics.py", "physics/band_gap_narrowing.py",
                 "scaps_compat/loader.py", "constants.py", "twod/microstructure.py",
                 "models/tandem_config.py")


@pytest.fixture(autouse=True)
def forbid_legacy_model_imports(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert not name.startswith(("perovskite_sim", "backend", "scipy.integrate")), name
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


@pytest.fixture(scope="module")
def catalog():
    return read_legacy_default_catalog(tuple(
        SourceDocument(name, (LEGACY / name).read_bytes()) for name in DEFAULT_PATHS))


@pytest.fixture(scope="module")
def resources():
    return ResourceLibrary(tuple(ResourceTable(path.stem, "nk", SourceDocument(
        str(path.relative_to(ROOT)), path.read_bytes()))
        for path in sorted((LEGACY / "data/nk").glob("*.csv"))))


def case(name, catalog):
    if name == "tandem":
        source = SourceDocument("tandem_lin2019.yaml", (CONFIGS / "tandem_lin2019.yaml").read_bytes())
        refs = {name: SourceDocument(name, (CONFIGS / name).read_bytes()) for name in
                ("nip_wideGap_FACs_1p77.yaml", "nip_SnPb_1p22.yaml")}
        return import_tandem(source, id="lin2019", references=refs, defaults=catalog), (source, *refs.values())
    path = ROOT / "perovskite-sim/configs/scaps_mirror_v2.yaml" if name == "scaps" else CONFIGS / "nip_MAPbI3.yaml"
    source = SourceDocument(str(path.relative_to(ROOT)), path.read_bytes())
    importer = import_scaps_device if name == "scaps" else import_standard_device
    return importer(source, id=name, defaults=catalog), (source,)


def prepared(input, context):
    resolver = resolve_tandem if isinstance(input, TandemInput) else resolve_device
    return resolver(input, context.defaults, context.resources, sources=context.sources)


def wire_input(input):
    # Use the resolver's existing DTO JSON export. Imported YAML quantities can
    # be Decimal; the generated number|string contract preserves these as text.
    return type(input).model_validate(input).model_dump(mode="json", exclude_unset=True)


def words(value):
    if isinstance(value, dict):
        return {name: words(part) for name, part in value.items()}
    if isinstance(value, (tuple, list)):
        return [words(part) for part in value]
    return [type(value).__name__, value.hex() if type(value) is float else value]


@pytest.fixture
def writer(tmp_path):
    with RunStore((tmp_path / "infrastructure-store").resolve()) as store:
        store.enqueue("untouched", opaque_identity={"H_input": "infrastructure",
            "H_physics": "unused", "H_execution": "not-a-simulation"},
            input_metadata={"purpose": "configuration preview infrastructure", "zero": 0})
        yield store


def client(writer, context=None):
    return TestClient(create_app(writer.root, configuration_preview=context))


def assert_preview(data, expected, context, kind):
    assert data["schema"] == "solarlab.configuration-preview.v1"
    assert data["kind"] == kind and data["status"] == "prepared_pending_dependencies"
    assert data["can_execute"] is False
    assert words(data["input"]) == words(expected.to_input().editing_data())
    assert words(data["resolved"]) == words(expected.to_mapping())
    assert data["identity"] == {"scope": "configuration_content", "content_sha256": expected.content_sha256,
        "default_catalog_sha256": context.defaults.content_sha256,
        "resource_library_sha256": context.resources.content_sha256}
    assert "H_execution" not in data["identity"]


@pytest.mark.parametrize("name", ["standard", "scaps", "tandem"])
def test_actual_dtos_post_matches_prepared_resolver(name, catalog, resources, writer):
    input, sources = case(name, catalog)
    context = ConfigurationPreviewContext(catalog, resources, sources)
    expected = prepared(input, context)
    kind = "tandem" if name == "tandem" else "device"
    raw = wire_input(input)
    before = words(raw)
    run, events = writer.get_run("untouched"), writer.events()
    with client(writer, context) as http:
        response = http.post(f"/configuration-preview/{kind}", json=raw)
        assert response.status_code == 200, response.text
        assert response.headers["cache-control"] == "no-store"
        assert_preview(response.json(), expected, context, kind)
        assert words(raw) == before
        assert http.get("/runs/untouched").json() == run
        assert http.get("/configuration-schema").status_code == 200
    assert writer.get_run("untouched") == run and writer.events() == events


def test_zero_signed_zero_presence_null_and_isolation(catalog, resources, writer):
    input, sources = case("standard", catalog)
    raw = wire_input(input)
    raw.pop("description", None)
    raw["settings"]["phi_left"] = -0.0
    raw["layers"][0]["parameters"].update(D_ion=0, optical_material=None)
    context = ConfigurationPreviewContext(catalog, resources, sources)
    expected = prepared(DeviceInput.model_validate(raw), context)
    with client(writer, context) as http:
        first = http.post("/configuration-preview/device", json=raw).json()
        assert_preview(first, expected, context, "device")
        assert first["input"]["settings"]["phi_left"].hex() == "-0x0.0p+0"
        # The existing unit normalizer owns the effective sign; the input sign
        # must survive independently, without changing that resolver behavior.
        assert first["resolved"]["settings"]["phi_left"].hex() == expected.to_mapping()["settings"]["phi_left"].hex()
        assert first["input"]["layers"][0]["parameters"]["D_ion"] == 0
        assert first["input"]["layers"][0]["parameters"]["optical_material"] is None
        assert "description" not in first["input"]
        first["resolved"]["settings"]["T"] = 999
        first["input"]["layers"].clear()
        again = http.post("/configuration-preview/device", json=raw).json()
        assert_preview(again, expected, context, "device")
        explicit = deepcopy(raw)
        explicit["description"] = None
        cleared = http.post("/configuration-preview/device", json=explicit).json()
        assert cleared["input"]["description"] is None
        assert cleared["identity"]["content_sha256"] != again["identity"]["content_sha256"]
        negative_integer_token = json.dumps(raw).replace('"phi_left": -0.0', '"phi_left": -0')
        special = http.post("/configuration-preview/device", content=negative_integer_token,
                            headers={"Content-Type": "application/json"}).json()
        assert special["input"]["settings"]["phi_left"].hex() == "-0x0.0p+0"


def test_equivalent_units_and_material_inheritance(catalog, resources, writer):
    input, sources = case("standard", catalog)
    input.materials = (NamedMaterialInput(id="shared", name="shared material", parameters=
        FullParameterInput(mu_n="2 cm^2/(V s)", optical_material="MAPbI3")),)
    raw = wire_input(input)
    raw["layers"][0]["material"] = "shared"
    raw["layers"][0]["parameters"].pop("mu_n", None)
    raw["layers"][0]["parameters"].pop("optical_material", None)
    raw["layers"][0]["thickness"] = "400 nm"
    context = ConfigurationPreviewContext(catalog, resources, sources)
    with client(writer, context) as http:
        inherited = http.post("/configuration-preview/device", json=raw).json()
        assert_preview(inherited, prepared(DeviceInput.model_validate(raw), context), context, "device")
        equivalent = deepcopy(raw)
        equivalent["layers"][0]["thickness"] = 4e-7
        same = http.post("/configuration-preview/device", json=equivalent).json()
        assert_preview(same, prepared(DeviceInput.model_validate(equivalent), context), context, "device")
        assert inherited["resolved"]["layers"][0]["thickness_m"] == same["resolved"]["layers"][0]["thickness_m"]
        raw["layers"][0]["parameters"].update(mu_n=0, optical_material=None)
        overridden = http.post("/configuration-preview/device", json=raw).json()
        assert_preview(overridden, prepared(DeviceInput.model_validate(raw), context), context, "device")
        assert overridden["input"]["layers"][0]["parameters"]["mu_n"] == 0
        assert overridden["identity"]["default_catalog_sha256"] == catalog.content_sha256


def test_missing_context_and_client_cannot_replace_catalog(catalog, resources, writer):
    input, _ = case("standard", catalog)
    with client(writer) as http:
        response = http.post("/configuration-preview/device", json=wire_input(input))
        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "configuration_context_missing"
        assert http.get("/configuration-schema").status_code == 200
    with pytest.raises(TypeError, match="ConfigurationPreviewContext"):
        create_app(writer.root, configuration_preview={"defaults": "request"})
    with client(writer, ConfigurationPreviewContext(catalog, resources)) as http:
        data = {**wire_input(input), "defaults": {"identity": "replacement"}, "resources": {}}
        error = http.post("/configuration-preview/device", json=data)
        assert error.status_code == 422
        assert {row["loc"][0] for row in error.json()["detail"]["field_errors"]} == {"defaults", "resources"}


def test_field_errors_and_unresolved_capability_gaps(catalog, resources, writer, monkeypatch):
    input, _ = case("standard", catalog)
    data = wire_input(input)
    context = ConfigurationPreviewContext(catalog, resources)
    with client(writer, context) as http:
        bad = deepcopy(data)
        bad["layers"][0]["parameters"]["mu_n"] = None
        response = http.post("/configuration-preview/device", json=bad)
        assert response.status_code == 422
        assert ["layers", 0, "parameters", "mu_n"] in [v["loc"] for v in response.json()["detail"]["field_errors"]]
        bad = deepcopy(data)
        bad["layers"][0]["material"] = "missing"
        response = http.post("/configuration-preview/device", json=bad)
        assert response.status_code == 422
        assert response.json()["detail"]["code"] == "configuration_resolution"
        assert "unknown named material" in response.json()["detail"]["message"]
        data["fixed_generation"] = "/a/user/supplied/server/path.json"
        monkeypatch.setattr(Path, "read_bytes", lambda *_: pytest.fail("request must not read a server path"))
        response = http.post("/configuration-preview/device", json=data)
        assert response.status_code == 422 and "missing supplied" in response.json()["detail"]["message"]
        data.pop("fixed_generation")
        data["settings"]["built_in_potential_mode"] = "semiconductor_work_function"
        result = http.post("/configuration-preview/device", json=data)
        assert result.status_code == 200, result.text
        assert "contact_potential_requires_behavior_or_physical_model_evaluation" in result.json()["resolved"]["capability_gaps"]
        assert result.json()["can_execute"] is False


@pytest.mark.parametrize("payload", [b'{', b'\xff', b'{"id":"a","id":"b"}', b'{"x":NaN}', b'{"x":1e999}'])
def test_malformed_json_is_structured(payload, catalog, resources, writer):
    with client(writer, ConfigurationPreviewContext(catalog, resources)) as http:
        response = http.post("/configuration-preview/device", content=payload,
                             headers={"Content-Type": "application/json"})
        assert response.status_code == 400 and response.json()["detail"]["can_execute"] is False


def test_transport_type_and_limit(catalog, resources, writer):
    with client(writer, ConfigurationPreviewContext(catalog, resources, max_input_bytes=8)) as http:
        assert http.post("/configuration-preview/device", content="{}").status_code == 415
        assert http.post("/configuration-preview/device", json={"oversized": True}).status_code == 413


@pytest.fixture(scope="module")
def node_client(tmp_path_factory):
    target = tmp_path_factory.mktemp("compiled-preview-client") / "configuration-client.mjs"
    script = """
import fs from 'node:fs';
import ts from 'typescript';
const source = fs.readFileSync(process.argv[1], 'utf8');
const result = ts.transpileModule(source, {compilerOptions: {
  module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2023, verbatimModuleSyntax: true
}});
fs.writeFileSync(process.argv[2], result.outputText.replace('./generated/configuration-schema', './configuration-schema.mjs'));
const metadata = ts.transpileModule(fs.readFileSync(process.argv[3], 'utf8'), {compilerOptions: {
  module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2023
}});
fs.writeFileSync(process.argv[2].replace('configuration-client.mjs', 'configuration-schema.mjs'), metadata.outputText);
"""
    subprocess.run(["node", "--input-type=module", "-e", script,
                    str(ROOT / "web/src/configuration-client.ts"), str(target),
                    str(ROOT / "web/src/generated/configuration-schema.ts")],
                   cwd=ROOT / "web", check=True, timeout=15, capture_output=True)
    return target


@pytest.mark.parametrize("name", ["standard", "scaps", "tandem"])
def test_real_loopback_generated_input_client(name, catalog, resources, writer, node_client, tmp_path):
    input, sources = case(name, catalog)
    if isinstance(input, DeviceInput):
        input.settings.phi_left = -0.0
    context = ConfigurationPreviewContext(catalog, resources, sources)
    raw = tmp_path / "input.json"
    wire = tmp_path / "wire.json"
    result = tmp_path / "response.json"
    raw.write_text(json.dumps(wire_input(input)))
    kind = "tandem" if name == "tandem" else "device"
    script = """
import fs from 'node:fs';
import { pathToFileURL } from 'node:url';
const client = await import(pathToFileURL(process.argv[1]));
const input = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const nativeFetch = globalThis.fetch;
globalThis.fetch = (endpoint, init) => {
  fs.writeFileSync(process.argv[3], init.body);
  return nativeFetch(endpoint, init);
};
const preview = process.argv[6] === 'device' ? client.previewDeviceConfiguration : client.previewTandemConfiguration;
const response = await preview(process.argv[5], input, AbortSignal.timeout(10000));
if (response.value.can_execute !== false) throw Error('unexpected execution');
if (process.argv[6] === 'device' && !Object.is(response.value.input.settings.phi_left, -0)) throw Error('signed zero lost');
fs.writeFileSync(process.argv[4], response.json);
"""
    with _HTTPServer(writer.root, configuration_preview=context) as server:
        endpoint = f"http://127.0.0.1:{server.port}/configuration-preview/{kind}"
        result_process = subprocess.run(["node", "--input-type=module", "-e", script,
            str(node_client), str(raw), str(wire), str(result), endpoint, kind],
            cwd=ROOT / "web", text=True, capture_output=True, timeout=20)
        assert result_process.returncode == 0, result_process.stderr
        sent = json.loads(wire.read_text())
        dto = (TandemInput if kind == "tandem" else DeviceInput).model_validate(sent)
        assert_preview(json.loads(result.read_text()), prepared(dto, context), context, kind)
        assert writer.get_run("untouched")["state"] == "queued"
    assert server.stopped
