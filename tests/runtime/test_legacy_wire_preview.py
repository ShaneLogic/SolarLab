"""Real local API/client/panel legacy exports; all fixtures are data only."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from solarlab.config.device_import import import_standard_device
from solarlab.config.scaps_input import import_scaps_device
from solarlab.config.tandem_input import import_tandem
from solarlab.config.schema import configuration_schema_representation
from solarlab.config.yaml import load_yaml_mapping
from solarlab.device.inputs import TandemInput
from solarlab.io.run_store import RunStore
from solarlab.materials.source import SourceDocument
from solarlab_server.configuration import ConfigurationPreviewContext
import test_configuration_preview as existing

ROOT = existing.ROOT
catalog = existing.catalog
resources = existing.resources
node_client = existing.node_client
forbid_legacy_model_imports = existing.forbid_legacy_model_imports


@pytest.fixture
def wire_store(tmp_path):
    with RunStore((tmp_path / "empty-preview-store").resolve()) as store:
        yield store
        assert store.events() == []  # No job, progress or terminal event was invented.


def headers():
    return {"Content-Type": "application/json",
            "X-Solarlab-Configuration-Schema": configuration_schema_representation()[1]}


def binding(source):
    return {"id": source.id, "sha256": source.sha256}


def request(input, sources):
    return {"schema_version": "solarlab.legacy-wire-preview-request.v1",
            "input": existing.wire_input(input), "source": binding(sources[0]),
            "references": {source.id: binding(source) for source in sources[1:]}}


def edit(input):
    child = input.top_cell if isinstance(input, TandemInput) else input
    index = 2 if child.source_format == "scaps" else 0
    child.layers[index].parameters.mu_n = "3 cm^2/(V s)"
    child.layers[index].parameters.D_ion = -0.0
    child.layers[index].parameters.optical_material = None
    child.layers[index].parameters.incoherent = False
    child.settings.phi_left = -0.0
    child.description = None


def reimport(result, input, sources, catalog):
    documents = result["documents"]
    assert result["can_execute"] is False and result["status"] == "prepared_data_only"
    assert result["report"]["strict_roundtrip_passed"] is True
    assert result["identity"]["source"] == binding(sources[0])
    assert result["identity"]["configuration_schema_sha256"] == configuration_schema_representation()[1]
    emitted = {}
    for document in documents:
        data = document["utf8"].encode("utf-8")
        assert len(data) == document["size_bytes"]
        assert hashlib.sha256(data).hexdigest() == document["sha256"]
        emitted[document["reference"]] = SourceDocument(document["source_id"], data)
    original = emitted.pop(None)
    if isinstance(input, TandemInput):
        return import_tandem(original, id=input.id, references=emitted, defaults=catalog)
    importer = import_scaps_device if input.source_format == "scaps" else import_standard_device
    return importer(original, id=input.id, defaults=catalog)


@pytest.mark.parametrize("name", ["standard", "scaps", "tandem"])
@pytest.mark.parametrize("edited", [False, True])
def test_bound_source_export_and_strict_reimport(name, edited, catalog, resources, wire_store):
    input, sources = existing.case(name, catalog)
    original_bytes = tuple(source.content for source in sources)
    if edited:
        edit(input)
    payload = request(input, sources)
    before = existing.words(payload)
    context = ConfigurationPreviewContext(catalog, resources, sources)
    with existing.client(wire_store, context) as http:
        response = http.post("/configuration-preview/legacy-wire", json=payload, headers=headers())
        assert response.status_code == 200, response.text
        assert response.headers["cache-control"] == "no-store"
        result = response.json()
        reopened = reimport(result, input, sources, catalog)
        assert reopened.id == input.id
        if not edited:
            assert {row["source_id"]: row["utf8"].encode() for row in result["documents"]} == {
                source.id: source.content for source in sources}
            assert result["report"]["differences"] == []
        else:
            child = reopened.top_cell if isinstance(reopened, TandemInput) else reopened
            index = 2 if child.source_format == "scaps" else 0
            assert dict(child.layers[index].parameters.normalized_items())["mu_n"] == .0003
            assert child.layers[index].parameters.optical_material is None
            assert child.layers[index].parameters.incoherent is False
            assert child.description is None
            assert result["report"]["differences"]
            original = next(row for row in result["documents"] if row["reference"] ==
                            (input.top_cell_reference if isinstance(input, TandemInput) else None))
            assert "-0.0" in original["utf8"]
        assert http.get("/configuration-schema").status_code == 200
    assert existing.words(payload) == before
    assert tuple(source.content for source in sources) == original_bytes


def test_no_edit_keeps_utf8_bom_and_crlf(catalog, resources, wire_store):
    _, sources = existing.case("standard", catalog)
    original = SourceDocument("line-endings.yaml", b"\xef\xbb\xbf" + sources[0].content.replace(b"\n", b"\r\n"))
    input = import_standard_device(original, id="line_endings", defaults=catalog)
    with existing.client(wire_store, ConfigurationPreviewContext(catalog, resources, (original,))) as http:
        response = http.post("/configuration-preview/legacy-wire", json=request(input, (original,)), headers=headers())
        assert response.status_code == 200, response.text
        assert response.json()["documents"][0]["utf8"].encode() == original.content


def test_omission_null_and_integer_negative_zero_are_distinct(catalog, resources, wire_store):
    input, sources = existing.case("standard", catalog)
    payload = request(input, sources)
    payload["input"].pop("description", None)
    payload["input"]["layers"][0]["parameters"].pop("mu_p", None)
    payload["input"]["settings"]["phi_left"] = -0.0
    with existing.client(wire_store, ConfigurationPreviewContext(catalog, resources, sources)) as http:
        raw = json.dumps(payload).replace('"phi_left": -0.0', '"phi_left": -0')
        response = http.post("/configuration-preview/legacy-wire", content=raw, headers=headers())
        assert response.status_code == 200, response.text
        result = response.json()
        assert "description" not in result["input"]
        assert "mu_p" not in result["input"]["layers"][0]["parameters"]
        wire = load_yaml_mapping(result["documents"][0]["utf8"].encode())
        assert "description" not in wire and "mu_p" not in wire["layers"][0]
        assert wire["device"]["phi_left"].is_signed()
        payload["input"]["description"] = None
        cleared = http.post("/configuration-preview/legacy-wire", json=payload, headers=headers()).json()
        assert cleared["input"]["description"] is None
        assert load_yaml_mapping(cleared["documents"][0]["utf8"].encode())["description"] is None


@pytest.mark.parametrize("failure,status,path", [
    ("unknown", 422, ["source", "id"]),
    ("mismatch", 409, ["source", "sha256"]),
    ("extra_source", 422, ["source", "path"]),
    ("extra_reference", 422, ["references"]),
    ("catalog", 422, ["defaults"]),
])
def test_source_selection_never_discovers_paths(failure, status, path, catalog, resources, wire_store, monkeypatch):
    input, sources = existing.case("standard", catalog)
    payload = request(input, sources)
    if failure == "unknown":
        payload["source"]["id"] = "/a/client/supplied/path.yaml"
    elif failure == "mismatch":
        payload["source"]["sha256"] = "0" * 64
    elif failure == "extra_source":
        payload["source"]["path"] = "/read/me.yaml"
    elif failure == "extra_reference":
        payload["references"]["unused.yaml"] = binding(sources[0])
    else:
        payload["defaults"] = {"replacement": True}
    expected_headers = headers()
    with existing.client(wire_store, ConfigurationPreviewContext(catalog, resources, sources)) as http:
        monkeypatch.setattr(Path, "read_bytes", lambda *_: pytest.fail("preview attempted server file discovery"))
        response = http.post("/configuration-preview/legacy-wire", json=payload, headers=expected_headers)
        assert response.status_code == status, response.text
        assert path in [row["loc"] for row in response.json()["detail"]["field_errors"]]
        assert "documents" not in response.json()


@pytest.mark.parametrize("failure", ["missing", "wrong_identity", "wrong_hash", "swapped"])
def test_tandem_requires_exact_reference_bindings(failure, catalog, resources, wire_store):
    input, sources = existing.case("tandem", catalog)
    payload = request(input, sources)
    refs = list(payload["references"])
    if failure == "missing":
        payload["references"].pop(refs[0])
    elif failure == "wrong_identity":
        payload["references"][refs[0]]["id"] = "unknown.yaml"
    elif failure == "wrong_hash":
        payload["references"][refs[0]]["sha256"] = "0" * 64
    else:
        payload["references"][refs[0]], payload["references"][refs[1]] = payload["references"][refs[1]], payload["references"][refs[0]]
    with existing.client(wire_store, ConfigurationPreviewContext(catalog, resources, sources)) as http:
        response = http.post("/configuration-preview/legacy-wire", json=payload, headers=headers())
        if failure == "swapped":
            assert response.status_code == 200, response.text
            assert response.json()["report"]["supported"] is False
            assert response.json()["documents"] == []
        else:
            assert response.status_code in {409, 422}, response.text
            assert response.json()["detail"]["can_execute"] is False


@pytest.mark.parametrize("failure", ["material", "spectrum", "wrong_original"])
def test_lossy_edit_returns_field_report_and_no_partial_document(failure, catalog, resources, wire_store):
    input, sources = existing.case("standard", catalog)
    payload = request(input, sources)
    if failure == "material":
        payload["input"]["materials"] = [{"id": "shared", "name": "Shared", "parameters": {"mu_n": 0}}]
        payload["input"]["layers"][0]["material"] = "shared"
    elif failure == "spectrum":
        payload["input"]["spectrum"] = "/not/a/server/read.json"
    else:
        _, other = existing.case("scaps", catalog)
        sources = (*sources, *other)
        payload["source"] = binding(other[0])
    with existing.client(wire_store, ConfigurationPreviewContext(catalog, resources, sources)) as http:
        response = http.post("/configuration-preview/legacy-wire", json=payload, headers=headers())
        assert response.status_code == 200, response.text
        value = response.json()
        assert value["status"] == "unsupported" and value["can_execute"] is False
        assert value["documents"] == [] and value["report"]["unsupported"]
        assert all(row["path"].startswith("$") for row in value["report"]["unsupported"])


def test_context_schema_and_existing_transport_errors(catalog, resources, wire_store):
    input, sources = existing.case("standard", catalog)
    payload = request(input, sources)
    with existing.client(wire_store) as http:
        assert http.post("/configuration-preview/legacy-wire", json=payload, headers=headers()).status_code == 503
    with existing.client(wire_store, ConfigurationPreviewContext(catalog, resources, sources)) as http:
        assert http.post("/configuration-preview/legacy-wire", json=payload).status_code == 428
        assert http.post("/configuration-preview/legacy-wire", json=payload,
                         headers={**headers(), "X-Solarlab-Configuration-Schema": "0" * 64}).status_code == 409
        for content in ('{"input": 1, "input": 2}', '{"x":NaN}', '{"x":1e999}'):
            assert http.post("/configuration-preview/legacy-wire", content=content, headers=headers()).status_code == 400
        bad = deepcopy(payload)
        bad["input"]["layers"][0]["parameters"]["mu_n"] = None
        response = http.post("/configuration-preview/legacy-wire", json=bad, headers=headers())
        assert response.status_code == 422
        assert any("mu_n" in field["loc"] for field in response.json()["detail"]["field_errors"])
        bad = deepcopy(payload)
        bad["input"]["schema_version"] = "unknown-input-schema"
        assert http.post("/configuration-preview/legacy-wire", json=bad, headers=headers()).status_code == 422


@pytest.fixture(scope="module")
def node_panel(node_client):
    target = node_client.parent / "configuration-preview.mjs"
    script = """
import fs from 'node:fs';
import ts from 'typescript';
const source = fs.readFileSync(process.argv[1], 'utf8');
const result = ts.transpileModule(source, {compilerOptions: {
  module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2023, verbatimModuleSyntax: true
}});
fs.writeFileSync(process.argv[2], result.outputText.replace('../configuration-client', './configuration-client.mjs'));
"""
    subprocess.run(["node", "--input-type=module", "-e", script,
        str(ROOT / "web/src/panels/configuration-preview.ts"), str(target)],
        cwd=ROOT / "web", check=True, timeout=15, capture_output=True)
    return target


@pytest.mark.parametrize("name", ["standard", "scaps", "tandem"])
@pytest.mark.parametrize("edited", [False, True])
def test_real_http_client_panel_download_reimports(name, edited, catalog, resources, wire_store, node_panel, tmp_path, record_property):
    input, sources = existing.case(name, catalog)
    if edited:
        edit(input)
    payload = request(input, sources)
    request_path, result_path = tmp_path / "request.json", tmp_path / "wire-result.json"
    request_path.write_text(json.dumps(payload))
    context = ConfigurationPreviewContext(catalog, resources, sources)
    script = """
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { pathToFileURL } from 'node:url';
import { JSDOM } from 'jsdom';
const { mountConfigurationPreviewPanel } = await import(pathToFileURL(process.argv[1]));
const request = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const kind = request.input.schema_version === 'solarlab.tandem-preparation.v1' ? 'tandem' : 'device';
const dom = new JSDOM('<main id="preview"></main>', { url: process.argv[4] });
globalThis.document = dom.window.document;
const root = document.getElementById('preview');
const downloads = [];
dom.window.HTMLAnchorElement.prototype.click = function () { downloads.push({ href: this.href, name: this.download }); };
const panel = mountConfigurationPreviewPanel(root, {
  endpoint: process.argv[4] + '/configuration-preview/' + kind, kind, input: request.input,
  legacyWire: { endpoint: process.argv[4] + '/configuration-preview/legacy-wire', source: request.source, references: request.references },
});
try {
  await panel.ready;
  assert.equal(root.querySelector('[data-panel]').dataset.state, 'ready');
  const button = label => [...root.querySelectorAll('button')].find(item => item.textContent === label);
  button('Preview legacy YAML').click();
  await panel.wireReady;
  assert.equal(root.querySelector('[data-section=legacy-wire]').dataset.state, 'ready', root.textContent);
  assert.equal(panel.wireDocument.value.can_execute, false);
  assert.equal(panel.wireDocument.value.report.strict_roundtrip_passed, true);
  assert.equal(panel.wireDocument.value.input.id, request.input.id);
  fs.writeFileSync(process.argv[3], panel.wireDocument.json);
  for (const item of root.querySelectorAll('[data-role=legacy-document] button')) item.click();
  assert.equal(downloads.length, panel.wireDocument.value.documents.length);
  const records = [];
  for (const [index, download] of downloads.entries()) {
    const bytes = new Uint8Array(await (await fetch(download.href)).arrayBuffer());
    const target = process.argv[3] + '.' + index + '.yaml';
    fs.writeFileSync(target, bytes);
    records.push({ filename: download.name, path: target });
  }
  fs.writeFileSync(process.argv[3] + '.downloads.json', JSON.stringify(records));
  console.log(JSON.stringify({ actual_HTTP: true, actual_client: true, actual_panel: true,
    download_sink: 'jsdom anchor with actual Blob bytes', real_browser: false, scientific_calls: 0, documents: downloads.length }));
} finally { panel.dispose(); dom.window.close(); }
"""
    with existing._HTTPServer(wire_store.root, configuration_preview=context) as server:
        endpoint = f"http://127.0.0.1:{server.port}"
        completed = subprocess.run(["node", "--input-type=module", "-e", script,
            str(node_panel), str(request_path), str(result_path), endpoint],
            cwd=ROOT / "web", text=True, capture_output=True, timeout=25)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        result = json.loads(result_path.read_text())
        reopened = reimport(result, input, sources, catalog)
        assert reopened.id == input.id
        downloads = json.loads(Path(str(result_path) + ".downloads.json").read_text())
        for row, document in zip(downloads, result["documents"], strict=True):
            assert Path(row["path"]).read_bytes() == document["utf8"].encode()
        if not edited:
            assert {row["source_id"]: row["utf8"].encode() for row in result["documents"]} == {
                source.id: source.content for source in sources}
    assert server.stopped
    record_property("legacy_source_kind", name)
    record_property("edited", edited)
    record_property("actual_local_API_client_panel", True)
    record_property("strict_reimport_passed", True)
    record_property("real_browser", False)
