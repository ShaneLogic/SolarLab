"""Real trusted sweep service and coordinate-specific error/preview transport."""
from copy import deepcopy
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from fastapi.testclient import TestClient
import pytest

from solarlab.materials.source import SourceDocument
from solarlab.sweeps.inputs import SweepInput
from solarlab.sweeps.preparation import prepare_sweep
from solarlab.sweeps.scaps import scaps_sweeps
from solarlab_server.app import create_app
from solarlab_server.configuration import ConfigurationPreviewContext
from test_configuration_preview import ROOT, case, words
from test_configuration_preview import resources as resources, writer as writer, forbid_legacy_model_imports as forbid_legacy_model_imports
from test_jv_configuration_preview import catalog as catalog
from test_spatial_configuration_preview import headers
from test_event_service import _HTTPServer


@pytest.fixture(scope="module")
def fixtures(catalog, resources):
    device, sources = case("scaps", catalog)
    declarations, refs = scaps_sweeps(device,
        SourceDocument("P00.manifest", (ROOT / "reproducibility/RefactorReferenceManifestV1.json").read_bytes()),
        SourceDocument("P00.contract", (ROOT / "reproducibility/ScapsComparisonContractV1.json").read_bytes()),
        interface_density_interpretation="historical_areal_input")
    context = ConfigurationPreviewContext(catalog, resources, sources, sweep_references=refs)
    return {item.id: item.model_dump(mode="json", exclude_unset=True) for item in declarations}, context


def test_point_input_content_and_reference_status_match_the_actual_resolver(fixtures, writer):
    values, context = fixtures
    raw = deepcopy(values["scaps_nt_cbo"])
    raw["axes"][0]["coordinates"] = raw["axes"][0]["coordinates"][-1:]
    raw["axes"][1]["coordinates"] = raw["axes"][1]["coordinates"][-2:]
    before = writer.get_run("untouched"), writer.events()
    with TestClient(create_app(writer.root, configuration_preview=context)) as http:
        result = http.post("/configuration-preview/sweep", json=raw, headers=headers())
        assert result.status_code == 200, result.text
        body = result.json()
        expected = prepare_sweep(SweepInput.model_validate(raw), context.defaults, context.resources, sources=context.sources, references=context.sweep_references)
        assert words(body["input"]) == words(expected.to_input().editing_data())
        assert words(body["resolved"]) == words(expected.to_mapping())
        assert body["identity"]["content_sha256"] == expected.content_sha256
        assert body["kind"] == "sweep" and body["can_execute"] is False
        assert body["resolved"]["reference_scope"]["interpretation"]["interface_density_choice"] == "historical_areal_input"
        assert all(p["reference"]["status"] == "reference_missing" for p in body["resolved"]["points"])
        point = body["resolved"]["points"][0]
        actual = http.post("/configuration-preview/device", json=point["applied_input"])
        assert actual.status_code == 200
        assert actual.json()["identity"]["content_sha256"] == point["content_sha256"]
        assert http.get("/configuration-schema").status_code == 200
    assert (writer.get_run("untouched"), writer.events()) == before


def test_schema_limits_target_and_coordinate_failures_remain_truthful(fixtures, writer):
    values, context = fixtures
    raw = deepcopy(values["scaps_nt_pvk_cb"])
    with TestClient(create_app(writer.root, configuration_preview=context)) as http:
        assert http.post("/configuration-preview/sweep", json=raw).status_code == 428
        assert http.post("/configuration-preview/sweep", json=raw, headers={**headers(), "X-Solarlab-Configuration-Schema": "0" * 64}).status_code == 409
        bad = deepcopy(raw)
        bad["axes"][0]["target"]["local_id"] = "missing"
        result = http.post("/configuration-preview/sweep", json=bad, headers=headers())
        assert result.status_code == 422
        assert result.json()["detail"]["field_errors"][0]["loc"] == ["axes", 0, "target"]
        bad = deepcopy(raw)
        bad["axes"][0]["coordinates"] = [{"kind": "value", "value": "1 nm"}]
        result = http.post("/configuration-preview/sweep", json=bad, headers=headers())
        assert result.status_code == 200
        point = result.json()["resolved"]["points"][0]
        assert point["status"] == "unresolved" and point["simulation_status"] == "not_started"
        actual = http.post("/configuration-preview/device", json=point["applied_input"])
        assert actual.status_code == 422
        # Extra-field union errors follow JSON object key order; compare the
        # complete path/error records independently of transport key ordering.
        assert sorted(json.dumps(field, sort_keys=True) for field in point["field_errors"]) == sorted(
            json.dumps(field, sort_keys=True) for field in actual.json()["detail"]["field_errors"])
        location = point["field_errors"][0]["loc"]
        assert location[:4] == ["layers", 2, "bulk_defects", 0]
        assert location[-2:] == ["distribution", "total_density_m3"]
        assert point["applied_input"]["layers"][2]["bulk_defects"][0]["distribution"]["total_density_m3"] == "1 nm"
        bad["axes"][0]["coordinates"] *= 2
        result = http.post("/configuration-preview/sweep", json=bad, headers=headers())
        assert result.status_code == 422 and "duplicate" in result.text


def test_real_loopback_sweep_and_missing_trusted_context(fixtures, writer):
    values, context = fixtures
    raw = deepcopy(values["scaps_cbo"])
    raw["axes"][0]["coordinates"] = [{"kind": "value", "value": "-200 meV"}]
    server = _HTTPServer(writer.root, configuration_preview=context)
    with server:
        endpoint = f"http://127.0.0.1:{server.port}/configuration-preview/sweep"
        with urlopen(Request(endpoint, data=json.dumps(raw).encode(), headers=headers(), method="POST"), timeout=10) as response:
            body = json.loads(response.read())
            assert response.status == 200
        assert body["resolved"]["points"][0]["effective_targets"]["CBO"]["value"] == 4.14
        raw["axes"][0]["target"]["owner_id"] = "unknown"
        with pytest.raises(HTTPError) as error:
            urlopen(Request(endpoint, data=json.dumps(raw).encode(), headers=headers(), method="POST"), timeout=10)
        assert error.value.code == 422
    assert server.stopped
    with TestClient(create_app(writer.root)) as http:
        assert http.post("/configuration-preview/sweep", json=raw, headers=headers()).status_code == 503
