"""J-V declarations through the real trusted service; no execution requests."""
from copy import deepcopy
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from fastapi.testclient import TestClient
import pytest

from solarlab.config.device_import import import_standard_device
from solarlab.config.legacy_defaults import read_legacy_default_catalog
from solarlab.experiments.jv.inputs import JVExperimentInput
from solarlab.experiments.jv.preparation import prepare_jv_experiment
from solarlab.materials.source import SourceDocument
from solarlab_server.app import create_app
from solarlab_server.configuration import ConfigurationPreviewContext
from test_configuration_preview import DEFAULT_PATHS, LEGACY, ROOT, case, wire_input, words
from test_configuration_preview import resources as resources, writer as writer, forbid_legacy_model_imports as forbid_legacy_model_imports
from test_spatial_configuration_preview import headers
from test_event_service import _HTTPServer


@pytest.fixture(scope="module")
def catalog():
    backend = SourceDocument("backend/main.py", (ROOT / "perovskite-sim/backend/main.py").read_bytes())
    return read_legacy_default_catalog(tuple(SourceDocument(name, (LEGACY / name).read_bytes()) for name in DEFAULT_PATHS),
        experiment_sources=(backend, SourceDocument("solver/tolerances.py", (LEGACY / "solver/tolerances.py").read_bytes())),
        jv_sources=(backend, *(SourceDocument(name, (LEGACY / name).read_bytes()) for name in
            ("experiments/jv_sweep.py", "experiments/waveform_jv.py", "experiments/dark_jv.py", "experiments/protocol.py"))))


def payload(catalog, **fields):
    device, sources = case("standard", catalog)
    return {"schema_version": "solarlab.experiment-preparation.v1", "id": "jv_http", "device": wire_input(device),
            "experiment": {"kind": "jv", **fields}}, sources


def assert_prepared(body, data, context):
    expected = prepare_jv_experiment(JVExperimentInput.model_validate(data), context.defaults, context.resources, sources=context.sources)
    assert body["status"] == "prepared_pending_dependencies" and body["can_execute"] is False
    assert words(body["input"]) == words(expected.to_input().editing_data())
    assert words(body["resolved"]) == words(expected.to_mapping())
    assert body["identity"]["content_sha256"] == expected.content_sha256
    assert body["resolved"]["identity"]["execution_identity"] is None
    assert body["resolved"]["experiment"]["protocol"]["executor_id"] is None
    return body["resolved"]["experiment"]


def test_shared_experiment_envelope_resolves_both_jv_kinds_and_preserves_spatial_route(catalog, resources, writer):
    data, sources = payload(catalog, V_max=-0.0, v_rate="40 mV/s", illuminated=False)
    context = ConfigurationPreviewContext(catalog, resources, sources)
    before = writer.get_run("untouched"), writer.events()
    with TestClient(create_app(writer.root, configuration_preview=context)) as http:
        response = http.post("/configuration-preview/experiment", json=data, headers=headers())
        assert response.status_code == 200, response.text
        resolved = assert_prepared(response.json(), data, context)
        assert resolved["controls"]["v_rate"] == 0.04
        assert words(response.json()["input"]["experiment"]["V_max"]) == words(-0.0)
        assert resolved["state_history"]["required"] is True
        data["experiment"] = {"kind": "dark_jv"}
        result = http.post("/configuration-preview/experiment", json=data, headers=headers())
        assert result.status_code == 200
        assert assert_prepared(result.json(), data, context)["controls"]["n_points"] == 60
        data["experiment"] = {"kind": "jv_2d", "microstructure": {}, "lateral_bc": None}
        spatial = http.post("/configuration-preview/experiment", json=data, headers=headers())
        assert spatial.status_code == 200
        assert spatial.json()["resolved"]["schema"] == "solarlab.resolved-spatial-experiment-preparation.v1"
        assert spatial.json()["resolved"]["experiment"]["controls"]["Ny_per_layer"] == 20
        assert spatial.json()["input"]["experiment"] == data["experiment"]
    assert (writer.get_run("untouched"), writer.events()) == before


def test_context_schema_branch_and_typed_field_failures(catalog, resources, writer):
    data, sources = payload(catalog)
    context = ConfigurationPreviewContext(catalog, resources, sources)
    with TestClient(create_app(writer.root, configuration_preview=context)) as http:
        assert http.post("/configuration-preview/experiment", json=data).status_code == 428
        stale = http.post("/configuration-preview/experiment", json=data, headers={**headers(), "X-Solarlab-Configuration-Schema": "0" * 64})
        assert stale.status_code == 409
        for patch, field in [({"kind": "jv_hysteresis"}, "kind"), ({"v_rate": "2 nm"}, "v_rate"),
                             ({"v_rate": 0}, "v_rate"), ({"n_points": 0}, "n_points"), ({"illuminated": None}, "illuminated"),
                             ({"kind": "dark_jv", "waveform": {}}, "waveform")]:
            bad = deepcopy(data)
            bad["experiment"].update(patch)
            result = http.post("/configuration-preview/experiment", json=bad, headers=headers())
            assert result.status_code == 422, result.text
            detail = result.json()["detail"]
            assert detail["status"] == "unresolved" and detail["can_execute"] is False
            assert ["experiment", field] in [row["loc"] for row in detail["field_errors"]]
    old = read_legacy_default_catalog(tuple(SourceDocument(name, (LEGACY / name).read_bytes()) for name in DEFAULT_PATHS),
        experiment_sources=(SourceDocument("backend/main.py", (ROOT / "perovskite-sim/backend/main.py").read_bytes()),
                            SourceDocument("solver/tolerances.py", (LEGACY / "solver/tolerances.py").read_bytes())))
    with TestClient(create_app(writer.root, configuration_preview=ConfigurationPreviewContext(old, resources))) as http:
        response = http.post("/configuration-preview/experiment", json=data, headers=headers())
        assert response.status_code == 503 and response.json()["detail"]["code"] == "configuration_experiment_context_missing"


def test_real_loopback_archive_waveform_and_order_error(catalog, resources, writer):
    archive = json.loads((ROOT / "tests/refactor/fixtures/jv_reference_requests.json").read_text())[0]
    original = archive["request"]
    source = SourceDocument(archive["provenance"]["source"], json.dumps(original["device"]).encode())
    device = import_standard_device(source, id="archive", defaults=catalog)
    data = {"schema_version": "solarlab.experiment-preparation.v1", "id": "archive_jv", "device": wire_input(device),
            "experiment": {"kind": "jv", **deepcopy(original["params"])}}
    context = ConfigurationPreviewContext(catalog, resources, (source,))
    server = _HTTPServer(writer.root, configuration_preview=context)
    with server:
        endpoint = f"http://127.0.0.1:{server.port}/configuration-preview/experiment"
        def post(value):
            return urlopen(Request(endpoint, data=json.dumps(value, allow_nan=False).encode(), headers=headers(), method="POST"), timeout=10)
        with post(data) as response:
            body = json.loads(response.read())
            assert response.status == 200
        resolved = assert_prepared(body, data, context)
        assert resolved["numerical_controls"] == {"rtol": 1e-4, "atol_m3": 1}
        declaration = resolved["protocol"]["declaration"]
        assert [phase["duration_s"] for phase in declaration["illumination_history"]] == [120, 30, .5, 55, 3, .5, 55]
        data["experiment"]["experiment_protocol"] = declaration
        declaration["sampling"]["values"][0] = "1 V"
        with pytest.raises(HTTPError) as error:
            post(data)
        assert error.value.code == 422
        detail = json.loads(error.value.read())["detail"]
        assert ["experiment", "experiment_protocol", "sampling", "values", 0] in [row["loc"] for row in detail["field_errors"]]
    assert server.stopped
