"""Trusted experiment preview over the existing service, including loopback HTTP."""
import builtins
from copy import deepcopy
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from fastapi.testclient import TestClient
import pytest

from solarlab.config.legacy_defaults import read_legacy_default_catalog
from solarlab.config.schema import configuration_schema_representation
from solarlab.experiments.two_dimensional.inputs import SpatialExperimentInput
from solarlab.experiments.two_dimensional.preparation import prepare_spatial_experiment
from solarlab.materials.source import SourceDocument
from solarlab_server.app import create_app
from solarlab_server.configuration import ConfigurationPreviewContext
from test_configuration_preview import DEFAULT_PATHS, LEGACY, ROOT, case, resources, writer, wire_input, words
from test_event_service import _HTTPServer


@pytest.fixture(scope="module")
def catalog():
    return read_legacy_default_catalog(tuple(SourceDocument(name, (LEGACY / name).read_bytes()) for name in DEFAULT_PATHS),
        experiment_sources=(SourceDocument("backend/main.py", (ROOT / "perovskite-sim/backend/main.py").read_bytes()),
                            SourceDocument("solver/tolerances.py", (LEGACY / "solver/tolerances.py").read_bytes())))


def payload(catalog, name="standard", **fields):
    device, sources = case(name, catalog)
    return {"schema_version": "solarlab.experiment-preparation.v1", "id": "spatial_http",
            "device": wire_input(device), "experiment": {"kind": "jv_2d", **fields}}, sources


def headers():
    return {"Content-Type": "application/json", "X-Solarlab-Configuration-Schema": configuration_schema_representation()[1]}


def assert_prepared(body, data, context):
    expected = prepare_spatial_experiment(SpatialExperimentInput.model_validate(data), context.defaults,
                                          context.resources, sources=context.sources)
    assert body["kind"] == "experiment" and body["can_execute"] is False
    assert body["status"] == "prepared_pending_dependencies"
    assert words(body["input"]) == words(expected.to_input().editing_data())
    assert words(body["resolved"]) == words(expected.to_mapping())
    assert body["identity"]["content_sha256"] == expected.content_sha256
    assert body["identity"]["configuration_schema_sha256"] == configuration_schema_representation()[1]


@pytest.mark.parametrize("name", ["standard", "scaps"])
def test_actual_service_composition_and_schema_binding(name, catalog, resources, writer, monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert not name.startswith(("perovskite_sim", "backend", "scipy.integrate")), name
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    data, sources = payload(catalog, name, lateral_length="0.5 um", V_max=-0.0, microstructure={}, lateral_bc=None)
    context = ConfigurationPreviewContext(catalog, resources, sources)
    run, events = writer.get_run("untouched"), writer.events()
    with TestClient(create_app(writer.root, configuration_preview=context)) as http:
        result = http.post("/configuration-preview/experiment", json=data, headers=headers())
        assert result.status_code == 200, result.text
        assert_prepared(result.json(), data, context)
        result.json()["resolved"]["device"]["layers"].clear()
        assert_prepared(http.post("/configuration-preview/experiment", json=data, headers=headers()).json(), data, context)
        old = http.post("/configuration-preview/device", json=data["device"])
        assert old.status_code == 200 and old.json()["kind"] == "device"
        assert http.get("/configuration-schema").content == configuration_schema_representation()[0]
    assert writer.get_run("untouched") == run and writer.events() == events


def test_missing_trusted_defaults_schema_mismatch_and_precise_errors(catalog, resources, writer):
    data, _ = payload(catalog)
    with TestClient(create_app(writer.root)) as http:
        assert http.post("/configuration-preview/experiment", json=data, headers=headers()).status_code == 503
    old_catalog = read_legacy_default_catalog(tuple(SourceDocument(name, (LEGACY / name).read_bytes()) for name in DEFAULT_PATHS))
    with TestClient(create_app(writer.root, configuration_preview=ConfigurationPreviewContext(old_catalog, resources))) as http:
        response = http.post("/configuration-preview/experiment", json=data, headers=headers())
        assert response.status_code == 503 and response.json()["detail"]["code"] == "configuration_experiment_context_missing"
    with TestClient(create_app(writer.root, configuration_preview=ConfigurationPreviewContext(catalog, resources))) as http:
        assert http.post("/configuration-preview/experiment", json=data).status_code == 428
        stale = http.post("/configuration-preview/experiment", json=data,
                          headers={**headers(), "X-Solarlab-Configuration-Schema": "0" * 64})
        assert stale.status_code == 409 and stale.json()["detail"]["code"] == "configuration_schema_mismatch"
        for field, value in (("Nx", 0), ("lateral_length", "1 ns"), ("atol", None)):
            bad = deepcopy(data); bad["experiment"][field] = value
            response = http.post("/configuration-preview/experiment", json=bad, headers=headers())
            assert response.status_code == 422
            detail = response.json()["detail"]
            assert ["experiment", field] in [error["loc"] for error in detail["field_errors"]]
            assert detail["status"] == "unresolved" and detail["can_execute"] is False
        bad = {**data, "defaults": catalog.to_mapping(), "server_path": "/tmp"}
        assert http.post("/configuration-preview/experiment", json=bad, headers=headers()).status_code == 422


def test_bounded_real_loopback_with_actual_preparation_and_grain_defaults(catalog, resources, writer):
    data, sources = payload(catalog)
    data["experiment"] = {"kind": "voc_grain_sweep", "grain_sizes_nm": [500, "1 um"], "illuminated": False}
    context = ConfigurationPreviewContext(catalog, resources, sources)
    server = _HTTPServer(writer.root, configuration_preview=context)
    with server:
        endpoint = f"http://127.0.0.1:{server.port}/configuration-preview/experiment"
        request = Request(endpoint, data=json.dumps(data, allow_nan=False).encode(), headers=headers(), method="POST")
        with urlopen(request, timeout=10) as response:
            raw = response.read()
            assert response.status == 200
        result = json.loads(raw)
        assert_prepared(result, data, context)
        assert result["resolved"]["experiment"]["controls"]["Ny_per_layer"] == 10
        bad = deepcopy(data); bad["experiment"]["grain_sizes_nm"] = [0]
        with pytest.raises(HTTPError) as error:
            urlopen(Request(endpoint, data=json.dumps(bad).encode(), headers=headers(), method="POST"), timeout=10)
        assert error.value.code == 422
        detail = json.loads(error.value.read())["detail"]
        assert ["experiment", "grain_sizes_nm", 0] in [row["loc"] for row in detail["field_errors"]]
    assert server.stopped
