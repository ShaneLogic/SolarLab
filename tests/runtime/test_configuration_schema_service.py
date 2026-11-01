"""Prepared schema bytes over the actual local app; no scientific worker."""

from hashlib import sha256
from pathlib import Path
import re

from fastapi.testclient import TestClient

from solarlab.config.schema import (
    configuration_schema_representation, export_configuration_schema,
)
from solarlab.io.run_store import RunStore
from solarlab_server.app import create_app

ROOT = Path(__file__).resolve().parents[2]


def test_schema_route_exact_bytes_generated_identity_and_read_only_lifecycle(tmp_path):
    root = (tmp_path / "store").resolve()
    with RunStore(root) as writer:
        identity = {name: "infrastructure:unused" for name in
                    ("H_input", "H_physics", "H_execution")}
        writer.enqueue("untouched", opaque_identity=identity,
                       input_metadata={"zero": 0, "false": False, "null": None})
        run, events = writer.get_run("untouched"), writer.events()
        app = create_app(root)
        with TestClient(app) as client:
            response = client.get("/configuration-schema")
            data, digest = configuration_schema_representation()
            assert response.status_code == 200
            assert response.headers["content-type"] == "application/json"
            assert response.content == data
            assert response.json() == export_configuration_schema()
            generated = (ROOT / "web/src/generated/configuration-schema.ts").read_text()
            match = re.search(r'export const configurationSchemaSha256 = "([a-f0-9]{64})";', generated)
            assert match is not None
            assert sha256(response.content).hexdigest() == digest == match[1]
            assert response.headers["etag"] == f'"{digest}"'
            detached = response.json()
            detached["dto_schemas"].clear()
            assert client.get("/configuration-schema").content == data
            assert client.post("/configuration-schema", json={"change": True}).status_code == 405
        assert app.state.run_store is None
        assert writer.get_run("untouched") == run
        assert writer.events() == events
