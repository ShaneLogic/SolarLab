"""The low-level port uses RunStore's existing fenced, byte-preserving API."""

import hashlib
import json
from typing import Any

import pytest

from solarlab.contracts.persistence import PersistencePort, PersistenceReader
from solarlab.io.artifacts import ArtifactPayload, bytes_artifact
from solarlab.io.run_store import AttemptToken, RunStore, StateConflict


IDENTITY = {"H_input": "port:input", "H_physics": "port:physics", "H_execution": "port:execution"}


def publish_result(port: PersistencePort[AttemptToken, ArtifactPayload]) -> dict[str, Any]:
    run = port.enqueue("run", opaque_identity=IDENTITY, input_metadata={"original": None})
    token = port.claim("run", expected_attempt_id=run["current_attempt_id"],
                       expected_generation=run["generation"], owner_id="owner")
    port.append_event(token, event_id="progress", payload={"completed": 1})
    artifact = port.publish(token, "result", bytes_artifact(b"\x00\xffunchanged\n", metadata={"unit": "1"}))
    terminal = port.complete(token, state="succeeded", result={"original_failure": None},
                             artifacts=(artifact["artifact_id"],))
    assert port.complete(token, state="succeeded", result={"original_failure": None},
                         artifacts=(artifact["artifact_id"],)) == terminal
    return port.artifact(artifact["artifact_id"])


def test_run_store_conforms_and_publishes_exact_bytes_through_port(tmp_path):
    with RunStore(tmp_path / "store") as store:
        port: PersistencePort[AttemptToken, ArtifactPayload] = store
        assert isinstance(store, PersistencePort)
        assert isinstance(store, PersistenceReader)
        artifact = publish_result(port)
        assert port.read_artifact(artifact["artifact_id"]) == b"\x00\xffunchanged\n"
        assert artifact["sha256"] == hashlib.sha256(b"\x00\xffunchanged\n").hexdigest()
        assert port.get_run("run")["request"]["identity"] == IDENTITY
        attempt, = port.attempts("run")
        assert attempt["state"] == "succeeded"
        assert attempt["terminal"]["artifacts"] == [{
            "artifact_id": artifact["artifact_id"], "manifest_sha256": artifact["manifest_sha256"],
        }]
        assert [row["kind"] for row in port.events(run_id="run")] == ["queued", "running", "progress", "terminal"]


def test_read_only_port_preserves_stored_records_and_refuses_writes(tmp_path):
    root = tmp_path / "store"
    with RunStore(root) as store:
        artifact = publish_result(store)
        recorded = json.dumps(store.attempts("run"), sort_keys=True)
    before = hashlib.sha256((root / "store.sqlite3").read_bytes()).hexdigest()
    with RunStore(root, read_only=True, immutable=True) as reader:
        port: PersistencePort[AttemptToken, ArtifactPayload] = reader
        assert json.dumps(port.attempts("run"), sort_keys=True) == recorded
        assert port.read_artifact(artifact["artifact_id"]) == b"\x00\xffunchanged\n"
        with pytest.raises(StateConflict, match="read-only RunStore"):
            port.enqueue("other", opaque_identity=IDENTITY, input_metadata={})
    assert hashlib.sha256((root / "store.sqlite3").read_bytes()).hexdigest() == before
