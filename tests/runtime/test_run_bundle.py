"""Real archive/filesystem/SQLite faults, read-only history and bounded IO."""

from copy import deepcopy
import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
import zipfile
from unittest.mock import patch

import numpy as np
import pytest

from solarlab.io.artifacts import ArtifactIntegrityError, array_artifact, bytes_artifact
from solarlab.io.run_bundle import RunBundle
from solarlab.io.run_store import RunStore, StateConflict
from solarlab.io.storage import require_local_storage
import solarlab.io.run_bundle as bundle_module
from test_run_store import IDENTITY, claim, enqueue
from test_run_store import store as store


def complete(store, *, retry=False):
    enqueue(store, old_schema="unrecognized_historical_result_v0", evidence="FIRST_SLICE")
    token = claim(store)
    if retry:
        store.complete(token, state="failed", result={"original_failure": True, "raw_signed_zero": -0.0})
        store.retry("run", expected_attempt_id=token.attempt_id, expected_generation=token.generation)
        token = claim(store)
    words = np.array([0x8000000000000000, 0x7FF8000000000123, 0x3FF0000000000000], dtype="<u8").view("<f8")
    artifact = store.publish(token, "state", array_artifact(words, unit="m^-3", metadata={"missing_reason": None}))
    raw = store.publish(token, "original", bytes_artifact(b"old encoded result\0\xff", metadata={"source": "unchanged old format"}))
    store.bind_sweep_point("sweep", "point1", run_id="run", coordinates={"temperature_K": 300.})
    store.complete(token, state="succeeded", result={"old_result_version": 0, "unknown_field": [None, -0.0],
                                                  "scientific_status": "uncertified"},
                   artifacts=(artifact["artifact_id"], raw["artifact_id"]))
    return token, artifact, raw, words


def tree_bytes(path):
    return {str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()}


def test_completed_archive_is_deterministic_bitwise_and_keeps_failed_attempts(store, tmp_path):
    _, artifact, raw, words = complete(store, retry=True)
    run, attempts, events = store.get_run("run"), store.attempts("run"), store.events(run_id="run")
    destination = tmp_path / "completed.slrun"
    first = store.archive_run("run", destination)
    assert store.archive_run("run", destination) == first
    assert not first["contains_active_database"]
    with zipfile.ZipFile(destination) as archive:
        assert set(archive.namelist()) == {"index.json", artifact["relative_path"], raw["relative_path"]}
    store.close()
    original_bundle = destination.read_bytes()
    with RunBundle(destination, expected_sha256=first["sha256"]) as reader:
        assert reader.get_run() == run and reader.attempts() == attempts and reader.events() == events
        assert len([row for row in reader.events() if row["kind"] == "terminal"]) == 2
        assert reader.attempts()[0]["state"] == "failed"
        assert reader.read_array(artifact["artifact_id"]).tobytes() == words.tobytes()
        assert reader.read_artifact(raw["artifact_id"]) == b"old encoded result\0\xff"
        assert reader.sweep_points()[0]["opaque_execution_identity"] == IDENTITY["H_execution"]
        changed = reader.get_run()
        changed["request"].clear()
        assert reader.get_run()["request"] == run["request"]
        assert not hasattr(reader, "complete") and not hasattr(reader, "retry")
    assert destination.read_bytes() == original_bundle


@pytest.mark.parametrize("state", ("queued", "running"))
def test_active_runs_cannot_be_archived(store, tmp_path, state):
    enqueue(store)
    if state == "running":
        claim(store)
    destination = tmp_path / "not-complete.slrun"
    with pytest.raises(StateConflict, match="completed"):
        store.archive_run("run", destination)
    assert not destination.exists()
    assert store.get_run("run")["state"] == state


def test_bounded_export_and_nonoverwriting_destination_preserve_valid_results(store, tmp_path):
    complete(store)
    destination = tmp_path / "limited.slrun"
    with pytest.raises(ValueError, match="byte bound"):
        store.archive_run("run", destination, max_bytes=10)
    assert not destination.exists()
    destination.write_bytes(b"user-owned unrelated result")
    with pytest.raises(StateConflict, match="different"):
        store.archive_run("run", destination)
    assert destination.read_bytes() == b"user-owned unrelated result"
    assert store.get_run("run")["state"] == "succeeded"


@pytest.mark.parametrize("damage", ("missing", "changed"))
def test_committed_artifact_damage_prevents_export_without_rewriting_the_terminal(store, tmp_path, damage):
    _, artifact, _, _ = complete(store)
    path = store.root / artifact["relative_path"]
    path.unlink() if damage == "missing" else path.write_bytes(b"corrupt")
    before = store.attempts("run")
    with pytest.raises(ArtifactIntegrityError):
        store.archive_run("run", tmp_path / "damaged.slrun")
    assert store.attempts("run") == before
    assert store.recover_artifacts(cleanup=True)["issues"]


@pytest.mark.parametrize("mutation", ("missing_member", "extra_member", "terminal", "owner", "metadata", "schema", "duplicate_terminal"))
def test_resealed_internally_corrupt_bundle_is_refused(store, tmp_path, mutation):
    complete(store)
    path = tmp_path / "source.slrun"
    store.archive_run("run", path)
    with zipfile.ZipFile(path) as source:
        members = {name: source.read(name) for name in source.namelist()}
    index = json.loads(members["index.json"])
    if mutation == "missing_member":
        del members[index["artifacts"][0]["relative_path"]]
    elif mutation == "extra_member":
        members["../escape"] = b"not extracted"
    elif mutation == "terminal":
        index["run"]["state"] = "failed"
    elif mutation == "owner":
        index["artifacts"][0]["generation"] += 1
    elif mutation == "metadata":
        index["artifacts"][0]["metadata"]["unit"] = "wrong"
    elif mutation == "schema":
        index["store_schema_version"] = 999
    else:
        event = deepcopy(index["events"][-1])
        event["sequence"] += 1
        index["events"].append(event)
    members["index.json"] = json.dumps(index, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    bad = tmp_path / "bad.slrun"
    with zipfile.ZipFile(bad, "w") as target:
        for name, data in members.items():
            target.writestr(name, data)
    digest = hashlib.sha256(bad.read_bytes()).hexdigest()  # fixture, not real source acceptance
    with pytest.raises(ArtifactIntegrityError):
        RunBundle(bad, expected_sha256=digest)
    assert not (tmp_path / "escape").exists()


def test_bundle_requires_external_digest_and_checks_later_member_damage(store, tmp_path):
    _, artifact, _, _ = complete(store)
    path = tmp_path / "complete.slrun"
    receipt = store.archive_run("run", path)
    with pytest.raises(ArtifactIntegrityError, match="digest"):
        RunBundle(path, expected_sha256="0" * 64)
    with pytest.raises(ArtifactIntegrityError, match="byte bound"):
        RunBundle(path, expected_sha256=receipt["sha256"], max_bytes=1)
    with RunBundle(path, expected_sha256=receipt["sha256"]) as reader:
        with zipfile.ZipFile(path) as zipped:
            info = zipped.getinfo(artifact["relative_path"])
            offset = info.header_offset + 30 + len(info.filename.encode()) + len(info.extra)
        with path.open("r+b") as stream:
            stream.seek(offset)
            stream.write(b"X")
        with pytest.raises(ArtifactIntegrityError):
            reader.read_artifact(artifact["artifact_id"])


@pytest.mark.parametrize("mutation", ["owner", "running_generation", "progress_generation", "queued_generation", "missing_running", "event_after_terminal", "retry_generation"])
def test_resealed_bundle_checks_the_whole_attempt_lifecycle(store, tmp_path, mutation):
    enqueue(store)
    first = claim(store)
    store.complete(first, state="failed", result={"reason": "original fixture failure"})
    store.retry("run", expected_attempt_id=first.attempt_id, expected_generation=first.generation)
    second = claim(store)
    store.append_event(second, event_id="sample", payload={"zero": -0.0})
    store.complete(second, state="succeeded", result={"fixture": True})
    path = tmp_path / "lifecycle.slrun"
    store.archive_run("run", path)
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    index = json.loads(members["index.json"])
    current = index["attempts"][-1]
    timeline = [e for e in index["events"] if e["attempt_id"] == second.attempt_id]
    if mutation == "owner":
        current["owner_id"] = "contradictory-owner"
    elif mutation.endswith("_generation") and mutation != "retry_generation":
        kind = mutation.removesuffix("_generation")
        next(e for e in timeline if e["kind"] == kind)["generation"] += 10
    elif mutation == "missing_running":
        index["events"] = [e for e in index["events"] if e is not timeline[1]]
    elif mutation == "event_after_terminal":
        event = deepcopy(timeline[-2])
        event.update(sequence=index["events"][-1]["sequence"] + 1, event_key="user:late")
        index["events"].append(event)
    else:
        for event in timeline:
            event["generation"] += 10
        current["generation"] += 10
        index["run"]["generation"] += 10
    members["index.json"] = json.dumps(index, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    altered = tmp_path / "resealed.slrun"
    with zipfile.ZipFile(altered, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    # Independent fixture digest permits testing envelope consistency; it is
    # not a claim that ordinary tampering bypasses an external trusted pin.
    with pytest.raises(ArtifactIntegrityError):
        RunBundle(altered, expected_sha256=hashlib.sha256(altered.read_bytes()).hexdigest())


def test_queued_cancellation_and_retry_have_distinct_valid_generations(store, tmp_path):
    queued = enqueue(store)
    store.cancel("run", expected_attempt_id=queued["current_attempt_id"], expected_generation=queued["generation"], reason="cancel before start")
    first = store.archive_run("run", tmp_path / "queued-cancel.slrun")
    with RunBundle(first["path"], expected_sha256=first["sha256"]) as reader:
        assert reader.attempts()[0]["owner_id"] is None
        assert [e["generation"] for e in reader.events()] == [0, 0]
    store.retry("run", expected_attempt_id=queued["current_attempt_id"], expected_generation=queued["generation"])
    token = claim(store)
    store.append_event(token, event_id="progress", payload={"kept": None})
    store.complete(token, state="failed", result={"reason": "interrupted", "ownership_invalidated": True})
    retry = store.archive_run("run", tmp_path / "retry.slrun")
    with RunBundle(retry["path"], expected_sha256=retry["sha256"]) as reader:
        assert [e["generation"] for e in reader.events()] == [0, 0, 1, 2, 2, 2]
        assert [a["state"] for a in reader.attempts()] == ["cancelled", "failed"]


@pytest.mark.parametrize("boundary", ("partial", "linked"))
def test_actual_archive_process_exit_and_idempotent_recovery(store, tmp_path, boundary):
    complete(store)
    root = store.root
    before = store.events(run_id="run")
    store.close()
    destination = tmp_path / "crash.slrun"
    code = '''
import os,sys
from pathlib import Path
import solarlab.io.run_bundle as bundle
from solarlab.io.run_store import RunStore
boundary = sys.argv[3]
if boundary == "partial":
    def interrupted(stream, path, expected):
        stream.seek(0)
        path.with_name("abandoned.part").write_bytes(stream.read(40))
        os._exit(73)
    bundle._publish = interrupted
else:
    bundle._sync_parent = lambda path: os._exit(74)
with RunStore(Path(sys.argv[1])) as store:
    store.archive_run("run", Path(sys.argv[2]))
'''
    child = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(root), str(destination), boundary],
                           capture_output=True, timeout=15)
    assert child.returncode == (73 if boundary == "partial" else 74), child.stderr
    assert destination.exists() == (boundary == "linked")
    sync_calls = []
    original_sync = bundle_module._sync_parent
    def sync(path):
        sync_calls.append(path)
        original_sync(path)
    with RunStore(root) as reopened, patch.object(bundle_module, "_sync_parent", sync):
        assert reopened.events(run_id="run") == before
        publication = reopened.archive_run("run", destination)
        assert reopened.recover_artifacts(cleanup=True)["issues"] == []
    assert destination in sync_calls  # Includes link-before-fsync crash retry.
    with RunBundle(destination, expected_sha256=publication["sha256"]) as reader:
        assert reader.get_run()["state"] == "succeeded"
    if boundary == "partial":
        assert (tmp_path / "abandoned.part").read_bytes() != destination.read_bytes()


def test_closed_v1_store_opens_immutable_and_refuses_every_mutating_api(store, tmp_path):
    token, artifact, _, words = complete(store)
    root = store.root
    original_run = store.get_run("run")
    store.close()
    before = tree_bytes(root)
    with RunStore(root, read_only=True, immutable=True) as reader:
        assert reader.get_run("run") == original_run
        assert reader.read_array(artifact["artifact_id"]).tobytes() == words.tobytes()
        assert reader.recover_artifacts()["issues"] == []
        for operation in (
            lambda: enqueue(reader, "new"),
            lambda: reader.complete(token, state="succeeded", result={}),
            lambda: reader.publish(token, "late", bytes_artifact(b"late", metadata={})),
            reader.recover_interrupted,
            lambda: reader.recover_artifacts(cleanup=True),
            lambda: reader.archive_run("run", tmp_path / "derived.slrun"),
        ):
            with pytest.raises(StateConflict):
                operation()
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            reader._db.execute("UPDATE runs SET state='failed'")
    assert tree_bytes(root) == before  # no new WAL or sidecars, no migration


def test_read_only_live_handle_sees_committed_events_but_never_recovers_or_writes(store):
    enqueue(store)
    with RunStore(store.root, read_only=True) as reader:
        token = claim(store)
        assert reader.get_run("run")["state"] == "running"
        store.complete(token, state="failed", result={"reason": "original failure"})
        assert reader.get_run("run")["state"] == "failed"
        assert reader.events() == store.events()
    with pytest.raises(StateConflict, match="WAL"):
        RunStore(store.root, read_only=True, immutable=True)


def test_known_cloud_active_root_refuses_but_closed_old_result_is_readable(store, tmp_path):
    complete(store)
    root = store.root
    store.close()
    cloud = tmp_path / "Library" / "Mobile Documents" / "old-results"
    with pytest.raises(ValueError, match="cloud-sync"):
        RunStore(cloud)
    assert not cloud.exists()
    shutil.copytree(root, cloud)  # explicit old-result fixture, not an active DB
    original = tree_bytes(cloud)
    with RunStore(cloud, read_only=True, immutable=True) as reader:
        assert reader.get_run("run")["state"] == "succeeded"
    assert tree_bytes(cloud) == original


def test_local_mount_guard_and_unknown_schema_refusal_preserve_files(tmp_path):
    require_local_storage(tmp_path)
    absent = tmp_path / "absent"
    with pytest.raises(FileNotFoundError):
        RunStore(absent, read_only=True)
    assert not absent.exists()
    root = tmp_path / "v1"
    with RunStore(root):
        pass
    with sqlite3.connect(root / "store.sqlite3") as db:
        db.execute("PRAGMA user_version=999")
    db.close()
    original = tree_bytes(root)
    with pytest.raises(StateConflict, match="unknown database/schema"):
        RunStore(root, read_only=True, immutable=True)
    assert tree_bytes(root) == original


@pytest.mark.parametrize("outcome", ("partial", "failed"))
def test_draft_result_schema_is_actually_stored_and_read_without_evidence_promotion(store, tmp_path, outcome):
    from solarlab.experiments.results import ExperimentResult
    from test_experiment_schemas import result

    enqueue(store)
    token = claim(store)
    document = result(outcome, "FIRST_SLICE")
    document["header"].update(run_id=token.run_id, attempt_id=token.attempt_id)
    prepared = ExperimentResult.from_document(document)
    store.complete(token, state="failed" if outcome == "failed" else "succeeded", result=prepared.export())
    # The archive can be placed in a sync directory only after completion.
    destination = tmp_path / "Library" / "Mobile Documents" / "complete.slrun"
    destination.parent.mkdir(parents=True)
    receipt = store.archive_run("run", destination)
    with RunBundle(destination, expected_sha256=receipt["sha256"]) as reader:
        restored = ExperimentResult.from_document(reader.attempts()[0]["result"])
        assert restored.canonical_json == prepared.canonical_json
        assert restored.export()["header"]["evidence"][0]["label"] == "FIRST_SLICE"
        assert restored.export()["header"]["outcome"] == outcome


def test_network_mount_refuses_even_when_parent_mount_is_local(tmp_path, monkeypatch):
    import solarlab.io.storage as storage

    monkeypatch.setattr(storage.sys, "platform", "darwin")
    output = f"disk on / (apfs, local)\nserver:/data on {tmp_path} (nfs, mounted by test)\n"
    monkeypatch.setattr(storage.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout=output))
    with pytest.raises(ValueError, match="verified local"):
        require_local_storage(tmp_path / "active")
    # Ambiguous overlapping mounts cannot be certified as local by choosing
    # the lexicographically larger boolean from an underlying local mount.
    output += f"other on {tmp_path} (apfs, local)\n"
    with pytest.raises(ValueError, match="verified local"):
        require_local_storage(tmp_path / "active")


def test_cloud_path_case_cannot_bypass_the_guard_on_case_insensitive_volumes(tmp_path):
    with pytest.raises(ValueError, match="cloud-sync"):
        require_local_storage(tmp_path / "library" / "mobile documents" / "active")
