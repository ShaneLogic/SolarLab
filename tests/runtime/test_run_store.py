"""Real local SQLite/filesystem contracts; no solver, service or scheduler.

Faults are injected at actual file/SQL boundaries. Numerical arrays here are
only small serialization fixtures, including signed zeros and raw NaN words.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading

import numpy as np
import pytest

from solarlab.io.artifacts import (
    ArtifactIntegrityError, ArtifactPayload, array_artifact, bytes_artifact,
)
from solarlab.io.run_store import RunStore, StateConflict, StaleAttempt

ROOT = Path(__file__).resolve().parents[2]
IDENTITY = {"H_input": "upstream:input", "H_physics": "upstream:physics",
            "H_execution": "upstream:execution", "registry": {"id": "caller.v1"}}


@pytest.fixture
def store(tmp_path):
    with RunStore((tmp_path / "store").resolve()) as value:
        yield value


def enqueue(store, run_id="run", **metadata):
    return store.enqueue(run_id, opaque_identity=IDENTITY, input_metadata=metadata)


def claim(store, run_id="run", owner="coordinator"):
    queued = store.get_run(run_id)
    return store.claim(run_id, owner_id=owner,
                       expected_attempt_id=queued["current_attempt_id"],
                       expected_generation=queued["generation"])


def cancel(store, run_id="run", reason="user_request"):
    current = store.get_run(run_id)
    return store.cancel(run_id, reason=reason,
                        expected_attempt_id=current["current_attempt_id"],
                        expected_generation=current["generation"])


def terminal_events(store, run_id="run"):
    return [event for event in store.events(run_id=run_id) if event["kind"] == "terminal"]


def test_request_idempotence_detaches_metadata_and_preserves_presence_types_and_zero(store):
    metadata = {"zero_int": 0, "positive": 0.0, "negative": -0.0, "nullable": None,
                "false": False, "nested": {"values": [1, None, -0.0]}}
    original = store.enqueue("r", opaque_identity=IDENTITY, input_metadata=metadata)
    assert store.enqueue("r", opaque_identity=IDENTITY, input_metadata=metadata) == original
    metadata["nested"]["values"][0] = 77
    value = store.get_run("r")["request"]["input_metadata"]
    assert "omitted" not in value and value["nullable"] is None
    assert type(value["zero_int"]) is int and type(value["positive"]) is float
    assert value["nested"]["values"][0] == 1 and value["false"] is False
    assert math.copysign(1, value["negative"]) == -1
    assert math.copysign(1, value["positive"]) == 1
    value["negative"] = 0.0
    with pytest.raises(StateConflict):
        store.enqueue("r", opaque_identity=IDENTITY, input_metadata=value)
    assert len(store.events(run_id="r")) == 1


def test_owner_generation_and_store_identity_fence_real_claims(store, tmp_path):
    queued = enqueue(store)
    token = claim(store)
    assert token.generation > queued["generation"]
    with pytest.raises(FrozenInstanceError):
        token.owner_id = "someone_else"
    for wrong in (replace(token, generation=token.generation + 1),
                  replace(token, owner_id="someone_else"),
                  replace(token, attempt_id="not_current")):
        with pytest.raises(StaleAttempt):
            store.complete(wrong, state="succeeded", result={})
    with RunStore((tmp_path / "other").resolve()) as other:
        with pytest.raises(StaleAttempt):
            other.complete(token, state="succeeded", result={})
    with pytest.raises(StaleAttempt):
        store.claim("run", owner_id="other", expected_attempt_id=queued["current_attempt_id"],
                    expected_generation=queued["generation"])
    assert store.get_run("run")["state"] == "running"


def test_terminal_event_atomic_idempotence_and_persistent_cursor(store):
    enqueue(store)
    token = claim(store)
    progress = store.append_event(token, event_id="sample1", payload={"value": -0.0})
    assert store.append_event(token, event_id="sample1", payload={"value": -0.0}) == progress
    with pytest.raises(StateConflict):
        store.append_event(token, event_id="sample1", payload={"value": 0.0})
    terminal = store.complete(token, state="succeeded", result={"optional": None})
    assert store.complete(token, state="succeeded", result={"optional": None}) == terminal
    with pytest.raises(StateConflict):
        store.complete(token, state="failed", result={"optional": None})
    with pytest.raises(StateConflict):
        store.append_event(token, event_id="late", payload={})
    path = store.root
    before = store.events()
    store.close()
    with RunStore(path) as reopened:
        assert reopened.events() == before
        assert reopened.events(after=progress, limit=1)[0]["sequence"] == terminal
        assert reopened.complete(token, state="succeeded", result={"optional": None}) == terminal
        assert len(terminal_events(reopened)) == 1
        assert reopened.events(after=terminal) == []


def test_cancel_queued_retry_preserves_attempt_and_rejects_old_cancellation(store):
    original = enqueue(store)
    sequence = cancel(store)
    assert cancel(store) == sequence
    old = store.attempts("run")[0]
    assert old["owner_id"] is None and old["state"] == "cancelled"
    assert old["result"]["process_exit_observed"] is False
    queued = store.retry("run", expected_attempt_id=original["current_attempt_id"],
                         expected_generation=original["generation"])
    assert queued["current_attempt_id"] != old["attempt_id"]
    with pytest.raises(StaleAttempt):
        store.cancel("run", expected_attempt_id=old["attempt_id"],
                     expected_generation=old["generation"], reason="user_request")
    new = claim(store)
    store.complete(new, state="succeeded", result={})
    assert store.attempts("run")[0] == old
    assert len(terminal_events(store)) == 2  # Exactly one for each preserved attempt.
    with pytest.raises(StateConflict):
        store.retry("run", expected_attempt_id=new.attempt_id, expected_generation=new.generation)


def test_interrupted_owner_retry_with_same_name_still_fences_old_attempt(store):
    enqueue(store)
    old = claim(store, owner="same-owner")
    record = store.publish(old, "old", bytes_artifact(b"old", metadata={}))
    store.recover_interrupted()
    previous = store.attempts("run")[0]
    store.retry("run", expected_attempt_id=old.attempt_id, expected_generation=old.generation)
    new = claim(store, owner="same-owner")
    assert new.owner_id == old.owner_id and new.generation > old.generation
    assert new.attempt_id != old.attempt_id
    with pytest.raises(StaleAttempt):
        store.complete(old, state="succeeded", result={}, artifacts=[record["artifact_id"]])
    with pytest.raises(StaleAttempt):
        store.publish(old, "late", bytes_artifact(b"late", metadata={}))
    with pytest.raises(StaleAttempt):
        store.complete(new, state="succeeded", result={}, artifacts=[record["artifact_id"]])
    assert store.get_run("run")["state"] == "running"
    store.complete(new, state="succeeded", result={})
    assert store.attempts("run")[0] == previous
    assert len(terminal_events(store)) == 2


@pytest.mark.parametrize("first", ["cancel", "success"])
def test_cancel_and_success_cannot_overwrite_each_other(store, first):
    enqueue(store)
    token = claim(store)
    if first == "cancel":
        sequence = cancel(store)
        with pytest.raises(StateConflict):
            store.complete(token, state="succeeded", result={})
        assert cancel(store) == sequence
    else:
        store.complete(token, state="succeeded", result={})
        with pytest.raises(StateConflict):
            cancel(store)
    assert len(terminal_events(store)) == 1


def test_two_real_connections_race_cancel_against_success(store):
    enqueue(store)
    token = claim(store)
    barrier = threading.Barrier(2)
    with RunStore(store.root) as other:
        def finish():
            barrier.wait(timeout=2)
            try:
                return store.complete(token, state="succeeded", result={})
            except StateConflict:
                return "rejected"

        def stop():
            barrier.wait(timeout=2)
            try:
                return other.cancel("run", expected_attempt_id=token.attempt_id,
                                    expected_generation=token.generation, reason="user_request")
            except StateConflict:
                return "rejected"

        with ThreadPoolExecutor(max_workers=2) as pool:
            a, b = pool.submit(finish), pool.submit(stop)
            results = [a.result(timeout=5), b.result(timeout=5)]
        assert results.count("rejected") == 1
        assert store.get_run("run") == other.get_run("run")
        assert len(terminal_events(other)) == 1


def test_byte_publication_is_independent_exact_and_non_overwriting(store):
    enqueue(store)
    token = claim(store)
    raw = b"\x00\xff\x80-0\nnull\n"
    metadata = {"unit": None, "missing_reason": None, "zero": -0.0}
    payload = bytes_artifact(raw, metadata=metadata)
    metadata["zero"] = 9
    published = store.publish(token, "raw", payload)
    assert store.publish(token, "raw", payload) == published
    assert (store.root / published["relative_path"]).read_bytes() == raw
    assert len(list((store.root / "objects").iterdir())) == 1
    with pytest.raises(StateConflict):
        store.read_artifact(published["artifact_id"])
    for changed in (bytes_artifact(raw + b"x", metadata={}),
                    bytes_artifact(raw, metadata={"unit": None})):
        with pytest.raises(StateConflict):
            store.publish(token, "raw", changed)
    sequence = store.complete(token, state="succeeded", result={}, artifacts=[published["artifact_id"]])
    assert store.read_artifact(published["artifact_id"]) == raw
    assert math.copysign(1, store.artifact(published["artifact_id"])["metadata"]["zero"]) == -1
    assert store.complete(token, state="succeeded", result={}, artifacts=[published["artifact_id"]]) == sequence


@pytest.mark.parametrize("array", [
    np.array([0, 0x8000000000000000, 0x7FF8000000000042, 0xFFF8000000000019], dtype=">u8").view(">f8"),
    np.arange(12, dtype="<i4").reshape(3, 4)[:, ::2],
    np.array([-0.0]), np.array(-0.0), np.empty((0, 3), dtype="<f4"),
    np.array([True, False]), np.array([complex(-0.0, 0.0)], dtype=np.complex128),
    np.array([0, 2, 255], dtype=np.uint8).view(np.bool_),
    np.array([-0.0, 1.0], dtype=np.longdouble),
])
def test_npy_round_trip_preserves_dtype_shape_and_every_array_word(store, array):
    enqueue(store)
    token = claim(store)
    original_words, original_shape, original_dtype = array.tobytes(order="C"), array.shape, array.dtype.str
    payload = array_artifact(array, unit="C", metadata={"missing_reason": None})
    if array.size:
        array[...] = 0
    record = store.publish(token, "values", payload)
    store.complete(token, state="succeeded", result={}, artifacts=[record["artifact_id"]])
    result = store.read_array(record["artifact_id"])
    assert result.dtype.str == original_dtype and result.shape == original_shape
    assert result.tobytes() == original_words
    assert result.flags.writeable is False
    with pytest.raises(ValueError):
        result.setflags(write=True)
    standard_reader = np.load(store.root / record["relative_path"], allow_pickle=False)
    assert standard_reader.tobytes() == original_words
    assert record["metadata"]["unit"] == "C" and record["metadata"]["missing_reason"] is None


@pytest.mark.parametrize("metadata", [
    {"x": float("nan")}, {"x": float("inf")}, {"x": (1, 2)}, {1: "key"},
    {"x": b"raw"}, {"x": Path("somewhere")}, {"x": np.float64(1)},
])
def test_unsupported_metadata_never_coerces_values(metadata):
    with pytest.raises(ValueError):
        bytes_artifact(b"bytes", metadata=metadata)


@pytest.mark.parametrize("value", [
    [1.0], np.array([object()], dtype=object), np.array(["1"]),
    np.array([(1, 2)], dtype=[("a", "i4"), ("b", "i4")]),
])
def test_unsupported_array_serialization_is_explicit(value):
    with pytest.raises(ValueError):
        array_artifact(value, unit="1", metadata={})


def test_npy_structure_and_byte_count_validated_before_reading_values():
    payload = array_artifact(np.array([1.0]), unit="A", metadata={})
    for change in ({"shape": [2]}, {"shape": [1.0]}, {"dtype": ">f8"}, {"unit": None}):
        info = {**json.loads(payload.metadata_json), **change}
        with pytest.raises(ArtifactIntegrityError):
            replace(payload, metadata_json=json.dumps(info))
    with pytest.raises(ArtifactIntegrityError):
        replace(payload, data=payload.data + b"trailing")
    with pytest.raises(ValueError):
        ArtifactPayload(b"bytes", "pickle", "{}")
    with pytest.raises(ValueError):
        ArtifactPayload(b"bytes", "bytes", '{"duplicate":0,"duplicate":1}')
    stream = io.BytesIO()
    np.save(stream, np.array([{}], dtype=object), allow_pickle=True)
    with pytest.raises(ArtifactIntegrityError):
        ArtifactPayload(stream.getvalue(), "npy", payload.metadata_json)


def test_publication_sql_failure_leaves_traceable_orphan_and_never_success(store):
    enqueue(store)
    enqueue(store, "queued")
    token = claim(store)
    store._db.execute("""CREATE TEMP TRIGGER publication_failure BEFORE UPDATE OF phase ON artifacts
                        WHEN NEW.phase='published' BEGIN SELECT RAISE(ABORT,'injected after publication'); END""")
    with pytest.raises(sqlite3.DatabaseError, match="injected after publication"):
        store.publish(token, "raw", bytes_artifact(b"original", metadata={}))
    record = dict(store._db.execute("SELECT * FROM artifacts").fetchone())
    assert record["phase"] == "reserved"
    assert (store.root / record["relative_path"]).read_bytes() == b"original"
    assert terminal_events(store) == []
    path = store.root
    store.close()
    with RunStore(path) as reopened:
        assert reopened.get_run("run")["state"] == "running"  # Opening is not recovery.
        assert reopened.recover_artifacts(cleanup=True)["removed_paths"] == []
        recovery = reopened.recover_interrupted()
        assert recovery["interrupted_attempts"] == [token.attempt_id]
        assert recovery["process_exit_observed"] is False
        assert reopened.get_run("queued")["state"] == "queued"
        assert reopened.recover_artifacts(cleanup=True)["removed_paths"] == [record["relative_path"]]
        assert reopened.artifact(record["artifact_id"])["phase"] == "collected"
        assert reopened.recover_artifacts(cleanup=True)["removed_paths"] == []
        with pytest.raises(StateConflict):
            reopened.complete(token, state="succeeded", result={})


def test_terminal_sql_failure_rolls_back_all_db_references_then_matching_retry_commits(store):
    enqueue(store)
    token = claim(store)
    record = store.publish(token, "raw", bytes_artifact(b"data", metadata={}))
    store._db.execute("""CREATE TEMP TRIGGER terminal_failure AFTER INSERT ON events
                        WHEN NEW.kind='terminal' BEGIN SELECT RAISE(ABORT,'injected before commit'); END""")
    with pytest.raises(sqlite3.DatabaseError, match="injected before commit"):
        store.complete(token, state="succeeded", result={"value": 0}, artifacts=[record["artifact_id"]])
    assert store.get_run("run")["state"] == "running"
    assert store.artifact(record["artifact_id"])["phase"] == "published"
    assert terminal_events(store) == []
    assert store.attempts("run")[0]["finished_ns"] is None
    store._db.execute("DROP TRIGGER terminal_failure")
    sequence = store.complete(token, state="succeeded", result={"value": 0}, artifacts=[record["artifact_id"]])
    assert store.complete(token, state="succeeded", result={"value": 0}, artifacts=[record["artifact_id"]]) == sequence
    assert store.read_artifact(record["artifact_id"]) == b"data"
    assert len(terminal_events(store)) == 1


def test_partial_write_cleanup_is_registered_and_keeps_user_files(store, monkeypatch):
    enqueue(store)
    token = claim(store)
    user = store.root / "objects" / "user.bin"
    user.write_bytes(b"do not delete")
    unrelated_partial = store.root / "partial" / ("f" * 32 + ".part")
    unrelated_partial.write_bytes(b"user partial")

    def short_write(fd, data):
        os.write(fd, data[:3])
        raise OSError("injected partial write")

    monkeypatch.setattr(store._files, "_write", short_write)
    with pytest.raises(OSError, match="injected partial write"):
        store.publish(token, "raw", bytes_artifact(b"full data", metadata={}))
    record = dict(store._db.execute("SELECT * FROM artifacts").fetchone())
    partial = store.root / "partial" / (record["artifact_id"] + ".part")
    assert partial.read_bytes() == b"ful"
    assert not (store.root / record["relative_path"]).exists()
    assert store.recover_artifacts(cleanup=True)["removed_paths"] == []
    store.recover_interrupted()
    report = store.recover_artifacts(cleanup=True)
    assert report["removed_paths"] == ["partial/" + partial.name]
    assert user.read_bytes() == b"do not delete"
    assert unrelated_partial.read_bytes() == b"user partial"


def test_crash_after_atomic_link_before_partial_unlink_recovers_both_generated_links(store, monkeypatch):
    enqueue(store)
    token = claim(store)

    def interrupted_unlink(directory, name):
        raise OSError("injected after atomic link")

    with monkeypatch.context() as patch:
        patch.setattr(store._files, "_unlink_regular", interrupted_unlink)
        with pytest.raises(OSError, match="injected after atomic link"):
            store.publish(token, "raw", bytes_artifact(b"both links", metadata={}))
    record = dict(store._db.execute("SELECT * FROM artifacts").fetchone())
    final = store.root / record["relative_path"]
    partial = store.root / "partial" / (record["artifact_id"] + ".part")
    assert final.stat().st_ino == partial.stat().st_ino and final.stat().st_nlink == 2
    assert final.read_bytes() == partial.read_bytes() == b"both links"
    assert record["phase"] == "reserved"
    store.recover_interrupted()
    assert set(store.recover_artifacts(cleanup=True)["removed_paths"]) == {
        record["relative_path"], "partial/" + partial.name,
    }
    assert not final.exists() and not partial.exists()


def test_actual_process_exit_without_close_reopens_wal_and_recovers_only_generated_orphan(tmp_path):
    path = (tmp_path / "crash").resolve()
    script = """
import os, sys
sys.path.insert(0, sys.argv[1])
from solarlab.io.run_store import RunStore
from solarlab.io.artifacts import bytes_artifact
s = RunStore(sys.argv[2])
q = s.enqueue('crashed', opaque_identity={'H_input':'i','H_physics':'p','H_execution':'e'}, input_metadata={})
t = s.claim('crashed', expected_attempt_id=q['current_attempt_id'], expected_generation=q['generation'], owner_id='child')
s.publish(t, 'raw', bytes_artifact(b'published', metadata={}))
os._exit(23)
"""
    child = subprocess.run([sys.executable, "-I", "-B", "-c", script, str(ROOT / "src"), str(path)],
                           capture_output=True, text=True, timeout=5)
    assert child.returncode == 23, child.stdout + child.stderr
    with RunStore(path) as recovered:
        assert recovered.get_run("crashed")["state"] == "running"
        assert len(recovered.recover_interrupted()["interrupted_attempts"]) == 1
        assert len(recovered.recover_artifacts(cleanup=True)["removed_paths"]) == 1
        assert recovered.attempts("crashed")[0]["result"]["reason"] == "interrupted"


@pytest.mark.parametrize("damage", ["missing", "corrupt", "metadata"])
def test_damaged_pending_artifact_fails_without_first_reporting_success(store, damage):
    enqueue(store)
    token = claim(store)
    record = store.publish(token, "raw", bytes_artifact(b"valid", metadata={}))
    path = store.root / record["relative_path"]
    if damage == "missing":
        path.unlink()
    elif damage == "corrupt":
        path.write_bytes(b"wrong")
    else:
        store._db.execute("UPDATE artifacts SET metadata_json=? WHERE artifact_id=?",
                          ('{"unit":"wrong"}', record["artifact_id"]))
    with pytest.raises(ArtifactIntegrityError):
        store.complete(token, state="succeeded", result={}, artifacts=[record["artifact_id"]])
    assert store.get_run("run")["state"] == "failed"
    assert terminal_events(store)[0]["payload"]["result"]["reason"] == "artifact_integrity"


@pytest.mark.parametrize("damage", ["missing", "corrupt", "metadata"])
def test_damaged_committed_result_is_reported_and_never_deleted_or_relabelled(store, damage):
    enqueue(store)
    token = claim(store)
    record = store.publish(token, "raw", bytes_artifact(b"valid", metadata={"unit": None}))
    store.complete(token, state="succeeded", result={}, artifacts=[record["artifact_id"]])
    path = store.root / record["relative_path"]
    if damage == "missing":
        path.unlink()
    elif damage == "corrupt":
        path.write_bytes(b"wrong")
    else:
        store._db.execute("UPDATE artifacts SET metadata_json='{}' WHERE artifact_id=?", (record["artifact_id"],))
    with pytest.raises(ArtifactIntegrityError):
        store.read_artifact(record["artifact_id"])
    audit = store.recover_artifacts(cleanup=True)
    assert len(audit["issues"]) == 1 and audit["removed_paths"] == []
    assert store.get_run("run")["state"] == "succeeded"  # Historical terminal is immutable.
    assert len(terminal_events(store)) == 1
    if damage != "missing":
        assert path.exists()


def test_cleanup_preserves_committed_and_active_publications(store):
    enqueue(store, "done")
    good = claim(store, "done")
    valid = store.publish(good, "valid", bytes_artifact(b"valid", metadata={}))
    store.complete(good, state="succeeded", result={}, artifacts=[valid["artifact_id"]])
    enqueue(store, "active")
    active = claim(store, "active")
    pending = store.publish(active, "pending", bytes_artifact(b"pending", metadata={}))
    report = store.recover_artifacts(cleanup=True)
    assert report["issues"] == [] and report["removed_paths"] == []
    assert report["retained_unreferenced"] == [pending["artifact_id"]]
    assert store.read_artifact(valid["artifact_id"]) == b"valid"
    assert (store.root / pending["relative_path"]).read_bytes() == b"pending"


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/b", "", "a\\b"])
def test_caller_names_cannot_become_write_paths(store, name):
    with pytest.raises(ValueError):
        enqueue(store, name)
    enqueue(store)
    with pytest.raises(ValueError):
        store.publish(claim(store), name, bytes_artifact(b"x", metadata={}))
    assert list((store.root / "objects").iterdir()) == []


@pytest.mark.parametrize("target", ["objects", "partial", "store.sqlite3", "store.sqlite3-wal",
                                     "store.sqlite3-shm", "store.sqlite3-journal", ".writer.lock"])
def test_reserved_symlinks_cannot_write_or_delete_outside_root(tmp_path, target):
    root, outside = (tmp_path / "store").resolve(), (tmp_path / "outside").resolve()
    root.mkdir()
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_bytes(b"user data")
    (root / target).symlink_to(outside if target in {"objects", "partial"} else sentinel)
    with pytest.raises((OSError, ValueError)):
        RunStore(root)
    assert sentinel.read_bytes() == b"user data"
    assert list(outside.iterdir()) == [sentinel]


def test_root_symlink_and_noncanonical_or_relative_paths_rejected(tmp_path):
    outside = (tmp_path / "outside").resolve()
    outside.mkdir()
    link = tmp_path / "link"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        RunStore(link)
    with pytest.raises(ValueError):
        RunStore(Path("relative"))
    assert list(outside.iterdir()) == []


def test_registered_artifact_symlink_is_not_followed_or_collected(store, tmp_path):
    outside = tmp_path / "user.dat"
    outside.write_bytes(b"valid")
    enqueue(store)
    token = claim(store)
    record = store.publish(token, "raw", bytes_artifact(b"valid", metadata={}))
    path = store.root / record["relative_path"]
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ArtifactIntegrityError):
        store.complete(token, state="succeeded", result={}, artifacts=[record["artifact_id"]])
    assert store.recover_artifacts(cleanup=True)["issues"]
    assert path.is_symlink() and outside.read_bytes() == b"valid"


def test_sweep_association_binds_complete_opaque_execution_and_preserves_coordinates(store):
    enqueue(store)
    point = store.bind_sweep_point("sweep", "point", run_id="run", coordinates={"x": -0.0, "y": None})
    assert store.bind_sweep_point("sweep", "point", run_id="run", coordinates={"x": -0.0, "y": None}) == point
    assert point["opaque_execution_identity"] == IDENTITY["H_execution"]
    with pytest.raises(StateConflict):
        store.bind_sweep_point("sweep", "point", run_id="run", coordinates={"x": 0.0, "y": None})
    different = {**IDENTITY, "H_execution": "other-protocol"}
    store.enqueue("different", opaque_identity=different, input_metadata={})
    with pytest.raises(StateConflict):
        store.bind_sweep_point("sweep", "point", run_id="different", coordinates={"x": -0.0, "y": None})
    token = claim(store)
    store.complete(token, state="succeeded", result={})
    assert store.sweep_points("sweep")[0]["state"] == "succeeded"
    assert store.sweep_points("absent") == []


def test_schema_states_and_foreign_keys_are_enforced_without_destructive_upgrade(store):
    enqueue(store)
    with pytest.raises(sqlite3.IntegrityError):
        store._db.execute("UPDATE runs SET state='done'")
    with pytest.raises(sqlite3.IntegrityError):
        store._db.execute("UPDATE runs SET current_attempt_id='foreign'")
    assert store.get_run("run")["state"] == "queued"
    path = store.root
    store._db.execute("PRAGMA user_version=2")
    store.close()
    with pytest.raises(StateConflict, match="schema"):
        RunStore(path)
    with sqlite3.connect(path / "store.sqlite3") as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        assert connection.execute("SELECT state FROM runs").fetchone()[0] == "queued"


def test_unknown_database_is_not_repurposed(tmp_path):
    path = (tmp_path / "store").resolve()
    path.mkdir()
    db = path / "store.sqlite3"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE user_data(value)")
        connection.execute("INSERT INTO user_data VALUES('preserve')")
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    with pytest.raises(StateConflict):
        RunStore(path)
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before


def test_lock_wait_is_bounded_and_released_after_contention(store):
    import fcntl

    with RunStore(store.root, timeout=0.03) as other:
        fcntl.flock(store._files.lock_fd, fcntl.LOCK_EX)
        try:
            with pytest.raises(TimeoutError):
                other.events()
        finally:
            fcntl.flock(store._files.lock_fd, fcntl.LOCK_UN)
        assert other.events() == []


def test_package_import_has_no_store_environment_or_solver_side_effect(tmp_path):
    script = """
import importlib.abc, os, pathlib, sys
sys.path.insert(0, sys.argv[1])
class Boundary(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy','scipy','sksundae','flint','perovskite_sim','backend','solarlab_server','solarlab_research'}:
            raise AssertionError('unexpected import: ' + fullname)
sys.meta_path.insert(0, Boundary())
def audit(event, args):
    if event == 'sqlite3.connect' or event in {'os.mkdir','subprocess.Popen','socket.connect'}:
        raise AssertionError('import side effect: ' + event)
    if event == 'open' and (args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)):
        raise AssertionError('import write: ' + str(args))
sys.addaudithook(audit)
import solarlab.io
import solarlab.io.run_store
import solarlab.io.artifacts
assert solarlab.io.__all__ == ()
assert not list(pathlib.Path('.').iterdir())
"""
    child = subprocess.run([sys.executable, "-I", "-B", "-c", script, str(ROOT / "src")],
                           cwd=tmp_path, capture_output=True, text=True, timeout=5)
    assert child.returncode == 0, child.stdout + child.stderr
