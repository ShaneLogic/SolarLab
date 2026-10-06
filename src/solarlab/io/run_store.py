"""Local SQLite RunStore preparation, independent of solvers and services.

Use one coordinating writer and explicit canonical local storage roots. WAL,
FULL synchronous commits, and a bounded advisory lock serialize cooperating
handles. Transactions do not contain artifact I/O. Open/close never claims a
worker has stopped: interrupted-attempt recovery is an explicit authority
decision, and queued attempts remain queued. A retry creates a new attempt;
each attempt has exactly one terminal event. A run reflects its current attempt.

H_input/H_physics/H_execution are opaque upstream strings, not identities
computed or scientifically validated here. This component remains
prepared_pending_dependencies for P03-05/P05-01 and runtime integration.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Any, Iterator, Sequence
import uuid

from solarlab.io.artifacts import (
    ArtifactIntegrityError, ArtifactPayload, _ArtifactFiles, _json_document,
    _load_npy, _parse_document,
)

__all__ = ["RunStore", "AttemptToken", "StateConflict", "StaleAttempt"]

_TERMINAL = {"succeeded", "failed", "cancelled"}
_APPLICATION_ID = 0x534C5253
_SCHEMA_VERSION = 1
_STATES = "('queued','running','succeeded','failed','cancelled')"
_SCHEMA = (
    "CREATE TABLE store_meta(singleton INTEGER PRIMARY KEY CHECK(singleton=1), store_id TEXT NOT NULL)",
    f"""CREATE TABLE runs(
        run_id TEXT PRIMARY KEY, request_json TEXT NOT NULL,
        state TEXT NOT NULL CHECK(state IN {_STATES}),
        generation INTEGER NOT NULL CHECK(generation>=0), current_attempt_id TEXT,
        created_ns INTEGER NOT NULL,
        FOREIGN KEY(current_attempt_id) REFERENCES attempts(attempt_id)
            DEFERRABLE INITIALLY DEFERRED)""",
    f"""CREATE TABLE attempts(
        attempt_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
        attempt_number INTEGER NOT NULL CHECK(attempt_number>0),
        state TEXT NOT NULL CHECK(state IN {_STATES}), owner_id TEXT,
        generation INTEGER NOT NULL CHECK(generation>=0),
        result_json TEXT, terminal_json TEXT, created_ns INTEGER NOT NULL,
        started_ns INTEGER, finished_ns INTEGER,
        UNIQUE(run_id,attempt_number),
        CHECK((state IN ('succeeded','failed','cancelled')) = (terminal_json IS NOT NULL)),
        CHECK((state IN ('succeeded','failed','cancelled')) = (finished_ns IS NOT NULL)))""",
    """CREATE TABLE events(
        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL REFERENCES runs(run_id),
        attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id),
        generation INTEGER NOT NULL, kind TEXT NOT NULL, event_key TEXT NOT NULL,
        payload_json TEXT NOT NULL, created_ns INTEGER NOT NULL,
        UNIQUE(attempt_id,event_key))""",
    "CREATE UNIQUE INDEX one_terminal_event ON events(attempt_id) WHERE kind='terminal'",
    "CREATE INDEX run_events ON events(run_id,sequence)",
    """CREATE TABLE artifacts(
        artifact_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
        attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id),
        generation INTEGER NOT NULL, name TEXT NOT NULL,
        relative_path TEXT NOT NULL UNIQUE, sha256 TEXT NOT NULL,
        size_bytes INTEGER NOT NULL CHECK(size_bytes>=0), format TEXT NOT NULL,
        metadata_json TEXT NOT NULL, manifest_sha256 TEXT NOT NULL,
        phase TEXT NOT NULL CHECK(phase IN ('reserved','published','committed','collected')),
        created_ns INTEGER NOT NULL, UNIQUE(attempt_id,name))""",
    """CREATE TABLE sweep_points(
        sweep_id TEXT NOT NULL, point_id TEXT NOT NULL,
        run_id TEXT NOT NULL REFERENCES runs(run_id),
        opaque_execution_identity TEXT NOT NULL, coordinates_json TEXT NOT NULL,
        PRIMARY KEY(sweep_id,point_id))""",
)


class StateConflict(RuntimeError):
    """The request contradicts an already recorded state or idempotency key."""


class StaleAttempt(StateConflict):
    """The attempt, store, or owner generation no longer has authority."""


def _identifier(value: str) -> str:
    if type(value) is not str or re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.:-]{0,127}", value) is None:
        raise ValueError("expected a bounded stable identifier, not a path")
    return value


def _generation(value: int) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("generation must be a nonnegative integer")
    return value


def _object(value: Any) -> str:
    if type(value) is not dict:
        raise ValueError("expected an explicit JSON object")
    return _json_document(value)


def _public(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    return {
        key.removesuffix("_json"): (_parse_document(value) if value is not None else None)
        if key.endswith("_json") else value
        for key, value in dict(row).items()
    }


@dataclass(frozen=True, slots=True)
class AttemptToken:
    store_id: str
    run_id: str
    attempt_id: str
    owner_id: str
    generation: int

    def __post_init__(self) -> None:
        for value in (self.store_id, self.run_id, self.attempt_id, self.owner_id):
            _identifier(value)
        _generation(self.generation)


class RunStore:
    """Explicit store at ``root/store.sqlite3`` with sibling WAL and artifacts.

No process is launched/killed, no environment is inferred, and no recovery is
implicit. POSIX local filesystems supporting SQLite WAL, fsync and hard links
are required. Keep the root private to cooperating store users. File/DB
atomicity and hostile replacement of the root are not promised.
"""

    def __init__(self, root: str | Path, *, timeout: float = 5.0) -> None:
        if type(timeout) not in {int, float} or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        import fcntl

        self._fcntl = fcntl
        self._timeout = timeout
        self._mutex = threading.Lock()
        self._closed = False
        self._files = _ArtifactFiles(Path(root))
        self.root = self._files.root
        self.database_path = self.root / "store.sqlite3"
        self._db: sqlite3.Connection
        try:
            with self._guard():
                self._files.check_database_files()
                try:
                    fd = os.open("store.sqlite3", os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
                                 0o600, dir_fd=self._files.root_fd)
                except FileExistsError:
                    pass
                else:
                    os.close(fd)
                self._db = sqlite3.connect(self.database_path, timeout=timeout,
                                           isolation_level=None, check_same_thread=False)
                self._db.row_factory = sqlite3.Row
                self._initialize()
        except BaseException:
            if hasattr(self, "_db"):
                self._db.close()
            self._files.close()
            self._closed = True
            raise

    @contextmanager
    def _guard(self) -> Iterator[None]:
        if not self._mutex.acquire(timeout=self._timeout):
            raise TimeoutError("store handle is busy")
        locked = False
        try:
            if self._closed:
                raise RuntimeError("RunStore is closed")
            deadline = time.monotonic() + self._timeout
            while True:
                try:
                    self._fcntl.flock(self._files.lock_fd, self._fcntl.LOCK_EX | self._fcntl.LOCK_NB)
                    locked = True
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("store writer is busy") from None
                    time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
            self._files.check_database_files()
            yield
        finally:
            if locked:
                self._fcntl.flock(self._files.lock_fd, self._fcntl.LOCK_UN)
            self._mutex.release()

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        self._db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self._db.execute("COMMIT")
        except BaseException:
            if self._db.in_transaction:
                self._db.execute("ROLLBACK")
            raise

    def _initialize(self) -> None:
        application = self._db.execute("PRAGMA application_id").fetchone()[0]
        version = self._db.execute("PRAGMA user_version").fetchone()[0]
        tables = self._db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        fresh = application == 0 and version == 0 and not tables
        if not fresh and (application != _APPLICATION_ID or version != _SCHEMA_VERSION):
            raise StateConflict("unknown database/schema; no destructive or implicit migration")
        self._db.execute("PRAGMA foreign_keys=ON")
        mode = self._db.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        if mode != "wal":
            raise RuntimeError("local SQLite WAL is required")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.execute("PRAGMA temp_store=MEMORY")
        if fresh:
            with self._transaction():
                for statement in _SCHEMA:
                    self._db.execute(statement)
                self._db.execute("INSERT INTO store_meta(singleton,store_id) VALUES(1,?)", (uuid.uuid4().hex,))
                self._db.execute(f"PRAGMA application_id={_APPLICATION_ID}")
                self._db.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")
        self.store_id = _identifier(self._db.execute("SELECT store_id FROM store_meta WHERE singleton=1").fetchone()[0])
        os.fsync(self._files.root_fd)

    def __enter__(self) -> RunStore:
        return self

    def __exit__(self, *exception: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        with self._guard():
            self._db.close()
            self._closed = True
        self._files.close()

    def _run(self, run_id: str) -> dict[str, Any]:
        row = self._db.execute("SELECT * FROM runs WHERE run_id=?", (_identifier(run_id),)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return dict(row)

    def _attempt(self, attempt_id: str) -> dict[str, Any]:
        row = self._db.execute("SELECT * FROM attempts WHERE attempt_id=?", (_identifier(attempt_id),)).fetchone()
        if row is None:
            raise KeyError(attempt_id)
        return dict(row)

    def _expected(self, run_id: str, attempt_id: str, generation: int) -> dict[str, Any]:
        run = self._run(run_id)
        if run["current_attempt_id"] != _identifier(attempt_id) or run["generation"] != _generation(generation):
            raise StaleAttempt("current attempt/generation differs from the request")
        return run

    def _token(self, token: AttemptToken, *, running: bool = True) -> dict[str, Any]:
        if not isinstance(token, AttemptToken) or token.store_id != self.store_id:
            raise StaleAttempt("foreign store or invalid attempt token")
        self._expected(token.run_id, token.attempt_id, token.generation)
        attempt = self._attempt(token.attempt_id)
        if (attempt["run_id"] != token.run_id or attempt["owner_id"] != token.owner_id
                or attempt["generation"] != token.generation):
            raise StaleAttempt("attempt owner/generation mismatch")
        if running and attempt["state"] != "running":
            raise StateConflict("attempt is not running; publication authority is revoked")
        return attempt

    def _event(self, attempt: dict[str, Any], kind: str, key: str, payload: str) -> int:
        existing = self._db.execute(
            "SELECT * FROM events WHERE attempt_id=? AND event_key=?", (attempt["attempt_id"], key),
        ).fetchone()
        if existing is not None:
            if existing["kind"] != kind or existing["payload_json"] != payload:
                raise StateConflict("event idempotency key has different content")
            return existing["sequence"]
        cursor = self._db.execute(
            "INSERT INTO events(run_id,attempt_id,generation,kind,event_key,payload_json,created_ns) VALUES(?,?,?,?,?,?,?)",
            (attempt["run_id"], attempt["attempt_id"], attempt["generation"], kind, key, payload, time.time_ns()),
        )
        assert cursor.lastrowid is not None
        return cursor.lastrowid

    def _queue(self, run_id: str, generation: int, number: int, payload: dict[str, Any]) -> str:
        identifier = uuid.uuid4().hex
        self._db.execute(
            "INSERT INTO attempts(attempt_id,run_id,attempt_number,state,generation,created_ns) VALUES(?,?,?,'queued',?,?)",
            (identifier, run_id, number, generation, time.time_ns()),
        )
        self._db.execute("UPDATE runs SET state='queued',current_attempt_id=?,generation=? WHERE run_id=?",
                         (identifier, generation, run_id))
        self._event(self._attempt(identifier), "queued", "queued", _object(payload))
        return identifier

    def enqueue(
        self, run_id: str, *, opaque_identity: dict[str, Any], input_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Idempotent by run_id and exact normalized storage document, not physics.

        Callers supply H_input/H_physics/H_execution from their authority. Extra
        registry/schema/source labels are retained; none are invented here.
        """
        _identifier(run_id)
        _object(opaque_identity)
        for name in ("H_input", "H_physics", "H_execution"):
            if type(opaque_identity.get(name)) is not str or not opaque_identity[name].strip():
                raise ValueError(f"opaque upstream {name} must be explicit")
        _object(input_metadata)
        request = _object({"identity_kind": "opaque_upstream", "identity": opaque_identity,
                           "input_metadata": input_metadata})
        with self._guard(), self._transaction():
            row = self._db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                self._db.execute("""INSERT INTO runs(run_id,request_json,state,generation,current_attempt_id,created_ns)
                                 VALUES(?,?,'queued',0,NULL,?)""",
                                 (run_id, request, time.time_ns()))
                self._queue(run_id, 0, 1, {})
            elif row["request_json"] != request:
                raise StateConflict("run_id was already used with different input/identity")
            return _public(self._run(run_id))

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self._guard():
            return _public(self._run(run_id))

    def attempts(self, run_id: str) -> list[dict[str, Any]]:
        with self._guard():
            self._run(run_id)
            return [_public(row) for row in self._db.execute(
                "SELECT * FROM attempts WHERE run_id=? ORDER BY attempt_number", (run_id,))]

    def claim(
        self, run_id: str, *, expected_attempt_id: str, expected_generation: int, owner_id: str,
    ) -> AttemptToken:
        _identifier(owner_id)
        with self._guard(), self._transaction():
            run = self._expected(run_id, expected_attempt_id, expected_generation)
            if run["state"] != "queued":
                raise StateConflict("only a queued attempt can be claimed")
            generation = run["generation"] + 1
            self._db.execute("UPDATE runs SET state='running',generation=? WHERE run_id=?", (generation, run_id))
            self._db.execute("UPDATE attempts SET state='running',owner_id=?,generation=?,started_ns=? WHERE attempt_id=?",
                             (owner_id, generation, time.time_ns(), expected_attempt_id))
            self._event(self._attempt(expected_attempt_id), "running", "running", _object({"owner_id": owner_id}))
            return AttemptToken(self.store_id, run_id, expected_attempt_id, owner_id, generation)

    def append_event(self, token: AttemptToken, *, event_id: str, payload: dict[str, Any]) -> int:
        key, encoded = "user:" + _identifier(event_id), _object(payload)
        with self._guard(), self._transaction():
            return self._event(self._token(token), "progress", key, encoded)

    def events(self, *, after: int = 0, limit: int = 100, run_id: str | None = None) -> list[dict[str, Any]]:
        """Persistent global sequence cursor, suitable for a future SSE adapter."""
        if type(after) is not int or after < 0 or type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("invalid event cursor/page size")
        with self._guard():
            if run_id is None:
                rows = self._db.execute("SELECT * FROM events WHERE sequence>? ORDER BY sequence LIMIT ?", (after, limit))
            else:
                rows = self._db.execute("SELECT * FROM events WHERE sequence>? AND run_id=? ORDER BY sequence LIMIT ?",
                                        (after, _identifier(run_id), limit))
            return [_public(row) for row in rows]

    @staticmethod
    def _manifest(record: dict[str, Any]) -> str:
        return _object({key: record[key] for key in (
            "artifact_id", "run_id", "attempt_id", "generation", "name", "relative_path",
            "sha256", "size_bytes", "format", "metadata_json",
        )})

    def _artifact(self, artifact_id: str) -> dict[str, Any]:
        row = self._db.execute("SELECT * FROM artifacts WHERE artifact_id=?", (_identifier(artifact_id),)).fetchone()
        if row is None:
            raise KeyError(artifact_id)
        record = dict(row)
        self._files.names(record)
        if hashlib.sha256(self._manifest(record).encode()).hexdigest() != record["manifest_sha256"]:
            raise ArtifactIntegrityError("artifact metadata/identity digest mismatch")
        return record

    def _verify(self, record: dict[str, Any]) -> bytes:
        data = self._files.read(record)
        ArtifactPayload(data, record["format"], record["metadata_json"])
        return data

    def publish(self, token: AttemptToken, name: str, payload: ArtifactPayload) -> dict[str, Any]:
        """Reserve a generated path, publish bytes, then mark publication ready.

        Publication is not a result commit. Failures leave the registered file
        window available for explicit recovery. A matching duplicate returns
        the same object; different bytes or metadata under that name conflict.
        """
        _identifier(name)
        if not isinstance(payload, ArtifactPayload):
            raise ValueError("publish requires a validated ArtifactPayload")
        payload = ArtifactPayload(payload.data, payload.format, payload.metadata_json)
        content_sha256, size_bytes = payload.sha256, len(payload.data)
        with self._guard():
            with self._transaction():
                self._token(token)
                row = self._db.execute("SELECT artifact_id FROM artifacts WHERE attempt_id=? AND name=?",
                                       (token.attempt_id, name)).fetchone()
                if row is None:
                    identifier = uuid.uuid4().hex
                    record = dict(artifact_id=identifier, run_id=token.run_id, attempt_id=token.attempt_id,
                                  generation=token.generation, name=name, sha256=content_sha256,
                                  size_bytes=size_bytes, format=payload.format,
                                  metadata_json=payload.metadata_json,
                                  relative_path=f"objects/{identifier}-{content_sha256}.bin")
                    digest = hashlib.sha256(self._manifest(record).encode()).hexdigest()
                    self._db.execute(
                        """INSERT INTO artifacts(artifact_id,run_id,attempt_id,generation,name,relative_path,
                           sha256,size_bytes,format,metadata_json,manifest_sha256,phase,created_ns)
                           VALUES(?,?,?,?,?,?,?,?,?,?,?,'reserved',?)""",
                        tuple(record[key] for key in ("artifact_id", "run_id", "attempt_id", "generation", "name",
                              "relative_path", "sha256", "size_bytes", "format", "metadata_json")) + (digest, time.time_ns()),
                    )
                else:
                    identifier = row["artifact_id"]
                record = self._artifact(identifier)
                if (record["sha256"] != content_sha256 or record["size_bytes"] != size_bytes
                        or record["format"] != payload.format or record["metadata_json"] != payload.metadata_json):
                    raise StateConflict("artifact name was already used with different content/metadata")
                if record["phase"] not in {"reserved", "published"}:
                    raise StateConflict("artifact publication is no longer pending")
            self._files.publish(record, payload.data)
            self._verify(record)
            with self._transaction():
                self._token(token)
                self._db.execute("UPDATE artifacts SET phase='published' WHERE artifact_id=?", (identifier,))
            return _public(self._artifact(identifier))

    def _finish(self, attempt: dict[str, Any], state: str, result: dict[str, Any], records: list[dict[str, Any]]) -> int:
        terminal = _object({"state": state, "result": result, "artifacts": [
            {"artifact_id": item["artifact_id"], "manifest_sha256": item["manifest_sha256"]} for item in records]})
        with self._transaction():
            self._expected(attempt["run_id"], attempt["attempt_id"], attempt["generation"])
            current = self._attempt(attempt["attempt_id"])
            if current["state"] in _TERMINAL:
                if current["terminal_json"] != terminal:
                    raise StateConflict("attempt already has a contradictory terminal result")
                return self._event(current, "terminal", "terminal", terminal)
            for item in records:
                self._db.execute("UPDATE artifacts SET phase='committed' WHERE artifact_id=?", (item["artifact_id"],))
            self._db.execute("UPDATE attempts SET state=?,result_json=?,terminal_json=?,finished_ns=? WHERE attempt_id=?",
                             (state, _object(result), terminal, time.time_ns(), attempt["attempt_id"]))
            self._db.execute("UPDATE runs SET state=? WHERE run_id=?", (state, attempt["run_id"]))
            return self._event(current, "terminal", "terminal", terminal)

    def complete(
        self, token: AttemptToken, *, state: str, result: dict[str, Any], artifacts: Sequence[str] = (),
    ) -> int:
        """Commit verified references, state and terminal event in one transaction.

        Matching retries return the original sequence. Integrity failure before
        a first terminal commit records a failed attempt and raises; SQL failure
        rolls back and leaves publication recoverable. Later file loss is an
        integrity error, never permission to rewrite a historical terminal.
        """
        if state not in _TERMINAL:
            raise ValueError("completion requires succeeded, failed, or cancelled")
        _object(result)
        if isinstance(artifacts, (str, bytes)):
            raise ValueError("artifacts must be an explicit sequence of IDs")
        identifiers = tuple(_identifier(item) for item in artifacts)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("duplicate artifact reference")
        with self._guard():
            attempt = self._token(token, running=False)
            records = []
            for identifier in identifiers:
                try:
                    record = self._artifact(identifier)
                    if (record["attempt_id"] != token.attempt_id or record["generation"] != token.generation
                            or record["phase"] not in {"published", "committed"}):
                        raise StaleAttempt("artifact is not published by this attempt/generation")
                    self._verify(record)
                except ArtifactIntegrityError as error:
                    if attempt["state"] == "running":
                        self._finish(attempt, "failed", {"reason": "artifact_integrity", "detail": str(error)}, [])
                    raise
                records.append(record)
            return self._finish(attempt, state, result, records)

    def cancel(
        self, run_id: str, *, expected_attempt_id: str, expected_generation: int, reason: str,
    ) -> int:
        """Revoke commit authority; this does not signal, join, or release a worker."""
        if type(reason) is not str or not reason.strip():
            raise ValueError("cancellation requires an explicit reason")
        with self._guard():
            self._expected(run_id, expected_attempt_id, expected_generation)
            attempt = self._attempt(expected_attempt_id)
            return self._finish(attempt, "cancelled", {
                "reason": reason, "ownership_invalidated": True, "process_exit_observed": False,
            }, [])

    def retry(self, run_id: str, *, expected_attempt_id: str, expected_generation: int) -> dict[str, Any]:
        """An explicit new queued attempt; never overwrite a prior failure/cancel."""
        with self._guard(), self._transaction():
            run = self._expected(run_id, expected_attempt_id, expected_generation)
            if run["state"] not in {"failed", "cancelled"}:
                raise StateConflict("only failed/cancelled runs can be retried")
            previous = self._attempt(expected_attempt_id)
            self._queue(run_id, run["generation"] + 1, previous["attempt_number"] + 1,
                        {"retry_of": expected_attempt_id})
            return _public(self._run(run_id))

    def recover_interrupted(self) -> dict[str, Any]:
        """Explicitly fail recorded running attempts and retain queued work.

        The caller decides to invalidate old ownership. This method observes no
        OS process and does not authorize releasing a native execution slot.
        """
        with self._guard():
            runs = [dict(row) for row in self._db.execute("SELECT * FROM runs ORDER BY run_id")]
            interrupted, queued = [], []
            for run in runs:
                if run["state"] == "running":
                    self._finish(self._attempt(run["current_attempt_id"]), "failed", {
                        "reason": "interrupted", "ownership_invalidated": True, "process_exit_observed": False,
                    }, [])
                    interrupted.append(run["current_attempt_id"])
                elif run["state"] == "queued":
                    queued.append(run["current_attempt_id"])
            return {"interrupted_attempts": interrupted, "queued_attempts_retained": queued,
                    "process_exit_observed": False}

    def artifact(self, artifact_id: str) -> dict[str, Any]:
        with self._guard():
            return _public(self._artifact(artifact_id))

    def read_artifact(self, artifact_id: str) -> bytes:
        with self._guard():
            record = self._artifact(artifact_id)
            if record["phase"] != "committed":
                raise StateConflict("artifact is not a committed result")
            return self._verify(record)

    def read_array(self, artifact_id: str) -> Any:
        with self._guard():
            record = self._artifact(artifact_id)
            if record["phase"] != "committed" or record["format"] != "npy":
                raise StateConflict("artifact is not a committed NPY result")
            return _load_npy(self._verify(record), _parse_document(record["metadata_json"]))

    def recover_artifacts(self, *, cleanup: bool = False) -> dict[str, Any]:
        """Audit known objects; optionally collect only unreferenced terminal work.

        Active publications, unknown files, and all committed results survive.
        Corrupt/missing references are reported, not relabelled as success or
        deleted. Cleanup failure and a crash after unlink are retryable.
        """
        if type(cleanup) is not bool:
            raise ValueError("cleanup must be explicit boolean")
        with self._guard():
            rows = self._db.execute("SELECT artifact_id FROM artifacts ORDER BY artifact_id").fetchall()
            issues, removed, retained = [], [], []
            for row in rows:
                identifier = row["artifact_id"]
                try:
                    record = self._artifact(identifier)
                    if record["phase"] == "collected":
                        continue
                    if record["phase"] == "committed":
                        self._verify(record)
                    elif cleanup and self._attempt(record["attempt_id"])["state"] in _TERMINAL:
                        removed.extend(self._files.remove_unreferenced(record))
                        with self._transaction():
                            self._db.execute("UPDATE artifacts SET phase='collected' WHERE artifact_id=?", (identifier,))
                    else:
                        retained.append(identifier)
                        if record["phase"] == "published":
                            self._verify(record)
                except (ArtifactIntegrityError, OSError) as error:
                    issues.append({"artifact_id": identifier, "error": str(error)})
            return {"issues": issues, "removed_paths": removed, "retained_unreferenced": retained}

    def bind_sweep_point(
        self, sweep_id: str, point_id: str, *, run_id: str, coordinates: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist a point/run association, not a scheduler or qualified cache.

        The full opaque execution identity is retained. Semantic validation and
        scientific reuse eligibility still belong to P03-05/P05-05/P06-05.
        """
        _identifier(sweep_id)
        _identifier(point_id)
        encoded = _object(coordinates)
        with self._guard(), self._transaction():
            run = self._run(run_id)
            identity = json.loads(run["request_json"])["identity"]["H_execution"]
            row = self._db.execute("SELECT * FROM sweep_points WHERE sweep_id=? AND point_id=?", (sweep_id, point_id)).fetchone()
            if row is None:
                self._db.execute("""INSERT INTO sweep_points(sweep_id,point_id,run_id,opaque_execution_identity,coordinates_json)
                                 VALUES(?,?,?,?,?)""", (sweep_id, point_id, run_id, identity, encoded))
                row = self._db.execute("SELECT * FROM sweep_points WHERE sweep_id=? AND point_id=?", (sweep_id, point_id)).fetchone()
            elif row["run_id"] != run_id or row["opaque_execution_identity"] != identity or row["coordinates_json"] != encoded:
                raise StateConflict("sweep point was already bound to different input/execution")
            return _public(row)

    def sweep_points(self, sweep_id: str) -> list[dict[str, Any]]:
        with self._guard():
            return [_public(row) for row in self._db.execute(
                """SELECT p.*,r.state,r.current_attempt_id FROM sweep_points p
                   JOIN runs r ON r.run_id=p.run_id WHERE sweep_id=? ORDER BY point_id""",
                (_identifier(sweep_id),))]
