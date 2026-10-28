"""Completed immutable run archives, with a digest-pinned read-only reader.

ZIP_STORED is a transport container, never an active database. It includes all
attempts/events and only committed, verified artifact references for one run.
Publication has no effect on the store's terminal state or scientific labels.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, BinaryIO
import uuid
import zipfile

from solarlab.io.artifacts import ArtifactIntegrityError, ArtifactPayload, _ArtifactFiles, _json_document, _load_npy
from solarlab.io.run_store import RunStore, StateConflict, _identifier, _public

__all__ = ["RunBundle"]

_TERMINAL = {"succeeded", "failed", "cancelled"}
_MAX_INDEX = 16 * 1024**2


def _digest(stream: BinaryIO) -> tuple[str, int]:
    stream.seek(0)
    result, size = hashlib.sha256(), 0
    while part := stream.read(1024**2):
        result.update(part)
        size += len(part)
    stream.seek(0)
    return result.hexdigest(), size


def _open(path: Path) -> BinaryIO:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ArtifactIntegrityError("bundle must be a regular independent file")
    return os.fdopen(fd, "rb")


def _sync_parent(path: Path) -> None:
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _publish(stream: BinaryIO, destination: Path, expected: str) -> None:
    """Copy an already complete local snapshot, then atomically link a name.

    A hard process crash may leave an unreferenced .part. Unknown files are
    preserved; an idempotent retry never treats such a partial as a bundle.
    """
    def verify_existing() -> None:
        with _open(destination) as existing:
            if _digest(existing)[0] != expected:
                raise StateConflict("bundle destination already contains different bytes")

    if destination.exists() or destination.is_symlink():
        verify_existing()
        _sync_parent(destination)  # A prior writer may have died immediately after link().
        return
    temporary = destination.with_name("." + destination.name + "." + uuid.uuid4().hex + ".part")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb") as output:
            stream.seek(0)
            while chunk := stream.read(1024**2):
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        with _open(temporary) as written:
            if _digest(written)[0] != expected:
                raise ArtifactIntegrityError("bundle copy failed digest verification")
        try:
            os.link(temporary, destination, follow_symlinks=False)
            _sync_parent(destination)
        except FileExistsError:
            verify_existing()
    finally:
        # Only this call's newly created unreferenced temporary is removed.
        temporary.unlink()
        _sync_parent(destination)


def _index_bytes(index: dict[str, Any]) -> bytes:
    encoded = json.dumps(index, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(encoded) > _MAX_INDEX:
        raise ValueError("run index exceeds the bounded archive limit")
    return encoded


def archive_run(store: RunStore, run_id: str, destination: Path, *, max_bytes: int) -> dict[str, Any]:
    if store.read_only:
        raise StateConflict("read-only RunStore cannot stage a new archive")
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("archive requires an explicit positive byte bound")
    if (not destination.is_absolute() or destination.resolve() != destination
            or destination.is_relative_to(store.root)):
        raise ValueError("bundle destination must be canonical, absolute and outside the active store")
    # tempfile is on the verified local store disk; never snapshot a live WAL
    # into the archive destination. The coordinator lock serializes writers.
    with tempfile.TemporaryFile(dir=store.root / "partial") as stream:
        with store._guard():
            run = _public(store._run(run_id))
            attempts = [_public(row) for row in store._db.execute("SELECT * FROM attempts WHERE run_id=? ORDER BY attempt_number", (run_id,))]
            if run["state"] not in _TERMINAL or any(row["state"] not in _TERMINAL for row in attempts):
                raise StateConflict("only a completed run with terminal attempts can be archived")
            events = [_public(row) for row in store._db.execute("SELECT * FROM events WHERE run_id=? ORDER BY sequence", (run_id,))]
            identifiers = store._db.execute("SELECT artifact_id FROM artifacts WHERE run_id=? AND phase='committed' ORDER BY artifact_id", (run_id,)).fetchall()
            artifacts = [store._artifact(row[0]) for row in identifiers]
            points = [_public(row) for row in store._db.execute("SELECT * FROM sweep_points WHERE run_id=? ORDER BY sweep_id,point_id", (run_id,))]
            index = {"schema": "solarlab.run-bundle.v1", "store_schema_version": 1, "store_id": store.store_id,
                     "run": run, "attempts": attempts, "events": events,
                     "artifacts": [_public(row) for row in artifacts], "sweep_points": points}
            encoded = _index_bytes(index)
            # Include ZIP headers/index overhead before reading large arrays.
            estimate = len(encoded) + sum(row["size_bytes"] + 1024 for row in artifacts) + 2048
            if estimate > max_bytes:
                raise ValueError("completed bundle exceeds the archive byte bound")
            with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                archive.writestr(zipfile.ZipInfo("index.json"), encoded)
                for row in artifacts:
                    data = store._verify(row)
                    archive.writestr(zipfile.ZipInfo(row["relative_path"]), data)
            stream.flush()
            os.fsync(stream.fileno())
        digest, size = _digest(stream)
        if size > max_bytes:
            raise ValueError("serialized bundle exceeds the archive byte bound")
        _publish(stream, destination, digest)
    return {"schema": "solarlab.run-bundle-publication.v1", "run_id": run_id, "store_id": store.store_id,
            "path": str(destination), "sha256": digest, "bytes": size, "state": run["state"], "contains_active_database": False}


class RunBundle:
    """Read a complete digest-pinned bundle without extracting or migrating it.

    Unknown experiment/result metadata is returned intact. Only the storage
    envelope is validated; success is never interpreted as scientific proof.
    """

    def __init__(self, path: str | Path, *, expected_sha256: str, max_bytes: int = 256 * 1024**2) -> None:
        if type(expected_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
            raise ValueError("bundle reader requires its independently supplied SHA-256")
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("bundle reader requires a positive byte bound")
        self._stream = _open(Path(path))
        try:
            if os.fstat(self._stream.fileno()).st_size > max_bytes:
                raise ArtifactIntegrityError("bundle exceeds the reader byte bound")
            if _digest(self._stream)[0] != expected_sha256:
                raise ArtifactIntegrityError("bundle digest mismatch")
            self._zip = zipfile.ZipFile(self._stream, "r")
            entries = self._zip.infolist()
            if (len({item.filename for item in entries}) != len(entries)
                    or any(item.compress_type != zipfile.ZIP_STORED for item in entries)):
                raise ArtifactIntegrityError("duplicate or compressed bundle members are unsupported")
            info = self._zip.getinfo("index.json")
            if info.file_size > _MAX_INDEX:
                raise ArtifactIntegrityError("oversized bundle index")
            raw = self._zip.read(info)

            def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
                if len({key for key, _ in items}) != len(items):
                    raise ArtifactIntegrityError("duplicate bundle index field")
                return dict(items)

            self._index = json.loads(raw, object_pairs_hook=pairs)
            if _index_bytes(self._index) != raw:
                raise ArtifactIntegrityError("noncanonical bundle index")
            self._validate()
            self.sha256 = expected_sha256
        except (KeyError, TypeError, ValueError, zipfile.BadZipFile, RecursionError) as error:
            self.close()
            if isinstance(error, ArtifactIntegrityError):
                raise
            raise ArtifactIntegrityError("invalid completed run bundle") from error
        except BaseException:
            self.close()
            raise

    def _validate(self) -> None:
        index = self._index
        if (type(index) is not dict or set(index) != {"schema", "store_schema_version", "store_id", "run", "attempts", "events", "artifacts", "sweep_points"}
                or index["schema"] != "solarlab.run-bundle.v1" or type(index["store_schema_version"]) is not int or index["store_schema_version"] != 1):
            raise ArtifactIntegrityError("unsupported bundle/storage schema")
        self.store_id = _identifier(index["store_id"])
        run = index["run"]
        self.run_id = _identifier(run["run_id"])
        for group in ("attempts", "events", "artifacts", "sweep_points"):
            if type(index[group]) is not list:
                raise ArtifactIntegrityError("bundle rows must be explicit lists")
            for row in index[group]:
                _json_document(row)
                if row["run_id"] != self.run_id:
                    raise ArtifactIntegrityError("bundle contains a foreign run")
        _json_document(run)
        if run["state"] not in _TERMINAL:
            raise ArtifactIntegrityError("bundle cannot contain an active run")
        attempts = {row["attempt_id"]: row for row in index["attempts"]}
        if len(attempts) != len(index["attempts"]) or run["current_attempt_id"] not in attempts:
            raise ArtifactIntegrityError("bundle attempt identity mismatch")
        current = attempts[run["current_attempt_id"]]
        if current["state"] != run["state"] or current["generation"] != run["generation"]:
            raise ArtifactIntegrityError("current run/attempt terminal mismatch")
        events = index["events"]
        sequences = [row["sequence"] for row in events]
        if sequences != sorted(set(sequences)) or any(type(number) is not int or number < 1 for number in sequences):
            raise ArtifactIntegrityError("event sequence is not strictly ordered")
        references: dict[str, tuple[str, str]] = {}
        queued_generation = 0
        previous_attempt: str | None = None
        for number, (key, attempt) in enumerate(attempts.items(), start=1):
            timeline = [row for row in events if row["attempt_id"] == key]
            queued = [row for row in timeline if row["kind"] == "queued"]
            running = [row for row in timeline if row["kind"] == "running"]
            terminal = [row for row in timeline if row["kind"] == "terminal"]
            if (type(attempt["attempt_number"]) is not int or attempt["attempt_number"] != number
                    or type(attempt["generation"]) is not int or len(queued) != 1 or len(running) > 1
                    or not timeline or timeline[0] is not queued[0] or timeline[-1]["kind"] != "terminal"
                    or queued[0]["event_key"] != "queued" or queued[0]["generation"] != queued_generation
                    or queued[0]["payload"] != ({} if previous_attempt is None else {"retry_of": previous_attempt})
                    or any(type(row["generation"]) is not int or row["kind"] not in {"queued", "running", "progress", "terminal"}
                           for row in timeline)):
                raise ArtifactIntegrityError("invalid attempt queue/retry lifecycle")
            if running:
                if (timeline[1] is not running[0] or running[0]["event_key"] != "running"
                        or running[0]["payload"] != {"owner_id": _identifier(attempt["owner_id"])}
                        or attempt["started_ns"] is None or attempt["generation"] != queued_generation + 1):
                    raise ArtifactIntegrityError("attempt owner/running lifecycle mismatch")
            elif (attempt["state"] != "cancelled" or attempt["owner_id"] is not None
                    or attempt["started_ns"] is not None or attempt["generation"] != queued_generation
                    or len(timeline) != 2):
                raise ArtifactIntegrityError("unstarted attempt must be a queued cancellation")
            if any(row["generation"] != attempt["generation"] for row in timeline[1:]):
                raise ArtifactIntegrityError("running/progress/terminal event generation mismatch")
            if (attempt["state"] not in _TERMINAL or len(terminal) != 1
                    or _json_document(terminal[0]["payload"]) != _json_document(attempt["terminal"])
                    or terminal[0]["generation"] != attempt["generation"]
                    or attempt["terminal"]["state"] != attempt["state"]
                    or _json_document(attempt["terminal"]["result"]) != _json_document(attempt["result"])):
                raise ArtifactIntegrityError("attempt does not have exactly one consistent terminal")
            queued_generation = attempt["generation"] + 1
            previous_attempt = key
            for ref in attempt["terminal"]["artifacts"]:
                if ref["artifact_id"] in references:
                    raise ArtifactIntegrityError("duplicate terminal artifact reference")
                references[ref["artifact_id"]] = (key, ref["manifest_sha256"])
        if previous_attempt != run["current_attempt_id"]:
            raise ArtifactIntegrityError("current attempt is not the last lifecycle")
        if any(row["attempt_id"] not in attempts for row in events):
            raise ArtifactIntegrityError("event refers to a foreign attempt")
        if len({(row["attempt_id"], row["event_key"]) for row in events}) != len(events):
            raise ArtifactIntegrityError("event idempotency key is duplicated")
        self._artifacts = {row["artifact_id"]: row for row in index["artifacts"]}
        if len(self._artifacts) != len(index["artifacts"]) or self._artifacts.keys() != references.keys():
            raise ArtifactIntegrityError("terminal/committed artifact coverage mismatch")
        names = {"index.json"}
        for key, row in self._artifacts.items():
            _ArtifactFiles.names(row)
            metadata = {**row, "metadata_json": _json_document(row["metadata"])}
            digest = hashlib.sha256(RunStore._manifest(metadata).encode()).hexdigest()
            if (row["phase"] != "committed" or digest != row["manifest_sha256"]
                    or references[key] != (row["attempt_id"], row["manifest_sha256"])
                    or row["generation"] != attempts[row["attempt_id"]]["generation"]):
                raise ArtifactIntegrityError("artifact owner/metadata binding differs")
            names.add(row["relative_path"])
            self.read_artifact(key)
        if set(self._zip.namelist()) != names:
            raise ArtifactIntegrityError("unknown or missing bundle member")
        for point in index["sweep_points"]:
            if point["opaque_execution_identity"] != run["request"]["identity"]["H_execution"]:
                raise ArtifactIntegrityError("sweep point execution identity mismatch")

    def get_run(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._index["run"]))

    def attempts(self) -> list[dict[str, Any]]:
        return json.loads(json.dumps(self._index["attempts"]))

    def events(self) -> list[dict[str, Any]]:
        return json.loads(json.dumps(self._index["events"]))

    def sweep_points(self) -> list[dict[str, Any]]:
        return json.loads(json.dumps(self._index["sweep_points"]))

    def artifact(self, artifact_id: str) -> dict[str, Any]:
        return json.loads(json.dumps(self._artifacts[artifact_id]))

    def read_artifact(self, artifact_id: str) -> bytes:
        row = self._artifacts[artifact_id]
        try:
            info = self._zip.getinfo(row["relative_path"])
            if info.file_size != row["size_bytes"]:
                raise ArtifactIntegrityError("bundle artifact size mismatch")
            data = self._zip.read(info)
        except (zipfile.BadZipFile, KeyError) as error:
            raise ArtifactIntegrityError("bundle artifact is missing or corrupt") from error
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ArtifactIntegrityError("bundle artifact digest mismatch")
        ArtifactPayload(data, row["format"], _json_document(row["metadata"]))
        return data

    def read_array(self, artifact_id: str) -> Any:
        row = self._artifacts[artifact_id]
        if row["format"] != "npy":
            raise StateConflict("artifact is not an NPY array")
        return _load_npy(self.read_artifact(artifact_id), row["metadata"])

    def close(self) -> None:
        if hasattr(self, "_zip"):
            self._zip.close()
        self._stream.close()

    def __enter__(self) -> RunBundle:
        return self

    def __exit__(self, *exception: Any) -> None:
        self.close()
