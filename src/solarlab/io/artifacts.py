"""Validated byte/NPY artifacts and confined, non-overwriting file publication.

NumPy is imported only for explicit array operations. Numerical words are never
converted to JSON or pickle. Metadata accepts strict JSON values, not implicit
conversions of paths, dataclasses, tuples, NumPy scalars, or nonfinite scalars.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any

__all__ = [
    "ArtifactIntegrityError", "ArtifactPayload", "bytes_artifact", "array_artifact",
]


class ArtifactIntegrityError(ValueError):
    """A recorded artifact is missing, unsafe, corrupt, or incorrectly described."""


def _json_document(value: Any) -> str:
    def check(item: Any, depth: int) -> None:
        if depth > 32:
            raise ValueError("metadata exceeds the supported nesting depth")
        if item is None or type(item) in {bool, int}:
            return
        if type(item) is str:
            item.encode("utf-8")  # Reject unpaired surrogates.
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is list:
            for child in item:
                check(child, depth + 1)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for key, child in item.items():
                check(key, depth + 1)
                check(child, depth + 1)
            return
        raise ValueError(f"unsupported metadata type/value: {type(item).__name__}")

    check(value, 0)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(encoded.encode("utf-8")) > 2**20:
        raise ValueError("metadata exceeds 1 MiB; use an independent artifact")
    return encoded


def _parse_document(text: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        if len({key for key, _ in items}) != len(items):
            raise ValueError("duplicate metadata key")
        return dict(items)

    value = json.loads(text, object_pairs_hook=pairs)
    _json_document(value)
    return value


def _metadata(value: Any) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError("artifact metadata must be an explicit JSON object")
    copy = _parse_document(_json_document(value))
    if "missing_reason" in copy and copy["missing_reason"] is not None:
        if type(copy["missing_reason"]) is not str or not copy["missing_reason"].strip():
            raise ValueError("missing_reason must be null or a nonempty string")
    return copy


def _load_npy(data: bytes, metadata: dict[str, Any]) -> Any:
    import numpy as np

    stream = io.BytesIO(data)
    try:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError("unsupported NPY version")
    except (ValueError, EOFError, OSError) as error:
        raise ArtifactIntegrityError("invalid non-pickle NPY payload") from error
    if dtype.kind not in "biufc" or dtype.hasobject or dtype.metadata:
        raise ArtifactIntegrityError("only plain numeric/boolean dtypes are supported")
    declared_shape = metadata.get("shape")
    if (type(declared_shape) is not list or any(type(n) is not int or n < 0 for n in declared_shape)
            or declared_shape != list(shape) or metadata.get("dtype") != dtype.str
            or metadata.get("order") != "C" or fortran):
        raise ArtifactIntegrityError("NPY shape/dtype/order differs from its metadata")
    if type(metadata.get("unit")) is not str or not metadata["unit"].strip():
        raise ArtifactIntegrityError("an array requires an explicit unit label")
    # Check the declared byte count before allocating anything from a file header.
    if math.prod(shape) * dtype.itemsize != len(data) - stream.tell():
        raise ArtifactIntegrityError("NPY byte count mismatch or trailing data")
    array = np.frombuffer(data, dtype=dtype, offset=stream.tell()).reshape(shape)
    # Immutable independent bytes, never a view into an externally mutable file.
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class ArtifactPayload:
    data: bytes
    format: str
    metadata_json: str

    def __post_init__(self) -> None:
        if type(self.data) is not bytes:
            raise ValueError("artifact payload must be immutable bytes")
        metadata = _metadata(_parse_document(self.metadata_json))
        if self.format == "npy":
            _load_npy(self.data, metadata)
        elif self.format == "bytes":
            if set(metadata) & {"shape", "dtype", "order"}:
                raise ValueError("array metadata requires a validated NPY artifact")
        else:
            raise ValueError("unsupported artifact serialization")
        object.__setattr__(self, "metadata_json", _json_document(metadata))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


def bytes_artifact(data: bytes, *, metadata: dict[str, Any]) -> ArtifactPayload:
    """Keep the exact bytes; metadata fields, including null, remain explicit."""
    return ArtifactPayload(data, "bytes", _json_document(_metadata(metadata)))


def array_artifact(
    array: Any, *, unit: str, metadata: dict[str, Any],
) -> ArtifactPayload:
    """Snapshot array values/shape/dtype as NPY, preserving all element words.

Strides are normalized to C order without dtype conversion. Object, structured,
and other unsupported dtypes fail explicitly. Unit semantics belong upstream.
"""
    import numpy as np

    if (type(array) is not np.ndarray or array.dtype.kind not in "biufc"
            or array.dtype.hasobject or array.dtype.metadata):
        raise ValueError("expected a plain numeric/boolean numpy.ndarray")
    info = _metadata(metadata)
    if set(info) & {"shape", "dtype", "order", "unit"}:
        raise ValueError("array structure and unit have dedicated arguments")
    words = array.tobytes(order="C")
    info.update(shape=list(array.shape), dtype=array.dtype.str, order="C", unit=unit)
    stream = io.BytesIO()
    np.lib.format.write_array_header_1_0(stream, {
        "descr": array.dtype.str, "fortran_order": False, "shape": array.shape,
    })
    stream.write(words)  # No elementwise cast/copy can normalize stored words.
    return ArtifactPayload(stream.getvalue(), "npy", _json_document(info))


class _ArtifactFiles:
    """Private POSIX file area; names come only from DB publication reservations.

The store root must be a local, canonical absolute path under a trusted parent.
All writers/recovery use the RunStore advisory lock. This is not a sandbox for
    an attacker who can replace the database or store directories themselves.
    """

    objects_fd: int
    partial_fd: int

    def __init__(self, root: Path) -> None:
        if os.name != "posix" or not hasattr(os, "O_NOFOLLOW"):
            raise NotImplementedError("this file publisher requires POSIX no-follow operations")
        if not root.is_absolute() or root.resolve() != root:
            raise ValueError("store root must be an explicit canonical absolute local path")
        root.mkdir(mode=0o700, exist_ok=True)  # Never create parents outside the root.
        self.root = root
        self._fds: list[int] = []
        try:
            self.root_fd = self._open_dir(root)
            for name in ("objects", "partial"):
                try:
                    os.mkdir(name, mode=0o700, dir_fd=self.root_fd)
                except FileExistsError:
                    pass
                setattr(self, name + "_fd", self._open_dir(name, self.root_fd))
            self.lock_fd = os.open(
                ".writer.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
                0o600, dir_fd=self.root_fd,
            )
            self._fds.append(self.lock_fd)
            self._regular(self.lock_fd, require_single_link=True)
            os.fsync(self.root_fd)
        except BaseException:
            self.close()
            raise

    def _open_dir(self, path: Any, parent: int | None = None) -> int:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        self._fds.append(fd)
        return fd

    @staticmethod
    def _regular(fd: int, *, require_single_link: bool = False) -> os.stat_result:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or (require_single_link and info.st_nlink != 1):
            raise ArtifactIntegrityError("expected an independent regular file")
        return info

    def check_database_files(self) -> None:
        for name in ("store.sqlite3", "store.sqlite3-wal", "store.sqlite3-shm", "store.sqlite3-journal"):
            try:
                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self.root_fd)
            except FileNotFoundError:
                continue
            try:
                self._regular(fd, require_single_link=True)
            finally:
                os.close(fd)

    @staticmethod
    def names(record: dict[str, Any]) -> tuple[str, str]:
        identifier, digest = record["artifact_id"], record["sha256"]
        if (type(identifier) is not str or re.fullmatch(r"[0-9a-f]{32}", identifier) is None
                or type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None):
            raise ArtifactIntegrityError("invalid generated artifact identity")
        name = identifier + "-" + digest + ".bin"
        if record["relative_path"] != "objects/" + name:
            raise ArtifactIntegrityError("artifact path differs from its generated identity")
        return name, identifier + ".part"

    def _read(self, directory: int, name: str, record: dict[str, Any]) -> bytes:
        try:
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
            with os.fdopen(fd, "rb") as stream:
                info = self._regular(stream.fileno())
                if info.st_size != record["size_bytes"]:
                    raise ArtifactIntegrityError("artifact size mismatch")
                data = stream.read()
        except OSError as error:
            raise ArtifactIntegrityError(f"artifact unavailable: {name}: {error.strerror}") from error
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ArtifactIntegrityError("artifact content digest mismatch")
        return data

    def read(self, record: dict[str, Any]) -> bytes:
        name, _ = self.names(record)
        return self._read(self.objects_fd, name, record)

    @staticmethod
    def _write(fd: int, data: bytes) -> None:
        remaining = memoryview(data)
        while remaining:
            count = os.write(fd, remaining)
            if count <= 0:
                raise OSError("artifact write made no progress")
            remaining = remaining[count:]
        os.fsync(fd)

    @staticmethod
    def _unlink_regular(directory: int, name: str) -> bool:
        try:
            info = os.stat(name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            return False
        if not stat.S_ISREG(info.st_mode):
            raise ArtifactIntegrityError("recovery refuses a symlink or non-regular file")
        os.unlink(name, dir_fd=directory)
        os.fsync(directory)
        return True

    def publish(self, record: dict[str, Any], data: bytes) -> None:
        name, partial = self.names(record)
        try:
            os.stat(name, dir_fd=self.objects_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            self.read(record)  # Never replace even a corrupt existing object.
            self._unlink_regular(self.partial_fd, partial)
            return
        try:
            fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=self.partial_fd)
        except FileExistsError:
            self._read(self.partial_fd, partial, record)  # A partial write is not retried blindly.
        else:
            try:
                self._write(fd, data)
            finally:
                os.close(fd)  # Retain failed writes for registered recovery.
        self._read(self.partial_fd, partial, record)
        os.link(partial, name, src_dir_fd=self.partial_fd, dst_dir_fd=self.objects_fd,
                follow_symlinks=False)
        os.fsync(self.objects_fd)
        self._unlink_regular(self.partial_fd, partial)
        self.read(record)

    def remove_unreferenced(self, record: dict[str, Any]) -> list[str]:
        name, partial = self.names(record)
        removed = []
        try:
            os.stat(name, dir_fd=self.objects_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            self.read(record)  # Preserve corrupt evidence; do not remove unknown content.
            if self._unlink_regular(self.objects_fd, name):
                removed.append("objects/" + name)
        if self._unlink_regular(self.partial_fd, partial):
            removed.append("partial/" + partial)
        return removed

    def close(self) -> None:
        for fd in reversed(self._fds):
            os.close(fd)
        self._fds.clear()
