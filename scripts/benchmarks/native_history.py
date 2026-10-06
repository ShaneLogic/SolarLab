"""Bounded lossless native JSONL recording; no numerical imports or implicit codec.

Pointers retain logical uncompressed offsets. Gzip is level 6, mtime 0, empty
filename, with Z_SYNC_FLUSH per record. A failed writer retains its readable
prefix and does not append a misleading footer. The outer artifact census still
owns logs/results and the total run cap; this writer reserves their original tail.
"""
from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import io
import json
from math import isfinite
import os
from pathlib import Path
from typing import Any, Callable, Literal, Mapping
import zlib

Encoding = Literal["raw", "gzip"]
METADATA_RESERVE = 1048576
# Empty final deflate block plus CRC32/ISIZE after a per-record sync flush.
# The actual final write is independently capacity checked as well.
GZIP_TRAILER_RESERVE = 10


class HistoryLimitError(ValueError):
    """A record, metadata publication or finalization would exceed its budget."""


@dataclass(frozen=True)
class HistoryPosition:
    logical_bytes: int
    encoded_bytes: int
    encoding: Encoding


def _integer(value: int, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _line(record: Mapping[str, Any]) -> bytes:
    return (json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n").encode()


def _reject_constant(text: str) -> Any:
    raise ValueError(f"non-finite JSON constant: {text}")


def _finite_float(text: str) -> float:
    value = float(text)
    if not isfinite(value):
        raise ValueError("non-finite JSON number")
    return value


class HistoryWriter:
    """Own one new history and its LastRecord/LastAccepted sidecars.

    Publication uses the retained runner's temporary-file/replace pattern after
    history flush. It is process-crash recovery, not a new power-loss durability
    guarantee. Metadata admission includes the transient old+new pointer copies.
    Unknown record kinds do not acquire a physical acceptance claim.
    """

    def __init__(self, path: Path, *, encoding: Encoding, total_output_bytes: int,
                 metadata_reserve_bytes: int = METADATA_RESERVE,
                 max_record_bytes: int = 1048576) -> None:
        if encoding not in ("raw", "gzip"):
            raise ValueError("encoding must be explicit raw or gzip")
        _integer(total_output_bytes, "total_output_bytes", 1)
        _integer(metadata_reserve_bytes, "metadata_reserve_bytes")
        _integer(max_record_bytes, "max_record_bytes", 1)
        self.path = Path(path)
        self.encoding = encoding
        self.total_output_bytes = total_output_bytes
        self.metadata_reserve_bytes = metadata_reserve_bytes
        self.max_record_bytes = max_record_bytes
        self.logical_bytes = self.encoded_bytes = self.records = 0
        self.closed = False
        self.container_complete: bool | None = False if encoding == "gzip" else None
        self.error: str | None = None
        self.first_exception: BaseException | None = None
        self._logical_hash = hashlib.sha256()
        self.cleanup_error: str | None = None
        self._sizes: dict[str, int] = {}
        self._buffer = io.BytesIO()
        self._gzip: gzip.GzipFile | None = None
        self._file: Any = None
        for name in ("LastRecord.json", "LastAccepted.json", "LastRecord.tmp", "LastAccepted.tmp", "FirstFailure.tmp"):
            if (self.path.parent / name).exists():
                self._buffer.close()
                raise FileExistsError(self.path.parent / name)
        try:
            first_failure = self.path.parent / "FirstFailure.json"
            if first_failure.exists():
                self._sizes["FirstFailure.json"] = first_failure.stat().st_size
                if self._sizes["FirstFailure.json"] > metadata_reserve_bytes:
                    raise HistoryLimitError("existing first failure exceeds metadata reservation")
            if encoding == "gzip":
                self._gzip = gzip.GzipFile(filename="", mode="wb", fileobj=self._buffer,
                                           compresslevel=6, mtime=0)
            header = self._take()
            self._admit(len(header), finishing=False)
            self._file = self.path.open("xb", buffering=0)
            self._write(header)
        except BaseException as error:
            self._fail(error)
            raise

    @property
    def position(self) -> HistoryPosition:
        return HistoryPosition(self.logical_bytes, self.encoded_bytes, self.encoding)

    @property
    def logical_sha256(self) -> str:
        """Digest of the complete record bytes successfully written so far."""
        return self._logical_hash.hexdigest()

    def _take(self) -> bytes:
        data = self._buffer.getvalue()
        self._buffer.seek(0)
        self._buffer.truncate(0)
        return data

    def _admit(self, size: int, *, finishing: bool) -> None:
        tail = GZIP_TRAILER_RESERVE if self.encoding == "gzip" and not finishing else 0
        if self.encoded_bytes + size + self.metadata_reserve_bytes + tail > self.total_output_bytes:
            raise HistoryLimitError("encoded history plus metadata/footer reservation exceeds cap")

    def _write(self, data: bytes) -> None:
        if data:
            written = self._file.write(data)
            self.encoded_bytes = self._file.tell()
            if written != len(data):
                raise OSError("short history write")
        self._file.flush()

    def _publication(self, row: dict[str, Any], logical: int, encoded: int) -> list[tuple[str, bytes]]:
        pointer = {"history_byte_offset_after_record": logical, "kind": row.get("kind"),
                   "record_sha256": row.get("record_sha256"),
                   "history_encoding": self.encoding, "encoded_byte_offset_after_record": encoded}
        values: list[tuple[str, Any]] = [("LastRecord.json", pointer)]
        if row.get("kind") == "voltage_lift_interval_charge" and row.get("observation", {}).get("passed") is True:
            values.append(("LastAccepted.json", {"pointer": pointer, "right": row["right"],
                                                "frame": row["coefficient_frame_identity"]}))
        plan = [(name, (json.dumps(value, allow_nan=False) + "\n").encode()) for name, value in values]
        self._admit_publications(plan)
        return plan

    def _admit_publications(self, plan: list[tuple[str, bytes]]) -> None:
        sizes = dict(self._sizes)
        for name, data in plan:
            if sum(sizes.values()) + len(data) > self.metadata_reserve_bytes:
                raise HistoryLimitError("pointer publication including temporary copy exceeds reservation")
            sizes[name] = len(data)

    def _publish(self, plan: list[tuple[str, bytes]]) -> None:
        for name, data in plan:
            target = self.path.parent / name
            temporary = target.with_suffix(".tmp")
            with temporary.open("xb") as stream:
                if stream.write(data) != len(data):
                    raise OSError("short pointer write")
                stream.flush()
            os.replace(temporary, target)
            self._sizes[name] = len(data)

    def publish_first_failure(self, value: Any) -> bool:
        """Keep the first supplied failure, including before a history refusal.

        This may be used after abort/finalization; it never changes the retained
        history or overwrites an earlier failure. The original metadata reserve
        includes the temporary publication copy and all owned sidecars.
        """
        target = self.path.parent / "FirstFailure.json"
        if target.exists():
            return False
        data = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
        plan = [("FirstFailure.json", data)]
        self._admit(0, finishing=True)
        self._admit_publications(plan)
        self._publish(plan)
        return True

    def abort(self, error: BaseException) -> None:
        """Close without copying a footer; retain the original failure identity."""
        self._fail(error)

    def append(self, record: Mapping[str, Any]) -> HistoryPosition:
        if self.closed:
            raise ValueError("history writer is closed")
        try:
            line = _line(record)
        except BaseException as error:
            self._fail(error)
            raise
        return self.append_line(line)

    def __call__(self, record: Mapping[str, Any]) -> HistoryPosition:
        return self.append(record)

    def append_line(self, line: bytes) -> HistoryPosition:
        """Preserve an already encoded complete JSON object without reserializing it."""
        if self.closed:
            raise ValueError("history writer is closed")
        try:
            if not isinstance(line, bytes) or len(line) > self.max_record_bytes:
                raise HistoryLimitError("record exceeds byte bound")
            if not line.endswith(b"\n") or line.count(b"\n") != 1:
                raise ValueError("one complete JSONL record is required")
            row = json.loads(line, parse_constant=_reject_constant, parse_float=_finite_float)
            if not isinstance(row, dict):
                raise ValueError("history records must be JSON objects")
            if row.get("kind") == "first_callback_failure":
                self.publish_first_failure(row)
            if self._gzip is None:
                encoded = line
            else:
                self._gzip.write(line)
                self._gzip.flush(zlib.Z_SYNC_FLUSH)
                encoded = self._take()
            self._admit(len(encoded), finishing=False)
            logical = self.logical_bytes + len(line)
            plan = self._publication(row, logical, self.encoded_bytes + len(encoded))
            self._write(encoded)
            self.logical_bytes = logical
            self._logical_hash.update(line)
            self.records += 1
            self._publish(plan)
            return self.position
        except BaseException as error:
            self._fail(error)
            raise

    def finish(self) -> HistoryPosition:
        if self.closed:
            if self.error is not None:
                raise ValueError("failed history cannot be finalized")
            return self.position
        try:
            if self._gzip is not None:
                self._gzip.close()
                trailer = self._take()
                self._admit(len(trailer), finishing=True)
                self._write(trailer)
            self._file.close()
            self._buffer.close()
            self.closed = True
            if self.encoding == "gzip":
                self.container_complete = True
            return self.position
        except BaseException as error:
            self._fail(error)
            raise

    def _fail(self, error: BaseException) -> None:
        if self.error is None:
            self.error = f"{type(error).__name__}: {error}"
            self.first_exception = error
        # Encoder output remains in memory on failure; never copy a footer for a
        # rejected/partially written record into the retained history file.
        try:
            if self._gzip is not None:
                self._gzip.close()
        except BaseException:
            pass
        try:
            if self._file is not None and not self._file.closed:
                self.encoded_bytes = self._file.tell()
                self._file.close()
        except BaseException as cleanup_error:
            self.cleanup_error = f"{type(cleanup_error).__name__}: {cleanup_error}"
        finally:
            self._buffer.close()
            self.closed = True

    def __enter__(self) -> HistoryWriter:
        return self

    def __exit__(self, kind: Any, error: Any, traceback: Any) -> None:
        if error is None:
            self.finish()
        elif not self.closed:
            self._fail(error)


def emit_record(record: Mapping[str, Any], emit: Callable[[Any], Any], *,
                logical_bytes: int, total_output_bytes: int) -> HistoryPosition:
    """Admission for both controller save paths; legacy callbacks stay raw.

    Only the explicit owned writer uses encoded accounting. A callback's return
    value never grants a budget bypass. Callers map HistoryLimitError to their
    existing path-specific ContractError and update counters after success only.
    """
    if isinstance(emit, HistoryWriter):
        if (emit.total_output_bytes != total_output_bytes
                or emit.metadata_reserve_bytes != METADATA_RESERVE
                or emit.logical_bytes != logical_bytes):
            raise HistoryLimitError("controller/writer budget or logical offset mismatch")
        return emit(record)
    size = len(_line(record))
    if logical_bytes + size > total_output_bytes - METADATA_RESERVE:
        raise HistoryLimitError("raw history plus original metadata reservation exceeds cap")
    emit(record)
    return HistoryPosition(logical_bytes + size, logical_bytes + size, "raw")


class HistoryReader:
    """Bounded binary lines for the existing native JSONL reader.

    EOF is not proof that a raw writer finished: raw container_complete is None.
    Gzip container_complete becomes True only after verified end-of-stream.
    JSON parsing stays with the original consumer; records() is a convenience
    with the same json.loads error behavior and explicit partial-tail retention.
    """

    def __init__(self, path: Path, *, encoding: Encoding, max_record_bytes: int,
                 max_logical_bytes: int) -> None:
        if encoding not in ("raw", "gzip"):
            raise ValueError("encoding must be explicit raw or gzip")
        _integer(max_record_bytes, "max_record_bytes", 1)
        _integer(max_logical_bytes, "max_logical_bytes")
        self.max_record_bytes = max_record_bytes
        self.max_logical_bytes = max_logical_bytes
        self.encoding = encoding
        self.logical_bytes = self.complete_lines = 0
        self.eof_seen = False
        self.container_complete: bool | None = False if encoding == "gzip" else None
        self.partial_tail = b""
        self.error: str | None = None
        self._raw = Path(path).open("rb")
        self._stream = gzip.GzipFile(fileobj=self._raw, mode="rb") if encoding == "gzip" else self._raw
        self.encoded_bytes_read = 0

    def __iter__(self) -> HistoryReader:
        return self

    def __next__(self) -> bytes:
        if self.error is not None:
            raise ValueError("history reader already failed")
        try:
            line = self._stream.readline(self.max_record_bytes + 1)
            self.encoded_bytes_read = self._raw.tell()
            if not line:
                if self.encoding == "gzip" and self.encoded_bytes_read == 0:
                    raise EOFError("gzip stream has no header or member")
                self.eof_seen = True
                if self.encoding == "gzip":
                    self.container_complete = True
                raise StopIteration
            if len(line) > self.max_record_bytes or self.logical_bytes + len(line) > self.max_logical_bytes:
                raise HistoryLimitError("decoded history exceeds read bound")
            self.logical_bytes += len(line)
            if line.endswith(b"\n"):
                self.complete_lines += 1
            else:
                self.partial_tail = line
            return line
        except StopIteration:
            raise
        except (OSError, EOFError, ValueError, zlib.error) as error:
            self.error = f"{type(error).__name__}: {error}"
            raise

    def records(self):
        for line in self:
            if line.endswith(b"\n"):
                try:
                    yield json.loads(line)
                except ValueError as error:
                    self.error = f"{type(error).__name__}: {error}"
                    raise

    def close(self) -> None:
        try:
            self._stream.close()
        finally:
            self._raw.close()

    def __enter__(self) -> HistoryReader:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
