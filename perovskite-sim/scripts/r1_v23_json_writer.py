"""Result-only atomic JSON writer with bounded text batches.

The first encoder pass rejects serialization failures before this writer opens
any file. A second pass writes exactly the legacy indented JSON plus one LF.
The input must remain unchanged for the call. Batching does not bound the size
of an individual token produced inside JSONEncoder (for example a huge string)
or its sorting state. This helper changes neither compact canonical nor SHA.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import tempfile
import time


_CHARACTER_LIMIT = 256 * 1024


def _encoder():
    # Keep the legacy writer's indentation, default separators and ASCII
    # escaping. Compact canonical separators would change persisted bytes.
    return json.JSONEncoder(indent=2, sort_keys=True, allow_nan=False)


def _note(primary, action, secondary):
    try:
        primary.add_note(f"{action} also failed: {type(secondary).__name__}: {secondary}")
    except BaseException:
        # Diagnostic annotation must never replace the original failure.
        pass


@contextmanager
def _phase(observer, name, value):
    def emit(event):
        if observer is not None:
            observer(name, event, {"result": value})

    emit("begin")
    try:
        yield
    except BaseException as primary:
        try:
            emit("error")
        except BaseException as secondary:
            _note(primary, "writer error observation", secondary)
        raise
    else:
        emit("end")


def _text_batches(value, limit):
    """Group text up to the fixed character limit; never collect all tokens."""
    parts = []
    length = 0
    for token in _encoder().iterencode(value):
        offset = 0
        while offset < len(token):
            take = min(limit - length, len(token) - offset)
            parts.append(token[offset:offset + take])
            length += take
            offset += take
            if length == limit:
                yield "".join(parts)
                parts = []
                length = 0
    # The legacy writer appends exactly one LF after the whole document.
    parts.append("\n")
    yield "".join(parts)


def _close_preserving_primary(stream):
    primary = sys.exc_info()[1]
    try:
        stream.close()
    except BaseException as secondary:
        if primary is None:
            raise
        _note(primary, "writer temporary close", secondary)


def write_result_json(path, value, *, phase_observer=None):
    """Persist one unchanged JSON tree and return scalar diagnostic stats.

    Validation precedes destination I/O; rejection cannot truncate an existing
    target. After a successful fsync and close, os.replace is the commit point.
    An observer failure after that replace propagates with the complete new
    file already published; the writer does not roll it back.

    Component timings are disjoint. ``other_elapsed_s`` retains all remaining
    overhead, including callbacks, temporary-file setup/close and cleanup.
    ``total_elapsed_s`` includes every component and must not be added again.
    """
    started = time.monotonic()
    path = Path(path)
    limit = _CHARACTER_LIMIT
    if type(limit) is not int or limit <= 0:
        raise ValueError("JSON character batch limit must be a positive integer")
    stats = {
        "method": "validated_iterencode_indented_json_utf8_v1",
        "character_batch_limit": limit,
        "bytes_written": 0,
        "write_batches": 0,
        "max_batch_characters": 0,
        "max_batch_bytes": 0,
        "validate_elapsed_s": 0.0,
        "stream_encoding_elapsed_s": 0.0,
        "utf8_elapsed_s": 0.0,
        "file_write_elapsed_s": 0.0,
        "flush_elapsed_s": 0.0,
        "fsync_elapsed_s": 0.0,
        "replace_elapsed_s": 0.0,
    }
    with _phase(phase_observer, "result_json_validate", value):
        tick = time.monotonic()
        for _ in _encoder().iterencode(value):
            pass
        _ = None  # Do not retain the validation pass's final, possibly large token.
        stats["validate_elapsed_s"] = time.monotonic() - tick

    temporary = None
    try:
        stream = tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                                             prefix=".pending-", delete=False)
        temporary = Path(stream.name)
        try:
            with _phase(phase_observer, "result_json_stream", value):
                batches = _text_batches(value, limit)
                while True:
                    tick = time.monotonic()
                    try:
                        text = next(batches)
                    except StopIteration:
                        stats["stream_encoding_elapsed_s"] += time.monotonic() - tick
                        break
                    stats["stream_encoding_elapsed_s"] += time.monotonic() - tick
                    tick = time.monotonic()
                    encoded = text.encode("utf-8")
                    stats["utf8_elapsed_s"] += time.monotonic() - tick
                    tick = time.monotonic()
                    written = stream.write(encoded)
                    stats["file_write_elapsed_s"] += time.monotonic() - tick
                    if type(written) is not int or written != len(encoded):
                        raise OSError("incomplete JSON temporary-file write")
                    stats["bytes_written"] += written
                    stats["write_batches"] += 1
                    stats["max_batch_characters"] = max(stats["max_batch_characters"], len(text))
                    stats["max_batch_bytes"] = max(stats["max_batch_bytes"], len(encoded))
            with _phase(phase_observer, "result_json_sync", value):
                tick = time.monotonic()
                stream.flush()
                stats["flush_elapsed_s"] = time.monotonic() - tick
                tick = time.monotonic()
                os.fsync(stream.fileno())
                stats["fsync_elapsed_s"] = time.monotonic() - tick
        finally:
            _close_preserving_primary(stream)
        with _phase(phase_observer, "result_json_replace", value):
            tick = time.monotonic()
            os.replace(temporary, path)
            stats["replace_elapsed_s"] = time.monotonic() - tick
            temporary = None
    finally:
        if temporary is not None:
            primary = sys.exc_info()[1]
            try:
                temporary.unlink(missing_ok=True)
            except BaseException as secondary:
                if primary is None:
                    raise
                _note(primary, "writer temporary cleanup", secondary)

    total = time.monotonic() - started
    components = sum(stats[name] for name in (
        "validate_elapsed_s", "stream_encoding_elapsed_s", "utf8_elapsed_s",
        "file_write_elapsed_s", "flush_elapsed_s", "fsync_elapsed_s", "replace_elapsed_s"))
    stats.update(
        total_elapsed_s=total,
        other_elapsed_s=total - components,
        timing_scope="disjoint_components_plus_other_equal_total; never_add_total_again; callback_cost_not_subtracted",
        memory_scope="bounded_text_batches_not_a_bound_on_JSONEncoder_token_or_sorting_memory",
    )
    return stats
