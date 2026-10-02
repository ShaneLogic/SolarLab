"""Small result-writer controls; no solver or saved large result is used."""
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import weakref

import pytest

from scripts import r1_v23_json_writer as writer


PHASES = ("result_json_validate", "result_json_stream", "result_json_sync", "result_json_replace")
COMPONENTS = ("validate_elapsed_s", "stream_encoding_elapsed_s", "utf8_elapsed_s",
              "file_write_elapsed_s", "flush_elapsed_s", "fsync_elapsed_s", "replace_elapsed_s")


def legacy_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def no_pending(directory):
    assert not list(directory.glob(".pending-*"))


class StreamProxy:
    def __init__(self, stream, *, fail=None, error=None, calls=None, close_error=None):
        self.stream = stream
        self.name = stream.name
        self.fail = fail
        self.error = error or OSError("injected " + str(fail))
        self.close_error = close_error
        self.calls = [] if calls is None else calls
        self.write_count = 0

    def write(self, raw):
        self.write_count += 1
        self.calls.append(("write", len(raw)))
        if self.fail == "write" and self.write_count == 2:
            raise self.error
        if self.fail == "short_write":
            return self.stream.write(raw[:-1])
        return self.stream.write(raw)

    def flush(self):
        self.calls.append(("flush", None))
        if self.fail == "flush":
            raise self.error
        return self.stream.flush()

    def fileno(self):
        return self.stream.fileno()

    def close(self):
        self.calls.append(("close", None))
        self.stream.close()
        if self.close_error is not None:
            raise self.close_error
        if self.fail == "close":
            raise self.error


def proxy_factory(monkeypatch, **kwargs):
    original = writer.tempfile.NamedTemporaryFile
    proxies = []

    def make(**options):
        assert options["prefix"] == ".pending-" and options["delete"] is False
        assert options["mode"] == "wb"
        proxy = StreamProxy(original(**options), **kwargs)
        proxies.append(proxy)
        return proxy

    monkeypatch.setattr(writer.tempfile, "NamedTemporaryFile", make)
    return proxies


@pytest.mark.parametrize("value", [
    None, True, 2**100, -0.0, (1, None, False),
    {"z": [], "a": {"z": 2, "a": 1}},
    {2: "two", -1: "negative", 10: "ten"},
    {"Unicode": "中文 café 🌍\n\t\0\\\"\u2028", "surrogate": "\ud800"},
    {"hi": [1.0, 0.0, -0.0], "lo": [math.ulp(1.) / 4, -0.0, 5e-324]},
])
def test_exact_legacy_bytes_sha_and_scalar_stats(tmp_path, value):
    path = tmp_path / "ResultV1.json"
    stats = writer.write_result_json(path, value)
    expected = legacy_bytes(value)
    assert path.read_bytes() == expected
    assert hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(expected).digest()
    assert path.read_bytes().endswith(b"\n") and not path.read_bytes().endswith(b"\n\n")
    assert stats["bytes_written"] == len(expected)
    assert all(type(item) in (str, int, float, bool) for item in stats.values())
    assert all(math.isfinite(item) and item >= 0 for item in stats.values() if type(item) is float)
    assert math.isclose(stats["total_elapsed_s"], stats["other_elapsed_s"] + sum(stats[k] for k in COMPONENTS))
    no_pending(tmp_path)


@pytest.mark.parametrize("limit", [1, 7, 64, 256])
def test_fixed_batches_aggregate_tokens_and_slice_large_strings(tmp_path, monkeypatch, limit):
    monkeypatch.setattr(writer, "_CHARACTER_LIMIT", limit)
    calls = []
    proxy_factory(monkeypatch, calls=calls)
    value = {"long": "中" * 150, "many": list(range(60)), "negative_zero": -0.0}
    path = tmp_path / "ResultV1.json"
    stats = writer.write_result_json(path, value)
    expected = legacy_bytes(value)
    assert path.read_bytes() == expected
    batches = [size for name, size in calls if name == "write"]
    assert all(0 < size <= limit for size in batches)
    assert sum(batches) == len(expected)
    assert stats["write_batches"] == len(batches) == math.ceil(len(expected) / limit)
    assert stats["max_batch_characters"] == stats["max_batch_bytes"] == max(batches)
    no_pending(tmp_path)


def test_negative_zero_low_word_remains_distinct(tmp_path):
    negative, positive = tmp_path / "negative.json", tmp_path / "positive.json"
    writer.write_result_json(negative, {"hi": 1.0, "lo": -0.0})
    writer.write_result_json(positive, {"hi": 1.0, "lo": 0.0})
    assert negative.read_bytes() != positive.read_bytes()
    assert b'"lo": -0.0' in negative.read_bytes()


@pytest.mark.parametrize("value,error", [
    (float("nan"), ValueError), ({"late": [1, float("inf")]}, ValueError),
    ({float("inf"): "key"}, ValueError), ({1: "integer", "a": "string"}, TypeError),
    ({(1, 2): "key"}, TypeError), ([1, b"bytes"], TypeError),
    ({"nested": Decimal("1.25")}, TypeError), ({"nested": object()}, TypeError),
])
def test_invalid_values_reject_before_writer_filesystem_access(tmp_path, monkeypatch, value, error):
    path = tmp_path / "ResultV1.json"
    path.write_bytes(b"existing\n")

    def forbidden(*args, **kwargs):
        raise AssertionError("writer touched filesystem before serialization rejection")

    monkeypatch.setattr(writer.tempfile, "NamedTemporaryFile", forbidden)
    monkeypatch.setattr(writer.os, "fsync", forbidden)
    monkeypatch.setattr(writer.os, "replace", forbidden)
    with pytest.raises(error):
        writer.write_result_json(path, value)
    assert path.read_bytes() == b"existing\n"
    with pytest.raises(error):
        writer.write_result_json(tmp_path / "missing-parent/ResultV1.json", value)
    no_pending(tmp_path)


def test_cycle_rejects_before_temporary_creation(tmp_path, monkeypatch):
    value = []; value.append(value)
    monkeypatch.setattr(writer.tempfile, "NamedTemporaryFile", lambda **kwargs: pytest.fail("unexpected temp"))
    with pytest.raises(ValueError, match="Circular"):
        writer.write_result_json(tmp_path / "ResultV1.json", value)
    assert not (tmp_path / "ResultV1.json").exists()


def second_encoder_fails(monkeypatch, error):
    original = writer._encoder
    calls = {"count": 0}

    class Broken:
        def iterencode(self, value):
            yield "["
            yield "0,1,2,3"
            raise error

    def factory():
        calls["count"] += 1
        return original() if calls["count"] == 1 else Broken()

    monkeypatch.setattr(writer, "_encoder", factory)
    monkeypatch.setattr(writer, "_CHARACTER_LIMIT", 2)
    return calls


@pytest.mark.parametrize("existing", [False, True])
def test_second_pass_failure_preserves_target_and_cleans_partial_temp(tmp_path, monkeypatch, existing):
    path = tmp_path / "ResultV1.json"
    if existing:
        path.write_bytes(b"existing\n")
    calls = second_encoder_fails(monkeypatch, ValueError("second pass failed"))
    events = []
    with pytest.raises(ValueError, match="second pass failed"):
        writer.write_result_json(path, [0, 1], phase_observer=lambda p, e, r: events.append((p, e)))
    assert calls["count"] == 2
    assert events == [(PHASES[0], "begin"), (PHASES[0], "end"), (PHASES[1], "begin"), (PHASES[1], "error")]
    assert path.read_bytes() == b"existing\n" if existing else not path.exists()
    no_pending(tmp_path)


@pytest.mark.parametrize("operation", ["write", "short_write", "flush", "fsync", "replace", "close"])
@pytest.mark.parametrize("existing", [False, True])
def test_io_errors_never_publish_partial_json(tmp_path, monkeypatch, operation, existing):
    path = tmp_path / "ResultV1.json"
    if existing:
        path.write_bytes(b"existing\n")
    monkeypatch.setattr(writer, "_CHARACTER_LIMIT", 7)
    proxy_factory(monkeypatch, fail=operation)
    if operation in ("fsync", "replace"):
        def fail(*args, **kwargs):
            raise OSError("injected " + operation)
        monkeypatch.setattr(writer.os, operation, fail)
    with pytest.raises(OSError):
        writer.write_result_json(path, {"long": list(range(30))})
    assert path.read_bytes() == b"existing\n" if existing else not path.exists()
    no_pending(tmp_path)


def test_success_flushes_fsyncs_closes_then_replaces_once(tmp_path, monkeypatch):
    path = tmp_path / "ResultV1.json"; path.write_bytes(b"old")
    calls = []; proxies = proxy_factory(monkeypatch, calls=calls)
    sync, replace = writer.os.fsync, writer.os.replace

    def observed_sync(fd):
        calls.append(("fsync", None)); return sync(fd)

    def observed_replace(source, target):
        assert Path(source).parent == path.parent and Path(target) == path
        assert proxies[0].stream.closed
        calls.append(("replace", None)); return replace(source, target)

    monkeypatch.setattr(writer.os, "fsync", observed_sync)
    monkeypatch.setattr(writer.os, "replace", observed_replace)
    writer.write_result_json(path, {"value": 1})
    assert [name for name, _ in calls if name != "write"] == ["flush", "fsync", "close", "replace"]
    assert path.read_bytes() == legacy_bytes({"value": 1})
    no_pending(tmp_path)


@pytest.mark.parametrize("error", [KeyboardInterrupt("interrupt"), SystemExit("exit")])
def test_base_exceptions_cleanup_and_preserve_original_identity(tmp_path, monkeypatch, error):
    path = tmp_path / "ResultV1.json"; path.write_bytes(b"old")
    second_encoder_fails(monkeypatch, error)
    with pytest.raises(type(error)) as caught:
        writer.write_result_json(path, [0])
    assert caught.value is error and path.read_bytes() == b"old"
    no_pending(tmp_path)


def test_close_and_unlink_cleanup_do_not_replace_primary_failure(tmp_path, monkeypatch):
    path = tmp_path / "ResultV1.json"; path.write_bytes(b"old")
    primary = ValueError("primary encoder failure")
    second_encoder_fails(monkeypatch, primary)
    proxy_factory(monkeypatch, close_error=OSError("secondary close failure"))
    unlink = Path.unlink

    def failed_unlink(target, *args, **kwargs):
        if target.name.startswith(".pending-"):
            raise PermissionError("secondary cleanup failure")
        return unlink(target, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", failed_unlink)
    try:
        with pytest.raises(ValueError) as caught:
            writer.write_result_json(path, [0])
        assert caught.value is primary
        assert any("temporary close" in note for note in primary.__notes__)
        assert any("temporary cleanup" in note for note in primary.__notes__)
        assert path.read_bytes() == b"old"
    finally:
        for pending in tmp_path.glob(".pending-*"):
            unlink(pending)


def test_observer_phases_roots_and_cost_are_explicit(tmp_path, monkeypatch):
    value = {"value": 1}; events = []; clock = {"now": 100.}
    monkeypatch.setattr(writer.time, "monotonic", lambda: clock["now"])

    def observe(phase, event, roots):
        assert set(roots) == {"result"} and roots["result"] is value
        events.append((phase, event)); clock["now"] += .125

    stats = writer.write_result_json(tmp_path / "ResultV1.json", value, phase_observer=observe)
    assert events == [(phase, event) for phase in PHASES for event in ("begin", "end")]
    assert stats["total_elapsed_s"] == stats["other_elapsed_s"] == 1.
    assert sum(stats[k] for k in COMPONENTS) == 0.


@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("event", ["begin", "end"])
def test_direct_observer_failure_propagates_with_defined_commit_point(tmp_path, phase, event):
    path = tmp_path / "ResultV1.json"; path.write_bytes(b"old")
    value = {"value": 1}

    def broken(actual_phase, actual_event, roots):
        if (actual_phase, actual_event) == (phase, event):
            raise RuntimeError("observer failed")

    with pytest.raises(RuntimeError, match="observer failed"):
        writer.write_result_json(path, value, phase_observer=broken)
    committed = (phase, event) == ("result_json_replace", "end")
    assert path.read_bytes() == (legacy_bytes(value) if committed else b"old")
    no_pending(tmp_path)


def test_error_observer_cannot_replace_encoder_error(tmp_path, monkeypatch):
    primary = ValueError("primary encoder failure")
    second_encoder_fails(monkeypatch, primary)

    def observer(phase, event, roots):
        if event == "error":
            raise KeyboardInterrupt("secondary observer interruption")

    with pytest.raises(ValueError) as caught:
        writer.write_result_json(tmp_path / "ResultV1.json", [0], phase_observer=observer)
    assert caught.value is primary
    assert any("error observation" in note for note in primary.__notes__)
    no_pending(tmp_path)


def test_stats_and_completed_phases_do_not_retain_input(tmp_path):
    class WeakDict(dict):
        pass
    value = WeakDict(value=1); reference = weakref.ref(value)
    stats = writer.write_result_json(tmp_path / "ResultV1.json", value, phase_observer=lambda *args: None)
    del value
    assert reference() is None and stats["bytes_written"] > 0
