"""Real-file qualification using the retained native prefix and accepted record."""
import ast
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess

import pytest

from scripts.benchmarks.native_history import (
    GZIP_TRAILER_RESERVE, METADATA_RESERVE, HistoryLimitError,
    HistoryReader, HistoryWriter, emit_record,
    INTEGER_IO_SCHEMA, INTEGER_IO_CODEC, INTEGER_HEX_TAG,
    configure_integer_io, decode_integer_values, integer_io_observed, integer_json_dumps,
)


@pytest.fixture
def original():
    name = os.environ.get("SOLARLAB_NATIVE_HISTORY_FIXTURE")
    manifest = (Path(name) if name is not None else
                Path(__file__).resolve().parents[1] / "fixtures/native_history/manifest.json")
    spec = json.loads(manifest.read_text())
    for key in ("prefix", "accepted", "live", "final", "reader"):
        spec[key]["path"] = str(manifest.parent / spec[key]["path"])
    data = {}
    for key in ("prefix", "accepted", "live", "final"):
        p = Path(spec[key]["path"])
        raw = p.read_bytes()
        if "stored_sha256" in spec[key]:
            assert hashlib.sha256(raw).hexdigest() == spec[key]["stored_sha256"]
        if spec[key].get("remove_suffix_bytes"):
            raw = raw[:-spec[key]["remove_suffix_bytes"]]
        if spec[key].get("encoding") == "gzip":
            raw = gzip.decompress(raw)
        assert hashlib.sha256(raw).hexdigest() == spec[key]["sha256"]
        data[key] = raw
    data["spec"] = spec
    return data


def reader(path, encoding, maximum=1048576):
    return HistoryReader(path, encoding=encoding, max_record_bytes=131072,
                          max_logical_bytes=maximum)


def integer_policy():
    return {"schema": INTEGER_IO_SCHEMA, "codec": INTEGER_IO_CODEC,
            "decimal_digits": 4300, "max_integer_bits": 131072}


@pytest.mark.parametrize("kind", ["voltage_lift_interval_charge", "voltage_lift_interval_charge_v2"])
def test_partition_history_v2_pointer_and_failed_successor(tmp_path, kind):
    """Storage semantics only: these synthetic records certify no physics."""
    path = tmp_path / "history.jsonl.gz"
    accepted = {"kind": kind, "observation": {"passed": True}, "right": {"point": "synthetic"},
                "coefficient_frame_identity": "a"*64, "record_sha256": "b"*64}
    failed = dict(accepted, observation={"passed": False}, record_sha256="c"*64)
    with HistoryWriter(path, encoding="gzip", total_output_bytes=2*METADATA_RESERVE) as writer:
        first = writer.append(accepted)
        pointer = (tmp_path / "LastAccepted.json").read_bytes()
        writer.append(failed)
        assert (tmp_path / "LastAccepted.json").read_bytes() == pointer
        assert json.loads(pointer)["pointer"]["history_byte_offset_after_record"] == first.logical_bytes
    with reader(path, "gzip") as stream:
        assert list(stream.records()) == [accepted, failed]
        assert stream.eof_seen and stream.container_complete


def test_bounded_integer_codec_preserves_exact_values_and_legacy_bytes(tmp_path):
    policy = configure_integer_io(integer_policy(), 1048576)
    small = {"a": [0, -1, True, None, -0.0, 1.25], "z": "unchanged"}
    options = {"allow_nan": False, "separators": (",", ":")}
    assert integer_json_dumps(small, **options) == json.dumps(small, **options)
    assert integer_json_dumps(small, policy, **options) == json.dumps(small, **options)
    large = 10**4400 + 1
    record = {"kind": "exact_integer_fixture", "signed": [large, -large, 0],
              "rational": [large, large+2]}
    path = tmp_path / "bounded.jsonl.gz"
    with HistoryWriter(path, encoding="gzip", total_output_bytes=2*METADATA_RESERVE,
                       integer_io_policy=policy) as writer:
        writer.append(record)
    with HistoryReader(path, encoding="gzip", max_record_bytes=1048576,
                       max_logical_bytes=1048576, integer_io_policy=policy) as decoded:
        assert list(decoded.records()) == [record]
        assert decoded.container_complete and decoded.eof_seen
    assert integer_io_observed(policy)["actual_decimal_digits"] == 4300
    with pytest.raises(ValueError, match="4300 digits"):
        json.dumps(large)


@pytest.mark.parametrize("change", [
    {"decimal_digits": 0}, {"max_integer_bits": True}, {"max_integer_bits": 131073},
    {"codec": "unknown"}, {"schema": "unknown"}, {"extra": 1},
])
def test_integer_policy_rejects_invalid_or_unbound_fields(change):
    with pytest.raises(ValueError, match="integer I/O policy"):
        configure_integer_io({**integer_policy(), **change}, 1048576)


@pytest.mark.parametrize("text", ["0x0", "-0x0", "+0x1", "0x01", "0X12", "0xff", "0x1"+"0"*32768])
def test_hex_integer_decode_rejects_noncanonical_small_and_oversize(text):
    policy = configure_integer_io(integer_policy(), 1048576)
    with pytest.raises(ValueError):
        decode_integer_values({INTEGER_HEX_TAG: text}, policy)


def test_integer_bit_bound_and_reserved_tag_fail_before_writing(tmp_path):
    policy = configure_integer_io(integer_policy(), 1048576)
    writer = HistoryWriter(tmp_path / "bounded.jsonl.gz", encoding="gzip",
                           total_output_bytes=2*METADATA_RESERVE, integer_io_policy=policy)
    with pytest.raises(HistoryLimitError, match=r"\$/oversize: 131073>131072"):
        writer.append({"kind": "fixture", "oversize": 1 << 131072})
    assert writer.closed and writer.records == 0 and writer.container_complete is False
    with pytest.raises(ValueError, match="reserved native integer tag"):
        integer_json_dumps({INTEGER_HEX_TAG: "0x1"}, policy)
    with pytest.raises(ValueError):
        decode_integer_values({INTEGER_HEX_TAG: hex(10**4400), "extra": 1}, policy)


def test_integer_policy_detects_process_setting_change(tmp_path):
    policy = configure_integer_io(integer_policy(), 1048576)
    try:
        sys.set_int_max_str_digits(640)
        with pytest.raises(HistoryLimitError, match="process decimal limit changed"):
            integer_json_dumps({"value": 1}, policy)
        assert integer_io_observed(policy, require_match=False)["actual_decimal_digits"] == 640
    finally:
        sys.set_int_max_str_digits(4300)


def old_reader(path, encoding, spec, accepted):
    source = Path(spec["reader"]["path"]).read_bytes()
    assert hashlib.sha256(source).hexdigest() == spec["reader"]["sha256"]
    module = ast.parse(source)
    if spec["reader"].get("fragment"):
        body = [node for node in module.body if not isinstance(node, ast.With)]
        loop = next(node for node in module.body if isinstance(node, ast.With))
    else:
        body = [node for node in module.body if 38 <= node.lineno <= 44]
        loop = next(node for node in module.body if isinstance(node, ast.With) and node.lineno == 45)
    loop.items[0].context_expr = ast.Call(ast.Name("open_history", ast.Load()), [], [])
    env = {"Counter": Counter, "defaultdict": defaultdict, "hashlib": hashlib,
           "json": json, "last_accepted": accepted, "open_history": lambda: reader(path, encoding)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body + [loop], type_ignores=[])),
                 str(spec["reader"]["path"]), "exec"), env)
    return env


@pytest.mark.parametrize("encoding", ["raw", "gzip"])
def test_native_bytes_values_and_accepted_pointer(tmp_path, original, encoding):
    path = tmp_path / ("history.jsonl.gz" if encoding == "gzip" else "history.jsonl")
    expected = original["prefix"] + original["accepted"]
    with HistoryWriter(path, encoding=encoding, total_output_bytes=2 * METADATA_RESERVE) as writer:
        for line in original["prefix"].splitlines(keepends=True):
            writer.append(json.loads(line))
        position = writer.append_line(original["accepted"])
        assert position.logical_bytes == len(expected)
        accepted = json.loads((tmp_path / "LastAccepted.json").read_text())
        assert accepted["pointer"]["history_byte_offset_after_record"] == len(expected)
        assert accepted["pointer"]["encoded_byte_offset_after_record"] == path.stat().st_size
        assert accepted["right"] == json.loads(original["accepted"])["right"]
        assert writer.container_complete is (False if encoding == "gzip" else None)
    with reader(path, encoding) as decoded:
        assert b"".join(decoded) == expected
        assert decoded.eof_seen and decoded.complete_lines == 17
        assert decoded.container_complete is (True if encoding == "gzip" else None)
    env = old_reader(path, encoding, original["spec"], accepted)
    assert env["records"] == 17 and env["parse_failure"] is None and not env["trailing"]
    assert env["accepted_pointer_record"]["right"] == accepted["right"]
    assert env["total_bytes"] == len(expected)
    with reader(path, encoding) as decoded:
        assert list(decoded.records()) == [json.loads(line) for line in expected.splitlines()]


def test_cap_admission_footer_and_metadata_peak(tmp_path):
    path = tmp_path / "too-small.gz"
    with pytest.raises(HistoryLimitError):
        HistoryWriter(path, encoding="gzip", total_output_bytes=10 + GZIP_TRAILER_RESERVE + 512 - 1,
                      metadata_reserve_bytes=512)
    assert not path.exists()
    path = tmp_path / "empty.gz"
    with HistoryWriter(path, encoding="gzip", total_output_bytes=532, metadata_reserve_bytes=512):
        pass
    assert path.stat().st_size == 20 and gzip.decompress(path.read_bytes()) == b""
    path = tmp_path / "bounded.gz"
    writer = HistoryWriter(path, encoding="gzip", total_output_bytes=600, metadata_reserve_bytes=512)
    writer.append({"kind": "tiny", "value": 0})
    before = path.read_bytes(); before_position = writer.position
    with pytest.raises(HistoryLimitError):
        writer.append({"kind": "too-large", "values": list(range(1000))})
    assert path.read_bytes() == before and writer.position == before_position
    assert writer.closed and writer.container_complete is False
    assert sum(p.stat().st_size for p in tmp_path.iterdir()) <= 600 + 20
    with pytest.raises(EOFError):
        gzip.decompress(path.read_bytes())
    with pytest.raises(ValueError, match="failed"):
        writer.finish()
    isolated = tmp_path / "metadata"; isolated.mkdir()
    writer = HistoryWriter(isolated / "raw", encoding="raw", total_output_bytes=200, metadata_reserve_bytes=1)
    with pytest.raises(HistoryLimitError, match="pointer"):
        writer.append({"kind": "x"})
    assert writer.position.logical_bytes == 0 and writer.path.stat().st_size == 0
    assert list(isolated.iterdir()) == [writer.path]


def test_publication_error_keeps_flushed_history_and_old_accepted(tmp_path, original, monkeypatch):
    writer = HistoryWriter(tmp_path / "history.gz", encoding="gzip", total_output_bytes=2 * METADATA_RESERVE)
    writer.append_line(original["accepted"])
    old = (tmp_path / "LastAccepted.json").read_bytes()
    replace = os.replace
    def fail_accepted(source, target):
        if Path(target).name == "LastAccepted.json":
            assert writer.path.stat().st_size == writer.position.encoded_bytes
            raise OSError("injected accepted publication failure")
        replace(source, target)
    monkeypatch.setattr("scripts.benchmarks.native_history.os.replace", fail_accepted)
    with pytest.raises(OSError, match="injected"):
        writer.append_line(original["accepted"])
    assert (tmp_path / "LastAccepted.json").read_bytes() == old
    assert (tmp_path / "LastAccepted.tmp").exists()
    current = json.loads((tmp_path / "LastRecord.json").read_text())
    assert current["history_byte_offset_after_record"] == 2 * len(original["accepted"])
    assert writer.closed and writer.container_complete is False
    assert sum(p.stat().st_size for p in tmp_path.iterdir()) <= writer.total_output_bytes


def test_reader_complete_live_corrupt_partial_and_malformed(tmp_path, original):
    final_path = Path(original["spec"]["final"]["path"])
    live, final = original["live"], original["final"]
    for key, data in (("live", live), ("final", final)):
        assert hashlib.sha256(data).hexdigest() == original["spec"][key]["sha256"]
    for name, data, error in [("live", live, EOFError), ("truncated", final[:-4], EOFError),
                              ("crc", final[:-8] + bytes([final[-8] ^ 1]) + final[-7:], gzip.BadGzipFile)]:
        path = tmp_path / name; path.write_bytes(data)
        with reader(path, "gzip") as stream:
            with pytest.raises(error):
                list(stream)
            assert not stream.eof_seen and stream.container_complete is False and stream.error
    partial = tmp_path / "partial"; partial.write_bytes(original["prefix"][:-1])
    with reader(partial, "raw") as stream:
        assert len(list(stream.records())) == 15
        assert stream.partial_tail == original["prefix"].splitlines(keepends=True)[-1][:-1]
        assert stream.eof_seen and stream.container_complete is None
    malformed = tmp_path / "malformed"; malformed.write_bytes(b'{"kind":}\n')
    with reader(malformed, "raw") as stream:
        with pytest.raises(json.JSONDecodeError):
            list(stream.records())
        assert stream.error and not stream.eof_seen
    with reader(final_path, "gzip", maximum=1) as stream:
        with pytest.raises(HistoryLimitError):
            list(stream)


def test_empty_gzip_is_incomplete_but_empty_member_is_complete(tmp_path):
    missing = tmp_path / "missing.gz"
    missing.write_bytes(b"")
    with reader(missing, "gzip") as stream:
        with pytest.raises(EOFError, match="no header or member"):
            list(stream)
        assert not stream.eof_seen and stream.container_complete is False
        assert stream.encoded_bytes_read == 0 and stream.error

    valid = tmp_path / "valid.gz"
    with HistoryWriter(valid, encoding="gzip", total_output_bytes=2 * METADATA_RESERVE):
        pass
    with reader(valid, "gzip") as stream:
        assert list(stream) == []
        assert stream.eof_seen and stream.container_complete is True
        assert stream.encoded_bytes_read > 0 and stream.error is None


def test_legacy_callback_budget_identity_zero_null_and_failures(tmp_path):
    record = {"kind": "typed", "zero": 0, "false": False, "null": None, "signed": -0.0}
    captured = []
    size = len((json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n").encode())
    position = emit_record(record, captured.append, logical_bytes=0,
                           total_output_bytes=METADATA_RESERVE + size)
    assert captured[0] is record and position.logical_bytes == position.encoded_bytes == size
    with pytest.raises(HistoryLimitError):
        emit_record(record, captured.append, logical_bytes=0, total_output_bytes=METADATA_RESERVE + size - 1)
    assert len(captured) == 1
    with pytest.raises(ValueError):
        emit_record({"bad": float("nan")}, captured.append, logical_bytes=0, total_output_bytes=2 * METADATA_RESERVE)
    def failed(_):raise OSError("original callback failure")
    with pytest.raises(OSError, match="original callback"):
        emit_record(record, failed, logical_bytes=0, total_output_bytes=2 * METADATA_RESERVE)
    with HistoryWriter(tmp_path / "typed.gz", encoding="gzip", total_output_bytes=2 * METADATA_RESERVE) as writer:
        actual = emit_record(record, writer, logical_bytes=0, total_output_bytes=2 * METADATA_RESERVE)
        assert actual.encoding == "gzip" and actual.logical_bytes == size
        with pytest.raises(HistoryLimitError, match="mismatch"):
            emit_record(record, writer, logical_bytes=0, total_output_bytes=2 * METADATA_RESERVE)
    with reader(writer.path, "gzip") as stream:
        value = list(stream.records())[0]
        assert value == record and value["null"] is None and value["false"] is False
        assert str(value["signed"]) == "-0.0"


def test_isolated_import_boundary():
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "import scripts.benchmarks.native_history; "
        "assert not any(name in sys.modules for name in ('numpy', 'scipy', 'flint', 'sksundae'))"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", code, str(Path(__file__).resolve().parents[2])],
        capture_output=True, text=True, timeout=1,
    )
    assert result.returncode == 0, result.stderr


def test_supplied_bytes_enforce_strict_finite_json(tmp_path):
    for index, literal in enumerate((b"NaN", b"Infinity", b"-Infinity", b"1e999")):
        path = tmp_path / f"invalid{index}.gz"
        writer = HistoryWriter(path, encoding="gzip", total_output_bytes=2 * METADATA_RESERVE)
        before = path.read_bytes()
        with pytest.raises(ValueError, match="non-finite"):
            writer.append_line(b'{"kind":"invalid","nested":[' + literal + b']}\n')
        assert path.read_bytes() == before and writer.position.logical_bytes == 0
        assert writer.closed and writer.container_complete is False and writer.error
        assert not (tmp_path / "LastRecord.json").exists()
