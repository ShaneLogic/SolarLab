"""Byte identity and failure compatibility for bounded canonical hashing."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
import hashlib
import sys

import numpy as np
import pytest

from perovskite_sim.experiments import one_dimensional_mechanism_r1_state as common


@dataclass(frozen=True)
class _Evidence:
    coordinate: object
    label: str
    children: tuple


@pytest.mark.parametrize("value", [
    None,
    True,
    False,
    0,
    2 ** 100,
    "",
    "\"\\/\b\f\n\r\t\x00",
    "中文é𝄞\ud800\udfff",
    [],
    {},
    (1, (2, None), [3, True]),
    {"z": [1, 2], "a": {"second": "x", "first": "y"}},
    {None: "none", True: "boolean", (1, 2): "tuple", 2: "integer"},
    {float("nan"): "allowed string key", float("inf"): "also a string key"},
    np.asarray([[1.0, -0.0], [2.5, 3.125]], dtype=np.float64),
    np.asarray([], dtype=np.float64).reshape(0, 2),
    np.asarray(4.25, dtype=np.float64),
    np.asarray(["中文", "é", "𝄞"]),
    np.asarray([(1, 2.5), (3, 4.5)], dtype=[("integer", "i8"), ("float", "f8")]),
    np.int64(-2 ** 63),
    np.uint64(2 ** 64 - 1),
    np.float32(1.234567),
    np.float64(-0.0),
    np.bool_(True),
    np.str_("中文"),
    _Evidence(
        np.asarray([0.5, -0.0]), "界面",
        (_Evidence(np.float64(1.5), "child", ()), {np.int64(2): np.bool_(False)}),
    ),
])
def test_streaming_digest_matches_existing_digest(value):
    assert common.streaming_digest(value) == common.digest(value)


def test_key_coercion_collision_order_and_unicode_remain_canonical():
    first = {1: "discarded", "1": "last", "𝄞": "音乐", "é": "accent"}
    second = {"é": "accent", "𝄞": "音乐", "1": "last"}
    expected = '{"1":"last","\\u00e9":"accent","\\ud834\\udd1e":"\\u97f3\\u4e50"}'
    assert common.canonical(first) == common.canonical(second) == expected
    expected_digest = hashlib.sha256(expected.encode("ascii")).hexdigest()
    assert common.streaming_digest(first) == common.streaming_digest(second) == expected_digest
    reverse_collision = {"1": "discarded", 1: "numeric key wins last"}
    assert common.canonical(reverse_collision) == '{"1":"numeric key wins last"}'
    assert common.streaming_digest(reverse_collision) == common.digest(reverse_collision)


def test_float_extremes_signed_zero_and_subnormals_preserve_exact_spelling():
    values = [
        0.0, -0.0, float.fromhex("0x0.0000000000001p-1022"),
        -float.fromhex("0x0.0000000000001p-1022"),
        sys.float_info.min, sys.float_info.max, -sys.float_info.max,
        1e-7, 1e-6, 1e15, 1e16, 1.2345678901234567,
    ]
    assert common.canonical(values).startswith("[0.0,-0.0,5e-324,-5e-324,")
    assert common.streaming_digest(values) == common.digest(values)
    assert common.streaming_digest(-0.0) != common.streaming_digest(0.0)


def test_random_finite_float_bit_patterns_match_legacy_encoding():
    # Exercise rounding and exponent spellings without assuming a distribution
    # of magnitudes representative of any particular physical experiment.
    bits = np.random.default_rng(3201).integers(
        0, np.iinfo(np.uint64).max, size=4096, dtype=np.uint64,
    )
    values = bits.view(np.float64)
    values = values[np.isfinite(values)]
    assert values.size > 4000
    assert common.streaming_digest({"values": values}) == common.digest({"values": values})


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("wrap", [
    lambda value: value,
    lambda value: {"nested": [value]},
    lambda value: np.float64(value),
    lambda value: np.asarray([value]),
    lambda value: _Evidence(value, "nonfinite", ()),
])
def test_nonfinite_values_rejected_by_both_paths(nonfinite, wrap):
    value = wrap(nonfinite)
    for calculate in (common.digest, common.streaming_digest):
        with pytest.raises(ValueError, match="Out of range float values are not JSON compliant"):
            calculate(value)


@pytest.mark.parametrize("value", [
    {1, 2},
    b"bytes",
    complex(1, 2),
    Decimal("1.25"),
    datetime(2026, 9, 28),
    object(),
    {"nested": object()},
    np.bytes_(b"numpy bytes"),
    np.asarray([complex(1, 2)]),
    # json_data intentionally does not recursively normalize ndarray.tolist().
    np.asarray([np.float32(1.25)], dtype=object),
])
def test_unsupported_values_keep_existing_type_error(value):
    with pytest.raises(TypeError) as original:
        common.digest(value)
    with pytest.raises(TypeError) as streaming:
        common.streaming_digest(value)
    assert str(streaming.value) == str(original.value)


def test_circular_normalization_remains_rejected():
    value = []
    value.append(value)
    for calculate in (common.digest, common.streaming_digest):
        with pytest.raises(RecursionError):
            calculate(value)


def test_streamed_bytes_equal_canonical_and_updates_are_bounded(monkeypatch):
    value = {
        "numeric_rows": [[float(i), -float(i), i % 2 == 0] for i in range(12000)],
        "large_token": "中文𝄞" * 12000,
        "tail": {"negative_zero": -0.0, "empty": []},
    }
    expected_bytes = common.canonical(value).encode("utf-8")
    real_sha256 = hashlib.sha256
    expected_digest = real_sha256(expected_bytes).hexdigest()
    updates = []

    class RecordingSHA256:
        def __init__(self):
            self.inner = real_sha256()

        def update(self, block):
            updates.append(block)
            self.inner.update(block)

        def hexdigest(self):
            return self.inner.hexdigest()

    monkeypatch.setattr(common.hashlib, "sha256", RecordingSHA256)
    assert common.streaming_digest(value) == expected_digest
    assert b"".join(updates) == expected_bytes
    assert max(map(len, updates)) <= 64 * 1024
    # Batching must reduce update overhead rather than hash every JSON token.
    assert 1 < len(updates) < 30


def test_streaming_path_does_not_call_whole_document_encoder(monkeypatch):
    value = {"evidence": [1.0, -0.0, np.float32(2.5)], "label": "中文"}
    expected = common.digest(value)

    def forbidden(*args, **kwargs):
        raise AssertionError("streaming hash materialized canonical JSON")

    monkeypatch.setattr(common, "canonical", forbidden)
    monkeypatch.setattr(common.json, "dumps", forbidden)
    assert common.streaming_digest(value) == expected
