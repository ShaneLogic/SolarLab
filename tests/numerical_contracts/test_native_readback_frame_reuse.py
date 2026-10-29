"""Exact, local frame decoding reuse; no integration or qualification.

The reference bodies below are from the pinned pre-change reader. An explicit
baseline path checks their provenance; a saved workload path adds original
producer samples without running the observer or changing source admission.
"""
import ast
from copy import deepcopy
from fractions import Fraction
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import textwrap
from types import MethodType
from unittest.mock import patch
import weakref

import pytest

from scripts.benchmarks import native_readback as reader
from scripts.benchmarks.native_readback import (
    _digest, _frame_word_values, _hex, _hex_vector, _require,
)

BASELINE_SHA = "1711a56c5ed5180964d6203d82a8909e68b4a5ff9b79f5455e6c170e9a9698c0"


def old_parent(self, frame, when, raw, *, rate=False):
    q0 = _frame_word_values(frame["q0_words_hex"], self.size)
    v0 = _frame_word_values(frame["v0_words_hex"], self.size)
    if rate:
        return tuple(v+Fraction(u) for v, u in zip(v0, raw))
    dt = Fraction(when)-Fraction(_hex(frame["t0_hex"]))
    return tuple(q+dt*v+Fraction(u) for q, v, u in zip(q0, v0, raw))


def old_sample(self, history):
    _require(history["segment_frame_identity"] == _digest(self.active), "segment_frame_sample_identity")
    if self.input_words == 12:
        _require(history.get("mapped_input_profile") == "frame-input-expansion12-v1"
                 and history.get("mapped_input_words") == 12, "frame_input_history_profile")
    else:
        _require("mapped_input_profile" not in history and "mapped_input_words" not in history,
                 "frame_input_unbound_history_profile")
    t = _hex(history["time_hex"])
    q = self.parent(self.active, t, _hex_vector(history["raw_solver_z_hex"], self.size))
    v = self.parent(self.active, t, _hex_vector(history["raw_solver_zdot_hex"], self.size), rate=True)
    _require(_frame_word_values(history["physical_cumulative_words_hex"], self.size, count=self.input_words) ==
             self.physical(q, _hex_vector(history["inputs_hex"], 2))
             and _frame_word_values(history["physical_rate_words_hex"], self.size, count=self.input_words) ==
             self.physical(v, _hex_vector(history["input_rates_hex"], 2), rate=True), "segment_frame_sample_physical_words")


class ReferenceCheck(reader.SegmentFrameCheck):
    parent = old_parent
    sample = old_sample


def reference_bodies(path):
    raw = Path(path).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == BASELINE_SHA
    before = ast.parse(raw)
    cls = next(n for n in before.body if isinstance(n, ast.ClassDef) and n.name == "SegmentFrameCheck")
    for copied in (old_parent, old_sample):
        original = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == copied.__name__[4:])
        reference = ast.parse(textwrap.dedent(inspect.getsource(copied))).body[0]
        assert ast.dump(ast.Module(body=original.body, type_ignores=[])) == ast.dump(ast.Module(body=reference.body, type_ignores=[]))
    # Everything outside sample (including parent, decoding, source guards,
    # endpoints, EOF and CRC handling) must be the exact pre-change syntax.
    after = ast.parse(Path(reader.__file__).read_bytes())
    current = next(n for n in after.body if isinstance(n, ast.ClassDef) and n.name == cls.name)
    original_sample = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "sample")
    current.body = [original_sample if isinstance(n, ast.FunctionDef) and n.name == "sample" else n for n in current.body]
    assert ast.dump(before) == ast.dump(after)


def words(values, count=4):
    remaining = list(map(Fraction, values))
    result = []
    for _ in range(count):
        layer = [float(value) for value in remaining]
        result.append([value.hex() for value in layer])
        remaining = [value-Fraction(word) for value, word in zip(remaining, layer)]
    assert not any(remaining), "fixture requires more words"
    return result


def fixture(count=4, size=6, when=0.5):
    q0 = tuple(Fraction((-1)**i*(i+1), 8)+Fraction(1, 2**60) for i in range(size))
    v0 = tuple(Fraction(i-2, 16)-Fraction(1, 2**70) for i in range(size))
    raw = [float(i-2)/32 for i in range(size)]
    rate = [float(3-i)/64 for i in range(size)]
    raw[-1] = -0.0
    frame = {"t0_hex": 0.25.hex(), "q0_words_hex": words(q0), "v0_words_hex": words(v0)}
    base = {"columns_hex": [float(i+1).hex() for i in range(size)],
            "lift_hex": [[0.125.hex(), (-0.25).hex()] for _ in range(size)],
            "reference_inputs_hex": [0.25.hex(), 2.0.hex()]}
    inputs, slopes = [0.625, 1.5], [0.0625, -0.125]
    q = tuple(a+(Fraction(when)-Fraction(0.25))*b+Fraction(u) for a, b, u in zip(q0, v0, raw))
    v = tuple(a+Fraction(u) for a, u in zip(v0, rate))
    physical_q = tuple((i+1)*a+Fraction(1, 8)*(Fraction(inputs[0])-Fraction(1, 4))
                       -Fraction(1, 4)*(Fraction(inputs[1])-2) for i, a in enumerate(q))
    physical_v = tuple((i+1)*a+Fraction(1, 8)*Fraction(slopes[0])
                       -Fraction(1, 4)*Fraction(slopes[1]) for i, a in enumerate(v))
    history = {"segment_frame_identity": _digest(frame), "time_hex": when.hex(),
               "raw_solver_z_hex": [u.hex() for u in raw], "raw_solver_zdot_hex": [u.hex() for u in rate],
               "physical_cumulative_words_hex": words(physical_q, count),
               "physical_rate_words_hex": words(physical_v, count),
               "inputs_hex": [u.hex() for u in inputs], "input_rates_hex": [u.hex() for u in slopes]}
    if count == 12:
        history.update(mapped_input_profile="frame-input-expansion12-v1", mapped_input_words=12)
    return {"size": size, "input_words": count, "active": frame, "base": base, "history": history}


def make(cls, data):
    check = object.__new__(cls)
    check.__dict__.update(deepcopy({k: v for k, v in data.items() if k != "history"}))
    return check


def outcome(call):
    try:
        return {"value": call(), "error": None}
    except Exception as error:
        return {"error": {"type": type(error).__name__, "message": str(error),
                          "code": getattr(error, "code", None),
                          "record_index": getattr(error, "record_index", None),
                          "logical_offset": getattr(error, "logical_offset", None)}}


def observe(cls, data):
    check, history = make(cls, data), deepcopy(data["history"])
    captures = []
    physical = check.physical
    def capture(parent, inputs, *, rate=False):
        result = physical(parent, inputs, rate=rate)
        captures.append({"rate": rate, "parent": parent, "physical": result,
                         "physical_words": words(result, check.input_words)})
        return result
    check.physical = capture
    state_before, history_before = deepcopy(check.__dict__), deepcopy(history)
    module = sys.modules[__name__] if cls is ReferenceCheck else reader
    with patch.object(module, "_frame_word_values", wraps=module._frame_word_values) as decode:
        result = outcome(lambda: check.sample(history))
    assert check.__dict__ == state_before
    assert history == history_before
    return {**result, "captures": captures}, decode.call_count


def test_pinned_reference_and_only_sample_changed():
    path = os.environ.get("SOLARLAB_FRAME_BASELINE_SOURCE")
    if not path:
        pytest.skip("explicit pinned baseline source required for provenance check")
    reference_bodies(path)


@pytest.mark.parametrize("count", [4, 12])
@pytest.mark.parametrize("size", [1, 6])
@pytest.mark.parametrize("when", [-0.0, 0.25, 0.5, 1.0, math.nextafter(1.0, 0.0), math.nextafter(1.0, 2.0)])
def test_exact_state_rate_and_words_with_six_to_four_decodes(count, size, when):
    data = fixture(count, size, when)
    before, old_count = observe(ReferenceCheck, data)
    after, new_count = observe(reader.SegmentFrameCheck, data)
    assert before == after and after["error"] is None
    assert len(after["captures"]) == 2
    assert (old_count, new_count) == (6, 4)


def damage(data, case):
    h, f = data["history"], data["active"]
    if case == "identity":
        h["segment_frame_identity"], h["time_hex"] = "changed", 0
        return "segment_frame_sample_identity"
    if case == "profile":
        h["mapped_input_words"], h["time_hex"] = 5, 0
        return "frame_input_history_profile" if data["input_words"] == 12 else "frame_input_unbound_history_profile"
    if case == "time":
        h["time_hex"], h["raw_solver_z_hex"] = "0x1p+0", []
        return "noncanonical_or_nonfinite_hex"
    if case == "raw_state":
        h["raw_solver_z_hex"][0], f["q0_words_hex"] = 0, []
        code = "hex_word_type"
    elif case == "q_shape":
        f["q0_words_hex"], h["raw_solver_zdot_hex"] = [], []
        code = "vector_shape"
    elif case == "q_canonical":
        f["q0_words_hex"][1][0], h["raw_solver_zdot_hex"] = 1.0.hex(), []
        code = "segment_frame_noncanonical_words"
    elif case == "v_shape":
        f["v0_words_hex"], f["t0_hex"], h["raw_solver_zdot_hex"] = [], 0, []
        code = "vector_shape"
    elif case == "v_canonical":
        f["v0_words_hex"][1][0], h["raw_solver_zdot_hex"] = 1.0.hex(), []
        code = "segment_frame_noncanonical_words"
    elif case == "anchor_time":
        f["t0_hex"], h["raw_solver_zdot_hex"] = 0, []
        code = "hex_word_type"
    elif case == "raw_rate":
        h["raw_solver_zdot_hex"][0], h["physical_cumulative_words_hex"] = 0, []
        code = "hex_word_type"
    elif case == "physical_state_shape":
        h["physical_cumulative_words_hex"], h["inputs_hex"] = [], [0, 0]
        code = "vector_shape"
    elif case == "state_inputs":
        h["inputs_hex"], h["physical_rate_words_hex"] = [0, 0], []
        code = "hex_word_type"
    elif case == "state_short_circuit":
        values = _frame_word_values(h["physical_cumulative_words_hex"], data["size"], count=data["input_words"])
        h["physical_cumulative_words_hex"] = words([value+1 for value in values], data["input_words"])
        h["physical_rate_words_hex"] = []
        code = "segment_frame_sample_physical_words"
    elif case == "physical_rate_shape":
        h["physical_rate_words_hex"], h["input_rates_hex"] = [], [0, 0]
        code = "vector_shape"
    elif case == "rate_inputs":
        h["input_rates_hex"] = [0, 0]
        code = "hex_word_type"
    elif case == "changed_frame":
        f["q0_words_hex"][0][0] = 1.0.hex()
        code = "segment_frame_sample_physical_words"
    else:
        raise AssertionError(case)
    h["segment_frame_identity"] = _digest(f)
    return code


@pytest.mark.parametrize("count", [4, 12])
@pytest.mark.parametrize("case", ["identity", "profile", "time", "raw_state", "q_shape", "q_canonical",
                                  "v_shape", "v_canonical", "anchor_time", "raw_rate", "physical_state_shape",
                                  "state_inputs", "state_short_circuit", "physical_rate_shape", "rate_inputs", "changed_frame"])
def test_identical_rejections_and_first_error_order(count, case):
    data = fixture(count)
    code = damage(data, case)
    before, _ = observe(ReferenceCheck, data)
    after, _ = observe(reader.SegmentFrameCheck, data)
    assert before == after
    assert after["error"]["code"] == code


@pytest.mark.parametrize("rate", [False, True])
@pytest.mark.parametrize("case", ["valid", "short_raw", "missing_q", "bad_q", "bad_v", "bad_t0", "bad_when", "bad_raw"])
def test_standalone_parent_keeps_validation_including_unused_q(rate, case):
    data = fixture()
    frame, when, raw = data["active"], 0.5, (0.25,)*data["size"]
    if case == "short_raw":
        raw = raw[:1]
    elif case == "missing_q":
        del frame["q0_words_hex"]
    elif case == "bad_q":
        frame["q0_words_hex"] = []
    elif case == "bad_v":
        frame["v0_words_hex"] = []
    elif case == "bad_t0":
        frame["t0_hex"] = "bad"
    elif case == "bad_when":
        when = object()
    elif case == "bad_raw":
        raw = (object(),)
    original = deepcopy(frame)
    results = [outcome(lambda cls=cls: make(cls, data).parent(frame, when, raw, rate=rate))
               for cls in (ReferenceCheck, reader.SegmentFrameCheck)]
    assert results[0] == results[1] and frame == original
    if case in ("missing_q", "bad_q", "bad_v", "bad_raw"):
        assert results[0]["error"] is not None


@pytest.mark.parametrize("subclass", [False, True])
@pytest.mark.parametrize("replace_after_state", [False, True])
def test_parent_override_keeps_both_public_calls(subclass, replace_after_state):
    data = fixture()
    results, calls = [], []
    for base in (ReferenceCheck, reader.SegmentFrameCheck):
        order = []
        def replacement(self, frame, when, raw, *, rate=False):
            order.append(("replacement", rate))
            raise ValueError("replacement parent refusal")
        def override(self, frame, when, raw, *, rate=False):
            order.append(("override", rate))
            if replace_after_state:
                self.parent = MethodType(replacement, self)
            return base.parent(self, frame, when, raw, rate=rate)
        cls = type("CustomCheck", (base,), {"parent": override}) if subclass else base
        check = make(cls, data)
        if not subclass:
            check.parent = MethodType(override, check)
        results.append(outcome(lambda: check.sample(data["history"])))
        calls.append(order)
    assert results[0] == results[1] and calls[0] == calls[1]
    assert calls[0] == [("override", False), ("replacement" if replace_after_state else "override", True)]


def test_decoded_values_have_no_retained_lifetime_or_cross_call_cache():
    class TrackedFraction(Fraction):
        __slots__ = ("__weakref__",)
    refs = []
    original_decode = reader._frame_word_values
    def tracked(*args, **kwargs):
        values = tuple(TrackedFraction(value) for value in original_decode(*args, **kwargs))
        refs.extend(weakref.ref(value) for value in values)
        return values
    data = fixture()
    check = make(reader.SegmentFrameCheck, data)
    before = deepcopy(check.__dict__)
    with patch.object(reader, "_frame_word_values", side_effect=tracked) as decode:
        for _ in range(3):
            check.sample(data["history"])
            gc.collect()
            assert all(ref() is None for ref in refs)
        assert decode.call_count == 12 and check.__dict__ == before
        check.active["q0_words_hex"][0][0] = 1.0.hex()
        changed = deepcopy(data["history"])
        changed["segment_frame_identity"] = _digest(check.active)
        with pytest.raises(reader.HistoryVerificationError, match="segment_frame_sample_physical_words"):
            check.sample(changed)
        check.active["v0_words_hex"] = []
        changed["segment_frame_identity"] = _digest(check.active)
        with pytest.raises(reader.HistoryVerificationError, match="vector_shape"):
            check.sample(changed)


def test_saved_producer_workload_exact_words_and_decode_count():
    path = os.environ.get("SOLARLAB_FRAME_WORKLOAD")
    if not path:
        pytest.skip("explicit bounded saved producer workload required")
    payload = json.loads(Path(path).read_bytes())
    assert payload["schema"] == "solarlab.frame-decode-fixed-workload.v1"
    counts = [0, 0]
    for row in payload["samples"]:
        assert _digest(row["history"]) == row["history_sha256"]
        data = {"size": payload["size"], "input_words": payload["input_words"], "base": payload["base"],
                "active": payload["frames"][row["frame_identity"]], "history": row["history"]}
        before, old_count = observe(ReferenceCheck, data)
        after, new_count = observe(reader.SegmentFrameCheck, data)
        assert before == after and after["error"] is None
        counts[0] += old_count
        counts[1] += new_count
    assert len(payload["samples"]) > 1
    assert counts == [6*len(payload["samples"]), 4*len(payload["samples"])]
