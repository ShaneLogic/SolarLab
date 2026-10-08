"""Strict, bounded readback of a completed voltage-lift native history.

This is artifact verification, not a replay or a new scientific qualification.
Only the current record, two endpoints and one interval's sample pointers are
retained. The caller owns source admission, process cleanup, archive publication
and the whole-process wall/RSS/output limits; the deadline here is the same
absolute ``time.perf_counter()`` deadline, not an additional timeout allowance.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import struct
import time
import zlib

from scripts.benchmarks.native_history import (
    HistoryReader, configure_integer_io, decode_integer_values, integer_io_observed,
)


class HistoryVerificationError(ValueError):
    """A bounded diagnostic, with the offending record and logical offset."""

    def __init__(self, code: str, record_index: int = 0, logical_offset: int = 0):
        self.code = code[:160]
        self.record_index = record_index
        self.logical_offset = logical_offset
        super().__init__(f"{self.code} (record={record_index}, offset={logical_offset})")


def _require(condition, code):
    if not condition:
        raise HistoryVerificationError(code)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _hex(value):
    _require(type(value) is str, "hex_word_type")
    number = float.fromhex(value)
    _require(math.isfinite(number) and number.hex() == value, "noncanonical_or_nonfinite_hex")
    return number


def _vector(value, size):
    _require(type(value) in (list, tuple) and len(value) == size, "vector_shape")
    return value


def _hex_vector(value, size):
    return tuple(_hex(item) for item in _vector(value, size))


def _word_bytes(value, size):
    return struct.pack(f"<{size}d", *_hex_vector(value, size))


def _buffer(value, size):
    _require(type(value) is bytes and len(value) == 8 * size, "native_buffer_shape")
    values = struct.unpack(f"<{size}d", value)
    _require(all(math.isfinite(v) for v in values), "native_buffer_nonfinite")
    return values


def _pairs(pairs):
    obj = {}
    for key, value in pairs:
        _require(key not in obj, "duplicate_json_key")
        obj[key] = value
    return obj


def _constant(_):
    raise HistoryVerificationError("nonfinite_json_constant")


def _finite(value):
    value = float(value)
    _require(math.isfinite(value), "nonfinite_json_number")
    return value


def _decode(value, depth=0):
    """Validate reserved tags without importing the numerical packet reader."""
    _require(depth <= 64, "json_nesting_limit")
    if type(value) is list:
        return [_decode(v, depth + 1) for v in value]
    if type(value) is not dict:
        return value
    tags = set(value) & {"tuple", "rational", "bytes_hex", "binary64"}
    if tags:
        _require(len(tags) == len(value) == 1, "ambiguous_observation_tag")
        key = next(iter(tags))
        data = value[key]
        if key == "tuple":
            _require(type(data) is list, "tuple_tag_type")
            return tuple(_decode(v, depth + 1) for v in data)
        if key == "binary64":
            return _hex(data)
        if key == "bytes_hex":
            _require(type(data) is str, "bytes_tag_type")
            binary = bytes.fromhex(data)
            _require(binary.hex() == data, "noncanonical_bytes_tag")
            return binary
        _require(type(data) is list and len(data) == 2
                 and all(type(v) is int for v in data) and data[1] > 0,
                 "rational_tag_shape")
        result = Fraction(*data)
        _require([result.numerator, result.denominator] == data, "noncanonical_rational_tag")
        return result
    return {key: _decode(item, depth + 1) for key, item in value.items()}


def _json(data, integer_io_policy=None):
    integer_io_observed(integer_io_policy)
    result = json.loads(data.decode("utf-8"), object_pairs_hook=_pairs,
                        parse_constant=_constant, parse_float=_finite)
    result = decode_integer_values(result, integer_io_policy)
    _require(type(result) is dict, "json_object_required")
    _decode(result)  # Validate every tag, including fields outside the projections below.
    return result


def _read_json(path, limit, checkpoint, integer_io_policy=None):
    checkpoint()
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    checkpoint()
    _require(len(data) <= limit, "metadata_byte_limit")
    return _json(data, integer_io_policy)


def _nonnegative(values, size):
    values = _vector(values, size)
    _require(all(type(v) is Fraction and v >= 0 for v in values), "nonnegative_rational_vector")
    return tuple(values)


def _enclosures(values, size):
    converted = []
    for value in _vector(values, size):
        _require(type(value) is dict and set(value) == {"center", "absolute_error_bound"}, "enclosure_structure")
        item = {}
        # Enclosure.payload uses untagged numerator/denominator lists. These
        # are distinct from observation_record's tagged standalone Fractions.
        for key, pair in value.items():
            _require(type(pair) is list and len(pair) == 2 and all(type(v) is int for v in pair)
                     and pair[1] > 0, "enclosure_rational_pair")
            rational = Fraction(*pair)
            _require([rational.numerator, rational.denominator] == pair, "enclosure_noncanonical_pair")
            item[key] = rational
        _require(item["absolute_error_bound"] >= 0, "enclosure_negative_radius")
        converted.append(item)
    return tuple(converted)


def _word_pair(value, size):
    _require(type(value) is dict and set(value) == {"high_hex", "low_hex"}, "double_word_shape")
    high = _hex_vector(value["high_hex"], size)
    low = _hex_vector(value["low_hex"], size)
    return tuple(Fraction(a) + Fraction(b) for a, b in zip(high, low))


def _snapshot_digest(value):
    def tagged(item):
        if type(item) is dict:
            return ["mapping", [[key, tagged(item[key])] for key in sorted(item)]]
        if type(item) is tuple:
            return ["tuple", [tagged(part) for part in item]]
        if type(item) is bytes:
            return ["bytes", item.hex()]
        if type(item) is float:
            return ["binary64", item.hex()]
        _require(type(item) in (str, int, bool, type(None)), "native_packet_type")
        return [type(item).__name__, item]
    return _digest(tagged(value))


_STAMP_NAMES = ("IDAGetNumSteps", "IDAGetCurrentTime", "IDAGetLastStep",
                "IDAGetLastOrder", "IDAGetCurrentStep", "IDAGetCurrentOrder")
_COUNTERS = ("num_steps", "residual_evals", "linear_setups", "error_test_fails",
             "nonlinear_iters", "nonlinear_conv_fails", "jacobian_evals")
_LEDGERS = ("raw_polynomial", "same_state_affine_tangent")


def _statuses(value, names):
    _require(type(value) is tuple and len(value) == len(names)
             and all(type(row) is tuple and len(row) == 2 and row[0] == name
                     and type(row[1]) is int and row[1] == 0
                     for row, name in zip(value, names)), "native_getter_status")


def _stamp(stamp):
    _require(all(type(stamp[k]) is int for k in ("nsteps", "kused", "kk"))
             and all(type(stamp[k]) is float and math.isfinite(stamp[k])
                     for k in ("tn", "hused", "hh")), "native_stamp_type")
    _statuses(stamp["statuses"], _STAMP_NAMES)
    return tuple(stamp[k] for k in ("nsteps", "tn", "hused", "kused", "hh", "kk"))


def _native_packet(packet, policy, size):
    """Structural/word checks from the maintained native packet contract.

    No polynomials or models are constructed. The frame identity is recomputed
    from all packet fields, including the native provenance and low-level bytes.
    """
    p = _decode(packet)
    _require(p["schema"] == "sksundae.ida.accepted-step-observation.v1"
             and p["dtype"] == "<f8" and p["shape"] == (size,), "native_packet_schema")
    binding, header = policy["binding_identity"], policy["header_sha256"]
    source = p["binding"]["source"]
    _require(type(source) is tuple and len(source) == 8
             and hashlib.sha256(repr(source).encode()).hexdigest() == binding
             and p["binding"]["identity"] == binding and source[0] == p["schema"]
             and source[1] == "scikit-sundae=1.1.3" and source[5] == header,
             "native_binding_identity")
    before = p["native_before"]
    _require(_stamp(before) == _stamp(p["native_after"]), "native_stamp_changed")
    owner, generation, q = p["owner"], p["generation"], before["kused"]
    _require(type(owner) is str and bool(owner) and type(generation) is int and generation >= 1
             and 1 <= q <= 5 and before["nsteps"] >= 1 and before["hused"] > 0,
             "native_owner_or_order")
    key = (binding, owner, generation, before["nsteps"], before["tn"].hex(),
           before["hused"].hex(), q, before["hh"].hex(), before["kk"])
    _require(p["step_key"] == key, "native_step_key")
    previous, endpoint = p["predecessor"], p["endpoint"]
    for point, steps in ((previous, before["nsteps"] - 1), (endpoint, before["nsteps"])):
        _require(point["owner"] == owner and point["generation"] == generation
                 and point["nsteps"] == steps and type(point["internal_t"]) is float
                 and type(point["output_t"]) is float
                 and point["internal_t"].hex() == point["output_t"].hex()
                 and type(point["native_status"]) is int and point["native_status"] in (0, 1),
                 "native_endpoint_context")
        _buffer(point["raw_y"], size)
        _buffer(point["raw_yp"], size)
        expected = (owner, generation, steps, point["internal_t"].hex(), point["output_t"].hex(),
                    hashlib.sha256(point["raw_y"] + point["raw_yp"]).hexdigest())
        _require(point["identity"] == expected, "native_endpoint_identity")
    _require(previous["internal_t"] < endpoint["internal_t"] == before["tn"]
             and p["raw_y"] == endpoint["raw_y"] and p["raw_yp"] == endpoint["raw_yp"]
             and p["native_left_terms"] == (before["tn"], -before["hused"])
             and p["clock_gap_terms"] == (before["tn"], -before["hused"], -previous["internal_t"]),
             "native_endpoint_or_clock")
    basis = p["basis"]
    _require(basis["kind"] == "ida75_phi_psi_copy_v1" and basis["dtype"] == "<f8"
             and basis["phi_shape"] == (q + 1, size) and basis["psi_shape"] == (q,)
             and type(basis["status"]) is int and basis["status"] == 0
             and basis["source_header_sha256"] == header and basis["source_config_sha256"] == source[6]
             and basis["build_identity"] == binding and type(basis["uround"]) is float
             and 0 < basis["uround"] < 1, "native_basis_binding")
    _buffer(basis["phi"], (q + 1) * size)
    _require(all(v != 0 for v in _buffer(basis["psi"], q)), "native_zero_psi")
    pred = p["predecessor_dky"]
    statuses = _vector(pred["statuses"], 2)
    _require(pred["query_t"] == previous["internal_t"]
             and pred["query_t_hex"] == previous["internal_t"].hex() and pred["orders"] == (0, 1)
             and all(type(v) is int for v in statuses) and type(pred["native_eligible"]) is bool
             and pred["native_eligible"] == all(v == 0 for v in statuses)
             and _stamp(pred["step_before"]) == _stamp(before)
             and _stamp(pred["step_after"]) == _stamp(before), "native_predecessor_getters")
    for code, buffer in zip(statuses, _vector(pred["buffers"], 2)):
        if code == 0:
            _buffer(buffer, size)
        else:
            _require(buffer is None, "failed_getter_buffer")
    for buffer in _vector(p["dky"], q + 1):
        _buffer(buffer, size)
    _buffer(p["estimated_local_errors"], size)
    _require(all(v > 0 for v in _buffer(p["error_weights"], size)), "native_error_weights")
    _statuses(p["vector_statuses"], tuple(f"IDAGetDky[{k}]" for k in range(q + 1))
              + ("IDAGetErrWeights", "IDAGetEstLocalErrors"))
    return p, _snapshot_digest(p)


class _Rows:
    def __init__(self, reader, max_records, checkpoint):
        self.reader, self.max_records, self.checkpoint = reader, max_records, checkpoint
        self.count = self.offset = self.crc = 0
        self.hash = hashlib.sha256()
        self.counts = Counter()
        self.last = None

    def take(self, kind=None, *, eof=False):
        self.checkpoint()
        try:
            line = next(self.reader)
        except StopIteration:
            _require(eof, "history_premature_eof")
            return None
        self.offset = self.reader.logical_bytes - len(line)
        self.count += 1
        _require(self.count <= self.max_records, "record_count_limit")
        _require(line.endswith(b"\n") and line.count(b"\n") == 1, "partial_or_multiline_record")
        row = _json(line, getattr(self.reader, "integer_io_policy", None))
        self.hash.update(line)
        self.crc = zlib.crc32(line, self.crc)
        self.checkpoint()
        _require(type(row.get("kind")) is str, "record_kind_required")
        self.counts[row["kind"]] += 1
        _require(not eof, "surplus_history_record")
        _require(kind is None or row["kind"] == kind, "record_order_expected_" + str(kind))
        self.last = {"history_byte_offset_after_record": self.reader.logical_bytes,
                     "kind": row["kind"], "record_sha256": row.get("record_sha256"),
                     "history_encoding": "gzip"}
        return row


def segment_startup_controls(request, segment):
    """Independently bind the sole permitted override, without producer imports."""
    controls = dict(request["controls"])
    _require("segment_startup_overrides" not in request, "segment_startup_unbound_override")
    if "segment_startup_policy" not in request:
        return controls
    policy = request["segment_startup_policy"]
    expected = {
        "schema": "solarlab.voltage-lift-segment-startup.v1",
        "ancestor_request_sha256": request["prior_request_sha256"],
        "base_controls_sha256": _digest(controls),
        "segments_sha256": _digest(request["segments"]),
        "time_weight_policy_sha256": _digest(request.get("time_weight_policy")),
        "overrides": {"slow_state_hold": {"first_step": 0.0}},
        "application": "fresh_segment_initialization_before_first_solve",
    }
    _require(type(policy) is dict and _digest(policy) == _digest(expected)
             and request["case_id"] == "DynamicAcceptorIonPublicDeviceV1"
             and [s["id"] for s in request["segments"]] == [
                 "dark_equilibrium_hold", "voltage_ramp", "slow_state_hold"]
             and segment in request["segments"]
             and controls.get("first_step") == 7.8125e-7
             and len(controls.get("atol", [])) == 45
             and controls.get("nonlin_conv_coef") == 1.024e-5
             and controls.get("nonlin_guard") == "first-correction-wrms-v1"
             and controls.get("nonlin_trace_capacity") == 4096
             and request.get("time_weight_policy", {}).get("kappa") == 1024,
             "segment_startup_policy_binding")
    if segment["id"] == "slow_state_hold":
        controls["first_step"] = 0.0
    return controls


class SegmentStartupCheck:
    """Bounded constructor/native-stat association, also usable on failed runs.

    Constructor words are source evidence, not a native requested-step getter.
    h0u independently checks explicit steps and records automatic selection;
    it cannot prove which automatic-start algorithm was executed.
    """
    def __init__(self, request):
        self.request = request
        self.enabled = "segment_startup_policy" in request
        self.expected = [segment_startup_controls(request, s) for s in request["segments"]]
        self.contexts = []

    def consume(self, row):
        kind = row.get("kind")
        if kind == "voltage_lift_initialization_controls":
            ordinal = len(self.contexts) + 1
            _require(self.enabled and ordinal <= len(self.expected), "segment_startup_unexpected_constructor")
            segment = self.request["segments"][ordinal - 1]
            controls = self.expected[ordinal - 1]
            _require(row["request_sha256"] == _digest(self.request)
                     and row["map_identity"] == self.request["map_identity"]
                     and row["segment_id"] == segment["id"]
                     and row["segment_sha256"] == _digest(segment)
                     and type(row["logical_initialization_index"]) is int
                     and row["logical_initialization_index"] == ordinal
                     and row["segment_startup_policy_sha256"] == _digest(self.request["segment_startup_policy"])
                     and _digest(row["constructor_controls"]) == row["constructor_controls_sha256"] == _digest(controls)
                     and row["requested_first_step_hex"] == float(controls["first_step"]).hex(),
                     "segment_startup_constructor_binding")
            self.contexts.append({"ordinal": ordinal, "segment_id": segment["id"],
                "constructor_record_sha256": _digest(row), "controls_sha256": _digest(controls),
                "requested_first_step_hex": row["requested_first_step_hex"],
                "native_owner": None, "actual_initial_step_hex": None, "accepted_steps": 0,
                "first_return_record_sha256": None})
        elif self.enabled and kind == "native_statistics":
            ordinal = row["logical_initialization_index"]
            _require(type(ordinal) is int and ordinal == len(self.contexts) and ordinal > 0,
                     "segment_startup_missing_constructor")
            context, raw = self.contexts[-1], row["raw_statistics"]
            _require(row["segment_id"] == context["segment_id"], "segment_startup_statistics_segment")
            owner = [raw["observation_owner"], raw["observation_generation"]]
            if row["phase"] == "initialization_return":
                _require(context["native_owner"] is None and raw["num_steps"] == 0,
                         "segment_startup_initialization_sequence")
                context["native_owner"] = owner
            else:
                _require(context["native_owner"] == owner, "segment_startup_native_owner")
            if row["phase"] in ("after_onestep", "onestep_exception"):
                h0, count = raw["initial_step"], raw["num_steps"]
                _require(type(count) is int and count >= context["accepted_steps"],
                         "segment_startup_step_count")
                if count > 0:
                    _require(type(h0) is float and math.isfinite(h0) and h0 > 0,
                             "segment_startup_actual_initial_step")
                    requested = self.expected[ordinal - 1]["first_step"]
                    _require(requested == 0 or h0.hex() == float(requested).hex(),
                             "segment_startup_explicit_step_not_applied")
                    _require(context["actual_initial_step_hex"] in (None, h0.hex()),
                             "segment_startup_actual_step_changed")
                    context["actual_initial_step_hex"] = h0.hex()
                if context["first_return_record_sha256"] is None:
                    context["first_return_record_sha256"] = _digest(row)
                    context["first_return_steps"] = count
                    context["raw_first_return_initial_step"] = h0
                context["accepted_steps"] = count

    def finish(self, *, complete):
        if self.enabled and complete:
            _require(len(self.contexts) == len(self.expected)
                     and all(c["native_owner"] is not None and c["accepted_steps"] > 0
                             and c["actual_initial_step_hex"] is not None for c in self.contexts),
                     "segment_startup_incomplete_application")
        return {"enabled": self.enabled, "initializations": self.contexts,
                "native_requested_first_step_getter": "unavailable",
                "scope": "source-bound constructor kwargs and actual IDAGetIntegratorStats h0u; no accuracy qualification"}


class _Protocol:
    def __init__(self, request, rows):
        self.request, self.rows = request, rows
        self.request_id = _digest(request)
        self.mapping, self.policy = request["voltage_lift_map"], request["interval_observation"]
        self.map_id, self.policy_id = request["map_identity"], _digest(self.policy)
        self.size, self.nodes = len(request["z0"]), request["numeric_packet"]["nodes"]
        _require(request["schema"] == "solarlab.voltage-lift-native-request.v1"
                 and 1 <= self.size <= 45 and type(self.nodes) is int and 2 <= self.nodes <= self.size
                 and self.map_id == _digest(self.mapping), "request_schema_or_map")
        _vector(request["zdot0"], self.size)
        for key, value in (("map_identity", self.map_id), ("protocol_sha256", _digest(request["segments"])),
                           ("controls_sha256", _digest(request["controls"])),
                           ("budgets_sha256", _digest(request["budgets"])),
                           ("sampling_sha256", _digest((request["observation_times"], request["quadrature"])))):
            _require(self.policy[key] == value, "request_policy_binding")
        _require(set(request["quadrature"]) == {"8", "16", "32"}, "quadrature_orders")
        for order in (8, 16, 32):
            table = request["quadrature"][str(order)]
            _require(all(type(v) in (float, int) and -1 < v < 1 for v in _vector(table["nodes"], order))
                     and all(type(v) in (float, int) and v > 0 for v in _vector(table["weights"], order)),
                     "quadrature_shape_or_domain")
        self.jacobians = self.steps = self.intervals = self.samples = self.queries = 0
        self.prefix = {k: ((Fraction(0),) * 3, (Fraction(0),) * 3) for k in _LEDGERS}
        self.coverage = []
        self.native_owner = None
        self.startup = SegmentStartupCheck(request)

    def _context(self, row):
        _require(row["request_sha256"] == self.request_id and row["map_identity"] == self.map_id,
                 "record_request_or_map")

    def _statistics(self, phase, returned=None, previous=None):
        row = self.rows.take()
        while row["kind"] == "jacobian_callback_context":
            _require(phase != "before_onestep", "callback_outside_native_call")
            self._context(row)
            self.jacobians += 1
            _require(row["segment_id"] == self.segment["id"]
                     and row["logical_initialization_index"] == self.ordinal
                     and row["callback_index"] == self.jacobians, "callback_context")
            _hex(row["time_hex"])
            _hex(row["cj_hex"])
            _hex_vector(row["z_hex"], self.size)
            _hex_vector(row["zdot_hex"], self.size)
            row = self.rows.take()
        _require(row["kind"] == "native_statistics" and row["phase"] == phase
                 and row["segment_id"] == self.segment["id"]
                 and row["logical_initialization_index"] == self.ordinal, "statistics_order_or_context")
        self.startup.consume(row)
        target = self.segment["start"] if phase == "initialization_return" else self.segment["end"]
        _require(_hex(row["requested_time_hex"]) == target, "statistics_requested_time")
        raw = row["raw_statistics"]
        _require(all(type(raw[k]) is int and raw[k] >= 0 for k in _COUNTERS)
                 and _hex(row["internal_time_hex"]) == raw["current_time"]
                 and row["method_state_valid"] is (raw["num_steps"] > 0), "statistics_values")
        if returned is not None:
            _require(_hex(row["returned_time_hex"]) == returned, "statistics_return_time")
        if phase == "before_onestep":
            _require(row["returned_time_hex"] is None, "statistics_before_return")
        if phase == "after_onestep":
            _require(_hex(row["returned_time_hex"]) == raw["current_time"], "statistics_after_return")
        owner = raw["observation_owner"], raw["observation_generation"]
        _require(type(owner[0]) is str and bool(owner[0]) and type(owner[1]) is int and owner[1] > 0,
                 "statistics_native_owner")
        if phase == "initialization_return":
            self.native_owner = owner
        else:
            _require(owner == self.native_owner, "native_owner_changed_within_segment")
        if previous is not None:
            _require(all(raw[k] >= previous[k] for k in _COUNTERS)
                     and row["work_since_before"] == {k: raw[k] - previous[k] for k in _COUNTERS},
                     "statistics_counter_delta")
            _require(raw["num_steps"] == previous["num_steps"] + 1, "onestep_not_consecutive")
            self.steps += 1
        return raw

    def _sample(self, when, origin, predecessor, *, initial=False):
        row = self.rows.take("voltage_lift_rate_pair")
        self._context(row)
        _require(row["record_sha256"] == _digest({k: v for k, v in row.items() if k != "record_sha256"}),
                 "sample_record_digest")
        _require(row["segment_id"] == self.segment["id"] and row["segment_sha256"] == _digest(self.segment)
                 and row["controls_sha256"] == self.policy["controls_sha256"]
                 and row["observation_policy_sha256"] == self.policy_id
                 and row["current_certificate_follows"] is (not initial), "sample_context")
        h = row["history"]
        side = (("continuous" if self.ordinal == 1 else "right") if initial
                else "left" if when == self.segment["end"] else "continuous")
        _require(h["schema"] == "solarlab.voltage-lift-sample.v2"
                 and h["transition_representation"] == "paired-endpoints-v1"
                 and h["map_identity"] == self.map_id and h["physical_model_identity"] == self.mapping["physical_model"]
                 and h["reference_digest"] == self.reference_digest and h["predecessor_identity"] == predecessor
                 and h["time_hex"] == float(when).hex() and h["origin"] == origin and h["event_side"] == side
                 and h["raw_coordinate_frame"] == "scaled-voltage-departure-v1"
                 and h["physical_rate_frame"] == "direct-map-push-forward-v1" and h["reference_embedded"] is False,
                 "sample_history_binding")
        _require(type(h["point_identity"]) is str and len(h["point_identity"]) == 64, "point_identity_shape")
        for name in ("physical_cumulative_words_hex", "physical_rate_words_hex"):
            for word in _vector(h[name], 4):
                _hex_vector(word, self.size)
        for name in ("raw_solver_z_hex", "raw_solver_zdot_hex"):
            _hex_vector(h[name], self.size)
        _hex_vector(h["inputs_hex"], 2)
        _hex_vector(h["input_rates_hex"], 2)
        _require(row["input_slope_hex"] == h["input_rates_hex"] and row["state_checks"]["passed"] is True
                 and row["state_checks"]["state_identity"] == h["point_identity"]
                 and row["state_checks"]["state_changed"] is False, "sample_state_gate")
        metrics, budgets = row["state_checks"], self.request["budgets"]
        for metric, cap in (("constraint_potential_bound_V", "constraint_potential_V"),
                            ("constraint_charge_bound_C", "constraint_charge_C"),
                            ("contact_max_relative", "contact_relative")):
            _require(type(metrics[metric]) in (int, float) and 0 <= metrics[metric] <= budgets[cap], "state_gate_values")
        _require(metrics["constraint_charge_bound_C"] == max(metrics["Poisson_absolute_constraint_charge_C"],
                 *_vector(metrics["metal_charge_error_bound_C"], 2), metrics["body_charge_error_bound_C"]),
                 "state_charge_bound_consistency")
        if "c_m3" in self.request["numeric_packet"]["variable_offsets"]:
            for name in ("ion_inventory_relative", "maximum_site_fraction"):
                _require(0 <= metrics[name] <= budgets[name], "dynamic_state_gate")
            _vector(metrics["ion_inventory_change_words"], 2)
            _require(0 <= metrics["minimum_f"] <= metrics["maximum_f"] <= 1, "dynamic_occupancy_gate")
        _require(row["physical_rate_consumption"] == "public RateView full mapped words"
                 and row["rate_projection_role"] == "counterfactual_first_word_only"
                 and row["rate_projection"] == h["physical_rate_projection"], "sample_rate_semantics")
        for name, observed_origin in (("raw", origin), ("physical_tangent", "physical_tangent")):
            obs = row[name]
            _require(obs["point_identity"] == h["point_identity"] and obs["origin"] == observed_origin
                     and obs["event_side"] == side, "observable_point_binding")
            _hex_vector(obs["physical_ydot_hex"], self.size)
            _hex_vector(obs["tangent_residual_hex"], self.size - len(set(self.request["numeric_packet"]["mass_columns"])))
            for field, size in (("metal_charge", 2), ("conduction_inward", 2), ("total_inward", 2),
                                ("body_charge", 1), ("body_charge_rate", 1), ("gauss_defect", 1),
                                ("interior_total_current", self.nodes - 1)):
                _word_pair(obs[field], size)
        rate = row["raw"]["physical_rate"]
        _require(rate["point_identity"] == h["point_identity"] and rate["source_identity"] == self.mapping["physical_model"]
                 and rate["mapping_identity"] == self.map_id and rate["frame"] == "mapped-coordinate-rate"
                 and rate["observed_origin"] == origin and rate["values_words_hex"] == h["physical_rate_words_hex"]
                 and rate["raw_coordinates_hex"] == h["raw_solver_z_hex"]
                 and rate["raw_rate_hex"] == h["raw_solver_zdot_hex"] and rate["input_rate_hex"] == h["input_rates_hex"]
                 and row["raw"]["physical_ydot_hex"] == h["physical_rate_words_hex"][0], "physical_rate_word_binding")
        _word_pair(row["raw"]["charge_integrands"], 3)
        _require(type(row["raw"]["linear_actions"]) is dict and bool(row["raw"]["linear_actions"]),
                 "missing_action_receipts")
        for action in row["raw"]["linear_actions"].values():
            _require(all(_hex(v) >= 0 for v in action["absolute_error_bound_hex"]), "action_error_words")
        if origin == "declared_polynomial":
            _require(row["coefficient_frame_identity"] == self.frame_id and row["polynomial_path_identity"] == self.path_id
                     and row["native_getter_used"] is False and row["native_output"] is False,
                     "polynomial_sample_binding")
        self.samples += 1
        return row

    @staticmethod
    def _pointer(row):
        h = row["history"]
        return {"record_sha256": row["record_sha256"], "point_identity": h["point_identity"],
                "time_s": _hex(h["time_hex"]), "origin": h["origin"], "segment_id": row["segment_id"]}

    def _certificate(self, sample):
        row = self.rows.take("voltage_lift_current_certificate")
        evidence = _decode(row["evidence"])
        h, rate = sample["history"], sample["raw"]["physical_rate"]
        _require(row["sample_record_sha256"] == sample["record_sha256"]
                 and evidence["point_identity"] == h["point_identity"] and evidence["path_identity"] == self.path_id
                 and evidence["passed"] is True and evidence["native_rate_or_state_changed"] is False
                 and evidence["DAE_accuracy_certified"] is False, "current_certificate_binding")
        rb = evidence["readback"]
        when = _hex(h["time_hex"])
        side = "right" if when == self.segment["start"] else "left" if when == self.segment["end"] else "continuous"
        _require(rb["point_identity"] == h["point_identity"] and rb["rate_identity"] == rate["identity"]
                 and rb["map_identity"] == self.map_id and rb["path_identity"] == self.path_id
                 and rb["binding_identity"] == self.initial_proof["source_identity"]
                 and rb["time_hex"] == h["time_hex"] and rb["event_side"] == side
                 and rb["state_changed"] is False and rb["native_source_provenance_claimed"] is False,
                 "current_readback_binding")
        for name, field, size in (("raw_z_words", "raw_solver_z_hex", self.size),
                                  ("raw_zdot_words", "raw_solver_zdot_hex", self.size),
                                  ("input_words", "inputs_hex", 2), ("input_rate_words", "input_rates_hex", 2)):
            _require(rb[name] == _word_bytes(h[field], size), "current_readback_words")
        _require(rb["physical_rate_words"] == tuple(_word_bytes(w, self.size) for w in h["physical_rate_words_hex"]),
                 "current_readback_rate_layers")
        for name in ("physical_state_absolute_error", "physical_rate_absolute_error", "state_action_error_bounds"):
            _nonnegative(rb[name], self.size)
        _enclosures(rb["physical_state_enclosures"], self.size)
        _require(rb["state_error_includes_actual_input_evaluation"] is True
                 and rb["do_not_add_same_input_error_twice"] is True
                 and evidence["input_error_included_once"] is True, "current_input_error_semantics")
        reported = tuple(_word_pair(sample[name]["total_inward"], 2) for name in ("raw", "physical_tangent"))
        _require(evidence["reported_currents_A"] == reported, "current_reported_high_low")
        references = (_enclosures(evidence["raw_reference_A"], 2),
                      _enclosures(evidence["same_state_exact_tangent_reference_A"], 2))
        errors = tuple(tuple(abs(v - ref["center"]) + ref["absolute_error_bound"] for v, ref in zip(values, refs))
                       for values, refs in zip(reported, references))
        combined = tuple(abs(reported[0][i] - reported[1][i]) + errors[0][i] + errors[1][i] for i in range(2))
        limit = Fraction(float(self.request["budgets"]["current_A"]) / 3)
        _require(evidence["source_and_readback_error_A"] == errors
                 and evidence["combined_current_bound_A"] == combined and evidence["original_allocation_A"] == limit
                 and all(v <= limit for v in combined), "current_bound_consistency")

    def _interval(self, left, right, frame):
        row = self.rows.take("voltage_lift_interval_charge")
        evidence = _decode(row["observation"])
        _require(row["segment_id"] == self.segment["id"] and row["left"] == self._pointer(left)
                 and row["right"] == self._pointer(right) and row["coefficient_frame_identity"] == self.frame_id
                 and evidence["frame_identity"] == self.frame_id and evidence["path_identity"] == self.path_id
                 and evidence["passed"] is True and row["endpoint_restore"] is None
                 and all(row[k] is False for k in ("normal_query_performed", "states_projected",
                                                  "reference_estimates_are_continuum_certificates")),
                 "interval_binding_or_gate")
        clock = evidence["clock"]
        for key, value in (("predecessor", frame["predecessor"]["internal_t"]),
                           ("tn", frame["endpoint"]["internal_t"]), ("hused", frame["native_before"]["hused"]),
                           ("segment_start", self.segment["start"]), ("segment_end", self.segment["end"])):
            _require(Fraction(*clock[key]) == Fraction(value), "interval_clock")
        _require(clock["owner"] == frame["owner"] and clock["generation"] == frame["generation"]
                 and clock["nsteps"] == frame["native_before"]["nsteps"]
                 and clock["kused"] == frame["native_before"]["kused"] and clock["segment_id"] == self.segment["id"],
                 "interval_clock_owner")
        native_left = Fraction(frame["native_before"]["tn"]) - Fraction(frame["native_before"]["hused"])
        _require(Fraction(*clock["native_left"]) == native_left
                 and Fraction(*clock["uncovered_strip"]) == max(Fraction(0), native_left - Fraction(frame["predecessor"]["internal_t"])),
                 "interval_clock_strip")
        inputs = evidence["input_error"]
        _require(inputs["path_identity"] == self.path_id and inputs["mapping_identity"] == self.map_id
                 and inputs["native_admitted"] is False and inputs["endpoint_error_still_required"] is True
                 and inputs["native_readback_error_still_required"] is True, "interval_input_binding")
        for name in ("raw_charge_integral_error_C", "tangent_charge_integral_error_C"):
            _nonnegative(inputs[name], 3)
        # Charge balances have three ledger rows; the L1 departure has two ports.
        _nonnegative(inputs["raw_tangent_L1_additional_error_C"], 2)
        _require(set(evidence["ledgers"]) == set(_LEDGERS), "missing_charge_ledger")
        for key in _LEDGERS:
            ledger = evidence["ledgers"][key]
            _require(ledger["passed"] is True and ledger["endpoint_debited_once"] is True
                     and all(v is True for v in _vector(ledger["row_checks"], 3)), "charge_ledger_gate")
            for name in ("saved_finite_charge_change_C", "nominal_change_C", "signed_integral_C", "signed_saved_defect_C"):
                ledger[name] = _enclosures(ledger[name], 3)
            bound = _nonnegative(ledger["interval_total_bound_C"], 3)
            absolute = _nonnegative(ledger["prefix_absolute_defect_C"], 3)
            error = _nonnegative(ledger["interval_reference_error_C"], 3)
            cumulative = _nonnegative(ledger["prefix_reference_error_C"], 3)
            budget = Fraction(self.request["budgets"]["charge_C"])
            # The controller records binary64(B/3); ChargePrefix also enforces
            # the exact 3*error<=B condition. Preserve both, including rounding.
            reference_share = Fraction(float(self.request["budgets"]["charge_C"]) / 3)
            _require(ledger["original_charge_budget_C"] == budget and ledger["original_reference_share_C"] == reference_share
                     and all(v <= budget for v in (*bound, *absolute))
                     and all(v <= reference_share and 3 * v <= budget for v in (*error, *cumulative)), "charge_budget_binding")
            components = ledger["error_components_C"]
            _require(set(components) == {"endpoint", "input", "clock_strip", "raw_tangent_departure", "integration"},
                     "charge_error_components")
            for values in components.values():
                _nonnegative(values, 3)
            integrals, nominal, actual = ledger["signed_integral_C"], ledger["nominal_change_C"], ledger["saved_finite_charge_change_C"]
            _require(all(v["absolute_error_bound"] == 0 for v in nominal)
                     and components["integration"] == tuple(v["absolute_error_bound"] for v in integrals)
                     and error == tuple(sum((v[i] for v in components.values()), Fraction(0)) for i in range(3))
                     and bound == tuple(abs(a["center"] - b["center"]) + e for a, b, e in zip(nominal, integrals, error)),
                     "charge_error_arithmetic")
            for a, b, defect in zip(actual, integrals, ledger["signed_saved_defect_C"]):
                _require(defect == {"center": a["center"] - b["center"],
                                    "absolute_error_bound": a["absolute_error_bound"] + b["absolute_error_bound"]},
                         "saved_charge_defect_words")
            old_abs, old_error = self.prefix[key]
            _require(absolute == tuple(a + b for a, b in zip(old_abs, bound))
                     and cumulative == tuple(a + b for a, b in zip(old_error, error)), "charge_prefix_reset_or_error")
            self.prefix[key] = absolute, cumulative
        calls = row["arithmetic_calls"]
        _require(type(calls) is list and bool(calls), "arithmetic_receipts_missing")
        for call in calls:
            if call.get("purpose") == "full-real-interval-precheck":
                # BallIntegrator.absolute_bound first checks a real enclosure;
                # this receipt is not an integral value/error estimate.
                _require(call == {"evaluations": 1, "purpose": "full-real-interval-precheck", "working_bits": 512}
                         and type(call["evaluations"]) is int and type(call["working_bits"]) is int,
                         "arithmetic_precheck_receipt")
                continue
            actual, requested = Fraction(*call["actual_error"]), Fraction(*call["requested_error"])
            _require(type(call["evaluations"]) is int and call["evaluations"] > 0
                     and 0 <= actual <= requested, "arithmetic_receipt_error")
        self.intervals += 1

    def run(self):
        row = self.rows.take("voltage_lift_reference")
        ref = row["reference"]
        self.reference_digest = _digest(ref)
        _require(row["request_sha256"] == self.request_id and row["reference_digest"] == self.reference_digest
                 and ref["schema"] == "solarlab.voltage-lift-reference.v1" and ref["map"] == self.mapping
                 and ref["map_identity"] == self.map_id and ref["physical_reference"] == self.request["numeric_packet"]["initial_reference"]
                 and ref["physical_reference_identity"] == self.mapping["physical_reference"]
                 and _digest(ref["physical_reference"]["payload"]) == ref["physical_reference"]["sha256"],
                 "reference_binding")
        predecessor = ref["physical_reference_identity"]
        last_time = _hex(ref["physical_reference"]["payload"]["time"])
        last_right = None
        previous_native = None
        for self.ordinal, self.segment in enumerate(self.request["segments"], 1):
            self.rows.checkpoint()
            start, end = self.segment["start"], self.segment["end"]
            _require(start == last_time and start < end, "protocol_segment_order")
            times = self.request["observation_times"][self.segment["id"]]
            _require(type(times) is list and bool(times) and times == sorted(times)
                     and all(start <= v <= end for v in times), "requested_sample_schedule")
            init = self.rows.take("voltage_lift_initialization_input")
            self.initial_proof = proof = init["proof"]
            _require(init["phase"] == "segment_initialization" and init["segment_id"] == self.segment["id"]
                     and _hex(init["time_hex"]) == start and proof["request_sha256"] == self.request_id
                     and proof["map_identity"] == self.map_id and proof["segment_sha256"] == _digest(self.segment)
                     and proof["predecessor_identity"] == predecessor
                     and proof["record_sha256"] == _digest({k: v for k, v in proof.items() if k != "record_sha256"}),
                     "initialization_binding")
            if self.ordinal == 1:
                for key in ("point_identity", "raw_z_hex", "raw_zdot_hex", "inputs_hex", "input_rates_hex",
                            "desired_physical_tangent_hex", "mapped_physical_rate_words_hex",
                            "desired_tangent_residual_SI", "represented_rate_residual_SI"):
                    _require(proof[key] == self.request["initial_preparation"][key], "initial_preparation_changed")
            if self.startup.enabled:
                self.startup.consume(self.rows.take("voltage_lift_initialization_controls"))
            stats = self._statistics("initialization_return", start)
            _require(stats["num_steps"] == 0, "initialization_step_counter")
            left = self._sample(start, "segment_initial", predecessor, initial=True)
            _require(left["history"]["point_identity"] == proof["point_identity"]
                     and left["history"]["raw_solver_z_hex"] == proof["raw_z_hex"]
                     and left["history"]["raw_solver_zdot_hex"] == proof["raw_zdot_hex"], "initialization_words")
            if last_right is not None:
                _require(left["history"]["physical_cumulative_words_hex"] == last_right["physical_cumulative_words_hex"],
                         "event_state_changed")
            cursor, segment_intervals = 0, 0
            while cursor < len(times) and times[cursor] == start:
                self._requested(self._pointer(left))
                cursor += 1
            now = start
            while now < end:
                before = self._statistics("before_onestep")
                _require(all(before[k] >= stats[k] for k in _COUNTERS)
                         and before["num_steps"] == stats["num_steps"], "statistics_between_steps")
                stats = self._statistics("after_onestep", previous=before)
                packet_row = self.rows.take("voltage_lift_native_observation_packet")
                _require(packet_row["segment_id"] == self.segment["id"], "packet_segment")
                frame, self.frame_id = _native_packet(packet_row["packet"], self.policy, self.size)
                previous, endpoint = frame["predecessor"], frame["endpoint"]
                then = endpoint["internal_t"]
                if previous_native is not None:
                    old = previous_native
                    same_epoch = (old["owner"], old["generation"]) == (frame["owner"], frame["generation"])
                    _require(old["endpoint"]["internal_t"].hex() == previous["internal_t"].hex(), "native_frame_time_gap")
                    if same_epoch:
                        _require(old["segment"]["id"] == self.segment["id"]
                                 and old["endpoint"]["identity"] == previous["identity"], "native_frame_predecessor_identity")
                    else:
                        _require(old["segment"]["id"] != self.segment["id"]
                                 and old["endpoint"]["internal_t"] == old["segment"]["end"]
                                 and previous["internal_t"] == self.segment["start"]
                                 and old["endpoint"]["raw_y"] == previous["raw_y"]
                                 and frame["native_before"]["nsteps"] == 1 and previous["nsteps"] == 0
                                 and old["segment"]["voltage"][1] == self.segment["voltage"][0]
                                 and old["segment"]["photons"][1] == self.segment["photons"][0]
                                 and frame["generation"] == (old["generation"] + 1 if old["owner"] == frame["owner"] else 1),
                                 "native_restart_epoch")
                        # Rates may change at a protocol derivative kink; do
                        # not equate the two independently retained rate sides.
                strip = Fraction(then) - Fraction(frame["native_before"]["hused"]) - Fraction(previous["internal_t"])
                _require(strip <= 0 or frame["predecessor_dky"]["native_eligible"] is True, "native_strip_getter_eligibility")
                _require(previous["internal_t"] == now < then <= end
                         and stats["num_steps"] == frame["native_before"]["nsteps"]
                         and stats["current_time"] == then
                         and stats["observation_owner"] == frame["owner"]
                         and stats["observation_generation"] == frame["generation"], "frame_chronology_or_statistics")
                for name, field in (("raw_y", "raw_solver_z_hex"), ("raw_yp", "raw_solver_zdot_hex")):
                    _require(previous[name] == _word_bytes(left["history"][field], self.size), "predecessor_word_binding")
                domain = _decode(self.rows.take("voltage_lift_polynomial_domain")["evidence"])
                self.path_id = domain["path_identity"]
                _require(type(self.path_id) is str and len(self.path_id) == 64 and domain["passed"] is True
                         and all(value is True for value in domain["checks"].values()), "polynomial_domain_gate")
                fields = set(self.request["numeric_packet"]["variable_offsets"])
                _require(set(domain["domain_bounds"]) == fields, "polynomial_domain_fields")
                _require(set(domain["checks"]) == ({"maximum_site_fraction", "ion_inventory"} if "c_m3" in fields else set()),
                         "polynomial_domain_checks")
                for bounds in domain["domain_bounds"].values():
                    for lo, hi in _vector(bounds, self.nodes):
                        _require(type(lo) is Fraction and type(hi) is Fraction and lo <= hi, "polynomial_domain_bounds")
                self._certificate(left)  # Deliberately re-certified in every new frame.
                origin = "stop_output" if endpoint["native_status"] == 1 else "native"
                right = self._sample(then, origin, left["history"]["point_identity"])
                _require(right["snapshot_status"] == endpoint["native_status"], "endpoint_status")
                for name, field in (("raw_y", "raw_solver_z_hex"), ("raw_yp", "raw_solver_zdot_hex")):
                    _require(endpoint[name] == _word_bytes(right["history"][field], self.size), "endpoint_word_binding")
                self._certificate(right)
                cache = {now: self._pointer(left), then: self._pointer(right)}

                def query(when):
                    self.rows.checkpoint()
                    _require(now <= when <= then, "query_outside_frame")
                    if when not in cache:
                        sample = self._sample(when, "declared_polynomial", left["history"]["point_identity"])
                        self._certificate(sample)
                        cache[when] = self._pointer(sample)
                        self.queries += 1
                    return cache[when]

                for order in (8, 16, 32):
                    for node in self.request["quadrature"][str(order)]["nodes"]:
                        query(now + (then - now) * (float(node) + 1) / 2)
                while cursor < len(times) and times[cursor] <= then:
                    self._requested(query(float(times[cursor])))
                    cursor += 1
                self._interval(left, right, frame)
                previous_native = {"endpoint": endpoint, "owner": frame["owner"],
                                   "generation": frame["generation"], "segment": self.segment}
                segment_intervals += 1
                left, now = right, then
            _require(cursor == len(times), "missing_requested_samples")
            last_right = left["history"]
            predecessor, last_time = last_right["point_identity"], now
            self.final_pointer = self._pointer(left)
            self.final_history, self.final_status = last_right, left["snapshot_status"]
            self.coverage.append({"segment_id": self.segment["id"], "intervals": segment_intervals,
                                  "requested_samples": cursor, "start_s": start, "end_s": end})
        _require(self.coverage and self.intervals > 0, "empty_protocol")
        self.startup.finish(complete=True)
        self.rows.take(eof=True)

    def _requested(self, pointer):
        row = self.rows.take("requested_sample")
        _require(row == {"kind": "requested_sample", **pointer}, "requested_sample_pointer")


def verify_history(folder: Path, *, max_record_bytes: int, max_logical_bytes: int,
                   max_records: int, deadline_monotonic: float, integer_io_policy=None) -> dict:
    """Verify a finalized, complete protocol, raising on every partial result.

    All limits are caller-selected stop rules. A receipt verifies stored words,
    record coverage and the recorded acceptance gates; it neither repeats their
    numerical proofs nor establishes full-device convergence or performance.
    """
    started = time.perf_counter()
    rows = None

    def checkpoint():
        _require(time.perf_counter() < deadline_monotonic, "deadline_exceeded")

    try:
        _require(all(type(v) is int and v > 0 for v in (max_record_bytes, max_logical_bytes, max_records)),
                 "positive_limits_required")
        _require(type(deadline_monotonic) in (int, float) and math.isfinite(deadline_monotonic), "finite_deadline_required")
        integer_io_policy = configure_integer_io(integer_io_policy, max_record_bytes)
        integer_observed = integer_io_observed(integer_io_policy)
        folder = Path(folder)
        checkpoint()
        _require(not any((folder / name).exists() for name in ("FirstFailure.json", "RunnerFailure.json", "PrimaryFailure.json",
                                                              "LastRecord.tmp", "LastAccepted.tmp", "FirstFailure.tmp")),
                 "failure_or_incomplete_publication")
        request = _read_json(folder / "NativeRequest.json", max_record_bytes, checkpoint)
        result = _read_json(folder / "NativeResult.json", max_record_bytes, checkpoint, integer_io_policy)
        history = result["history"]
        _require(history.get("integer_io") == integer_observed, "integer_io_policy_or_applied_limit_mismatch")
        if integer_observed is not None:
            producer = _read_json(folder / "IntegerIOObserved.json", max_record_bytes, checkpoint)
            _require(producer.get("role") == "producer" and type(producer.get("pid")) is int
                     and producer["pid"] > 0
                     and {key: producer.get(key) for key in integer_observed} == integer_observed,
                     "producer_integer_io_observation_mismatch")
        else:
            _require(not (folder / "IntegerIOObserved.json").exists(), "unbound_integer_io_observation")
        _require(result["status"] == "completed_bounded_voltage_lift_native_pilot" and result["complete_protocol"] is True
                 and result.get("first_failure") is None and history["closed"] is True
                 and history["container_complete"] is True and history["error"] is None
                 and history["encoding"] == "gzip" and history["path"] == "NativeHistory.jsonl.gz", "incomplete_native_result")
        path = folder / history["path"]
        before_stat = path.stat()
        _require(before_stat.st_size == history["encoded_bytes"]
                 and before_stat.st_size <= request["budgets"]["total_output_bytes"], "encoded_byte_count_or_cap")
        with HistoryReader(path, encoding="gzip", max_record_bytes=max_record_bytes,
                            max_logical_bytes=max_logical_bytes, integer_io_policy=integer_io_policy) as reader:
            rows = _Rows(reader, max_records, checkpoint)
            protocol = _Protocol(request, rows)
            _require(result["request_sha256"] == protocol.request_id and result["map_identity"] == protocol.map_id
                     and result["observation_policy_sha256"] == protocol.policy_id, "result_request_binding")
            protocol.run()
            _require(reader.eof_seen and reader.container_complete is True and not reader.partial_tail, "container_incomplete")
            _require(history["records"] == rows.count and history["logical_bytes"] == reader.logical_bytes
                     and history["logical_sha256"] == rows.hash.hexdigest(), "logical_count_or_digest")
        counts = result["counts"]
        expected = {"history_bytes": history["logical_bytes"], "history_encoded_bytes": history["encoded_bytes"],
                    "native_steps": protocol.steps, "onestep_returns": protocol.intervals,
                    "normal_queries": 0, "polynomial_queries": protocol.queries,
                    "initializations": len(protocol.coverage), "jacobian": protocol.jacobians}
        _require(all(type(counts[k]) is int and counts[k] == value for k, value in expected.items()), "result_counters")
        _require(type(counts["residual"]) is int and 0 <= counts["residual"] <= request["budgets"]["residual_calls"]
                 and protocol.steps <= request["budgets"]["native_steps"], "native_work_caps")
        _require(result["last_numerical"] == protocol.final_pointer
                 and result["last_physically_accepted"] == protocol.final_pointer, "result_final_pointers")
        attempt = result["last_attempt"]
        _require(attempt["phase"] == "native_return" and attempt["success"] is True
                 and attempt["segment_id"] == protocol.final_pointer["segment_id"]
                 and attempt["time_hex"] == protocol.final_history["time_hex"]
                 and attempt["status"] == protocol.final_status
                 and attempt["z_hex"] == protocol.final_history["raw_solver_z_hex"]
                 and attempt["zdot_hex"] == protocol.final_history["raw_solver_zdot_hex"], "result_last_attempt")
        for key, (absolute, error) in protocol.prefix.items():
            _require(_decode(result["cumulative_absolute_charge_bounds_C"][key]) == absolute
                     and _decode(result["cumulative_observation_reference_bounds_C"][key]) == error, "result_charge_prefix")
        last = _read_json(folder / "LastRecord.json", max_record_bytes, checkpoint)
        accepted = _read_json(folder / "LastAccepted.json", max_record_bytes, checkpoint)
        _require({k: last[k] for k in rows.last} == rows.last
                 and last["encoded_byte_offset_after_record"] + 10 == history["encoded_bytes"]
                 and accepted == {"pointer": last, "right": protocol.final_pointer, "frame": protocol.frame_id},
                 "final_sidecar_binding")
        container_hash, size = hashlib.sha256(), 0
        # A compressed-byte pass supplies the full container digest without a
        # second decompression or an in-memory copy of the history.
        with path.open("rb") as stream:
            _require(stream.read(10) == b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff", "gzip_writer_header")
            stream.seek(0)
            while True:
                checkpoint()
                chunk = stream.read(1048576)
                if not chunk:
                    break
                size += len(chunk)
                _require(size <= history["encoded_bytes"], "container_grew")
                container_hash.update(chunk)
            stream.seek(-10, 2)
            trailer = stream.read(10)
        _require(trailer == b"\x03\x00" + struct.pack("<II", rows.crc, history["logical_bytes"] % (1 << 32)),
                 "gzip_final_flush_crc_or_size")
        after_stat = path.stat()
        _require(size == history["encoded_bytes"] and
                 (before_stat.st_dev, before_stat.st_ino, before_stat.st_size, before_stat.st_mtime_ns) ==
                 (after_stat.st_dev, after_stat.st_ino, after_stat.st_size, after_stat.st_mtime_ns), "container_changed")
        checkpoint()
        return {"schema": "solarlab.native-history-verification.v1", "verified": True, "complete_protocol": True,
                "status": "verified_complete_native_history", "request_sha256": protocol.request_id,
                "map_identity": protocol.map_id, "observation_policy_sha256": protocol.policy_id,
                "history": {"path": history["path"], "logical_bytes": history["logical_bytes"],
                            "encoded_bytes": size, "records": rows.count, "logical_sha256": rows.hash.hexdigest(),
                            "container_sha256": container_hash.hexdigest(), "eof_seen": True, "crc_verified": True,
                            "container_complete": True, "partial_tail_bytes": 0},
                "coverage": {"segments": protocol.coverage, "intervals": protocol.intervals,
                             "rate_pairs": protocol.samples, "quadrature_orders": [8, 16, 32], "ports": 2,
                             "physical_word_layers": 4, "record_counts": dict(rows.counts)},
                "pointers": {"last_record": last, "last_accepted": accepted}, "elapsed_s": time.perf_counter() - started,
                **({"segment_startup": protocol.startup.finish(complete=True)} if protocol.startup.enabled else {}),
                **({"integer_io": {"role": "independent_reader", **integer_io_observed(integer_io_policy)}}
                   if integer_io_policy is not None else {}),
                "scope": "stored artifact integrity, complete request coverage and recorded gate consistency; no numerical replay",
                "scientific_qualification": False}
    except HistoryVerificationError as error:
        if rows is not None and error.record_index == 0:
            raise HistoryVerificationError(error.code, rows.count, rows.offset) from error
        raise
    except (OSError, EOFError, ValueError, KeyError, TypeError, IndexError, RecursionError, OverflowError, struct.error, zlib.error) as error:
        # Do not include a malformed giant row or native words in diagnostics.
        raise HistoryVerificationError("invalid_artifact_" + type(error).__name__,
                                       0 if rows is None else rows.count,
                                       0 if rows is None else rows.offset) from error
