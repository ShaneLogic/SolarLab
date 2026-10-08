"""Small artifact-corruption tests; fixtures are not a physical native run.

Only stdlib and the recording/readback modules are imported. The fixture has two
event sides, three accepted intervals, both current words, and repeated queries
across the 8/16/32 tables. It asserts infrastructure integrity, not numerical
correctness of the invented states or the recorded scientific assertions.
"""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import zlib

from scripts.benchmarks.native_history import HistoryWriter, encode_integer_values, integer_io_observed, integer_json_dumps
from scripts.benchmarks.native_readback import HistoryVerificationError, verify_history, _Protocol, _native_packet


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def tag(value):
    if type(value) is Fraction:
        return {"rational": [value.numerator, value.denominator]}
    if type(value) is bytes:
        return {"bytes_hex": value.hex()}
    if type(value) is float:
        return {"binary64": value.hex()}
    if type(value) is tuple:
        return {"tuple": [tag(v) for v in value]}
    if type(value) is dict:
        return {k: tag(v) for k, v in value.items()}
    if type(value) is list:
        return [tag(v) for v in value]
    return value


def frame_id(value):
    def typed(v):
        if type(v) is dict:
            return ["mapping", [[k, typed(v[k])] for k in sorted(v)]]
        if type(v) is tuple:
            return ["tuple", [typed(w) for w in v]]
        if type(v) is bytes:
            return ["bytes", v.hex()]
        if type(v) is float:
            return ["binary64", v.hex()]
        return [type(v).__name__, v]
    return digest(typed(value))


def words(value, size=6):
    return [float(value).hex()] * size


def binary(values):
    return struct.pack(f"<{len(values)}d", *(float.fromhex(v) for v in values))


def double(size):
    return {"high_hex": words(1, size), "low_hex": words(2**-60, size)}


class SegmentFrameReadbackTests(unittest.TestCase):
    """Six-component invented data test the independent reader, not IDA."""

    def make(self):
        from scripts.benchmarks.native_readback import SegmentFrameCheck

        n = 6; zero = words(0, n); rate = [float(2*(i+1)).hex() for i in range(n)]
        segment = dict(id="unit", start=0.0, end=1.0, voltage=[0.0, 0.0], photons=[0.0, 0.0])
        mapping = dict(physical_model="a"*64, physical_reference="b"*64, layout="c"*64,
                       columns_hex=words(1, n), rows_hex=words(1, n), lift_hex=[words(0, 2)]*n,
                       reference_inputs_hex=words(0, 2))
        controls = dict(rtol=0.125, atol=[0.5]*n, first_step=0.125)
        request = dict(controls=controls, voltage_lift_map=mapping, map_identity=digest(mapping),
                       prior_request_sha256="d"*64, segments=[segment], observation_times={"unit": [0.0, 1.0]},
                       quadrature={}, weight_certificate={}, z0=[0.0]*n)
        request["segment_frame_policy"] = dict(schema="solarlab.segment-frame-policy.v1", name="fixed-affine-state-rate-v1",
            ancestor_request_sha256=request["prior_request_sha256"], parent_map_identity=request["map_identity"],
            controls_sha256=digest(controls), segments_sha256=digest([segment]),
            sampling_sha256=digest((request["observation_times"], request["quadrature"])), weight_certificate_sha256=digest({}),
            segment_startup_policy_sha256=digest(None), qualification_policy_sha256=digest(None),
            application="one_fixed_live_origin_at_each_existing_fresh_segment_initialization",
            weight_rounding="RN-exact-parent-q_then-unfused-SV-v1", requested_first_steps_unchanged=True,
            actual_automatic_first_step_parity_claimed=False)
        fields = {"n_m3": [0, 2], "p_m3": [2, 4], "phi_V": [4, 6]}
        request["numeric_packet"] = {"variable_offsets": fields, "initial_reference": {"payload": {
            "layout": "c"*64, "fields": {k: {"high": words(1, 2), "low": words(0, 2)} for k in fields}}}}
        rid = digest(request)
        parent = dict(request_sha256=rid, predecessor_identity="b"*64, segment_sha256=digest(segment), time_hex=0.0.hex(),
                      raw_z_hex=zero, raw_zdot_hex=rate, inputs_hex=words(0, 2), input_rates_hex=words(0, 2),
                      state_changed=False, native_initialization_performed=False, mapped_physical_rate_words_hex=[rate, zero, zero, zero])
        parent["record_sha256"] = digest(parent)
        frame = dict(schema="solarlab.segment-affine-frame.v1", parent_map_identity=request["map_identity"],
                     segment_sha256=digest(segment), logical_initialization_index=1, predecessor_identity="b"*64,
                     parent_input_sha256=parent["record_sha256"], t0_hex=0.0.hex(), size=n,
                     q0_words_hex=[zero]*4, v0_words_hex=[rate, zero, zero, zero],
                     law="q=Q0+(t-t0)*V0+u;qdot=V0+udot", weight_rounding="RN-exact-parent-q_then-unfused-SV-v1")
        framed = dict(mapping, segment_frame=frame, raw_coordinate_meaning="fixed-segment affine remainder u; native history is u",
            physical_coordinate_meaning="Point.y is the first retained word of S*(Q0+(t-t0)*V0+u)+L*(a-a_ref)")
        ancestry = dict(request_sha256=rid, parent_map_identity=request["map_identity"], predecessor_identity="b"*64,
                        q0_words_hex=[zero]*4)
        record = dict(kind="voltage_lift_segment_frame", request_sha256=rid, parent_map_identity=request["map_identity"],
                      frame=frame, frame_identity=digest(frame), map=framed, map_identity=digest(framed),
                      ancestry=ancestry, preparation_frame=dict(frame, v0_words_hex=[zero]*4, parent_input_sha256=digest(ancestry)),
                      parent_input=parent, native_initial_z_hex=zero, native_initial_zdot_hex=zero,
                      physical_handoff_words_hex=[zero]*4, physical_rate_handoff_words_hex=[rate, zero, zero, zero])
        record["record_sha256"] = digest(record)
        check = SegmentFrameCheck(request); check.begin(record, segment, 1, "b"*64)
        weights = dict(schema="sksundae.ida.parent-affine-weight-state.v1", owner="unit", generation=1, size=n,
                       frame_identity=digest(frame), build_identity="e"*64, installed=True, setter="IDAWFtolerances", setter_status=0,
                       t0_hex=0.0.hex(), rtol_hex=controls["rtol"].hex(), atol=binary(words(0.5, n)),
                       q0_words=tuple(binary(w) for w in frame["q0_words_hex"]),
                       v0_words=tuple(binary(w) for w in frame["v0_words_hex"]), callback_calls=1, callback_status=0,
                       failure_component=None, basis_time_hex=0.0.hex(), basis_u=binary(zero), rounded_parent=binary(zero),
                       computed_weights=binary(words(2, n)), weight_rounding=frame["weight_rounding"], raw_tolerance_getter=False)
        phi = [0.25*(i+1) for i in range(n)]*2+[0.5*(i+1) for i in range(n)]
        packet = dict(owner="unit", generation=1, parent_weight_state=weights, binding={"identity": "e"*64},
                      predecessor={"internal_t": 0.0}, native_before=dict(kused=2, nsteps=1, hused=0.5, tn=0.5),
                      basis={"phi": struct.pack('<18d', *phi), "psi": struct.pack('<2d', 0.5, 1.0)},
                      error_weights=binary(words(2, n)))
        return check, record, packet

    def test_independent_full_polynomial_and_operative_parent_weights(self):
        check, _, packet = self.make(); check.packet(packet)
        for i, row in enumerate(check.physical_polynomials, 1):
            self.assertEqual(row, (1+Fraction(5*i, 4), Fraction(3*i, 2), Fraction(i, 4)))
        self.assertEqual(check.finish()["independent_physical_polynomials"], 1)
        self.assertEqual(check.frames[0]["weight_packets"], 1)

    def test_foreign_generation_bad_weights_and_endpoint_basis_rejected(self):
        for key, value in (("generation", 2), ("frame_identity", "f"*64),
                           ("computed_weights", binary(words(3))), ("q0_words", (binary(words(1)),)*4)):
            check, _, packet = self.make(); packet["parent_weight_state"][key] = value
            with self.assertRaises(HistoryVerificationError):
                check.packet(packet)
        check, _, packet = self.make()
        changed = packet["parent_weight_state"]
        changed.update(basis_time_hex=0.5.hex(), basis_u=binary(words(0.25)))
        parent = [i+0.25 for i in range(1, 7)]
        changed["rounded_parent"] = struct.pack('<6d', *parent)
        changed["computed_weights"] = struct.pack('<6d', *(1/(0.125*q+0.5) for q in parent))
        packet["error_weights"] = changed["computed_weights"]
        with self.assertRaisesRegex(HistoryVerificationError, "operative_weight_basis"):
            check.packet(packet)


class QualificationProtocolSizeTests(unittest.TestCase):
    def test_extended_protocol_consumes_validated_request_size(self):
        # This isolates protocol wiring from the separately tested metadata
        # validator; none of these synthetic requests is a qualified device.
        for size in (51, 85, 99, 165, 195, 325):
            with self.subTest(size=size):
                request = fixture()[0]
                request.update(z0=[0.0]*size, zdot0=[1.0]*size,
                               qualification_policy={"synthetic_wiring_only": True})
                target = "scripts.benchmarks.native_readback.qualification_coordinate_count"
                with patch(target, return_value=size) as check:
                    protocol = _Protocol(request, None)
                self.assertEqual(protocol.size, size)
                self.assertTrue(check.call_count >= 1)
                self.assertTrue(all(call.args == (request,) for call in check.call_args_list))
                with patch(target, return_value=size+1), self.assertRaisesRegex(
                        HistoryVerificationError, "request_schema_or_map"):
                    _Protocol(request, None)
                with self.assertRaisesRegex(HistoryVerificationError, "qualification_request_binding"):
                    _Protocol(request, None)
                request.pop("qualification_policy")
                with self.assertRaisesRegex(HistoryVerificationError, "request_schema_or_map"):
                    _Protocol(request, None)


def point_id(segment, when):
    return digest(["synthetic-point", segment["id"], float(when).hex()])


def initial_fields(segment):
    start = segment["start"]
    return {"point_identity": point_id(segment, start), "raw_z_hex": words(start), "raw_zdot_hex": words(1),
            "inputs_hex": words(0, 2), "input_rates_hex": words(0, 2), "desired_physical_tangent_hex": words(1),
            "mapped_physical_rate_words_hex": [words(1), words(2**-60), words(0), words(0)],
            "desired_tangent_residual_SI": double(6), "represented_rate_residual_SI": double(6)}


def fixture(policy_factory=None):
    size = 6
    source = ("sksundae.ida.accepted-step-observation.v1", "scikit-sundae=1.1.3",
              "synthetic-library", "synthetic-compiler", "synthetic-source", "a" * 64, "b" * 64, "test-only")
    binding = hashlib.sha256(repr(source).encode()).hexdigest()
    segments = [{"id": "a", "start": 0.0, "end": 1.0, "voltage": [0, 0], "photons": [0, 0]},
                {"id": "b", "start": 1.0, "end": 2.0, "voltage": [0, 0], "photons": [0, 0]}]
    reference = {"payload": {"time": 0.0.hex()}}
    reference["sha256"] = digest(reference["payload"])
    mapping = {"schema": "solarlab.affine-voltage-map.v1", "physical_model": "c" * 64,
               "physical_reference": "d" * 64}
    request = {"schema": "solarlab.voltage-lift-native-request.v1", "voltage_lift_map": mapping,
               "map_identity": digest(mapping), "z0": [0.0] * size, "zdot0": [1.0] * size,
               "numeric_packet": {"nodes": 2, "mass_columns": [0, 1, 2, 3],
                                  "variable_offsets": {"n_m3": [0, 2], "p_m3": [2, 4], "phi_V": [4, 6]},
                                  "initial_reference": reference},
               "controls": {"nthreads": 1, "max_step": 1.0}, "segments": segments,
               "budgets": {"charge_C": 1.0, "current_A": 1.0, "residual_calls": 1000,
                           "native_steps": 100, "total_output_bytes": 8 * 1048576,
                           "constraint_potential_V": 1.0, "constraint_charge_C": 1.0, "contact_relative": 1.0},
               "observation_times": {"a": [0.0, 0.25, 1.0], "b": [1.0, 1.5, 2.0]},
               "quadrature": {str(n): {"nodes": [-1 + 2 * (i + 1) / (n + 1) for i in range(n)],
                                       "weights": [2 / n] * n} for n in (8, 16, 32)},
               "initial_preparation": initial_fields(segments[0])}
    policy = {"map_identity": request["map_identity"], "protocol_sha256": digest(segments),
              "controls_sha256": digest(request["controls"]), "budgets_sha256": digest(request["budgets"]),
              "sampling_sha256": digest((request["observation_times"], request["quadrature"])),
              "binding_identity": binding, "header_sha256": source[5]}
    if policy_factory is not None:
        policy = policy_factory(request, policy)
    request["interval_observation"] = policy
    request_id, policy_id = digest(request), digest(policy)
    ref = {"schema": "solarlab.voltage-lift-reference.v1", "map": mapping,
           "map_identity": request["map_identity"], "physical_reference": reference,
           "physical_reference_identity": mapping["physical_reference"]}
    ref_id = digest(ref)
    rows = [{"kind": "voltage_lift_reference", "request_sha256": request_id,
             "reference_digest": ref_id, "reference": ref}]
    zeros = (Fraction(0),) * 3
    enclosure = {"center": [0, 1], "absolute_error_bound": [0, 1]}
    current = Fraction(1) + Fraction(2**-60)
    all_currents = ((current, current), (current, current))
    current_enclosures = ({"center": [current.numerator, current.denominator], "absolute_error_bound": [0, 1]},) * 2
    counters = ("num_steps", "residual_evals", "linear_setups", "error_test_fails",
                "nonlinear_iters", "nonlinear_conv_fails", "jacobian_evals")
    predecessor, queries, total_intervals = mapping["physical_reference"], 0, 0
    pointer = None

    def sample(segment, ordinal, when, origin, previous, frame=None, path=None):
        side = ("continuous" if ordinal == 1 else "right") if origin == "segment_initial" else (
            "left" if when == segment["end"] else "continuous")
        pid = point_id(segment, when)
        h = {"schema": "solarlab.voltage-lift-sample.v2", "transition_representation": "paired-endpoints-v1",
             "map_identity": request["map_identity"], "physical_model_identity": mapping["physical_model"],
             "reference_digest": ref_id, "point_identity": pid, "predecessor_identity": previous,
             "time_hex": float(when).hex(), "inputs_hex": words(0, 2), "input_rates_hex": words(0, 2),
             "raw_solver_z_hex": words(when), "raw_solver_zdot_hex": words(1),
             "physical_cumulative_words_hex": [words(when), words(2**-60), words(0), words(0)],
             "physical_rate_words_hex": [words(1), words(2**-60), words(0), words(0)],
             "physical_rate_projection": {"projected_rate_hex": words(1)}, "origin": origin, "event_side": side,
             "raw_coordinate_frame": "scaled-voltage-departure-v1", "physical_rate_frame": "direct-map-push-forward-v1",
             "reference_embedded": False}
        obs = {"point_identity": pid, "event_side": side, "physical_ydot_hex": words(1),
               "tangent_residual_hex": words(0, 2), **{k: double(n) for k, n in
                (("metal_charge", 2), ("conduction_inward", 2), ("total_inward", 2),
                 ("body_charge", 1), ("body_charge_rate", 1), ("gauss_defect", 1), ("interior_total_current", 1))}}
        rate = {"identity": digest(["rate", pid]), "point_identity": pid, "source_identity": mapping["physical_model"],
                "mapping_identity": request["map_identity"], "frame": "mapped-coordinate-rate", "observed_origin": origin,
                "values_words_hex": h["physical_rate_words_hex"], "raw_coordinates_hex": words(when),
                "raw_rate_hex": words(1), "input_rate_hex": words(0, 2)}
        row = {"kind": "voltage_lift_rate_pair", "request_sha256": request_id, "map_identity": request["map_identity"],
               "segment_id": segment["id"], "segment_sha256": digest(segment), "controls_sha256": policy["controls_sha256"],
               "observation_policy_sha256": policy_id, "current_certificate_follows": origin != "segment_initial",
               "history": h, "input_slope_hex": words(0, 2), "snapshot_status": 1 if origin == "stop_output" else 0,
               "state_checks": {"passed": True, "state_identity": pid, "state_changed": False,
                                "constraint_potential_bound_V": 0.0, "constraint_charge_bound_C": 0.0,
                                "contact_max_relative": 0.0, "Poisson_absolute_constraint_charge_C": 0.0,
                                "metal_charge_error_bound_C": [0.0, 0.0], "body_charge_error_bound_C": 0.0},
               "physical_rate_consumption": "public RateView full mapped words",
               "rate_projection_role": "counterfactual_first_word_only", "rate_projection": h["physical_rate_projection"],
               "raw": {**deepcopy(obs), "origin": origin, "physical_rate": rate, "charge_integrands": double(3),
                       "linear_actions": {"sample": {"absolute_error_bound_hex": words(0, 1)}}},
               "physical_tangent": {**deepcopy(obs), "origin": "physical_tangent"}}
        if origin == "declared_polynomial":
            row.update(coefficient_frame_identity=frame, polynomial_path_identity=path, native_getter_used=False, native_output=False)
        row["record_sha256"] = digest(row)
        return row

    def ptr(pair):
        return {"record_sha256": pair["record_sha256"], "point_identity": pair["history"]["point_identity"],
                "time_s": float.fromhex(pair["history"]["time_hex"]), "origin": pair["history"]["origin"],
                "segment_id": pair["segment_id"]}

    def certificate(pair, segment, path):
        h, rate = pair["history"], pair["raw"]["physical_rate"]
        when = float.fromhex(h["time_hex"])
        rb = {"point_identity": h["point_identity"], "rate_identity": rate["identity"], "path_identity": path,
              "map_identity": request["map_identity"], "binding_identity": digest(["segment-source", segment["id"]]),
              "time_hex": h["time_hex"], "event_side": "right" if when == segment["start"] else
              "left" if when == segment["end"] else "continuous", "state_changed": False,
              "native_source_provenance_claimed": False, "raw_z_words": binary(h["raw_solver_z_hex"]),
              "raw_zdot_words": binary(h["raw_solver_zdot_hex"]), "input_words": binary(h["inputs_hex"]),
              "input_rate_words": binary(h["input_rates_hex"]),
              "physical_rate_words": tuple(binary(w) for w in h["physical_rate_words_hex"]),
              **{k: (Fraction(0),) * size for k in
                 ("physical_state_absolute_error", "physical_rate_absolute_error", "state_action_error_bounds")},
              "physical_state_enclosures": (enclosure,) * size}
        rb.update(state_error_includes_actual_input_evaluation=True, do_not_add_same_input_error_twice=True)
        evidence = {"point_identity": h["point_identity"], "path_identity": path, "passed": True,
                    "native_rate_or_state_changed": False, "DAE_accuracy_certified": False, "readback": rb,
                    "reported_currents_A": all_currents, "raw_reference_A": current_enclosures,
                    "same_state_exact_tangent_reference_A": current_enclosures,
                    "source_and_readback_error_A": ((Fraction(0), Fraction(0)),) * 2,
                    "combined_current_bound_A": (Fraction(0), Fraction(0)),
                    "original_allocation_A": Fraction(1.0 / 3), "input_error_included_once": True}
        return {"kind": "voltage_lift_current_certificate", "sample_record_sha256": pair["record_sha256"],
                "evidence": tag(evidence)}

    for ordinal, segment in enumerate(segments, 1):
        owner = f"synthetic-owner-{ordinal}"
        proof = {**initial_fields(segment), "source_identity": digest(["segment-source", segment["id"]]),
                 "request_sha256": request_id, "map_identity": request["map_identity"],
                 "segment_sha256": digest(segment), "predecessor_identity": predecessor}
        proof["record_sha256"] = digest(proof)
        rows.append({"kind": "voltage_lift_initialization_input", "phase": "segment_initialization",
                     "segment_id": segment["id"], "time_hex": segment["start"].hex(), "proof": proof})

        def statistics(phase, when, steps, previous=None):
            raw = {k: steps if k == "num_steps" else 0 for k in counters}
            raw.update(current_time=when, observation_owner=owner, observation_generation=1)
            row = {"kind": "native_statistics", "segment_id": segment["id"], "logical_initialization_index": ordinal,
                   "phase": phase, "requested_time_hex": float(segment["start"] if phase == "initialization_return"
                                                              else segment["end"]).hex(),
                   "returned_time_hex": None if phase == "before_onestep" else float(when).hex(),
                   "internal_time_hex": float(when).hex(), "method_state_valid": steps > 0, "raw_statistics": raw}
            if previous is not None:
                row["work_since_before"] = {k: raw[k] - previous[k] for k in counters}
            rows.append(row)
            return raw

        now = segment["start"]
        statistics("initialization_return", now, 0)
        left = sample(segment, ordinal, now, "segment_initial", predecessor)
        rows.append(left)
        rows.append({"kind": "requested_sample", **ptr(left)})
        cursor = 1
        for step, then in enumerate(([0.5, 1.0] if ordinal == 1 else [2.0]), 1):
            before = statistics("before_onestep", now, step - 1)
            statistics("after_onestep", then, step, before)
            stamp = {"nsteps": step, "tn": then, "hused": then - now, "kused": 1,
                     "hh": then - now, "kk": 1,
                     "statuses": tuple((name, 0) for name in
                        ("IDAGetNumSteps", "IDAGetCurrentTime", "IDAGetLastStep", "IDAGetLastOrder",
                         "IDAGetCurrentStep", "IDAGetCurrentOrder"))}

            def native_point(t, nsteps, status):
                y, yp = binary(words(t)), binary(words(1))
                return {"owner": owner, "generation": 1, "nsteps": nsteps, "internal_t": t, "output_t": t,
                        "native_status": status, "raw_y": y, "raw_yp": yp,
                        "identity": (owner, 1, nsteps, t.hex(), t.hex(), hashlib.sha256(y + yp).hexdigest())}

            previous = native_point(now, step - 1, 0)
            endpoint = native_point(then, step, int(then == segment["end"]))
            packet = {"schema": source[0], "dtype": "<f8", "shape": (size,), "owner": owner, "generation": 1,
                      "binding": {"source": source, "identity": binding}, "native_before": stamp, "native_after": stamp,
                      "step_key": (binding, owner, 1, step, then.hex(), (then - now).hex(), 1, (then - now).hex(), 1),
                      "predecessor": previous, "endpoint": endpoint, "raw_y": endpoint["raw_y"], "raw_yp": endpoint["raw_yp"],
                      "native_left_terms": (then, -(then - now)), "clock_gap_terms": (then, -(then - now), -now),
                      "basis": {"kind": "ida75_phi_psi_copy_v1", "dtype": "<f8", "phi_shape": (2, size), "psi_shape": (1,),
                                "status": 0, "source_header_sha256": source[5], "source_config_sha256": source[6],
                                "build_identity": binding, "uround": 2**-53,
                                "phi": binary(words(0, 2 * size)), "psi": binary(words(then - now, 1))},
                      "predecessor_dky": {"query_t": now, "query_t_hex": now.hex(), "orders": (0, 1),
                                          "statuses": (0, 0), "native_eligible": True, "step_before": stamp, "step_after": stamp,
                                          "buffers": (previous["raw_y"], previous["raw_yp"])},
                      "dky": (endpoint["raw_y"], endpoint["raw_yp"]), "estimated_local_errors": binary(words(0)),
                      "error_weights": binary(words(1)), "vector_statuses": (("IDAGetDky[0]", 0), ("IDAGetDky[1]", 0),
                                          ("IDAGetErrWeights", 0), ("IDAGetEstLocalErrors", 0))}
            fid, path = frame_id(packet), digest(["synthetic-path", segment["id"], step])
            rows.append({"kind": "voltage_lift_native_observation_packet", "segment_id": segment["id"], "packet": tag(packet)})
            rows.append({"kind": "voltage_lift_polynomial_domain", "evidence": tag({"path_identity": path, "passed": True,
                         "checks": {}, "domain_bounds": {name: ((Fraction(1), Fraction(2)),) * 2
                                                         for name in request["numeric_packet"]["variable_offsets"]}})})
            rows.append(certificate(left, segment, path))
            right = sample(segment, ordinal, then, "stop_output" if endpoint["native_status"] == 1 else "native",
                           left["history"]["point_identity"])
            rows.extend((right, certificate(right, segment, path)))
            cache = {now: ptr(left), then: ptr(right)}

            def query(t):
                nonlocal queries
                if t not in cache:
                    pair = sample(segment, ordinal, t, "declared_polynomial", left["history"]["point_identity"], fid, path)
                    rows.extend((pair, certificate(pair, segment, path)))
                    cache[t] = ptr(pair)
                    queries += 1
                return cache[t]

            for order in (8, 16, 32):
                for node in request["quadrature"][str(order)]["nodes"]:
                    query(now + (then - now) * (float(node) + 1) / 2)
            times = request["observation_times"][segment["id"]]
            while cursor < len(times) and times[cursor] <= then:
                rows.append({"kind": "requested_sample", **query(float(times[cursor]))})
                cursor += 1
            error = (Fraction(1, 1000),) * 3
            prefix = (Fraction(total_intervals + 1, 1000),) * 3
            ledger = {**{k: (enclosure,) * 3 for k in ("saved_finite_charge_change_C", "nominal_change_C",
                                                      "signed_integral_C", "signed_saved_defect_C")},
                      "interval_total_bound_C": error, "prefix_absolute_defect_C": prefix,
                      "interval_reference_error_C": error, "prefix_reference_error_C": prefix,
                      "original_charge_budget_C": Fraction(1), "original_reference_share_C": Fraction(1.0 / 3),
                      "error_components_C": {k: error if k == "endpoint" else zeros for k in
                         ("endpoint", "input", "clock_strip", "raw_tangent_departure", "integration")},
                      "row_checks": (True,) * 3, "passed": True, "endpoint_debited_once": True}
            clock = {k: [Fraction(v).numerator, Fraction(v).denominator] for k, v in
                     (("predecessor", now), ("tn", then), ("hused", then - now),
                      ("segment_start", segment["start"]), ("segment_end", segment["end"]))}
            clock.update(owner=owner, generation=1, nsteps=step, kused=1, segment_id=segment["id"])
            clock.update(native_left=[Fraction(now).numerator, Fraction(now).denominator], uncovered_strip=[0, 1])
            observation = {"frame_identity": fid, "path_identity": path, "clock": clock, "passed": True,
                           "ledgers": {k: deepcopy(ledger) for k in ("raw_polynomial", "same_state_affine_tangent")}}
            observation["input_error"] = {"path_identity": path, "mapping_identity": request["map_identity"],
                                          "native_admitted": False, "endpoint_error_still_required": True,
                                          "native_readback_error_still_required": True,
                                          **{k: zeros for k in ("raw_charge_integral_error_C", "tangent_charge_integral_error_C")},
                                          "raw_tangent_L1_additional_error_C": zeros[:2]}
            rows.append({"kind": "voltage_lift_interval_charge", "segment_id": segment["id"],
                         "left": ptr(left), "right": ptr(right), "coefficient_frame_identity": fid,
                         "observation": tag(observation), "endpoint_restore": None, "normal_query_performed": False,
                         "states_projected": False, "reference_estimates_are_continuum_certificates": False,
                         "arithmetic_calls": [{"evaluations": 1, "actual_error": [0, 1], "requested_error": [1, 32]},
                                              {"evaluations": 1, "purpose": "full-real-interval-precheck", "working_bits": 512}]})
            total_intervals += 1
            left, now = right, then
        predecessor = left["history"]["point_identity"]
        pointer = ptr(left)
    result = {"status": "completed_bounded_voltage_lift_native_pilot", "complete_protocol": True,
              "request_sha256": request_id, "map_identity": request["map_identity"], "observation_policy_sha256": policy_id,
              "counts": {"native_steps": total_intervals, "onestep_returns": total_intervals, "initializations": len(segments),
                         "polynomial_queries": queries, "normal_queries": 0, "jacobian": 0, "residual": 0},
              "last_numerical": pointer, "last_physically_accepted": pointer,
              "last_attempt": {"phase": "native_return", "success": True, "segment_id": "b", "time_hex": 2.0.hex(),
                               "status": 1, "z_hex": words(2), "zdot_hex": words(1)},
              "cumulative_absolute_charge_bounds_C": {k: tag((Fraction(total_intervals, 1000),) * 3)
                                                     for k in ("raw_polynomial", "same_state_affine_tangent")},
              "cumulative_observation_reference_bounds_C": {k: tag((Fraction(total_intervals, 1000),) * 3)
                                                            for k in ("raw_polynomial", "same_state_affine_tangent")}}
    return request, rows, result


class NativeReadbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = fixture()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.request, self.rows, self.result = deepcopy(self.template)

    def publish(self, integer_io_policy=None):
        writer = HistoryWriter(self.folder / "NativeHistory.jsonl.gz", encoding="gzip", total_output_bytes=8 * 1048576,
                               integer_io_policy=integer_io_policy)
        for row in self.rows:
            writer.append(row)
        writer.finish()
        self.result["history"] = {"encoding": "gzip", "path": writer.path.name, "logical_bytes": writer.logical_bytes,
                                  "encoded_bytes": writer.encoded_bytes, "records": writer.records,
                                  "logical_sha256": writer.logical_sha256, "closed": True, "container_complete": True, "error": None}
        self.result["counts"].update(history_bytes=writer.logical_bytes, history_encoded_bytes=writer.encoded_bytes)
        if integer_io_policy is not None:
            observed = integer_io_observed(writer.integer_io_policy)
            self.result["history"]["integer_io"] = observed
            (self.folder / "IntegerIOObserved.json").write_text(json.dumps({"role": "producer", "pid": os.getpid(), **observed}))
        for name, value in (("NativeRequest.json", self.request), ("NativeResult.json", self.result)):
            (self.folder / name).write_text(integer_json_dumps(value, integer_io_policy, allow_nan=False))

    def verify(self, **limits):
        return verify_history(self.folder, **{"max_record_bytes": 1048576, "max_logical_bytes": 4 * 1048576,
                                             "max_records": 5000, "deadline_monotonic": time.perf_counter() + 5, **limits})

    def bad(self, code=None, **limits):
        with self.assertRaises(HistoryVerificationError) as ctx:
            self.verify(**limits)
        if code is not None:
            self.assertEqual(ctx.exception.code, code)
        self.assertLess(len(str(ctx.exception)), 240)
        return ctx.exception

    def test_complete_event_and_recertification_and_cached_quadrature(self):
        self.publish()
        receipt = self.verify()
        self.assertTrue(receipt["verified"])
        self.assertFalse(receipt["scientific_qualification"])
        self.assertEqual(receipt["coverage"]["intervals"], 3)
        self.assertEqual(receipt["coverage"]["record_counts"]["requested_sample"], 6)
        self.assertEqual(receipt["coverage"]["record_counts"]["voltage_lift_current_certificate"],
                         receipt["coverage"]["rate_pairs"] - 2 + 3)
        self.assertLess(self.result["counts"]["polynomial_queries"], 3 * 56 + 2)
        self.assertEqual(receipt["history"]["container_sha256"],
                         hashlib.sha256((self.folder / "NativeHistory.jsonl.gz").read_bytes()).hexdigest())
        self.assertTrue(receipt["history"]["crc_verified"])

    def test_integer_policy_full_reader_and_independent_application(self):
        policy = {"schema": "solarlab.native-integer-io.v1", "codec": "decimal-with-bounded-hex-v1",
                  "decimal_digits": 4300, "max_integer_bits": 131072}
        large = 10**4400 + 1
        row = next(row for row in self.rows if row["kind"] == "voltage_lift_interval_charge")
        row["io_fixture_extra"] = {"rational": [large, large+2]}
        self.result["io_fixture_extra"] = {"rational": [-large, large+2]}
        self.publish(integer_io_policy=policy)
        receipt = self.verify(integer_io_policy=policy)
        self.assertTrue(receipt["verified"])
        self.assertEqual(receipt["integer_io"], {"role": "independent_reader", "policy": policy,
                                                 "actual_decimal_digits": 4300})
        # A fresh reader process must itself configure and report the policy.
        code = """import json, sys, time
from pathlib import Path
from scripts.benchmarks.native_readback import verify_history
policy=json.loads(sys.argv[2])
r=verify_history(Path(sys.argv[1]), max_record_bytes=1048576, max_logical_bytes=4*1048576,
                 max_records=5000, deadline_monotonic=time.perf_counter()+5, integer_io_policy=policy)
print(json.dumps(r['integer_io']))
"""
        other = subprocess.run([sys.executable, "-B", "-c", code, str(self.folder), json.dumps(policy)],
                               capture_output=True, text=True, timeout=10)
        self.assertEqual(other.returncode, 0, other.stderr)
        self.assertEqual(json.loads(other.stdout), receipt["integer_io"])

    def test_integer_policy_mismatch_and_forged_producer_observation(self):
        policy = {"schema": "solarlab.native-integer-io.v1", "codec": "decimal-with-bounded-hex-v1",
                  "decimal_digits": 4300, "max_integer_bits": 131072}
        self.publish(integer_io_policy=policy)
        self.bad("integer_io_policy_or_applied_limit_mismatch")
        self.bad("integer_io_policy_or_applied_limit_mismatch", integer_io_policy={**policy, "max_integer_bits": 65536})
        path = self.folder / "IntegerIOObserved.json"
        forged = json.loads(path.read_text())
        forged["actual_decimal_digits"] = 0
        path.write_text(json.dumps(forged))
        self.bad("producer_integer_io_observation_mismatch", integer_io_policy=policy)

    def test_import_does_not_load_numerical_backends(self):
        # Other numerical tests may already have imported these modules in the
        # host runner. Only a fresh interpreter can test this import boundary.
        code = """import json, sys
sys.path.insert(0, sys.argv[1])
from scripts.benchmarks import native_readback
names = ('numpy', 'scipy', 'flint', 'sksundae', 'scripts.benchmarks.native_observation',
         'scripts.benchmarks.coupled_device_prototype')
print(json.dumps([name for name in names if name in sys.modules]))
"""
        result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(Path(__file__).resolve().parents[2])],
                                check=True, capture_output=True, text=True, timeout=3)
        self.assertEqual(json.loads(result.stdout), [])

    def test_record_logical_and_count_limits(self):
        self.publish()
        for limits in ({"max_record_bytes": 100}, {"max_logical_bytes": 20000}, {"max_records": 9}):
            with self.subTest(limits=limits):
                self.bad(**limits)

    def test_deadline_checked_during_read(self):
        self.publish()
        self.bad("deadline_exceeded", deadline_monotonic=0.0)
        clock = iter([0.0] * 35 + [20.0] * 5000)
        with patch("scripts.benchmarks.native_readback.time.perf_counter", side_effect=lambda: next(clock)):
            self.bad("deadline_exceeded", deadline_monotonic=10.0)

    def test_missing_low_word_or_layer_is_rejected_even_with_new_record_digest(self):
        for layer in (False, True):
            with self.subTest(layer=layer):
                self.rows = deepcopy(self.template[1])
                pair = next(row for row in self.rows if row["kind"] == "voltage_lift_rate_pair")
                if layer:
                    pair["history"]["physical_cumulative_words_hex"].pop()
                else:
                    pair["raw"]["total_inward"].pop("low_hex")
                pair["record_sha256"] = digest({k: v for k, v in pair.items() if k != "record_sha256"})
                with tempfile.TemporaryDirectory() as temp:
                    self.folder = Path(temp)
                    self.publish()
                    self.bad("vector_shape" if layer else "double_word_shape")

    def test_unknown_reordered_and_missing_records_fail(self):
        for change in ("unknown", "swap", "remove_certificate", "remove_requested"):
            with self.subTest(change=change):
                self.rows = deepcopy(self.template[1])
                i = next(i for i, row in enumerate(self.rows) if row["kind"] == "voltage_lift_current_certificate")
                if change == "unknown":
                    self.rows[i] = {"kind": "unrecognized"}
                elif change == "swap":
                    self.rows[i], self.rows[i + 1] = self.rows[i + 1], self.rows[i]
                elif change == "remove_requested":
                    self.rows.pop(next(i for i, r in enumerate(self.rows) if r["kind"] == "requested_sample"))
                else:
                    self.rows.pop(i)
                with tempfile.TemporaryDirectory() as temp:
                    self.folder = Path(temp)
                    self.publish()
                    self.bad()

    def test_packet_and_certificate_word_corruption(self):
        packet = next(row for row in self.rows if row["kind"] == "voltage_lift_native_observation_packet")
        packet["packet"]["endpoint"]["raw_yp"]["bytes_hex"] = binary(words(2)).hex()
        self.publish()
        self.bad("native_endpoint_identity")

    def test_current_low_word_changes_cannot_hide_behind_passed(self):
        cert = next(row for row in self.rows if row["kind"] == "voltage_lift_current_certificate")
        cert["evidence"]["reported_currents_A"]["tuple"][0]["tuple"][0] = {"rational": [1, 1]}
        self.publish()
        self.bad("current_reported_high_low")

    def test_ambiguous_tag_and_nonfinite_hex_fail(self):
        for ambiguity in (True, False):
            self.rows = deepcopy(self.template[1])
            cert = next(row for row in self.rows if row["kind"] == "voltage_lift_current_certificate")
            cert["extra"] = {"binary64": "inf", **({"another": 1} if ambiguity else {})}
            with tempfile.TemporaryDirectory() as temp:
                self.folder = Path(temp)
                self.publish()
                self.bad("ambiguous_observation_tag" if ambiguity else "noncanonical_or_nonfinite_hex")

    def test_failed_result_is_not_complete_even_with_valid_gzip(self):
        self.result["complete_protocol"] = False
        self.publish()
        self.bad("incomplete_native_result")

    def test_sidecar_and_result_prefix_binding(self):
        self.publish()
        p = self.folder / "LastAccepted.json"
        value = json.loads(p.read_text())
        value["right"]["time_s"] = 1.0
        p.write_text(json.dumps(value))
        self.bad("final_sidecar_binding")

    def test_charge_prefix_cannot_reset_at_event(self):
        interval = self.rows[-1]
        ledger = interval["observation"]["ledgers"]["raw_polynomial"]
        ledger["prefix_absolute_defect_C"] = tag((Fraction(1, 1000),) * 3)
        self.publish()
        self.bad("charge_prefix_reset_or_error")

    def test_exact_third_and_encoded_binary64_share_are_distinct(self):
        interval = next(row for row in self.rows if row["kind"] == "voltage_lift_interval_charge")
        interval["observation"]["ledgers"]["raw_polynomial"]["original_reference_share_C"] = tag(Fraction(1, 3))
        self.publish()
        self.bad("charge_budget_binding")

    def test_recorded_passed_cannot_hide_failed_state_gate(self):
        sample = next(row for row in self.rows if row["kind"] == "voltage_lift_rate_pair")
        sample["state_checks"]["constraint_potential_bound_V"] = 2.0
        sample["record_sha256"] = digest({k: v for k, v in sample.items() if k != "record_sha256"})
        self.publish()
        self.bad("state_gate_values")

    def test_native_owner_cannot_change_inside_segment(self):
        stats = next(row for row in self.rows if row["kind"] == "native_statistics" and row["phase"] == "after_onestep")
        stats["raw_statistics"]["observation_owner"] = "another-owner"
        self.publish()
        self.bad("native_owner_changed_within_segment")

    def test_restart_new_owner_must_begin_at_generation_one(self):
        for row in self.rows:
            if row.get("segment_id") == "b" and row["kind"] == "native_statistics":
                row["raw_statistics"]["observation_generation"] = 2
        packet = next(row["packet"] for row in self.rows if row["kind"] == "voltage_lift_native_observation_packet"
                      and row["segment_id"] == "b")
        packet["generation"] = 2
        packet["step_key"]["tuple"][2] = 2
        for name in ("predecessor", "endpoint"):
            packet[name]["generation"] = 2
            packet[name]["identity"]["tuple"][1] = 2
        self.publish()
        self.bad("native_restart_epoch")

    def test_uncovered_strip_requires_eligible_native_predecessor_getters(self):
        packet = next(row["packet"] for row in self.rows if row["kind"] == "voltage_lift_native_observation_packet")
        for stamp in (packet["native_before"], packet["native_after"],
                      packet["predecessor_dky"]["step_before"], packet["predecessor_dky"]["step_after"]):
            stamp["hused"] = tag(0.25)
        packet["step_key"]["tuple"][5] = 0.25.hex()
        packet["native_left_terms"]["tuple"][1] = tag(-0.25)
        packet["clock_gap_terms"]["tuple"][1] = tag(-0.25)
        packet["predecessor_dky"].update(statuses=tag((-25, -25)), buffers=tag((None, None)), native_eligible=False)
        self.publish()
        self.bad("native_strip_getter_eligibility")

    def test_truncated_footer_crc_and_trailing_junk(self):
        self.publish()
        path = self.folder / "NativeHistory.jsonl.gz"
        original = path.read_bytes()
        for data in (original[:-1], original[:-8] + bytes([original[-8] ^ 1]) + original[-7:], original + b"junk"):
            with self.subTest(length=len(data)):
                path.write_bytes(data)
                self.bad()

    def test_strict_json_and_partial_tail(self):
        self.publish()
        # A tiny manually encoded first record exercises strict parsing before
        # later schema checks; no producer/model is invoked to generate it.
        path = self.folder / "NativeHistory.jsonl.gz"
        for raw, code in ((b'{"kind":"x","kind":"x"}\n', "duplicate_json_key"),
                          (b'{"kind":"x","value":NaN}\n', "nonfinite_json_constant"),
                          (b'{"kind":"x","value":1e999}\n', "nonfinite_json_number"),
                          (b'{"kind":"x"}', "partial_or_multiline_record")):
            with self.subTest(code=code):
                compressor = zlib.compressobj(wbits=31)
                encoded = compressor.compress(raw) + compressor.flush()
                path.write_bytes(encoded)
                self.result["history"]["encoded_bytes"] = len(encoded)
                (self.folder / "NativeResult.json").write_text(json.dumps(self.result))
                self.bad(code)



# NativeAttempt06 S0, first complete interval, record126 at logical3260611.
# Historical saved data only: no model/solver is constructed by these checks.
# Exact record SHA b937e7d2d41d06b4f4cbad923def8a3ed5ea46d6c693b3b3a4923f0886228498.
# The bundle retains its actual request, native packet, domain and endpoint rows.
_NATIVE06_BUNDLE_SHA256 = '3aa5d6a789f48144fe68a78aaacbfde04717955eee0775451579475f3ce7a2b8'


def saved_native_interval():
    raw = (Path(__file__).with_name("fixtures") / "native_interval_charge.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == _NATIVE06_BUNDLE_SHA256
    data = json.loads(raw)
    interval_line = (json.dumps(data["interval"], separators=(",", ":"), allow_nan=False) + "\n").encode()
    assert hashlib.sha256(interval_line).hexdigest() == "b937e7d2d41d06b4f4cbad923def8a3ed5ea46d6c693b3b3a4923f0886228498"
    return data


def verify_saved_interval(data):
    class SavedRows:
        def take(self, kind):
            if data["interval"]["kind"] != kind:
                raise AssertionError("saved record kind")
            return deepcopy(data["interval"])

    verifier = _Protocol(data["request"], SavedRows())
    verifier.segment = data["request"]["segments"][0]
    verifier.path_id = data["domain"]["evidence"]["path_identity"]
    frame, verifier.frame_id = _native_packet(data["packet"], verifier.policy, verifier.size)
    verifier._interval(data["left"], data["right"], frame)
    return verifier


class RecordedNativeIntervalTests(unittest.TestCase):
    def test_real_complete_interval_and_adjacent_arithmetic_receipts(self):
        result = verify_saved_interval(saved_native_interval())
        self.assertEqual(result.intervals, 1)

    def test_real_input_error_dimensions_remain_strict(self):
        for key in ("raw_charge_integral_error_C", "tangent_charge_integral_error_C", "raw_tangent_L1_additional_error_C"):
            for change in ("missing", "extra"):
                with self.subTest(field=key, change=change):
                    data = saved_native_interval()
                    values = data["interval"]["observation"]["input_error"][key]["tuple"]
                    if change == "missing":
                        values.pop()
                    else:
                        values.append({"rational": [0, 1]})
                    with self.assertRaisesRegex(HistoryVerificationError, "vector_shape"):
                        verify_saved_interval(data)

    def test_real_input_errors_reject_negative_values(self):
        for key in ("raw_charge_integral_error_C", "tangent_charge_integral_error_C", "raw_tangent_L1_additional_error_C"):
            with self.subTest(field=key):
                data = saved_native_interval()
                data["interval"]["observation"]["input_error"][key]["tuple"][0] = {"rational": [-1, 1]}
                with self.assertRaisesRegex(HistoryVerificationError, "nonnegative_rational_vector"):
                    verify_saved_interval(data)

    def test_real_precheck_receipt_is_not_an_integral_error_estimate(self):
        for field, value in (("working_bits", 256), ("evaluations", True), ("evaluations", 0), ("actual_error", [0, 1])):
            with self.subTest(field=field, value=value):
                data = saved_native_interval()
                call = next(row for row in data["interval"]["arithmetic_calls"] if row.get("purpose") == "full-real-interval-precheck")
                call[field] = value
                with self.assertRaisesRegex(HistoryVerificationError, "arithmetic_precheck_receipt"):
                    verify_saved_interval(data)

    def test_real_integral_error_budget_is_still_enforced(self):
        data = saved_native_interval()
        call = next(row for row in data["interval"]["arithmetic_calls"] if "actual_error" in row)
        bound = Fraction(*call["requested_error"])
        call["actual_error"] = [2 * bound.numerator, bound.denominator]
        with self.assertRaisesRegex(HistoryVerificationError, "arithmetic_receipt_error"):
            verify_saved_interval(data)

def _untag_fixture(value):
    """Test-only inverse for invented records; no producer or numerical imports."""
    if isinstance(value, list):
        return [_untag_fixture(v) for v in value]
    if isinstance(value, dict):
        if set(value) == {"rational"}:
            return Fraction(*value["rational"])
        if set(value) == {"tuple"}:
            return tuple(_untag_fixture(v) for v in value["tuple"])
        if set(value) == {"binary64"}:
            return float.fromhex(value["binary64"])
        if set(value) == {"bytes_hex"}:
            return bytes.fromhex(value["bytes_hex"])
        return {k: _untag_fixture(v) for k, v in value.items()}
    return value


def _partition_fixture_policy(request, policy):
    """Set the synthetic policy before the original fixture hashes its rows."""
    request["budgets"]["native_steps"] = 200000
    ledgers = ("raw_polynomial", "same_state_affine_tangent")
    channels = tuple(f"{k}.{r}.{v}" for k in ledgers for r in ("device", "left_metal", "right_metal")
                     for v in ("total", "reference"))
    quantum, maximum_slack = Fraction(1, 2**120), Fraction(200000, 2**120)
    base = {**policy, "schema": "solarlab.interval-observation-policy.v1",
            "budgets_sha256": digest(request["budgets"]), "backend_modules": {},
            "arithmetic": {"python_flint": "0.8.0", "bits": 256, "evaluations": 2048, "depth": 16},
            "quadrature_error_allocation": "charge_C/12 * actual_interval/full_protocol / 32 per integral",
            "sampling": "original requested and 8/16/32 times, explicitly declared polynomial reconstruction",
            "history": "actual accepted predecessor/endpoint plus source-owned immutable phi/psi/Dky and error metadata",
            "qualification": "represented numerical path only; original independent time/state/space refinements remain pending"}
    constants = {"schema": "solarlab.nonnegative-upper-sum.v1", "quantum_bits": 120, "max_terms": 200000,
                 "max_increment_bits": 65536, "max_evidence_bytes": 8*1024*1024,
                 "max_slack_per_channel": [maximum_slack.numerator, maximum_slack.denominator], "channel_ids": list(channels)}
    return {**base, "schema": "solarlab.interval-observation-policy.v2",
        "parent_policy_sha256": digest(base), "charge_refinement": {"name": "fixed4-upper120-v1", "partition_cells": 4,
            "upper_sum": constants, "prefix_semantics": "exact signed local evidence; non-resetting upward dyadic upper bounds; original gates"}}


def partition_fixture(*, zero=False):
    """Synthetic constant-function certificates; no physical/native calls.

    Policy, moments and prefixes do not use producer or reader check helpers.
    """
    request, rows, result = fixture(_partition_fixture_policy)
    request_id = digest(request)
    ledgers = ("raw_polynomial", "same_state_affine_tangent")
    channels = tuple(f"{k}.{r}.{v}" for k in ledgers for r in ("device", "left_metal", "right_metal")
                     for v in ("total", "reference"))
    quantum, maximum_slack = Fraction(1, 2**120), Fraction(200000, 2**120)
    constants = request["interval_observation"]["charge_refinement"]["upper_sum"]
    upper_policy = digest({"schema": constants["schema"], "source_identity": request_id,
        "channels": [{"id": c, "unit": "C", "increment": "interval_total_bound_C" if c.endswith(".total")
                      else "interval_reference_error_C", "role": "nonnegative_upper_bound"} for c in channels],
        "quantum_bits": 120, "max_terms": 200000,
        "max_slack_per_channel_hex": [hex(maximum_slack.numerator), hex(maximum_slack.denominator)],
        "max_increment_bits": 65536, "max_evidence_bytes": 8*1024*1024})
    local_fields = ("saved_finite_charge_change_C", "nominal_change_C", "signed_integral_C", "error_components_C",
                    "signed_saved_defect_C", "interval_total_bound_C", "interval_reference_error_C",
                    "original_charge_budget_C", "original_reference_share_C")
    numbers, events, index = [0]*12, [0]*12, 0
    for row in rows:
        if row["kind"] != "voltage_lift_interval_charge":
            continue
        row["kind"] = "voltage_lift_interval_charge_v2"
        obs = _untag_fixture(row["observation"])
        before, now, h = (Fraction(*obs["clock"][k]) for k in ("predecessor", "tn", "hused"))
        left = (before-now)/h
        allocation = Fraction(request["budgets"]["charge_C"])/12*(now-before)/Fraction(request["segments"][-1]["end"])/32
        calls = [{"evaluations": 1, "actual_error": [0, 1], "requested_error": [1, 32]}]

        def constant_bound(a, b, epsilon):
            value = Fraction(0) if zero else Fraction(1, 100000)
            moment, upper = (b-a)*value**2, (b-a)*value
            start = len(calls)
            calls.append({"evaluations": 1, "purpose": "full-real-interval-precheck", "working_bits": 512})
            if not zero:
                goal = epsilon**2/(8*(b-a))
                calls.append({"evaluations": 1, "actual_error": [0, 1], "requested_error": [goal.numerator, goal.denominator],
                              "purpose": "L2-moment", "working_bits": 512, "absolute_integrand": False})
            excess = Fraction(0) if zero else epsilon
            return {"upper": [upper.numerator, upper.denominator], "method": "Cauchy-Schwarz from a certified squared integral",
                    "squared_integral": {"center": [moment.numerator, moment.denominator], "absolute_error_bound": [0, 1]},
                    "numerical_excess_bound": [excess.numerator, excess.denominator], "is_integral_value_estimate": False}, (start, len(calls))

        ports = []
        for port in ("left_metal", "right_metal"):
            original, original_calls = constant_bound(left, Fraction(0), allocation)
            cells = []
            for j in range(4):
                a, b = left*(1-Fraction(j, 4)), left*(1-Fraction(j+1, 4))
                bound, call_range = constant_bound(a, b, allocation/4)
                cells.append({"left": a, "right": b, "allocation_C": allocation/4, "bound": bound, "call_range": call_range})
            total = sum((Fraction(*c["bound"]["upper"]) for c in cells), Fraction(0))
            ports.append({"port": port, "integrand_identity": digest({"schema": "solarlab.terminal-departure.v1",
                "path_identity": obs["path_identity"], "port": port,
                "formula": "hused*(endpoint_carrier_rate+raw_metal-tangent_metal_polynomial-tangent_metal_flux)"}),
                "absolute_error_C": allocation, "original": original, "original_call_range": original_calls,
                "cells": tuple(cells), "partition_upper_C": total, "retained_upper_C": min(Fraction(*original["upper"]), total)})
        carrier, body = (Fraction(1, 10**8), Fraction(2, 10**8)), Fraction(3, 10**8)
        obs["terminal_partition"] = {"schema": "solarlab.terminal-partition.v1", "path_identity": obs["path_identity"],
            "normalized_interval": (left, Fraction(0)), "cells_per_port": 4, "ports": tuple(ports),
            "endpoint_carrier_L1_upper_C": carrier, "body_L1_upper_C": body}
        obs["raw_tangent_L1_current_bounds"] = tuple(p["original"] for p in ports)
        raw_ledger = obs["ledgers"]["raw_polynomial"]
        raw_ledger["error_components_C"]["raw_tangent_departure"] = (body, *(p["retained_upper_C"]+c for p,c in zip(ports,carrier)))
        for k in ledgers:
            ledger = obs["ledgers"][k]
            local = tuple(sum((v[i] for v in ledger["error_components_C"].values()), Fraction(0)) for i in range(3))
            ledger["interval_total_bound_C"] = ledger["interval_reference_error_C"] = local
        index += 1
        for j, value in enumerate(v for k in ledgers for i in range(3)
                                  for v in (obs["ledgers"][k]["interval_total_bound_C"][i], obs["ledgers"][k]["interval_reference_error_C"][i])):
            exact = numbers[j]*quantum+value
            ratio = exact/quantum
            n = -(-ratio.numerator//ratio.denominator)
            events[j] += int(n*quantum > exact)
            numbers[j] = n
        for j, k in enumerate(ledgers):
            obs["ledgers"][k]["prefix_absolute_defect_C"] = tuple(n*quantum for n in numbers[j*6:j*6+6:2])
            obs["ledgers"][k]["prefix_reference_error_C"] = tuple(n*quantum for n in numbers[j*6+1:j*6+6:2])
        local = {"schema": "solarlab.native-charge-local-evidence.v1", "source_identity": request_id,
                 "frame_identity": obs["frame_identity"], "path_identity": obs["path_identity"], "clock": obs["clock"],
                 "ledgers": {k: {f: obs["ledgers"][k][f] for f in local_fields} for k in ledgers}}
        local_bytes = json.dumps(encode_integer_values(tag(local), {"max_integer_bits": 65536}),
                                sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        summary = {"schema": "solarlab.native-upper-prefix-state.v1", "policy_sha256": upper_policy,
                   "source_identity": request_id, "channel_ids": channels, "index": index,
                   "upper_numerators": tuple(numbers), "rounded_terms": tuple(events),
                   "slack_upper_bounds": tuple(n*quantum for n in events)}
        obs["upper_sum"] = {**summary, "schema": "solarlab.nonnegative-upper-step.v1",
                            "interval_sha256": hashlib.sha256(local_bytes).hexdigest()}
        row["observation"], row["arithmetic_calls"] = tag(obs), calls
    for j, key in enumerate(ledgers):
        result["cumulative_absolute_charge_bounds_C"][key] = tag(tuple(n*quantum for n in numbers[j*6:j*6+6:2]))
        result["cumulative_observation_reference_bounds_C"][key] = tag(tuple(n*quantum for n in numbers[j*6+1:j*6+6:2]))
    result["cumulative_representation"] = tag(summary)
    return request, rows, result


class PartitionReadbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = partition_fixture()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.case_number = 0
        self.reset()

    def reset(self, template=None):
        self.case_number += 1
        self.folder = Path(self.temp.name)/str(self.case_number)
        self.folder.mkdir()
        self.request, self.rows, self.result = deepcopy(self.template if template is None else template)

    publish = NativeReadbackTests.publish
    verify = NativeReadbackTests.verify
    bad = NativeReadbackTests.bad

    def interval(self, index=0):
        return [r for r in self.rows if r["kind"].startswith("voltage_lift_interval_charge")][index]

    def mutate_observation(self, change, index=0):
        row = self.interval(index)
        obs = _untag_fixture(row["observation"])
        change(obs)
        row["observation"] = tag(obs)

    def test_complete_partition_and_nonresetting_twelve_prefixes(self):
        self.publish()
        receipt = self.verify()
        self.assertEqual(receipt["charge_refinement"]["intervals"], 3)
        self.assertEqual(receipt["charge_refinement"]["channels"], 12)
        self.assertEqual(receipt["charge_refinement"]["rounded_terms"], [3]*12)

    def test_exact_zero_range_shortcut(self):
        self.reset(partition_fixture(zero=True))
        self.publish()
        self.assertTrue(self.verify()["verified"])

    def test_legacy_and_v2_record_kinds_cannot_mix(self):
        self.interval()["kind"] = "voltage_lift_interval_charge"
        self.publish()
        self.bad("record_order_expected_voltage_lift_interval_charge_v2")
        self.reset(fixture())
        self.interval()["kind"] = "voltage_lift_interval_charge_v2"
        self.publish()
        self.bad("record_order_expected_voltage_lift_interval_charge")

    def test_frozen_parent_and_refinement_constants(self):
        for field in ("parent", "quantum", "channels", "max_terms", "quantum_float", "precision_float"):
            with self.subTest(field=field):
                self.reset()
                policy = self.request["interval_observation"]
                if field == "parent":
                    policy["parent_policy_sha256"] = "0"*64
                elif field == "quantum_float":
                    policy["charge_refinement"]["upper_sum"]["quantum_bits"] = 120.0
                elif field == "precision_float":
                    policy["arithmetic"]["bits"] = 256.0
                    base = {k: v for k, v in policy.items() if k not in ("parent_policy_sha256", "charge_refinement")}
                    base["schema"] = "solarlab.interval-observation-policy.v1"
                    policy["parent_policy_sha256"] = digest(base)
                elif field == "channels":
                    policy["charge_refinement"]["upper_sum"]["channel_ids"].reverse()
                else:
                    policy["charge_refinement"]["upper_sum"]["quantum_bits" if field == "quantum" else "max_terms"] -= 1
                self.publish()
                self.bad("charge_refinement_parent_policy" if field == "precision_float" else
                         "charge_refinement_constants" if field == "quantum_float" else None)

    def test_working_precision_is_an_integer_schema_field(self):
        row = self.interval()
        port = _untag_fixture(row["observation"])["terminal_partition"]["ports"][0]
        row["arithmetic_calls"][port["original_call_range"][0]+1]["working_bits"] = 512.0
        self.publish()
        self.bad("partition_moment_radius")

    def test_cells_cannot_be_missing_reordered_or_wrong_clock(self):
        for change in ("missing", "clock", "overlap", "port"):
            with self.subTest(change=change):
                self.reset()
                def corrupt(obs):
                    part = obs["terminal_partition"]
                    port = part["ports"][0]
                    if change == "missing":
                        port["cells"] = port["cells"][:-1]
                    elif change == "clock":
                        port["cells"][0]["left"] += Fraction(1, 100)
                    elif change == "overlap":
                        port["cells"][1]["call_range"] = port["cells"][0]["call_range"]
                    else:
                        part["ports"] = tuple(reversed(part["ports"]))
                self.mutate_observation(corrupt)
                self.publish()
                self.bad()

    def test_outward_root_radius_and_minimum_are_independent_checks(self):
        for change in ("root", "radius", "minimum", "allocation"):
            with self.subTest(change=change):
                self.reset()
                def corrupt(obs):
                    port = obs["terminal_partition"]["ports"][0]
                    cell = port["cells"][0]
                    if change == "root":
                        cell["bound"]["upper"] = [0, 1]
                    elif change == "radius":
                        cell["bound"]["squared_integral"]["absolute_error_bound"] = [1, 1]
                    elif change == "minimum":
                        port["retained_upper_C"] = Fraction(0)
                    else:
                        cell["allocation_C"] *= 2
                self.mutate_observation(corrupt)
                self.publish()
                self.bad()

    def test_endpoint_carrier_debit_cannot_be_dropped(self):
        self.mutate_observation(lambda o: o["terminal_partition"].update(endpoint_carrier_L1_upper_C=(Fraction(0),)*2))
        self.publish()
        self.bad("partition_charge_debits")

    def test_prefix_understatement_extra_quantum_reset_and_slack(self):
        for change in ("under", "over", "reset", "slack", "count", "source"):
            with self.subTest(change=change):
                self.reset()
                def corrupt(obs):
                    cert = obs["upper_sum"]
                    if change in ("under", "over", "reset"):
                        values = list(cert["upper_numerators"])
                        values[0] = 0 if change == "reset" else values[0]+(-1 if change == "under" else 1)
                        cert["upper_numerators"] = tuple(values)
                    elif change == "slack":
                        cert["slack_upper_bounds"] = (Fraction(0),)*12
                    elif change == "count":
                        cert["index"] = 1
                    else:
                        cert["source_identity"] = "0"*64
                self.mutate_observation(corrupt, index=2)
                self.publish()
                self.bad()

    def test_signed_evidence_digest_and_final_representation(self):
        self.mutate_observation(lambda o: o["upper_sum"].update(interval_sha256="0"*64))
        self.publish()
        self.bad("charge_upper_local_evidence")
        self.reset()
        summary = _untag_fixture(self.result["cumulative_representation"])
        summary["rounded_terms"] = (0,)*12
        self.result["cumulative_representation"] = tag(summary)
        self.publish()
        self.bad("result_charge_representation")

    def test_new_representation_does_not_accept_incomplete_or_failed_result(self):
        for key, value in (("complete_protocol", False), ("first_failure", {"reason": "original gate"})):
            with self.subTest(key=key):
                self.reset()
                self.result[key] = value
                self.publish()
                self.bad("incomplete_native_result")


if __name__ == "__main__":
    unittest.main()
