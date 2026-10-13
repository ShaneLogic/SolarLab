"""Metadata-only P02 tests; every numeric packet here is a synthetic layout."""
import copy
from fractions import Fraction
import json
import math
from pathlib import Path
import unittest

from scripts.benchmarks import qualification_policy as q

FIXTURES = Path(__file__).with_name("fixtures")
ORIGINS = json.loads((FIXTURES / "qualification_origins.json").read_text())
GRIDS = json.loads((FIXTURES / "qualification_grids.json").read_text())


def packet(policy):
    """A unit fixture, not a constructed or solved physical device."""
    variables, rows = q.layout_offsets(policy)
    definition = policy["origin"]["definition"]
    x = GRIDS[str(policy["intervals"])]["x_m"]
    edges = [0.0] + [(a+b)/2 for a, b in zip(x[:-1], x[1:])] + [definition["length"]]
    scales = [None]*policy["coordinates"]
    for name, (lo, hi) in variables.items():
        scales[lo:hi] = [float.fromhex(policy["origin"]["field_denominator_ceilings"][name]["scale_hex"])]*(hi-lo)
    return {"definition": definition, "intervals": policy["intervals"], "nodes": policy["nodes"],
            "variable_offsets": variables, "equation_offsets": rows, "x_m": x,
            "cell_edges_m": edges, "volumes_m3": [definition["area"]*(b-a) for a, b in zip(edges[:-1], edges[1:])],
            "face_areas_m2": [definition["area"]]*policy["intervals"],
            "column_scaling": scales, "row_scaling": [1.0]*policy["coordinates"]}


class QualificationMetadataTests(unittest.TestCase):
    def test_all_original_fourteen_axes_retain_current_denominator_origin(self):
        cases = json.loads((FIXTURES / "qualification_cases.json").read_text())
        self.assertEqual(len(cases), 14)
        for case in cases:
            with self.subTest(case=case["id"]):
                policy = q.make_policy(ORIGINS[case["label"]], case["intervals"], case["axis"])
                self.assertEqual(policy["coordinates"], case["expected_coordinate_count"])
                self.assertEqual(policy["plan_basis"], {"rtol": case["basis_rtol"], "component_atol": case["basis_component_atol"]})
                self.assertEqual(policy["max_step"], case["max_step_s"])
                fixture = packet(policy)
                controls = q.raw_controls(policy, fixture)
                factor = Fraction(*policy["tolerance_factor"])*policy["pre_time_multiplier"]
                self.assertLessEqual(Fraction(controls["rtol"]), Fraction(float.fromhex(policy["origin"]["applied_rtol_hex"]))*factor)
                self.assertNotEqual(controls["rtol"], case["basis_rtol"])
                for name, (lo, hi) in fixture["variable_offsets"].items():
                    ceiling = Fraction(policy["origin"]["field_denominator_ceilings"][name]["applied_atol_SI_exact"])*factor
                    for index in range(lo, hi):
                        self.assertLessEqual(Fraction(controls["atol"][index])*Fraction(fixture["column_scaling"][index]), ceiling)
                self.assertFalse(policy["native_execution_authorized"])

    def test_no_arbitrary_mesh_axis_or_n8_reinterpretation(self):
        for mesh, axis in [(8, "base"), (128, "base"), (17, "base"), (True, "base"),
                           (32, "base"), (16, "mesh32"), (64, "joint_finest"), (16, None)]:
            with self.subTest(mesh=mesh, axis=axis), self.assertRaises(ValueError):
                q.make_policy(ORIGINS["B"], mesh, axis)

    def test_ancestor_and_original_science_cannot_be_relabelled(self):
        for mutate in [lambda d: d.update(baseline_request_sha256="0"*64),
                       lambda d: d["budgets"].update(charge_C=1.0),
                       lambda d: d["definition"].update(temperature=301.0),
                       lambda d: d.update(applied_rtol_hex=0.0001.hex())]:
            origin = copy.deepcopy(ORIGINS["B"]); mutate(origin)
            with self.assertRaisesRegex(ValueError, "unbound_origin"):
                q.make_policy(origin, 16, "base")

    def test_policy_tampering_and_extra_fields_reject(self):
        good = q.make_policy(ORIGINS["B"], 16, "base")
        for field, value in [("coordinates", 84), ("tolerance_factor", [2, 1]),
                             ("max_step", 0.1), ("native_execution_authorized", True),
                             ("unbound_override", {})]:
            bad = copy.deepcopy(good); bad[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "policy_binding"):
                q.validate_policy(bad)

    def test_layout_and_geometric_weight_negatives(self):
        policy = q.make_policy(ORIGINS["B"], 16, "base")
        good = packet(policy)
        mutations = [lambda d: d["variable_offsets"].update(f=[0, 17]),
                     lambda d: d["equation_offsets"].update(z_n_contact=[0, 2]),
                     lambda d: d.update(nodes=16),
                     lambda d: d["volumes_m3"].__setitem__(0, d["volumes_m3"][0]*2),
                     lambda d: d["column_scaling"].__setitem__(0, d["column_scaling"][0]*2),
                     lambda d: d["row_scaling"].__setitem__(0, math.nan),
                     lambda d: d["x_m"].__setitem__(1, math.nextafter(d["x_m"][1], math.inf)),
                     lambda d: d["face_areas_m2"].pop()]
        for mutate in mutations:
            bad = copy.deepcopy(good); mutate(bad)
            with self.assertRaises(ValueError):
                q.validate_layout(bad, policy)

    def test_tolerance_axes_are_strict_and_step_axes_do_not_change_weights(self):
        base = q.make_policy(ORIGINS["B"], 16, "base")
        a = q.raw_controls(base, packet(base))
        for axis in ["step_middle", "step_tight"]:
            policy = q.make_policy(ORIGINS["B"], 16, axis)
            b = q.raw_controls(policy, packet(policy))
            self.assertEqual({k: v for k, v in a.items() if k != "max_step"},
                             {k: v for k, v in b.items() if k != "max_step"})
        for axis, divisor in [("tolerance_middle", 10), ("tolerance_tight", 100)]:
            policy = q.make_policy(ORIGINS["B"], 16, axis)
            b = q.raw_controls(policy, packet(policy))
            self.assertLessEqual(Fraction(b["rtol"]), Fraction(a["rtol"])/divisor)
            self.assertTrue(all(Fraction(x) <= Fraction(y)/divisor for x, y in zip(b["atol"], a["atol"], strict=True)))



def _initialization_metadata(*, framed, count=4):
    """Manufactured two-coordinate metadata, never a device or trajectory."""
    size, zero = 2, [0.0.hex()]*2
    first = {"id": "prepared", "start": 0.0, "end": 1.0,
             "voltage": [0.5, 1.0], "photons": [2.0, 2.5]}
    base = {"physical_model": "b"*64, "physical_reference": "r"*64,
            "layout": "l"*64, "columns_hex": [2.0.hex(), 0.5.hex()],
            "rows_hex": [1.0.hex()]*size, "lift_hex": [[1.0.hex(), (-0.25).hex()], [(-0.5).hex(), 1.0.hex()]],
            "reference_inputs_hex": [0.0.hex(), 1.0.hex()],
            "raw_coordinate_meaning": "scaled voltage departures; other fields are scaled physical remainders",
            "physical_coordinate_meaning": "Point.y is the first word of S*z+L*(a-a_ref)"}
    if count == 12:
        base.update(mapped_input_profile="frame-input-expansion12-v1", mapped_input_words=12)
    request = {"map_identity": q.digest(base), "voltage_lift_map": base, "segments": [first]}
    if framed:
        request["segment_frame_policy"] = {"schema": "solarlab.segment-frame-policy.v1"}
    context_id = q.digest(request)
    rates = [1.0.hex(), 0.5.hex()]
    residual = {"high_hex": zero, "low_hex": zero}
    parent = {"schema": "solarlab.voltage-lift-initial-input.v1", "source_identity": "manufactured",
              "request_sha256": context_id, "map_identity": request["map_identity"],
              "point_identity": "p"*64, "predecessor_identity": base["physical_reference"],
              "segment_id": first["id"], "segment_sha256": q.digest(first),
              "time_hex": first["start"].hex(), "inputs_hex": [0.5.hex(), 2.0.hex()],
              "input_rates_hex": [0.5.hex(), 0.5.hex()],
              "raw_z_hex": zero, "raw_zdot_hex": rates,
              "mapped_physical_rate_words_hex": [[2.375.hex(), 0.5.hex()]] + [list(zero) for _ in range(count-1)],
              "desired_tangent_residual_SI": residual, "represented_rate_residual_SI": residual,
              "state_changed": False, "native_initialization_performed": False, "native_steps": 0}
    record = copy.deepcopy(parent)
    initial = None
    if framed:
        ancestry = {"request_sha256": context_id, "parent_map_identity": request["map_identity"],
                    "predecessor_identity": base["physical_reference"], "q0_words_hex": [zero]*4}
        seed = {"schema": "solarlab.segment-affine-frame.v1", "parent_map_identity": request["map_identity"],
                "segment_sha256": q.digest(first), "logical_initialization_index": 1,
                "predecessor_identity": base["physical_reference"], "parent_input_sha256": q.digest(ancestry),
                "t0_hex": 0.0.hex(), "size": size, "q0_words_hex": [zero]*4, "v0_words_hex": [zero]*4,
                "law": "q=Q0+(t-t0)*V0+u;qdot=V0+udot", "weight_rounding": "RN-exact-parent-q_then-unfused-SV-v1"}
        def framed_map(frame):
            return dict(base, segment_frame=frame,
                        raw_coordinate_meaning="fixed-segment affine remainder u; native history is u",
                        physical_coordinate_meaning="Point.y is the first retained word of S*(Q0+(t-t0)*V0+u)+L*(a-a_ref)")
        parent["map_identity"] = q.digest(framed_map(seed))
        parent["record_sha256"] = q.digest(parent)
        frame = dict(seed, parent_input_sha256=parent["record_sha256"], v0_words_hex=[rates, zero, zero, zero])
        mapping = framed_map(frame)
        record = dict(parent, schema="solarlab.voltage-lift-initial-input.v2", map_identity=q.digest(mapping),
                      raw_zdot_hex=zero, segment_frame_identity=q.digest(frame), parent_coordinate_rate_hex=rates)
        record.pop("record_sha256")
        initial = {"kind": "voltage_lift_segment_frame", "request_sha256": context_id,
                   "parent_map_identity": request["map_identity"], "map_identity": q.digest(mapping),
                   "frame": frame, "frame_identity": q.digest(frame), "map": mapping,
                   "ancestry": ancestry, "preparation_frame": seed, "parent_input": parent,
                   "physical_handoff_words_hex": [[0.25.hex(), 0.75.hex()]] + [list(zero) for _ in range(count-1)],
                   "physical_rate_handoff_words_hex": parent["mapped_physical_rate_words_hex"],
                   "native_initial_z_hex": zero, "native_initial_zdot_hex": zero,
                   "requested_h0_unchanged": True, "automatic_h0_parent_rate_limiter_parity": False,
                   "native_initialization_performed": False}
        initial["record_sha256"] = q.digest(initial)
    record["record_sha256"] = q.digest(record)
    request.update(z0=[0.0]*size, zdot0=[float.fromhex(v) for v in record["raw_zdot_hex"]],
                   actual_initial_identity=record["point_identity"], initial_preparation=record,
                   preparation_context_sha256=context_id)
    if initial is not None:
        request["initial_segment_frame"] = initial
    return request, size


def _resign_initial_frame(request):
    """Keep all envelope identities valid so negatives reach physical checks."""
    initial = request["initial_segment_frame"]
    parent, frame, record = initial["parent_input"], initial["frame"], request["initial_preparation"]
    parent["record_sha256"] = q.digest({k: v for k, v in parent.items() if k != "record_sha256"})
    frame["parent_input_sha256"] = parent["record_sha256"]
    initial["map"]["segment_frame"] = frame
    initial["frame_identity"] = q.digest(frame)
    initial["map_identity"] = q.digest(initial["map"])
    record.update(map_identity=initial["map_identity"], segment_frame_identity=initial["frame_identity"])
    record["record_sha256"] = q.digest({k: v for k, v in record.items() if k != "record_sha256"})
    initial["record_sha256"] = q.digest({k: v for k, v in initial.items() if k != "record_sha256"})


def _qualified_metadata(label="B", axis="base"):
    """Synthetic raw/affine control ancestry; no physical preparation is done."""
    policy = q.make_policy(ORIGINS[label], q.AXES[axis][0], axis)
    numeric = packet(policy)
    numeric["physical_reference_fields"] = {name: {"high": [0.0]*(hi-lo), "low": [0.0]*(hi-lo)}
                                            for name, (lo, hi) in numeric["variable_offsets"].items()}
    numeric.update(layout_identity="a"*64, initial_identity="b"*64, definition_identity="c"*64)
    origin = policy["origin"]
    common = {key: copy.deepcopy(origin[key]) for key in
              ("segments", "observation_times", "quadrature", "budgets", "mandatory", "physical_domain_policy")}
    common.update(case_id=origin["case_id"], numeric_packet=numeric, qualification_policy=policy)
    raw = dict(common, schema="solarlab.real-device-native-request.v1", controls=q.raw_controls(policy, numeric))
    affine_controls = {k: v for k, v in raw["controls"].items() if k not in ("constraints_idx", "constraints_type")}
    affine = dict(common, schema="solarlab.affine-native-request.v1", controls=affine_controls,
                  qualification_ancestor=raw, prior_request_sha256=q.digest(raw),
                  weight_certificate={"old_controls_sha256": q.digest(raw["controls"]),
                                      "rtol": affine_controls["rtol"], "atol": affine_controls["atol"]})
    controls = copy.deepcopy(affine_controls)
    if label == "B":
        controls.update(rtol=controls["rtol"]/1024, atol=[v/1024 for v in controls["atol"]],
                        nonlin_conv_coef=1.024e-5, nonlin_guard="first-correction-wrms-v1",
                        nonlin_trace_capacity=4096, first_step=7.8125e-7)
    request = dict(common, schema="solarlab.voltage-lift-native-request.v1", qualification_ancestor=affine,
                   prior_request_sha256=q.digest(affine), previous_controls=affine_controls, controls=controls,
                   original_budgets=copy.deepcopy(origin["budgets"]),
                   weight_certificate={"rtol": controls["rtol"], "atol": controls["atol"]})
    if label == "B":
        request.update(nonlinear_control_refinement={"value": 1e-8},
                       nonlinear_guard_policy={"policy": "first-correction-wrms-v1", "trace_capacity": 4096},
                       startup_step_policy={"value": 7.8125e-7},
                       time_weight_policy={"schema": "solarlab.voltage-lift-time-weights.v2",
                                           "qualification_policy_sha256": q.digest(policy), "kappa": 1024},
                       segment_startup_policy={"schema": "solarlab.voltage-lift-segment-startup.v2",
                                               "qualification_policy_sha256": q.digest(policy),
                                               "overrides": {"slow_state_hold": {"first_step": 2.0**-32}}})
    lift = [[0.0.hex(), 0.0.hex()] for _ in range(policy["coordinates"])]
    lo, _ = numeric["variable_offsets"]["phi_V"]
    for i, x in enumerate(numeric["x_m"], lo):
        lift[i][0] = (-x/numeric["definition"]["length"]).hex()
    first = request["segments"][0]
    mapping = {"schema": "solarlab.affine-voltage-map.v1", "physical_model": "d"*64,
               "physical_reference": numeric["initial_identity"], "layout": numeric["layout_identity"],
               "columns_hex": [float(v).hex() for v in numeric["column_scaling"]],
               "rows_hex": [float(v).hex() for v in numeric["row_scaling"]], "lift_hex": lift,
               "reference_inputs_hex": [float(first["voltage"][0]).hex(), float(first["photons"][0]).hex()],
               "mapped_input_profile": "frame-input-expansion12-v1", "mapped_input_words": 12}
    request.update(voltage_lift_map=mapping, map_identity=q.digest(mapping))
    return request


def _physical_qualified_metadata(label="B", profile=(1e-5, 1e-8)):
    """Synthetic complete control ancestry; no model or actual case input."""
    from scripts.benchmarks import physical_tolerance as pt
    request = _qualified_metadata(label)
    numeric = request["numeric_packet"]
    for name, values in numeric["physical_reference_fields"].items():
        value = 0.0 if name == "phi_V" else 0.5 if name == "f" else 1.0
        values["high"] = [value]*len(values["high"])
    numeric["initial_reference"] = {"payload": {"fields": {name: {
        key: [v.hex() for v in values] for key, values in pair.items()}
        for name, pair in numeric["physical_reference_fields"].items()}}}
    affine = request["qualification_ancestor"]
    raw = affine["qualification_ancestor"]
    scales = list(map(Fraction, numeric["column_scaling"]))
    upper = [Fraction(v) for name, (lo, hi) in sorted(numeric["variable_offsets"].items(), key=lambda row: row[1][0])
             for v in numeric["physical_reference_fields"][name]["high"]]
    amounts = [s*Fraction(a) for s, a in zip(scales, raw["controls"]["atol"])]
    rr = q.downward(min([Fraction(raw["controls"]["rtol"])] + [a/(2*u) for a, u in zip(amounts, upper) if u]))
    aa = [q.downward((a-Fraction(rr)*u)/s) for a, u, s in zip(amounts, upper, scales)]
    previous = {k: v for k, v in raw["controls"].items() if k not in ("constraints_idx", "constraints_type")}
    previous.update(rtol=rr, atol=aa)
    affine.update(controls=previous, prior_request_sha256=q.digest(raw),
        weight_certificate={"old_controls_sha256": q.digest(raw["controls"]), "rtol": rr, "atol": aa})
    for key in ("nonlinear_control_refinement", "nonlinear_guard_policy", "startup_step_policy",
                "time_weight_policy", "segment_startup_policy"):
        request.pop(key, None)
    policy = pt.make_policy(numeric, {"r": profile[0], "alpha": profile[1]})
    request.update(prior_request_sha256=q.digest(affine), previous_controls=previous,
                   controls=pt.controls(previous, policy), physical_tolerance_policy=policy,
                   weight_certificate=policy, native_admission=False)
    return request


class PhysicalQualifiedControlTests(unittest.TestCase):
    def test_physical_profiles_return_exact_size_without_old_tolerance_compression(self):
        from scripts.benchmarks import physical_tolerance as pt
        for label, size in (("S0", 51), ("B", 85)):
            for profile in pt.PROFILES:
                request = _physical_qualified_metadata(label, profile)
                self.assertEqual(q.check_applied_ceiling(request), size)
                self.assertIs(type(q.check_applied_ceiling(request)), int)
                self.assertEqual(request["controls"]["rtol"], profile[0])
                self.assertEqual(request["controls"]["atol"], [a/s for a, s in zip(
                    request["physical_tolerance_policy"]["atol_SI"], request["numeric_packet"]["column_scaling"])])
                self.assertEqual(request["controls"]["nonlin_conv_coef"], 1e-8)
                self.assertEqual(request["budgets"], request["original_budgets"])

    def test_physical_qualification_requires_complete_initialization_and_rejects_mixed_controls(self):
        for mutate in (lambda r: r.update(z0=[]), lambda r: r.update(initial_preparation={}),
                       lambda r: r.update(time_weight_policy={}),
                       lambda r: r["controls"].update(nonlin_conv_coef=65536e-8),
                       lambda r: r["controls"].update(max_step=0.5)):
            request = _physical_qualified_metadata()
            mutate(request)
            with self.assertRaises((ValueError, KeyError)):
                q.check_applied_ceiling(request)

    def test_real_factory_and_reader_select_physical_option_with_all_frame_words(self):
        from dataclasses import asdict
        import numpy as np
        from scripts.benchmarks import coupled_device_prototype as producer
        from scripts.benchmarks import native_readback as reader
        from scripts.benchmarks import physical_tolerance as pt
        from scripts.benchmarks.precision_prototype import PrimitiveExpansion
        request = _physical_qualified_metadata()
        request["segment_frame_policy"] = producer._segment_frame_policy(request, "fixed-affine-state-rate-v1")
        request["frame_input_policy"] = producer._frame_input_policy(request, "frame-input-expansion12-v1")
        self.assertEqual(reader.frame_input_word_count(request), 12)
        row = request["segments"][0]
        segment = producer.ProtocolSegment(row["id"], row["start"], row["end"], tuple(row["voltage"]), tuple(row["photons"]))
        size = request["qualification_policy"]["coordinates"]
        zero = PrimitiveExpansion(tuple(np.zeros(size) for _ in range(4)))
        frame = producer.SegmentAffineFrame(request["map_identity"], q.digest(asdict(segment)), 1,
                                            "a"*64, "b"*64, segment.start, zero, zero)
        captured, marker = {}, object()
        def constructor(residual, **kwargs):
            captured.update(kwargs)
            return marker
        result, receipt = producer._voltage_lift_segment_solver(constructor, request, segment, 1,
            marker, marker, marker, [], frame=frame)
        self.assertIs(result, marker)
        self.assertNotIn("parent_weight_frame", captured)
        self.assertEqual(captured["physical_weight_frame"], pt.weight_option(request, frame.payload(), asdict(segment)))
        self.assertEqual(receipt["physical_weight_frame"], captured["physical_weight_frame"])
        self.assertEqual(reader.segment_startup_controls(request, row), receipt["constructor_controls"])
        self.assertEqual(captured["nonlin_conv_coef"], 1e-8)
        foreign = asdict(segment)
        foreign["photons"] = (foreign["photons"][0]+1.0, foreign["photons"][1])
        rebound_frame = dict(frame.payload(), segment_sha256=q.digest(foreign))
        with self.assertRaisesRegex(ValueError, "segment_binding"):
            pt.weight_option(request, rebound_frame, foreign)
        # This is an in-memory constructor-argument spy, not IDA or a model.

    def test_physical_profile_binds_the_failed_parent_without_kappa_equivalence(self):
        from scripts.benchmarks import coupled_device_prototype as producer
        from scripts.benchmarks import native_readback as reader
        from scripts.benchmarks import physical_tolerance as pt
        old = _physical_qualified_metadata()
        for key in ("physical_tolerance_policy", "qualification_policy", "qualification_ancestor"):
            old.pop(key)
        old["controls"] = copy.deepcopy(old["previous_controls"])
        old["weight_certificate"] = {"rtol": old["controls"]["rtol"], "atol": old["controls"]["atol"]}
        old["voltage_lift_map"].pop("mapped_input_profile")
        old["voltage_lift_map"].pop("mapped_input_words")
        old["map_identity"] = q.digest(old["voltage_lift_map"])
        old["segment_frame_policy"] = producer._segment_frame_policy(old, "fixed-affine-state-rate-v1")
        failure = {"manufactured_failure_fixture": True}
        parent = {"request_json": json.dumps(old), "native_result_json": json.dumps({
            "request_sha256": q.digest(old), "complete_protocol": False,
            "reason": "primitive_expansion_capacity_exceeded", "first_failure": failure}),
            "first_failure_json": json.dumps({"kind": "first_callback_failure", "failure": failure})}
        current = copy.deepcopy(old)
        policy = pt.make_policy(current["numeric_packet"], {"r": 1e-5, "alpha": 1e-8})
        current.update(physical_tolerance_policy=policy, weight_certificate=policy,
                       controls=pt.controls(current["previous_controls"], policy), frame_input_parent=parent)
        current["voltage_lift_map"].update(mapped_input_profile="frame-input-expansion12-v1", mapped_input_words=12)
        current["map_identity"] = q.digest(current["voltage_lift_map"])
        current["segment_frame_policy"] = producer._segment_frame_policy(current, "fixed-affine-state-rate-v1")
        current["frame_input_policy"] = producer._frame_input_policy(current, "frame-input-expansion12-v1")
        a, b = producer._frame_input_parent_binding(current), reader._frame_input_parent(current)
        self.assertEqual(a, b)
        self.assertEqual(a["control_refinement"]["old_failed_request_sha256"], q.digest(old))
        self.assertFalse(a["control_refinement"]["effective_nonlinear_equivalence"])
        self.assertEqual(reader.frame_input_word_count(current), 12)
        self.assertEqual(parent["request_json"], json.dumps(old))
        for kind in ("control", "physics", "successful_parent"):
            bad = copy.deepcopy(current)
            if kind == "control":
                bad["controls"]["nonlin_conv_coef"] *= 65536
            elif kind == "physics":
                bad["segments"][0]["voltage"][0] += 1
            else:
                result = json.loads(bad["frame_input_parent"]["native_result_json"])
                result["complete_protocol"] = True
                bad["frame_input_parent"]["native_result_json"] = json.dumps(result)
            with self.assertRaises((ValueError, producer.ContractError)):
                producer._frame_input_parent_binding(bad)
            with self.assertRaises(ValueError):
                reader._frame_input_parent(bad)

    def test_declared_twelve_word_charge_reader_uses_complete_physical_words(self):
        from scripts.benchmarks import coupled_device_prototype as producer
        from scripts.benchmarks.native_observation import prepare_interval_observation_policy
        from scripts.benchmarks.native_readback import PhysicalChargeCheck
        from perovskite_sim.constants import Q
        request = _physical_qualified_metadata()
        request["segment_frame_policy"] = producer._segment_frame_policy(request, "fixed-affine-state-rate-v1")
        request["frame_input_policy"] = producer._frame_input_policy(request, "frame-input-expansion12-v1")
        request["interval_observation"] = prepare_interval_observation_policy(request,
            binding_identity="a"*64, header_sha256="b"*64, backend_modules={})
        check = PhysicalChargeCheck(request)
        self.assertEqual(check.input_words, 12)
        size = request["qualification_policy"]["coordinates"]
        left = {"history": {"physical_cumulative_words_hex": [[0.0.hex()]*size for _ in range(12)]}}
        right = copy.deepcopy(left)
        index = request["numeric_packet"]["variable_offsets"]["p_m3"][0]+1
        right["history"]["physical_cumulative_words_hex"][0][index] = (2.0**-20).hex()
        right["history"]["physical_cumulative_words_hex"][1][index] = (2.0**-80).hex()
        expected = Fraction(Q)*Fraction(request["numeric_packet"]["volumes_m3"][1])*(Fraction(1, 2**20)+Fraction(1, 2**80))
        self.assertEqual(check.saved_change(left, right), (expected, Fraction(0), Fraction(0)))
        right["history"]["physical_cumulative_words_hex"].pop()
        with self.assertRaises(ValueError):
            check.saved_change(left, right)


class QualifiedControlTests(unittest.TestCase):
    def test_original_fourteen_choices_keep_defaults_and_allow_explicit_hold(self):
        for label in ("S0", "B"):
            for axis, (intervals, _, _) in q.AXES.items():
                policy = q.make_policy(ORIGINS[label], intervals, axis)
                original = q.options(policy)
                self.assertTrue(q.matches_options(policy, original))
                chosen = copy.deepcopy(original)
                chosen["segment_startup_overrides"] = {"slow_state_hold": {"first_step": 2.0**-32}}
                self.assertEqual(q.matches_options(policy, chosen), label == "B")
                self.assertEqual(q.options(policy), original)

    def test_invalid_hold_and_other_options_are_rejected(self):
        policy = q.make_policy(ORIGINS["B"], 16, "base")
        for value in (-0.0, False, True, -2.0**-32, 1e-8, math.nan, math.inf, "0"):
            chosen = q.options(policy)
            chosen["segment_startup_overrides"]["slow_state_hold"]["first_step"] = value
            self.assertFalse(q.matches_options(policy, chosen), repr(value))
        for name, value in (("nonlin_conv_coef", 1e-6), ("first_step", 0.0), ("extra", True)):
            chosen = q.options(policy)
            chosen[name] = value
            self.assertFalse(q.matches_options(policy, chosen))
        chosen = q.options(policy)
        chosen["segment_startup_overrides"]["extra"] = {}
        self.assertFalse(q.matches_options(policy, chosen))

    def test_framed_and_unframed_initialization_keep_distinct_identities(self):
        for framed in (False, True):
            request, size = _initialization_metadata(framed=framed)
            before = copy.deepcopy(request)
            q.check_initialization(request, size)
            self.assertEqual(request, before)
            if framed:
                frame = request["initial_segment_frame"]
                self.assertEqual(len({request["map_identity"], frame["frame_identity"], frame["map_identity"]}), 3)

    def test_frame_payload_id_cannot_replace_framed_map_id(self):
        request, size = _initialization_metadata(framed=True)
        record = request["initial_preparation"]
        record["map_identity"] = request["initial_segment_frame"]["frame_identity"]
        record["record_sha256"] = q.digest({k: v for k, v in record.items() if k != "record_sha256"})
        with self.assertRaisesRegex(ValueError, "initial_record_binding"):
            q.check_initialization(request, size)

    def test_frame_output_context_and_words_are_bound(self):
        mutations = [lambda r: r.pop("initial_segment_frame"),
                     lambda r: r["initial_segment_frame"]["frame"].update(size=3),
                     lambda r: r["initial_segment_frame"]["frame"].update(logical_initialization_index=True),
                     lambda r: r["initial_segment_frame"]["frame"].update(predecessor_identity="wrong"),
                     lambda r: r["initial_segment_frame"]["map"].update(physical_model="wrong"),
                     lambda r: r["initial_segment_frame"]["frame"]["v0_words_hex"][0].__setitem__(0, 2.0.hex()),
                     lambda r: r.update(preparation_context_sha256="wrong")]
        for mutate in mutations:
            request, size = _initialization_metadata(framed=True)
            mutate(request)
            with self.assertRaises(ValueError):
                q.check_initialization(request, size)
        request, size = _initialization_metadata(framed=False)
        request["initial_segment_frame"] = {}
        with self.assertRaisesRegex(ValueError, "unbound_initial_frame"):
            q.check_initialization(request, size)

    def test_physical_handoff_accepts_exact_four_and_twelve_words(self):
        for count in (4, 12):
            request, size = _initialization_metadata(framed=True, count=count)
            before = copy.deepcopy(request)
            q.check_initialization(request, size)
            self.assertEqual(request, before)

    def test_missing_all_matching_rates_cannot_pass(self):
        request, size = _initialization_metadata(framed=True)
        initial = request["initial_segment_frame"]
        initial.pop("physical_rate_handoff_words_hex")
        initial["parent_input"].pop("mapped_physical_rate_words_hex")
        request["initial_preparation"].pop("mapped_physical_rate_words_hex")
        _resign_initial_frame(request)
        with self.assertRaisesRegex(ValueError, "initial_frame_word_shape"):
            q.check_initialization(request, size)

    def test_matching_rate_arrays_must_be_well_formed_and_physical(self):
        wrong = [[3.0.hex(), 0.5.hex()]] + [[0.0.hex()]*2 for _ in range(3)]
        noncanonical = [[2.5.hex(), 0.5.hex()], [(-0.125).hex(), 0.0.hex()]] + [[0.0.hex()]*2 for _ in range(2)]
        cases = [(None, "word_shape"), ([], "word_shape"),
                 ([[0.0.hex()]]*4, "vector_shape"),
                 ([["nan", 0.0.hex()]]*4, "nonfinite_word"),
                 ([["invalid", 0.0.hex()]]*4, "invalid_hex"),
                 ([[False, 0.0.hex()]]*4, "vector_shape"),
                 (noncanonical, "noncanonical_word"), (wrong, "physical_handoff")]
        for words, reason in cases:
            with self.subTest(reason=reason):
                request, size = _initialization_metadata(framed=True)
                initial = request["initial_segment_frame"]
                initial["physical_rate_handoff_words_hex"] = copy.deepcopy(words)
                initial["parent_input"]["mapped_physical_rate_words_hex"] = copy.deepcopy(words)
                request["initial_preparation"]["mapped_physical_rate_words_hex"] = copy.deepcopy(words)
                _resign_initial_frame(request)
                with self.assertRaisesRegex(ValueError, "initial_frame_"+reason):
                    q.check_initialization(request, size)

    def test_missing_or_wrong_state_handoff_cannot_pass(self):
        for value, reason in ((None, "word_shape"),
                              ([[0.0.hex()]*2 for _ in range(4)], "physical_handoff")):
            request, size = _initialization_metadata(framed=True)
            initial = request["initial_segment_frame"]
            if value is None:
                initial.pop("physical_handoff_words_hex")
            else:
                initial["physical_handoff_words_hex"] = value
            _resign_initial_frame(request)
            with self.assertRaisesRegex(ValueError, "initial_frame_"+reason):
                q.check_initialization(request, size)

    def test_right_sided_initial_inputs_rates_and_time_are_bound(self):
        for name, value in (("inputs_hex", [0.0.hex(), 2.0.hex()]),
                            ("input_rates_hex", [0.0.hex(), 0.5.hex()]),
                            ("time_hex", 0.5.hex())):
            request, size = _initialization_metadata(framed=True)
            request["initial_segment_frame"]["parent_input"][name] = copy.deepcopy(value)
            request["initial_preparation"][name] = copy.deepcopy(value)
            _resign_initial_frame(request)
            with self.assertRaisesRegex(ValueError, "initial_frame_input_binding"):
                q.check_initialization(request, size)

    def test_qualified_profile_is_non_circular_and_has_an_independent_reader(self):
        from scripts.benchmarks import coupled_device_prototype as producer
        from scripts.benchmarks import native_readback as reader
        for label in ("S0", "B"):
            for axis in q.AXES:
                request = _qualified_metadata(label, axis)
                q.check_applied_ceiling(request)
                request["segment_frame_policy"] = producer._segment_frame_policy(request, "fixed-affine-state-rate-v1")
                policy = producer._frame_input_policy(request, "frame-input-expansion12-v1")
                self.assertEqual(policy["schema"], "solarlab.frame-input-policy.v2")
                self.assertNotIn("failed_parent", policy)
                request["frame_input_policy"] = policy
                before = copy.deepcopy(request)
                self.assertEqual(reader.frame_input_word_count(request), 12)
                self.assertEqual(request, before)

    def test_qualified_profile_cannot_borrow_a_failed_parent_or_changed_map(self):
        from scripts.benchmarks import coupled_device_prototype as producer
        from scripts.benchmarks import native_readback as reader
        request = _qualified_metadata()
        request["segment_frame_policy"] = producer._segment_frame_policy(request, "fixed-affine-state-rate-v1")
        request["frame_input_policy"] = producer._frame_input_policy(request, "frame-input-expansion12-v1")
        mutations = [lambda r: r.update(frame_input_parent={}),
                     lambda r: r["frame_input_policy"].update(extra=True),
                     lambda r: r["frame_input_policy"].update(schema="solarlab.frame-input-policy.v1"),
                     lambda r: r["voltage_lift_map"].update(physical_model="e"*64),
                     lambda r: r["voltage_lift_map"]["lift_hex"][0].__setitem__(1, 1.0.hex()),
                     lambda r: r["controls"].update(max_step=0.1),
                     lambda r: r["qualification_policy"].update(intervals=32),
                     lambda r: r["numeric_packet"].update(initial_identity="wrong")]
        for mutate in mutations:
            changed = copy.deepcopy(request)
            mutate(changed)
            with self.assertRaises((reader.HistoryVerificationError, ValueError)):
                reader.frame_input_word_count(changed)

    def test_both_actual_upper_guards_admit_the_bound_hold_selection(self):
        # Manufactured API sentinels isolate the guards, never source/genuine
        # model readiness. The actual14-case constructors still need admission.
        from dataclasses import asdict
        from types import SimpleNamespace
        from unittest.mock import patch
        from scripts.benchmarks import coupled_device_prototype as producer
        class ReachedAfterGuard(Exception):
            pass
        for step in (0.0, 2.0**-32, 1e-8):
            request = _qualified_metadata()
            policy = request["qualification_policy"]
            selected = q.options(policy)
            selected["segment_startup_overrides"] = {"slow_state_hold": {"first_step": step}}
            segments = tuple(producer.ProtocolSegment(row["id"], row["start"], row["end"],
                             tuple(row["voltage"]), tuple(row["photons"])) for row in request["segments"])
            definition = SimpleNamespace(id=request["case_id"])
            def stop():
                raise ReachedAfterGuard
            model = SimpleNamespace(definition=definition, intervals=16,
                                    layout=SimpleNamespace(size=85), numeric_packet=stop)
            mapping = SimpleNamespace(model=model, mapped_input_profile=None)
            outcome = ReachedAfterGuard if step in (0.0, 2.0**-32) else producer.ContractError
            with patch.object(producer, "_qualification_scope", return_value=policy):
                with self.assertRaises(outcome):
                    producer.prepare_voltage_lift_native_request(mapping, segments, request["qualification_ancestor"], **selected)
                model.numeric_packet = lambda: request["numeric_packet"]
                mapping.identity = request["map_identity"]
                request["segment_startup_policy"]["overrides"] = selected["segment_startup_overrides"]
                self.assertEqual(q.digest([asdict(s) for s in segments]), q.digest(request["segments"]))
                with patch.object(q, "check_applied_ceiling", return_value=85), \
                     patch.object(producer, "voltage_lift_wrms_policy", side_effect=ReachedAfterGuard):
                    with self.assertRaises(outcome):
                        producer.validate_voltage_lift_native_request(mapping, segments, request)


if __name__ == "__main__":
    unittest.main()
