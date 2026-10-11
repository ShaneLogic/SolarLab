"""Tiny synthetic-only qualification analysis tests; no scientific fixtures.

All metadata, values, message IDs and receipts here are manufactured unit-test
data. None is an operation grant, accepted device run, or reference oracle.
"""
from copy import deepcopy
from fractions import Fraction as F
import gzip
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

from scripts.benchmarks import qualification_comparison as q
from scripts.benchmarks.hysteresis_current_bounds import _expm1_bounds


class ArithmeticTests(unittest.TestCase):
    def test_flat_SG_is_exact_linear_for_both_carriers(self):
        for carrier in ("electron", "hole"):
            self.assertEqual(q.sg_interpolate(2, 8, 0, F(1, 3), carrier=carrier), q.Span.of(4))

    def test_sg_constant_field_formula_and_carrier_sign(self):
        # expm1(2u)=expm1(u)*(exp(u)+1), giving the independent midpoint weight.
        ep = q.Span.of(_expm1_bounds(F(1)))+1
        expected = 2+6*(q.Span.of(1)/(ep+1))
        value = q.sg_interpolate(2, 8, 2, F(1, 2), carrier="electron")
        self.assertLessEqual(value.lo, expected.hi)
        self.assertGreaterEqual(value.hi, expected.lo)
        self.assertLess(value.hi, 5)
        hole = q.sg_interpolate(2, 8, 2, F(1, 2), carrier="hole")
        self.assertGreater(hole.lo, 5)

    def test_sg_low_word_and_near_zero_cancellation(self):
        delta = F(1, 2**120)
        value = q.sg_interpolate(1, 1+delta, F(1, 2**140), F(1, 2), carrier="electron")
        self.assertGreater(value.lo, 1)
        self.assertLess(value.hi, 1+delta)
        self.assertLess((value-q.Span.of(1+delta/2)).absolute_upper, delta/1000)

    def test_contacts_exact_and_no_extrapolation(self):
        for s, expected in ((0, 2), (1, 8)):
            self.assertEqual(q.sg_interpolate(2, 8, 400, s, carrier="electron"), q.Span.of(expected))
        with self.assertRaisesRegex(q.ComparisonError, "sg_scope"):
            q.sg_interpolate(2, 8, 0, -1, carrier="electron")

    def test_log_interval_preserves_sub_float_difference(self):
        delta = F(1, 2**100)
        log = q.log_exact(1+delta)
        self.assertGreater(log.lo, 0)
        self.assertLessEqual(log.hi, delta)
        self.assertGreaterEqual(log.lo, delta-delta*delta/2)
        self.assertEqual(q.log_exact(1), q.Span.of(0))

    def test_same_physical_positions_on_different_meshes(self):
        a = {"phi_V": [0, F(1, 3), 1], "n_m3": [2]*3, "p_m3": [3]*3}
        b = {"phi_V": [0, F(1, 4), F(1, 2), 1], "n_m3": [2]*4, "p_m3": [3]*4}
        left = q.reconstruct([0, F(1, 3), 1], a, 1)
        right = q.reconstruct([0, F(1, 4), F(1, 2), 1], b, 1)
        self.assertEqual(left, right)
        self.assertEqual(len(left), 19)
        self.assertEqual(left[1]["phi_V"], q.Span.of(F(1, 18)))

    def test_trap_interpolation_requires_explicit_rule(self):
        fields = {"phi_V": [0, 0], "n_m3": [1, 1], "p_m3": [1, 1], "f": [0, 1]}
        with self.assertRaisesRegex(q.ComparisonError, "trap_reconstruction"):
            q.reconstruct([0, 1], fields, 1)
        got = q.reconstruct([0, 1], fields, 1, trap_rule="linear_nodal_occupancy_v1")
        self.assertEqual(got[9]["f"], q.Span.of(F(1, 2)))
        with self.assertRaisesRegex(q.ComparisonError, "interface"):
            q.reconstruct([0, 1], fields, 1, trap_rule="linear_nodal_occupancy_v1", support=("a", "b"))
        fields["f"][1] = F(101, 100)
        with self.assertRaisesRegex(q.ComparisonError, "no_clipping"):
            q.reconstruct([0, 1], fields, 1, trap_rule="linear_nodal_occupancy_v1")

    def test_full_reference_and_all_remainder_words(self):
        request = {"z0": [0], "frame_input_policy": {"mapped_input_words": 12},
                   "voltage_lift_map": {"physical_model": "test-only-model"},
                   "numeric_packet": {"variable_offsets": {"n_m3": [0, 1]}, "initial_reference": {
                    "payload": {"fields": {"n_m3": {"high": [1.0.hex()], "low": [(2.0**-70).hex()]}}}}}}
        words = [[0.0.hex()] for _ in range(12)]
        words[0] = [(2.0**-100).hex()]
        history = {"physical_model_identity": "test-only-model", "physical_cumulative_words_hex": words,
                   "physical_rate_words_hex": [[0.0.hex()] for _ in range(12)]}
        fields, rates = q.full_state(request, history)
        self.assertEqual(fields["n_m3"][0], 1+F(1, 2**70)+F(1, 2**100))
        self.assertEqual(rates["n_m3"], (F(0),))
        broken = deepcopy(history); broken["physical_model_identity"] = "other"
        with self.assertRaisesRegex(q.ComparisonError, "physical_source"):
            q.full_state(request, broken)

    def test_polynomial_state_rate_exact(self):
        fields, rates = q.polynomial_state(((F(1), F(2), F(3)),), 1, F(1, 2), F(3, 4), {"phi_V": (0, 1)})
        self.assertEqual(fields["phi_V"], (F(3, 4),))
        self.assertEqual(rates["phi_V"], (F(-2),))

    def test_rational_budget_and_independent_shares(self):
        args = dict(metric="absolute", budget=F(3, 10), input_error=F(1, 10), reference_error=F(1, 10))
        got = q.compare_metric(F(1, 10), 0, **args)
        self.assertTrue(got["passed"])
        self.assertEqual(got["total_upper"], F(3, 10))
        args["input_error"] += F(1, 10**100)
        self.assertFalse(q.compare_metric(F(1, 10), 0, **args)["passed"])

    def test_reference_absence_zero_and_undefined_denominators(self):
        args = dict(metric="absolute", budget=0, input_error=0, reference_error=0)
        self.assertTrue(q.compare_metric(1, 1, **args)["passed"])
        self.assertFalse(q.compare_metric(1+F(1, 2**100), 1, **args)["passed"])
        args["reference_error"] = None
        with self.assertRaisesRegex(q.ComparisonError, "independent_reference"):
            q.compare_metric(1, 1, **args)
        for metric, scale in (("log_ratio", None), ("declared_scale", 0)):
            with self.assertRaises(q.ComparisonError):
                q.compare_metric(1, 0, metric=metric, scale=scale, budget=1, input_error=0, reference_error=0)

    def test_charge_prefix_uses_absolute_defects_not_cancellation(self):
        first = q.accumulate_charge(((0, 0),)*3, (F(1, 100), 0, 0), (0, 0, 0), F(1, 100))
        self.assertTrue(first[0]["passed"])
        previous = tuple((r["prefix_absolute_bound_C"], r["prefix_uncertainty_C"]) for r in first)
        second = q.accumulate_charge(previous, (-F(1, 100), 0, 0), (0, 0, 0), F(1, 100))
        self.assertEqual(second[0]["prefix_absolute_bound_C"], F(1, 50))
        self.assertFalse(second[0]["passed"])

    def test_event_sides_never_merged(self):
        left = {"key": q.observation_key("ramp", 1, "left"), "fields": {"phi_V": []}}
        right = {"key": q.observation_key("hold", 1, "right"), "fields": {"phi_V": []}}
        with self.assertRaisesRegex(q.ComparisonError, "event_mismatch"):
            q.validate_pair_clock(left, right)

    def test_actual_refinement_and_joint_unknown(self):
        self.assertFalse(q.actual_refinement("time", [0, 1], [0, 1], [1, 1], [1, 1])["passed"])
        self.assertTrue(q.actual_refinement("space", [0, 1], [0, F(1, 2), 1], [1], [1])["passed"])
        self.assertFalse(q.joint_confirmation(None)["passed"])
        self.assertFalse(q.joint_confirmation({"interaction": True, "independently_accepted": True})["passed"])

    def test_dimensionless_metrics_preserve_physical_field_units(self):
        fields = {"phi_V": [0, 0], "n_m3": [1, 1], "p_m3": [2, 2], "c_m3": [3, 3], "f": [0.5, 0.5]}
        currents = {k: [0, 0] for k in ("raw_total_inward_A", "same_state_tangent_total_inward_A")}
        snap = {"key": ("test-only", F(0), "continuous"), "mesh": [0, 1], "fields": fields, "observables": currents}
        uncertainty = {k: {"input": 0, "reference": 0} for k in (*fields, *currents)}
        thresholds = dict(potential_max_abs_V=1, carrier_max_abs_log_ratio=1,
                          ion_max_abs_change_over_declared_c0=1, trap_max_abs_occupancy_change=1, current_A=1)
        result = q.score_clock(snap, snap, thresholds=thresholds, uncertainty=uncertainty,
                               c0=3, thermal_voltage=1, trap_rule="linear_nodal_occupancy_v1")
        for row in result["rows"]:
            if row["channel"] in ("n_m3", "p_m3", "c_m3"):
                self.assertEqual((row["unit"], row["field_unit"]), ("1", "m^-3"))
            elif row["channel"] == "phi_V":
                self.assertEqual((row["unit"], row["field_unit"]), ("V", "V"))

    def test_tolerance_tightening_can_keep_the_same_maximum_native_h(self):
        coarse = {"rtol": F(1, 10), "atol": [F(1, 20)], "max_step": F(1, 4)}
        fine = {**coarse, "rtol": F(1, 100), "atol": [F(1, 200)]}
        result = q.actual_refinement("tolerance", [0, 1], [0, 1], [F(1, 4)]*2,
            [F(1, 4), F(1, 8), F(1, 8)], tolerance_controls={"coarse": coarse, "fine": fine, "scales": [3]})
        self.assertTrue(result["passed"])
        self.assertEqual(result["coarse_max_h"], result["fine_max_h"])
        self.assertEqual(result["physical_control_tightening"]["largest_physical_atol_ratio"], F(1, 10))
        self.assertFalse(q.actual_refinement("time", [0, 1], [0, 1], [F(1, 4)]*2,
                                             [F(1, 4), F(1, 8), F(1, 8)])["passed"])

    def test_nominal_ceiling_and_boolean_do_not_prove_pairwise_tightening(self):
        coarse = {"rtol": F(1, 10), "atol": [F(1, 20)], "max_step": 1}
        fine = {**coarse, "rtol": F(1, 100), "atol": [F(1, 200)]}
        for invalid in (coarse, {**fine, "atol": coarse["atol"]}, {**fine, "rtol": coarse["rtol"]},
                        {**fine, "max_step": F(1, 2)}):
            with self.assertRaises(q.ComparisonError):
                q.physical_control_tightening(coarse, invalid, [3])
        for evidence in (None, True):
            with self.assertRaises(q.ComparisonError):
                q.actual_refinement("tolerance", [0, 1], [0, 1], [1], [1], tolerance_controls=evidence)

    def test_step_summary_storage_is_independent_of_interval_count(self):
        steps = q.StepSummary()
        for _ in range(1000):
            steps.add(F(1, 8), 2)
        self.assertEqual(steps.record(), {"count": 1000, "minimum_h": F(1, 8), "maximum_h": F(1, 8), "orders": {"2": 1000}})
        self.assertEqual(len(vars(steps)), 4)

    def test_metadata_only_uniform_physical_laws(self):
        packet = {"x_m": [0, F(1, 2), 1], "volumes_m3": [F(1, 4), F(1, 2), F(1, 4)],
                  "definition": {"id": "S0NeutralPublicDeviceV1", "dynamic": False,
                     "temperature": 1, "area": 1, "epsilon": 1, "trap_density": 0,
                     "mu_n": 1, "mu_p": 1}}
        fields = {"phi_V": [0, 0, 0], "n_m3": [1, 1, 1], "p_m3": [1, 1, 1]}
        rates = {k: [0, 0, 0] for k in fields}
        result = q.physical_observables(packet, fields, rates, [0, 0], {"Q": 1, "K_B": 1})
        self.assertEqual(result["raw_total_inward_A"], (q.Span.of(0),)*2)
        self.assertEqual(result["same_state_tangent_total_inward_A"], (q.Span.of(0),)*2)
        self.assertEqual(result["gauss_defect_C"], 0)
        fields["p_m3"][1] += F(1, 2**100)
        changed = q.physical_observables(packet, fields, rates, [0, 0], {"Q": 1, "K_B": 1})
        self.assertEqual(changed["body_charge_C"], F(1, 2**101))
        self.assertNotEqual(changed["gauss_defect_C"], 0)

    def test_physical_current_signs_and_finite_gauss_are_independent(self):
        packet = {"x_m": [0, 0.5, 1], "volumes_m3": [0.25, 0.5, 0.25],
                  "definition": {"id": "S0NeutralPublicDeviceV1", "dynamic": False, "temperature": 1,
                                 "area": 1, "epsilon": 1, "trap_density": 0, "mu_n": 1, "mu_p": 1}}
        delta = F(1, 2**100)
        fields = {"phi_V": [0, 0, 0], "n_m3": [1, 1, 1], "p_m3": [1, 1+delta, 1]}
        result = q.physical_observables(packet, fields, {k: [0]*3 for k in fields}, [0, 0], {"Q": 1, "K_B": 1})
        self.assertEqual(result["raw_total_inward_A"], (q.Span.of(-2*delta),)*2)
        self.assertEqual(result["same_state_tangent_total_inward_A"], (q.Span.of(0),)*2)
        self.assertEqual(result["gauss_defect_C"], delta/2)

    def test_nonfinite_boolean_and_nonpositive_population_rejection(self):
        for value in (True, float("nan"), float("inf")):
            with self.assertRaises(q.ComparisonError): q.exact(value)
        with self.assertRaises(q.ComparisonError):
            q.sg_interpolate(0, 1, 1, F(1, 2), carrier="electron")

    def test_manufactured_dynamic_ions_remain_blocking_at_contacts(self):
        packet = {"x_m": [0, 0.5, 1], "volumes_m3": [0.25, 0.5, 0.25],
                  "definition": {"id": "DynamicAcceptorIonPublicDeviceV1", "dynamic": True, "temperature": 1,
                    "area": 1, "epsilon": 1, "trap_density": 0, "mu_n": 1, "mu_p": 1,
                    "ion_initial": 1, "ion_capacity": 4, "capture_n": 0, "capture_p": 0,
                    "n1": 1, "p1": 1, "diffusion_ion": 1}}
        fields = {"phi_V": [0, 0, 0], "n_m3": [1]*3, "p_m3": [1]*3,
                  "c_m3": [F(1), F(3, 2), F(2)], "f": [F(1, 2)]*3}
        r = q.physical_observables(packet, fields, {k: [0]*3 for k in fields}, [0, 0], {"Q": 1, "K_B": 1})
        self.assertEqual(r["raw_total_inward_A"], (q.Span.of(0),)*2)
        balance = sum(r["same_state_tangent_total_inward_A"], q.Span.of(0))
        self.assertLessEqual(balance.lo, 0)
        self.assertGreaterEqual(balance.hi, 0)
        self.assertLess(balance.absolute_upper, F(1, 10**80))


class BoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = os.environ.get("SOLARLAB_QUALIFICATION_ANALYSIS_SOURCE")
        if directory is None:
            raise unittest.SkipTest("separate prepared source packet path required")
        cls.directory = Path(directory)
        # Mutable manufactured fixtures belong on local temporary storage.
        # The prepared source packet remains in its original evidence directory.
        fixtures = tempfile.TemporaryDirectory(prefix="solarlab_qualification_tests_")
        cls.addClassCleanup(fixtures.cleanup)
        cls.fixture_directory = Path(fixtures.name)
        cls.modules = {}
        for name in ("Analyze", "ReadCases"):
            spec = importlib.util.spec_from_file_location("test_only_"+name, cls.directory/(name+".py"))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            cls.modules[name] = module

    def test_loaded_module_requires_declared_path_and_current_bytes(self):
        from types import SimpleNamespace
        m = self.modules["Analyze"]
        with tempfile.TemporaryDirectory(dir=self.fixture_directory) as folder:
            reviewed = Path(folder)/"reviewed.py"
            shadow = Path(folder)/"shadow.py"
            reviewed.write_text("# manufactured module\n")
            shadow.write_bytes(reviewed.read_bytes())
            binding = m.pin(reviewed)
            m.validate_module_source(SimpleNamespace(__file__=str(reviewed)), binding)
            # Equal bytes at an unreviewed import path are not the bound module.
            with self.assertRaisesRegex(ValueError, "actual_selected_reader_dependency"):
                m.validate_module_source(SimpleNamespace(__file__=str(shadow)), binding)
            reviewed.write_text("# changed module body\n")
            consumed = m.METADATA_READ_BYTES
            with self.assertRaisesRegex(ValueError, "actual_selected_reader_dependency"):
                m.validate_module_source(SimpleNamespace(__file__=str(reviewed)), binding)
            self.assertGreater(m.METADATA_READ_BYTES, consumed)

    def test_stale_source_int_float_and_signed_zero(self):
        m = self.modules["Analyze"]
        with tempfile.TemporaryDirectory(dir=self.fixture_directory) as folder:
            p = Path(folder)/"synthetic.json"
            p.write_text('{"test_only":true,"x":0}')
            binding = m.pin(p)
            self.assertEqual(m.json_pin(binding)["x"], 0)
            for changed in ('{"test_only":true,"x":0.0}', '{"test_only":true,"x":-0.0}', '{"test_only":true,"x":1}'):
                p.write_text(changed)
                with self.assertRaises(ValueError): m.json_pin(binding)
            self.assertNotEqual(m.canonical({"x": 0.0}), m.canonical({"x": -0.0}))

    def test_bound_uncertainty_integer_codec_preserves_large_exact_ratio(self):
        m = self.modules["Analyze"]
        from scripts.benchmarks.native_history import integer_json_dumps
        policy = {"schema": "solarlab.native-integer-io.v1", "codec": "decimal-with-bounded-hex-v1",
                  "decimal_digits": 4300, "max_integer_bits": 131072}
        with tempfile.TemporaryDirectory(dir=self.fixture_directory) as folder:
            path = Path(folder)/"test-only-uncertainty.json"
            large = 10**4401+1
            path.write_text(integer_json_dumps({"test_only": True, "upper": {"rational": [large, large+2]}}, policy))
            self.assertEqual(m.observation_pin(m.pin(path), policy)["upper"], F(large, large+2))

    def test_incomplete_and_wrong_request_rejection(self):
        m = self.modules["ReadCases"]
        request = {"qualification_policy": {"test_only": True}}
        result = {"request_sha256": m.reader._digest(request), "status": "completed_bounded_voltage_lift_native_pilot",
                  "complete_protocol": True, "first_failure": None, "history": {"encoding": "gzip", "closed": True,
                   "container_complete": True, "error": None, "path": "NativeHistory.jsonl.gz"}}
        m.validate_complete(request, result, result["request_sha256"])
        for field, value in (("closed", False), ("container_complete", False), ("error", "test-only-error")):
            broken = deepcopy(result); broken["history"][field] = value
            with self.assertRaisesRegex(q.ComparisonError, "incomplete_history"):
                m.validate_complete(request, broken, result["request_sha256"])
        with self.assertRaisesRegex(q.ComparisonError, "stale_request"):
            m.validate_complete(request, result, "test-only-wrong-id")

    def test_tiny_encoded_words_and_partial_tail(self):
        m = self.modules["ReadCases"]
        with tempfile.TemporaryDirectory(dir=self.fixture_directory) as folder:
            p = Path(folder)/"synthetic-window.jsonl"
            raw = json.dumps(m.encode({"schema": "solarlab.qualification-window.v1", "test_only": True,
                                       "low_word": F(1, 2**100)})).encode()+b"\n"
            p.write_bytes(raw)
            limits = {"window_bytes": 4096, "spool_bytes": 4096, "integer_io_policy": None}
            from collections import Counter
            read = list(m.windows(p, limits, lambda: None, Counter()))
            self.assertEqual(read[0]["low_word"], F(1, 2**100))
            p.write_bytes(raw[:-1])
            with self.assertRaisesRegex(q.ComparisonError, "partial_spool"):
                list(m.windows(p, limits, lambda: None, Counter()))

    def test_real_boundary_rejects_missing_actor_messages_and_case_proof(self):
        m = self.modules["Analyze"]
        with self.assertRaises((ValueError, KeyError)):
            m.validate_authority({}, {}, {}, operation_pin={}, admission_pin={}, packet_pin={},
                                 grant_message=None, start_message=None, actor={"role": "test-only"})
        accepted = {"authority": "Root", "case": "test-only", "accepted": True,
                    "complete_protocol_accepted": False, "request_sha256": "test-only", "source_packet": {}}
        with self.assertRaisesRegex(ValueError, "accepted_complete_case"):
            m.validate_case_binding("test-only", accepted, {}, {}, "test-only", {})

    def test_actual_finite_dispatcher_on_manufactured_windows(self):
        from collections import Counter
        m = self.modules["Analyze"]
        material = {"id": "S0NeutralPublicDeviceV1", "dynamic": False, "temperature": 1,
                    "area": 1, "epsilon": 1, "trap_density": 0, "mu_n": 1, "mu_p": 1, "ion_initial": 0}
        packet = {"x_m": [0, 1], "volumes_m3": [0.5, 0.5], "definition": material,
                  "variable_offsets": {"n_m3": [0, 2], "p_m3": [2, 4], "phi_V": [4, 6]}}
        request = {"numeric_packet": packet, "budgets": {"current_A": 1}}
        segment = {"id": "test-only-segment", "start": 0, "end": 1, "voltage": [0, 0], "photons": [0, 0]}
        def window(left, right):
            return {"segment": segment, "ordinal": 1, "left": F(left), "right": F(right), "tn": F(right),
                    "h": F(right)-F(left), "coefficients": [(F(1),)]*4+[(F(0),)]*2, "samples": []}
        thresholds = {"potential_max_abs_V": F(1, 1000), "carrier_max_abs_log_ratio": F(1, 100),
                      "ion_max_abs_change_over_declared_c0": F(1, 100), "trap_max_abs_occupancy_change": F(1, 100)}
        uncertainty = {k: {"input": F(0), "reference": F(0)} for k in
                       ("phi_V", "n_m3", "p_m3", "raw_total_inward_A", "same_state_tangent_total_inward_A")}
        rows, counters = [], Counter()
        result = m.compare_pair(iter([window(0, 1)]), iter([window(0, F(1, 2)), window(F(1, 2), 1)]),
            request, request, job={"axis": "time", "cases": ["test-only-coarse", "test-only-fine"], "finest_pair": True},
            route={"original_accuracy": {"finest_pair": thresholds}}, uncertainty=uncertainty,
            constants={"Q": 1, "K_B": 1}, trap_rule=None, counters=counters, checkpoint=lambda: None, emit=rows.append)
        self.assertTrue(result["passed"])
        self.assertEqual(result["clocks"], 3)
        self.assertEqual(counters["current_evaluations"], 6)
        self.assertFalse(result["scientific_qualification"])
        self.assertEqual(rows[-1]["key"], ("test-only-segment", F(1), "left"))

    def test_authority_contract_pure_manufactured_metadata(self):
        m = self.modules["Analyze"]
        actor = m.ROOT_ACTOR.copy()
        packet = {"test_only": True, "sha256": "a"*64}
        op_pin, ad_pin = {"test_only": "operation"}, {"test_only": "admission"}
        op = {"test_only": True, "schema": "solarlab.original-twelve-analysis-operation.v1", "actor": actor,
              "operation_id": "synthetic-unit-only", "source_packet": packet,
              "scope": "original_fourteen_saved_histories_twelve_finite_comparisons"}
        ad = {"test_only": True, "actor": actor, "operation_id": op["operation_id"], "operation": op_pin,
              "source_packet": packet, "Root_grant_message_id": "msg_aa", "allowed": True, "scientific_producer_calls": 0}
        start = {"test_only": True, "actor": actor, "operation_id": op["operation_id"], "operation": op_pin,
                 "admission": ad_pin, "Root_START_message_id": "msg_bb", "Root_grant_message_id": "msg_aa", "started": True}
        kwargs = dict(operation_pin=op_pin, admission_pin=ad_pin, packet_pin=packet,
                      grant_message="msg_aa", start_message="msg_bb", actor=actor)
        m.validate_authority(op, ad, start, **kwargs)
        for changed in ({**kwargs, "start_message": "msg_aa"}, {**kwargs, "packet_pin": {"test_only": "stale"}}):
            with self.assertRaises(ValueError): m.validate_authority(op, ad, start, **changed)

    def test_source_subclass_keeps_original_predicates(self):
        import ast
        tree = ast.parse((self.directory/"ReadCases.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_ExtractingProtocol")
        for name in ("_sample", "_certificate", "_interval"):
            method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
            self.assertTrue(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and
                n.func.attr == name and isinstance(n.func.value, ast.Call) and
                isinstance(n.func.value.func, ast.Name) and n.func.value.func.id == "super"
                for n in ast.walk(method)))

    def test_every_metadata_pin_and_source_load_is_counted_and_capped(self):
        m = self.modules["Analyze"]
        original_cap = m.METADATA_CEILING
        with tempfile.TemporaryDirectory(dir=self.fixture_directory) as folder:
            p = Path(folder)/"test_only.py"; raw = b"value = 'test-only'\n"; p.write_bytes(raw)
            before = m.METADATA_READ_BYTES
            binding = m.pin(p)
            self.assertEqual(m.METADATA_READ_BYTES-before, len(raw))
            cached = m.PinnedMetadata(binding)
            read_before = m.METADATA_READ_BYTES
            self.assertEqual(cached.read_bytes(), raw)
            self.assertEqual(cached.read_text(), raw.decode())
            self.assertEqual(m.METADATA_READ_BYTES, read_before)
            self.assertEqual(m.load_path("_qa_test_only_source", binding).value, "test-only")
            try:
                m.METADATA_CEILING = m.METADATA_READ_BYTES+len(raw)-1
                for read in (lambda: m.pin(p), lambda: m.read_pin(binding), lambda: m.PinnedMetadata(binding),
                             lambda: m.load_path("_qa_capped_test_source", binding)):
                    with self.assertRaisesRegex(ValueError, "aggregate_metadata_read_cap"):
                        read()
            finally:
                m.METADATA_CEILING = original_cap

    def test_original_delegated_predicates_use_bounded_io(self):
        import ast
        m = self.modules["Analyze"]
        spec = m.json_pin(m.pin(self.directory/"Request.json"))
        packet = m.json_pin(spec["qualification_source_packet"])
        before = m.METADATA_READ_BYTES
        consumers = m.qualification_consumers(spec, packet)
        self.assertGreater(m.METADATA_READ_BYTES, before)
        source = m.read_pin(spec["qualification_job_preparer"])
        tree = ast.parse(source)
        original = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "prepare_jobs")
        self.assertEqual(consumers.prepare_jobs.__code__.co_name, original.name)
        old_cap = m.METADATA_CEILING
        try:
            m.METADATA_CEILING = m.METADATA_READ_BYTES+1
            with self.assertRaisesRegex(ValueError, "aggregate_metadata_read_cap"):
                consumers.prepare_jobs({})
        finally:
            m.METADATA_CEILING = old_cap
        with self.assertRaisesRegex(ValueError, "all_fourteen_complete_cases_required"):
            consumers.prepare_jobs({})

    def test_failed_digest_retains_actual_read_prefix(self):
        from collections import Counter
        m = self.modules["ReadCases"]
        with tempfile.TemporaryDirectory(dir=self.fixture_directory) as folder:
            p = Path(folder)/"test-only-bytes"; p.write_bytes(b"x"*27)
            counters = Counter()
            def stop(): raise RuntimeError("test-only-checkpoint")
            with self.assertRaisesRegex(RuntimeError, "test-only-checkpoint"):
                m.digest_file(p, 32, stop, counters)
            self.assertEqual(counters["encoded_hash_bytes"], 27)
            self.assertEqual(m.accounting(counters)["encoded_hash_bytes"]["exact_total"], 27)

    def test_failed_decoding_and_oversized_line_keep_incremental_counts(self):
        from collections import Counter
        m = self.modules["ReadCases"]
        with tempfile.TemporaryDirectory(dir=self.fixture_directory) as folder:
            p = Path(folder)/"test-only-lines"; p.write_bytes(b"invalid-json\n")
            counters = Counter()
            with m._CountedHistoryReader(p, encoding="raw", max_record_bytes=64, max_logical_bytes=64,
                    counters=counters, counter_key="decoded_history_bytes", checkpoint=lambda: None) as stream:
                rows = m._ObservedRows(stream, 2, lambda: None, counters=counters)
                with self.assertRaises(ValueError): rows.take()
            self.assertEqual(counters["decoded_history_bytes"], 13)
            self.assertEqual(counters["decoded_records"], 1)
            counters = Counter()
            with m._CountedHistoryReader(p, encoding="raw", max_record_bytes=2, max_logical_bytes=64,
                    counters=counters, counter_key="decoded_history_bytes", checkpoint=lambda: None) as stream:
                with self.assertRaises(ValueError): next(stream)
            self.assertEqual(counters["decoded_history_bytes"], 3)

    def test_interrupted_decoder_reports_unknown_tail_without_zero_total(self):
        from collections import Counter
        from types import SimpleNamespace
        import io
        m = self.modules["ReadCases"]; counters = Counter(); raw = io.BytesIO(b"test-only")
        class Failure:
            def readline(self, size):
                raw.read(3)
                raise OSError("test-only-decoder-failure")
        counted = m._ReadlineCounter(SimpleNamespace(_raw=raw), Failure(), counters, "decoded_history_bytes", lambda: None)
        with self.assertRaisesRegex(OSError, "test-only-decoder-failure"): counted.readline(8)
        measured = m.accounting(counters)
        self.assertIsNone(measured["decoded_history_bytes"]["exact_total"])
        self.assertFalse(measured["decoded_history_bytes"]["complete_measurement"])
        self.assertEqual(measured["history_encoded_scan_bytes"]["exact_total"], 3)

    def test_spool_failure_is_charged_before_json_or_checkpoint(self):
        from collections import Counter
        m = self.modules["ReadCases"]
        with tempfile.TemporaryDirectory(dir=self.fixture_directory) as folder:
            p = Path(folder)/"test-only-spool"; p.write_bytes(b"invalid-json\n")
            limits = {"window_bytes": 64, "spool_bytes": 64, "integer_io_policy": None}
            counters = Counter()
            with self.assertRaises(ValueError): list(m.windows(p, limits, lambda: None, counters))
            self.assertEqual(counters["spool_read_bytes"], 13)
            def stop(): raise RuntimeError("test-only-stop")
            counters = Counter()
            with self.assertRaisesRegex(RuntimeError, "test-only-stop"): list(m.windows(p, limits, stop, counters))
            self.assertEqual(counters["spool_read_bytes"], 13)

    def test_serialization_is_bounded_before_materializing_large_JSON(self):
        m = self.modules["ReadCases"]
        value = {"test_only": True, "low_word": F(1, 2**120)}
        self.assertIn(b"rational", m.serialized(value, None, 4096))
        with self.assertRaisesRegex(q.ComparisonError, "serialization_memory_cap"):
            m.serialized({"test_only": "x"*1000}, None, 512)

    def test_observed_controls_must_be_complete_and_bound(self):
        m = self.modules["Analyze"]
        segment = {"id": "test-only-segment", "start": 0, "end": 1}
        request = {"controls": {"rtol": 0.1, "atol": [0.02], "max_step": 0.25}, "segments": [segment]}
        row = {"segment_sha256": m.canonical(segment), "constructor_controls": request["controls"],
               "rtol_hex": 0.1.hex(), "atol_hex": [0.02.hex()], "observed_intervals": 1,
               **{k: "c"*64 for k in ("frame_identity", "constructor_record_sha256", "packet_chain_sha256")}}
        evidence = {"test_only": True, "complete_original_predicates": True, "request_sha256": m.canonical(request),
                    "history_sha256": "a"*64, "reader_sha256": "b"*64, "intervals": 1,
                    "observed_controls": {segment["id"]: row}}
        m.validate_control_observations(request, evidence, "a"*64, "b"*64)
        for key, value in (("history_sha256", "d"*64), ("reader_sha256", "d"*64),
                           ("complete_original_predicates", False), ("observed_controls", {})):
            bad = deepcopy(evidence); bad[key] = value
            with self.assertRaises(ValueError): m.validate_control_observations(request, bad, "a"*64, "b"*64)

        for key, value in (("rtol_hex", 0.01.hex()), ("atol_hex", [0.2.hex()]), ("observed_intervals", 0)):
            bad = deepcopy(evidence); bad["observed_controls"][segment["id"]][key] = value
            with self.assertRaises(ValueError): m.validate_control_observations(request, bad, "a"*64, "b"*64)

    def test_actual_control_observation_collector_rejects_unobserved_or_changed_weights(self):
        from types import SimpleNamespace
        import struct
        m = self.modules["ReadCases"]
        protocol = m._ExtractingProtocol.__new__(m._ExtractingProtocol)
        protocol.size, protocol.segment, protocol.control_observations = 2, {"id": "test-only"}, {}
        constructor = {"segment_sha256": m.reader._digest(protocol.segment),
                       "constructor_controls": {"rtol": 0.1, "atol": [0.2, 0.3]}}
        protocol.rows = SimpleNamespace(constructor_records={constructor["segment_sha256"]: constructor})
        weights = {"callback_calls": 0, "callback_status": 0, "installed": True, "setter": "IDAWFtolerances",
                   "setter_status": 0, "frame_identity": "a"*64, "rtol_hex": 0.1.hex(),
                   "atol": struct.pack("<2d", 0.2, 0.3)}
        sf = SimpleNamespace(latest_weights=weights)
        with self.assertRaisesRegex(q.ComparisonError, "observed_effective_controls_missing"):
            protocol.observe_controls(sf, "b"*64)
        weights["callback_calls"] = 1
        protocol.observe_controls(sf, "b"*64)
        protocol.observe_controls(sf, "c"*64)
        row = protocol.control_observations["test-only"]
        self.assertEqual(row["observed_intervals"], 2)
        self.assertEqual(row["atol_hex"], [0.2.hex(), 0.3.hex()])
        weights["rtol_hex"] = 0.01.hex()
        with self.assertRaisesRegex(q.ComparisonError, "changed_observed_controls"):
            protocol.observe_controls(sf, "d"*64)

    def test_metadata_read_error_keeps_prefix_and_unknown_tail(self):
        from unittest.mock import patch
        m = self.modules["Analyze"]
        previous_unknown = m.METADATA_IO_UNKNOWN
        with tempfile.TemporaryDirectory(dir=self.fixture_directory) as folder:
            p = Path(folder)/"test-only-io"; p.write_bytes(b"test-only-data")
            normal_open = Path.open
            class Interrupted:
                def __init__(self, stream): self.stream, self.calls = stream, 0
                def __enter__(self): return self
                def __exit__(self, *args): self.stream.close()
                def fileno(self): return self.stream.fileno()
                def read(self, size):
                    self.calls += 1
                    if self.calls == 1: return self.stream.read(2)
                    raise OSError("test-only-primary-IO-error")
            def opening(path, *args, **kwargs):
                stream = normal_open(path, *args, **kwargs)
                return Interrupted(stream) if path == p else stream
            before = m.METADATA_READ_BYTES
            try:
                with patch.object(Path, "open", opening):
                    with self.assertRaisesRegex(OSError, "test-only-primary-IO-error"):
                        m.pin(p)
                self.assertEqual(m.METADATA_READ_BYTES-before, 2)
                self.assertIsNone(m.metadata_accounting()["exact_read_bytes"])
            finally:
                m.METADATA_IO_UNKNOWN = previous_unknown


if __name__ == "__main__":
    unittest.main()
