"""Bounded protocol/current/operator tests; no device integration is performed."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


REPO = Path(__file__).resolve().parents[2]
ARCHIVE = Path("/Users/bytedance/Library/Mobile Documents/com~apple~CloudDocs/projects/solarlab")
SOURCE = REPO/"scripts/benchmarks/hysteresis_reference.py"
spec = importlib.util.spec_from_file_location("hysteresis_reference_test_subject", SOURCE)
hr = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = hr
spec.loader.exec_module(hr)


class RecordingAdapter:
    """Analytic synthetic fixture only; seven arbitrary carried state blocks."""

    def __init__(self):
        self.initial_calls = 0
        self.calls = []

    def initial_state(self):
        self.initial_calls += 1
        return tuple(range(1, 8))

    def advance(self, state, control):
        self.calls.append((control.binding(), state))
        end = tuple(x+control.duration for x in state)
        return hr.Segment((0., control.duration), (state, end), True, {"fixture": "analytic_synthetic"})

    def observe(self, state, phase, physical_time, side):
        t = float(physical_time)
        v = phase.voltage(physical_time)
        return {"phase_id": phase.id, "time_s": t, "event_side": side,
                "voltage_V": v, "voltage_rate_V_s": phase.voltage_rate,
                "state": list(state), "junction_polarity": 1,
                "D_C_m2": [t*t+v]*2, "Ddot_A_m2": [2*t+phase.voltage_rate]*2,
                "J_n_A_m2": [1., 1.], "J_p_A_m2": [2., 2.],
                "J_ion_A_m2": [0., 0.], "J_cond_A_m2": [3., 3.]}


class FrozenFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = hr.checked_json(REPO/hr.CONTRACT, hr.CONTRACT_SHA256)
        cls.gates = hr.checked_json(REPO/hr.ANALYTIC, hr.ANALYTIC_SHA256)
        cls.manifest = json.loads((REPO/"reproducibility/RefactorReferenceManifestV1.json").read_text())
        cls.request = cls.manifest["hi"]["nominal_request"]

    def timeline(self, rate=.04, level=0):
        request = copy.deepcopy(self.request)
        request["params"]["v_rate"] = rate
        return hr.make_timeline(self.contract, request, level)


class TestSourcesAndInput(FrozenFixture):
    def test_frozen_gates_are_actual_sources(self):
        self.assertEqual(self.contract["status"], "frozen_standard_reference_qualification_pending")
        self.assertEqual(self.gates["status"], "frozen")
        self.assertIn("NC06", [x["id"] for x in self.gates["gates"]])

    def test_complete_45_records_and_43_potential_requests(self):
        cases = self.manifest["hi"]["cases"]
        self.assertEqual(len(cases), 45)
        self.assertEqual(len({x["request_sha256"] for x in cases}), 43)
        self.assertEqual(hr.digest(cases), self.contract["reference_set"]["cases_payload_sha256"])
        self.assertEqual([x["id"] for x in cases], self.contract["reference_set"]["required_case_ids"])

    def test_only_registered_coordinate_changes_reproduce_all_saved_request_hashes(self):
        for case in self.manifest["hi"]["cases"]:
            with self.subTest(case=case["id"]):
                request = copy.deepcopy(self.request)
                absorber = next(x for x in request["device"]["layers"] if x["role"] == "absorber")
                absorber["D_ion"], absorber["P0"] = case["D_ion_m2_s"], case["c0_m3"]
                request["params"]["v_rate"] = case["scan_rate_V_s"]
                self.assertEqual(hr.digest(request), case["request_sha256"])

    def test_exact_archived_requests_distinguish_D0_and_c0(self):
        d0 = hr.load_inputs(REPO, ARCHIVE, "hi_diffusivity_00")
        c0 = hr.load_inputs(REPO, ARCHIVE, "hi_density_00")
        for bundle in (d0, c0):
            self.assertEqual(bundle["request"]["params"]["waveform_controls"], {"atol_m3": 1, "rtol": .0001})
            self.assertEqual(bundle["request"]["params"]["n_points"], 111)
        d_layer = d0["request"]["device"]["layers"][1]
        c_layer = c0["request"]["device"]["layers"][1]
        self.assertEqual((d_layer["D_ion"], d_layer["P0"]), (0, 1e25))
        self.assertEqual((c_layer["D_ion"], c_layer["P0"]), (2.585e-18, 0))
        self.assertNotEqual(d0["case"]["request_sha256"], c0["case"]["request_sha256"])

    def test_nominal_saved_request_is_bound(self):
        bundle = hr.load_inputs(REPO, ARCHIVE, "hi_rate_40")
        self.assertEqual(bundle["request"], self.request)
        self.assertEqual(bundle["coverage"]["records"], 45)

    def test_unknown_case_rejected(self):
        with self.assertRaisesRegex(hr.ReferenceError, "45 original"):
            hr.load_inputs(REPO, ARCHIVE, "unregistered_shorter_protocol")

    def test_wrong_frozen_source_digest_rejected(self):
        with self.assertRaisesRegex(hr.ReferenceError, "source mismatch"):
            hr.checked_json(REPO/hr.CONTRACT, "0"*64)


class TestTimeline(FrozenFixture):
    def test_exact_physical_phases_for_all_three_rates(self):
        for rate, end in ((.02, 374), (.04, 264), (.08, 209)):
            with self.subTest(rate=rate):
                phases = self.timeline(rate)
                self.assertEqual(tuple(p.id for p in phases), hr.PHASE_IDS)
                self.assertEqual([len(p.controls) for p in phases], [1, 1, 1, 440, 1, 1, 440])
                self.assertEqual(phases[0].start, 0)
                self.assertEqual(phases[-1].stop, end)
                self.assertEqual([float(p.stop-p.start) for p in phases[:3]], [120., 30., .5])
                self.assertEqual(phases[4].stop-phases[4].start, 3)
                self.assertEqual(phases[5].stop-phases[5].start, Fraction(1, 2))

    def test_every_restart_has_local_zero_and_exact_continuity(self):
        for phase in self.timeline():
            for index, c in enumerate(phase.controls):
                self.assertEqual(c.voltage(0), float(c.v_start))
                self.assertEqual(c.voltage(c.duration), float(c.v_stop))
                self.assertEqual(c.binding()["local_time_origin_s"], 0)
                if index:
                    before = phase.controls[index-1]
                    self.assertEqual(before.stop, c.start)
                    self.assertEqual(before.v_stop, c.v_start)

    def test_literal_caps_do_not_follow_observation_count(self):
        for level, expected in enumerate((.00625, .003125, .0015625)):
            phases = self.timeline(level=level)
            self.assertEqual(phases[3].controls[0].max_step_s, expected)
            self.assertEqual(phases[6].controls[-1].max_step_s, expected)
            self.assertEqual(phases[0].controls[0].max_step_s, (6., 3., 1.5)[level])
            self.assertTrue(all(c.max_rhs_calls == 100000 for p in phases for c in p.controls))

    def test_only_max_step_changes_across_time_levels(self):
        schedules = []
        for level in range(3):
            rows = [c.binding() for p in self.timeline(level=level) for c in p.controls]
            for row in rows:
                del row["max_step_s"]
            schedules.append(rows)
        self.assertEqual(schedules[0], schedules[1])
        self.assertEqual(schedules[1], schedules[2])

    def test_observation_grids_are_exact_subsets(self):
        for count, step in ((111, 4), (221, 2), (441, 1)):
            indices = hr.observation_indices(count)
            self.assertEqual(len(indices), count)
            self.assertEqual(indices, tuple(range(0, 441, step)))

    def test_unsupported_or_coerced_observations_fail(self):
        for count in (True, 111., 112, 442, 0, -1):
            with self.subTest(count=count), self.assertRaises(hr.ReferenceError):
                hr.observation_indices(count)

    def test_changed_physical_protocol_and_rate_fail(self):
        for changes in ({"dark_seed_s": 119}, {"turnaround_s": 0}, {"branch_dwell_s": 0},
                        {"uniform_generation_rate_m3_s": 0}, {"start_voltage_V": 0}):
            request = copy.deepcopy(self.request)
            request["params"]["waveform"].update(changes)
            with self.subTest(changes=changes), self.assertRaises(hr.ReferenceError):
                hr.make_timeline(self.contract, request, 0)
        with self.assertRaises(hr.ReferenceError):
            self.timeline(rate=.03)

    def test_invalid_clock_or_step_level_rejected(self):
        for level in (True, -1, 3, 1.):
            with self.assertRaises(hr.ReferenceError):
                self.timeline(level=level)
        control = self.timeline()[3].controls[0]
        for t in (-1e-8, control.duration+1e-8, float("nan")):
            with self.assertRaises(hr.ReferenceError):
                control.voltage(t)

    def test_phase_gap_or_missing_phase_rejected(self):
        for change in ("gap", "missing"):
            contract = copy.deepcopy(self.contract)
            profile = contract["refinement_plan"]["future_decoupled_reference_driver"]["phase_profiles"][1]
            if change == "gap":
                profile["phases"][1]["start_s"] += 1
            else:
                profile["phases"].pop()
            with self.subTest(change=change), self.assertRaises(hr.ReferenceError):
                hr.make_timeline(contract, self.request, 0)


class TestHistoryAndObservations(FrozenFixture):
    def test_full_state_and_material_history_are_carried_once(self):
        adapter = RecordingAdapter()
        segments, events = [], []
        phases = self.timeline()
        history = hr.collect_history(adapter, phases, lambda c, s: segments.append(s), events.append)
        self.assertEqual(adapter.initial_calls, 1)
        self.assertEqual(len(adapter.calls), 885)
        self.assertEqual(history["initial_state"], tuple(range(1, 8)))
        self.assertEqual(history["phases"]["reverse_ramp"]["states"][-1], tuple(i+264 for i in range(1, 8)))
        self.assertEqual(len(history["events"]), 6)
        self.assertEqual(events[0]["kind"], "initial_state")
        self.assertEqual([e["kind"] for e in events if "kind" in e].count("phase_end"), 7)
        for before, after in zip(segments, segments[1:]):
            self.assertEqual(before.states[-1], after.states[0])

    def test_all_output_grids_reuse_one_integration_history(self):
        adapter = RecordingAdapter()
        history = hr.collect_history(adapter, self.timeline(), lambda *x: None, lambda x: None)
        identity = hr.digest(adapter.calls)
        all_outputs = [hr.observe_branches(adapter, history, n) for n in (111, 221, 441)]
        self.assertEqual(len(adapter.calls), 885)
        self.assertEqual(hr.digest(adapter.calls), identity)
        for count, output in zip((111, 221, 441), all_outputs):
            for direction in ("forward", "reverse"):
                self.assertEqual(len(output["instantaneous_ramp_side_v1"][direction]), count)
                self.assertEqual(len(output["legacy_interval_endpoint"][direction]), 111)
        self.assertEqual(all_outputs[0]["legacy_interval_endpoint"], all_outputs[2]["legacy_interval_endpoint"])

    def test_first_point_is_present_and_named_not_smoothed(self):
        adapter = RecordingAdapter()
        history = hr.collect_history(adapter, self.timeline(), lambda *x: None, lambda x: None)
        output = hr.observe_branches(adapter, history, 111)
        instant = output["instantaneous_ramp_side_v1"]["forward"][0]
        legacy = output["legacy_interval_endpoint"]["forward"][0]
        self.assertEqual(instant["time_s"], 150.5)
        self.assertEqual(instant["event_side"], "right")
        self.assertAlmostEqual(legacy["left_solar_current_A_m2"]-instant["left_solar_current_A_m2"], .54, places=9)
        self.assertIn("missing", legacy["historical_reference"])
        self.assertIsNone(instant["uncertainty_A_m2"])

    def test_failure_preserves_raw_result_and_does_not_retry(self):
        adapter = RecordingAdapter()
        adapter.advance = Mock(return_value=hr.Segment((), (), False, {"message": "budget stop"}))
        saved = []
        with self.assertRaisesRegex(hr.ReferenceError, "budget stop"):
            hr.collect_history(adapter, self.timeline(), lambda c, s: saved.append(s), lambda x: None)
        self.assertEqual(len(saved), 1)
        self.assertEqual(adapter.advance.call_count, 1)

    def test_reset_or_truncated_state_is_rejected(self):
        for reset in (True, False):
            adapter = RecordingAdapter()
            def bad(state, c):
                initial = tuple(x+1 for x in state) if reset else state
                final = state if reset else state[:-1]
                return hr.Segment((0., c.duration), (initial, final), True, {})
            adapter.advance = bad
            with self.subTest(reset=reset), self.assertRaises(hr.ReferenceError):
                hr.collect_history(adapter, self.timeline(), lambda *x: None, lambda x: None)

    def test_interval_current_cannot_cross_a_physical_event(self):
        adapter = RecordingAdapter()
        phases = self.timeline()
        before = adapter.observe((1,), phases[0], phases[0].stop, "left")
        after = adapter.observe((1,), phases[1], phases[1].stop, "left")
        with self.assertRaisesRegex(hr.ReferenceError, "one physical phase"):
            hr.current_value(after, previous=before)

    def test_event_charge_is_saved_without_finite_impulse_current(self):
        adapter = RecordingAdapter()
        phases = self.timeline()
        before = adapter.observe((1, 2), phases[0], phases[0].stop, "left")
        after = adapter.observe((1, 2), phases[1], phases[1].start, "right")
        event = hr.event_record(before, after)
        self.assertEqual(event["delta_D_C_m2"], [-1., -1.])
        self.assertEqual(event["electrode_impulse_charge_C_m2"], {"left": -1., "right": 1.})
        self.assertIsNone(event["impulse_current_A_m2"])
        after["state"] = [1, 3]
        with self.assertRaises(hr.ReferenceError):
            hr.event_record(before, after)

    def test_zero_power_is_unknown_in_each_named_HI_definition(self):
        row = {"voltage_V": 0., "left_solar_current_A_m2": 0.}
        data = {"instantaneous_ramp_side_v1": {"forward": [row], "reverse": [row]}}
        metrics = hr.power_summary(data, "instantaneous_ramp_side_v1")
        self.assertIsNone(metrics["HI_P"])
        self.assertIsNone(metrics["HI_normalized"])
        self.assertEqual(metrics["reference_power_precondition"], "unknown")


class TestThinBridgeAndAnalyticCurrent(FrozenFixture):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        sys.path.insert(0, str(REPO/"perovskite-sim"))

    def test_bridge_requests_raw_accepted_mesh_and_literal_cap(self):
        import numpy as np
        bridge = hr.ProductionBridge.__new__(hr.ProductionBridge)
        control = self.timeline()[3].controls[0]
        state = (1., 2., 3., 4., 0., 0.)
        solve = Mock(return_value=SimpleNamespace(success=True, t=np.array([0., control.duration]),
            y=np.array([state, state]).T, message="synthetic", numerical_diagnostics=None))
        bridge.np, bridge.jv = np, SimpleNamespace(run_transient=solve)
        bridge.x, bridge.stack = np.array([0., 1.]), object()
        bridge.mat = SimpleNamespace(P_ion0=np.zeros(2))
        bridge.widths, bridge.initial_inventory = np.array([.5, .5]), 0.
        bridge.jacobian = lambda *args: None
        bridge.numerics = {"rtol": 1e-4, "atol_m3": 1}
        with patch.object(hr, "radau_history", return_value=(object, [0., control.duration], [state, state])):
            result = bridge.advance(state, control)
        self.assertTrue(result.success)
        kwargs = solve.call_args.kwargs
        self.assertIsNone(kwargs["t_eval"])
        self.assertEqual(kwargs["max_step"], .00625)
        self.assertEqual(kwargs["max_nfev"], 100000)
        self.assertEqual(kwargs["t_span"], (0., .125))
        self.assertEqual(kwargs["atol"], 1)

    def test_inventory_failure_inside_a_control_interval_is_retained(self):
        import numpy as np
        bridge = hr.ProductionBridge.__new__(hr.ProductionBridge)
        control = self.timeline()[3].controls[0]
        start = (1., 2., 3., 4., 2., 2.)
        # Endpoint inventory recovers: inspecting endpoints alone would miss this.
        middle = (1., 2., 3., 4., 3., 3.)
        bridge.np = np
        bridge.x, bridge.stack = np.array([0., 1.]), object()
        bridge.mat = SimpleNamespace(P_ion0=np.full(2, 2.))
        bridge.widths, bridge.initial_inventory = np.array([.5, .5]), 2.
        bridge.numerics = {"rtol": 1e-4, "atol_m3": 1}
        bridge.jacobian = lambda *args: None
        bridge.jv = SimpleNamespace(run_transient=Mock(return_value=SimpleNamespace(
            success=True, t=np.array([0., .0625, .125]), y=np.array([start, middle, start]).T,
            message="synthetic", numerical_diagnostics=None)))
        with patch.object(hr, "radau_history", return_value=(object, [0., .0625, .125], [start, middle, start])):
            result = bridge.advance(start, control)
        self.assertFalse(result.success)
        self.assertEqual(result.states[1], middle)
        self.assertEqual(result.diagnostics["max_inventory_relative_drift_over_accepted_knots"], .5)

    def test_frozen_NC06_capacitor_ramps_and_two_orientations(self):
        import numpy as np
        from perovskite_sim.constants import EPS_0
        from perovskite_sim.physics.poisson import factor_poisson
        gate = next(x for x in self.gates["gates"] if x["id"] == "NC06")
        case = gate["fixed_inputs"]["capacitor"]
        x = np.linspace(0., case["length"], 5)
        factor = factor_poisson(x, np.full_like(x, case["epsilon"]/EPS_0))
        for slope in case["slopes"]:
            for orientation in (-1, 1):
                with self.subTest(slope=slope, orientation=orientation):
                    _, Ddot, residual = hr.differentiated_poisson(factor, np.zeros_like(x), -orientation*slope)
                    actual = case["area"]*Ddot
                    expected = orientation*case["C"]*slope
                    np.testing.assert_allclose(actual, expected, atol=gate["algebraic_atol"], rtol=gate["algebraic_rtol"])
                    np.testing.assert_allclose(residual, 0., atol=gate["algebraic_atol"], rtol=0.)

    def test_differentiated_poisson_includes_full_charge_rate(self):
        import numpy as np
        from perovskite_sim.constants import EPS_0
        from perovskite_sim.physics.poisson import factor_poisson
        x = np.linspace(0., 1., 9)
        factor = factor_poisson(x, np.ones_like(x)/EPS_0)
        # epsilon=1, rho_dot=2 gives phi_dot=x*(1-x), Ddot=2*x_face-1.
        phi, Ddot, residual = hr.differentiated_poisson(factor, np.full_like(x, 2.), 0.)
        np.testing.assert_allclose(phi, x*(1-x), atol=1e-13, rtol=1e-12)
        np.testing.assert_allclose(Ddot, x[1:]+x[:-1]-1, atol=1e-13, rtol=1e-12)
        np.testing.assert_allclose(residual, 0., atol=1e-13, rtol=0.)


class TestAdmission(unittest.TestCase):
    def test_missing_or_mismatched_admission_never_executes(self):
        plan = {"identity_sha256": "a"*64, "resources": {"wall_s": 600},
                "environment": {"model_environment": {k: "1" for k in hr.THREAD_KEYS}}}
        with self.assertRaises(hr.ReferenceError):
            hr.verify_admission(plan, {})
        admitted = {"schema": "solarlab.hysteresis_reference_admission.v1", "execution_authorized": True,
                    "identity_sha256": plan["identity_sha256"], "resources": plan["resources"],
                    "root_authorization_message": "synthetic_test_only", "bound_analytic_test_receipt_sha256": "b"*64,
                    "purpose": "unqualified_reference_diagnostic"}
        hr.verify_admission(plan, admitted)
        for change in ({"identity_sha256": "c"*64}, {"execution_authorized": False},
                       {"resources": {"wall_s": 900}}, {"purpose": "candidate_acceptance"},
                       {"bound_analytic_test_receipt_sha256": None}):
            with self.subTest(change=change), self.assertRaises(hr.ReferenceError):
                hr.verify_admission(plan, {**admitted, **change})
        plan["environment"]["model_environment"]["OMP_NUM_THREADS"] = "8"
        with self.assertRaisesRegex(hr.ReferenceError, "thread"):
            hr.verify_admission(plan, admitted)


if __name__ == "__main__":
    unittest.main()
