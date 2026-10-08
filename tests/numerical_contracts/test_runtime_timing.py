"""Synthetic aggregate-clock and recording tests; no model/native/readback calls."""
import ast
from collections.abc import Mapping
import copy
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.benchmarks import native_runner as runner
from scripts.benchmarks.native_history import emit_record
from scripts.benchmarks.runtime_timing import PHASES, RuntimeTiming

ADMISSION_PREFIX_OUTCOMES = []


class FakeClock:
    def __init__(self, scale=1):
        self.now = 0
        self.scale = scale

    def __call__(self):
        return self.now

    def advance(self, amount):
        self.now += amount * self.scale


class RuntimeTimingTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp(prefix="timing-", dir=os.environ.get("SOLARLAB_TIMING_TEST_ROOT")))
        if os.environ.get("SOLARLAB_TIMING_KEEP_FIXTURES") != "1":
            self.addCleanup(shutil.rmtree, self.folder)

    def read_timing(self, folder=None):
        return json.loads(((folder or self.folder) / "RuntimeTiming.json").read_text())

    def assert_partition(self, result):
        phases = result["phases"]
        self.assertEqual(tuple(phases), PHASES)
        self.assertEqual(result["partition_exclusive_ns"], phases["recorded_run"]["inclusive_ns"])
        for row in phases.values():
            for name in ("calls", "exceptions", "inclusive_ns", "exclusive_ns"):
                self.assertIs(type(row[name]), int)
                self.assertGreaterEqual(row[name], 0)
        self.assertLess(len(json.dumps(result).encode()), 16384)

    def test_nested_and_disjoint_spans_partition_without_double_counting(self):
        clock = FakeClock()
        timing = RuntimeTiming(clock=clock)
        with timing.phase("recorded_run"):
            clock.advance(2)
            with timing.phase("controller"):
                clock.advance(3)
                with timing.phase("ida_calls"):
                    clock.advance(7)
                    with timing.phase("history_io"):
                        clock.advance(11)
                    clock.advance(13)
                clock.advance(17)
                with timing.phase("observation"):
                    clock.advance(19)
                    with timing.phase("history_io"):
                        clock.advance(23)
                    clock.advance(29)
                clock.advance(31)
            clock.advance(37)
            with timing.phase("artifact_io"):
                clock.advance(41)
            clock.advance(43)
        result = timing.snapshot("returned_completed")
        self.assert_partition(result)
        self.assertEqual(result["partition_exclusive_ns"], 276)
        self.assertEqual(result["phases"]["controller"]["exclusive_ns"], 51)
        self.assertEqual(result["phases"]["ida_calls"]["inclusive_ns"], 31)
        self.assertEqual(result["phases"]["ida_calls"]["exclusive_ns"], 20)
        self.assertEqual(result["phases"]["history_io"]["calls"], 2)

    def test_exception_closes_each_span_and_preserves_exception_identity(self):
        clock = FakeClock()
        timing = RuntimeTiming(clock=clock)
        sentinel = RuntimeError("synthetic callback failure")
        with self.assertRaises(RuntimeError) as caught:
            with timing.phase("recorded_run"):
                clock.advance(5)
                with timing.phase("ida_calls"):
                    clock.advance(7)
                    raise sentinel
        self.assertIs(caught.exception, sentinel)
        result = timing.snapshot("raised")
        self.assert_partition(result)
        self.assertEqual(result["phases"]["ida_calls"]["exceptions"], 1)
        self.assertEqual(result["phases"]["recorded_run"]["exceptions"], 1)
        self.assertFalse(result["phases"]["observation"]["entered"])

    def test_unknown_recursive_and_open_snapshot_are_rejected(self):
        timing = RuntimeTiming(clock=FakeClock())
        with self.assertRaises(ValueError):
            with timing.phase("unregistered"):
                self.fail("unsupported phase entered")
        with timing.phase("recorded_run"):
            with self.assertRaises(ValueError):
                with timing.phase("recorded_run"):
                    self.fail("recursive phase entered")
            with self.assertRaises(ValueError):
                timing.snapshot("returned_completed")
        self.assert_partition(timing.snapshot("returned_completed"))

    def synthetic_controller(self, timing, clock, events, *, failure=None, returned_failure=False):
        def controller(writer):
            events.append("controller")
            with timing.phase("ida_calls"):
                events.append("fake_ida")
                clock.advance(2)
                with timing.phase("history_io"):
                    emit_record({"kind": "synthetic_timing_fixture", "value": 7}, writer,
                                logical_bytes=0, total_output_bytes=2**21)
                    events.append("history")
                    clock.advance(3)
                clock.advance(5)
                if failure is not None:
                    raise failure
            with timing.phase("observation"):
                events.append("fake_observation")
                clock.advance(11)
            clock.advance(13)
            result = {"status": runner.COMPLETED_STATUS, "diagnostic": {"exact_input": [0, None, -7]},
                      "counts": {"history_bytes": writer.logical_bytes}, "complete_protocol": True}
            if returned_failure:
                result.update(status="failed_bounded_voltage_lift_native_pilot", complete_protocol=False,
                              first_failure={"phase": "synthetic", "reason": "retained failure"})
            return result
        return controller

    def execute(self, folder, *, scale=1, failure=None, returned_failure=False):
        folder.mkdir(exist_ok=True)
        clock = FakeClock(scale)
        timing = RuntimeTiming(clock=clock)
        events = []
        result = runner.run_recorded(folder, self.synthetic_controller(
            timing, clock, events, failure=failure, returned_failure=returned_failure),
            total_output_bytes=2**21, input_names=(), timing=timing)
        return result, events

    def test_clock_values_do_not_change_controller_result_or_call_order(self):
        first, events = self.execute(self.folder / "first", scale=1)
        second, other_events = self.execute(self.folder / "second", scale=1000)
        self.assertEqual(first, second)
        self.assertEqual(events, ["controller", "fake_ida", "history", "fake_observation"])
        self.assertEqual(events, other_events)
        a = self.read_timing(self.folder / "first")
        b = self.read_timing(self.folder / "second")
        self.assert_partition(a)
        self.assert_partition(b)
        self.assertEqual(b["partition_exclusive_ns"], a["partition_exclusive_ns"] * 1000)
        self.assertFalse(a["scientific_decisions_use_timing"])

    def test_controller_exception_publishes_totals_and_original_failure(self):
        sentinel = RuntimeError("retained synthetic failure")
        with self.assertRaises(RuntimeError) as caught:
            self.execute(self.folder, failure=sentinel)
        self.assertIs(caught.exception, sentinel)
        result = self.read_timing()
        self.assert_partition(result)
        self.assertEqual(result["recorded_call_outcome"], "raised")
        self.assertEqual(result["phases"]["ida_calls"]["inclusive_ns"], 10)
        self.assertEqual(result["phases"]["ida_calls"]["exclusive_ns"], 7)
        self.assertEqual(json.loads((self.folder / "RunnerFailure.json").read_text())["reason"], str(sentinel))

    def test_returned_controller_failure_keeps_its_status(self):
        result, _ = self.execute(self.folder, returned_failure=True)
        self.assertFalse(result["complete_protocol"])
        timing = self.read_timing()
        self.assert_partition(timing)
        self.assertEqual(timing["recorded_call_outcome"], "returned_other_status")
        self.assertTrue((self.folder / "PrimaryFailure.json").exists())

    def test_history_finalization_exception_retains_aggregate(self):
        sentinel = OSError("synthetic footer refusal")
        with patch.object(runner.HistoryWriter, "finish", side_effect=sentinel):
            with self.assertRaises(OSError) as caught:
                self.execute(self.folder)
        self.assertIs(caught.exception, sentinel)
        timing = self.read_timing()
        self.assert_partition(timing)
        self.assertEqual(timing["phases"]["history_io"]["exceptions"], 1)
        self.assertFalse((self.folder / "NativeResult.json").exists())

    def test_timing_publication_error_does_not_mask_primary_exception(self):
        original = runner._write_document
        sentinel = RuntimeError("original synthetic failure")
        def publish(folder, name, *args, **kwargs):
            if name == "RuntimeTiming.json":
                raise OSError("synthetic timing publication refusal")
            return original(folder, name, *args, **kwargs)
        with patch.object(runner, "_write_document", side_effect=publish):
            with self.assertRaises(RuntimeError) as caught:
                self.execute(self.folder, failure=sentinel)
        self.assertIs(caught.exception, sentinel)
        self.assertTrue(any("aggregate timing publication failed" in note for note in sentinel.__notes__))
        self.assertTrue((self.folder / "RunnerFailure.json").exists())
        self.assertFalse((self.folder / "RuntimeTiming.json").exists())

    def test_existing_timing_artifact_refuses_new_controller_and_is_unchanged(self):
        old = b'{"old_timing_evidence":true}\n'
        (self.folder / "RuntimeTiming.json").write_bytes(old)
        events = []
        with self.assertRaises(FileExistsError):
            runner.run_recorded(self.folder, lambda writer: events.append("must_not_run"),
                                total_output_bytes=2**21, input_names=())
        self.assertEqual(events, [])
        self.assertEqual((self.folder / "RuntimeTiming.json").read_bytes(), old)

    def test_timing_binding_and_original_admission_guards_without_model(self):
        # Execute only the production admission prefix, stopping before its
        # observation imports; material validation/sampling/policy construction
        # are explicit metadata stubs. Never import or call the full controller.
        controller_path = Path(runner.__file__).with_name("coupled_device_prototype.py")
        policy_path = Path(sys.modules[runner.HistoryWriter.__module__].__file__).with_name("native_observation.py")
        timing_path = Path(sys.modules[RuntimeTiming.__module__].__file__).resolve()
        controller = next(node for node in ast.parse(controller_path.read_text()).body
                          if isinstance(node, ast.FunctionDef) and node.name == "run_voltage_lift_native_pilot")
        stop = next(i for i, node in enumerate(controller.body)
                    if isinstance(node, ast.ImportFrom) and node.module == "scripts.benchmarks.native_observation")
        controller.body = controller.body[:stop] + ast.parse("return metadata_policy(request, admission)").body
        controller.name = "metadata_prefix"
        controller.returns = None
        for arg in controller.args.args + controller.args.posonlyargs + controller.args.kwonlyargs:
            arg.annotation = None
        policy = next(node for node in ast.parse(policy_path.read_text()).body
                      if isinstance(node, ast.FunctionDef) and node.name == "validate_interval_observation_admission")
        class MetadataRejection(ValueError):
            pass
        digest = lambda value: sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        policy_scope = {"__file__": str(policy_path), "Mapping": Mapping, "Path": Path,
                        "sha256": sha256, "digest": digest, "ContractError": MetadataRejection,
                        "prepare_interval_observation_policy": lambda request, **kwargs: copy.deepcopy(request["interval_observation"])}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[policy], type_ignores=[])),
                     "<timing-policy-metadata-only>", "exec"), policy_scope)
        scope = {"__file__": str(controller_path), "RuntimeTiming": RuntimeTiming,
                 "Path": Path, "sys": sys, "sha256": sha256, "digest": digest,
                 "ContractError": MetadataRejection,
                 "validate_voltage_lift_native_request": lambda *args: None,
                 "AffineSamplingContext": lambda model, request: SimpleNamespace(
                     request_sha256=digest(request), request_copy=lambda: copy.deepcopy(request), segments=()),
                 "metadata_policy": policy_scope["validate_interval_observation_admission"]}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[controller], type_ignores=[])),
                     "<timing-controller-admission-prefix-only>", "exec"), scope)
        paths = {controller_path.resolve(), timing_path,
                 *(policy_path.with_name(name).resolve() for name in
                   ("native_observation.py", "interval_observation.py", "coupled_device_prototype.py"))}
        pins = {str(path): sha256(path.read_bytes()).hexdigest() for path in paths}
        cases = [
            ("missing_timing_pin", False, False, "runtime_timing_executing_source_not_bound"),
            ("wrong_timing_pin", True, False, "voltage_lift_native_source_changed"),
            ("bound_without_policy", True, False, "full_word_native_observation_policy_unqualified"),
            ("bound_policy_without_review", True, True, "native_observation_independent_review_missing"),
        ]
        for label, bound, with_policy, expected in cases:
            with self.subTest(case=label):
                request = {"synthetic_metadata_only": True, "segments": []}
                admission = {"map_identity": "synthetic-map", "voltage_lift_native_authorized": True,
                             "coordinator_message": "msg_unit_metadata_no_native_authority", "source_sha256": dict(pins)}
                if not bound:
                    admission["source_sha256"].pop(str(timing_path))
                if label == "wrong_timing_pin":
                    admission["source_sha256"][str(timing_path)] = "0" * 64
                if with_policy:
                    request["interval_observation"] = {"binding_identity": "a" * 64,
                        "header_sha256": "b" * 64, "backend_modules": {}}
                    admission.update(interval_observation_authorized=True,
                                     interval_observation_policy_sha256=digest(request["interval_observation"]))
                admission["request_sha256"] = digest(request)
                unchanged = copy.deepcopy(request)
                with self.assertRaisesRegex(MetadataRejection, expected):
                    scope["metadata_prefix"](SimpleNamespace(model=None, identity="synthetic-map"), (),
                                             request, admission, lambda record: self.fail("unadmitted output"))
                self.assertEqual(request, unchanged)
                ADMISSION_PREFIX_OUTCOMES.append({"case": label, "expected_rejection": expected,
                                                  "full_controller_or_model_executed": False})


if __name__ == "__main__":
    unittest.main()
