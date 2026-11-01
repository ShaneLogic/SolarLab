"""Focused source and separately admitted native initial-weight checks."""
from collections.abc import Mapping
from hashlib import sha256
import ast
import gc
import importlib
import json
import os
from pathlib import Path
import struct
import unittest

ROOT = Path(__file__).resolve().parents[2]
BINDING = ROOT / "scripts/benchmarks/ida_observation"


def frozen(value):
    if isinstance(value, Mapping):
        return tuple(sorted((key, frozen(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(map(frozen, value))
    if type(value) is float:
        return struct.pack("<d", value)
    return value


class TestInitialWeightSource(unittest.TestCase):
    def test_pins_and_real_callback_route(self):
        pins = json.loads((BINDING / "SourcePinsV1.json").read_text())
        for name, key in (("scikit_sundae_1_1_3.patch", "patch_sha256"),
                          ("ida_observation_copy.h", "copy_header_sha256")):
            self.assertEqual(sha256((BINDING / name).read_bytes()).hexdigest(), pins[key])
        self.assertEqual(pins["initial_parent_weight_evaluation"]["allowed_native_steps"], 0)
        header = (BINDING / "ida_observation_copy.h").read_text()
        function = header.split("static int sl_ida75_evaluate_initial_weights(", 1)[1]
        self.assertIn("m->ida_efun(m->ida_phi[0], output, m->ida_user_data)", function)
        self.assertIn("memcmp(record, m, sizeof(*m))", function)
        self.assertIn("m->ida_ewt", function)
        self.assertNotIn("IDASolve(", function)
        self.assertNotIn("IDAReInit(", function)

    def test_prepared_public_api_is_callable_source(self):
        path = os.environ.get("ERROR_WEIGHT_PREPARED_SOURCE")
        if not path:
            self.skipTest("Explicit source preparation required; native code is not imported.")
        prepared = Path(path)
        pins = json.loads((BINDING / "SourcePinsV1.json").read_text())
        for name, entry in pins["files"].items():
            self.assertEqual(sha256((prepared / name).read_bytes()).hexdigest(),
                             entry["patched_sha256"])
        tree = ast.parse((prepared / "src/sksundae/ida/_solver.py").read_text())
        methods = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        self.assertIn("evaluate_initial_parent_weights", methods)


@unittest.skipUnless(os.environ.get("SOLARLAB_IDA_OBSERVATION_NATIVE_TEST") == "1",
                     "A new immutable build and explicit native admission are required.")
class TestInitialWeightNative(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        expected = os.environ.get("ERROR_WEIGHT_EXPECTED_INSTALL")
        if not expected:
            raise RuntimeError("The admitted complete installation must be explicit.")
        cls.numpy = importlib.import_module("numpy")
        module = importlib.import_module("sksundae._cy_ida")
        actual = Path(module.__file__).resolve()
        if not actual.is_relative_to(Path(expected).resolve()):
            raise RuntimeError("Refuse an old or unbound native extension.")
        cls.IDA = importlib.import_module("sksundae.ida").IDA
        if not callable(getattr(cls.IDA, "evaluate_initial_parent_weights", None)):
            raise RuntimeError("Admitted binding lacks the required actual callback API.")

    @staticmethod
    def frame(size=2, time=0.5):
        high = [2.0, -3.0][:size]
        zeros = [0.0] * size
        return {"schema": "sksundae.ida.parent-affine-weights.v1",
                "frame_identity": "6" * 64, "size": size, "t0_hex": time.hex(),
                "q0_words_hex": [[v.hex() for v in high],
                                 [(2.0**-55).hex()] * size,
                                 [v.hex() for v in zeros], [v.hex() for v in zeros]],
                "v0_words_hex": [[v.hex() for v in zeros] for _ in range(4)],
                "weight_rounding": "RN-exact-parent-q_then-unfused-SV-v1"}

    @staticmethod
    def residual(t, y, yp, output):
        output[:] = yp + y

    def test_actual_initial_state_weights_and_generation(self):
        from fractions import Fraction
        np = self.numpy
        initial, rate = np.array([0.25, 1.5]), np.array([-0.25, -1.5])
        frame = self.frame()
        atol = np.array([1e-8, 2e-8])
        fail_first_attempt = [False]

        def residual(t, y, yp, output):
            if fail_first_attempt[0]:
                raise RuntimeError("declared failed first native attempt")
            self.residual(t, y, yp, output)

        solver = self.IDA(residual, rtol=1e-6, atol=atol,
                          parent_weight_frame=frame)
        try:
            solver.init_step(0.5, initial, rate)
            before = solver.statistics()
            for expected_calls in (1, 2):
                record = solver.evaluate_initial_parent_weights()
                self.assertEqual(record["native_initial_state"], initial.tobytes())
                self.assertEqual(record["native_initial_rate"], rate.tobytes())
                self.assertTrue(record["native_memory_phi_and_ewt_unchanged"])
                self.assertFalse(record["solver_applied_weights"])
                self.assertEqual(record["native_steps"], 0)
                self.assertEqual(record["parent_weight_state"]["callback_calls"], expected_calls)
                weights = np.frombuffer(record["weights"], dtype="<f8")
                for i, raw in enumerate(initial):
                    q = sum((Fraction.from_float(float.fromhex(row[i]))
                             for row in frame["q0_words_hex"]), Fraction.from_float(float(raw)))
                    expected = 1.0 / (1e-6 * abs(float(q)) + float(atol[i]))
                    self.assertEqual(float(weights[i]).hex(), expected.hex())
            after = solver.statistics()
            self.assertEqual(frozen({k: v for k, v in before.items() if k != "parent_weight_state"}),
                             frozen({k: v for k, v in after.items() if k != "parent_weight_state"}))
            solver.init_step(1.5, initial, rate, parent_weight_frame=self.frame(time=1.5))
            fresh = solver.evaluate_initial_parent_weights()
            self.assertEqual(fresh["generation"], record["generation"] + 1)
            self.assertEqual(fresh["parent_weight_state"]["callback_calls"], 1)
            self.assertEqual(fresh["time_hex"], (1.5).hex())
            fail_first_attempt[0] = True
            # Prospective failed first-solve attempt; no accepted step is added.
            with self.assertRaisesRegex(RuntimeError, "declared failed first native attempt"):
                solver.step(1.500001, method="onestep")
            failed = solver.statistics()
            self.assertEqual(failed["num_steps"], 0)
            with self.assertRaisesRegex(RuntimeError, "initial_parent_weight_phase_unavailable"):
                solver.evaluate_initial_parent_weights()
            self.assertEqual(frozen(failed), frozen(solver.statistics()))
        finally:
            del solver
            gc.collect()

    def test_uninitialized_and_default_off_reject(self):
        np = self.numpy
        solver = self.IDA(self.residual, parent_weight_frame=self.frame())
        try:
            with self.assertRaisesRegex(RuntimeError, "initialized"):
                solver.evaluate_initial_parent_weights()
        finally:
            del solver
        solver = self.IDA(self.residual)
        try:
            solver.init_step(0.5, np.array([1.0]), np.array([-1.0]))
            before = solver.statistics()
            with self.assertRaisesRegex(RuntimeError, "parent_weight_evaluation_requires_opt_in"):
                solver.evaluate_initial_parent_weights()
            self.assertEqual(frozen(before), frozen(solver.statistics()))
        finally:
            del solver
            gc.collect()

    def test_busy_and_after_one_manufactured_step_reject(self):
        np = self.numpy
        holder, busy_rejections = {}, []

        def residual(t, y, yp, output):
            try:
                holder["solver"].evaluate_initial_parent_weights()
            except RuntimeError as error:
                busy_rejections.append("operation_in_progress" in str(error))
            else:
                busy_rejections.append(False)
            output[:] = yp + y

        solver = self.IDA(residual, rtol=1e-6, atol=1e-8,
                          parent_weight_frame=self.frame(size=1),
                          first_step=1e-6, max_step=1e-6)
        holder["solver"] = solver
        try:
            solver.init_step(0.5, np.array([1.0]), np.array([-1.0]))
            # Exactly one admitted manufactured accepted step, not a device trajectory.
            result = solver.step(0.500001, method="onestep")
            self.assertTrue(result.success)
            self.assertTrue(busy_rejections and all(busy_rejections))
            before = solver.statistics()
            self.assertEqual(before["num_steps"], 1)
            with self.assertRaisesRegex(RuntimeError, "initial_parent_weight_phase_unavailable"):
                solver.evaluate_initial_parent_weights()
            self.assertEqual(frozen(before), frozen(solver.statistics()))
        finally:
            holder.clear()
            del solver
            gc.collect()
