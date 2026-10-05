"""Exact dense-factor checks and mocked harness wiring; no integrator runs."""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from scipy.linalg import LinAlgWarning, lu_factor, lu_solve


REPO = Path(__file__).resolve().parents[2]
ARCHIVE = Path("/Users/bytedance/Library/Mobile Documents/com~apple~CloudDocs/projects/solarlab")
spec = importlib.util.spec_from_file_location("factor_cache_test_subject",
                                           REPO/"scripts/benchmarks/hysteresis_reference.py")
hr = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = hr
spec.loader.exec_module(hr)


def dense_backend():
    owner = SimpleNamespace(nlu=0)
    def original(matrix):
        owner.nlu += 1
        return lu_factor(matrix, overwrite_a=True)
    return owner, original


def matrix(dtype=float):
    return np.array([[4., 1., -0., -.25], [2., 5., -.2, 0.],
                     [0., 1., 6., 1.], [.125, 0., 2., 7.]], dtype=dtype)


class TestExactDenseLUCache(unittest.TestCase):
    def assert_words(self, actual, expected):
        self.assertEqual(actual.dtype.str, expected.dtype.str)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.tobytes(order="C"), expected.tobytes(order="C"))

    def assert_factors(self, actual, expected):
        for a, b in zip(actual, expected, strict=True):
            self.assert_words(a, b)

    def test_real_complex_hits_solutions_and_factor_immutability(self):
        owner, original = dense_backend()
        cache = hr.ExactDenseLUCache(original)
        for dtype in (np.float64, np.complex128):
            a = matrix(dtype)
            expected = lu_factor(a.copy(), overwrite_a=True)
            miss, hit = cache(a.copy()), cache(a.copy())
            self.assert_factors(miss, expected)
            self.assert_factors(hit, expected)
            for factors in (miss, hit):
                before = tuple(x.tobytes() for x in factors)
                for multiplier in (1., 2., -3.):
                    rhs = np.arange(1., 5.).astype(dtype)*multiplier
                    wanted = lu_solve(expected, rhs.copy(), overwrite_b=True)
                    actual = lu_solve(factors, rhs.copy(), overwrite_b=True)
                    self.assert_words(actual, wanted)
                self.assertEqual(tuple(x.tobytes() for x in factors), before)
        self.assertEqual(owner.nlu, 2)
        self.assertEqual({k: cache.statistics[k] for k in ("requests", "hits", "misses", "retained_entries")},
                         {"requests": 4, "hits": 2, "misses": 2, "retained_entries": 2})

    def test_fortran_overwrite_and_caller_mutation_cannot_poison_cache(self):
        for dtype in (np.float64, np.complex128):
            owner, original = dense_backend()
            cache = hr.ExactDenseLUCache(original)
            baseline = matrix(dtype)
            caller = np.asfortranarray(baseline)
            expected = lu_factor(baseline.copy(), overwrite_a=True)
            returned = cache(caller)
            self.assertTrue(np.shares_memory(caller, returned[0]))
            self.assertNotEqual(caller.tobytes(), baseline.tobytes())
            caller[:] = 99.
            returned[1][:] = 0
            hit = cache(baseline.copy())
            self.assert_factors(hit, expected)
            self.assertEqual(owner.nlu, 1)
            self.assertFalse(np.shares_memory(caller, hit[0]))
            self.assertFalse(np.shares_memory(returned[1], hit[1]))
            with self.assertRaises(ValueError):
                hit[0][0, 0] = 0.
            caller[:] = baseline
            caller.real[3, 0] = np.nextafter(caller.real[3, 0], np.inf)
            cache(caller)
            self.assertEqual(owner.nlu, 2)

    def test_nonlocal_one_ulp_and_both_signed_zero_words_miss(self):
        owner, original = dense_backend()
        cache = hr.ExactDenseLUCache(original)
        a = matrix()
        cache(a.copy())
        a[3, 0] = np.nextafter(a[3, 0], np.inf)
        self.assert_factors(cache(a.copy()), lu_factor(a.copy(), overwrite_a=True))
        b = a.copy()
        b[0, 2] = 0.
        self.assertTrue(np.array_equal(a, b))
        self.assertNotEqual(a.tobytes(), b.tobytes())
        self.assert_factors(cache(b.copy()), lu_factor(b.copy(), overwrite_a=True))
        c = matrix(complex)
        c[0, 2] = complex(0., -0.)
        cache(c.copy())
        c[0, 2] = complex(0., 0.)
        self.assert_factors(cache(c.copy()), lu_factor(c.copy(), overwrite_a=True))
        self.assertEqual(owner.nlu, 5)
        self.assertEqual(cache.statistics["hits"], 0)

    def test_shape_replacement_and_two_entry_memory_bound(self):
        owner, original = dense_backend()
        cache = hr.ExactDenseLUCache(original)
        for n in (4, 2, 6, 3):
            for dtype in (np.float64, np.complex128):
                a = np.eye(n, dtype=dtype)*3
                self.assert_factors(cache(a.copy()), lu_factor(a.copy(), overwrite_a=True))
                self.assertLessEqual(cache.statistics["retained_entries"], 2)
                self.assertLessEqual(cache.statistics["retained_bytes"], 48*6**2+2*6*8)
        self.assertEqual(owner.nlu, 8)
        self.assertEqual(cache.statistics["retained_entries"], 2)
        pivot_bytes = lu_factor(np.eye(3), overwrite_a=True)[1].nbytes
        expected_bytes = 2*(3**2*8+3**2*16)+2*pivot_bytes
        self.assertEqual(cache.statistics["retained_bytes"], expected_bytes)

    def test_unsupported_inputs_keep_original_results_or_errors(self):
        class ArraySubclass(np.ndarray):
            pass
        readonly = matrix()
        readonly.flags.writeable = False
        factories = (lambda: matrix(np.float32), lambda: matrix(np.complex64),
                     lambda: matrix().tolist(), lambda: matrix().view(ArraySubclass),
                     lambda: readonly, lambda: np.arange(6.).reshape(3, 2)+1.,
                     lambda: np.empty((0, 0)), lambda: matrix().astype(">f8"),
                     lambda: np.eye(8)[::2, ::2], lambda: np.array([[np.inf]]),
                     lambda: np.array([[np.nan]]), lambda: np.array([1., 2.]))
        for make in factories:
            owner, original = dense_backend()
            cache = hr.ExactDenseLUCache(original)
            try:
                expected = lu_factor(make(), overwrite_a=True)
            except Exception as error:
                for _ in range(2):
                    with self.assertRaises(type(error)) as seen:
                        cache(make())
                    self.assertEqual(str(seen.exception), str(error))
            else:
                for _ in range(2):
                    self.assert_factors(cache(make()), expected)
            self.assertEqual(owner.nlu, 2)
            self.assertEqual(cache.statistics["bypasses"], 2)
            self.assertEqual(cache.statistics["retained_entries"], 0)

    def test_singular_warnings_keep_filters_and_never_become_hits(self):
        owner, original = dense_backend()
        cache = hr.ExactDenseLUCache(original)
        singular = np.array([[1., 2.], [2., 4.]])
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always", LinAlgWarning)
            for _ in range(2):
                cache(singular.copy())
        self.assertEqual(len(seen), 2)
        self.assertTrue(all(w.category is LinAlgWarning for w in seen))
        self.assertEqual(str(seen[0].message), str(seen[1].message))
        with warnings.catch_warnings():
            warnings.simplefilter("error", LinAlgWarning)
            with self.assertRaises(LinAlgWarning):
                cache(singular.copy())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LinAlgWarning)
            cache(singular.copy())
        self.assertEqual(owner.nlu, 4)
        self.assertEqual(cache.statistics["hits"], 0)
        self.assertEqual(cache.statistics["uncacheable_results"], 3)
        self.assertEqual(cache.statistics["retained_entries"], 0)

    def test_failure_and_nonfinite_results_are_not_reused(self):
        failure = RuntimeError("backend failure witness")
        original = Mock(side_effect=failure)
        cache = hr.ExactDenseLUCache(original)
        for _ in range(2):
            with self.assertRaises(RuntimeError) as seen:
                cache(matrix())
            self.assertIs(seen.exception, failure)
        self.assertEqual(original.call_count, 2)
        self.assertEqual(cache.statistics["retained_entries"], 0)
        factors = (np.array([[np.inf]]), np.array([0], dtype=np.int32))
        original = Mock(return_value=factors)
        cache = hr.ExactDenseLUCache(original)
        for _ in range(2):
            self.assertIs(cache(np.array([[1.]])), factors)
        self.assertEqual(original.call_count, 2)
        self.assertEqual(cache.statistics["uncacheable_results"], 2)

    def test_distinct_solver_caches_have_no_shared_entries(self):
        first_owner, first = dense_backend()
        second_owner, second = dense_backend()
        a, b = hr.ExactDenseLUCache(first), hr.ExactDenseLUCache(second)
        a(matrix())
        a(matrix())
        b(matrix())
        self.assertEqual((first_owner.nlu, second_owner.nlu), (1, 1))
        self.assertEqual((a.statistics["hits"], b.statistics["hits"]), (1, 0))
        self.assertFalse(np.shares_memory(a(matrix())[0], b(matrix())[0]))


class TestFactorPolicyWiring(unittest.TestCase):
    def test_plan_default_stays_absent_and_optin_is_identity_bound(self):
        with patch.object(hr, "source_binding", return_value={}), patch.object(hr, "environment_binding", return_value={}):
            args = (REPO, ARCHIVE, "hi_diffusivity_00", 240, 2, 2)
            default = hr.build_plan(*args, prebias_first_step_s=4e-14)
            explicit_none = hr.build_plan(*args, prebias_first_step_s=4e-14, factor_cache_policy=None)
            enabled = hr.build_plan(*args, prebias_first_step_s=4e-14,
                                    factor_cache_policy=hr.EXACT_DENSE_FACTOR_POLICY)
        self.assertEqual(default, explicit_none)
        self.assertNotIn("factor_cache_policy", default["numerics"])
        self.assertEqual(enabled["numerics"]["factor_cache_policy"], hr.EXACT_DENSE_FACTOR_POLICY)
        self.assertNotEqual(default["identity_sha256"], enabled["identity_sha256"])
        del enabled["numerics"]["factor_cache_policy"]
        for p in (default, enabled):
            del p["identity_sha256"]
        self.assertEqual(default, enabled)

    def test_invalid_policy_is_rejected_before_construction_or_loading(self):
        for value in (True, False, 1, "", "approximate", {}):
            with self.subTest(value=value), patch.object(hr, "load_inputs") as load:
                with self.assertRaisesRegex(hr.ReferenceError, "factor cache"):
                    hr.build_plan(REPO, ARCHIVE, "unused", 240, 2, 2, factor_cache_policy=value)
                load.assert_not_called()
            with patch("scipy.integrate.Radau", Mock()) as constructor:
                with self.assertRaises(hr.ReferenceError):
                    hr.radau_history([1.], factor_cache_policy=value)
                constructor.assert_not_called()

    def test_constructor_wraps_only_lu_and_preserves_controller_and_solver_scope(self):
        class ConstructorOnly:
            def __init__(self, *args, **kwargs):
                self.args, self.kwargs = args, kwargs
                self.nlu = 0
                def original(a):
                    self.nlu += 1
                    return lu_factor(a, overwrite_a=True)
                self.lu = self.original_lu = original
                self.solve_lu = self.original_solve_lu = lambda factors, b: lu_solve(factors, b, overwrite_b=True)
            def _step_impl(self):
                raise AssertionError("mock controller must never be called")
        with patch("scipy.integrate.Radau", ConstructorOnly):
            default, _, _ = hr.radau_history([1.])
            uncached = default("fun", 0., [1.], 1.)
            self.assertIs(uncached.lu, uncached.original_lu)
            self.assertEqual(default.factor_cache_reports, [])
            method, times, states = hr.radau_history([1.], first_step=4e-14,
                                                   factor_cache_policy=hr.EXACT_DENSE_FACTOR_POLICY)
            first, second = method("fun", 0., [1.], 1.), method("fun", 0., [1.], 1.)
            self.assertIs(method._step_impl, ConstructorOnly._step_impl)
            self.assertEqual(first.kwargs, {"first_step": 4e-14})
            self.assertIs(first.solve_lu, first.original_solve_lu)
            first.lu(matrix())
            first.lu(matrix())
            second.lu(matrix())
            self.assertEqual((first.nlu, second.nlu), (1, 1))
            self.assertEqual([r["hits"] for r in method.factor_cache_reports], [1, 0])
            self.assertEqual((times, states), ([0.], [(1.,)]))

    def test_bridge_preserves_failure_history_and_factor_counters(self):
        bundle = hr.load_inputs(REPO, ARCHIVE, "hi_diffusivity_00")
        control = hr.make_timeline(bundle["contract"], bundle["request"], 2)[3].controls[0]
        state = (1., 2., 3., 4., 0., 0.)
        bridge = hr.ProductionBridge.__new__(hr.ProductionBridge)
        bridge.np, bridge.x, bridge.stack = np, np.array([0., 1.]), object()
        bridge.mat = SimpleNamespace(P_ion0=np.zeros(2))
        bridge.widths, bridge.initial_inventory = np.array([.5, .5]), 0.
        bridge.numerics = {"rtol": 1e-6, "atol_m3": .01,
                           "factor_cache_policy": hr.EXACT_DENSE_FACTOR_POLICY}
        bridge.jacobian = object()
        counters = {"requests": 4, "hits": 2, "misses": 2}
        method = SimpleNamespace(factor_cache_reports=[counters])
        for raised in (False, True):
            solver = Mock(side_effect=RuntimeError("retained failure") if raised else None,
                          return_value=SimpleNamespace(success=False, t=np.array([]), y=np.array([]),
                                                       message="retained failure", nlu=2))
            bridge.jv = SimpleNamespace(run_transient=solver)
            with patch.object(hr, "radau_history", return_value=(method, [0., .001], [state, state])) as recorder:
                segment = bridge.advance(state, control)
            self.assertFalse(segment.success)
            self.assertEqual(segment.local_times, (0., .001))
            self.assertEqual(segment.states, (state, state))
            self.assertEqual(segment.diagnostics["factor_cache"]["solvers"], [counters])
            self.assertEqual(recorder.call_args.kwargs, {"factor_cache_policy": hr.EXACT_DENSE_FACTOR_POLICY})
            self.assertEqual(solver.call_args.kwargs["max_step"], control.max_step_s)
            self.assertIs(solver.call_args.kwargs["jacobian"], bridge.jacobian)

    def test_cli_and_supervisor_preserve_explicit_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/"plan.json"
            with patch.object(hr, "build_plan", return_value={}) as build:
                hr.main(["plan", "--repo", str(REPO), "--archive", str(ARCHIVE),
                         "--case", "hi_diffusivity_00", "--output", str(output),
                         "--factor-cache-policy", hr.EXACT_DENSE_FACTOR_POLICY])
            self.assertEqual(build.call_args.kwargs["factor_cache_policy"], hr.EXACT_DENSE_FACTOR_POLICY)
            args = SimpleNamespace(output=Path(temporary)/"run", repo=REPO, archive=ARCHIVE,
                                   case="hi_diffusivity_00", grid=240, step_level=2, solver_level=2,
                                   admission=Path(temporary)/"Admission.json", prebias_first_step_s=4e-14,
                                   factor_cache_policy=hr.EXACT_DENSE_FACTOR_POLICY)
            child = Mock(returncode=0)
            child.poll.return_value = 0
            with patch.object(hr.subprocess, "Popen", return_value=child) as launch:
                self.assertEqual(hr.supervise(args, {}), 0)
            argv = launch.call_args.args[0]
            self.assertEqual(argv[argv.index("--factor-cache-policy")+1], hr.EXACT_DENSE_FACTOR_POLICY)
            self.assertEqual(argv[argv.index("--prebias-first-step-s")+1], repr(4e-14))


if __name__ == "__main__":
    unittest.main()
