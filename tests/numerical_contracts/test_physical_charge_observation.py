"""Non-authorizing AST/stdlib fixtures for the physical-charge observer branch.

Actual production arithmetic and observer method bodies execute with explicit
manufactured charge-action/integral providers. No model, Point, physical kernel,
Arb, native solver, history or request materialization is imported or executed.
"""
import ast
import copy
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
import math
from pathlib import Path
from types import MappingProxyType, SimpleNamespace as NS
from typing import Mapping
import unittest
import __future__


REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "scripts/benchmarks"
ENV = dict(__name__=__name__, dataclass=dataclass, Fraction=Fraction, math=math,
           sha256=sha256, json=json, Mapping=Mapping, MappingProxyType=MappingProxyType)


def execute(nodes, filename, namespace=ENV):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(filename), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)


def definitions(filename, names):
    tree = ast.parse(filename.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    execute(nodes, filename)
    return tree


constants = REPO / "perovskite-sim/perovskite_sim/constants.py"
constant_nodes = [n for n in ast.parse(constants.read_text()).body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id in {"Q", "K_B"} for t in n.targets)]
assert len(constant_nodes) == 2
execute(constant_nodes, constants)
definitions(SOURCE / "contract_prototype.py", {"ContractError"})
INTERVAL_TREE = definitions(SOURCE / "interval_observation.py", {
    "rational", "ratio_payload", "identity", "Polynomial", "Enclosure", "ChargePrefix", "SignedChargePrefix",
    "public_action_enclosures", "contained_subinterval", "rational_sqrt_upper", "SlabPathObserver"})
policy_path = SOURCE / "physical_tolerance.py"
policy_tree = ast.parse(policy_path.read_text())
execute([n for n in policy_tree.body if isinstance(n, (ast.Assign, ast.FunctionDef))], policy_path)
OBSERVER_TREE = definitions(SOURCE / "native_observation.py", {
    "prepare_physical_charge_prefix", "physical_charge_summary", "physical_charge_passed",
    "prepare_interval_observation_policy", "AcceptedIntervalObserver",
    "observation_record", "restore_observation_record"})
ENV["CHARGE_LEDGERS"] = ("raw_polynomial", "same_state_affine_tangent")
ENV["propagated_input_error"] = lambda binding, prepared: binding.fixture_inputs
ENV["endpoint_charge_mismatch"] = lambda model, path, left, right: (Fraction(0),)*3

F, E = Fraction, ENV["Enclosure"]
ContractError = ENV["ContractError"]
SignedChargePrefix, ChargePrefix = ENV["SignedChargePrefix"], ENV["ChargePrefix"]
SlabPathObserver, Observer = ENV["SlabPathObserver"], ENV["AcceptedIntervalObserver"]
P = ENV["Polynomial"]
ZERO = (F(0),)*3
ZERO_E = (E(0),)*3


def request_fixture(dynamic=True):
    """Manufactured metadata only; never a native request or Root authority."""
    names = ("n_m3", "p_m3", "c_m3", "f", "phi_V") if dynamic else ("n_m3", "p_m3", "phi_V")
    references = {k: {"high": [0.5 if k == "f" else 0.0 if k == "phi_V" else 2.0]*3,
                      "low": [0.0]*3} for k in names}
    packet = {"definition": {"id": "DynamicAcceptorIonPublicDeviceV1" if dynamic else "S0NeutralPublicDeviceV1",
                            "dynamic": dynamic, "temperature": 300.0, "area": 3.0, "length": 2.0,
                            "epsilon": 5.0, "ni": 2.0, "ion_initial": 20.0 if dynamic else 0.0,
                            "trap_density": 4.0 if dynamic else 0.0},
              "column_scaling": [1.0]*(3*len(names)),
              "variable_offsets": {k: [3*i, 3*i+3] for i, k in enumerate(names)},
              "physical_reference_fields": references,
              "initial_reference": {"payload": {"fields": {k: {w: [v.hex() for v in values]
                  for w, values in pair.items()} for k, pair in references.items()}}}}
    policy = ENV["make_policy"](packet, {"r": 1e-4, "alpha": 1e-8})
    previous = {"rtol": 1e-9, "atol": [1e-11]*(3*len(names)), "max_step": 0.0}
    qref = F(ENV["Q"])*6*(20 if dynamic else 2)
    budgets = {"charge_C": float(qref/10**10), "native_steps": 200000}
    mapping = {"physical_model": "fixture-source", "layout": "fixture-layout",
               "physical_reference": "fixture-reference"}
    request = {"numeric_packet": packet, "physical_tolerance_policy": policy,
               "previous_controls": previous, "controls": ENV["controls"](previous, policy),
               "weight_certificate": policy, "budgets": budgets, "original_budgets": dict(budgets),
               "native_admission": False, "voltage_lift_map": mapping,
               "map_identity": ENV["digest"](mapping), "segments": [],
               "observation_times": [], "quadrature": []}
    request["interval_observation"] = ENV["prepare_interval_observation_policy"](request,
        binding_identity="fixture-binding", header_sha256="fixture-header", backend_modules={})
    return request


def action_result(values):
    # Exercise the real full-word reducer; the low word must not be dropped.
    low = F(1, 2**180)
    return NS(value=NS(words=(NS(flat=[v.center-low for v in values]), NS(flat=[low]*len(values))),
                       shape=(len(values),)),
              absolute_error_bound=NS(values=NS(flat=[v.radius for v in values])))


class Fixture:
    def __init__(self, dynamic=True, profile=True):
        self.request = request_fixture(dynamic)
        self.budget = ENV["physical_charge_budgets"](self.request)[0]
        if not profile:
            del self.request["physical_tolerance_policy"]
            del self.request["interval_observation"]["physical_charge"]
        self.actual, self.flux, self.raw, self.nominal = ZERO_E, ZERO_E, ZERO_E, ZERO
        self.input_error, self.clock_error, self.gap = ZERO, ZERO, ZERO
        self.integration_calls, self.action_calls = [], []
        self.left, self.right, self.increment = NS(time=0.0), NS(time=1.0), object()
        self.frame = NS(identity="fixture-frame")
        self.clock = NS(predecessor=F(0), tn=F(1), payload=lambda: {"predecessor": F(0), "tn": F(1)})
        self.path = NS(identity="fixture-path", source_identity="fixture-source",
                       layout_identity="fixture-layout", clock=self.clock)
        definition = NS(**self.request["numeric_packet"]["definition"])
        self.model = NS(definition=definition, source_identity=self.path.source_identity,
            layout=NS(identity=self.path.layout_identity),
            linear_forms={"increment": {k: NS(identity="fixture-"+k) for k in ("body_charge", "metal_charge")}})
        def action(name, right, *, increment, left):
            self.action_calls.append((name, right, increment, left))
            return action_result(self.actual[:1] if name == "body_charge" else self.actual[1:])
        self.model.linear_action = action
        self.context = NS(request_copy=lambda: copy.deepcopy(self.request),
                          request_sha256=ENV["digest"](self.request))
        self.binding = NS(context=self.context)
        self.prepared = NS(path=self.path, request_identity=self.context.request_sha256,
                           map_identity=self.request["map_identity"], coefficient_frame_identity=self.frame.identity)
        self.root_map = copy.deepcopy(self.request["voltage_lift_map"])
        self.active_map = copy.deepcopy(self.root_map)
        self.binding.adapter = NS(mapping=NS(identity=self.prepared.map_identity,
            payload=lambda include_frame=True: copy.deepcopy(self.active_map if include_frame else self.root_map)))
        # Exact production type, deliberately allocated without its physical constructor.
        self.slab = object.__new__(SlabPathObserver)
        self.slab.model = self.model
        def integrate(arithmetic, absolute_error, **kwargs):
            self.integration_calls.append((arithmetic, absolute_error, kwargs))
            return {"path_identity": self.path.identity, "raw_polynomial": self.raw,
                    "same_state_affine_tangent": self.flux, "raw_tangent_charge_L1_upper_C": self.gap,
                    "raw_tangent_L1_upper_bounds": tuple(E(v) for v in self.gap)}
        self.slab.integrate = integrate
        self.slab.strip_current_debit = lambda arithmetic: {
            "raw_polynomial": self.clock_error, "same_state_affine_tangent": self.clock_error}
        self.observer = object.__new__(Observer)
        self.observer.binding, self.observer.prepared = self.binding, self.prepared
        self.observer.frame, self.observer.observer, self.observer.arithmetic = self.frame, self.slab, "fixture-arithmetic"
        self.prefixes = {k: ChargePrefix.start((self.request["budgets"]["charge_C"],)*3)
                         for k in ENV["CHARGE_LEDGERS"]}
        if profile:
            self.prefixes["physical_charge"] = ENV["prepare_physical_charge_prefix"](self.request)

    def observe(self, **kwargs):
        self.binding.fixture_inputs = {"raw_charge_integral_error_C": self.input_error,
                                       "tangent_charge_integral_error_C": self.input_error}
        self.slab.affine = {"body_charge": {"left": ZERO[:1], "right": self.nominal[:1]},
                           "metal_charge": {"left": ZERO[1:], "right": self.nominal[1:]}}
        self.prefixes, evidence = self.observer.charge_evidence((self.left, None), (self.right, self.increment),
            self.prefixes, absolute_error=self.budget/100, charge_budget=self.request["budgets"]["charge_C"], **kwargs)
        return evidence

    def selected(self, evidence):
        return ENV["physical_charge_passed"](self.request, evidence,
            frame_identity=self.frame.identity, path_identity=self.path.identity,
            map_identity=self.binding.adapter.mapping.identity, map_payload=self.binding.adapter.mapping.payload())

    def bind_active_frame(self):
        self.active_map = dict(self.root_map,
            segment_frame={"parent_map_identity": self.request["map_identity"], "t0_hex": 0.0.hex()},
            raw_coordinate_meaning="fixed-segment affine remainder u; native history is u",
            physical_coordinate_meaning="Point.y is the first retained word of S*(Q0+(t-t0)*V0+u)+L*(a-a_ref)")
        self.binding.adapter.mapping.identity = ENV["digest"](self.active_map)
        self.prepared.map_identity = self.binding.adapter.mapping.identity


class PhysicalChargeObservationTests(unittest.TestCase):
    def test_distinct_active_and_root_maps_use_the_actual_parent_contract(self):
        f = Fixture()
        f.bind_active_frame()
        self.assertNotEqual(f.prepared.map_identity, f.request["map_identity"])
        evidence = f.observe()
        self.assertTrue(f.selected(evidence))
        row = evidence["physical_charge"]
        self.assertEqual(row["map_identity"], ENV["digest"](f.active_map))
        self.assertEqual(row["root_map_identity"], ENV["digest"](f.root_map))

    def test_distinct_map_rejects_wrong_active_or_parent_before_integrating(self):
        for kind in ("active", "parent"):
            f = Fixture()
            f.bind_active_frame()
            if kind == "active":
                f.binding.adapter.mapping.identity = "0"*64
            else:
                f.root_map["physical_reference"] = "wrong-parent"
            with self.assertRaises(ContractError):
                f.observe()
            self.assertEqual(f.integration_calls, [])
        f = Fixture()
        f.bind_active_frame()
        evidence = f.observe()
        f.active_map["segment_frame"]["parent_map_identity"] = "0"*64
        f.binding.adapter.mapping.identity = ENV["digest"](f.active_map)
        evidence["physical_charge"]["map_identity"] = f.binding.adapter.mapping.identity
        with self.assertRaisesRegex(ContractError, "parent_map"):
            f.selected(evidence)

    def test_actual_saved_charge_and_single_existing_flux_integral(self):
        for dynamic in (False, True):
            with self.subTest(dynamic=dynamic):
                f = Fixture(dynamic)
                f.actual = f.flux = (E(2*f.budget), E(-3*f.budget), E(f.budget))
                f.raw = (E(900*f.budget),)*3
                f.gap = (100*f.budget,)*3
                evidence = f.observe()
                self.assertTrue(f.selected(evidence))
                self.assertFalse(evidence["passed"])
                self.assertFalse(evidence["ledgers"]["raw_polynomial"]["passed"])
                self.assertFalse(evidence["ledgers"]["same_state_affine_tangent"]["passed"])
                row = evidence["physical_charge"]
                self.assertEqual(row["saved_finite_charge_change_C"], f.actual)
                self.assertEqual(row["physical_flux_integral_C"], f.flux)
                self.assertEqual(row["signed_balance_C"], ZERO)
                self.assertEqual(row["interval_total_bound_C"], ZERO)
                self.assertEqual(len(f.integration_calls), 1)
                self.assertEqual([v[0] for v in f.action_calls], ["body_charge", "metal_charge"])
                self.assertTrue(all(r is f.right and inc is f.increment and l is f.left
                                    for _, r, inc, l in f.action_calls))

    def test_all_positive_uncertainties_once_without_endpoint_double_charge(self):
        f = Fixture()
        b = f.budget
        f.actual, f.flux = (E(2*b, b/10),)*3, (E(2*b-b/4, b/20),)*3
        f.input_error, f.clock_error = (b/40,)*3, (b/80,)*3
        row = f.observe()["physical_charge"]
        self.assertTrue(row["passed"])
        self.assertEqual(row["interval_total_bound_C"], (7*b/16,)*3)
        self.assertEqual(row["prefix"]["total_error_C"], (3*b/16,)*3)
        self.assertEqual(set(row["error_components_C"]),
            {"saved_charge_evaluation", "physical_flux_integration", "input", "clock_strip"})
        self.assertIn("physical-integrand evaluation", row["integration_uncertainty_role"])

    def test_signed_cancellation_and_absolute_sum_only_diagnostic(self):
        f = Fixture()
        for sign in (1, -1, 1, -1):
            f.actual = (E(sign*3*f.budget/4),)*3
            evidence = f.observe()
            self.assertTrue(f.selected(evidence))
        prefix = evidence["physical_charge"]["prefix"]
        self.assertEqual(prefix["signed_balance_C"], ZERO)
        self.assertEqual(prefix["cumulative_interval_absolute_balance_C"], (3*f.budget,)*3)
        self.assertEqual(prefix["intervals"], 4)

    def test_uncertainties_cannot_cancel_and_failure_latches(self):
        f = Fixture()
        for sign in (1, -1):
            f.actual = (E(sign*f.budget/4, 3*f.budget/8),)*3
            f.flux = (E(0, f.budget/12),)*3
            f.input_error = f.clock_error = (f.budget/24,)*3
            row = f.observe()["physical_charge"]
            self.assertEqual(row["passed"], sign == 1)
        self.assertEqual(row["prefix"]["signed_balance_C"], ZERO)
        self.assertEqual(row["prefix"]["total_error_C"], (13*f.budget/12,)*3)

    def test_each_component_interval_failure_cannot_be_recovered(self):
        for i in range(3):
            with self.subTest(component=i):
                f = Fixture()
                for sign in (1, -1):
                    f.actual = tuple(E(sign*2*f.budget if j == i else 0) for j in range(3))
                    evidence = f.observe()
                    self.assertFalse(f.selected(evidence))
                    self.assertEqual(evidence["physical_charge"]["row_checks"], tuple(j != i for j in range(3)))
                self.assertEqual(evidence["physical_charge"]["prefix"]["total_bound_C"], ZERO)

    def test_exact_total_and_integral_radius_boundaries(self):
        for total_extra, radius_extra, expected in ((0, 0, True), (F(1, 2**150), 0, False),
                                                     (0, F(1, 2**150), False)):
            f = Fixture()
            f.actual = (E(f.budget*(F(2, 3)+total_extra)),)*3
            f.flux = (E(0, f.budget*(F(1, 3)+radius_extra)),)*3
            self.assertEqual(f.selected(f.observe()), expected)
        f = Fixture()
        f.flux = (E(0, f.budget/4),)*3
        self.assertTrue(f.selected(f.observe()))
        self.assertFalse(f.selected(f.observe()))  # cumulative third share only

    def test_profile_prefix_missing_budget_or_count_mismatch_fails_before_integration(self):
        for mutation in ("missing", "budget", "count", "policy"):
            f = Fixture()
            if mutation == "missing":
                del f.prefixes["physical_charge"]
            elif mutation == "budget":
                f.prefixes["physical_charge"] = SignedChargePrefix.start((2*f.budget,)*3)
            elif mutation == "count":
                f.prefixes["physical_charge"], _ = f.prefixes["physical_charge"].append(ZERO_E, ZERO_E, ZERO, ZERO)
            else:
                del f.request["interval_observation"]["physical_charge"]
            with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                f.observe()
            self.assertEqual(f.integration_calls, [])

    def test_mismatched_frame_map_source_and_slab_scope_rejected(self):
        for target, field, value in (("prepared", "coefficient_frame_identity", "other"),
                                    ("prepared", "map_identity", "other"),
                                    ("prepared", "request_identity", "other"),
                                    ("path", "source_identity", "other"),
                                    ("definition", "id", "other-geometry"),
                                    ("definition", "epsilon", 6.0)):
            f = Fixture()
            obj = f.model.definition if target == "definition" else getattr(f, target)
            setattr(obj, field, value)
            with self.subTest(target=target, field=field), self.assertRaises(ContractError):
                f.observe()
            self.assertEqual(f.integration_calls, [])

    def test_incomplete_interval_and_missing_or_invalid_uncertainty_rejected(self):
        f = Fixture()
        with self.assertRaises(ContractError):
            f.observe(subinterval=(0, F(1, 2)))
        self.assertEqual(f.integration_calls, [])
        for bad in ((None, 0, 0), (-1, 0, 0), (float("nan"), 0, 0), (0, 0), (True, 0, 0)):
            with self.subTest(bad=bad), self.assertRaises((ContractError, ValueError, TypeError)):
                f = Fixture()
                f.input_error = bad
                f.observe()
        f = Fixture()
        f.flux = (E(0), E(0))
        with self.assertRaises(ContractError):
            f.observe()

    def test_saved_endpoint_clocks_must_match_the_native_frame(self):
        for side in ("left", "right"):
            f = Fixture()
            getattr(f, side).time = 0.5
            with self.subTest(side=side), self.assertRaises(ContractError):
                f.observe()
            self.assertEqual(f.integration_calls, [])

    def test_missing_or_mismatched_new_evidence_or_changed_latch_rejected(self):
        f = Fixture()
        good = f.observe()
        for key, value in (("policy", {}), ("request_sha256", "other"), ("map_identity", "other"),
                           ("path_identity", "other"), ("frame_identity", "other"), ("clock", {})):
            bad = copy.deepcopy(good)
            bad["physical_charge"][key] = value
            with self.subTest(key=key), self.assertRaises(ContractError):
                f.selected(bad)
        bad = copy.deepcopy(good)
        del bad["physical_charge"]
        with self.assertRaises(ContractError):
            f.selected(bad)
        bad = copy.deepcopy(good)
        bad["physical_charge"]["prefix"]["failed"] = (True, False, False)
        with self.assertRaises(ContractError):
            f.selected(bad)
        for key in ("physical_flux_integral_C", "saved_finite_charge_change_C", "error_components_C"):
            bad = copy.deepcopy(good)
            del bad["physical_charge"][key]
            with self.subTest(missing=key), self.assertRaises(ContractError):
                f.selected(bad)
        bad = copy.deepcopy(good)
        bad["physical_charge"]["row_checks"] = (1, 1, 1)
        with self.assertRaises(ContractError):
            f.selected(bad)

    def test_legacy_branch_and_lossless_new_record_round_trip(self):
        f = Fixture(profile=False)
        evidence = f.observe()
        self.assertEqual(set(f.prefixes), set(ENV["CHARGE_LEDGERS"]))
        self.assertNotIn("physical_charge", evidence)
        self.assertTrue(evidence["passed"])
        self.assertIsNone(ENV["prepare_physical_charge_prefix"](f.request))
        f.prefixes["physical_charge"] = SignedChargePrefix.start((1,)*3)
        with self.assertRaises(ContractError):
            f.observe()
        f = Fixture()
        evidence = f.observe()
        encoded = ENV["observation_record"](evidence)
        restored = ENV["restore_observation_record"](json.loads(json.dumps(encoded)))
        self.assertTrue(f.selected(restored))
        self.assertEqual(restored["physical_charge"]["prefix"], evidence["physical_charge"]["prefix"])


class PhysicalFluxSignTests(unittest.TestCase):
    def test_actual_body_charge_form_both_variants_and_area_once(self):
        tree = ast.parse((SOURCE / "coupled_device_prototype.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AffineCoupledSlab")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_physical_forms")
        charge = next(n for n in method.body if isinstance(n, ast.FunctionDef) and n.name == "charge")
        body = next(n for n in method.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "body" for t in n.targets))
        fields = {"p_m3": (11, 13, 17), "n_m3": (5, 7, 9), "c_m3": (22, 23, 24),
                  "f": (F(1, 4), F(1, 2), F(3, 4))}
        for dynamic, expected in ((False, 78), (True, 90)):
            env = dict(ENV, m=NS(dynamic=dynamic), N=3, kind="state", q=F(2), nt=F(4), c0=F(20),
                       volumes=(F(3, 2), F(3), F(3, 2)),
                       LinearTerm=lambda row, field, node, factors, sign=1: NS(field=field, node=node, factors=factors, sign=sign))
            execute([charge, body], "actual_body_charge_AST", env)
            value = sum(t.sign*math.prod(t.factors)*(fields[t.field][t.node] if t.field else 1) for t in env["body"])
            self.assertEqual(value, expected)

    def test_actual_metal_flux_polynomial_and_integral_row_both_variants(self):
        cls = next(n for n in INTERVAL_TREE.body if isinstance(n, ast.ClassDef) and n.name == "SlabPathObserver")
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        selected = []
        for n in init.body:
            if isinstance(n, ast.Assign) and any(
                    (isinstance(t, ast.Name) and t.id in {"voltage_rate", "capacitance"}) or
                    (isinstance(t, ast.Attribute) and t.attr in {"capture_charge_rate", "tangent_metal_polynomial"}) for t in n.targets):
                selected.append(n)
        self.assertEqual(len(selected), 4)
        for dynamic, expected in ((False, (F(-12), F(207, 2), F(-183, 2))),
                                   (True, (F(-48), F(213, 2), F(-117, 2)))):
            slab = object.__new__(SlabPathObserver)
            slab.q, slab.area, slab.length = F(2), F(3), F(2)
            slab.weights, slab.volumes = ((F(1), F(1, 2), F(0)), (F(0), F(1, 2), F(1))), (F(3, 2), F(3), F(3, 2))
            slab.m, slab.model = NS(epsilon=5, trap_density=4 if dynamic else 0), NS(count=3)
            slab.capture_difference = (P((1,)), P(), P((2,))) if dynamic else (P(),)*3
            slab.endpoint_carrier_rate, slab.raw_metal = (P(),)*2, (P((900,)), P((-800,)))
            slab.fluxes = lambda a, u, analytic=False: ((F(7), F(11)), (F(2), F(-5)) if dynamic else (F(0), F(0)))
            clock = NS(predecessor=F(0), tn=F(1), hused=F(1), coordinate=lambda t: t-1)
            path = NS(clock=clock, inputs=(P((0, 13)), P()), identity="manufactured-constant-flux",
                      require_covered=lambda: None, integrate=lambda p: p.integral(-1, 0))
            slab.path = path
            execute(selected, "actual_tangent_polynomial_AST", dict(ENV, self=slab, path=path, h=F(1), model=slab.model))
            class ConstantArithmetic:
                """Exact constant integrands only; not a production quadrature substitute."""
                def __init__(self):
                    self.calls = []
                    self.integration_count = 0
                number = staticmethod(F)
                polynomial = staticmethod(lambda p, u: p.at(u))
                def integrate(self, function, lo, hi, allocation):
                    a, b = function(lo, False), function(hi, False)
                    assert a == b
                    self.integration_count += 1
                    self.calls.append("constant-integral")
                    return E(a*(hi-lo))
                def absolute_bound(self, function, lo, hi, allocation):
                    a, b = function(lo, False), function(hi, False)
                    assert a == b
                    self.calls.append("constant-absolute-bound")
                    return NS(upper=abs(a)*(hi-lo))
            arithmetic = ConstantArithmetic()
            row = slab.integrate(arithmetic, F(1, 100))["same_state_affine_tangent"]
            self.assertEqual(tuple(v.center for v in row), expected)
            self.assertEqual(sum(v.center for v in row), 0)
            self.assertEqual(tuple(v.radius for v in row), ZERO)
            self.assertEqual(arithmetic.integration_count, 4)


if __name__ == "__main__":
    unittest.main()
