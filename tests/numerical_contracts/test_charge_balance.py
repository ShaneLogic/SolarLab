"""Manufactured physical-charge arithmetic; no physical model or flux proof.

Execute the actual pure definitions without importing the observation module's
physical dependencies. These tests do not admit any numerical trajectory.
"""
import ast
from dataclasses import dataclass
from fractions import Fraction
import math
from pathlib import Path
import unittest


REPO = Path(__file__).resolve().parents[2]
NAMESPACE = dict(__name__=__name__, dataclass=dataclass, Fraction=Fraction, math=math)
for filename, names in (
        ("contract_prototype.py", {"ContractError"}),
        ("interval_observation.py", {"rational", "Enclosure", "ChargePrefix", "SignedChargePrefix"})):
    path = REPO / "scripts/benchmarks" / filename
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in nodes} == names
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), NAMESPACE)

ContractError = NAMESPACE["ContractError"]
Enclosure = NAMESPACE["Enclosure"]
ChargePrefix = NAMESPACE["ChargePrefix"]
SignedChargePrefix = NAMESPACE["SignedChargePrefix"]
ZERO = (Fraction(0),)*3
ZERO_ENCLOSURES = (Enclosure(0),)*3


class SignedChargePrefixTests(unittest.TestCase):
    def test_independently_supplied_exact_balances_and_uncertainty_fields(self):
        prefix = SignedChargePrefix.start((1, 1, 1))
        changes = tuple(Enclosure(value, Fraction(1, 100)) for value in
                        (Fraction(3, 5), Fraction(5, 6), Fraction(-7, 8)))
        integrals = tuple(Enclosure(value, Fraction(1, 200)) for value in
                          (Fraction(1, 10), Fraction(1, 3), Fraction(-1, 2)))
        result, passed = prefix.append(changes, integrals, (Fraction(1, 400),)*3, (Fraction(1, 800),)*3)
        self.assertEqual(passed, (True,)*3)
        self.assertEqual(result.signed_balances, (Fraction(1, 2), Fraction(1, 2), Fraction(-3, 8)))
        self.assertEqual(result.absolute_balances, (Fraction(1, 2), Fraction(1, 2), Fraction(3, 8)))
        self.assertEqual(result.charge_errors, (Fraction(1, 100),)*3)
        self.assertEqual(result.quadrature_errors, (Fraction(1, 200),)*3)
        self.assertEqual(result.input_errors, (Fraction(1, 400),)*3)
        self.assertEqual(result.clock_errors, (Fraction(1, 800),)*3)
        self.assertEqual(result.total_errors, (Fraction(3, 160),)*3)
        self.assertEqual(result.total_bounds, (Fraction(83, 160), Fraction(83, 160), Fraction(63, 160)))
        self.assertEqual(result.intervals, 1)
        self.assertEqual(prefix.intervals, 0)
        self.assertEqual(prefix.signed_balances, ZERO)

    def test_signed_cancellation_has_no_hidden_absolute_prefix_gate(self):
        prefix = SignedChargePrefix.start((1, 1, 1))
        legacy = ChargePrefix.start((1, 1, 1))
        for sign in (1, -1, 1, -1):
            changes = (Enclosure(sign * Fraction(3, 4)),)*3
            prefix, passed = prefix.append(changes, ZERO_ENCLOSURES, ZERO, ZERO)
            legacy, legacy_passed = legacy.append(changes, ZERO_ENCLOSURES, ZERO)
            self.assertEqual(passed, (True,)*3)
        self.assertEqual(prefix.signed_balances, ZERO)
        self.assertEqual(prefix.absolute_balances, (Fraction(3),)*3)
        self.assertEqual(prefix.total_bounds, ZERO)
        self.assertEqual(prefix.failed, (False,)*3)
        self.assertEqual(legacy_passed, (False,)*3)
        self.assertEqual(legacy.absolute_defects, (Fraction(3),)*3)

    def test_all_uncertainties_accumulate_despite_signed_cancellation(self):
        prefix = SignedChargePrefix.start((1, 1, 1))
        for sign in (1, -1):
            prefix, passed = prefix.append((Enclosure(sign * Fraction(1, 4), Fraction(3, 8)),)*3,
                (Enclosure(0, Fraction(1, 12)),)*3, (Fraction(1, 24),)*3, (Fraction(1, 24),)*3)
            self.assertEqual(passed, (sign == 1,)*3)
        self.assertEqual(prefix.signed_balances, ZERO)
        self.assertEqual(prefix.charge_errors, (Fraction(3, 4),)*3)
        self.assertEqual(prefix.quadrature_errors, (Fraction(1, 6),)*3)
        self.assertEqual(prefix.input_errors, (Fraction(1, 12),)*3)
        self.assertEqual(prefix.clock_errors, (Fraction(1, 12),)*3)
        self.assertEqual(prefix.total_errors, (Fraction(13, 12),)*3)

    def test_only_combined_integration_bound_uses_the_one_third_subbudget(self):
        for field in ("charge", "input", "clock"):
            with self.subTest(field=field):
                charge = Fraction(3, 4) if field == "charge" else 0
                inputs = (Fraction(3, 4),)*3 if field == "input" else ZERO
                clocks = (Fraction(3, 4),)*3 if field == "clock" else ZERO
                prefix, passed = SignedChargePrefix.start((1, 1, 1)).append(
                    (Enclosure(0, charge),)*3, ZERO_ENCLOSURES, inputs, clocks)
                self.assertEqual(passed, (True,)*3)
                self.assertEqual(prefix.total_bounds, (Fraction(3, 4),)*3)
        prefix, passed = SignedChargePrefix.start((1, 1, 1)).append(
            ZERO_ENCLOSURES, (Enclosure(0, Fraction(5, 12)),)*3, ZERO, ZERO)
        self.assertEqual(passed, (False,)*3)
        self.assertEqual(prefix.total_bounds, (Fraction(5, 12),)*3)

    def test_separately_known_integrand_error_is_included_in_integral_radius(self):
        quadrature = Fraction(1, 6)
        integrand_contribution = Fraction(1, 6)
        for extra in (0, Fraction(1, 2**150)):
            with self.subTest(integrand_excess=extra):
                combined = quadrature + integrand_contribution + extra
                result, passed = SignedChargePrefix.start((1, 1, 1)).append(
                    (Enclosure(Fraction(1, 4)),)*3,
                    (Enclosure(Fraction(1, 4), combined),)*3, ZERO, ZERO)
                self.assertEqual(result.signed_balances, ZERO)
                self.assertEqual(result.quadrature_errors, (combined,)*3)
                self.assertEqual(result.total_errors, (combined,)*3)
                self.assertEqual(passed, (extra == 0,)*3)

    def test_cumulative_quadrature_gate(self):
        prefix = SignedChargePrefix.start((1, 1, 1))
        prefix, passed = prefix.append(ZERO_ENCLOSURES, (Enclosure(0, Fraction(1, 4)),)*3, ZERO, ZERO)
        self.assertEqual(passed, (True,)*3)
        prefix, passed = prefix.append(ZERO_ENCLOSURES, (Enclosure(0, Fraction(1, 4)),)*3, ZERO, ZERO)
        self.assertEqual(passed, (False,)*3)
        self.assertEqual(prefix.total_bounds, (Fraction(1, 2),)*3)

    def test_each_body_and_metal_component_is_independent(self):
        budgets = (Fraction(1), Fraction(2), Fraction(3))
        for component in range(3):
            for field in ("balance", "charge", "quadrature", "input", "clock"):
                with self.subTest(component=component, field=field):
                    changes, integrals = list(ZERO_ENCLOSURES), list(ZERO_ENCLOSURES)
                    inputs, clocks = list(ZERO), list(ZERO)
                    value = budgets[component] + Fraction(1, 100)
                    if field == "balance": changes[component] = Enclosure(value)
                    if field == "charge": changes[component] = Enclosure(0, value)
                    if field == "quadrature": integrals[component] = Enclosure(0, budgets[component]/3 + Fraction(1, 100))
                    if field == "input": inputs[component] = value
                    if field == "clock": clocks[component] = value
                    _, passed = SignedChargePrefix.start(budgets).append(changes, integrals, inputs, clocks)
                    self.assertEqual(passed, tuple(i != component for i in range(3)))
        _, passed = SignedChargePrefix.start((1, 1, 1)).append(
            (Enclosure(Fraction(3, 2)), Enclosure(Fraction(-3, 2)), Enclosure(0)), ZERO_ENCLOSURES, ZERO, ZERO)
        self.assertEqual(passed, (False, False, True))

    def test_failed_interval_cannot_be_recovered_by_later_opposite_centers(self):
        prefix = SignedChargePrefix.start((1, 1, 1))
        for center in (2, -1, -1):
            prefix, passed = prefix.append((Enclosure(center),)*3, ZERO_ENCLOSURES, ZERO, ZERO)
            self.assertEqual(passed, (False,)*3)
            self.assertEqual(prefix.failed, (True,)*3)
        self.assertEqual(prefix.total_bounds, ZERO)

    def test_failed_prefix_latches_even_when_each_interval_is_in_budget(self):
        prefix = SignedChargePrefix.start((1, 1, 1))
        for center, expected in ((Fraction(3, 4), True), (Fraction(3, 4), False), (Fraction(-3, 4), False)):
            prefix, passed = prefix.append((Enclosure(center),)*3, ZERO_ENCLOSURES, ZERO, ZERO)
            self.assertEqual(passed, (expected,)*3)
        self.assertEqual(prefix.total_bounds, (Fraction(3, 4),)*3)

    def test_exact_budget_and_quadrature_boundaries(self):
        epsilon = Fraction(1, 2**150)
        for over in (0, epsilon):
            with self.subTest(total_excess=over):
                _, passed = SignedChargePrefix.start((1, 1, 1)).append(
                    (Enclosure(Fraction(2, 3)+over),)*3, (Enclosure(0, Fraction(1, 3)),)*3, ZERO, ZERO)
                self.assertEqual(passed, (over == 0,)*3)
            with self.subTest(quadrature_excess=over):
                _, passed = SignedChargePrefix.start((1, 1, 1)).append(
                    ZERO_ENCLOSURES, (Enclosure(0, Fraction(1, 3)+over),)*3, ZERO, ZERO)
                self.assertEqual(passed, (over == 0,)*3)
        prefix, passed = SignedChargePrefix.start(ZERO).append(ZERO_ENCLOSURES, ZERO_ENCLOSURES, ZERO, ZERO)
        self.assertEqual(passed, (True,)*3)
        _, passed = prefix.append((Enclosure(epsilon),)*3, ZERO_ENCLOSURES, ZERO, ZERO)
        self.assertEqual(passed, (False,)*3)

    def test_invalid_missing_nonfinite_negative_and_shape_inputs(self):
        invalid = (None, True, "1", -1, Fraction(-1, 10), float("inf"), float("-inf"), float("nan"))
        for bad in invalid:
            with self.subTest(value=repr(bad)):
                with self.assertRaises(ContractError): SignedChargePrefix.start((bad, 1, 1))
                for target in ("input", "clock"):
                    inputs = (bad, 0, 0) if target == "input" else ZERO
                    clocks = (bad, 0, 0) if target == "clock" else ZERO
                    with self.assertRaises(ContractError):
                        SignedChargePrefix.start((1, 1, 1)).append(ZERO_ENCLOSURES, ZERO_ENCLOSURES, inputs, clocks)
        for bad in (None, (), (1, 1), (1, 1, 1, 1), {"body": 1, "left": 1, "right": 1}, "111"):
            with self.subTest(shape=repr(bad)):
                with self.assertRaises(ContractError): SignedChargePrefix.start(bad)
                for index in range(4):
                    args = [ZERO_ENCLOSURES, ZERO_ENCLOSURES, ZERO, ZERO]
                    args[index] = bad
                    with self.assertRaises(ContractError): SignedChargePrefix.start((1, 1, 1)).append(*args)
        for bad in (None, 0, Fraction(0), {"center": 0, "radius": 0}):
            with self.subTest(enclosure=repr(bad)):
                with self.assertRaises(ContractError):
                    SignedChargePrefix.start((1, 1, 1)).append((bad, Enclosure(0), Enclosure(0)), ZERO_ENCLOSURES, ZERO, ZERO)
        for center, radius in ((float("nan"), 0), (float("inf"), 0), (0, -1), (0, float("inf"))):
            with self.assertRaises(ContractError): Enclosure(center, radius)
        with self.assertRaises(TypeError):
            SignedChargePrefix.start((1, 1, 1)).append(ZERO_ENCLOSURES, ZERO_ENCLOSURES, ZERO)


if __name__ == "__main__":
    unittest.main(verbosity=2)
