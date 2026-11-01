"""Synthetic comparison contract tests; never import or execute a solver."""

from copy import deepcopy
import math
import unittest

from reference_comparison import IDENTITY_FIELDS, compare, hysteresis_from_power


def example(kind="scalar", coordinates=None, values=None):
    axes = {"scalar": [], "curve": [{"name": "voltage", "unit": "V"}],
            "parameter_grid": [{"name": "Nt", "unit": "cm^-3"}, {"name": "CBO", "unit": "eV"}]}[kind]
    coordinates = coordinates if coordinates is not None else [[]]
    values = values if values is not None else [2.0]
    identity = {field: "a" * 64 for field in IDENTITY_FIELDS}
    identity["driver"] = "synthetic_recorded_driver_v1"
    table = {"identity": identity, "kind": kind, "scope": "synthetic_control", "quantity": "J_total", "unit": "A/m^2",
             "axes": axes, "run_started_at": "2026-10-02T00:00:00Z", "created_at": "2026-10-03T00:00:00Z",
             "points": [{"coordinates": c, "value": v, "status": "ok", "origin": "native"}
                        for c, v in zip(coordinates, values)]}
    gate = {"id": "synthetic_gate", "version": "1", "status": "frozen", "scope": table["scope"],
            "kind": kind, "quantity": table["quantity"], "unit": table["unit"], "axes": axes,
            "evidence_level": "numerical", "reference_identity": deepcopy(identity), "candidate_identity": deepcopy(identity),
            "expected_coordinates": coordinates, "atol": 0.01, "rtol": 0.001, "scale": "abs_reference",
            "reference_error_abs": 0.0, "reference_error_evidence": "exact synthetic numbers; no physical certification",
            "rule": "sum", "zero_policy": "absolute_limit", "frozen_at": "2026-10-01T00:00:00Z",
            "independent_review": {"status": "approved", "reviewer": "synthetic_fixture", "evidence": "test contract only"}}
    return deepcopy(table), deepcopy(table), deepcopy(gate)


class ReferenceComparisonTests(unittest.TestCase):
    def assert_rejected(self, reference, candidate, gate, code):
        result = compare(reference, candidate, gate)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["code"], code, result)
        self.assertTrue(result["detail"])
        self.assertEqual(result["rows"], [])

    def test_scalar_mixed_budget_and_margin(self):
        ref, new, gate = example()
        new["points"][0]["value"] = 2.005
        result = compare(ref, new, gate)
        self.assertTrue(result["accepted"])
        row = result["rows"][0]
        self.assertAlmostEqual(row["limit"], 0.012)
        self.assertAlmostEqual(row["absolute_error"], 0.005)
        self.assertAlmostEqual(row["margin"], 0.007)

    def test_limit_is_inclusive(self):
        ref, new, gate = example(values=[1.0])
        gate.update(atol=0.125, rtol=0)
        new["points"][0]["value"] = 1.125
        self.assertTrue(compare(ref, new, gate)["accepted"])
        new["points"][0]["value"] = math.nextafter(1.125, math.inf)
        self.assertFalse(compare(ref, new, gate)["accepted"])

    def test_candidate_cannot_set_relative_scale(self):
        ref, new, gate = example(values=[1.0])
        gate.update(atol=0, rtol=0.1)
        new["points"][0]["value"] = 1.11
        result = compare(ref, new, gate)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["rows"][0]["limit"], 0.1)

    def test_fixed_physical_scale_is_supported(self):
        ref, new, gate = example(values=[0.0])
        gate.update(atol=0, rtol=0.001, scale=100.0)
        new["points"][0]["value"] = 0.08
        self.assertTrue(compare(ref, new, gate)["accepted"])

    def test_curve_is_joined_by_coordinate(self):
        ref, new, gate = example("curve", [[-1.0], [0.0], [1.2]], [-4.0, 0.0, 8.0])
        new["points"].reverse()
        self.assertTrue(compare(ref, new, gate)["accepted"])
        self.assertEqual([r["coordinates"] for r in compare(ref, new, gate)["rows"]], gate["expected_coordinates"])

    def test_curve_position_swap_is_not_equivalence(self):
        ref, new, gate = example("curve", [[0.0], [1.0]], [1.0, 10.0])
        new["points"][0]["value"], new["points"][1]["value"] = 10.0, 1.0
        result = compare(ref, new, gate)
        self.assertEqual(result["counts"], {"failed": 2})

    def test_two_parameter_grid_joins_both_coordinates(self):
        ref, new, gate = example("parameter_grid", [[1e9, -0.2], [1e9, 0.1], [1e10, -0.2], [1e10, 0.1]], [4.0, 8.0, 2.0, 6.0])
        new["points"] = [new["points"][i] for i in [3, 0, 2, 1]]
        self.assertTrue(compare(ref, new, gate)["accepted"])

    def test_cbo_minus_point_two_cannot_be_point_zero_five(self):
        ref, new, gate = example("parameter_grid", [[1e9, -0.2]], [4.0])
        new["points"][0]["coordinates"][1] = 0.05
        self.assert_rejected(ref, new, gate, "coordinate_mismatch")

    def test_no_implicit_coordinate_rounding(self):
        ref, new, gate = example("curve", [[0.2]], [1.0])
        new["points"][0]["coordinates"] = [math.nextafter(0.2, 0.0)]
        self.assert_rejected(ref, new, gate, "coordinate_mismatch")

    def test_large_integer_coordinates_do_not_collapse(self):
        ref, new, gate = example("curve", [[2**53], [2**53 + 1]], [1.0, 2.0])
        new["points"].reverse()
        self.assertTrue(compare(ref, new, gate)["accepted"])

    def test_missing_extra_and_duplicate_coordinates_rejected(self):
        for change, code in [("missing", "coordinate_mismatch"), ("extra", "coordinate_mismatch"), ("duplicate", "duplicate_coordinate")]:
            with self.subTest(change=change):
                ref, new, gate = example("curve", [[0.0], [1.0]], [1.0, 2.0])
                if change == "missing":
                    new["points"].pop()
                else:
                    point = deepcopy(new["points"][0])
                    if change == "extra":
                        point["coordinates"] = [3.0]
                    new["points"].append(point)
                self.assert_rejected(ref, new, gate, code)

    def test_both_table_and_gate_coordinates_must_be_nonempty_unique(self):
        ref, new, gate = example("curve", [[0.0]], [1.0])
        gate["expected_coordinates"] = []
        self.assert_rejected(ref, new, gate, "incomplete_gate")
        gate["expected_coordinates"] = [[0.0], [0.0]]
        self.assert_rejected(ref, new, gate, "duplicate_coordinate")

    def test_source_and_every_execution_identity_field_are_checked(self):
        for side in ("reference", "candidate"):
            for field in IDENTITY_FIELDS:
                with self.subTest(side=side, field=field):
                    ref, new, gate = example()
                    table = ref if side == "reference" else new
                    table["identity"][field] = "other_driver" if field == "driver" else "b" * 64
                    self.assert_rejected(ref, new, gate, "identity_mismatch")

    def test_explicitly_bound_new_source_can_be_compared(self):
        ref, new, gate = example()
        gate["candidate_identity"]["source_sha256"] = "b" * 64
        new["identity"]["source_sha256"] = "b" * 64
        self.assertTrue(compare(ref, new, gate)["accepted"])

    def test_unknown_identity_is_not_a_wildcard(self):
        ref, new, gate = example()
        new["identity"]["initial_state_sha256"] = None
        gate["candidate_identity"]["initial_state_sha256"] = None
        self.assert_rejected(ref, new, gate, "missing_identity")

    def test_missing_thresholds_or_reference_budget_fail_closed(self):
        for field in ("atol", "rtol", "scale", "reference_error_abs", "reference_error_evidence", "rule", "zero_policy"):
            with self.subTest(field=field):
                ref, new, gate = example()
                del gate[field]
                self.assert_rejected(ref, new, gate, "missing_threshold")

    def test_unreviewed_proposal_cannot_accept(self):
        ref, new, gate = example()
        gate["status"] = "awaiting_independent_review"
        self.assert_rejected(ref, new, gate, "unreviewed_gate")
        gate["status"] = "frozen"
        gate["independent_review"]["status"] = "pending"
        self.assert_rejected(ref, new, gate, "unreviewed_gate")

    def test_results_dependent_gate_and_timezone_ambiguity_rejected(self):
        ref, new, gate = example()
        gate["frozen_at"] = new["created_at"]
        self.assert_rejected(ref, new, gate, "posthoc_gate")
        gate["frozen_at"] = "2026-09-01T00:00:00"
        self.assert_rejected(ref, new, gate, "invalid_time")

    def test_gate_between_run_start_and_artifact_creation_is_posthoc(self):
        ref, new, gate = example()
        gate["frozen_at"] = "2026-10-02T12:00:00Z"
        self.assert_rejected(ref, new, gate, "posthoc_gate")
        del new["run_started_at"]
        self.assert_rejected(ref, new, gate, "invalid_time")

    def test_reference_error_must_fit_one_third_budget(self):
        ref, new, gate = example(values=[1.0])
        gate.update(atol=0.09, rtol=0, reference_error_abs=0.031)
        self.assert_rejected(ref, new, gate, "reference_budget_exceeded")
        gate["reference_error_abs"] = 0.03
        self.assertTrue(compare(ref, new, gate)["accepted"])

    def test_missing_reference_failed_solver_and_unknown_remain_distinct(self):
        ref, new, gate = example("curve", [[0.0], [1.0], [2.0], [3.0]], [1.0] * 4)
        ref["points"][0].update(status="reference_missing", value=None, reason="blank source cell")
        new["points"][0].update(status="solver_failed", value=None, reason="nonlinear convergence")
        new["points"][1].update(status="solver_failed", value=None, reason="time budget")
        ref["points"][2].update(status="unknown", value=None, reason="unresolved input identity")
        result = compare(ref, new, gate)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["reference_status_counts"], {"reference_missing": 1, "ok": 2, "unknown": 1})
        self.assertEqual(result["candidate_status_counts"], {"solver_failed": 2, "ok": 2})
        self.assertEqual(result["counts"], {"unresolved": 3, "passed": 1})
        self.assertEqual(result["rows"][0]["candidate_reason"], "nonlinear convergence")

    def test_unknown_needs_reason_and_cannot_carry_zero_fill(self):
        ref, new, gate = example()
        new["points"][0].update(status="unknown", value=0.0, reason="missing")
        self.assert_rejected(ref, new, gate, "invalid_status")
        new["points"][0].update(value=None, reason="")
        self.assert_rejected(ref, new, gate, "missing_reason")

    def test_nonfinite_values_boolean_and_negative_budget_rejected(self):
        for bad in (math.nan, math.inf, -math.inf, True, "1.0"):
            with self.subTest(bad=bad):
                ref, new, gate = example()
                new["points"][0]["value"] = bad
                self.assert_rejected(ref, new, gate, "invalid_number")
        ref, new, gate = example()
        gate["atol"] = -1.0
        self.assert_rejected(ref, new, gate, "invalid_number")

    def test_unit_metric_branch_scope_and_axis_name_mismatches_rejected(self):
        for field, value in [("unit", "mA/cm^2"), ("quantity", "HI_normalized"), ("scope", "reverse_branch"),
                             ("axes", [{"name": "time", "unit": "s"}])]:
            with self.subTest(field=field):
                ref, new, gate = example("curve", [[0.0]], [1.0])
                new[field] = value
                self.assert_rejected(ref, new, gate, "measurement_mismatch")

    def test_interpolation_does_not_count_as_measured_or_computed_points(self):
        for side in ("reference", "candidate"):
            ref, new, gate = example()
            table = ref if side == "reference" else new
            table["points"][0]["origin"] = "PCHIP"
            self.assert_rejected(ref, new, gate, "unqualified_point")

    def test_zero_signal_uses_declared_absolute_limit(self):
        ref, new, gate = example(values=[0.0])
        new["points"][0]["value"] = 0.005
        result = compare(ref, new, gate)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["rows"][0]["scale"], 0.0)
        new["points"][0]["value"] = 0.02
        self.assertFalse(compare(ref, new, gate)["accepted"])

    def test_existing_max_envelope_is_not_relaxed_into_sum(self):
        ref, new, gate = example(values=[1.0])
        gate.update(atol=0.1, rtol=0.1, rule="max")
        new["points"][0]["value"] = 1.15
        self.assertFalse(compare(ref, new, gate)["accepted"])
        gate["rule"] = "sum"
        self.assertTrue(compare(ref, new, gate)["accepted"])

    def test_bitwise_historical_migration_preserves_signed_zero(self):
        ref, new, gate = example(values=[0.0])
        gate.update(rule="bitwise", atol=0, rtol=0, evidence_level="historical")
        self.assertTrue(compare(ref, new, gate)["accepted"])
        new["points"][0]["value"] = -0.0
        result = compare(ref, new, gate)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["rows"][0]["reason"], "numeric_payload_mismatch")

    def test_bitwise_cannot_certify_new_physics_or_different_numerics(self):
        ref, new, gate = example()
        gate.update(rule="bitwise", atol=0, rtol=0)
        self.assert_rejected(ref, new, gate, "invalid_bitwise_gate")
        gate["evidence_level"] = "historical"
        gate["candidate_identity"]["numerics_sha256"] = new["identity"]["numerics_sha256"] = "b" * 64
        self.assert_rejected(ref, new, gate, "invalid_bitwise_gate")

    def test_bitwise_does_not_round_large_integers_or_change_numeric_type(self):
        ref, new, gate = example(values=[2**53])
        gate.update(rule="bitwise", atol=0, rtol=0, evidence_level="historical")
        new["points"][0]["value"] = 2**53 + 1
        result = compare(ref, new, gate)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["rows"][0]["absolute_error"], 1)
        new["points"][0]["value"] = float(2**53)
        self.assertFalse(compare(ref, new, gate)["accepted"])
        new["points"][0]["value"] = 2**53
        self.assertTrue(compare(ref, new, gate)["accepted"])

    def test_hysteresis_definitions_are_separate(self):
        paper = hysteresis_from_power(10.0, 20.0, "HI_P", forward_error_abs=0, reverse_error_abs=0)
        normalized = hysteresis_from_power(10.0, 20.0, "HI_normalized", forward_error_abs=0, reverse_error_abs=0)
        self.assertEqual(paper["value"], 1.0)
        self.assertEqual(normalized["value"], 0.5)

    def test_hysteresis_zero_denominators_are_unknown_not_zero(self):
        for definition, forward, reverse in [("HI_P", 0.0, 1.0), ("HI_normalized", 1.0, 0.0), ("HI_P", 0.0, 0.0)]:
            with self.subTest(definition=definition):
                result = hysteresis_from_power(forward, reverse, definition)
                self.assertEqual(result["status"], "unknown")
                self.assertIsNone(result["value"])
                self.assertEqual(result["reason"], "zero_denominator")

    def test_nonzero_unresolved_power_denominator_is_not_accepted(self):
        for definition, fwd, rev, f_error, r_error in [
            ("HI_P", 0.01, 2.0, 0.01, 0.001),
            ("HI_normalized", 2.0, 0.01, 0.001, 0.02),
        ]:
            with self.subTest(definition=definition):
                result = hysteresis_from_power(fwd, rev, definition, forward_error_abs=f_error, reverse_error_abs=r_error)
                self.assertIsNone(result["value"])
                self.assertEqual(result["reason"], "denominator_unresolved")

    def test_hi_needs_supplied_uncertainty_and_propagates_it(self):
        self.assertEqual(hysteresis_from_power(10.0, 20.0, "HI_P")["reason"], "power_uncertainty_missing")
        result = hysteresis_from_power(10.0, 20.0, "HI_P", forward_error_abs=1.0, reverse_error_abs=2.0)
        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["absolute_error_bound"], 22.0 / 9.0 - 2.0)
        tiny = hysteresis_from_power(10.0, 20.0, "HI_P", forward_error_abs=1e-20, reverse_error_abs=1e-20)
        self.assertGreater(tiny["absolute_error_bound"], 0)


if __name__ == "__main__":
    unittest.main()
