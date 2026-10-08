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


if __name__ == "__main__":
    unittest.main()
