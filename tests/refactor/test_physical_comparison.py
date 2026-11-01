"""Synthetic comparator checks; these are not reference-device qualifications."""

import copy
import unittest

from physical_comparison import (
    DEFINITION_FIELDS, NORMS, compare_physical, definition_fingerprint,
)


def rule(limit, unit="1", scale=1.0, relative=0.0):
    return {"atol": limit, "rtol": relative, "scale": scale, "unit": unit,
            "basis": "Declared exact synthetic fixture; not a device acceptance allocation"}


def fixture(kind, values, thresholds, **parameters):
    identity = {k+"_sha256": "a"*64 for k in
                ("source", "input", "protocol", "initial_state", "model", "numerics", "environment")}
    identity["driver"] = "exact_synthetic_comparator_fixture"
    gate = {"kind": kind, "scope": "synthetic_fixture", "quantity": "test_observable",
            "comparison_target": {"kind": "analytic_fixture", "id": "declared_fixture"},
            "unit": "1", "norm": NORMS[kind], "support": "declared_fixture_support",
            "mask": None, "zero_policy": None, "thresholds": thresholds,
            "source_bindings": [{"path": __file__, "sha256": "a"*64}],
            "status": "frozen", "frozen_at": "2026-10-01T00:00:00Z",
            "independent_review": {"status": "approved", "evidence": "synthetic_test_fixture_only"},
            "reference_qualified": True, "reference_identity": identity,
            "candidate_identity": {**identity, "source_sha256": "b"*64},
            "reference_errors": {k: {"bound": 0, "unit": v["unit"], "evidence_sha256": "a"*64}
                                 for k, v in thresholds.items()}, **parameters}
    gate["definition_sha256"] = definition_fingerprint(gate)
    shared = {key: copy.deepcopy(gate[key]) for key in DEFINITION_FIELDS}
    shared["definition_sha256"] = gate["definition_sha256"]
    for key in ("volumes", "components", "coordinates"):
        if key in gate:
            shared[key] = copy.deepcopy(gate[key])
    reference = {**copy.deepcopy(shared), "identity": identity, "values": copy.deepcopy(values)}
    candidate = {**copy.deepcopy(shared), "identity": gate["candidate_identity"],
                 "values": copy.deepcopy(values), "run_started_at": "2026-10-02T00:00:00Z"}
    return reference, candidate, gate


def potential(**kw):
    return fixture("potential", [0.0, 1.0, 2.0], {"rms": rule(1e-6, "V"), "max": rule(1e-6, "V")},
                   unit="V", mask=[True]*3, volumes=[1, 2, 1], zero_policy="fixed_physical_scale",
                   gauge=kw.get("gauge", {"kind": "fixed_contact", "index": 0, "value_V": 0.0, "absolute_limit_V": 0.0}))


def density(values=(1.0, 2.0, 0.0), volumes=(1, 2, 1), components=("A", "A", "Z"), zero=(2,)):
    scales = {c: sum(v*w for v, w, name in zip(values, volumes, components) if name == c) for c in components}
    limits = {"log_rms": rule(1e-6), "log_max": rule(1e-6), "zero_density_max": rule(0, "m-3")}
    limits.update({"inventory:"+c: rule(0, "1", scale, 1e-6) for c, scale in scales.items()})
    return fixture("density", list(values), limits, unit="m-3", mask=[True]*len(values), volumes=list(volumes),
                   volume_unit="m3", inventory_unit="1", components=list(components), zero_cells=list(zero),
                   zero_policy="explicit_zero_cells_no_floor")


def complex_fixture(values=None, zero=()):
    values = [[1.0, 0.0], [-1.0, 1e-6]] if values is None else values
    return fixture("complex", values,
                   {"real": rule(1e-4, "S m-2"), "imag": rule(1e-4, "S m-2"),
                    "difference": rule(1e-4, "S m-2"), "phase": rule(1e-3, "rad")},
                   unit="S m-2", mask="all_frequencies", coordinates=list(range(1, len(values)+1)),
                   zero_policy="explicit_zero_else_resolved_phase", zero_indices=list(zero),
                   phase_floor=1e-8, phase_floor_evidence={"value": 1e-8, "sha256": "c"*64})


class PhysicalComparisonTests(unittest.TestCase):
    def test_potential_normal_and_nonuniform_error(self):
        a, b, g = potential()
        b["values"][1] += 1e-7
        self.assertTrue(compare_physical(a, b, g)["accepted"])
        b["values"][1] += 1e-4
        result = compare_physical(a, b, g)
        self.assertFalse(result["accepted"])
        self.assertFalse(result["metrics"]["rms"]["passed"])

    def test_floating_gauge_shift_and_fixed_contact_are_distinct(self):
        a, b, g = potential(gauge={"kind": "cv_zero_mean", "physical_gauge_freedom": True})
        b["values"] = [x+16 for x in b["values"]]
        self.assertTrue(compare_physical(a, b, g)["accepted"])
        a, b, g = potential()
        b["values"] = [x+16 for x in b["values"]]
        self.assertEqual(compare_physical(a, b, g)["code"], "gauge")

    def test_zero_potential_has_no_norm_of_phi_denominator(self):
        a, b, g = potential()
        a["values"] = b["values"] = [0.0]*3
        self.assertTrue(compare_physical(a, b, g)["accepted"])

    def test_density_normal_zero_and_negative_branches(self):
        a, b, g = density()
        b["values"][1] += 1e-7
        self.assertTrue(compare_physical(a, b, g)["accepted"])
        b["values"][2] = 1e-100
        self.assertFalse(compare_physical(a, b, g)["accepted"])
        b["values"][0] = -1
        self.assertEqual(compare_physical(a, b, g)["code"], "density_domain")

    def test_weak_density_has_no_artificial_floor(self):
        a, b, g = density((1e-200, 2e-200), (1, 1), ("A", "A"), ())
        b["values"][1] = 2.000001e-200
        self.assertTrue(compare_physical(a, b, g)["accepted"])
        b["values"][1] = 2.1e-200
        self.assertFalse(compare_physical(a, b, g)["accepted"])

    def test_density_large_integer_motion_is_not_rounded_to_binary64(self):
        a, b, g = density((2**60,), (1,), ("A",), ())
        g["thresholds"]["log_max"]["atol"] = 1e-20
        g["definition_sha256"] = definition_fingerprint(g)
        a["definition_sha256"] = b["definition_sha256"] = g["definition_sha256"]
        b["values"][0] += 1
        self.assertFalse(compare_physical(a, b, g)["accepted"])

    def test_inventory_cannot_cancel_between_disconnected_components(self):
        a, b, g = density((1.0, 1.0), (1, 1), ("A", "B"), ())
        b["values"] = [1.0+5e-7, 1.0-5e-7]
        for name in ("inventory:A", "inventory:B"):
            g["thresholds"][name]["rtol"] = 1e-8
        g["definition_sha256"] = definition_fingerprint(g)
        a["definition_sha256"] = b["definition_sha256"] = g["definition_sha256"]
        result = compare_physical(a, b, g)
        self.assertTrue(result["metrics"]["log_max"]["passed"])
        self.assertFalse(result["metrics"]["inventory:A"]["passed"])
        self.assertFalse(result["metrics"]["inventory:B"]["passed"])

    def test_conservative_refinement_preserves_cells_and_both_traces(self):
        limits = {"l1": rule(1e-6), "inventory:A": rule(1e-6), "inventory:B": rule(1e-6),
                  "trace:A:left": rule(1e-6, "m-3"), "trace:A:right": rule(1e-6, "m-3")}
        a, b, g = fixture("cells", [1.0, 3.0], limits, unit="m-3", mask="all_cells_and_both_interface_sides",
                          zero_policy="absolute_integrals_no_floor", area=1.0, area_unit="m2", edge_unit="m", integral_unit="1",
                          partitions=[{"id": "A", "left": 0, "right": .5}, {"id": "B", "left": .5, "right": 1}])
        a.update(edges=[0, .5, 1], representation="cell_average", traces={"A:left": 1, "A:right": 3})
        b.update(edges=[0, .25, .5, .75, 1], values=[1, 1, 3, 3], representation="cell_average", traces=dict(a["traces"]))
        self.assertTrue(compare_physical(a, b, g)["accepted"])
        b["traces"] = {"A:left": 2, "A:right": 2}
        self.assertFalse(compare_physical(a, b, g)["accepted"])
        b["edges"] = [0, .25, .6, .75, 1]
        self.assertEqual(compare_physical(a, b, g)["code"], "interface")

    def test_complex_circular_phase_crosses_branch_cut(self):
        a, b, g = complex_fixture()
        b["values"][1][1] = -1e-6
        result = compare_physical(a, b, g)
        self.assertTrue(result["accepted"])
        self.assertLess(result["metrics"]["phase"]["value"], 3e-6)

    def test_nd_overlap_requires_geometry_evidence_and_exact_conservation(self):
        a, b, g = fixture("cells", [1.0, 3.0],
                          {"l1": rule(1e-6), "inventory:A": rule(1e-6), "inventory:B": rule(1e-6)},
                          unit="m-3", mask="all_cells_and_both_interface_sides", zero_policy="absolute_integrals_no_floor",
                          volume_unit="m3", integral_unit="1", overlap=[[1, 1, 0], [0, 0, 2]],
                          reference_volumes=[2, 2], candidate_volumes=[1, 1, 2],
                          reference_components=["A", "B"], candidate_components=["A", "A", "B"], interface_ids=[],
                          projection_evidence={"status": "approved", "sha256": "c"*64, "geometry_verified": True})
        a.update(volumes=[2, 2], components=["A", "B"], representation="cell_average", traces={})
        b.update(values=[1, 1, 3], volumes=[1, 1, 2], components=["A", "A", "B"], representation="cell_average", traces={})
        self.assertTrue(compare_physical(a, b, g)["accepted"])
        g["overlap"][0][0] = .5
        g["definition_sha256"] = definition_fingerprint(g)
        a["definition_sha256"] = b["definition_sha256"] = g["definition_sha256"]
        self.assertEqual(compare_physical(a, b, g)["code"], "projection")

    def test_complex_component_error_cannot_expand_its_scale(self):
        a, b, g = complex_fixture()
        b["values"][0] = [10.0, 0.0]
        self.assertFalse(compare_physical(a, b, g)["accepted"])

    def test_complex_weak_phase_and_explicit_zero(self):
        a, b, g = complex_fixture([[1e-10, 0.0]])
        self.assertEqual(compare_physical(a, b, g)["status"], "unresolved")
        a, b, g = complex_fixture([[0.0, 0.0]], (0,))
        self.assertTrue(compare_physical(a, b, g)["accepted"])
        b["values"][0][0] = 1e-100
        self.assertEqual(compare_physical(a, b, g)["code"], "zero_signal_changed")

    def test_phase_relative_error_and_unbound_floor_reject(self):
        for field in ("phase_floor", "phase_floor_evidence"):
            a, b, g = complex_fixture()
            g.pop(field)
            self.assertFalse(compare_physical(a, b, g)["accepted"])
        a, b, g = complex_fixture()
        g["thresholds"]["phase"]["rtol"] = 0.1
        g["definition_sha256"] = definition_fingerprint(g)
        a["definition_sha256"] = b["definition_sha256"] = g["definition_sha256"]
        self.assertEqual(compare_physical(a, b, g)["code"], "phase_relative_error")

    def test_every_required_scope_identity_error_and_numeric_field_rejects_missing(self):
        for key in DEFINITION_FIELDS:
            a, b, g = potential(); b.pop(key)
            self.assertFalse(compare_physical(a, b, g)["accepted"], key)
        for key in ("reference_identity", "candidate_identity", "reference_errors", "independent_review", "reference_qualified"):
            a, b, g = potential(); g.pop(key)
            self.assertFalse(compare_physical(a, b, g)["accepted"], key)
        for key in ("atol", "rtol", "scale", "unit", "basis"):
            a, b, g = potential(); g["thresholds"]["max"].pop(key)
            self.assertFalse(compare_physical(a, b, g)["accepted"], key)

    def test_source_unit_rule_mutation_posthoc_and_error_budget_reject(self):
        a, b, g = potential(); b["identity"] = {**b["identity"], "input_sha256": "c"*64}
        self.assertEqual(compare_physical(a, b, g)["code"], "identity_mismatch")
        a, b, g = potential(); b["unit"] = "C"
        self.assertEqual(compare_physical(a, b, g)["code"], "measurement_mismatch")
        a, b, g = potential(); g["thresholds"]["max"]["scale"] = 100
        self.assertEqual(compare_physical(a, b, g)["code"], "definition_mismatch")
        a, b, g = potential(); g["frozen_at"] = b["run_started_at"]
        g["definition_sha256"] = definition_fingerprint(g)
        a["definition_sha256"] = b["definition_sha256"] = g["definition_sha256"]
        self.assertEqual(compare_physical(a, b, g)["code"], "posthoc_gate")
        a, b, g = potential(); g["reference_errors"]["max"]["bound"] = 1e-6
        g["definition_sha256"] = definition_fingerprint(g)
        a["definition_sha256"] = b["definition_sha256"] = g["definition_sha256"]
        self.assertEqual(compare_physical(a, b, g)["code"], "reference_budget")

    def test_M_preserves_signed_zero_types_and_only_declared_metadata_exceptions(self):
        values = {"numeric": [1, -0.0], "metadata": {"import_path": "old", "protocol": "fixed"}}
        a, b, g = fixture("semantic", values, {"mismatch": rule(0)}, mask="entire_payload",
                          zero_policy="exact_words", change_kind="M", metadata_exceptions=["import_path"])
        b["values"]["metadata"]["import_path"] = "new"
        self.assertTrue(compare_physical(a, b, g)["accepted"])
        b["values"]["numeric"][1] = 0.0
        self.assertFalse(compare_physical(a, b, g)["accepted"])
        a, b, g = fixture("semantic", values, {"mismatch": rule(0)}, mask="entire_payload",
                          zero_policy="exact_words", change_kind="N", metadata_exceptions=[])
        self.assertEqual(compare_physical(a, b, g)["code"], "word_equality_is_M_only")

    def test_full_R1_is_not_a_generic_physical_comparison(self):
        a, b, g = potential(); g["comparison_target"]["family"] = "R1_full"
        self.assertEqual(compare_physical(a, b, g)["code"], "R1_contract_required")
