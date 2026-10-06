"""Small metadata-only failures; no scientific modules or fixtures are loaded."""

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from scripts.benchmarks.reference_map import (
    MapError, check_comparison, digest, validate_comparison_scopes, validate_map,
)


class ReferenceMapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "reproducibility").mkdir()
        coverage = {
            "capabilities": [
                {"id": "science", "route_id": "s", "origin_key": "test:science"},
                {"id": "helper", "route_id": "e", "origin_key": "export:helper"},
            ],
            "migration_routes": {
                "s": {"migration_tasks": ["P05-03"]},
                "e": {"migration_tasks": ["P03-01"]},
            },
        }
        manifest = {"scaps": {"groups": []}, "hi": {"cases": []},
                    "capability_reconciliation": {
                        "slice_references": [{"capability_id": "science"}],
                        "experiment_bindings": []}}
        self.sources = {}
        self.write_source("coverage", "coverage.json", coverage)
        self.write_source("gates", "gates.json", {
            "q": {"comparison": "absolute_linf", "limit": 1e-4, "units": "V"}})
        self.write_source("code", "oracle.py", "def compare():\n    assert 1 == 1\n")
        (self.root / "reproducibility/RefactorReferenceManifestV1.json").write_text(json.dumps(manifest))
        target = {"kind": "original_refinement_lane", "capability_id": "science",
                  "scope_id": "test:science"}
        value = {"comparison": "absolute_linf", "limit": 1e-4, "units": "V"}
        binding = {"source_id": "gates", "pointer": "/q", "value_sha256": digest(value)}
        self.record = {
            "capability_id": "science", "route_id": "s", "origin_key": "test:science",
            "target": target, "state": "original_rule_bound_pending_evidence",
            "citations": [{"source_id": "code", "symbol": "compare"}],
            "rule_ids": ["q"],
            "active_contract": None,
            "obligations": [{"prospective": True, "artifact": "future/scope.json",
                             "tasks": ["P05-03"], "missing": ["reference error"]}],
        }
        self.mapping = {
            "schema": "solarlab.refactor_comparison_map.v1", "sources": self.sources,
            "inputs": {"coverage_source_id": "coverage", "manifest_payload_hashes": {
                "/scaps/groups": digest([]), "/hi/cases": digest([])}},
            "scientific_records": [self.record], "test_link_audit": [],
            "rule_catalog": {"q": {"binding": binding, "gate_value": value,
                                    "unit": "V", "norm": "absolute_linf",
                                    "scale": {"kind": "absolute_original_units"},
                                    "zero_policy": {"kind": "absolute_comparison_no_division"}}},
            "engineering_routes": {"e": {"tasks": ["P03-01"], "physics_qualification": False}},
        }

    def write_source(self, sid, name, content):
        raw = content.encode() if isinstance(content, str) else json.dumps(content).encode()
        (self.root / name).write_bytes(raw)
        self.sources[sid] = {"root": "repository", "path": name,
                             "sha256": hashlib.sha256(raw).hexdigest()}

    def validate(self, mapping=None):
        return validate_map(mapping or self.mapping, self.root, self.root)

    def test_complete_partition_retains_pending_evidence(self):
        result = self.validate()
        self.assertEqual(result["scientific_records"], 1)
        self.assertEqual(result["engineering_capabilities"], 1)
        self.assertEqual(result["active_comparisons"], 0)
        with self.assertRaisesRegex(MapError, "comparison_blocked"):
            check_comparison(self.record, {})

    def test_dangling_source_symbol_and_pointer(self):
        for citation in ({"source_id": "absent"}, {"source_id": "code", "symbol": "absent"},
                         {"source_id": "gates", "pointer": "/absent", "value_sha256": "0" * 64}):
            with self.subTest(citation=citation):
                mapping = copy.deepcopy(self.mapping)
                mapping["scientific_records"][0]["citations"] = [citation]
                with self.assertRaises(MapError):
                    self.validate(mapping)

    def test_source_and_protected_payload_mutations_fail(self):
        (self.root / "oracle.py").write_text("def compare(): pass\n")
        with self.assertRaisesRegex(MapError, "source_hash_mismatch"):
            self.validate()
        self.write_source("code", "oracle.py", "def compare():\n    assert 1 == 1\n")
        self.mapping["inputs"]["manifest_payload_hashes"]["/hi/cases"] = "0" * 64
        with self.assertRaisesRegex(MapError, "input_payload_drift"):
            self.validate()

    def test_wrong_route_scope_and_obligation_fail(self):
        for field, value in (("route_id", "e"), ("origin_key", "other"),
                             ("target", {"capability_id": "other", "scope_id": "wrong"})):
            with self.subTest(field=field):
                mapping = copy.deepcopy(self.mapping)
                mapping["scientific_records"][0][field] = value
                with self.assertRaises(MapError):
                    self.validate(mapping)
        self.record["obligations"][0]["tasks"] = ["P09-01"]
        with self.assertRaisesRegex(MapError, "wrong_downstream_tasks"):
            self.validate()

    def test_gate_units_norm_scale_zero_and_threshold_cannot_drift(self):
        for field in ("unit", "norm", "scale", "zero_policy", "gate_value"):
            for value in (None, "wrong"):
                with self.subTest(field=field, value=value):
                    mapping = copy.deepcopy(self.mapping)
                    mapping["rule_catalog"]["q"][field] = value
                    with self.assertRaises(MapError):
                        self.validate(mapping)

    def test_engineering_route_cannot_claim_physics(self):
        self.mapping["engineering_routes"]["e"]["physics_qualification"] = True
        with self.assertRaisesRegex(MapError, "engineering_claims_physics"):
            self.validate()

    def active(self):
        identity = {field + "_sha256": "a" * 64 for field in
                    ("source", "input", "protocol", "initial_state", "model", "numerics", "environment")}
        identity["driver"] = "synthetic_fixture_only"
        contract = {"target": self.record["target"], "source_identity": identity,
                    "unit": "V", "norm": "absolute_linf", "scale": "fixed_reference",
                    "zero_policy": "absolute", "threshold": 1e-4, "review_status": "approved",
                    "review_evidence": "frozen/gate.json", "reference_qualified": True,
                    "reference_error": 1e-6, "frozen_at": "2026-10-05T00:00:00Z"}
        record = copy.deepcopy(self.record)
        record["active_contract"] = contract
        request = {**contract, "run_started_at": "2026-10-05T00:00:01Z"}
        return record, request

    def test_every_active_identity_quantity_field_is_required_and_exact(self):
        record, request = self.active()
        self.assertTrue(check_comparison(record, request)["metadata_eligible"])
        for field in ("target", "source_identity", "unit", "norm", "scale", "zero_policy"):
            with self.subTest(field=field):
                missing = dict(request)
                missing.pop(field)
                with self.assertRaises(MapError):
                    check_comparison(record, missing)
                wrong = {**request, field: "other"}
                with self.assertRaises(MapError):
                    check_comparison(record, wrong)

    def test_unknown_threshold_error_review_and_posthoc_freeze_fail(self):
        for field, value in (("threshold", None), ("reference_error", None),
                             ("threshold", float("nan")), ("reference_error", 1.0),
                             ("reference_qualified", False), ("review_status", "pending"),
                             ("review_evidence", None)):
            with self.subTest(field=field):
                record, request = self.active()
                record["active_contract"][field] = value
                with self.assertRaises(MapError):
                    check_comparison(record, request)
        record, request = self.active()
        request["run_started_at"] = "2026-10-04T00:00:00Z"
        with self.assertRaisesRegex(MapError, "posthoc"):
            check_comparison(record, request)

    def test_word_equality_is_only_M_with_metadata_exceptions(self):
        record, request = self.active()
        record["active_contract"].update(norm="word_equal", threshold=0, reference_error=0, change_kind="N")
        request["norm"] = "word_equal"
        with self.assertRaisesRegex(MapError, "M_only"):
            check_comparison(record, request)
        record["active_contract"]["change_kind"] = "M"
        with self.assertRaisesRegex(MapError, "metadata_exceptions"):
            check_comparison(record, request)
        record["active_contract"]["metadata_exceptions"] = ["timestamp"]
        self.assertFalse(check_comparison(record, request)["scientific_qualification_granted"])

    def test_missing_identity_component_and_dangling_rule_fail(self):
        record, request = self.active()
        record["active_contract"]["source_identity"] = {"source_sha256": "a" * 64}
        request["source_identity"] = record["active_contract"]["source_identity"]
        with self.assertRaisesRegex(MapError, "source_identity_incomplete"):
            check_comparison(record, request)
        self.record["rule_ids"] = ["absent"]
        with self.assertRaisesRegex(MapError, "dangling_rule"):
            self.validate()

    def test_active_contract_must_match_its_frozen_gate_artifact(self):
        record, request = self.active()
        policy = copy.deepcopy(record["active_contract"])
        self.write_source("active_gate", "active_gate.json", {"gate": policy})
        record["active_contract"]["gate_binding"] = {
            "source_id": "active_gate", "pointer": "/gate", "value_sha256": digest(policy)}
        record["active_contract"]["validation_run_started_at"] = request["run_started_at"]
        record["state"] = "active"
        self.mapping["scientific_records"] = [record]
        self.assertEqual(self.validate()["active_comparisons"], 1)
        record["active_contract"]["threshold"] = 2e-4
        with self.assertRaisesRegex(MapError, "active_gate_content_mismatch"):
            self.validate()

    def test_quoted_assertion_cannot_differ_from_bound_source(self):
        text = "    assert 1 == 1"
        rule = self.mapping["rule_catalog"]["q"]
        rule.pop("gate_value")
        rule["binding"] = {"source_id": "code", "symbol": "compare", "lines": [2, 2],
                           "text_sha256": hashlib.sha256(text.encode()).hexdigest()}
        rule["expression"] = "    assert 1 == 2"
        with self.assertRaisesRegex(MapError, "assertion_expression_mismatch"):
            self.validate()

    def test_configuration_consumer_keeps_its_scientific_parent(self):
        path = self.root / "reproducibility/RefactorReferenceManifestV1.json"
        manifest = json.loads(path.read_text())
        manifest["capability_reconciliation"]["configured_capability_bindings"] = [{
            "capability_id": "helper", "benchmark_reference_ids": ["science"],
            "refinement_reference_ids": []}]
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(MapError, "consumer_link_coverage"):
            self.validate()
        self.mapping["consumer_scientific_links"] = [{
            "capability_id": "helper", "route_id": "e", "scientific_record_ids": ["science"],
            "scientific_qualification_granted": False}]
        self.assertEqual(self.validate()["consumer_scientific_links"], 1)
        self.mapping["consumer_scientific_links"][0]["scientific_record_ids"] = []
        with self.assertRaisesRegex(MapError, "wrong_consumer_science_links"):
            self.validate()


class ComparisonScopeTests(unittest.TestCase):
    def setUp(self):
        self.base = ReferenceMapTests("test_complete_partition_retains_pending_evidence")
        self.base.setUp()
        self.addCleanup(self.base.doCleanups)
        self.base.record["comparison_scope_id"] = "scope-science"
        self.scopes = {
            "schema": "solarlab.refactor_comparison_scopes.v1", "sources": self.base.sources,
            "rules": {"exact": {
                "id": "exact", "status": "numeric_parameters_frozen", "quantity": "semantic payload",
                "unit": "typed payload", "norm": "exact_structure_and_words", "scale": 0,
                "zero_policy": "exact", "threshold": 0, "scope_boundary": "synthetic engineering fixture only",
                "parameter_justification": "Exact metadata identity; no physical tolerance",
                "comparator": {"source_id": "code", "symbol": "compare"},
            }},
            "records": [{"id": "scope-science", "capability_id": "science", "route_id": "s",
                         "origin_key": "test:science", "disposition": "engineering_only",
                         "positive_physical_obligation": False, "engineering_expected": {"count": 1},
                         "classification_basis": "Synthetic schema-only fixture",
                         "rule_ids": ["exact"], "active_comparison": None, "scientific_qualification_granted": False,
                         "remaining_inputs": [{"field": "source bound M output", "needed_from": "fixture",
                                               "blocked_use": "claiming execution was checked"}]}],
        }

    def validate(self):
        return validate_comparison_scopes(self.scopes, self.base.mapping, self.base.root, self.base.root)

    def test_scoped_partition_keeps_qualification_closed(self):
        self.assertEqual(self.validate()["records"], 1)
        self.assertEqual(self.validate()["active_new_comparisons"], 0)

    def test_unknown_numeric_field_and_vague_pending_record_reject(self):
        for field in ("quantity", "unit", "norm", "scale", "threshold", "zero_policy"):
            saved = self.scopes["rules"]["exact"].pop(field)
            with self.assertRaises(MapError):
                self.validate()
            self.scopes["rules"]["exact"][field] = saved
        self.scopes["records"][0]["remaining_inputs"][0].pop("needed_from")
        with self.assertRaisesRegex(MapError, "scope_vague_missing_input"):
            self.validate()

    def test_changed_source_wrong_scope_and_unadmitted_active_gate_reject(self):
        self.scopes["sources"]["code"]["sha256"] = "0"*64
        with self.assertRaisesRegex(MapError, "scope_source_changed"):
            self.validate()
        self.base.write_source("code", "oracle.py", "def compare():\n    assert 1 == 1\n")
        self.scopes["records"][0]["route_id"] = "other"
        with self.assertRaisesRegex(MapError, "scope_wrong_route"):
            self.validate()
        self.scopes["records"][0]["route_id"] = "s"
        self.scopes["records"][0]["active_comparison"] = {"threshold": 1}
        with self.assertRaisesRegex(MapError, "scope_unadmitted_active_comparison"):
            self.validate()

    def test_dangling_comparator_and_unmarked_missing_parameters_reject(self):
        rule = self.scopes["rules"]["exact"]
        rule["comparator"]["symbol"] = "absent"
        with self.assertRaisesRegex(MapError, "scope_dangling_symbol"):
            self.validate()
        rule["comparator"]["symbol"] = "compare"
        rule["status"] = "missing_parameters"
        with self.assertRaisesRegex(MapError, "scope_unmarked_missing_parameters"):
            self.validate()
        rule.update(active=False, missing_fields=["numeric reference scale"])
        self.assertEqual(self.validate()["rules"], 1)

    def test_original_rule_value_is_not_silently_rebound(self):
        rule = self.scopes["rules"]["exact"]
        rule.update(status="original_contract_only", original_rule_ids=["q"],
                    original_rules_sha256=digest({"q": self.base.mapping["rule_catalog"]["q"]}))
        self.assertEqual(self.validate()["rules"], 1)
        self.base.mapping["rule_catalog"]["q"]["gate_value"]["limit"] = 0.1
        with self.assertRaisesRegex(MapError, "scope_original_rule_changed"):
            self.validate()

    def test_actual_scope_links_and_protected_payloads(self):
        repo = Path(__file__).resolve().parents[2]
        scopes = json.loads((repo/"reproducibility/RefactorComparisonScopesV1.json").read_text())
        mapping = json.loads((repo/"reproducibility/RefactorComparisonMapV1.json").read_text())
        archive = Path(os.environ.get("SOLARLAB_ARCHIVE", str(
            Path.home()/"Library/Mobile Documents/com~apple~CloudDocs/projects/solarlab")))
        if not archive.is_dir():
            self.skipTest("Source-bound archive is verified separately when available")
        result = validate_comparison_scopes(scopes, mapping, repo, archive)
        self.assertEqual(result["records"], 170)
        manifest = json.loads((repo/"reproducibility/RefactorReferenceManifestV1.json").read_text())
        self.assertEqual(digest(manifest["scaps"]["groups"]), scopes["protected_manifest_payload_sha256"]["/scaps/groups"])
        self.assertEqual(digest(manifest["hi"]["cases"]), scopes["protected_manifest_payload_sha256"]["/hi/cases"])
        self.assertEqual(sum(r["disposition"] == "engineering_only" for r in scopes["records"]), 2)

    def test_actual_short_R1_source_is_bound_without_inventing_a_test(self):
        import ast
        repo = Path(__file__).resolve().parents[2]
        scopes = json.loads((repo/"reproducibility/RefactorComparisonScopesV1.json").read_text())
        rule = scopes["rules"]["R1_short_original"]
        source = scopes["sources"]["r1_short"]
        raw = (repo/source["path"]).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), source["sha256"])
        function = next(n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name == "run_coupled")
        limits = next(ast.literal_eval(n.value) for n in ast.walk(function)
                      if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "limits" for t in n.targets))
        self.assertEqual(rule["source_limits"], limits)
        self.assertIsNone(rule["existing_test_for_wrapper"])
        self.assertEqual(rule["times_s"], [0, 1e-8, 1e-6, 1e-4])
        self.assertIn("Not R1-1", rule["scope_boundary"])


class NumericParameterPreparationTests(unittest.TestCase):
    """Actual source-bound parameters stay separate from reference admission."""

    @classmethod
    def setUpClass(cls):
        cls.repo = Path(__file__).resolve().parents[2]
        cls.scopes_path = cls.repo / "reproducibility/RefactorComparisonScopesV1.json"
        cls.scopes = json.loads(cls.scopes_path.read_text())
        cls.mapping_path = cls.repo / "reproducibility/RefactorComparisonMapV1.json"
        cls.mapping = json.loads(cls.mapping_path.read_text())
        cls.manifest = json.loads((cls.repo / "reproducibility/RefactorReferenceManifestV1.json").read_text())
        cls.analytic = json.loads((cls.repo / "reproducibility/RefactorAnalyticReferencesV1.json").read_text())
        cls.packets = [p for row in cls.scopes["records"] for p in row.get("parameter_bindings", [])]

    def test_parameter_sources_and_analytic_pointers_are_bound(self):
        import ast
        for packet in self.packets:
            self.assertIn(packet["family_rule_id"], self.scopes["rules"])
            self.assertTrue(packet["applicability"])
            self.assertTrue(packet["remaining_parameters"])
            for citation in packet["citations"]:
                source = self.scopes["sources"][citation["source_id"]]
                if source["root"] != "repository":
                    continue  # The existing archive-aware scope test checks the plan.
                raw = (self.repo / source["path"]).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), source["sha256"])
                if "lines" in citation:
                    lo, hi = citation["lines"]
                    text = "\n".join(raw.decode().splitlines()[lo - 1:hi])
                    self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), citation["text_sha256"])
                if "symbol" in citation:
                    self.assertIn(citation["symbol"], {n.name for n in ast.walk(ast.parse(raw))
                                                      if isinstance(n, (ast.FunctionDef, ast.ClassDef))})
            if "analytic_reference_binding" in packet:
                binding = packet["analytic_reference_binding"]
                index = int(binding["pointer"].rsplit("/", 1)[1])
                original = self.analytic["definitions"][index]
                self.assertEqual(original["id"], binding["definition_id"])
                self.assertEqual(digest(original), binding["value_sha256"])

    def test_limits_are_finite_with_named_units_and_fixed_scales(self):
        import math

        def check(value):
            if isinstance(value, dict):
                if "atol" in value:
                    self.assertTrue(value["unit"])
                    for key in ("atol", "rtol"):
                        self.assertIn(type(value[key]), (int, float))
                        self.assertTrue(math.isfinite(value[key]) and value[key] >= 0)
                    scale = value.get("scale", value.get("fixed_reference_scale"))
                    self.assertIn(type(scale), (int, float))
                    self.assertTrue(math.isfinite(scale) and scale >= 0)
                for child in value.values():
                    check(child)
            elif isinstance(value, list):
                for child in value:
                    check(child)

        for packet in self.packets:
            check(packet["limits"])
            eligibility = packet["reference_eligibility"]
            self.assertIsNone(eligibility["reference_error_bound"])
            self.assertFalse(eligibility["reference_qualified"])
            self.assertIsNone(eligibility["active_comparison"])
            self.assertEqual(eligibility["default_reference_error_share"], "1/3")
        self.assertEqual({p["family_rule_id"] for p in self.packets}, {
            "potential_packet", "density_inventory_packet", "current_packet",
            "operator_packet", "JV_metric_packet", "complex_packet",
            "projection_packet", "spectrum_packet", "transient_packet", "thermal_packet",
        })

    def test_affine_geometry_is_original_and_numeric_definition_does_not_admit_use(self):
        from physical_comparison import compare_physical, definition_fingerprint, validate_definition
        from reference_comparison import Rejected
        original = next(d for d in self.analytic["definitions"]
                        if d["id"] == "poisson_affine_51_stored_nodes")
        for packet in self.packets:
            if "physical_definition" not in packet:
                continue
            gate = copy.deepcopy(packet["physical_definition"])
            self.assertTrue(validate_definition(gate))
            self.assertEqual(gate["coordinates_m"], original["inputs"]["x_m"]["values"])
            weights = original["inputs"]["CV_weights"]["binary64"]
            self.assertEqual(gate["volumes"], weights["values"] if isinstance(weights, dict) else weights)
            self.assertEqual(len(gate["volumes"]), 51)
            self.assertTrue(all(gate["mask"]))
            self.assertEqual(gate["gauge"]["value_V"], 0.0)
            gate["definition_sha256"] = definition_fingerprint(gate)
            table = {"definition_sha256": gate["definition_sha256"]}
            result = compare_physical(table, table, gate)
            self.assertFalse(result["accepted"])
            self.assertEqual(result["code"], "unreviewed_gate")
            gate.pop("thresholds")
            with self.assertRaises(Rejected):
                validate_definition(gate)

    def test_source_inventory_zero_and_phase_branches_remain_distinct(self):
        import math
        cases = {p["case_id"]: p for p in self.packets}
        inventory = cases["mobile_single_and_dual_absorber_inventory"]
        values = inventory["known_parameters"]
        self.assertEqual(values["positive_inventory_m2"], 4e15)
        self.assertEqual(values["negative_inventory_m2_dual"], 2.8e15)
        self.assertEqual(inventory["limits"]["positive_drift"]["scale"], 4e15)
        self.assertEqual(inventory["limits"]["negative_drift_dual"]["scale"], 2.8e15)
        self.assertEqual(inventory["limits"]["inactive_density"]["atol"], 0.0)
        self.assertEqual(inventory["limits"]["positive_drift"]["operator"], "lt")
        lockin = cases["ion_aware_lockin_three_frequencies"]
        self.assertEqual(lockin["limits"]["frequency_phase"]["atol"], 0.01 * math.pi / 180.0)
        self.assertEqual(lockin["limits"]["frequency_phase"]["unit"], "rad")
        self.assertEqual(lockin["limits"]["frequency_phase"]["rtol"], 0.0)
        self.assertEqual(lockin["numeric_parameter_status"], "partially_specified")
        transient = cases["original_clean_exponential_fit_and_zero_perturbation"]
        self.assertIsNone(transient["known_parameters"]["zero_case_lifetime"])
        self.assertEqual(transient["known_parameters"]["zero_case_samples"], 100)

    def test_links_and_protected_payloads_match_and_every_scope_stays_closed(self):
        scopes_hash = hashlib.sha256(self.scopes_path.read_bytes()).hexdigest()
        map_hash = hashlib.sha256(self.mapping_path.read_bytes()).hexdigest()
        self.assertEqual(self.mapping["comparison_scopes"]["sha256"], scopes_hash)
        self.assertEqual(self.manifest["capability_reconciliation"]["comparison_scopes"]["sha256"], scopes_hash)
        self.assertEqual(self.manifest["capability_reconciliation"]["comparison_map"]["sha256"], map_hash)
        protected = self.scopes["protected_manifest_payload_sha256"]
        self.assertEqual(digest(self.manifest["scaps"]["groups"]), protected["/scaps/groups"])
        self.assertEqual(digest(self.manifest["hi"]["cases"]), protected["/hi/cases"])
        self.assertEqual(sum(r["status"] == "missing_parameters" for r in self.scopes["rules"].values()), 10)
        for row in self.mapping["scientific_records"]:
            with self.assertRaisesRegex(MapError, "comparison_blocked"):
                check_comparison(row, {})


class ComparisonScaleClosureTests(unittest.TestCase):
    """Concrete source geometry/operators, without importing the old engine."""

    @classmethod
    def setUpClass(cls):
        cls.repo = Path(__file__).resolve().parents[2]
        cls.scopes = json.loads((cls.repo / "reproducibility/RefactorComparisonScopesV1.json").read_text())
        cls.mapping = json.loads((cls.repo / "reproducibility/RefactorComparisonMapV1.json").read_text())
        cls.closures = {c["id"]: c for row in cls.scopes["records"] for c in row.get("scale_closures", [])}

    def test_original_full_endpoint_measure_and_signed_boundary_are_bound(self):
        import ast
        from fractions import Fraction
        from physical_comparison import _potential, validate_definition

        item = self.closures["foundation_v1_uniform44_potential"]
        gate = item["physical_definition"]
        self.assertTrue(validate_definition(gate))
        x, w = gate["coordinates_m"], gate["volumes"]
        self.assertEqual(len(x), 45)
        self.assertEqual(w[0], x[1] - x[0])
        self.assertEqual(w[-1], x[-1] - x[-2])
        self.assertNotEqual(w[0], (x[1] - x[0]) / 2)
        self.assertGreater(sum(w), x[-1] - x[0])
        source = self.scopes["sources"]["legacy_dual_measure"]
        raw = (self.repo / source["path"]).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), source["sha256"])
        function = next(n for n in ast.parse(raw).body
                        if isinstance(n, ast.FunctionDef) and n.name == "dual_cell_widths")
        self.assertIn("w[0] = dx[0]", ast.unparse(function))
        self.assertIn("w[-1] = dx[-1]", ast.unparse(function))
        material = self.scopes["sources"]["legacy_material_geometry"]
        tree = ast.parse((self.repo / material["path"]).read_bytes())
        boundary = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == "poisson_right_boundary")
        self.assertIn("mat.V_bi_bc - mat.junction_polarity * float(V_app)", ast.unparse(boundary))
        # A right-end perturbation discriminates the full and half measures.
        reference = {"values": [0.0] * 45, "volumes": w}
        candidate = {"values": [0.0] * 44 + [1e-8], "volumes": w}
        measured = _potential(reference, candidate, gate)
        expected_squared = Fraction(w[-1]) * Fraction(1e-8) ** 2 / sum(map(Fraction, w))
        self.assertEqual(measured["rms"][1], expected_squared)
        affine = next(p["physical_definition"] for row in self.scopes["records"]
                      for p in row.get("parameter_bindings", []) if "physical_definition" in p)
        self.assertNotEqual(affine["comparison_target"], gate["comparison_target"])
        self.assertEqual(len(affine["volumes"]), 51)
        self.assertEqual(affine["zero_policy"], "fixed_physical_scale")

    def test_zero_species_definition_and_unreviewed_use_are_separate(self):
        from fractions import Fraction
        from physical_comparison import _density, compare_physical, definition_fingerprint, validate_definition

        gate = copy.deepcopy(self.closures["foundation_v1_uniform44_zero_positive_ion"]["physical_definition"])
        self.assertTrue(validate_definition(gate))
        reference = {"values": [0.0] * 45, "volumes": gate["volumes"], "components": gate["components"]}
        candidate = copy.deepcopy(reference)
        candidate["values"][10] = 0.25
        metrics = _density(reference, candidate, gate)
        self.assertEqual(metrics["zero_density_max"], Fraction(0.25))
        self.assertEqual(metrics["inventory:positive_ion_device"], Fraction(gate["volumes"][10]) / 4)
        self.assertEqual(gate["thresholds"]["zero_density_max"]["atol"], 0)
        gate["definition_sha256"] = definition_fingerprint(gate)
        for table in (reference, candidate):
            table["definition_sha256"] = gate["definition_sha256"]
        result = compare_physical(reference, candidate, gate)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["code"], "unreviewed_gate")

    def test_all47_original_lane_definitions_reuse_exact_operators_and_limits(self):
        from collections import Counter
        lanes = [r for r in self.scopes["records"] if r["origin_key"].startswith("lane:")]
        self.assertEqual(len(lanes), 47)
        counts = Counter()
        for row in lanes:
            definition = row["original_numeric_definition"]
            group = self.scopes["rules"][definition["rule_group_id"]]
            self.assertEqual(definition["original_rules_sha256"], group["original_rules_sha256"])
            ids = [rid for values in definition["operators"].values() for rid in values]
            self.assertCountEqual(ids, group["original_rule_ids"])
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(definition["remaining_numeric_parameters_for_original_target"], [])
            self.assertFalse(definition["new_N_applicability_approved"])
            self.assertIsNone(definition["eligibility"]["reference_error_bound"])
            for operator, values in definition["operators"].items():
                for rid in values:
                    bound = self.mapping["rule_catalog"][rid]["gate_value"]
                    self.assertEqual(operator, bound.get("comparison", bound.get("operator")))
                    self.assertTrue(bound["units"])
                    counts[operator] += 1
        self.assertEqual(dict(counts), self.scopes["comparison_scale_closure"]["original_lane_operator_counts"])
        self.assertEqual(counts["absolute_linf"], 299)
        self.assertEqual(counts["relative_linf"] + counts["pointwise_relative_linf"], 78)

    def test_reference_only_TPV_scale_units_and_gap_categories(self):
        import ast
        tpv = self.closures["original_tpv_reference_amplitude_waveform"]
        definition = tpv["definition"]
        self.assertEqual(definition["amplitude_scale_formula"], "A_ref=max(abs(reference.delta_V))")
        self.assertFalse(definition["scale_depends_on_candidate_output"])
        self.assertEqual(definition["reference_terminal_charge_voltage_error"]["limit_formula"], "min(1e-6,0.01*A_ref)")
        source = self.scopes["sources"]["original_tpv_refinement"]
        tree = ast.parse((self.repo / source["path"]).read_bytes())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "_require_converged_waveform")
        amplitude = next(n for n in fn.body if isinstance(n, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == "amplitude" for t in n.targets))
        self.assertIn("fine.delta_V", ast.unparse(amplitude.value))
        self.assertNotIn("coarse", ast.unparse(amplitude.value))
        units = self.closures["EL_exact_output_units"]["definition"]
        self.assertEqual(units["spectrum_output_unit"], "photons m-2 s-1 nm-1")
        self.assertEqual(units["per_nm_to_per_m_multiplier"], 1e9)
        self.assertEqual(units["lambda_nm_to_m_multiplier"], 1e-9)
        self.assertEqual(len(self.scopes["records"]), 170)
        for row in self.scopes["records"]:
            gap = row["gap_classification"]
            self.assertIsNone(gap["reference_error_bound"])
            self.assertFalse(gap["fixed_discrete_target_requires_continuum_qualification"])
            self.assertTrue(gap["reference_eligibility_is_separate"])
            for need in gap["scope_specific_prior_gaps"]:
                self.assertTrue(need["categories"])
                self.assertTrue(need["needed_from"])
        for closure in self.closures.values():
            self.assertFalse(closure["eligibility"]["reference_qualified"])
            self.assertIsNone(closure["eligibility"]["reference_error_bound"])


if __name__ == "__main__":
    unittest.main()
