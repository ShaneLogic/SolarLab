"""Small metadata-only failures; no scientific modules or fixtures are loaded."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.benchmarks.reference_map import MapError, check_comparison, digest, validate_map


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


if __name__ == "__main__":
    unittest.main()
