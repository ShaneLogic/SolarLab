"""Validate P00 comparison links without importing or running scientific code.

The map records original assertions and blocked prospective obligations.  A
valid map is planning evidence, not permission to use an unqualified reference.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path


class MapError(ValueError):
    """A comparison link or its declared eligibility is invalid."""


def digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode()).hexdigest()


def pointer(value, path):
    if not isinstance(path, str) or (path and not path.startswith("/")):
        raise MapError("invalid_pointer")
    try:
        for part in path.split("/")[1:]:
            part = part.replace("~1", "/").replace("~0", "~")
            value = value[int(part)] if isinstance(value, list) else value[part]
    except (KeyError, IndexError, ValueError, TypeError) as exc:
        raise MapError(f"dangling_pointer: {path}") from exc
    return value


def _require(condition, message):
    if not condition:
        raise MapError(message)


def _symbols(tree):
    result = set()

    def visit(nodes, prefix=""):
        for node in nodes:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                name = prefix + node.name
                result.add(name)
                visit(node.body, name + ".")

    visit(tree.body)
    return result


def check_comparison(record, supplied):
    """Check an explicitly active binding; never invent missing qualification.

    This checks metadata only.  The bound original comparator still has to run.
    """
    contract = record.get("active_contract")
    _require(isinstance(contract, dict), "comparison_blocked: no active contract")
    for field in ("target", "source_identity", "unit", "norm", "scale", "zero_policy"):
        _require(contract.get(field) is not None, f"active_missing_{field}")
        _require(field in supplied and supplied[field] is not None,
                 f"missing_{field}")
        _require(supplied[field] == contract[field], f"mismatched_{field}")
    _require(contract["target"] == record["target"], "wrong_scope")
    identity = contract["source_identity"]
    _require(isinstance(identity, dict), "source_identity_incomplete")
    for field in ("source", "input", "protocol", "initial_state", "model", "numerics", "environment"):
        value = identity.get(field + "_sha256")
        _require(isinstance(value, str) and len(value) == 64
                 and all(c in "0123456789abcdef" for c in value), "source_identity_incomplete")
    _require(isinstance(identity.get("driver"), str) and bool(identity["driver"]),
             "source_identity_incomplete")
    for field in ("threshold", "reference_error"):
        value = contract.get(field)
        _require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                 f"unknown_{field}")
    _require(contract.get("review_status") == "approved", "unreviewed_comparison")
    _require(bool(contract.get("review_evidence")), "review_evidence_missing")
    try:
        frozen = datetime.fromisoformat(contract["frozen_at"].replace("Z", "+00:00"))
        started = datetime.fromisoformat(supplied["run_started_at"].replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise MapError("freeze_or_start_missing") from exc
    _require(frozen.tzinfo is not None and started.tzinfo is not None
             and frozen < started, "posthoc_or_ambiguous_freeze")
    _require(contract.get("reference_qualified") is True,
             "reference_qualification_missing")
    _require(contract.get("reference_error") is not None,
             "reference_error_missing")
    _require(contract["reference_error"] <= contract["threshold"] / 3,
             "reference_error_budget_exceeded")
    if contract.get("norm") == "word_equal":
        _require(contract.get("change_kind") == "M", "word_equality_is_M_only")
        _require(contract.get("threshold") == 0, "word_equality_threshold")
        _require(isinstance(contract.get("metadata_exceptions"), list),
                 "metadata_exceptions_missing")
    return {"metadata_eligible": True, "scientific_qualification_granted": False}


def validate_comparison_scopes(scopes, mapping, repo, archive):
    """Validate the small prospective rule table, not the scientific results.

    Existing lane rules retain their own meaning. Missing physical scale/floor
    values remain named blockers rather than being filled by test truth values.
    """
    _require(scopes.get("schema") == "solarlab.refactor_comparison_scopes.v1", "scope_schema")
    repo, archive = Path(repo), Path(archive)
    cache, trees = {}, {}
    sources = scopes["sources"]
    for sid, entry in sources.items():
        _require(entry.get("root") in {"repository", "archive"}, "scope_source_root")
        path = (repo if entry["root"] == "repository" else archive)/entry["path"]
        _require(path.is_file(), "scope_source_missing")
        raw = path.read_bytes()
        _require(hashlib.sha256(raw).hexdigest() == entry["sha256"], "scope_source_changed")
        cache[sid] = raw

    def cite(ref):
        _require(ref.get("source_id") in cache, "scope_dangling_source")
        raw = cache[ref["source_id"]]
        if "symbol" in ref:
            sid = ref["source_id"]
            if sid not in trees:
                trees[sid] = _symbols(ast.parse(raw))
            _require(ref["symbol"] in trees[sid], "scope_dangling_symbol")
        if "lines" in ref:
            lo, hi = ref["lines"]; lines = raw.decode().splitlines()
            _require(1 <= lo <= hi <= len(lines), "scope_source_lines")
            _require(hashlib.sha256("\n".join(lines[lo-1:hi]).encode()).hexdigest() == ref["text_sha256"],
                     "scope_source_text_changed")

    rules = scopes["rules"]
    for rid, rule in rules.items():
        _require(rule.get("id") == rid, "scope_rule_identity")
        _require(rule.get("status") in {"numeric_parameters_frozen", "original_contract_only", "missing_parameters"},
                 "scope_rule_status")
        cite(rule["comparator"])
        _require(bool(rule.get("scope_boundary")), "scope_rule_boundary")
        for binding in rule.get("citations", []):
            cite(binding)
        if rule["status"] == "numeric_parameters_frozen":
            for key in ("quantity", "unit", "norm", "scale", "zero_policy", "threshold"):
                _require(rule.get(key) is not None, "scope_missing_"+key)
            def finite_numbers(value):
                if isinstance(value, dict):
                    return bool(value) and all(finite_numbers(v) for v in value.values())
                if isinstance(value, list):
                    return bool(value) and all(finite_numbers(v) for v in value)
                return type(value) in (int, float) and math.isfinite(value) and value >= 0
            _require(finite_numbers(rule["scale"]), "scope_unknown_scale")
            _require(finite_numbers(rule["threshold"]), "scope_unknown_threshold")
            _require(bool(rule.get("parameter_justification")), "scope_parameter_basis")
        elif rule["status"] == "missing_parameters":
            _require(bool(rule.get("missing_fields")) and rule.get("active") is False, "scope_unmarked_missing_parameters")
        if rule.get("original_rule_ids"):
            _require(set(rule["original_rule_ids"]) <= mapping["rule_catalog"].keys(), "scope_original_rule_missing")
            _require(rule.get("original_rules_sha256") == digest({k: mapping["rule_catalog"][k] for k in rule["original_rule_ids"]}),
                     "scope_original_rule_changed")

    original = {r["capability_id"]: r for r in mapping["scientific_records"]}
    records = scopes["records"]
    _require(len({r["capability_id"] for r in records}) == len(records) and
             {r["capability_id"] for r in records} == original.keys(), "scope_record_coverage")
    for row in records:
        old = original[row["capability_id"]]
        _require(row["route_id"] == old["route_id"] and row["origin_key"] == old["origin_key"], "scope_wrong_route")
        _require(old.get("comparison_scope_id") == row["id"], "scope_link_mismatch")
        _require(set(row["rule_ids"]) <= rules.keys() and bool(row["rule_ids"]), "scope_dangling_rule")
        _require(row.get("active_comparison") is None and row.get("scientific_qualification_granted") is False,
                 "scope_unadmitted_active_comparison")
        _require(bool(row.get("classification_basis")), "scope_classification_basis")
        for binding in row.get("citations", []):
            cite(binding)
        _require(bool(row.get("remaining_inputs")), "scope_missing_qualification_obligation")
        for need in row["remaining_inputs"]:
            _require(all(need.get(k) for k in ("field", "needed_from", "blocked_use")), "scope_vague_missing_input")
        if row.get("disposition") == "engineering_only":
            _require(row.get("positive_physical_obligation") is False and bool(row.get("engineering_expected")),
                     "scope_engineering_disposition")
        else:
            _require(row.get("positive_physical_obligation") is True, "scope_physics_obligation_erased")
    return {"records": len(records), "rules": len(rules), "source_files_checked": len(cache),
            "active_new_comparisons": 0, "scientific_qualification_granted": False}


def validate_map(mapping, repo, archive):
    """Resolve every citation and partition against the frozen coverage index."""
    repo, archive = Path(repo), Path(archive)
    _require(mapping.get("schema") == "solarlab.refactor_comparison_map.v1", "schema")
    documents, trees, blobs = {}, {}, {}
    sources = mapping["sources"]

    def source_bytes(source_id):
        _require(source_id in sources, f"dangling_source: {source_id}")
        if source_id not in blobs:
            source = sources[source_id]
            _require(source["root"] in {"repository", "archive"}, "source_root")
            base = repo if source["root"] == "repository" else archive
            path = base / source["path"]
            _require(path.is_file(), f"missing_source: {path}")
            content = path.read_bytes()
            _require(hashlib.sha256(content).hexdigest() == source["sha256"],
                     f"source_hash_mismatch: {path}")
            blobs[source_id] = content
        return blobs[source_id]

    def document(source_id):
        if source_id not in documents:
            content = source_bytes(source_id)
            if Path(sources[source_id]["path"]).suffix in {".yaml", ".yml"}:
                import yaml
                documents[source_id] = yaml.safe_load(content)
            else:
                documents[source_id] = json.loads(content)
        return documents[source_id]

    def citation(ref):
        _require(not ref.get("prospective"), "prospective_citation_used_as_existing")
        sid = ref["source_id"]
        content = source_bytes(sid)
        if "pointer" in ref:
            value = pointer(document(sid), ref["pointer"])
            _require(digest(value) == ref["value_sha256"], "pointer_value_mismatch")
        if "symbol" in ref:
            if sid not in trees:
                trees[sid] = _symbols(ast.parse(content))
            _require(ref["symbol"] in trees[sid],
                     f"dangling_symbol: {sources[sid]['path']}::{ref['symbol']}")
        if "lines" in ref:
            lo, hi = ref["lines"]
            lines = content.decode().splitlines()
            _require(1 <= lo <= hi <= len(lines), "invalid_line_range")
            text = "\n".join(lines[lo - 1:hi])
            _require(hashlib.sha256(text.encode()).hexdigest() == ref["text_sha256"],
                     "assertion_text_mismatch")

    coverage = document(mapping["inputs"]["coverage_source_id"])
    caps = {row["id"]: row for row in coverage["capabilities"]}
    routes = coverage["migration_routes"]
    manifest = json.loads((repo / "reproducibility/RefactorReferenceManifestV1.json").read_bytes())
    for path, expected in mapping["inputs"]["manifest_payload_hashes"].items():
        _require(digest(pointer(manifest, path)) == expected, f"input_payload_drift: {path}")
    reconciliation = manifest["capability_reconciliation"]
    link = reconciliation.get("comparison_map")
    if link:
        raw_map = (repo / link["path"]).read_bytes()
        _require(hashlib.sha256(raw_map).hexdigest() == link["sha256"], "map_link_hash_mismatch")
        _require(digest(json.loads(raw_map)) == digest(mapping), "map_argument_mismatch")
    expected = {row["capability_id"] for key in ("slice_references", "experiment_bindings")
                for row in reconciliation[key]}
    records = mapping["scientific_records"]
    _require(len({row["capability_id"] for row in records}) == len(records),
             "duplicate_capability")
    _require({row["capability_id"] for row in records} == expected,
             "scientific_coverage_mismatch")
    states = {"original_rule_bound_pending_evidence", "original_assertions_bound",
              "blocked_prospective", "active"}
    for record in records:
        cid = record["capability_id"]
        _require(cid in caps, "dangling_capability")
        _require(record["route_id"] == caps[cid]["route_id"], "wrong_route")
        _require(record["origin_key"] == caps[cid]["origin_key"], "wrong_origin_scope")
        _require(record["state"] in states, "invalid_state")
        target = record["target"]
        _require(target.get("capability_id") == cid
                 and target.get("scope_id") == record["origin_key"],
                 "wrong_scope")
        _require(target.get("kind") in {"original_refinement_lane", "original_scoped_assertions",
                                        "prospective_comparison"}, "invalid_target")
        for ref in record["citations"]:
            citation(ref)
        for rule_id in record["rule_ids"]:
            _require(rule_id in mapping["rule_catalog"], "dangling_rule")
            rule = mapping["rule_catalog"][rule_id]
            for field in ("unit", "norm", "scale", "zero_policy"):
                _require(rule.get(field) is not None, f"missing_rule_{field}")
            citation(rule["binding"])
            if "expression" in rule:
                _require(hashlib.sha256(rule["expression"].encode()).hexdigest()
                         == rule["binding"].get("text_sha256"), "assertion_expression_mismatch")
            if "gate_value" in rule:
                value = pointer(document(rule["binding"]["source_id"]), rule["binding"]["pointer"])
                _require(value == rule["gate_value"], "rule_changed_original_gate")
                _require(rule["unit"] == value.get("units", "1"), "rule_unit_mismatch")
                _require(rule["norm"] == value.get("comparison", value.get("operator")),
                         "rule_norm_mismatch")
                _require(value.get("limit") is not None, "unknown_original_threshold")
                relative = value.get("comparison") in {"relative_linf", "pointwise_relative_linf"}
                floor = value.get("relative_floor", 1e-30)
                scale = ({"kind": "original_symmetric_pair_scale", "relative_floor": floor}
                         if relative else {"kind": "absolute_original_units"})
                zero = ({"kind": "original_relative_floor", "value": floor}
                        if relative else {"kind": "absolute_comparison_no_division"})
                _require(rule["scale"] == scale, "rule_scale_mismatch")
                _require(rule["zero_policy"] == zero, "rule_zero_policy_mismatch")
        if record["state"] == "active":
            contract = record.get("active_contract", {})
            _require("gate_binding" in contract and "pointer" in contract["gate_binding"],
                     "active_gate_binding_missing")
            binding = contract["gate_binding"]
            citation(binding)
            bound = pointer(document(binding["source_id"]), binding["pointer"])
            policy = {key: value for key, value in contract.items()
                      if key not in {"gate_binding", "validation_run_started_at"}}
            _require(bound == policy, "active_gate_content_mismatch")
            # Eligibility is exercised with a separately supplied actual run start.
            probe = {**contract, "run_started_at": contract.get("validation_run_started_at")}
            check_comparison(record, probe)
        else:
            _require(record.get("active_contract") is None, "blocked_record_has_active_contract")
            obligations = record["obligations"]
            _require(bool(obligations), "missing_blocked_obligation")
            for obligation in obligations:
                _require(obligation.get("prospective") is True, "obligation_not_prospective")
                _require(bool(obligation.get("artifact")) and bool(obligation.get("missing")),
                         "incomplete_obligation")
                _require(set(obligation["tasks"]) == set(routes[record["route_id"]]["migration_tasks"]),
                         "wrong_downstream_tasks")
    engineering = mapping["engineering_routes"]
    for cid, cap in caps.items():
        if cid in expected:
            continue
        rid = cap["route_id"]
        _require(rid in engineering, "missing_engineering_route")
        _require(engineering[rid]["tasks"] == routes[rid]["migration_tasks"],
                 "wrong_engineering_tasks")
        _require(engineering[rid]["physics_qualification"] is False,
                 "engineering_claims_physics")
    expected_consumers = {}
    for row in reconciliation.get("configured_capability_bindings", []):
        expected_consumers[row["capability_id"]] = set(row["benchmark_reference_ids"] + row["refinement_reference_ids"])
    for row in reconciliation.get("registered_test_bindings", []):
        expected_consumers[row["capability_id"]] = set(row["benchmark_reference_ids"])
    for row in reconciliation["slice_references"]:
        if row.get("executor_id"):
            expected_consumers.setdefault(row["executor_id"], set()).add(row["capability_id"])
    for cid, cap in caps.items():
        if cap.get("kind") == "reviewed_model_or_combination_boundary":
            expected_consumers[cid] = set()
    consumers = mapping.get("consumer_scientific_links", [])
    _require(len({x["capability_id"] for x in consumers}) == len(consumers), "duplicate_consumer")
    _require({x["capability_id"] for x in consumers} == set(expected_consumers), "consumer_link_coverage")
    for row in consumers:
        cid = row["capability_id"]
        _require(set(row["scientific_record_ids"]) == expected_consumers[cid], "wrong_consumer_science_links")
        _require(set(row["scientific_record_ids"]) <= expected, "dangling_consumer_science_link")
        _require(row["route_id"] == caps[cid]["route_id"], "wrong_consumer_route")
        _require(row["scientific_qualification_granted"] is False, "consumer_claims_physics")
        if caps[cid].get("kind") == "reviewed_model_or_combination_boundary":
            _require(row.get("original_support_boundary") == caps[cid]["support_boundary"],
                     "model_support_boundary_changed")
            _require(bool(row.get("prospective_artifact")), "model_obligation_missing")
    for item in mapping["test_link_audit"]:
        _require(item["capability_id"] in expected, "unknown_test_audit_capability")
        for ref in item["citations"]:
            citation(ref)
        _require(bool(item["chains"]) or bool(item.get("missing_test_artifact")),
                 "missing_test_disposition")
    scope_link = mapping.get("comparison_scopes")
    if scope_link:
        scope_bytes = (repo / scope_link["path"]).read_bytes()
        _require(hashlib.sha256(scope_bytes).hexdigest() == scope_link["sha256"], "scope_link_hash_mismatch")
        validate_comparison_scopes(json.loads(scope_bytes), mapping, repo, archive)
    return {"status": "valid_planning_map", "scientific_records": len(records),
            "engineering_capabilities": len(caps) - len(expected),
            "sources_checked": len(blobs), "active_comparisons": sum(r["state"] == "active" for r in records),
            "consumer_scientific_links": len(consumers),
            "scientific_qualification_granted": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    args = parser.parse_args()
    mapping = json.loads((args.repo / "reproducibility/RefactorComparisonMapV1.json").read_bytes())
    print(json.dumps(validate_map(mapping, args.repo, args.archive), indent=2))


if __name__ == "__main__":
    main()
