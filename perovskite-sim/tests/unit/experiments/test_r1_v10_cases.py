"""Contract and cost boundaries for the new, finite P3/P4 case set."""
from copy import deepcopy
import json

import pytest

from scripts import run_r1_v10_cases as runner


def request(case_id="P3MigrationD32"):
    kind, control, n, factor, steps = runner.CASES[case_id]
    return {"schema": runner.SCHEMA, "scope": runner.SCOPE, "case_id": case_id, "kind": kind,
            "case": {"control": control, "intervals": n, "nonlinear_factor": factor,
                     "time_substeps": steps, "amplitude_V": .005, "fault": "none",
                     "times_s": runner.SHORT_TIMES},
            "inputs": runner.INPUTS, "certificate_limits": runner.LIMITS,
            "solver_limits": runner.SOLVER_LIMITS, "relative_engineering_limits": runner.RELATIVE_LIMITS,
            "preparation": {"nonlinear_factor": .1, "time_substeps": [1, 2, 4]},
            "expected_rows": sum(1 + 4 * x for x in steps), "source_requires_clean_commit": True,
            "numerical_retries": 0, "baseline_repeats": 1, "pair_repeats": 1,
            "absolute_engineering_qualification": False, "scientifically_accepted": False,
            "formal_R1_2_qualification": False}


def load(tmp_path, value):
    path = tmp_path / "Request.json"
    path.write_text(json.dumps(value))
    return runner.load_request(path, runner.sha(path), tmp_path / "Result")


@pytest.mark.parametrize("case_id,rows", [("P3MigrationD32", 59), ("P3TightB256", 227),
                                        ("P3TimeB256", 115), ("P3GridB128", 227)])
def test_original_strict_cases_have_exact_finite_extent(tmp_path, case_id, rows):
    value = load(tmp_path, request(case_id))
    assert value["expected_rows"] == rows
    assert value["preparation"] == {"nonlinear_factor": .1, "time_substeps": [1, 2, 4]}
    assert value["case"]["nonlinear_factor"] == .01


@pytest.mark.parametrize("key,value", [
    ("expected_rows", 794), ("pair_repeats", 2), ("numerical_retries", 1),
    ("absolute_engineering_qualification", True), ("scientifically_accepted", True),
    ("preparation", {"nonlinear_factor": .01, "time_substeps": [2, 4, 8]}),
    ("solver_limits", {**runner.SOLVER_LIMITS, "newton": 200}),
    ("certificate_limits", {**runner.LIMITS, "nonlinear_residual": .051}),
    ("relative_engineering_limits", {**runner.RELATIVE_LIMITS, "elapsed_ratio": 5.1}),
])
def test_changed_extent_solver_acceptance_or_qualification_rejected(tmp_path, key, value):
    body = deepcopy(request())
    body[key] = value
    with pytest.raises(ValueError):
        load(tmp_path, body)


@pytest.mark.parametrize("key,value", [("control", "B"), ("intervals", 64),
    ("nonlinear_factor", .1), ("time_substeps", [1, 2, 4]), ("amplitude_V", .01),
    ("times_s", [0., 1e-9, 1e-8, 1e-6, 1e-3]), ("fault", "omit_ion")])
def test_unapproved_physical_axes_rejected(tmp_path, key, value):
    body = deepcopy(request())
    body["case"][key] = value
    with pytest.raises(ValueError):
        load(tmp_path, body)


def test_long_window_cannot_be_satisfied_by_a_short_case(tmp_path):
    with pytest.raises(ValueError, match="134 historical"):
        load(tmp_path, request("P4LongD16"))


def test_digest_and_external_location_are_checked(tmp_path):
    path = tmp_path / "Request.json"
    path.write_text(json.dumps(request()))
    with pytest.raises(ValueError, match="identity"):
        runner.load_request(path, "0" * 64, tmp_path / "Result")
    with pytest.raises(ValueError, match="outside"):
        runner.load_request(path, runner.sha(path), tmp_path)


def summary(mode="baseline"):
    return {"schema": "R1V10CaseRunV1", "case_id": "P3TightB256", "mode": mode,
            "source_commit": "a" * 40, "source_content_sha256": "b" * 64,
            "request_sha256": "c" * 64, "precision_budget_sha256": "d" * 64,
            "runtime_identity": {"numpy": "2.1.3", "scipy": "1.15.3"},
            "baseline_usable": mode == "baseline", "extent": {"complete": True},
            "cost": {"elapsed_s": 10., "peak_rss_bytes": 1000, "bytes_per_row": 200.}}


def test_failed_prefix_never_supplies_a_full_run_denominator():
    baseline, pair = summary(), summary("compensated")
    baseline["extent"]["complete"] = False
    result = runner.relative_cost(pair, baseline)
    assert result["applicable"] is False and result["qualified"] is None and not result["ratios"]


def test_original_ratio_limits_are_inclusive_and_not_the_4_8_target():
    baseline, pair = summary(), summary("compensated")
    pair["cost"] = {"elapsed_s": 50., "peak_rss_bytes": 2000, "bytes_per_row": 600.}
    result = runner.relative_cost(pair, baseline)
    assert result["qualified"] and result["ratios"] == runner.RELATIVE_LIMITS
    pair["cost"]["elapsed_s"] += 1e-10
    result = runner.relative_cost(pair, baseline)
    assert not result["qualified"] and result["failed_metrics"] == ["elapsed_ratio"]


@pytest.mark.parametrize("key", ["case_id", "source_commit", "source_content_sha256", "request_sha256",
                                  "precision_budget_sha256", "runtime_identity"])
def test_reference_identity_cannot_cross_cases_or_sources(key):
    baseline, pair = summary(), summary("compensated")
    pair[key] = "changed"
    with pytest.raises(ValueError, match="same-source"):
        runner.relative_cost(pair, baseline)


def test_counted_artifacts_include_nested_manifests_and_their_own_metadata(tmp_path):
    child = tmp_path / "Analysis"
    child.mkdir()
    (child / "ManifestV1.json").write_text("{}\n")
    (child / "Rows.jsonl").write_text('{"row":0}\n')
    value = {"cost": {}}
    runner.finalize_manifest(tmp_path, value)
    manifest = json.loads((tmp_path / "ManifestV1.json").read_text())
    assert "Analysis/ManifestV1.json" in manifest
    assert value["cost"]["case_artifact_bytes"] == sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file())
    assert "absolute_engineering_check" not in value


@pytest.mark.parametrize("key", ["intervals", "control_label", "times_s", "policy", "prepared_sha256", "execution_axes"])
def test_actual_producer_axes_are_not_replaced_by_request_metadata(key):
    case = request()["case"]
    record = {"schema": "R1ControlledStepV2", "intervals": 32, "control_label": "D",
              "amplitude_V": .005, "times_s": runner.SHORT_TIMES,
              "prepared_sha256": "a" * 64, "policy": {"newton": 100},
              "execution_axes": {"intervals": 32, "time_substeps": [2, 4, 8]}}
    runner.validate_result_request(record, case, "a" * 64, {"newton": 100}, pair=True)
    record[key] = "changed"
    with pytest.raises(ValueError, match="actual producer"):
        runner.validate_result_request(record, case, "a" * 64, {"newton": 100}, pair=True)
