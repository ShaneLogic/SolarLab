"""Fixed-request accounting and unavailable-termination counterexamples."""
import hashlib
import json
import shutil

import pytest

from perovskite_sim.experiments.one_dimensional_mechanism_r1_study_request import (
    build_execution_plan, canonical_bytes, load_execution_plan,
)
from perovskite_sim.experiments.one_dimensional_mechanism_r1_failure_witness import build_failure_witness
from perovskite_sim.experiments.one_dimensional_mechanism_r1_failure_reconstruction import rebuild_failure_witness
from perovskite_sim.experiments.one_dimensional_mechanism_r1_result_validation import failure_scope_report
from tests.unit.experiments.test_one_dimensional_mechanism_r1_physics_study import runner, study
from tests.unit.experiments.test_r1_v4_failure_witness import inputs


def test_result_directory_cannot_supply_its_own_external_plan(tmp_path):
    result = tmp_path / "result"
    result.mkdir()
    plan = build_execution_plan({}, ["short"], {}, {"Case": {"scope": "example"}})
    path = result / "StudyPlanV1.json"
    path.write_bytes(canonical_bytes(plan))
    anchor = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="outside the result"):
        load_execution_plan(path, anchor, result_directory=result)
    outside = tmp_path / "CallerPlanV1.json"
    outside.write_bytes(path.read_bytes())
    assert load_execution_plan(outside, anchor, result_directory=result) == plan


def _formal_records(runner, study, monkeypatch, *, fail=True):
    cases = {key: {"scope": "counterexample"} for key in ("Good", "Bad", "Unstarted")}
    study.plan = build_execution_plan({}, ["short"], {}, cases)
    study.plan_sha256, study.planned_cases = "a" * 64, cases
    study.run_class = "formal"
    runner.write_json(study.output / "CaseInventoryV2.json", cases)
    study.case("Good", cases["Good"], lambda directory: {"certificate": {"certified": True}})
    def bad(directory):
        if fail:
            raise ValueError("recorded failure")
        return {"certificate": {"certified": True}}
    study.case("Bad", cases["Bad"], bad)
    assert study.finish() == (1 if fail else 2)
    monkeypatch.setattr(runner, "verify_study_source", lambda *a, **kw: None)
    study.verifying = True
    study.verified_cases = {key: (runner.sha(study.latest(key) / "ManifestV1.json"), {})
                            for key in ("Good", "Bad")}
    return cases


def test_fixed_request_failed_case_cannot_be_relabelled_unstarted(runner, study, monkeypatch):
    _formal_records(runner, study, monkeypatch)
    # Content-level counterexample intentionally accepts a new manifest; the
    # separate integration control holds the original manifest fixed.
    shutil.rmtree(study.output / "Bad")
    for path in [study.output / "StudySummaryV1.json", *sorted((study.output / "Invocations").glob("*.json"))]:
        value = json.loads(path.read_text())
        value["cases"] = [row for row in value["cases"] if row["case"] != "Bad"]
        value.update(active_case_count=1, diagnostic_failure_count=0, diagnostic_failed_case_count=0)
        runner.write_json(path, value)
    runner.write_json(study.output / "FailureIndexV1.json", {"schema": "R1PhysicsStudyFailureIndexV1", "cases": []})
    runner.seal(study.output)
    with pytest.raises(ValueError, match="attempt|missing"):
        study.finish()


def test_invocation_attempt_count_is_local_to_each_resume():
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_study_request import verify_invocation_attempts
    planned = {key: {"scope": "example"} for key in ("Earlier", "New", "NotStarted")}
    recorded = {key + "/AttemptV1": planned[key] for key in ("Earlier", "New")}
    invocation = {"attempted_cases": 1, "attempt_count_scope": "new_attempts_in_this_invocation",
                  "invocation_attempts": [{"case": "New", "directory": "New/AttemptV1"}],
                  "active_case_count": 2, "missing_cases": ["NotStarted"]}
    assert verify_invocation_attempts(invocation, planned_cases=planned,
                                     recorded_attempts=recorded) == {"New/AttemptV1"}
    resumed = {**invocation, "attempted_cases": 0, "invocation_attempts": []}
    assert verify_invocation_attempts(resumed, planned_cases=planned, recorded_attempts=recorded) == set()


def test_retained_interrupted_attempt_is_counted_without_claiming_completion(runner, study):
    cases = {"Interrupted": {"scope": "example"}, "NotStarted": {"scope": "example"}}
    study.plan = build_execution_plan({}, ["short"], {}, cases)
    study.plan_sha256, study.planned_cases, study.run_class = "a" * 64, cases, "formal"
    runner.write_json(study.output / "CaseInventoryV2.json", cases)
    def interrupt(directory):
        raise KeyboardInterrupt("bounded interruption fixture")
    with pytest.raises(KeyboardInterrupt):
        study.case("Interrupted", cases["Interrupted"], interrupt)
    assert study.finish() == 1
    value = json.loads((study.output / "StudySummaryV1.json").read_text())
    assert value["attempted_cases"] == 1 and value["active_case_count"] == 0
    assert value["interrupted_attempts"][0]["case"] == "Interrupted"
    study.verify_recorded_invocations()


@pytest.mark.parametrize("field", ["missing_cases", "diagnostic_failure_count", "declared_cases_available"])
def test_published_current_extent_must_match_recomputed_extent(runner, study, monkeypatch, field):
    _formal_records(runner, study, monkeypatch)
    path = study.output / "StudySummaryV1.json"
    value = json.loads(path.read_text())
    if field == "declared_cases_available":
        value["requirements"][field] = True
    else:
        value[field] = [] if field == "missing_cases" else 0
    runner.write_json(path, value)
    runner.seal(study.output)
    with pytest.raises(ValueError, match="published.*(extent|count|requirement)"):
        study.finish()


@pytest.mark.parametrize("state_field", ["coordinate", "attempted_state", "diagnostics", "independent_physics"])
def test_unavailable_newton_witness_cannot_retain_terminal_state(state_field):
    arguments = inputs()
    arguments["result"]["failure"]["numerical_evidence"] = {
        "schema": "R1NewtonFailureWitnessV1", "terminal_state_available": False,
        "witness_collection_error": {"type": "RuntimeError", "message": "unavailable"},
        state_field: {} if state_field != "coordinate" else [1.],
    }
    with pytest.raises(ValueError, match="unavailable.*(state|field)"):
        build_failure_witness(**arguments)
    with pytest.raises(ValueError, match="unavailable.*(state|field)"):
        rebuild_failure_witness(None, 16, None, None, arguments["result"])


def test_true_missing_newton_state_stays_unknown_and_keeps_its_reason():
    arguments = inputs()
    numerical = {"schema": "R1NewtonFailureWitnessV1", "terminal_state_available": False,
                 "witness_collection_error": {"type": "OSError", "message": "collector unavailable"}}
    arguments["result"]["failure"]["numerical_evidence"] = numerical
    witness = build_failure_witness(**arguments)
    assert witness["terminal_evidence"]["missing_reason"] == numerical["witness_collection_error"]
    report = rebuild_failure_witness(None, 16, None, None, arguments["result"])
    assert report["available"] is False and report["terminal_state_physics_recomputed"] is None


def test_failed_prefix_is_not_a_recomputed_newton_terminal_state():
    arguments = inputs()
    row = arguments["result"]["accepted_steps"][-1]
    row["physical_checks_passed"] = False
    report = failure_scope_report(arguments["result"], {
        "checked_row_count": 1, "physical_limit_violations": [{"row": 0}], "physical_limits_satisfied": False})
    assert report["saved_physical_failure_demonstrated"] is True
    assert report["terminal_state_physics_verified"] is None
    assert report["saved_last_row_physics_verified"] is True
    assert report["whole_failed_run_physical_limits_satisfied"] is False
