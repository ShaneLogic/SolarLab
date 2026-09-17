"""Independent real-CLI regressions for V3 study failure replay and scope."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tests.integration.test_r1_physics_study_formal import (
    ENV, digest, frozen_project, read, reseal, write,
)

pytestmark = pytest.mark.slow
KEY = "Short/N16/D"


def summary(stdout):
    found = []
    for index, character in enumerate(stdout):
        if character != "{" or index and stdout[index-1] != "\n":
            continue
        try:
            value, _ = json.JSONDecoder().raw_decode(stdout[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("schema") == "R1PhysicsStudySummaryV2":
            found.append(value)
    return found[-1] if found else None


def verified_failure(result):
    report = summary(result.stdout)
    assert result.returncode == 1, result.stdout + result.stderr
    assert report is not None, result.stdout + result.stderr
    assert report["requirements"]["verification_completed_without_error"], result.stdout + result.stderr
    assert not report["study_exit_passed"]
    row = next(row for row in report["cases"] if row["case"] == KEY)
    assert row["status"] == "failed" and row["scientific_checks_passed"] is False
    return row


@pytest.fixture(scope="module")
def short_study(tmp_path_factory):
    base = tmp_path_factory.mktemp("r1-v3-failed-study-review")
    checkout = base / "checkout"
    checkout.mkdir()
    project, revision = frozen_project(checkout)
    plan = base / "CallerPlanV1.json"
    dependencies = Path(np.__file__).resolve().parents[1]

    def launch(output, *, planning=False, verifying=False, resuming=False):
        command = [sys.executable, "-I", "-S", str(project / "scripts/run_one_dimensional_mechanism_r1_controlled.py"),
            "--project", str(project), "--source-commit", revision, "--dependency-path", str(dependencies),
            "--runner", "physics-study", "--", "--formal", "--output-dir", str(output), "--grids", "16",
            "--section", "short", "--case", KEY, "--plan-file", str(plan)]
        command += ["--plan-only"] if planning else ["--request-sha256", digest(plan)]
        if verifying or resuming:
            command += ["--verify" if verifying else "--resume", "--manifest-sha256", digest(output / "ManifestV1.json")]
        return subprocess.run(command, cwd=project, env=ENV, capture_output=True, text=True, timeout=180)

    output = base / "Baseline"
    planned = launch(output, planning=True)
    assert planned.returncode == 0 and not output.exists(), planned.stdout + planned.stderr
    produced = launch(output)
    assert produced.returncode == 0, produced.stdout + produced.stderr
    checked = launch(output, verifying=True)
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert summary(checked.stdout)["requirements"]["all_recorded_cases_checked"]
    return SimpleNamespace(output=output, launch=launch, plan=plan, request_sha256=digest(plan))


def copied(study, tmp_path):
    output = tmp_path / "copy"
    shutil.copytree(study.output, output)
    return output


def failed_prefix(output, *, persistence=False, durable_rows=2, post_append=False):
    """Retain real accepted states; alter only saved interruption provenance."""
    case = output / KEY / "AttemptV1"
    partial = deepcopy(read(case / "ResultV1.json"))
    partial.pop("sha256", None)
    partial["accepted_steps"] = partial["accepted_steps"][:2]
    for name in ("output_states", "regular_currents", "finite_step_averages", "accepted_state_arrays", "charge_integral"):
        partial.pop(name, None)
    partial["certificate"] = {"certified": False, "reasons": ["bounded_review_interruption"]}
    partial["failure"] = {"type": "InjectedStop", "message": "bounded review interruption"}
    if persistence:
        row = partial["accepted_steps"][-1]
        failure = {"type": "OSError", "message": "review write failure", "reason": "accepted_step_persistence_failed",
                   "record_index": 1, "time_s": row["time_s"], "substeps": row["substeps"]}
        partial["persistence_failure"] = deepcopy(failure)
        row["persistence_failure"] = deepcopy(failure)
        partial["failure"] = deepcopy(failure)
        partial["certificate"] = {"certified": False, "scope": partial["scope"],
            "reasons": ["accepted_step_persistence_failed"], "failed_record_index": 1}
    failure = {"type": "R1RunError", "message": "bounded review interruption", "partial_result": partial}
    write(case / "FailureV1.json", failure)
    completion = read(case / "CompletionV1.json")
    completion.update(status="failed", scientific_checks_passed=False,
                      failure={key: value for key, value in failure.items() if key != "partial_result"})
    write(case / "CompletionV1.json", completion)
    (case / "ResultV1.json").unlink()
    rows = deepcopy(partial["accepted_steps"][:durable_rows])
    if post_append:
        rows[-1].pop("persistence_failure", None)
    (case / "AcceptedStepsV1.jsonl").write_text("".join(json.dumps(row, allow_nan=False)+"\n" for row in rows))
    published = read(output / "StudySummaryV1.json")
    for row in published["cases"]:
        if row["case"] == KEY:
            row.update(status="failed", scientific_checks_passed=False)
    published.update(diagnostic_failure_count=1, diagnostic_failed_case_count=1, exit_code=1, study_exit_passed=False)
    write(output / "StudySummaryV1.json", published)
    reseal(case)
    reseal(output)
    return case


def test_real_failed_prefix_reports_its_checked_and_unobserved_scope(short_study, tmp_path):
    output = copied(short_study, tmp_path)
    failed_prefix(output)
    row = verified_failure(short_study.launch(output, verifying=True))
    assert row["independently_verified"] is True
    assert row["failure_scope"]["saved_row_count"] == row["failure_scope"]["persisted_row_count"] == 2
    assert row["failure_scope"]["complete_requested_schedule"] is False
    assert row["failure_scope"]["original_execution_extent_verified"] is False
    assert row["failure_scope"]["failure_origin_status"] == "not_reconstructed_from_saved_prefix"


@pytest.mark.parametrize("erase", ["missing", "empty", "all_partial"])
def test_erasing_reconstruction_never_hides_retained_corrupt_failed_states(short_study, tmp_path, erase):
    output = copied(short_study, tmp_path)
    case = failed_prefix(output)
    assert verified_failure(short_study.launch(output, verifying=True))["independently_verified"]
    failure = read(case / "FailureV1.json")
    failure["partial_result"]["accepted_steps"][1]["state"]["n_m3"][1] *= 2.
    (case / "AcceptedStepsV1.jsonl").write_text("".join(json.dumps(row)+"\n" for row in failure["partial_result"]["accepted_steps"]))
    write(case / "FailureV1.json", failure)
    reseal(case); reseal(output)
    detected = short_study.launch(output, verifying=True)
    assert detected.returncode == 1 and "accepted state 1" in detected.stderr
    if erase == "missing":
        failure["partial_result"].pop("physics_reconstruction")
    elif erase == "empty":
        failure["partial_result"]["physics_reconstruction"] = {}
    else:
        failure["partial_result"] = None
    write(case / "FailureV1.json", failure)
    reseal(case); reseal(output)
    rejected = short_study.launch(output, verifying=True)
    assert rejected.returncode == 1
    assert "present failed states require exact physical reconstruction" in rejected.stderr
    checked = summary(rejected.stdout)
    assert checked is None or not checked["requirements"]["verification_completed_without_error"]


@pytest.mark.parametrize("durable_rows,post_append", [(1, False), (2, True)])
def test_protocol_valid_final_write_failure_preserves_raw_and_saved_counts(short_study, tmp_path, durable_rows, post_append):
    output = copied(short_study, tmp_path)
    failed_prefix(output, persistence=True, durable_rows=durable_rows, post_append=post_append)
    row = verified_failure(short_study.launch(output, verifying=True))
    assert row["independently_verified"] is True
    scope = row["failure_scope"]
    assert scope["saved_row_count"] == 2 and scope["persisted_row_count"] == durable_rows
    assert scope["last_observed_row_persisted"] is (durable_rows == 2)
    assert scope["original_execution_extent_verified"] is False
    assert scope["post_callback_persistence_annotation_replayed"] is False


def test_missing_saved_tail_requires_specific_persistence_evidence(short_study, tmp_path):
    output = copied(short_study, tmp_path)
    failed_prefix(output, durable_rows=1)
    result = short_study.launch(output, verifying=True)
    assert result.returncode == 1
    assert "failed saved rows differ from the raw persisted prefix" in result.stderr


def test_no_saved_scientific_data_never_claims_independent_physics_replay(short_study, tmp_path):
    output = copied(short_study, tmp_path)
    case = failed_prefix(output)
    failure = read(case / "FailureV1.json")
    failure["partial_result"] = {"schema": "UnreconstructableDiagnosticV1", "message": "no saved state"}
    write(case / "FailureV1.json", failure)
    (case / "AcceptedStepsV1.jsonl").unlink()
    reseal(case); reseal(output)
    row = verified_failure(short_study.launch(output, verifying=True))
    assert row["independently_verified"] is False
    assert row["failure_scope"]["saved_row_count"] == 0
    assert row["failure_scope"]["unverified_present_fields"] == ["message", "schema"]


def test_formal_interrupted_attempt_is_preserved_without_implicit_retry(short_study, tmp_path):
    output = copied(short_study, tmp_path)
    case = output / KEY / "AttemptV1"
    for name in ("CompletionV1.json", "ManifestV1.json", "ResultV1.json"):
        (case / name).unlink()
    original_rows = (case / "AcceptedStepsV1.jsonl").read_bytes()
    reseal(output)
    result = short_study.launch(output, resuming=True)
    assert result.returncode == 1, result.stdout + result.stderr
    assert not (case.parent / "AttemptV2").exists()
    assert (case / "AcceptedStepsV1.jsonl").read_bytes() == original_rows
    assert read(output / "FailureIndexV1.json")["cases"]


def test_formal_second_attempt_cannot_hide_earlier_attempt(short_study, tmp_path):
    output = copied(short_study, tmp_path)
    shutil.copytree(output / KEY / "AttemptV1", output / KEY / "AttemptV2")
    reseal(output)
    result = short_study.launch(output, verifying=True)
    assert result.returncode == 1
    assert "repeated attempt" in result.stderr
