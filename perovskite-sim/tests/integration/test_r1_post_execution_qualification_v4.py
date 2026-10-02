"""Real launcher serialization and post-execution review, without device claims.

The real device calculation has no independently reviewed response budgets.
Completing the analysis must therefore report unknown qualification, while
leaving its original timestamped preparation and entire collection unchanged.
Analytic positive scientific cases are covered separately by library tests.
"""
from pathlib import Path
import shutil

import pytest

from tests.integration.test_r1_physics_study_formal import (
    formal_study, digest, read, write, reseal, verify,
)

pytestmark = pytest.mark.slow


def file_identities(directory):
    return {p.relative_to(directory).as_posix(): digest(p) for p in directory.rglob("*") if p.is_file()}


def test_real_cli_can_review_completed_collection_without_repreparing_or_mutating_it(formal_study, tmp_path):
    study = formal_study
    before = file_identities(study.output)
    prepared = read(study.output/"Preparation/N16/AttemptV1/ResultV1.json")
    assert prepared["created_utc"] and len(prepared["sha256"]) == 64
    approval = tmp_path/"ReviewedInputsV1.json"
    write(approval, {"schema": "R1StudyQualificationInputsV1", "current_budgets": {},
        "turnover_evidence": {}, "double_domain_evidence": {}, "trusted_evidence": {}})
    plan, output = tmp_path/"AnalysisRequestV1.json", tmp_path/"AnalysisV1"
    standard = read(study.output/"StudyRequestV1.json")["standard_binding"]["candidate_standard_sha256"]
    common = ["--section", "prepare", "--section", "dc-ac", "--qualify",
        "--manifest-sha256", study.anchor, "--qualification-file", str(approval),
        "--qualification-sha256", digest(approval), "--analysis-dir", str(output),
        "--analysis-plan-file", str(plan), "--analysis-approved-standard-sha256", standard,
        "--analysis-section", "reconstruct", "--analysis-case", "Frequency/N16"]
    planned = study.launch(study.output, *common, "--analysis-plan-only")
    assert planned.returncode == 0, planned.stdout + planned.stderr
    assert not output.exists()
    result = study.launch(study.output, *common, "--analysis-request-sha256", digest(plan))
    assert result.returncode == 2, result.stdout + result.stderr
    report = read(output/"AnalysisResultV1.json")
    assert report["analysis_completed"] is True
    assert report["new_physical_solves_in_derivation"] == 0
    assert report["requested_device_qualifications_passed"] is False
    assert report["study_exit_passed"] is False
    assert report["request"]["collection_manifest_sha256"] == study.anchor
    assert report["request"]["qualification_inputs_sha256"] == digest(approval)
    assert report["verification_receipt"]["scientific_acceptance_inferred"] is False
    assert report["results"]["Frequency/N16"]["device_frequency_window_certified"] is False
    assert file_identities(study.output) == before
    replay = study.launch(study.output, *common, "--analysis-request-sha256", digest(plan),
        "--verify-analysis", "--analysis-manifest-sha256", digest(output/"ManifestV1.json"))
    assert replay.returncode == 2, replay.stdout + replay.stderr
    assert file_identities(study.output) == before
    receipt = read(output/"VerificationReceiptV1.json")
    receipt["diagnostic_failure_count"] += 1
    write(output/"VerificationReceiptV1.json", receipt)
    reseal(output)
    altered_receipt = study.launch(study.output, *common, "--analysis-request-sha256", digest(plan),
        "--verify-analysis", "--analysis-manifest-sha256", digest(output/"ManifestV1.json"))
    assert altered_receipt.returncode == 1 and "verification receipt differs" in altered_receipt.stderr
    absent = tmp_path/"AbsentCollection"
    rejected = study.launch(absent, *common, "--analysis-plan-only")
    assert rejected.returncode == 1 and not absent.exists()
    changed = read(approval)
    changed["trusted_evidence"]["unreviewed"] = "f"*64
    write(approval, changed)
    refused = study.launch(study.output, *common, "--analysis-plan-only")
    assert refused.returncode == 1 and "digest mismatch" in refused.stderr
    assert file_identities(study.output) == before


@pytest.mark.parametrize("relative", ["StudySummaryV1.json", "Invocations/InvocationV1.json"])
def test_request_digest_claims_are_verified_even_after_a_coordinated_reseal(formal_study, tmp_path, relative):
    output = tmp_path/"ChangedClaim"
    shutil.copytree(formal_study.output, output)
    value = read(output/relative)
    value["study_request_sha256"] = "f"*64
    write(output/relative, value)
    reseal(output)
    result = verify(formal_study, output)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "study_request_sha256" in result.stderr


def test_coordinated_root_source_claims_cannot_replace_the_selected_commit(formal_study, tmp_path):
    output = tmp_path/"ChangedSourceClaim"
    shutil.copytree(formal_study.output, output)
    for name in ("ExecutionSourceV1.json", "SourceReadsV1.json"):
        value = read(output/name)
        value["source_commit"] = "f"*40
        write(output/name, value)
    reseal(output)
    result = verify(formal_study, output)
    assert result.returncode == 1 and "caller-selected frozen source" in result.stderr


def test_verifying_an_absent_collection_never_creates_it(formal_study, tmp_path):
    absent = tmp_path/"Absent"
    result = verify(formal_study, absent, anchor="f"*64)
    assert result.returncode == 1 and not absent.exists()
