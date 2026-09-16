"""The current physical acceptance cannot be downgraded to historical inspection."""

from pathlib import Path
import pytest

from perovskite_sim.experiments import one_dimensional_mechanism_r1_evidence as evidence
from tests.unit.experiments.test_one_dimensional_mechanism_r1_controlled import repository
from tests.unit.experiments.test_one_dimensional_mechanism_r1_evidence_revision_four import (
    PROJECT, make_bundle, write, seal,
)


@pytest.mark.parametrize("revision", [1, 2, 3, 4, 5, True, 6.0])
def test_current_acceptance_rejects_all_historical_revision_selections(tmp_path, revision):
    with pytest.raises(ValueError, match="requires evidence revision 6"):
        evidence.verify_acceptance(tmp_path, required_evidence_revision=revision)


@pytest.mark.parametrize("revision", [6, 7, True])
def test_historical_inspection_cannot_accept_current_revision(tmp_path, revision):
    with pytest.raises(ValueError, match="historical inspection supports only"):
        evidence.inspect_legacy_evidence(tmp_path, required_evidence_revision=revision)


def test_early_rejection_remains_integrity_readable_but_not_accepted(tmp_path):
    failure = {"type": "InputRejected", "message": "execution did not begin"}
    write(tmp_path / "CompletionV1.json", {
        "schema": "R1StageOneCompletionV1", "stage_scope": "R1-rejected", "stage": "step",
        "status": "failed", "failure": failure, "evidence_revision": 6,
        "run_class": "rejected_before_execution", "physical_execution_started": False,
        "accepted_record_count": 0, "observed_record_count": 0, "persisted_record_count": 0,
        "persisted_finite_step_count": 0, "physical_passed_finite_step_count": 0,
    })
    write(tmp_path / "FailureV1.json", failure)
    seal(tmp_path)
    completion, count = evidence.verify_output(tmp_path)
    assert completion["status"] == "failed" and count == 2
    with pytest.raises(ValueError, match="physical execution did not start"):
        evidence.verify_acceptance(tmp_path, expected_manifest_sha256=evidence.sha256(tmp_path / "ManifestV1.json"))


def test_missing_policy_artifact_is_a_structured_unavailable_error(tmp_path):
    with pytest.raises(ValueError, match="artifact unavailable"):
        evidence.read_json(tmp_path / "ProtocolV1.json")
    with pytest.raises(ValueError, match="artifact unavailable"):
        evidence.sha256(tmp_path / "PhysicsProtocolV1.md")


def test_external_commit_alone_cannot_select_a_different_physics_implementation(repository, tmp_path):
    source_repository = repository
    # The tiny source fixture has genuine pinned data and a genuine external
    # commit, but deliberately only a stub physics package.
    output = make_bundle(source_repository, tmp_path / "six")
    for name, source in (
        ("OperatorCriterionDecisionV2.md", "docs/OneDimensionalMechanismR1OperatorCriterionDecisionV2.md"),
        ("AdditionalFailuresV1.json", "reproducibility/OneDimensionalMechanismR1AdditionalFailuresV1.json"),
        ("PhysicsProtocolV1.md", "docs/OneDimensionalMechanismR1PhysicsProtocolV1.md"),
    ):
        (output / name).write_bytes((PROJECT / source).read_bytes())
    completion = evidence.read_json(output / "CompletionV1.json")
    completion["evidence_revision"] = 6
    write(output / "CompletionV1.json", completion)
    seal(output)
    with pytest.raises(ValueError, match="trusted installed package"):
        evidence.verify_acceptance(output, expected_manifest_sha256=evidence.sha256(output / "ManifestV1.json"),
            expected_source_commit=source_repository[2], source_repository=source_repository[0])
