"""The current physical acceptance cannot be downgraded to historical inspection."""

from pathlib import Path
import pytest

from perovskite_sim.experiments import one_dimensional_mechanism_r1_evidence as evidence
from tests.unit.experiments.test_one_dimensional_mechanism_r1_controlled import repository
from tests.unit.experiments.test_one_dimensional_mechanism_r1_controlled import commit
from tests.unit.experiments.test_one_dimensional_mechanism_r1_stage_one_cli import runner
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


@pytest.mark.parametrize("producer_revision,revision", [(5, 4), (6, 4), (6, 5)])
def test_resealed_legacy_revision_cannot_downgrade_anchored_producer(
    repository, tmp_path, producer_revision, revision, runner, monkeypatch, capsys,
):
    root, project, _ = repository
    producer = project / "scripts/run_one_dimensional_mechanism_r1_stage_one.py"
    # The producer's format declaration belongs to its committed source,
    # independent of the adversarial CompletionV1 edit below.
    with producer.open("a") as stream:
        stream.write("\nPRODUCER_FORMAT = {\"schema\": \"R1StageOneCompletionV1\", \"evidence_revision\": " + str(producer_revision) + "}\n")
    source_commit = commit(root)
    output = make_bundle((root, project, source_commit), tmp_path / "downgraded")
    completion = evidence.read_json(output / "CompletionV1.json")
    completion["evidence_revision"] = revision
    write(output / "CompletionV1.json", completion)
    seal(output)
    with pytest.raises(ValueError, match="downgrades its anchored producer format"):
        evidence.inspect_legacy_evidence(output,
            required_evidence_revision=revision, expected_manifest_sha256=evidence.sha256(output / "ManifestV1.json"),
            expected_source_commit=source_commit, source_repository=root)
    monkeypatch.setattr(runner, "PROJECT", project)
    assert runner.main(["verify", "--mode", "legacy-inspect", "--output-dir", str(output),
                        "--required-evidence-revision", str(revision),
                        "--expected-manifest-sha256", evidence.sha256(output / "ManifestV1.json"),
                        "--expected-source-commit", source_commit]) == 1
    assert "downgrades its anchored producer format" in capsys.readouterr().err


def test_cli_success_uses_scientific_acceptance_not_saved_status(runner, monkeypatch, tmp_path):
    report = {"status": "passed", "stage": "prepare", "verification": {"scientifically_accepted": False}}
    monkeypatch.setattr(runner, "verify_acceptance", lambda *a, **k: (report, 1))
    assert runner.main(["verify", "--mode", "acceptance", "--output-dir", str(tmp_path)]) == 1
    monkeypatch.setattr(runner, "inspect_legacy_evidence", lambda *a, **k: (report, 1))
    assert runner.main(["verify", "--mode", "legacy-inspect", "--required-evidence-revision", "4",
                        "--output-dir", str(tmp_path)]) == 2
    report["verification"]["scientifically_accepted"] = True
    assert runner.main(["verify", "--mode", "acceptance", "--output-dir", str(tmp_path)]) == 0


def test_external_commit_alone_cannot_select_a_different_physics_implementation(repository, tmp_path):
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_checkout import R1_DECLARATIONS
    source_repository = repository
    # The tiny source fixture has genuine pinned data and a genuine external
    # commit, but deliberately only a stub physics package.
    output = make_bundle(source_repository, tmp_path / "six")
    for name, source in [(entry[1], relative) for relative, entry in R1_DECLARATIONS.items()] + [
        ("AdditionalFailuresV1.json", "reproducibility/OneDimensionalMechanismR1AdditionalFailuresV1.json"),
        ("AdditionalFailuresV2.json", "reproducibility/OneDimensionalMechanismR1AdditionalFailuresV2.json")]:
        (output / name).write_bytes((PROJECT / source).read_bytes())
    completion = evidence.read_json(output / "CompletionV1.json")
    completion["evidence_revision"] = 6
    write(output / "CompletionV1.json", completion)
    seal(output)
    with pytest.raises(ValueError, match="trusted installed package"):
        evidence.verify_acceptance(output, expected_manifest_sha256=evidence.sha256(output / "ManifestV1.json"),
            expected_source_commit=source_repository[2], source_repository=source_repository[0])
