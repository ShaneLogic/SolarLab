"""Trust-boundary regressions: exact source anchors and classified declarations."""
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from perovskite_sim.experiments import one_dimensional_mechanism_r1_checkout as checkout
from perovskite_sim.experiments import one_dimensional_mechanism_r1_evidence as evidence
from perovskite_sim.experiments.one_dimensional_mechanism_r1_binding import standard_binding_record
from tests.unit.experiments.test_one_dimensional_mechanism_r1_controlled import (
    repository, launch, commit, PROJECT,
)
from tests.unit.experiments.test_one_dimensional_mechanism_r1_evidence_revision_four import (
    make_bundle, write, seal, canonical_seal,
)


@pytest.mark.parametrize("name,reader_reference", [
    ("docs/r1/OneDimensionalMechanismR1WaiverV1.md", False),
    ("docs/SpatialCriterionAddendumV1.md", True),
])
def test_declaration_namespace_and_actual_reader_references_require_classification(repository, name, reader_reference):
    root, project, _ = repository
    assert launch(repository).returncode == 0
    path = project / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("Unapproved scientific waiver.\n")
    if reader_reference:
        with (project / "scripts/run_one_dimensional_mechanism_r1_stage_one.py").open("a") as stream:
            stream.write("\nDECLARED_CRITERION = " + repr(name) + "\n")
    changed = root, project, commit(root)
    result = launch(changed)
    assert result.returncode == 1
    assert "declaration registry coverage mismatch" in result.stdout + result.stderr
    assert name in result.stdout + result.stderr


def test_unrelated_document_is_not_silently_promoted_to_a_research_contract(repository):
    root, project, _ = repository
    path = project / "docs/unrelated-user-interface.md"
    path.write_text("An unrelated UI document.\n")
    assert launch((root, project, commit(root))).returncode == 0


def test_increment_contract_is_both_registered_and_byte_pinned(repository):
    root, project, _ = repository
    relative = "docs/InterfaceDefectTransientIncrementContract.md"
    assert relative in checkout.R1_DECLARATIONS
    assert relative in checkout.REQUIRED_SOURCE_ANCHORS
    with (project / relative).open("a") as stream:
        stream.write("\nThe previous physical limits are waived.\n")
    result = launch((root, project, commit(root)))
    assert result.returncode == 1
    assert "pinned digest" in result.stdout + result.stderr


def test_standard_identity_does_not_self_approve_and_external_mismatch_is_rejected():
    context = SimpleNamespace(source_commit="candidate", read_bytes=lambda name: (PROJECT / name).read_bytes())
    candidate = standard_binding_record(context)
    assert candidate["candidate_internal_consistency"]
    assert not candidate["matches_external_approval"]
    assert not candidate["scientific_qualification_asserted"]
    approved = standard_binding_record(context, approved_standard_sha256=candidate["candidate_standard_sha256"])
    assert approved["matches_external_approval"]
    assert not approved["scientific_qualification_asserted"]
    with pytest.raises(ValueError, match="caller-approved standard"):
        standard_binding_record(context, approved_standard_sha256="0" * 64)


@pytest.mark.parametrize("revision", [1, 2, 3])
def test_modern_source_cannot_select_legacy_checks_with_a_relabel(repository, tmp_path, revision):
    root, project, _ = repository
    with (project / "scripts/run_one_dimensional_mechanism_r1_stage_one.py").open("a") as stream:
        stream.write('\nPRODUCER = {"schema": "R1StageOneCompletionV1", "evidence_revision": 6}\n')
    source = root, project, commit(root)
    output = make_bundle(source, tmp_path / "modern")
    completion = evidence.read_json(output / "CompletionV1.json")
    completion["evidence_revision"] = revision
    write(output / "CompletionV1.json", completion)
    seal(output)
    anchor = evidence.sha256(output / "ManifestV1.json")
    with pytest.raises(ValueError, match="downgrades its anchored producer format"):
        evidence.inspect_legacy_evidence(output, required_evidence_revision=revision,
            expected_source_commit=source[2], source_repository=root, expected_manifest_sha256=anchor)
    inspected, _ = evidence.inspect_legacy_evidence(output,
        required_evidence_revision=revision, expected_manifest_sha256=anchor)
    report = inspected["verification"]
    assert not report["source_identity_verified"]
    assert not report["producer_format_verified"]
    assert not report["scientifically_accepted"]
    assert report["result_checks"] is None


def test_attachment_byte_checks_do_not_attest_the_attachment_claims(tmp_path):
    extra = tmp_path / "ConvergenceCertificateV1.md"
    extra.write_text("All three axes converged; mechanism identified.\n")
    seal(tmp_path)
    scope = evidence.evidence_file_scope(tmp_path)
    assert scope["byte_consistency_file_count"] == 1
    assert scope["unclassified_attachments"] == [extra.name]
    assert not scope["unclassified_attachment_contents_verified"]


def test_failed_prepare_existing_state_still_binds_its_source(repository, tmp_path):
    output = make_bundle(repository, tmp_path / "failed-prepare")
    completion = evidence.read_json(output / "CompletionV1.json")
    failure = {"type": "PostPreparationFailure", "message": "after state write"}
    completion.update(status="failed", failure=failure)
    write(output / "CompletionV1.json", completion)
    write(output / "FailureV1.json", failure)
    seal(output)
    arguments = dict(required_evidence_revision=4, expected_source_commit=repository[2],
                     source_repository=repository[0])
    valid, _ = evidence.inspect_legacy_evidence(output,
        expected_manifest_sha256=evidence.sha256(output / "ManifestV1.json"), **arguments)
    assert valid["status"] == "failed" and not valid["verification"]["scientifically_accepted"]
    prepared = evidence.read_json(output / "PreparedStateV1.json")
    prepared["source"]["source_commit"] = "0" * 40
    prepared["source"] = canonical_seal(prepared["source"])
    write(output / "PreparedStateV1.json", canonical_seal(prepared))
    seal(output)
    with pytest.raises(ValueError, match="prepared execution source differs"):
        evidence.inspect_legacy_evidence(output,
            expected_manifest_sha256=evidence.sha256(output / "ManifestV1.json"), **arguments)
