"""Adversarial acceptance with real prepared physics and independent source roots."""

import copy
from pathlib import Path

import pytest

from perovskite_sim.experiments import one_dimensional_mechanism_r1_evidence as evidence
from perovskite_sim.experiments import one_dimensional_mechanism_r1_state as common
from perovskite_sim.experiments.one_dimensional_mechanism_r1_protocol import r1_policy
from perovskite_sim.models.config_loader import load_device_from_yaml
from tests.fixtures.r1_reference import approved_r1_binding
from tests.unit.experiments.test_one_dimensional_mechanism_r1_controlled import repository
from tests.unit.experiments.test_one_dimensional_mechanism_r1_evidence_revision_four import (
    source_repository, make_bundle, canonical_seal, seal, write, accept, FIXTURE, PROJECT,
)


CRITERION = "docs/OneDimensionalMechanismR1OperatorCriterionDecisionV2.md"
ADDITIONAL = "reproducibility/OneDimensionalMechanismR1AdditionalFailuresV1.json"


@pytest.fixture(scope="module")
def numerical_preparation():
    return common.prepare_common_state(load_device_from_yaml(PROJECT / FIXTURE), 16,
                                       approved_r1_binding(), policy=r1_policy()).to_dict()


@pytest.fixture
def bundle(source_repository, numerical_preparation, tmp_path):
    output = make_bundle(source_repository, tmp_path / "five")
    state = copy.deepcopy(numerical_preparation)
    state["source"] = evidence.read_json(output / "PreparedStateV1.json")["source"]
    write(output / "PreparedStateV1.json", canonical_seal(state))
    (output / "OperatorCriterionDecisionV2.md").write_bytes((PROJECT / CRITERION).read_bytes())
    (output / "AdditionalFailuresV1.json").write_bytes((PROJECT / ADDITIONAL).read_bytes())
    protocol = evidence.read_json(output / "ProtocolV1.json")
    protocol["criterion_sha256"] = evidence.sha256(output / "OperatorCriterionDecisionV2.md")
    protocol["additional_failures_sha256"] = evidence.sha256(output / "AdditionalFailuresV1.json")
    write(output / "ProtocolV1.json", protocol)
    completion = evidence.read_json(output / "CompletionV1.json")
    completion["evidence_revision"] = 5
    write(output / "CompletionV1.json", completion)
    seal(output)
    return output


def check(output, source_repository):
    return accept(output, source_repository, required_evidence_revision=5)


def fail_record(output):
    completion = evidence.read_json(output / "CompletionV1.json")
    failure = {"type": "NumericalFailure", "message": "deliberate failed control"}
    completion.update(status="failed", failure=failure)
    write(output / "CompletionV1.json", completion)
    write(output / "FailureV1.json", failure)
    seal(output)


def test_revision_five_recertifies_saved_preparation_without_new_dc_solve(bundle, source_repository, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("acceptance must not run another DC optimizer")
    monkeypatch.setattr(common, "solve_r1_dc", forbidden)
    completion, _ = check(bundle, source_repository)
    assert completion["verification"]["legacy"] is True
    assert "time integration not rerun" in " ".join(completion["verification"]["limits"])


@pytest.mark.parametrize("part", ["preparation_checks", "dc_state", "preparation_policy"])
def test_resealed_and_reanchored_certification_forgery_is_recomputed(bundle, source_repository, part):
    state = evidence.read_json(bundle / "PreparedStateV1.json")
    if part == "preparation_checks":
        state[part]["forged_metric"] = 0.0
    elif part == "dc_state":
        state[part]["certificate"]["maximum_normalized_residual"] = 0.5
    else:
        state[part]["maximum_dc_normalized_residual"] = 0.5
    write(bundle / "PreparedStateV1.json", canonical_seal(state))
    seal(bundle)
    with pytest.raises(ValueError, match="recomputed physics|declared R1 policy"):
        check(bundle, source_repository)


@pytest.mark.parametrize("revision", [1, 3, 4])
def test_failed_records_cannot_choose_a_weaker_revision(bundle, source_repository, revision):
    fail_record(bundle)
    completion = evidence.read_json(bundle / "CompletionV1.json")
    completion["evidence_revision"] = revision
    write(bundle / "CompletionV1.json", completion)
    seal(bundle)
    with pytest.raises(ValueError, match="externally required revision"):
        check(bundle, source_repository)


@pytest.mark.parametrize("field,value", [
    ("run_class", "development"), ("source_commit", "0" * 40),
    ("runtime", {"isolated": False}),
])
def test_failed_records_do_not_bypass_source_or_runtime_checks(bundle, source_repository, field, value):
    fail_record(bundle)
    execution = evidence.read_json(bundle / "ExecutionSourceV1.json")
    execution[field] = value
    write(bundle / "ExecutionSourceV1.json", execution)
    seal(bundle)
    with pytest.raises(ValueError, match="controlled|source|runtime"):
        check(bundle, source_repository)


def test_honest_failed_record_keeps_failed_status_after_source_verification(bundle, source_repository):
    fail_record(bundle)
    completion, _ = check(bundle, source_repository)
    assert completion["status"] == "failed"
    assert completion["verification"]["source_commit_anchor"] == source_repository[2]


@pytest.mark.parametrize("failed", [False, True])
def test_operator_decision_cannot_change_on_passed_or_failed_records(bundle, source_repository, failed):
    if failed:
        fail_record(bundle)
    (bundle / "OperatorCriterionDecisionV2.md").write_text("Automatic pass permitted\n")
    protocol = evidence.read_json(bundle / "ProtocolV1.json")
    protocol["criterion_sha256"] = evidence.sha256(bundle / "OperatorCriterionDecisionV2.md")
    write(bundle / "ProtocolV1.json", protocol)
    seal(bundle)
    with pytest.raises(ValueError, match="operator criterion"):
        check(bundle, source_repository)


def test_legacy_four_has_explicit_limits_and_cannot_use_default_acceptance(source_repository, tmp_path):
    output = make_bundle(source_repository, tmp_path / "four")
    assert accept(output, source_repository)[0]["verification"]["legacy"] is True
    with pytest.raises(ValueError, match="externally required revision"):
        evidence.verify_acceptance(output, expected_manifest_sha256=evidence.sha256(output / "ManifestV1.json"),
            expected_source_commit=source_repository[2], source_repository=source_repository[0])


@pytest.mark.parametrize("relative,pin,artifact,protocol_field,error", [
    (CRITERION, "PINNED_OPERATOR_CRITERION_SHA256", "OperatorCriterionDecisionV2.md", "criterion_sha256", "pinned decision"),
    (ADDITIONAL, "PINNED_ADDITIONAL_FAILURES_SHA256", "AdditionalFailuresV1.json", "additional_failures_sha256", "pinned registry"),
    ("reproducibility/OneDimensionalMechanismR1DynamicsInputV1.json", "PINNED_STUDY_INPUT_SHA256", "StudyInputV1.json", "input_sha256", "pinned protocol"),
    ("docs/OneDimensionalMechanismR1DynamicsV1.md", "PINNED_EXECUTION_CONTRACT_SHA256", "ExecutionContractV1.md", "contract_sha256", "pinned contract"),
])
def test_candidate_repin_cannot_override_trusted_verifier(
    source_repository, tmp_path, relative, pin, artifact, protocol_field, error,
):
    import hashlib
    import json
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_binding as policy
    from tests.unit.experiments.test_one_dimensional_mechanism_r1_controlled import commit, BINDING

    root, project, _ = source_repository
    path = project / relative
    if path.suffix == ".json":
        changed = json.loads(path.read_text())
        for key in ("known_nonconvergence", "known_physical_gate_failures"):
            if key in changed:
                changed[key] = []
        path.write_text(json.dumps(changed, indent=2) + "\n")
    else:
        path.write_text(path.read_text() + "\nUnreviewed automatic acceptance permitted.\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest != getattr(policy, pin)
    binding_code = project / BINDING
    binding_code.write_text(binding_code.read_text().replace(getattr(policy, pin), digest))
    changed_repository = root, project, commit(root)
    output = make_bundle(changed_repository, tmp_path / "repinned")
    for name, source in (("OperatorCriterionDecisionV2.md", CRITERION), ("AdditionalFailuresV1.json", ADDITIONAL)):
        (output / name).write_bytes((project / source).read_bytes())
    protocol = evidence.read_json(output / "ProtocolV1.json")
    protocol["criterion_sha256"] = evidence.sha256(output / "OperatorCriterionDecisionV2.md")
    protocol["additional_failures_sha256"] = evidence.sha256(output / "AdditionalFailuresV1.json")
    protocol[protocol_field] = evidence.sha256(output / artifact)
    write(output / "ProtocolV1.json", protocol)
    completion = evidence.read_json(output / "CompletionV1.json")
    completion["evidence_revision"] = 5
    write(output / "CompletionV1.json", completion)
    fail_record(output)
    # Both the candidate source and all bundle hashes agree with the new pin.
    # This verifier's independently installed constant must still reject it.
    with pytest.raises(ValueError, match=error):
        check(output, changed_repository)
