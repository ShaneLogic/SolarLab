"""Externally selected commit, frozen contract, and imported preparation chain.

Synthetic records below exercise identity checks; they never claim a physical
simulation or independent approval. Source snapshots come from real Git commits.
"""

import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import pytest

from perovskite_sim.experiments import one_dimensional_mechanism_r1_evidence as evidence
from perovskite_sim.experiments.one_dimensional_mechanism_r1_checkout import source_content_digest
from tests.fixtures.r1_reference import approved_r1_binding
from tests.unit.experiments.test_one_dimensional_mechanism_r1_controlled import (
    repository, launch, commit, git, INPUT, PROJECT,
)


CONTRACT = "docs/OneDimensionalMechanismR1DynamicsV1.md"
FIXTURE = "tests/fixtures/configs/dynamic_interface_defect_ion_transient_absorber_only.yaml"


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def seal(output):
    write(output / "ManifestV1.json", {
        path.relative_to(output).as_posix(): {
            "sha256": evidence.sha256(path), "bytes": path.stat().st_size,
        }
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.relative_to(output).as_posix() != "ManifestV1.json"
    })


def canonical_seal(value):
    payload = {key: child for key, child in value.items() if key != "sha256"}
    return {**payload, "sha256": hashlib.sha256(evidence._canonical(payload).encode()).hexdigest()}


def make_bundle(repository, output):
    root, project, revision = repository
    result = launch(repository, "--output", str(output))
    assert result.returncode == 0, result.stdout + result.stderr
    for name in evidence.COMMON_ARTIFACTS:
        if not (output / name).exists():
            write(output / name, {})
    for name, relative in (
        ("StudyInputV1.json", INPUT), ("ExecutionContractV1.md", CONTRACT),
        ("SourceFixtureV1.yaml", FIXTURE),
    ):
        (output / name).write_bytes((project / relative).read_bytes())
    execution = evidence.read_json(output / "ExecutionSourceV1.json")
    binding = approved_r1_binding()
    write(output / "ReferenceBindingV1.json", binding)
    protocol = {
        "stage_scope": "R1-1", "stage": "prepare", "intervals": 16,
        "input_sha256": evidence.sha256(output / "StudyInputV1.json"),
        "contract_sha256": evidence.sha256(output / "ExecutionContractV1.md"),
        "reference_file_sha256": evidence.sha256(output / "ReferenceBindingV1.json"),
        "reference_binding_sha256": binding["sha256"],
        "approved_reference_binding_sha256": binding["sha256"],
        "source_commit": revision, "source_content_sha256": execution["source_content_sha256"],
        "run_class": "formal", "preparation": None,
    }
    write(output / "ProtocolV1.json", protocol)
    prefix = "perovskite-sim/perovskite_sim/"
    source = canonical_seal({
        "files": {name[len(prefix):]: item["sha256"]
                  for name, item in execution["required_sources"].items()
                  if name.startswith(prefix) and name.endswith(".py")},
        "study_input": {"path": INPUT, "sha256": protocol["input_sha256"]},
        "run_class": "formal", "source_commit": revision,
        "source_content_sha256": execution["source_content_sha256"],
    })
    write(output / "PreparedStateV1.json", canonical_seal({
        "schema": "R1CommonStateV1", "source": source, "intervals": 16,
        "fixed_reference": binding, "reference_sha256": binding["sha256"],
        "preparation_checks": {"certified": True, "reasons": []},
        "dc_state": {"certificate": {"certified": True, "reasons": [], "optimizer_success": True}},
        "environment": {"note": "synthetic identity fixture; no simulation performed"},
    }))
    write(output / "CompletionV1.json", {
        "schema": "R1StageOneCompletionV1", "stage_scope": "R1-1", "stage": "prepare",
        "status": "passed", "failure": None, "evidence_revision": 4, "run_class": "formal",
    })
    seal(output)
    return output


@pytest.fixture
def source_repository(repository):
    root, project, _ = repository
    for relative in (CONTRACT, FIXTURE):
        (project / relative).write_bytes((PROJECT / relative).read_bytes())
    return root, project, commit(root)


@pytest.fixture
def prepared_bundle(source_repository, tmp_path):
    return make_bundle(source_repository, tmp_path / "prepared")


def accept(output, repository, **kwargs):
    defaults = {
        "expected_manifest_sha256": evidence.sha256(output / "ManifestV1.json"),
        "required_evidence_revision": 4,
        "expected_source_commit": repository[2], "source_repository": repository[0],
    }
    defaults.update(kwargs)
    return evidence.verify_acceptance(output, **defaults)


def consume(prepared, target):
    shutil.copytree(prepared, target)
    shutil.copytree(prepared, target / "PreparationV1")
    protocol = evidence.read_json(target / "ProtocolV1.json")
    protocol.update(stage="zero-check", prepared_file_sha256=evidence.sha256(target / "PreparedStateV1.json"))
    protocol["preparation"] = {
        "manifest_sha256": evidence.sha256(prepared / "ManifestV1.json"),
        "source_commit": protocol["source_commit"],
        "source_content_sha256": protocol["source_content_sha256"],
        "run_class": "formal", "evidence_revision": 4,
    }
    write(target / "ProtocolV1.json", protocol)
    completion = evidence.read_json(target / "CompletionV1.json")
    completion["stage"] = "zero-check"
    write(target / "CompletionV1.json", completion)
    write(target / "ZeroExcitationV1.json", {})
    seal(target)
    return target


def test_revision_four_matches_the_separately_selected_commit_and_contract(prepared_bundle, source_repository):
    completion, _ = accept(prepared_bundle, source_repository)
    assert completion["status"] == "passed"
    assert completion["verification"]["source_commit_anchor"] == source_repository[2]
    assert completion["verification"]["legacy"] is True
    assert "signatures or independent approval" in " ".join(completion["verification"]["limits"])
    assert "verification" not in evidence.read_json(prepared_bundle / "CompletionV1.json")


@pytest.mark.parametrize("anchor", [None, "HEAD", "HEAD^{commit}", "a" * 39, "A" * 40, "0" * 40])
def test_source_anchor_cannot_be_selected_by_bundle_or_git_expression(prepared_bundle, source_repository, anchor):
    with pytest.raises(ValueError, match="source commit anchor"):
        accept(prepared_bundle, source_repository, expected_source_commit=anchor)


def test_annotated_tag_object_is_not_a_commit_anchor(prepared_bundle, source_repository):
    root = source_repository[0]
    git(root, "-c", "user.name=R1 Test", "-c", "user.email=r1-test@example.invalid",
        "tag", "-a", "review", "-m", "tag object is not the commit")
    tag = git(root, "rev-parse", "review")
    assert tag != source_repository[2]
    with pytest.raises(ValueError, match="exact Git commit"):
        accept(prepared_bundle, source_repository, expected_source_commit=tag)


def test_another_valid_commit_cannot_authorize_this_snapshot(prepared_bundle, source_repository):
    root, project, _ = source_repository
    (root / "note.txt").write_text("new committed content")
    other = commit(root)
    with pytest.raises(ValueError, match="caller source commit anchor"):
        accept(prepared_bundle, source_repository, expected_source_commit=other)


@pytest.mark.parametrize("field,value", [
    ("source_commit", "0" * 40), ("source_content_sha256", "0" * 64), ("run_class", "development"),
])
def test_protocol_source_claims_are_bound_to_execution(prepared_bundle, source_repository, field, value):
    protocol = evidence.read_json(prepared_bundle / "ProtocolV1.json")
    protocol[field] = value
    write(prepared_bundle / "ProtocolV1.json", protocol)
    seal(prepared_bundle)
    with pytest.raises(ValueError, match="protocol source identity"):
        accept(prepared_bundle, source_repository)


@pytest.mark.parametrize("field,value", [("stage", "step"), ("stage_scope", "R1-2")])
def test_protocol_stage_claims_must_match_completion(prepared_bundle, source_repository, field, value):
    protocol = evidence.read_json(prepared_bundle / "ProtocolV1.json")
    protocol[field] = value
    write(prepared_bundle / "ProtocolV1.json", protocol)
    seal(prepared_bundle)
    with pytest.raises(ValueError, match="protocol completion identity mismatch: " + field):
        accept(prepared_bundle, source_repository)


def test_contract_only_commit_is_rejected_by_trusted_verifier_pin(prepared_bundle, source_repository):
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_checkout import _commit_tree, _commit_bytes

    root, project, _ = source_repository
    with (project / CONTRACT).open("a") as stream:
        stream.write("\nAn ignored shadow copy is rejected even when it carries an unchanged package.\n")
    changed = root, project, commit(root)
    # Production startup also rejects this commit. Build the adversarial source
    # metadata directly to isolate the verifier's independent constant check.
    startup = launch(changed)
    assert startup.returncode == 1
    assert "pinned" in startup.stderr
    output = prepared_bundle
    files = _commit_bytes(root, _commit_tree(root, changed[2]))
    identities = {name: {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
                  for name, raw in files.items()}
    with zipfile.ZipFile(output / "SourceV1.zip", "w") as archive:
        for name, raw in files.items():
            archive.writestr(name, raw)
    write(output / "SourceManifestV1.json", identities)
    execution = evidence.read_json(output / "ExecutionSourceV1.json")
    execution.update(source_commit=changed[2], observed_commit=changed[2])
    execution["required_sources"] = {name: identities[name] for name in execution["required_sources"]}
    execution["source_content_sha256"] = source_content_digest(execution["required_sources"])
    write(output / "ExecutionSourceV1.json", execution)
    (output / "ExecutionContractV1.md").write_bytes((project / CONTRACT).read_bytes())
    protocol = evidence.read_json(output / "ProtocolV1.json")
    protocol.update(source_commit=changed[2], source_content_sha256=execution["source_content_sha256"],
                    contract_sha256=evidence.sha256(output / "ExecutionContractV1.md"))
    write(output / "ProtocolV1.json", protocol)
    state = evidence.read_json(output / "PreparedStateV1.json")
    state["source"].update(source_commit=changed[2], source_content_sha256=execution["source_content_sha256"])
    state["source"] = canonical_seal(state["source"])
    write(output / "PreparedStateV1.json", canonical_seal(state))
    seal(output)
    # All bundle identities, source bytes, and caller commit agree. Only the
    # independent fixed contract policy rejects the changed declaration.
    with pytest.raises(ValueError, match="verifier's pinned contract"):
        accept(output, changed)


def test_source_zip_cannot_be_resealed_against_unchanged_commit(prepared_bundle, source_repository):
    output = prepared_bundle
    with zipfile.ZipFile(output / "SourceV1.zip") as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    files[".gitignore"] += b"# unapproved snapshot bytes\n"
    with zipfile.ZipFile(output / "SourceV1.zip", "w") as archive:
        for name, raw in files.items():
            archive.writestr(name, raw)
    write(output / "SourceManifestV1.json", {
        name: {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        for name, raw in files.items()
    })
    seal(output)
    with pytest.raises(ValueError, match="caller commit blobs"):
        accept(output, source_repository)


@pytest.mark.parametrize("field,value", [("run_class", "development"), ("source_commit", "0" * 40),
                                        ("source_content_sha256", "0" * 64)])
def test_prepared_source_cannot_be_resealed_as_another_execution(prepared_bundle, source_repository, field, value):
    state = evidence.read_json(prepared_bundle / "PreparedStateV1.json")
    state["source"][field] = value
    state["source"] = canonical_seal(state["source"])
    write(prepared_bundle / "PreparedStateV1.json", canonical_seal(state))
    seal(prepared_bundle)
    with pytest.raises(ValueError, match="prepared execution source"):
        accept(prepared_bundle, source_repository)


def test_prepare_payload_rehash_is_required(prepared_bundle, source_repository):
    state = evidence.read_json(prepared_bundle / "PreparedStateV1.json")
    state["environment"] = {"forged": True}
    write(prepared_bundle / "PreparedStateV1.json", state)
    seal(prepared_bundle)
    with pytest.raises(ValueError, match="canonical payload digest"):
        accept(prepared_bundle, source_repository)


def test_archived_preparation_has_an_independent_external_anchor(prepared_bundle, source_repository, tmp_path):
    output = consume(prepared_bundle, tmp_path / "zero")
    expected = evidence.sha256(prepared_bundle / "ManifestV1.json")
    completion, _ = accept(output, source_repository, expected_prepared_manifest_sha256=expected)
    assert completion["status"] == "passed"
    for anchor in (None, "0" * 64):
        with pytest.raises(ValueError, match="external.*digest|external anchor"):
            accept(output, source_repository, expected_prepared_manifest_sha256=anchor)


@pytest.mark.parametrize("artifact", ["environment", "preparation_checks", "dc_state"])
def test_inherited_unrecomputed_fields_cannot_replace_the_anchored_parent(
    prepared_bundle, source_repository, tmp_path, artifact,
):
    output = consume(prepared_bundle, tmp_path / "zero")
    original_anchor = evidence.sha256(prepared_bundle / "ManifestV1.json")
    state = evidence.read_json(output / "PreparedStateV1.json")
    if artifact == "dc_state":
        state[artifact]["certificate"]["unrecomputed_note"] = "injected"
    else:
        state[artifact]["unrecomputed_note"] = "injected"
    changed = canonical_seal(state)
    for directory in (output, output / "PreparationV1"):
        write(directory / "PreparedStateV1.json", changed)
    seal(output / "PreparationV1")
    protocol = evidence.read_json(output / "ProtocolV1.json")
    protocol["prepared_file_sha256"] = evidence.sha256(output / "PreparedStateV1.json")
    protocol["preparation"]["manifest_sha256"] = evidence.sha256(output / "PreparationV1/ManifestV1.json")
    write(output / "ProtocolV1.json", protocol)
    seal(output)
    with pytest.raises(ValueError, match="external anchor"):
        accept(output, source_repository, expected_prepared_manifest_sha256=original_anchor)


def test_formal_child_cannot_consume_a_development_preparation(prepared_bundle, source_repository, tmp_path):
    completion = evidence.read_json(prepared_bundle / "CompletionV1.json")
    completion["run_class"] = "development"
    write(prepared_bundle / "CompletionV1.json", completion)
    execution = evidence.read_json(prepared_bundle / "ExecutionSourceV1.json")
    execution["run_class"] = "development"
    write(prepared_bundle / "ExecutionSourceV1.json", execution)
    seal(prepared_bundle)
    output = consume(prepared_bundle, tmp_path / "zero")
    # Relabel the child; the independently anchored archived parent still fails.
    for filename in ("CompletionV1.json", "ExecutionSourceV1.json"):
        value = evidence.read_json(output / filename)
        value["run_class"] = "formal"
        write(output / filename, value)
    seal(output)
    with pytest.raises(ValueError, match="formal acceptance requires controlled execution"):
        accept(output, source_repository,
               expected_prepared_manifest_sha256=evidence.sha256(prepared_bundle / "ManifestV1.json"))


def test_consumer_state_must_be_exact_parent_bytes(prepared_bundle, source_repository, tmp_path):
    output = consume(prepared_bundle, tmp_path / "zero")
    state = evidence.read_json(output / "PreparedStateV1.json")
    state["environment"]["forged"] = True
    write(output / "PreparedStateV1.json", canonical_seal(state))
    protocol = evidence.read_json(output / "ProtocolV1.json")
    protocol["prepared_file_sha256"] = evidence.sha256(output / "PreparedStateV1.json")
    write(output / "ProtocolV1.json", protocol)
    seal(output)
    with pytest.raises(ValueError, match="differs from the anchored parent"):
        accept(output, source_repository,
               expected_prepared_manifest_sha256=evidence.sha256(prepared_bundle / "ManifestV1.json"))


@pytest.mark.parametrize("revision", [1, 2, 3])
def test_legacy_selection_is_explicit_and_marked(prepared_bundle, source_repository, revision):
    completion = evidence.read_json(prepared_bundle / "CompletionV1.json")
    completion["evidence_revision"] = revision
    write(prepared_bundle / "CompletionV1.json", completion)
    seal(prepared_bundle)
    with pytest.raises(ValueError, match="externally required revision"):
        accept(prepared_bundle, source_repository)
    accepted, _ = evidence.verify_acceptance(
        prepared_bundle, expected_manifest_sha256=evidence.sha256(prepared_bundle / "ManifestV1.json"),
        required_evidence_revision=revision,
    )
    assert accepted["verification"]["legacy"] is True
    assert "no external source commit" in " ".join(accepted["verification"]["limits"])
    with pytest.raises(ValueError, match="legacy acceptance does not implement"):
        accept(prepared_bundle, source_repository, required_evidence_revision=revision)


def make_ledger(output, repository, destination, *, top=True, entry=True):
    run = {
        "manifest_sha256": evidence.sha256(output / "ManifestV1.json"),
        "stage": "prepare", "recorded_status": "passed", "stage_scope": "R1-1",
        "source_manifest_sha256": evidence.sha256(output / "SourceManifestV1.json"),
        "study_input_sha256": evidence.sha256(output / "StudyInputV1.json"),
        "reference_binding_sha256": approved_r1_binding()["sha256"],
        "required_evidence_revision": 4,
    }
    if entry:
        run["source_commit"] = repository[2]
    value = {"schema": "R1EvidenceLedgerV1", "entries": {"prepare": run}}
    if top:
        value["source_commit"] = repository[2]
    write(destination, value)
    return destination


def accept_ledger(output, repository, ledger, **kwargs):
    return evidence.verify_acceptance(
        output, ledger=ledger, ledger_sha256=evidence.sha256(ledger), run_id="prepare",
        source_repository=repository[0], required_evidence_revision=4, **kwargs,
    )


@pytest.mark.parametrize("entry", [False, True])
def test_anchored_ledger_can_supply_the_source_commit(prepared_bundle, source_repository, tmp_path, entry):
    ledger = make_ledger(prepared_bundle, source_repository, tmp_path / "LedgerV1.json", entry=entry)
    completion, _ = accept_ledger(prepared_bundle, source_repository, ledger)
    assert completion["verification"]["source_commit_anchor"] == source_repository[2]
    assert accept_ledger(prepared_bundle, source_repository, ledger,
                         expected_source_commit=source_repository[2])[0]["status"] == "passed"


@pytest.mark.parametrize("location", ["top", "entry"])
def test_ledger_source_claims_are_checked_against_caller_and_each_other(
    prepared_bundle, source_repository, tmp_path, location,
):
    ledger = make_ledger(prepared_bundle, source_repository, tmp_path / "LedgerV1.json")
    value = evidence.read_json(ledger)
    target = value if location == "top" else value["entries"]["prepare"]
    target["source_commit"] = "0" * 40
    write(ledger, value)
    with pytest.raises(ValueError, match="source commit disagrees"):
        accept_ledger(prepared_bundle, source_repository, ledger,
                      expected_source_commit=source_repository[2])
    with pytest.raises(ValueError, match="source commit disagrees"):
        accept_ledger(prepared_bundle, source_repository, ledger)


@pytest.mark.parametrize("entry", [False, True])
def test_source_anchor_cannot_be_inferred_from_an_unselected_ledger_entry(
    prepared_bundle, source_repository, tmp_path, entry,
):
    ledger = make_ledger(prepared_bundle, source_repository, tmp_path / "LedgerV1.json", top=False, entry=entry)
    with pytest.raises(ValueError, match="source commit"):
        accept_ledger(prepared_bundle, source_repository, ledger)
    assert accept_ledger(prepared_bundle, source_repository, ledger,
                         expected_source_commit=source_repository[2])[0]["status"] == "passed"


def test_anchored_ledger_revision_does_not_override_external_selection(prepared_bundle, source_repository, tmp_path):
    ledger = make_ledger(prepared_bundle, source_repository, tmp_path / "LedgerV1.json")
    value = evidence.read_json(ledger)
    value["entries"]["prepare"]["required_evidence_revision"] = 3
    write(ledger, value)
    with pytest.raises(ValueError, match="ledger evidence revision"):
        accept_ledger(prepared_bundle, source_repository, ledger)
