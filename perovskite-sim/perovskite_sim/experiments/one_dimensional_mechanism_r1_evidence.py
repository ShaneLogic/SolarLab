"""Externally anchored R1 artifact and controlled-source identity checks.

Current acceptance requires revision six and re-evaluates saved state equations,
physical currents and declared execution parameters. Older formats are available
only through explicit historical inspection. No mode grants independent approval. Trust this
verifier, its Python startup and dependency installation, its fixed policy
constants, and the Git object database selected outside the bundle.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile


def _required_bytes(path):
    try:
        return Path(path).read_bytes()
    except OSError as exc:
        raise ValueError("required evidence artifact unavailable: " + Path(path).name) from exc


def sha256(path):
    return hashlib.sha256(_required_bytes(path)).hexdigest()


COMMON_ARTIFACTS = frozenset({
    "CompletionV1.json", "StudyInputV1.json", "SourceFixtureV1.yaml",
    "ExecutionContractV1.md", "EnvironmentV1.json", "ResolvedStackV1.json",
    "ProtocolV1.json", "ReferenceBindingV1.json", "PreparedStateV1.json",
    "SourceV1.zip", "SourceManifestV1.json", "SourceChangesV1.patch",
})


def _reject_constant(value):
    raise ValueError(f"non-finite JSON value: {value}")


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    return _read_json_bytes(_required_bytes(path))


def _read_json_bytes(raw):
    return json.loads(
        raw.decode("utf-8"),
        parse_constant=_reject_constant, object_pairs_hook=_unique_pairs,
    )


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest_argument(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest supplied outside the bundle")
    return value


def _verification_anchor(output, expected_manifest_sha256=None, ledger=None,
                         ledger_sha256=None, run_id=None):
    """Resolve a caller-trusted anchor; a bundle never supplies its own trust."""
    if expected_manifest_sha256 is not None:
        if ledger is not None or ledger_sha256 is not None or run_id is not None:
            raise ValueError("choose a manifest digest or a separately anchored ledger")
        return _digest_argument(expected_manifest_sha256, "expected manifest digest"), None, None
    if ledger is None or ledger_sha256 is None or not run_id:
        raise ValueError("acceptance requires an external manifest digest or ledger, ledger digest and run id")
    ledger = Path(ledger).resolve(strict=True)
    if ledger.is_relative_to(output.resolve()):
        raise ValueError("the acceptance ledger must be outside the bundle")
    if sha256(ledger) != _digest_argument(ledger_sha256, "expected ledger digest"):
        raise ValueError("external ledger digest mismatch")
    value = read_json(ledger)
    if not isinstance(value, dict) or value.get("schema") != "R1EvidenceLedgerV1":
        raise ValueError("invalid R1 evidence ledger")
    if not isinstance(value.get("entries"), dict):
        raise ValueError("invalid R1 evidence ledger entries")
    entry = value["entries"].get(run_id)
    if not isinstance(entry, dict):
        raise ValueError("run id is absent from the anchored ledger")
    required = {"manifest_sha256", "stage", "recorded_status", "stage_scope",
                "source_manifest_sha256", "study_input_sha256", "reference_binding_sha256"}
    if not required <= entry.keys():
        raise ValueError("anchored ledger entry lacks required identities")
    return (_digest_argument(entry["manifest_sha256"], "ledger manifest digest"),
            entry, value.get("source_commit"))


def verify_output(output):
    """Check a sealed bundle's bytes; do not promote it to new physics evidence."""
    output = output.resolve(strict=True)
    entries = read_json(output / "ManifestV1.json")
    if not isinstance(entries, dict) or "CompletionV1.json" not in entries:
        raise ValueError("manifest lacks the completion record")
    observed = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*") if path.is_file()
        and path.relative_to(output).as_posix() != "ManifestV1.json"
    }
    if observed != set(entries):
        raise ValueError("manifest does not cover exactly the saved files")
    for name, entry in entries.items():
        path = (output / name).resolve()
        if (
            Path(name).is_absolute() or not path.is_relative_to(output)
            or not path.is_file() or not isinstance(entry, dict)
            or path.stat().st_size != entry.get("bytes")
            or sha256(path) != entry.get("sha256")
        ):
            raise ValueError(f"artifact identity mismatch: {name}")
    completion = read_json(output / "CompletionV1.json")
    rejected = (completion.get("stage_scope") == "R1-rejected"
                and completion.get("status") == "failed"
                and completion.get("run_class") == "rejected_before_execution"
                and completion.get("physical_execution_started") is False
                and all(completion.get(name, 0) == 0 for name in (
                    "accepted_record_count", "observed_record_count", "persisted_record_count",
                    "persisted_finite_step_count", "physical_passed_finite_step_count")))
    if (
        completion.get("schema") != "R1StageOneCompletionV1"
        or not (rejected or completion.get("stage_scope") == "R1-1" or (
            completion.get("stage_scope") == "R1-2-physics"
            and completion.get("run_class") == "formal"
            and completion.get("evidence_revision") == 6) or (
            completion.get("stage_scope") == "R1-2-development"
            and completion.get("run_class") == "development"))
        or completion.get("stage") not in ("prepare", "zero-check", "step")
        or completion.get("status") not in ("passed", "failed")
        or (completion["status"] == "passed") != (completion.get("failure") is None)
    ):
        raise ValueError("invalid R1-1 completion record")
    if completion["status"] == "passed":
        required = set(COMMON_ARTIFACTS)
        if completion["stage"] == "zero-check":
            required.add("ZeroExcitationV1.json")
        if completion["stage"] == "step":
            required.update(("StepResultV1.json", "AcceptedStepsV1.json"))
        if not required <= entries.keys():
            raise ValueError("passed completion lacks required R1-1 evidence")
        source = read_json(output / "SourceManifestV1.json")
        if not isinstance(source, dict) or not source:
            raise ValueError("passed completion has empty source coverage")
        if completion.get("evidence_revision", 1) >= 2:
            if "ExecutionSourceV1.json" not in entries:
                raise ValueError("passed completion lacks execution checkout evidence")
    elif "FailureV1.json" not in entries:
        raise ValueError("failed completion lacks its failure record")
    return completion, len(entries)


def _producer_evidence_revision(source_bytes):
    """Read a producer's literal completion format without executing its code.

    This is read only after source ZIP/commit identity checks. It cannot be
    selected by editing a bundle's CompletionV1 or ProtocolV1 labels.
    """
    import ast
    try:
        tree = ast.parse(source_bytes)
    except (SyntaxError, UnicodeDecodeError) as exc:
        raise ValueError("producer source cannot declare its evidence format") from exc
    revisions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            entries = {key.value: value for key, value in zip(node.keys, node.values)
                       if isinstance(key, ast.Constant) and isinstance(key.value, str)}
            schema, revision = entries.get("schema"), entries.get("evidence_revision")
            if (isinstance(schema, ast.Constant) and schema.value == "R1StageOneCompletionV1"
                    and isinstance(revision, ast.Constant) and type(revision.value) is int):
                revisions.append(revision.value)
    return max(revisions, default=None)


def _verify_archived_source_identity(output, committed_source):
    """Bind legacy source bytes to a caller's commit without executing them."""
    source = read_json(output / "SourceManifestV1.json")
    expected = {name: {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
                for name, raw in committed_source.items()}
    if source != expected:
        raise ValueError("source snapshot differs from the caller commit blobs")
    try:
        with zipfile.ZipFile(output / "SourceV1.zip") as archive:
            if len(archive.namelist()) != len(expected) or set(archive.namelist()) != set(expected):
                raise ValueError("source ZIP and trusted commit coverage differ")
            for name, raw in committed_source.items():
                if archive.read(name) != raw:
                    raise ValueError("source ZIP differs from caller commit: " + name)
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise ValueError("invalid controlled source ZIP") from exc


def evidence_file_scope(output):
    """Byte-consistent extra attachments are not silently scientific certificates."""
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_checkout import R1_DECLARATIONS
    known = {
        "CompletionV1.json", "ManifestV1.json", "SourceManifestV1.json", "SourceV1.zip", "SourceChangesV1.patch",
        "StudyInputV1.json", "ProtocolV1.json", "ReferenceBindingV1.json", "SourceFixtureV1.yaml",
        "EnvironmentV1.json", "ExecutionSourceV1.json", "AdditionalFailuresV1.json", "AdditionalFailuresV2.json",
        "PreparedStateV1.json", "PreparedStateV1.npz", "StepResultV1.json", "StepResultV1.npz",
        "AcceptedStepsV1.json", "AcceptedStepsV1.npz", "ZeroExcitationV1.json", "ZeroExcitationV1.npz",
        "FailureV1.json", "FailedResultV1.json", "FailedResultV1.npz", "FailureSerializationV1.json",
        *(declaration[1] for declaration in R1_DECLARATIONS.values()),
    }
    entries = read_json(Path(output) / "ManifestV1.json")
    # A consuming bundle includes one explicitly checked PreparationV1 parent.
    unclassified = [name for name in entries
                    if name not in known and not (name.startswith("PreparationV1/")
                                                 and name.removeprefix("PreparationV1/") in known)]
    return {"byte_consistency_file_count": len(entries),
            "unclassified_attachments": sorted(unclassified),
            "unclassified_attachment_contents_verified": False,
            "scope": "file_bytes_and_classified_record_content_are_separate_checks"}


def _verify_controlled_evidence(output, completion, *, expected_source_commit=None,
                                committed_source=None, pin_contract=False):
    """Check the verifier-selected controlled format and optional source root."""
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_checkout import (
        REQUIRED_SOURCE_ANCHORS, LEGACY_SOURCE_ANCHORS, source_content_digest, R1_DECLARATIONS,
    )
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_binding as binding_policy

    revision = completion.get("evidence_revision", 1)
    permitted_scopes = ("R1-1", "R1-2-physics") if revision >= 6 else ("R1-1",)
    if completion.get("stage_scope") not in permitted_scopes:
        raise ValueError("formal acceptance rejects development or unexecuted scope")
    execution = read_json(output / "ExecutionSourceV1.json")
    if (execution.get("schema") != "R1ExecutionSourceV2"
            or execution.get("run_class") != "formal" or completion.get("run_class") != "formal"):
        raise ValueError("formal acceptance requires controlled execution evidence")
    commit = execution.get("source_commit")
    if (not isinstance(commit, str) or len(commit) not in (40, 64)
            or any(c not in "0123456789abcdef" for c in commit)
            or execution.get("observed_commit") != commit):
        raise ValueError("controlled execution source commit mismatch")
    if expected_source_commit is not None and commit != expected_source_commit:
        raise ValueError("execution source differs from the caller source commit anchor")
    runtime = execution.get("runtime")
    if (not isinstance(runtime, dict) or runtime.get("isolated") is not True
            or runtime.get("no_site") is not True
            or runtime.get("project_bytecode_cache_used") is not False
            or runtime.get("project_loader") != "FrozenSourceLoader"):
        raise ValueError("controlled execution runtime evidence is incomplete")
    source = read_json(output / "SourceManifestV1.json")
    required = execution.get("required_sources")
    revision = completion.get("evidence_revision", 1)
    fifth_anchors = (*LEGACY_SOURCE_ANCHORS,
        "docs/OneDimensionalMechanismR1OperatorCriterionDecisionV2.md",
        "reproducibility/OneDimensionalMechanismR1AdditionalFailuresV1.json")
    anchors = REQUIRED_SOURCE_ANCHORS if revision >= 6 else fifth_anchors if revision == 5 else LEGACY_SOURCE_ANCHORS
    expected = {"perovskite-sim/" + name for name in anchors}
    if revision < 6 and isinstance(required, dict):
        # Old formats may have been produced with a superset of the old roots.
        # Accept only this known extension, never arbitrary bundle-selected roots.
        expected.update("perovskite-sim/" + name for name in REQUIRED_SOURCE_ANCHORS
                        if "perovskite-sim/" + name in required)
    expected.update(name for name in source
                    if name.startswith("perovskite-sim/perovskite_sim/") and name.endswith(".py"))
    if not isinstance(required, dict) or set(required) != expected or not expected <= source.keys():
        raise ValueError("execution source coverage lacks the required package sources or anchors")
    if any(required[name] != source[name] for name in expected):
        raise ValueError("required source identities disagree with the source manifest")
    if execution.get("required_source_content_mismatches") != []:
        raise ValueError("formal execution records source content mismatches")
    if source_content_digest(required) != execution.get("source_content_sha256"):
        raise ValueError("execution source content digest mismatch")
    if committed_source is not None:
        committed_identities = {
            name: {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
            for name, raw in committed_source.items()
        }
        if source != committed_identities:
            raise ValueError("source snapshot differs from the caller commit blobs")
    try:
        with zipfile.ZipFile(output / "SourceV1.zip") as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or set(names) != set(source):
                raise ValueError("source ZIP and source manifest coverage differ")
            for name, identity in source.items():
                if (not isinstance(identity, dict) or Path(name).is_absolute()
                        or ".." in Path(name).parts
                        or archive.getinfo(name).file_size != identity.get("bytes")
                        or hashlib.sha256(archive.read(name)).hexdigest() != identity.get("sha256")):
                    raise ValueError("source ZIP identity mismatch: " + name)
            producer_revision = (_producer_evidence_revision(committed_source[
                "perovskite-sim/scripts/run_one_dimensional_mechanism_r1_stage_one.py"])
                if committed_source is not None else None)
            if producer_revision is not None and revision < producer_revision:
                raise ValueError("bundle evidence revision downgrades its anchored producer format")
            input_name = "perovskite-sim/reproducibility/OneDimensionalMechanismR1DynamicsInputV1.json"
            if archive.read(input_name) != _required_bytes(output / "StudyInputV1.json"):
                raise ValueError("study input differs from the controlled source snapshot")
            contract_name = "perovskite-sim/docs/OneDimensionalMechanismR1DynamicsV1.md"
            if archive.read(contract_name) != _required_bytes(output / "ExecutionContractV1.md"):
                raise ValueError("contract differs from the controlled source snapshot")
            if revision >= 5:
                criterion = _required_bytes(output / "OperatorCriterionDecisionV2.md")
                if archive.read("perovskite-sim/" + binding_policy.OPERATOR_CRITERION_RELATIVE_PATH) != criterion:
                    raise ValueError("operator criterion differs from the controlled source snapshot")
                criterion_pins = {binding_policy.PINNED_OPERATOR_CRITERION_SHA256}
                if revision < 6:
                    criterion_pins.add("0adf2d3fec785febb9747c47e62a532fb23fabe6d58d08925e8de5573f657b9b")
                if hashlib.sha256(criterion).hexdigest() not in criterion_pins:
                    raise ValueError("operator criterion does not match this verifier's pinned decision")
                additional = _required_bytes(output / "AdditionalFailuresV1.json")
                if archive.read("perovskite-sim/" + binding_policy.ADDITIONAL_FAILURES_RELATIVE_PATH) != additional:
                    raise ValueError("additional failures differ from the controlled source snapshot")
                if hashlib.sha256(additional).hexdigest() != binding_policy.PINNED_ADDITIONAL_FAILURES_SHA256:
                    raise ValueError("additional failures do not match this verifier's pinned registry")
            if revision >= 6:
                additional_v2 = _required_bytes(output / "AdditionalFailuresV2.json")
                if (archive.read("perovskite-sim/" + binding_policy.ADDITIONAL_FAILURES_V2_RELATIVE_PATH) != additional_v2
                        or hashlib.sha256(additional_v2).hexdigest() != binding_policy.PINNED_ADDITIONAL_FAILURES_V2_SHA256):
                    raise ValueError("additional failures V2 differ from the pinned registry")
                physics = _required_bytes(output / "PhysicsProtocolV1.md")
                if archive.read("perovskite-sim/" + binding_policy.PHYSICS_PROTOCOL_RELATIVE_PATH) != physics:
                    raise ValueError("physics protocol differs from controlled source snapshot")
                for relative, digest in binding_policy.FROZEN_PHYSICS_DECLARATIONS.items():
                    if hashlib.sha256(archive.read("perovskite-sim/" + relative)).hexdigest() != digest:
                        raise ValueError("physics declaration differs from trusted pinned decision: " + relative)
                    if archive.read("perovskite-sim/" + relative) != _required_bytes(output / R1_DECLARATIONS[relative][1]):
                        raise ValueError("exported declaration differs from controlled source snapshot: " + relative)
                # The caller's Git anchor proves content identity, not that an
                # arbitrary candidate implementation matches this verifier.
                package = Path(__file__).resolve().parents[1]
                installed = {"perovskite-sim/perovskite_sim/" + item.relative_to(package).as_posix(): item
                             for item in package.rglob("*.py")}
                archived = {name for name in source if name.startswith("perovskite-sim/perovskite_sim/")
                            and name.endswith(".py")}
                if archived != set(installed):
                    raise ValueError("recorded package differs from trusted installed package coverage")
                for name, path in installed.items():
                    if archive.read(name) != path.read_bytes():
                        raise ValueError("recorded package differs from trusted installed package: " + name)
            study = read_json(output / "StudyInputV1.json")
            fixture_name = "perovskite-sim/" + study["fixture"]
            fixture_bytes = _required_bytes(output / "SourceFixtureV1.yaml")
            if (archive.read(fixture_name) != fixture_bytes
                    or hashlib.sha256(fixture_bytes).hexdigest() != study["fixture_sha256"]):
                raise ValueError("source fixture differs from the controlled snapshot or pinned input")
    except (zipfile.BadZipFile, KeyError, OSError) as exc:
        raise ValueError("invalid controlled source ZIP") from exc
    input_pins = {binding_policy.PINNED_STUDY_INPUT_SHA256}
    if revision == 3:
        # Reviewed 794dd7e/84f9131 legacy input. This exception cannot select
        # revision five or promote the old evidence to current certification.
        input_pins.add("6a06878c467efe1b4d1a597ffcc45a8ad566ff347c4cdf868af094a36e8b64ca")
    if sha256(output / "StudyInputV1.json") not in input_pins:
        raise ValueError("study input does not match this verifier's pinned protocol")
    contract_pins = {binding_policy.PINNED_EXECUTION_CONTRACT_SHA256}
    if revision < 6:
        contract_pins.add("a0a918d175c7989db60d5924b7600baadd1929cc54c7ac66b3ec3ccdcc1f9955")
    if pin_contract and sha256(output / "ExecutionContractV1.md") not in contract_pins:
        raise ValueError("contract does not match this verifier's pinned contract")
    study, protocol = read_json(output / "StudyInputV1.json"), read_json(output / "ProtocolV1.json")
    if revision >= 6:
        expected_physics = binding_policy.PINNED_PHYSICS_PROTOCOL_SHA256
        if (protocol.get("physics_protocol_sha256") != expected_physics
                or sha256(output / "PhysicsProtocolV1.md") != expected_physics):
            raise ValueError("protocol physical acceptance identity mismatch")
    if revision >= 5 and protocol.get("criterion_sha256") != sha256(output / "OperatorCriterionDecisionV2.md"):
        raise ValueError("protocol operator criterion identity mismatch")
    if revision >= 5 and protocol.get("additional_failures_sha256") != sha256(output / "AdditionalFailuresV1.json"):
        raise ValueError("protocol additional failures identity mismatch")
    for field in ("stage", "stage_scope"):
        if protocol.get(field) != completion[field]:
            raise ValueError("protocol completion identity mismatch: " + field)
    binding = read_json(output / "ReferenceBindingV1.json")
    payload = {key: value for key, value in binding.items() if key != "sha256"}
    # ReferenceBindingV1 predates the compact-JSON common-state encoding.
    payload_sha256 = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if payload_sha256 != binding.get("sha256"):
        raise ValueError("reference binding canonical payload digest mismatch")
    expected_reference = study["fixed_reference_binding_sha256"]
    if any(value != expected_reference for value in (
        binding["sha256"], protocol.get("reference_binding_sha256"),
        protocol.get("approved_reference_binding_sha256"),
    )):
        raise ValueError("reference identities disagree with the pinned study input")
    for field, filename in (("input_sha256", "StudyInputV1.json"),
                            ("contract_sha256", "ExecutionContractV1.md"),
                            ("reference_file_sha256", "ReferenceBindingV1.json")):
        if protocol.get(field) != sha256(output / filename):
            raise ValueError("protocol artifact identity mismatch: " + field)
    if expected_source_commit is not None:
        for field, expected_value in (
            ("source_commit", expected_source_commit),
            ("source_content_sha256", execution["source_content_sha256"]),
            ("run_class", "formal"),
        ):
            if protocol.get(field) != expected_value:
                raise ValueError("protocol source identity mismatch: " + field)
    return execution


def _source_anchor(expected_source_commit, source_repository):
    """Read regular blobs from an explicitly selected exact commit object."""
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_checkout import (
        _commit_bytes, _commit_tree, _git,
    )

    if (not isinstance(expected_source_commit, str)
            or len(expected_source_commit) not in (40, 64)
            or any(c not in "0123456789abcdef" for c in expected_source_commit)):
        raise ValueError("revision-four acceptance requires a full caller source commit anchor")
    # This default belongs to the installed/trusted verifier, never to a bundle
    # path or a source-repository field supplied by the archived execution.
    repository = (Path(__file__).resolve().parents[3] if source_repository is None
                  else Path(source_repository).resolve(strict=True))
    try:
        if _git(repository, "cat-file", "-t", expected_source_commit).strip() != b"commit":
            raise ValueError("caller source anchor is not an exact Git commit object")
        tree = _commit_tree(repository, expected_source_commit)
        if not tree:
            raise ValueError("caller source commit has no tracked source files")
        return expected_source_commit, _commit_bytes(repository, tree)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("caller source commit anchor is unavailable or invalid: " + str(exc)) from exc


def _verify_prepared_record(output, execution, *, classify_current=True):
    """Bind the saved state to the accepted source, without rerunning its DC."""
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_result_contract import verify_prepared_metadata
    prepared = read_json(output / "PreparedStateV1.json")
    if classify_current:
        verify_prepared_metadata(prepared)
    if not isinstance(prepared, dict) or prepared.get("schema") != "R1CommonStateV1":
        raise ValueError("invalid revision-four prepared-state record")
    payload = {key: value for key, value in prepared.items() if key != "sha256"}
    if hashlib.sha256(_canonical(payload).encode()).hexdigest() != prepared.get("sha256"):
        raise ValueError("prepared-state canonical payload digest mismatch")
    source = prepared.get("source")
    if not isinstance(source, dict):
        raise ValueError("prepared state lacks execution source identity")
    source_payload = {key: value for key, value in source.items() if key != "sha256"}
    if hashlib.sha256(_canonical(source_payload).encode()).hexdigest() != source.get("sha256"):
        raise ValueError("prepared execution source canonical digest mismatch")
    prefix = "perovskite-sim/perovskite_sim/"
    expected_files = {
        name[len(prefix):]: identity["sha256"]
        for name, identity in execution["required_sources"].items()
        if name.startswith(prefix) and name.endswith(".py")
    }
    expected_source = {
        "files": expected_files,
        "study_input": {
            "path": "reproducibility/OneDimensionalMechanismR1DynamicsInputV1.json",
            "sha256": sha256(output / "StudyInputV1.json"),
        },
        "run_class": "formal", "source_commit": execution["source_commit"],
        "source_content_sha256": execution["source_content_sha256"],
    }
    if source_payload != expected_source:
        raise ValueError("prepared execution source differs from the controlled source anchor")
    protocol = read_json(output / "ProtocolV1.json")
    binding = read_json(output / "ReferenceBindingV1.json")
    if (prepared.get("fixed_reference") != binding
            or prepared.get("reference_sha256") != binding["sha256"]
            or prepared.get("intervals") != protocol.get("intervals")):
        raise ValueError("prepared state and protocol identities disagree")
    for label, certificate in (
        ("preparation checks", prepared.get("preparation_checks")),
        ("DC certificate", prepared.get("dc_state", {}).get("certificate")),
    ):
        if (not isinstance(certificate, dict) or certificate.get("certified") is not True
                or certificate.get("reasons") != []):
            raise ValueError("prepared state lacks passed " + label)
    if prepared["dc_state"]["certificate"].get("optimizer_success") is not True:
        raise ValueError("prepared DC optimizer was not accepted")
    return prepared


def _verify_preparation_chain(output, completion, execution, *, source_repository,
                              expected_prepared_manifest_sha256):
    """Check an externally pinned archived parent and the exact imported bytes."""
    protocol = read_json(output / "ProtocolV1.json")
    revision = completion["evidence_revision"]
    prepared = _verify_prepared_record(output, execution, classify_current=revision >= 6)
    if revision >= 5:
        from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import verify_prepared_physics
        from perovskite_sim.models.config_loader import load_device_from_yaml

        verify_prepared_physics(prepared, load_device_from_yaml(output / "SourceFixtureV1.yaml"),
                                read_json(output / "ReferenceBindingV1.json"))
    parent = output / "PreparationV1"
    if completion["stage"] == "prepare":
        if (protocol.get("preparation", "missing") is not None or parent.exists()
                or expected_prepared_manifest_sha256 is not None):
            raise ValueError("prepare evidence must not claim an imported preparation chain")
        return
    expected = _digest_argument(expected_prepared_manifest_sha256, "external prepared manifest digest")
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError("formal consuming evidence lacks the archived preparation package")
    parent_completion = read_json(parent / "CompletionV1.json")
    if parent_completion.get("stage") != "prepare" or parent_completion.get("status") != "passed":
        raise ValueError("preparation chain must refer to a passed prepare stage")
    parent_verifier = verify_acceptance if revision >= 6 else inspect_legacy_evidence
    parent_completion, _ = parent_verifier(
        parent, expected_manifest_sha256=expected,
        expected_source_commit=execution["source_commit"],
        source_repository=source_repository, required_evidence_revision=revision,
    )
    parent_execution = read_json(parent / "ExecutionSourceV1.json")
    expected_identity = {
        "manifest_sha256": expected, "source_commit": execution["source_commit"],
        "source_content_sha256": execution["source_content_sha256"],
        "run_class": "formal", "evidence_revision": revision,
    }
    if protocol.get("preparation") != expected_identity:
        raise ValueError("protocol preparation identity differs from the external parent anchor")
    if parent_execution["source_content_sha256"] != execution["source_content_sha256"]:
        raise ValueError("preparation and consuming source content differ")
    # These exact bytes include the inherited preparation checks, environment,
    # and DC certificate. Their recorded provenance cannot be silently replaced
    # while preserving an externally selected parent manifest identity.
    for name in ("PreparedStateV1.json", "StudyInputV1.json", "ExecutionContractV1.md",
                 "ReferenceBindingV1.json", "SourceFixtureV1.yaml"):
        if _required_bytes(output / name) != _required_bytes(parent / name):
            raise ValueError("imported preparation artifact differs from the anchored parent: " + name)
    if revision >= 5:
        for name in ("OperatorCriterionDecisionV2.md", "AdditionalFailuresV1.json"):
            if _required_bytes(output / name) != _required_bytes(parent / name):
                raise ValueError("imported policy artifact differs from the anchored parent: " + name)
    if revision >= 6 and _required_bytes(output / "PhysicsProtocolV1.md") != _required_bytes(parent / "PhysicsProtocolV1.md"):
        raise ValueError("imported physics protocol differs from anchored parent")
    if protocol.get("prepared_file_sha256") != sha256(output / "PreparedStateV1.json"):
        raise ValueError("protocol prepared-state file identity mismatch")


def _verify_anchored_evidence(output, *, expected_manifest_sha256=None, ledger=None,
                      ledger_sha256=None, run_id=None, required_evidence_revision=6,
                      expected_source_commit=None, source_repository=None,
                      expected_prepared_manifest_sha256=None, approved_standard_sha256=None):
    output = Path(output).resolve(strict=True)
    expected, entry, ledger_source_commit = _verification_anchor(
        output, expected_manifest_sha256, ledger, ledger_sha256, run_id,
    )
    if sha256(output / "ManifestV1.json") != expected:
        raise ValueError("bundle manifest differs from the supplied external anchor")
    completion, count = verify_output(output)
    result_checks = None
    execution_parameters = None
    if required_evidence_revision not in (1, 2, 3, 4, 5, 6):
        raise ValueError("unsupported externally required evidence revision")
    if completion.get("stage_scope") == "R1-rejected":
        raise ValueError("physical execution did not start; required scientific inputs and trajectory unavailable")
    if required_evidence_revision >= 4 or expected_source_commit is not None or ledger_source_commit is not None:
        if expected_source_commit is None:
            expected_source_commit = ledger_source_commit
        elif ledger_source_commit is not None and expected_source_commit != ledger_source_commit:
            raise ValueError("caller source commit disagrees with the anchored ledger source commit")
        if (entry is not None and "source_commit" in entry
                and entry["source_commit"] != expected_source_commit):
            raise ValueError("ledger entry source commit disagrees with the external source anchor")
        source_commit, committed_source = _source_anchor(expected_source_commit, source_repository)
    else:
        source_commit = committed_source = None
    if required_evidence_revision < 4 and expected_prepared_manifest_sha256 is not None:
        raise ValueError("legacy inspection before revision four does not verify preparation-chain anchors")
    # A failure is evidence too: it cannot bypass format, source or policy checks.
    if completion.get("evidence_revision", 1) != required_evidence_revision:
        raise ValueError("bundle evidence revision differs from the externally required revision")
    producer_revision = None
    if committed_source is not None:
        producer_revision = _producer_evidence_revision(committed_source.get(
            "perovskite-sim/scripts/run_one_dimensional_mechanism_r1_stage_one.py", b""))
        if producer_revision is not None and required_evidence_revision < producer_revision:
            raise ValueError("bundle evidence revision downgrades its anchored producer format")
    if required_evidence_revision < 4 and committed_source is not None:
        _verify_archived_source_identity(output, committed_source)
        if required_evidence_revision == 3:
            _verify_controlled_evidence(output, completion, expected_source_commit=source_commit,
                                        committed_source=committed_source)
    elif required_evidence_revision >= 4:
        execution = _verify_controlled_evidence(
            output, completion, expected_source_commit=source_commit,
            committed_source=committed_source, pin_contract=True,
        )
        if (completion["status"] == "passed" or completion["stage"] != "prepare"
                or (output / "PreparedStateV1.json").exists()):
            _verify_preparation_chain(
                output, completion, execution, source_repository=source_repository,
                expected_prepared_manifest_sha256=expected_prepared_manifest_sha256,
            )
        else:
            protocol = read_json(output / "ProtocolV1.json")
            if (protocol.get("preparation", "missing") is not None
                    or (output / "PreparationV1").exists() or expected_prepared_manifest_sha256 is not None):
                raise ValueError("failed prepare evidence must not claim an imported preparation chain")
        if required_evidence_revision >= 6:
            from perovskite_sim.experiments.one_dimensional_mechanism_r1_execution_validation import verify_execution_parameters
            execution_parameters = verify_execution_parameters(output, completion)
        if required_evidence_revision >= 5:
            from perovskite_sim.experiments.one_dimensional_mechanism_r1_result_validation import verify_result_records
            result_checks = verify_result_records(output, completion)
            if (required_evidence_revision >= 6 and completion["status"] == "passed"
                    and not result_checks.get("scientifically_accepted", False)):
                raise ValueError("passed record lacks current physical acceptance")
    standard = None
    if required_evidence_revision >= 6:
        from types import SimpleNamespace
        from perovskite_sim.experiments.one_dimensional_mechanism_r1_binding import standard_binding_record
        selected = SimpleNamespace(source_commit=source_commit,
            read_bytes=lambda relative: committed_source["perovskite-sim/" + str(relative)])
        standard = standard_binding_record(selected, approved_standard_sha256=approved_standard_sha256)
    elif approved_standard_sha256 is not None:
        raise ValueError("historical inspection cannot assert current standard approval")
    if entry is not None:
        if ("required_evidence_revision" in entry
                and entry["required_evidence_revision"] != required_evidence_revision):
            raise ValueError("ledger evidence revision differs from the externally required revision")
        actual = {
            "stage": completion["stage"], "stage_scope": completion["stage_scope"],
            "recorded_status": completion["status"],
            "source_manifest_sha256": sha256(output / "SourceManifestV1.json"),
            "study_input_sha256": sha256(output / "StudyInputV1.json"),
            "reference_binding_sha256": read_json(output / "ReferenceBindingV1.json")["sha256"],
        }
        if any(entry[key] != value for key, value in actual.items()):
            raise ValueError("bundle identities disagree with the anchored ledger entry")
    # Return annotations without changing the sealed completion record. Legacy
    # selection is explicit and cannot acquire revision-four certificate weight.
    limits = ["external anchors are not signatures or independent approval",
              "historical environment, timestamps and optimizer history are recorded provenance, not independently verified"]
    if required_evidence_revision < 4:
        limits.append("legacy revision: no current pinned contract or preparation-chain acceptance")
        if committed_source is None:
            limits.append("no external source commit anchor; byte inspection only; no producer-format claim")
    if required_evidence_revision < 5:
        limits.append("legacy revision: preparation numerical claims, result records and operator decision are not recertified")
    else:
        limits.append("record consistency checked; time integration not rerun")
        if completion["status"] == "passed" or completion["stage"] != "prepare":
            limits.append("saved preparation physics reevaluated")
        else:
            limits.append("failed preparation remains failed; only present preparation content reevaluated")
    if completion["status"] == "failed":
        limits.append("failed record remains failed; successful-evidence structure not certified")
    if required_evidence_revision < 6:
        limits.append("historical inspection only; not eligible for current scientific acceptance")
    elif completion["stage"] == "step":
        limits.append("saved states, rates, currents and declared equation residuals independently reevaluated; no time integration replay")
    return {**completion, "verification": {
        "required_evidence_revision": required_evidence_revision,
        "legacy": required_evidence_revision < 6,
        "mode": "physical_acceptance" if required_evidence_revision >= 6 else "historical_inspection",
        "scientifically_accepted": bool(required_evidence_revision >= 6 and completion["status"] == "passed"
                                        and result_checks and result_checks.get("scientifically_accepted")),
        "execution_parameters": execution_parameters,
        "source_commit_anchor": source_commit,
        "source_identity_verified": committed_source is not None,
        "producer_format_verified": producer_revision is not None,
        "standard_binding": standard,
        "file_scope": evidence_file_scope(output),
        "result_checks": result_checks,
        "limits": limits,
    }}, count


def verify_acceptance(output, *, expected_manifest_sha256=None, ledger=None,
                      ledger_sha256=None, run_id=None, required_evidence_revision=6,
                      expected_source_commit=None, source_repository=None,
                      expected_prepared_manifest_sha256=None, approved_standard_sha256=None):
    """Current physical acceptance; historical contracts cannot lower this gate."""
    if type(required_evidence_revision) is not int or required_evidence_revision != 6:
        raise ValueError("current physical acceptance requires evidence revision 6; use inspect_legacy_evidence for historical inspection")
    return _verify_anchored_evidence(output, expected_manifest_sha256=expected_manifest_sha256,
        ledger=ledger, ledger_sha256=ledger_sha256, run_id=run_id,
        required_evidence_revision=6, expected_source_commit=expected_source_commit,
        source_repository=source_repository,
        expected_prepared_manifest_sha256=expected_prepared_manifest_sha256,
        approved_standard_sha256=approved_standard_sha256)


def inspect_legacy_evidence(output, *, expected_manifest_sha256=None, ledger=None,
                            ledger_sha256=None, run_id=None, required_evidence_revision=5,
                            expected_source_commit=None, source_repository=None,
                            expected_prepared_manifest_sha256=None):
    """Inspect externally anchored revisions 1-5 without current science approval."""
    if type(required_evidence_revision) is not int or required_evidence_revision not in (1, 2, 3, 4, 5):
        raise ValueError("historical inspection supports only evidence revisions 1 through 5")
    return _verify_anchored_evidence(output, expected_manifest_sha256=expected_manifest_sha256,
        ledger=ledger, ledger_sha256=ledger_sha256, run_id=run_id,
        required_evidence_revision=required_evidence_revision,
        expected_source_commit=expected_source_commit, source_repository=source_repository,
        expected_prepared_manifest_sha256=expected_prepared_manifest_sha256)
