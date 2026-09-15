"""Externally anchored R1 artifact and controlled-source identity checks.

Acceptance verifies identities and declared execution structure. It neither
reruns physics nor grants independent approval. The caller must trust this
verifier, its Python startup and dependency installation, its fixed policy
constants, and the Git object database selected outside the bundle.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
    return _read_json_bytes(Path(path).read_bytes())


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
    if (
        completion.get("schema") != "R1StageOneCompletionV1"
        or completion.get("stage_scope") != "R1-1"
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


def _verify_controlled_evidence(output, completion, *, expected_source_commit=None,
                                committed_source=None, pin_contract=False):
    """Check the verifier-selected controlled format and optional source root."""
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_checkout import (
        REQUIRED_SOURCE_ANCHORS, source_content_digest,
    )
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_binding as binding_policy

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
    expected = {"perovskite-sim/" + name for name in REQUIRED_SOURCE_ANCHORS}
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
            input_name = "perovskite-sim/reproducibility/OneDimensionalMechanismR1DynamicsInputV1.json"
            if archive.read(input_name) != (output / "StudyInputV1.json").read_bytes():
                raise ValueError("study input differs from the controlled source snapshot")
            contract_name = "perovskite-sim/docs/OneDimensionalMechanismR1DynamicsV1.md"
            if archive.read(contract_name) != (output / "ExecutionContractV1.md").read_bytes():
                raise ValueError("contract differs from the controlled source snapshot")
            study = read_json(output / "StudyInputV1.json")
            fixture_name = "perovskite-sim/" + study["fixture"]
            fixture_bytes = (output / "SourceFixtureV1.yaml").read_bytes()
            if (archive.read(fixture_name) != fixture_bytes
                    or hashlib.sha256(fixture_bytes).hexdigest() != study["fixture_sha256"]):
                raise ValueError("source fixture differs from the controlled snapshot or pinned input")
    except (zipfile.BadZipFile, KeyError) as exc:
        raise ValueError("invalid controlled source ZIP") from exc
    if sha256(output / "StudyInputV1.json") != binding_policy.PINNED_STUDY_INPUT_SHA256:
        raise ValueError("study input does not match this verifier's pinned protocol")
    if pin_contract and sha256(output / "ExecutionContractV1.md") != binding_policy.PINNED_EXECUTION_CONTRACT_SHA256:
        raise ValueError("contract does not match this verifier's pinned contract")
    study, protocol = read_json(output / "StudyInputV1.json"), read_json(output / "ProtocolV1.json")
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


def _verify_prepared_record(output, execution):
    """Bind the saved state to the accepted source, without rerunning its DC."""
    prepared = read_json(output / "PreparedStateV1.json")
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
    _verify_prepared_record(output, execution)
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
    parent_completion, _ = verify_acceptance(
        parent, expected_manifest_sha256=expected,
        expected_source_commit=execution["source_commit"],
        source_repository=source_repository, required_evidence_revision=4,
    )
    parent_execution = read_json(parent / "ExecutionSourceV1.json")
    expected_identity = {
        "manifest_sha256": expected, "source_commit": execution["source_commit"],
        "source_content_sha256": execution["source_content_sha256"],
        "run_class": "formal", "evidence_revision": 4,
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
        if (output / name).read_bytes() != (parent / name).read_bytes():
            raise ValueError("imported preparation artifact differs from the anchored parent: " + name)
    if protocol.get("prepared_file_sha256") != sha256(output / "PreparedStateV1.json"):
        raise ValueError("protocol prepared-state file identity mismatch")


def verify_acceptance(output, *, expected_manifest_sha256=None, ledger=None,
                      ledger_sha256=None, run_id=None, required_evidence_revision=4,
                      expected_source_commit=None, source_repository=None,
                      expected_prepared_manifest_sha256=None):
    output = Path(output).resolve(strict=True)
    expected, entry, ledger_source_commit = _verification_anchor(
        output, expected_manifest_sha256, ledger, ledger_sha256, run_id,
    )
    if sha256(output / "ManifestV1.json") != expected:
        raise ValueError("bundle manifest differs from the supplied external anchor")
    completion, count = verify_output(output)
    if required_evidence_revision not in (1, 2, 3, 4):
        raise ValueError("unsupported externally required evidence revision")
    if required_evidence_revision == 4:
        if expected_source_commit is None:
            expected_source_commit = ledger_source_commit
        elif ledger_source_commit is not None and expected_source_commit != ledger_source_commit:
            raise ValueError("caller source commit disagrees with the anchored ledger source commit")
        if (entry is not None and "source_commit" in entry
                and entry["source_commit"] != expected_source_commit):
            raise ValueError("ledger entry source commit disagrees with the external source anchor")
        source_commit, committed_source = _source_anchor(expected_source_commit, source_repository)
    else:
        if expected_source_commit is not None or expected_prepared_manifest_sha256 is not None:
            raise ValueError("legacy acceptance does not implement source or preparation anchors; select revision four")
        source_commit = committed_source = None
    if completion["status"] == "passed":
        if completion.get("evidence_revision", 1) != required_evidence_revision:
            raise ValueError("bundle evidence revision differs from the externally required revision")
        if required_evidence_revision == 3:
            _verify_controlled_evidence(output, completion)
        elif required_evidence_revision == 4:
            execution = _verify_controlled_evidence(
                output, completion, expected_source_commit=source_commit,
                committed_source=committed_source, pin_contract=True,
            )
            _verify_preparation_chain(
                output, completion, execution, source_repository=source_repository,
                expected_prepared_manifest_sha256=expected_prepared_manifest_sha256,
            )
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
    limits = ["identity and declared execution checks; physics not rerun",
              "external anchors are not signatures or independent approval"]
    if required_evidence_revision < 4:
        limits.append("legacy revision: no external source commit, pinned contract, or preparation-chain acceptance")
    if completion["status"] == "failed":
        limits.append("failed record remains failed; successful-evidence structure not certified")
    return {**completion, "verification": {
        "required_evidence_revision": required_evidence_revision,
        "legacy": required_evidence_revision < 4,
        "source_commit_anchor": source_commit,
        "limits": limits,
    }}, count
