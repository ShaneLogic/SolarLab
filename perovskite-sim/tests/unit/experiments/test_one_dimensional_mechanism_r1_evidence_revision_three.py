"""Externally selected source-evidence format and explicit development status."""

import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from tests.fixtures.r1_reference import approved_r1_binding
from tests.unit.experiments.test_one_dimensional_mechanism_r1_stage_one_cli import runner
from tests.unit.experiments.test_one_dimensional_mechanism_r1_controlled import (
    repository, launch, commit, INPUT, PROJECT,
)


@pytest.fixture
def controlled_bundle(runner, repository, tmp_path):
    """A real controlled source snapshot with synthetic, non-physical records.

    The externally anchored verifier checks identity and declared structure;
    this fixture deliberately makes no assertion that it ran a simulation.
    """
    root, project, _ = repository
    fixture = "tests/fixtures/configs/dynamic_interface_defect_ion_transient_absorber_only.yaml"
    (project / fixture).write_bytes((PROJECT / fixture).read_bytes())
    repository = root, project, commit(root)
    output = tmp_path / "controlled_bundle"
    result = launch(repository, "--output", str(output))
    assert result.returncode == 0, result.stderr
    for name in runner.COMMON_ARTIFACTS:
        if not (output / name).exists():
            runner.write_json(output / name, {})
    (output / "StudyInputV1.json").write_bytes((project / INPUT).read_bytes())
    (output / "ExecutionContractV1.md").write_bytes((project / "docs/OneDimensionalMechanismR1DynamicsV1.md").read_bytes())
    (output / "SourceFixtureV1.yaml").write_bytes((project / fixture).read_bytes())
    binding = approved_r1_binding()
    runner.write_json(output / "ReferenceBindingV1.json", binding)
    runner.write_json(output / "ProtocolV1.json", {
        "stage_scope": "R1-1", "stage": "prepare",
        "input_sha256": runner.sha256(output / "StudyInputV1.json"),
        "contract_sha256": runner.sha256(output / "ExecutionContractV1.md"),
        "reference_file_sha256": runner.sha256(output / "ReferenceBindingV1.json"),
        "reference_binding_sha256": binding["sha256"],
        "approved_reference_binding_sha256": binding["sha256"],
    })
    runner.write_json(output / "CompletionV1.json", {
        "schema": "R1StageOneCompletionV1", "stage_scope": "R1-1", "stage": "prepare",
        "status": "passed", "failure": None, "evidence_revision": 3, "run_class": "formal",
    })
    runner.manifest(output)
    return output


def accept(runner, output):
    return runner.verify_acceptance(
        output, expected_manifest_sha256=runner.sha256(output / "ManifestV1.json"),
        required_evidence_revision=3,
    )


def test_controlled_revision_three_matches_external_anchor(runner, controlled_bundle):
    assert accept(runner, controlled_bundle)[0]["status"] == "passed"


@pytest.mark.parametrize("revision", [None, 1, 2])
def test_bundle_cannot_choose_its_own_weaker_evidence_contract(runner, controlled_bundle, revision):
    completion = runner.read_json(controlled_bundle / "CompletionV1.json")
    if revision is None:
        completion.pop("evidence_revision")
    else:
        completion["evidence_revision"] = revision
    runner.write_json(controlled_bundle / "CompletionV1.json", completion)
    (controlled_bundle / "ExecutionSourceV1.json").unlink()
    runner.manifest(controlled_bundle)
    with pytest.raises(ValueError, match="revision|execution checkout"):
        accept(runner, controlled_bundle)


@pytest.mark.parametrize("mutation", ["empty_zip", "unrelated_manifest", "empty_execution",
                                      "missing_package", "wrong_source_digest", "development",
                                      "cached_loader", "wrong_reference", "payload_forgery", "changed_fixture"])
def test_resealed_structure_cannot_satisfy_strict_source_contract(runner, controlled_bundle, mutation):
    output = controlled_bundle
    if mutation == "empty_zip":
        with zipfile.ZipFile(output / "SourceV1.zip", "w"):
            pass
    elif mutation == "unrelated_manifest":
        runner.write_json(output / "SourceManifestV1.json", {"unrelated.py": {"bytes": 0, "sha256": "0"*64}})
    elif mutation == "wrong_reference":
        protocol = runner.read_json(output / "ProtocolV1.json")
        protocol["approved_reference_binding_sha256"] = "0" * 64
        runner.write_json(output / "ProtocolV1.json", protocol)
    elif mutation == "payload_forgery":
        binding = runner.read_json(output / "ReferenceBindingV1.json")
        binding["f_ref"][0] += 5e-5
        runner.write_json(output / "ReferenceBindingV1.json", binding)
    elif mutation == "changed_fixture":
        (output / "SourceFixtureV1.yaml").write_text("modified: true\n")
    else:
        execution = runner.read_json(output / "ExecutionSourceV1.json")
        if mutation == "empty_execution":
            execution = {}
        elif mutation == "missing_package":
            execution["required_sources"].pop("perovskite-sim/perovskite_sim/probe.py")
        elif mutation == "wrong_source_digest":
            execution["source_content_sha256"] = "0" * 64
        elif mutation == "development":
            execution["run_class"] = "development"
        elif mutation == "cached_loader":
            execution["runtime"]["project_bytecode_cache_used"] = True
        runner.write_json(output / "ExecutionSourceV1.json", execution)
    runner.manifest(output)
    with pytest.raises(ValueError, match="source|controlled|reference"):
        accept(runner, output)


def test_direct_computation_requires_explicit_development(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_record_execution_source", lambda path: None)
    output = tmp_path / "rejected"
    assert runner.main(["prepare", "--reference", "unused.json", "--output-dir", str(output)]) == 1
    assert "controlled -I -S launcher" in runner.read_json(output / "FailureV1.json")["message"]
    assert runner.verify_output(output)[0]["status"] == "failed"


def test_historical_failure_is_reported_without_waiving_acceptance(runner):
    study = runner.read_json(runner.INPUT_PATH)
    args = SimpleNamespace(stage="step", intervals=64, control="D", nonlinear_factor=1., amplitude=.005)
    policy = SimpleNamespace(refinement_substeps=(1, 2, 4))
    failure = {"message": "contact_internal_current_spread_relative_exceeds_limit"}
    report = runner._historical_case_observation(study, args, policy, failure)
    assert report["waives_checks"] is False
    assert report["matching_historical_cases"][0]["category"] == "known_physical_gate_failures"
    assert report["matching_historical_cases"][0]["outcome"] == "historical_signature_recurred"
    assert runner._historical_case_observation(study, args, policy, None)["matching_historical_cases"][0]["outcome"] == "previously_failed_case_now_passed"
    policy.refinement_substeps = (2, 4, 8)
    assert runner._historical_case_observation(study, args, policy, failure)["matching_historical_cases"] == []


def test_complex_nonfinite_failure_keeps_components_and_raw_npz(runner, tmp_path):
    value = np.complex128(complex(float("nan"), float("inf")))
    runner._record_failure_result(tmp_path, {"probe": value})
    assert runner.read_json(tmp_path / "FailedResultV1.json")["probe"] == {
        "real": {"nonfinite": "nan"}, "imag": {"nonfinite": "inf"}}
    with np.load(tmp_path / "FailedResultV1.npz", allow_pickle=False) as archive:
        assert np.isnan(archive["data.probe"].real) and np.isposinf(archive["data.probe"].imag)
