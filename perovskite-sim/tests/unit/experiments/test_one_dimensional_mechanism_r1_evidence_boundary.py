"""Externally anchored identity and complete, nonfinite-safe failure streams."""

import json
import subprocess

import numpy as np
import pytest

from tests.fixtures.r1_reference import approved_r1_binding
from tests.unit.experiments.test_one_dimensional_mechanism_r1_stage_one_cli import runner


def test_capture_and_commit_use_same_checkout_despite_git_environment(runner, tmp_path, monkeypatch):
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_checkout import git_environment
    expected_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=runner.PROJECT,
                                              env=git_environment(), text=True).strip()
    expected_patch = subprocess.check_output(["git", "diff", "HEAD"], cwd=runner.PROJECT,
                                             env=git_environment())
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "unrelated.git"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(tmp_path / "unrelated-index"))
    output = tmp_path / "capture"
    output.mkdir()
    context = runner._record_execution_source(output)
    assert context.commit == expected_commit
    assert (output / "SourceChangesV1.patch").read_bytes() == expected_patch
    assert runner.read_json(output / "ExecutionSourceV1.json")["observed_commit"] == expected_commit
    assert runner.read_json(output / "SourceManifestV1.json")


@pytest.fixture
def bundle(runner, tmp_path):
    # Structurally complete evidence is not itself a physical calculation.
    # These tests deliberately separate checksum/anchor matching from physics.
    output = tmp_path / "bundle"
    output.mkdir()
    for name in runner.COMMON_ARTIFACTS:
        (output / name).write_text("{}\n")
    runner.write_json(output / "CompletionV1.json", {
        "schema": "R1StageOneCompletionV1", "stage_scope": "R1-1", "stage": "prepare",
        "status": "passed", "failure": None,
    })
    runner.write_json(output / "SourceManifestV1.json", {"example.py": {"sha256": "0"*64, "bytes": 1}})
    runner.write_json(output / "ReferenceBindingV1.json", approved_r1_binding())
    runner.manifest(output)
    return output


def ledger_for(runner, bundle, destination):
    entry = {
        "manifest_sha256": runner.sha256(bundle / "ManifestV1.json"),
        "stage": "prepare", "recorded_status": "passed", "stage_scope": "R1-1",
        "source_manifest_sha256": runner.sha256(bundle / "SourceManifestV1.json"),
        "study_input_sha256": runner.sha256(bundle / "StudyInputV1.json"),
        "reference_binding_sha256": approved_r1_binding()["sha256"],
    }
    runner.write_json(destination, {"schema": "R1EvidenceLedgerV1", "entries": {"run-one": entry}})
    return destination, runner.sha256(destination)


def test_integrity_does_not_claim_authenticated_provenance(runner, bundle, capsys):
    assert runner.main(["verify", "--output-dir", str(bundle)]) == 0
    output = capsys.readouterr().out
    assert "checksums consistent" in output
    assert "provenance not authenticated" in output
    assert "artifact identities" not in output


def test_resealed_changes_fail_against_prior_external_digest(runner, bundle):
    expected = runner.sha256(bundle / "ManifestV1.json")
    assert runner.verify_acceptance(bundle, expected_manifest_sha256=expected)[0]["status"] == "passed"
    runner.write_json(bundle / "PreparedStateV1.json", {"sheet_charge_C_m2": [1.2345e-9]})
    runner.manifest(bundle)
    assert runner.verify_output(bundle)[0]["status"] == "passed"
    with pytest.raises(ValueError, match="external anchor"):
        runner.verify_acceptance(bundle, expected_manifest_sha256=expected)


def test_acceptance_requires_explicit_external_anchor(runner, bundle, capsys):
    assert runner.main(["verify", "--mode", "acceptance", "--output-dir", str(bundle)]) == 1
    assert "requires an external" in capsys.readouterr().err


def test_ledger_identity_requires_separately_held_digest(runner, bundle, tmp_path):
    path, digest = ledger_for(runner, bundle, tmp_path / "LedgerV1.json")
    assert runner.verify_acceptance(bundle, ledger=path, ledger_sha256=digest, run_id="run-one")[0]["status"] == "passed"
    ledger = runner.read_json(path)
    ledger["entries"]["run-one"]["manifest_sha256"] = "0" * 64
    runner.write_json(path, ledger)
    with pytest.raises(ValueError, match="ledger digest mismatch"):
        runner.verify_acceptance(bundle, ledger=path, ledger_sha256=digest, run_id="run-one")


def test_bundled_ledger_cannot_authorize_itself(runner, bundle):
    path, digest = ledger_for(runner, bundle, bundle / "LedgerV1.json")
    with pytest.raises(ValueError, match="outside the bundle"):
        runner.verify_acceptance(bundle, ledger=path, ledger_sha256=digest, run_id="run-one")


@pytest.mark.parametrize("key,value", [("stage", "step"), ("recorded_status", "failed"),
                                       ("reference_binding_sha256", "0"*64)])
def test_matching_manifest_cannot_hide_ledger_identity_disagreement(runner, bundle, tmp_path, key, value):
    path, _ = ledger_for(runner, bundle, tmp_path / "LedgerV1.json")
    ledger = runner.read_json(path)
    ledger["entries"]["run-one"][key] = value
    runner.write_json(path, ledger)
    with pytest.raises(ValueError, match="identities disagree"):
        runner.verify_acceptance(bundle, ledger=path, ledger_sha256=runner.sha256(path), run_id="run-one")


def test_empty_source_manifest_cannot_pass_structural_verification(runner, bundle):
    runner.write_json(bundle / "SourceManifestV1.json", {})
    runner.manifest(bundle)
    with pytest.raises(ValueError, match="empty source"):
        runner.verify_output(bundle)


def test_atomic_evidence_failure_preserves_previous_json_and_removes_temporary_files(runner, tmp_path, monkeypatch):
    path = tmp_path / "AcceptedStepsV1.json"
    runner._write_evidence(path, [{"old": 1}])
    def fail_replace(*args):
        raise OSError("injected persistence failure")
    monkeypatch.setattr(runner.os, "replace", fail_replace)
    with pytest.raises(OSError, match="persistence"):
        runner._write_evidence(path, [{"new": 2}])
    assert runner.read_json(path) == [{"old": 1}]
    assert not list(tmp_path.glob(".r1-*"))


@pytest.mark.parametrize("write_failure", [False, True])
def test_failed_row_stream_counts_and_physical_reason_survive(runner, tmp_path, monkeypatch, write_failure):
    import threadpoolctl
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_protocol as protocol
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import digest

    reference = tmp_path / "ReferenceBindingV1.json"
    runner.write_json(reference, approved_r1_binding())
    prepared = {"schema": "R1CommonStateV1"}
    prepared["sha256"] = digest(prepared)
    state = tmp_path / "PreparedStateV1.json"
    runner.write_json(state, prepared)
    rows = [
        {"phase": "0+", "solver_accepted": False, "physical_checks_passed": True},
        {"phase": "accepted_regular_step", "solver_accepted": True, "physical_checks_passed": False,
         "physical": {"gauss_normalized": float("nan")},
         "physical_checks": {"reasons": ["gauss_normalized"]}},
    ]
    original_writer = runner._write_evidence
    def writing(path, value):
        if write_failure and path.name == "AcceptedStepsV1.json" and len(value) == 2:
            raise OSError("injected stream failure")
        return original_writer(path, value)
    monkeypatch.setattr(runner, "_write_evidence", writing)
    def fail_run(*args, accepted_step_observer, **kwargs):
        accepted_step_observer(rows[0])
        result = {"accepted_steps": rows, "certificate": {"certified": False, "reasons": ["gauss_normalized"]}}
        try:
            accepted_step_observer(rows[1])
        except OSError as exc:
            result["persistence_failure"] = {"message": str(exc)}
        raise protocol.R1RunError("gauss_normalized", result)
    monkeypatch.setattr(protocol, "run_r1_step", fail_run)
    monkeypatch.setattr(runner, "_record_execution_source", lambda output: None)
    monkeypatch.setattr(threadpoolctl, "threadpool_info", lambda: [{"num_threads": 1}])
    output = tmp_path / "run"
    assert runner.main(["step", "--control", "D", "--reference", str(reference),
                        "--prepared", str(state), "--output-dir", str(output)]) == 1
    completion = runner.read_json(output / "CompletionV1.json")
    streamed = runner.read_json(output / "AcceptedStepsV1.json")
    failed = runner.read_json(output / "FailedResultV1.json")
    assert completion["observed_record_count"] == 2
    assert completion["persisted_record_count"] == completion["accepted_record_count"] == len(streamed)
    assert len(streamed) == (1 if write_failure else 2)
    assert failed["certificate"]["reasons"] == ["gauss_normalized"]
    assert failed["accepted_steps"][1]["physical"]["gauss_normalized"] == {"nonfinite": "nan"}
    assert "gauss_normalized" in runner.read_json(output / "FailureV1.json")["message"]
    if write_failure:
        assert "stream failure" in failed["persistence_failure"]["message"]
    else:
        assert streamed == failed["accepted_steps"]
    with np.load(output / "FailedResultV1.npz", allow_pickle=False) as arrays:
        assert np.isnan(arrays["data.accepted_steps.1.physical.gauss_normalized"])
