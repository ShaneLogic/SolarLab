"""R1-1 execution preserves prerequisite identities and failed-run evidence."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from perovskite_sim.models.config_loader import load_device_from_yaml
from tests.fixtures.r1_reference import (
    APPROVED_R1_BINDING_PATH, alternate_r1_binding, approved_r1_binding,
)
from tests.integration.test_one_dimensional_mechanism_r1 import FIXTURE


PROJECT = Path(__file__).resolve().parents[3]
SCRIPT = PROJECT / "scripts/run_one_dimensional_mechanism_r1_stage_one.py"


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(PROJECT / "scripts"))
    spec = importlib.util.spec_from_file_location("r1_stage_one_runner", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_list_is_read_only_and_scoped_to_r1_1(runner, tmp_path, capsys):
    output = tmp_path / "must_not_be_created"
    assert runner.main(["list", "--output-dir", str(output)]) == 0
    assert not output.exists()
    text = capsys.readouterr().out
    assert "Scope: R1-1 only" in text
    assert "convergence matrix belong to R1-2" in text


@pytest.mark.parametrize("arguments", [
    ["prepare"],
    ["zero-check", "--reference", "reference.json"],
    ["step", "--reference", "reference.json", "--prepared", "state.json"],
    ["step", "--reference", "reference.json", "--prepared", "state.json", "--control", "E"],
    ["prepare", "--reference", "reference.json", "--intervals", "4"],
])
def test_missing_or_out_of_scope_choices_fail_before_execution(runner, tmp_path, arguments):
    output = tmp_path / "not_started"
    with pytest.raises(SystemExit) as error:
        runner.main([*arguments, "--output-dir", str(output)])
    assert error.value.code == 2
    assert not output.exists()


def test_existing_evidence_is_never_overwritten(runner, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "CompletionV1.json"
    sentinel.write_bytes(b"original evidence\n")
    with pytest.raises(SystemExit) as error:
        runner.main(["prepare", "--reference", "reference.json", "--output-dir", str(output)])
    assert error.value.code == 2
    assert sentinel.read_bytes() == b"original evidence\n"
    assert sorted(path.name for path in output.iterdir()) == ["CompletionV1.json"]


@pytest.fixture
def failed_bundle(runner, tmp_path):
    output = tmp_path / "failed"
    output.mkdir()
    failure = {"type": "RuntimeError", "message": "deliberate test failure"}
    runner.write_json(output / "FailureV1.json", failure)
    runner.write_json(output / "CompletionV1.json", {
        "schema": "R1StageOneCompletionV1", "stage_scope": "R1-1", "stage": "step",
        "status": "failed", "failure": failure,
    })
    runner.manifest(output)
    return output


def test_verified_failed_bundle_remains_failed(runner, failed_bundle, capsys):
    completion, count = runner.verify_output(failed_bundle)
    assert completion["status"] == "failed"
    assert count == 2
    assert runner.main(["verify", "--output-dir", str(failed_bundle)]) == 1
    assert "recorded step status: failed" in capsys.readouterr().out


@pytest.mark.parametrize("mutation", ["modified", "missing", "extra", "escaped_path"])
def test_verify_rejects_incomplete_or_changed_artifacts(runner, failed_bundle, mutation, tmp_path):
    if mutation == "modified":
        (failed_bundle / "FailureV1.json").write_text("{}\n")
    elif mutation == "missing":
        (failed_bundle / "FailureV1.json").unlink()
    elif mutation == "extra":
        (failed_bundle / "UnsealedV1.json").write_text("{}\n")
    else:
        outside = tmp_path / "OutsideV1.json"
        outside.write_text("{}\n")
        entries = runner.read_json(failed_bundle / "ManifestV1.json")
        entries["../OutsideV1.json"] = {"bytes": outside.stat().st_size, "sha256": runner.sha256(outside)}
        runner.write_json(failed_bundle / "ManifestV1.json", entries)
    with pytest.raises(ValueError, match="manifest|identity mismatch"):
        runner.verify_output(failed_bundle)


def test_resealing_failure_as_passed_cannot_omit_required_evidence(runner, failed_bundle):
    completion = runner.read_json(failed_bundle / "CompletionV1.json")
    completion.update(status="passed", failure=None)
    runner.write_json(failed_bundle / "CompletionV1.json", completion)
    runner.manifest(failed_bundle)
    with pytest.raises(ValueError, match="required R1-1 evidence"):
        runner.verify_output(failed_bundle)


@pytest.mark.parametrize("text", ['{"value":NaN}', '{"value":1,"value":2}'])
def test_ambiguous_or_nonfinite_input_json_rejected(runner, tmp_path, text):
    path = tmp_path / "InputV1.json"
    path.write_text(text)
    with pytest.raises(ValueError, match="JSON"):
        runner.read_json(path)


def test_failed_numeric_result_remains_strict_json_and_lossless_npz(runner, tmp_path):
    original = np.array([1.0, np.nan, np.inf, -np.inf])
    runner._record_failure_result(tmp_path, {"state": original, "residual": float("nan")})
    saved = runner.read_json(tmp_path / "FailedResultV1.json")
    assert saved["residual"] == {"nonfinite": "nan"}
    assert saved["state"] == [1.0, {"nonfinite": "nan"}, {"nonfinite": "inf"}, {"nonfinite": "-inf"}]
    with np.load(tmp_path / "FailedResultV1.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["data.state"], original)


@pytest.mark.parametrize("stage", ["prepare", "zero-check", "step"])
@pytest.mark.parametrize("forged_hash", [False, True], ids=["resealed_alternate", "forged_approved_hash"])
def test_unapproved_reference_rejected_before_cli_dispatch(
    runner, tmp_path, monkeypatch, stage, forged_hash,
):
    import threadpoolctl
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_protocol as protocol
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_state as common
    from perovskite_sim.experiments.one_dimensional_mechanism_r1 import validate_binding

    alternate = alternate_r1_binding()
    validate_binding(alternate, load_device_from_yaml(FIXTURE))
    if forged_hash:
        alternate["sha256"] = approved_r1_binding()["sha256"]
    reference = tmp_path / "AlternateReferenceV1.json"
    runner.write_json(reference, alternate)
    external = {
        "schema": "R1CommonStateV1", "fixed_reference": alternate,
        "reference_sha256": alternate["sha256"],
    }
    external["sha256"] = common.digest(external)
    prepared = tmp_path / "ExternalStateV1.json"
    runner.write_json(prepared, external)

    def forbidden(*args, **kwargs):
        raise AssertionError("unapproved reference reached a CLI computation stage")

    monkeypatch.setattr(common, "prepare_common_state", forbidden)
    monkeypatch.setattr(protocol, "check_zero_excitation", forbidden)
    monkeypatch.setattr(protocol, "run_r1_step", forbidden)
    monkeypatch.setattr(runner, "_record_execution_source", lambda output: None)
    # This unit case isolates admission from host BLAS availability. The real
    # subprocess workflow below observes and verifies the actual backend.
    monkeypatch.setattr(threadpoolctl, "threadpool_info", lambda: [{"num_threads": 1, "source": "test-double"}])
    output = tmp_path / "rejected"
    arguments = [stage, "--reference", str(reference), "--output-dir", str(output)]
    if stage != "prepare":
        arguments += ["--prepared", str(prepared)]
    if stage == "step":
        arguments += ["--control", "D"]
    assert runner.main(arguments) == 1
    failure = runner.read_json(output / "FailureV1.json")
    expected = "reference identity mismatch" if forged_hash else "not the approved study reference"
    assert expected in failure["message"]
    assert runner.verify_output(output)[0]["status"] == "failed"
    assert not (output / "StepResultV1.json").exists()


def test_caller_input_cannot_repin_approved_reference(runner, tmp_path, monkeypatch):
    alternate = alternate_r1_binding()
    reference = tmp_path / "AlternateReferenceV1.json"
    runner.write_json(reference, alternate)
    overridden = runner.read_json(runner.INPUT_PATH)
    overridden["fixed_reference_binding_sha256"] = alternate["sha256"]
    input_path = tmp_path / "AlternateInputV1.json"
    runner.write_json(input_path, overridden)
    monkeypatch.setattr(runner, "_record_execution_source", lambda output: None)
    output = tmp_path / "rejected"
    assert runner.main([
        "prepare", "--reference", str(reference), "--input", str(input_path),
        "--output-dir", str(output),
    ]) == 1
    failure = runner.read_json(output / "FailureV1.json")
    assert "input differs from the supported versioned protocol" in failure["message"]
    assert runner.verify_output(output)[0]["status"] == "failed"


def test_failure_preserves_accepted_observations_and_exception_result(runner, tmp_path, monkeypatch):
    import threadpoolctl
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_protocol as protocol
    from perovskite_sim.experiments.one_dimensional_mechanism_r1_state import digest

    reference = tmp_path / "ReferenceBindingV1.json"
    runner.write_json(reference, approved_r1_binding())
    # The runner only decodes the content hash; the production protocol owns
    # live physical validation. This test injects a failure at that boundary.
    prepared = {"schema": "R1CommonStateV1"}
    prepared["sha256"] = digest(prepared)
    prepared_path = tmp_path / "PreparedStateV1.json"
    runner.write_json(prepared_path, prepared)
    accepted = {"phase": "accepted_regular_step", "time_s": 1e-9, "state": {"n_m3": [1.0]}}

    def fail_after_acceptance(*args, accepted_step_observer, **kwargs):
        accepted_step_observer(accepted)
        raise protocol.R1RunError("injected gate failure", {
            "accepted_steps": [accepted], "residual": np.array([np.nan]),
            "certificate": {"certified": False, "reasons": ["injected_gate"]},
        })

    monkeypatch.setattr(protocol, "run_r1_step", fail_after_acceptance)
    monkeypatch.setattr(runner, "_record_execution_source", lambda output: None)
    # Isolate the failure writer from host BLAS availability; real CLI
    # integration below independently observes the actual backend.
    monkeypatch.setattr(threadpoolctl, "threadpool_info", lambda: [{"num_threads": 1, "source": "test-double"}])
    output = tmp_path / "interrupted"
    assert runner.main([
        "step", "--control", "D", "--reference", str(reference), "--prepared", str(prepared_path),
        "--output-dir", str(output),
    ]) == 1
    assert runner.read_json(output / "AcceptedStepsV1.json") == [accepted]
    failed = runner.read_json(output / "FailedResultV1.json")
    assert failed["certificate"]["reasons"] == ["injected_gate"]
    assert runner.verify_output(output)[0]["status"] == "failed"
    assert (output / "ReferenceBindingV1.json").read_bytes() == reference.read_bytes()
    assert (output / "PreparedStateV1.json").read_bytes() == prepared_path.read_bytes()


@pytest.mark.slow
def test_real_cli_prepare_zero_step_and_verify_share_one_state(tmp_path):
    reference = tmp_path / "ReferenceBindingV1.json"
    reference.write_text(json.dumps(approved_r1_binding(), separators=(",", ":")))
    assert reference.read_bytes() != APPROVED_R1_BINDING_PATH.read_bytes()
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(PROJECT))
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        environment[key] = "1"

    def execute(stage, output, *arguments):
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), stage, "--output-dir", str(output), *arguments],
            cwd=PROJECT, env=environment, capture_output=True, text=True, timeout=120,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr

    preparation = tmp_path / "preparation"
    execute("prepare", preparation, "--reference", str(reference), "--intervals", "16")
    prepared_path = preparation / "PreparedStateV1.json"
    prepared = json.loads(prepared_path.read_text())
    arguments = ["--reference", str(reference), "--prepared", str(prepared_path), "--intervals", "16"]
    zero = tmp_path / "zero"
    execute("zero-check", zero, *arguments)
    zero_result = json.loads((zero / "ZeroExcitationV1.json").read_text())
    assert zero_result["prepared_sha256"] == prepared["sha256"]
    assert set(zero_result["controls"]) == set("ABCD")
    assert zero_result["certificate"]["certified"]
    assert not zero_result["certificate"]["relative_dynamic_current_certified"]
    step = tmp_path / "step"
    execute("step", step, *arguments, "--control", "D", "--amplitude", "0.005")
    result = json.loads((step / "StepResultV1.json").read_text())
    assert result["prepared_sha256"] == prepared["sha256"]
    assert result["certificate"]["certified"]
    step_protocol = json.loads((step / "ProtocolV1.json").read_text())
    assert step_protocol["approved_reference_binding_sha256"] == approved_r1_binding()["sha256"]
    assert step_protocol["reference_binding_sha256"] == step_protocol["approved_reference_binding_sha256"]
    assert result["initial_event"]["impulse_charge_C_m2"] == pytest.approx(2.21354695425e-6, rel=1e-12)
    assert len(result["accepted_steps"]) == 31
    assert json.loads((step / "AcceptedStepsV1.json").read_text()) == result["accepted_steps"]
    with np.load(step / "StepResultV1.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["data.output_states.positive_m3"], result["output_states"]["positive_m3"])
    for output in (preparation, zero, step):
        assert (output / "ReferenceBindingV1.json").read_bytes() == reference.read_bytes()
        assert (output / "PreparedStateV1.json").read_bytes() == prepared_path.read_bytes()
        execute("verify", output)
