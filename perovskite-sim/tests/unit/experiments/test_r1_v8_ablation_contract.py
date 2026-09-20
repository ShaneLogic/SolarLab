"""Ablation contract, failure persistence, and local observer semantics."""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_r1_v8_ablation as ablation
from perovskite_sim.experiments import interface_defect_transient as transient
from perovskite_sim.experiments import one_dimensional_mechanism_r1_state as states


@pytest.mark.parametrize("times", [[0., float("nan")], [0., float("inf")], [0., 100.], [0.], [0., 1., .5]])
def test_invalid_or_out_of_bound_ablation_schedule_is_rejected(times):
    with pytest.raises(ValueError, match="bounded ablation"):
        ablation.validate_times(times)


def test_solver_attempt_capture_preserves_args_and_restores_after_failure(monkeypatch):
    calls = []
    def fail(*args, **kwargs):
        calls.append((args, kwargs))
        raise RuntimeError("controlled solver failure")
    monkeypatch.setattr(transient, "_solve_step", fail)
    monkeypatch.setattr(states, "snapshot", lambda system, previous: previous)
    record = {"accepted_steps": [{}, {}]}
    previous = {"precision_phi_V_hi": [.1], "precision_phi_V_lo": [1e-20]}
    system = SimpleNamespace(rebase_evidence={"reference_after": "quantized", "physical_previous_unchanged": True})
    coordinate = np.zeros(3)
    policy = object()
    with pytest.raises(RuntimeError, match="controlled solver"):
        with ablation.capture_solver_attempt(record, [0., 1e-9]):
            transient._solve_step(system, coordinate, previous, .005, 2.5e-10, policy, check_jacobian=False)
    assert transient._solve_step is fail
    assert calls[0][0][0] is system and calls[0][0][1] is coordinate and calls[0][0][2] is previous
    assert calls[0][0][-1] is policy
    assert record["last_solver_attempt"]["row_index"] == 2
    assert record["last_solver_attempt"]["physical_previous"] == previous
    assert record["last_solver_attempt"]["reference_quantization"] == system.rebase_evidence


def test_nonfinite_failed_witness_is_preserved_as_explicit_tagged_values():
    exc = ValueError("nonfinite failure")
    exc.result = {"residual": [float("inf"), float("nan")]}
    evidence = ablation.failure_record(exc, stage="integration")
    assert evidence["numerical_evidence"]["residual"] == [{"nonfinite_float": "inf"}, {"nonfinite_float": "nan"}]
    assert evidence["nonfinite_evidence_paths"] == ["numerical_evidence.residual[0]", "numerical_evidence.residual[1]"]
    json.dumps(evidence, allow_nan=False)


def test_io_or_unknown_execution_failure_is_not_a_scientific_ablation_outcome():
    for stage in ("observer_persistence", "accepted_state_diagnostics", "integration"):
        result = {"execution_status": "failed", "complete": False,
                  "failure": {"stage": stage, "type": "OSError", "numerical_evidence": {}}}
        assert not ablation.bounded_scientific_outcome(result)
    actual_numerical_failure = {"execution_status": "failed", "complete": False,
        "failure": {"stage": "integration", "original_numerical_solver_error": True,
                    "numerical_evidence": {"schema": "R1NewtonFailureWitnessV1"}}}
    assert ablation.bounded_scientific_outcome(actual_numerical_failure)


@pytest.mark.parametrize("stage", ["preparation", "replay"])
def test_runner_failure_retains_summary_manifest_and_existing_result(monkeypatch, tmp_path, stage):
    """Mock execution only to test error-artifact handling, not physical validity."""
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_checkout as checkout
    from perovskite_sim.experiments import one_dimensional_mechanism_r1_precision as precision
    import threadpoolctl
    flags = {key: getattr(ablation.sys.flags, key) for key in dir(ablation.sys.flags)
             if not key.startswith("_") and isinstance(getattr(ablation.sys.flags, key), int)}
    monkeypatch.setattr(ablation.sys, "flags", SimpleNamespace(**{**flags, "isolated": 1, "no_site": 1}))
    project = Path(__file__).resolve().parents[3]
    request = {"case": {"times_s": [0., 1e-9]}}
    request_file = tmp_path / "request.json"
    request_file.write_text(json.dumps(request))
    args = SimpleNamespace(output=tmp_path / "run", request_file=request_file,
        request_sha256=hashlib.sha256(request_file.read_bytes()).hexdigest(),
        expected_commit="a" * 40, source_sha256="b" * 64)
    monkeypatch.setattr(ablation.prototype, "load_contract", lambda *args: (request, request_file.read_bytes()))
    monkeypatch.setattr(ablation.prototype, "source_snapshot", lambda *args: {"source_commit": "a" * 40})
    monkeypatch.setattr(checkout, "require_r1_checkout", lambda **kwargs: SimpleNamespace(
        source_content_sha256="b" * 64, read_bytes=lambda path: (project / path).read_bytes()))
    monkeypatch.setattr(precision, "precision_context", nullcontext)
    monkeypatch.setattr(threadpoolctl, "threadpool_info", lambda: [{"num_threads": 1}])
    def fail(*args, **kwargs):
        raise RuntimeError("forced " + stage + " failure")
    if stage == "preparation":
        monkeypatch.setattr(states, "prepare_common_state", fail)
    else:
        monkeypatch.setattr(states, "prepare_common_state", lambda *args, **kwargs: SimpleNamespace(to_dict=lambda: {}))
        monkeypatch.setattr(ablation, "run_trace", lambda *args, **kwargs: {
            "execution_status": "completed", "complete": True, "intervention_rows": 3, "accepted_steps": []})
        monkeypatch.setattr(ablation, "replay_trace", fail)
    assert ablation.run(args) == 1
    summary = json.loads((args.output / "SummaryV1.json").read_text())
    assert not summary["diagnostic_evidence_complete"]
    assert summary["runner_failure"]["stage"] == stage
    manifest = json.loads((args.output / "ManifestV1.json").read_text())
    assert "SummaryV1.json" in manifest and "RunnerFailureV1.json" in manifest
    assert ("ResultV1.json" in manifest) is (stage == "replay")
