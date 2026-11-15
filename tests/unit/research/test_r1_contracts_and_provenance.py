"""Unit tests for ``solarlab_research.r1`` contracts/provenance and CLI scripts (P07-01, P07-02).

Covers:
* Frozen 16-item V53 certificate limits, solver budgets, relative engineering limits,
  controls ``A-D``, and 13-channel operator schema validator.
* 12-per-decade observation grid and conservative ``(4, 8, 16)`` substep row counts
  (``3727`` on full 100 s window; ``59`` on 2-interval short window).
* ``R1StudyRequest`` hashing/tamper rejection and strict IDA ``R1-L2`` exclusion (``P07-02.4``)
  plus short-window non-qualification.
* Immutable ``ReleaseBundle`` provenance verification (``scientific_qualification=False``)
  and materialized bundle directory roundtrip.
* Non-relabeling historical readers for ``R1CommonStateV1``, ``R1CommonStatePairV2``,
  and ``R1FailureWitnessV1.json`` (preserving ``step309_failed`` status).
* Module CLI entrypoints under ``solarlab_research.scripts``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from solarlab.reproducibility.release import write_release_bundle

from solarlab_research.r1 import (
    R1_CONSERVATIVE_SUBSTEPS,
    R1_CONTROLS,
    R1_FULL_WINDOW_EXPECTED_ROWS,
    R1_OPERATOR_CHANNELS,
    R1_RELATIVE_ENGINEERING_LIMITS,
    R1_SOLVER_BUDGETS,
    R1_V53_CERTIFICATE_LIMITS,
    assess_r1_qualification,
    build_r1_study_request,
    evaluate_v53_certificate,
    expected_substep_rows,
    load_r1_study_request,
    load_verified_release_bundle,
    observation_times_12_per_decade,
    read_historical_r1_artifact,
    synthetic_research_release_bundle,
    validate_operator_channel_packet,
    verify_historical_r1_failure_witness,
    verify_r1_release_source,
    verify_r1_study_request,
)
from solarlab_research.scripts import (
    cbo_scan as cbo_scan_cli,
    r1_short_window as r1_short_window_cli,
    r1_verify_sources as r1_verify_sources_cli,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
HIST_COMMON_V1 = REPO_ROOT / "tests" / "refactor" / "fixtures" / "r1_common_state_preparation.json"
HIST_PAIR_V2 = (
    REPO_ROOT / "tests" / "refactor" / "fixtures" / "r1_pair_common_state_preparation.json"
)
HIST_WITNESS = REPO_ROOT / "tests" / "fixtures" / "refactor" / "R1FailureWitnessV1.json"


def test_frozen_v53_limits_channels_and_observation_row_counts() -> None:
    assert len(R1_V53_CERTIFICATE_LIMITS) == 16
    assert R1_V53_CERTIFICATE_LIMITS["nonlinear_residual"] == 0.05
    assert R1_V53_CERTIFICATE_LIMITS["analytic_jacobian_error"] == 3.0e-4
    assert len(R1_OPERATOR_CHANNELS) == 13
    assert set(R1_CONTROLS) == {"A", "B", "C", "D"}
    assert R1_CONTROLS["A"] == {"nu_I": 0, "nu_t": 0}
    assert R1_CONTROLS["D"] == {"nu_I": 1, "nu_t": 1}
    assert R1_SOLVER_BUDGETS["maximum_newton_iterations"] == 100
    assert R1_RELATIVE_ENGINEERING_LIMITS["elapsed_ratio"] == 5.0

    full_times = observation_times_12_per_decade()
    assert full_times[0] == 0.0
    assert full_times[1] == 1.0e-9
    assert full_times[-1] == 100.0
    assert len(full_times) == 134  # 0.0 + 133 log points across 11 decades
    assert expected_substep_rows(full_times, R1_CONSERVATIVE_SUBSTEPS) == R1_FULL_WINDOW_EXPECTED_ROWS

    short_times = (0.0, 1.0e-9, 1.0e-8)
    assert expected_substep_rows(short_times, R1_CONSERVATIVE_SUBSTEPS) == 59

    passing_metrics = {key: lim * 0.5 for key, lim in R1_V53_CERTIFICATE_LIMITS.items()}
    cert = evaluate_v53_certificate(passing_metrics)
    assert cert.passed is True
    assert cert.failed_items == ()

    failing_metrics = {**passing_metrics, "nonlinear_residual": 0.0532305}
    cert_fail = evaluate_v53_certificate(failing_metrics)
    assert cert_fail.passed is False
    assert cert_fail.failed_items == ("nonlinear_residual",)

    valid_packet: dict[str, object] = {ch: np.zeros((59, 2), dtype=float) for ch in R1_OPERATOR_CHANNELS}
    shapes = validate_operator_channel_packet(valid_packet, expected_rows=59)
    assert len(shapes) == 13
    assert all(sh == (59, 2) for sh in shapes.values())

    with pytest.raises(ValueError, match="operator packet mismatch"):
        validate_operator_channel_packet({k: v for k, v in valid_packet.items() if k != "sheet_charge"})
    with pytest.raises(ValueError, match="row count"):
        validate_operator_channel_packet(valid_packet, expected_rows=60)


def test_study_request_and_ida_r1_l2_exclusion() -> None:
    bundle = synthetic_research_release_bundle()
    prep_sha = "a" * 64
    short_req = build_r1_study_request(
        study_id="short_test",
        control="D",
        times_s=(0.0, 1.0e-9, 1.0e-8),
        release_bundle_sha256=bundle.sha256,
        prepared_device_sha256=prep_sha,
    )
    raw = json.dumps(short_req.to_mapping())
    reloaded = load_r1_study_request(raw)
    assert reloaded == short_req

    tampered = short_req.to_mapping()
    tampered["voltage_step_V"] = 0.01
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_r1_study_request(tampered)

    passing_cert = evaluate_v53_certificate({k: v * 0.1 for k, v in R1_V53_CERTIFICATE_LIMITS.items()})

    # Short-window run passes interface verification ONLY; never claims R1-L2 or R1-L3
    short_qual = assess_r1_qualification(
        short_req,
        baseline_certificate=passing_cert,
        compensated_certificate=passing_cert,
        baseline_rows=59,
        compensated_rows=59,
        interpolated_rows=False,
        release_verified=True,
    )
    assert short_qual.qualification_level == "short_window_interface_verification_only"
    assert short_qual.short_window_interface_verified is True
    assert short_qual.r1_l2_certified is False
    assert short_qual.r1_l3_device_qualified is False

    # P07-02.4: IDA integrator on full 100 s window is strictly excluded from R1-L2
    full_times = observation_times_12_per_decade()
    ida_req = build_r1_study_request(
        study_id="ida_comparison",
        control="D",
        times_s=full_times,
        integrator="ida",
        release_bundle_sha256=bundle.sha256,
        prepared_device_sha256=prep_sha,
    )
    ida_qual = assess_r1_qualification(
        ida_req,
        baseline_certificate=passing_cert,
        compensated_certificate=passing_cert,
        baseline_rows=R1_FULL_WINDOW_EXPECTED_ROWS,
        compensated_rows=R1_FULL_WINDOW_EXPECTED_ROWS,
        interpolated_rows=False,
        release_verified=True,
    )
    assert ida_qual.qualification_level == "unqualified"
    assert ida_qual.r1_l2_certified is False
    assert ida_qual.r1_l3_device_qualified is False
    assert any("P07-02.4" in reason for reason in ida_qual.reasons)

    # Interpolated rows are also rejected even for conservative_backward_euler
    be_full_req = build_r1_study_request(
        study_id="be_full",
        control="D",
        times_s=full_times,
        integrator="conservative_backward_euler",
        release_bundle_sha256=bundle.sha256,
        prepared_device_sha256=prep_sha,
    )
    interp_qual = assess_r1_qualification(
        be_full_req,
        baseline_certificate=passing_cert,
        compensated_certificate=passing_cert,
        baseline_rows=R1_FULL_WINDOW_EXPECTED_ROWS,
        compensated_rows=R1_FULL_WINDOW_EXPECTED_ROWS,
        interpolated_rows=True,
        release_verified=True,
    )
    assert interp_qual.r1_l2_certified is False
    assert any("interpolated" in reason for reason in interp_qual.reasons)


def test_release_bundle_provenance_and_non_relabeling_historical_readers(tmp_path: Path) -> None:
    bundle = synthetic_research_release_bundle(
        build_source_commit="19c19776ff0b13beb791683697035594718ff317"
    )
    bundle_dir = tmp_path / "release_bundle"
    write_release_bundle(bundle, bundle_dir)

    loaded = load_verified_release_bundle(bundle_dir, expected_bundle_sha256=bundle.sha256)
    binding = verify_r1_release_source(
        loaded,
        expected_bundle_sha256=bundle.sha256,
        inputs={"study": "provenance_test"},
    )
    assert binding.build_source_commit == "19c19776ff0b13beb791683697035594718ff317"
    assert binding.scientific_qualification is False
    assert (
        binding.job_provenance_envelope["record"]["git_commit"]
        == "19c19776ff0b13beb791683697035594718ff317"
    )
    assert "scientific_qualification" not in binding.job_provenance_envelope
    assert "scientific_qualification" not in binding.job_provenance_envelope["record"]

    with pytest.raises(ValueError, match="differs from expected"):
        verify_r1_release_source(loaded, expected_bundle_sha256="f" * 64, inputs={})

    # Non-relabeling historical readers
    s_v1 = read_historical_r1_artifact(HIST_COMMON_V1)
    assert s_v1.schema == "R1CommonStateV1"
    assert s_v1.relabelled is False
    assert s_v1.metadata["intervals"] == 4
    assert s_v1.metadata["node_count"] == 4

    s_v2 = read_historical_r1_artifact(HIST_PAIR_V2)
    assert s_v2.schema == "R1CommonStatePairV2"
    assert s_v2.relabelled is False
    assert s_v2.metadata["decoded_field_count"] == 17
    expected_seed_sha = json.loads(HIST_PAIR_V2.read_text(encoding="utf-8"))["seed_preparation"]["sha256"]
    assert s_v2.metadata["seed_preparation_sha256"] == expected_seed_sha

    witness_audit = verify_historical_r1_failure_witness(HIST_WITNESS)
    assert witness_audit.relabelled is False
    assert witness_audit.step183_accepted_status == "accepted"
    assert witness_audit.step183_failed_status == "failed"
    assert witness_audit.step309_original_status == "failed"
    assert witness_audit.step309_failure_type == "InterfaceDefectTransientError"
    assert witness_audit.step309_nonlinear_residual == pytest.approx(0.0532305)
    assert witness_audit.step309_dt_s == pytest.approx(0.00011393063141266287)

    # Attempt to relabel historical step309 residual as passing (<= 0.05) must fail
    tampered_witness = json.loads(HIST_WITNESS.read_text(encoding="utf-8"))
    tampered_witness["cases"]["step309_failed"]["nonlinear_residual"] = 0.01
    with pytest.raises(ValueError, match="cannot be relabeled as passing"):
        verify_historical_r1_failure_witness(json.dumps(tampered_witness))


def test_module_cli_scripts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # 1. r1_verify_sources CLI
    bundle = synthetic_research_release_bundle()
    bundle_dir = tmp_path / "cli_bundle"
    write_release_bundle(bundle, bundle_dir)

    rc = r1_verify_sources_cli.main([
        "--historical",
        str(HIST_COMMON_V1),
        "--historical",
        str(HIST_PAIR_V2),
        "--witness",
        str(HIST_WITNESS),
        "--bundle-dir",
        str(bundle_dir),
        "--expected-bundle-sha256",
        bundle.sha256,
    ])
    assert rc == 0
    out_sources = json.loads(capsys.readouterr().out)
    assert out_sources["schema"] == "solarlab_research.r1_verify_sources_cli.v1"
    assert len(out_sources["historical_artifacts"]) == 2
    assert out_sources["witness_audit"]["step309_original_status"] == "failed"
    assert out_sources["witness_audit"]["relabelled"] is False
    assert out_sources["release_bundle"]["scientific_qualification"] is False

    # 2. cbo_scan CLI (coarse 2-point sweep for fast unit execution)
    rc_cbo = cbo_scan_cli.main([
        "--delta-ec",
        "0.0",
        "0.1",
        "--intervals",
        "4",
        "6",
        "4",
        "--voltage-points",
        "7",
    ])
    assert rc_cbo == 0
    out_cbo = json.loads(capsys.readouterr().out)
    assert out_cbo["schema"] == "solarlab_research.cbo_scan_cli.v1"
    assert out_cbo["complete"] is True
    assert len(out_cbo["points"]) == 2

    # 3. r1_short_window CLI
    rc_r1 = r1_short_window_cli.main([
        "--control",
        "D",
        "--intervals",
        "3",
        "2",
        "--bundle-dir",
        str(bundle_dir),
        "--expected-bundle-sha256",
        bundle.sha256,
    ])
    assert rc_r1 == 0
    out_r1 = json.loads(capsys.readouterr().out)
    assert out_r1["schema"] == "solarlab_research.r1_short_window_cli.v1"
    assert out_r1["baseline_passed"] is True
    assert out_r1["compensated_passed"] is True
    assert out_r1["baseline_rows"] == 59
    assert out_r1["compensated_rows"] == 59
    assert out_r1["deterministic_replay_match"] is True
    assert out_r1["r1_l2_certified"] is False
    assert out_r1["r1_l3_device_qualified"] is False
