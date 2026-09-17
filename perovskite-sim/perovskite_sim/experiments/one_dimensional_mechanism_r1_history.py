"""Finite, caller-selected historical verification; never current approval.

Only a separately selected frozen producer checkout is executable. Source
code from an evidence ZIP is data, never the implementation of this API.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

from perovskite_sim.experiments.one_dimensional_mechanism_r1_checkout import _git


HISTORICAL_PRODUCERS = {
    "3928259a684b0d4c6c07dc05490bfea09401b015": {
        "evidence_revision": 6,
        "response_contract_sha256": "d6c7c4a1cafebefa75a8729b23f9fdaac570faf148b0ba895a288896b80d2da3",
        "finite_step_current_metric": "internal_and_physical_contact_faces",
    },
    "815c2fe7c5d884381b2ecaefa4d39b355e1b4904": {
        "evidence_revision": 6,
        "response_contract_sha256": "d61d0385985d07f306af6f0d013a97e29c2eb20c232e6a4bb3a2a8f343e1422f",
        "finite_step_current_metric": "internal_and_physical_contact_faces",
    },
}


def _digest(value, label):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(label + " must be a SHA-256 value")
    return value


def historical_verifier_command(output, *, expected_source_commit, producer_project,
                                expected_manifest_sha256, python_executable, dependency_path,
                                expected_prepared_manifest_sha256=None, kind="stage-one"):
    """Build a command only after binding a known producer, never from bundle labels."""
    if expected_source_commit not in HISTORICAL_PRODUCERS:
        raise ValueError("historical producer is outside the explicit compatibility set")
    if kind not in ("stage-one", "physics-study"):
        raise ValueError("unsupported historical evidence kind")
    project = Path(producer_project).resolve(strict=True)
    if _git(project, "rev-parse", "HEAD").decode().strip() != expected_source_commit:
        raise ValueError("historical verifier checkout differs from selected producer commit")
    if _git(project, "diff", "HEAD", "--name-only", "--", ".").strip():
        raise ValueError("historical verifier checkout has modified tracked source")
    response = project / "docs/OneDimensionalMechanismR1ResponseV1.md"
    if hashlib.sha256(response.read_bytes()).hexdigest() != HISTORICAL_PRODUCERS[expected_source_commit]["response_contract_sha256"]:
        raise ValueError("historical declaration differs from its trusted profile")
    output = Path(output).resolve(strict=True)
    expected_manifest_sha256 = _digest(expected_manifest_sha256, "external historical manifest digest")
    if hashlib.sha256((output / "ManifestV1.json").read_bytes()).hexdigest() != expected_manifest_sha256:
        raise ValueError("historical manifest differs from the caller's anchor")
    arguments = [str(Path(python_executable).resolve(strict=True)), "-I", "-S",
        str(project / "scripts/run_one_dimensional_mechanism_r1_controlled.py"),
        "--project", str(project), "--source-commit", expected_source_commit,
        "--runner", kind, "--dependency-path", str(Path(dependency_path).resolve(strict=True)), "--"]
    if kind == "stage-one":
        arguments += ["verify", "--mode", "acceptance", "--required-evidence-revision", "6",
            "--output-dir", str(output), "--expected-manifest-sha256", expected_manifest_sha256,
            "--expected-source-commit", expected_source_commit]
        if expected_prepared_manifest_sha256 is not None:
            arguments += ["--prepared-manifest-sha256", _digest(expected_prepared_manifest_sha256, "historical preparation digest")]
    else:
        request = json.loads((output / "StudyRequestV1.json").read_text())
        if request.get("source", {}).get("source_commit") != expected_source_commit:
            raise ValueError("historical study producer differs from the caller's source")
        arguments += ["--formal", "--verify", "--section", "all", "--output-dir", str(output),
            "--manifest-sha256", expected_manifest_sha256, "--grids", *map(str, request["grids"]),
            "--matrix-controls", *request["matrix_controls"], "--window", request["window"],
            "--first-time-s", str(request["window_extensions"]["first_time_s"]),
            "--last-time-s", str(request["window_extensions"]["last_time_s"])]
        if len(request["frequency_Hz"]) == 66:
            arguments.append("--extended-frequency")
    return arguments


def inspect_historical_evidence(output, *, expected_source_commit, producer_project,
                                expected_manifest_sha256, python_executable, dependency_path,
                                expected_prepared_manifest_sha256=None, kind="stage-one", timeout_s=300):
    """Recompute through a selected old contract and explicitly withhold V3 qualification."""
    command = historical_verifier_command(output, expected_source_commit=expected_source_commit,
        producer_project=producer_project, expected_manifest_sha256=expected_manifest_sha256,
        python_executable=python_executable, dependency_path=dependency_path,
        expected_prepared_manifest_sha256=expected_prepared_manifest_sha256, kind=kind)
    environment = os.environ.copy()
    environment.update({key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")})
    try:
        result = subprocess.run(command, cwd=producer_project, env=environment, capture_output=True,
                                text=True, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired as exc:
        raise ValueError("historical verification did not complete within its selected time budget") from exc
    if hashlib.sha256((Path(output) / "ManifestV1.json").read_bytes()).hexdigest() != expected_manifest_sha256:
        raise ValueError("historical verifier changed the anchored evidence manifest")
    reports = []
    for index, character in enumerate(result.stdout):
        if character != "{" or (index and result.stdout[index - 1] != "\n"):
            continue
        try:
            report, _ = json.JSONDecoder().raw_decode(result.stdout[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(report, dict):
            reports.append(report)
    if kind == "stage-one":
        reports = [r for r in reports if "result_checks" in r and "scientifically_accepted" in r]
        completed = bool(reports and reports[-1].get("source_commit_anchor") == expected_source_commit)
    else:
        reports = [r for r in reports if r.get("schema") == "R1PhysicsStudySummaryV2"]
        completed = bool(reports and reports[-1].get("requirements", {}).get("verification_completed_without_error")
                         and reports[-1].get("requirements", {}).get("all_recorded_cases_checked")
                         and not reports[-1].get("not_recomputed_in_this_invocation"))
    if not completed:
        raise ValueError("historical source-bound reconstruction failed: " + result.stderr.strip()[-1000:])
    return {
        "schema": "R1HistoricalInspectionV1", "mode": "historical_inspection",
        "producer_source_commit": expected_source_commit,
        "verifier_source_commit": expected_source_commit,
        "profile": dict(HISTORICAL_PRODUCERS[expected_source_commit]),
        "manifest_sha256": expected_manifest_sha256, "root_manifest_unchanged": True,
        "reconstruction_completed": True, "producer_exit_code": result.returncode,
        "producer_report": reports[-1],
        "scientifically_accepted": False, "eligible_for_v3_acceptance": False,
        "scope": "selected_historical_producer_contract; not_current_standard_approval_or_R1_2_qualification",
        "command": command,
    }
