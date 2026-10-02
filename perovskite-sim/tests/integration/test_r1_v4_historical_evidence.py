"""Real old producers retain failure and scope under caller-held anchors."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from perovskite_sim.experiments.one_dimensional_mechanism_r1_checkout import git_environment
from perovskite_sim.experiments.one_dimensional_mechanism_r1_history import (
    HISTORICAL_PRODUCERS, historical_verifier_command, inspect_historical_evidence,
)

pytestmark = pytest.mark.slow
PROJECT = Path(__file__).resolve().parents[2]
THREADS = {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                  "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module", params=tuple(HISTORICAL_PRODUCERS))
def old_producer(request, tmp_path_factory):
    directory = tmp_path_factory.mktemp("r1-v4-historical")
    repository = directory / "source"
    subprocess.run(["git", "clone", "--shared", "--no-checkout", "--quiet", str(PROJECT.parent), str(repository)],
                   env=git_environment(), check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repository), "checkout", "--quiet", "--detach", request.param],
                   env=git_environment(), check=True, capture_output=True)
    project = repository / "perovskite-sim"
    dependency = Path(np.__file__).resolve().parents[1]
    prefix = [sys.executable, "-I", "-S", str(project / "scripts/run_one_dimensional_mechanism_r1_controlled.py"),
              "--project", str(project), "--source-commit", request.param, "--dependency-path", str(dependency)]
    arguments = dict(expected_source_commit=request.param, producer_project=project,
                     python_executable=sys.executable, dependency_path=dependency)
    return directory, project, prefix, arguments


def run(prefix, args, project):
    result = subprocess.run([*prefix, *args], cwd=project, env={**os.environ, **THREADS},
                            capture_output=True, text=True, timeout=600)
    assert result.returncode in (0, 1), result.stdout + result.stderr
    return result


def test_real_prepare_passing_and_failed_steps_preserve_historical_verdict(old_producer):
    directory, project, prefix, arguments = old_producer
    preparation = directory / "prepare"
    reference = project / "tests/fixtures/OneDimensionalMechanismR1ReferenceBindingV1.json"
    produced = run(prefix, ["--", "prepare", "--output-dir", str(preparation), "--reference", str(reference),
                           "--intervals", "32"], project)
    assert produced.returncode == 0, produced.stdout + produced.stderr
    prepare_anchor = digest(preparation / "ManifestV1.json")
    prepared = inspect_historical_evidence(preparation, **arguments,
                                          expected_manifest_sha256=prepare_anchor, timeout_s=600)
    assert prepared["historical_contract_passed"] is True
    assert prepared["diagnostic_failure_count"] == 0
    for amplitude, expected_failure in (("0.005", False), ("0.01", True)):
        output = directory / ("failed-step" if expected_failure else "passing-step")
        produced = run(prefix, ["--", "step", "--output-dir", str(output), "--reference", str(reference),
            "--prepared", str(preparation / "PreparedStateV1.json"), "--prepared-manifest-sha256", prepare_anchor,
            "--control", "D", "--intervals", "32", "--nonlinear-factor", "1" if expected_failure else "0.1", "--amplitude", amplitude], project)
        assert produced.returncode == int(expected_failure), produced.stdout + produced.stderr
        anchor = digest(output / "ManifestV1.json")
        report = inspect_historical_evidence(output, **arguments, expected_manifest_sha256=anchor,
            expected_prepared_manifest_sha256=prepare_anchor, timeout_s=600)
        assert report["reconstruction_completed"]
        assert report["diagnostic_failure_count"] == int(expected_failure)
        assert report["historical_contract_passed"] is (not expected_failure)
        assert not report["scientifically_accepted"] and not report["eligible_for_current_acceptance"]
        assert not report["original_requested_case_set_verified"]
        assert digest(output / "ManifestV1.json") == anchor
        with pytest.raises(ValueError, match="caller.s anchor"):
            historical_verifier_command(output, **arguments, expected_manifest_sha256="0" * 64)
        with pytest.raises(ValueError, match="reconstruction failed"):
            inspect_historical_evidence(output, **arguments, expected_manifest_sha256=anchor,
                expected_prepared_manifest_sha256="0" * 64, timeout_s=600)


def test_real_physics_study_keeps_configuration_failures_and_anchor(old_producer):
    directory, project, prefix, arguments = old_producer
    if arguments["expected_source_commit"] != "815c2fe7c5d884381b2ecaefa4d39b355e1b4904":
        pytest.skip("one representative archived study format; other profiles exercised above")
    output = directory / "study"
    produced = run(prefix, ["--runner", "physics-study", "--", "--output-dir", str(output), "--formal",
        "--section", "prepare", "--section", "short", "--section", "compare", "--grids", "16", "64",
        "--matrix-controls", "D"], project)
    assert produced.returncode == 1, produced.stdout + produced.stderr
    summary = json.loads((output / "StudySummaryV1.json").read_text())
    assert summary["diagnostic_failure_count"] == 4
    anchor = digest(output / "ManifestV1.json")
    report = inspect_historical_evidence(output, **arguments, expected_manifest_sha256=anchor,
                                         kind="physics-study", timeout_s=1200)
    assert report["diagnostic_failure_count"] == 4
    assert report["recorded_case_count"] == summary["active_case_count"]
    assert report["reconstruction_completed"] and not report["historical_contract_passed"]
    assert not report["eligible_for_current_acceptance"]
    # The negative retains the original caller anchor, rather than accepting a new seal.
    manifest = output / "ManifestV1.json"
    manifest.write_bytes(manifest.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="caller.s anchor"):
        historical_verifier_command(output, **arguments, expected_manifest_sha256=anchor, kind="physics-study")
