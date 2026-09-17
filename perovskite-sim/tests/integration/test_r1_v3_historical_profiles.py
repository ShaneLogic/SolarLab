"""Known-source compatibility uses a real old checkout, never bundled code."""
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


@pytest.fixture(scope="module", params=tuple(HISTORICAL_PRODUCERS))
def historical_prepare(request, tmp_path_factory):
    root = tmp_path_factory.mktemp("r1-real-historical-profile")
    repository = root / "source"
    subprocess.run(["git", "clone", "--shared", "--no-checkout", "--quiet", str(PROJECT.parent),
                    str(repository)], env=git_environment(), check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repository), "checkout", "--quiet", "--detach", request.param],
                   env=git_environment(), check=True, capture_output=True)
    project = repository / "perovskite-sim"
    output = root / "prepare"
    dependency = Path(np.__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update({key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")})
    command = [sys.executable, "-I", "-S", str(project / "scripts/run_one_dimensional_mechanism_r1_controlled.py"),
        "--project", str(project), "--source-commit", request.param, "--dependency-path", str(dependency), "--",
        "prepare", "--output-dir", str(output), "--intervals", "16", "--reference",
        str(project / "tests/fixtures/OneDimensionalMechanismR1ReferenceBindingV1.json")]
    result = subprocess.run(command, cwd=project, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return output, dict(expected_source_commit=request.param, producer_project=project,
        expected_manifest_sha256=hashlib.sha256((output / "ManifestV1.json").read_bytes()).hexdigest(),
        python_executable=sys.executable, dependency_path=dependency)


def test_real_392_and_815_checks_preserve_source_and_withhold_v3_qualification(historical_prepare):
    output, arguments = historical_prepare
    before = (output / "ManifestV1.json").read_bytes()
    report = inspect_historical_evidence(output, **arguments, timeout_s=120)
    assert report["verification_completed_without_invocation_error"]
    assert report["reconstruction_completed"]
    assert report["producer_report"]["scientifically_accepted"]
    assert report["producer_source_commit"] == arguments["expected_source_commit"]
    assert report["verifier_source_commit"] == arguments["expected_source_commit"]
    assert not report["scientifically_accepted"]
    assert not report["eligible_for_v3_acceptance"]
    assert (output / "ManifestV1.json").read_bytes() == before


def test_historical_dispatch_does_not_choose_the_bundle_source_or_accept_a_new_anchor(historical_prepare):
    output, arguments = historical_prepare
    unknown = dict(arguments, expected_source_commit="f" * 40)
    with pytest.raises(ValueError, match="explicit compatibility set"):
        historical_verifier_command(output, **unknown)
    other = next(c for c in HISTORICAL_PRODUCERS if c != arguments["expected_source_commit"])
    with pytest.raises(ValueError, match="checkout differs"):
        historical_verifier_command(output, **dict(arguments, expected_source_commit=other))
    with pytest.raises(ValueError, match="caller.s anchor"):
        historical_verifier_command(output, **dict(arguments, expected_manifest_sha256="0" * 64))


def test_dirty_historical_verifier_cannot_repin_its_own_approved_profile(historical_prepare):
    output, arguments = historical_prepare
    source = arguments["producer_project"] / "docs/OneDimensionalMechanismR1ResponseV1.md"
    original = source.read_bytes()
    try:
        source.write_bytes(original + b"\nWaive all original physical criteria.\n")
        with pytest.raises(ValueError, match="modified tracked source"):
            historical_verifier_command(output, **arguments)
    finally:
        source.write_bytes(original)
