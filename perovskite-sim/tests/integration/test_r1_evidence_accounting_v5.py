"""Real formal partial runs, resumes and a retained failed comparison."""
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tests.integration.test_r1_physics_study_formal import (
    ENV, digest, frozen_project, read, reseal, write,
)

pytestmark = pytest.mark.slow
FAILED_CASE = "Compare/D/N16N64"


@pytest.fixture(scope="module")
def bounded_study(tmp_path_factory):
    base = tmp_path_factory.mktemp("r1-v5-accounting")
    checkout = base / "checkout"
    checkout.mkdir()
    project, revision = frozen_project(checkout)
    plan = base / "CallerPlanV1.json"
    output = base / "ResultV1"
    command = [sys.executable, "-I", "-S", str(project / "scripts/run_one_dimensional_mechanism_r1_controlled.py"),
        "--project", str(project), "--source-commit", revision, "--dependency-path", str(Path(np.__file__).resolve().parents[1]),
        "--runner", "physics-study", "--", "--formal", "--grids", "16", "64",
        "--section", "prepare", "--section", "short", "--section", "compare", "--plan-file", str(plan)]
    for case in ("Preparation/N16", "Preparation/N64", "Short/N16/D", "Short/N64/D", FAILED_CASE):
        command += ["--case", case]

    def launch(destination, *extra):
        args = command + ["--output-dir", str(destination), *extra]
        if "--plan-only" not in extra:
            args += ["--request-sha256", digest(plan)]
        return subprocess.run(args, env=ENV, cwd=project, capture_output=True, text=True, timeout=300)

    planned = launch(output, "--plan-only")
    assert planned.returncode == 0, planned.stdout + planned.stderr
    assert len(read(plan)["cases"]) == 7
    snapshots = []
    for index in range(3):
        extra = ["--max-cases", "2"] if index < 2 else []
        if index:
            extra += ["--resume", "--manifest-sha256", digest(output / "ManifestV1.json")]
        produced = launch(output, *extra)
        expected = 2 if index < 2 else 1
        assert produced.returncode == expected, produced.stdout + produced.stderr
        snapshots.append(read(output / "StudySummaryV1.json"))
        if index == 0:
            checked = launch(output, "--verify", "--manifest-sha256", digest(output / "ManifestV1.json"))
            assert checked.returncode == 2 and not checked.stderr, checked.stdout + checked.stderr
    return SimpleNamespace(output=output, plan=plan, launch=launch, snapshots=snapshots,
                           anchor=digest(output / "ManifestV1.json"))


def test_partial_resumes_count_only_new_attempts_and_keep_the_real_failure(bounded_study):
    snapshots = bounded_study.snapshots
    assert [record["attempted_cases"] for record in snapshots] == [2, 2, 3]
    assert [record["active_case_count"] for record in snapshots] == [2, 4, 7]
    assert [len(record["missing_cases"]) for record in snapshots] == [5, 3, 0]
    assert [record["diagnostic_failure_count"] for record in snapshots] == [0, 0, 1]
    assert all(len(record["invocation_attempts"]) == record["attempted_cases"] for record in snapshots)
    checked = bounded_study.launch(bounded_study.output, "--verify", "--manifest-sha256", bounded_study.anchor)
    assert checked.returncode == 1 and not checked.stderr, checked.stdout + checked.stderr
    assert digest(bounded_study.output / "ManifestV1.json") == bounded_study.anchor


def test_failure_erasure_is_refused_with_original_anchor_and_with_new_reseal(bounded_study, tmp_path):
    output = tmp_path / "Erased"
    shutil.copytree(bounded_study.output, output)
    original_plan = (output / "StudyPlanV1.json").read_bytes()
    original_inventory = (output / "CaseInventoryV2.json").read_bytes()
    shutil.rmtree(output / FAILED_CASE)
    for path in [output / "StudySummaryV1.json", *sorted((output / "Invocations").glob("*.json"))]:
        value = read(path)
        value["cases"] = [row for row in value["cases"] if row["case"] != FAILED_CASE]
        if value["active_case_count"] == 7:
            value.update(active_case_count=6, diagnostic_failure_count=0, diagnostic_failed_case_count=0)
        write(path, value)
    write(output / "FailureIndexV1.json", {"schema": "R1PhysicsStudyFailureIndexV1", "cases": []})
    reseal(output)
    assert (output / "StudyPlanV1.json").read_bytes() == original_plan
    assert (output / "CaseInventoryV2.json").read_bytes() == original_inventory
    anchored = bounded_study.launch(output, "--verify", "--manifest-sha256", bounded_study.anchor)
    assert anchored.returncode == 1 and "external study manifest" in anchored.stderr
    resealed = bounded_study.launch(output, "--verify", "--manifest-sha256", digest(output / "ManifestV1.json"))
    assert resealed.returncode == 1 and "invocation attempt is missing" in resealed.stderr


def test_archived_plan_is_not_an_external_request(bounded_study):
    result = bounded_study.launch(bounded_study.output, "--verify", "--manifest-sha256", bounded_study.anchor,
                                  "--plan-file", str(bounded_study.output / "StudyPlanV1.json"))
    assert result.returncode == 1 and "outside the result directory" in result.stderr
