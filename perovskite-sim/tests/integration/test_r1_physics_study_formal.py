"""Exercise the real controlled study and adversarial resealed AC evidence."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


pytestmark = pytest.mark.slow
PROJECT = Path(__file__).resolve().parents[2]
THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
ENV = {**{k: v for k, v in os.environ.items() if not k.startswith("GIT_")},
       **{key: "1" for key in THREADS}, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reseal(directory):
    # Independent implementation: the attacker supplies new case/root hashes,
    # so detecting byte edits alone cannot satisfy these negative tests.
    write(directory / "ManifestV1.json", {
        path.relative_to(directory).as_posix(): digest(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path != directory / "ManifestV1.json"
    })


def git(directory, *arguments):
    result = subprocess.run(["git", "-C", str(directory), *arguments], env=ENV,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.strip()


def frozen_project(directory):
    root = Path(git(PROJECT, "rev-parse", "--show-toplevel"))
    allowed = {"perovskite_sim", "scripts", "tests", "docs", "reproducibility", "configs"}
    for _ in range(3):
        names = git(root, "ls-files", "--cached", "--others", "--exclude-standard", "--", "perovskite-sim").splitlines()
        paths = [Path(name) for name in sorted(set(names))
                 if len(Path(name).parts) > 1 and (Path(name).parts[1] in allowed or Path(name).name == "pyproject.toml")
                 and not {"outputs", ".venv", "__pycache__", ".pytest_cache", "node_modules"}.intersection(Path(name).parts)]
        captured = {path: (root / path).read_bytes() for path in paths
                    if (root / path).is_file() and not (root / path).is_symlink()}
        if all((root / path).read_bytes() == raw for path, raw in captured.items()):
            break
    else:
        pytest.fail("source did not stabilize while creating the independent test checkout")
    for path, raw in captured.items():
        target = directory / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    git(directory, "init", "-q")
    git(directory, "add", "-A")
    git(directory, "-c", "core.hooksPath=" + os.devnull, "-c", "user.name=R1 Formal Test",
        "-c", "user.email=r1-formal-test@example.invalid", "commit", "-qm", "Frozen integration test source")
    return directory / "perovskite-sim", git(directory, "rev-parse", "HEAD")


@pytest.fixture(scope="module")
def formal_study(tmp_path_factory):
    base = tmp_path_factory.mktemp("r1-formal-study")
    checkout = base / "checkout"
    checkout.mkdir()
    project, revision = frozen_project(checkout)
    dependencies = Path(np.__file__).resolve().parent.parent
    launcher = project / "scripts/run_one_dimensional_mechanism_r1_controlled.py"

    def launch(output, *extra):
        return subprocess.run([
            sys.executable, "-I", "-S", str(launcher), "--project", str(project),
            "--source-commit", revision, "--dependency-path", str(dependencies),
            "--runner", "physics-study", "--", "--formal", "--output-dir", str(output),
            "--grids", "16", *extra,
        ], env=ENV, cwd=checkout, capture_output=True, text=True, timeout=180)

    output = base / "evidence"
    result = launch(output, "--section", "prepare", "--section", "dc-ac")
    assert result.returncode == 0, result.stdout + result.stderr
    assert read(output / "StudyRequestV1.json")["run_class"] == "formal"
    assert read(output / "StudySummaryV1.json")["active_case_count"] == 6
    assert not read(output / "StudySummaryV1.json")["study_exit_passed"]
    return SimpleNamespace(output=output, project=project, revision=revision, launch=launch,
                           anchor=digest(output / "ManifestV1.json"))


def verify(study, output, anchor=None):
    return study.launch(output, "--section", "prepare", "--section", "dc-ac", "--verify",
                        "--manifest-sha256", anchor or digest(output / "ManifestV1.json"))


def test_formal_dc_ac_evidence_verifies_with_an_external_anchor_without_writes(formal_study):
    study = formal_study
    before = {p.relative_to(study.output).as_posix(): digest(p) for p in study.output.rglob("*") if p.is_file()}
    result = verify(study, study.output, study.anchor)
    assert result.returncode == 0, result.stdout + result.stderr
    assert {p.relative_to(study.output).as_posix(): digest(p) for p in study.output.rglob("*") if p.is_file()} == before
    rejected = verify(study, study.output, "0" * 64)
    assert rejected.returncode == 1
    assert "external study manifest" in rejected.stderr


@pytest.mark.parametrize("tamper", [False, True], ids=["unchanged-reseal-control", "published-ac-and-verdict-forgery"])
def test_resealed_ac_publication_and_pass_flags_are_replayed(formal_study, tmp_path, tamper):
    output = tmp_path / "copied"
    shutil.copytree(formal_study.output, output)
    case = output / "AC/N16/D/AttemptV1"
    result = read(case / "ResultV1.json")
    completion = read(case / "CompletionV1.json")
    if tamper:
        published = result["admittance_S_m2"]
        assert isinstance(published, list) and "real" in published[0]
        published[0]["real"] *= 1000.
        result["checks"] = {key: [True] * len(value) for key, value in result["checks"].items()}
        result["numerically_eligible_frequency_points"] = [True] * len(published)
        completion.update(status="completed", scientific_checks_passed=True, failure=None)
    write(case / "ResultV1.json", result)
    write(case / "CompletionV1.json", completion)
    reseal(case)
    reseal(output)
    verification = verify(formal_study, output)
    if tamper:
        assert verification.returncode == 1
        error = verification.stderr.lower()
        assert any(term in error for term in ("response", "admittance", "replay", "reconstruct", "ac")), error
        assert "manifest mismatch" not in error and "source changed" not in error
        subset = formal_study.launch(output, "--section", "prepare", "--verify",
                                     "--manifest-sha256", digest(output / "ManifestV1.json"))
        assert subset.returncode != 0, (
            "A forged unselected AC case must be rejected or explicitly remain unverified; "
            "selecting only prepare cannot turn it into a successful study verification.\n" + subset.stdout)
    else:
        assert verification.returncode == 0, verification.stdout + verification.stderr


def test_false_saved_verdict_cannot_replace_recomputed_science(formal_study, tmp_path):
    output = tmp_path / "false-verdict"
    shutil.copytree(formal_study.output, output)
    case = output / "AC/N16/D/AttemptV1"
    completion = read(case / "CompletionV1.json")
    completion["scientific_checks_passed"] = False
    write(case / "CompletionV1.json", completion)
    reseal(case)
    reseal(output)
    result = verify(formal_study, output)
    assert result.returncode == 1
    assert "scientific verdict differs from recomputed" in result.stderr


@pytest.mark.parametrize("state,exit_code", [("failed", 1), ("missing", 2)])
def test_repeated_subset_resume_retains_failure_or_missing_inventory(formal_study, tmp_path, state, exit_code):
    output = tmp_path / state
    shutil.copytree(formal_study.output, output)
    case = output / "AC/N16/D/AttemptV1"
    if state == "missing":
        shutil.rmtree(case.parent)
    else:
        failure = {"type": "InjectedStop", "message": "bounded saved failure", "partial_result": None}
        completion = read(case / "CompletionV1.json")
        completion.update(status="failed", scientific_checks_passed=False,
                          failure={key: value for key, value in failure.items() if key != "partial_result"})
        write(case / "CompletionV1.json", completion)
        write(case / "FailureV1.json", failure)
        (case / "ResultV1.json").unlink()
        reseal(case)
    reseal(output)
    for _ in range(2):
        result = formal_study.launch(output, "--section", "prepare", "--resume",
                                     "--manifest-sha256", digest(output / "ManifestV1.json"))
        assert result.returncode == exit_code, result.stdout + result.stderr
        summary = read(output / "StudySummaryV1.json")
        assert not summary["study_exit_passed"]
        if state == "missing":
            assert "AC/N16/D" in summary["missing_cases"]
        else:
            assert any(item["completion"]["case"] == "AC/N16/D" for item in read(output / "FailureIndexV1.json")["cases"])
