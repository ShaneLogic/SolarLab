"""Formal R1 source admission rejects shadow copies and missing coverage."""

from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import zipfile

import pytest

import perovskite_sim
from perovskite_sim.experiments import one_dimensional_mechanism_r1_checkout as checkout


PROJECT = Path(__file__).resolve().parents[3]
RUNNER = PROJECT / "scripts/run_one_dimensional_mechanism_r1_stage_one.py"


@pytest.fixture(scope="module")
def context():
    return checkout.require_r1_checkout(project=PROJECT, runner=RUNNER)


def test_actual_git_worktree_covers_all_current_package_sources(context):
    assert context.root == PROJECT.parent.resolve()
    assert context.project == PROJECT.resolve()
    assert len(context.commit) == 40
    package_sources = {
        path.relative_to(context.root).as_posix()
        for path in (PROJECT / "perovskite_sim").rglob("*.py")
    }
    assert package_sources <= context.required_sources.keys()
    assert set(context.required_sources) <= set(context.tracked_paths)
    for path in (
        "scripts/run_one_dimensional_mechanism_r1_stage_one.py",
        "scripts/run_one_dimensional_mechanism_r1.py",
        "scripts/run_one_dimensional_mechanism_r0.py",
        checkout.STUDY_INPUT_RELATIVE_PATH,
        "docs/OneDimensionalMechanismR1DynamicsV1.md",
        "tests/fixtures/configs/dynamic_interface_defect_ion_transient_absorber_only.yaml",
    ):
        assert "perovskite-sim/" + path in context.required_sources
    assert context.to_dict()["observed_commit"] == context.commit
    assert "approval" in context.to_dict()["scope"]


@pytest.mark.parametrize("argument", ["project", "runner"])
def test_unexpected_cli_location_is_rejected(tmp_path, argument):
    arguments = {"project": PROJECT, "runner": RUNNER, argument: tmp_path}
    with pytest.raises(checkout.R1CheckoutError, match="expected tracked checkout path"):
        checkout.require_r1_checkout(**arguments)


def test_repository_environment_cannot_redirect_checkout_identity(monkeypatch, tmp_path):
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "unrelated_git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path))
    monkeypatch.setenv("GIT_INDEX_FILE", str(tmp_path / "unrelated_index"))
    assert checkout.require_r1_checkout().root == PROJECT.parent.resolve()


def test_unexpected_imported_package_origin_is_rejected(monkeypatch, tmp_path):
    shadow = tmp_path / "shadow_module.py"
    shadow.write_text("# This file is never executed.\n")
    monkeypatch.setitem(sys.modules, "perovskite_sim.shadow_module", SimpleNamespace(__file__=str(shadow)))
    with pytest.raises(checkout.R1CheckoutError, match="unexpected imported package origin"):
        checkout.require_r1_checkout()


def test_injected_package_search_path_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(perovskite_sim, "__path__", [str(PROJECT / "perovskite_sim"), str(tmp_path)])
    with pytest.raises(checkout.R1CheckoutError, match="unexpected imported package search path"):
        checkout.require_r1_checkout()


def _probe_context(project):
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(project))
    return subprocess.run(
        [sys.executable, "-c", (
            "from pathlib import Path; "
            "from perovskite_sim.experiments import one_dimensional_mechanism_r1_checkout as c; "
            "print(c.__file__); "
            "c.require_r1_checkout(project=Path.cwd(), "
            "runner=Path.cwd()/'scripts/run_one_dimensional_mechanism_r1_stage_one.py')"
        )],
        cwd=project, env=environment, capture_output=True, text=True, timeout=30,
    )


def test_whole_project_copy_under_ignored_outputs_is_rejected():
    output = PROJECT / "outputs"
    output.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="r1_checkout_shadow_", dir=output) as directory:
        shadow = Path(directory)
        for name in ("perovskite_sim", "scripts", "docs", "reproducibility"):
            shutil.copytree(PROJECT / name, shadow / name, ignore=shutil.ignore_patterns("__pycache__"))
        fixture = Path("tests/fixtures/configs/dynamic_interface_defect_ion_transient_absorber_only.yaml")
        (shadow / fixture).parent.mkdir(parents=True)
        shutil.copyfile(PROJECT / fixture, shadow / fixture)
        study_path = shadow / checkout.STUDY_INPUT_RELATIVE_PATH
        study = json.loads(study_path.read_text())
        study["fixed_reference_binding_sha256"] = "0" * 64
        study_path.write_text(json.dumps(study))
        copied_binding = shadow / "perovskite_sim/experiments/one_dimensional_mechanism_r1_binding.py"
        assert copied_binding.read_bytes() == (PROJECT / copied_binding.relative_to(shadow)).read_bytes()
        # The weaker samefile-only guard is satisfied by this whole copy.
        assert study_path.samefile(copied_binding.resolve().parents[2] / checkout.STUDY_INPUT_RELATIVE_PATH)
        result = _probe_context(shadow)
        assert result.returncode != 0
        assert str(shadow) in result.stdout
        assert "expected tracked checkout path" in result.stderr


def test_package_only_copy_without_git_has_a_clear_execution_error(tmp_path):
    shutil.copytree(PROJECT / "perovskite_sim", tmp_path / "perovskite_sim", ignore=shutil.ignore_patterns("__pycache__"))
    result = _probe_context(tmp_path)
    assert result.returncode != 0
    assert "R1CheckoutError" in result.stderr
    assert "requires a Git checkout" in result.stderr
    assert "FileNotFoundError" not in result.stderr


@pytest.fixture
def evidence(context, tmp_path):
    manifest = {name: dict(entry) for name, entry in context.required_sources.items()}
    (tmp_path / "SourceManifestV1.json").write_text(json.dumps(manifest))
    with zipfile.ZipFile(tmp_path / "SourceV1.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in manifest:
            archive.write(context.root / name, name)
    return tmp_path


def test_source_capture_matches_required_execution_bytes(context, evidence):
    result = checkout.validate_source_coverage(evidence, context)
    assert result["certified"]
    assert result["source_file_count"] == result["required_execution_source_count"] == len(context.required_sources)


def test_empty_source_manifest_is_rejected(context, evidence):
    (evidence / "SourceManifestV1.json").write_text("{}")
    with pytest.raises(checkout.R1CheckoutError, match="nonempty object"):
        checkout.validate_source_coverage(evidence, context)


@pytest.mark.parametrize("missing", [
    "scripts/run_one_dimensional_mechanism_r1_stage_one.py",
    "scripts/run_one_dimensional_mechanism_r1.py",
    checkout.STUDY_INPUT_RELATIVE_PATH,
    "docs/OneDimensionalMechanismR1DynamicsV1.md",
    "tests/fixtures/configs/dynamic_interface_defect_ion_transient_absorber_only.yaml",
    "perovskite_sim/physics/poisson.py",
])
def test_missing_required_execution_source_is_rejected(context, evidence, missing):
    path = evidence / "SourceManifestV1.json"
    entries = json.loads(path.read_text())
    del entries["perovskite-sim/" + missing]
    path.write_text(json.dumps(entries))
    with pytest.raises(checkout.R1CheckoutError, match="lacks required execution sources"):
        checkout.validate_source_coverage(evidence, context)


@pytest.mark.parametrize("field,value", [("sha256", "0" * 64), ("bytes", 0)])
def test_wrong_captured_source_identity_is_rejected(context, evidence, field, value):
    path = evidence / "SourceManifestV1.json"
    entries = json.loads(path.read_text())
    entries["perovskite-sim/" + checkout.STUDY_INPUT_RELATIVE_PATH][field] = value
    path.write_text(json.dumps(entries))
    with pytest.raises(checkout.R1CheckoutError, match="source manifest identity mismatch"):
        checkout.validate_source_coverage(evidence, context)


def test_archive_bytes_must_match_manifest_and_execution_source(context, evidence):
    changed = "perovskite-sim/" + checkout.STUDY_INPUT_RELATIVE_PATH
    with zipfile.ZipFile(evidence / "SourceV1.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in context.required_sources:
            raw = (context.root / name).read_bytes()
            archive.writestr(name, b"X" + raw[1:] if name == changed else raw)
    with pytest.raises(checkout.R1CheckoutError, match="source archive identity mismatch"):
        checkout.validate_source_coverage(evidence, context)


def test_empty_required_coverage_context_is_rejected(context, evidence):
    with pytest.raises(checkout.R1CheckoutError, match="no required source coverage"):
        checkout.validate_source_coverage(evidence, replace(context, required_sources={}))
