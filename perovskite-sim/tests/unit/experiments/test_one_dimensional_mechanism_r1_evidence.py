"""Reference preparation must consume complete, matching GEO evidence."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

from perovskite_sim.models.config_loader import load_device_from_yaml


PROJECT = Path(__file__).resolve().parents[3]


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(PROJECT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "r1_evidence_runner", PROJECT / "scripts/run_one_dimensional_mechanism_r1.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def inputs():
    study = json.loads((PROJECT / "reproducibility/OneDimensionalMechanismR1InputV1.json").read_text())
    return study, load_device_from_yaml(PROJECT / study["fixture"])


@pytest.fixture
def evidence(tmp_path, runner, inputs):
    study, stack = inputs
    folder = tmp_path / "geometry"
    folder.mkdir()
    runner.write_json(folder / "CompletionV1.json",
                      {"stage": "geometry", "status": "passed", "failure": None})
    runner.write_json(folder / "StudyInputV1.json", study)
    runner.write_json(folder / "ResolvedStackV1.json", stack)
    runner.write_json(folder / "GeometryV1.json",
                      runner.geometry_records(stack, study["geometry_intervals"]))
    shutil.copyfile(PROJECT / "docs/OneDimensionalMechanismR1GeometryV1.md",
                    folder / "ExecutionContractV1.md")
    sources = list((PROJECT / "perovskite_sim").rglob("*.py")) + [
        PROJECT / runner.GEOMETRY_TEST_PATH, PROJECT / study["fixture"],
    ]
    runner.write_json(folder / "SourceManifestV1.json", {
        p.relative_to(PROJECT.parent).as_posix(): {"sha256": runner.sha256(p)}
        for p in sources
    })
    # Test-only evidence fixture. A real CLI-generated certificate is exercised below.
    names = [
        f"{name}[{n}]" for n in [4, 16, 32, 64]
        for name in [
            "test_geo_01_02_geometry_inventory",
            "test_geo_03_flux_telescopes_and_rate_uses_same_volumes",
            "test_geo_05_constant_charge_poisson",
            "test_geo_06_dielectric_sheet_and_series_capacitance",
        ]
    ] + [
        "test_geo_04_half_occupied_diffusion_cosine_decay",
        "test_geo_05_spatial_convergence_nonquadratic_charge",
    ]
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite", tests=str(len(names)),
                          failures="0", errors="0", skipped="0")
    for name in names:
        ET.SubElement(suite, "testcase", classname=runner.GEOMETRY_TEST_CLASS, name=name)
    ET.ElementTree(root).write(folder / "TestsV1.xml")
    (folder / "TestsV1.log").write_text("Test-only fixture\n")
    runner.manifest(folder)
    return folder


def test_valid_geometry_is_bound_without_directory_name_dependency(evidence, runner, inputs, tmp_path):
    relocated = tmp_path / "arbitrary_name"
    shutil.copytree(evidence, relocated)
    result = runner.require_geometry_evidence(relocated, *inputs)
    assert result["manifest_sha256"] == runner.sha256(relocated / "ManifestV1.json")
    assert len(result["required_test_cases"]) == 18
    assert set(result["required_artifacts"]) == runner.REQUIRED_GEOMETRY_ARTIFACTS


@pytest.mark.parametrize("stage", ["r0-regression", "ac", "coupled", "prepare", None])
def test_wrong_stage_rejected(evidence, runner, inputs, stage):
    runner.write_json(evidence / "CompletionV1.json",
                      {"stage": stage, "status": "passed", "failure": None})
    runner.manifest(evidence)
    with pytest.raises(ValueError, match="geometry-stage"):
        runner.require_geometry_evidence(evidence, *inputs)


@pytest.mark.parametrize("field,value", [("status", "failed"), ("failure", {"type": "RuntimeError"})])
def test_unsuccessful_geometry_rejected(evidence, runner, inputs, field, value):
    completion = {"stage": "geometry", "status": "passed", "failure": None}
    completion[field] = value
    runner.write_json(evidence / "CompletionV1.json", completion)
    runner.manifest(evidence)
    with pytest.raises(ValueError, match="geometry-stage"):
        runner.require_geometry_evidence(evidence, *inputs)


@pytest.mark.parametrize("name", [
    "GeometryV1.json", "TestsV1.xml", "TestsV1.log", "SourceManifestV1.json",
    "StudyInputV1.json", "ResolvedStackV1.json", "ExecutionContractV1.md",
    "CompletionV1.json",
])
def test_required_artifact_cannot_be_omitted_from_manifest(evidence, runner, inputs, name):
    entries = json.loads((evidence / "ManifestV1.json").read_text())
    del entries[name]
    runner.write_json(evidence / "ManifestV1.json", entries)
    with pytest.raises(ValueError, match="required evidence"):
        runner.require_geometry_evidence(evidence, *inputs)


@pytest.mark.parametrize("name", ["GeometryV1.json", "TestsV1.xml", "StudyInputV1.json"])
def test_unsealed_changes_and_missing_files_rejected(evidence, runner, inputs, name):
    path = evidence / name
    original = path.read_bytes()
    path.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="identity mismatch"):
        runner.require_geometry_evidence(evidence, *inputs)
    path.unlink()
    with pytest.raises(ValueError, match="identity mismatch"):
        runner.require_geometry_evidence(evidence, *inputs)


@pytest.mark.parametrize("name,value,message", [
    ("StudyInputV1.json", {}, "study input"),
    ("ResolvedStackV1.json", {}, "resolved stack"),
    ("GeometryV1.json", [], "physical grids"),
    ("SourceManifestV1.json", {}, "source differs"),
])
def test_resealed_mismatched_records_rejected(evidence, runner, inputs, name, value, message):
    runner.write_json(evidence / name, value)
    runner.manifest(evidence)
    with pytest.raises(ValueError, match=message):
        runner.require_geometry_evidence(evidence, *inputs)


def test_stale_contract_rejected(evidence, runner, inputs):
    (evidence / "ExecutionContractV1.md").write_text("old contract\n")
    runner.manifest(evidence)
    with pytest.raises(ValueError, match="execution contract"):
        runner.require_geometry_evidence(evidence, *inputs)


@pytest.mark.parametrize("mutation", [
    "failure", "error", "skipped", "count", "missing_case", "duplicate", "wrong_class", "empty",
])
def test_incomplete_or_unsuccessful_test_evidence_rejected(evidence, runner, inputs, mutation):
    path = evidence / "TestsV1.xml"
    tree = ET.parse(path)
    suite = tree.getroot()[0]
    if mutation in ("failure", "error", "skipped"):
        # Even a falsely zero suite summary must not hide a nonpassing case.
        ET.SubElement(suite[0], mutation)
    elif mutation == "count":
        suite.set("tests", "0")
    elif mutation == "missing_case":
        suite.remove(suite[0])
        suite.set("tests", str(len(suite)))
    elif mutation == "duplicate":
        suite[1].set("name", suite[0].get("name"))
    elif mutation == "wrong_class":
        suite[0].set("classname", "tests.unrelated")
    else:
        suite.clear()
    tree.write(path)
    runner.manifest(evidence)
    with pytest.raises(ValueError, match="geometry tests"):
        runner.require_geometry_evidence(evidence, *inputs)


def test_manifest_paths_cannot_escape_evidence_directory(evidence, runner, inputs, tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    entries = json.loads((evidence / "ManifestV1.json").read_text())
    entries["../outside.json"] = {"sha256": runner.sha256(outside), "bytes": 2}
    runner.write_json(evidence / "ManifestV1.json", entries)
    with pytest.raises(ValueError, match="identity mismatch"):
        runner.require_geometry_evidence(evidence, *inputs)


def run_cli(*args):
    return subprocess.run(
        [sys.executable, str(PROJECT / "scripts/run_one_dimensional_mechanism_r1.py"), *map(str, args)],
        cwd=PROJECT,
        env=dict(os.environ, PYTHONPATH=str(PROJECT), PYTHONDONTWRITEBYTECODE="1",
                 OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1"),
        capture_output=True, text=True, timeout=120,
    )


def test_cli_rejects_wrong_stage_before_reference_solve(evidence, runner, tmp_path):
    runner.write_json(evidence / "CompletionV1.json", {"stage": "r0-regression", "status": "passed"})
    runner.manifest(evidence)
    output = tmp_path / "rejected"
    completed = run_cli("prepare", "--geometry-evidence", evidence, "--output-dir", output)
    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert json.loads((output / "CompletionV1.json").read_text())["status"] == "failed"
    assert "geometry-stage" in json.loads((output / "FailureV1.json").read_text())["message"]
    assert not list(output.glob("Reference*"))


@pytest.mark.slow
def test_real_geometry_to_preparation_cli_passes(tmp_path):
    geometry, preparation = tmp_path / "geometry", tmp_path / "preparation"
    completed = run_cli("geometry", "--output-dir", geometry)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    completed = run_cli("prepare", "--geometry-evidence", geometry, "--output-dir", preparation)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    record = json.loads((preparation / "GeometryPrerequisiteV1.json").read_text())
    assert record["stage"] == "geometry"
    binding = json.loads((preparation / "ReferenceBindingV1.json").read_text())
    assert binding["reference_intervals"] == 512
    assert binding["rungs"][-1]["difference"] <= 1e-8
