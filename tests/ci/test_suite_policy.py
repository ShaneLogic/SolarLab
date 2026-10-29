"""Regression guards for test discovery, skip precision and CI ownership."""
from __future__ import annotations

import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from suite_support import changed_bound_core_sources, clean_core_skip_reason
from .policy import Selection, audit, audit_files, matches, workflow_selections


ROOT = Path(__file__).resolve().parents[2]


def test_every_root_test_file_has_an_ordinary_ci_job_or_reasoned_exclusion():
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    exclusions = json.loads(Path(__file__).with_name("exclusions.json").read_text())
    selections = workflow_selections(workflow)
    report = audit(ROOT, selections, exclusions)
    print(json.dumps({key: report[key] for key in (
        "test_files", "uncovered_files", "duplicate_files", "conflicting_exclusions")}))
    assert report["uncovered_files"] == [], report
    assert report["duplicate_files"] == {}, report
    assert report["conflicting_exclusions"] == [], report
    assert {selection.job for selection in selections} >= {"static", "core", "packages", "backend", "frontend"}


def test_guard_detects_new_unassigned_directory_and_frontend_marker_gap(tmp_path):
    folder = tmp_path / "tests/new_area"
    folder.mkdir(parents=True)
    test = folder / "test_new.py"
    test.write_text("def test_new(): pass\n")
    policy = {"files": {}}
    assert audit(tmp_path, (Selection("core", ("tests/unit",), "not slow"),), policy)["uncovered_files"]
    assert audit(tmp_path, (Selection("frontend", ("tests",), "frontend"),), policy)["uncovered_files"]
    assert not audit(tmp_path, (Selection("core", ("tests",), "not slow"),), policy)["uncovered_files"]


@pytest.mark.parametrize("command", [
    "python -m pytest tests/unit/test_named.py",
    "python -m pytest tests/unit -k not_a_failure",
    "python -m pytest tests/unit --deselect tests/unit/test_named.py::test_failure",
])
def test_guard_rejects_file_lists_and_hidden_deselection(command):
    with pytest.raises(ValueError):
        workflow_selections({"jobs": {"core": {"steps": [{"run": command}]}}})


def test_optional_job_cannot_hide_an_uncovered_file():
    value = {"jobs": {"coverage": {"if": "manual", "steps": [{"run": "python -m pytest tests"}]}}}
    assert workflow_selections(value) == ()
    assert matches("not slow and frontend", {"frontend"})
    assert not matches("not slow and frontend", {"slow", "frontend"})


def test_ci_bookkeeping_uses_the_pytest_namespace():
    source = (ROOT / ".github/workflows/ci.yml").read_text()
    workflow = yaml.safe_load(source)
    assert "SOLARLAB_CI_JOB" not in source
    for job in ("core", "packages", "backend", "frontend"):
        body = workflow["jobs"][job]
        environments = [body.get("env", {}), *(step.get("env", {}) for step in body["steps"])]
        assert [env["PYTEST_CI_JOB"] for env in environments if "PYTEST_CI_JOB" in env] == [job]


def test_frontend_prerequisite_is_enforced_with_pytest_ci_job(monkeypatch, request):
    hook = next(plugin for plugin in request.config.pluginmanager.get_plugins()
                if getattr(plugin, "__file__", None) == str(ROOT / "tests/conftest.py"))
    markers = []
    item = SimpleNamespace(get_closest_marker=lambda name: name == "frontend", add_marker=markers.append)
    monkeypatch.setattr(hook.shutil, "which", lambda _: None)
    monkeypatch.setenv("PYTEST_CI_JOB", "frontend")
    with pytest.raises(pytest.UsageError, match="frontend contracts require Node"):
        hook.pytest_collection_modifyitems([item])
    monkeypatch.delenv("PYTEST_CI_JOB")
    hook.pytest_collection_modifyitems([item])
    assert len(markers) == 1
    assert "frontend contracts require Node" in markers[0].kwargs["reason"]


def test_duplicate_ci_jobs_fail_for_a_synthetic_file():
    sources = {"tests/test_overlap.py": "def test_overlap(): pass\n"}
    jobs = (Selection("core", ("tests",), "not slow"), Selection("packages", ("tests",), "not slow"))
    with pytest.raises(ValueError, match="duplicate CI job assignments"):
        audit_files(sources, jobs, {"files": {}})


def test_covered_file_cannot_also_be_excluded():
    sources = {"tests/test_overlap.py": "def test_overlap(): pass\n"}
    jobs = (Selection("core", ("tests",), "not slow"),)
    with pytest.raises(ValueError, match="covered files conflict with exclusions"):
        audit_files(sources, jobs, {"files": {"tests/test_overlap.py": "historical reason"}})


def test_missing_excluded_file_fails():
    with pytest.raises(ValueError, match="stale or unexplained exclusion"):
        audit_files({}, (), {"files": {"tests/test_missing.py": "historical reason"}})


def test_explicit_ci_ignore_requires_a_current_reason_and_removes_assignment():
    sources = {"tests/test_archive.py": "def test_archive(): pass\n"}
    jobs = workflow_selections({"jobs": {"core": {"steps": [{
        "run": "python -m pytest tests --ignore tests/test_archive.py"}]}}})
    with pytest.raises(ValueError, match="undocumented CI exclusion"):
        audit_files(sources, jobs, {"files": {}})
    result = audit_files(sources, jobs, {"files": {"tests/test_archive.py": "requires archive"}})
    assert result["assignments"] == {"tests/test_archive.py": []}
    assert result["uncovered_files"] == []


def test_clean_source_precondition_reports_only_changed_bound_python(monkeypatch, tmp_path):
    outputs = iter(("src/solarlab/device/compile_2d.py\0src/solarlab/data.txt\0",
                    "src/solarlab/new.py\0"))
    monkeypatch.setattr("suite_support.subprocess.run", lambda *a, **kw:
                        SimpleNamespace(returncode=0, stdout=next(outputs)))
    paths = changed_bound_core_sources(tmp_path)
    assert paths == ("src/solarlab/device/compile_2d.py", "src/solarlab/new.py")
    reason = clean_core_skip_reason(paths)
    assert reason and all(path in reason for path in paths)
    assert clean_core_skip_reason(()) is None


def test_parity_budget_excludes_unrelated_time_and_keeps_aggregate_limit(monkeypatch):
    import sys

    source = ROOT / "tests/unit/assembly_parity/conftest.py"
    spec = importlib.util.spec_from_file_location("parity_budget_under_test", source)
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as patch:
        patch.setattr(sys, "path", list(sys.path))
        spec.loader.exec_module(module)
        ticks = iter((0.0, 2.0, 1000.0, 1003.0))
        patch.setattr(module, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
        budget = module.assembly_budget.__wrapped__()
        elapsed = next(budget)
        for _ in range(2):
            case = module.assembly_case_time.__wrapped__(elapsed)
            next(case)
            with pytest.raises(StopIteration):
                next(case)
        assert elapsed == [5.0]
        with pytest.raises(StopIteration):
            next(budget)
        rejected = module.assembly_budget.__wrapped__()
        next(rejected)[0] = 60.0
        with pytest.raises(AssertionError, match="exceeded 60 s"):
            next(rejected)
