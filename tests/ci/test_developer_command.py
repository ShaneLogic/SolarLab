"""The developer green command must preserve CI selection and real failures."""
from dataclasses import asdict
import json
import os
import sys

import pytest
import yaml

from scripts import test_all
from .policy import Selection, workflow_selections


def test_developer_command_reuses_every_ci_selection_and_isolated_group():
    workflow = yaml.safe_load((test_all.ROOT / ".github/workflows/ci.yml").read_text())
    expected = workflow_selections(workflow)
    actual = test_all.selections(test_all.ROOT)
    assert [asdict(group) for group in actual] == [asdict(group) for group in expected]
    assert {group.isolated_group for group in actual} >= {
        "scan_execution", "supervisor_control_failure", "run_client"}
    assert all(path == "tests" or path.startswith("tests/") for group in actual for path in group.directories)


def test_environment_pins_threads_and_removes_inherited_selection(monkeypatch):
    for variable in test_all.THREADS:
        monkeypatch.setenv(variable, "8")
    for variable in ("PYTEST_CI_JOB", "PYTEST_CI_ISOLATED_GROUP", "PYTEST_ADDOPTS", "PYTHONPATH", "PYTHONHOME"):
        monkeypatch.setenv(variable, "foreign")
    isolated = test_all.environment(Selection("backend", ("tests/runtime/test_scan_execution.py",), "not slow",
                                               isolated_group="scan_execution"))
    assert isolated["PYTEST_CI_JOB"] == "backend"
    assert isolated["PYTEST_CI_ISOLATED_GROUP"] == "scan_execution"
    bulk = test_all.environment(Selection("packages", ("tests/integration",), "not slow"))
    assert bulk["PYTEST_CI_JOB"] == "packages"
    assert "PYTEST_CI_ISOLATED_GROUP" not in bulk
    assert all(bulk[name] == "1" for name in test_all.THREADS)
    assert not {"PYTEST_ADDOPTS", "PYTHONPATH", "PYTHONHOME"} & bulk.keys()


@pytest.mark.parametrize("bad_report", ["missing", "malformed"])
def test_missing_or_malformed_report_cannot_reuse_an_earlier_pass(tmp_path, monkeypatch, bad_report):
    report = tmp_path / "selection_1.xml"
    report.write_text('<testsuites><testsuite><testcase name="stale"/></testsuite></testsuites>')

    def process(*args, **kwargs):
        assert not report.exists()
        if bad_report == "malformed":
            report.write_text("truncated report")
        from types import SimpleNamespace
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(test_all.subprocess, "run", process)
    rows = test_all.run_groups(tmp_path, (Selection("core", ("tests",), "not slow"),), tmp_path)
    assert rows[0]["status"] == "failed"
    assert test_all.summarize(rows)["status"] == "failed"


def test_counts_keep_strict_xfail_and_subtests_separate(tmp_path):
    report = tmp_path / "report.xml"
    report.write_text('''<testsuites><testsuite tests="12">
        <testcase name="pass"/><testcase name="fail"><failure/></testcase>
        <testcase name="error"><error/></testcase>
        <testcase name="skip"><skipped type="pytest.skip"/></testcase>
        <testcase name="xfail"><skipped type="pytest.xfail"/></testcase>
        </testsuite></testsuites>''')
    assert test_all.result_counts(report) == {"passed": 1, "failed": 1, "errors": 1, "skipped": 1, "xfailed": 1}


@pytest.mark.parametrize("first_passes", [True, False])
def test_real_process_isolation_runs_all_groups_and_propagates_failure(tmp_path, monkeypatch, capsys, first_passes):
    tests = tmp_path / "tests"
    tests.mkdir()
    for name, passes in (("first", first_passes), ("second", True)):
        (tests / f"test_{name}.py").write_text(
            "import json, os\nfrom pathlib import Path\n"
            f"def test_{name}():\n"
            f"    Path({str(tmp_path / (name + '.json'))!r}).write_text(json.dumps(\n"
            "        [os.getpid(), os.getenv('PYTEST_CI_JOB'), os.getenv('PYTEST_CI_ISOLATED_GROUP')]))\n"
            f"    assert {passes!r}\n"
        )
    selections = (Selection("backend", ("tests/test_first.py",), "not slow", isolated_group="first"),
                  Selection("frontend", ("tests/test_second.py",), "not slow", isolated_group="second"))
    monkeypatch.setattr(test_all, "ROOT", tmp_path)
    monkeypatch.setattr(test_all, "selections", lambda root: selections)
    assert test_all.main(["--output-dir", str(tmp_path / "results")]) == (0 if first_passes else 1)
    records = [json.loads((tmp_path / (name + ".json")).read_text()) for name in ("first", "second")]
    assert records[0][0] != records[1][0] and all(record[0] != os.getpid() for record in records)
    assert [record[1:] for record in records] == [["backend", "first"], ["frontend", "second"]]
    summary = json.loads((tmp_path / "results/summary.json").read_text())
    assert [row["exit_code"] for row in summary["selections"]] == [0 if first_passes else 1, 0]
    assert all(row["command"][0] == sys.executable for row in summary["selections"])
    assert summary["counts"]["passed"] == (2 if first_passes else 1)
    assert summary["counts"]["failed"] == (0 if first_passes else 1)
    assert test_all.LEGACY in capsys.readouterr().out
