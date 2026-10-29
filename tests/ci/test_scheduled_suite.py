"""Contracts for the separate scheduled workflow; production gates stay unchanged."""
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from scripts.ci import scheduled_suite as scheduled


def item(path, name="test_case", markers=()):
    return {"path": path, "nodeid": path + "::" + name, "markers": list(markers)}


def inventory():
    return {
        "root": {"items": [
            item("tests/ci/test_guard.py"),
            item("tests/unit/test_kernel.py"),
            item("tests/integration/test_package.py"),
            item("tests/runtime/test_pool.py"),
            item("tests/runtime/test_scan_execution.py"),
            item("tests/refactor/test_browser.py", markers=("frontend",)),
            item("tests/runtime/test_run_client_contract.py", markers=("frontend",)),
            item("tests/numerical_contracts/test_slow_case.py", markers=("slow",)),
        ]},
        "legacy": {"items": [
            item("tests/unit/test_one.py"),
            item("tests/unit/physics/test_two.py"),
            item("tests/regression/test_twod_validation.py", "test_a", ("slow",)),
            item("tests/regression/test_twod_validation.py", "test_b", ("slow",)),
            item("tests/regression/test_other.py", "test_a", ("slow",)),
            item("tests/regression/test_other.py", "test_b", ("slow",)),
        ]},
    }


def test_actual_push_partition_is_reused_without_losing_process_isolation():
    shards = scheduled.build_shards(inventory())
    rows = {row["id"]: row for row in shards["coverage"]}
    assert rows["coverage-root-backend"]["ignored"] == [
        "tests/runtime/test_scan_execution.py"
    ]
    assert rows["coverage-root-backend-scan_execution"]["count"] == 1
    assert rows["coverage-root-frontend-run_client"]["count"] == 1
    assert rows["coverage-root-frontend"]["ignored"] == [
        "tests/runtime/test_run_client_contract.py"
    ]
    assert sum(row["count"] for row in shards["coverage"]) == 9
    assert sum(row["count"] for row in shards["slow"]) == 5
    assert len(shards["slow"]) == 4  # Two expensive nodes; other file fixtures shared.
    assert Counter(row["suite"] for row in shards["slow"]) == {"root": 1, "legacy": 3}


def test_new_unowned_root_directory_fails_the_default_suite_plan():
    data = inventory()
    data["root"]["items"].append(item("tests/unowned/test_new.py"))
    with pytest.raises(ValueError, match="partition mismatch"):
        scheduled.build_shards(data)


def test_large_slow_file_is_partitioned_without_dropping_parameterized_nodes():
    data = {"legacy": {"items": [
        item("tests/integration/test_large.py", f"test_case[{i}]", ("slow",))
        for i in range(52)
    ]}}
    shards = scheduled.build_shards(data)["slow"]
    assert [row["count"] for row in shards] == [24, 24, 4]
    assert len({node for row in shards for node in row["nodeids"]}) == 52


def test_duplicate_coverage_is_not_silently_deduplicated():
    shard = {"id": "a", "suite": "root", "nodeids": ["test.py::test_x"]}
    with pytest.raises(ValueError, match="partition mismatch"):
        scheduled.verify_partition([shard, dict(shard, id="b")], [("root", "test.py::test_x")])


@pytest.mark.parametrize("requested", ["", "a,a", "missing", "a,", "a;echo hidden"])
def test_manual_shard_selection_rejects_invalid_or_unknown_ids(requested):
    with pytest.raises(ValueError):
        scheduled.choose([{"id": "a"}, {"id": "b"}], requested)


def test_manual_shard_selection_preserves_explicit_subset_and_all():
    rows = [{"id": "a"}, {"id": "b"}]
    assert scheduled.choose(rows, "b") == [rows[1]]
    assert scheduled.choose(rows, "all") == rows


def test_coverage_filters_current_source_and_installed_copy_not_foreign_producers(tmp_path):
    config = tmp_path / "coverage.ini"
    scheduled.coverage_config(config)
    text = config.read_text()
    assert "src/solarlab" in text and "perovskite-sim/perovskite_sim" in text
    assert "site-packages/solarlab" in text
    assert "patch = subprocess" in text
    assert "*" not in text
    assert "fail_under" not in text and "fail-under" not in text


@pytest.mark.parametrize(("expected", "body", "code"), [
    (["test_probe.py::absent"], "assert True", 4),
    (["test_probe.py::test_probe"], "assert False, 'failure must remain visible'", 1),
])
def test_scheduled_runner_refuses_drift_and_preserves_failures(tmp_path, expected, body, code):
    sentinel = tmp_path / "executed"
    (tmp_path / "test_probe.py").write_text(
        "from pathlib import Path\ndef test_probe():\n"
        f"    Path({str(sentinel)!r}).write_text('ran')\n    {body}\n"
    )
    expectation = tmp_path / "expected.json"
    expectation.write_text(json.dumps(expected))
    result = subprocess.run([
        sys.executable, str(Path(scheduled.__file__)), "_pytest",
        "--out", str(tmp_path / "actual.json"), "--expected", str(expectation),
        "--", "test_probe.py", "-q", "-p", "no:cacheprovider",
    ], cwd=tmp_path, text=True, capture_output=True, timeout=30)
    assert result.returncode == code, result.stdout + result.stderr
    assert sentinel.exists() is (code == 1)
    data = json.loads((tmp_path / "actual.json").read_text())
    assert data["exit_code"] == code
    if code == 1:
        assert any(row["outcome"] == "failed" for row in data["reports"])


def test_workflow_triggers_are_bounded_and_have_no_threshold_gate():
    path = scheduled.ROOT / ".github/workflows/scheduled-suite.yml"
    text = path.read_text()
    workflow = yaml.load(text, Loader=yaml.BaseLoader)
    assert set(workflow["on"]) == {"push", "schedule", "workflow_dispatch"}
    assert workflow["on"]["push"] == {
        "branches": ["devel"],
        "paths": [".github/workflows/scheduled-suite.yml", "scripts/ci/scheduled_suite.py"],
    }
    plan = workflow["jobs"]["plan"]["steps"]
    environment = next(step["env"] for step in plan if step.get("id") == "plan")
    for name in ("RUN_SLOW", "RUN_COVERAGE"):
        assert "github.event_name != 'workflow_dispatch'" in environment[name]
    assert workflow["concurrency"]["cancel-in-progress"] == "false"
    assert "run_id" in workflow["concurrency"]["group"]
    assert "fail-under" not in text
    for name in ("slow", "coverage"):
        assert workflow["jobs"][name]["strategy"]["fail-fast"] == "false"
        assert int(workflow["jobs"][name]["timeout-minutes"]) == 120
    for variable in scheduled.THREADS:
        assert workflow["env"][variable] == "1"


@pytest.mark.parametrize("stale", [False, True])
def test_installation_audit_refuses_stale_package_bytes(tmp_path, monkeypatch, stale):
    root = tmp_path / "checkout"
    source = root / "src/solarlab/__init__.py"
    source.parent.mkdir(parents=True)
    source.write_text("current_source = True\n")
    prefix = tmp_path / "environment"
    installed = prefix / "lib/solarlab/__init__.py"
    installed.parent.mkdir(parents=True)
    installed.write_text("old_source = True\n" if stale else source.read_text())
    with monkeypatch.context() as patch:
        patch.setattr(scheduled, "ROOT", root)
        patch.setattr(scheduled, "PACKAGE_ROOTS", {"solarlab": "src/solarlab"})
        patch.setattr(sys, "prefix", str(prefix))
        patch.setattr(scheduled.importlib.util, "find_spec", lambda _: SimpleNamespace(origin=str(installed)))
        patch.setattr(scheduled.subprocess, "check_output", lambda *a, **k: b"src/solarlab/__init__.py\0")
        if stale:
            with pytest.raises(ValueError, match="installed package differs from checkout"):
                scheduled.verify_installation()
        else:
            assert scheduled.verify_installation()["solarlab"]["files_verified"] == 1
