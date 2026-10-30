"""Contracts for the separate scheduled workflow; production gates stay unchanged."""
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

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


def test_clock_bound_files_run_once_in_explicit_parent_only_coverage_shards():
    data = inventory()
    for path in scheduled.PARENT_ONLY_COVERAGE:
        data["root"]["items"].append(item(path))
    rows = {row["id"]: row for row in scheduled.build_shards(data)["coverage"]}
    for job, file in [("core", "test_g2_route_launcher.py"),
                      ("backend", "test_ionic_jv_job_service.py")]:
        isolated = next(row for row in rows.values() if row["paths"] == [next(
            path for path in scheduled.PARENT_ONLY_COVERAGE if path.endswith(file)
        )])
        assert isolated["ci_job"] == job and isolated["subprocess_coverage"] is False
        assert isolated["coverage_limitation"]
        assert isolated["paths"][0] in rows["coverage-root-" + job]["ignored"]
        assert sum(isolated["nodeids"][0] in row["nodeids"] for row in rows.values()) == 1
    assert rows["coverage-root-backend-scan_execution"].get("subprocess_coverage", True)


def test_large_slow_function_keeps_all_parameterized_cases_together():
    data = {"legacy": {"items": [
        item("tests/integration/test_large.py", f"test_case[{i}]", ("slow",))
        for i in range(52)
    ]}}
    shards = scheduled.build_shards(data)["slow"]
    assert [row["count"] for row in shards] == [52]
    assert shards[0]["paths"] == ["tests/integration/test_large.py::test_case"]
    assert shards[0]["function_counts"] == {"tests/integration/test_large.py::test_case": 52}
    assert len({node for row in shards for node in row["nodeids"]}) == 52


def test_large_slow_file_splits_only_between_functions():
    data = {"legacy": {"items": [
        item("tests/integration/test_large.py", f"test_{name}[{i}]", ("slow",))
        for name, count in (("a", 20), ("b", 20), ("c", 12)) for i in range(count)
    ]}}
    shards = scheduled.build_shards(data)["slow"]
    assert [row["count"] for row in shards] == [20, 20, 12]
    assert all(len(row["paths"]) == 1 and "[" not in row["paths"][0] for row in shards)


def test_float_repr_drift_does_not_change_function_selectors_or_shard_ids():
    path = "tests/integration/test_physical_interface_cbo_scan.py"
    plans = []
    for value in ("-0.04056782763682157", "-0.040567827636821575"):
        plans.append(scheduled.build_shards({"legacy": {"items": [
            item(path, f"TestProbe::test_case[{value}]", ("slow",)),
            item(path, "TestProbe::test_case[1.0]", ("slow",)),
        ]}})["slow"])
    left, right = plans[0][0], plans[1][0]
    assert len(plans[0]) == len(plans[1]) == 1
    assert left["id"] == right["id"]
    assert left["paths"] == right["paths"] == [path + "::TestProbe::test_case"]
    assert left["function_counts"] == right["function_counts"] == {path + "::TestProbe::test_case": 2}
    assert left["nodeids"] != right["nodeids"]  # Raw observations remain inspectable.


def test_integration_coverage_retains_each_file_once_in_its_own_shard():
    data = {"legacy": {"items": [
        item("tests/integration/test_first.py", f"test_case[{i}]") for i in range(3)
    ] + [item("tests/integration/test_second.py"), item("tests/unit/test_small.py")]}}
    rows = scheduled.build_shards(data)["coverage"]
    integration = [row for row in rows if row["id"].startswith("coverage-legacy-integration-")]
    assert len(integration) == 2
    assert sorted(row["count"] for row in integration) == [1, 3]
    assert all(len(row["paths"]) == 1 for row in integration)
    assert sum(row["count"] for row in rows) == 5


def test_cigs_slow_file_is_split_by_function_with_parameters_kept_together():
    path = "tests/regression/test_grading_cigs_notch.py"
    data = {"legacy": {"items": [
        item(path, f"test_thickness[{v}]", ("slow",)) for v in (0.5, 1.0, 2.0)
    ] + [item(path, "test_production", ("slow",)), item(path, "test_observations", ("slow",))]}}
    rows = scheduled.build_shards(data)["slow"]
    assert sorted(row["count"] for row in rows) == [1, 1, 3]
    assert all(len(row["paths"]) == 1 and "[" not in row["paths"][0] for row in rows)


def test_function_cannot_span_shards_even_when_raw_node_partition_is_exact():
    nodes = ["test.py::test_x[0.1]", "test.py::test_x[0.2]"]
    shards = [{"id": str(i), "suite": "root", "nodeids": [node]} for i, node in enumerate(nodes)]
    with pytest.raises(ValueError, match="test function split across shards"):
        scheduled.verify_partition(shards, [("root", node) for node in nodes])


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


def test_parent_only_coverage_keeps_parent_trace_and_leaves_children_uninstrumented(tmp_path):
    config = tmp_path / "coverage.ini"
    scheduled.coverage_config(config, subprocesses=False)
    assert "patch = subprocess" not in config.read_text()
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import coverage, subprocess, sys\n"
        "assert coverage.Coverage.current() is not None\n"
        "subprocess.run([sys.executable, '-c', "
        "'import coverage; assert coverage.Coverage.current() is None'], check=True)\n"
    )
    env = scheduled.pinned_environment()
    env["COVERAGE_FILE"] = str(tmp_path / "coverage.data")
    result = subprocess.run([sys.executable, "-m", "coverage", "run", "--rcfile=" + str(config), str(probe)],
                            env=env, cwd=tmp_path, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(("kind", "job", "source_bound"), [
    ("coverage", "core", True),
    ("coverage", "backend", False),
    ("slow", None, False),
])
def test_core_source_binding_follows_installation_audit_only_in_its_child(
    tmp_path, monkeypatch, kind, job, source_bound,
):
    monkeypatch.setattr(scheduled, "ROOT", tmp_path)
    monkeypatch.setattr(scheduled, "SUITES", {"root": tmp_path})
    monkeypatch.setattr(scheduled, "commit", lambda: "frozen")
    monkeypatch.setenv("PYTHONPATH", "/foreign/source")
    monkeypatch.delenv("PYTEST_CI_JOB", raising=False)
    (tmp_path / "uv.lock").write_text("locked")
    shard = {"id": "case", "suite": "root", "paths": ["tests"],
             "markers": "not slow", "nodeids": ["tests/test_one.py::test_one"],
             "ignored": [], "ci_job": job}
    plan = tmp_path / "plan.json"
    scheduled.write(plan, {"commit": "frozen", "lock_sha256": scheduled.sha256(tmp_path / "uv.lock"),
                           "selected_shards": {kind: [shard]}})
    audited = []

    def audit():
        # Source binding must not make the installation guard audit itself.
        assert scheduled.os.environ["PYTHONPATH"] == "/foreign/source"
        audited.append(True)
        return {"solarlab": {"files_verified": 1}}

    def execute(command, **kwargs):
        assert audited == [True]
        assert kwargs["env"].get("PYTHONPATH") == (
            str(tmp_path / "src") if source_bound else None
        )
        assert kwargs["env"].get("PYTEST_CI_JOB") == job
        assert ("coverage" in command) is (kind == "coverage")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(scheduled, "verify_installation", audit)
    monkeypatch.setattr(scheduled.subprocess, "run", execute)

    def slow(_shard, directory, environment, _result):
        (directory / "pytest.log").write_text("")
        return execute([sys.executable], env=environment).returncode

    monkeypatch.setattr(scheduled, "run_slow_nodes", slow)
    out = tmp_path / "out"
    assert scheduled.run_shard(SimpleNamespace(plan=plan, kind=kind, shard="case", out=out)) == 0
    result = scheduled.read(out / "result.json")
    assert result["installed_source_verification"] == {"solarlab": {"files_verified": 1}}
    assert result.get("execution_source") == (str(tmp_path / "src") if source_bound else None)


def test_source_bound_coverage_never_runs_after_installation_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduled, "ROOT", tmp_path)
    monkeypatch.setattr(scheduled, "commit", lambda: "frozen")
    (tmp_path / "uv.lock").write_text("locked")
    plan = tmp_path / "plan.json"
    scheduled.write(plan, {"commit": "frozen", "lock_sha256": scheduled.sha256(tmp_path / "uv.lock"),
                           "selected_shards": {"coverage": [{"id": "case", "ci_job": "core", "nodeids": []}]}})

    def stale():
        raise ValueError("installed package differs from checkout")

    monkeypatch.setattr(scheduled, "verify_installation", stale)
    monkeypatch.setattr(scheduled.subprocess, "run", lambda *a, **k: pytest.fail("stale package executed"))
    with pytest.raises(ValueError, match="installed package differs"):
        scheduled.run_shard(SimpleNamespace(plan=plan, kind="coverage", shard="case", out=tmp_path / "out"))


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


@pytest.mark.parametrize(("expected_counts", "code"), [
    ((2, 1), 0), ((1, 1), 4), ((3, 1), 4), ((1, 2), 4),
])
def test_runner_accepts_repr_drift_but_requires_exact_counts_for_each_function(
    tmp_path, expected_counts, code,
):
    sentinel = tmp_path / "executed"
    (tmp_path / "test_probe.py").write_text(
        "import pytest\nfrom pathlib import Path\n"
        "@pytest.mark.parametrize('x', [1, 2], ids=['-0.040567827636821575', 'other'])\n"
        "def test_probe(x):\n"
        f"    with Path({str(sentinel)!r}).open('a') as out: out.write(str(x))\n"
        "def test_other():\n"
        f"    with Path({str(sentinel)!r}).open('a') as out: out.write('3')\n"
    )
    expected = [f"test_probe.py::{name}[plan-{i}]"
                for name, count in zip(("test_probe", "test_other"), expected_counts)
                for i in range(count)]
    expected[0] = "test_probe.py::test_probe[-0.04056782763682157]"
    expectation = tmp_path / "expected.json"
    expectation.write_text(json.dumps(expected))
    result = subprocess.run([
        sys.executable, str(Path(scheduled.__file__)), "_pytest",
        "--out", str(tmp_path / "actual.json"), "--expected", str(expectation),
        "--", "test_probe.py", "-q", "-p", "no:cacheprovider",
    ], cwd=tmp_path, text=True, capture_output=True, timeout=30)
    assert result.returncode == code, result.stdout + result.stderr
    if code == 0:
        assert sentinel.read_text() == "123"
    else:
        assert "scheduled selection drift" in result.stderr
        assert not sentinel.exists()  # No body runs after even a per-function count mismatch.


def test_partial_receipts_preserve_named_failures_and_the_next_started_case(tmp_path):
    output = tmp_path / "actual.json"
    journal = tmp_path / "actual.jsonl"
    (tmp_path / "test_probe.py").write_text(
        "import json\nfrom pathlib import Path\n"
        "def test_first():\n    assert False, 'preserve this failure before a later timeout'\n"
        "def test_second():\n"
        f"    current = json.loads(Path({str(output)!r}).read_text())\n"
        "    assert current['status'] == 'running' and 'exit_code' not in current\n"
        f"    rows = [json.loads(line) for line in Path({str(journal)!r}).read_text().splitlines()]\n"
        "    failed = next(row for row in rows if row.get('outcome') == 'failed')\n"
        "    assert failed['nodeid'] == 'test_probe.py::test_first'\n"
        "    assert 'preserve this failure' in failed['longrepr']\n"
        "    starts = [row['nodeid'] for row in rows if row['event'] == 'start']\n"
        "    assert starts == ['test_probe.py::test_first', 'test_probe.py::test_second']\n"
    )
    result = subprocess.run([
        sys.executable, str(Path(scheduled.__file__)), "_pytest", "--out", str(output),
        "--", "test_probe.py", "-q", "-p", "no:cacheprovider",
    ], cwd=tmp_path, text=True, capture_output=True, timeout=30)
    assert result.returncode == 1, result.stdout + result.stderr
    data = json.loads(output.read_text())
    assert data["status"] == "completed" and data["exit_code"] == 1
    assert [(row["nodeid"], row["outcome"]) for row in data["reports"] if row["when"] == "call"] == [
        ("test_probe.py::test_first", "failed"), ("test_probe.py::test_second", "passed"),
    ]


def test_slow_watchdog_caps_use_measured_costs_without_changing_test_gates():
    assert scheduled.slow_node_seconds("legacy", "tests/regression/test_grading_cigs_notch.py") == 1200
    assert scheduled.slow_node_seconds("legacy", "tests/integration/test_physical_interface_cbo_scan.py") == 3600
    assert scheduled.slow_node_seconds("legacy", "tests/regression/test_interface_balance_ridge.py") == 2820
    assert scheduled.slow_node_seconds("root", "tests/test_new.py") == 1200
    assert all(1200 <= scheduled.slow_node_seconds("legacy", path) <= 3600
               for path in scheduled.MEASURED_SLOW_NODE_SECONDS)


def test_slow_watchdog_records_timeout_failure_and_continues_to_next_case(tmp_path, monkeypatch):
    (tmp_path / "pytest.ini").write_text("[pytest]\nmarkers = slow: watchdog probe\n")
    (tmp_path / "test_probe.py").write_text(
        "import os, time, pytest\npytestmark = pytest.mark.slow\n"
        "def test_first():\n    time.sleep(30)\n"
        "@pytest.mark.parametrize('x', [1], ids=['-0.040567827636821575'])\n"
        "def test_second(x):\n    assert os.environ['OMP_NUM_THREADS'] == '1'\n"
    )
    nodes = ["test_probe.py::test_first", "test_probe.py::test_second[-0.04056782763682157]"]
    local_second = "test_probe.py::test_second[-0.040567827636821575]"
    out = tmp_path / "out"
    out.mkdir()
    scheduled.write(out / "expected.json", nodes)
    shard = {"id": "probe", "suite": "legacy", "paths": ["test_probe.py"],
             "markers": "slow", "ignored": [], "node_wall_seconds": 5}
    monkeypatch.setattr(scheduled, "SUITES", {"legacy": tmp_path})
    original_watchdog = scheduled.run_with_watchdog

    def interrupted_receipt(command, **kwargs):
        value = original_watchdog(command, **kwargs)
        if value["timeout"]:
            with (Path(kwargs["log"]).parent / "tests.jsonl").open("a") as stream:
                stream.write('{"interrupted":')
        return value

    monkeypatch.setattr(scheduled, "run_with_watchdog", interrupted_receipt)
    result = {"status": "started"}
    code = scheduled.run_slow_nodes(shard, out, scheduled.pinned_environment(), result)
    assert code == 1
    data = scheduled.read(out / "tests.json")
    assert [(r["nodeid"], r["outcome"]) for r in data["node_outcomes"]] == [
        (nodes[0], "timeout"), (local_second, "passed"),
    ]
    assert len(data["node_outcomes"][0]["receipt_errors"]) == 1
    assert data["status"] == "completed" and data["exit_code"] == 1
    timeout = next(r for r in data["reports"] if r["outcome"] == "timeout")
    assert timeout["wasxfail"] is None and timeout["when"] == "watchdog"
    xml = ET.parse(out / "junit.xml")
    assert len(xml.findall(".//testcase")) == 2
    assert len(xml.findall(".//failure[@type='timeout']")) == 1
    assert not xml.findall(".//skipped")


def test_slow_preflight_refuses_count_drift_before_any_node_body(tmp_path, monkeypatch):
    sentinel = tmp_path / "executed"
    (tmp_path / "test_probe.py").write_text(
        "from pathlib import Path\ndef test_probe():\n"
        f"    Path({str(sentinel)!r}).write_text('ran')\n"
    )
    out = tmp_path / "out"
    out.mkdir()
    scheduled.write(out / "expected.json", ["test_probe.py::absent"])
    shard = {"id": "probe", "suite": "legacy", "paths": ["test_probe.py"],
             "markers": "not slow", "ignored": [], "node_wall_seconds": 10}
    monkeypatch.setattr(scheduled, "SUITES", {"legacy": tmp_path})
    assert scheduled.run_slow_nodes(shard, out, scheduled.pinned_environment(), {}) == 4
    assert not sentinel.exists()
    assert "scheduled selection drift" in (out / "pytest.log").read_text()


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
