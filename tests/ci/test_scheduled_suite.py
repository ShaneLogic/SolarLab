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
            item("tests/runtime/test_supervisor_control_failure.py"),
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
        "tests/runtime/test_scan_execution.py",
        "tests/runtime/test_supervisor_control_failure.py",
    ]
    assert rows["coverage-root-backend-scan_execution"]["count"] == 1
    assert rows["coverage-root-backend-supervisor_control_failure"]["count"] == 1
    assert rows["coverage-root-frontend-run_client"]["count"] == 1
    assert rows["coverage-root-frontend"]["ignored"] == [
        "tests/runtime/test_run_client_contract.py"
    ]
    assert sum(row["count"] for row in shards["coverage"]) == 10
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


def test_coverage_watchdog_designation_names_only_dedicated_legacy_files():
    assert scheduled.LEGACY_COVERAGE_NODE_SECONDS == {
        "tests/integration/test_interface_flux_balance.py": 1200,
    }
    for path, seconds in scheduled.LEGACY_COVERAGE_NODE_SECONDS.items():
        assert (scheduled.SUITES["legacy"] / path).is_file()
        assert not (scheduled.SUITES["root"] / path).exists()
        assert Path(path).parent.as_posix() in scheduled.FILE_COVERAGE_SHARDS
        assert seconds == scheduled.slow_node_seconds("legacy", path)
        assert (3 + 1) * seconds == 80 * 60  # Three nodes and collection; step is 110 min.


def test_coverage_watchdog_preserves_every_other_plan_byte_and_all_node_ids(monkeypatch):
    path = "tests/integration/test_interface_flux_balance.py"
    data = inventory()
    data["legacy"]["items"].extend(item(path, f"test_{i}") for i in range(3))
    data["legacy"]["items"].append(item(path, "test_slow", ("slow",)))
    data["root"]["items"].append(item(path))  # A same-named root file is never designated.
    current = scheduled.build_shards(data)
    with monkeypatch.context() as patch:
        patch.setattr(scheduled, "LEGACY_COVERAGE_NODE_SECONDS", {})
        previous = scheduled.build_shards(data)
    designated = next(row for row in current["coverage"]
                      if row["id"] == "coverage-legacy-integration-interface_flux_balance")
    assert designated.pop("node_wall_seconds") == 1200
    assert json.dumps(current, sort_keys=True) == json.dumps(previous, sort_keys=True)


def test_coverage_watchdog_refuses_shared_files_and_growth_beyond_step_margin():
    path = "tests/integration/test_interface_flux_balance.py"
    with pytest.raises(ValueError, match="dedicated legacy file"):
        scheduled.make_shard("coverage", "legacy", [path, "tests/other.py"], "not slow", [item(path)])
    with pytest.raises(ValueError, match="exceed 90 minutes"):
        scheduled.make_shard("coverage", "legacy", [path], "not slow",
                             [item(path, f"test_{i}") for i in range(4)])


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
        nodeid = shard["nodeids"][0]
        reports = [{"nodeid": nodeid, "when": phase, "outcome": "passed"}
                   for phase in ("setup", "call", "teardown")]
        scheduled.write(out / "tests.json", {
            "items": [{"nodeid": nodeid}], "reports": reports, "status": "completed",
            "exit_code": 0, "collection_complete": True, "execution_complete": True,
            "recorder_errors": [], "started_nodeids": [nodeid],
        })
        (out / "tests.jsonl").write_text("".join(
            json.dumps(row) + "\n" for row in [
                {"event": "start", "nodeid": nodeid},
                *({"event": "report", **report} for report in reports),
            ]
        ))
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
    commands = []

    def interrupted_receipt(command, **kwargs):
        commands.append(command)
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
    # The designated coverage stack dump never reaches slow shards.
    assert len(commands) == 3 and not any("faulthandler" in part for c in commands for part in c)
    assert "stack_dump_seconds" not in result["watchdog"]
    timeout = next(r for r in data["reports"] if r["outcome"] == "timeout")
    assert timeout["wasxfail"] is None and timeout["when"] == "watchdog"
    xml = ET.parse(out / "junit.xml")
    assert len(xml.findall(".//testcase")) == 2
    assert len(xml.findall(".//failure[@type='timeout']")) == 1
    assert not xml.findall(".//skipped")


@pytest.mark.parametrize("interruption", ["timeout", "hard_exit"])
def test_designated_coverage_watchdog_closes_timeout_and_keeps_later_subprocess_data(
    tmp_path, monkeypatch, interruption,
):
    from coverage import CoverageData

    path = "tests/integration/test_interface_flux_balance.py"
    suite = tmp_path / "perovskite-sim"
    test = suite / path
    test.parent.mkdir(parents=True)
    (suite / "pytest.ini").write_text("[pytest]\n")
    probe = tmp_path / "src/solarlab/probe.py"
    probe.parent.mkdir(parents=True)
    probe.write_text("def parent():\n    return 11\ndef child():\n    return 23\n"
                     "if __name__ == '__main__':\n    assert child() == 23\n")
    stop = "time.sleep(600)" if interruption == "timeout" else "os._exit(0)"
    test.write_text(
        "import os, runpy, subprocess, sys, time, pytest\n"
        "@pytest.fixture(scope='module')\ndef blocked():\n"
        f"    {stop}\n"
        "def test_first(blocked):\n    pytest.fail('blocked setup returned')\n"
        "def test_second():\n"
        f"    assert runpy.run_path({str(probe)!r})['parent']() == 11\n"
        f"    subprocess.run([sys.executable, {str(probe)!r}], check=True)\n"
    )
    shard = scheduled.build_shards({"legacy": {"items": [
        item(path, "test_first"), item(path, "test_second"),
    ]}})["coverage"][0]
    assert shard["node_wall_seconds"] == 1200
    # Local synthetic deadline, never the production table. The passing node
    # took up to 4.7 s on an oversubscribed host, so 30 s keeps a wide margin.
    shard["node_wall_seconds"] = 30
    monkeypatch.setattr(scheduled, "ROOT", tmp_path)
    monkeypatch.setattr(scheduled, "SUITES", {"legacy": suite})
    monkeypatch.setattr(scheduled, "commit", lambda: "frozen")
    monkeypatch.setattr(scheduled, "verify_installation", lambda: {"synthetic": {"files_verified": 1}})
    (tmp_path / "uv.lock").write_text("locked")
    plan = tmp_path / "plan.json"
    scheduled.write(plan, {"commit": "frozen", "lock_sha256": scheduled.sha256(tmp_path / "uv.lock"),
                           "selected_shards": {"coverage": [shard]}})
    out = tmp_path / "out"
    assert scheduled.run_shard(SimpleNamespace(plan=plan, kind="coverage", shard=shard["id"], out=out)) == 1
    result = scheduled.read(out / "result.json")
    assert result["status"] == "failed" and result["exit_code"] == 1
    assert result["receipt_complete"] is (interruption == "timeout")
    assert result["not_started_nodeids"] == []
    assert result["incomplete_nodeids"] == ([] if interruption == "timeout" else [shard["nodeids"][0]])
    assert result["coverage_data"] and result["subprocess_coverage"]
    assert "patch = subprocess" in (out / "coverage.ini").read_text()
    data = scheduled.read(out / "tests.json")
    first = "timeout" if interruption == "timeout" else "failed"
    assert data["status"] == "completed" and data["exit_code"] == 1
    assert [(row["nodeid"], row["outcome"]) for row in data["node_outcomes"]] == [
        (shard["nodeids"][0], first), (shard["nodeids"][1], "passed"),
    ]
    assert any(row["when"] == "watchdog" and row["outcome"] == first for row in data["reports"])
    journal = [json.loads(line) for line in (out / "tests.jsonl").read_text().splitlines()]
    assert [row["nodeid"] for row in journal if row["event"] == "start"] == shard["nodeids"]
    assert [row["outcome"] for row in journal if row["event"] == "outcome"] == [first, "passed"]
    xml = ET.parse(out / "junit.xml")
    assert len(xml.findall(".//testcase")) == 2
    assert len(xml.findall(f".//failure[@type='{first}']")) == 1
    assert not xml.findall(".//skipped")
    parent_data, child_data = set(), set()
    for filename in out.glob("coverage.data.*"):
        coverage = CoverageData(basename=str(filename))
        coverage.read()
        lines = coverage.lines(str(probe)) or []
        if 2 in lines:
            parent_data.add(filename)
        if 4 in lines:
            child_data.add(filename)
    assert parent_data and child_data and parent_data.isdisjoint(child_data)
    assert scheduled.read(out / "nodes/0001/tests.json")["execution_complete"]
    assert all(not p.read_text() for p in out.rglob("*.recorder-errors.jsonl"))
    assert result["watchdog"]["stack_dump_seconds"] == 15
    if interruption == "timeout":
        dump = (out / "nodes/0000/pytest.log").read_text()
        assert "Timeout (0:00:15)!" in dump and "in blocked" in dump


def _watchdog_receipt_tree(out, rows, items=None, child_cases=1):
    """Write the files run_slow_nodes leaves for a timeout and a completed node."""
    suites = ET.Element("testsuites")
    aggregate = {"items": [{"nodeid": nodeid} for nodeid in items or [r[0] for r in rows]],
                 "collection_reports": [], "reports": [], "node_outcomes": [],
                 "status": "completed", "exit_code": 1, "seconds": 1.0}
    journal = []
    for index, (nodeid, outcome) in enumerate(rows):
        node = out / "nodes" / f"{index:04d}"
        node.mkdir(parents=True)
        timeout = outcome == "timeout"
        row = {"nodeid": nodeid, "outcome": outcome, "receipt_errors": [],
               "returncode": -15 if timeout else int(outcome == "failed"),
               "timeout": timeout, "seconds": 2.0}
        if timeout:
            aggregate["reports"].append(scheduled.watchdog_failure(
                suites, nodeid, "timeout", 2.0, "Scheduled node exceeded its wall cap"))
        else:
            reports = [{"nodeid": nodeid, "when": phase,
                        "outcome": "failed" if outcome == "failed" and phase != "setup" else "passed"}
                       for phase in ("setup", "call", "teardown")]
            scheduled.write(node / "tests.json", {
                "items": [{"nodeid": nodeid}], "reports": reports, "status": "completed",
                "exit_code": row["returncode"], "collection_complete": True,
                "execution_complete": True, "recorder_errors": [], "started_nodeids": [nodeid]})
            (node / "tests.jsonl").write_text("".join(
                json.dumps({"event": "report", **report}) + "\n" for report in reports))
            child = ET.Element("testsuite", name="pytest")
            for _ in range(child_cases):
                ET.SubElement(child, "testcase", classname=nodeid.partition("::")[0],
                              name=nodeid.partition("::")[2])
            ET.ElementTree(child).write(node / "junit.xml")
            suites.append(child)
            aggregate["reports"].extend(reports)
        aggregate["node_outcomes"].append(row)
        journal += [{"event": "start", "nodeid": nodeid}, {"event": "outcome", **row}]
    return aggregate, journal, suites


def _write_watchdog_receipt_tree(out, aggregate, journal, suites):
    scheduled.write(out / "tests.json", aggregate)
    (out / "tests.jsonl").write_text("".join(json.dumps(row) + "\n" for row in journal))
    ET.ElementTree(suites).write(out / "junit.xml")


@pytest.mark.parametrize("corruption", [
    None, "teardown-error-case", "function-count", "status", "exit-code", "passing-timeout",
    "outcome-order", "duplicate-start", "journal-outcome", "extra-testcase",
    "duplicate-timeout-label", "unlabelled-timeout", "mislabelled-timeout", "empty-child-junit",
    "missing-child-junit", "unparseable-child-junit", "child-receipt-errors", "child-missing-phase",
    "never-started",
])
def test_watchdog_coverage_receipt_rejects_false_completeness(tmp_path, corruption):
    first = "tests/integration/test_interface_flux_balance.py::test_first"
    second = "tests/integration/test_interface_flux_balance.py::test_second"
    rows = [(first, "timeout"), (second, "failed" if corruption == "teardown-error-case" else "passed")]
    items, expected, exit_code = None, [first, second], 1
    if corruption == "outcome-order":
        rows, items = rows[::-1], [first, second]
    if corruption == "never-started":
        expected = items = [first, second, second + "_late"]
    cases = {"teardown-error-case": 2, "empty-child-junit": 0}.get(corruption, 1)
    aggregate, journal, suites = _watchdog_receipt_tree(tmp_path, rows, items, child_cases=cases)
    outcome_rows = [row for row in journal if row["event"] == "outcome"]
    if corruption == "function-count":
        expected = [first]
    elif corruption == "status":
        aggregate["status"] = "running"
    elif corruption == "exit-code":
        exit_code = 2
    elif corruption == "passing-timeout":
        aggregate["exit_code"] = exit_code = 0
    elif corruption == "duplicate-start":
        journal.insert(0, {"event": "start", "nodeid": first})
    elif corruption == "journal-outcome":
        outcome_rows[1]["seconds"] = 3.0
    elif corruption == "extra-testcase":
        ET.SubElement(suites[0], "testcase", classname="extra", name="case")
    elif corruption == "duplicate-timeout-label":
        aggregate["reports"].append(dict(aggregate["reports"][0]))
    elif corruption == "unlabelled-timeout":
        aggregate["reports"] = [r for r in aggregate["reports"] if r["when"] != "watchdog"]
    elif corruption == "mislabelled-timeout":
        aggregate["reports"][0]["outcome"] = "failed"
    elif corruption == "missing-child-junit":
        (tmp_path / "nodes/0001/junit.xml").unlink()
    elif corruption == "unparseable-child-junit":
        (tmp_path / "nodes/0001/junit.xml").write_text("<testsuite><testcase")
    elif corruption == "child-receipt-errors":
        aggregate["node_outcomes"][1]["receipt_errors"] = outcome_rows[1]["receipt_errors"] = ["torn"]
    elif corruption == "child-missing-phase":
        child = scheduled.read(tmp_path / "nodes/0001/tests.json")
        child["reports"].pop()
        scheduled.write(tmp_path / "nodes/0001/tests.json", child)
        (tmp_path / "nodes/0001/tests.jsonl").write_text("")
    _write_watchdog_receipt_tree(tmp_path, aggregate, journal, suites)
    result = scheduled._watchdog_coverage_receipt(tmp_path, expected, exit_code)
    if corruption in (None, "teardown-error-case"):
        # A failed call plus a teardown error yields two testcases for one node.
        assert result["receipt_complete"] and result["receipt_errors"] == []
    else:
        assert not result["receipt_complete"] and result["receipt_errors"]
    if corruption == "never-started":
        assert result["not_started_nodeids"] == [second + "_late"]


def test_incomplete_watchdog_coverage_receipt_fails_a_shard_whose_runner_exited_zero(
    tmp_path, monkeypatch,
):
    path = "tests/integration/test_interface_flux_balance.py"
    shard = scheduled.build_shards({"legacy": {"items": [item(path)]}})["coverage"][0]
    monkeypatch.setattr(scheduled, "ROOT", tmp_path)
    monkeypatch.setattr(scheduled, "SUITES", {"legacy": tmp_path})
    monkeypatch.setattr(scheduled, "commit", lambda: "frozen")
    monkeypatch.setattr(scheduled, "verify_installation", lambda: {"synthetic": {"files_verified": 1}})
    monkeypatch.setattr(scheduled, "run_slow_nodes", lambda *args, **kwargs: 0)
    (tmp_path / "uv.lock").write_text("locked")
    plan = tmp_path / "plan.json"
    scheduled.write(plan, {"commit": "frozen", "lock_sha256": scheduled.sha256(tmp_path / "uv.lock"),
                           "selected_shards": {"coverage": [shard]}})
    out = tmp_path / "out"
    (out).mkdir()
    (out / "pytest.log").write_text("")
    assert scheduled.run_shard(SimpleNamespace(plan=plan, kind="coverage", shard=shard["id"], out=out)) == 1
    result = scheduled.read(out / "result.json")
    assert result["status"] == "failed" and not result["receipt_complete"]
    assert result["not_started_nodeids"] == shard["nodeids"]


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
    aggregate = scheduled.read(out / "tests.json")
    assert aggregate["status"] == "incomplete" and aggregate["exit_code"] == 4
    assert not aggregate["collection_complete"] and not aggregate["execution_complete"]
    assert aggregate["not_started_nodeids"] == ["test_probe.py::absent"]


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


def test_coverage_shards_upload_runtime_evidence_for_success_and_failure():
    workflow = yaml.load(
        (scheduled.ROOT / ".github/workflows/scheduled-suite.yml").read_text(), Loader=yaml.BaseLoader,
    )
    steps = workflow["jobs"]["coverage"]["steps"]
    run = next(step for step in steps if "--kind coverage" in step.get("run", ""))
    upload = next(step for step in steps if step.get("uses", "").startswith("actions/upload-artifact@"))
    uploaded = upload["with"]["path"].rstrip("/")
    assert uploaded == "${{ runner.temp }}/coverage-result"
    assert '--out "$RUNNER_TEMP/coverage-result"' in run["run"]
    assert run["env"]["SCHEDULED_RUNTIME_EVIDENCE_DIR"] == (
        "${{ startsWith(matrix.id, 'coverage-root-backend') && "
        "format('{0}/coverage-result/runtime-evidence/{1}', runner.temp, matrix.id) || '' }}"
    )
    assert "if" not in run and upload["if"] == "always()"
    assert upload["with"]["name"] == "coverage-data-${{ matrix.id }}"


def test_runtime_evidence_is_not_a_node_receipt_or_coverage_shard(tmp_path, monkeypatch):
    from coverage import CoverageData

    nodeid = "test_probe.py::test_probe"
    shard = {"id": "probe", "count": 1, "nodeids": [nodeid]}
    plan = tmp_path / "plan.json"
    scheduled.write(plan, {"commit": "frozen", "selected_shards": {"coverage": [shard]},
                           "all_shards": {"coverage": [shard]}})
    directory = tmp_path / "coverage-shards/coverage-data-probe"
    reports = [{"nodeid": nodeid, "when": phase, "outcome": "passed"}
               for phase in ("setup", "call", "teardown")]
    scheduled.write(directory / "tests.json", {
        "items": [{"nodeid": nodeid}], "reports": reports, "status": "completed", "exit_code": 0,
        "collection_complete": True, "execution_complete": True,
        "recorder_errors": [], "started_nodeids": [nodeid],
    })
    (directory / "tests.jsonl").write_text("".join(
        json.dumps({"event": "report", **row}) + "\n" for row in reports
    ))
    receipt = scheduled._coverage_receipt(directory, [nodeid], 0)
    assert receipt["receipt_complete"]
    result = {"shard": shard, "commit": "frozen", "status": "passed", "coverage_data": True,
              "plan_sha256": scheduled.sha256(plan)}
    scheduled.write(directory / "result.json", result)
    evidence = directory / "runtime-evidence/probe"
    labels = ["malformed-not_mapping", "malformed-missing_name", "guard-cancel",
              "guard-publication-error", "guard-publication-wall", "guard-missing-counter"]
    for label in labels:
        scheduled.write(evidence / (label + ".json"), {
            "workers": {"attempt": {"receipt.json": {"content": {"store_state": "failed"}},
                                    "fixture-clock.json": {"content": {"elapsed_offset_seconds": 2.0}}}},
            "resource_samples": [], "events": [],
        })
    retained = {path: path.read_bytes() for path in evidence.glob("*.json")}
    assert scheduled._coverage_receipt(directory, [nodeid], 0) == receipt

    source = tmp_path / "probe.py"
    source.write_text("value = 1\n")
    data = CoverageData(basename=str(directory / "coverage.data.probe"))
    data.add_lines({str(source): {1}})
    data.write()
    monkeypatch.setattr(scheduled, "commit", lambda: "frozen")
    monkeypatch.setattr(scheduled, "verify_installation", lambda: {})
    monkeypatch.setattr(scheduled, "coverage_config", lambda path: path.write_text("[run]\nparallel = true\n"))
    output = tmp_path / "coverage-report"
    scheduled.combine(SimpleNamespace(plan=plan, data=directory.parent, out=output))
    scope = scheduled.read(output / "scope.json")
    assert scope["full_default_run_passed"] is True
    assert scope["shard_results"] == {"probe": result}
    assert scheduled.read(directory / "tests.json")["items"] == [{"nodeid": nodeid}]
    assert scheduled.read(output / "coverage.json")["totals"]["covered_lines"] == 1
    assert {path: path.read_bytes() for path in evidence.glob("*.json")} == retained


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
