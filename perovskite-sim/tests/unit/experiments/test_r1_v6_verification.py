"""Evidence accounting must not turn partial or differently sourced runs green."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


def load_script(name):
    path = Path(__file__).resolve().parents[3] / "scripts" / name
    spec = spec_from_file_location(name.replace(".py", ""), path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verification = load_script("run_r1_v6_verification.py")
preparation = load_script("prepare_r1_v6_evidence.py")


def test_snapshot_reads_root_relative_paths_and_rejects_later_dirty_source(tmp_path):
    project = tmp_path / "perovskite-sim"
    project.mkdir()
    (project / "solver.py").write_text("value = 1\n")
    (tmp_path / "README.md").write_text("tracked root document\n")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-qm", "fixture"], cwd=tmp_path, check=True)
    result = verification.snapshot(project)
    assert set(result["file_hashes"]) == {"README.md", "perovskite-sim/solver.py"}
    (project / "solver.py").write_text("value = 2\n")
    with pytest.raises(ValueError, match="clean frozen"):
        verification.snapshot(project, result["source_commit"])


def test_passed_call_without_teardown_remains_incomplete():
    assert verification.phase_status({"call": {"outcome": "passed"}}) == "incomplete"
    assert verification.phase_status({}) == "not_run"
    assert verification.phase_status({"call": {"outcome": "passed"},
                                      "teardown": {"outcome": "failed"}}) == "error"


def test_junit_preserves_parameterized_raw_ids_and_accounts_for_teardown_error(tmp_path):
    xml = tmp_path / "results.xml"
    xml.write_text('''<testsuites><testsuite>
      <testcase classname="tests.test_one.Test" name="test_value[x.y]">
       <properties><property name="raw_nodeid" value="tests/test_one.py::Test::test_value[x.y]"/></properties>
      </testcase>
      <testcase classname="tests.test_one.Test" name="test_value[x.y]">
       <properties><property name="raw_nodeid" value="tests/test_one.py::Test::test_value[x.y]"/></properties>
       <error message="teardown failed"/>
      </testcase>
    </testsuite></testsuites>''')
    raw = {"tests/test_one.py::Test::test_value[x.y]": {
        "junit_key": ["tests.test_one.Test", "test_value[x.y]"], "status": "error"}}
    result = verification.junit_assessment(xml, raw)
    assert result["consistent"]
    assert result["unique_raw_ids"] == 1
    raw["tests/test_one.py::Test::test_value[x.y]"]["status"] = "passed"
    assert not verification.junit_assessment(xml, raw)["consistent"]


def test_unmapped_or_missing_junit_ids_never_pass(tmp_path):
    raw = {"tests/test_a.py::test_ok": {"junit_key": ["tests.test_a", "test_ok"], "status": "passed"}}
    assert not verification.junit_assessment(tmp_path / "missing.xml", raw)["consistent"]
    xml = tmp_path / "foreign.xml"
    xml.write_text('<testsuites><testcase classname="other" name="test_other"/></testsuites>')
    result = verification.junit_assessment(xml, raw)
    assert not result["consistent"]
    assert result["missing_ids"] == ["tests/test_a.py::test_ok"]
    assert len(result["unmapped_rows"]) == 1


def test_historical_manifest_tamper_is_recorded(tmp_path):
    payload = tmp_path / "Result.json"
    payload.write_text('{"value": 1}\n')
    preparation.write(tmp_path / "ManifestV1.json", {"files": {"Result.json": preparation.identity(payload)}})
    assert preparation.verify_delivery(tmp_path)["passed"]
    payload.write_text('{"value": 2}\n')
    result = preparation.verify_delivery(tmp_path)
    assert not result["passed"]
    assert result["failures"] == [{"file": "Result.json", "reason": "sha256_mismatch"}]


def test_response_request_rejects_mixed_sources_before_producing_request(tmp_path):
    batches = []
    for i in range(2):
        batch = tmp_path / f"batch{i}"
        batch.mkdir()
        verification.write(batch / "ManifestV1.json", {})
        verification.write(batch / "SourceReceiptV1.json", {
            "source_commit": str(i) * 40, "source_content_sha256": str(i) * 64})
        batches.append(batch)
    out = tmp_path / "requests"
    with pytest.raises(ValueError, match="one source"):
        verification.response_requests(SimpleNamespace(batch=batches, output=out, dc_bundle=None, methodology=None))
    assert not (out / "DCRequestV1.json").exists()


def test_summary_lists_unstarted_shards_as_not_run(tmp_path, capsys):
    plan = {"source_identity": {"all_tracked_content_sha256": "a" * 64},
            "expected_raw_ids": ["tests/a.py::test_one", "tests/b.py::test_two"],
            "shards": {"A": {"expected_raw_ids": ["tests/a.py::test_one"]},
                       "B": {"expected_raw_ids": ["tests/b.py::test_two"]}}, "dependency_exclusions": []}
    plan_file = tmp_path / "plan.json"
    verification.write(plan_file, plan)
    out = tmp_path / "run"
    verification.write(out / "A/RawResultsV1.json", {"results": {
        "tests/a.py::test_one": {"status": "passed", "phases": {"call": {"outcome": "passed"}}}}})
    verification.write(out / "A/ResultV1.json", {"process_exit_code": 0})
    verification.summarize(SimpleNamespace(plan=plan_file, plan_sha256=verification.sha(plan_file),
                                           output=out, baseline_results=None))
    result = verification.read(out / "VerificationSummaryV1.json")
    assert result["counts"]["not_run"] == 1
    assert result["actual_started_count"] == 1
    assert not result["all_selected_passed"]
    assert not result["all_planned_ids_reported"]
    assert not result["verification_integrity_passed"]


def test_isolated_end_to_end_runner_retains_failure_error_and_skip(tmp_path):
    repo = tmp_path / "repo"
    project = repo / "perovskite-sim"
    tests = project / "tests"
    tests.mkdir(parents=True)
    (tests / "test_accounts.py").write_text('''import pytest
def test_pass():
    assert True
def test_failure():
    assert False
@pytest.mark.skip(reason="explicit test fixture")
def test_skip():
    pass
@pytest.fixture
def broken():
    raise RuntimeError("setup failed")
def test_error(broken):
    pass
''')
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-qm", "fixture"], cwd=repo, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    prior = tmp_path / "prior.json"
    verification.write(prior, {"shards": {"A": {"files": ["tests/test_accounts.py"]}},
                               "dependency_exclusions": [], "expected_raw_ids": []})
    affected = tmp_path / "affected.json"
    verification.write(affected, {"arguments": ["tests/test_accounts.py"]})
    freeze_args = SimpleNamespace(project=project, python=sys.executable,
        dependency_path=[str(Path(pytest.__file__).resolve().parents[1])], expected_commit=commit,
        prior_plan=prior, affected=affected, extra_file=[], output=tmp_path / "freeze")
    verification.freeze(freeze_args)
    plan = freeze_args.output / "PlanV1.json"
    plan_data = verification.read(plan)
    run_args = SimpleNamespace(project=project, python=freeze_args.python,
        dependency_path=freeze_args.dependency_path, plan=plan, plan_sha256=verification.sha(plan),
        expected_source_sha256=plan_data["source_identity"]["all_tracked_content_sha256"],
        shard=None, output=tmp_path / "run")
    assert verification.run(run_args) == 1
    result = verification.read(run_args.output / "A/RawResultsV1.json")
    assert result["selected_count"] == result["started_count"] == 4
    assert result["counts"] == {"passed": 1, "failure": 1, "error": 1, "skipped": 1,
                                "incomplete": 0, "not_run": 0}
    assert verification.read(run_args.output / "A/JUnitAssessmentV1.json")["consistent"]
    verification.summarize(SimpleNamespace(plan=plan, plan_sha256=verification.sha(plan),
        output=run_args.output, baseline_results=None))
    summary = verification.read(run_args.output / "VerificationSummaryV1.json")
    assert summary["all_planned_ids_reported"]
    assert summary["verification_integrity_passed"]
    assert not summary["all_selected_passed"]
