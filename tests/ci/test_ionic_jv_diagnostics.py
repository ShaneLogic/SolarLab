"""The opt-in ionic HTTP diagnostic must retain failures and every attempt."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts.ci import ionic_jv_diagnostics as diagnostics


def test_workflow_diagnostics_are_manual_and_keep_both_python_versions():
    workflow = yaml.safe_load((diagnostics.ROOT / ".github/workflows/ci.yml").read_text())
    job = workflow["jobs"]["ionic-jv-diagnostics"]
    assert job["if"] == "github.event_name == 'workflow_dispatch' && inputs.run_ionic_jv_diagnostics"
    assert job["strategy"]["matrix"]["python"] == ["3.11.16", "3.13.15"]
    # PyYAML's YAML 1.1 loader calls the workflow's `on` key True.
    inputs = workflow.get("on", workflow.get(True))["workflow_dispatch"]["inputs"]
    assert inputs["run_ionic_jv_diagnostics"]["default"] is False
    assert inputs["ionic_jv_diagnostic_repetitions"]["default"] == 20
    timing, = [step for step in job["steps"] if "scripts/ci/ionic_jv_timing.py" in step.get("run", "")]
    assert timing["if"] == "always()"
    assert timing["env"]["IONIC_JV_DIAGNOSTIC_REPETITIONS"] == "${{ inputs.ionic_jv_diagnostic_repetitions }}"
    assert 'ionic-jv-diagnostics-${{ matrix.python }}/uncensored' in timing["run"]
    upload, = [step for step in job["steps"] if "actions/upload-artifact@" in step.get("uses", "")]
    assert upload["if"] == "always()"
    assert "ionic-jv-diagnostics-${{ matrix.python }}" in upload["with"]["path"]
    assert upload["with"]["include-hidden-files"] is True


@pytest.mark.parametrize("first_code", [0, 1])
def test_every_attempt_has_distinct_evidence_and_failures_do_not_stop_the_diagnostic(tmp_path, monkeypatch, first_code):
    for name in ("PYTEST_ADDOPTS", "PYTEST_CI_ISOLATED_GROUP", "SOLARLAB_IONIC_JV_HTTP_SAVED"):
        monkeypatch.setenv(name, "must-not-be-inherited")
    calls = []

    def process(command, **kwargs):
        calls.append((command, kwargs))
        number = len(calls)
        env = kwargs["env"]
        assert env["PYTEST_CI_JOB"] == "backend" and all(env[key] == "1" for key in diagnostics.THREADS)
        assert not {"PYTEST_ADDOPTS", "PYTEST_CI_ISOLATED_GROUP", "SOLARLAB_IONIC_JV_HTTP_SAVED"} & env.keys()
        evidence = Path(env["SOLARLAB_IONIC_JV_HTTP_EVIDENCE"])
        (evidence / "Receipt.json").write_text(str(number))
        report = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--junitxml=")))
        code = first_code if number == 1 else 0
        detail = "<failure/>" if code else ""
        report.write_text(f'<testsuites><testsuite><testcase>{detail}</testcase></testsuite></testsuites>')
        return SimpleNamespace(returncode=code)

    monkeypatch.setattr(diagnostics.subprocess, "run", process)
    output = tmp_path / "diagnostic"
    assert diagnostics.main(["--repetitions", "3", "--output-dir", str(output)]) == first_code
    assert len(calls) == 3
    assert [p.read_text() for p in sorted(output.glob("attempt_*/evidence/Receipt.json"))] == ["1", "2", "3"]
    assert len({next(arg for arg in command if arg.startswith("--basetemp=")) for command, _ in calls}) == 3


@pytest.mark.parametrize("count", ["0", "-1", "1.5"])
def test_invalid_repeat_count_cannot_start_work(tmp_path, monkeypatch, count):
    monkeypatch.setattr(diagnostics.subprocess, "run", lambda *args, **kwargs: pytest.fail("must not start"))
    with pytest.raises(SystemExit) as error:
        diagnostics.main(["--repetitions", count, "--output-dir", str(tmp_path / "unused")])
    assert error.value.code == 2
    assert not (tmp_path / "unused").exists()


def test_existing_evidence_is_never_overwritten(tmp_path):
    (tmp_path / "evidence").write_text("retained")
    with pytest.raises(ValueError, match="previous evidence is preserved"):
        diagnostics.diagnose(1, tmp_path)
    assert (tmp_path / "evidence").read_text() == "retained"
