"""Uncensored samples, conservative wall derivation and failed evidence retention."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.ci import ionic_jv_timing as timing


def row(seconds, *, ceiling=400., status="passed"):
    return {"status": status, "measurement": {"status": status, "child_wall_seconds": seconds,
            "request": {"limits": {"wall_seconds": ceiling}}}}


@pytest.mark.parametrize("seconds,expected", [
    (list(range(1, 21)), 128),
    (list(range(20, 40)), 152),
    ([1.] * 19 + [200.], 400),
    ([32.1] * 19 + [40.2], 129),
])
def test_allowance_uses_floor_p95_max_and_rounds_up(seconds, expected):
    result = timing.summarize([row(value) for value in seconds])
    assert result["status"] == "passed" and result["completed"] == 20
    assert result["recommended_wall_seconds"] == expected
    assert result["child_wall_seconds"]["max"] == max(seconds)


def test_insufficient_provisional_allowance_cannot_be_reported_as_validated():
    result = timing.summarize([row(33., ceiling=128.) for _ in range(20)])
    assert result["status"] == "failed" and result["completed"] == 20
    assert result["recommended_wall_seconds"] == 132 and result["allowance_sufficient"] is False
    assert result["reason"] == "installed_allowance_below_uncensored_derivation"


def test_numerical_or_process_failure_cannot_disappear_from_uncensored_percentiles():
    result = timing.summarize([row(10.) for _ in range(19)] + [row(25., status="failed")])
    assert result["status"] == "failed" and result["requested"] == 20 and result["completed"] == 19
    assert result["recommended_wall_seconds"] is None and "child_wall_seconds" not in result


@pytest.mark.parametrize("first_code", [0, 1])
@pytest.mark.parametrize("curve_measurement", [False, True])
def test_timing_runs_every_fresh_child_without_a_whole_job_timeout(tmp_path, monkeypatch, first_code, curve_measurement):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        assert command[1:4] == ["-m", "scripts.ci.ionic_jv_timing", "--single"]
        assert "timeout" not in kwargs
        assert kwargs["env"]["OPENBLAS_NUM_THREADS"] == "1"
        assert ("--diagnostic-curve-64" in command) is curve_measurement
        directory = Path(command[4])
        code = first_code if len(calls) == 1 else 0
        data = row(10., ceiling=128., status="failed" if code else "passed")["measurement"]
        (directory / "Timing.json").write_text(json.dumps(data))
        return SimpleNamespace(returncode=code)

    monkeypatch.setattr(timing.subprocess, "run", run)
    output = tmp_path / "timing"
    arguments = ["--repetitions", "3", "--output-dir", str(output)]
    if curve_measurement:
        arguments.append("--diagnostic-curve-64")
    assert timing.main(arguments) == first_code
    assert len(calls) == len({command[4] for command, _ in calls}) == 3
    summary = json.loads((output / "Summary.json").read_text())
    assert summary["requested"] == 3 and summary["completed"] == 3 - first_code
    if curve_measurement:
        assert summary["recommended_wall_seconds"] is None and summary["recommendation_status"] == "measurement_only"
    assert len(list(output.glob("attempt_*/Timing.json"))) == 3


def test_timing_preserves_previous_evidence(tmp_path):
    saved = tmp_path / "Timing.json"
    saved.write_text("retained")
    with pytest.raises(ValueError, match="previous evidence is preserved"):
        timing.diagnose(1, tmp_path)
    assert saved.read_text() == "retained"


@pytest.mark.parametrize("seconds", [[], [0.], [-1.], [float("nan")], [float("inf")]])
def test_invalid_measurements_cannot_drive_a_ceiling(seconds):
    with pytest.raises(ValueError, match="positive finite"):
        timing.distribution(seconds)
