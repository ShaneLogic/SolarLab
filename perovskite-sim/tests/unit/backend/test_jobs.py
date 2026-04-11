"""Unit tests for the in-process job registry that backs the SSE job API."""
from __future__ import annotations
import time
import pytest
from backend.jobs import JobRegistry, JobStatus
from backend.progress import ProgressReporter


def _noop_job(reporter: ProgressReporter) -> dict:
    reporter.report("noop", 0, 3)
    reporter.report("noop", 1, 3)
    reporter.report("noop", 2, 3)
    reporter.report("noop", 3, 3)
    return {"ok": True}


def _crashing_job(reporter: ProgressReporter) -> dict:
    reporter.report("crash", 0, 1)
    raise RuntimeError("boom")


def test_submit_runs_to_completion():
    reg = JobRegistry()
    job_id = reg.submit(_noop_job)
    assert isinstance(job_id, str) and len(job_id) > 0

    events = []
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        ev = reg.next_event(job_id, timeout=0.05)
        if ev is None:
            break
        # Ignore drain-timeout sentinel — keep looping.
        if type(ev).__name__ == "_DrainTimeout":
            continue
        events.append(ev)

    assert [e.current for e in events] == [0, 1, 2, 3]
    status, result, error = reg.status(job_id)
    assert status == JobStatus.DONE
    assert result == {"ok": True}
    assert error is None


def test_job_captures_errors():
    reg = JobRegistry()
    job_id = reg.submit(_crashing_job)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        reg.next_event(job_id, timeout=0.05)
        status, _, _ = reg.status(job_id)
        if status != JobStatus.RUNNING:
            break
    status, result, error = reg.status(job_id)
    assert status == JobStatus.ERROR
    assert result is None
    assert error is not None and "boom" in error


def test_unknown_job_id_raises():
    reg = JobRegistry()
    with pytest.raises(KeyError):
        reg.status("nope")
    with pytest.raises(KeyError):
        reg.next_event("nope", timeout=0.0)
