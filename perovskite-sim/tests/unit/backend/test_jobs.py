"""Unit tests for the in-process job registry that backs the SSE job API."""
from __future__ import annotations
import time
import threading
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


def test_wait_timeout_keeps_the_running_job_and_then_joins_it():
    registry = JobRegistry()
    entered = threading.Event()
    release = threading.Event()

    def blocked_job(_reporter):
        entered.set()
        assert release.wait(5.0)
        return {"released": True}

    job_id = registry.submit(blocked_job)
    try:
        assert entered.wait(2.0)
        with pytest.raises(TimeoutError):
            registry.wait(job_id, timeout=0.0)
        assert registry.status(job_id)[0] == JobStatus.RUNNING
    finally:
        release.set()
        status, result, error = registry.wait(job_id, timeout=5.0)
    assert status == JobStatus.DONE
    assert result == {"released": True}
    assert error is None
    assert not registry._jobs[job_id].thread.is_alive()


def test_wait_reports_worker_failure_after_cleanup():
    registry = JobRegistry()
    job_id = registry.submit(_crashing_job)
    status, result, error = registry.wait(job_id, timeout=5.0)
    assert status == JobStatus.ERROR
    assert result is None
    assert "boom" in error
    assert not registry._jobs[job_id].thread.is_alive()


def test_wait_rejects_unknown_job():
    with pytest.raises(KeyError):
        JobRegistry().wait("missing", timeout=0.0)


@pytest.mark.parametrize("timeout", [-1.0, float("nan"), float("inf")])
def test_wait_rejects_invalid_timeout(timeout):
    with pytest.raises(ValueError, match="timeout"):
        JobRegistry().wait("missing", timeout=timeout)
