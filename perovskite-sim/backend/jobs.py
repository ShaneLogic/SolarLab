"""In-process job registry for streaming experiment progress over SSE.

A `Job` is a Python callable that takes a `ProgressReporter` and returns a
JSON-serializable result dict. The registry spawns one worker thread per
submitted job, lets the worker report progress through the reporter, and
captures the final result or any raised exception.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import queue
import threading
import traceback
import uuid
from typing import Any, Callable, Dict, Optional, Tuple

from backend.progress import ProgressEvent, ProgressReporter


class JobStatus(str, Enum):
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass
class Job:
    job_id: str
    status: JobStatus = JobStatus.RUNNING
    result: Optional[dict] = None
    error: Optional[str] = None
    reporter: ProgressReporter = field(default_factory=ProgressReporter)
    thread: Optional[threading.Thread] = None


class JobRegistry:
    """Thread-safe dict of jobs keyed by UUID.

    Lifecycle:
      - `submit(fn)` spawns a worker thread that runs `fn(reporter)`.
      - `next_event(job_id, timeout)` blocks for the next progress event,
        returns None when the job is done, re-raises KeyError for unknown ids.
      - `status(job_id)` returns the current state and final result/error.

    Single-process and in-memory. If the FastAPI worker restarts, all jobs
    are lost — acceptable for runs that take seconds to a few minutes.
    """

    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, fn: Callable[[ProgressReporter], dict]) -> str:
        job_id = uuid.uuid4().hex
        job = Job(job_id=job_id)
        with self._lock:
            self._jobs[job_id] = job

        def _worker() -> None:
            try:
                result = fn(job.reporter)
                job.result = result
                job.status = JobStatus.DONE
            except Exception as exc:  # noqa: BLE001
                job.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                job.status = JobStatus.ERROR
            finally:
                job.reporter.finish()

        t = threading.Thread(target=_worker, name=f"job-{job_id[:8]}", daemon=True)
        job.thread = t
        t.start()
        return job_id

    def next_event(
        self,
        job_id: str,
        timeout: float = 0.25,
    ) -> Optional[ProgressEvent]:
        """Return the next ProgressEvent, or None when the stream is done.

        Raises KeyError for an unknown job id. On a drain timeout this
        method returns the `_DRAIN_TIMEOUT` sentinel instead of None so
        SSE handlers can distinguish "keep the connection open" from
        "worker finished" — tests that wait on a fast-finishing job will
        never hit the timeout branch.
        """
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            job = self._jobs[job_id]
        try:
            return job.reporter.drain(timeout=timeout)
        except queue.Empty:
            return _DRAIN_TIMEOUT  # type: ignore[return-value]

    def status(
        self,
        job_id: str,
    ) -> Tuple[JobStatus, Optional[dict], Optional[str]]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            job = self._jobs[job_id]
        return job.status, job.result, job.error


class _DrainTimeout:
    """Distinct sentinel returned by `next_event` on a drain timeout.

    The SSE handler checks `is _DRAIN_TIMEOUT` to emit a keep-alive comment
    rather than closing the stream. ProgressEvent consumers never see it —
    pytest-shaped tests only compare against `None`.
    """


_DRAIN_TIMEOUT = _DrainTimeout()
