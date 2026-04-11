"""Thread-safe progress pub/sub used by the SSE job channel.

An experiment running on a worker thread calls `reporter.report(...)` after
each unit of work. The SSE endpoint handler drains the reporter on the main
request thread and emits Server-Sent-Event frames.
"""
from __future__ import annotations
from dataclasses import dataclass
import queue
import threading
import time
from typing import Optional


@dataclass(frozen=True)
class ProgressEvent:
    """One progress update. Immutable so it can be passed across threads safely."""
    stage: str
    current: int
    total: int
    eta_s: Optional[float]
    message: str = ""


class ProgressReporter:
    """Thread-safe FIFO queue of ProgressEvent objects.

    - Producers call `report(stage, current, total, message)` on the worker thread.
    - Consumers call `drain(timeout)` on the SSE thread to fetch the next event.
    - `finish()` posts a None sentinel that signals the stream is over.
    """

    _DONE: object = object()

    def __init__(self) -> None:
        self._q: "queue.Queue[object]" = queue.Queue()
        self._lock = threading.Lock()
        self._first_report_time: Optional[float] = None
        self._first_report_current: int = 0

    def report(
        self,
        stage: str,
        current: int,
        total: int,
        message: str = "",
    ) -> None:
        """Post a progress update, computing a best-effort ETA on the fly."""
        now = time.monotonic()
        with self._lock:
            if self._first_report_time is None:
                self._first_report_time = now
                self._first_report_current = current
                eta: Optional[float] = None
            else:
                elapsed = now - self._first_report_time
                done = max(1, current - self._first_report_current)
                remaining = max(0, total - current)
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = remaining / rate if rate > 0 else None
        self._q.put(ProgressEvent(
            stage=stage, current=current, total=total, eta_s=eta, message=message,
        ))

    def finish(self) -> None:
        """Post the end-of-stream sentinel."""
        self._q.put(self._DONE)

    def drain(self, timeout: float = 0.1) -> Optional[ProgressEvent]:
        """Block up to `timeout` seconds for the next event.

        Returns the event, `None` when the done sentinel is observed, or
        raises `queue.Empty` if the timeout elapses with nothing to deliver.
        """
        item = self._q.get(timeout=timeout)
        if item is self._DONE:
            return None
        return item  # type: ignore[return-value]
