"""Unit tests for the progress pub/sub primitive used by the SSE job channel."""
from __future__ import annotations
import queue
import time
from backend.progress import ProgressEvent, ProgressReporter


def test_progress_event_defaults():
    ev = ProgressEvent(stage="jv_forward", current=3, total=30, eta_s=None)
    assert ev.stage == "jv_forward"
    assert ev.current == 3
    assert ev.total == 30
    assert ev.eta_s is None
    assert ev.message == ""


def test_reporter_captures_events_in_order():
    reporter = ProgressReporter()
    reporter.report("jv_forward", 0, 30)
    reporter.report("jv_forward", 1, 30, message="step 1")
    reporter.report("jv_forward", 2, 30)

    drained: list[ProgressEvent] = []
    while True:
        try:
            ev = reporter.drain(timeout=0.0)
        except queue.Empty:
            break
        if ev is None:
            break
        drained.append(ev)

    assert [(e.stage, e.current, e.total, e.message) for e in drained] == [
        ("jv_forward", 0, 30, ""),
        ("jv_forward", 1, 30, "step 1"),
        ("jv_forward", 2, 30, ""),
    ]


def test_reporter_eta_monotone_decreasing():
    reporter = ProgressReporter()
    reporter.report("impedance", 0, 10)
    time.sleep(0.05)
    reporter.report("impedance", 5, 10)
    time.sleep(0.05)
    reporter.report("impedance", 9, 10)

    events: list[ProgressEvent] = []
    while True:
        try:
            ev = reporter.drain(timeout=0.0)
        except queue.Empty:
            break
        if ev is None:
            break
        events.append(ev)

    assert events[0].eta_s is None
    etas = [e.eta_s for e in events[1:]]
    assert all(eta is not None and eta >= 0.0 for eta in etas)
    assert etas[0] >= etas[-1]


def test_reporter_finish_marks_done():
    reporter = ProgressReporter()
    reporter.report("impedance", 0, 3)
    reporter.finish()
    drained: list = []
    while True:
        ev = reporter.drain(timeout=0.0)
        drained.append(ev)
        if ev is None:
            break
    assert drained[-1] is None
