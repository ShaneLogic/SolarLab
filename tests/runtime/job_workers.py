"""Actual process barriers before the production registered solarlab steady worker."""
from pathlib import Path
import time


def gated_steady(request, context):
    context.progress({"stage": "ready"})
    deadline = time.monotonic() + 8
    while not Path("release").exists() and time.monotonic() < deadline:
        context.check_cancel()
        time.sleep(0.005)
    from solarlab_server.registered_worker import run_steady
    return run_steady(request, context)


def failing_steady(request, context):
    context.consume_manifest(request, engine="solarlab")
    context.progress({"stage": "ready"})
    raise RuntimeError("injected steady worker driver failure")


def stuck_steady(request, context):
    import signal
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    context.progress({"stage": "ready"})
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        time.sleep(0.01)
    raise RuntimeError("orphaned test worker ended its independent bound")
