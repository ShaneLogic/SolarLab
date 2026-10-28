"""Actual process barriers before the production registered legacy worker."""
from pathlib import Path
import time


def gated_legacy(request, context):
    context.progress({"stage": "ready"})
    deadline = time.monotonic() + 8
    while not Path("release").exists() and time.monotonic() < deadline:
        context.check_cancel()
        time.sleep(.005)
    from solarlab_server.registered_worker import run_legacy
    return run_legacy(request, context)


def stuck_legacy(request, context):
    import signal
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    context.progress({"stage": "ready"})
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        time.sleep(.01)
    raise RuntimeError("orphaned test worker ended its independent bound")
