"""Durable barriers for supervisor-control tests; no scientific computation."""

import os
from pathlib import Path
import time

from solarlab.experiments.worker import WorkerResult, _atomic_json


def barrier(request, context):
    context.counters(accepted_steps=0, calls=0)
    _atomic_json(Path("ready.json"), {"pid": os.getpid()})
    while not Path("release").exists():
        context.check_cancel()
        time.sleep(0.005)
    return WorkerResult.from_dict({"ok": True})
