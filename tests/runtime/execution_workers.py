"""Only infrastructure fixtures; these are not scientific registry entries."""

import json
import multiprocessing
import os
from pathlib import Path
import resource
import signal
import struct
import sys
import time

from solarlab.experiments.worker import WorkerResult
from solarlab.experiments.worker import _atomic_json
from solarlab.io.artifacts import bytes_artifact


def _ready(context):
    forbidden = {"scipy", "sksundae", "flint", "perovskite_sim"} & set(sys.modules)
    if forbidden:
        raise AssertionError("unexpected scientific import: " + repr(forbidden))
    context.counters(accepted_steps=0, calls=0)
    context.progress({"stage": "ready", "pid": os.getpid(), "forbidden_modules_loaded": sorted(forbidden)})


def normal(request, context):
    _ready(context)
    payload = json.loads(request.request.content)
    count = payload.get("progress_count", 100)
    for index in range(count):
        context.counters(accepted_steps=index + 1, calls=index + 1)
        context.progress({"stage": "fixture", "current": index + 1, "total": count})
    artifact = bytes_artifact(struct.pack("<2d", 0.0, -0.0), metadata={"unit": "fixture_words", "missing_reason": None})
    return WorkerResult.from_dict({"ok": True, "explicit_null": None, "negative_zero": -0.0},
                                  artifacts=(("raw_words", artifact),))


def supervised_client(request, context):
    """Real durable-client fixture with an observed release/cancel boundary."""
    forbidden = {"scipy", "sksundae", "flint", "perovskite_sim"} & set(sys.modules)
    if forbidden:
        raise AssertionError("unexpected scientific import: " + repr(forbidden))
    # Intentionally report no cumulative counters. Unobserved values must remain
    # null through the supervisor, store, HTTP client and panel.
    context.progress({"stage": "ready", "pid": os.getpid(),
                      "forbidden_modules_loaded": sorted(forbidden), "zero": 0,
                      "negative_zero": -0.0, "unobserved": None,
                      "large_integer": 18446744073709551617,
                      "scientific_qualification": False})
    while not Path("release").exists():
        context.check_cancel()
        time.sleep(0.005)
    if json.loads(request.request.content)["outcome"] == "failed":
        raise ValueError("supervised client infrastructure failure")
    artifact = bytes_artifact(struct.pack("<2d", 0.0, -0.0) + b"\x00supervised",
        metadata={"unit": "fixture_words", "missing_reason": None, "scientific_qualification": False})
    return WorkerResult.from_dict({"zero": 0, "negative_zero": -0.0, "tau": None,
        "large_integer": 18446744073709551617, "scientific_qualification": False},
        artifacts=(("raw_words", artifact),))


def cooperative(request, context):
    _ready(context)
    while True:
        context.check_cancel()
        time.sleep(0.005)


def uncooperative(request, context):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    _ready(context)
    while True:
        time.sleep(0.005)


def delayed_success(request, context):
    _ready(context)
    # Deliberately return success after a soft cancellation: the parent must win.
    time.sleep(0.18)
    return WorkerResult.from_dict({"returned_after_delay": True})


def barrier_success(request, context):
    _ready(context)
    # Ignore soft stop deliberately; return only after the test has established
    # the cancellation/generation ordering, never after a guessed sleep.
    while not Path("release").exists():
        time.sleep(0.005)
    return WorkerResult.from_dict({"returned_after_release": True})


def exception(request, context):
    _ready(context)
    raise ValueError("infrastructure fixture exception")


def crash(request, context):
    _ready(context)
    os._exit(17)


def counter_limit(request, context):
    _ready(context)
    context.counters(accepted_steps=3, calls=2)
    return WorkerResult.from_dict({"unreachable": True})


def missing_counters(request, context):
    return WorkerResult.from_dict({"counts_were_not_observed": True})


def disk_limit(request, context):
    _ready(context)
    remaining = b"x" * 32768
    while remaining:
        remaining = remaining[os.write(1, remaining):]
    return WorkerResult.from_dict({"log_written": True})


def nested_pool(request, context):
    _ready(context)
    child = multiprocessing.get_context("spawn").Process(target=os.getpid)
    child.start()  # A daemonic worker must reject nested worker creation.
    child.join(0.2)
    return WorkerResult.from_dict({"unexpected_nested_worker": True})


def malformed_terminal(request, context):
    """Deliberately bypass the normal encoder to exercise parent validation."""
    _ready(context)
    source = json.loads(Path("launch.json").read_text())
    kind = json.loads(request.request.content)["malformed"]
    entry = None if kind == "not_mapping" else {"file": "artifact-0.bin"}
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    _atomic_json(Path("terminal.json"), {"token": source["token"],
        "worker_source_sha256": source["binding"]["source_sha256"], "state": "succeeded", "result": {},
        "artifacts": [entry], "counters": {"accepted_steps": 0, "calls": 0},
        "child_peak_rss_bytes": peak if sys.platform == "darwin" else peak * 1024})
    os._exit(0)


def finalize_after_release(request, context):
    """Small real output, paused so the parent can set an exact retained load."""
    _ready(context)
    while not Path("release").exists():
        context.check_cancel()
        time.sleep(0.005)
    config = json.loads(request.request.content)
    data = b"x" * config["bytes"]
    if config["kind"] == "artifact":
        return WorkerResult.from_dict({"large_output": "artifact"},
            artifacts=(("payload", bytes_artifact(data, metadata={"scope": "infrastructure"})),))
    return WorkerResult.from_dict({"large_output": data.decode("ascii")})
