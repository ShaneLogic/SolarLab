"""Real spawned infrastructure probes; none is a scientific executor."""

import json
import os
from pathlib import Path
import signal
import time

from solarlab.experiments.worker import WorkerResult
from solarlab.io.artifacts import bytes_artifact

_LOCAL_MUTATIONS = []


def point(request, context):
    binding = context.consume_manifest(request, engine="solarlab")
    payload = json.loads(request.request.content)
    if binding.execution.protocol.content != request.request.content:
        raise ValueError("fixture protocol differs from actual request")
    if _LOCAL_MUTATIONS:
        raise AssertionError("state leaked between points")
    _LOCAL_MUTATIONS.append(payload["point"])
    assert os.environ["OMP_NUM_THREADS"] == os.environ["OPENBLAS_NUM_THREADS"] == "1"
    context.counters(accepted_steps=0, calls=0)
    context.progress({"stage": "ready", "pid": os.getpid()})
    action = payload.get("action", "normal")
    if action == "exception":
        raise ValueError("point-local deliberate failure after state mutation")
    if action == "crash":
        os._exit(17)
    if action in {"stuck", "late", "release"}:
        if action == "stuck":
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        deadline = time.monotonic() + 8  # Even an orphaned test worker is bounded.
        while not Path("release").exists() and time.monotonic() < deadline:
            if action == "release":
                context.check_cancel()
            time.sleep(.005)
    if action == "counter":
        context.counters(accepted_steps=3, calls=3)
    if action == "nested_scan":
        from solarlab.experiments.execution import ScanRunner
        ScanRunner(None, None, (), max_in_flight=1)
    context.counters(accepted_steps=1, calls=1)
    initial = binding.execution.initial_state
    words = bytes.fromhex(json.loads(initial.content)["words"])
    return WorkerResult.from_dict({"point": payload["point"], "initial_state_sha256": initial.sha256,
        "scientific_qualification": False, "evidence_level": "unclassified", "fresh_process_state": True},
        artifacts=(("words", bytes_artifact(words, metadata={"unit": "fixture", "missing_reason": None})),))


def crash_coordinator(database, work, raw, bindings, executions, checkpoint):
    """Abrupt real parent exit leaves a running intent and a queued child."""
    from solarlab.experiments.execution import ProcessSupervisor, ScanSpec
    from solarlab.io.run_store import RunStore

    store = RunStore(Path(database))
    pool = ProcessSupervisor(store, Path(work), bindings, max_active=1, max_pending=1,
        shared_rss_bytes=2 * 1024**3, shared_output_bytes=64 * 1024**2, poll_seconds=.01)
    scan = pool.scan(ScanSpec.from_bytes(raw), executions=executions, max_in_flight=2)
    deadline = time.monotonic() + 6
    while time.monotonic() < deadline:
        status = scan.advance()
        if status["points"][0]["state"] == "succeeded" and status["points"][1]["state"] == "running":
            run_id = status["points"][1]["run_id"]
            if any(e["payload"].get("ui_progress", {}).get("stage") == "ready" for e in store.events(run_id=run_id)):
                Path(checkpoint).write_text(json.dumps(status))
                os._exit(72)
        time.sleep(.01)
    os._exit(73)
