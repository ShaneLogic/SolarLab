"""Retain infrastructure failures without replacing their original exception."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from solarlab.experiments.execution import ProcessSupervisor
from solarlab.io.run_store import RunStore


def record_resource_samples(supervisor: ProcessSupervisor, monkeypatch: Any) -> list[dict[str, Any]]:
    """Observe the real sampler; do not replace or retry its measurements."""
    samples: list[dict[str, Any]] = []
    original = supervisor._sample

    def sampled() -> None:
        try:
            original()
        finally:
            now = time.monotonic()
            samples.append({"monotonic": now, "rss_observation": supervisor._rss_observation(),
                "parent_rss_bytes": supervisor._parent_rss,
                "shared_rss_bytes": supervisor._shared_rss,
                "sampled_family_pids": list(supervisor._sampled_family_pids),
                "jobs": [{"run_id": job.handle.run_id, "attempt_id": job.handle.attempt_id,
                    "limits": asdict(job.limits), "started_monotonic": job.started,
                    "owned_wall_seconds": None if job.started is None else now - job.started,
                    "child_peak_rss_bytes": job.peak_rss, "output_peak_bytes": job.output_peak,
                    "counters": dict(job.counts), "stop_reason": job.stop_reason,
                    "signals": list(job.signals)} for job in supervisor._active.values()]})

    monkeypatch.setattr(supervisor, "_sample", sampled)
    return samples


def preserve_runtime_evidence(root: Path, store_name: str, label: str,
                              samples: list[dict[str, Any]]) -> str:
    """Called after owner cleanup so every available final receipt is retained."""
    value: dict[str, Any] = {"store": str(root / store_name), "work": str(root / "work"),
        "resource_samples": samples, "workers": {}, "events": []}
    for directory in sorted((root / "work").glob("*")):
        if not directory.is_dir():
            continue
        documents: dict[str, Any] = {}
        for name in ("launch.json", "started.json", "spawned.json", "receipt.json",
                     "terminal.json", "stop.json", "finalization.json", "fixture-clock.json", "worker.log"):
            path = directory / name
            if not path.exists():
                documents[name] = {"missing": True}
                continue
            try:
                data = path.read_bytes()
            except OSError as error:
                documents[name] = {"path": str(path), "read_error": str(error)}
                continue
            try:
                content: Any = json.loads(data)
            except (ValueError, UnicodeError):
                content = {"unparsed_text": data.decode("utf-8", errors="replace")}
            documents[name] = {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "content": content}
        value["workers"][directory.name] = documents
    try:
        with RunStore(root / store_name, read_only=True) as store:
            value["events"] = store.events(limit=1000)
    except Exception as error:
        value["store_read_error"] = f"{type(error).__name__}: {error}"
    output = Path(os.environ.get("SCHEDULED_RUNTIME_EVIDENCE_DIR") or str(root))
    path = output / (label + ".json")
    value["evidence_path"] = str(path)
    text = json.dumps(value, indent=2, allow_nan=False)
    try:
        output.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
    except OSError as error:
        value["evidence_write_error"] = str(error)
        text = json.dumps(value, indent=2, allow_nan=False)
    return text
