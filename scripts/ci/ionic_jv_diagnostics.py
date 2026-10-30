"""Repeat the real ionic HTTP module, preserving every attempt for diagnosis.

Each repetition uses a fresh pytest process, its own temporary directory and
the existing optional HTTP evidence sink. Failures remain failures; all requested
repetitions run before the final nonzero status is returned.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
from typing import Any
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
MODULE = "tests/runtime/test_ionic_jv_job_service.py"
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")


def environment(evidence: Path) -> dict[str, str]:
    env = dict(os.environ)
    for key in ("PYTEST_ADDOPTS", "PYTEST_CI_ISOLATED_GROUP", "PYTHONPATH", "PYTHONHOME",
                "SOLARLAB_IONIC_JV_HTTP_SAVED", "SOLARLAB_IONIC_JV_HTTP_COMMIT",
                "SOLARLAB_IONIC_JV_HTTP_ARCHIVE"):
        env.pop(key, None)
    env.update({key: "1" for key in THREADS})
    env.update(PYTEST_CI_JOB="backend", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               SOLARLAB_IONIC_JV_HTTP_EVIDENCE=str(evidence))
    return env


def diagnose(repetitions: int, output: Path) -> dict[str, Any]:
    if repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    if output.exists() and any(output.iterdir()):
        raise ValueError("diagnostic output must be empty; previous evidence is preserved")
    output.mkdir(parents=True, exist_ok=True)
    result_counts = runpy.run_path(str(ROOT / "scripts/test_all.py"))["result_counts"]
    summary: dict[str, Any] = {"status": "running", "module": MODULE,
        "python": sys.executable, "python_version": sys.version,
        "requested_repetitions": repetitions, "attempts": [],
        "finding": "unresolved; native terminal evidence must be classified"}

    def save() -> None:
        (output / "Summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    save()
    for index in range(1, repetitions + 1):
        directory = output / f"attempt_{index:02d}"
        directory.mkdir()
        evidence = directory / "evidence"
        evidence.mkdir()
        report, log = directory / "Tests.xml", directory / "Tests.log"
        command = [sys.executable, "-m", "pytest", "-q", "-ra", "-p", "no:cacheprovider", MODULE,
            "--basetemp=" + str(directory / "pytest"), "--junitxml=" + str(report), "--durations=20"]
        row: dict[str, Any] = {"attempt": index, "command": command, "evidence": str(evidence),
            "junit": str(report), "log": str(log), "status": "running",
            "started_utc": datetime.now(timezone.utc).isoformat()}
        summary["attempts"].append(row)
        save()
        print(f"[{index}/{repetitions}] ionic HTTP: {directory}", flush=True)
        started = time.monotonic()
        try:
            with log.open("w") as stream:
                result = subprocess.run(command, cwd=ROOT, env=environment(evidence),
                    stdout=stream, stderr=subprocess.STDOUT, check=False)
            row["exit_code"] = result.returncode
            row["counts"] = result_counts(report)
            row["status"] = "passed" if (result.returncode == 0 and row["counts"]["passed"] > 0
                and row["counts"]["failed"] == row["counts"]["errors"] == 0) else "failed"
        except (OSError, ET.ParseError) as error:
            row.update(status="failed", error=f"{type(error).__name__}: {error}")
        row["seconds"] = time.monotonic() - started
        save()
        print(f"[{index}/{repetitions}] {row['status']}: {row.get('counts', row.get('error'))}", flush=True)
    counts: Counter[str] = Counter()
    for row in summary["attempts"]:
        counts.update(row.get("counts", {}))
    summary["counts"] = dict(counts)
    summary["status"] = "passed" if all(row["status"] == "passed" for row in summary["attempts"]) else "failed"
    save()
    print(f"Ionic HTTP diagnostics: {summary['status']} — {repetitions} attempts; {dict(counts)}", flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.repetitions < 1:
        parser.error("repetitions must be a positive integer")
    summary = diagnose(args.repetitions, args.output_dir.resolve())
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
