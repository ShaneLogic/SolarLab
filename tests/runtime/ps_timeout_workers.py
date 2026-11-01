"""Inject only the child's startup identity query, leaving parent sampling real."""
import os
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

from solarlab.experiments import worker


def identity_timeout(*args, misses):
    directory = Path(args[3])
    original = subprocess.run
    argv = ["ps", "-p", str(os.getpid()), "-o", "pid=,ppid=,lstart=,rss="]
    observations = []

    def observed(command, *positional, **options):
        if command != argv:
            return original(command, *positional, **options)
        assert options["timeout"] == 0.5
        row = {"pid": os.getpid(), "ppid": os.getppid(), "argv": command,
               "timeout": options["timeout"], "started_record_exists": (directory / "started.json").exists()}
        observations.append(row)
        began = time.monotonic()
        try:
            if len(observations) <= misses:
                # Spend the real subprocess deadline, then raise its exact exception.
                time.sleep(options["timeout"])
                row["outcome"] = "injected_timeout"
                raise subprocess.TimeoutExpired(command, options["timeout"])
            # A deterministic parser fixture separates this retry contract from
            # unrelated natural sampler misses under load; parent RSS stays real.
            row["outcome"] = "controlled_success"
            return subprocess.CompletedProcess(command, 0,
                stdout=f"{os.getpid()} {os.getppid()} Mon Jan 01 00:00:00 2024 1024\n", stderr="")
        finally:
            row["seconds"] = time.monotonic() - began
            worker._atomic_json(directory / "identity-probe.json", {"observations": observations})

    with patch.object(subprocess, "run", observed):
        worker._child_main(*args)
