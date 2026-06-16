# perovskite_sim/autoloop/provenance.py
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from perovskite_sim.autoloop.types import Provenance


def config_hash(path: Path) -> str:
    """SHA-256 of a config file's bytes (content-addressed reproducibility)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*args: str) -> str:
    """Run a git command from the repo root; '' on failure (no exception)."""
    try:
        return subprocess.run(
            ["git", *args],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def stamp(*, run_id: str, config_path: Path, flags: dict[str, str],
          seed: int, timestamp: str) -> Provenance:
    """Build a Provenance record. ``timestamp`` is passed in (ISO-8601),
    never generated here, so a run is reproducible/replayable."""
    sha = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    return Provenance(
        run_id=run_id,
        git_sha=sha or "unknown",
        git_dirty=dirty,
        config_hash=config_hash(config_path),
        flags=dict(flags),
        seed=seed,
        timestamp=timestamp,
    )
