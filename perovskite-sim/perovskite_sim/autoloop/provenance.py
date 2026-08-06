# perovskite_sim/autoloop/provenance.py
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from perovskite_sim.autoloop.types import Provenance


_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_GIT_TIMEOUT_S = 10.0


def config_hash(path: Path) -> str:
    """SHA-256 of a config file's bytes (content-addressed reproducibility).
    Returns a sentinel hash of b'' when the file does not exist (e.g. in unit tests
    where the orchestrator is called with a notional config path).
    """
    p = Path(path)
    data = p.read_bytes() if p.exists() else b""
    return hashlib.sha256(data).hexdigest()


def _git(*args: str) -> str | None:
    """Run bounded Git metadata queries; ``None`` means unavailable."""
    try:
        return subprocess.run(
            ["git", *args],
            capture_output=True, text=True, check=True,
            cwd=_PACKAGE_ROOT,
            timeout=_GIT_TIMEOUT_S,
        ).stdout.strip()
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
    ):
        return None


def stamp(*, run_id: str, config_path: Path, flags: dict[str, str],
          seed: int, timestamp: str) -> Provenance:
    """Build a Provenance record. ``timestamp`` is passed in (ISO-8601),
    never generated here, so a run is reproducible/replayable."""
    sha = _git("rev-parse", "HEAD")
    git_root_raw = _git("rev-parse", "--show-toplevel")
    status: str | None = None
    if git_root_raw:
        try:
            package_pathspec = _PACKAGE_ROOT.relative_to(Path(git_root_raw))
        except ValueError:
            pass
        else:
            # Scope the query to the simulated package and tracked files. A
            # repository-wide status can block indefinitely when unrelated
            # OneDrive paths are dataless placeholders. The active config has
            # its own content hash, so untracked research output is irrelevant
            # to this provenance bit.
            status = _git(
                "status",
                "--porcelain",
                "--untracked-files=no",
                "--",
                package_pathspec.as_posix(),
            )
    # Unknown status is conservatively dirty; never label an uninspectable
    # working tree as clean.
    dirty = status is None or bool(status)
    return Provenance(
        run_id=run_id,
        git_sha=sha or "unknown",
        git_dirty=dirty,
        config_hash=config_hash(config_path),
        flags=dict(flags),
        seed=seed,
        timestamp=timestamp,
    )
