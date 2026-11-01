"""Explicit admission for fixtures that require a known checkout source record."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import subprocess

from solarlab.reproducibility.job_provenance import provenance_owner


@contextmanager
def checked_provenance_owner() -> Iterator[None]:
    """Verify installed bytes before the test supplies its checkout and lock.

    Ordinary installed producers still have unknown source history unless a
    caller explicitly supplies admission. Each use owns a separate job so
    distinct fixture inputs cannot accidentally reuse the preceding record.
    """
    root = Path(__file__).resolve().parents[2]
    for package in ("solarlab", "solarlab_research", "solarlab_server"):
        spec = importlib.util.find_spec(package)
        if spec is None or spec.origin is None:
            raise ValueError("fixture package has no source origin: " + package)
        installed = Path(spec.origin).resolve().parent
        names = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z",
             "--", "src/" + package], cwd=root,
        ).decode().split("\0")
        for name in filter(None, names):
            source = root / name
            target = installed / source.relative_to(root / "src" / package)
            if not target.is_file() or target.read_bytes() != source.read_bytes():
                raise ValueError("installed fixture source differs from test checkout: " + name)
    with provenance_owner(source_root=root, dirty_policy="record"):
        yield
