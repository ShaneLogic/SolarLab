"""Test prerequisites; production source admission remains authoritative."""
from __future__ import annotations

from pathlib import Path
import subprocess


def changed_bound_core_sources(root: Path) -> tuple[str, ...]:
    """List changed Python sources that the runtime binds to committed core."""
    commands = (
        ["git", "diff", "--name-only", "-z", "HEAD", "--", "src/solarlab"],
        ["git", "ls-files", "--others", "--exclude-standard", "-z", "--", "src/solarlab"],
    )
    changed: set[str] = set()
    for command in commands:
        result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=10)
        if result.returncode:
            raise RuntimeError("a Git checkout is required to verify committed core sources")
        changed.update(path for path in result.stdout.split("\0") if path.endswith(".py"))
    return tuple(sorted(changed))


def clean_core_skip_reason(paths: tuple[str, ...]) -> str | None:
    if not paths:
        return None
    return (
        "requires clean committed core sources; uncommitted bound files: "
        + ", ".join(paths)
        + "; run this source-binding test in a clean checkout"
    )
