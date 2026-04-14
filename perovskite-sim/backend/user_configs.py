"""User-preset filesystem helpers for Phase 2b layer builder.

Kept in its own module so filename validation and shipped-name reservation
can be unit-tested without spinning up FastAPI. The module is the single
authority on:

  * the user-preset filename grammar (a strict ASCII regex);
  * which names are reserved by shipped presets;
  * atomic file creation that prevents TOCTOU collisions on first save.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

USER_FILENAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

CONFIGS_ROOT = Path(__file__).resolve().parent.parent / "configs"
USER_CONFIGS_ROOT = CONFIGS_ROOT / "user"


def validate_user_filename(name: str) -> None:
    """Raise ValueError if name is not a safe user-preset filename."""
    if not isinstance(name, str) or not USER_FILENAME_RE.match(name):
        raise ValueError(
            f"Invalid filename {name!r}: must match {USER_FILENAME_RE.pattern}"
        )


def is_shipped_name(name: str) -> bool:
    """Return True if name collides with a shipped (top-level) preset.

    Callers must pass the bare name, without an extension.
    """
    if "." in name or "/" in name:
        return False
    return (CONFIGS_ROOT / f"{name}.yaml").exists()


def write_user_config(
    name: str,
    body: dict,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a validated user config atomically.

    Uses ``os.O_EXCL`` on first save so two concurrent saves of the same
    name cannot both succeed (the second raises ``FileExistsError``). A
    second save with ``overwrite=True`` truncates the existing file.

    Raises:
        ValueError: filename does not match USER_FILENAME_RE.
        FileExistsError: name collides with a shipped preset, or the
            target already exists and ``overwrite`` is False.
    """
    validate_user_filename(name)
    if is_shipped_name(name):
        raise FileExistsError(f"{name!r} is reserved by a shipped preset")
    USER_CONFIGS_ROOT.mkdir(parents=True, exist_ok=True)
    target = USER_CONFIGS_ROOT / f"{name}.yaml"
    flags = os.O_WRONLY | os.O_CREAT | (os.O_TRUNC if overwrite else os.O_EXCL)
    try:
        fd = os.open(target, flags, 0o644)
    except FileExistsError:
        raise FileExistsError(f"{name!r} already exists")
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(body, f, default_flow_style=False, sort_keys=False)
    return target


def list_user_configs() -> list[str]:
    """Return a sorted list of user-preset bare names (no extension)."""
    if not USER_CONFIGS_ROOT.is_dir():
        return []
    return sorted(p.stem for p in USER_CONFIGS_ROOT.glob("*.yaml"))
