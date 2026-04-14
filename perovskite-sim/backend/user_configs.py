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
