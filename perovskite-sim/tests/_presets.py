"""Resolve a device preset by file name across the shipped and fixture roots.

Shipped research presets live in ``configs/``; the historical presets used by
the regression lanes live in ``tests/fixtures/configs/``.
"""
from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
PRESET_ROOTS = (
    _ROOT / "configs",
    _ROOT / "configs" / "twod",
    _ROOT / "tests" / "fixtures" / "configs",
    _ROOT / "tests" / "fixtures" / "configs" / "twod",
)


def preset_path(name: str) -> Path:
    for root in PRESET_ROOTS:
        candidate = root / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no preset named {name!r} under {[str(r) for r in PRESET_ROOTS]}")
