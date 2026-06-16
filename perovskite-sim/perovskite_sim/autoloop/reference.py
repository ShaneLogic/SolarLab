# perovskite_sim/autoloop/reference.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Protocol


class ReferenceSource(Protocol):
    def base_metrics(self) -> dict: ...
    def sweep(self, sheet: str) -> Optional[dict]: ...
    def sweep_sheets(self) -> list: ...


class ScapsReferenceSource:
    """Wraps a scaps_reference.json ({base_model} + {sweeps}). Default ground truth."""

    def __init__(self, path):
        self.path = Path(path)
        self._data = json.loads(self.path.read_text(encoding="utf-8"))

    def base_metrics(self) -> dict:
        return dict(self._data["base_model"])

    def sweep(self, sheet: str) -> Optional[dict]:
        return self._data["sweeps"].get(sheet)

    def sweep_sheets(self) -> list:
        return list(self._data["sweeps"].keys())


def build_reference_source(path) -> ReferenceSource:
    """Dispatch on file shape: scaps-json -> ScapsReferenceSource;
    reference descriptor ({scaps, lab}) -> TieredReferenceSource (Task 3)."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if "base_model" in data and "sweeps" in data:
        return ScapsReferenceSource(path)
    raise ValueError(f"unrecognised reference file shape: {path}")
