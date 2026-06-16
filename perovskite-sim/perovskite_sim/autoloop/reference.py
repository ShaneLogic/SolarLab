# perovskite_sim/autoloop/reference.py
from __future__ import annotations

import csv
import json
import logging
import statistics
from pathlib import Path
from typing import Optional, Protocol

from perovskite_sim.experiments.jv_sweep import compute_metrics

logger = logging.getLogger(__name__)

_VALID_UNITS = {"mA/cm2", "A/m2"}
_VALID_SIGN = {"positive", "negative"}
_VALID_AGG = {"median", "mean", "champion"}


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


def _parse_jv_csv(path: Path) -> tuple[list, list]:
    """Read the first two numeric columns (V, J) from a CSV; skip header/junk rows."""
    V, J = [], []
    for row in csv.reader(path.read_text(encoding="utf-8").splitlines()):
        if len(row) < 2:
            continue
        try:
            v, j = float(row[0]), float(row[1])
        except ValueError:
            continue   # header or non-numeric row
        V.append(v)
        J.append(j)
    if len(V) < 2:
        raise ValueError(f"{path}: fewer than 2 numeric (V, J) rows")
    return V, J


def _metrics_to_base(m) -> dict:
    return {"Voc_V": m.V_oc, "Jsc_mA_cm2": m.J_sc / 10.0,
            "FF_percent": m.FF * 100.0, "PCE_percent": m.PCE * 100.0}


class LabReferenceSource:
    """Ingest measured J-V CSV(s) -> base {Voc,Jsc,FF,PCE} via compute_metrics.
    Tiered use: supplies base only (sweep -> None)."""

    def __init__(self, jv_path, *, units: str = "mA/cm2", sign: str = "positive",
                 aggregate: str = "median"):
        if units not in _VALID_UNITS:
            raise ValueError(f"units must be one of {_VALID_UNITS}, got {units!r}")
        if sign not in _VALID_SIGN:
            raise ValueError(f"sign must be one of {_VALID_SIGN}, got {sign!r}")
        if aggregate not in _VALID_AGG:
            raise ValueError(f"aggregate must be one of {_VALID_AGG}, got {aggregate!r}")
        self.jv_path = Path(jv_path)
        self.units, self.sign, self.aggregate = units, sign, aggregate
        self._base = self._compute_base()

    def _csv_files(self) -> list:
        if self.jv_path.is_dir():
            files = sorted(self.jv_path.glob("*.csv"))
            if not files:
                raise ValueError(f"no .csv files in {self.jv_path}")
            return files
        return [self.jv_path]

    def _device_metrics(self, csv_path: Path):
        V, J = _parse_jv_csv(csv_path)
        scale = 10.0 if self.units == "mA/cm2" else 1.0
        J_Am2 = [j * scale for j in J]
        return compute_metrics(V, J_Am2, assume_jsc_positive=(self.sign == "positive"))

    def _compute_base(self) -> dict:
        metrics = []
        for f in self._csv_files():
            try:
                m = self._device_metrics(f)
            except Exception as exc:          # noqa: BLE001 — logged, not swallowed
                logger.warning("lab device %s unreadable: %r — skipped", f, exc)
                continue
            if not m.voc_bracketed:
                logger.warning("lab device %s did not bracket V_oc — skipped", f)
                continue
            metrics.append(m)
        if not metrics:
            raise ValueError(f"no valid (V_oc-bracketed) lab device in {self.jv_path}")
        if self.aggregate == "champion":
            return _metrics_to_base(max(metrics, key=lambda m: m.PCE))
        agg = statistics.median if self.aggregate == "median" else statistics.mean
        return {"Voc_V": agg([m.V_oc for m in metrics]),
                "Jsc_mA_cm2": agg([m.J_sc for m in metrics]) / 10.0,
                "FF_percent": agg([m.FF for m in metrics]) * 100.0,
                "PCE_percent": agg([m.PCE for m in metrics]) * 100.0}

    def base_metrics(self) -> dict:
        return dict(self._base)

    def sweep(self, sheet: str):
        return None

    def sweep_sheets(self) -> list:
        return []


class TieredReferenceSource:
    """base_metrics() from base_source (Lab); sweep()/sweep_sheets() from sweep_source (SCAPS)."""

    def __init__(self, base_source: ReferenceSource, sweep_source: ReferenceSource):
        self._base = base_source
        self._sweep = sweep_source

    def base_metrics(self) -> dict:
        return self._base.base_metrics()

    def sweep(self, sheet: str):
        return self._sweep.sweep(sheet)

    def sweep_sheets(self) -> list:
        return self._sweep.sweep_sheets()


def build_reference_source(path) -> ReferenceSource:
    """Dispatch on file shape: scaps-json -> ScapsReferenceSource;
    reference descriptor ({scaps, lab}) -> TieredReferenceSource (Task 3)."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if "base_model" in data and "sweeps" in data:
        return ScapsReferenceSource(path)
    if "scaps" in data and "lab" in data:
        base_dir = path.parent
        scaps = ScapsReferenceSource(base_dir / data["scaps"])
        lab_cfg = data["lab"]
        lab = LabReferenceSource(
            base_dir / lab_cfg["jv_csv"],
            units=lab_cfg.get("units", "mA/cm2"),
            sign=lab_cfg.get("sign", "positive"),
            aggregate=lab_cfg.get("aggregate", "median"))
        return TieredReferenceSource(base_source=lab, sweep_source=scaps)
    raise ValueError(f"unrecognised reference file shape: {path}")
