# Autoloop Stage 4b — L3 Real-Lab-Data Ingest Seam — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pluggable `ReferenceSource` seam so the autoloop can score parity against real measured device data (tiered: lab base + SCAPS sweeps) instead of only the SCAPS reference — with zero plumbing change to guardian/ladder/boulder.

**Architecture:** A new `reference.py` with a `ReferenceSource` protocol + `ScapsReferenceSource` / `LabReferenceSource` / `TieredReferenceSource` + a content-dispatch factory `build_reference_source(path)` (scaps-json → Scaps; reference descriptor → Tiered). `scorecard.score_parity` and `_probe_worker._badness` swap their raw `json.loads(ref)` for the factory; everything else is untouched. `LabReferenceSource` ingests measured J-V CSVs via the simulator's own `compute_metrics`.

**Tech Stack:** Python 3.9+, csv, statistics, dataclasses. Reuses `experiments.jv_sweep.compute_metrics` + the Stage-1 scorecard. No new third-party deps.

---

## Design contract (read before starting)

- **Spec:** `docs/superpowers/specs/2026-06-16-autoloop-stage4b-l3-data-seam-design.md`.
- **Ground truth today:** `scorecard.score_parity` + `_probe_worker._badness` do `json.loads(reference_path)` then read `["base_model"]` (dict `{Voc_V, Jsc_mA_cm2, FF_percent, PCE_percent}`) and `["sweeps"][sheet]` (`{"points":[{x, Voc_V, Jsc_mA_cm2, FF_percent, PCE_percent}]}`). The seam returns those same shapes.
- **`compute_metrics`** (verified): `perovskite_sim.experiments.jv_sweep.compute_metrics(V, J, *, assume_jsc_positive=True) -> JVMetrics(V_oc, J_sc, FF, PCE, voc_bracketed)`. `J_sc` in **A/m²**, `FF`/`PCE` are **fractions**. It does `np.asarray` internally → lists are fine.
- **base_model schema mapping** (lab → scaps base shape): `Voc_V=V_oc`, `Jsc_mA_cm2=J_sc/10`, `FF_percent=FF*100`, `PCE_percent=PCE*100`.
- **Sign:** `"positive"` = J>0 at V=0 (active-cell, the sim convention) → `assume_jsc_positive=True`.
- **Run all commands from `perovskite-sim/`.** Tests default to `-m 'not slow'`.

## File Structure

```
perovskite_sim/autoloop/
  reference.py      ReferenceSource protocol + Scaps/Lab/Tiered sources + build_reference_source
  scorecard.py      score_parity: json.loads -> build_reference_source
  _probe_worker.py  _badness: json.loads -> build_reference_source
tests/unit/autoloop/
  test_reference_scaps.py
  test_reference_lab.py
  test_reference_tiered.py
  test_scorecard_via_seam.py
tests/integration/
  lab_data_example/device_01.csv, device_02.csv, device_03.csv   (synthetic fixtures)
  scaps_lab_tiered.json                                          (descriptor fixture)
  test_autoloop_l3_seam.py   (slow)
```

---

## Task 1: ReferenceSource protocol + ScapsReferenceSource + factory (scaps-only)

**Files:**
- Create: `perovskite_sim/autoloop/reference.py`
- Test: `tests/unit/autoloop/test_reference_scaps.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/autoloop/test_reference_scaps.py
import json
import pytest
from perovskite_sim.autoloop.reference import (
    ScapsReferenceSource, build_reference_source,
)

_SCAPS = {
    "base_model": {"Voc_V": 1.17, "Jsc_mA_cm2": 26.28, "FF_percent": 87.0, "PCE_percent": 26.7},
    "sweeps": {"CHI_ETL": {"x_name": "dEc", "n_points": 1,
                           "points": [{"x": 0.0, "Voc_V": 1.25, "Jsc_mA_cm2": 26.3,
                                       "FF_percent": 90.0, "PCE_percent": 29.6}]}},
}


def _write(tmp_path, obj, name="scaps_reference.json"):
    p = tmp_path / name
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


def test_scaps_source_base_and_sweep(tmp_path):
    src = ScapsReferenceSource(_write(tmp_path, _SCAPS))
    assert src.base_metrics()["Voc_V"] == 1.17
    assert src.sweep("CHI_ETL")["points"][0]["Voc_V"] == 1.25
    assert src.sweep("MISSING") is None
    assert src.sweep_sheets() == ["CHI_ETL"]


def test_factory_dispatches_scaps_json(tmp_path):
    src = build_reference_source(_write(tmp_path, _SCAPS))
    assert isinstance(src, ScapsReferenceSource)


def test_factory_raises_on_junk(tmp_path):
    with pytest.raises(ValueError):
        build_reference_source(_write(tmp_path, {"nonsense": 1}, "junk.json"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_reference_scaps.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'perovskite_sim.autoloop.reference'`.

- [ ] **Step 3: Write `reference.py` (scaps + factory only)**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_reference_scaps.py`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add perovskite-sim/perovskite_sim/autoloop/reference.py perovskite-sim/tests/unit/autoloop/test_reference_scaps.py
git commit -m "feat(autoloop): add ReferenceSource seam + ScapsReferenceSource + factory (Stage 4b)"
```

---

## Task 2: LabReferenceSource (J-V CSV adapter) + fixtures

**Files:**
- Modify: `perovskite_sim/autoloop/reference.py`
- Create: `tests/integration/lab_data_example/device_01.csv`, `device_02.csv`, `device_03.csv`, `unbracketed.csv`
- Test: `tests/unit/autoloop/test_reference_lab.py`

- [ ] **Step 1: Create the synthetic J-V fixtures**

`tests/integration/lab_data_example/device_01.csv`:
```
V,J_mA_cm2
0.0,24.0
0.2,23.9
0.4,23.8
0.6,23.6
0.8,23.0
0.9,21.5
1.0,16.0
1.05,9.0
1.1,0.5
1.12,-6.0
1.15,-20.0
```

`tests/integration/lab_data_example/device_02.csv` (slightly higher Jsc):
```
V,J_mA_cm2
0.0,25.0
0.4,24.8
0.8,24.0
0.9,22.5
1.0,17.0
1.05,9.5
1.1,1.0
1.13,-6.0
1.16,-20.0
```

`tests/integration/lab_data_example/device_03.csv` (slightly lower Jsc):
```
V,J_mA_cm2
0.0,23.0
0.4,22.8
0.8,22.0
0.9,20.5
1.0,15.0
1.05,8.0
1.09,0.5
1.11,-6.0
1.14,-20.0
```

`tests/integration/lab_data_example/unbracketed.csv` (never crosses zero — bad data):
```
V,J_mA_cm2
0.0,24.0
0.5,23.0
1.0,20.0
1.2,15.0
```

- [ ] **Step 2: Write the failing test**

```python
# tests/unit/autoloop/test_reference_lab.py
import pytest
from pathlib import Path
from perovskite_sim.autoloop.reference import LabReferenceSource

FIX = Path(__file__).resolve().parents[2] / "tests" / "integration" / "lab_data_example"
D1 = FIX / "device_01.csv"


def test_single_device_base_metrics_sane():
    src = LabReferenceSource(D1)
    b = src.base_metrics()
    assert 1.0 <= b["Voc_V"] <= 1.2
    assert 20.0 <= b["Jsc_mA_cm2"] <= 28.0
    assert 0.0 < b["FF_percent"] < 100.0
    assert b["PCE_percent"] > 0.0
    assert src.sweep("CHI_ETL") is None and src.sweep_sheets() == []


def test_units_conversion_agrees():
    # Same curve declared in A/m2 (x10) must give the same metrics.
    import csv, tempfile, os
    rows = [(float(r[0]), float(r[1]) * 10.0)
            for r in csv.reader(D1.read_text().splitlines()) if _num(r)]
    d = tempfile.mkdtemp()
    p = Path(d) / "dev_am2.csv"
    p.write_text("V,J\n" + "\n".join(f"{v},{j}" for v, j in rows), encoding="utf-8")
    a = LabReferenceSource(D1, units="mA/cm2").base_metrics()
    b = LabReferenceSource(p, units="A/m2").base_metrics()
    assert abs(a["Voc_V"] - b["Voc_V"]) < 1e-6
    assert abs(a["Jsc_mA_cm2"] - b["Jsc_mA_cm2"]) < 1e-6


def _num(r):
    try:
        float(r[0]); float(r[1]); return True
    except (ValueError, IndexError):
        return False


def test_directory_aggregate_median_vs_champion():
    med = LabReferenceSource(FIX, aggregate="median").base_metrics()
    champ = LabReferenceSource(FIX, aggregate="champion").base_metrics()
    # the unbracketed.csv is skipped; 3 valid devices remain
    assert med["Jsc_mA_cm2"] != champ["Jsc_mA_cm2"] or med["PCE_percent"] <= champ["PCE_percent"]
    assert champ["PCE_percent"] >= med["PCE_percent"]   # champion = best PCE


def test_unbracketed_only_raises(tmp_path):
    p = tmp_path / "only_bad.csv"
    p.write_text("V,J\n0.0,24\n0.5,23\n1.0,20\n", encoding="utf-8")
    with pytest.raises(ValueError):
        LabReferenceSource(p)


def test_unknown_config_raises():
    with pytest.raises(ValueError):
        LabReferenceSource(D1, units="furlongs")
    with pytest.raises(ValueError):
        LabReferenceSource(D1, sign="sideways")
    with pytest.raises(ValueError):
        LabReferenceSource(D1, aggregate="vibes")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_reference_lab.py`
Expected: FAIL — `ImportError: cannot import name 'LabReferenceSource'`.

- [ ] **Step 4: Add `LabReferenceSource` to `reference.py`**

Add imports at the top:

```python
import csv
import logging
import statistics
from perovskite_sim.experiments.jv_sweep import compute_metrics

logger = logging.getLogger(__name__)

_VALID_UNITS = {"mA/cm2", "A/m2"}
_VALID_SIGN = {"positive", "negative"}
_VALID_AGG = {"median", "mean", "champion"}
```

Add the class + helpers:

```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_reference_lab.py`
Expected: PASS (5 tests).

- [ ] **Step 6: Commit**

```bash
git add perovskite-sim/perovskite_sim/autoloop/reference.py perovskite-sim/tests/unit/autoloop/test_reference_lab.py perovskite-sim/tests/integration/lab_data_example/
git commit -m "feat(autoloop): add LabReferenceSource (measured J-V CSV -> base via compute_metrics)"
```

---

## Task 3: TieredReferenceSource + descriptor dispatch

**Files:**
- Modify: `perovskite_sim/autoloop/reference.py`
- Create: `tests/integration/scaps_lab_tiered.json`
- Test: `tests/unit/autoloop/test_reference_tiered.py`

- [ ] **Step 1: Create the descriptor fixture**

`tests/integration/scaps_lab_tiered.json`:
```json
{
  "scaps": "scaps_reference.json",
  "lab": {"jv_csv": "lab_data_example/", "units": "mA/cm2", "sign": "positive", "aggregate": "median"}
}
```
(Paths are relative to this descriptor's own directory: `tests/integration/`.)

- [ ] **Step 2: Write the failing test**

```python
# tests/unit/autoloop/test_reference_tiered.py
import json
from pathlib import Path
from perovskite_sim.autoloop.reference import (
    TieredReferenceSource, ScapsReferenceSource, LabReferenceSource,
    build_reference_source,
)

INTEG = Path(__file__).resolve().parents[2] / "tests" / "integration"


def test_tiered_base_from_lab_sweeps_from_scaps():
    scaps = ScapsReferenceSource(INTEG / "scaps_reference.json")
    lab = LabReferenceSource(INTEG / "lab_data_example")
    t = TieredReferenceSource(base_source=lab, sweep_source=scaps)
    assert t.base_metrics() == lab.base_metrics()            # base from lab
    assert t.sweep("Nd_ETL") == scaps.sweep("Nd_ETL")        # sweeps from scaps
    assert t.sweep_sheets() == scaps.sweep_sheets()


def test_factory_dispatches_descriptor_to_tiered():
    src = build_reference_source(INTEG / "scaps_lab_tiered.json")
    assert isinstance(src, TieredReferenceSource)
    # base comes from the lab fixtures, not the scaps base_model
    scaps_base = ScapsReferenceSource(INTEG / "scaps_reference.json").base_metrics()
    assert src.base_metrics()["Voc_V"] != scaps_base["Voc_V"]
    assert src.sweep("Nd_ETL") is not None                   # SCAPS sweep present
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_reference_tiered.py`
Expected: FAIL — `ImportError: cannot import name 'TieredReferenceSource'`.

- [ ] **Step 4: Add `TieredReferenceSource` + extend the factory**

Add the class to `reference.py`:

```python
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
```

Extend `build_reference_source` — replace the `raise` with the descriptor branch:

```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_reference_tiered.py`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add perovskite-sim/perovskite_sim/autoloop/reference.py perovskite-sim/tests/unit/autoloop/test_reference_tiered.py perovskite-sim/tests/integration/scaps_lab_tiered.json
git commit -m "feat(autoloop): add TieredReferenceSource + descriptor dispatch (real base + SCAPS sweeps)"
```

---

## Task 4: Refactor scorecard + _probe_worker to read via the seam

**Files:**
- Modify: `perovskite_sim/autoloop/scorecard.py` (`score_parity`, lines 45/49/71)
- Modify: `perovskite_sim/autoloop/_probe_worker.py` (`_badness`, lines 31/37/49)
- Test: `tests/unit/autoloop/test_scorecard_via_seam.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/autoloop/test_scorecard_via_seam.py
import json
from pathlib import Path
from perovskite_sim.autoloop.scorecard import score_parity
from perovskite_sim.autoloop.types import ParityScore

INTEG = Path(__file__).resolve().parents[2] / "tests" / "integration"

_SCAPS = {
    "base_model": {"Voc_V": 1.17, "Jsc_mA_cm2": 26.28, "FF_percent": 87.0, "PCE_percent": 26.7},
    "sweeps": {"CHI_ETL": {"n_points": 2, "points": [
        {"x": -0.5, "Voc_V": 0.83, "Jsc_mA_cm2": 26.3, "FF_percent": 82.0, "PCE_percent": 18.0},
        {"x": 0.0, "Voc_V": 1.25, "Jsc_mA_cm2": 26.3, "FF_percent": 90.0, "PCE_percent": 29.6}]}},
}


def test_score_parity_plain_scaps_unchanged(tmp_path):
    # Regression: a plain scaps_reference.json behaves exactly as pre-4b.
    p = tmp_path / "scaps_reference.json"
    p.write_text(json.dumps(_SCAPS), encoding="utf-8")
    score = score_parity(
        reference_path=p, config_path=tmp_path / "c.yaml",
        run_point=lambda axis, x: ({-0.5: 0.83, 0.0: 1.25}[x], 263.0, 0.86, 0.27, True),
        base_point=lambda: (1.17, 262.8, 0.87, 0.267, True))
    assert isinstance(score, ParityScore)
    assert score.base_deltas["V_oc"] == 0.0          # 1.17 - 1.17 (scaps base)
    assert score.per_sweep["CHI_ETL"].voc_closure_pct == 100.0


def test_score_parity_tiered_uses_lab_base(tmp_path):
    # A descriptor reference -> base_deltas measured against the LAB base, not SCAPS.
    from perovskite_sim.autoloop.reference import build_reference_source
    lab_base = build_reference_source(INTEG / "scaps_lab_tiered.json").base_metrics()
    score = score_parity(
        reference_path=INTEG / "scaps_lab_tiered.json", config_path=tmp_path / "c.yaml",
        run_point=lambda axis, x: (1.0, 250.0, 0.8, 0.2, True),
        base_point=lambda: (lab_base["Voc_V"], lab_base["Jsc_mA_cm2"] * 10.0,
                            lab_base["FF_percent"] / 100.0, lab_base["PCE_percent"] / 100.0, True))
    assert abs(score.base_deltas["V_oc"]) < 1e-9      # sim base == lab base -> zero delta
    assert score.per_sweep["Nd_ETL"].n_points > 0     # SCAPS sweeps still scored
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_scorecard_via_seam.py`
Expected: FAIL — `test_score_parity_tiered_uses_lab_base` errors (score_parity does raw `json.loads` and chokes on the descriptor's missing `["sweeps"]`/`["base_model"]`).

- [ ] **Step 3: Refactor `score_parity`**

In `scorecard.py`, add the import:
```python
from perovskite_sim.autoloop.reference import build_reference_source
```
Replace line 45 and the two read sites:
```python
    source = build_reference_source(reference_path)
    per_sweep: dict[str, SweepScore] = {}

    for sheet, axis in SHEET_TO_AXIS.items():
        sweep = source.sweep(sheet)
        if sweep is None:
            ...
```
And the base block:
```python
    # base-point absolute deltas (solarlab - reference)
    bm = source.base_metrics()
    voc, jsc_A, ff, pce, _brk = base_point()
    base_deltas = { ... }   # unchanged body
```

- [ ] **Step 4: Refactor `_probe_worker._badness`**

In `_probe_worker.py`, add the import:
```python
from perovskite_sim.autoloop.reference import build_reference_source
```
Replace line 31 and the two read sites:
```python
    source = build_reference_source(payload["reference"])

    if payload["gap_kind"] == "trend":
        sheet = payload["gap_sweep"]
        axis = SHEET_TO_AXIS[sheet]
        sl, scaps = [], []
        for pt in source.sweep(sheet)["points"]:
            ...
```
And the absolute branch:
```python
    bm = source.base_metrics()
    ref_v = float(bm[_METRIC_KEY[payload["gap_metric"]]])
    ...   # unchanged unit handling
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_scorecard_via_seam.py tests/unit/autoloop/test_scorecard.py`
Expected: PASS — both the new seam tests AND the existing scorecard tests (bit-identical on plain scaps paths).

- [ ] **Step 6: Commit**

```bash
git add perovskite-sim/perovskite_sim/autoloop/scorecard.py perovskite-sim/perovskite_sim/autoloop/_probe_worker.py perovskite-sim/tests/unit/autoloop/test_scorecard_via_seam.py
git commit -m "refactor(autoloop): scorecard + probe_worker read ground truth via ReferenceSource seam"
```

---

## Task 5: Integration smoke + docs

**Files:**
- Create: `tests/integration/test_autoloop_l3_seam.py`
- Modify: `perovskite-sim/CLAUDE.md`, `README.md`

- [ ] **Step 1: Write the failing integration test (slow)**

```python
# tests/integration/test_autoloop_l3_seam.py
import pytest
from pathlib import Path
from perovskite_sim.autoloop.orchestrator import guardian_once
from perovskite_sim.autoloop.reference import build_reference_source

REPO_ROOT = Path(__file__).resolve().parents[1]
DESCRIPTOR = REPO_ROOT / "tests" / "integration" / "scaps_lab_tiered.json"
CFG = REPO_ROOT / "configs" / "scaps_mirror_v2.yaml"


@pytest.mark.slow
def test_guardian_scores_against_lab_base(tmp_path):
    report = guardian_once(
        ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
        reference_path=DESCRIPTOR, config_path=CFG, cycle=0,
        timestamp="2026-06-16T00:00:00Z", l0_paths=["tests/unit/autoloop"], baseline=None)
    assert report["overall"] is not None
    # the descriptor's lab base differs from the SCAPS base_model -> the loop
    # is anchoring absolutes to the (synthetic) lab device, end-to-end.
    lab_voc = build_reference_source(DESCRIPTOR).base_metrics()["Voc_V"]
    from perovskite_sim.autoloop.reference import ScapsReferenceSource
    scaps_voc = ScapsReferenceSource(REPO_ROOT / "tests" / "integration" / "scaps_reference.json").base_metrics()["Voc_V"]
    assert lab_voc != scaps_voc
```

- [ ] **Step 2: Run test to verify it fails (or is collected)**

Run: `cd perovskite-sim && python -m pytest -q -m slow tests/integration/test_autoloop_l3_seam.py`
Expected: FAIL until the seam is wired (Task 4); once green, confirms the guardian scores parity through a tiered descriptor end-to-end (real solver, bounded — one parity score).

- [ ] **Step 3: Docs**

Add to the **Autoloop** section of `perovskite-sim/CLAUDE.md`:

```markdown
### Stage 4b — L3 real-lab-data ingest seam

`autoloop/reference.py` makes ground truth pluggable: `build_reference_source(path)`
returns a `ScapsReferenceSource` for a `scaps_reference.json`, or a
`TieredReferenceSource` (LabReferenceSource base + SCAPS sweeps) for a reference
**descriptor** `{"scaps": "...json", "lab": {"jv_csv": "...", "units", "sign", "aggregate"}}`.
`LabReferenceSource` ingests measured J-V CSV(s) → base {Voc,Jsc,FF,PCE} via the
simulator's own `compute_metrics` (skips V_oc-unbracketed devices; median/champion/mean
aggregate). `scorecard` + `_probe_worker` read through the factory — guardian/ladder/
boulder are untouched. To anchor absolutes to hardware, point `--reference` at a descriptor:

    python scripts/autoloop_run.py --once --reference tests/integration/scaps_lab_tiered.json

Default `--reference` stays scaps_reference.json (pure SCAPS, bit-identical). Real lab
CSVs replace the synthetic `tests/integration/lab_data_example/*.csv` fixtures.
```

Add to `README.md` (next to the other autoloop lines):

```markdown
- **Autoloop L3 lab data** — point `--reference` at a tiered descriptor
  (`scaps_lab_tiered.json`) to score absolutes against measured J-V (LabReferenceSource)
  while keeping SCAPS trend sweeps. Default stays pure-SCAPS.
```

- [ ] **Step 4: Run tests**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop && python -m pytest -q -m slow tests/integration/test_autoloop_l3_seam.py`
Expected: all green. Also `python -m pytest -q` (full default suite) — confirm no import/collection regression.

- [ ] **Step 5: Commit**

```bash
git add perovskite-sim/tests/integration/test_autoloop_l3_seam.py perovskite-sim/CLAUDE.md README.md
git commit -m "feat(autoloop): L3 seam integration smoke + docs (Stage 4b)"
```

---

## Self-Review

**Spec coverage** (vs `2026-06-16-autoloop-stage4b-l3-data-seam-design.md`):
- §2 `ReferenceSource` protocol + `ScapsReferenceSource` + `build_reference_source` dispatch → Tasks 1/3. ✓
- §3 `LabReferenceSource` (J-V CSV → compute_metrics, units/sign/aggregate, skip-unbracketed, validation, base_model-schema output) → Task 2. ✓
- §4 wiring (scorecard + worker via factory; `--reference <descriptor>`) → Task 4 + Task 5 docs. ✓
- §5 fixtures (synthetic CSVs + descriptor) → Tasks 2/3. ✓
- §6 error handling (missing path → raise; malformed/unbracketed → skip/raise; unknown config → raise; plain scaps bit-identical) → Tasks 1/2 + the regression test in Task 4. ✓
- §7 testing (dispatch, lab metrics/units/sign/aggregate/skip, bit-identical regression, worker base via lab, slow smoke) → every task. ✓
- §8 deferred (EQE, lab sweeps, weighted scoring, vendor parsers, 4c) → correctly NOT built.

**Placeholder scan:** none — complete code/tests/commands/fixtures. The Task 4 refactor shows the exact lines to swap with the surrounding unchanged body called out.

**Type consistency:** `ReferenceSource.base_metrics()/sweep()/sweep_sheets()` consistent across all three sources + both read sites + tests. `LabReferenceSource(jv_path, units, sign, aggregate)` signature consistent Tasks 2/3 + factory. `build_reference_source(path)` consistent Tasks 1/3/4/5. base dict keys `{Voc_V, Jsc_mA_cm2, FF_percent, PCE_percent}` match the scaps `base_model` schema the scorecard already reads. `compute_metrics(V, J, *, assume_jsc_positive)` + `JVMetrics(.V_oc/.J_sc/.FF/.PCE/.voc_bracketed)` are verified real symbols.

---

## Execution Handoff

After saving, choose execution:

1. **Subagent-Driven (recommended)** — fresh subagent per task, two-stage review (same background-workflow as Stages 1–4a).
2. **Inline Execution** — batch tasks in this session with checkpoints.
