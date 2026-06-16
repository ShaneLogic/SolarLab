# Autoloop Stage 4b — L3 Real-Lab-Data Ingest Seam — Design

**Date:** 2026-06-16
**Status:** Design approved, pre-planning
**Parent design:** `docs/superpowers/specs/2026-06-16-autoloop-research-pipeline-design.md` (L3 data-ingest seam)
**Builds on:** Stages 1–4a (guardian/attribution/auto-implement/boulder), all merged to `main`. Reuses `scorecard.score_parity`, `_probe_worker`, `experiments.jv_sweep.compute_metrics`.

**Scope note:** Stage 4 split = 4a boulder (done), **4b L3 data seam (this)**, 4c L4 design-search (separate later spec).

---

## 1. Problem & scope

The whole loop scores parity against `tests/integration/scaps_reference.json` — SCAPS
is the ground truth. 4b makes **real measured device data** a pluggable ground-truth
source so the loop can anchor absolutes to hardware.

**Current state:** no real measured data exists in the repo (verified — only the
simulator's own `eqe.py` and a partner-PPT figure). So 4b **architects the seam + a
real-data adapter + example fixtures now**; real measurements drop in later with zero
rework (the "partner data blocked" reality from project memory).

**Decisions (locked during brainstorming):**

| Axis | Decision |
|------|----------|
| Ground-truth model | **Tiered.** Real data overrides the **base** `{Voc,Jsc,FF,PCE}`; SCAPS still supplies the trend **sweeps** (real data = single-device base points, rarely parametric series). |
| Lab input format | **Measured J-V CSV** `(V, J)` per device → `{Voc,Jsc,FF,PCE}` via the simulator's own `compute_metrics` (apples-to-apples extraction). |
| Threading | **Content-dispatch factory** — `build_reference_source(path)` reads the file: scaps-json → `ScapsReferenceSource`; reference descriptor → `TieredReferenceSource`. `scorecard`/`_probe_worker` swap `json.loads(ref)` → the factory; guardian/ladder/boulder are **untouched**. |

**Explicitly deferred:** EQE ingestion; real parametric *sweeps* from the lab;
uncertainty-*weighted* scoring; vendor-specific instrument parsers; 4c (L4).

## 2. The `ReferenceSource` seam

New `perovskite_sim/autoloop/reference.py`:

```python
class ReferenceSource(Protocol):
    def base_metrics(self) -> dict          # {"Voc_V","Jsc_mA_cm2","FF_percent","PCE_percent"}
    def sweep(self, sheet: str) -> Optional[dict]   # {"points":[{x,Voc_V,Jsc_mA_cm2,FF_percent,PCE_percent}]} or None
    def sweep_sheets(self) -> list[str]
```

- **`ScapsReferenceSource(path)`** — wraps `scaps_reference.json` (`base_model` + `sweeps`). Default → bit-identical to today.
- **`LabReferenceSource(jv_path, units, sign, aggregate)`** — measured J-V → base (§4). `sweep()`→None.
- **`TieredReferenceSource(base_source, sweep_source)`** — `base_metrics()` from base (Lab); `sweep()`/`sweep_sheets()` from sweep (SCAPS).

**Content-dispatch factory** (the zero-threading-churn trick):
```python
def build_reference_source(path) -> ReferenceSource:
    data = json.loads(Path(path).read_text())
    if "base_model" in data and "sweeps" in data:
        return ScapsReferenceSource(path)
    if "scaps" in data and "lab" in data:            # reference descriptor
        scaps = ScapsReferenceSource(<dir>/data["scaps"])
        lab = LabReferenceSource(<dir>/data["lab"]["jv_csv"],
                                 units=data["lab"].get("units","mA/cm2"),
                                 sign=data["lab"].get("sign","positive"),
                                 aggregate=data["lab"].get("aggregate","median"))
        return TieredReferenceSource(base_source=lab, sweep_source=scaps)
    raise ValueError(f"unrecognised reference file shape: {path}")
```
(Descriptor paths resolve relative to the descriptor's own directory.)

## 3. LabReferenceSource (J-V CSV adapter)

`LabReferenceSource(jv_path, *, units="mA/cm2", sign="positive", aggregate="median")`:

- **Ingest:** `jv_path` = a single J-V CSV (one device) or a directory of per-device CSVs. Each CSV = two columns `(V[volts], J[current density])` (header optional; parse the two numeric columns).
- **Per device → metrics:**
  1. normalize J → A/m² (`× 10` if `units=="mA/cm2"`; `× 1` if `"A/m2"`).
  2. `compute_metrics(V, J_Am2, assume_jsc_positive=(sign=="positive"))` → `JVMetrics`.
  3. **skip + log** any device with `voc_bracketed=False` (curve never crossed zero → bad/clipped; don't feed a sentinel-zero base).
- **Aggregate** bracketed devices (`median` default | `champion`=best PCE | `mean`) → `base_metrics()` in the **exact scaps base_model schema**:
  ```python
  {"Voc_V": V_oc, "Jsc_mA_cm2": J_sc/10.0, "FF_percent": FF*100.0, "PCE_percent": PCE*100.0}
  ```
- `sweep()`→None, `sweep_sheets()`→[]. Device spread carried as optional uncertainty (report-only).
- **Validation at construction:** unknown `units`/`sign`/`aggregate` → raise; path missing → raise; no valid (bracketed) device → raise (never silently anchor a base on nothing).
- **Sign default** `"positive"` (J>0 at V=0, active-cell — matches the sim + the J-V convention memory); `"negative"` flips a load-convention export.

## 4. Wiring

Two one-line read swaps; the factory makes both lab-aware transparently:

- **`scorecard.score_parity`:** `source = build_reference_source(reference_path)`; per-sweep loop → `source.sweep(sheet)`; base_deltas → `source.base_metrics()`.
- **`_probe_worker._badness`:** same swap — trend-gap badness reads `source.sweep(sheet)["points"]` (SCAPS); absolute-base-gap badness compares the sim base to `source.base_metrics()` (Lab). An absolute-base gap now means **sim-vs-real-device** (the L3 calibration target).

**Going live — no new flag:** point the existing `--reference` at a descriptor:
```bash
python scripts/autoloop_run.py --once|--attribute|--implement|--boulder \
    --reference tests/integration/scaps_lab_tiered.json
```
Default `--reference` stays `scaps_reference.json` (pure SCAPS, bit-identical). The whole
pipeline anchors absolutes to hardware with zero change beyond the seam.

## 5. Fixtures (committed; replace with real files later)

- `tests/integration/lab_data_example/device_01.csv` — synthetic diode J-V (V 0→1.2, crosses zero at V_oc≈1.1) so the adapter + drop-in path are exercised. (A 2nd/3rd device CSV for aggregate tests.)
- `tests/integration/scaps_lab_tiered.json` — descriptor: `{"scaps": "scaps_reference.json", "lab": {"jv_csv": "lab_data_example/", "units": "mA/cm2", "sign": "positive", "aggregate": "median"}}`.

## 6. Error handling

- Descriptor → missing scaps/lab path → surfaced `FileNotFoundError`.
- Malformed CSV (non-numeric / wrong columns / <2 rows) → skip + log; no valid device → raise.
- All devices unbracketed → raise.
- Unknown `units`/`sign`/`aggregate` → raise at construction with a clear message.
- Plain `scaps_reference.json` → `ScapsReferenceSource`, bit-identical, lab code untouched.
- **Never** silently fall back to the SCAPS base when lab data is configured but unreadable — fail loud (a silent fallback would mask a bad data dir).

## 7. Testing

- `reference.py`: `build_reference_source` dispatch (scaps-json→Scaps, descriptor→Tiered, junk→raises); `ScapsReferenceSource` base/sweep round-trip; `TieredReferenceSource` (base=lab, sweep=scaps).
- `LabReferenceSource`: synthetic J-V CSV → metrics within tol of a known curve; units mA/cm²↔A/m² agree; sign flip; aggregate median vs champion across 2–3 device CSVs; skip-unbracketed (clipped curve excluded); no-valid-device → raises; unknown-config → raises.
- **Regression (critical):** `score_parity` given a plain scaps path is **bit-identical** to pre-4b (the seam preserves SCAPS behavior). Run the existing autoloop scorecard tests unchanged.
- `_probe_worker`: base-badness against a tiered descriptor uses the lab base.
- integration smoke (slow): real `--once --reference <descriptor>` on `scaps_mirror_v2` → guardian scores parity with the lab-base override; assert it ran + `base_deltas` reflect the lab base (≠ SCAPS base). Bounded.

## 8. Out of scope / deferred

- EQE ingestion (J-V only).
- Real parametric *sweeps* from the lab (would extend `LabReferenceSource.sweep()`).
- Uncertainty-*weighted* scoring (carry uncertainty; point-compare stays).
- Vendor-specific instrument parsers (generic 2-column CSV now).
- 4c (L4 design-search).

## 9. Build order (staged tasks for writing-plans)

1. `reference.py` — `ReferenceSource` Protocol + `ScapsReferenceSource` + `build_reference_source` (scaps-json dispatch only; preserves current behavior) (+ tests).
2. `LabReferenceSource` (J-V CSV → compute_metrics, units/sign/aggregate, skip-unbracketed, validation) + synthetic CSV fixtures (+ tests).
3. `TieredReferenceSource` + descriptor dispatch in the factory + descriptor fixture (+ tests).
4. Refactor `scorecard.score_parity` + `_probe_worker._badness` to read via `build_reference_source` (+ bit-identical regression test).
5. integration smoke (`--reference <descriptor>`) + docs (README / CLAUDE.md).
