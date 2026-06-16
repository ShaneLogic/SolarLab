# Autoloop Stage 2 — Attribution Leg — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic attribution leg that diagnoses a gap's cause (bug / numerics / physics / data) by auto-ablating `SOLARLAB_*` flags + grid + dark-limiting probes, behind a pluggable `Attributor` seam — read-only (writes a `Hypothesis` to the ledger).

**Architecture:** Extends `perovskite_sim/autoloop/`. An ablation harness runs probe *variants* through an injected `ProbeRunner` (real = subprocess with env flags set; fake = canned) and records a **badness scalar (lower = closer to the SCAPS reference)** per probe. A `HeuristicAttributor` classifies the resulting matrix into a `Hypothesis` with an honest `uncertain` fallback and a negatives-guard. No solver/physics code changes.

**Tech Stack:** Python 3.9+, dataclasses, subprocess, json. Reuses Stage 1 (`Gap`/`Hypothesis`/`Ledger`/`build_run_callables`/`SHEET_TO_AXIS`/`provenance.stamp`/`_PKG_ROOT`). No new third-party deps.

---

## Design contract (read before starting)

- **Spec:** `docs/superpowers/specs/2026-06-16-autoloop-stage2-attribution-design.md`.
- **Badness scalar (the uniform metric):** every probe returns a float where **lower = closer to the SCAPS reference**. A flag that *improves* the gap yields `delta = variant_val - baseline_val < 0`. Definitions (computed by the real worker; canned by the fake):
  - **gap badness, trend gap** (`gap.kind == "trend"`): `100 - voc_closure_pct` for the gap's sweep sheet (0 = perfect closure).
  - **gap badness, absolute base gap** (`gap.kind == "absolute"`): `abs(base_metric - reference_base_metric)` for `gap.metric` (0 = exact).
  - **dark badness** (`measure == "dark"`): `abs(dark_J_sc)` (expect ~0; large = bug).
- **`ProbeRunner.run(variant: dict) -> float`** where `variant = {"env_flags": dict[str,str], "jv_overrides": dict, "measure": "gap"|"dark"}`. Returns the badness scalar. Raising is allowed — the harness wraps it.
- **Stage 1 APIs (verified, on `main`):**
  - `types.Gap(id, metric, sweep, sweep_point, solarlab_val, reference_val, gap_mag, kind, status, found_cycle, last_attempt_cycle, mechanism=None)` with `.with_status(...)`.
  - `types.Hypothesis(gap_id, cause, mechanism, evidence_for=(), evidence_against=(), verifier_votes=0, verdict="uncertain", cycle=0)`.
  - `ledger.Ledger(root, gaps=(), hypotheses=(), negatives=())`, `.load(root)`, `.save()`, `.add_gap`, `.add_hypothesis`, `.add_negative`, `.is_refuted(approach)->bool`; attrs `.gaps`, `.hypotheses`, `.negatives`.
  - `ladder.build_run_callables(config_path, jv_kwargs) -> (run_point, base_point)`, `ladder.DEFAULT_JV_KWARGS`, `ladder._PKG_ROOT`.
  - `scorecard.SHEET_TO_AXIS`, `scorecard._voc_closure(sl_vocs, scaps_vocs)`.
  - `provenance.stamp(*, run_id, config_path, flags, seed, timestamp) -> Provenance`.
- **Limiting probe scope:** Stage 2 ships the **dark J=0** limiting probe only. The rad-only detailed-balance-ceiling probe is **deferred** (needs `radiative_limit` config plumbing) — noted, not silently dropped.
- **Run all commands from `perovskite-sim/`.** Tests default to `-m 'not slow'`.

## File Structure

```
perovskite_sim/autoloop/
  types.py          + AblationProbe, AblationMatrix, Gap.with_mechanism
  ablation.py       ProbeRunner protocol + CANDIDATE_FLAGS + bucket_for_gap + run_ablation
  _probe_worker.py  CLI: set env -> compute badness -> print {"metric": ...}
  subprocess_probe.py  SubprocessProbeRunner (real ProbeRunner)
  attribution.py    Attributor protocol + HeuristicAttributor + thresholds
  orchestrator.py   + attribute_top_gap(...)
scripts/autoloop_run.py  + --attribute
tests/unit/autoloop/
  test_types_ablation.py
  test_ablation.py
  test_attribution.py
  test_orchestrator_attribution.py
tests/integration/
  test_autoloop_attribution.py   (slow, real subprocess)
```

---

## Task 1: Ablation types + `Gap.with_mechanism`

**Files:**
- Modify: `perovskite_sim/autoloop/types.py`
- Test: `tests/unit/autoloop/test_types_ablation.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/autoloop/test_types_ablation.py
import dataclasses
import pytest
from perovskite_sim.autoloop.types import Gap, AblationProbe, AblationMatrix


def _gap():
    return Gap(id="g", metric="V_oc", sweep="Nd_ETL", sweep_point=0.0,
               solarlab_val=30.0, reference_val=70.0, gap_mag=0.4, kind="trend",
               status="open", found_cycle=0, last_attempt_cycle=0, mechanism=None)


def test_gap_with_mechanism_returns_new_instance():
    g = _gap()
    g2 = g.with_mechanism("flag SOLARLAB_IFACE_PROJ term")
    assert g.mechanism is None                     # original unchanged
    assert g2.mechanism == "flag SOLARLAB_IFACE_PROJ term"
    assert g2.id == g.id


def test_ablation_probe_is_frozen():
    p = AblationProbe(name="SOLARLAB_IFACE_PROJ", kind="flag", variant={"flag": "X"},
                      baseline_val=40.0, variant_val=25.0, delta=-15.0, ok=True)
    assert p.delta == -15.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.delta = 0.0  # type: ignore[misc]


def test_ablation_matrix_holds_probes():
    p = AblationProbe(name="grid_n80", kind="grid", variant={"n_points": 80},
                      baseline_val=40.0, variant_val=41.0, delta=1.0, ok=True)
    m = AblationMatrix(gap_id="g", baseline_val=40.0, probes=(p,))
    assert m.probes[0].kind == "grid"
    assert m.skipped == ()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_types_ablation.py`
Expected: FAIL — `ImportError: cannot import name 'AblationProbe'`.

- [ ] **Step 3: Add to `types.py`**

Add `with_mechanism` to `Gap` (right after `with_status`):

```python
    def with_mechanism(self, mechanism: str) -> "Gap":
        return dataclasses.replace(self, mechanism=mechanism)
```

Append the two new dataclasses to `types.py` (after `Provenance`):

```python
@dataclass(frozen=True)
class AblationProbe:
    """One ablation variant's effect on the gap's badness scalar."""
    name: str
    kind: str            # "flag" | "grid" | "limiting"
    variant: dict        # the variant applied (env_flags / jv_overrides summary)
    baseline_val: float
    variant_val: float
    delta: float         # variant_val - baseline_val (negative = closer to reference)
    ok: bool
    note: str = ""


@dataclass(frozen=True)
class AblationMatrix:
    gap_id: str
    baseline_val: float
    probes: tuple[AblationProbe, ...]
    skipped: tuple[str, ...] = ()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_types_ablation.py`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add perovskite-sim/perovskite_sim/autoloop/types.py perovskite-sim/tests/unit/autoloop/test_types_ablation.py
git commit -m "feat(autoloop): add ablation types + Gap.with_mechanism (Stage 2)"
```

---

## Task 2: Ablation harness

**Files:**
- Create: `perovskite_sim/autoloop/ablation.py`
- Test: `tests/unit/autoloop/test_ablation.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/autoloop/test_ablation.py
from perovskite_sim.autoloop.types import Gap
from perovskite_sim.autoloop.ablation import (
    CANDIDATE_FLAGS, bucket_for_gap, run_ablation,
)


def _gap(sweep="Nd_ETL", kind="trend"):
    return Gap(id=f"trend:{sweep}:V_oc", metric="V_oc", sweep=sweep, sweep_point=0.0,
               solarlab_val=30.0, reference_val=70.0, gap_mag=0.4, kind=kind,
               status="open", found_cycle=0, last_attempt_cycle=0, mechanism=None)


class _FakeRunner:
    """Returns canned badness keyed by a stable variant signature."""
    def __init__(self, table):
        self.table = table
        self.calls = []

    def run(self, variant):
        self.calls.append(variant)
        key = _key(variant)
        if isinstance(self.table.get(key), Exception):
            raise self.table[key]
        return self.table[key]


def _key(variant):
    flags = ",".join(sorted(variant.get("env_flags", {})))
    jv = ",".join(f"{k}={v}" for k, v in sorted(variant.get("jv_overrides", {}).items()))
    return f"{variant.get('measure')}|{flags}|{jv}"


def test_bucket_for_gap():
    assert bucket_for_gap(_gap("Nd_ETL")) == "interface"
    assert bucket_for_gap(_gap("CHI_ETL")) == "interface"
    base = _gap("base", "absolute")
    assert bucket_for_gap(base) == "base"


def test_run_ablation_builds_matrix_with_flag_grid_dark_probes():
    g = _gap("Nd_ETL")
    table = {
        "gap||": 40.0,                                   # baseline
        "gap|SOLARLAB_IFACE_PROJ|": 22.0,                # physics lever (improves)
        "gap|SOLARLAB_IFACE_PLANE|": 41.0,
        "gap|SOLARLAB_INTERFACE_PLANE_STATE|": 39.0,
        "gap||n_points=80": 40.5,                        # grid stable
        "dark||illuminated=False": 0.01,                 # dark ~0
    }
    m = run_ablation(g, _FakeRunner(table))
    kinds = {p.kind for p in m.probes}
    assert kinds == {"flag", "grid", "limiting"}
    assert m.baseline_val == 40.0
    proj = next(p for p in m.probes if p.name == "SOLARLAB_IFACE_PROJ")
    assert proj.delta == -18.0                           # 22 - 40


def test_run_ablation_marks_failed_probe_not_ok():
    g = _gap("Nd_ETL")
    table = {
        "gap||": 40.0,
        "gap|SOLARLAB_IFACE_PROJ|": RuntimeError("solver diverged"),
        "gap|SOLARLAB_IFACE_PLANE|": 41.0,
        "gap|SOLARLAB_INTERFACE_PLANE_STATE|": 39.0,
        "gap||n_points=80": 40.5,
        "dark||illuminated=False": 0.01,
    }
    m = run_ablation(g, _FakeRunner(table))
    proj = next(p for p in m.probes if p.name == "SOLARLAB_IFACE_PROJ")
    assert proj.ok is False
    assert "solver diverged" in proj.note
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_ablation.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'perovskite_sim.autoloop.ablation'`.

- [ ] **Step 3: Write `ablation.py`**

```python
# perovskite_sim/autoloop/ablation.py
from __future__ import annotations

import math
from typing import Protocol

from perovskite_sim.autoloop.types import AblationMatrix, AblationProbe, Gap

# Candidate physics flags per gap bucket. Data, not logic — extend as the
# campaign learns which flags lever which gaps. Logged, never silently capped.
CANDIDATE_FLAGS: dict[str, list[str]] = {
    "interface": ["SOLARLAB_IFACE_PROJ", "SOLARLAB_IFACE_PLANE", "SOLARLAB_INTERFACE_PLANE_STATE"],
    "base":      ["SOLARLAB_DOS_BAND"],
}

_INTERFACE_SWEEPS = {"Nt_PVK ETL", "CHI_ETL", "Nd_ETL"}

GRID_N_POINTS = 80   # the high-resolution grid for the numerics probe


class ProbeRunner(Protocol):
    def run(self, variant: dict) -> float: ...


def bucket_for_gap(gap: Gap) -> str:
    """Map a gap to a candidate-flag bucket."""
    if gap.sweep in _INTERFACE_SWEEPS:
        return "interface"
    return "base"


def _safe_run(runner: ProbeRunner, variant: dict) -> tuple[float, bool, str]:
    try:
        return float(runner.run(variant)), True, ""
    except Exception as exc:           # noqa: BLE001 — recorded, not swallowed
        return math.nan, False, f"{type(exc).__name__}: {exc}"


def run_ablation(gap: Gap, probe_runner: ProbeRunner) -> AblationMatrix:
    """Run flag + grid + dark-limiting probes; record badness deltas.

    Badness is lower = closer to the SCAPS reference, so a flag probe with a
    negative delta improves the gap. A failing probe is recorded ok=False
    (not raised) so attribution can proceed on the surviving signal.
    """
    bucket = bucket_for_gap(gap)
    base_val, base_ok, base_note = _safe_run(
        probe_runner, {"env_flags": {}, "jv_overrides": {}, "measure": "gap"})

    probes: list[AblationProbe] = []
    skipped: list[str] = []

    for flag in CANDIDATE_FLAGS.get(bucket, []):
        val, ok, note = _safe_run(
            probe_runner, {"env_flags": {flag: "1"}, "jv_overrides": {}, "measure": "gap"})
        probes.append(AblationProbe(
            name=flag, kind="flag", variant={"flag": flag},
            baseline_val=base_val, variant_val=val,
            delta=(val - base_val), ok=(ok and base_ok), note=note))
    if bucket not in CANDIDATE_FLAGS:
        skipped.append(f"no candidate flags for bucket '{bucket}'")

    gval, gok, gnote = _safe_run(
        probe_runner, {"env_flags": {}, "jv_overrides": {"n_points": GRID_N_POINTS}, "measure": "gap"})
    probes.append(AblationProbe(
        name=f"grid_n{GRID_N_POINTS}", kind="grid", variant={"n_points": GRID_N_POINTS},
        baseline_val=base_val, variant_val=gval,
        delta=(gval - base_val), ok=(gok and base_ok), note=gnote))

    dval, dok, dnote = _safe_run(
        probe_runner, {"env_flags": {}, "jv_overrides": {"illuminated": False}, "measure": "dark"})
    probes.append(AblationProbe(
        name="dark_jsc", kind="limiting", variant={"illuminated": False},
        baseline_val=0.0, variant_val=dval, delta=dval, ok=dok,
        note=(dnote or "dark J_sc; expect ~0")))

    return AblationMatrix(gap_id=gap.id, baseline_val=base_val,
                          probes=tuple(probes), skipped=tuple(skipped))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_ablation.py`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add perovskite-sim/perovskite_sim/autoloop/ablation.py perovskite-sim/tests/unit/autoloop/test_ablation.py
git commit -m "feat(autoloop): add ablation harness (flag/grid/dark probes, injected runner)"
```

---

## Task 3: Subprocess probe worker + runner

**Files:**
- Create: `perovskite_sim/autoloop/_probe_worker.py`
- Create: `perovskite_sim/autoloop/subprocess_probe.py`
- Test: `tests/unit/autoloop/test_subprocess_probe.py`

The worker is the real badness computer; the runner shells out to it with env flags set (so module-level flag reads + the `MaterialArrays` cache see them). Unit-tested at the wiring level (command construction + JSON parse); end-to-end correctness is the Task 6 slow smoke.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/autoloop/test_subprocess_probe.py
import json
from perovskite_sim.autoloop.types import Gap
from perovskite_sim.autoloop.subprocess_probe import SubprocessProbeRunner


def _gap():
    return Gap(id="trend:Nd_ETL:V_oc", metric="V_oc", sweep="Nd_ETL", sweep_point=0.0,
               solarlab_val=30.0, reference_val=70.0, gap_mag=0.4, kind="trend",
               status="open", found_cycle=0, last_attempt_cycle=0, mechanism=None)


def test_runner_parses_metric_from_worker_stdout(monkeypatch, tmp_path):
    captured = {}

    class _CP:
        returncode = 0
        stdout = json.dumps({"metric": 12.5})
        stderr = ""

    def _fake_run(cmd, **kw):
        captured["cmd"] = cmd
        captured["env_has_flag"] = kw.get("env", {}).get("SOLARLAB_IFACE_PROJ")
        captured["cwd"] = kw.get("cwd")
        return _CP()

    monkeypatch.setattr("perovskite_sim.autoloop.subprocess_probe.subprocess.run", _fake_run)
    runner = SubprocessProbeRunner(config_path=tmp_path / "c.yaml",
                                   reference_path=tmp_path / "r.json", gap=_gap())
    val = runner.run({"env_flags": {"SOLARLAB_IFACE_PROJ": "1"},
                      "jv_overrides": {}, "measure": "gap"})
    assert val == 12.5
    assert captured["env_has_flag"] == "1"                 # flag injected into env
    assert "_probe_worker" in " ".join(captured["cmd"])
    from perovskite_sim.autoloop.ladder import _PKG_ROOT
    assert captured["cwd"] == str(_PKG_ROOT)               # cwd = package root


def test_runner_raises_on_worker_failure(monkeypatch, tmp_path):
    class _CP:
        returncode = 1
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr("perovskite_sim.autoloop.subprocess_probe.subprocess.run",
                        lambda cmd, **kw: _CP())
    runner = SubprocessProbeRunner(config_path=tmp_path / "c.yaml",
                                   reference_path=tmp_path / "r.json", gap=_gap())
    import pytest
    with pytest.raises(RuntimeError):
        runner.run({"env_flags": {}, "jv_overrides": {}, "measure": "gap"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_subprocess_probe.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'perovskite_sim.autoloop.subprocess_probe'`.

- [ ] **Step 3: Write the runner and worker**

```python
# perovskite_sim/autoloop/subprocess_probe.py
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from perovskite_sim.autoloop.ladder import _PKG_ROOT
from perovskite_sim.autoloop.types import Gap


@dataclass
class SubprocessProbeRunner:
    """Real ProbeRunner: runs one variant in a fresh interpreter with env flags
    set, so module-level SOLARLAB_* reads + the MaterialArrays cache pick them
    up. Returns the badness scalar printed by _probe_worker."""
    config_path: Path
    reference_path: Path
    gap: Gap

    def run(self, variant: dict) -> float:
        env = dict(os.environ)
        env.update(variant.get("env_flags", {}))
        payload = {
            "config": str(self.config_path),
            "reference": str(self.reference_path),
            "gap_sweep": self.gap.sweep,
            "gap_metric": self.gap.metric,
            "gap_kind": self.gap.kind,
            "jv_overrides": variant.get("jv_overrides", {}),
            "measure": variant.get("measure", "gap"),
        }
        proc = subprocess.run(
            ["python", "-m", "perovskite_sim.autoloop._probe_worker", json.dumps(payload)],
            capture_output=True, text=True, env=env, cwd=str(_PKG_ROOT),
        )
        if proc.returncode != 0:
            raise RuntimeError(f"probe worker failed (rc={proc.returncode}): "
                               f"{proc.stderr.strip()[-400:]}")
        return float(json.loads(proc.stdout)["metric"])
```

```python
# perovskite_sim/autoloop/_probe_worker.py
"""Probe worker (Stage 2). Runs ONE ablation variant and prints its badness.

Invoked as: python -m perovskite_sim.autoloop._probe_worker '<json payload>'
Badness (lower = closer to the SCAPS reference):
  measure="gap", trend gap     -> 100 - V_oc closure% for the sweep sheet
  measure="gap", absolute gap  -> |base_metric - reference_base_metric|
  measure="dark"               -> |dark J_sc|
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from perovskite_sim.autoloop.ladder import DEFAULT_JV_KWARGS, build_run_callables
from perovskite_sim.autoloop.scorecard import SHEET_TO_AXIS, _voc_closure

_METRIC_KEY = {"V_oc": "Voc_V", "J_sc": "Jsc_mA_cm2", "FF": "FF_percent", "PCE": "PCE_percent"}


def _badness(payload: dict) -> float:
    config = Path(payload["config"])
    jv = {**DEFAULT_JV_KWARGS, **payload.get("jv_overrides", {})}
    run_point, base_point = build_run_callables(config, jv_kwargs=jv)

    if payload["measure"] == "dark":
        # base point under no illumination -> J_sc should be ~0
        _voc, jsc, _ff, _pce, _brk = base_point()
        return abs(jsc)

    ref = json.loads(Path(payload["reference"]).read_text(encoding="utf-8"))

    if payload["gap_kind"] == "trend":
        sheet = payload["gap_sweep"]
        axis = SHEET_TO_AXIS[sheet]
        sl, scaps = [], []
        for pt in ref["sweeps"][sheet]["points"]:
            x = float(pt["x"])
            voc, _j, _f, _p, brk = run_point(axis, x)
            if brk and voc == voc:
                sl.append(voc)
                scaps.append(float(pt["Voc_V"]))
        closure = _voc_closure(sl, scaps)
        return 100.0 if closure != closure else max(0.0, 100.0 - closure)

    # absolute base gap
    voc, jsc_A, ff, pce, _brk = base_point()
    sl_map = {"V_oc": voc, "J_sc": jsc_A, "FF": ff, "PCE": pce}
    bm = ref["base_model"]
    ref_v = float(bm[_METRIC_KEY[payload["gap_metric"]]])
    if payload["gap_metric"] == "J_sc":
        ref_v *= 10.0
    elif payload["gap_metric"] in ("FF", "PCE"):
        ref_v /= 100.0
    return abs(sl_map[payload["gap_metric"]] - ref_v)


def main(argv: list[str]) -> int:
    payload = json.loads(argv[0])
    print(json.dumps({"metric": _badness(payload)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_subprocess_probe.py`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add perovskite-sim/perovskite_sim/autoloop/_probe_worker.py perovskite-sim/perovskite_sim/autoloop/subprocess_probe.py perovskite-sim/tests/unit/autoloop/test_subprocess_probe.py
git commit -m "feat(autoloop): add subprocess probe worker + runner (real badness via env-set sweep)"
```

---

## Task 4: Heuristic attributor

**Files:**
- Create: `perovskite_sim/autoloop/attribution.py`
- Test: `tests/unit/autoloop/test_attribution.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/autoloop/test_attribution.py
from perovskite_sim.autoloop.types import Gap, AblationProbe, AblationMatrix
from perovskite_sim.autoloop.attribution import HeuristicAttributor
from perovskite_sim.autoloop.ledger import Ledger
from perovskite_sim.autoloop.types import NegativeResult


def _gap():
    return Gap(id="trend:Nd_ETL:V_oc", metric="V_oc", sweep="Nd_ETL", sweep_point=0.0,
               solarlab_val=30.0, reference_val=70.0, gap_mag=0.4, kind="trend",
               status="open", found_cycle=0, last_attempt_cycle=0, mechanism=None)


def _matrix(*, flag_delta=0.0, grid_delta=0.0, dark_val=0.0):
    probes = (
        AblationProbe("SOLARLAB_IFACE_PROJ", "flag", {"flag": "SOLARLAB_IFACE_PROJ"},
                      40.0, 40.0 + flag_delta, flag_delta, True),
        AblationProbe("grid_n80", "grid", {"n_points": 80},
                      40.0, 40.0 + grid_delta, grid_delta, True),
        AblationProbe("dark_jsc", "limiting", {"illuminated": False},
                      0.0, dark_val, dark_val, True),
    )
    return AblationMatrix(gap_id="trend:Nd_ETL:V_oc", baseline_val=40.0, probes=probes)


def _empty_ledger(tmp_path):
    return Ledger(root=tmp_path)


def test_numerics_when_grid_sensitive(tmp_path):
    h = HeuristicAttributor()
    hyp = h.attribute(_gap(), _matrix(grid_delta=-25.0), _empty_ledger(tmp_path))
    assert hyp.cause == "numerics"
    assert hyp.verdict == "confirmed"


def test_physics_when_flag_improves(tmp_path):
    h = HeuristicAttributor()
    hyp = h.attribute(_gap(), _matrix(flag_delta=-18.0), _empty_ledger(tmp_path))
    assert hyp.cause == "physics"
    assert "SOLARLAB_IFACE_PROJ" in hyp.mechanism


def test_bug_when_dark_current_nonzero(tmp_path):
    h = HeuristicAttributor()
    hyp = h.attribute(_gap(), _matrix(dark_val=15.0), _empty_ledger(tmp_path))
    assert hyp.cause == "bug"


def test_uncertain_when_no_dominant_lever(tmp_path):
    h = HeuristicAttributor()
    hyp = h.attribute(_gap(), _matrix(flag_delta=-0.2, grid_delta=0.3, dark_val=0.01),
                      _empty_ledger(tmp_path))
    assert hyp.cause == "uncertain"
    assert hyp.verdict == "uncertain"


def test_negatives_guard_downgrades_refuted_physics(tmp_path):
    led = Ledger(root=tmp_path)
    led.add_negative(NegativeResult(approach="flag SOLARLAB_IFACE_PROJ term",
                                    why_failed="x", evidence="y"))
    h = HeuristicAttributor()
    hyp = h.attribute(_gap(), _matrix(flag_delta=-18.0), led)
    assert hyp.verdict == "uncertain"             # refuted mechanism never confirmed
    assert "refuted" in " ".join(hyp.evidence_against).lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_attribution.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'perovskite_sim.autoloop.attribution'`.

- [ ] **Step 3: Write `attribution.py`**

```python
# perovskite_sim/autoloop/attribution.py
from __future__ import annotations

from typing import Protocol

from perovskite_sim.autoloop.ledger import Ledger
from perovskite_sim.autoloop.types import AblationMatrix, Gap, Hypothesis

# Thresholds (badness units; tunable). A probe must move badness by more than
# the tolerance to count as a signal.
GRID_TOL = 5.0     # grid-convergence shift this large -> numerics artifact
FLAG_TOL = 5.0     # a flag improving badness by this much -> physics lever
DARK_TOL = 1.0     # dark J_sc magnitude (A/m^2) above this -> bug
DOMINANCE = 2.0    # winning |signal| must beat the runner-up by this factor


class Attributor(Protocol):
    def attribute(self, gap: Gap, matrix: AblationMatrix, negatives: Ledger) -> Hypothesis: ...


def _ok(probes, kind):
    return [p for p in probes if p.kind == kind and p.ok and p.delta == p.delta]


class HeuristicAttributor:
    """Rule-based attributor over an AblationMatrix. Honest: falls back to
    'uncertain' with no dominant lever, and never confirms a mechanism that
    matches a refuted approach in the negatives ledger."""

    def attribute(self, gap: Gap, matrix: AblationMatrix, negatives: Ledger) -> Hypothesis:
        grids = _ok(matrix.probes, "grid")
        flags = _ok(matrix.probes, "flag")
        darks = _ok(matrix.probes, "limiting")

        grid_sig = max((abs(p.delta) for p in grids), default=0.0)
        best_flag = min(flags, key=lambda p: p.delta, default=None)   # most negative = best improvement
        flag_sig = -best_flag.delta if best_flag is not None and best_flag.delta < 0 else 0.0
        dark_sig = max((p.variant_val for p in darks), default=0.0)

        # 1. numerics
        if grid_sig > GRID_TOL and grid_sig >= DOMINANCE * flag_sig:
            return Hypothesis(
                gap_id=gap.id, cause="numerics",
                mechanism=f"grid-convergence sensitive (n_points->80 shifts badness by {grid_sig:.3g})",
                evidence_for=(f"grid delta {grid_sig:.3g} > tol {GRID_TOL}",),
                verifier_votes=1, verdict="confirmed")

        # 2. physics
        if best_flag is not None and flag_sig > FLAG_TOL:
            mechanism = f"flag {best_flag.name} term"
            if negatives.is_refuted(mechanism):
                return Hypothesis(
                    gap_id=gap.id, cause="physics", mechanism=mechanism,
                    evidence_for=(f"{best_flag.name} improves badness by {flag_sig:.3g}",),
                    evidence_against=("matches a REFUTED approach in the negatives ledger",),
                    verifier_votes=0, verdict="uncertain")
            return Hypothesis(
                gap_id=gap.id, cause="physics", mechanism=mechanism,
                evidence_for=(f"{best_flag.name} improves badness by {flag_sig:.3g} (> tol {FLAG_TOL})",),
                verifier_votes=1, verdict="confirmed")

        # 3. bug
        if dark_sig > DARK_TOL:
            return Hypothesis(
                gap_id=gap.id, cause="bug",
                mechanism=f"limiting-case violation: dark J_sc = {dark_sig:.3g} (expect ~0)",
                evidence_for=(f"dark J_sc {dark_sig:.3g} > tol {DARK_TOL}",),
                verifier_votes=1, verdict="confirmed")

        # 4. uncertain (honest fallback)
        return Hypothesis(
            gap_id=gap.id, cause="uncertain",
            mechanism="no single ablation lever identified",
            evidence_against=(f"grid {grid_sig:.3g}, best-flag {flag_sig:.3g}, dark {dark_sig:.3g} "
                              f"all below dominance",),
            verifier_votes=0, verdict="uncertain")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_attribution.py`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add perovskite-sim/perovskite_sim/autoloop/attribution.py perovskite-sim/tests/unit/autoloop/test_attribution.py
git commit -m "feat(autoloop): add HeuristicAttributor (numerics/physics/bug/uncertain + negatives-guard)"
```

---

## Task 5: Orchestrator attribution pass

**Files:**
- Modify: `perovskite_sim/autoloop/orchestrator.py`
- Test: `tests/unit/autoloop/test_orchestrator_attribution.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/autoloop/test_orchestrator_attribution.py
from perovskite_sim.autoloop.types import Gap, AblationMatrix, AblationProbe, Hypothesis
from perovskite_sim.autoloop.ledger import Ledger
from perovskite_sim.autoloop.orchestrator import attribute_top_gap


def _gap(gid, mag, status="open"):
    return Gap(id=gid, metric="V_oc", sweep="Nd_ETL", sweep_point=0.0,
               solarlab_val=30.0, reference_val=70.0, gap_mag=mag, kind="trend",
               status=status, found_cycle=0, last_attempt_cycle=0, mechanism=None)


class _FakeAttributor:
    def attribute(self, gap, matrix, negatives):
        return Hypothesis(gap_id=gap.id, cause="physics", mechanism="flag X term",
                          verdict="confirmed", verifier_votes=1)


def _fake_ablation(gap, probe_runner):
    return AblationMatrix(gap_id=gap.id, baseline_val=40.0,
                          probes=(AblationProbe("f", "flag", {}, 40.0, 22.0, -18.0, True),))


def test_attribute_top_gap_picks_highest_open_and_records(tmp_path):
    led = Ledger(root=tmp_path / "ledger")
    led.add_gap(_gap("trend:Nd_ETL:V_oc", 0.4))
    led.add_gap(_gap("absolute:base:V_oc", 0.1))
    led.save()

    hyp = attribute_top_gap(
        ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
        config_path=tmp_path / "c.yaml", reference_path=tmp_path / "r.json",
        cycle=1, timestamp="2026-06-16T00:00:00Z",
        probe_runner=object(), attributor=_FakeAttributor(),
        run_ablation_fn=_fake_ablation,
    )
    assert hyp is not None and hyp.gap_id == "trend:Nd_ETL:V_oc"   # highest gap_mag

    led2 = Ledger.load(tmp_path / "ledger")
    assert any(h.gap_id == "trend:Nd_ETL:V_oc" for h in led2.hypotheses)
    g = next(g for g in led2.gaps if g.id == "trend:Nd_ETL:V_oc")
    assert g.mechanism == "flag X term"                            # confirmed -> mechanism written
    assert (tmp_path / "out" / "attr-1" / "hypothesis.json").exists()


def test_attribute_top_gap_skips_non_open_and_returns_none(tmp_path):
    led = Ledger(root=tmp_path / "ledger")
    led.add_gap(_gap("g1", 0.9, status="blocked"))
    led.add_gap(_gap("g2", 0.8, status="refuted"))
    led.save()

    hyp = attribute_top_gap(
        ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
        config_path=tmp_path / "c.yaml", reference_path=tmp_path / "r.json",
        cycle=1, timestamp="2026-06-16T00:00:00Z",
        probe_runner=object(), attributor=_FakeAttributor(),
        run_ablation_fn=_fake_ablation,
    )
    assert hyp is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_orchestrator_attribution.py`
Expected: FAIL — `ImportError: cannot import name 'attribute_top_gap'`.

- [ ] **Step 3: Add `attribute_top_gap` to `orchestrator.py`**

Add imports at the top (alongside existing imports):

```python
from perovskite_sim.autoloop.ablation import run_ablation as _run_ablation
from perovskite_sim.autoloop.types import Hypothesis
```

Append the function:

```python
def attribute_top_gap(*, ledger_root: Path, outputs_root: Path,
                      config_path: Path, reference_path: Path,
                      cycle: int, timestamp: str,
                      probe_runner, attributor,
                      flags: Optional[dict[str, str]] = None, seed: int = 0,
                      run_ablation_fn=None) -> Optional[Hypothesis]:
    """One attribution pass: pick the top open gap, ablate, attribute, record.

    Read-only re: code — writes only the ledger + run artifacts.
    """
    ledger_root = Path(ledger_root)
    led = Ledger.load(ledger_root)

    open_gaps = [g for g in led.gaps if g.status == "open"]
    if not open_gaps:
        return None
    gap = max(open_gaps, key=lambda g: g.gap_mag)

    run_ablation = run_ablation_fn or _run_ablation
    matrix = run_ablation(gap, probe_runner)
    hyp = attributor.attribute(gap, matrix, led)

    led.add_hypothesis(hyp)
    if hyp.verdict == "confirmed":
        led.add_gap(gap.with_mechanism(hyp.mechanism))   # add_gap replaces on id
    led.save()

    run_dir = Path(outputs_root) / f"attr-{cycle}"
    run_dir.mkdir(parents=True, exist_ok=True)
    prov = stamp(run_id=f"attr-{cycle}", config_path=config_path,
                 flags=flags or {}, seed=seed, timestamp=timestamp)
    (run_dir / "hypothesis.json").write_text(
        json.dumps(dataclasses.asdict(hyp), indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "matrix.json").write_text(
        json.dumps(dataclasses.asdict(matrix), indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "provenance.json").write_text(
        json.dumps(dataclasses.asdict(prov), indent=2, sort_keys=True), encoding="utf-8")
    return hyp
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_orchestrator_attribution.py`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add perovskite-sim/perovskite_sim/autoloop/orchestrator.py perovskite-sim/tests/unit/autoloop/test_orchestrator_attribution.py
git commit -m "feat(autoloop): add attribute_top_gap orchestrator pass (read-only, records Hypothesis)"
```

---

## Task 6: CLI `--attribute` + integration smoke + docs

**Files:**
- Modify: `scripts/autoloop_run.py`
- Create: `tests/integration/test_autoloop_attribution.py`
- Modify: `perovskite-sim/CLAUDE.md` (Autoloop section), `README.md`

- [ ] **Step 1: Write the failing integration test (real subprocess, slow)**

```python
# tests/integration/test_autoloop_attribution.py
import pytest
from pathlib import Path
from perovskite_sim.autoloop.orchestrator import guardian_once, attribute_top_gap
from perovskite_sim.autoloop.attribution import HeuristicAttributor
from perovskite_sim.autoloop.subprocess_probe import SubprocessProbeRunner
from perovskite_sim.autoloop.ledger import Ledger

REPO_ROOT = Path(__file__).resolve().parents[1]
REF = REPO_ROOT / "tests" / "integration" / "scaps_reference.json"
CFG = REPO_ROOT / "configs" / "scaps_mirror_v2.yaml"


@pytest.mark.slow
def test_attribution_pass_produces_hypothesis(tmp_path):
    # First a guardian cycle to populate the gap ledger.
    guardian_once(ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
                  reference_path=REF, config_path=CFG, cycle=0,
                  timestamp="2026-06-16T00:00:00Z", l0_paths=["tests/unit/autoloop"],
                  baseline=None)
    led = Ledger.load(tmp_path / "ledger")
    open_gaps = [g for g in led.gaps if g.status == "open"]
    if not open_gaps:
        pytest.skip("no open gaps produced on this config — nothing to attribute")

    top = max(open_gaps, key=lambda g: g.gap_mag)
    runner = SubprocessProbeRunner(config_path=CFG, reference_path=REF, gap=top)
    hyp = attribute_top_gap(
        ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
        config_path=CFG, reference_path=REF, cycle=1,
        timestamp="2026-06-16T00:00:00Z",
        probe_runner=runner, attributor=HeuristicAttributor())
    assert hyp is not None
    assert hyp.cause in {"bug", "numerics", "physics", "data", "uncertain"}
    assert (tmp_path / "out" / "attr-1" / "hypothesis.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd perovskite-sim && python -m pytest -q -m slow tests/integration/test_autoloop_attribution.py`
Expected: FAIL — `ImportError` until CLI wiring lands, or a runtime error from the worker; once green it confirms the real subprocess path end-to-end (runtime: several real sweeps, order minutes).

- [ ] **Step 3: Wire the CLI + docs**

In `scripts/autoloop_run.py`, add the `--attribute` flag to `parse_args`:

```python
    ap.add_argument("--attribute", action="store_true",
                    help="run one attribution pass on the top open gap")
```

In `main`, dispatch on it (insert before the existing guardian path):

```python
    if ns.attribute:
        from perovskite_sim.autoloop.orchestrator import attribute_top_gap
        from perovskite_sim.autoloop.attribution import HeuristicAttributor
        from perovskite_sim.autoloop.subprocess_probe import SubprocessProbeRunner
        from perovskite_sim.autoloop.ledger import Ledger
        led = Ledger.load(ns.ledger_root)
        open_gaps = [g for g in led.gaps if g.status == "open"]
        if not open_gaps:
            print(json.dumps({"attributed": None, "reason": "no open gaps"}))
            return 0
        top = max(open_gaps, key=lambda g: g.gap_mag)
        runner = SubprocessProbeRunner(config_path=ns.config, reference_path=ns.reference, gap=top)
        hyp = attribute_top_gap(
            ledger_root=ns.ledger_root, outputs_root=ns.outputs_root,
            config_path=ns.config, reference_path=ns.reference, cycle=ns.cycle,
            timestamp=iso_timestamp_utc(), probe_runner=runner,
            attributor=HeuristicAttributor())
        import dataclasses
        print(json.dumps({"attributed": dataclasses.asdict(hyp)}, indent=2, sort_keys=True))
        return 0
```

Add to the **Autoloop** section of `perovskite-sim/CLAUDE.md`:

```markdown
### Stage 2 — attribution leg (deterministic)

`autoloop/ablation.py` + `autoloop/attribution.py` diagnose the top open gap
(read-only). `run_ablation` toggles candidate `SOLARLAB_*` flags + a grid
(n_points→80) + a dark-J probe through a `ProbeRunner` (real =
`SubprocessProbeRunner` → `_probe_worker`, env-set fresh interpreter; fake for
tests), recording a badness scalar (lower = closer to SCAPS). `HeuristicAttributor`
classifies: grid-sensitive→numerics, flag-improves→physics, dark-J≠0→bug, else
uncertain — with a negatives-guard (never confirms a refuted mechanism). Writes a
`Hypothesis` to the ledger; the LLM adapter + multi-skeptic verify are deferred.

    cd perovskite-sim
    python scripts/autoloop_run.py --once        # populate the gap ledger
    python scripts/autoloop_run.py --attribute   # diagnose the top open gap
```

Add to `README.md` (next to the Stage 1 autoloop line):

```markdown
- **Autoloop attribution** (`python perovskite-sim/scripts/autoloop_run.py --attribute`) —
  diagnoses the top open gap (bug / numerics / physics / uncertain) by ablating
  physics flags + grid + dark-current probes; records a Hypothesis. Read-only.
```

- [ ] **Step 4: Run tests**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop && python -m pytest -q -m slow tests/integration/test_autoloop_attribution.py`
Expected: all green. Also `python -m pytest -q` (full default suite) — confirm no import/collection regression from the new modules.

- [ ] **Step 5: Commit**

```bash
git add perovskite-sim/scripts/autoloop_run.py perovskite-sim/tests/integration/test_autoloop_attribution.py perovskite-sim/CLAUDE.md README.md
git commit -m "feat(autoloop): wire --attribute CLI + integration smoke + docs (Stage 2)"
```

---

## Self-Review

**Spec coverage** (vs `2026-06-16-autoloop-stage2-attribution-design.md`):
- §2 two injected seams (ProbeRunner, Attributor) → Tasks 2/3/4. ✓
- §3 module layout → Tasks 1–6 create exactly those files. ✓
- §4 ablation harness + CANDIDATE_FLAGS + badness + failed-probe handling → Task 2. ✓
- §4 SubprocessProbeRunner + _probe_worker (env-set, cwd=_PKG_ROOT) → Task 3. ✓
- §5 heuristic branches + negatives-guard + dominance + uncertain fallback → Task 4. ✓
- §6 attribute_top_gap (top open gap, confirmed→mechanism, artifacts) + CLI → Tasks 5/6. ✓
- §7 error handling (failed probe ok=False, attributor never raises, no-open-gaps→None) → Tasks 2/4/5. ✓
- §8 testing (per-branch, fakes, slow smoke) → every task. ✓
- §9 deferred (LLM adapter, rad-only ceiling, residual-by-channel) → correctly NOT built; rad-only narrowing flagged in the design contract.

**Placeholder scan:** none — every code/test step is complete; the one narrowing (limiting = dark-only, rad-only deferred) is stated explicitly, not a TODO.

**Type consistency:** `AblationProbe`/`AblationMatrix` fields defined in Task 1 are used identically in Tasks 2/4/5. `ProbeRunner.run(variant)` signature consistent across Tasks 2 (protocol), 3 (real), tests (fake). `attribute(gap, matrix, negatives)` consistent Tasks 4/5. `attribute_top_gap(... run_ablation_fn=)` injection matches the Task 5 test. `Hypothesis` fields match the Stage 1 `types.py` definition. `_PKG_ROOT`/`build_run_callables`/`SHEET_TO_AXIS`/`_voc_closure`/`DEFAULT_JV_KWARGS` are all real Stage 1 symbols (verified on `main`).

---

## Execution Handoff

After saving, choose execution:

1. **Subagent-Driven (recommended)** — fresh subagent per task, two-stage review (same workflow as Stage 1).
2. **Inline Execution** — batch tasks in this session with checkpoints.
