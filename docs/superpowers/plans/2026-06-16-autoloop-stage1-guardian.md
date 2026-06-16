# Autoloop Stage 1 — The Guardian — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the deterministic spine of the autoloop — ledgers, provenance, parity scorecard, the L0–L2 ladder runner, the G1–G3 gate stack, and a sense-and-record guardian orchestrator + CLI — with no agents, runnable in CI.

**Architecture:** A new `perovskite_sim/autoloop/` subpackage. Pure-Python, deterministic, importable. It wraps the existing simulation primitives (`run_jv_sweep`, `compute_metrics`, `load_scaps_yaml`, `apply_sweep_point`) and the SCAPS reference data (`tests/integration/scaps_reference.json`) to score parity, ranks discrepancies into a persistent JSON gap ledger, and exposes a gate stack that a CI run (and later, the cognition legs) call to detect regressions. Cognition, change-implementation, and the continuous boulder are explicitly out of scope (Stages 2–4).

**Tech Stack:** Python 3.9+, dataclasses, `json`, `hashlib`, `subprocess` (for the pytest gate), numpy. No new third-party dependency. pytest for tests.

---

## Design context (read before starting)

- **Repo root vs package.** `git rev-parse --show-toplevel` returns the **SolarLab root** (the `perovskite-sim/` tree is a subdirectory). `provenance.py` must resolve git from the repo root, not from `perovskite-sim/`.
- **All commands run from `perovskite-sim/`** unless stated. Tests run with `pytest` (defaults to `-m 'not slow'`, no coverage — see `pyproject.toml`).
- **Frozen dataclasses everywhere** (project invariant). Never mutate; produce new instances. Ledger updates are append + status-transition entries, never in-place edits.
- **Existing primitives (verified signatures):**
  - `perovskite_sim.experiments.jv_sweep.run_jv_sweep(stack, N_grid=100, v_rate=0.1, n_points=50, rtol=1e-4, atol=1e-6, V_max=None, illuminated=True, ...) -> JVResult` with `.metrics_fwd: JVMetrics`.
  - `JVMetrics(V_oc, J_sc, FF, PCE, voc_bracketed)` — frozen; `J_sc` in **A/m²**.
  - `compute_metrics(V, J, *, assume_jsc_positive=True) -> JVMetrics`.
  - `perovskite_sim.scaps_compat.load_scaps_yaml(path: str|Path) -> DeviceStack`.
  - `perovskite_sim.sweeps.device_parameter_sweep.SweepPoint(name, axis, label, updates: dict)` and `apply_sweep_point(base_stack, sp) -> DeviceStack`.
- **Reference data** `tests/integration/scaps_reference.json`:
  ```
  {
    "base_model": {"Voc_V", "Jsc_mA_cm2", "FF_percent", "PCE_percent"},
    "sweeps": {
      "CHI_ETL":   {"x_name", "x_unit", "n_points", "points": [{"x","Voc_V","Jsc_mA_cm2","FF_percent","PCE_percent"}, ...]},
      "Nd_ETL":    {...}, "Nt_PVK ETL": {...}, "Nt_C_PVK": {...},
      "Et_*": {...}   // unmapped in Stage 1
    }
  }
  ```
  SCAPS `Jsc_mA_cm2` → A/m² is `× 10`. SCAPS `FF_percent`/`PCE_percent` → fraction is `/ 100`.
- **Sweep→axis map** (verified, from `scripts/run_scaps_v2_regression.py`) — only these four sheets are scoreable in Stage 1:
  ```python
  SHEET_TO_AXIS = {
      "CHI_ETL":     "etl_delta_ec_eV",
      "Nd_ETL":      "etl_doping_cm3",
      "Nt_PVK ETL":  "interface_defect_N_t_cm2",
      "Nt_C_PVK":    "absorber_defect_density_cm3",
  }
  ```
- **Parity config:** `configs/scaps_mirror_v2.yaml`. **JV kwargs:** `dict(N_grid=30, n_points=20, v_rate=5.0, V_max=1.6)` (the campaign envelope).
- **Negative-results seeds** (from project memory — must be pre-loaded so the loop never re-tries them):
  - `"DOS-cap projection target → false convergence, high residual at bulk node 19"`
  - `"BBD face-density term → V=0.08 blow-up"`
  - `"1.40 V_bi fudge → +106 mV unexplained over derived V_bi, rejected"`
  - `"two-sided additive mirror interface channel → measured no-op"`
  - `"shared-occupancy on bulk-node densities → CBO trend collapse 80→22%"`

---

## File Structure

```
perovskite_sim/autoloop/
  __init__.py        # public exports
  types.py           # Gap, Hypothesis, NegativeResult, ParityScore, SweepScore, LadderResult, GateVerdict, Provenance
  ledger.py          # Ledger: load/save/add/dedup over the three ledger files
  seeds.py           # SEED_NEGATIVE_RESULTS list + seed_negative_results(ledger)
  provenance.py      # stamp() -> Provenance (git SHA, config hash, flags, seed, timestamp)
  scorecard.py       # score_parity(config, reference, jv_kwargs) -> ParityScore ; gaps_from_score(...)
  ladder.py          # run_l0 / run_l1 / run_l2 / run_ladder -> LadderResult
  gates.py           # gate_g1/g2/g3 + run_gate_stack(...) -> list[GateVerdict] ; g0/g4/g5 deferred stubs
  orchestrator.py    # guardian_once(...) : sense -> score -> rank -> record -> gate verdict
scripts/
  autoloop_run.py    # CLI entry: python scripts/autoloop_run.py --once
tests/unit/autoloop/
  __init__.py
  test_types.py
  test_ledger.py
  test_seeds.py
  test_provenance.py
  test_scorecard.py
  test_ladder.py
  test_gates.py
  test_orchestrator.py
tests/integration/
  test_autoloop_guardian.py   # end-to-end --once smoke (marked slow)
outputs/autoloop/<run-id>/    # provenance-stamped artifacts (gitignored except a baseline)
docs/autoloop/ledger/         # human-readable ledger mirror (created by ledger.save)
```

Each file has one responsibility; `types.py` is the shared vocabulary every other module imports, so define it first and keep names stable.

---

## Task 1: Package skeleton + shared types

**Files:**
- Create: `perovskite_sim/autoloop/__init__.py`
- Create: `perovskite_sim/autoloop/types.py`
- Create: `tests/unit/autoloop/__init__.py`
- Test: `tests/unit/autoloop/test_types.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/autoloop/test_types.py
import dataclasses
import pytest
from perovskite_sim.autoloop.types import (
    Gap, Hypothesis, NegativeResult, SweepScore, ParityScore,
    LadderResult, GateVerdict, Provenance,
)


def test_gap_is_frozen_and_has_dedup_key():
    g = Gap(
        id="cbo:voc:base", metric="V_oc", sweep="CHI_ETL", sweep_point=0.0,
        solarlab_val=1.10, reference_val=1.17, gap_mag=0.07, kind="absolute",
        status="open", found_cycle=0, last_attempt_cycle=0, mechanism=None,
    )
    assert g.status == "open"
    with pytest.raises(dataclasses.FrozenInstanceError):
        g.status = "closed"  # type: ignore[misc]


def test_gap_with_status_returns_new_instance():
    g = Gap(
        id="x", metric="V_oc", sweep="Nd_ETL", sweep_point=1e15,
        solarlab_val=1.0, reference_val=1.1, gap_mag=0.1, kind="trend",
        status="open", found_cycle=0, last_attempt_cycle=0, mechanism=None,
    )
    g2 = g.with_status("blocked")
    assert g.status == "open"          # original unchanged
    assert g2.status == "blocked"
    assert g2.id == g.id


def test_negative_result_dedup_key_is_normalised():
    a = NegativeResult(approach="DOS-Cap  Projection", why_failed="x", evidence="y", never_retry=True)
    b = NegativeResult(approach="dos-cap projection", why_failed="z", evidence="w", never_retry=True)
    assert a.dedup_key() == b.dedup_key()


def test_parity_score_overall_in_unit_range():
    s = ParityScore(
        overall=0.0, base_deltas={}, per_sweep={
            "CHI_ETL": SweepScore(sweep="CHI_ETL", voc_closure_pct=80.0, n_points=10, n_bracketed=8),
        },
    )
    assert 0.0 <= s.overall <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/autoloop/test_types.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'perovskite_sim.autoloop'`.

- [ ] **Step 3: Write the implementation**

```python
# perovskite_sim/autoloop/__init__.py
"""Autoloop — continuous autonomous research-loop orchestrator (Stage 1: guardian)."""
from perovskite_sim.autoloop.types import (
    Gap, Hypothesis, NegativeResult, SweepScore, ParityScore,
    LadderResult, GateVerdict, Provenance,
)

__all__ = [
    "Gap", "Hypothesis", "NegativeResult", "SweepScore", "ParityScore",
    "LadderResult", "GateVerdict", "Provenance",
]
```

```python
# perovskite_sim/autoloop/types.py
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Optional


def _norm(text: str) -> str:
    """Normalise free text for dedup: lowercase, collapse whitespace."""
    return " ".join(text.lower().split())


@dataclass(frozen=True)
class Gap:
    """One open discrepancy between SolarLab and the reference."""
    id: str
    metric: str            # "V_oc" | "J_sc" | "FF" | "PCE"
    sweep: str             # reference sweep key, or "base"
    sweep_point: float
    solarlab_val: float
    reference_val: float
    gap_mag: float         # ranking magnitude (>= 0)
    kind: str              # "trend" | "absolute"
    status: str            # "open" | "closed" | "refuted" | "blocked"
    found_cycle: int
    last_attempt_cycle: int
    mechanism: Optional[str] = None

    def with_status(self, status: str, *, last_attempt_cycle: Optional[int] = None) -> "Gap":
        return dataclasses.replace(
            self, status=status,
            last_attempt_cycle=self.last_attempt_cycle if last_attempt_cycle is None else last_attempt_cycle,
        )


@dataclass(frozen=True)
class Hypothesis:
    """One attribution attempt for a gap (populated by Stage 2; defined here for stability)."""
    gap_id: str
    cause: str             # "bug" | "numerics" | "physics" | "data"
    mechanism: str
    evidence_for: tuple[str, ...] = ()
    evidence_against: tuple[str, ...] = ()
    verifier_votes: int = 0
    verdict: str = "uncertain"   # "confirmed" | "refuted" | "uncertain"
    cycle: int = 0


@dataclass(frozen=True)
class NegativeResult:
    """A refuted/failed approach the loop must never re-try."""
    approach: str
    why_failed: str
    evidence: str
    never_retry: bool = True

    def dedup_key(self) -> str:
        return _norm(self.approach)


@dataclass(frozen=True)
class SweepScore:
    sweep: str
    voc_closure_pct: float   # 100 * sl_voc_range / scaps_voc_range over bracketed points
    n_points: int
    n_bracketed: int


@dataclass(frozen=True)
class ParityScore:
    overall: float                       # 0..1, mean clamped closure across scored sweeps
    base_deltas: dict[str, float]        # metric -> (solarlab - reference) at base point
    per_sweep: dict[str, SweepScore]


@dataclass(frozen=True)
class LadderResult:
    l0_pass: bool                # numerics / unit tests
    l1_pass: bool                # limiting cases
    score: Optional[ParityScore] # L2 parity (None if L0/L1 short-circuited)
    details: dict


@dataclass(frozen=True)
class GateVerdict:
    name: str
    passed: bool
    reason: str


@dataclass(frozen=True)
class Provenance:
    run_id: str
    git_sha: str
    git_dirty: bool
    config_hash: str
    flags: dict[str, str]
    seed: int
    timestamp: str           # ISO-8601, passed in (not generated, for reproducibility)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/autoloop/test_types.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add perovskite_sim/autoloop/__init__.py perovskite_sim/autoloop/types.py tests/unit/autoloop/__init__.py tests/unit/autoloop/test_types.py
git commit -m "feat(autoloop): add Stage 1 shared types (Gap/Hypothesis/NegativeResult/scores)"
```

---

## Task 2: Ledger — persistence + dedup

**Files:**
- Create: `perovskite_sim/autoloop/ledger.py`
- Test: `tests/unit/autoloop/test_ledger.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/autoloop/test_ledger.py
from perovskite_sim.autoloop.types import Gap, NegativeResult
from perovskite_sim.autoloop.ledger import Ledger


def _gap(gid="g1", status="open"):
    return Gap(id=gid, metric="V_oc", sweep="CHI_ETL", sweep_point=0.0,
               solarlab_val=1.1, reference_val=1.17, gap_mag=0.07, kind="absolute",
               status=status, found_cycle=0, last_attempt_cycle=0, mechanism=None)


def test_add_and_roundtrip(tmp_path):
    led = Ledger(root=tmp_path)
    led.add_gap(_gap())
    led.add_negative(NegativeResult(approach="bad idea", why_failed="x", evidence="y"))
    led.save()

    led2 = Ledger.load(tmp_path)
    assert [g.id for g in led2.gaps] == ["g1"]
    assert led2.is_refuted("Bad  Idea") is True   # normalised match


def test_add_gap_dedups_on_id(tmp_path):
    led = Ledger(root=tmp_path)
    led.add_gap(_gap(status="open"))
    led.add_gap(_gap(status="blocked"))    # same id -> replaces, does not append twice
    assert len(led.gaps) == 1
    assert led.gaps[0].status == "blocked"


def test_is_refuted_false_for_unknown(tmp_path):
    led = Ledger(root=tmp_path)
    assert led.is_refuted("never seen this") is False


def test_save_writes_three_json_files_and_markdown_mirror(tmp_path):
    led = Ledger(root=tmp_path)
    led.add_gap(_gap())
    led.save()
    assert (tmp_path / "gaps.json").exists()
    assert (tmp_path / "hypotheses.json").exists()
    assert (tmp_path / "negative_results.json").exists()
    assert (tmp_path / "LEDGER.md").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/autoloop/test_ledger.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'perovskite_sim.autoloop.ledger'`.

- [ ] **Step 3: Write the implementation**

```python
# perovskite_sim/autoloop/ledger.py
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Iterable

from perovskite_sim.autoloop.types import Gap, Hypothesis, NegativeResult


class Ledger:
    """Persistent gap / hypothesis / negative-result store.

    Append-and-replace semantics: gaps dedup on ``id`` (latest wins),
    negative results dedup on the normalised approach key. Updates produce
    new dataclass instances; nothing is mutated in place.
    """

    def __init__(self, root: Path,
                 gaps: Iterable[Gap] = (),
                 hypotheses: Iterable[Hypothesis] = (),
                 negatives: Iterable[NegativeResult] = ()):
        self.root = Path(root)
        self.gaps: list[Gap] = list(gaps)
        self.hypotheses: list[Hypothesis] = list(hypotheses)
        self.negatives: list[NegativeResult] = list(negatives)

    # ---- mutation (append/replace only) ----
    def add_gap(self, gap: Gap) -> None:
        self.gaps = [g for g in self.gaps if g.id != gap.id] + [gap]

    def add_hypothesis(self, hyp: Hypothesis) -> None:
        self.hypotheses = list(self.hypotheses) + [hyp]

    def add_negative(self, neg: NegativeResult) -> None:
        if not self.is_refuted(neg.approach):
            self.negatives = list(self.negatives) + [neg]

    # ---- queries ----
    def is_refuted(self, approach: str) -> bool:
        key = " ".join(approach.lower().split())
        return any(n.dedup_key() == key for n in self.negatives)

    # ---- persistence ----
    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        _dump(self.root / "gaps.json", self.gaps)
        _dump(self.root / "hypotheses.json", self.hypotheses)
        _dump(self.root / "negative_results.json", self.negatives)
        (self.root / "LEDGER.md").write_text(self._markdown(), encoding="utf-8")

    @classmethod
    def load(cls, root: Path) -> "Ledger":
        root = Path(root)
        return cls(
            root=root,
            gaps=_load(root / "gaps.json", Gap),
            hypotheses=_load(root / "hypotheses.json", Hypothesis),
            negatives=_load(root / "negative_results.json", NegativeResult),
        )

    def _markdown(self) -> str:
        lines = ["# Autoloop ledger\n", "## Open gaps\n"]
        for g in sorted(self.gaps, key=lambda x: -x.gap_mag):
            lines.append(f"- `{g.id}` [{g.status}] {g.metric}@{g.sweep} gap={g.gap_mag:.4g} "
                         f"(SL {g.solarlab_val:.4g} vs ref {g.reference_val:.4g})")
        lines.append("\n## Refuted approaches (never retry)\n")
        for n in self.negatives:
            lines.append(f"- {n.approach} — {n.why_failed}")
        return "\n".join(lines) + "\n"


def _dump(path: Path, items: list) -> None:
    path.write_text(
        json.dumps([dataclasses.asdict(i) for i in items], indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _load(path: Path, cls):
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    fields = {f.name for f in dataclasses.fields(cls)}
    out = []
    for d in raw:
        kw = {k: v for k, v in d.items() if k in fields}
        # JSON lists -> tuples for frozen tuple fields
        for f in dataclasses.fields(cls):
            if f.name in kw and isinstance(kw[f.name], list) and "tuple" in str(f.type):
                kw[f.name] = tuple(kw[f.name])
        out.append(cls(**kw))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/autoloop/test_ledger.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add perovskite_sim/autoloop/ledger.py tests/unit/autoloop/test_ledger.py
git commit -m "feat(autoloop): add Ledger with append/replace persistence + dedup"
```

---

## Task 3: Negative-results seeds

**Files:**
- Create: `perovskite_sim/autoloop/seeds.py`
- Test: `tests/unit/autoloop/test_seeds.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/autoloop/test_seeds.py
from perovskite_sim.autoloop.ledger import Ledger
from perovskite_sim.autoloop.seeds import SEED_NEGATIVE_RESULTS, seed_negative_results


def test_seeds_cover_known_refuted_approaches():
    approaches = " ".join(n.approach.lower() for n in SEED_NEGATIVE_RESULTS)
    for needle in ["dos-cap", "bbd", "1.40 v_bi", "two-sided", "shared-occupancy"]:
        assert needle in approaches


def test_seed_is_idempotent(tmp_path):
    led = Ledger(root=tmp_path)
    seed_negative_results(led)
    n_first = len(led.negatives)
    seed_negative_results(led)              # second call must not duplicate
    assert len(led.negatives) == n_first
    assert led.is_refuted("DOS-cap projection target") is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/autoloop/test_seeds.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'perovskite_sim.autoloop.seeds'`.

- [ ] **Step 3: Write the implementation**

```python
# perovskite_sim/autoloop/seeds.py
from __future__ import annotations

from perovskite_sim.autoloop.ledger import Ledger
from perovskite_sim.autoloop.types import NegativeResult

SEED_NEGATIVE_RESULTS: list[NegativeResult] = [
    NegativeResult(
        approach="DOS-cap projection target",
        why_failed="false convergence, high residual at bulk node 19",
        evidence="project_scaps_validation_parked memory; TE-ceiling analysis 2026-06-14",
    ),
    NegativeResult(
        approach="BBD face-density interface term",
        why_failed="V=0.08 blow-up",
        evidence="project_scaps_validation_parked memory (E8 prototype)",
    ),
    NegativeResult(
        approach="1.40 V_bi fudge",
        why_failed="+106 mV unexplained over derived flat-band V_bi; fails G4 honest-residual",
        evidence="project_scaps_root_cause_reanalysis memory 2026-06-07",
    ),
    NegativeResult(
        approach="two-sided additive mirror interface channel",
        why_failed="measured no-op (mirror pair minority-limited, ~uA/m2)",
        evidence="perovskite-sim CLAUDE.md interface_two_sided note",
    ),
    NegativeResult(
        approach="shared-occupancy on bulk-node densities",
        why_failed="CBO trend collapse 80->22%; trap algebra invisible under bulk-node sampling",
        evidence="perovskite-sim CLAUDE.md interface_shared_occupancy note",
    ),
]


def seed_negative_results(ledger: Ledger) -> None:
    """Idempotently load the known-refuted approaches into the ledger."""
    for neg in SEED_NEGATIVE_RESULTS:
        ledger.add_negative(neg)   # add_negative already dedups on normalised key
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/autoloop/test_seeds.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add perovskite_sim/autoloop/seeds.py tests/unit/autoloop/test_seeds.py
git commit -m "feat(autoloop): seed negative-results ledger from known-refuted approaches"
```

---

## Task 4: Provenance stamping

**Files:**
- Create: `perovskite_sim/autoloop/provenance.py`
- Test: `tests/unit/autoloop/test_provenance.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/autoloop/test_provenance.py
from perovskite_sim.autoloop.provenance import stamp, config_hash


def test_config_hash_is_stable_and_content_addressed(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("a: 1\n", encoding="utf-8")
    h1 = config_hash(p)
    h2 = config_hash(p)
    assert h1 == h2 and len(h1) == 64        # sha256 hex
    p.write_text("a: 2\n", encoding="utf-8")
    assert config_hash(p) != h1


def test_stamp_captures_git_and_flags(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("x: 1\n", encoding="utf-8")
    prov = stamp(
        run_id="run-test",
        config_path=cfg,
        flags={"SOLARLAB_DOS_BAND": "1"},
        seed=1234,
        timestamp="2026-06-16T00:00:00Z",
    )
    assert prov.run_id == "run-test"
    assert prov.seed == 1234
    assert prov.timestamp == "2026-06-16T00:00:00Z"
    assert prov.flags == {"SOLARLAB_DOS_BAND": "1"}
    assert isinstance(prov.git_sha, str) and len(prov.git_sha) >= 7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/autoloop/test_provenance.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'perovskite_sim.autoloop.provenance'`.

- [ ] **Step 3: Write the implementation**

```python
# perovskite_sim/autoloop/provenance.py
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from perovskite_sim.autoloop.types import Provenance


def config_hash(path: Path) -> str:
    """SHA-256 of a config file's bytes (content-addressed reproducibility)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*args: str) -> str:
    """Run a git command from the repo root; '' on failure (no exception)."""
    try:
        return subprocess.run(
            ["git", *args],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def stamp(*, run_id: str, config_path: Path, flags: dict[str, str],
          seed: int, timestamp: str) -> Provenance:
    """Build a Provenance record. ``timestamp`` is passed in (ISO-8601),
    never generated here, so a run is reproducible/replayable."""
    sha = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    return Provenance(
        run_id=run_id,
        git_sha=sha or "unknown",
        git_dirty=dirty,
        config_hash=config_hash(config_path),
        flags=dict(flags),
        seed=seed,
        timestamp=timestamp,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/autoloop/test_provenance.py -v`
Expected: PASS (2 tests). (`git_sha` resolves from the SolarLab repo root because git walks up from CWD.)

- [ ] **Step 5: Commit**

```bash
git add perovskite_sim/autoloop/provenance.py tests/unit/autoloop/test_provenance.py
git commit -m "feat(autoloop): add provenance stamping (git SHA, config hash, flags, seed)"
```

---

## Task 5: Parity scorecard

**Files:**
- Create: `perovskite_sim/autoloop/scorecard.py`
- Test: `tests/unit/autoloop/test_scorecard.py`

This module mirrors the V_oc-closure logic of `scripts/run_scaps_v2_regression.py:summarize` but lives in the package and emits `ParityScore` + `Gap`s. The heavy sim call (`run_jv_sweep`) is injected as a callable so unit tests run without the solver.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/autoloop/test_scorecard.py
import json
from perovskite_sim.autoloop.scorecard import (
    SHEET_TO_AXIS, score_parity, gaps_from_score,
)
from perovskite_sim.autoloop.types import ParityScore


def _fake_reference(tmp_path):
    ref = {
        "base_model": {"Voc_V": 1.17, "Jsc_mA_cm2": 26.28, "FF_percent": 87.0, "PCE_percent": 26.7},
        "sweeps": {
            "CHI_ETL": {"x_name": "delta_E_C_eV", "x_unit": "eV", "n_points": 2,
                        "points": [
                            {"x": -0.5, "Voc_V": 0.83, "Jsc_mA_cm2": 26.3, "FF_percent": 82.0, "PCE_percent": 18.0},
                            {"x":  0.0, "Voc_V": 1.25, "Jsc_mA_cm2": 26.3, "FF_percent": 90.0, "PCE_percent": 29.6},
                        ]},
        },
    }
    p = tmp_path / "ref.json"
    p.write_text(json.dumps(ref), encoding="utf-8")
    return p


def test_score_parity_perfect_closure(tmp_path):
    ref = _fake_reference(tmp_path)

    # Fake solver: return SL metrics equal to the SCAPS reference at each point.
    def fake_run(axis, x):
        pt = {-0.5: (0.83, 263.0, 0.82, 18.0), 0.0: (1.25, 263.0, 0.90, 29.6)}[x]
        return pt  # (V_oc, J_sc_A_m2, FF_frac, PCE_frac) -- bracketed

    score = score_parity(
        reference_path=ref, config_path=tmp_path / "unused.yaml",
        run_point=lambda axis, x: (*fake_run(axis, x), True),
        base_point=lambda: (1.17, 262.8, 0.87, 0.267, True),
    )
    assert isinstance(score, ParityScore)
    assert score.per_sweep["CHI_ETL"].voc_closure_pct == 100.0
    assert score.per_sweep["CHI_ETL"].n_bracketed == 2
    assert 0.99 <= score.overall <= 1.0


def test_gaps_from_score_emits_gap_when_closure_low(tmp_path):
    score = ParityScore(
        overall=0.3, base_deltas={"V_oc": -0.07},
        per_sweep={"Nd_ETL": __import__("perovskite_sim.autoloop.types",
                   fromlist=["SweepScore"]).SweepScore("Nd_ETL", 30.0, 5, 4)},
    )
    gaps = gaps_from_score(score, cycle=0, closure_target=70.0, base_tol={"V_oc": 0.02})
    ids = {g.id for g in gaps}
    assert any("Nd_ETL" in i for i in ids)      # low-closure sweep -> gap
    assert any("base" in i and "V_oc" in i for i in ids)  # base abs delta -> gap


def test_sheet_to_axis_has_four_scoreable_sheets():
    assert set(SHEET_TO_AXIS) == {"CHI_ETL", "Nd_ETL", "Nt_PVK ETL", "Nt_C_PVK"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/autoloop/test_scorecard.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'perovskite_sim.autoloop.scorecard'`.

- [ ] **Step 3: Write the implementation**

```python
# perovskite_sim/autoloop/scorecard.py
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Callable, Optional

from perovskite_sim.autoloop.types import Gap, ParityScore, SweepScore

# Only these reference sheets have a SolarLab sweep-axis mapping (Stage 1).
SHEET_TO_AXIS: dict[str, str] = {
    "CHI_ETL":     "etl_delta_ec_eV",
    "Nd_ETL":      "etl_doping_cm3",
    "Nt_PVK ETL":  "interface_defect_N_t_cm2",
    "Nt_C_PVK":    "absorber_defect_density_cm3",
}

# Callable injected by the orchestrator (real solver) or a test (fake).
#   run_point(axis, x)  -> (V_oc, J_sc_A_m2, FF_frac, PCE_frac, bracketed)
#   base_point()        -> (V_oc, J_sc_A_m2, FF_frac, PCE_frac, bracketed)
RunPoint = Callable[[str, float], tuple[float, float, float, float, bool]]
BasePoint = Callable[[], tuple[float, float, float, float, bool]]


def _voc_closure(sl_vocs: list[float], scaps_vocs: list[float]) -> float:
    if len(sl_vocs) < 2:
        return float("nan")
    sl_range = (max(sl_vocs) - min(sl_vocs)) * 1000.0
    scaps_range = (max(scaps_vocs) - min(scaps_vocs)) * 1000.0
    if scaps_range <= 0.0:
        return float("nan")
    return 100.0 * sl_range / scaps_range


def score_parity(*, reference_path: Path, config_path: Path,
                 run_point: RunPoint, base_point: BasePoint,
                 skip_log: Optional[list[str]] = None) -> ParityScore:
    """Score SolarLab parity against the SCAPS reference JSON.

    ``run_point`` / ``base_point`` are injected so the math is testable
    without the solver; the orchestrator wires the real ``run_jv_sweep``.
    Unmapped reference sheets are skipped and logged (no silent cap).
    """
    ref = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    per_sweep: dict[str, SweepScore] = {}

    for sheet, axis in SHEET_TO_AXIS.items():
        sweep = ref["sweeps"].get(sheet)
        if sweep is None:
            if skip_log is not None:
                skip_log.append(f"reference missing sweep '{sheet}' — skipped")
            continue
        sl_vocs, scaps_vocs, n_brk = [], [], 0
        pts = sweep["points"]
        for pt in pts:
            x = float(pt["x"])
            voc_sl, _jsc, _ff, _pce, bracketed = run_point(axis, x)
            if bracketed and voc_sl == voc_sl:   # not NaN
                sl_vocs.append(voc_sl)
                scaps_vocs.append(float(pt["Voc_V"]))
                n_brk += 1
        per_sweep[sheet] = SweepScore(
            sweep=sheet,
            voc_closure_pct=_voc_closure(sl_vocs, scaps_vocs),
            n_points=len(pts),
            n_bracketed=n_brk,
        )

    # base-point absolute deltas (solarlab - reference)
    bm = ref["base_model"]
    voc, jsc_A, ff, pce, _brk = base_point()
    base_deltas = {
        "V_oc": voc - float(bm["Voc_V"]),
        "J_sc": jsc_A - float(bm["Jsc_mA_cm2"]) * 10.0,
        "FF":   ff - float(bm["FF_percent"]) / 100.0,
        "PCE":  pce - float(bm["PCE_percent"]) / 100.0,
    }

    closures = [s.voc_closure_pct for s in per_sweep.values() if s.voc_closure_pct == s.voc_closure_pct]
    overall = max(0.0, min(1.0, mean(closures) / 100.0)) if closures else 0.0
    return ParityScore(overall=overall, base_deltas=base_deltas, per_sweep=per_sweep)


def gaps_from_score(score: ParityScore, *, cycle: int,
                    closure_target: float = 70.0,
                    base_tol: Optional[dict[str, float]] = None) -> list[Gap]:
    """Emit a ranked Gap per under-target sweep + per out-of-tolerance base metric."""
    base_tol = base_tol or {"V_oc": 0.02, "J_sc": 10.0, "FF": 0.03, "PCE": 0.02}
    gaps: list[Gap] = []

    for sheet, s in score.per_sweep.items():
        if s.voc_closure_pct == s.voc_closure_pct and s.voc_closure_pct < closure_target:
            mag = (closure_target - s.voc_closure_pct) / 100.0
            gaps.append(Gap(
                id=f"trend:{sheet}:V_oc", metric="V_oc", sweep=sheet, sweep_point=float("nan"),
                solarlab_val=s.voc_closure_pct, reference_val=closure_target, gap_mag=mag,
                kind="trend", status="open", found_cycle=cycle, last_attempt_cycle=cycle,
            ))

    for metric, delta in score.base_deltas.items():
        tol = base_tol.get(metric, float("inf"))
        if abs(delta) > tol:
            gaps.append(Gap(
                id=f"absolute:base:{metric}", metric=metric, sweep="base", sweep_point=0.0,
                solarlab_val=delta, reference_val=0.0, gap_mag=abs(delta),
                kind="absolute", status="open", found_cycle=cycle, last_attempt_cycle=cycle,
            ))

    gaps.sort(key=lambda g: -g.gap_mag)
    return gaps
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/autoloop/test_scorecard.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add perovskite_sim/autoloop/scorecard.py tests/unit/autoloop/test_scorecard.py
git commit -m "feat(autoloop): add parity scorecard (V_oc closure + base deltas -> gaps)"
```

---

## Task 6: Ladder runner (L0/L1/L2)

**Files:**
- Create: `perovskite_sim/autoloop/ladder.py`
- Test: `tests/unit/autoloop/test_ladder.py`

The ladder wires the real solver into the scorecard's injected callables and runs L0 (pytest subset) and L1 (limiting cases). Heavy calls are guarded so unit tests exercise the wiring with fakes; the real run is covered by the Task 10 integration smoke.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/autoloop/test_ladder.py
from perovskite_sim.autoloop.ladder import run_l0, build_run_callables, run_ladder
from perovskite_sim.autoloop.types import LadderResult


def test_run_l0_reports_pass_on_green_subprocess(monkeypatch):
    class _CP:
        returncode = 0
        stdout = "5 passed"
        stderr = ""
    monkeypatch.setattr("perovskite_sim.autoloop.ladder.subprocess.run",
                        lambda *a, **k: _CP())
    ok, detail = run_l0(["tests/unit/autoloop"])
    assert ok is True
    assert "passed" in detail


def test_run_l0_reports_fail_on_red_subprocess(monkeypatch):
    class _CP:
        returncode = 1
        stdout = "1 failed"
        stderr = ""
    monkeypatch.setattr("perovskite_sim.autoloop.ladder.subprocess.run",
                        lambda *a, **k: _CP())
    ok, _ = run_l0(["tests/unit/autoloop"])
    assert ok is False


def test_run_ladder_short_circuits_when_l0_fails(monkeypatch, tmp_path):
    monkeypatch.setattr("perovskite_sim.autoloop.ladder.run_l0", lambda paths: (False, "1 failed"))
    res = run_ladder(reference_path=tmp_path / "ref.json", config_path=tmp_path / "c.yaml",
                     l0_paths=["tests/unit/autoloop"])
    assert isinstance(res, LadderResult)
    assert res.l0_pass is False
    assert res.score is None        # L2 not reached
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/autoloop/test_ladder.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'perovskite_sim.autoloop.ladder'`.

- [ ] **Step 3: Write the implementation**

```python
# perovskite_sim/autoloop/ladder.py
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from perovskite_sim.autoloop.scorecard import (
    SHEET_TO_AXIS, score_parity, RunPoint, BasePoint,
)
from perovskite_sim.autoloop.types import LadderResult, ParityScore

# Default JV envelope — the campaign parity setting (verified).
DEFAULT_JV_KWARGS = dict(N_grid=30, n_points=20, v_rate=5.0, V_max=1.6)


def run_l0(paths: list[str]) -> tuple[bool, str]:
    """L0 numerics gate: run a fast pytest subset; pass iff returncode == 0."""
    proc = subprocess.run(
        ["python", "-m", "pytest", "-q", "-m", "not slow", *paths],
        capture_output=True, text=True,
    )
    ok = proc.returncode == 0
    tail = (proc.stdout or proc.stderr).strip().splitlines()
    return ok, (tail[-1] if tail else "")


def run_l1_limiting_cases(run_voc_radiative_only, detailed_balance_ceiling: float,
                          run_dark_jsc) -> tuple[bool, dict]:
    """L1 physics-sanity gate. Two cheap limiting checks:

    - rad-only V_oc must not exceed the detailed-balance ceiling.
    - dark J_sc must be ~0 (no photocurrent without light).
    Both probes are injected so the module stays solver-agnostic/testable.
    """
    voc_rad = run_voc_radiative_only()
    dark_jsc = run_dark_jsc()
    ok = (voc_rad <= detailed_balance_ceiling + 1e-6) and (abs(dark_jsc) < 1.0)  # A/m^2
    return ok, {"voc_radiative_only": voc_rad, "ceiling": detailed_balance_ceiling,
                "dark_jsc": dark_jsc}


def build_run_callables(config_path: Path,
                        jv_kwargs: Optional[dict] = None) -> tuple[RunPoint, BasePoint]:
    """Wire the real solver into the scorecard's injected interface.

    Imports the heavy sim lazily so importing this module is cheap.
    """
    from perovskite_sim.experiments.jv_sweep import run_jv_sweep
    from perovskite_sim.scaps_compat import load_scaps_yaml
    from perovskite_sim.sweeps.device_parameter_sweep import SweepPoint, apply_sweep_point

    jv_kwargs = {**DEFAULT_JV_KWARGS, **(jv_kwargs or {})}
    base_stack = load_scaps_yaml(config_path)

    def _metrics(stack):
        res = run_jv_sweep(stack, **jv_kwargs)
        m = res.metrics_fwd
        return (m.V_oc, m.J_sc, m.FF, m.PCE, m.voc_bracketed)

    def run_point(axis: str, x: float):
        sp = SweepPoint("p", axis, f"{x:.3e}", {axis: x})
        try:
            return _metrics(apply_sweep_point(base_stack, sp))
        except Exception:
            return (float("nan"), float("nan"), float("nan"), float("nan"), False)

    def base_point():
        return _metrics(base_stack)

    return run_point, base_point


def run_ladder(*, reference_path: Path, config_path: Path,
               l0_paths: list[str],
               run_point: Optional[RunPoint] = None,
               base_point: Optional[BasePoint] = None,
               l1: Optional[tuple[bool, dict]] = None) -> LadderResult:
    """Run L0 -> L1 -> L2 with fail-fast short-circuiting."""
    l0_ok, l0_detail = run_l0(l0_paths)
    if not l0_ok:
        return LadderResult(l0_pass=False, l1_pass=False, score=None,
                            details={"l0": l0_detail})

    l1_ok, l1_detail = (True, {}) if l1 is None else l1
    if not l1_ok:
        return LadderResult(l0_pass=True, l1_pass=False, score=None,
                            details={"l0": l0_detail, "l1": l1_detail})

    if run_point is None or base_point is None:
        run_point, base_point = build_run_callables(config_path)

    skip_log: list[str] = []
    score = score_parity(reference_path=reference_path, config_path=config_path,
                         run_point=run_point, base_point=base_point, skip_log=skip_log)
    return LadderResult(l0_pass=True, l1_pass=True, score=score,
                        details={"l0": l0_detail, "l1": l1_detail, "skipped": skip_log})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/autoloop/test_ladder.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add perovskite_sim/autoloop/ladder.py tests/unit/autoloop/test_ladder.py
git commit -m "feat(autoloop): add L0/L1/L2 ladder runner with fail-fast short-circuit"
```

---

## Task 7: Gate stack (G1–G3 deterministic; G0/G4/G5 deferred)

**Files:**
- Create: `perovskite_sim/autoloop/gates.py`
- Test: `tests/unit/autoloop/test_gates.py`

In Stage 1 the gates operate on a `LadderResult` plus a committed **baseline** `ParityScore` (the parity floor). G0 (legacy bit-identical) and G4/G5 (cognition) require a proposed flag-gated change + agents, so they are explicit `deferred` stubs that raise — the orchestrator never calls them in Stage 1.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/autoloop/test_gates.py
import pytest
from perovskite_sim.autoloop.types import LadderResult, ParityScore, SweepScore
from perovskite_sim.autoloop.gates import run_gate_stack, gate_g4_deferred


def _score(overall, closure):
    return ParityScore(overall=overall, base_deltas={"V_oc": 0.0},
                       per_sweep={"CHI_ETL": SweepScore("CHI_ETL", closure, 10, 8)})


def test_gates_pass_when_no_regression():
    res = LadderResult(l0_pass=True, l1_pass=True, score=_score(0.80, 80.0), details={})
    verdicts = run_gate_stack(res, baseline=_score(0.78, 78.0), regression_tol=0.01)
    assert all(v.passed for v in verdicts)


def test_g1_fails_when_l0_red():
    res = LadderResult(l0_pass=False, l1_pass=False, score=None, details={})
    verdicts = {v.name: v for v in run_gate_stack(res, baseline=_score(0.78, 78.0))}
    assert verdicts["G1_numerics"].passed is False
    assert verdicts["G3_scorecard"].passed is False   # cannot improve without a score


def test_g3_fails_on_parity_regression():
    res = LadderResult(l0_pass=True, l1_pass=True, score=_score(0.70, 70.0), details={})
    verdicts = {v.name: v for v in run_gate_stack(res, baseline=_score(0.80, 80.0),
                                                  regression_tol=0.01)}
    assert verdicts["G3_scorecard"].passed is False    # 0.70 < 0.80 - 0.01


def test_g4_g5_are_deferred_stubs():
    with pytest.raises(NotImplementedError):
        gate_g4_deferred(mechanism=None, residual=None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/autoloop/test_gates.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'perovskite_sim.autoloop.gates'`.

- [ ] **Step 3: Write the implementation**

```python
# perovskite_sim/autoloop/gates.py
from __future__ import annotations

from typing import Optional

from perovskite_sim.autoloop.types import GateVerdict, LadderResult, ParityScore


def run_gate_stack(result: LadderResult, *, baseline: Optional[ParityScore] = None,
                   regression_tol: float = 0.01) -> list[GateVerdict]:
    """Deterministic G1-G3. G0/G4/G5 are deferred (Stage 3, cognition).

    G1 numerics    : L0 pytest subset green.
    G2 limiting    : L1 limiting cases hold.
    G3 scorecard   : overall parity did not regress vs baseline beyond tol.
    """
    g1 = GateVerdict("G1_numerics", result.l0_pass,
                     "pytest subset green" if result.l0_pass else "L0 unit/numerics failed")
    g2 = GateVerdict("G2_limiting", result.l1_pass,
                     "limiting cases hold" if result.l1_pass else "L1 limiting case violated")

    if result.score is None:
        g3 = GateVerdict("G3_scorecard", False, "no parity score (L0/L1 short-circuited)")
    elif baseline is None:
        g3 = GateVerdict("G3_scorecard", True, f"no baseline; overall={result.score.overall:.3f}")
    else:
        regressed = result.score.overall < baseline.overall - regression_tol
        g3 = GateVerdict(
            "G3_scorecard", not regressed,
            f"overall {result.score.overall:.3f} vs baseline {baseline.overall:.3f} "
            f"(tol {regression_tol})",
        )
    return [g1, g2, g3]


def all_passed(verdicts: list[GateVerdict]) -> bool:
    return all(v.passed for v in verdicts)


# ---- Deferred gates (Stage 3 — require a proposed flag-gated change + cognition) ----
def gate_g0_deferred(*args, **kwargs):
    raise NotImplementedError("G0 legacy-bit-identical needs a proposed flagged change (Stage 3)")


def gate_g4_deferred(*, mechanism, residual):
    raise NotImplementedError("G4 honest-residual fudge-guard needs a cognition mechanism (Stage 3)")


def gate_g5_deferred(*args, **kwargs):
    raise NotImplementedError("G5 adversarial-verify needs the cognition skeptics (Stage 3)")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/autoloop/test_gates.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add perovskite_sim/autoloop/gates.py tests/unit/autoloop/test_gates.py
git commit -m "feat(autoloop): add deterministic gate stack G1-G3 (+ deferred G0/G4/G5 stubs)"
```

---

## Task 8: Guardian orchestrator

**Files:**
- Create: `perovskite_sim/autoloop/orchestrator.py`
- Test: `tests/unit/autoloop/test_orchestrator.py`

`guardian_once` ties the spine together: run the ladder, score, rank gaps into the ledger, run the gate stack, stamp provenance, persist. No implement/land (Stage 3). Heavy pieces injected for testability.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/autoloop/test_orchestrator.py
from perovskite_sim.autoloop.types import LadderResult, ParityScore, SweepScore
from perovskite_sim.autoloop.orchestrator import guardian_once


def _score(overall=0.40):
    return ParityScore(overall=overall, base_deltas={"V_oc": -0.07},
                       per_sweep={"Nd_ETL": SweepScore("Nd_ETL", 30.0, 5, 4)})


def test_guardian_once_records_gaps_and_returns_verdicts(tmp_path):
    fake_ladder = LadderResult(l0_pass=True, l1_pass=True, score=_score(), details={})
    report = guardian_once(
        ledger_root=tmp_path / "ledger",
        outputs_root=tmp_path / "out",
        reference_path=tmp_path / "ref.json",
        config_path=tmp_path / "c.yaml",
        cycle=0,
        timestamp="2026-06-16T00:00:00Z",
        run_ladder_fn=lambda **kw: fake_ladder,
        baseline=_score(0.80),       # current 0.40 << baseline 0.80 -> G3 fails
    )
    assert report["gate_passed"] is False           # parity regressed
    assert any("Nd_ETL" in g for g in report["gap_ids"])
    assert (tmp_path / "ledger" / "gaps.json").exists()
    assert (tmp_path / "out" / "run-0" / "report.json").exists()


def test_guardian_once_seeds_negative_results(tmp_path):
    fake_ladder = LadderResult(l0_pass=True, l1_pass=True, score=_score(0.9), details={})
    guardian_once(
        ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
        reference_path=tmp_path / "ref.json", config_path=tmp_path / "c.yaml",
        cycle=0, timestamp="2026-06-16T00:00:00Z",
        run_ladder_fn=lambda **kw: fake_ladder, baseline=None,
    )
    from perovskite_sim.autoloop.ledger import Ledger
    led = Ledger.load(tmp_path / "ledger")
    assert led.is_refuted("DOS-cap projection target") is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/autoloop/test_orchestrator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'perovskite_sim.autoloop.orchestrator'`.

- [ ] **Step 3: Write the implementation**

```python
# perovskite_sim/autoloop/orchestrator.py
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Callable, Optional

from perovskite_sim.autoloop.gates import run_gate_stack, all_passed
from perovskite_sim.autoloop.ladder import run_ladder as _run_ladder
from perovskite_sim.autoloop.ledger import Ledger
from perovskite_sim.autoloop.provenance import stamp
from perovskite_sim.autoloop.scorecard import gaps_from_score
from perovskite_sim.autoloop.seeds import seed_negative_results
from perovskite_sim.autoloop.types import LadderResult, ParityScore


def guardian_once(*, ledger_root: Path, outputs_root: Path,
                  reference_path: Path, config_path: Path,
                  cycle: int, timestamp: str,
                  l0_paths: Optional[list[str]] = None,
                  baseline: Optional[ParityScore] = None,
                  flags: Optional[dict[str, str]] = None,
                  seed: int = 0,
                  run_ladder_fn: Optional[Callable[..., LadderResult]] = None) -> dict:
    """One guardian cycle: sense -> score -> rank -> record -> gate. No landing.

    Returns a JSON-serialisable report and persists the ledger + run artifacts.
    """
    ledger_root = Path(ledger_root)
    run_dir = Path(outputs_root) / f"run-{cycle}"
    run_dir.mkdir(parents=True, exist_ok=True)
    l0_paths = l0_paths or ["tests/unit", "tests/integration"]

    led = Ledger.load(ledger_root)
    seed_negative_results(led)               # idempotent

    runner = run_ladder_fn or _run_ladder
    result = runner(reference_path=reference_path, config_path=config_path, l0_paths=l0_paths)

    gap_ids: list[str] = []
    if result.score is not None:
        for g in gaps_from_score(result.score, cycle=cycle):
            if led.is_refuted(g.id):         # never resurface a refuted approach
                continue
            led.add_gap(g)
            gap_ids.append(g.id)

    verdicts = run_gate_stack(result, baseline=baseline)
    prov = stamp(run_id=f"run-{cycle}", config_path=config_path,
                 flags=flags or {}, seed=seed, timestamp=timestamp)

    led.save()
    report = {
        "cycle": cycle,
        "provenance": dataclasses.asdict(prov),
        "l0_pass": result.l0_pass,
        "l1_pass": result.l1_pass,
        "overall": None if result.score is None else result.score.overall,
        "gap_ids": gap_ids,
        "verdicts": [dataclasses.asdict(v) for v in verdicts],
        "gate_passed": all_passed(verdicts),
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True),
                                         encoding="utf-8")
    return report
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/autoloop/test_orchestrator.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add perovskite_sim/autoloop/orchestrator.py tests/unit/autoloop/test_orchestrator.py
git commit -m "feat(autoloop): add guardian orchestrator (sense/score/rank/record/gate)"
```

---

## Task 9: CLI entry point

**Files:**
- Create: `scripts/autoloop_run.py`
- Modify: `perovskite_sim/autoloop/__init__.py` (export `guardian_once`)
- Test: `tests/unit/autoloop/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/autoloop/test_cli.py
import importlib.util
import sys
from pathlib import Path

CLI = Path(__file__).resolve().parents[3] / "scripts" / "autoloop_run.py"


def _load_cli():
    spec = importlib.util.spec_from_file_location("autoloop_run", CLI)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["autoloop_run"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_cli_parse_args_defaults():
    mod = _load_cli()
    ns = mod.parse_args(["--once"])
    assert ns.once is True
    assert ns.cycle == 0


def test_cli_build_timestamp_is_iso():
    mod = _load_cli()
    ts = mod.iso_timestamp_utc()
    assert ts.endswith("Z") and "T" in ts
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/autoloop/test_cli.py -v`
Expected: FAIL — `FileNotFoundError` / `ModuleNotFoundError` for `scripts/autoloop_run.py`.

- [ ] **Step 3: Write the implementation**

First export `guardian_once`:

```python
# perovskite_sim/autoloop/__init__.py  (append guardian_once to imports + __all__)
from perovskite_sim.autoloop.orchestrator import guardian_once  # noqa: E402

__all__ += ["guardian_once"]
```

Then the CLI:

```python
# scripts/autoloop_run.py
#!/usr/bin/env python
"""Autoloop guardian CLI (Stage 1).

Run one sense-and-record guardian cycle against the SCAPS reference:

    python scripts/autoloop_run.py --once

Exits non-zero if the gate stack fails (CI-friendly).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

from perovskite_sim.autoloop import guardian_once

REPO_ROOT = Path(__file__).resolve().parents[1]                      # perovskite-sim/
DEFAULT_REFERENCE = REPO_ROOT / "tests" / "integration" / "scaps_reference.json"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "scaps_mirror_v2.yaml"
DEFAULT_LEDGER = REPO_ROOT.parent / "docs" / "autoloop" / "ledger"   # SolarLab/docs/...
DEFAULT_OUTPUTS = REPO_ROOT.parent / "outputs" / "autoloop"
DEFAULT_BASELINE = REPO_ROOT / "tests" / "integration" / "autoloop_parity_baseline.json"


def iso_timestamp_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Autoloop guardian (Stage 1)")
    ap.add_argument("--once", action="store_true", help="run a single cycle")
    ap.add_argument("--cycle", type=int, default=0)
    ap.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--ledger-root", type=Path, default=DEFAULT_LEDGER)
    ap.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS)
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--l0-paths", nargs="*", default=["tests/unit/autoloop"])
    return ap.parse_args(argv)


def _load_baseline(path: Path):
    if not path.exists():
        return None
    from perovskite_sim.autoloop.types import ParityScore, SweepScore
    raw = json.loads(path.read_text(encoding="utf-8"))
    per = {k: SweepScore(**v) for k, v in raw.get("per_sweep", {}).items()}
    return ParityScore(overall=raw["overall"], base_deltas=raw.get("base_deltas", {}),
                       per_sweep=per)


def main(argv: list[str] | None = None) -> int:
    ns = parse_args(argv if argv is not None else sys.argv[1:])
    report = guardian_once(
        ledger_root=ns.ledger_root, outputs_root=ns.outputs_root,
        reference_path=ns.reference, config_path=ns.config,
        cycle=ns.cycle, timestamp=iso_timestamp_utc(),
        l0_paths=ns.l0_paths, baseline=_load_baseline(ns.baseline),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/autoloop/test_cli.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/autoloop_run.py perovskite_sim/autoloop/__init__.py tests/unit/autoloop/test_cli.py
git commit -m "feat(autoloop): add guardian CLI (--once, CI exit code)"
```

---

## Task 10: End-to-end guardian smoke + docs

**Files:**
- Create: `tests/integration/test_autoloop_guardian.py`
- Modify: `.gitignore` (ignore `outputs/autoloop/` artifacts)
- Modify: `perovskite-sim/CLAUDE.md` (add an Autoloop section)
- Modify: `README.md` (one-line mention + command)

- [ ] **Step 1: Write the failing test (real solver, marked slow)**

```python
# tests/integration/test_autoloop_guardian.py
import pytest
from pathlib import Path
from perovskite_sim.autoloop import guardian_once

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_guardian_once_real_solver_produces_report(tmp_path):
    """Full sense-and-record cycle on the real scaps_mirror_v2 config.

    Asserts the ladder ran the real solver, produced a parity score, and
    wrote a report — without asserting a specific parity number (that is
    the moving target the loop tracks).
    """
    report = guardian_once(
        ledger_root=tmp_path / "ledger",
        outputs_root=tmp_path / "out",
        reference_path=REPO_ROOT / "tests" / "integration" / "scaps_reference.json",
        config_path=REPO_ROOT / "configs" / "scaps_mirror_v2.yaml",
        cycle=0,
        timestamp="2026-06-16T00:00:00Z",
        l0_paths=["tests/unit/autoloop"],     # keep L0 fast inside the smoke
        baseline=None,
    )
    assert report["overall"] is not None
    assert 0.0 <= report["overall"] <= 1.0
    assert (tmp_path / "out" / "run-0" / "report.json").exists()
    assert (tmp_path / "ledger" / "gaps.json").exists()
```

- [ ] **Step 2: Run test to verify it fails (or is collected)**

Run: `pytest -m slow tests/integration/test_autoloop_guardian.py -v`
Expected: FAIL first if any wiring bug remains; once green, it confirms the real solver path. (Runtime: a handful of `run_jv_sweep` calls at `N_grid=30` — order ~1–2 min. If it exceeds budget, reduce `SHEET_TO_AXIS` to `CHI_ETL` only for the smoke via a future `sheets=` kwarg — do NOT silently cap; log it.)

- [ ] **Step 3: Make it pass + wire docs/gitignore**

Add to `.gitignore`:

```
# Autoloop run artifacts (ledger is tracked under docs/autoloop/, outputs are not)
outputs/autoloop/
```

Add an **Autoloop** section to `perovskite-sim/CLAUDE.md` (under a new top-level `## Autoloop` heading), text:

```markdown
## Autoloop (Stage 1 — guardian)

`perovskite_sim/autoloop/` is the deterministic spine of the continuous
research-loop orchestrator (design: `docs/superpowers/specs/2026-06-16-autoloop-research-pipeline-design.md`).
Stage 1 ships the guardian only — no agents, no auto-implementation.

- `scorecard.py` scores SolarLab vs the SCAPS reference (`tests/integration/scaps_reference.json`)
  by V_oc trend closure per sweep + base-point absolute deltas → `ParityScore`.
- `ladder.py` runs L0 (pytest subset) → L1 (limiting cases) → L2 (scorecard), fail-fast.
- `gates.py` G1–G3 are deterministic regression barriers; G0/G4/G5 are deferred
  stubs (need a proposed flag-gated change + cognition, Stage 3) and raise if called.
- `ledger.py` persists gap / hypothesis / negative-result ledgers under
  `docs/autoloop/ledger/`; the negative-results ledger is seeded from known-refuted
  approaches (`seeds.py`) so the loop never re-tries them.

Run one guardian cycle (exits non-zero if the gate stack fails):

    cd perovskite-sim
    python scripts/autoloop_run.py --once

Scoreable sweeps are the four with a SolarLab axis mapping (`scorecard.SHEET_TO_AXIS`);
unmapped reference sheets are skipped and logged. Stages 2–4 (cognition attribution,
auto-implement, continuous boulder) are not built yet.
```

Add to `README.md` (in the existing tooling/commands area), one line:

```markdown
- **Autoloop guardian** (`python perovskite-sim/scripts/autoloop_run.py --once`) — runs the
  L0–L2 ladder, scores SolarLab-vs-SCAPS parity, and records regressions to the gap ledger.
```

- [ ] **Step 4: Run the full fast suite + the slow smoke**

Run: `pytest tests/unit/autoloop -v && pytest -m slow tests/integration/test_autoloop_guardian.py -v`
Expected: all green. Also run `pytest -q` (full default suite) to confirm no collection or import regressions from the new package.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_autoloop_guardian.py .gitignore perovskite-sim/CLAUDE.md README.md
git commit -m "test(autoloop): end-to-end guardian smoke + docs (README/CLAUDE.md in sync)"
```

---

## Self-Review

**Spec coverage (vs `2026-06-16-autoloop-research-pipeline-design.md` §12 Stage 1 = "Spine + gates + ledgers + scorecard"):**
- Ledgers (gap/hypothesis/negative-result) → Tasks 1–3. ✓
- Provenance → Task 4. ✓
- Scorecard (trend closure + base deltas + gap ranking) → Task 5. ✓
- Ladder L0/L1/L2 → Task 6. ✓
- Gate stack (G1–G3 deterministic; G0/G4/G5 deferred per "needs cognition") → Task 7. ✓
- Anti-thrash dedup (spine refuses refuted approaches) → Ledger `is_refuted` (Task 2) + orchestrator skip (Task 8) + seeds (Task 3). ✓
- Sense-and-record orchestrator + CLI + CI exit code → Tasks 8–9. ✓
- Docs-in-sync (README + CLAUDE.md) per user feedback rule → Task 10. ✓
- Out of scope (cognition, implement, boulder, L3 calibration, L4 search) → correctly NOT built; G0/G4/G5 are explicit deferred stubs.

**Placeholder scan:** No TBD/TODO. Every code step shows complete code; every test step shows real assertions; every run step shows the exact command + expected outcome. The one conditional ("if the smoke exceeds budget, add a `sheets=` kwarg") is a documented future option, not a placeholder in shipped code.

**Type consistency:** `Gap`, `ParityScore`, `SweepScore`, `LadderResult`, `GateVerdict`, `Provenance` are defined once in `types.py` (Task 1) and imported unchanged everywhere. `run_point`/`base_point` signatures `(axis,x)->(V_oc,J_sc,FF,PCE,bracketed)` are consistent between `scorecard.py` (Task 5) and `ladder.build_run_callables` (Task 6). `guardian_once`'s `run_ladder_fn` matches `ladder.run_ladder`'s keyword signature (`reference_path`, `config_path`, `l0_paths`). `Ledger.is_refuted` (Task 2) is used by the orchestrator (Task 8) with the same normalised-key semantics seeded in Task 3.

**Known approximation (called out, not hidden):** `scorecard.py` re-implements the V_oc-closure math from `scripts/run_scaps_v2_regression.py:summarize` rather than importing it (script modules are not importable package API). The two should not drift; a future refactor can lift `summarize` into the package and have both call it.

---

## Execution Handoff

After saving, choose execution:

1. **Subagent-Driven (recommended)** — fresh subagent per task, two-stage review between tasks.
2. **Inline Execution** — batch tasks in this session with checkpoints.
