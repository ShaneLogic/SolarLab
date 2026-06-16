# Autoloop Stage 4c — L4 Design-Search — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a parity-gated, advisory design-search that searches the device-design space (band offsets / doping / defects) for designs maximizing PCE, via a pluggable Optimizer seam with a no-dependency seeded random-search default.

**Architecture:** A new `search.py` with `DesignKnob`/`Trial`/`SearchResult` types, an `Optimizer` protocol + `RandomSearchOptimizer`, `make_design_objective` (apply_sweep_point → run_jv_sweep → PCE), and `run_design_search` that refuses unless `score_parity().overall ≥ parity_target` then runs the optimizer and writes an advisory report. Nothing is applied to any config — designs are proposed only.

**Tech Stack:** Python 3.9+, dataclasses, random, math. Reuses `scaps_compat.load_scaps_yaml`, `sweeps.device_parameter_sweep.SweepPoint`/`apply_sweep_point`, `experiments.jv_sweep.run_jv_sweep`, `autoloop.ladder.build_run_callables`/`DEFAULT_JV_KWARGS`, `autoloop.scorecard.score_parity`. No new third-party deps.

---

## Design contract (read before starting)

- **Spec:** `docs/superpowers/specs/2026-06-16-autoloop-stage4c-design-search-design.md`.
- **Verified APIs:**
  - `apply_sweep_point(base_stack, SweepPoint(name, axis, label, updates: dict)) -> DeviceStack` — `updates` may carry MULTIPLE axes (the existing coupled sweeps prove it).
  - `run_jv_sweep(stack, **jv_kwargs).metrics_fwd` → `JVMetrics(V_oc, J_sc, FF, PCE, voc_bracketed)`; `PCE` is a fraction.
  - `scaps_compat.load_scaps_yaml(path) -> DeviceStack`.
  - `autoloop.ladder.build_run_callables(config_path, jv_kwargs=None) -> (run_point, base_point)`; `autoloop.ladder.DEFAULT_JV_KWARGS`.
  - `autoloop.scorecard.score_parity(*, reference_path, config_path, run_point, base_point) -> ParityScore` with `.overall` (0..1).
- **Design knobs** are `apply_sweep_point` axes: `etl_delta_ec_eV`, `htl_delta_ev_eV`, `etl_doping_cm3`, `absorber_doping_cm3`, `absorber_defect_depth_eV`, `absorber_defect_density_cm3`.
- **Module imports the solver at top level** (search is CLI-invoked, not a hot import) so tests can monkeypatch `search.run_jv_sweep` / `search.load_scaps_yaml` / `search.apply_sweep_point`.
- **Run all commands from `perovskite-sim/`.** Tests default to `-m 'not slow'`.

## File Structure

```
perovskite_sim/autoloop/
  search.py   DesignKnob, Trial, SearchResult, SearchNotTrusted, DEFAULT_DESIGN_SPACE,
              RandomSearchOptimizer, make_design_objective, run_design_search, _default_parity
scripts/autoloop_run.py  + --search [--budget] [--parity-target]
tests/unit/autoloop/
  test_search_types.py
  test_search_optimizer.py
  test_search_objective.py
  test_search_orchestration.py
tests/integration/
  test_autoloop_search.py   (slow)
```

---

## Task 1: search types + DEFAULT_DESIGN_SPACE

**Files:**
- Create: `perovskite_sim/autoloop/search.py`
- Test: `tests/unit/autoloop/test_search_types.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/autoloop/test_search_types.py
import dataclasses
import pytest
from perovskite_sim.autoloop.search import (
    DesignKnob, Trial, SearchResult, SearchNotTrusted, DEFAULT_DESIGN_SPACE,
)


def test_design_knob_frozen():
    k = DesignKnob(axis="etl_doping_cm3", low=1e15, high=1e19, scale="log")
    assert k.scale == "log"
    with pytest.raises(dataclasses.FrozenInstanceError):
        k.low = 0.0  # type: ignore[misc]


def test_trial_and_result():
    t = Trial(design={"etl_doping_cm3": 1e17}, pce=0.27, bracketed=True)
    r = SearchResult(best=t, trials=(t,), n_evaluated=1, parity_overall=0.9, budget=10)
    assert r.best.pce == 0.27 and r.n_evaluated == 1


def test_search_not_trusted_is_runtimeerror():
    assert issubclass(SearchNotTrusted, RuntimeError)


def test_default_design_space_axes():
    axes = {k.axis for k in DEFAULT_DESIGN_SPACE}
    assert axes == {"etl_delta_ec_eV", "htl_delta_ev_eV",
                    "etl_doping_cm3", "absorber_defect_density_cm3"}
    assert all(k.low < k.high for k in DEFAULT_DESIGN_SPACE)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_search_types.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'perovskite_sim.autoloop.search'`.

- [ ] **Step 3: Write `search.py` (types + space + imports)**

```python
# perovskite_sim/autoloop/search.py
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Protocol

from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.sweeps.device_parameter_sweep import SweepPoint, apply_sweep_point
from perovskite_sim.experiments.jv_sweep import run_jv_sweep
from perovskite_sim.autoloop.ladder import DEFAULT_JV_KWARGS, build_run_callables
from perovskite_sim.autoloop.scorecard import score_parity

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DesignKnob:
    axis: str
    low: float
    high: float
    scale: str = "linear"     # "linear" | "log"


@dataclass(frozen=True)
class Trial:
    design: dict
    pce: float
    bracketed: bool


@dataclass(frozen=True)
class SearchResult:
    best: Optional[Trial]
    trials: tuple
    n_evaluated: int
    parity_overall: float
    budget: int


class SearchNotTrusted(RuntimeError):
    """Raised when the model's parity is below the trust threshold."""


DEFAULT_DESIGN_SPACE = [
    DesignKnob("etl_delta_ec_eV", -0.3, 0.3, "linear"),
    DesignKnob("htl_delta_ev_eV", -0.3, 0.3, "linear"),
    DesignKnob("etl_doping_cm3", 1e15, 1e19, "log"),
    DesignKnob("absorber_defect_density_cm3", 1e14, 1e17, "log"),
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_search_types.py`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add perovskite-sim/perovskite_sim/autoloop/search.py perovskite-sim/tests/unit/autoloop/test_search_types.py
git commit -m "feat(autoloop): add design-search types + DEFAULT_DESIGN_SPACE (Stage 4c)"
```

---

## Task 2: RandomSearchOptimizer

**Files:**
- Modify: `perovskite_sim/autoloop/search.py`
- Test: `tests/unit/autoloop/test_search_optimizer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/autoloop/test_search_optimizer.py
import pytest
from perovskite_sim.autoloop.search import DesignKnob, RandomSearchOptimizer


SPACE = [DesignKnob("a", 0.0, 10.0, "linear"), DesignKnob("b", 1e2, 1e6, "log")]


def _obj(design):
    # PCE proxy = value of knob "a"; always bracketed
    return (design["a"], True)


def test_seeded_determinism():
    o1 = RandomSearchOptimizer(seed=7).optimize(_obj, SPACE, budget=8)
    o2 = RandomSearchOptimizer(seed=7).optimize(_obj, SPACE, budget=8)
    assert [t.design for t in o1] == [t.design for t in o2]


def test_samples_within_bounds_and_sorted():
    trials = RandomSearchOptimizer(seed=3).optimize(_obj, SPACE, budget=20)
    assert len(trials) == 20
    for t in trials:
        assert 0.0 <= t.design["a"] <= 10.0
        assert 1e2 <= t.design["b"] <= 1e6      # log knob stays in range
    pces = [t.pce for t in trials]
    assert pces == sorted(pces, reverse=True)    # sorted by PCE desc


def test_budget_must_be_positive():
    with pytest.raises(ValueError):
        RandomSearchOptimizer().optimize(_obj, SPACE, budget=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_search_optimizer.py`
Expected: FAIL — `ImportError: cannot import name 'RandomSearchOptimizer'`.

- [ ] **Step 3: Add `Optimizer` protocol + `RandomSearchOptimizer` to `search.py`**

```python
class Optimizer(Protocol):
    def optimize(self, objective: Callable[[dict], tuple], space: list, budget: int) -> tuple: ...


class RandomSearchOptimizer:
    """Seeded uniform random search over the design space (no dependency)."""

    def __init__(self, *, seed: int = 0):
        self.seed = seed

    def _sample(self, rng: random.Random, space: list) -> dict:
        out = {}
        for k in space:
            r = rng.random()
            if k.scale == "log":
                out[k.axis] = 10.0 ** (math.log10(k.low) + r * (math.log10(k.high) - math.log10(k.low)))
            else:
                out[k.axis] = k.low + r * (k.high - k.low)
        return out

    def optimize(self, objective, space, budget) -> tuple:
        if budget <= 0:
            raise ValueError(f"budget must be > 0, got {budget}")
        rng = random.Random(self.seed)
        trials = []
        for _ in range(budget):
            design = self._sample(rng, space)
            pce, bracketed = objective(design)
            trials.append(Trial(design=design, pce=pce, bracketed=bracketed))
        return tuple(sorted(trials, key=lambda t: t.pce, reverse=True))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_search_optimizer.py`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add perovskite-sim/perovskite_sim/autoloop/search.py perovskite-sim/tests/unit/autoloop/test_search_optimizer.py
git commit -m "feat(autoloop): add RandomSearchOptimizer (seeded, no-dep, pluggable seam)"
```

---

## Task 3: make_design_objective

**Files:**
- Modify: `perovskite_sim/autoloop/search.py`
- Test: `tests/unit/autoloop/test_search_objective.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/autoloop/test_search_objective.py
from perovskite_sim.autoloop import search


class _Metrics:
    def __init__(self, pce, bracketed):
        self.PCE = pce
        self.voc_bracketed = bracketed


class _Result:
    def __init__(self, m):
        self.metrics_fwd = m


def test_objective_returns_pce_when_bracketed(monkeypatch, tmp_path):
    monkeypatch.setattr(search, "load_scaps_yaml", lambda p: "BASE_STACK")
    monkeypatch.setattr(search, "apply_sweep_point", lambda base, sp: "SWEPT")
    monkeypatch.setattr(search, "run_jv_sweep", lambda stack, **kw: _Result(_Metrics(0.27, True)))
    obj = search.make_design_objective(tmp_path / "c.yaml", {"N_grid": 30})
    pce, bracketed = obj({"etl_doping_cm3": 1e17})
    assert pce == 0.27 and bracketed is True


def test_objective_zero_when_unbracketed(monkeypatch, tmp_path):
    monkeypatch.setattr(search, "load_scaps_yaml", lambda p: "BASE")
    monkeypatch.setattr(search, "apply_sweep_point", lambda base, sp: "SWEPT")
    monkeypatch.setattr(search, "run_jv_sweep", lambda stack, **kw: _Result(_Metrics(0.0, False)))
    obj = search.make_design_objective(tmp_path / "c.yaml", {})
    assert obj({"x": 1.0}) == (0.0, False)


def test_objective_zero_and_logged_on_exception(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(search, "load_scaps_yaml", lambda p: "BASE")
    def _boom(base, sp): raise RuntimeError("solver diverged")
    monkeypatch.setattr(search, "apply_sweep_point", _boom)
    obj = search.make_design_objective(tmp_path / "c.yaml", {})
    import logging
    with caplog.at_level(logging.WARNING):
        pce, bracketed = obj({"x": 1.0})
    assert pce == 0.0 and bracketed is False
    assert "design eval failed" in caplog.text          # logged, not swallowed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_search_objective.py`
Expected: FAIL — `AttributeError: module ... has no attribute 'make_design_objective'`.

- [ ] **Step 3: Add `make_design_objective` to `search.py`**

```python
def make_design_objective(config_path, jv_kwargs: dict):
    """Returns objective(design: dict) -> (pce, bracketed). Applies the design to
    the base stack and runs a J-V; a failed/unbracketed design scores PCE=0."""
    base = load_scaps_yaml(config_path)

    def objective(design: dict) -> tuple:
        try:
            stack = apply_sweep_point(base, SweepPoint("design", "multi", "design", dict(design)))
            m = run_jv_sweep(stack, **jv_kwargs).metrics_fwd
            return (m.PCE if m.voc_bracketed else 0.0, bool(m.voc_bracketed))
        except Exception as exc:                       # logged, not swallowed
            logger.warning("design eval failed %s: %r", design, exc)
            return (0.0, False)

    return objective
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_search_objective.py`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add perovskite-sim/perovskite_sim/autoloop/search.py perovskite-sim/tests/unit/autoloop/test_search_objective.py
git commit -m "feat(autoloop): add make_design_objective (apply design -> run_jv_sweep -> PCE)"
```

---

## Task 4: run_design_search + parity gate + report

**Files:**
- Modify: `perovskite_sim/autoloop/search.py`
- Test: `tests/unit/autoloop/test_search_orchestration.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/autoloop/test_search_orchestration.py
import json
import pytest
from perovskite_sim.autoloop.search import (
    run_design_search, DesignKnob, SearchNotTrusted, RandomSearchOptimizer,
)

SPACE = [DesignKnob("a", 0.0, 10.0, "linear")]


def test_refuses_when_not_parity_trusted(tmp_path):
    with pytest.raises(SearchNotTrusted):
        run_design_search(
            config_path=tmp_path / "c.yaml", reference_path=tmp_path / "r.json",
            outputs_root=tmp_path / "out", timestamp="t", space=SPACE, budget=5,
            parity_target=0.9, parity_fn=lambda: 0.5,           # below target
            objective=lambda d: (d["a"], True), optimizer=RandomSearchOptimizer(seed=1))


def test_runs_when_trusted_and_writes_report(tmp_path):
    result = run_design_search(
        config_path=tmp_path / "c.yaml", reference_path=tmp_path / "r.json",
        outputs_root=tmp_path / "out", timestamp="2026-06-16T00:00:00Z",
        space=SPACE, budget=12, parity_target=0.9,
        parity_fn=lambda: 0.95,                                  # trusted
        objective=lambda d: (d["a"], True), optimizer=RandomSearchOptimizer(seed=1))
    assert result.n_evaluated == 12
    assert result.parity_overall == 0.95
    assert result.best.pce == max(t.pce for t in result.trials)  # best = top PCE
    # advisory report written, nothing applied
    report = tmp_path / "out" / "search-2026-06-16T00:00:00Z" / "result.json"
    assert report.exists()
    data = json.loads(report.read_text())
    assert data["budget"] == 12 and data["n_evaluated"] == 12


def test_all_unbracketed_does_not_crash(tmp_path):
    result = run_design_search(
        config_path=tmp_path / "c.yaml", reference_path=tmp_path / "r.json",
        outputs_root=tmp_path / "out", timestamp="t", space=SPACE, budget=4,
        parity_target=0.5, parity_fn=lambda: 0.9,
        objective=lambda d: (0.0, False), optimizer=RandomSearchOptimizer(seed=1))
    assert result.best.pce == 0.0
    assert all(not t.bracketed for t in result.trials)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_search_orchestration.py`
Expected: FAIL — `ImportError: cannot import name 'run_design_search'`.

- [ ] **Step 3: Add `_default_parity` + `run_design_search` to `search.py`**

```python
import dataclasses
import json


def _default_parity(config_path, reference_path) -> Callable[[], float]:
    def fn() -> float:
        run_point, base_point = build_run_callables(config_path)
        return score_parity(reference_path=reference_path, config_path=config_path,
                            run_point=run_point, base_point=base_point).overall
    return fn


def run_design_search(*, config_path, reference_path, outputs_root, timestamp,
                      space=None, budget: int = 50, parity_target: float = 0.90,
                      optimizer=None, objective=None, parity_fn=None) -> SearchResult:
    """Parity-gated, advisory design search. Refuses unless parity >= target,
    then runs the optimizer and writes an advisory report. Applies nothing."""
    space = space if space is not None else DEFAULT_DESIGN_SPACE
    overall = (parity_fn or _default_parity(config_path, reference_path))()
    if overall < parity_target:
        raise SearchNotTrusted(
            f"model parity {overall:.3f} < target {parity_target} — "
            "refuse to optimize an untrusted model")

    optimizer = optimizer or RandomSearchOptimizer()
    objective = objective or make_design_objective(config_path, DEFAULT_JV_KWARGS)
    trials = optimizer.optimize(objective, space, budget)
    result = SearchResult(best=(trials[0] if trials else None), trials=trials,
                          n_evaluated=len(trials), parity_overall=overall, budget=budget)

    run_dir = Path(outputs_root) / f"search-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "parity_overall": overall, "parity_target": parity_target, "budget": budget,
        "n_evaluated": result.n_evaluated,
        "best": (dataclasses.asdict(result.best) if result.best else None),
        "trials": [dataclasses.asdict(t) for t in trials],
        "note": "ADVISORY — proposed designs, nothing applied to any config",
    }
    (run_dir / "result.json").write_text(json.dumps(payload, indent=2, sort_keys=True),
                                         encoding="utf-8")
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_search_orchestration.py`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add perovskite-sim/perovskite_sim/autoloop/search.py perovskite-sim/tests/unit/autoloop/test_search_orchestration.py
git commit -m "feat(autoloop): add run_design_search (parity gate + advisory report)"
```

---

## Task 5: CLI `--search` + integration smoke + docs

**Files:**
- Modify: `scripts/autoloop_run.py`
- Create: `tests/integration/test_autoloop_search.py`
- Modify: `perovskite-sim/CLAUDE.md`, `README.md`

- [ ] **Step 1: Write the failing integration test (slow)**

```python
# tests/integration/test_autoloop_search.py
import pytest
from pathlib import Path
from perovskite_sim.autoloop.search import run_design_search, DesignKnob

REPO_ROOT = Path(__file__).resolve().parents[1]
REF = REPO_ROOT / "tests" / "integration" / "scaps_reference.json"
CFG = REPO_ROOT / "configs" / "scaps_mirror_v2.yaml"


@pytest.mark.slow
def test_real_search_low_parity_gate(tmp_path):
    # Low parity-target so the gate passes (real parity ~0.69); tiny budget.
    cfg_before = CFG.read_text(encoding="utf-8")
    result = run_design_search(
        config_path=CFG, reference_path=REF, outputs_root=tmp_path / "out",
        timestamp="2026-06-16T00:00:00Z",
        space=[DesignKnob("etl_doping_cm3", 1e15, 1e19, "log")],
        budget=3, parity_target=0.5)
    assert result.n_evaluated == 3
    assert result.best is not None
    assert result.parity_overall >= 0.5             # gate passed
    assert CFG.read_text(encoding="utf-8") == cfg_before    # advisory — nothing applied
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd perovskite-sim && python -m pytest -q -m slow tests/integration/test_autoloop_search.py`
Expected: FAIL until the module is complete; once green it confirms the real search runs 3 sweeps behind the parity gate and applies nothing. Runtime: a few minutes (3 real sweeps + one parity score).

- [ ] **Step 3: Wire the CLI + docs**

In `scripts/autoloop_run.py`, add flags to `parse_args`:

```python
    ap.add_argument("--search", action="store_true",
                    help="run a parity-gated, advisory device-design search")
    ap.add_argument("--budget", type=int, default=50, help="design-search eval budget")
```
(`--parity-target` already exists from Stage 4a.)

Add the dispatch in `main` (before the `--boulder` block):

```python
    if ns.search:
        import dataclasses
        from perovskite_sim.autoloop.search import run_design_search, SearchNotTrusted
        try:
            result = run_design_search(
                config_path=ns.config, reference_path=ns.reference,
                outputs_root=ns.outputs_root, timestamp=iso_timestamp_utc(),
                budget=ns.budget, parity_target=ns.parity_target)
        except SearchNotTrusted as exc:
            print(json.dumps({"search": None, "error": str(exc)}))
            return 1
        print(json.dumps({"search": {
            "parity_overall": result.parity_overall, "n_evaluated": result.n_evaluated,
            "best": (dataclasses.asdict(result.best) if result.best else None),
            "top": [dataclasses.asdict(t) for t in result.trials[:5]]}},
            indent=2, sort_keys=True, default=str))
        return 0
```

Add to the **Autoloop** section of `perovskite-sim/CLAUDE.md`:

```markdown
### Stage 4c — L4 design-search (advisory)

`autoloop/search.py` searches the device-design space (band offsets / doping /
defects via `apply_sweep_point`) to maximize PCE, via a pluggable `Optimizer` seam
(ships a no-dep seeded `RandomSearchOptimizer`; optuna/pymoo adapters plug in later).
`run_design_search` REFUSES unless `score_parity().overall >= parity_target` (don't
optimize an untrusted model), then reports the best design + top trials **advisorily**
— nothing is applied to any config. Each design eval = a real J-V sweep, so use a
modest `--budget`.

    cd perovskite-sim
    python scripts/autoloop_run.py --search --budget 50 --parity-target 0.9

Note: the model's current parity (~0.69) is below 0.9, so search refuses by default
until the boulder/converge loop improves parity — the correct trust coupling.
```

Add to `README.md` (next to the other autoloop lines):

```markdown
- **Autoloop design-search** (`python perovskite-sim/scripts/autoloop_run.py --search`) —
  parity-gated, advisory search of the device-design space for max-PCE designs (no-dep
  random search; pluggable for optuna later). Reports designs; applies nothing.
```

- [ ] **Step 4: Run tests**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop && python -m pytest -q -m slow tests/integration/test_autoloop_search.py`
Expected: all green. Also `python -m pytest -q` (full default suite) — confirm no import/collection regression.

- [ ] **Step 5: Commit**

```bash
git add perovskite-sim/scripts/autoloop_run.py perovskite-sim/tests/integration/test_autoloop_search.py perovskite-sim/CLAUDE.md README.md
git commit -m "feat(autoloop): wire --search CLI + integration smoke + docs (Stage 4c)"
```

---

## Self-Review

**Spec coverage** (vs `2026-06-16-autoloop-stage4c-design-search-design.md`):
- §3 types (`DesignKnob`/`Trial`/`SearchResult`/`SearchNotTrusted`) + `DEFAULT_DESIGN_SPACE` → Task 1. ✓
- §4 `Optimizer` protocol + `RandomSearchOptimizer` (seeded, linear/log, sorted, budget guard) → Task 2. ✓
- §4 `make_design_objective` (apply_sweep_point → run_jv_sweep → PCE; fail→0+log) → Task 3. ✓
- §5 `run_design_search` + parity gate (`SearchNotTrusted`) + advisory report → Task 4. ✓
- §6 CLI `--search [--budget] [--parity-target]` → Task 5. ✓
- §7 error handling (not-trusted→raise/exit 1; eval fail→0; budget≤0→raise; all-unbracketed→no crash) → Tasks 2/3/4. ✓
- §8 testing (frozen types, seeded determinism, monkeypatched objective, gate-refuse, advisory report, slow smoke with low parity-target) → every task. ✓
- §9 deferred (optuna/pymoo, multi-objective, LHS, auto-apply, non-box) → correctly NOT built.

**Placeholder scan:** none — complete code/tests/commands. The smoke's `--parity-target 0.5` is explained (gate passes at the model's ~0.69 to exercise the search), not a placeholder.

**Type consistency:** `DesignKnob(axis, low, high, scale)`, `Trial(design, pce, bracketed)`, `SearchResult(best, trials, n_evaluated, parity_overall, budget)` defined Task 1, used identically Tasks 2/4 + tests. `RandomSearchOptimizer(seed=).optimize(objective, space, budget) -> tuple[Trial]` consistent Tasks 2/4 + tests. `make_design_objective(config_path, jv_kwargs) -> objective(design)->(pce,bracketed)` consistent Tasks 3/4. `run_design_search(... parity_fn, objective, optimizer)` injection matches the Task 4 tests. `apply_sweep_point`/`run_jv_sweep`/`load_scaps_yaml`/`build_run_callables`/`score_parity`/`DEFAULT_JV_KWARGS` are verified real symbols.

---

## Execution Handoff

After saving, choose execution:

1. **Subagent-Driven (recommended)** — fresh subagent per task, two-stage review (same background-workflow as Stages 1–4b).
2. **Inline Execution** — batch tasks in this session with checkpoints.
