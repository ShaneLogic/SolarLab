# Autoloop Stage 4c — L4 Design-Search — Design

**Date:** 2026-06-16
**Status:** Design approved, pre-planning
**Parent design:** `docs/superpowers/specs/2026-06-16-autoloop-research-pipeline-design.md` (L4 design-search)
**Builds on:** Stages 1–4b (all merged to `main`). Reuses `sweeps.device_parameter_sweep.SweepPoint`/`apply_sweep_point`, `experiments.jv_sweep.run_jv_sweep`/`compute_metrics`, `scaps_compat.load_scaps_yaml`, `autoloop.scorecard.score_parity`, `autoloop.ladder.build_run_callables`/`DEFAULT_JV_KWARGS`.

**Scope note:** the last Stage-4 sub-project (4a boulder + 4b L3 seam done). 4c = the optimization leg.

---

## 1. Problem & scope

Stages 1–4b make the model *trustworthy*; 4c *uses* it — searches the device-design
space for designs that maximize PCE, **gated on the model being parity-trusted**, and
reports them **advisorily** (designs are a human call; never auto-applied).

**Decisions (locked during brainstorming):**

| Axis | Decision |
|------|----------|
| Optimizer | **Pluggable `Optimizer` seam + a no-dependency seeded random-search default.** optuna/pymoo adapters plug in later behind the same interface. (Matches the ship-deterministic-core / pluggable-backend pattern of Stages 2/4b. optuna/pymoo are NOT installed; scipy is.) |
| Objective | **PCE (single).** Multi-objective/Pareto deferred (pymoo). |
| Trust gate | Search **refuses unless `score_parity(...).overall ≥ parity_target`** — don't optimize an untrusted model. |
| Output | **Advisory** — best design + top trials reported; nothing applied to any config. |

**Explicitly deferred:** optuna/pymoo adapters; multi-objective/Pareto; Latin-hypercube
stratification; auto-applying a design; non-box constraints.

## 2. Flow

```
load base stack (config)
        │
        ▼
  PARITY GATE: score_parity(...).overall ≥ parity_target ?
        │  no → raise SearchNotTrusted ("won't optimize an untrusted model")
        ▼ yes
  optimizer.optimize(objective=evaluate_design, space=DesignSpace, budget=N) → trials
        │
        ▼
  ADVISORY report: best design + top-N trials + the gate parity score
  (json + md to outputs/autoloop/search-<ts>/; NOTHING applied)
```

**Coupling (important):** the model's current parity (~0.69 from 4b validation) is below
a 0.9 trust bar → design-search **refuses by default** until the loop (boulder/converge)
improves parity. This is the correct dependency — optimization rides on top of trust.

## 3. Module layout + types

```
perovskite_sim/autoloop/
  search.py   DesignKnob, Trial, SearchResult, DesignSpace, DEFAULT_DESIGN_SPACE,
              Optimizer protocol, RandomSearchOptimizer, make_design_objective,
              run_design_search, SearchNotTrusted
scripts/autoloop_run.py  + --search [--budget] [--parity-target]
```

```python
@dataclass(frozen=True)
class DesignKnob:
    axis: str                # apply_sweep_point axis, e.g. "etl_doping_cm3"
    low: float
    high: float
    scale: str = "linear"    # "linear" | "log"

@dataclass(frozen=True)
class Trial:
    design: dict             # {axis: value}
    pce: float               # 0.0 if the design didn't bracket V_oc / didn't converge
    bracketed: bool

@dataclass(frozen=True)
class SearchResult:
    best: Optional[Trial]
    trials: tuple            # all, sorted by pce desc
    n_evaluated: int
    parity_overall: float    # parity at gate time
    budget: int

class SearchNotTrusted(RuntimeError):
    pass
```

`DesignSpace = list[DesignKnob]`.

`DEFAULT_DESIGN_SPACE` (a starting box over the real `apply_sweep_point` axes; user tunes):
```python
DEFAULT_DESIGN_SPACE = [
    DesignKnob("etl_delta_ec_eV", -0.3, 0.3, "linear"),
    DesignKnob("htl_delta_ev_eV", -0.3, 0.3, "linear"),
    DesignKnob("etl_doping_cm3", 1e15, 1e19, "log"),
    DesignKnob("absorber_defect_density_cm3", 1e14, 1e17, "log"),
]
```

## 4. Optimizer (pluggable) + objective

**`Optimizer` protocol:** `optimize(objective, space, budget) -> tuple[Trial, ...]`.

**`RandomSearchOptimizer(seed=0)`** — seeded → deterministic + testable:
```python
def optimize(self, objective, space, budget):
    if budget <= 0:
        raise ValueError("budget must be > 0")
    rng = random.Random(self.seed)
    trials = [Trial(d, *objective(d))
              for d in (self._sample(rng, space) for _ in range(budget))]
    return tuple(sorted(trials, key=lambda t: t.pce, reverse=True))

def _sample(self, rng, space):
    out = {}
    for k in space:
        r = rng.random()
        if k.scale == "log":
            out[k.axis] = 10.0 ** (math.log10(k.low) + r * (math.log10(k.high) - math.log10(k.low)))
        else:
            out[k.axis] = k.low + r * (k.high - k.low)
    return out
```
(Plain uniform; Latin-hypercube stratification = a later nicety, noted.)

**The real objective (injected — optimizer stays solver-free in tests):**
```python
def make_design_objective(config_path, jv_kwargs):
    base = load_scaps_yaml(config_path)
    def objective(design: dict) -> tuple[float, bool]:
        try:
            stack = apply_sweep_point(base, SweepPoint("design", "multi", "design", design))
            m = run_jv_sweep(stack, **jv_kwargs).metrics_fwd
            return (m.PCE if m.voc_bracketed else 0.0, m.voc_bracketed)
        except Exception as exc:                  # logged, not swallowed
            logger.warning("design eval failed %s: %r", design, exc)
            return (0.0, False)
    return objective
```
`apply_sweep_point` accepts a multi-key updates dict (the existing coupled sweeps prove it). A failed/unbracketed design scores PCE=0 — never crashes the search.

## 5. Orchestration

```python
def run_design_search(*, config_path, reference_path, outputs_root, timestamp,
                      space=DEFAULT_DESIGN_SPACE, budget=50, parity_target=0.90,
                      optimizer=None, objective=None, parity_fn=None) -> SearchResult:
    overall = (parity_fn or _default_parity(config_path, reference_path))()
    if overall < parity_target:
        raise SearchNotTrusted(f"parity {overall:.3f} < target {parity_target} — "
                               "refuse to optimize an untrusted model")
    optimizer = optimizer or RandomSearchOptimizer()
    objective = objective or make_design_objective(config_path, DEFAULT_JV_KWARGS)
    trials = optimizer.optimize(objective, space, budget)
    result = SearchResult(best=(trials[0] if trials else None), trials=trials,
                          n_evaluated=len(trials), parity_overall=overall, budget=budget)
    # write advisory report (json + md) to outputs/autoloop/search-<ts>/ ; apply nothing
    return result
```
`_default_parity` scores via `build_run_callables` + `score_parity` against `reference_path`
(SCAPS default, or a 4b tiered descriptor). `parity_fn`/`objective`/`optimizer` are injected
so the orchestration is unit-testable without the solver.

## 6. CLI

```bash
python scripts/autoloop_run.py --search [--budget 50] [--parity-target 0.9] \
    [--reference tests/integration/scaps_lab_tiered.json]
```
Prints the best design + top trials + the gate parity. **Advisory** — nothing applied.
Exit 0 on success; 1 if not parity-trusted (`SearchNotTrusted`).

## 7. Error handling

- Not parity-trusted → `SearchNotTrusted`, CLI prints the refusal, exit 1.
- A design eval fails (non-converge / exception) → PCE=0 + logged; search continues.
- `budget ≤ 0` → `ValueError`.
- All designs unbracketed → `best` is a PCE=0 trial; the report flags "no design bracketed V_oc — widen the space / check the config". No crash.

## 8. Testing

- Frozen types (`DesignKnob`/`Trial`/`SearchResult`).
- `RandomSearchOptimizer`: same seed → identical trials (determinism); linear + log samples within `[low, high]`; trials sorted by PCE desc; `budget` respected; `budget≤0` raises.
- `make_design_objective`: monkeypatch `run_jv_sweep` → returns `(PCE, bracketed)`; a raising run → `(0.0, False)` + logged.
- `run_design_search` with injected fakes: parity-gate **raises `SearchNotTrusted`** when `overall < target`; trusted → sorted trials + `best` = top PCE + advisory report written; all-unbracketed → no crash, flagged.
- integration smoke (slow): real `--search --budget 3 --parity-target 0.5` on `scaps_mirror_v2` → gate passes (0.69 > 0.5), 3 real sweeps, ≥1 trial returned, **no config mutated**. Bounded.

## 9. Out of scope / deferred

- optuna / pymoo `Optimizer` adapters (seam ready).
- Multi-objective / Pareto (single PCE now).
- Latin-hypercube stratification.
- Auto-applying a searched design (advisory only — a human call).
- Non-box constraints (only `[low, high]` per knob).

## 10. Build order (staged tasks for writing-plans)

1. `search.py` types — `DesignKnob`, `Trial`, `SearchResult`, `SearchNotTrusted`, `DesignSpace`, `DEFAULT_DESIGN_SPACE`.
2. `RandomSearchOptimizer` (seeded; linear/log sampling; sorted; budget guard) (+ tests).
3. `make_design_objective` (apply_sweep_point → run_jv_sweep → PCE; fail→0) (+ tests).
4. `run_design_search` + parity gate + advisory report (+ injected-fake tests).
5. CLI `--search` + integration smoke + docs (README / CLAUDE.md).
