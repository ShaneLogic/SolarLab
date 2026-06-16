# Autoloop Stage 2 — Attribution Leg (deterministic) — Design

**Date:** 2026-06-16
**Status:** Design approved, pre-planning
**Parent design:** `docs/superpowers/specs/2026-06-16-autoloop-research-pipeline-design.md` (§8 attribute-gap, §12 stage 2)
**Builds on:** Stage 1 guardian (merged to `main`): `perovskite_sim/autoloop/` spine — `Gap`/`Hypothesis`/`Ledger`/`ParityScore`/`build_run_callables`/`SHEET_TO_AXIS`.

---

## 1. Problem & scope

Stage 1 senses discrepancies and ranks them into a gap ledger, but cannot say
*why* a gap exists. Stage 2 adds the **first cognition leg — built
deterministic** — that attributes a gap to a cause (bug / numerics / physics /
data) by auto-ablating the `SOLARLAB_*` physics flags plus limiting-case and
grid-convergence probes, then classifying the evidence with a heuristic.

**Scope decisions (locked during brainstorming):**

| Axis | Decision |
|------|----------|
| Cognition seam | **Pluggable `Attributor` interface.** The reasoning backend is swappable; Stage 2 ships a deterministic adapter, the LLM adapter plugs in later behind the same interface. |
| Backend shipped | **Deterministic `HeuristicAttributor`** (rule-based decision tree over the ablation matrix) + a test fake. No LLM dependency this increment. |
| Side effects | **Read-only re: code.** Produces a `Hypothesis` written to the ledger + run artifacts. Changes no solver/physics code (that is Stage 3). |

**Explicitly deferred (honest):** the LLM `Attributor` adapter and multi-skeptic
adversarial verify are the next increment (Stage 2.5 / fold into Stage 3). Stage 2
ships the ablation harness + the seam + the heuristic.

## 2. Flow

```
top open gap (gap ledger; not refuted, not blocked)
        │
        ▼
  ABLATION HARNESS  ── deterministic ──▶  AblationMatrix
   (toggle SOLARLAB_* flags · grid 20→80 ·    (probe → baseline/variant/delta)
    limiting cases)  via injected ProbeRunner
        │
        ▼
   ATTRIBUTOR (pluggable seam)
   ├─ HeuristicAttributor   (ships now: rules over the matrix)
   └─ LLMAttributor         (later, same interface)
        │
        ▼
   Hypothesis{cause, mechanism, evidence_for/against, verdict, confidence}
        │
        ▼
   ledger.add_hypothesis  +  gap.mechanism if confirmed   (read-only re: code)
```

**Two injected seams keep Stage 2 fully unit-testable at zero LLM/solver cost**
(mirrors Stage 1's injected `run_point`/`base_point`):

- **`ProbeRunner`** — runs one ablation variant. Real = subprocess with env flag
  set; fake = canned. Subprocess is required because some `SOLARLAB_*` flags are
  read at module import (e.g. `_QSS_V_TH_MS` in `solver/mol.py`) and the
  `MaterialArrays` cache is built once — an in-process env toggle would not take.
- **`Attributor`** — reads the matrix → `Hypothesis`. Real = heuristic; fake = canned.

## 3. Module layout (extends `perovskite_sim/autoloop/`)

```
ablation.py      run_ablation(gap, ..., probe_runner) -> AblationMatrix
                 SubprocessProbeRunner ; CANDIDATE_FLAGS map
attribution.py   Attributor protocol + HeuristicAttributor (reuses types.Hypothesis)
_probe_worker.py CLI worker: set env -> load config -> run sweep -> print {"metric": ...}
types.py         + AblationProbe, AblationMatrix  (Hypothesis already exists)
orchestrator.py  + attribute_top_gap(...) : pick gap -> ablate -> attribute -> record
scripts/autoloop_run.py  + --attribute (one attribution pass on the top gap)
```

Reuses Stage 1 unchanged: `Gap`, `Hypothesis`, `Ledger`, `SHEET_TO_AXIS`,
`build_run_callables`, `provenance.stamp`. No Stage 1 rewrites.

## 4. The ablation harness

`run_ablation(gap, config_path, reference_path, jv_kwargs, probe_runner) ->
AblationMatrix`. Measures one baseline, then runs probes; each probe is a
`(env_flags, jv_overrides)` variant and records how the gap's metric moved.

**Probe kinds:**

| Kind | Varies | Signal |
|------|--------|--------|
| `flag` | toggle one candidate `SOLARLAB_*` ON | gap moves toward reference → that physics term is the lever |
| `grid` | `n_points` 20 → 80 | large shift → numerics / interp artifact |
| `limiting` | rad-only V_oc vs detailed-balance ceiling; dark J=0 | violated → bug / mis-physics |

**Candidate-flag map** (data, per gap kind; start small, **log what was tried —
no silent cap**):
```python
CANDIDATE_FLAGS = {
    "interface": ["SOLARLAB_IFACE_PROJ", "SOLARLAB_IFACE_PLANE", "SOLARLAB_INTERFACE_PLANE_STATE"],
    "base":      ["SOLARLAB_DOS_BAND"],
}
# gap -> bucket: sweep in {Nt_PVK ETL, CHI_ETL, Nd_ETL} -> "interface"; sweep=="base" -> "base"
```

**The ProbeRunner seam:** `run(env_flags: dict[str,str], jv_overrides: dict) ->
float` returns the gap's metric under that variant (V_oc closure for a trend gap;
base V_oc delta for an absolute base gap — computed via the Stage 1 scorecard
helpers).

- **`SubprocessProbeRunner`** (real): `python -m perovskite_sim.autoloop._probe_worker`
  with `(config, gap spec, env flags, jv overrides)`; the worker sets env, runs the
  sweep via `build_run_callables`, computes the gap metric, prints
  `{"metric": ...}`. The harness parses stdout.
- **fake** (tests): a `dict[variant_key -> metric]` — zero solver.

A probe that fails (non-convergence, worker crash) is recorded `ok=False` + note;
the harness continues. The attributor treats a failed probe as no-signal.

**Types (added to `types.py`, frozen):**
```python
@dataclass(frozen=True)
class AblationProbe:
    name: str
    kind: str            # "flag" | "grid" | "limiting"
    variant: dict        # env_flags + jv_overrides applied
    baseline_val: float
    variant_val: float
    delta: float         # variant_val - baseline_val
    ok: bool
    note: str = ""

@dataclass(frozen=True)
class AblationMatrix:
    gap_id: str
    baseline_val: float
    probes: tuple[AblationProbe, ...]
    skipped: tuple[str, ...] = ()
```

## 5. The Attributor seam + heuristic

`Attributor` protocol: `attribute(gap: Gap, matrix: AblationMatrix, negatives) ->
Hypothesis`. Ships `HeuristicAttributor`. Thresholds are explicit constants (no
magic numbers), tunable.

**Decision logic** — ordered, first strong signal wins:

1. **Numerics** — a `grid` probe shifts the metric by more than `grid_tol` →
   `cause="numerics"`, mechanism *"grid-convergence sensitive (n_points 20→80
   shifts metric by Δ)"*.
2. **Physics** — else the largest `flag` probe that moves the gap *toward* the
   reference by > `flag_tol` → `cause="physics"`, mechanism *"flag X term"*,
   `evidence_for` = that probe.
3. **Bug** — else a `limiting` case violated (rad-only V_oc > ceiling, or
   dark J ≠ 0) → `cause="bug"`, mechanism = the violation.
4. **Uncertain** — else no dominant lever → `cause="uncertain"`,
   `verdict="uncertain"`. **Honest fallback: never invent a cause.**

**Honesty guards (attribution-level G4 spirit):**

- **Negatives-guard:** if the implied mechanism matches a refuted approach
  (`ledger.is_refuted`), it is never emitted `confirmed` → downgraded to
  `uncertain` + note. The heuristic cannot "rediscover" DOS-cap / BBD /
  shared-occupancy.
- **Dominance check:** `verdict="confirmed"` only when the winning probe clearly
  separates from the next; else `uncertain`. This *is* the heuristic's verify;
  multi-skeptic adversarial verify is the LLM adapter's job (deferred).
- `confidence` = signal separation; `evidence_against` = probes that worsened the gap.

Output: the existing `Hypothesis(gap_id, cause, mechanism, evidence_for,
evidence_against, verifier_votes, verdict, cycle)` (`verifier_votes` = count of
supporting probes for the heuristic).

## 6. Orchestrator pass + CLI

`attribute_top_gap(*, ledger_root, outputs_root, config_path, reference_path,
cycle, timestamp, probe_runner, attributor, jv_kwargs=None) -> Hypothesis | None`:

1. Load ledger → pick top open, non-refuted, non-blocked gap (max `gap_mag`).
   None → return None.
2. `run_ablation` → matrix.
3. `attributor.attribute(gap, matrix, negatives=ledger)` → Hypothesis.
4. `ledger.add_hypothesis(hyp)`; if `verdict=="confirmed"`, set `gap.mechanism`
   (gap stays `open` — Stage 3 acts on it).
5. `provenance.stamp`; artifacts → `outputs/autoloop/<run>/attribution/`
   (matrix.json + hypothesis.json); save ledger.

Read-only re: code — writes only ledger + artifacts. Adds `Gap.with_mechanism`
(via `dataclasses.replace`) to `types.py`.

**CLI:** `autoloop_run.py --attribute` runs one pass on the top gap. Exit 0 on
success (hypothesis produced, or no open gaps); 1 on error. (`--once` populates
gaps; `--attribute` diagnoses the top one.)

## 7. Error handling

- No open gaps → `None`; CLI prints "no open gaps"; exit 0.
- Probe / subprocess-worker failure → probe `ok=False` + note; harness continues
  with available signal.
- Attributor **never raises** — `uncertain` fallback on no signal.
- Refuted-match → downgraded (Section 5).
- Subprocess worker is invoked with the package root as cwd (same `_PKG_ROOT`
  fix as Stage 1's `run_l0`) so it resolves config/imports regardless of caller cwd.

## 8. Testing

- **`ablation`** — `run_ablation` with a fake ProbeRunner (canned variant→metric)
  → asserts matrix structure, deltas, baseline, and `ok=False` handling for a
  failing probe. CANDIDATE_FLAGS bucketing unit-tested.
- **`attribution`** — one test per branch (numerics / physics / bug / uncertain) +
  negatives-guard downgrade (a physics mechanism matching a seeded refuted
  approach → `uncertain`) + dominance→uncertain.
- **`orchestrator`** — `attribute_top_gap` with fakes → picks top gap, writes a
  Hypothesis, sets `gap.mechanism` on confirmed, skips refuted/blocked gaps,
  returns None when no open gaps.
- **integration smoke (slow)** — real `SubprocessProbeRunner` on one
  guardian-produced gap → asserts a Hypothesis was produced + recorded (not a
  specific cause — that is the moving target the loop tracks).

## 9. Out of scope / deferred

- LLM `Attributor` adapter + multi-skeptic adversarial verify (next increment).
- Residual-by-recombination-channel decomposition (needs solver hooks; the cheap
  flag/grid/limiting probes are sufficient signal for Stage 2).
- Any code/physics change (Stage 3 — auto-implement).
- Marking a gap `blocked` after repeated `uncertain` (kept simple: just record).

## 10. Build order (staged tasks for writing-plans)

1. `types.py` — `AblationProbe`, `AblationMatrix`, `Gap.with_mechanism`.
2. `ablation.py` — `run_ablation` + `CANDIDATE_FLAGS` + fake-driven tests.
3. `_probe_worker.py` + `SubprocessProbeRunner`.
4. `attribution.py` — `Attributor` protocol + `HeuristicAttributor` + per-branch tests.
5. `orchestrator.attribute_top_gap` + tests.
6. CLI `--attribute` + integration smoke + docs (README / CLAUDE.md in sync).
