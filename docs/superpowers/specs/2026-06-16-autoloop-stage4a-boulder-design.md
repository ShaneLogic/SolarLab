# Autoloop Stage 4a — Boulder Driver — Design

**Date:** 2026-06-16
**Status:** Design approved, pre-planning
**Parent design:** `docs/superpowers/specs/2026-06-16-autoloop-research-pipeline-design.md` (§12 stage 4 — boulder)
**Builds on:** Stages 1–3 (guardian + attribution + auto-implement), all merged to `main`. Reuses `guardian_once`, `attribute_top_gap`, `implement_top_confirmed`, the gap/hypothesis ledger, and the G0–G4 gate stack.

**Scope note:** Stage 4 splits into three independent sub-projects — **4a boulder driver** (this spec), **4b L3 real-lab-data ingest seam**, **4c L4 design-search**. 4a is the continuous driver; 4b/4c are later, separate specs.

---

## 1. Problem & scope

Stages 1–3 already chain manually (`--once → --attribute → --implement`). 4a makes
that chain **self-driving**: a loop that senses, diagnoses, and proposes/lands fixes
across gaps until a stop condition — turning the three CLIs into one orchestrated
boulder.

**Decisions (locked during brainstorming):**

| Axis | Decision |
|------|----------|
| Mode | **Sweep (dry-run) default; `--converge` opt-in.** Sweep drains current gaps into a batch proposal report (no commits). Converge auto-applies landable fixes, re-senses the evolved config, and loops until a stop condition. |
| Implementation | In-process Python loop over the existing orchestrator functions. **No subprocess/CLI shelling.** |
| Advancement | **Gap status transitions.** Processed-but-not-landed gaps → new `attempted` status so the top-gap pickers move on. |
| Landing (converge) | Reuses Stage 3 `commit_promotion` → commits to the **current branch** (refuses main/dirty). Boulder refuses `--converge` on main early. Never pushes. |

**Explicitly deferred:** auto-push/PR; L3 (4b); L4 (4c); scheduling/cron (an external
scheduler invokes `--boulder`); parallel cycles.

## 2. The two modes

```
SWEEP (default, dry-run):                  CONVERGE (--converge, the real loop):
 guardian_once (sense, refresh gaps)        repeat:
 while an `open` gap remains:                 guardian_once (status-preserving re-sense)
   attribute_top_gap (diagnose)               if overall ≥ parity_target: STOP success
   implement_top_confirmed(apply=False)       gap = top open; if none: STOP drained
   record proposal; gap → attempted           attribute → implement(apply=True)
 emit batch report (no commits)               if applied: landed++, reject=0
                                              else: gap → attempted; if gates_failed: reject++
                                              if reject ≥ reject_streak: STOP halt
                                              if cycles ≥ max_cycles: STOP cap
```

- **Sweep** = one drain of current gaps → a report of {gap, diagnosis, proposed fix, dry-run gate verdicts}. Bounded by gap count. Zero commits.
- **Converge** = re-sense after each landed fix so the config *evolves*; loop until a stop condition. Commits accumulate on the current branch.

## 3. Advancement + the termination-critical re-sense

All three orchestrator functions pick the **top `open` gap**. The boulder advances by
**status transitions**, processing strictly top-down:

```
gap = top open gap                           # boulder reads ledger; none → drained/complete
attribute_top_gap(...)                        # picks G (top open), writes its Hypothesis
result = implement_top_confirmed(apply=converge)   # picks G iff confirmed; else no_confirmed
record BoulderProposal(G, hyp, result)
if result.status == "applied": G already → closed   # converge only; config evolved
else:                          G → attempted        # dry-run / uncertain / not-promotable / gates-failed
```

Because the boulder transitions each gap before the next iteration, the internal
"top open" pick always lands on the intended gap — no refactor of
`attribute_top_gap`/`implement_top_confirmed` to accept an explicit gap.

**Termination-critical detail (converge re-sense):** converge re-runs `guardian_once`
each cycle on the evolved config. But `guardian_once` currently does `add_gap`
(replace-by-id), so a re-sense would **resurrect an `attempted`/`refuted` gap back to
`open`** → infinite loop on an unfixable gap. **Fix:** `guardian_once` gains a
**status-preserving merge** — if a gap id already exists with a non-`open` status
(`attempted`/`blocked`/`refuted`/`closed`), keep that status; only genuinely-new gaps
enter as `open`. This is what lets converge drain and terminate.

## 4. Stop conditions (converge)

Checked each cycle; all tunable, sensible defaults:

| Stop | Condition | Default |
|------|-----------|---------|
| **success** | `score.overall ≥ parity_target` | 0.90 |
| **drained** | no `open` gap remains | — |
| **cap** | cycle count ≥ `max_cycles` | 10 |
| **halt** | consecutive `gates_failed` ≥ `reject_streak` | 3 |

A landed fix resets the reject counter (progress). Sweep has no stop conditions — one
bounded drain.

## 5. Module layout + types

```
perovskite_sim/autoloop/
  orchestrator.py   + run_boulder(...)
  types.py          + BoulderProposal, BoulderResult, "attempted" as a valid Gap.status
  scorecard.py / orchestrator guardian_once  → status-preserving gap merge
scripts/autoloop_run.py  + --boulder [--converge] [--parity-target] [--max-cycles] [--reject-streak]
```

```python
@dataclass(frozen=True)
class BoulderProposal:
    gap_id: str
    cause: str
    mechanism: str
    device_key: Optional[str]
    gate_status: str          # implement status: dry_run|applied|gates_failed|not_promotable|no_confirmed
    landed: bool
    note: str = ""

@dataclass(frozen=True)
class BoulderResult:
    mode: str                 # "sweep" | "converge"
    cycles: int
    proposals: tuple          # of BoulderProposal
    landed_count: int
    stop_reason: str          # "sweep_complete"|"success"|"drained"|"cap"|"halt"
    final_overall: Optional[float]
```

`run_boulder` signature (heavy fns injected so the loop is unit-testable without the
solver):
```python
def run_boulder(*, ledger_root, outputs_root, config_path, reference_path, timestamp,
                converge=False, parity_target=0.90, max_cycles=10, reject_streak=3,
                guardian_fn=guardian_once, attribute_fn=attribute_top_gap,
                implement_fn=implement_top_confirmed, git_cwd=None) -> BoulderResult
```

## 6. CLI + landing

```bash
python scripts/autoloop_run.py --boulder              # sweep: drain → batch report, no commits
python scripts/autoloop_run.py --boulder --converge   # auto-apply + loop until stop (current branch ≠ main)
   [--parity-target 0.9] [--max-cycles 10] [--reject-streak 3]
```

- **Converge refuses to start on `main`/`master`** (early guard, before any work) — create an `autoloop/<date>` branch first. Each landed fix = one `commit_promotion` commit on the current branch. Never pushes.
- Exit 0 on sweep / converge-success / drained; 1 on halt or the converge-on-main guard.
- Per-cycle artifacts → `outputs/autoloop/boulder-<run>/cycle-N/`; final `BoulderResult` → JSON + a markdown report.

## 7. Error handling

- `--converge` on main/dirty → refuse before any work, exit 1.
- Parity ≥ target on first sense → sweep: empty report; converge: stop `success` at cycle 0.
- An unexpected exception inside a cycle → caught; halt + emit the partial `BoulderResult` (never blind-continue past an unknown error).
- Long-running (~30–60 min/cycle real solver); prints per-cycle progress.

## 8. Testing

- **`run_boulder` with injected fakes** (`guardian_fn`/`attribute_fn`/`implement_fn`) — loop logic, zero solver:
  - sweep: a sequence of gaps → drains all, proposals recorded, gaps → `attempted`, no commits.
  - converge: stops on each of `success` (rising overall), `drained` (all attempted), `cap` (never improves to max_cycles), `halt` (repeated gates_failed).
  - converge-on-main refused.
- **status-preserving merge** in `guardian_once` — unit test: re-adding a gap whose id exists with `attempted` keeps it attempted (the termination guarantee); a brand-new gap enters `open`.
- **`BoulderProposal`/`BoulderResult`** frozen + serialization.
- integration smoke (slow): real **sweep** on `scaps_mirror_v2` → one drain → report contains the `trend:Nd_ETL:V_oc → interface_plane_projection` proposal (validation confirmed this lever). Converge real run is hours → manual-only, NOT a CI smoke (documented).

## 9. Out of scope / deferred

- Converge auto-push / PR creation (human reviews the branch).
- 4b (L3 real-lab-data seam) + 4c (L4 design-search) — separate specs.
- Scheduling/cron (an external scheduler invokes `--boulder`).
- Parallel cycles (cycles are sequential; the solver is the bottleneck).

## 10. Build order (staged tasks for writing-plans)

1. `types.py` — `BoulderProposal`, `BoulderResult` (+ document `attempted` status).
2. `guardian_once` status-preserving gap merge (+ test).
3. `run_boulder` sweep mode (+ injected-fake tests).
4. `run_boulder` converge mode + stop conditions + main-guard (+ tests).
5. CLI `--boulder [--converge ...]` + sweep integration smoke + docs (README / CLAUDE.md).
