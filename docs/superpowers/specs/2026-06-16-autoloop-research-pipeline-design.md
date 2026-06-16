# Autoloop — Continuous Autonomous Research-Loop Orchestrator

**Date:** 2026-06-16
**Status:** Design approved, pre-planning
**Scope:** A self-driving development loop for the SolarLab / `perovskite-sim` physics simulator that ties together the numerical, physics-model, experiment-calibration, and optimization concerns into one gated, continuously-running pipeline.

---

## 1. Problem

SolarLab already owns every *manual* piece of a research loop — parity runners
(`run_scaps_validation.py`, `run_scaps_full_regression.py`), golden-master
generation (`qss_golden_master.py`), scoring (`scaps_absolute_scorecard.py`),
attribution probes (`probes/`, the hand-driven `outputs/scaps_e7_probe_*`,
`kill_auger`, `cascade` runs), physics ablation dials (`SOLARLAB_*` env flags),
and design/material sweeps (`run_device_parameter_sweep.py`,
`run_material_screening.py`). What is missing is **orchestration**: there is no
CI, no optimizer dependency, and no standing loop. The 36-agent root-cause
analysis and the e7 probe series were one-shot manual invocations.

**Goal:** wire these pieces into a continuously-running, gated loop that senses
the biggest open discrepancy, diagnoses its cause, proposes and implements a
flag-gated fix, hard-gates it, and lands it on a review branch — accumulating
knowledge so it never re-tries a refuted approach.

## 2. Decisions (locked during brainstorming)

| Axis | Decision |
|------|----------|
| **Loop goal** | Full standing pipeline across the L0–L4 ladder (numerics → physics-sanity → reference-parity → experiment-calibration → optimization). |
| **Autonomy** | Auto-implement with guardrails: the loop writes the flag-gated change, runs the full gate stack, and lands only if every gate is green. Human reviews the resulting branch/PR after the fact. |
| **Ground truth** | SCAPS-1D / synthetic today (partner lab data gated, e.g. `Nt_C_PVK`). Build the L3 data-ingest seam now so real measurements drop in later with no rework. |
| **Cadence** | Continuous "boulder" loop: keep finding the next-biggest gap until a stop condition (parity target met / K dry rounds / token budget / reject-streak). |
| **Engine** | Approach A — deterministic Python *spine* + agentic *cognition*. The parts that must be correct (gates, scoring, provenance, anti-thrash dedup) are code; the parts that need judgment (novel cause, novel fix) are agents. The spine re-checks everything an agent does before landing. |
| **Execution model** | Headless Python orchestrator drives spine steps directly and invokes cognition via `claude -p --output-format json` (structured-schema subprocess), so "continuous" is real and schedulable. |

## 3. The loop ladder (conceptual frame)

The four concerns are run as **nested loops, cheap-inner → expensive-outer, each
with its own gate**. The system never enters an outer loop on a broken inner one.

| Loop | Question | Gate |
|------|----------|------|
| L0 numerics | Solver converges? bit-stable? conserves charge/current? | unit tests + golden-master regression |
| L1 physics sanity | Single device physically sane? limiting cases hold? | detailed-balance ceiling, dark J=0, monotonic trends |
| L2 reference parity | Matches SCAPS / analytic? | trend match first, absolutes "approach" |
| L3 experiment | Matches measured device? | fit free params through the data seam |
| L4 optimization | Best design? | only on a parity-trusted model |

Every observed discrepancy is **attributed to a layer (bug / numerics / physics /
data) before the next loop is spent.** This is a debugging hierarchy, not a
pipeline.

## 4. Architecture — spine + cognition

```
 ┌─ SPINE (Python, deterministic) ───────────────┐
 │  ladder · scorecard · gates · ledgers ·         │   must be correct → code
 │  provenance · stop-conditions · data-seam       │
 └───────────────▲────────────────┬───────────────┘
        proposes  │                │  re-checks everything
 ┌────────────────┴── COGNITION ───▼───────────────┐
 │  attribution · fix-proposal · adversarial-verify │  needs judgment → agents
 │  (multi-agent fan-out, schema-validated output)  │
 └──────────────────────────────────────────────────┘
```

**The cycle (one boulder turn):**

`Sense → Attribute → Propose → Implement → Gate → Land → Record → Next`

1. **Sense** — spine runs the ladder, scores parity, ranks open gaps, picks the biggest *not-refuted* one.
2. **Attribute** — cognition auto-ablates `SOLARLAB_*` flags + limiting cases, classifies cause, and **adversarially verifies** the mechanism (refute before trust).
3. **Propose** — cognition designs one flag-gated change (default-OFF), with the residual-reconciliation arithmetic.
4. **Implement** — cognition writes it; legacy path stays bit-identical.
5. **Gate** — spine re-runs the ladder through the hard gate stack.
6. **Land** — all green + gap actually closed → atomic commit on the boulder branch. Else reject.
7. **Record** — outcome (incl. negative results) → ledgers.
8. **Next** — pick the next gap, or stop.

## 5. Repo layout

```
perovskite_sim/autoloop/
  orchestrator.py   boulder driver: cycle loop + stop conditions
  ladder.py         L0–L2 runner (wraps run_scaps_*_regression.py)
  scorecard.py      parity score + gap ranking (wraps scaps_absolute_scorecard)
  gates.py          the gate stack (hard barriers)
  ledger.py         gap / hypothesis / negative-result ledgers (JSON)
  provenance.py     commit + config + flags + seed stamping
  datasource.py     L3 ingest seam (SCAPS now, real data later)
scripts/autoloop_run.py        entry: --once | --continuous
.claude/workflows/
  attribute-gap.js  cognition: ablate + diagnose + verify
  propose-fix.js    cognition: design flag-gated change
outputs/autoloop/<run-id>/     provenance-stamped cycle artifacts
docs/autoloop/ledger/          human-readable ledger mirror
```

## 6. State — the three ledgers (anti-thrash memory)

JSON in repo (committed) + markdown mirror. **Seeded from existing project
memory** (DOS-cap, BBD blow-up, the 1.40 V_bi fudge are already known-refuted).
Updates are immutable: new entries + logged status transitions, never in-place
edits.

**Gap ledger** — open discrepancies, ranked:
```
{id, metric, sweep_point, solarlab_val, scaps_val, gap_mag,
 kind: trend|absolute, status: open|closed|refuted|blocked,
 found_cycle, last_attempt_cycle, mechanism?}
```

**Hypothesis ledger** — attribution attempts per gap:
```
{gap_id, cause: bug|numerics|physics|data, mechanism,
 evidence_for[], evidence_against[], verifier_votes,
 verdict: confirmed|refuted|uncertain, cycle}
```

**Negative-results ledger** — the memory that stops thrash:
```
{approach, why_failed, evidence, never_retry: true}
```
Seeds: `"DOS-cap projection → false convergence, high residual node 19"`,
`"BBD face-density term → V=0.08 blow-up"`,
`"1.40 V_bi fudge → +106 mV unexplained, rejected"`.

**Anti-thrash rules (spine-enforced, deterministic):**

- The propose-leg **must read** negative-results first; the spine **dedups** every proposal against it (analogous to a loop-until-dry `seen` set). A refuted match is auto-rejected with no agent vote.
- A gap refuted K times → `blocked`, skipped until a human or new data unblocks it (`Nt_C_PVK` = partner-data-blocked).
- Confirmed-cause gaps keep their verified mechanism, so the propose-leg starts from the mechanism, not from scratch.

**Provenance:** every cycle stamped with `git SHA · config hash · active flags ·
RNG seed · solver tolerances · data version · timestamp`. Artifacts land in
`outputs/autoloop/<run-id>/`. Ledger and provenance writes commit immediately and
re-verify (OneDrive sync resurrects/overwrites unstaged files).

## 7. The gate stack (guardrails)

A proposed change must pass **all** gates to land. Ordered cheap→costly,
fail-fast.

| Gate | Barrier | Engine |
|------|---------|--------|
| **G0 legacy bit-identical** | flag OFF → golden-masters byte-for-byte vs pre-change. Zero default-path regression. | deterministic |
| **G1 numerics** | pytest green · Newton converges · charge/current conserved · RHS finite · no new stall | deterministic |
| **G2 limiting cases** | rad-only V_oc ≤ detailed-balance ceiling · dark J=0 · grid-converged (n_points 20→80 stable, no interp artifact) | deterministic |
| **G3 scorecard improved** | targeted gap closed (trend up) AND nothing else regressed past tolerance. Net parity gain. | deterministic |
| **G4 honest-residual (fudge-guard)** | change ships a mechanism whose **magnitude reconciles the residual** (e.g. 137 mV = 84 + 53 kT·ln). Closes-the-gap-but-mechanism-doesn't-add-up → REJECT. | cognition supplies, **spine checks arithmetic** |
| **G5 adversarial verify** | ≥ majority of independent skeptics fail to refute the mechanism, each via a distinct lens | cognition, majority vote |

G0–G3 are pure deterministic Python. G4–G5 consume cognition output but the spine
enforces *presence + arithmetic* as a hard check — even the soft gates have a hard
core.

**Stop conditions:** parity target met · K dry rounds (all open gaps
blocked/refuted) · token budget exhausted · M-reject streak → halt + flag human.

**Branch model:** the boulder runs on an `autoloop/<date>` branch, one atomic
conventional commit per landed change (with decision trailers). The human reviews
the accumulated commits and merges. The loop never touches `main` unreviewed.

## 8. Cognition legs

Each is invoked `claude -p --output-format json` and returns schema-validated JSON
the spine re-checks. **Hard contract:** cognition *requests* sim runs; the spine
*executes and records* them — an agent never runs-and-reports its own gate.

**`attribute-gap`** — automates the e7_probe / kill_auger / cascade work, per gap:
- *Ablate* — spine toggles each relevant `SOLARLAB_*`, runs limiting cases + grid-convergence + residual-by-channel decomposition → ablation matrix.
- *Diagnose* — agent reads the matrix → cause + mechanism.
- *Verify* — N skeptics refute via distinct lenses (correctness / numerics-artifact / data-support / limiting-case); majority must fail to refute.
- → `{cause, mechanism, evidence_for[], evidence_against[], verdict, confidence}`.

This is the 36-agent run, now auto-fired per-gap and scoped to a single gap.

**`propose-fix`** — confirmed mechanism → one flag-gated change spec (module, term,
flag, default-OFF) + the residual-reconciliation arithmetic that feeds G4. Reads
the negative-results ledger (spine pre-filters refuted approaches).

**`implement`** — writes the change + the flag-ON test, legacy bit-identical → diff.

## 9. L3 data seam and L4 search

**L3 seam** (`datasource.py`): a single interface
`get_reference(metric, point) → {value, source, uncertainty}`. Today it returns
SCAPS table + synthetic values; a real-lab loader (J-V / EQE CSV) drops in behind
the same interface with zero spine rework. L3 calibration is a scipy/optuna fit of
free parameters to whatever the seam returns; flipping the source from SCAPS to
hardware retargets the fit automatically.

**L4 search:** optuna/pymoo over the device-design space (layer thickness, doping,
band offsets), objective PCE (or multi-objective), **gated on scorecard ≥
parity-threshold** so the loop never optimizes an untrusted model. Output is
**advisory** — designs are reported, not auto-landed, because a design choice is a
human call (a different risk class than an auto-implemented physics fix).

## 10. Error handling

- Solver non-convergence mid-cycle → logged as a numerics-gap, not a crash; the cycle continues.
- Cognition agent dies / returns null → retry once, then skip the gap and record.
- Gate rejection → normal flow; recorded, next gap.
- Ledger / provenance writes → commit immediately and re-verify (OneDrive overwrite gotcha).
- Refuted-retry → spine dedup blocks it before any sim runs.
- M-reject streak / suspected infinite loop → hard halt + human flag.

## 11. Testing the loop itself

- **Unit:** scorecard math, gate booleans, ledger dedup (all deterministic).
- **Guardrail golden-master:** a known-good change MUST pass; the known-fudge (1.40 V_bi) MUST be rejected by G4 — a regression test on the guardrails themselves.
- **Mock cognition:** inject canned diagnosis / diff to exercise spine logic at zero LLM cost.
- **Dry-run mode:** a full cycle with no landing, CI-safe.

## 12. Build order (staged)

1. **Spine + gates + ledgers + scorecard** — the guardian. Deterministic, CI-able, no agents. (Closes the current CI gap.)
2. **`attribute-gap` cognition leg** — read-only; produces diagnoses, proposes nothing to code yet.
3. **`propose-fix` + `implement` + G4/G5 + branch landing** — auto-implement with guardrails goes live.
4. **Boulder driver (continuous) + L3 seam wired + L4 search.**

## 13. Out of scope / deferred

- Auto-landing to `main` (always human-reviewed branch merge).
- Auto-landing of L4 design choices (advisory only).
- Tunneling and other high-stiffness physics deferred per existing `feedback_defer_stiffness` guidance.
- Real partner-lab data ingestion (seam built now; loader added when data unblocks).
