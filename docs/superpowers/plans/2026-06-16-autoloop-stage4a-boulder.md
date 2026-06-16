# Autoloop Stage 4a — Boulder Driver — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the continuous boulder driver that chains sense→attribute→implement — a sweep (dry-run) mode that drains current gaps into a batch report, and a `--converge` mode that auto-applies landable fixes, re-senses the evolved config, and loops until a stop condition.

**Architecture:** A single `run_boulder(...)` in-process loop over three injected step-functions (`sense`/`attribute`/`implement`, each wrapping the Stage 1–3 orchestrator functions with their deps bound). It advances by transitioning processed gaps to a new `attempted` status; `guardian_once` gains a status-preserving merge so a converge re-sense can't resurrect an attempted/refuted gap (the termination guarantee). The CLI wires the real step-functions; tests inject fakes so the loop logic runs with zero solver.

**Tech Stack:** Python 3.9+, dataclasses, json. Reuses Stages 1–3 (`guardian_once`/`attribute_top_gap`/`implement_top_confirmed`/`Ledger`/`Gap.with_status`/`provenance._git`). No new third-party deps.

---

## Design contract (read before starting)

- **Spec:** `docs/superpowers/specs/2026-06-16-autoloop-stage4a-boulder-design.md`.
- **Step-function contracts** (the boulder is pure loop logic over these; CLI binds the real ones, tests inject fakes):
  - `sense(cycle: int) -> float | None` — runs the guardian, returns `score.overall` (parity, 0..1) or None.
  - `attribute(cycle: int) -> None` — diagnoses the top open gap, writes its Hypothesis to the ledger.
  - `implement(cycle: int, apply: bool) -> ImplementResult` — proposes (dry-run) or lands (apply) the top confirmed gap.
- **Advancement:** after processing a gap, if `implement` did NOT return `"applied"`, the boulder sets that gap → `"attempted"` so the next top-open pick advances. An `"applied"` gap is already `"closed"` by `implement_top_confirmed`.
- **Stage 1–3 APIs (verified on `main`):**
  - `ledger.Ledger.load(root)` / `.save()` / `.add_gap(g)` (replace-by-id) / `.gaps` / `.hypotheses`.
  - `types.Gap.with_status(status)`; statuses used: `open`/`closed`/`refuted`/`blocked` + new `attempted`.
  - `types.ImplementResult(status, hypothesis_gap_id, device_key, gate_verdicts, committed_sha, note)`; status ∈ {applied, dry_run, gates_failed, no_confirmed, not_promotable}.
  - `orchestrator.guardian_once(...) -> dict` with `["overall"]`; gap-add loop at lines 46–52.
  - `provenance._git(*args) -> str`.
- **Run all commands from `perovskite-sim/`.** Tests default to `-m 'not slow'`.

## File Structure

```
perovskite_sim/autoloop/
  types.py          + BoulderProposal, BoulderResult
  orchestrator.py   guardian_once: status-preserving merge ; + run_boulder
scripts/autoloop_run.py  + --boulder [--converge] [--parity-target] [--max-cycles] [--reject-streak]
tests/unit/autoloop/
  test_types_boulder.py
  test_guardian_status_preserving.py
  test_boulder_sweep.py
  test_boulder_converge.py
tests/integration/
  test_autoloop_boulder.py   (slow, real sweep)
```

---

## Task 1: Boulder result types

**Files:**
- Modify: `perovskite_sim/autoloop/types.py`
- Test: `tests/unit/autoloop/test_types_boulder.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/autoloop/test_types_boulder.py
import dataclasses
import pytest
from perovskite_sim.autoloop.types import BoulderProposal, BoulderResult


def test_boulder_proposal_frozen():
    p = BoulderProposal(gap_id="g", cause="physics", mechanism="flag X term",
                        device_key="interface_plane_projection", gate_status="dry_run",
                        landed=False)
    assert p.landed is False
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.landed = True  # type: ignore[misc]


def test_boulder_result_defaults():
    r = BoulderResult(mode="sweep", cycles=3, proposals=(), landed_count=0,
                      stop_reason="sweep_complete", final_overall=None)
    assert r.mode == "sweep" and r.landed_count == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_types_boulder.py`
Expected: FAIL — `ImportError: cannot import name 'BoulderProposal'`.

- [ ] **Step 3: Append to `types.py`** (after `ImplementResult`)

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
    proposals: tuple
    landed_count: int
    stop_reason: str          # "sweep_complete"|"success"|"drained"|"cap"|"halt"
    final_overall: Optional[float]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_types_boulder.py`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add perovskite-sim/perovskite_sim/autoloop/types.py perovskite-sim/tests/unit/autoloop/test_types_boulder.py
git commit -m "feat(autoloop): add BoulderProposal + BoulderResult types (Stage 4a)"
```

---

## Task 2: Status-preserving gap merge in guardian_once

**Files:**
- Modify: `perovskite_sim/autoloop/orchestrator.py` (the gap-add loop, lines 46–52)
- Test: `tests/unit/autoloop/test_guardian_status_preserving.py`

This is the termination guarantee: a converge re-sense must NOT resurrect an `attempted`/`refuted`/`blocked`/`closed` gap back to `open`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/autoloop/test_guardian_status_preserving.py
from perovskite_sim.autoloop.types import Gap, ParityScore, SweepScore, LadderResult
from perovskite_sim.autoloop.ledger import Ledger
from perovskite_sim.autoloop.orchestrator import guardian_once


def _gap(gid, status):
    return Gap(id=gid, metric="V_oc", sweep="Nd_ETL", sweep_point=0.0,
               solarlab_val=30.0, reference_val=70.0, gap_mag=0.4, kind="trend",
               status=status, found_cycle=0, last_attempt_cycle=0, mechanism=None)


def _ladder_with_nd_etl_gap(**kw):
    # A ladder result whose scorecard yields a trend:Nd_ETL:V_oc gap (low closure).
    score = ParityScore(overall=0.4, base_deltas={},
                        per_sweep={"Nd_ETL": SweepScore("Nd_ETL", 30.0, 5, 4)})
    return LadderResult(l0_pass=True, l1_pass=True, score=score, details={})


def test_resense_does_not_resurrect_attempted_gap(tmp_path):
    # Seed the ledger with the gap already marked attempted.
    led = Ledger(root=tmp_path / "ledger")
    led.add_gap(_gap("trend:Nd_ETL:V_oc", "attempted"))
    led.save()

    guardian_once(ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
                  reference_path=tmp_path / "r.json", config_path=tmp_path / "c.yaml",
                  cycle=1, timestamp="2026-06-16T00:00:00Z", l0_paths=["tests/unit/autoloop"],
                  baseline=None, run_ladder_fn=_ladder_with_nd_etl_gap)

    led2 = Ledger.load(tmp_path / "ledger")
    g = next(g for g in led2.gaps if g.id == "trend:Nd_ETL:V_oc")
    assert g.status == "attempted"     # preserved, NOT reset to open


def test_new_gap_enters_open(tmp_path):
    led = Ledger(root=tmp_path / "ledger"); led.save()   # empty
    guardian_once(ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
                  reference_path=tmp_path / "r.json", config_path=tmp_path / "c.yaml",
                  cycle=0, timestamp="2026-06-16T00:00:00Z", l0_paths=["tests/unit/autoloop"],
                  baseline=None, run_ladder_fn=_ladder_with_nd_etl_gap)
    led2 = Ledger.load(tmp_path / "ledger")
    g = next(g for g in led2.gaps if g.id == "trend:Nd_ETL:V_oc")
    assert g.status == "open"          # brand-new gap enters open
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_guardian_status_preserving.py`
Expected: FAIL — `test_resense_does_not_resurrect_attempted_gap` fails: the gap is reset to `open` by the current `add_gap`.

- [ ] **Step 3: Edit the gap-add loop in `guardian_once`**

Replace lines 46–52 (the `gap_ids` / `for g in gaps_from_score(...)` block):

```python
    gap_ids: list[str] = []
    if result.score is not None:
        existing = {g.id: g for g in led.gaps}
        for g in gaps_from_score(result.score, cycle=cycle):
            if led.is_refuted(g.id):          # never resurface a refuted approach
                continue
            prior = existing.get(g.id)
            if prior is not None and prior.status != "open":
                continue   # preserve attempted/blocked/refuted/closed — do NOT
                           # resurrect to open (the boulder converge termination guard)
            led.add_gap(g)
            gap_ids.append(g.id)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_guardian_status_preserving.py`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add perovskite-sim/perovskite_sim/autoloop/orchestrator.py perovskite-sim/tests/unit/autoloop/test_guardian_status_preserving.py
git commit -m "feat(autoloop): status-preserving gap merge in guardian_once (boulder termination)"
```

---

## Task 3: run_boulder — sweep mode

**Files:**
- Modify: `perovskite_sim/autoloop/orchestrator.py`
- Test: `tests/unit/autoloop/test_boulder_sweep.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/autoloop/test_boulder_sweep.py
from perovskite_sim.autoloop.types import Gap, Hypothesis, ImplementResult
from perovskite_sim.autoloop.ledger import Ledger
from perovskite_sim.autoloop.orchestrator import run_boulder


def _gap(gid, mag):
    return Gap(id=gid, metric="V_oc", sweep="Nd_ETL", sweep_point=0.0,
               solarlab_val=30.0, reference_val=70.0, gap_mag=mag, kind="trend",
               status="open", found_cycle=0, last_attempt_cycle=0, mechanism=None)


def test_sweep_drains_all_gaps_and_records_proposals(tmp_path):
    led = Ledger(root=tmp_path / "ledger")
    led.add_gap(_gap("g_hi", 0.5))
    led.add_gap(_gap("g_lo", 0.2))
    led.save()

    def sense(cycle):
        return 0.4

    def attribute(cycle):
        # write a confirmed hypothesis for the current top open gap
        led = Ledger.load(tmp_path / "ledger")
        top = max((g for g in led.gaps if g.status == "open"), key=lambda g: g.gap_mag)
        led.add_hypothesis(Hypothesis(gap_id=top.id, cause="physics",
                                      mechanism="flag X term", verdict="confirmed"))
        led.save()

    def implement(cycle, apply):
        led = Ledger.load(tmp_path / "ledger")
        top = max((g for g in led.gaps if g.status == "open"), key=lambda g: g.gap_mag)
        return ImplementResult("dry_run", top.id, "interface_plane_projection", (), None)

    result = run_boulder(ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
                         timestamp="2026-06-16T00:00:00Z", converge=False,
                         sense=sense, attribute=attribute, implement=implement)

    assert result.mode == "sweep"
    assert result.stop_reason == "sweep_complete"
    assert result.cycles == 2                       # both gaps drained
    assert {p.gap_id for p in result.proposals} == {"g_hi", "g_lo"}
    assert result.landed_count == 0                 # dry-run, nothing committed
    # both gaps marked attempted
    led2 = Ledger.load(tmp_path / "ledger")
    assert all(g.status == "attempted" for g in led2.gaps)


def test_sweep_empty_when_no_gaps(tmp_path):
    led = Ledger(root=tmp_path / "ledger"); led.save()
    result = run_boulder(ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
                         timestamp="2026-06-16T00:00:00Z", converge=False,
                         sense=lambda c: 0.95, attribute=lambda c: None,
                         implement=lambda c, apply: None)
    assert result.cycles == 0 and result.proposals == ()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_boulder_sweep.py`
Expected: FAIL — `ImportError: cannot import name 'run_boulder'`.

- [ ] **Step 3: Add `run_boulder` to `orchestrator.py`**

Add imports (with the others at the top):

```python
from perovskite_sim.autoloop.types import BoulderProposal, BoulderResult
```

Append the function:

```python
def _boulder_top_open(ledger_root):
    led = Ledger.load(ledger_root)
    open_gaps = [g for g in led.gaps if g.status == "open"]
    return max(open_gaps, key=lambda g: g.gap_mag) if open_gaps else None


def _boulder_proposal(ledger_root, gap, result) -> BoulderProposal:
    led = Ledger.load(ledger_root)
    hyp = next((h for h in led.hypotheses if h.gap_id == gap.id), None)
    status = result.status if result is not None else "no_confirmed"
    return BoulderProposal(
        gap_id=gap.id,
        cause=(hyp.cause if hyp else "uncertain"),
        mechanism=(hyp.mechanism if hyp else ""),
        device_key=(result.device_key if result is not None else None),
        gate_status=status,
        landed=(status == "applied"))


def _boulder_mark_attempted(ledger_root, gap):
    led = Ledger.load(ledger_root)
    led.add_gap(gap.with_status("attempted"))
    led.save()


def run_boulder(*, ledger_root, outputs_root, timestamp, converge: bool = False,
                parity_target: float = 0.90, max_cycles: int = 10, reject_streak: int = 3,
                sense, attribute, implement) -> BoulderResult:
    """Continuous driver. sweep (dry-run) drains current gaps into proposals;
    converge auto-applies + re-senses + loops to a stop condition. The three
    step-functions are injected (CLI binds real ones; tests inject fakes)."""
    ledger_root = Path(ledger_root)
    proposals: list[BoulderProposal] = []

    if not converge:
        sense(0)
        cycle = 0
        while True:
            gap = _boulder_top_open(ledger_root)
            if gap is None:
                break
            attribute(cycle)
            result = implement(cycle, False)
            proposals.append(_boulder_proposal(ledger_root, gap, result))
            _boulder_mark_attempted(ledger_root, gap)
            cycle += 1
        return BoulderResult("sweep", cycle, tuple(proposals), 0, "sweep_complete", None)

    cycle = 0
    landed = 0
    reject = 0
    overall = None
    stop = "cap"
    while cycle < max_cycles:
        overall = sense(cycle)
        if overall is not None and overall >= parity_target:
            stop = "success"
            break
        gap = _boulder_top_open(ledger_root)
        if gap is None:
            stop = "drained"
            break
        attribute(cycle)
        result = implement(cycle, True)
        proposals.append(_boulder_proposal(ledger_root, gap, result))
        if result is not None and result.status == "applied":
            landed += 1
            reject = 0
        else:
            _boulder_mark_attempted(ledger_root, gap)
            if result is not None and result.status == "gates_failed":
                reject += 1
        if reject >= reject_streak:
            stop = "halt"
            break
        cycle += 1
    return BoulderResult("converge", cycle, tuple(proposals), landed, stop, overall)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_boulder_sweep.py`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add perovskite-sim/perovskite_sim/autoloop/orchestrator.py perovskite-sim/tests/unit/autoloop/test_boulder_sweep.py
git commit -m "feat(autoloop): add run_boulder sweep mode (drain gaps -> proposals)"
```

---

## Task 4: run_boulder — converge stop conditions

**Files:**
- Test: `tests/unit/autoloop/test_boulder_converge.py` (converge logic already in `run_boulder` from Task 3; this task pins all four stop conditions)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/autoloop/test_boulder_converge.py
from perovskite_sim.autoloop.types import Gap, Hypothesis, ImplementResult
from perovskite_sim.autoloop.ledger import Ledger
from perovskite_sim.autoloop.orchestrator import run_boulder


def _seed(tmp_path, *gaps):
    led = Ledger(root=tmp_path / "ledger")
    for gid, mag in gaps:
        led.add_gap(Gap(id=gid, metric="V_oc", sweep="Nd_ETL", sweep_point=0.0,
                        solarlab_val=30.0, reference_val=70.0, gap_mag=mag, kind="trend",
                        status="open", found_cycle=0, last_attempt_cycle=0, mechanism=None))
    led.save()


def _confirm_top(tmp_path):
    led = Ledger.load(tmp_path / "ledger")
    og = [g for g in led.gaps if g.status == "open"]
    if og:
        top = max(og, key=lambda g: g.gap_mag)
        led.add_hypothesis(Hypothesis(gap_id=top.id, cause="physics",
                                      mechanism="flag X term", verdict="confirmed"))
        led.save()


def test_converge_stops_success_when_parity_met(tmp_path):
    _seed(tmp_path, ("g", 0.5))
    result = run_boulder(ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
                         timestamp="t", converge=True, parity_target=0.9,
                         sense=lambda c: 0.95, attribute=lambda c: None,
                         implement=lambda c, a: None)
    assert result.stop_reason == "success" and result.cycles == 0


def test_converge_stops_drained_when_no_open_gaps(tmp_path):
    _seed(tmp_path)   # no gaps
    result = run_boulder(ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
                         timestamp="t", converge=True, parity_target=0.9,
                         sense=lambda c: 0.5, attribute=lambda c: None,
                         implement=lambda c, a: None)
    assert result.stop_reason == "drained"


def test_converge_lands_then_drains(tmp_path):
    _seed(tmp_path, ("g", 0.5))

    def implement(cycle, apply):
        # land the fix: close the gap (as implement_top_confirmed would on apply)
        led = Ledger.load(tmp_path / "ledger")
        g = next(x for x in led.gaps if x.id == "g")
        led.add_gap(g.with_status("closed"))
        led.save()
        return ImplementResult("applied", "g", "interface_plane_projection", (), "sha123")

    result = run_boulder(ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
                         timestamp="t", converge=True, parity_target=0.99, max_cycles=5,
                         sense=lambda c: 0.5, attribute=lambda c: _confirm_top(tmp_path),
                         implement=implement)
    assert result.landed_count == 1
    assert result.stop_reason == "drained"        # after landing, no open gaps left


def test_converge_stops_cap_when_never_improves(tmp_path):
    _seed(tmp_path, ("g", 0.5))

    def implement(cycle, apply):
        return ImplementResult("not_promotable", "g", None, (), None)

    result = run_boulder(ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
                         timestamp="t", converge=True, parity_target=0.9, max_cycles=3,
                         # re-seed an open gap each sense so it never drains
                         sense=lambda c: (_seed(tmp_path, ("g2", 0.4)) or 0.5),
                         attribute=lambda c: None, implement=implement)
    assert result.stop_reason == "cap" and result.cycles == 3


def test_converge_stops_halt_on_reject_streak(tmp_path):
    _seed(tmp_path, ("g", 0.5))

    def implement(cycle, apply):
        return ImplementResult("gates_failed", "g", "interface_plane_projection", (), None)

    result = run_boulder(ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
                         timestamp="t", converge=True, parity_target=0.9, max_cycles=10,
                         reject_streak=2,
                         sense=lambda c: (_seed(tmp_path, (f"g{c}", 0.4)) or 0.5),
                         attribute=lambda c: _confirm_top(tmp_path), implement=implement)
    assert result.stop_reason == "halt"
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_boulder_converge.py`
Expected: PASS (5 tests) — the converge logic landed in Task 3. If any fails, fix `run_boulder` converge branch (do NOT weaken the test). Note: `test_converge_stops_cap` re-seeds a fresh open gap on each `sense` to keep the loop from draining, isolating the `max_cycles` cap.

- [ ] **Step 3: (only if Step 2 showed a failure) fix run_boulder**

If a stop condition misfires, the bug is in the converge branch ordering (parity check before drained check before reject/cap). Ensure: parity → drained → process → reject → cap, exactly as in Task 3's code.

- [ ] **Step 4: Re-run**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_boulder_converge.py`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add perovskite-sim/tests/unit/autoloop/test_boulder_converge.py
git commit -m "test(autoloop): pin run_boulder converge stop conditions (success/drained/cap/halt)"
```

---

## Task 5: CLI `--boulder` + sweep smoke + docs

**Files:**
- Modify: `scripts/autoloop_run.py`
- Create: `tests/integration/test_autoloop_boulder.py`
- Modify: `perovskite-sim/CLAUDE.md`, `README.md`

- [ ] **Step 1: Write the failing integration test (slow, real sweep)**

```python
# tests/integration/test_autoloop_boulder.py
import pytest
from pathlib import Path
from perovskite_sim.autoloop.orchestrator import (
    run_boulder, guardian_once, attribute_top_gap, implement_top_confirmed)
from perovskite_sim.autoloop.attribution import HeuristicAttributor
from perovskite_sim.autoloop.subprocess_probe import SubprocessProbeRunner
from perovskite_sim.autoloop.gates_impl import make_implement_gate_runner

REPO_ROOT = Path(__file__).resolve().parents[1]
REF = REPO_ROOT / "tests" / "integration" / "scaps_reference.json"
CFG = REPO_ROOT / "configs" / "scaps_mirror_v2.yaml"


@pytest.mark.slow
def test_boulder_sweep_real(tmp_path):
    lr, orr = tmp_path / "ledger", tmp_path / "out"

    def sense(cycle):
        rep = guardian_once(ledger_root=lr, outputs_root=orr, reference_path=REF,
                            config_path=CFG, cycle=cycle, timestamp="t",
                            l0_paths=["tests/unit/autoloop"], baseline=None)
        return rep["overall"]

    def attribute(cycle):
        attribute_top_gap(ledger_root=lr, outputs_root=orr, config_path=CFG,
                          reference_path=REF, cycle=cycle, timestamp="t",
                          probe_runner_factory=lambda g: SubprocessProbeRunner(
                              config_path=CFG, reference_path=REF, gap=g),
                          attributor=HeuristicAttributor())

    def implement(cycle, apply):
        def _measure(edit, gap):
            return SubprocessProbeRunner(config_path=edit.config_path, reference_path=REF,
                                         gap=gap).run({"env_flags": {}, "jv_overrides": {}, "measure": "gap"})
        return implement_top_confirmed(
            ledger_root=lr, outputs_root=orr, config_path=CFG, reference_path=REF,
            cycle=cycle, timestamp="t",
            gate_runner=make_implement_gate_runner(measure_badness=_measure), apply=apply)

    cfg_before = CFG.read_text(encoding="utf-8")
    result = run_boulder(ledger_root=lr, outputs_root=orr, timestamp="t",
                         converge=False, sense=sense, attribute=attribute, implement=implement)
    assert result.mode == "sweep" and result.stop_reason == "sweep_complete"
    # validation established Nd_ETL is the surfaced gap with the IFACE_PROJ lever
    assert any(p.gap_id == "trend:Nd_ETL:V_oc" for p in result.proposals)
    assert CFG.read_text(encoding="utf-8") == cfg_before     # sweep commits nothing
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd perovskite-sim && python -m pytest -q -m slow tests/integration/test_autoloop_boulder.py`
Expected: FAIL until the CLI wiring lands; once green it confirms the real sweep drains gaps + leaves the config untouched. Runtime: minutes (real parity score + one ablation), bounded by the 240s probe timeout.

- [ ] **Step 3: Wire the CLI + docs**

In `scripts/autoloop_run.py`, add flags to `parse_args`:

```python
    ap.add_argument("--boulder", action="store_true",
                    help="run the continuous boulder (sweep dry-run unless --converge)")
    ap.add_argument("--converge", action="store_true",
                    help="with --boulder/--implement: auto-apply landable fixes and loop")
    ap.add_argument("--parity-target", type=float, default=0.90)
    ap.add_argument("--max-cycles", type=int, default=10)
    ap.add_argument("--reject-streak", type=int, default=3)
```

(Note: `--converge` is shared with `--implement`; that's fine — `--implement` ignores it, only `--boulder` reads it.)

Add the dispatch in `main` (before the `--implement` block):

```python
    if ns.boulder:
        import dataclasses
        from perovskite_sim.autoloop.orchestrator import (
            run_boulder, guardian_once, attribute_top_gap, implement_top_confirmed)
        from perovskite_sim.autoloop.attribution import HeuristicAttributor
        from perovskite_sim.autoloop.subprocess_probe import SubprocessProbeRunner
        from perovskite_sim.autoloop.gates_impl import make_implement_gate_runner
        from perovskite_sim.autoloop.provenance import _git

        if ns.converge:
            branch = _git("rev-parse", "--abbrev-ref", "HEAD")
            if branch in ("main", "master"):
                print(json.dumps({"boulder": None,
                                  "error": f"refuse --boulder --converge on '{branch}'; "
                                           "create an autoloop branch first"}))
                return 1

        def sense(cycle):
            rep = guardian_once(ledger_root=ns.ledger_root, outputs_root=ns.outputs_root,
                                reference_path=ns.reference, config_path=ns.config,
                                cycle=cycle, timestamp=iso_timestamp_utc(),
                                l0_paths=ns.l0_paths, baseline=None)
            return rep["overall"]

        def attribute(cycle):
            attribute_top_gap(ledger_root=ns.ledger_root, outputs_root=ns.outputs_root,
                              config_path=ns.config, reference_path=ns.reference, cycle=cycle,
                              timestamp=iso_timestamp_utc(),
                              probe_runner_factory=lambda g: SubprocessProbeRunner(
                                  config_path=ns.config, reference_path=ns.reference, gap=g),
                              attributor=HeuristicAttributor())

        def implement(cycle, apply):
            def _measure(edit, gap):
                return SubprocessProbeRunner(config_path=edit.config_path,
                                             reference_path=ns.reference, gap=gap).run(
                    {"env_flags": {}, "jv_overrides": {}, "measure": "gap"})
            return implement_top_confirmed(
                ledger_root=ns.ledger_root, outputs_root=ns.outputs_root, config_path=ns.config,
                reference_path=ns.reference, cycle=cycle, timestamp=iso_timestamp_utc(),
                gate_runner=make_implement_gate_runner(measure_badness=_measure), apply=apply)

        result = run_boulder(ledger_root=ns.ledger_root, outputs_root=ns.outputs_root,
                             timestamp=iso_timestamp_utc(), converge=ns.converge,
                             parity_target=ns.parity_target, max_cycles=ns.max_cycles,
                             reject_streak=ns.reject_streak,
                             sense=sense, attribute=attribute, implement=implement)
        print(json.dumps({"boulder": dataclasses.asdict(result)}, indent=2,
                         sort_keys=True, default=str))
        return 1 if result.stop_reason == "halt" else 0
```

Add to the **Autoloop** section of `perovskite-sim/CLAUDE.md`:

```markdown
### Stage 4a — boulder driver (continuous loop)

`autoloop/orchestrator.run_boulder` chains guardian->attribute->implement.
**Sweep** (default) drains current open gaps into a batch proposal report (dry-run,
no commits), advancing via a new `attempted` gap status. **`--converge`** auto-applies
each landable fix, re-senses the evolved config (guardian_once preserves
attempted/refuted statuses so it terminates), and loops until parity-target / drained /
max-cycles / reject-streak. Converge refuses to start on main/master and commits via
Stage 3 commit_promotion on the current branch (never pushes).

    cd perovskite-sim
    python scripts/autoloop_run.py --boulder            # sweep: drain -> report, no commits
    git checkout -b autoloop/$(date +%F)
    python scripts/autoloop_run.py --boulder --converge --max-cycles 5   # auto-apply loop
```

Add to `README.md` (next to the other autoloop lines):

```markdown
- **Autoloop boulder** (`python perovskite-sim/scripts/autoloop_run.py --boulder [--converge]`) —
  the continuous driver: sweep drains gaps into a proposal report; --converge auto-applies
  fixes and loops until the parity target (on an autoloop branch, never main).
```

- [ ] **Step 4: Run tests**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop && python -m pytest -q -m slow tests/integration/test_autoloop_boulder.py`
Expected: all green. Also `python -m pytest -q` (full default suite) — confirm no import/collection regression.

- [ ] **Step 5: Commit**

```bash
git add perovskite-sim/scripts/autoloop_run.py perovskite-sim/tests/integration/test_autoloop_boulder.py perovskite-sim/CLAUDE.md README.md
git commit -m "feat(autoloop): wire --boulder[--converge] CLI + sweep smoke + docs (Stage 4a)"
```

---

## Self-Review

**Spec coverage** (vs `2026-06-16-autoloop-stage4a-boulder-design.md`):
- §2 two modes (sweep/converge) → Task 3 (`run_boulder`). ✓
- §3 status-transition advancement + status-preserving re-sense → Task 2 (guardian merge) + Task 3 (`_boulder_mark_attempted`). ✓
- §4 four converge stop conditions (success/drained/cap/halt) → Task 4 (all pinned). ✓
- §5 `BoulderProposal`/`BoulderResult` + `run_boulder` signature → Tasks 1/3. ✓
- §6 CLI `--boulder [--converge ...]` + converge-on-main guard + commit-on-current-branch → Task 5. ✓
- §7 error handling (converge-on-main refused; parity-already-met → success cycle 0) → Tasks 4/5. ✓
- §8 testing (injected-fake loop tests; status-preserving merge; sweep smoke) → every task. ✓
- §9 deferred (auto-push, 4b/4c, scheduling, parallel) → correctly NOT built.

**Placeholder scan:** none — complete code/tests/commands. Task 4 Step 3 is conditional ("only if Step 2 failed") because the converge logic lands in Task 3; this is explicit, not a placeholder.

**Type consistency:** `BoulderProposal`/`BoulderResult` (Task 1) used in Task 3's `run_boulder` + tests. `run_boulder(sense, attribute, implement, converge, parity_target, max_cycles, reject_streak)` signature consistent Tasks 3/4/5 + tests. `sense(cycle)->float|None`, `attribute(cycle)->None`, `implement(cycle, apply)->ImplementResult` contracts consistent across all tests + the CLI wiring. `Gap.with_status("attempted")`, `Ledger.add_gap/load/save`, `ImplementResult.status/device_key`, `guardian_once(...)["overall"]` are all real Stage 1–3 symbols (verified on `main`).

---

## Execution Handoff

After saving, choose execution:

1. **Subagent-Driven (recommended)** — fresh subagent per task, two-stage review (same background-workflow as Stages 1–3).
2. **Inline Execution** — batch tasks in this session with checkpoints.
