# Autoloop Stage 5.2 — G5 Multi-Skeptic Adversarial Verify — Design

**Date:** 2026-06-17
**Status:** Design approved, pre-planning
**Parent design:** `docs/superpowers/specs/2026-06-16-autoloop-research-pipeline-design.md` (§7 G5 gate, §8 adversarial verify)
**Builds on:** Stages 1–4c + 5.1 (all merged to `main`). Reuses 5.1's `cognition.CognitionRuntime`/`_validate`, the Stage-2 `gate_g5_deferred` stub, `orchestrator.attribute_top_gap`, `types.Hypothesis`/`NegativeResult`, `ledger.add_negative`.

**Scope note:** cognition legs = 5.1 (runtime + attributor, done), **5.2 (G5 verify, this)**, 5.3 (LLM codegen). 5.2 adjudicates 5.1's leads.

---

## 1. Problem & scope

5.1's `LLMAttributor` emits novel-cause **leads** (`verdict="uncertain"`) — nothing acts
on them. 5.2 adds the **G5 multi-skeptic adversarial verifier** that adjudicates a lead:
N diverse-lens skeptics each try to **refute** the mechanism; survives → `confirmed`
(Stage-3-actionable); majority-refute → `refuted` (+ negatives ledger). Fills the
`gate_g5_deferred` stub.

**Decisions (locked during brainstorming):**

| Axis | Decision |
|------|----------|
| Wiring | **Post-attribution lead-adjudicator.** Runs right after attribution on LLM leads (`verdict=="uncertain"` AND `cause != "uncertain"`); promotes/refutes them. Also fills `gate_g5` as a thin wrapper. |
| Skeptics | **N=3 diverse lenses** (physical-plausibility / numerical-artifact / data-support), each conservative (`refuted=true` if no solid support). Same cost as identical refuters, catches more failure modes. |
| Vote | **Strict majority of skeptics that ran**, with a **quorum** of 2. <quorum (skeptics errored) → stays `uncertain` (re-verify later; never a false confirm/refute). |
| Refuted handling | `verdict="refuted"` → `add_negative(mechanism)` → 5.1's LLM never re-proposes it (anti-thrash). |
| Default | No verifier (deterministic). `--verify` opt-in, typically with `--llm`. |

**Explicitly deferred:** 5.3 LLM codegen; G5 as a primary pre-land gate (wrapper only);
verifying heuristic causes; tuning N/lenses.

## 2. Architecture

```
attribute_top_gap(..., verifier=None):
   hyp = attributor.attribute(gap, matrix, led)
   if verifier and hyp.verdict == "uncertain" and hyp.cause != "uncertain":   # an LLM lead
       hyp = verifier.verify(hyp, gap, matrix)
   led.add_hypothesis(hyp)
   if hyp.verdict == "refuted":   led.add_negative(NegativeResult(hyp.mechanism, "refuted by G5 skeptics", ...))
   if hyp.verdict == "confirmed": led.add_gap(gap.with_mechanism(hyp.mechanism))   # Stage-3-actionable
```

New `perovskite_sim/autoloop/verify.py` reusing 5.1's `CognitionRuntime`.

## 3. MultiSkepticVerifier (`verify.py`)

```python
SKEPTIC_LENSES = ["physical-plausibility", "numerical-artifact", "data-support"]   # N=3
VOTE_SCHEMA = {"required": ["refuted"]}   # consumed by cognition._validate; {refuted: bool, reason: str}


def refute_prompt(hyp, gap, matrix, lens) -> str:
    # presents the claim (cause+mechanism), the gap, the ablation matrix probes, the lens,
    # and: "Try to REFUTE this from the <lens> lens. Default refuted=true if you cannot find
    #       solid support. Output ONLY JSON {refuted: true|false, reason: <one sentence>}."


class MultiSkepticVerifier:
    def __init__(self, runtime, *, lenses=SKEPTIC_LENSES, quorum=2):
        self.runtime = runtime; self.lenses = lenses; self.quorum = quorum

    def verify(self, hyp, gap, matrix) -> Hypothesis:
        ran = []          # list of (lens, refuted: bool, reason)
        for lens in self.lenses:
            try:
                v = self.runtime.complete(refute_prompt(hyp, gap, matrix, lens), VOTE_SCHEMA)
                ran.append((lens, bool(v["refuted"]), str(v.get("reason", ""))))
            except Exception as exc:                       # errored skeptic EXCLUDED (not a refute)
                logger.warning("G5 skeptic %s failed: %r", lens, exc)
        if len(ran) < self.quorum:                          # can't adjudicate → leave the lead alone
            logger.warning("G5 quorum not met (%d/%d) — leaving %s uncertain", len(ran), self.quorum, hyp.gap_id)
            return hyp
        refutes = sum(1 for _, r, _ in ran if r)
        verdict = "refuted" if refutes > len(ran) / 2 else "confirmed"   # strict majority
        ev_for = tuple(f"G5 {lens}: {reason}" for lens, r, reason in ran if not r)
        ev_against = tuple(f"G5 {lens}: {reason}" for lens, r, reason in ran if r)
        return dataclasses.replace(
            hyp, verdict=verdict, verifier_votes=len(ran) - refutes,
            evidence_for=hyp.evidence_for + ev_for, evidence_against=hyp.evidence_against + ev_against)


def gate_g5_verify(hyp, gap, matrix, verifier) -> GateVerdict:   # fills the gate_g5_deferred stub
    out = verifier.verify(hyp, gap, matrix)
    return GateVerdict("G5_adversarial_verify", out.verdict == "confirmed",
                       f"verdict={out.verdict}, votes={out.verifier_votes}")
```

**Safety:** an errored skeptic is **excluded, never counted as a refutation** — a claude
outage degrades to `uncertain` (re-verify later), never a false confirm/refute, never
poisons the negatives ledger. Confirmation needs a genuine quorum that couldn't refute.

## 4. Wiring + CLI

- `orchestrator.attribute_top_gap` gains `verifier=None`; runs verify on LLM leads, then
  `add_negative` on refute / `with_mechanism` on confirm (§2). Covers `--boulder` (its
  attribute step calls `attribute_top_gap`).
- CLI: `--verify` → `_build_verifier(ns)` returns `MultiSkepticVerifier(ClaudeCliRuntime(ns.llm_model))`
  (else `None`), wired into `--attribute` + `--boulder`. Typically with `--llm`; `--verify`
  alone is a harmless no-op (no leads). Default: no verifier (deterministic).

```bash
python scripts/autoloop_run.py --attribute --llm --verify    # diagnose novel cause + adjudicate it
python scripts/autoloop_run.py --boulder --llm --verify
```

## 5. Error handling

- Quorum not met (skeptics errored) → lead stays `uncertain`; no negative, no confirm.
- Verify only runs on LLM leads (`cause != "uncertain"`); heuristic no-ops skipped.
- Refuted → `add_negative` (anti-thrash; 5.1's LLM reads negatives).
- The verifier never crashes the loop (per-skeptic try/except + quorum guard).

## 6. Testing

- **`verify`:** FakeRuntime → all-fail-to-refute → `confirmed`; majority-refute → `refuted`;
  <quorum (a runtime that raises for 2 of 3 lenses) → stays `uncertain`; an errored skeptic
  is excluded (1 runs + refutes, but <quorum → uncertain, NOT refuted); `evidence_for/against`
  + `verifier_votes` populated. `refute_prompt` contains the mechanism + the lens name.
- **`gate_g5_verify`:** confirmed → `passed=True`; refuted → `passed=False`.
- **`orchestrator`:** `attribute_top_gap` with a fake attributor emitting an LLM lead
  (`uncertain`, `cause="physics"`) + a fake verifier → confirmed → `gap.mechanism` set;
  refuted → `add_negative` called (assert via reloaded ledger); a heuristic no-op
  (`cause=="uncertain"`) → verifier NOT called (spy).
- **integration smoke (opt-in):** `@pytest.mark.slow` + `skipif(not which("claude") or not
  os.environ.get("SOLARLAB_LLM_SMOKE"))` → real `MultiSkepticVerifier` on a synthetic lead →
  a valid `{confirmed,refuted,uncertain}` verdict. (Cost: N nested claude calls → opt-in.)

## 7. Out of scope / deferred

- 5.3 LLM codegen for non-promotable levers.
- G5 as a primary pre-land gate (wrapper only here).
- Verifying heuristic (non-LLM) causes.
- Tuning N / the lens set / the quorum beyond the defaults.

## 8. Build order (staged tasks for writing-plans)

1. `verify.py` — `SKEPTIC_LENSES`, `VOTE_SCHEMA`, `refute_prompt`, `MultiSkepticVerifier` (quorum + exclude-errored + strict-majority), `gate_g5_verify` (+ tests, FakeRuntime).
2. `orchestrator.attribute_top_gap` verifier wiring (verify LLM leads → confirm/refute + `add_negative`) (+ tests).
3. CLI `--verify` + `_build_verifier` into `--attribute` + `--boulder` + opt-in smoke + docs (README / CLAUDE.md).
