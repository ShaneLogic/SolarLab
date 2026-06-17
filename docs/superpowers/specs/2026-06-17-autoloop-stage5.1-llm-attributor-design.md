# Autoloop Stage 5.1 — Cognition Runtime + LLM Attributor — Design

**Date:** 2026-06-17
**Status:** Design approved, pre-planning
**Parent design:** `docs/superpowers/specs/2026-06-16-autoloop-research-pipeline-design.md` (§8 cognition legs, the deferred LLM attributor)
**Builds on:** Stages 1–4c (all merged to `main`). Reuses the Stage-2 `Attributor` seam (`attribute(gap, matrix, negatives) -> Hypothesis`), `HeuristicAttributor`, `ablation.CANDIDATE_FLAGS`, `types.Hypothesis`/`AblationMatrix`/`Gap`, `ledger.Ledger.is_refuted`.

**Scope note:** the LLM cognition legs decompose into **5.1 runtime + LLM attributor (this)**, **5.2 G5 multi-skeptic verify**, **5.3 LLM codegen implement** — built in that order (runtime foundational; codegen riskiest). 5.1 ships the runtime its successors reuse.

**Feasibility (verified):** `claude` CLI present at `~/.local/bin/claude` (v2.1.179) with `-p/--print` + `--output-format json` + `--model` → a headless cognition runtime is real.

---

## 1. Problem & scope

The deterministic `HeuristicAttributor` can only diagnose causes in its pre-coded
flag menu; it returns `uncertain` when no known flag is the lever. 5.1 adds an
**LLM attributor that engages exactly there** — reads the same ablation matrix and
proposes a **novel** root cause the heuristic can't reach — plus the **pluggable
`CognitionRuntime`** seam (the foundation 5.2/5.3 reuse).

**Decisions (locked during brainstorming):**

| Axis | Decision |
|------|----------|
| Role | **Fallback when the heuristic is uncertain.** Heuristic runs first (free); the LLM is called ONLY on its `uncertain` verdicts → bounded cost, clear division of labor. |
| Trust | **Always a lead (`verdict="uncertain"`).** The LLM proposes; it never auto-confirms. Confirmation is G5's job (5.2) / human review. Stage 3 acts only on `confirmed`, so nothing acts on an unverified LLM claim. |
| Runtime | **Pluggable `CognitionRuntime`** — `FakeRuntime` (tests) + `ClaudeCliRuntime` (`claude -p --output-format json`), schema-validated in-process, timeout-guarded, 1 retry. |
| Default | `--llm` **opt-in.** Default attributor stays deterministic (no LLM, no cost). |

**Explicitly deferred:** G5 multi-skeptic verify (5.2); LLM codegen (5.3); LLM
confirming a cause; agentic experiment-direction (LLM requesting new ablations);
LLM in the implement/propose step.

## 2. Architecture

```
attribute(gap, matrix, negatives):
   hyp = HeuristicAttributor.attribute(...)
   if hyp.verdict != "uncertain":  return hyp        # heuristic found a flag lever → no LLM
   # heuristic uncertain → LLM fallback on the SAME matrix
   try:
       out = runtime.complete(build_attribution_prompt(gap, matrix, negatives), ATTRIBUTION_SCHEMA)
   except Exception:
       return hyp                                     # never block/crash on the LLM
   → Hypothesis(cause, mechanism, evidence, verdict="uncertain")   # ALWAYS a lead
```

Two new modules:
```
perovskite_sim/autoloop/
  cognition.py        CognitionRuntime protocol + FakeRuntime + ClaudeCliRuntime + _validate
  llm_attribution.py  ATTRIBUTION_SCHEMA + build_attribution_prompt + LLMAttributor
scripts/autoloop_run.py  + --llm [--llm-model] ; _build_attributor(ns) wired into --attribute + --boulder
```

## 3. CognitionRuntime seam (`cognition.py`)

```python
class CognitionRuntime(Protocol):
    def complete(self, prompt: str, schema: dict) -> dict: ...
```

**`ClaudeCliRuntime(model="sonnet", timeout_s=180)`:**
1. `subprocess.run(["claude","-p",prompt,"--output-format","json","--model",model], capture_output=True, text=True, timeout=timeout_s)`. `TimeoutExpired` / rc≠0 → `RuntimeError`.
2. Parse claude's envelope: `json.loads(stdout)["result"]` → the assistant text.
3. Strip ```` ```json ```` fences → `json.loads` → `_validate(obj, schema)`.
4. On parse/validate failure → **retry once** (append "Return ONLY the JSON object, no prose."); still bad → `RuntimeError`.
5. Return the validated dict.

`_validate(obj, schema)` — in-process, no dependency: checks `schema["required"]` keys present + `cause` ∈ the enum; raises `ValueError` on mismatch.

**`FakeRuntime(canned)`** — `complete` returns `canned` (dict) or `canned(prompt)` (callable). Tests + zero cost.

## 4. LLM attributor (`llm_attribution.py`)

**`ATTRIBUTION_SCHEMA`:**
```python
ATTRIBUTION_SCHEMA = {
    "required": ["cause", "mechanism"],
    "cause_enum": ["bug", "numerics", "physics", "data"],
}
```
(consumed by `_validate`).

**`build_attribution_prompt(gap, matrix, negatives)`** assembles, as text:
- the gap (metric, sweep, SolarLab-vs-reference value, kind);
- the **ablation matrix** — each probe: name, kind (flag/grid/limiting), delta, ok;
- the **candidate flag menu** (`ablation.CANDIDATE_FLAGS`) — "these were tried; none a clear lever";
- the **negatives ledger** (refuted approaches) — "REFUTED — do NOT propose any of these";
- instruction: *"The deterministic heuristic found no known-flag lever for this gap. Propose the single most likely **novel** root cause — a physics term / numerical issue / bug / data problem **not behind any existing flag**. Be specific + mechanistic (name the term, equation, or site). Output ONLY a JSON object: {cause, mechanism, evidence_for, evidence_against, confidence}."*

**`LLMAttributor`:**
```python
class LLMAttributor:
    def __init__(self, runtime: CognitionRuntime, *, heuristic=None):
        self.runtime = runtime
        self.heuristic = heuristic or HeuristicAttributor()

    def attribute(self, gap, matrix, negatives) -> Hypothesis:
        hyp = self.heuristic.attribute(gap, matrix, negatives)
        if hyp.verdict != "uncertain":
            return hyp
        try:
            out = self.runtime.complete(
                build_attribution_prompt(gap, matrix, negatives), ATTRIBUTION_SCHEMA)
        except Exception as exc:                          # never block on the LLM
            logger.warning("LLM attribution failed for %s: %r — keeping heuristic uncertain",
                           gap.id, exc)
            return hyp
        mechanism = out["mechanism"]
        lead = (f"LLM novel-cause lead (confidence {out.get('confidence', '?')})",)
        ev_for = tuple(out.get("evidence_for", ())) + lead
        ev_against = tuple(out.get("evidence_against", ()))
        if negatives.is_refuted(mechanism):
            ev_against = ev_against + ("matches a REFUTED approach in the negatives ledger",)
        return Hypothesis(gap_id=gap.id, cause=out["cause"], mechanism=mechanism,
                          evidence_for=ev_for, evidence_against=ev_against,
                          verifier_votes=0, verdict="uncertain")   # ALWAYS a lead
```

## 5. CLI wiring

`--llm` + `--llm-model` (default `sonnet`). A helper:
```python
def _build_attributor(ns):
    if not ns.llm:
        return HeuristicAttributor()
    from perovskite_sim.autoloop.cognition import ClaudeCliRuntime
    from perovskite_sim.autoloop.llm_attribution import LLMAttributor
    return LLMAttributor(ClaudeCliRuntime(model=ns.llm_model))
```
wired into **both** the `--attribute` and `--boulder` dispatch (they already build an
attributor). Default = `HeuristicAttributor` (deterministic, no LLM, no cost).

```bash
python scripts/autoloop_run.py --attribute --llm            # LLM fallback on uncertain gaps
python scripts/autoloop_run.py --boulder --llm --llm-model sonnet
```

## 6. Error handling

- `claude` CLI missing → `FileNotFoundError` → `LLMAttributor` catches → heuristic uncertain + log (graceful degrade; `--llm` is safe even where claude isn't installed).
- timeout / bad-JSON-after-retry / rc≠0 → `RuntimeError` → caught → heuristic uncertain.
- The LLM **never confirms**, never crashes the loop; the negatives-guard is always applied.

## 7. Testing

- **`cognition`:** `ClaudeCliRuntime` with monkeypatched `subprocess.run` → parses `{"result": '{"cause":"physics","mechanism":"missing Auger term"}'}` → validated dict; markdown-fenced result stripped + parsed; invalid-JSON first → retries once → good second; still-bad → raises; `TimeoutExpired`/rc≠0 → raises. `_validate` rejects missing-required + bad `cause` enum. `FakeRuntime` returns canned.
- **`llm_attribution`:** heuristic-confirmed → runtime NOT called (spy FakeRuntime records calls); heuristic-uncertain → runtime called → `uncertain` Hypothesis carrying the LLM cause/mechanism + the lead tag; runtime raises → falls back to heuristic uncertain; LLM mechanism matching a seeded refuted approach → `evidence_against` flags it (still uncertain). `build_attribution_prompt` contains the gap id, ≥1 matrix probe name, a negatives approach, and a CANDIDATE_FLAGS flag.
- **integration smoke (opt-in only):** `@pytest.mark.slow` + `skipif(not shutil.which("claude") or not os.environ.get("SOLARLAB_LLM_SMOKE"))` so CI never nests `claude`; when run, calls the real `ClaudeCliRuntime` on a synthetic uncertain matrix → asserts a parseable `uncertain` Hypothesis with a non-empty mechanism. (Honest: a real-LLM test is cost + nondeterministic → opt-in.)

## 8. Out of scope / deferred

- G5 multi-skeptic adversarial verify (5.2).
- LLM codegen for non-promotable levers (5.3).
- LLM confirming a cause (always uncertain in 5.1).
- Agentic experiment-direction (LLM requesting new ablations).
- LLM in the implement / propose-fix step (Stage 3 stays flag-promotion).

## 9. Build order (staged tasks for writing-plans)

1. `cognition.py` — `CognitionRuntime` protocol + `FakeRuntime` + `_validate` + `ClaudeCliRuntime` (+ tests, monkeypatched subprocess).
2. `llm_attribution.py` — `ATTRIBUTION_SCHEMA` + `build_attribution_prompt` + `LLMAttributor` (+ tests, FakeRuntime).
3. CLI `--llm`/`--llm-model` + `_build_attributor` wired into `--attribute` + `--boulder` + opt-in real smoke + docs (README / CLAUDE.md).
