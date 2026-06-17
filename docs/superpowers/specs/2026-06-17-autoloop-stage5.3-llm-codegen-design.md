# Autoloop Stage 5.3 — LLM Codegen for Non-Promotable Levers — Design

**Date:** 2026-06-17
**Status:** Design approved, pre-planning
**Parent design:** `docs/superpowers/specs/2026-06-16-autoloop-research-pipeline-design.md` (cognition legs)
**Builds on:** Stages 1–4c + 5.1 + 5.2 (all merged to `main`). Reuses 5.1's `cognition.CognitionRuntime`/`FakeRuntime`, Stage-3 `promote`/`implement_top_confirmed`/`commit_promotion`/gate stack, Stage-2 ablation matrix, `types.Hypothesis`/`NegativeResult`/`GateVerdict`.

**Scope note:** cognition legs = 5.1 (runtime + attributor, done), 5.2 (G5 verify, done), **5.3 (LLM codegen, this — the last leg).**

---

## 1. Problem & scope

When G5 (5.2) promotes an LLM lead to `confirmed` but its mechanism has no existing
flag in Stage-3's `promote.FLAG_TO_CONFIG_KEY`, `implement_top_confirmed` dead-ends at
`ImplementResult("not_promotable", ..., note=hyp.mechanism)` (`orchestrator.py:251`) —
mechanism noted, nothing acted on. 5.3 routes that confirmed-but-not-promotable gap to
**codegen**: an LLM writes a flag-gated band-parameter transform, the deterministic spine
verifies it, and (with `--apply`) lands it on a **fresh feature branch** for human merge.

This is the highest-risk leg — an LLM writes real physics code. Safety rests entirely on
the autoloop philosophy: **a new lever defaults OFF, so the spine guarantees it is inert
until explicitly enabled, and the human merges the branch.** Default behavior with no
codegen is deterministic, no LLM, no cost (opt-in `--codegen`, exactly like 5.1's `--llm`).

**Decisions (locked during brainstorming):**

| Axis | Decision |
|------|----------|
| Autonomy | A verified, fully-gate-passing patch auto-commits to a **fresh `feat/autoloop-gen-<gapslug>` branch** (refuses main/current, never pushes); human merges. Default = dry-run (candidate patch + gate report, no commit). |
| Codegen target | A **sandboxed `autoloop/generated/` module** plus ONE pre-wired flag-gated hook in the solver. Core solver files are never touched by codegen → G0 (flag-OFF bit-identical) is *structural*. |
| Hook seam | **Build-time material-array transform.** The hook fires once at the end of `solver/mol.build_material_arrays` (not per Newton iteration) → lowest stiffness risk; the proven V_oc locus (where the DOS-fold lives). |
| LLM write surface | The LLM writes **only the body of one function** — `adjust_material_arrays(arrays, ctx) -> arrays`. It never touches flag plumbing, config keys, or the hook call. |

**Explicitly deferred:** per-node RHS / boundary hooks; the 2D path (`build_material_arrays_2d`);
the LLM proposing *new* flags or multi-function levers; auto-push / auto-merge; tuning.

## 2. Architecture (flow)

```
not_promotable confirmed gap
  -> codegen_top_not_promotable(*, ..., codegen, apply=False, ...)
       hyp = the confirmed not-promotable hypothesis (top by gap_mag)
       lever = codegen.generate(gap, hyp, matrix)        # GeneratedLever(body, rationale)
       splice lever.body into autoloop/generated/lever.py between sentinels (template
           owns the def header; _LeverContext lives in the never-overwritten _ctx module)
       verdicts = gate_runner(...)  # G6 build FIRST -> G2 limiting ->
                                    # G3 badness-improves (flag ON)
       if not all passed:
           restore identity lever.py; add_negative(mechanism, "codegen gate(s) failed: ...")
           -> CodegenResult("gates_failed", ...)
       elif apply:
           commit lever.py (+ provenance) to a NEW feat/autoloop-gen-<slug> branch
           restore identity lever.py on the working branch; mark gap closed
           -> CodegenResult("applied", branch=..., sha=...)
       else:                                              # dry-run default
           write candidate patch + report; restore identity lever.py
           -> CodegenResult("dry_run", ...)
```

G5 already adjudicated the *cause* upstream (5.2); 5.3 verifies the *generated code*.

## 3. Codegen seam (`autoloop/codegen.py`) — the deferred-LLM part

```python
@dataclass(frozen=True)
class GeneratedLever:
    body: str          # source of the adjust_material_arrays function (def included)
    rationale: str     # one-paragraph why, for the provenance/report

CODEGEN_SCHEMA = {"required": ["body", "rationale"]}   # consumed by cognition._validate

class Codegen(Protocol):
    def generate(self, gap, hyp, matrix) -> GeneratedLever: ...

class FakeCodegen:                 # tests: returns a canned body
    def __init__(self, body, rationale="fake"): ...
    def generate(self, gap, hyp, matrix) -> GeneratedLever: ...

class ClaudeCodegen:               # live: prompts the 5.1 runtime
    def __init__(self, runtime: CognitionRuntime): ...
    def generate(self, gap, hyp, matrix) -> GeneratedLever:
        # prompt presents the confirmed cause+mechanism, the gap, the ablation matrix,
        # the EXACT function contract (signature, MaterialArrays fields it may shift,
        # "return arrays unchanged if unsure", "pure function, no I/O, no globals"),
        # and asks for ONLY {body, rationale} JSON. Validates + strips fences (5.1).
        # Any runtime failure raises -> caller degrades to no-op (dry-run, no commit).
```

**Containment:** the prompt fixes the signature and forbids touching anything but the
function body. The body is the *only* artifact written, and only into
`autoloop/generated/lever.py`.

## 4. Pre-wired hook (the deterministic part 5.3 builds first)

5.3 wires, once and deterministically (NOT via the LLM):

1. **Generic flag + config key.** `models/device.py`: `autoloop_generated_lever: bool = False`
   (default OFF). Loaders (`scaps_compat/loader.py`, `models/config_loader.py`) read the
   `autoloop_generated_lever` device key with the existing string-coerce idiom.
2. **Hook call** at the end of `solver/mol.build_material_arrays`, guarded exactly like the
   DOS flag (`mol.py:750-751`):
   ```python
   if getattr(stack, "autoloop_generated_lever", False) or os.environ.get("SOLARLAB_AUTOLOOP_GEN") == "1":
       from perovskite_sim.autoloop.generated.lever import adjust_material_arrays
       arrays = adjust_material_arrays(arrays, _LeverContext(x=x, stack=stack))
   ```
   The import is *inside* the guard → when the flag is off the generated module is never
   even imported, so a broken body cannot affect the legacy path.
3. **Default identity lever.** `autoloop/generated/lever.py` ships with
   `def adjust_material_arrays(arrays, ctx): return arrays` and `autoloop/generated/__init__.py`.
4. **`_LeverContext`** (frozen) carries `x` (grid) and `stack` (DeviceStack) — read-only
   inputs the body may use to compute a transform.

Because the flag is OFF in every legacy/parity config, the hook is never called →
**structurally bit-identical → G0 trivially green.** The LLM only ever overwrites the
identity body in `lever.py`.

## 5. Gates (`gates_impl.py` additions)

A new **G6 build gate runs FIRST** (cheap fail-fast before the expensive physics gates):

- **imports/compiles:** the candidate is AST-validated (body-only statements; no imports /
  denylisted names / dunders / `def`) then imported in a FRESH SUBPROCESS (never the parent
  process); a `SyntaxError`/`ImportError` → G6 fail.
- **flag-OFF bit-identical:** reuse the existing Stage-3 G0 golden-suite runner (with the
  flag off, the new lever module is not imported → must be bit-identical).
- **flag-ON runs:** a parity J-V sweep with `SOLARLAB_AUTOLOOP_GEN=1` completes, all metrics
  finite (no NaN/Inf), `voc_bracketed` holds.

The implemented flag-ON stack is **G6 → G2 limiting (limiting cases hold) → G3 (badness
improves vs the sense-time baseline)**. G1 numerics is folded into G6's flag-OFF
bit-identical + flag-ON finiteness check; G4 realized-reconciles-predicted is N/A — a
brand-new LLM lever has no prior ablation prediction, so G3-improves is the honest benefit
check. Any failure → restore identity body + `add_negative` (anti-thrash) + `gates_failed`.

## 6. Landing (`orchestrator.codegen_top_not_promotable` + CLI)

- New orchestrator fn mirrors `implement_top_confirmed` but for the not-promotable branch:
  selects the top confirmed gap whose mechanism is *not* promotable, runs §2.
- **Branch isolation:** on `--apply`, a `commit_generated_lever(...)` helper creates a fresh
  `feat/autoloop-gen-<gapslug>` branch off the current HEAD, commits `generated/lever.py`
  (+ a provenance note: gap, mechanism, rationale, gate verdicts), then returns to the
  original branch and restores the identity body there. **Refuses to commit onto main or
  the current branch; never pushes.** (Extends Stage-4a's main-guard discipline.)
- **CLI:** `--codegen` (opt-in, typically with `--llm`) builds `ClaudeCodegen(ClaudeCliRuntime(ns.llm_model))`
  (else `None`); `--apply` reused for the commit; default dry-run writes the candidate patch
  + gate report to `outputs_root`. Wired into a `--codegen` dispatch (and, later, the boulder
  loop — out of scope for this stage's wiring beyond the standalone command).

```bash
cd perovskite-sim
python scripts/autoloop_run.py --codegen --llm            # dry-run: candidate patch + gate report
python scripts/autoloop_run.py --codegen --llm --apply    # commit to a fresh feat/autoloop-gen-* branch
```

## 7. Error handling

- Codegen runtime failure (claude missing/timeout/bad JSON) → raises → caller treats as
  no-lever → dry-run no-op, never a commit.
- A body that breaks the flag-OFF path → impossible by construction (import is guarded), but
  G6's bit-identical check is the backstop; fail → restore + negative.
- A body that crashes/NaNs flag-ON → G6 flag-ON check fails → restore + negative.
- Every path restores the identity `lever.py` on the working branch (try/finally), so a run
  never leaves a non-identity body behind on the working tree.
- Refuse to commit on main/current branch; never push (human merges).

## 8. Testing

- **`codegen`:** `FakeCodegen` returns a canned body; `GeneratedLever`/`CODEGEN_SCHEMA`
  shapes; `ClaudeCodegen.generate` validates + fence-strips (FakeRuntime); a runtime that
  raises → `generate` propagates (caller degrades).
- **hook:** with the flag off, `build_material_arrays` output is unchanged and the generated
  module is not imported; with the flag on + identity body, output is byte-identical; with a
  non-identity body, the targeted array field shifts as written.
- **gates:** G6 — identity body → pass + bit-identical; a syntactically broken body → fail
  (import); a body that NaNs flag-ON → fail; reuse-path G2/G3 exercised via the existing
  gate tests with the flag on.
- **orchestrator/landing:** `codegen_top_not_promotable` with a `FakeCodegen` + fake gate
  runner: all-pass + `--apply` → fresh-branch commit (fake committer, assert branch name +
  identity restored on working branch); gate-fail → `add_negative` (assert via reloaded
  ledger); no codegen → dry-run no-op; refuses main.
- **CLI:** `_build_codegen(ns)` → `None` without `--codegen`, `ClaudeCodegen` with it.
- **integration smoke (opt-in):** `@pytest.mark.slow` + `skipif(not which("claude") or not
  SOLARLAB_LLM_SMOKE)` → real `ClaudeCodegen` on a synthetic not-promotable gap → a
  syntactically valid body that passes G6 bit-identical (flag off). (Cost → opt-in.)

## 9. Out of scope / deferred

- Per-node RHS / boundary-condition hooks (stiffness); the 2D `build_material_arrays_2d` path.
- LLM proposing *new* flags, new config keys, or multi-function levers.
- Auto-push / auto-merge of the generated branch; boulder-loop wiring of `--codegen`.
- Tuning the prompt / the gate thresholds beyond reuse of the Stage-3 defaults.

## 10. Build order (staged tasks for writing-plans)

1. **Pre-wired hook (deterministic):** `device.py` flag + loaders + `build_material_arrays`
   guarded hook + `autoloop/generated/{__init__,lever}.py` identity + `_LeverContext`
   (+ tests: flag-off bit-identical, flag-on identity byte-identical, non-identity body shifts).
2. **`codegen.py`:** `GeneratedLever`, `CODEGEN_SCHEMA`, `Codegen` protocol, `FakeCodegen`,
   `ClaudeCodegen` (+ tests).
3. **G6 build gate** in `gates_impl.py` (imports + reuse-G0 bit-identical + flag-ON runs)
   (+ tests).
4. **`orchestrator.codegen_top_not_promotable`** + `commit_generated_lever` (fresh-branch,
   refuse-main, identity-restore) (+ tests).
5. **CLI `--codegen` + `_build_codegen`** + opt-in smoke + docs (CLAUDE.md / README) (+ tests).
