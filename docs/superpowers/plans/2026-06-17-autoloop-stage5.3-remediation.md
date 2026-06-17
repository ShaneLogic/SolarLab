# Autoloop Stage 5.3 — Codegen Remediation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** Fix the defects the adversarial review found in the committed Stage 5.3 codegen leg so the feature actually works and its containment premise is real, reusing the sound scaffolding (flag-OFF hook, codegen seam, types).

**Architecture:** Redesign the lever artifact as a stable template with a spliced body + a non-overwritten `_ctx` module; move G6's import/compile + flag-ON checks into a subprocess + AST-validate the body (real containment); fix the probe wiring; add runtime-failure degradation; persist the dry-run candidate; reconcile the gate stack with the spec.

**Tech Stack:** Python 3.9+, dataclasses, ast, importlib, subprocess. Reuses Stage 5.3 modules + Stage-2 `SubprocessProbeRunner`/`_probe_worker`.

**Source of findings:** the review result at the workflow output for run `weuyw51ci` (also summarized below). Spec: `docs/superpowers/specs/2026-06-17-autoloop-stage5.3-llm-codegen-design.md`.

**Run all pytest from `perovskite-sim/`.** Default suite excludes `-m slow`. Before each task, READ the current code in the cited files — do not assume; the committed impl is on `main` (HEAD `0f4cbe2`).

**Branch:** work on `feat/autoloop-stage5.3-fix` (created by the controller). Commit per task. Do not push.

---

## Task 1: Lever contract redesign (CRITICAL — feature can't produce a working lever)

**Defect:** `orchestrator.codegen_top_not_promotable` does `target.write_text(lever.body)` = whole file, but the prompt/`GeneratedLever.body` docstring says "body statements only, no def line." Obey → `return` at module scope → SyntaxError; full-`def` → file drops `_LeverContext` → the `mol.py` hook's `from ...lever import adjust_material_arrays, _LeverContext` → ImportError. Tests pass only because FakeCodegen emits full-def bodies + in-process reload keeps the stale `_LeverContext`.

**Files:** `perovskite_sim/autoloop/generated/_ctx.py` (create), `perovskite_sim/autoloop/generated/lever.py` (restructure), `perovskite_sim/autoloop/codegen.py` (splice helper + body-only contract), `perovskite_sim/solver/mol.py` (hook import), `perovskite_sim/autoloop/orchestrator.py` (splice not overwrite). Tests: `tests/unit/autoloop/test_codegen.py`, `tests/unit/autoloop/test_generated_hook.py`.

- [ ] **Step 1 — failing test.** In `test_codegen.py` add:
```python
def test_splice_produces_importable_module(tmp_path):
    from perovskite_sim.autoloop.codegen import splice_lever_body, LEVER_TEMPLATE
    body = "import dataclasses\n" if False else "return dataclasses.replace(arrays, chi=arrays.chi + 0.0)"
    src = splice_lever_body(LEVER_TEMPLATE, body)
    f = tmp_path / "lever.py"; f.write_text(src)
    import ast; ast.parse(src)                       # must be syntactically valid
    assert "def adjust_material_arrays(arrays, ctx):" in src
    assert "return dataclasses.replace(arrays, chi=arrays.chi + 0.0)" in src
    assert "_LeverContext" not in src                # _LeverContext lives in _ctx, not spliced file
```
Run `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_codegen.py::test_splice_produces_importable_module` → FAIL (no `splice_lever_body`/`LEVER_TEMPLATE`).

- [ ] **Step 2 — implement.**
  - Create `generated/_ctx.py` with the frozen `_LeverContext` (move it out of `lever.py`; `@dataclass(frozen=True)` with `x` and `stack` fields — copy the current definition).
  - Rewrite `generated/lever.py` as the canonical template: top imports (`import dataclasses` + `from perovskite_sim.autoloop.generated._ctx import _LeverContext`), then:
    ```python
    def adjust_material_arrays(arrays, ctx):
        # >>> AUTOLOOP BODY
        return arrays
        # <<< AUTOLOOP BODY
    ```
  - In `codegen.py`: add `LEVER_TEMPLATE` (the exact text of the identity `lever.py`) and `splice_lever_body(template, body) -> str` that replaces the region between the `# >>> AUTOLOOP BODY` / `# <<< AUTOLOOP BODY` sentinels with the LLM body, indented 8 spaces (under the def). `GeneratedLever.body` doc + the `ClaudeCodegen` prompt = "ONLY the function body statements (e.g. `return dataclasses.replace(arrays, chi=...)`), no `def`, no imports."
  - `mol.py` hook: import `adjust_material_arrays` from `...generated.lever` and `_LeverContext` from `...generated._ctx` (so `_ctx` is never overwritten).
  - `orchestrator.codegen_top_not_promotable`: write `splice_lever_body(LEVER_TEMPLATE, lever.body)` to `generated/lever.py` (not the raw body).

- [ ] **Step 3 — test passes** + update `test_generated_hook.py` so a non-identity body is supplied as **body-only statements** (spliced) and the flag-ON path imports both names. Run the two test files → PASS.

- [ ] **Step 4 — commit:** `git add -A && git commit -m "fix(autoloop): lever template + body-splice + _ctx module (Stage 5.3 contract, CRITICAL)"`

## Task 2: Real containment — AST-validate body + subprocess G6 (CRITICAL/safety)

**Defect:** G6 imports the LLM module in the PARENT process → arbitrary module-level code executes at import with full FS/network. "Sandbox" was only a path convention.

**Files:** `perovskite_sim/autoloop/codegen.py` (AST validator), `perovskite_sim/autoloop/gates_impl.py` (`gate_g6_build` → subprocess import). Tests: `test_codegen.py`, `tests/unit/autoloop/test_gates_g6.py`.

- [ ] **Step 1 — failing tests.**
```python
def test_validate_lever_body_rejects_imports_and_dangerous_calls():
    from perovskite_sim.autoloop.codegen import validate_lever_body
    import pytest
    for bad in ["import os\nreturn arrays",
                "__import__('os').system('x')\nreturn arrays",
                "open('/tmp/x','w').write('h')\nreturn arrays"]:
        with pytest.raises(ValueError):
            validate_lever_body(bad)
    validate_lever_body("return dataclasses.replace(arrays, chi=arrays.chi + 0.0)")  # ok
```
Run → FAIL (no `validate_lever_body`).

- [ ] **Step 2 — implement.**
  - `validate_lever_body(body) -> None`: `ast.parse` the body wrapped in a dummy `def`; walk the tree, raise `ValueError` on any `ast.Import`/`ast.ImportFrom`, any `ast.Attribute`/`ast.Name` referencing a denylist (`os`, `sys`, `subprocess`, `open`, `exec`, `eval`, `compile`, `__import__`, `globals`, `locals`, `getattr`, `setattr`), or any dunder name. Call it in `ClaudeCodegen.generate` (reject → raise) and before splicing in the orchestrator.
  - `gate_g6_build`: run the import/compile check (and the flag-ON finiteness) in a SUBPROCESS — reuse the `SubprocessProbeRunner`/`_probe_worker` path so the candidate module is imported in a child interpreter, never the parent. The parent only reads the child's structured result.

- [ ] **Step 3 — tests pass.** Add a `test_gates_g6.py` case: a body that passes AST but the subprocess import is exercised (use the real subprocess path with a tmp identity body). Run → PASS.

- [ ] **Step 4 — commit:** `git add -A && git commit -m "fix(autoloop): AST-validate lever body + run G6 import in subprocess (Stage 5.3 containment, CRITICAL)"`

## Task 3: Fix the live `--codegen` probe wiring (CRITICAL — path never runs)

**Defect:** `_flag_on`/`_realized` build `SubprocessProbeRunner(..., gap=None)` → `.run` derefs `self.gap.sweep` → AttributeError → always `(False, ...)` → G6 flag-ON never passes → every real apply = `gates_failed`. `measure="base"` is unrecognized.

**Files:** `scripts/autoloop_run.py` (the `--codegen` dispatch closures), possibly `perovskite_sim/autoloop/orchestrator.py` (thread the selected gap out). Tests: `tests/integration/test_autoloop_codegen.py`.

- [ ] **Step 1 — failing test.** Add an opt-out-free unit-style test that drives the CLI gate closures with a FakeCodegen + a monkeypatched probe (no real solver), asserting `_flag_on` returns `(True, ...)` for a finite stub and that a real `gap` is passed (not None). Run → FAIL.

- [ ] **Step 2 — implement.** Thread the selected `gap` from `codegen_top_not_promotable` to the gate closures; build `SubprocessProbeRunner` with that real `gap` (or a synthetic `Gap` mirroring it) and `measure="gap"` (the only mode `_probe_worker` supports). `_realized` likewise uses a valid gap.

- [ ] **Step 3 — test passes.** Run → PASS.

- [ ] **Step 4 — commit:** `git add -A && git commit -m "fix(autoloop): wire real gap + measure=gap into --codegen probes (Stage 5.3, CRITICAL)"`

## Task 4: Runtime-failure degradation + tracked-only dirty-tree guard (IMPORTANT)

**Defect:** `codegen.generate` + `commit` are uncaught → an LLM failure / dirty-tree refusal crashes the dispatch (spec §7 promised a no-op). The dirty-tree guard flags untracked `??` (e.g. `outputs/`) → raises on essentially every real apply.

**Files:** `perovskite_sim/autoloop/orchestrator.py` (`codegen_top_not_promotable`, `commit_generated_lever`). Tests: `tests/unit/autoloop/test_orchestrator_codegen.py`.

- [ ] **Step 1 — failing tests.** (a) a `FakeCodegen` whose `generate` raises → `codegen_top_not_promotable` returns a no-op `CodegenResult` (status e.g. `"no_target"`/`"dry_run"`), no exception. (b) `commit_generated_lever` with untracked files present in the tree does NOT raise on the untracked entries (only tracked modifications block). Run → FAIL.

- [ ] **Step 2 — implement.** Wrap `codegen.generate(...)` in try/except → no-op result (mirror `LLMAttributor`'s degrade pattern). Wrap the `commit(...)` call → on failure return `gates_failed`/`dry_run` with the error in `rationale`. In the dirty-tree guard, filter `git status --porcelain` to TRACKED modifications (drop `??` lines).

- [ ] **Step 3 — tests pass.** Run → PASS.

- [ ] **Step 4 — commit:** `git add -A && git commit -m "fix(autoloop): degrade codegen failures to no-op + ignore untracked in dirty guard (Stage 5.3)"`

## Task 5: Persist the dry-run candidate patch to outputs_root (IMPORTANT)

**Defect:** `outputs_root`/`reference_path`/`timestamp` are dead params; the candidate lever body + gate report are never written (discarded when identity is restored). `CodegenResult` has no `body`. Spec §2/§6 violated.

**Files:** `perovskite_sim/autoloop/types.py` (`CodegenResult.body`), `perovskite_sim/autoloop/orchestrator.py`. Tests: `tests/unit/autoloop/test_orchestrator_codegen.py`.

- [ ] **Step 1 — failing test.** After a dry-run, assert `outputs_root/codegen-<cycle>/lever.py` and `report.json` exist and contain the spliced body + verdicts; assert `CodegenResult.body` is populated. Run → FAIL.

- [ ] **Step 2 — implement.** Add `body: Optional[str] = None` to `CodegenResult`. In the dry-run + gates_failed branches, write the spliced candidate `lever.py` + a `report.json` (gap, mechanism, rationale, verdicts) under `Path(outputs_root)/f"codegen-{cycle}"` (mirror `guardian_once`'s run_dir). Populate `CodegenResult.body`.

- [ ] **Step 3 — test passes.** Run → PASS.

- [ ] **Step 4 — commit:** `git add -A && git commit -m "fix(autoloop): persist dry-run candidate lever + report to outputs_root (Stage 5.3, spec §2/§6)"`

## Task 6: Reconcile the gate stack with the spec (IMPORTANT)

**Defect:** impl runs only G6 + G3; spec §5 specifies G6 → G1 → G2 → G3 → G4. No limiting-case gate on the highest-risk leg.

**Files:** `perovskite_sim/autoloop/gates_impl.py` (`make_codegen_gate_runner`), the spec doc. Tests: `tests/unit/autoloop/test_gates_g6.py`.

- [ ] **Step 1 — failing test.** Assert `make_codegen_gate_runner`'s verdict list includes a G2 limiting-case verdict (radiative-ceiling / dark-Jsc sanity) in addition to G6 + G3. Run → FAIL.

- [ ] **Step 2 — implement.** Add the G2 limiting-case verdict (reuse the existing G2 limiting check from `gates_impl`) to the codegen gate runner, ordered G6 → G2 → G3. Update spec §2/§5 + the design table to document the final codegen stack and the explicit rationale for omitting G1 (folded into G6 finiteness) and G4 (no prior ablation prediction for a novel lever).

- [ ] **Step 3 — test passes.** Run → PASS.

- [ ] **Step 4 — commit:** `git add -A && git commit -m "fix(autoloop): add G2 limiting to codegen gate stack + reconcile spec §5 (Stage 5.3)"`

## Task 7: Minors batch (IMPORTANT-adjacent + housekeeping)

**Files:** `perovskite_sim/autoloop/orchestrator.py`, `perovskite_sim/autoloop/promote.py`, `perovskite_sim/autoloop/generated/lever.py` docstring. Tests: existing files.

- [ ] **Step 1 — failing tests** for each: (a) after a codegen run, a fresh import of `generated.lever` returns the identity result (no stale-module residue) — add `sys.modules.pop`/`reload` of the identity in the `finally`. (b) a codegen-applied gap is marked `"attempted"` (not `"closed"`) and records the branch/sha. (c) `promote.is_promotable(hyp, ledger) -> bool` exists and `propose_promotion` no longer swallows `FileNotFoundError` (revert the contract-broadening; codegen routing uses `is_promotable`). Run → FAIL.

- [ ] **Step 2 — implement** all three + fix the `lever.py` "no cycle" docstring claim (state it's re-entrant-safe only because `solver.mol` is fully imported by call time) + use ASCII `pass`/`fail` (not `✓`/`✗`) in the `commit_generated_lever` commit-message gate summary.

- [ ] **Step 3 — tests pass.** Run `cd perovskite-sim && python -m pytest -q tests/unit/autoloop` → all green.

- [ ] **Step 4 — commit:** `git add -A && git commit -m "fix(autoloop): codegen minors — stale-module reload, attempted status, is_promotable, ascii provenance (Stage 5.3)"`

---

## Self-Review

**Finding coverage:** lever contract (CRIT) → T1; in-process exec / containment (CRIT/safety) → T2; live probe path (CRIT) → T3; runtime degradation + dirty-guard (IMP) → T4; dry-run persistence (IMP, spec §2/§6) → T5; gate-stack deviation (IMP, spec §5) → T6; stale-module + closed-status + promote.py + ascii + no-cycle (minors) → T7. All 2 critical + 4 important + 5 minor findings mapped.

**Placeholder scan:** none — each task has concrete tests + fix direction + exact commit. (Agents read current code for verbatim edits, since the committed 5.3 source is the substrate.)

**Type consistency:** `splice_lever_body(template, body)`, `LEVER_TEMPLATE`, `validate_lever_body(body)`, `_LeverContext` in `generated/_ctx.py`, `CodegenResult.body`, `is_promotable(hyp, ledger)` — names used consistently across tasks. `measure="gap"` is the `_probe_worker`-supported mode.
