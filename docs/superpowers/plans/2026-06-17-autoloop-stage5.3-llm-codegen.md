# Autoloop Stage 5.3 — LLM Codegen for Non-Promotable Levers — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When G5 confirms a cause whose mechanism has no existing promotable flag, let an LLM write a flag-gated band-parameter transform into a sandboxed `autoloop/generated/lever.py`; the deterministic spine verifies it (G6 build + flag-OFF bit-identical + flag-ON improves) and, with `--apply`, commits it to a fresh `feat/autoloop-gen-*` branch for human merge.

**Architecture:** A pre-wired, default-OFF flag (`autoloop_generated_lever` / env `SOLARLAB_AUTOLOOP_GEN`) gates a single hook at the end of `solver/mol.build_material_arrays`; with the flag off the generated module is never imported → structurally bit-identical. The LLM writes ONLY the body of `adjust_material_arrays(arrays, ctx)`. `codegen_top_not_promotable` routes a confirmed-but-not-promotable gap through codegen → gate stack → fresh-branch commit. Default behavior is deterministic, no LLM, no cost (opt-in `--codegen`, like 5.1's `--llm`).

**Tech Stack:** Python 3.9+, numpy, dataclasses, subprocess (git). Reuses 5.1 `cognition.CognitionRuntime`/`ClaudeCliRuntime`/`FakeRuntime`, Stage-3 `promote.propose_promotion`/`parse_lever`, `gates_impl.gate_g0_bit_identical`/`gap_baseline_badness`, `orchestrator.commit_promotion` (template) + `_parse_porcelain_paths`, `types.{Gap,Hypothesis,NegativeResult,GateVerdict,AblationMatrix}`. No new third-party deps.

---

## Design contract (read before starting)

Verified on `main` (quote exactly — do NOT invent symbols):

- `solver/mol.py`: `MaterialArrays` is `@dataclass(frozen=True)` (use `dataclasses.replace`). Its fields include `chi`, `Eg`, `ni_sq`, `tau_n`, `tau_p`, `B_rad` (per-node `np.ndarray`). `build_material_arrays(x: np.ndarray, stack: DeviceStack) -> MaterialArrays` ends with a single multi-line `return MaterialArrays(...)` at **line 1223** (the constructor spans ~1223–~1295, the last statement of the function). `import os` (line 4) and `from dataclasses import dataclass` (line 2) are present. The DOS flag guard idiom (lines 749–752):
  ```python
  _dos_band = bool(
      getattr(stack, "dos_band_potentials", True)
      or os.environ.get("SOLARLAB_DOS_BAND") == "1"
  ) and sim_mode.name != "legacy"
  ```
- `models/device.py`: `@dataclass(frozen=True) class DeviceStack` (line 52) holds `dos_band_potentials: bool = True` (line 90). `electrical_layers(stack)` exists.
- Loaders both `return DeviceStack(...)` with the idiom (`models/config_loader.py:151-154`, `scaps_compat/loader.py:141-144`):
  ```python
  dos_band_potentials=(
      str(dev.get("dos_band_potentials", True)).strip().lower()
      in ("true", "1", "yes", "on")
  ),
  ```
- `autoloop/types.py`: `Gap` (frozen) has `.with_status(status, *, last_attempt_cycle=None)` and `.with_mechanism(mechanism)`. `Hypothesis(gap_id, cause, mechanism, evidence_for=(), evidence_against=(), verifier_votes=0, verdict="uncertain", cycle=0, predicted_delta=0.0)`. `NegativeResult(approach, why_failed, evidence, never_retry=True)`. `GateVerdict(name, passed, reason)`. `ImplementResult(status, hypothesis_gap_id, device_key, gate_verdicts, committed_sha, note="")`. `AblationMatrix(gap_id, baseline_val, probes, skipped=())`.
- `autoloop/promote.py`: `parse_lever(mechanism) -> Optional[str]` (regex `SOLARLAB_[A-Z_]+`); `propose_promotion(hypothesis, ledger, config_path) -> Optional[ConfigEdit]` returns `None` when not confirmed / no SOLARLAB flag / flag not in `FLAG_TO_CONFIG_KEY` / refuted. **"Not promotable" ≡ `propose_promotion(...) is None` for a confirmed hypothesis.**
- `autoloop/gates_impl.py`: `gate_g0_bit_identical(golden_runner: Callable[[], tuple[bool,str]]) -> GateVerdict`; `gap_baseline_badness(gap) -> float`; `make_implement_gate_runner(*, measure_badness, l0_runner=None, regression_paths=None)`. `run_l0` is imported from `autoloop.ladder`.
- `autoloop/orchestrator.py`: `commit_promotion(edit, gap, hypothesis, verdicts, *, git_cwd=None) -> str` (refuse main/master, dirty-tree guard, scoped `git commit -- <path>`). Helper `_parse_porcelain_paths([line]) -> list[str]` exists. `implement_top_confirmed` returns `ImplementResult("not_promotable", gap.id, None, (), None, note=hyp.mechanism)` at line 251 — the dead-end this stage fills. Imports at top include `dataclasses, subprocess, Path, Callable/Optional`, `propose_promotion/apply_edit/revert_edit`, `Ledger`, `NegativeResult`, the `types` block.
- `autoloop/cognition.py`: `ClaudeCliRuntime(model="sonnet")`, `FakeRuntime(canned)`, `.complete(prompt, schema) -> dict` (validates `schema["required"]`, strips markdown fences).
- `scripts/autoloop_run.py`: `_build_attributor(ns)` / `_build_verifier(ns)` (return `None` unless the flag is set, else build over `ClaudeCliRuntime(model=ns.llm_model)`); `parse_args` adds `--llm`, `--llm-model` (default `sonnet`), `--verify`, `--apply`; `--implement` dispatch builds `make_implement_gate_runner(measure_badness=...)` and calls `implement_top_confirmed(...)`.
- **Run all commands from `perovskite-sim/`.** Default suite excludes `-m slow`. The hook test uses the real build pattern from `tests/unit/solver/test_interface_plane_closure.py`:
  ```python
  from perovskite_sim.discretization.grid import multilayer_grid, Layer
  from perovskite_sim.models.device import electrical_layers
  from perovskite_sim.scaps_compat import load_scaps_yaml
  from perovskite_sim.solver.mol import build_material_arrays
  _V2 = "configs/scaps_mirror_v2.yaml"
  def _build(stack):
      elec = electrical_layers(stack)
      x = multilayer_grid([Layer(thickness=L.thickness, N=30 // len(elec)) for L in elec])
      return x, build_material_arrays(x, stack)
  ```

## File Structure

```
perovskite_sim/autoloop/generated/__init__.py   (new) package marker
perovskite_sim/autoloop/generated/lever.py      (new) _LeverContext + identity adjust_material_arrays
perovskite_sim/models/device.py                 + autoloop_generated_lever: bool = False
perovskite_sim/models/config_loader.py          + loader kwarg
perovskite_sim/scaps_compat/loader.py           + loader kwarg
perovskite_sim/solver/mol.py                     + flag-gated hook at end of build_material_arrays
perovskite_sim/autoloop/codegen.py              (new) GeneratedLever, Codegen, FakeCodegen, ClaudeCodegen
perovskite_sim/autoloop/gates_impl.py           + gate_g6_build + make_codegen_gate_runner
perovskite_sim/autoloop/types.py                + CodegenResult
perovskite_sim/autoloop/orchestrator.py         + codegen_top_not_promotable + commit_generated_lever + _gap_slug
scripts/autoloop_run.py                          + --codegen + _build_codegen + dispatch
tests/unit/autoloop/test_generated_hook.py      (new)
tests/unit/autoloop/test_codegen.py             (new)
tests/unit/autoloop/test_gates_g6.py            (new)
tests/unit/autoloop/test_orchestrator_codegen.py(new)
tests/integration/test_autoloop_codegen.py      (new, opt-in slow)
```

---

## Task 1: Pre-wired flag-gated hook (deterministic seam)

**Files:**
- Create: `perovskite_sim/autoloop/generated/__init__.py`, `perovskite_sim/autoloop/generated/lever.py`
- Modify: `perovskite_sim/models/device.py`, `perovskite_sim/models/config_loader.py`, `perovskite_sim/scaps_compat/loader.py`, `perovskite_sim/solver/mol.py`
- Test: `tests/unit/autoloop/test_generated_hook.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/autoloop/test_generated_hook.py
import dataclasses
import numpy as np

from perovskite_sim.discretization.grid import multilayer_grid, Layer
from perovskite_sim.models.device import electrical_layers
from perovskite_sim.scaps_compat import load_scaps_yaml
from perovskite_sim.solver.mol import build_material_arrays
from perovskite_sim.autoloop.generated.lever import adjust_material_arrays, _LeverContext

_V2 = "configs/scaps_mirror_v2.yaml"


def _build(stack):
    elec = electrical_layers(stack)
    x = multilayer_grid([Layer(thickness=L.thickness, N=30 // len(elec)) for L in elec])
    return x, build_material_arrays(x, stack)


def test_identity_default_returns_same_object():
    sentinel = object()
    assert adjust_material_arrays(sentinel, _LeverContext(x=None, stack=None)) is sentinel


def test_flag_off_bit_identical(monkeypatch):
    monkeypatch.delenv("SOLARLAB_AUTOLOOP_GEN", raising=False)
    base = load_scaps_yaml(_V2)
    _, a_off = _build(dataclasses.replace(base, autoloop_generated_lever=False))
    _, a_on = _build(dataclasses.replace(base, autoloop_generated_lever=True))  # identity body
    assert np.array_equal(a_off.chi, a_on.chi)
    assert np.array_equal(a_off.Eg, a_on.Eg)
    assert np.array_equal(a_off.ni_sq, a_on.ni_sq)


def test_flag_on_nonidentity_body_shifts_chi(monkeypatch):
    monkeypatch.delenv("SOLARLAB_AUTOLOOP_GEN", raising=False)
    base = load_scaps_yaml(_V2)
    _, a_off = _build(dataclasses.replace(base, autoloop_generated_lever=False))
    import perovskite_sim.autoloop.generated.lever as lev

    def _shift(arrays, ctx):
        return dataclasses.replace(arrays, chi=arrays.chi + 0.1)

    monkeypatch.setattr(lev, "adjust_material_arrays", _shift)
    _, a_on = _build(dataclasses.replace(base, autoloop_generated_lever=True))
    assert np.allclose(a_on.chi, a_off.chi + 0.1)


def test_env_flag_triggers_hook(monkeypatch):
    base = load_scaps_yaml(_V2)
    _, a_off = _build(dataclasses.replace(base, autoloop_generated_lever=False))
    import perovskite_sim.autoloop.generated.lever as lev
    monkeypatch.setattr(lev, "adjust_material_arrays",
                        lambda arrays, ctx: dataclasses.replace(arrays, Eg=arrays.Eg + 0.05))
    monkeypatch.setenv("SOLARLAB_AUTOLOOP_GEN", "1")
    _, a_on = _build(dataclasses.replace(base, autoloop_generated_lever=False))  # off on stack, on via env
    assert np.allclose(a_on.Eg, a_off.Eg + 0.05)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_generated_hook.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'perovskite_sim.autoloop.generated'` (and, once that exists, `TypeError`/`unexpected keyword argument 'autoloop_generated_lever'` from `dataclasses.replace`).

- [ ] **Step 3a: Create the generated package**

```python
# perovskite_sim/autoloop/generated/__init__.py
# Autoloop Stage 5.3 generated-lever sandbox package. The LLM writes ONLY the
# body of adjust_material_arrays in lever.py; nothing else lives here.
```

```python
# perovskite_sim/autoloop/generated/lever.py
"""Sandboxed extension point for autoloop Stage 5.3 LLM codegen.

`solver/mol.build_material_arrays` calls `adjust_material_arrays` ONCE on the
assembled MaterialArrays, but only when the `autoloop_generated_lever` device
flag (or env `SOLARLAB_AUTOLOOP_GEN=1`) is set. The default body is the identity
transform; with the flag off (every legacy/parity config) this module is never
imported, so the solver is bit-identical. Stage 5.3 codegen overwrites ONLY the
body of `adjust_material_arrays` — never the signature, the context, or the hook."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _LeverContext:
    """Read-only inputs a generated lever may use: the spatial grid `x` and the
    `DeviceStack` (`stack`). Typed as `object` to keep this module dependency-light
    (no solver/models import → no cycle when imported inside the hook guard)."""
    x: object
    stack: object


def adjust_material_arrays(arrays, ctx):
    """Identity transform (default). A generated lever returns
    `dataclasses.replace(arrays, <field>=...)` with shifted band parameters
    (chi / Eg / ni_sq / tau_n / tau_p / B_rad ...). MUST be pure: no I/O, no
    globals, no mutation of `arrays` (it is frozen) — return a replaced copy."""
    return arrays
```

- [ ] **Step 3b: Add the flag to `DeviceStack`**

In `perovskite_sim/models/device.py`, immediately after the line `    dos_band_potentials: bool = True` (line 90), insert:

```python
    # Autoloop Stage 5.3 codegen lever (2026-06). When True (or env
    # ``SOLARLAB_AUTOLOOP_GEN=1``), build_material_arrays calls the sandboxed
    # ``autoloop.generated.lever.adjust_material_arrays`` once on the assembled
    # MaterialArrays. Default False = the generated module is never imported →
    # bit-identical. The autoloop writes the lever body; a human merges the branch.
    autoloop_generated_lever: bool = False
```

- [ ] **Step 3c: Add the loader kwarg (both loaders)**

In `perovskite_sim/models/config_loader.py`, immediately after the `dos_band_potentials=( ... ),` block (lines 151-154), insert:

```python
        autoloop_generated_lever=(
            str(dev.get("autoloop_generated_lever", False)).strip().lower()
            in ("true", "1", "yes", "on")
        ),
```

In `perovskite_sim/scaps_compat/loader.py`, immediately after the `dos_band_potentials=( ... ),` block (lines 141-144), insert the identical kwarg block.

- [ ] **Step 3d: Wire the hook into `build_material_arrays`**

In `perovskite_sim/solver/mol.py`, change the opening of the final return (line 1223):

```python
    return MaterialArrays(
```
to:
```python
    arrays = MaterialArrays(
```

Then locate the closing `)` that ends that `MaterialArrays(...)` constructor (the final statement of `build_material_arrays`, ~line 1295) and immediately after it append (4-space indent, matching the function body):

```python

    if getattr(stack, "autoloop_generated_lever", False) or os.environ.get("SOLARLAB_AUTOLOOP_GEN") == "1":
        from perovskite_sim.autoloop.generated.lever import (
            adjust_material_arrays, _LeverContext)
        arrays = adjust_material_arrays(arrays, _LeverContext(x=x, stack=stack))
    return arrays
```

(The import is *inside* the guard, so with the flag off the generated module is never imported — the legacy path cannot be affected by a broken body.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_generated_hook.py`
Expected: PASS (4 tests).

- [ ] **Step 5: Run the solver regression guard to confirm flag-OFF bit-identity**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop tests/unit/solver/test_interface_plane_closure.py`
Expected: the autoloop suite green; `test_interface_plane_closure.py` shows the **same 1 pre-existing failure** as on `main` (`test_mat_caches_require_parity_configuration`, a stale `fcbff18` baseline) and no NEW failures. (The hook adds a default-False field; it must not change any other result.)

- [ ] **Step 6: Commit**

```bash
git add perovskite-sim/perovskite_sim/autoloop/generated/__init__.py \
        perovskite-sim/perovskite_sim/autoloop/generated/lever.py \
        perovskite-sim/perovskite_sim/models/device.py \
        perovskite-sim/perovskite_sim/models/config_loader.py \
        perovskite-sim/perovskite_sim/scaps_compat/loader.py \
        perovskite-sim/perovskite_sim/solver/mol.py \
        perovskite-sim/tests/unit/autoloop/test_generated_hook.py
git commit -m "feat(autoloop): pre-wired flag-gated codegen hook in build_material_arrays (Stage 5.3)"
```

---

## Task 2: Codegen seam (`codegen.py`)

**Files:**
- Create: `perovskite_sim/autoloop/codegen.py`
- Test: `tests/unit/autoloop/test_codegen.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/autoloop/test_codegen.py
import pytest
from perovskite_sim.autoloop.types import Gap, Hypothesis, AblationMatrix, AblationProbe
from perovskite_sim.autoloop.codegen import (
    GeneratedLever, CODEGEN_SCHEMA, FakeCodegen, ClaudeCodegen, codegen_prompt,
)


def _gap():
    return Gap(id="trend:Et_PVK ETL:V_oc", metric="V_oc", sweep="Et_PVK ETL", sweep_point=0.0,
               solarlab_val=30.0, reference_val=70.0, gap_mag=0.4, kind="trend",
               status="open", found_cycle=0, last_attempt_cycle=0, mechanism=None)


def _hyp():
    return Hypothesis(gap_id="trend:Et_PVK ETL:V_oc", cause="physics",
                      mechanism="missing band-tail Urbach absorption", verdict="confirmed",
                      evidence_for=("ablation: dark J nonzero",))


def _matrix():
    return AblationMatrix(gap_id="trend:Et_PVK ETL:V_oc", baseline_val=40.0,
                          probes=(AblationProbe("SOLARLAB_IFACE_PROJ", "flag", {}, 40.0, 39.9, -0.1, True),))


class _Runtime:
    def __init__(self, obj):
        self.obj = obj
        self.last_prompt = None
    def complete(self, prompt, schema):
        self.last_prompt = prompt
        if isinstance(self.obj, Exception):
            raise self.obj
        return dict(self.obj)


def test_fakecodegen_returns_lever():
    lev = FakeCodegen("return arrays", rationale="noop").generate(_gap(), _hyp(), _matrix())
    assert isinstance(lev, GeneratedLever)
    assert lev.body == "return arrays" and lev.rationale == "noop"


def test_codegen_schema_keys():
    assert CODEGEN_SCHEMA == {"required": ["body", "rationale"]}


def test_claudecodegen_parses_runtime_output():
    rt = _Runtime({"body": "return dataclasses.replace(arrays, chi=arrays.chi)", "rationale": "why"})
    lev = ClaudeCodegen(rt).generate(_gap(), _hyp(), _matrix())
    assert lev.body.startswith("return dataclasses.replace") and lev.rationale == "why"


def test_prompt_contains_contract_and_mechanism():
    p = codegen_prompt(_gap(), _hyp(), _matrix())
    assert "adjust_material_arrays" in p and "Urbach" in p
    assert "chi" in p and "JSON" in p and "return arrays unchanged" in p


def test_claudecodegen_propagates_runtime_error():
    with pytest.raises(RuntimeError):
        ClaudeCodegen(_Runtime(RuntimeError("claude down"))).generate(_gap(), _hyp(), _matrix())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_codegen.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'perovskite_sim.autoloop.codegen'`.

- [ ] **Step 3: Write `codegen.py`**

```python
# perovskite_sim/autoloop/codegen.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from perovskite_sim.autoloop.cognition import CognitionRuntime

CODEGEN_SCHEMA = {"required": ["body", "rationale"]}   # consumed by cognition._validate

# Fields a generated lever may shift (per-node arrays on the frozen MaterialArrays).
_ALLOWED_FIELDS = ("chi", "Eg", "ni_sq", "tau_n", "tau_p", "B_rad", "alpha")


@dataclass(frozen=True)
class GeneratedLever:
    body: str          # source statements for the adjust_material_arrays body (NO def line)
    rationale: str     # one-paragraph why, for the provenance/report


class Codegen(Protocol):
    def generate(self, gap, hyp, matrix=None) -> GeneratedLever: ...


class FakeCodegen:
    """Test seam: returns a canned body, ignoring inputs."""
    def __init__(self, body: str, rationale: str = "fake"):
        self.body = body
        self.rationale = rationale

    def generate(self, gap, hyp, matrix=None) -> GeneratedLever:
        return GeneratedLever(body=self.body, rationale=self.rationale)


def codegen_prompt(gap, hyp, matrix=None) -> str:
    probes = ""
    if matrix is not None and getattr(matrix, "probes", ()):
        probes = "\n".join(f"  - {p.name} [{p.kind}]: delta={p.delta:.4g} ok={p.ok}"
                           for p in matrix.probes)
    ev = "; ".join(hyp.evidence_for) if hyp.evidence_for else "(none)"
    return (
        "You write ONE pure Python function body for a perovskite drift-diffusion "
        "solver extension point. The function is:\n\n"
        "    def adjust_material_arrays(arrays, ctx):\n"
        "        # arrays: a frozen MaterialArrays (per-node numpy arrays). ctx.x is the\n"
        "        # spatial grid, ctx.stack is the DeviceStack. Return a NEW arrays via\n"
        "        # dataclasses.replace(arrays, <field>=...). MUST be pure: no I/O, no\n"
        "        # globals, no mutation. Use `import dataclasses` and `import numpy as np`\n"
        "        # at the top of the body if needed.\n\n"
        f"CONFIRMED CAUSE: cause={hyp.cause}; mechanism={hyp.mechanism}\n"
        f"GAP: metric={gap.metric}, sweep={gap.sweep}, "
        f"solarlab={gap.solarlab_val:.4g} vs reference={gap.reference_val:.4g}\n"
        f"EVIDENCE: {ev}\n"
        f"{('ABLATION PROBES (delta<0 = improved):' + chr(10) + probes) if probes else ''}\n\n"
        f"You may shift ONLY these MaterialArrays fields: {', '.join(_ALLOWED_FIELDS)}.\n"
        "If you cannot justify a change from the evidence, return arrays unchanged "
        "(`return arrays`). Output ONLY a JSON object: "
        '{"body": "<python statements, no def line>", "rationale": "<one paragraph>"}')


class ClaudeCodegen:
    """Live codegen over the 5.1 cognition runtime."""
    def __init__(self, runtime: CognitionRuntime):
        self.runtime = runtime

    def generate(self, gap, hyp, matrix=None) -> GeneratedLever:
        out = self.runtime.complete(codegen_prompt(gap, hyp, matrix), CODEGEN_SCHEMA)
        return GeneratedLever(body=str(out["body"]), rationale=str(out.get("rationale", "")))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_codegen.py`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add perovskite-sim/perovskite_sim/autoloop/codegen.py perovskite-sim/tests/unit/autoloop/test_codegen.py
git commit -m "feat(autoloop): add codegen seam (Codegen/FakeCodegen/ClaudeCodegen, Stage 5.3)"
```

---

## Task 3: G6 build gate + codegen gate runner

**Files:**
- Modify: `perovskite_sim/autoloop/gates_impl.py`
- Test: `tests/unit/autoloop/test_gates_g6.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/autoloop/test_gates_g6.py
from perovskite_sim.autoloop.types import Gap
from perovskite_sim.autoloop.gates_impl import gate_g6_build, make_codegen_gate_runner


def _gap():
    return Gap(id="trend:x:V_oc", metric="V_oc", sweep="x", sweep_point=0.0,
               solarlab_val=30.0, reference_val=70.0, gap_mag=0.4, kind="trend",
               status="open", found_cycle=0, last_attempt_cycle=0, mechanism=None)


def test_g6_identity_passes():
    v = gate_g6_build(golden_runner=lambda: (True, "golden green"),
                      flag_on_runner=lambda: (True, "voc_bracketed"))
    assert v.name == "G6_build" and v.passed is True


def test_g6_import_failure_fails():
    v = gate_g6_build(golden_runner=lambda: (True, "x"), flag_on_runner=lambda: (True, "x"),
                      lever_module="perovskite_sim.autoloop.generated.__does_not_exist__")
    assert v.passed is False and "import" in v.reason.lower()


def test_g6_flag_off_not_bit_identical_fails():
    v = gate_g6_build(golden_runner=lambda: (False, "golden diff"),
                      flag_on_runner=lambda: (True, "ok"))
    assert v.passed is False and "OFF" in v.reason


def test_g6_flag_on_run_fails():
    v = gate_g6_build(golden_runner=lambda: (True, "ok"),
                      flag_on_runner=lambda: (False, "NaN in J_sc"))
    assert v.passed is False and "ON" in v.reason


def test_codegen_gate_runner_g6_then_g3():
    runner = make_codegen_gate_runner(
        golden_runner=lambda: (True, "ok"), flag_on_runner=lambda: (True, "ok"),
        realized_badness=lambda gap: 10.0)   # baseline badness for this gap = 100-30 = 70 -> improves
    verdicts = runner(_gap(), None, None)
    assert [v.name for v in verdicts] == ["G6_build", "G3_improves"]
    assert all(v.passed for v in verdicts)


def test_codegen_gate_runner_short_circuits_on_g6_fail():
    runner = make_codegen_gate_runner(
        golden_runner=lambda: (False, "diff"), flag_on_runner=lambda: (True, "ok"),
        realized_badness=lambda gap: 10.0)
    verdicts = runner(_gap(), None, None)
    assert [v.name for v in verdicts] == ["G6_build"] and verdicts[0].passed is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_gates_g6.py`
Expected: FAIL — `ImportError: cannot import name 'gate_g6_build'`.

- [ ] **Step 3: Add G6 + the codegen gate runner to `gates_impl.py`**

Append to `perovskite_sim/autoloop/gates_impl.py`:

```python
def gate_g6_build(*, golden_runner: Callable[[], tuple[bool, str]],
                  flag_on_runner: Callable[[], tuple[bool, str]],
                  lever_module: str = "perovskite_sim.autoloop.generated.lever") -> GateVerdict:
    """G6 (codegen): the generated lever must (1) import/compile, (2) keep the
    legacy suite green with the flag OFF (reuse G0's golden suite), (3) run a
    flag-ON parity sweep to a finite, voc-bracketed result. Cheap fail-fast
    before the G1/G3 physics checks. Runners are injected so unit tests run
    without the solver."""
    import importlib
    try:
        mod = importlib.import_module(lever_module)
        importlib.reload(mod)
    except Exception as exc:                       # SyntaxError/ImportError/etc.
        return GateVerdict("G6_build", False, f"lever import/compile failed: {exc!r}")
    ok_off, d_off = golden_runner()
    if not ok_off:
        return GateVerdict("G6_build", False, f"flag-OFF not bit-identical: {d_off}")
    ok_on, d_on = flag_on_runner()
    if not ok_on:
        return GateVerdict("G6_build", False, f"flag-ON run failed: {d_on}")
    return GateVerdict("G6_build", True, f"import ok; OFF[{d_off}]; ON[{d_on}]")


def make_codegen_gate_runner(*, golden_runner, flag_on_runner, realized_badness,
                             lever_module: str = "perovskite_sim.autoloop.generated.lever"):
    """Codegen gate stack: G6 (build/import + flag-OFF bit-identical + flag-ON
    runs finite) then G3 (flag-ON badness improves vs the gap's sense-time
    baseline). Short-circuits if G6 fails. Returns a callable(gap, hyp, lever).
    `realized_badness(gap) -> float` re-measures the flag-ON gap badness; injected
    so unit tests run without the solver. (G4 reconcile is N/A: a brand-new lever
    has no prior ablation prediction — G3-improves is the honest benefit check.)"""
    def gate_runner(gap, hyp, lever):
        verdicts = [gate_g6_build(golden_runner=golden_runner, flag_on_runner=flag_on_runner,
                                  lever_module=lever_module)]
        if verdicts[0].passed:
            realized = realized_badness(gap)
            base = gap_baseline_badness(gap)
            verdicts.append(GateVerdict("G3_improves", realized < base,
                                        f"badness {realized:.3g} vs baseline {base:.3g}"))
        return verdicts

    return gate_runner
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_gates_g6.py`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add perovskite-sim/perovskite_sim/autoloop/gates_impl.py perovskite-sim/tests/unit/autoloop/test_gates_g6.py
git commit -m "feat(autoloop): add G6 build gate + codegen gate runner (Stage 5.3)"
```

---

## Task 4: Orchestrator routing + fresh-branch commit

**Files:**
- Modify: `perovskite_sim/autoloop/types.py` (add `CodegenResult`), `perovskite_sim/autoloop/orchestrator.py`
- Test: `tests/unit/autoloop/test_orchestrator_codegen.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/autoloop/test_orchestrator_codegen.py
from pathlib import Path
import subprocess
from perovskite_sim.autoloop.types import Gap, Hypothesis, GateVerdict, CodegenResult
from perovskite_sim.autoloop.ledger import Ledger
from perovskite_sim.autoloop.codegen import FakeCodegen, GeneratedLever
from perovskite_sim.autoloop.orchestrator import codegen_top_not_promotable, commit_generated_lever


def _gap(gid="trend:Et_PVK ETL:V_oc"):
    return Gap(id=gid, metric="V_oc", sweep="Et_PVK ETL", sweep_point=0.0,
               solarlab_val=30.0, reference_val=70.0, gap_mag=0.4, kind="trend",
               status="open", found_cycle=0, last_attempt_cycle=0, mechanism=None)


def _confirmed_hyp(gid, mechanism):
    return Hypothesis(gap_id=gid, cause="physics", mechanism=mechanism, verdict="confirmed")


def _setup(tmp_path, mechanism="missing band-tail Urbach absorption"):
    led = Ledger(root=tmp_path / "ledger")
    led.add_gap(_gap())
    led.add_hypothesis(_confirmed_hyp("trend:Et_PVK ETL:V_oc", mechanism))
    led.save()


def _lever_file(tmp_path):
    p = tmp_path / "lever.py"
    p.write_text("def adjust_material_arrays(arrays, ctx):\n    return arrays\n", encoding="utf-8")
    return p


def _passing_runner(gap, hyp, lever):
    return [GateVerdict("G6_build", True, "ok"), GateVerdict("G3_improves", True, "ok")]


def _failing_runner(gap, hyp, lever):
    return [GateVerdict("G6_build", False, "flag-ON run failed: NaN")]


def _common(tmp_path):
    return dict(ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
                config_path=tmp_path / "c.yaml", reference_path=tmp_path / "r.json",
                cycle=1, timestamp="t")


def test_applied_commits_branch_and_restores_identity(tmp_path):
    _setup(tmp_path)
    lever = _lever_file(tmp_path)
    identity = lever.read_text(encoding="utf-8")
    calls = {}

    def _committer(target, gen_lever, gap, hyp, verdicts, *, git_cwd=None):
        calls["body_at_commit"] = Path(target).read_text(encoding="utf-8")
        return ("feat/autoloop-gen-trend-et-pvk-etl-v-oc", "abc123")

    res = codegen_top_not_promotable(
        **_common(tmp_path), codegen=FakeCodegen("def adjust_material_arrays(arrays, ctx):\n    return arrays\n"),
        gate_runner=_passing_runner, apply=True, committer=_committer, lever_path=lever)
    assert res.status == "applied" and res.branch.startswith("feat/autoloop-gen-")
    assert res.committed_sha == "abc123"
    assert lever.read_text(encoding="utf-8") == identity         # identity restored on working branch


def test_gates_failed_adds_negative(tmp_path):
    _setup(tmp_path)
    lever = _lever_file(tmp_path)
    res = codegen_top_not_promotable(
        **_common(tmp_path), codegen=FakeCodegen("def adjust_material_arrays(arrays, ctx):\n    return arrays\n"),
        gate_runner=_failing_runner, apply=True, committer=None, lever_path=lever)
    assert res.status == "gates_failed"
    led = Ledger.load(tmp_path / "ledger")
    assert led.is_refuted("missing band-tail Urbach absorption")


def test_dry_run_no_commit(tmp_path):
    _setup(tmp_path)
    lever = _lever_file(tmp_path)
    identity = lever.read_text(encoding="utf-8")
    res = codegen_top_not_promotable(
        **_common(tmp_path), codegen=FakeCodegen("def adjust_material_arrays(arrays, ctx):\n    return arrays\n"),
        gate_runner=_passing_runner, apply=False, lever_path=lever)
    assert res.status == "dry_run" and res.committed_sha is None
    assert lever.read_text(encoding="utf-8") == identity


def test_promotable_mechanism_is_not_a_target(tmp_path):
    # mechanism carries an existing promotable flag -> Stage 3's job, NOT codegen.
    _setup(tmp_path, mechanism="enable SOLARLAB_IFACE_PROJ interface projection")
    lever = _lever_file(tmp_path)
    res = codegen_top_not_promotable(
        **_common(tmp_path), codegen=FakeCodegen("x"), gate_runner=_passing_runner,
        apply=False, lever_path=lever)
    assert res.status == "no_target"


def test_commit_generated_lever_fresh_branch(tmp_path):
    # real tiny git repo: identity committed on a feature branch, codegen commits
    # the new body to a fresh feat/autoloop-gen-* branch, returns to origin branch.
    repo = tmp_path / "repo"
    repo.mkdir()
    def git(*a):
        return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True, check=True)
    git("init", "-q"); git("config", "user.email", "t@t"); git("config", "user.name", "t")
    target = repo / "lever.py"
    target.write_text("def adjust_material_arrays(arrays, ctx):\n    return arrays\n", encoding="utf-8")
    git("add", "lever.py"); git("commit", "-q", "-m", "init")
    git("checkout", "-q", "-b", "feat/work")
    target.write_text("def adjust_material_arrays(arrays, ctx):\n    return arrays  # gen\n", encoding="utf-8")
    lever = GeneratedLever(body="return arrays  # gen", rationale="why")
    branch, sha = commit_generated_lever(target, lever, _gap(), _confirmed_hyp("g", "m"),
                                         [GateVerdict("G6_build", True, "ok")], git_cwd=repo)
    assert branch == "feat/autoloop-gen-trend-et-pvk-etl-v-oc"
    cur = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo,
                         capture_output=True, text=True).stdout.strip()
    assert cur == "feat/work"                                    # returned to origin branch
    branches = subprocess.run(["git", "branch"], cwd=repo, capture_output=True, text=True).stdout
    assert "feat/autoloop-gen-trend-et-pvk-etl-v-oc" in branches


def test_commit_generated_lever_refuses_main(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    def git(*a):
        return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True, check=True)
    git("init", "-q", "-b", "main"); git("config", "user.email", "t@t"); git("config", "user.name", "t")
    target = repo / "lever.py"
    target.write_text("x\n", encoding="utf-8")
    git("add", "lever.py"); git("commit", "-q", "-m", "init")
    target.write_text("y\n", encoding="utf-8")
    import pytest
    with pytest.raises(RuntimeError):
        commit_generated_lever(target, GeneratedLever("y", "r"), _gap(),
                               _confirmed_hyp("g", "m"), [], git_cwd=repo)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_orchestrator_codegen.py`
Expected: FAIL — `ImportError: cannot import name 'CodegenResult'` / `codegen_top_not_promotable`.

- [ ] **Step 3a: Add `CodegenResult` to `types.py`**

Append to `perovskite_sim/autoloop/types.py`:

```python
@dataclass(frozen=True)
class CodegenResult:
    status: str         # "applied"|"dry_run"|"gates_failed"|"no_target"|"refuted"
    gap_id: Optional[str]
    branch: Optional[str]
    gate_verdicts: tuple
    committed_sha: Optional[str]
    rationale: Optional[str] = None
```

- [ ] **Step 3b: Add `CodegenResult` to the orchestrator import + write the new functions**

In `perovskite_sim/autoloop/orchestrator.py`, add `CodegenResult` to the `from perovskite_sim.autoloop.types import (...)` block (line 18-21):

```python
from perovskite_sim.autoloop.types import (
    BoulderProposal, BoulderResult,
    CodegenResult, ConfigEdit, Hypothesis, ImplementResult, LadderResult, NegativeResult, ParityScore,
)
```

Append these functions to `orchestrator.py`:

```python
_DEFAULT_LEVER_PATH = Path(__file__).resolve().parent / "generated" / "lever.py"


def _gap_slug(gap_id: str) -> str:
    """git-branch-safe slug from a gap id."""
    slug = "".join(c if c.isalnum() else "-" for c in gap_id.lower())
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")[:48] or "lever"


def commit_generated_lever(target_path, lever, gap, hypothesis, verdicts, *, git_cwd=None):
    """Commit the (already-written) generated lever body to a FRESH
    feat/autoloop-gen-<slug> branch off the current HEAD, then return to the
    original branch (git restores the identity body on it). Refuses main/master
    and a dirty tree other than the lever file; never pushes. -> (branch, sha)."""
    cwd = str(git_cwd) if git_cwd is not None else None

    def _git(*args, check=True):
        return subprocess.run(["git", *args], capture_output=True, text=True, cwd=cwd, check=check)

    origin = _git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if origin in ("main", "master"):
        raise RuntimeError(f"refuse to branch/commit from '{origin}'; create an autoloop branch first")

    git_root = str(Path(_git("rev-parse", "--show-toplevel").stdout.strip()).resolve())
    tgt_abs = str(Path(target_path).resolve())
    dirty = [ln for ln in _git("status", "--porcelain").stdout.splitlines() if ln]
    stray = []
    for line in dirty:
        for p in _parse_porcelain_paths([line]):
            if str((Path(git_root) / p).resolve()) != tgt_abs:
                stray.append(line)
                break
    if stray:
        raise RuntimeError(f"refuse to commit: working tree has unrelated changes: {stray[:3]}")

    branch = f"feat/autoloop-gen-{_gap_slug(gap.id)}"
    _git("checkout", "-b", branch)
    try:
        rel = str(Path(tgt_abs).relative_to(git_root))
        _git("add", rel)
        gate_summary = " ".join(f"{v.name}{'✓' if v.passed else '✗'}" for v in verdicts)
        msg = (f"feat(autoloop): codegen lever for gap {gap.id}\n\n"
               f"Auto-generated by autoloop Stage 5.3 from a confirmed, non-promotable hypothesis.\n"
               f"Mechanism: {hypothesis.mechanism}\n"
               f"Rationale: {lever.rationale}\n"
               f"Gates: {gate_summary}\n"
               f"Gap: {gap.id} | Hypothesis-cycle: {hypothesis.cycle}\n"
               f"Flag-gated (autoloop_generated_lever / SOLARLAB_AUTOLOOP_GEN); default OFF. "
               f"Review before merge.")
        _git("commit", "-q", "-m", msg, "--", rel)
        sha = _git("rev-parse", "HEAD").stdout.strip()
    finally:
        _git("checkout", origin)   # restore origin branch (target reverts to identity)
    return branch, sha


def codegen_top_not_promotable(*, ledger_root: Path, outputs_root: Path, config_path,
                               reference_path, cycle: int, timestamp: str, codegen,
                               gate_runner, apply: bool = False, committer=None,
                               git_cwd=None, lever_path=None) -> CodegenResult:
    """One codegen pass: top confirmed gap whose mechanism is NOT promotable ->
    LLM writes a lever body into autoloop/generated/lever.py -> gate -> (restore +
    report | fresh-branch commit). Always restores the identity body on the
    working branch (try/finally)."""
    led = Ledger.load(Path(ledger_root))
    confirmed_ids = {h.gap_id for h in led.hypotheses if h.verdict == "confirmed"}
    candidates = [g for g in led.gaps
                  if g.status == "open" and g.id in confirmed_ids
                  and not g.id.startswith("coverage:")]

    def _hyp(g):
        return next(h for h in led.hypotheses if h.gap_id == g.id and h.verdict == "confirmed")

    # codegen handles ONLY the not-promotable confirmed gaps (promotable -> Stage 3).
    candidates = [g for g in candidates if propose_promotion(_hyp(g), led, config_path) is None]
    if not candidates:
        return CodegenResult("no_target", None, None, (), None, None)
    gap = max(candidates, key=lambda g: g.gap_mag)
    hyp = _hyp(gap)
    if led.is_refuted(hyp.mechanism):
        return CodegenResult("refuted", gap.id, None, (), None, None)

    lever = codegen.generate(gap, hyp, None)
    target = Path(lever_path) if lever_path is not None else _DEFAULT_LEVER_PATH
    identity = target.read_text(encoding="utf-8")
    try:
        target.write_text(lever.body, encoding="utf-8")
        verdicts = list(gate_runner(gap, hyp, lever))
        if not all(v.passed for v in verdicts):
            led.add_negative(NegativeResult(
                approach=hyp.mechanism,
                why_failed="codegen gate(s) failed: " + ",".join(v.name for v in verdicts if not v.passed),
                evidence=f"autoloop Stage 5.3 codegen cycle {cycle}"))
            led.save()
            return CodegenResult("gates_failed", gap.id, None, tuple(verdicts), None, lever.rationale)
        if apply:
            commit = committer or commit_generated_lever
            branch, sha = commit(target, lever, gap, hyp, verdicts, git_cwd=git_cwd)
            led.add_gap(gap.with_status("closed").with_mechanism(hyp.mechanism))
            led.save()
            return CodegenResult("applied", gap.id, branch, tuple(verdicts), sha, lever.rationale)
        return CodegenResult("dry_run", gap.id, None, tuple(verdicts), None, lever.rationale)
    finally:
        target.write_text(identity, encoding="utf-8")   # ALWAYS restore identity on the working branch
```

(`propose_promotion`, `NegativeResult`, `subprocess`, `Path`, `_parse_porcelain_paths` are already imported/defined in `orchestrator.py`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_orchestrator_codegen.py`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add perovskite-sim/perovskite_sim/autoloop/types.py perovskite-sim/perovskite_sim/autoloop/orchestrator.py perovskite-sim/tests/unit/autoloop/test_orchestrator_codegen.py
git commit -m "feat(autoloop): route not-promotable confirmed gaps to codegen + fresh-branch commit (Stage 5.3)"
```

---

## Task 5: CLI `--codegen` + opt-in smoke + docs

**Files:**
- Modify: `scripts/autoloop_run.py`
- Create: `tests/integration/test_autoloop_codegen.py`
- Modify: `perovskite-sim/CLAUDE.md`, `README.md`

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_autoloop_codegen.py
import importlib.util
import os
import shutil
import sys
from pathlib import Path
import pytest

CLI = Path(__file__).resolve().parents[1] / "scripts" / "autoloop_run.py"


def _load_cli():
    spec = importlib.util.spec_from_file_location("autoloop_run", CLI)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["autoloop_run"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_build_codegen_none_without_flag():
    mod = _load_cli()
    assert mod._build_codegen(mod.parse_args(["--codegen"])) is None or \
        mod._build_codegen(mod.parse_args([])) is None  # no --codegen -> None


def test_build_codegen_with_flag():
    mod = _load_cli()
    from perovskite_sim.autoloop.codegen import ClaudeCodegen
    assert isinstance(mod._build_codegen(mod.parse_args(["--codegen"])), ClaudeCodegen)


@pytest.mark.slow
@pytest.mark.skipif(not shutil.which("claude") or not os.environ.get("SOLARLAB_LLM_SMOKE"),
                    reason="opt-in real-LLM smoke (set SOLARLAB_LLM_SMOKE=1 + claude installed)")
def test_real_codegen_returns_lever():
    from perovskite_sim.autoloop.cognition import ClaudeCliRuntime
    from perovskite_sim.autoloop.codegen import ClaudeCodegen, GeneratedLever
    from perovskite_sim.autoloop.types import Gap, Hypothesis
    gap = Gap(id="g", metric="V_oc", sweep="x", sweep_point=0.0, solarlab_val=30.0,
              reference_val=70.0, gap_mag=0.4, kind="trend", status="open",
              found_cycle=0, last_attempt_cycle=0, mechanism=None)
    hyp = Hypothesis(gap_id="g", cause="physics",
                     mechanism="missing band-tail Urbach absorption", verdict="confirmed")
    lev = ClaudeCodegen(ClaudeCliRuntime(model="sonnet")).generate(gap, hyp, None)
    assert isinstance(lev, GeneratedLever) and isinstance(lev.body, str) and lev.body
```

**Note (test_build_codegen_none_without_flag):** the first test asserts `--codegen` *absent* → `None`. Because `parse_args(["--codegen"])` would set it True, write the no-flag case as `mod._build_codegen(mod.parse_args([])) is None` — fix the test to the single clear assertion before running:

```python
def test_build_codegen_none_without_flag():
    mod = _load_cli()
    assert mod._build_codegen(mod.parse_args([])) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd perovskite-sim && python -m pytest -q tests/integration/test_autoloop_codegen.py`
Expected: FAIL — `AttributeError: module 'autoloop_run' has no attribute '_build_codegen'` (real-LLM smoke skipped by default).

- [ ] **Step 3a: Wire the CLI**

In `scripts/autoloop_run.py` `parse_args`, after the `--verify` argument:

```python
    ap.add_argument("--codegen", action="store_true",
                    help="codegen a flag-gated lever for a confirmed, non-promotable cause "
                         "(dry-run unless --apply; commits to a fresh feat/autoloop-gen-* branch)")
```

Add the helper next to `_build_verifier`:

```python
def _build_codegen(ns):
    if not getattr(ns, "codegen", False):
        return None
    from perovskite_sim.autoloop.cognition import ClaudeCliRuntime
    from perovskite_sim.autoloop.codegen import ClaudeCodegen
    return ClaudeCodegen(ClaudeCliRuntime(model=ns.llm_model))
```

Add a dispatch block in `main`, immediately after the `if ns.implement:` block (before the final `report = guardian_once(...)`):

```python
    if ns.codegen:
        import dataclasses
        from perovskite_sim.autoloop.orchestrator import codegen_top_not_promotable
        from perovskite_sim.autoloop.gates_impl import make_codegen_gate_runner
        from perovskite_sim.autoloop.ladder import run_l0
        from perovskite_sim.autoloop.subprocess_probe import SubprocessProbeRunner

        codegen = _build_codegen(ns)
        if codegen is None:
            print(json.dumps({"codegen": None, "error": "use --codegen together with --llm-capable runtime"}))
            return 1

        def _golden():
            return run_l0(["tests/regression"])

        def _flag_on():
            # flag-ON parity sweep must complete finite; SubprocessProbeRunner
            # returns a badness float (inf/failure -> not ok).
            try:
                val = SubprocessProbeRunner(config_path=ns.config, reference_path=ns.reference,
                                            gap=None).run({"env_flags": {"SOLARLAB_AUTOLOOP_GEN": "1"},
                                                           "jv_overrides": {}, "measure": "base"})
                import math
                return (math.isfinite(val), f"badness={val}")
            except Exception as exc:
                return (False, repr(exc))

        def _realized(gap):
            return SubprocessProbeRunner(config_path=ns.config, reference_path=ns.reference,
                                         gap=gap).run({"env_flags": {"SOLARLAB_AUTOLOOP_GEN": "1"},
                                                       "jv_overrides": {}, "measure": "gap"})

        gate_runner = make_codegen_gate_runner(golden_runner=_golden, flag_on_runner=_flag_on,
                                               realized_badness=_realized)
        result = codegen_top_not_promotable(
            ledger_root=ns.ledger_root, outputs_root=ns.outputs_root, config_path=ns.config,
            reference_path=ns.reference, cycle=ns.cycle, timestamp=iso_timestamp_utc(),
            codegen=codegen, gate_runner=gate_runner, apply=ns.apply)
        print(json.dumps({"codegen": dataclasses.asdict(result)}, indent=2, sort_keys=True, default=str))
        return 1 if result.status == "gates_failed" and ns.apply else 0
```

(If `SubprocessProbeRunner.run`'s `measure="base"` / `gap=None` contract differs, adapt to the existing probe API — the unit tests do not exercise this real path; the opt-in smoke covers `ClaudeCodegen.generate` only.)

- [ ] **Step 3b: Docs**

Add to the **Autoloop** section of `perovskite-sim/CLAUDE.md`:

```markdown
### Stage 5.3 — LLM codegen for non-promotable levers (cognition leg 3, last)

When G5 (5.2) confirms a cause whose mechanism has no existing promotable flag,
`implement_top_confirmed` dead-ends at `not_promotable`. Stage 5.3 routes that gap
to `orchestrator.codegen_top_not_promotable`: `autoloop/codegen.py:ClaudeCodegen`
(over the 5.1 `CognitionRuntime`) writes ONLY the body of `adjust_material_arrays`
into the sandboxed `autoloop/generated/lever.py`. A pre-wired, default-OFF flag
(`autoloop_generated_lever` / env `SOLARLAB_AUTOLOOP_GEN`) gates a single hook at the
end of `solver/mol.build_material_arrays` (import-inside-guard → with the flag off the
generated module is never imported → structurally bit-identical). The codegen gate stack
is G6 build (import/compile + flag-OFF bit-identical via G0 + flag-ON parity sweep runs
finite) then G3 (flag-ON badness improves). On `--apply`, `commit_generated_lever` commits
the lever to a **fresh `feat/autoloop-gen-<gapslug>` branch** off HEAD (refuses main/current,
never pushes) and restores the identity body on the working branch; a human merges. Opt-in:

    cd perovskite-sim
    python scripts/autoloop_run.py --codegen --llm            # dry-run: candidate lever + gate report
    python scripts/autoloop_run.py --codegen --llm --apply    # commit to a fresh feat/autoloop-gen-* branch

Default OFF (no flag, no LLM, no cost). This is the last cognition leg — the full
sense → attribute → verify → implement/codegen → land loop is now complete.
```

Add to `README.md` (next to the G5-verify bullet):

```markdown
- **Autoloop codegen** (`--codegen --llm`) — for a confirmed cause with no existing flag,
  an LLM writes a flag-gated band-parameter lever; the spine verifies it (build + flag-OFF
  bit-identical + flag-ON improves) and commits it to a fresh branch for human merge. Opt-in,
  default OFF.
```

- [ ] **Step 4: Run tests**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop tests/integration/test_autoloop_codegen.py`
Expected: all green (real-LLM smoke skipped without `SOLARLAB_LLM_SMOKE`). Then `python -m pytest -q` (full default suite) — confirm no NEW failures beyond the 2 pre-existing `fcbff18` solver-baseline failures (`test_flat_band_contacts::test_low_doped_etl_flat_band_eliminates_pseudo_crossing`, `test_interface_plane_closure::test_mat_caches_require_parity_configuration`).

- [ ] **Step 5: Commit**

```bash
git add perovskite-sim/scripts/autoloop_run.py perovskite-sim/tests/integration/test_autoloop_codegen.py perovskite-sim/CLAUDE.md README.md
git commit -m "feat(autoloop): wire --codegen (Stage 5.3) into CLI + opt-in smoke + docs"
```

---

## Self-Review

**Spec coverage** (vs `2026-06-17-autoloop-stage5.3-llm-codegen-design.md`):
- §4 pre-wired hook (flag on `DeviceStack` + both loaders + guarded hook in `build_material_arrays` + `generated/{__init__,lever}.py` identity + `_LeverContext`) → Task 1. ✓
- §3 codegen seam (`GeneratedLever`, `CODEGEN_SCHEMA`, `Codegen`, `FakeCodegen`, `ClaudeCodegen`, prompt with fixed signature + allowed fields + "return unchanged if unsure") → Task 2. ✓
- §5 gates (G6 build = import + flag-OFF bit-identical via G0 + flag-ON runs finite; then G3 improves; G4 documented N/A for a novel lever) → Task 3. ✓
- §2/§6 routing (`codegen_top_not_promotable` selects confirmed not-promotable gaps; refute/confirm; identity-restore try/finally) + branch isolation (`commit_generated_lever`: fresh branch, refuse main, return to origin, never push) → Task 4. ✓
- §6 CLI `--codegen` + `_build_codegen` + dispatch; §8 opt-in smoke; docs → Task 5. ✓
- §7 error handling (runtime raise → propagates → caller no-commit; gate-fail → add_negative; finally restore; refuse main) → Tasks 2/4. ✓
- §9 deferred (RHS/boundary hooks, 2D path, new-flag proposal, auto-push, boulder wiring) → correctly NOT built.

**Placeholder scan:** none — every code/test step is complete and verbatim. The one runtime-API caveat (`SubprocessProbeRunner.run` `measure`/`gap=None` for the real `--codegen` path) is flagged explicitly and is not exercised by unit tests; the implementer adapts it to the existing probe API at Task 5 Step 3a. The duplicated `test_build_codegen_none_without_flag` is corrected inline before running.

**Type consistency:** `Codegen.generate(gap, hyp, matrix=None) -> GeneratedLever` consistent across `FakeCodegen`/`ClaudeCodegen`/orchestrator (passes `None`) + tests. `GeneratedLever(body, rationale)`, `CODEGEN_SCHEMA={"required":["body","rationale"]}`, `gate_g6_build(*, golden_runner, flag_on_runner, lever_module=...)`, `make_codegen_gate_runner(*, golden_runner, flag_on_runner, realized_badness, lever_module=...) -> callable(gap, hyp, lever)`, `codegen_top_not_promotable(..., codegen, gate_runner, apply, committer, git_cwd, lever_path) -> CodegenResult`, `commit_generated_lever(target_path, lever, gap, hypothesis, verdicts, *, git_cwd=None) -> (branch, sha)`, `CodegenResult(status, gap_id, branch, gate_verdicts, committed_sha, rationale=None)` all match between definitions, the orchestrator call sites, the CLI, and every test. Reused real symbols verified in the Design contract: `propose_promotion`, `gap_baseline_badness`, `gate_g0_bit_identical` (pattern), `_parse_porcelain_paths`, `commit_promotion` (template), `Ledger.add_negative`/`is_refuted`/`add_gap`, `gap.with_status`/`with_mechanism`, `MaterialArrays` frozen fields (`chi`/`Eg`/`ni_sq`), `multilayer_grid`/`electrical_layers`/`load_scaps_yaml`.

---

## Execution Handoff

After saving, choose execution:

1. **Subagent-Driven (recommended)** — fresh subagent per task, two-stage review (the background-workflow used for prior stages).
2. **Inline Execution** — batch tasks in this session with checkpoints.
