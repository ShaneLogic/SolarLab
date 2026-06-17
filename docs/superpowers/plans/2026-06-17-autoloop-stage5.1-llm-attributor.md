# Autoloop Stage 5.1 — Cognition Runtime + LLM Attributor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pluggable `CognitionRuntime` (headless `claude -p --json` + a test fake) and an `LLMAttributor` that composes over the heuristic — calling the LLM only on `uncertain` gaps to propose a novel cause, always as a `verdict="uncertain"` lead (never auto-confirmed).

**Architecture:** A new `cognition.py` (runtime seam + `ClaudeCliRuntime` + `FakeRuntime` + in-process schema validation) and `llm_attribution.py` (`LLMAttributor` wrapping `HeuristicAttributor`: heuristic first; on uncertain, call `runtime.complete(prompt, schema)` and emit an uncertain Hypothesis carrying the LLM's novel cause). The CLI gains `--llm` (opt-in; default deterministic). Any runtime failure degrades to the heuristic's uncertain — the LLM never confirms and never crashes the loop.

**Tech Stack:** Python 3.9+, subprocess, json, dataclasses. Reuses Stage-2 `HeuristicAttributor`, `ablation.CANDIDATE_FLAGS`, `types.Hypothesis`, `ledger.Ledger.is_refuted`. No new third-party deps.

---

## Design contract (read before starting)

- **Spec:** `docs/superpowers/specs/2026-06-17-autoloop-stage5.1-llm-attributor-design.md`.
- **Feasibility verified:** `claude` CLI at `~/.local/bin/claude` (v2.1.179) supports `-p` + `--output-format json` + `--model`. `--output-format json` returns an envelope `{"type":"result","result":"<assistant text>", ...}` — the `result` field is the assistant's text.
- **Stage-2 APIs (verified on `main`):**
  - `autoloop.attribution.HeuristicAttributor().attribute(gap, matrix, negatives) -> Hypothesis`.
  - `autoloop.ablation.CANDIDATE_FLAGS: dict[str, list[str]]`.
  - `autoloop.types.Hypothesis(gap_id, cause, mechanism, evidence_for=(), evidence_against=(), verifier_votes=0, verdict="uncertain", cycle=0, predicted_delta=0.0)`.
  - `autoloop.types.AblationMatrix(gap_id, baseline_val, probes: tuple[AblationProbe], skipped)`; `AblationProbe(name, kind, variant, baseline_val, variant_val, delta, ok, note)`.
  - `autoloop.ledger.Ledger.is_refuted(approach) -> bool`; `.negatives` (list of `NegativeResult(approach, ...)`).
- **Run all commands from `perovskite-sim/`.** Tests default to `-m 'not slow'`.

## File Structure

```
perovskite_sim/autoloop/
  cognition.py        CognitionRuntime protocol + FakeRuntime + ClaudeCliRuntime + _validate
  llm_attribution.py  ATTRIBUTION_SCHEMA + build_attribution_prompt + LLMAttributor
scripts/autoloop_run.py  + --llm/--llm-model + _build_attributor wired into --attribute & --boulder
tests/unit/autoloop/
  test_cognition.py
  test_llm_attribution.py
tests/integration/
  test_autoloop_llm.py   (opt-in slow)
```

---

## Task 1: CognitionRuntime seam

**Files:**
- Create: `perovskite_sim/autoloop/cognition.py`
- Test: `tests/unit/autoloop/test_cognition.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/autoloop/test_cognition.py
import json
import subprocess
import pytest
from perovskite_sim.autoloop import cognition
from perovskite_sim.autoloop.cognition import ClaudeCliRuntime, FakeRuntime, _validate

SCHEMA = {"required": ["cause", "mechanism"], "cause_enum": ["bug", "numerics", "physics", "data"]}


def _envelope(text):
    class _CP:
        returncode = 0
        stdout = json.dumps({"type": "result", "result": text})
        stderr = ""
    return _CP()


def test_validate_accepts_good_and_rejects_bad():
    _validate({"cause": "physics", "mechanism": "x"}, SCHEMA)        # no raise
    with pytest.raises(ValueError):
        _validate({"mechanism": "x"}, SCHEMA)                        # missing cause
    with pytest.raises(ValueError):
        _validate({"cause": "vibes", "mechanism": "x"}, SCHEMA)      # bad enum


def test_fake_runtime_returns_canned():
    assert FakeRuntime({"cause": "bug", "mechanism": "m"}).complete("p", SCHEMA)["cause"] == "bug"


def test_claude_runtime_parses_envelope(monkeypatch):
    captured = {}
    def _run(cmd, **kw):
        captured["cmd"] = cmd
        return _envelope('{"cause": "physics", "mechanism": "missing Auger term"}')
    monkeypatch.setattr(cognition.subprocess, "run", _run)
    out = ClaudeCliRuntime(model="sonnet").complete("diagnose", SCHEMA)
    assert out["mechanism"] == "missing Auger term"
    assert "--output-format" in captured["cmd"] and "json" in captured["cmd"]
    assert "sonnet" in captured["cmd"]


def test_claude_runtime_strips_markdown_fence(monkeypatch):
    monkeypatch.setattr(cognition.subprocess, "run",
                        lambda cmd, **kw: _envelope('```json\n{"cause":"bug","mechanism":"m"}\n```'))
    assert ClaudeCliRuntime().complete("p", SCHEMA)["cause"] == "bug"


def test_claude_runtime_retries_once_then_succeeds(monkeypatch):
    calls = {"n": 0}
    def _run(cmd, **kw):
        calls["n"] += 1
        return _envelope("not json" if calls["n"] == 1 else '{"cause":"data","mechanism":"m"}')
    monkeypatch.setattr(cognition.subprocess, "run", _run)
    assert ClaudeCliRuntime().complete("p", SCHEMA)["cause"] == "data"
    assert calls["n"] == 2                                          # retried once


def test_claude_runtime_raises_after_retry(monkeypatch):
    monkeypatch.setattr(cognition.subprocess, "run", lambda cmd, **kw: _envelope("never json"))
    with pytest.raises(RuntimeError):
        ClaudeCliRuntime().complete("p", SCHEMA)


def test_claude_runtime_raises_on_timeout(monkeypatch):
    def _run(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw.get("timeout"))
    monkeypatch.setattr(cognition.subprocess, "run", _run)
    with pytest.raises(RuntimeError):
        ClaudeCliRuntime(timeout_s=1).complete("p", SCHEMA)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_cognition.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'perovskite_sim.autoloop.cognition'`.

- [ ] **Step 3: Write `cognition.py`**

```python
# perovskite_sim/autoloop/cognition.py
from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)


class CognitionRuntime(Protocol):
    def complete(self, prompt: str, schema: dict) -> dict: ...


def _validate(obj: dict, schema: dict) -> None:
    """In-process schema check (no jsonschema dep): required keys + cause enum."""
    if not isinstance(obj, dict):
        raise ValueError("LLM output is not a JSON object")
    for key in schema.get("required", ()):
        if key not in obj:
            raise ValueError(f"LLM output missing required key {key!r}")
    enum = schema.get("cause_enum")
    if enum is not None and obj.get("cause") not in enum:
        raise ValueError(f"cause {obj.get('cause')!r} not in {enum}")


def _strip_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


@dataclass
class FakeRuntime:
    """Test runtime: returns a canned dict (or callable(prompt) -> dict)."""
    canned: object

    def complete(self, prompt: str, schema: dict) -> dict:
        out = self.canned(prompt) if callable(self.canned) else self.canned
        return dict(out)


@dataclass
class ClaudeCliRuntime:
    """Headless cognition via `claude -p --output-format json`. Schema-validated,
    timeout-guarded, retries once on parse/validation failure."""
    model: str = "sonnet"
    timeout_s: float = 180.0

    def _run(self, prompt: str) -> dict:
        proc = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "json", "--model", self.model],
            capture_output=True, text=True, timeout=self.timeout_s)
        if proc.returncode != 0:
            raise RuntimeError(f"claude rc={proc.returncode}: {proc.stderr.strip()[-300:]}")
        text = json.loads(proc.stdout)["result"]
        return json.loads(_strip_fence(text))

    def complete(self, prompt: str, schema: dict) -> dict:
        try:
            obj = self._run(prompt)
            _validate(obj, schema)
            return obj
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"claude timed out after {self.timeout_s}s") from exc
        except (json.JSONDecodeError, ValueError, KeyError):
            # one retry with an explicit JSON-only nudge
            try:
                obj = self._run(prompt + "\n\nReturn ONLY the JSON object, no prose, no markdown.")
                _validate(obj, schema)
                return obj
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"claude timed out after {self.timeout_s}s") from exc
            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                raise RuntimeError(f"claude returned unparseable/invalid output: {exc}") from exc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_cognition.py`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add perovskite-sim/perovskite_sim/autoloop/cognition.py perovskite-sim/tests/unit/autoloop/test_cognition.py
git commit -m "feat(autoloop): add CognitionRuntime seam (ClaudeCliRuntime + FakeRuntime, Stage 5.1)"
```

---

## Task 2: LLMAttributor

**Files:**
- Create: `perovskite_sim/autoloop/llm_attribution.py`
- Test: `tests/unit/autoloop/test_llm_attribution.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/autoloop/test_llm_attribution.py
from perovskite_sim.autoloop.types import Gap, AblationProbe, AblationMatrix, Hypothesis, NegativeResult
from perovskite_sim.autoloop.ledger import Ledger
from perovskite_sim.autoloop.cognition import FakeRuntime
from perovskite_sim.autoloop.llm_attribution import (
    LLMAttributor, build_attribution_prompt, ATTRIBUTION_SCHEMA,
)


def _gap():
    return Gap(id="trend:Et_PVK ETL:V_oc", metric="V_oc", sweep="Et_PVK ETL", sweep_point=0.0,
               solarlab_val=30.0, reference_val=70.0, gap_mag=0.4, kind="trend",
               status="open", found_cycle=0, last_attempt_cycle=0, mechanism=None)


def _uncertain_matrix():
    # all probes flat -> heuristic returns "uncertain"
    probes = (
        AblationProbe("SOLARLAB_IFACE_PROJ", "flag", {}, 40.0, 39.9, -0.1, True),
        AblationProbe("grid_n80", "grid", {}, 40.0, 40.1, 0.1, True),
        AblationProbe("dark_jsc", "limiting", {}, 0.0, 0.01, 0.01, True),
    )
    return AblationMatrix(gap_id="trend:Et_PVK ETL:V_oc", baseline_val=40.0, probes=probes)


class _SpyRuntime:
    def __init__(self, out):
        self.out = out
        self.calls = 0
    def complete(self, prompt, schema):
        self.calls += 1
        self.last_prompt = prompt
        return dict(self.out)


def test_heuristic_confirmed_skips_llm(tmp_path):
    # a flag probe that strongly improves -> heuristic confirms physics -> LLM NOT called
    from perovskite_sim.autoloop.types import AblationMatrix as AM, AblationProbe as AP
    matrix = AM(gap_id="g", baseline_val=40.0, probes=(
        AP("SOLARLAB_IFACE_PROJ", "flag", {}, 40.0, 20.0, -20.0, True),
        AP("grid_n80", "grid", {}, 40.0, 40.0, 0.0, True),
        AP("dark_jsc", "limiting", {}, 0.0, 0.0, 0.0, True)))
    spy = _SpyRuntime({"cause": "physics", "mechanism": "x"})
    hyp = LLMAttributor(spy).attribute(_gap(), matrix, Ledger(root=tmp_path))
    assert spy.calls == 0
    assert hyp.verdict == "confirmed"          # heuristic handled it


def test_uncertain_calls_llm_and_returns_lead(tmp_path):
    spy = _SpyRuntime({"cause": "physics", "mechanism": "missing band-tail Urbach absorption",
                       "confidence": 0.6})
    hyp = LLMAttributor(spy).attribute(_gap(), _uncertain_matrix(), Ledger(root=tmp_path))
    assert spy.calls == 1
    assert hyp.cause == "physics"
    assert "Urbach" in hyp.mechanism
    assert hyp.verdict == "uncertain"          # ALWAYS a lead
    assert any("LLM novel-cause lead" in e for e in hyp.evidence_for)


def test_llm_failure_falls_back_to_heuristic(tmp_path):
    class _Boom:
        def complete(self, prompt, schema): raise RuntimeError("claude missing")
    hyp = LLMAttributor(_Boom()).attribute(_gap(), _uncertain_matrix(), Ledger(root=tmp_path))
    assert hyp.cause == "uncertain" and hyp.verdict == "uncertain"   # heuristic's uncertain


def test_refuted_mechanism_flagged(tmp_path):
    led = Ledger(root=tmp_path)
    led.add_negative(NegativeResult(approach="missing band-tail Urbach absorption",
                                    why_failed="x", evidence="y"))
    spy = _SpyRuntime({"cause": "physics", "mechanism": "missing band-tail Urbach absorption"})
    hyp = LLMAttributor(spy).attribute(_gap(), _uncertain_matrix(), led)
    assert hyp.verdict == "uncertain"
    assert any("refuted" in e.lower() for e in hyp.evidence_against)


def test_prompt_contains_context(tmp_path):
    led = Ledger(root=tmp_path)
    led.add_negative(NegativeResult(approach="DOS-cap projection", why_failed="x", evidence="y"))
    prompt = build_attribution_prompt(_gap(), _uncertain_matrix(), led)
    assert "Et_PVK ETL" in prompt                       # the gap
    assert "SOLARLAB_IFACE_PROJ" in prompt              # a matrix probe / flag menu
    assert "DOS-cap projection" in prompt               # the negatives ledger
    assert "JSON" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_llm_attribution.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'perovskite_sim.autoloop.llm_attribution'`.

- [ ] **Step 3: Write `llm_attribution.py`**

```python
# perovskite_sim/autoloop/llm_attribution.py
from __future__ import annotations

import logging

from perovskite_sim.autoloop.ablation import CANDIDATE_FLAGS
from perovskite_sim.autoloop.attribution import HeuristicAttributor
from perovskite_sim.autoloop.cognition import CognitionRuntime
from perovskite_sim.autoloop.types import Hypothesis

logger = logging.getLogger(__name__)

ATTRIBUTION_SCHEMA = {
    "required": ["cause", "mechanism"],
    "cause_enum": ["bug", "numerics", "physics", "data"],
}


def build_attribution_prompt(gap, matrix, negatives) -> str:
    probes = "\n".join(
        f"  - {p.name} [{p.kind}]: delta={p.delta:.4g} ok={p.ok}" for p in matrix.probes)
    flags = ", ".join(sorted({f for fl in CANDIDATE_FLAGS.values() for f in fl}))
    refuted = "\n".join(f"  - {n.approach}" for n in getattr(negatives, "negatives", []))
    return (
        "You are diagnosing why a perovskite drift-diffusion simulator (SolarLab) "
        "disagrees with its reference on one metric. The deterministic heuristic found "
        "NO known-flag lever for this gap.\n\n"
        f"GAP: metric={gap.metric}, sweep={gap.sweep}, kind={gap.kind}, "
        f"solarlab={gap.solarlab_val:.4g} vs reference={gap.reference_val:.4g}.\n\n"
        f"ABLATION MATRIX (baseline badness {matrix.baseline_val:.4g}; "
        "delta<0 means that variant improved the gap):\n" + probes + "\n\n"
        f"FLAGS ALREADY TRIED (none a clear lever): {flags}\n\n"
        "REFUTED approaches — do NOT propose any of these:\n" + (refuted or "  (none)") + "\n\n"
        "Propose the single most likely NOVEL root cause — a physics term, numerical "
        "issue, bug, or data problem NOT behind any existing flag. Be specific and "
        "mechanistic (name the term, equation, or site).\n"
        "Output ONLY a JSON object: "
        '{"cause": "bug|numerics|physics|data", "mechanism": "<specific>", '
        '"evidence_for": ["..."], "evidence_against": ["..."], "confidence": 0.0}')


class LLMAttributor:
    """Composes over HeuristicAttributor: runs the heuristic first; on an
    'uncertain' verdict, asks the LLM for a NOVEL cause. The LLM result is ALWAYS
    a verdict='uncertain' lead (never auto-confirmed). Any runtime failure
    degrades to the heuristic's uncertain — the LLM never blocks the loop."""

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
        except Exception as exc:                       # never block on the LLM
            logger.warning("LLM attribution failed for %s: %r — keeping heuristic uncertain",
                           gap.id, exc)
            return hyp
        mechanism = out["mechanism"]
        ev_for = tuple(out.get("evidence_for", ())) + (
            f"LLM novel-cause lead (confidence {out.get('confidence', '?')})",)
        ev_against = tuple(out.get("evidence_against", ()))
        if negatives.is_refuted(mechanism):
            ev_against = ev_against + ("matches a REFUTED approach in the negatives ledger",)
        return Hypothesis(gap_id=gap.id, cause=out["cause"], mechanism=mechanism,
                          evidence_for=ev_for, evidence_against=ev_against,
                          verifier_votes=0, verdict="uncertain")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_llm_attribution.py`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add perovskite-sim/perovskite_sim/autoloop/llm_attribution.py perovskite-sim/tests/unit/autoloop/test_llm_attribution.py
git commit -m "feat(autoloop): add LLMAttributor (heuristic-fallback novel-cause lead, Stage 5.1)"
```

---

## Task 3: CLI `--llm` wiring + opt-in smoke + docs

**Files:**
- Modify: `scripts/autoloop_run.py`
- Create: `tests/integration/test_autoloop_llm.py`
- Modify: `perovskite-sim/CLAUDE.md`, `README.md`

- [ ] **Step 1: Write the failing test (CLI helper + opt-in smoke)**

```python
# tests/integration/test_autoloop_llm.py
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


def test_build_attributor_default_is_heuristic():
    mod = _load_cli()
    ns = mod.parse_args(["--attribute"])
    from perovskite_sim.autoloop.attribution import HeuristicAttributor
    assert isinstance(mod._build_attributor(ns), HeuristicAttributor)


def test_build_attributor_llm_flag():
    mod = _load_cli()
    ns = mod.parse_args(["--attribute", "--llm", "--llm-model", "sonnet"])
    from perovskite_sim.autoloop.llm_attribution import LLMAttributor
    assert isinstance(mod._build_attributor(ns), LLMAttributor)


@pytest.mark.slow
@pytest.mark.skipif(not shutil.which("claude") or not os.environ.get("SOLARLAB_LLM_SMOKE"),
                    reason="opt-in real-LLM smoke (set SOLARLAB_LLM_SMOKE=1 + claude installed)")
def test_real_llm_attributor_produces_lead(tmp_path):
    from perovskite_sim.autoloop.cognition import ClaudeCliRuntime
    from perovskite_sim.autoloop.llm_attribution import LLMAttributor
    from perovskite_sim.autoloop.types import Gap, AblationProbe, AblationMatrix
    from perovskite_sim.autoloop.ledger import Ledger
    gap = Gap(id="trend:Et_PVK ETL:V_oc", metric="V_oc", sweep="Et_PVK ETL", sweep_point=0.0,
              solarlab_val=30.0, reference_val=70.0, gap_mag=0.4, kind="trend",
              status="open", found_cycle=0, last_attempt_cycle=0, mechanism=None)
    matrix = AblationMatrix(gap_id=gap.id, baseline_val=40.0, probes=(
        AblationProbe("SOLARLAB_IFACE_PROJ", "flag", {}, 40.0, 39.9, -0.1, True),
        AblationProbe("grid_n80", "grid", {}, 40.0, 40.1, 0.1, True),
        AblationProbe("dark_jsc", "limiting", {}, 0.0, 0.0, 0.0, True)))
    hyp = LLMAttributor(ClaudeCliRuntime(model="sonnet")).attribute(gap, matrix, Ledger(root=tmp_path))
    assert hyp.verdict == "uncertain"
    assert hyp.mechanism and hyp.cause in {"bug", "numerics", "physics", "data"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd perovskite-sim && python -m pytest -q tests/integration/test_autoloop_llm.py`
Expected: FAIL — `_build_attributor` does not exist yet (the real-LLM test is skipped by default).

- [ ] **Step 3: Wire the CLI + docs**

In `scripts/autoloop_run.py`, add flags to `parse_args`:

```python
    ap.add_argument("--llm", action="store_true",
                    help="use the LLM attributor (fallback on gaps the heuristic can't diagnose)")
    ap.add_argument("--llm-model", default="sonnet", help="model for --llm (default sonnet)")
```

Add the helper near the top of `main` (module-level, after `parse_args`):

```python
def _build_attributor(ns):
    from perovskite_sim.autoloop.attribution import HeuristicAttributor
    if not getattr(ns, "llm", False):
        return HeuristicAttributor()
    from perovskite_sim.autoloop.cognition import ClaudeCliRuntime
    from perovskite_sim.autoloop.llm_attribution import LLMAttributor
    return LLMAttributor(ClaudeCliRuntime(model=ns.llm_model))
```

In the `--attribute` dispatch, replace `attributor=HeuristicAttributor()` with
`attributor=_build_attributor(ns)`. In the `--boulder` dispatch, the `attribute(cycle)`
closure builds `attributor=HeuristicAttributor()` — replace it with
`attributor=_build_attributor(ns)` (compute once before the closure:
`_attr = _build_attributor(ns)` and reference `_attr`).

Add to the **Autoloop** section of `perovskite-sim/CLAUDE.md`:

```markdown
### Stage 5.1 — LLM attributor (cognition leg 1)

`autoloop/cognition.py` is the pluggable `CognitionRuntime` seam (`ClaudeCliRuntime`
runs `claude -p --output-format json`, schema-validated, timeout + 1 retry; `FakeRuntime`
for tests). `autoloop/llm_attribution.py:LLMAttributor` composes over `HeuristicAttributor`:
the heuristic runs first, and the LLM is called ONLY on `uncertain` gaps to propose a
NOVEL cause beyond the flag menu. The LLM result is ALWAYS `verdict="uncertain"` (a lead —
confirmation is the deferred G5 / human review); any runtime failure degrades to the
heuristic's uncertain (the LLM never confirms, never blocks the loop). Opt-in:

    cd perovskite-sim
    python scripts/autoloop_run.py --attribute --llm --llm-model sonnet
    python scripts/autoloop_run.py --boulder --llm

Default (no --llm) stays fully deterministic, no LLM, no cost. G5 multi-skeptic verify
(5.2) and LLM codegen (5.3) are deferred behind this runtime seam.
```

Add to `README.md` (next to the other autoloop lines):

```markdown
- **Autoloop LLM attributor** (`--attribute --llm`) — when the deterministic heuristic
  can't diagnose a gap, an LLM proposes a novel root cause (always a verdict=uncertain
  lead; never auto-confirmed). Opt-in; default stays deterministic.
```

- [ ] **Step 4: Run tests**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop tests/integration/test_autoloop_llm.py`
Expected: all green (the real-LLM smoke is skipped without `SOLARLAB_LLM_SMOKE`). Also `python -m pytest -q` (full default suite) — confirm no import/collection regression.

- [ ] **Step 5: Commit**

```bash
git add perovskite-sim/scripts/autoloop_run.py perovskite-sim/tests/integration/test_autoloop_llm.py perovskite-sim/CLAUDE.md README.md
git commit -m "feat(autoloop): wire --llm attributor into CLI + opt-in smoke + docs (Stage 5.1)"
```

---

## Self-Review

**Spec coverage** (vs `2026-06-17-autoloop-stage5.1-llm-attributor-design.md`):
- §3 `CognitionRuntime` protocol + `FakeRuntime` + `ClaudeCliRuntime` (envelope parse, fence strip, validate, retry, timeout) + `_validate` → Task 1. ✓
- §4 `ATTRIBUTION_SCHEMA` + `build_attribution_prompt` (gap + matrix + flag menu + negatives) + `LLMAttributor` (heuristic-first, uncertain→LLM, always-uncertain lead, negatives-guard, fail→fallback) → Task 2. ✓
- §5 CLI `--llm`/`--llm-model` + `_build_attributor` wired into `--attribute` + `--boulder` → Task 3. ✓
- §6 error handling (claude missing/timeout/bad-json → heuristic fallback; never confirm) → Tasks 1/2. ✓
- §7 testing (validate, fake, envelope/fence/retry/timeout; heuristic-skip, uncertain-lead, fallback, refuted-flag, prompt-content; opt-in real smoke) → every task. ✓
- §8 deferred (G5, codegen, confirm, agentic, implement-step) → correctly NOT built.

**Placeholder scan:** none — complete code/tests/commands. The opt-in smoke's skip condition is explicit, not a placeholder.

**Type consistency:** `CognitionRuntime.complete(prompt, schema) -> dict` consistent Tasks 1/2 + tests. `FakeRuntime(canned)`/`ClaudeCliRuntime(model, timeout_s)` consistent. `LLMAttributor(runtime, heuristic=)` + `.attribute(gap, matrix, negatives) -> Hypothesis` matches the Stage-2 `Attributor` seam so it drops into `attribute_top_gap`'s injected `attributor`. `ATTRIBUTION_SCHEMA` `{required, cause_enum}` consumed by `_validate`. `_build_attributor(ns)` returns the seam type both CLI dispatches inject. `Hypothesis(... verdict="uncertain")`, `HeuristicAttributor`, `CANDIDATE_FLAGS`, `Ledger.is_refuted`/`.negatives` are verified real symbols.

---

## Execution Handoff

After saving, choose execution:

1. **Subagent-Driven (recommended)** — fresh subagent per task, two-stage review (same background-workflow as prior stages).
2. **Inline Execution** — batch tasks in this session with checkpoints.
