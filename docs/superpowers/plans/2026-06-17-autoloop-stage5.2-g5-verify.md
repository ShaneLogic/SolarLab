# Autoloop Stage 5.2 — G5 Multi-Skeptic Adversarial Verify — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `MultiSkepticVerifier` (N diverse-lens skeptics over the 5.1 cognition runtime) that adjudicates an LLM novel-cause lead — promoting it to `confirmed` if it survives, or `refuted` (+ negatives ledger) if a majority refutes — wired post-attribution and filling the `gate_g5` stub.

**Architecture:** A new `verify.py` reuses 5.1's `CognitionRuntime`: each of N lenses (physical-plausibility / numerical-artifact / data-support) is prompted to refute the hypothesis mechanism; strict majority of a quorum decides. `attribute_top_gap` gains an optional `verifier` that runs on LLM leads (`verdict=="uncertain"` AND `cause != "uncertain"`) before the ledger write, calling `add_negative` on refute. CLI `--verify` (opt-in, with `--llm`). Errored skeptics are excluded (never a false refute); no quorum → stays uncertain.

**Tech Stack:** Python 3.9+, dataclasses. Reuses 5.1 `cognition.CognitionRuntime`/`FakeRuntime`, `types.Hypothesis`/`NegativeResult`/`GateVerdict`, `gate_g5_deferred` stub, `orchestrator.attribute_top_gap`. No new third-party deps.

---

## Design contract (read before starting)

- **Spec:** `docs/superpowers/specs/2026-06-17-autoloop-stage5.2-g5-verify-design.md`.
- **5.1 + Stage-2 APIs (verified on `main`):**
  - `cognition.CognitionRuntime.complete(prompt, schema) -> dict`; `cognition.FakeRuntime(canned)`; `cognition._validate` consumes `{"required": [...], "cause_enum": [...]}`.
  - `types.Hypothesis(gap_id, cause, mechanism, evidence_for=(), evidence_against=(), verifier_votes=0, verdict="uncertain", cycle=0, predicted_delta=0.0)` — frozen; use `dataclasses.replace`.
  - `types.NegativeResult(approach, why_failed, evidence, never_retry=True)`; `types.GateVerdict(name, passed, reason)`.
  - `ledger.Ledger.add_negative`, `.add_hypothesis`, `.add_gap`; `gap.with_mechanism(mechanism)`.
  - `orchestrator.attribute_top_gap(*, ledger_root, outputs_root, config_path, reference_path, cycle, timestamp, probe_runner_factory, attributor, flags=None, seed=0, run_ablation_fn=None)` — body: `hyp = attributor.attribute(gap, matrix, led)` then `led.add_hypothesis(hyp)` / `if confirmed: add_gap(with_mechanism)` / `save`. `NegativeResult` is already imported in orchestrator.py (Stage 3).
- **Run all commands from `perovskite-sim/`.** Tests default to `-m 'not slow'`.

## File Structure

```
perovskite_sim/autoloop/
  verify.py         SKEPTIC_LENSES, VOTE_SCHEMA, refute_prompt, MultiSkepticVerifier, gate_g5_verify
  orchestrator.py   attribute_top_gap: + verifier param (verify LLM leads -> confirm/refute + add_negative)
scripts/autoloop_run.py  + --verify + _build_verifier into --attribute & --boulder
tests/unit/autoloop/
  test_verify.py
  test_orchestrator_verify.py
tests/integration/
  test_autoloop_g5.py   (opt-in slow)
```

---

## Task 1: MultiSkepticVerifier

**Files:**
- Create: `perovskite_sim/autoloop/verify.py`
- Test: `tests/unit/autoloop/test_verify.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/autoloop/test_verify.py
from perovskite_sim.autoloop.types import Gap, AblationProbe, AblationMatrix, Hypothesis
from perovskite_sim.autoloop.verify import (
    SKEPTIC_LENSES, MultiSkepticVerifier, refute_prompt, gate_g5_verify,
)


def _gap():
    return Gap(id="trend:Et_PVK ETL:V_oc", metric="V_oc", sweep="Et_PVK ETL", sweep_point=0.0,
               solarlab_val=30.0, reference_val=70.0, gap_mag=0.4, kind="trend",
               status="open", found_cycle=0, last_attempt_cycle=0, mechanism=None)


def _matrix():
    return AblationMatrix(gap_id="trend:Et_PVK ETL:V_oc", baseline_val=40.0, probes=(
        AblationProbe("SOLARLAB_IFACE_PROJ", "flag", {}, 40.0, 39.9, -0.1, True),))


def _lead():
    return Hypothesis(gap_id="trend:Et_PVK ETL:V_oc", cause="physics",
                      mechanism="missing band-tail Urbach absorption", verdict="uncertain")


class _ScriptedRuntime:
    """Returns a sequence of votes (or raises) keyed by call order."""
    def __init__(self, votes):
        self.votes = list(votes)
        self.calls = 0
    def complete(self, prompt, schema):
        v = self.votes[self.calls]
        self.calls += 1
        if isinstance(v, Exception):
            raise v
        return dict(v)


def test_all_fail_to_refute_confirms():
    rt = _ScriptedRuntime([{"refuted": False, "reason": "plausible"}] * 3)
    hyp = MultiSkepticVerifier(rt).verify(_lead(), _gap(), _matrix())
    assert hyp.verdict == "confirmed"
    assert hyp.verifier_votes == 3
    assert any("Urbach" in hyp.mechanism for _ in [0])


def test_majority_refute_refutes():
    rt = _ScriptedRuntime([{"refuted": True, "reason": "no support"},
                           {"refuted": True, "reason": "artifact"},
                           {"refuted": False, "reason": "maybe"}])
    hyp = MultiSkepticVerifier(rt).verify(_lead(), _gap(), _matrix())
    assert hyp.verdict == "refuted"
    assert any("artifact" in e for e in hyp.evidence_against)


def test_below_quorum_stays_uncertain():
    # only 1 of 3 skeptics succeeds (2 raise) -> < quorum(2) -> unchanged uncertain
    rt = _ScriptedRuntime([RuntimeError("x"), {"refuted": True, "reason": "r"}, RuntimeError("y")])
    hyp = MultiSkepticVerifier(rt).verify(_lead(), _gap(), _matrix())
    assert hyp.verdict == "uncertain"          # NOT refuted on a single errored-heavy run


def test_errored_skeptic_excluded_not_counted_as_refute():
    # 2 succeed (both fail-to-refute) + 1 errors -> quorum met, no refutes -> confirmed
    rt = _ScriptedRuntime([{"refuted": False, "reason": "ok"}, RuntimeError("x"),
                           {"refuted": False, "reason": "ok"}])
    hyp = MultiSkepticVerifier(rt).verify(_lead(), _gap(), _matrix())
    assert hyp.verdict == "confirmed" and hyp.verifier_votes == 2


def test_refute_prompt_has_mechanism_and_lens():
    p = refute_prompt(_lead(), _gap(), _matrix(), "numerical-artifact")
    assert "Urbach" in p and "numerical-artifact" in p and "JSON" in p


def test_gate_g5_verify_maps_verdict():
    rt = _ScriptedRuntime([{"refuted": False, "reason": "ok"}] * 3)
    v = gate_g5_verify(_lead(), _gap(), _matrix(), MultiSkepticVerifier(rt))
    assert v.name == "G5_adversarial_verify" and v.passed is True


def test_lenses_default_is_three():
    assert len(SKEPTIC_LENSES) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_verify.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'perovskite_sim.autoloop.verify'`.

- [ ] **Step 3: Write `verify.py`**

```python
# perovskite_sim/autoloop/verify.py
from __future__ import annotations

import dataclasses
import logging

from perovskite_sim.autoloop.cognition import CognitionRuntime
from perovskite_sim.autoloop.types import GateVerdict, Hypothesis

logger = logging.getLogger(__name__)

SKEPTIC_LENSES = ["physical-plausibility", "numerical-artifact", "data-support"]
VOTE_SCHEMA = {"required": ["refuted"]}

_LENS_QUESTION = {
    "physical-plausibility": "Could this mechanism physically produce the observed gap in a "
                             "perovskite drift-diffusion model?",
    "numerical-artifact": "Could the gap be a numerical/discretisation artifact rather than this "
                          "mechanism? Does the ablation evidence rule that out?",
    "data-support": "Do the ablation probes actually SUPPORT this mechanism, or is the claim "
                    "unsupported by the evidence shown?",
}


def refute_prompt(hyp, gap, matrix, lens: str) -> str:
    probes = "\n".join(f"  - {p.name} [{p.kind}]: delta={p.delta:.4g} ok={p.ok}"
                       for p in matrix.probes)
    return (
        f"You are a skeptic reviewing a proposed root-cause via the '{lens}' lens.\n\n"
        f"CLAIM: cause={hyp.cause}; mechanism={hyp.mechanism}\n"
        f"GAP: metric={gap.metric}, sweep={gap.sweep}, "
        f"solarlab={gap.solarlab_val:.4g} vs reference={gap.reference_val:.4g}\n"
        f"ABLATION EVIDENCE (delta<0 = that variant improved the gap):\n{probes}\n\n"
        f"{_LENS_QUESTION.get(lens, 'Is the claim well-supported?')}\n"
        "Try to REFUTE the claim from your lens. Default refuted=true if you cannot find solid "
        "support. Output ONLY a JSON object: "
        '{"refuted": true|false, "reason": "<one sentence>"}')


class MultiSkepticVerifier:
    """N diverse-lens skeptics each try to refute a hypothesis mechanism. Strict
    majority of a quorum decides; errored skeptics are excluded (never counted as
    a refutation); below quorum the hypothesis is returned unchanged (uncertain)."""

    def __init__(self, runtime: CognitionRuntime, *, lenses=None, quorum: int = 2):
        self.runtime = runtime
        self.lenses = lenses or SKEPTIC_LENSES
        self.quorum = quorum

    def verify(self, hyp, gap, matrix) -> Hypothesis:
        ran = []   # (lens, refuted, reason) for skeptics that succeeded
        for lens in self.lenses:
            try:
                v = self.runtime.complete(refute_prompt(hyp, gap, matrix, lens), VOTE_SCHEMA)
                ran.append((lens, bool(v["refuted"]), str(v.get("reason", ""))))
            except Exception as exc:               # excluded, NOT a refutation
                logger.warning("G5 skeptic %s failed: %r", lens, exc)
        if len(ran) < self.quorum:
            logger.warning("G5 quorum not met (%d/%d) — leaving %s uncertain",
                           len(ran), self.quorum, hyp.gap_id)
            return hyp
        refutes = sum(1 for _, r, _ in ran if r)
        verdict = "refuted" if refutes > len(ran) / 2 else "confirmed"
        ev_for = tuple(f"G5 {lens}: {reason}" for lens, r, reason in ran if not r)
        ev_against = tuple(f"G5 {lens}: {reason}" for lens, r, reason in ran if r)
        return dataclasses.replace(
            hyp, verdict=verdict, verifier_votes=len(ran) - refutes,
            evidence_for=hyp.evidence_for + ev_for,
            evidence_against=hyp.evidence_against + ev_against)


def gate_g5_verify(hyp, gap, matrix, verifier: MultiSkepticVerifier) -> GateVerdict:
    """Thin wrapper filling the gate_g5 stub (pre-land reuse)."""
    out = verifier.verify(hyp, gap, matrix)
    return GateVerdict("G5_adversarial_verify", out.verdict == "confirmed",
                       f"verdict={out.verdict}, votes={out.verifier_votes}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_verify.py`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add perovskite-sim/perovskite_sim/autoloop/verify.py perovskite-sim/tests/unit/autoloop/test_verify.py
git commit -m "feat(autoloop): add MultiSkepticVerifier (G5 diverse-lens adversarial verify, Stage 5.2)"
```

---

## Task 2: Wire the verifier into attribute_top_gap

**Files:**
- Modify: `perovskite_sim/autoloop/orchestrator.py` (`attribute_top_gap`)
- Test: `tests/unit/autoloop/test_orchestrator_verify.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/autoloop/test_orchestrator_verify.py
import dataclasses
from perovskite_sim.autoloop.types import Gap, Hypothesis, AblationMatrix, AblationProbe
from perovskite_sim.autoloop.ledger import Ledger
from perovskite_sim.autoloop.orchestrator import attribute_top_gap


def _gap(gid="trend:Et_PVK ETL:V_oc"):
    return Gap(id=gid, metric="V_oc", sweep="Et_PVK ETL", sweep_point=0.0,
               solarlab_val=30.0, reference_val=70.0, gap_mag=0.4, kind="trend",
               status="open", found_cycle=0, last_attempt_cycle=0, mechanism=None)


def _fake_ablation(gap, probe_runner):
    return AblationMatrix(gap_id=gap.id, baseline_val=40.0,
                          probes=(AblationProbe("f", "flag", {}, 40.0, 39.9, -0.1, True),))


class _LeadAttributor:
    """Returns an LLM lead (uncertain + a real cause)."""
    def attribute(self, gap, matrix, negatives):
        return Hypothesis(gap_id=gap.id, cause="physics",
                          mechanism="missing band-tail Urbach absorption", verdict="uncertain")


class _NoLeadAttributor:
    """Heuristic no-op uncertain (cause uncertain, no real mechanism)."""
    def attribute(self, gap, matrix, negatives):
        return Hypothesis(gap_id=gap.id, cause="uncertain",
                          mechanism="no single ablation lever identified", verdict="uncertain")


class _Verifier:
    def __init__(self, verdict):
        self.verdict = verdict
        self.calls = 0
    def verify(self, hyp, gap, matrix):
        self.calls += 1
        return dataclasses.replace(hyp, verdict=self.verdict)


def _setup(tmp_path):
    led = Ledger(root=tmp_path / "ledger")
    led.add_gap(_gap())
    led.save()


def test_lead_confirmed_sets_mechanism(tmp_path):
    _setup(tmp_path)
    v = _Verifier("confirmed")
    attribute_top_gap(ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
                      config_path=tmp_path / "c.yaml", reference_path=tmp_path / "r.json",
                      cycle=1, timestamp="t", probe_runner_factory=lambda g: object(),
                      attributor=_LeadAttributor(), run_ablation_fn=_fake_ablation, verifier=v)
    assert v.calls == 1
    led = Ledger.load(tmp_path / "ledger")
    g = next(g for g in led.gaps if g.id == "trend:Et_PVK ETL:V_oc")
    assert g.mechanism == "missing band-tail Urbach absorption"   # confirmed -> mechanism set


def test_lead_refuted_adds_negative(tmp_path):
    _setup(tmp_path)
    attribute_top_gap(ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
                      config_path=tmp_path / "c.yaml", reference_path=tmp_path / "r.json",
                      cycle=1, timestamp="t", probe_runner_factory=lambda g: object(),
                      attributor=_LeadAttributor(), run_ablation_fn=_fake_ablation,
                      verifier=_Verifier("refuted"))
    led = Ledger.load(tmp_path / "ledger")
    assert led.is_refuted("missing band-tail Urbach absorption")  # refuted -> negatives ledger


def test_heuristic_noop_skips_verifier(tmp_path):
    _setup(tmp_path)
    v = _Verifier("confirmed")
    attribute_top_gap(ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
                      config_path=tmp_path / "c.yaml", reference_path=tmp_path / "r.json",
                      cycle=1, timestamp="t", probe_runner_factory=lambda g: object(),
                      attributor=_NoLeadAttributor(), run_ablation_fn=_fake_ablation, verifier=v)
    assert v.calls == 0                                            # cause=="uncertain" -> not verified


def test_no_verifier_is_unchanged(tmp_path):
    _setup(tmp_path)
    hyp = attribute_top_gap(ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
                            config_path=tmp_path / "c.yaml", reference_path=tmp_path / "r.json",
                            cycle=1, timestamp="t", probe_runner_factory=lambda g: object(),
                            attributor=_LeadAttributor(), run_ablation_fn=_fake_ablation)
    assert hyp.verdict == "uncertain"                             # no verifier -> lead stays
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_orchestrator_verify.py`
Expected: FAIL — `attribute_top_gap` has no `verifier` parameter (`TypeError: unexpected keyword argument 'verifier'`).

- [ ] **Step 3: Edit `attribute_top_gap`**

Add `verifier=None` to the signature (after `run_ablation_fn=None`):

```python
                      run_ablation_fn=None, verifier=None) -> Optional[Hypothesis]:
```

Replace the block from `hyp = attributor.attribute(gap, matrix, led)` through `led.save()`:

```python
    hyp = attributor.attribute(gap, matrix, led)

    # G5 (Stage 5.2): adjudicate an LLM novel-cause lead before recording it.
    if verifier is not None and hyp.verdict == "uncertain" and hyp.cause != "uncertain":
        hyp = verifier.verify(hyp, gap, matrix)

    led.add_hypothesis(hyp)
    if hyp.verdict == "refuted":
        led.add_negative(NegativeResult(
            approach=hyp.mechanism, why_failed="refuted by G5 multi-skeptic verify",
            evidence=f"attribution cycle {cycle}"))
    if hyp.verdict == "confirmed":
        led.add_gap(gap.with_mechanism(hyp.mechanism))   # add_gap replaces on id
    led.save()
```

(`NegativeResult` is already imported in `orchestrator.py`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_orchestrator_verify.py`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add perovskite-sim/perovskite_sim/autoloop/orchestrator.py perovskite-sim/tests/unit/autoloop/test_orchestrator_verify.py
git commit -m "feat(autoloop): wire G5 verifier into attribute_top_gap (confirm/refute LLM leads)"
```

---

## Task 3: CLI `--verify` + opt-in smoke + docs

**Files:**
- Modify: `scripts/autoloop_run.py`
- Create: `tests/integration/test_autoloop_g5.py`
- Modify: `perovskite-sim/CLAUDE.md`, `README.md`

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_autoloop_g5.py
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


def test_build_verifier_none_without_flag():
    mod = _load_cli()
    assert mod._build_verifier(mod.parse_args(["--attribute"])) is None


def test_build_verifier_with_flag():
    mod = _load_cli()
    from perovskite_sim.autoloop.verify import MultiSkepticVerifier
    assert isinstance(mod._build_verifier(mod.parse_args(["--attribute", "--verify"])),
                      MultiSkepticVerifier)


@pytest.mark.slow
@pytest.mark.skipif(not shutil.which("claude") or not os.environ.get("SOLARLAB_LLM_SMOKE"),
                    reason="opt-in real-LLM smoke (set SOLARLAB_LLM_SMOKE=1 + claude installed)")
def test_real_g5_verify_returns_verdict(tmp_path):
    from perovskite_sim.autoloop.cognition import ClaudeCliRuntime
    from perovskite_sim.autoloop.verify import MultiSkepticVerifier
    from perovskite_sim.autoloop.types import Gap, AblationProbe, AblationMatrix, Hypothesis
    gap = Gap(id="g", metric="V_oc", sweep="Et_PVK ETL", sweep_point=0.0, solarlab_val=30.0,
              reference_val=70.0, gap_mag=0.4, kind="trend", status="open",
              found_cycle=0, last_attempt_cycle=0, mechanism=None)
    matrix = AblationMatrix(gap_id="g", baseline_val=40.0,
                            probes=(AblationProbe("f", "flag", {}, 40.0, 39.9, -0.1, True),))
    lead = Hypothesis(gap_id="g", cause="physics",
                      mechanism="missing band-tail Urbach absorption", verdict="uncertain")
    out = MultiSkepticVerifier(ClaudeCliRuntime(model="sonnet")).verify(lead, gap, matrix)
    assert out.verdict in {"confirmed", "refuted", "uncertain"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd perovskite-sim && python -m pytest -q tests/integration/test_autoloop_g5.py`
Expected: FAIL — `_build_verifier` does not exist (real-LLM smoke skipped by default).

- [ ] **Step 3: Wire the CLI + docs**

In `scripts/autoloop_run.py`, add the flag to `parse_args`:

```python
    ap.add_argument("--verify", action="store_true",
                    help="adjudicate LLM novel-cause leads with the G5 multi-skeptic verifier")
```

Add the helper next to `_build_attributor`:

```python
def _build_verifier(ns):
    if not getattr(ns, "verify", False):
        return None
    from perovskite_sim.autoloop.cognition import ClaudeCliRuntime
    from perovskite_sim.autoloop.verify import MultiSkepticVerifier
    return MultiSkepticVerifier(ClaudeCliRuntime(model=ns.llm_model))
```

In the `--attribute` dispatch, add `verifier=_build_verifier(ns)` to the `attribute_top_gap(...)` call. In the `--boulder` dispatch, compute `_verifier = _build_verifier(ns)` next to `_attr = _build_attributor(ns)`, and add `verifier=_verifier` to the `attribute_top_gap(...)` call inside the `attribute(cycle)` closure.

Add to the **Autoloop** section of `perovskite-sim/CLAUDE.md`:

```markdown
### Stage 5.2 — G5 multi-skeptic verify (cognition leg 2)

`autoloop/verify.py:MultiSkepticVerifier` adjudicates an LLM novel-cause lead (5.1):
N=3 diverse-lens skeptics (physical-plausibility / numerical-artifact / data-support),
each prompted via the `CognitionRuntime` to REFUTE the mechanism (default refuted if no
solid support). Strict majority of a quorum (≥2 skeptics that ran) decides: survives →
`verdict="confirmed"` (now Stage-3-actionable); majority-refute → `verdict="refuted"` +
`add_negative` (5.1's LLM never re-proposes it). An errored skeptic is EXCLUDED (never a
false refute); below quorum the lead stays `uncertain` (re-verify later). Wired into
`attribute_top_gap` (verifies LLM leads only — `cause != "uncertain"`) and fills the
`gate_g5` stub via `gate_g5_verify`. Opt-in, with `--llm`:

    cd perovskite-sim
    python scripts/autoloop_run.py --attribute --llm --verify
    python scripts/autoloop_run.py --boulder --llm --verify

5.3 (LLM codegen) is the last deferred leg.
```

Add to `README.md` (next to the LLM attributor line):

```markdown
- **Autoloop G5 verify** (`--attribute --llm --verify`) — N diverse-lens skeptics
  adversarially adjudicate an LLM novel-cause lead: survives → confirmed (actionable),
  majority-refute → refuted + recorded as never-retry. Opt-in.
```

- [ ] **Step 4: Run tests**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop tests/integration/test_autoloop_g5.py`
Expected: all green (real-LLM smoke skipped without `SOLARLAB_LLM_SMOKE`). Also `python -m pytest -q` (full default suite) — confirm no import/collection regression.

- [ ] **Step 5: Commit**

```bash
git add perovskite-sim/scripts/autoloop_run.py perovskite-sim/tests/integration/test_autoloop_g5.py perovskite-sim/CLAUDE.md README.md
git commit -m "feat(autoloop): wire --verify (G5) into CLI + opt-in smoke + docs (Stage 5.2)"
```

---

## Self-Review

**Spec coverage** (vs `2026-06-17-autoloop-stage5.2-g5-verify-design.md`):
- §3 `SKEPTIC_LENSES` + `VOTE_SCHEMA` + `refute_prompt` + `MultiSkepticVerifier` (quorum, exclude-errored, strict-majority, evidence) + `gate_g5_verify` → Task 1. ✓
- §2/§4 `attribute_top_gap` verifier wiring (verify LLM leads only; confirm→mechanism; refute→add_negative) → Task 2. ✓
- §4 CLI `--verify` + `_build_verifier` into `--attribute` + `--boulder` → Task 3. ✓
- §5 error handling (quorum → uncertain; errored excluded; refute → negative) → Tasks 1/2. ✓
- §6 testing (all-fail/majority-refute/below-quorum/excluded-errored/prompt-content/gate-map; orchestrator confirm/refute/noop-skip/no-verifier; opt-in smoke) → every task. ✓
- §7 deferred (5.3, primary pre-land gate, heuristic-cause verify, tuning) → correctly NOT built.

**Placeholder scan:** none — complete code/tests/commands. The opt-in smoke skip is explicit.

**Type consistency:** `MultiSkepticVerifier(runtime, lenses=, quorum=).verify(hyp, gap, matrix) -> Hypothesis` consistent Tasks 1/2/3 + tests. `refute_prompt(hyp, gap, matrix, lens)`, `gate_g5_verify(hyp, gap, matrix, verifier)`, `VOTE_SCHEMA={"required":["refuted"]}` consistent. `attribute_top_gap(..., verifier=None)` matches the Task-2 tests + the CLI calls. `_build_verifier(ns)` returns the seam type both dispatches inject. `dataclasses.replace` on the frozen `Hypothesis`, `NegativeResult`, `GateVerdict`, `Ledger.add_negative`/`is_refuted`, `gap.with_mechanism`, `cognition.FakeRuntime`/`CognitionRuntime` are verified real symbols.

---

## Execution Handoff

After saving, choose execution:

1. **Subagent-Driven (recommended)** — fresh subagent per task, two-stage review (same background-workflow as prior stages).
2. **Inline Execution** — batch tasks in this session with checkpoints.
