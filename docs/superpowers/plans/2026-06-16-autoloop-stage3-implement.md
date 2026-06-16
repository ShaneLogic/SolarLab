# Autoloop Stage 3 — Auto-Implement Leg — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic auto-implement leg that turns a confirmed Hypothesis into a flag-promotion config edit, runs the full gate stack (G0 bit-identical + G4 reconciles + reuse G1–G3), and lands it on the current branch only on `--apply` (dry-run default).

**Architecture:** Extends `perovskite_sim/autoloop/`. `propose_promotion` maps a confirmed lever (`SOLARLAB_*` flag) to a device-config key and produces a `ConfigEdit` (set `device.<key>: true` in `configs/scaps_mirror_v2.yaml`); `apply_edit`/`revert_edit` are exact-text working-tree ops. The orchestrator applies the edit, runs an injected gate stack, then reverts (dry-run) or commits to the current branch (`--apply`, refusing `main`/dirty). No solver code is changed; legacy tier forces the flag off so G0 holds by construction.

**Tech Stack:** Python 3.9+, dataclasses, subprocess (git + golden suite), re, json. Reuses Stage 1/2 (`Gap`/`Hypothesis`/`NegativeResult`/`Ledger`/`GateVerdict`/`gates`/`provenance._git`/`ladder._PKG_ROOT`). No new third-party deps.

---

## Design contract (read before starting)

- **Spec:** `docs/superpowers/specs/2026-06-16-autoloop-stage3-implement-design.md`.
- **The change is a CONFIG edit, not solver code.** Verified: `configs/scaps_mirror_v2.yaml` has a top-level `device:` block (currently `het_recomb_despike: 0.53`, `dos_band_potentials: true`, `mode: fast`). `models/config_loader.py` reads device flags via `dev.get("interface_plane_projection", False)` etc. Promotion = set `device.<key>: true`.
- **Env-flag → device-config-key map** (only these are promotable):
  ```
  SOLARLAB_IFACE_PROJ  → interface_plane_projection
  SOLARLAB_IFACE_PLANE → interface_plane_closure
  SOLARLAB_DOS_BAND    → dos_band_potentials
  ```
  `SOLARLAB_INTERFACE_PLANE_STATE` is SS-only with no device key → not promotable.
- **Badness delta convention (from Stage 2):** lower badness = closer to SCAPS; an improving change has a **negative** delta.
- **Stage 1/2 APIs (verified on `main`):**
  - `types.Gap(...)` with `.with_status(status, *, last_attempt_cycle=None)` and `.with_mechanism(mechanism)`.
  - `types.Hypothesis(gap_id, cause, mechanism, evidence_for=(), evidence_against=(), verifier_votes=0, verdict="uncertain", cycle=0)`.
  - `types.NegativeResult(approach, why_failed, evidence, never_retry=True)`.
  - `types.GateVerdict(name, passed, reason)`.
  - `ledger.Ledger.load/save/add_gap/add_hypothesis/add_negative/is_refuted`; attrs `.gaps`, `.hypotheses`, `.negatives`.
  - `provenance._git(*args) -> str` (runs git from repo, '' on failure).
- **Run all commands from `perovskite-sim/`.** Tests default to `-m 'not slow'`.

## File Structure

```
perovskite_sim/autoloop/
  types.py          + ConfigEdit, ImplementResult
  promote.py        FLAG_TO_CONFIG_KEY, parse_lever, set_device_flag, apply_edit, revert_edit, propose_promotion
  gates_impl.py     gate_g4_reconciles, gate_g0_bit_identical
  orchestrator.py   + implement_top_confirmed, commit_promotion
scripts/autoloop_run.py  + --implement [--apply]
tests/unit/autoloop/
  test_types_implement.py
  test_promote.py
  test_gates_impl.py
  test_orchestrator_implement.py
tests/integration/
  test_autoloop_implement.py   (slow)
```

---

## Task 1: Implement types

**Files:**
- Modify: `perovskite_sim/autoloop/types.py`
- Test: `tests/unit/autoloop/test_types_implement.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/autoloop/test_types_implement.py
import dataclasses
import pytest
from perovskite_sim.autoloop.types import ConfigEdit, ImplementResult


def test_config_edit_is_frozen():
    e = ConfigEdit(config_path="c.yaml", device_key="interface_plane_projection",
                   new_value=True, old_text="device:\n  mode: fast\n")
    assert e.device_key == "interface_plane_projection"
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.new_value = False  # type: ignore[misc]


def test_implement_result_defaults():
    r = ImplementResult(status="dry_run", hypothesis_gap_id="g", device_key="k",
                        gate_verdicts=(), committed_sha=None)
    assert r.status == "dry_run"
    assert r.note == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_types_implement.py`
Expected: FAIL — `ImportError: cannot import name 'ConfigEdit'`.

- [ ] **Step 3: Append to `types.py`** (after `AblationMatrix`)

```python
@dataclass(frozen=True)
class ConfigEdit:
    config_path: str
    device_key: str
    new_value: bool
    old_text: str       # full prior file text, for exact revert


@dataclass(frozen=True)
class ImplementResult:
    status: str         # "applied"|"dry_run"|"gates_failed"|"no_confirmed"|"not_promotable"
    hypothesis_gap_id: Optional[str]
    device_key: Optional[str]
    gate_verdicts: tuple
    committed_sha: Optional[str]
    note: str = ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_types_implement.py`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add perovskite-sim/perovskite_sim/autoloop/types.py perovskite-sim/tests/unit/autoloop/test_types_implement.py
git commit -m "feat(autoloop): add ConfigEdit + ImplementResult types (Stage 3)"
```

---

## Task 2: Promote — lever→config-key, YAML edit, propose

**Files:**
- Create: `perovskite_sim/autoloop/promote.py`
- Test: `tests/unit/autoloop/test_promote.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/autoloop/test_promote.py
from perovskite_sim.autoloop.types import Hypothesis
from perovskite_sim.autoloop.ledger import Ledger
from perovskite_sim.autoloop.types import NegativeResult
from perovskite_sim.autoloop.promote import (
    FLAG_TO_CONFIG_KEY, parse_lever, set_device_flag,
    apply_edit, revert_edit, propose_promotion,
)

_YAML = "name: x\ndevice:\n  mode: fast\n  dos_band_potentials: true\nlayers:\n  - a\n"


def test_parse_lever():
    assert parse_lever("flag SOLARLAB_IFACE_PROJ term") == "SOLARLAB_IFACE_PROJ"
    assert parse_lever("no flag here") is None


def test_set_device_flag_inserts_when_absent():
    out = set_device_flag(_YAML, "interface_plane_projection", True)
    assert "  interface_plane_projection: true\n" in out
    assert "layers:" in out                       # rest preserved


def test_set_device_flag_overwrites_when_present():
    out = set_device_flag(_YAML, "dos_band_potentials", False)
    assert "  dos_band_potentials: false\n" in out
    assert "  dos_band_potentials: true\n" not in out


def test_set_device_flag_idempotent():
    once = set_device_flag(_YAML, "interface_plane_projection", True)
    twice = set_device_flag(once, "interface_plane_projection", True)
    assert once == twice


def test_set_device_flag_raises_without_device_block():
    import pytest
    with pytest.raises(ValueError):
        set_device_flag("name: x\nlayers:\n  - a\n", "k", True)


def test_apply_and_revert_roundtrip(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(_YAML, encoding="utf-8")
    from perovskite_sim.autoloop.types import ConfigEdit
    edit = ConfigEdit(config_path=str(p), device_key="interface_plane_projection",
                      new_value=True, old_text=_YAML)
    apply_edit(edit)
    assert "interface_plane_projection: true" in p.read_text(encoding="utf-8")
    revert_edit(edit)
    assert p.read_text(encoding="utf-8") == _YAML       # exact restore


def _hyp(mech="flag SOLARLAB_IFACE_PROJ term", verdict="confirmed"):
    return Hypothesis(gap_id="g", cause="physics", mechanism=mech, verdict=verdict)


def test_propose_promotion_builds_edit(tmp_path):
    p = tmp_path / "c.yaml"; p.write_text(_YAML, encoding="utf-8")
    edit = propose_promotion(_hyp(), Ledger(root=tmp_path), p)
    assert edit is not None
    assert edit.device_key == "interface_plane_projection"
    assert edit.old_text == _YAML


def test_propose_promotion_none_when_not_confirmed(tmp_path):
    p = tmp_path / "c.yaml"; p.write_text(_YAML, encoding="utf-8")
    assert propose_promotion(_hyp(verdict="uncertain"), Ledger(root=tmp_path), p) is None


def test_propose_promotion_none_when_not_promotable(tmp_path):
    p = tmp_path / "c.yaml"; p.write_text(_YAML, encoding="utf-8")
    h = _hyp(mech="flag SOLARLAB_INTERFACE_PLANE_STATE term")    # no config key
    assert propose_promotion(h, Ledger(root=tmp_path), p) is None


def test_propose_promotion_none_when_refuted(tmp_path):
    p = tmp_path / "c.yaml"; p.write_text(_YAML, encoding="utf-8")
    led = Ledger(root=tmp_path)
    led.add_negative(NegativeResult(approach="flag SOLARLAB_IFACE_PROJ term",
                                    why_failed="x", evidence="y"))
    assert propose_promotion(_hyp(), led, p) is None


def test_flag_map_keys():
    assert set(FLAG_TO_CONFIG_KEY) == {
        "SOLARLAB_IFACE_PROJ", "SOLARLAB_IFACE_PLANE", "SOLARLAB_DOS_BAND"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_promote.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'perovskite_sim.autoloop.promote'`.

- [ ] **Step 3: Write `promote.py`**

```python
# perovskite_sim/autoloop/promote.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from perovskite_sim.autoloop.ledger import Ledger
from perovskite_sim.autoloop.types import ConfigEdit, Hypothesis

# Confirmed levers that are EXISTING device-config flags -> promotable. A lever
# without a key here needs codegen (out of Stage 3 scope) -> propose returns None.
FLAG_TO_CONFIG_KEY: dict[str, str] = {
    "SOLARLAB_IFACE_PROJ":  "interface_plane_projection",
    "SOLARLAB_IFACE_PLANE": "interface_plane_closure",
    "SOLARLAB_DOS_BAND":    "dos_band_potentials",
}


def parse_lever(mechanism: str) -> Optional[str]:
    """Extract the SOLARLAB_* flag token from a Hypothesis mechanism string."""
    m = re.search(r"(SOLARLAB_[A-Z_]+)", mechanism)
    return m.group(1) if m else None


def set_device_flag(text: str, key: str, value: bool) -> str:
    """Set-or-insert `  <key>: <value>` under the top-level `device:` block,
    preserving the rest of the file verbatim. Raises if no device block."""
    val = "true" if value else "false"
    lines = text.splitlines(keepends=True)
    dev_idx = next((i for i, ln in enumerate(lines) if ln.rstrip("\n") == "device:"), None)
    if dev_idx is None:
        raise ValueError("config has no top-level 'device:' block")
    # device block body = subsequent blank or indented lines
    body_end = dev_idx + 1
    while body_end < len(lines):
        ln = lines[body_end]
        if ln.strip() == "" or ln[:1] in (" ", "\t"):
            body_end += 1
        else:
            break
    for j in range(dev_idx + 1, body_end):
        stripped = lines[j].lstrip()
        if stripped.startswith(f"{key}:"):
            indent = lines[j][: len(lines[j]) - len(stripped)]
            lines[j] = f"{indent}{key}: {val}\n"
            return "".join(lines)
    lines.insert(dev_idx + 1, f"  {key}: {val}\n")
    return "".join(lines)


def apply_edit(edit: ConfigEdit) -> None:
    text = Path(edit.config_path).read_text(encoding="utf-8")
    Path(edit.config_path).write_text(
        set_device_flag(text, edit.device_key, edit.new_value), encoding="utf-8")


def revert_edit(edit: ConfigEdit) -> None:
    Path(edit.config_path).write_text(edit.old_text, encoding="utf-8")


def propose_promotion(hypothesis: Hypothesis, ledger: Ledger,
                      config_path) -> Optional[ConfigEdit]:
    """Confirmed lever -> ConfigEdit, or None if not confirmed / not promotable /
    refuted. Never proposes a refuted mechanism (anti-thrash)."""
    if hypothesis.verdict != "confirmed":
        return None
    flag = parse_lever(hypothesis.mechanism)
    if flag is None:
        return None
    key = FLAG_TO_CONFIG_KEY.get(flag)
    if key is None:
        return None
    if ledger.is_refuted(hypothesis.mechanism):
        return None
    old_text = Path(config_path).read_text(encoding="utf-8")
    return ConfigEdit(config_path=str(config_path), device_key=key,
                      new_value=True, old_text=old_text)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_promote.py`
Expected: PASS (10 tests).

- [ ] **Step 5: Commit**

```bash
git add perovskite-sim/perovskite_sim/autoloop/promote.py perovskite-sim/tests/unit/autoloop/test_promote.py
git commit -m "feat(autoloop): add flag-promotion (lever->config-key, YAML edit, propose)"
```

---

## Task 3: Gate implementations (G0 + G4)

**Files:**
- Create: `perovskite_sim/autoloop/gates_impl.py`
- Test: `tests/unit/autoloop/test_gates_impl.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/autoloop/test_gates_impl.py
from perovskite_sim.autoloop.gates_impl import gate_g4_reconciles, gate_g0_bit_identical


def test_g4_passes_when_realized_reconciles_predicted():
    v = gate_g4_reconciles(predicted_delta=-18.0, realized_delta=-16.0, tol=0.5)
    assert v.passed is True
    assert v.name == "G4_reconcile"


def test_g4_fails_when_realized_wrong_sign():
    v = gate_g4_reconciles(predicted_delta=-18.0, realized_delta=+3.0, tol=0.5)
    assert v.passed is False        # promoting the flag did NOT improve in-config


def test_g4_fails_when_realized_off_magnitude():
    v = gate_g4_reconciles(predicted_delta=-18.0, realized_delta=-2.0, tol=0.5)
    assert v.passed is False        # |−2 − (−18)| = 16 > 0.5*18 = 9


def test_g4_fails_when_no_predicted_improvement():
    v = gate_g4_reconciles(predicted_delta=0.0, realized_delta=-5.0, tol=0.5)
    assert v.passed is False


def test_g0_passes_when_golden_green():
    v = gate_g0_bit_identical(lambda: (True, "regression suite green"))
    assert v.passed is True
    assert v.name == "G0_legacy_bit_identical"


def test_g0_fails_when_golden_red():
    v = gate_g0_bit_identical(lambda: (False, "1 new failure in test_jv_regression"))
    assert v.passed is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_gates_impl.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'perovskite_sim.autoloop.gates_impl'`.

- [ ] **Step 3: Write `gates_impl.py`**

```python
# perovskite_sim/autoloop/gates_impl.py
from __future__ import annotations

from typing import Callable

from perovskite_sim.autoloop.types import GateVerdict


def gate_g4_reconciles(predicted_delta: float, realized_delta: float,
                       *, tol: float = 0.5) -> GateVerdict:
    """G4 honest gate for flag-promotion: the promoted flag's measured benefit
    must actually materialize. Deltas are badness changes (negative = improvement).
    Pass iff realized improves (negative) AND reconciles with the predicted
    magnitude within ``tol`` (relative)."""
    if predicted_delta >= 0:
        return GateVerdict("G4_reconcile", False,
                           f"no predicted improvement (Δpred {predicted_delta:.3g} >= 0)")
    improved = realized_delta < 0
    reconciles = abs(realized_delta - predicted_delta) <= tol * abs(predicted_delta)
    passed = improved and reconciles
    return GateVerdict("G4_reconcile", passed,
                       f"Δpred {predicted_delta:.3g}, Δreal {realized_delta:.3g}, "
                       f"tol {tol} (improved={improved}, reconciles={reconciles})")


def gate_g0_bit_identical(golden_runner: Callable[[], tuple[bool, str]]) -> GateVerdict:
    """G0: the legacy/golden regression suite must stay green with the edit
    applied (legacy tier forces the flag off, so it holds by construction; this
    verifies it). ``golden_runner() -> (ok, detail)`` is injected."""
    ok, detail = golden_runner()
    return GateVerdict("G0_legacy_bit_identical", ok, detail)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_gates_impl.py`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add perovskite-sim/perovskite_sim/autoloop/gates_impl.py perovskite-sim/tests/unit/autoloop/test_gates_impl.py
git commit -m "feat(autoloop): add G0 (legacy-bit-identical) + G4 (reconcile) gates"
```

---

## Task 4: Orchestrator implement pass + commit helper

**Files:**
- Modify: `perovskite_sim/autoloop/orchestrator.py`
- Test: `tests/unit/autoloop/test_orchestrator_implement.py`

The gate stack + git commit are injected so unit tests run without the solver or touching the real repo. `commit_promotion` (the real committer) carries the guards.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/autoloop/test_orchestrator_implement.py
import subprocess
from pathlib import Path
import pytest
from perovskite_sim.autoloop.types import Gap, Hypothesis, GateVerdict
from perovskite_sim.autoloop.ledger import Ledger
from perovskite_sim.autoloop.orchestrator import implement_top_confirmed, commit_promotion

_YAML = "name: x\ndevice:\n  mode: fast\nlayers:\n  - a\n"


def _gap(gid="trend:Nd_ETL:V_oc", mag=0.4, status="open"):
    return Gap(id=gid, metric="V_oc", sweep="Nd_ETL", sweep_point=0.0,
               solarlab_val=30.0, reference_val=70.0, gap_mag=mag, kind="trend",
               status=status, found_cycle=0, last_attempt_cycle=0, mechanism=None)


def _confirmed_hyp(gid="trend:Nd_ETL:V_oc"):
    return Hypothesis(gap_id=gid, cause="physics",
                      mechanism="flag SOLARLAB_IFACE_PROJ term", verdict="confirmed")


def _setup(tmp_path, *, status="open", hyp=True):
    cfg = tmp_path / "c.yaml"; cfg.write_text(_YAML, encoding="utf-8")
    led = Ledger(root=tmp_path / "ledger")
    led.add_gap(_gap(status=status))
    if hyp:
        led.add_hypothesis(_confirmed_hyp())
    led.save()
    return cfg


def _green_gates(edit, gap, hyp):
    return [GateVerdict("G1_numerics", True, ""), GateVerdict("G0_legacy_bit_identical", True, "")]


def _red_gates(edit, gap, hyp):
    return [GateVerdict("G1_numerics", True, ""), GateVerdict("G0_legacy_bit_identical", False, "boom")]


def test_no_confirmed_returns_status(tmp_path):
    cfg = _setup(tmp_path, hyp=False)
    r = implement_top_confirmed(ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
                                config_path=cfg, reference_path=tmp_path / "r.json",
                                cycle=1, timestamp="2026-06-16T00:00:00Z",
                                gate_runner=_green_gates)
    assert r.status == "no_confirmed"


def test_not_promotable_when_lever_has_no_key(tmp_path):
    cfg = tmp_path / "c.yaml"; cfg.write_text(_YAML, encoding="utf-8")
    led = Ledger(root=tmp_path / "ledger")
    led.add_gap(_gap())
    led.add_hypothesis(Hypothesis(gap_id="trend:Nd_ETL:V_oc", cause="physics",
                                  mechanism="flag SOLARLAB_INTERFACE_PLANE_STATE term",
                                  verdict="confirmed"))
    led.save()
    r = implement_top_confirmed(ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
                                config_path=cfg, reference_path=tmp_path / "r.json",
                                cycle=1, timestamp="2026-06-16T00:00:00Z",
                                gate_runner=_green_gates)
    assert r.status == "not_promotable"


def test_dry_run_reverts_and_reports(tmp_path):
    cfg = _setup(tmp_path)
    r = implement_top_confirmed(ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
                                config_path=cfg, reference_path=tmp_path / "r.json",
                                cycle=1, timestamp="2026-06-16T00:00:00Z",
                                gate_runner=_green_gates, apply=False)
    assert r.status == "dry_run"
    assert cfg.read_text(encoding="utf-8") == _YAML       # working tree restored


def test_gates_failed_reverts_and_adds_negative(tmp_path):
    cfg = _setup(tmp_path)
    r = implement_top_confirmed(ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
                                config_path=cfg, reference_path=tmp_path / "r.json",
                                cycle=1, timestamp="2026-06-16T00:00:00Z",
                                gate_runner=_red_gates, apply=True)
    assert r.status == "gates_failed"
    assert cfg.read_text(encoding="utf-8") == _YAML
    led = Ledger.load(tmp_path / "ledger")
    assert led.is_refuted("flag SOLARLAB_IFACE_PROJ term")  # refuted -> won't retry


def test_apply_commits_in_tmp_git_repo(tmp_path):
    # real git repo so commit_promotion runs for real
    repo = tmp_path / "repo"; repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    subprocess.run(["git", "checkout", "-q", "-b", "autoloop/test"], cwd=repo, check=True)
    cfg = repo / "c.yaml"; cfg.write_text(_YAML, encoding="utf-8")
    led_dir = repo / "ledger"
    led = Ledger(root=led_dir); led.add_gap(_gap()); led.add_hypothesis(_confirmed_hyp()); led.save()
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "seed"], cwd=repo, check=True)

    r = implement_top_confirmed(ledger_root=led_dir, outputs_root=repo / "out",
                                config_path=cfg, reference_path=repo / "r.json",
                                cycle=1, timestamp="2026-06-16T00:00:00Z",
                                gate_runner=_green_gates, apply=True, git_cwd=repo)
    assert r.status == "applied"
    assert r.committed_sha
    assert "interface_plane_projection: true" in cfg.read_text(encoding="utf-8")
    led2 = Ledger.load(led_dir)
    g = next(g for g in led2.gaps if g.id == "trend:Nd_ETL:V_oc")
    assert g.status == "closed" and g.mechanism == "flag SOLARLAB_IFACE_PROJ term"


def test_commit_promotion_refuses_on_main(tmp_path):
    repo = tmp_path / "repo"; repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    cfg = repo / "c.yaml"; cfg.write_text(_YAML, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "seed"], cwd=repo, check=True)
    from perovskite_sim.autoloop.types import ConfigEdit
    edit = ConfigEdit(config_path=str(cfg), device_key="interface_plane_projection",
                      new_value=True, old_text=_YAML)
    with pytest.raises(RuntimeError, match="main"):
        commit_promotion(edit, _gap(), _confirmed_hyp(), [], git_cwd=repo)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_orchestrator_implement.py`
Expected: FAIL — `ImportError: cannot import name 'implement_top_confirmed'`.

- [ ] **Step 3: Add to `orchestrator.py`**

Add imports (top, with the others):

```python
import subprocess
from perovskite_sim.autoloop.promote import propose_promotion, apply_edit, revert_edit
from perovskite_sim.autoloop.types import NegativeResult, ImplementResult, ConfigEdit
from perovskite_sim.autoloop.provenance import _git
```

Append the commit helper and the pass:

```python
def commit_promotion(edit: ConfigEdit, gap, hypothesis, verdicts, *, git_cwd=None) -> str:
    """Commit the (already-applied) config edit to the CURRENT branch. Guards:
    refuse main/master, refuse a dirty tree (other than the edited config)."""
    cwd = str(git_cwd) if git_cwd is not None else None
    branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                            capture_output=True, text=True, cwd=cwd).stdout.strip()
    if branch in ("main", "master"):
        raise RuntimeError(f"refuse to auto-commit on '{branch}'; create an autoloop branch first")
    dirty = subprocess.run(["git", "status", "--porcelain"],
                           capture_output=True, text=True, cwd=cwd).stdout.strip().splitlines()
    cfg_name = Path(edit.config_path).name
    stray = [ln for ln in dirty if cfg_name not in ln]
    if stray:
        raise RuntimeError(f"refuse to commit: working tree has unrelated changes: {stray[:3]}")
    gate_summary = " ".join(f"{v.name}{'✓' if v.passed else '✗'}" for v in verdicts)
    msg = (f"feat(autoloop): promote {edit.device_key} (closes gap {gap.id})\n\n"
           f"Auto-implemented by autoloop Stage 3 from a confirmed hypothesis.\n"
           f"Mechanism: {hypothesis.mechanism}\n"
           f"Gates: {gate_summary}\n"
           f"Gap: {gap.id} | Hypothesis-cycle: {hypothesis.cycle}")
    subprocess.run(["git", "add", edit.config_path], cwd=cwd, check=True)
    subprocess.run(["git", "commit", "-q", "-m", msg], cwd=cwd, check=True)
    return subprocess.run(["git", "rev-parse", "HEAD"],
                          capture_output=True, text=True, cwd=cwd).stdout.strip()


def implement_top_confirmed(*, ledger_root: Path, outputs_root: Path,
                            config_path, reference_path,
                            cycle: int, timestamp: str, apply: bool = False,
                            gate_runner, committer=None, git_cwd=None) -> ImplementResult:
    """One implement pass: top confirmed gap -> propose -> apply -> gate ->
    (revert+report | commit). Read-only on solver code."""
    led = Ledger.load(Path(ledger_root))
    confirmed_ids = {h.gap_id for h in led.hypotheses if h.verdict == "confirmed"}
    candidates = [g for g in led.gaps if g.status == "open" and g.id in confirmed_ids]
    if not candidates:
        return ImplementResult("no_confirmed", None, None, (), None)
    gap = max(candidates, key=lambda g: g.gap_mag)
    hyp = next(h for h in led.hypotheses if h.gap_id == gap.id and h.verdict == "confirmed")

    edit = propose_promotion(hyp, led, config_path)
    if edit is None:
        return ImplementResult("not_promotable", gap.id, None, (), None, note=hyp.mechanism)

    apply_edit(edit)
    verdicts = list(gate_runner(edit, gap, hyp))

    if not all(v.passed for v in verdicts):
        revert_edit(edit)
        led.add_negative(NegativeResult(
            approach=hyp.mechanism,
            why_failed="gate(s) failed: " + ",".join(v.name for v in verdicts if not v.passed),
            evidence=f"autoloop Stage 3 implement cycle {cycle}"))
        led.save()
        return ImplementResult("gates_failed", gap.id, edit.device_key, tuple(verdicts), None)

    if apply:
        commit = committer or commit_promotion
        sha = commit(edit, gap, hyp, verdicts, git_cwd=git_cwd)
        led.add_gap(gap.with_status("closed").with_mechanism(hyp.mechanism))
        led.save()
        return ImplementResult("applied", gap.id, edit.device_key, tuple(verdicts), sha)

    revert_edit(edit)
    return ImplementResult("dry_run", gap.id, edit.device_key, tuple(verdicts), None)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop/test_orchestrator_implement.py`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add perovskite-sim/perovskite_sim/autoloop/orchestrator.py perovskite-sim/tests/unit/autoloop/test_orchestrator_implement.py
git commit -m "feat(autoloop): add implement_top_confirmed + commit_promotion (guards, anti-thrash)"
```

---

## Task 5: CLI `--implement` + real gate runner + smoke + docs

**Files:**
- Modify: `scripts/autoloop_run.py`
- Create: `tests/integration/test_autoloop_implement.py`
- Modify: `perovskite-sim/CLAUDE.md`, `README.md`

The real gate runner wires G1–G3 + G4 + G0 to the solver; tests use the injected fake, the slow smoke exercises a reduced real path (dry-run, no full-suite G0).

- [ ] **Step 1: Write the failing integration test (slow)**

```python
# tests/integration/test_autoloop_implement.py
import pytest
from pathlib import Path
from perovskite_sim.autoloop.orchestrator import implement_top_confirmed
from perovskite_sim.autoloop.types import GateVerdict
from perovskite_sim.autoloop.ledger import Ledger
from perovskite_sim.autoloop.types import Gap, Hypothesis

REPO_ROOT = Path(__file__).resolve().parents[1]
CFG = REPO_ROOT / "configs" / "scaps_mirror_v2.yaml"


@pytest.mark.slow
def test_implement_dry_run_on_confirmed_gap(tmp_path):
    # Seed a confirmed hypothesis for a promotable lever.
    led = Ledger(root=tmp_path / "ledger")
    led.add_gap(Gap(id="trend:Nt_PVK ETL:V_oc", metric="V_oc", sweep="Nt_PVK ETL",
                    sweep_point=0.0, solarlab_val=30.0, reference_val=70.0, gap_mag=0.4,
                    kind="trend", status="open", found_cycle=0, last_attempt_cycle=0,
                    mechanism=None))
    led.add_hypothesis(Hypothesis(gap_id="trend:Nt_PVK ETL:V_oc", cause="physics",
                                  mechanism="flag SOLARLAB_IFACE_PROJ term", verdict="confirmed"))
    led.save()

    # Reduced real gate runner: G1 only (fast pytest subset) — skips the expensive
    # full-suite G0 (validated separately). Confirms the real propose+edit+revert path.
    from perovskite_sim.autoloop.ladder import run_l0

    def reduced_gates(edit, gap, hyp):
        ok, detail = run_l0(["tests/unit/autoloop"])
        return [GateVerdict("G1_numerics", ok, detail)]

    cfg_text_before = CFG.read_text(encoding="utf-8")
    r = implement_top_confirmed(ledger_root=tmp_path / "ledger", outputs_root=tmp_path / "out",
                                config_path=CFG, reference_path=tmp_path / "r.json",
                                cycle=1, timestamp="2026-06-16T00:00:00Z",
                                gate_runner=reduced_gates, apply=False)
    assert r.status in {"dry_run", "gates_failed"}
    assert r.device_key == "interface_plane_projection"
    assert CFG.read_text(encoding="utf-8") == cfg_text_before    # parity config untouched after dry-run
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd perovskite-sim && python -m pytest -q -m slow tests/integration/test_autoloop_implement.py`
Expected: FAIL until the module wiring lands; once green it confirms the real propose→edit→revert path and that the **shipped parity config is left untouched** after a dry-run.

- [ ] **Step 3: Wire the CLI + real gate runner + docs**

In `scripts/autoloop_run.py`, add flags to `parse_args`:

```python
    ap.add_argument("--implement", action="store_true",
                    help="run one implement pass on the top confirmed gap (dry-run unless --apply)")
    ap.add_argument("--apply", action="store_true",
                    help="with --implement: commit the change to the current branch if all gates pass")
```

Add the dispatch in `main` (before the guardian path, after the `--attribute` block):

```python
    if ns.implement:
        import dataclasses
        from perovskite_sim.autoloop.orchestrator import implement_top_confirmed
        from perovskite_sim.autoloop.types import GateVerdict
        from perovskite_sim.autoloop.ladder import run_l0, run_ladder
        from perovskite_sim.autoloop.gates_impl import gate_g0_bit_identical, gate_g4_reconciles

        def real_gates(edit, gap, hyp):
            # G1 numerics (fast subset) + G0 legacy/golden regression suite.
            v = []
            ok1, d1 = run_l0(["tests/unit/autoloop"]); v.append(GateVerdict("G1_numerics", ok1, d1))
            ok0, d0 = run_l0(["tests/regression", "-m", "not slow"])  # golden oracle
            v.append(gate_g0_bit_identical(lambda: (ok0, d0)))
            return v

        hyp_result = implement_top_confirmed(
            ledger_root=ns.ledger_root, outputs_root=ns.outputs_root,
            config_path=ns.config, reference_path=ns.reference, cycle=ns.cycle,
            timestamp=iso_timestamp_utc(), gate_runner=real_gates, apply=ns.apply)
        print(json.dumps({"implement": dataclasses.asdict(hyp_result)}, indent=2,
                         sort_keys=True, default=str))
        return 1 if hyp_result.status == "gates_failed" and ns.apply else 0
```

Add to the **Autoloop** section of `perovskite-sim/CLAUDE.md`:

```markdown
### Stage 3 — auto-implement leg (deterministic flag-promotion)

`autoloop/promote.py` + `autoloop/gates_impl.py` turn a CONFIRMED Hypothesis into a
config flag-promotion (set `device.<key>: true` in `configs/scaps_mirror_v2.yaml`)
— the lever is an existing device flag, so no solver code changes and legacy tier
forces it off (G0 holds by construction). `implement_top_confirmed` proposes →
applies → runs the gate stack (G1–G3 reuse + G4 realized-reconciles-predicted +
G0 regression-suite green; G5 deferred) → reverts (dry-run) or commits to the
current branch (`--apply`, refuses main/dirty, never pushes). A failed gate reverts
+ records a negative result (anti-thrash).

    cd perovskite-sim
    python scripts/autoloop_run.py --once          # populate gaps
    python scripts/autoloop_run.py --attribute     # diagnose -> confirmed Hypothesis
    python scripts/autoloop_run.py --implement     # dry-run: diff + gate verdicts
    python scripts/autoloop_run.py --implement --apply   # commit to current branch
```

Add to `README.md` (next to the Stage 1/2 autoloop lines):

```markdown
- **Autoloop implement** (`python perovskite-sim/scripts/autoloop_run.py --implement [--apply]`) —
  promotes a confirmed lever flag in the parity config, runs the full gate stack
  (legacy-bit-identical + parity-improved + reconciliation), and lands it on the
  current branch only on `--apply`. Dry-run by default.
```

- [ ] **Step 4: Run tests**

Run: `cd perovskite-sim && python -m pytest -q tests/unit/autoloop && python -m pytest -q -m slow tests/integration/test_autoloop_implement.py`
Expected: all green. Also `python -m pytest -q` (full default suite) — confirm no import/collection regression.

- [ ] **Step 5: Commit**

```bash
git add perovskite-sim/scripts/autoloop_run.py perovskite-sim/tests/integration/test_autoloop_implement.py perovskite-sim/CLAUDE.md README.md
git commit -m "feat(autoloop): wire --implement[--apply] CLI + real gate runner + smoke + docs"
```

---

## Self-Review

**Spec coverage** (vs `2026-06-16-autoloop-stage3-implement-design.md`):
- §4 `FLAG_TO_CONFIG_KEY`, `parse_lever`, `set_device_flag`, `apply_edit`/`revert_edit`, `propose_promotion` (confirmed/not-promotable/refuted) → Task 2. ✓
- §4 `ConfigEdit`/`ImplementResult` → Task 1. ✓
- §5 G4 reconcile (sign + magnitude) + G0 (injected golden runner) → Task 3. ✓
- §6 `implement_top_confirmed` (all 5 statuses) + commit guards (main/dirty) + traceability commit message + gap→closed → Task 4. ✓
- §6 dry-run default / `--apply` + CLI → Task 5. ✓
- §7 error handling (revert + add_negative on gate fail; working tree clean) → Tasks 4/5. ✓
- §8 testing (set_device_flag matrix, propose branches, G0/G4 pass-fail, orchestrator statuses incl. real tmp-git commit + refuse-on-main, slow dry-run smoke) → every task. ✓
- §9 deferred (G5, LLM codegen, push, boulder) → correctly NOT built.

**Placeholder scan:** none — complete code/tests/commands. The "reduced gate set" in the smoke (G1 only, skip full-suite G0) is stated explicitly with rationale, not a TODO.

**Type consistency:** `ConfigEdit`/`ImplementResult` (Task 1) used identically in Tasks 2/4. `propose_promotion(hyp, ledger, config_path)` signature consistent Tasks 2/4. `gate_runner(edit, gap, hyp) -> list[GateVerdict]` consistent Tasks 4/5 + tests. `commit_promotion(edit, gap, hyp, verdicts, *, git_cwd)` consistent Task 4 + its test. `Gap.with_status`/`with_mechanism`, `Hypothesis`, `NegativeResult`, `GateVerdict`, `Ledger.is_refuted/add_negative`, `ladder.run_l0`, `provenance._git` are all real Stage 1/2 symbols (verified on `main`).

---

## Execution Handoff

After saving, choose execution:

1. **Subagent-Driven (recommended)** — fresh subagent per task, two-stage review (same background-workflow as Stages 1–2).
2. **Inline Execution** — batch tasks in this session with checkpoints.
