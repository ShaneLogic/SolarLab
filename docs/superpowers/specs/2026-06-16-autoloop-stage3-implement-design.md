# Autoloop Stage 3 — Auto-Implement Leg (deterministic flag-promotion) — Design

**Date:** 2026-06-16
**Status:** Design approved, pre-planning
**Parent design:** `docs/superpowers/specs/2026-06-16-autoloop-research-pipeline-design.md` (§7 gate stack, §8 propose/implement, §12 stage 3)
**Builds on:** Stage 1 (guardian) + Stage 2 (attribution), both merged to `main`. Reuses `Gap`/`Hypothesis`/`Ledger`/`build_run_callables`/`scorecard`/`gates(G1–G3)`/`ladder._PKG_ROOT`/`provenance`.

---

## 1. Problem & scope

Stage 2 produces a `confirmed` Hypothesis naming a *lever* (an existing
`SOLARLAB_*` flag that improves a gap). Stage 3 is the first leg that **acts on
it by changing the repo** — but only in the safest possible form.

**Scope decisions (locked during brainstorming):**

| Axis | Decision |
|------|----------|
| Change type | **Deterministic flag-promotion.** The lever is an *existing* device-config flag; the change = set that flag in the parity config. **No codegen.** |
| Landing default | **Dry-run.** Default writes the edit, runs gates, prints the diff + verdicts, then reverts. `--apply` commits to the current branch only when all gates pass. |
| Change model | **Tier/config-gated** — flag default-off for legacy tier (the existing `mol.py` tier gate forces it off), set true on the parity config. So G0 bit-identical is provable. |
| Gates built | **G0 (legacy-bit-identical) + G4 (realized≈predicted) — new, deterministic.** Reuse G1–G3. **G5 (multi-skeptic) deferred → LLM.** |
| Autonomy | Auto-implement with guardrails; human reviews the branch. Never `main`, never push. |

**The key realization:** for flag-promotion the change is a **config edit, not
solver code.** Stage 2's levers already map to device-config flags the loader
parses (`interface_plane_projection`, `dos_band_potentials`, `flat_band_contacts`,
`interface_plane_closure`, `het_recomb_despike`). "Implement the fix" = set the
confirmed flag's key in `configs/scaps_mirror_v2.yaml`. Legacy tier forces the
flag off regardless → **G0 bit-identical holds by construction**, the suite just
verifies it.

**Explicitly deferred:** G5 multi-skeptic verify (LLM); LLM codegen for
non-promotable levers; auto-push; Stage 4 boulder chaining.

## 2. Flow

```
top open gap with a CONFIRMED Hypothesis (mechanism = "flag X term")
        │
        ▼
  PROPOSE  ── env-flag→config-key map + negatives-guard ──▶  ConfigEdit | None
        │                                                    (None = not promotable)
        ▼
  WRITE  → working-tree config edit (set device flag = true)
        │
        ▼
  GATE STACK   G1 numerics · G2 limiting · G3 scorecard-improved ·
               G4 realized≈predicted Δ · G0 legacy-bit-identical   (G5 deferred)
        │
        ├── any fail → revert edit + add_negative (refuted) → status gates_failed
        │
        ├── all green + dry-run → revert edit, report diff + verdicts → status dry_run
        │
        └── all green + --apply → commit to current branch (≠main, clean tree),
                                  gap.status=closed + mechanism + sha → status applied
```

## 3. Module layout (extends `perovskite_sim/autoloop/`)

```
promote.py       FLAG_TO_CONFIG_KEY map + parse_lever + propose_promotion(hyp, ledger) -> ConfigEdit|None
                 + set_device_flag(text, key, value) + apply_edit / revert_edit
gates_impl.py    gate_g0_bit_identical(...) + gate_g4_reconciles(...) (fill Stage-1 stubs)
orchestrator.py  + implement_top_confirmed(..., apply: bool)
types.py         + ConfigEdit, ImplementResult
scripts/autoloop_run.py  + --implement [--apply]
```

Read-only on solver code; edits only a config + (on `--apply`) commits to a branch.

## 4. Propose (`promote.py`)

`propose_promotion(hypothesis, ledger) -> ConfigEdit | None`:

1. **Parse the lever** from `hypothesis.mechanism` (`"flag SOLARLAB_IFACE_PROJ term"` → `SOLARLAB_IFACE_PROJ`).
2. **Map env-flag → device config key** (data):
   ```python
   FLAG_TO_CONFIG_KEY = {
       "SOLARLAB_IFACE_PROJ":  "interface_plane_projection",
       "SOLARLAB_IFACE_PLANE": "interface_plane_closure",
       "SOLARLAB_DOS_BAND":    "dos_band_potentials",
   }
   ```
   **Not every lever is promotable:** `SOLARLAB_INTERFACE_PLANE_STATE` is an
   SS-only runtime param with no device-config key → return `None`
   ("needs codegen, out of scope"), recorded. Honest.
3. **Negatives-guard (defense in depth):** `ledger.is_refuted(mechanism)` → return None.
4. Build `ConfigEdit{config_path, device_key, new_value=True, old_text}` (`old_text` =
   full prior file text, for exact revert).

**Apply / revert:**
- `apply_edit(edit)` — `set_device_flag(text, key, True)`: set-or-insert
  `  <key>: true` under the `device:` block (format-preserving line edit, no new
  dep). The dry-run diff is the human-visible safety.
- `revert_edit(edit)` — write `old_text` back verbatim.

**Types:**
```python
@dataclass(frozen=True)
class ConfigEdit:
    config_path: str
    device_key: str
    new_value: bool
    old_text: str

@dataclass(frozen=True)
class ImplementResult:
    status: str   # "applied"|"dry_run"|"gates_failed"|"no_confirmed"|"not_promotable"
    hypothesis_gap_id: Optional[str]
    device_key: Optional[str]
    gate_verdicts: tuple   # of GateVerdict
    committed_sha: Optional[str]
    note: str = ""
```

`set_device_flag` is the one piece of real logic → unit-tested hard (key absent →
insert; key present → overwrite; idempotent re-set; `device:` block present/missing).

## 5. Gate stack (`gates_impl.py` fills the Stage-1 stubs)

Ordered cheap→costly, fail-fast, all must pass:

| Gate | For a flag-promotion | Engine |
|------|----------------------|--------|
| G1 numerics | Stage-1 `run_l0` pytest subset green with the edit | reuse |
| G2 limiting | Stage-1 L1 limiting cases hold | reuse |
| G3 scorecard-improved | re-score parity with the flag on → `overall` up vs baseline, nothing regressed | reuse |
| **G4 realized≈predicted** | re-measure the gap's badness with the flag on; realized Δ must be a real improvement AND reconcile with Stage 2's predicted Δ (same sign, within `RECONCILE_TOL`). The lever must DO what attribution said. | **new, deterministic** |
| **G0 legacy-bit-identical** | run the legacy/golden regression suite with the edit → green, no NEW failures vs the two known pre-existing (`test_flat_band_contacts::...`, `test_interface_plane_closure::...`), no baseline shift on legacy-tier configs. Most expensive → last. | **new, deterministic** |
| G5 adversarial-verify | — | **deferred → LLM** (stub raises) |

- **G0** verifies the tier gate: `mol.py` forces the flag off on legacy, so legacy
  goldens are unaffected by construction; G0 runs the regression suite as the oracle.
- **G4** is the honest gate for flag-promotion. The master design's
  residual-reconciliation (137 mV = 84+53) is for codegen; for a flag it degrades
  to "the promoted flag's measured benefit actually materializes in-config." A flag
  whose ablation Δ doesn't show up → G4 rejects.

Gates are injected (gate-runner callables) so the orchestrator unit tests run
without the solver; the real runners are exercised by the slow smoke.

## 6. Orchestrator + landing (`orchestrator.implement_top_confirmed`)

1. Load ledger → top **open gap with a `confirmed` Hypothesis** (max `gap_mag`).
   None → `no_confirmed`.
2. `propose_promotion` → `ConfigEdit` | None. None → `not_promotable` (recorded).
3. `apply_edit`.
4. Run gate stack.
5. **Fail** → `revert_edit`; `add_negative` (flag-for-this-gap now refuted → anti-thrash);
   `gates_failed`.
6. **Green + apply** → commit to current branch; `gap.status="closed"` + mechanism +
   sha; `applied`.
7. **Green + dry-run** → `revert_edit` (clean tree); `dry_run` + diff + verdicts.

**Landing guards (safety):**
- Refuse `--apply` on `main`/`master` (human creates `autoloop/<date>` first).
- Refuse `--apply` if the working tree is dirty (else the commit captures unrelated edits).
- Commit message links traceability: `closes gap <id>`, mechanism, per-gate verdict
  summary, hypothesis cycle. **Never pushes.**
- All paths leave the working tree clean unless a commit succeeded.

**CLI:**
```bash
python scripts/autoloop_run.py --implement          # dry-run: diff + verdicts, no commit
python scripts/autoloop_run.py --implement --apply   # commit to current branch (refuses on main/dirty)
```
Exit 0 for dry_run/applied/no_confirmed/not_promotable; 1 only on a guard error
(apply-on-main, dirty tree). Gate outcome is in the JSON report.

## 7. Error handling

- No confirmed hypothesis / not promotable → clean status, exit 0.
- Gate failure → revert + `add_negative` + report (`gates_failed`).
- Malformed YAML edit → revert + error.
- `--apply` guard violation (main / dirty) → no commit, error, exit 1.
- Every non-committed path restores the working tree.

## 8. Testing

- **`promote`** — `set_device_flag` (absent→insert, present→overwrite, idempotent,
  no-device-block); `propose_promotion` (lever parse, map-hit, map-miss→not_promotable,
  negatives-guard refuse); `apply_edit`/`revert_edit` round-trip.
- **`gates_impl`** — G0 + G4 pass/fail with injected fakes (G4: same-sign+in-tol pass,
  wrong-sign/out-of-tol fail; G0: green→pass, new-failure→fail).
- **`orchestrator`** — `implement_top_confirmed` with fakes → each status; gates_failed
  reverts + adds negative; dry_run reverts + reports diff; **applied commits in a tmp
  `git init` repo** (real git, throwaway); apply-on-main refused; dirty-tree refused.
- **integration smoke (slow)** — real `--implement` **dry-run** on a confirmed gap with
  a **reduced gate set** (skip the expensive full-suite G0, validated separately) →
  asserts a diff + verdicts produced, working tree clean after. Honest about G0 cost.

## 9. Out of scope / deferred

- G5 multi-skeptic adversarial verify (LLM).
- LLM codegen for non-promotable levers (SS-only flags, novel terms).
- Auto-push; PR creation.
- Stage 4 boulder chaining (sense→attribute→implement loop + stop conditions + L3/L4).

## 10. Build order (staged tasks for writing-plans)

1. `types.py` — `ConfigEdit`, `ImplementResult`.
2. `promote.py` — `FLAG_TO_CONFIG_KEY`, `parse_lever`, `set_device_flag`, `apply_edit`/`revert_edit`, `propose_promotion`.
3. `gates_impl.py` — `gate_g4_reconciles` + `gate_g0_bit_identical` (injected runners).
4. `orchestrator.implement_top_confirmed` (+ git-commit helper, guards).
5. CLI `--implement [--apply]` + integration smoke + docs (README / CLAUDE.md in sync).
