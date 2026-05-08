# Frontend Wiring for Stage B(c.x) Physics — Implementation Plan

**Date**: 2026-05-04
**Companion spec**: `docs/superpowers/specs/2026-05-04-frontend-bcx-wiring-design.md`
**Status**: Plan only — implementation gated on user approval.

## Refinements approved 2026-05-04

The base spec is unchanged; these refinements (A–D) are folded into the relevant tasks below.

- **A. Beginner-safe UI**: Advanced 2D Physics section collapsed by default; short tooltips on every field; sentinel values (`null`/`0`) rendered as a placeholder hint ("(disabled)" / "(0.0 — disabled)"); the combined demo preset is labeled `# Demonstration preset — not a physical default` in its YAML header and the device-card display name says "(demo)".
- **B. Naming clarity**: UI labels use **Top contact (HTL side)** and **Bottom contact (ETL side)** instead of `left`/`right`. The mapping `device.S_*_left → MaterialArrays2D.S_*_top` is documented in helper text under the Outer-contacts subsection: "2D top maps to the YAML/1D `_left` fields; 2D bottom maps to `_right`. The 1D solver names them by transport-axis end; the 2D solver names them by physical orientation."
- **C. Active-physics summary**: `active-physics.ts` is the single frontend SoT; unit-tested with one case per tier × per relevant flag combination; `jv-2d-pane.ts` shows compact form `Active physics: Microstructure, Robin contacts, μ(E), Photon recycling` (comma-separated, alphabetical-by-fragment-name); when no upgrade is active, shows `Active physics: baseline 2D drift-diffusion`.
- **D. Validation gates**: per-task validation listed below; `pytest -m "slow or regression"` is conditionally required only if a task touches solver-adjacent code paths or modifies a config the regression suite loads.

## Task list

| #   | Task                                                              | Effort  | Rollback point  |
|-----|-------------------------------------------------------------------|---------|-----------------|
| T1  | Extend frontend types for the 10 new fields                       | XS      | After T1 commit |
| T2  | Add `PER_RHS_KEYS` to tier-gating                                 | XS      | After T2 commit |
| T3  | New `active-physics.ts` helper + unit tests                       | S       | After T3 commit |
| T4  | New "Advanced 2D Physics" group in config-editor.ts               | M       | After T4 commit |
| T5  | jv-2d-pane.ts active-physics summary line                         | XS      | After T5 commit |
| T6  | Device-card tier badge (compact, hover tooltip)                   | S       | After T6 commit |
| T7  | Two demo presets: `field_mobility_demo.yaml`, `bcx_combined_demo.yaml` | S       | After T7 commit |
| T8  | Backend dispatch test for advanced-physics fields                 | S       | After T8 commit |
| T9  | CLAUDE.md frontend-section update                                 | XS      | After T9 commit |

Each task is a **single, independently revertable commit**. T1 → T9 in order — later tasks depend on earlier ones.

## T1 — Extend frontend types

**Goal**: TypeScript types catch up with the YAML schema. No runtime behaviour change; this task should be invisible to the running app.

**Files touched**:
- `perovskite-sim/frontend/src/types.ts` — extend `LayerConfig` and the device-level part of `DeviceConfig` with optional fields.

**Specific changes**:
- `LayerConfig`: add `v_sat_n?: number`, `v_sat_p?: number`, `ct_beta_n?: number`, `ct_beta_p?: number`, `pf_gamma_n?: number`, `pf_gamma_p?: number`. All optional, all `number`.
- `DeviceConfig['device']`: add `S_n_left?: number | null`, `S_p_left?: number | null`, `S_n_right?: number | null`, `S_p_right?: number | null`. The `null` distinguishes "explicitly set to disabled" from "absent" (matches `_opt_S` in `models/config_loader.py:128–132`).

**Tests**:
- `npm run build` — `tsc` passes with the extended types.
- `npm run build` again with a known-good preset round-trip (e.g. `selective_contacts_demo.yaml` after a fetch+save) to confirm no schema regression.

**Rollback**: revert the single commit; no other module imports these new keys yet.

## T2 — Tier-gating for the new keys

**Goal**: hide the 10 new keys under FAST and LEGACY tiers (mirrors how TMM/dual-ion/trap-profile/temperature keys are already gated). No behaviour change yet because no field renders these keys.

**Files touched**:
- `perovskite-sim/frontend/src/workstation/tier-gating.ts`.

**Specific changes**:
- Add named constant `PER_RHS_KEYS` at module top:
```
const PER_RHS_KEYS = [
  // Robin (B(c.1))
  'S_n_left', 'S_p_left', 'S_n_right', 'S_p_right',
  // Caughey-Thomas + Poole-Frenkel (B(c.2))
  'v_sat_n', 'v_sat_p',
  'ct_beta_n', 'ct_beta_p',
  'pf_gamma_n', 'pf_gamma_p',
] as const
```
- Append `...PER_RHS_KEYS` to both `FAST_HIDDEN` and `LEGACY_HIDDEN`.
- Update the file-level docstring to say "any per-RHS hook flag (rr / μ(E) / Robin / selective contacts) is FULL-only — see `mode.py:71–86`".

**Tests**:
- `npm run build` — clean.
- New frontend unit `frontend/src/workstation/tier-gating.test.ts` (if a test framework is wired) or a console-driven check confirming `isFieldVisible('S_n_left', 'fast') === false` and `isFieldVisible('S_n_left', 'full') === true`. If no JS test framework exists, defer the unit test until T3 (where vitest gets bootstrapped) — note this dependency in the task ordering.

**Rollback**: revert the single commit; `PER_RHS_KEYS` are simply not in the hidden sets — they'd render under all tiers, which is harmless because no field-renderer references them yet.

## T3 — `active-physics.ts` helper + unit tests

**Goal**: a pure function `describeActivePhysics(device: DeviceConfig): string` that returns the same active-physics fragment ordering as backend `_describe_active_physics(stack)` (`backend/main.py:65–80`). Single frontend source of truth for what's running.

**Files touched (NEW)**:
- `perovskite-sim/frontend/src/active-physics.ts`
- `perovskite-sim/frontend/src/active-physics.test.ts`

**Specific changes**:
- Pure function, no I/O. Resolves tier (`legacy | fast | full`) → flag bundle (mirrors `mode.py` constants), then walks fragments alphabetically by display name and returns a comma-separated string.
- Display fragments (in alphabetical order by display name, matching backend convention):
  - `μ(E)` (only if FULL and any layer has `v_sat_n>0|v_sat_p>0|pf_gamma_n>0|pf_gamma_p>0`)
  - `Microstructure` (if `device.microstructure?.grain_boundaries?.length > 0`)
  - `Photon recycling` (if FAST or FULL)
  - `Reabsorption` (if FULL)
  - `Robin contacts` (only if FULL and any of the four `S_*` is non-null and non-zero)
  - `TMM` (if any layer has `optical_material`)
- When no fragment fires, returns the literal string `"baseline 2D drift-diffusion"`.
- Compact display form per refinement C: `"Active physics: <fragments comma-joined>"` or `"Active physics: baseline 2D drift-diffusion"`.

**Note on backend parity**: backend produces strings like `"FULL  band offsets · TE · TMM · dual ions · trap profile · T-scaling · photon recycling · PR reabsorption · μ(E) · Robin contacts"`. Frontend deliberately uses a SHORTER comma-joined form for UI compactness — this is OK because the full backend string is still surfaced in `Run.activePhysics` after a run completes (and shown in run details on hover). Frontend display = pre-run preview; backend display = post-run truth. Document this divergence in the file-level docstring.

**Unit tests** (`active-physics.test.ts`):
1. legacy tier → `"baseline 2D drift-diffusion"`
2. fast tier with no TMM → `"Active physics: Photon recycling"` (PR is on in FAST)
3. full tier with TMM but no optional flags → `"Active physics: Photon recycling, Reabsorption, TMM"`
4. full tier with TMM + microstructure → adds `Microstructure`
5. full tier + Robin only (S_n_left=1e3, others null) → adds `Robin contacts`
6. full tier + μ(E) only (one layer's v_sat_n=1e2) → adds `μ(E)`
7. full tier + all four hooks → all six fragments comma-joined
8. full tier + Robin where all four S are 0 → no `Robin contacts` fragment (zero S = inactive)

If a JS test framework isn't already set up, **add `vitest` as a dev dep** as part of T3 (lightweight, ~zero config). This unblocks T2's deferred test too.

**Tests for the build**:
- `npm run build` — clean.
- `npm test active-physics` — all 8 unit tests pass.

**Rollback**: revert the commit; nothing else imports `active-physics.ts` yet.

## T4 — "Advanced 2D Physics" field group in config-editor.ts

**Goal**: render the 10 new fields under a 5th group, collapsed by default, with the top/bottom UI naming and short tooltips.

**Files touched**:
- `perovskite-sim/frontend/src/config-editor.ts`.

**Specific changes**:

5th group:

```
Advanced 2D Physics  [collapsed by default]
├── Outer contacts (Robin)              [device-level]
│     Top contact / HTL side
│       S_n  [maps to YAML S_n_left]    placeholder "(disabled)"  tooltip: "Surface recombination velocity for electrons at the top contact (HTL side). Higher = more ohmic; very low ≈ blocking. Leave blank for the default Dirichlet ohmic contact."
│       S_p  [maps to YAML S_p_left]    placeholder "(disabled)"  tooltip: "…holes…"
│     Bottom contact / ETL side
│       S_n  [maps to YAML S_n_right]   placeholder "(disabled)"  tooltip: "…electrons at the bottom contact (ETL side)…"
│       S_p  [maps to YAML S_p_right]   placeholder "(disabled)"  tooltip: "…holes at the bottom contact…"
│     Helper text (small italic):
│       "2D top maps to YAML/1D `_left` fields; 2D bottom maps to `_right`. The 1D
│        solver names contacts by transport-axis end; the 2D solver names them by
│        physical orientation."
└── Field-dependent mobility μ(E)        [per-layer]
      For each electrical layer:
        v_sat_n   placeholder "(0.0 — disabled)"  tooltip: "Caughey-Thomas saturation velocity for electrons. Caps drift mobility under high field. Typical perovskite ~1e5 m/s. Set 0 to disable."
        v_sat_p   placeholder "(0.0 — disabled)"  tooltip: "…holes…"
        ct_beta_n placeholder "2.0"               tooltip: "Caughey-Thomas exponent. β=2 for silicon electrons (Canali); β=1 for silicon holes (Thornber). Default 2."
        ct_beta_p placeholder "2.0"               tooltip: "…holes…"
        pf_gamma_n placeholder "(0.0 — disabled)" tooltip: "Poole-Frenkel coefficient for electrons (m^0.5/V^0.5). Field-assisted hopping; typical disordered HTL ~3e-4. Set 0 to disable."
        pf_gamma_p placeholder "(0.0 — disabled)" tooltip: "…holes…"
```

Implementation notes:
- The new group respects the existing `tier` argument to `renderGroup()`; tier-gating from T2 hides the whole group under FAST/LEGACY.
- The group `<details>` element is `<details>` (not `<details open>`) — closed by default.
- Tooltip rendering reuses the existing config-editor pattern (likely `title=` on the input, or whatever the codebase already does — match local style).
- Single-layer-edit mode (FULL only, config-editor.ts:237) already supported — μ(E) fields appear in the focused-layer view as well as the all-layers expanded view.

**Tests**:
- `npm run build` — clean.
- Manual UI smoke (or `webapp-testing` skill): load `selective_contacts_demo.yaml`, see the four S fields populated under Top/Bottom; collapse and re-expand the group; switch tier full → fast → full and see the group disappear/reappear; the YAML values persist across the toggle.
- `pytest tests/integration/backend/` — no change expected, just confirms no regression.

**Rollback**: revert the commit; UI returns to 4 groups.

## T5 — jv-2d-pane.ts active-physics summary line

**Goal**: one read-only line above the existing pane-hint that previews what physics is active.

**Files touched**:
- `perovskite-sim/frontend/src/workstation/panes/jv-2d-pane.ts`.

**Specific changes**:
- Import `describeActivePhysics` from T3.
- Above the existing `<div class="pane-hint">`, add:
  ```
  <div class="active-physics-summary" id="jv2d-active-physics"></div>
  ```
- In `mountJV2DPane`, populate this div on mount and on every device change (the pane already takes a `getActiveDevice()` callback, so refresh on each "Run" click is a clean hook point — also refresh on `change` of the tier dropdown if that's reachable from this pane scope).
- Compact display per refinement C: `Active physics: <comma-joined fragments>` or `Active physics: baseline 2D drift-diffusion`.
- No new inputs, no new run params — strictly informational.

**Tests**:
- `npm run build` — clean.
- Manual smoke: load FULL-tier device with TMM + Robin S → see all listed; switch to LEGACY → see "baseline 2D drift-diffusion".

**Rollback**: revert the commit; pane returns to its current state.

## T6 — Device-card tier badge

**Goal**: small tier badge on the device card header (e.g. `[FULL]`, `[FAST]`, `[LEGACY]`) with the active-physics string as a hover tooltip. Helps users see the tier at a glance without scrolling to the editor's mode dropdown.

**Files touched**:
- `perovskite-sim/frontend/src/device-panel.ts` (most likely; confirm at implementation time).

**Specific changes**:
- One-element badge component rendered next to the device name.
- Class names follow existing card styling. Three badge colour-classes (`tier-full`, `tier-fast`, `tier-legacy`) — colours per the current design system, not new colours.
- Hover tooltip = the full `describeActivePhysics(device)` string.

**Tests**:
- `npm run build` — clean.
- Manual smoke: tier dropdown change → badge updates immediately; hover the badge → see active-physics string.

**Rollback**: revert the commit; device card returns to its current state.

## T7 — Demo presets

**Goal**: two new YAML presets — one for B(c.2), one for the combined B(c.x) demo. **Both label themselves as demos**, not physical defaults.

**Files touched (NEW)**:
- `perovskite-sim/configs/field_mobility_demo.yaml`
- `perovskite-sim/configs/twod/bcx_combined_demo.yaml`

**`field_mobility_demo.yaml`** (B(c.2)):
- Header comment: `# Demonstration preset — exercises Caughey-Thomas + Poole-Frenkel μ(E) on the spiro-OMeTAD HTL and the absorber. Not a physical default; values chosen to show a measurable but stable shift vs the constant-mobility baseline. See docs/.../2026-04-29-2d-stage-b-c2-field-mobility-design.md.`
- Based on `nip_MAPbI3_tmm.yaml` (so PR + TMM are already valid).
- HTL layer: `pf_gamma_p: 3.0e-4` (literature-typical for spiro), `pf_gamma_n: 0.0`, `v_sat_n: 0`, `v_sat_p: 0`.
- Absorber layer: `v_sat_n: 1.0e5`, `v_sat_p: 1.0e5`, `ct_beta_n: 2.0`, `ct_beta_p: 2.0`.
- ETL layer: defaults (all zero).
- `device.mode: full`.

**`twod/bcx_combined_demo.yaml`** (combined):
- Header comment: `# Demonstration preset — all four B(c.x) per-RHS hooks active at moderate values. NOT a physical default; chosen to stay out of bisection-budget regimes (no aggressive-blocking S, no extreme v_sat). For physics studies, derive from a published parameter set instead.`
- Based on `configs/twod/nip_MAPbI3_singleGB.yaml` (Stage B(a) microstructure).
- Add `device.contacts: { left: {S_n: 1.0e3, S_p: 1.0e3}, right: {S_n: 1.0e3, S_p: 1.0e3} }` (matched, moderate).
- Per absorber layer: `v_sat_n: 1.0e2`, `v_sat_p: 1.0e2`, `pf_gamma_*: 0.0`. (Modest CT cap — measurable shift but doesn't stress Newton.)
- `device.mode: full`.

**Tests**:
- `pytest tests/integration/backend/test_jv_2d_advanced_physics.py` (NEW, see T8) loads both presets and runs `kind=jv_2d` on each at coarse grid (Ny_per_layer=5, Nx=4, V_max=0.6, V_step=0.2) — asserts result is finite and active_physics string contains the expected fragments.
- Optional: hand-spot-check both load cleanly into the workstation UI.

**Rollback**: delete the two YAML files.

## T8 — Backend dispatch test

**Goal**: pin the contract that the backend accepts a device with all 10 new fields populated and surfaces them in `active_physics`.

**Files touched (NEW)**:
- `perovskite-sim/tests/integration/backend/test_jv_2d_advanced_physics.py`.

**Specific changes**:
- Two `@pytest.mark.integration` tests:
  1. `test_jv_2d_field_mobility_demo_runs` — load `field_mobility_demo.yaml`, POST to `/api/jobs` with `kind=jv_2d` at coarse grid, stream the result, assert SSE delivers a finite result and `active_physics` contains `"μ(E)"` and `"Photon recycling"`.
  2. `test_jv_2d_bcx_combined_demo_runs` — same shape, on `twod/bcx_combined_demo.yaml`. Assert active_physics contains `"Microstructure"`, `"Robin contacts"`, `"μ(E)"`, `"Photon recycling"`, `"Reabsorption"`.
- Both tests use the existing FastAPI test client pattern (look at `tests/integration/backend/test_voc_grain_sweep_api.py` as the reference).

**Tests**:
- `pytest tests/integration/backend/test_jv_2d_advanced_physics.py -v` — both pass.
- `pytest tests/integration/backend/` — full backend suite remains green.

**Rollback**: delete the test file.

## T9 — CLAUDE.md frontend section update

**Goal**: the frontend section of `perovskite-sim/CLAUDE.md` describes the new group, the gating, and the active-physics helper. This is documentation drift insurance.

**Files touched**:
- `perovskite-sim/CLAUDE.md` (frontend section).
- (No README change — README scope policy keeps unbuilt plans / advanced-feature minutiae out of the public README.)

**Specific changes**:
- One paragraph describing the "Advanced 2D Physics" group, the Top/Bottom mapping, and the FULL-only tier gate.
- One sentence linking `active-physics.ts` as the frontend mirror of `_describe_active_physics`, with the note that the frontend uses a compact comma-joined form vs the backend's bullet form.

**Tests**:
- `pytest -q` — green (no code change).
- Reading-level smoke: scan the CLAUDE.md section to confirm it stays under the existing voice/style.

**Rollback**: revert the docs commit.

## Validation gates (per refinement D)

| Gate                                                      | When run                       | Required for |
|-----------------------------------------------------------|--------------------------------|--------------|
| `npm run build` (tsc + vite, clean)                       | After every frontend task      | T1, T2, T3, T4, T5, T6 |
| `npm test` (vitest, active-physics + tier-gating)         | After T3 lands; reruns later   | T3, T4, T5 (regression check) |
| Manual UI smoke: tier toggle group visibility             | After T4 + T6                  | T4, T6 |
| `pytest tests/integration/backend/`                       | After T7, T8                   | T7, T8 |
| `pytest -q` (default; integration + unit, no slow)        | After T8 lands; final          | T8, T9 |
| `pytest -m "slow or regression"` (slow regression suite)  | **Conditional** — only if any task touches solver-adjacent code paths or ships a config the existing regression loads | (probably none) |

**Expectation**: zero numerical-regression changes. We're not touching solver code; we're surfacing existing fields. The slow regression suite is therefore NOT a required gate for this work, but I will run it once at the end as a paranoid final check before the merge.

## Expected UI layout (post-implementation)

```
┌─ Workstation ──────────────────────────────────────────────────────────┐
│  Device tree                  │  J–V Sweep (2D)  pane                  │
│                               │  ┌─────────────────────────────────┐   │
│  ▼ MAPbI3 [FULL]      <-T6    │  │ 2D J–V Sweep (Stage A — …)      │   │
│      "(hover)" Active physics:│  │                                 │   │
│       Microstructure, Robin   │  │  Active physics: Microstructure │   │
│       contacts, μ(E),         │  │   Robin contacts, μ(E),          │   │
│       Photon recycling        │  │   Photon recycling      <-T5     │   │
│                               │  │                                 │   │
│  ▶ Device editor              │  │  ┌──────────────────────────┐   │   │
│      Geometry                 │  │  │ Lateral length:    500   │   │   │
│      Transport                │  │  │ Nx:                10    │   │   │
│      Recombination            │  │  │ ...etc                   │   │   │
│      Ions & Optics            │  │  └──────────────────────────┘   │   │
│                               │  │                                 │   │
│      ▼ Advanced 2D Physics    │  │  ▶ Run 2D J–V                   │   │
│         <-T4 collapsed by dflt│  └─────────────────────────────────┘   │
│         (FULL tier only)      │                                        │
│         Outer contacts (Robin)│                                        │
│         Top contact / HTL:    │                                        │
│           S_n: ____ (disabled)│                                        │
│           S_p: ____ (disabled)│                                        │
│         Bottom contact / ETL: │                                        │
│           S_n: ____ (disabled)│                                        │
│           S_p: ____ (disabled)│                                        │
│         Helper: 2D top maps to│                                        │
│         YAML/1D _left fields. │                                        │
│                               │                                        │
│         Field-dep mobility μ(E)                                        │
│         For absorber:         │                                        │
│           v_sat_n: 0.0 (dsbl) │                                        │
│           v_sat_p: 0.0 (dsbl) │                                        │
│           ct_beta_n: 2.0      │                                        │
│           ct_beta_p: 2.0      │                                        │
│           pf_gamma_n: 0.0 (d) │                                        │
│           pf_gamma_p: 0.0 (d) │                                        │
└────────────────────────────────────────────────────────────────────────┘
```

## Preset definitions (final form for review)

### `configs/field_mobility_demo.yaml`

```yaml
# Demonstration preset — exercises Caughey-Thomas + Poole-Frenkel μ(E) on
# the spiro-OMeTAD HTL and the absorber. NOT a physical default; values
# chosen to show a measurable but stable shift vs the constant-mobility
# baseline. See docs/superpowers/specs/2026-04-29-2d-stage-b-c2-field-
# mobility-design.md for the physics derivation.
device:
  mode: full
  V_bi: 1.0
  Phi: 1.0e21
layers:
  - name: glass
    role: substrate
    # …same as nip_MAPbI3_tmm.yaml…
  - name: spiro_HTL
    role: HTL
    # …same as nip_MAPbI3_tmm.yaml plus:
    pf_gamma_p: 3.0e-4
  - name: MAPbI3
    role: absorber
    # …same as nip_MAPbI3_tmm.yaml plus:
    v_sat_n: 1.0e5
    v_sat_p: 1.0e5
    ct_beta_n: 2.0
    ct_beta_p: 2.0
  - name: PCBM
    role: ETL
    # …same as nip_MAPbI3_tmm.yaml…
```

### `configs/twod/bcx_combined_demo.yaml`

```yaml
# Demonstration preset — all four B(c.x) per-RHS hooks active at MODERATE
# values. NOT a physical default; chosen to stay out of bisection-budget
# regimes (no aggressive-blocking S, no extreme v_sat). For real physics
# studies, derive from a published parameter set instead. Builds on
# configs/twod/nip_MAPbI3_singleGB.yaml (Stage B(a) microstructure).
device:
  mode: full
  V_bi: 1.0
  Phi: 1.0e21
  contacts:
    left:
      S_n: 1.0e3
      S_p: 1.0e3
    right:
      S_n: 1.0e3
      S_p: 1.0e3
microstructure:
  grain_boundaries:
    - x_position: 250e-9
      width: 5e-9
      tau_n: 5.0e-8
      tau_p: 5.0e-8
      layer_role: absorber
layers:
  # …copied from configs/twod/nip_MAPbI3_singleGB.yaml; absorber adds:
  #   v_sat_n: 1.0e2
  #   v_sat_p: 1.0e2
```

(Final field values reviewed at implementation time; YAML coordinates are illustrative.)

## Rollback plan

Each task is a single commit. Rollback is `git revert <commit>` for the affected task. Tasks are ordered so partial rollback (revert the last N commits) leaves the codebase in a consistent state:

- Revert T9 → docs are stale but UI works.
- Revert T8 + T9 → no backend test, UI works, presets still load.
- Revert T7 + T8 + T9 → no demo presets, UI still works, tier-gated group still renders.
- Revert T6 + T7 + T8 + T9 → no badge, no presets, no tests, UI group still renders.
- Revert T5 + above → no pane summary; everything else works.
- Revert T4 + above → no UI for advanced fields; the types and helper are dead weight, harmless.
- Revert T3 + above → no active-physics helper; T2 tier-gating still applies but no field-renderer references it.
- Revert T2 + above → no tier gating for new keys; harmless because no fields render.
- Revert T1 + above → back to pre-spec state.

Single commit per task means clean atomic rollback at each level.

## Effort estimate

| Task | Effort estimate (focused work) |
|------|-------------------------------|
| T1   | 10 min                        |
| T2   | 10 min                        |
| T3   | 45 min (helper + 8 unit tests, vitest setup if needed) |
| T4   | 90 min (UI group, tooltips, top/bottom mapping, manual smoke) |
| T5   | 15 min                        |
| T6   | 30 min                        |
| T7   | 30 min                        |
| T8   | 30 min                        |
| T9   | 15 min                        |
| **Total** | **~5 hrs focused work**   |

## Approval gate

Plan describes 9 atomic tasks, expected UI layout, preset definitions, and rollback at each level. **No code changes proposed yet.** Awaiting explicit approval before starting T1.
