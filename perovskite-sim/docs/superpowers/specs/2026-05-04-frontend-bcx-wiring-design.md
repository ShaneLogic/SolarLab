# Frontend Wiring for Stage B(c.x) Physics — Design Spec

**Date**: 2026-05-04
**Status**: Spec only — implementation gated on user review.
**Scope**: Surface the Stage B(c.1) Robin contacts, B(c.2) field-dependent mobility μ(E), and B(c.3) radiative reabsorption / photon recycling controls in the workstation UI. Add demo presets. No numerical-solver or backend-dispatch changes.

## 1. Background — what already exists

**Backend / config-loader (no changes required):**
- `device:` block accepts `S_n_left`, `S_p_left`, `S_n_right`, `S_p_right` (or nested `device.contacts: { left: {S_n, S_p}, right: {S_n, S_p} }`). Source: `models/config_loader.py:118–149`.
- `layers[]` accept `v_sat_n`, `v_sat_p`, `ct_beta_n`, `ct_beta_p`, `pf_gamma_n`, `pf_gamma_p`. Source: `models/config_loader.py:97–102`.
- `device.mode: legacy | fast | full` controls the SimulationMode tier — and `use_photon_recycling`, `use_radiative_reabsorption`, `use_field_dependent_mobility`, `use_selective_contacts` are bundled into the tier (only FULL turns the four per-RHS flags on). Source: `models/mode.py:71–86`.
- `kind=jv_2d` already reads `device` (full DeviceConfig with all the above) and exposes `microstructure` as an inline override. Source: `backend/main.py:860–920`.
- `_describe_active_physics(stack)` is already plumbed into the `jv_2d` SSE result payload, surfacing the active flags as a human string. Source: `backend/main.py:65–80`.

**Frontend (current gaps):**
- `config-editor.ts` has 4 field groups (Geometry / Transport / Recombination / Ions & Optics). No per-layer μ(E) fields. No device-level Robin S fields. The `device.mode` tier dropdown is already wired (config-editor.ts:255).
- `tier-gating.ts` hides FAST/LEGACY-incompatible keys; the four B(c.x) per-RHS hooks are not currently in this map.
- `jv-2d-pane.ts` exposes a single GB inline as a per-run override but otherwise reads everything from the active DeviceConfig. There's no advanced-physics surface.
- Existing presets: `selective_contacts_demo.yaml` (B(c.1)), `radiative_limit.yaml` (B(c.3) physics-limit reference). No B(c.2) demo. No combined demo.

**Implication**: All four B(c.x) flags are configurable through YAML today, but invisible in the UI. The wiring task is **frontend-only** — surface the existing configuration knobs.

## 2. UX design (proposed)

### 2.1 Where the controls live

**Decision**: Put new fields in the **device editor** (`config-editor.ts`), not in `jv-2d-pane.ts`. Rationale:
- These are device properties, not experiment knobs — once a user sets `S_n_left=1e-4`, every experiment on that device should see it.
- Save-As preset already serialises `DeviceConfig`; new fields automatically persist.
- Avoids cluttering the 2D pane (per user's UX guidance).
- Matches how the existing `device.mode` tier dropdown works.

The 2D pane gets **one read-only summary line**: "Active physics: TMM · TE · PR · μ(E) · Robin contacts" sourced from the device's resolved tier — clarifies for the user what's about to run without exposing parameters they shouldn't change here.

### 2.2 Tier gating

Add to `tier-gating.ts:HIDDEN_BY_TIER`:

```
PER_RHS_KEYS = [
  'S_n_left', 'S_p_left', 'S_n_right', 'S_p_right',          // Robin (B(c.1))
  'v_sat_n', 'v_sat_p',                                       // CT (B(c.2))
  'ct_beta_n', 'ct_beta_p',                                   //   "
  'pf_gamma_n', 'pf_gamma_p',                                 // PF (B(c.2))
]
```

These are hidden under LEGACY and FAST (both disable the flags), visible under FULL. This mirrors the existing TMM / dual-ion / trap-profile / temperature gating exactly. **Default tier remains `full`**, so the average user sees the controls.

When a user lowers tier from FULL → FAST, the fields disappear from the UI but the YAML values stay on disk (the loader preserves them silently — they're "enable-if-configured" no-ops in lower tiers, by design).

### 2.3 Field placement

```
Device card (existing 4 groups + 1 new):
├── Geometry              (existing)
├── Transport             (existing — μ_n, μ_p go here today)
├── Recombination         (existing)
├── Ions & Optics         (existing)
└── Advanced 2D physics   (NEW — collapsed by default, FULL-tier-only)
    ├── Outer contacts (Robin)            ← device-level, 4 fields
    │     S_n_left, S_p_left, S_n_right, S_p_right
    └── Field-dependent mobility μ(E)     ← per-layer, 6 fields × layer
          v_sat_n, v_sat_p              (Caughey-Thomas saturation velocities)
          ct_beta_n, ct_beta_p          (CT exponent — β=2 default)
          pf_gamma_n, pf_gamma_p        (Poole-Frenkel coefficient — m⁰·⁵·V⁻⁰·⁵)
```

Photon recycling and radiative reabsorption have no parameters — they're tier-flags only. **No new parameter fields for B(c.3).** Instead, add:
- A small "tier" badge on the device card header — e.g. `[FULL · TMM · PR · μ(E) · Robin]` — read-only, sourced from `_describe_active_physics`-style logic on the frontend (mirrors backend).

### 2.4 Beginner safety

- All new fields default to inactive sentinels (`null` / `0` / `0.0`):
  - Robin: 4 fields default to `null` (not present in YAML) → outer-contact code path stays Dirichlet, identical to today.
  - μ(E): `v_sat_*=0` and `pf_gamma_*=0` defaults disable both CT and PF on that layer (`apply_field_mobility` is a no-op when both are 0).
- Tooltips on each field, ~1 sentence each, plain-English physics:
  - `S_n_left`: "Surface recombination velocity for electrons at the left contact (m/s). Higher = more ohmic; very low = blocking. Leave blank for the default (Dirichlet) ohmic contact."
  - `v_sat_n`: "Caughey-Thomas saturation velocity for electrons (m/s). Caps electron drift mobility under high field. Typical perovskite ~1e5 m/s. Leave 0 to disable."
  - …etc.
- Group is collapsed by default. A small "ⓘ Why FULL only?" hint inside the legend points to mode.py docs.

### 2.5 Active-physics indicator on jv-2d-pane

Add one line above the existing "Stage A" pane-hint:

> **Active physics**: TMM · TE · PR · μ(E) · Robin contacts (FULL tier)

Rendered from the device's tier + flags. If lowered to LEGACY: `Active physics: (LEGACY tier — all upgrades off)`. This is the same string already returned in `Run.activePhysics` after a run, just shown *before* the run too.

## 3. Preset strategy

| Preset                                            | Status   | What it demonstrates                                |
|---------------------------------------------------|----------|-----------------------------------------------------|
| `configs/selective_contacts_demo.yaml`            | exists   | B(c.1) Robin (matched + wrong-sign S)               |
| `configs/radiative_limit.yaml`                    | exists   | B(c.3) PR/rr V_oc-boost reference                   |
| `configs/twod/nip_MAPbI3_uniform.yaml`            | exists   | Stage A 2D parity baseline (BL)                     |
| `configs/twod/nip_MAPbI3_singleGB.yaml`           | exists   | B(a) microstructure                                 |
| **NEW** `configs/field_mobility_demo.yaml`        | propose  | B(c.2) μ(E) — non-zero v_sat / pf_gamma in HTL+absorber |
| **NEW** `configs/twod/bcx_combined_demo.yaml`     | propose  | All four hooks on at once for the workstation demo  |

Both new presets prefix with `nip_MAPbI3_tmm.yaml` so PR is meaningfully testable and inherit the FULL tier. The combined demo uses moderate (non-extreme) S values to avoid Newton-stall regimes — explicitly NOT the aggressive-blocking S values used in the regression test.

## 4. Backend / API compatibility

**No backend changes needed** — the schema already accepts every field above. Specifically:

- `POST /api/jobs` body `device` is round-tripped through `_coerce_numbers` (handles YAML 1.1 `1e-3`-as-string quirk) → built into `DeviceStack` via `DeviceConfig.from_dict` → all S/v_sat/etc. fields populate `MaterialParams` and `DeviceStack` correctly.
- `POST /api/configs/user` (Phase 2b user-preset save-as) already round-trips arbitrary device fields, so save-as preserves new fields automatically.
- `GET /api/configs/{name}` returns the raw YAML; `_coerce_numbers` runs there too.

**Schema verification step** in implementation: after the UI changes land, run `pytest tests/integration/backend/` to confirm no regression in JSON-round-trip; add one new test that POSTs a device with all 10 new fields populated and asserts the run succeeds (or surface in the active-physics string).

## 5. Files likely to be touched

```
frontend/src/config-editor.ts                  — add Advanced-2D-physics group, 10 new field defs
frontend/src/workstation/tier-gating.ts        — add PER_RHS_KEYS to FAST_HIDDEN/LEGACY_HIDDEN
frontend/src/workstation/panes/jv-2d-pane.ts   — add active-physics summary line above the pane-hint (NO new inputs here)
frontend/src/types.ts                          — extend DeviceConfig type with the 10 optional fields
frontend/src/active-physics.ts                 — NEW small helper that mirrors _describe_active_physics on the frontend; called from device card header + jv-2d-pane summary
configs/field_mobility_demo.yaml               — NEW B(c.2) demo preset
configs/twod/bcx_combined_demo.yaml            — NEW combined demo preset
docs/superpowers/specs/2026-05-04-frontend-bcx-wiring-design.md   — this file
docs/superpowers/plans/2026-05-04-frontend-bcx-wiring-plan.md     — implementation plan, separate doc
perovskite-sim/CLAUDE.md                       — frontend section: list new fields and their gating
```

**Out of scope** — explicitly:
- `frontend/src/workstation/panes/voc-grain-sweep-pane.ts` — already runs through the device config; nothing to change.
- `backend/main.py` — no new dispatch parameters.
- `perovskite_sim/twod/*` — no numerical-solver changes.
- 1D experiment panes — same flags work there too, but the user's request is 2D-focused. If/when we revisit, the same Advanced-2D-physics group will pick them up automatically (the flags are device-level, not experiment-level).

## 6. Risks

| # | Risk                                                                | Mitigation                                                                                                  |
|---|---------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| 1 | Tier=LEGACY/FAST + non-null S or v_sat → flag silently ignored      | Tier-gating hides the fields under those tiers. If a YAML preset sets them, value persists but is inert (matches existing TMM / trap-profile behaviour).        |
| 2 | YAML 1.1 `1e-3` → string contamination on round-trip                | `_coerce_numbers` already handles this for arbitrary nested keys. No new code.                              |
| 3 | UI clutter: 6 new fields × N layers in the editor                   | New group is collapsed by default. Single-layer-edit mode already exists in FULL tier (config-editor.ts:237) so users can drill into one layer at a time.       |
| 4 | Beginners enable Robin S=1e-4 (extreme) → 2D solver hangs           | Default sentinels (null) preserve current behaviour. Tooltip suggests "matched-carrier S ~ 1e3 m/s" range. The 200k `max_nfev` cap on `run_transient_2d` converts a hang into a fast `RuntimeError` even for accidental extremes.    |
| 5 | active-physics string drifts between backend and frontend           | Single source of truth on backend (`_describe_active_physics`); frontend helper is a thin tier→flags map driven by the same `SimulationMode` constants exported via `frontend/src/types.ts`. Add one frontend test pinning the mapping.    |
| 6 | New `bcx_combined_demo.yaml` enables four hooks at once → flaky on TMM | Use moderate S (~1e3 m/s), small v_sat (~1e2 m/s), moderate γ_PF (~1e-4 (V/m)⁻⁰·⁵) — far from the aggressive-blocking regime that exercised the 200k cap. Verify the combined preset runs to V_max=1.2 V on `kind=jv_2d` without hitting bisection beyond depth 2. |
| 7 | `S` fields default to `null` in TS but YAML serialiser may write `null` literally | `config-editor.ts` already drops null/empty fields when serialising back to YAML for save-as (Phase 2b convention). Audit during implementation.   |

## 7. Validation plan

Before declaring complete:

1. **Frontend build**: `npm run build` clean (tsc + vite). Bundle size delta < 5 KB for new code.
2. **Backend dispatch tests**: `pytest tests/integration/backend/` green. Add `test_jv_2d_accepts_advanced_physics_params` that POSTs a device with all 10 new fields populated and asserts the SSE result includes `active_physics` containing `Robin contacts` and `μ(E)`.
3. **2D regression suite (slow)**: `pytest -m "slow or regression"` green — confirms no numerical drift from the spec-described changes. Expected: zero changes to numerical regression values (we're not touching solver code).
4. **UI smoke**: manual or `webapp-testing` skill: load `configs/twod/bcx_combined_demo.yaml`, see all four hooks listed in the active-physics summary, click "Run 2D J–V" with default grid, watch it complete in <2 minutes wall, verify the resulting `Run.activePhysics` string.
5. **Tier-gate UI test**: switch tier `full → fast → legacy`, confirm the Advanced 2D physics group disappears and reappears, and the underlying YAML values persist across the toggle.

## 8. Implementation task list (in order)

1. **types.ts**: extend `DeviceConfig.device` with optional `S_n_left`, `S_p_left`, `S_n_right`, `S_p_right`. Extend `LayerConfig` with optional `v_sat_n/p`, `ct_beta_n/p`, `pf_gamma_n/p`.
2. **tier-gating.ts**: add `PER_RHS_KEYS` const, include in `FAST_HIDDEN` and `LEGACY_HIDDEN`. Update the file's docstring to call out which mode flag gates which set.
3. **active-physics.ts** (NEW): pure function `describeActivePhysics(device: DeviceConfig): string` that mirrors backend `_describe_active_physics` (same tier resolution, same fragment ordering). Unit-tested with a small fixture matching the backend test cases.
4. **config-editor.ts**: add the 5th field group "Advanced 2D physics" with the 10 fields; respect the new tier-gating; collapsed by default; tooltips on each field. Single-layer-edit mode (FULL only) already supported.
5. **jv-2d-pane.ts**: add the one-line active-physics summary above the existing `pane-hint` div.
6. **Device card header**: add the FULL/FAST/LEGACY badge + active-physics tooltip on hover.
7. **field_mobility_demo.yaml** + **bcx_combined_demo.yaml**: add the two demo presets under `configs/`.
8. **Tests**:
   - Backend: `tests/integration/backend/test_jv_2d_advanced_physics.py` (1 test).
   - Frontend: a small unit test for `active-physics.ts` (covers all three tiers + a couple flag combinations).
9. **Docs**: update `perovskite-sim/CLAUDE.md` Phase 2b / frontend section to describe the new group, the gating, and the active-physics helper. Add a one-line entry to the README's frontend feature list.
10. **Smoke run**: load `bcx_combined_demo.yaml`, click Run, verify result + active-physics string. Capture screenshot for the spec doc as a follow-up.

## 9. Out-of-scope (deferred)

- Trap-profile `N_t`-shape fields (B(a) physics, not B(c.x)) — separate ticket.
- Per-layer photon-recycling overrides — current backend uses a global flag; per-layer would need solver work.
- 1D experiment panes — same fields work there but the user has explicitly scoped this work as 2D.
- A "physics presets quick-pick" dropdown inside the device card — nice-to-have, propose for a future iteration once we have user feedback on the current group.
- Animation / visualisation of the per-RHS hook source terms (G_rad, μ(E) corrections) inside the workstation — separate diagnostic work.

## 10. Approval gate

Spec describes UI surface, presets, and tasks. **No code changes proposed yet.** Awaiting explicit approval before producing the implementation plan (`docs/superpowers/plans/2026-05-04-frontend-bcx-wiring-plan.md`) and starting work.
