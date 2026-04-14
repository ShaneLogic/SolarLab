# Layer Builder UI — Design Spec

**Date:** 2026-04-14
**Status:** Draft → awaiting user review
**Scope:** Phase 2b of the workstation physics upgrade — give full-tier users a
visual stack builder (add / remove / reorder / role-edit layers) backed by a
template library and user-preset save path. Builds on Phase 2a (TMM optics
activation).

> **Out of scope (deferred to later specs):**
> - **Phase 3 — Tandem cell support** (dual-absorber stacks with tunnel
>   junctions, sub-cell current matching). Requires solver-level changes.
> - User-preset rename and delete (filesystem CRUD beyond create/read).
> - In-UI editing of n,k optical data (CSVs remain reference data).
> - Physical-sanity validation that requires real semiconductor knowledge
>   (bandgap vs chi consistency, etc.) — those live in `config_loader.py`
>   and run server-side at dispatch.
> - Plot-side TMM/Beer-Lambert toggle (deferred from Phase 2a, still deferred).

---

## 1. Goal

Make custom device stacks a first-class workflow in the workstation. Today,
the Device pane renders a fixed accordion of whatever layers a preset YAML
ships with; users cannot add, remove, or reorder layers from the UI, and the
role dropdown only knows about `HTL`/`absorber`/`ETL` (Phase 2a's
`substrate`/`front_contact`/`back_contact` roles are unreachable from the UI).

Phase 2b ships:
1. A vertical, SCAPS-inspired **stack visualizer** that renders the layer
   stack as a sun-side-up cross-section with role-colored blocks, log-scale
   heights, drag handles, hover-revealed insert/delete affordances, and inline
   interface-recombination strips.
2. A split-view **detail editor** that shows only the currently selected
   layer's parameter groups (replacing the all-layers accordion in full tier).
3. A **template library** of starter layers (TiO₂ ETL, spiro HTL, Ag back
   contact, etc.) so users can build a working device from scratch in a few
   clicks.
4. A **Save-As / user-preset namespace** so custom stacks survive page
   reloads and appear as named presets in the dropdown.
5. **Structural validation** (substrate-first, exactly-one-absorber, unique
   names, positive thicknesses) hard-blocked at the UI layer with inline
   error feedback.
6. A **dirty-state pill** (`● modified`) and a **Reset** button that reverts
   to the loaded preset baseline.
7. A **YAML download** escape hatch for sharing configs outside the app.

The Python solver does not change. The backend gains two new endpoints
(`POST /api/configs/user`, `GET /api/layer-templates`) plus one auto-scan
recursion update; everything else is a frontend reorganization.

## 2. Background — what already exists

```
perovskite-sim/
├── frontend/src/
│   ├── config-editor.ts             (renders all layers as an accordion)
│   ├── device-panel.ts              (preset dropdown + Reset)
│   ├── workstation/
│   │   ├── panes/device-pane.ts     (host container; mode dropdown lives here)
│   │   └── tier-gating.ts           (per-field tier visibility — Phase 2a)
│   ├── api.ts                        (typed fetch wrappers)
│   └── types.ts                      (DeviceConfig, LayerConfig)
│
├── backend/main.py                  (FastAPI; /api/configs auto-scans
│                                     configs/*.yaml; /api/optical-materials
│                                     auto-scans data/nk/*.csv — Phase 2a)
│
├── perovskite_sim/data/
│   └── nk/                           (Phase 2a — n,k CSVs + manifest.yaml)
│
└── configs/
    ├── nip_MAPbI3.yaml              (shipped, Beer-Lambert)
    ├── nip_MAPbI3_tmm.yaml          (shipped, Phase 2a TMM-active)
    ├── pin_MAPbI3.yaml
    ├── pin_MAPbI3_tmm.yaml
    ├── ionmonger_benchmark.yaml
    ├── driftfusion_benchmark.yaml
    ├── cigs_baseline.yaml
    └── cSi_homojunction.yaml
```

**Tier gates (Phase 2a):** `optical_material` and `incoherent` are visible
only when `tier === 'full'`. Phase 2b extends this with one new helper
`isLayerBuilderEnabled(tier)` that returns `tier === 'full'`. Fast and legacy
tiers continue to render the existing accordion editor unchanged.

**Solver assumptions that this spec relies on:**
- The solver is orientation-agnostic. n-i-p (`ETL / absorber / HTL`) and p-i-n
  (`HTL / absorber / ETL`) both work without code changes. The built-in
  voltage `V_bi_eff` is derived from the Fermi-level difference across the
  full stack and flips sign automatically.
- Layer count is unconstrained.
- `role: substrate` layers are filtered out of the electrical grid by
  `electrical_layers()` in `models/device.py` (Phase 2a).

## 3. Architecture

```
┌─ Frontend (full tier only) ─────────────────────────────────────────┐
│ Device pane                                                         │
│  ┌──────────────────────────────────┐  ┌─────────────────────────┐  │
│  │ Stack Visualizer (~280px col)    │  │ Layer Detail Editor     │  │
│  │  ☀ AM1.5G + ↓ rays               │  │  Name + Role dropdown   │  │
│  │  Vertical layer cards            │  │  Geometry & Optics      │  │
│  │   - log-scale heights            │◄─┤  Transport              │  │
│  │   - role color palette           │  │  Recombination          │  │
│  │   - drag handle, ✕ delete        │  │  Ions & Optics          │  │
│  │   - inter-layer "+" inserts      │  │  (only for selected     │  │
│  │   - interface strips inline      │  │   layer — not all)      │  │
│  │  Legend / Add… / Save As / ↓YAML │  │                         │  │
│  └──────────────────────────────────┘  └─────────────────────────┘  │
│  Header: device name · "● modified" · "TMM active · N layers"       │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
       ┌──────────────┼──────────────┬───────────────┐
       ▼              ▼              ▼               ▼
  GET /configs   GET /optical-  GET /layer-      POST /configs/user
  (recurses     materials      templates        (Save As, atomic
   user/)       (Phase 2a)     (NEW)             O_EXCL|O_CREAT)

┌─ Backend ───────────────────────────────────────────────────────────┐
│ /api/configs           — recurses configs/ + configs/user/          │
│ /api/layer-templates   — auto-scan data/layer_templates.yaml        │
│ /api/configs/user      — POST a validated YAML to configs/user/     │
│   • filename regex ^[a-zA-Z0-9_-]{1,64}$                            │
│   • 409 if name collides with a shipped preset                      │
│   • 400 if body fails Pydantic structural validation                │
└─────────────────────────────────────────────────────────────────────┘

┌─ perovskite_sim ────────────────────────────────────────────────────┐
│ data/                                                               │
│  ├─ nk/                  (Phase 2a)                                 │
│  └─ layer_templates.yaml (NEW) — per-template defaults + citations  │
│ configs/                                                            │
│  ├─ *.yaml               (shipped, read-only by convention)         │
│  └─ user/                (NEW) — user-saved presets land here       │
└─────────────────────────────────────────────────────────────────────┘
```

**Tier-mode safety net.** The visualizer + detail-editor split is rendered
only when `mode === 'full'`. Fast and legacy tiers render the existing
accordion editor unchanged (no regression risk for benchmark workflows).
Even in full tier, loading a Phase 2a TMM preset (`nip_MAPbI3_tmm.yaml`)
just renders the same layers — there is no migration, no schema change.
Phase 2b is a strictly additive overlay over the same `DeviceConfig` shape.

## 4. Stack visualizer

### 4a. Visual scheme

Inspired by SCAPS-1D's left-panel layer view and Sentaurus Device Editor's
cross-section preview, adapted to the workstation's horizontal pane budget:

- **Sun side up.** A `☀ AM1.5G` label and five ↓ arrows above the stack
  show illumination direction.
- **Vertical layer cards.** Each layer renders as one rounded rectangle.
  Cards stack top-to-bottom in YAML order, which is also optical order.
- **Log-scale heights.** A 1 mm glass substrate next to a 50 nm TiO₂ layer
  would be unreadable at linear scale. The visualizer maps `log10(thickness)`
  from `[1 nm, 1 mm]` to a fixed pixel range so every layer is visible while
  the absorber still appears as the fattest block.
- **Role colors** (CSS variables in `style.css`):

  | Role | Background | Foreground |
  |---|---|---|
  | `substrate` | grey gradient | dark grey |
  | `front_contact` | warm gold gradient | brown |
  | `ETL` | light blue gradient | navy |
  | `absorber` | deep purple gradient | white |
  | `HTL` | mauve gradient | maroon |
  | `back_contact` | metallic gold gradient | white |

  A WCAG-AA contrast pass is part of the implementation checklist; sub- and
  superscripts (MAPbI₃, TiO₂, χ, E_g, μ_n) use a dedicated `.sym` class with
  `font-feature-settings: "sups", "subs"` and a slightly larger size to stay
  readable at the small card scale.
- **Selected-layer affordance.** The selected card gets a 2 px orange
  outline plus a soft `box-shadow` glow. The detail editor on the right
  side reflects the selected layer.
- **Drag handle (`⋮⋮`).** Visible on the left of every card. Native HTML5
  `dragstart`/`dragover`/`drop` events; on drop, the card moves and the
  interfaces array is reconciled.
- **↑↓ reorder buttons.** Hover-revealed on the right of each card next to
  the delete button. Keyboard-focusable; serve as the accessible reorder
  path for users who cannot drag (and as a fallback if browser DnD
  misbehaves). They drive the same `onAction({type:'reorder', from, to})`
  path as drag-and-drop, so reconciliation logic is shared.
- **Delete (`✕`).** Hover-revealed on the right of each card. Confirms only
  if the card is the only absorber.
- **Inter-layer `+` buttons.** A 2 px-tall gap sits between every pair of
  adjacent cards; on hover it expands and a circular `+` button appears
  centered in the gap. Clicking opens the Add Layer dialog, with the
  `insertAtIdx` already set.
- **Interface strips.** Between every pair of adjacent layers, a thin
  blue-tinted strip shows `◆ v_n=… v_p=… m/s` (or `◆ uses default 0 m/s`
  in yellow if both are zero). Click to expand into an inline two-input
  editor.
- **Optical material pill.** Each card displays a small pill on the right
  with the layer's `optical_material` key (e.g. `MAPbI3`, `TiO2`). Cards
  with no `optical_material` set show no pill.
- **Legend.** A single row of color chips below the stack mapping color to
  role.
- **Stack-level actions.** A row of three buttons below the legend:
  `＋ Add layer…`, `Save as…`, `↓ YAML`.

### 4b. Two-column layout

In full tier, the Device pane uses a CSS grid with two columns:
`280px 1fr`. The visualizer is the navigator; the detail editor on the
right shows only the currently selected layer's parameter groups (Geometry
& Optics, Transport, Recombination, Ions & Optics — same group structure
as today, but for one layer instead of all layers).

Below ~1100 px viewport width (laptop split-screen scenario), the layout
collapses to a single column: visualizer on top, detail editor below.

### 4c. Header pill states

The Device pane header shows the device name, then up to two pills:
- `● modified` — appears when `state.current` differs from `state.loaded`.
  Cleared by Reset, by reloading the preset, or by a successful Save-As.
- `TMM active · N layers` — Phase 2a, unchanged.

## 5. Add Layer dialog

A modal triggered by the `＋ Add layer…` button or any inter-layer `+`.
Two tabs:

### 5a. Template tab (default)

Lists all entries from `/api/layer-templates`. Each entry shows:
- Name (e.g. `TiO2 ETL`)
- Role badge
- One-line description
- Source citation (small grey footnote)

Clicking a template inserts a new `LayerConfig` populated from the template
at `insertAtIdx`. The new layer becomes the selected layer immediately.

### 5b. Blank tab

Two inputs: `name` and `role` (dropdown including all six role values).
Numeric fields default to zero. The user fills them in via the detail
editor after insertion.

## 6. Save As dialog

A modal triggered by the `Save as…` button.

- **Filename input.** Live-validated against `^[a-zA-Z0-9_-]{1,64}$`.
  Invalid characters render a red border and a tooltip explaining the rule.
- **Collision check.** As the user types (debounced 250 ms), a probe
  `HEAD /api/configs/user/{name}` runs. If the name collides with a
  **shipped** preset, the dialog renders an inline 409-style error and
  disables Submit. If it collides with another **user** preset, the dialog
  shows a yellow warning and the Submit button label changes from `Save`
  to `Overwrite`.
- **Submit.** POSTs `state.current` to `/api/configs/user` with the chosen
  filename. On success, the preset dropdown re-fetches `/api/configs`,
  switches to the new entry under `User presets`, and clears the dirty
  pill (`state.loaded` is updated to match).

The backend is the source of truth for collision and atomicity (uses
`os.O_EXCL | os.O_CREAT` on first save, plain `O_TRUNC` on overwrite).
The probe is a UX nicety, not a security boundary.

## 7. YAML download

The `↓ YAML` button serializes `state.current` to YAML in the browser
(via `js-yaml`, already a transitive dependency of the existing config
parser) and triggers a download with filename `<device-name>.yaml`.
This is purely client-side; no backend round-trip.

## 8. Detail editor changes

`config-editor.ts` shrinks from rendering the full layer accordion to
rendering one layer's parameter groups. Concretely:

- The `renderDeviceEditor` function gains a `selectedLayerIdx: number`
  parameter and only renders that one layer's groups.
- The role dropdown gains the Phase 2a values:
  `substrate`, `front_contact`, `ETL`, `absorber`, `HTL`, `back_contact`.
- The interface-recombination section is removed (now lives inline in the
  visualizer).
- The Device-level fields (`Mode`, `T`, `V_bi`, `Φ`) stay above the
  visualizer in their own group.
- All existing field types (numeric inputs, optical-material select,
  incoherent checkbox) are unchanged.
- `readDeviceEditor` is updated to read the single visible layer and merge
  it back into `state.current.layers[selectedLayerIdx]` immutably.

In fast / legacy tiers, `renderDeviceEditor` renders the same all-layers
accordion as today (no behavior change).

## 9. Layer template library

### 9a. New file: `perovskite_sim/data/layer_templates.yaml`

Hand-authored library of starter layers. Each entry has a `role`,
`optical_material`, ballpark electrical fields, and a one-line
`description`.

```yaml
TiO2_ETL:
  role: ETL
  optical_material: TiO2
  description: "Compact TiO₂ ETL for n-i-p stacks (planar)"
  source: "Liu et al. 2014, Nature 501, 395"
  defaults:
    thickness: 5.0e-8           # 50 nm
    eps_r: 30
    chi: 4.0
    Eg: 3.2
    mu_n: 1.0e-7
    mu_p: 1.0e-7
    N_D: 1.0e22
    N_A: 0
    ni: 1.0e10
    tau_n: 1.0e-7
    tau_p: 1.0e-7
    D_ion: 0
    P_lim: 0
    P0: 0
    alpha: 0
    incoherent: false

SnO2_ETL:
  # ...

spiro_HTL:
  role: HTL
  optical_material: spiro_OMeTAD
  description: "Spiro-OMeTAD HTL (n-i-p)"
  source: "Saliba et al. 2016, Energy Environ. Sci. 9, 1989"
  defaults:
    thickness: 2.0e-7
    # ... etc

PEDOT_PSS_HTL:
  # p-i-n HTL

C60_ETL:
  # p-i-n ETL

PCBM_ETL:
  # alternative p-i-n ETL

FTO_front_contact:
  # ...
ITO_front_contact:
  # ...
Ag_back_contact:
  # ...
Au_back_contact:
  # ...

glass_substrate:
  role: substrate
  optical_material: glass
  description: "BK7-equivalent glass substrate (1 mm)"
  source: "Schott BK7 datasheet"
  defaults:
    thickness: 1.0e-3
    incoherent: true
    # electrical defaults are zero; substrate is filtered out of the
    # electrical grid by electrical_layers() in models/device.py

MAPbI3_absorber:
  # absorber starter — same params as nip_MAPbI3.yaml
```

The exact electrical defaults are written during implementation against
published cell measurements. Each template's `source:` field is mandatory
and is surfaced as the dialog footnote.

### 9b. New endpoint: `GET /api/layer-templates`

```python
@app.get("/api/layer-templates")
def list_layer_templates() -> dict:
    """Return the parsed layer templates library."""
    path = Path(__file__).parent.parent / "perovskite_sim" / "data" / "layer_templates.yaml"
    with path.open() as f:
        templates = yaml.safe_load(f) or {}
    return {"templates": templates}
```

The frontend caches the response for the lifetime of the page.

## 10. User-preset save endpoint

### 10a. New module: `backend/user_configs.py`

```python
"""User-preset filesystem operations.

Kept in its own module so filename validation and shipped-name
reservation can be unit-tested without spinning up FastAPI.
"""
import os
import re
from pathlib import Path
import yaml

USER_FILENAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
CONFIGS_ROOT = Path(__file__).parent.parent / "configs"
USER_CONFIGS_ROOT = CONFIGS_ROOT / "user"


def validate_user_filename(name: str) -> None:
    """Raise ValueError if name is not a safe user-preset filename."""
    if not USER_FILENAME_RE.match(name):
        raise ValueError(
            f"Invalid filename {name!r}: must match {USER_FILENAME_RE.pattern}"
        )


def is_shipped_name(name: str) -> bool:
    """Return True if name collides with a shipped preset (top-level configs/)."""
    return (CONFIGS_ROOT / f"{name}.yaml").exists()


def write_user_config(name: str, body: dict, *, overwrite: bool = False) -> Path:
    """Write a validated user config atomically.

    Uses os.O_EXCL on first save to prevent TOCTOU races; honors overwrite=True
    for explicit user-confirmed overwrites.
    """
    validate_user_filename(name)
    if is_shipped_name(name):
        raise FileExistsError(f"{name!r} collides with a shipped preset")
    USER_CONFIGS_ROOT.mkdir(parents=True, exist_ok=True)
    target = USER_CONFIGS_ROOT / f"{name}.yaml"
    flags = os.O_WRONLY | os.O_CREAT | (os.O_TRUNC if overwrite else os.O_EXCL)
    fd = os.open(target, flags, 0o644)
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(body, f, default_flow_style=False, sort_keys=False)
    return target
```

### 10b. New endpoint: `POST /api/configs/user`

```python
@app.post("/api/configs/user")
def save_user_config(payload: dict) -> dict:
    """Write a user-edited config to configs/user/."""
    name = payload.get("name")
    body = payload.get("config")
    overwrite = bool(payload.get("overwrite", False))
    if not name or not body:
        raise HTTPException(400, "name and config required")
    try:
        validate_user_filename(name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if is_shipped_name(name):
        raise HTTPException(409, f"{name!r} is reserved (shipped preset)")
    try:
        write_user_config(name, body, overwrite=overwrite)
    except FileExistsError:
        raise HTTPException(409, f"{name!r} already exists; pass overwrite=true")
    return {"saved": name}
```

### 10c. Modified endpoint: `GET /api/configs`

The existing endpoint scans `configs/*.yaml`. It is updated to recurse
exactly one level so `configs/user/*.yaml` is also returned, with each
entry tagged `{"name": "...", "namespace": "shipped" | "user"}`. The
frontend uses the namespace tag to render the dropdown in two groups.

## 11. State management and data flow

The Device pane owns one piece of state:

```typescript
interface DevicePaneState {
  loaded: Readonly<DeviceConfig>      // immutable baseline from the last fetch / save
  current: Readonly<DeviceConfig>     // the working copy
  selectedLayerIdx: number
}
```

All edits replace `current` immutably (spread + `Object.freeze` in dev mode
to catch accidental mutation). `loaded` is touched only by:
- The initial fetch when a preset is selected.
- A successful Save-As (then `loaded := current`).
- A Reset action (then `current := loaded`).

`reconcileInterfaces(oldLayers, newLayers, oldInterfaces)` is the single
helper for keeping `interfaces.length === layers.length - 1` invariant
across insert/delete/reorder. It:
- For an insert at index `i`, inserts `[0, 0]` at `i - 1` if `i > 0`,
  appends `[0, 0]` if inserting at the end.
- For a delete at index `i`, drops the interface at `i - 1` (or `i` if
  deleting the first layer, which removes the interface to the right).
- For a reorder, walks the new layer order and looks up each adjacent
  pair in the old interface map by `(left.name, right.name)`; surviving
  pairs keep their values, new pairs get `[0, 0]`.

## 12. Validation

`stack-validator.ts` exposes one pure function:

```typescript
export function validate(config: DeviceConfig): ValidationReport
```

`ValidationReport` is `{ errors: LayerError[], warnings: LayerWarning[] }`
where each entry has a `layerIdx`, a `field` (or null for stack-level), and
a human-readable `message`.

### 12a. Hard-blocking errors

| Rule | Message |
|---|---|
| Stack must contain exactly one layer with `role === 'absorber'` | "Stack needs exactly one absorber layer (found {n})" |
| Layer names must be unique within the stack | "Duplicate layer name {name}" |
| Every layer's `thickness > 0` | "Thickness must be positive" |
| At most one substrate layer | "At most one substrate layer is allowed" |
| If any layer has `role === 'substrate'`: it must be at index 0 | "Substrate must be the first layer" |
| If any layer has `role === 'substrate'`: it must have `incoherent: true` | "Substrate must be marked incoherent" |
| If any layer has `role === 'substrate'`: it must have a non-empty `optical_material` | "Substrate must have an optical material" |

Errors disable the Run button and the Save-As button. They render as a red
border on the offending layer card plus a tooltip with the message; a
toolbar banner above the visualizer summarizes the first error.

### 12b. Soft warnings

| Rule | Message |
|---|---|
| A new interface row has `(v_n, v_p) === (0, 0)` | "Interface uses default 0 m/s" |
| Some but not all layers have `optical_material` set | "Mixed TMM / Beer-Lambert layers — TMM-less layers fall back per Phase 2a" |
| Stack has zero layers with `optical_material` set in full tier | "TMM is dormant — set optical_material to enable" |

Warnings render as yellow pills next to the affected layer / interface; they
do not disable any button.

### 12c. Out of scope (backend's responsibility)

- Bandgap vs electron affinity sanity (`chi + Eg < 0` etc.)
- Doping vs intrinsic concentration consistency
- Mobility positivity beyond zero (the existing `config_loader.py` enforces this)
- Anything that requires real semiconductor physics knowledge

These continue to be raised by `models/config_loader.py` at job-dispatch
time as `ValueError`, surfaced through the existing SSE `error` channel.

## 13. Frontend file inventory

**New files:**

| File | Purpose |
|---|---|
| `frontend/src/stack/stack-visualizer.ts` | Renders the visualizer column. Pure render function; emits actions through `onAction(action)` so the parent owns state. |
| `frontend/src/stack/stack-layer-card.ts` | Renders one layer card (role colors, log-scale height, drag handle, delete, optical pill). |
| `frontend/src/stack/stack-interface-strip.ts` | Renders the inline interface row between two layers; click to expand into a two-input editor. |
| `frontend/src/stack/add-layer-dialog.ts` | Modal with Template / Blank tabs. Returns a new `LayerConfig` to the caller. |
| `frontend/src/stack/save-as-dialog.ts` | Filename input, live regex check, collision probe, submit. |
| `frontend/src/stack/stack-validator.ts` | Pure `validate(config)` function; returns `ValidationReport`. |
| `frontend/src/stack/reconcile-interfaces.ts` | Pure helper for keeping `interfaces` length in sync after insert/delete/reorder. |
| `frontend/src/stack/dirty-state.ts` | Pure `isDirty(loaded, current)` helper (deep equal, key-order independent). |
| `frontend/src/stack/__tests__/stack-validator.test.ts` | Vitest, ~12 cases. |
| `frontend/src/stack/__tests__/reconcile-interfaces.test.ts` | Vitest, ~6 cases. |
| `frontend/src/stack/__tests__/dirty-state.test.ts` | Vitest, ~4 cases. |
| `frontend/src/stack/__tests__/log-scale-height.test.ts` | Vitest, ~5 cases. |

**Modified files:**

| File | Change |
|---|---|
| `frontend/src/workstation/panes/device-pane.ts` | Two-column layout in full tier (CSS grid). Owns `DevicePaneState`. Routes visualizer actions to immutable state updates. Renders the header dirty pill. |
| `frontend/src/config-editor.ts` | Shrunk to render only the selected layer in full tier; unchanged in fast/legacy. Role dropdown gains the Phase 2a values. |
| `frontend/src/api.ts` | `fetchLayerTemplates`, `saveUserConfig`, `checkUserConfigExists`. |
| `frontend/src/types.ts` | `LayerTemplate`, `ValidationReport`, `LayerError`, `LayerWarning`. Role literal-union extended. `DeviceConfig` and `LayerConfig` marked `Readonly` at the type level. |
| `frontend/src/workstation/tier-gating.ts` | New `isLayerBuilderEnabled(tier)` helper. |
| `frontend/src/style.css` | Visualizer column styles, role color CSS variables, drag/hover affordances, narrow-viewport breakpoint at 1100 px. |
| `frontend/src/device-panel.ts` | Render preset dropdown as two groups (`Shipped presets`, `User presets`) using the `namespace` tag from `/api/configs`. |
| `frontend/src/panels/tutorial.ts` | New "Custom Stacks" section explaining the builder. |
| `frontend/src/panels/parameters.ts` | Note that role now includes `front_contact`/`back_contact`/`substrate`. |

## 14. Backend file inventory

**New files:**
- `backend/user_configs.py` (~80 lines)
- `perovskite_sim/data/layer_templates.yaml` (~12 entries)
- `tests/unit/backend/test_user_configs.py`
- `tests/integration/backend/test_user_configs_api.py`
- `tests/e2e/layer-builder.spec.ts`

**Modified files:**
- `backend/main.py` — recurse `configs/user/` in `/api/configs` (one new
  glob); add `/api/layer-templates` (~10 lines); add `/api/configs/user`
  (~20 lines); import `user_configs`.
- `CLAUDE.md` — one-line note on Phase 2b activation.

## 15. Tests

### 15a. Unit (frontend, Vitest)

- `stack-validator.test.ts` — one happy-path case + one violation per rule
  in §12a (12 cases total). Includes both n-i-p and p-i-n stacks to
  confirm the validator is orientation-agnostic.
- `reconcile-interfaces.test.ts` — insert at start / middle / end, delete
  at start / middle / end, reorder forward and backward; assert that
  `interfaces.length === layers.length - 1` and that surviving adjacent
  pairs keep their values.
- `dirty-state.test.ts` — `isDirty(c, c)` is false; `isDirty(c, edited)`
  is true; key-order independence on JSON-serialized representations.
- `log-scale-height.test.ts` — height function is finite for `[1 nm, 1 mm]`,
  preserves ordering, and never returns 0 / NaN / negative values.

Field-set superset test:
- `config-editor-superset.test.ts` — render the same `LayerConfig` through
  both the legacy/fast accordion path and the full-tier single-layer path;
  collect the rendered field `id` set from each (e.g. `layer-0-thickness`,
  `layer-0-mu_n`, …) and assert that the full-tier set is a strict
  superset of the legacy set. This catches accidental field drops in the
  shrunk editor without depending on DOM structure.

### 15b. Unit (backend, pytest)

- `tests/unit/backend/test_user_configs.py` —
  - `validate_user_filename` accepts only `^[a-zA-Z0-9_-]{1,64}$`.
  - Rejects `..`, `/`, `.`, empty, 65-char names, unicode letters.
  - `is_shipped_name` returns `True` for every file in `configs/*.yaml`.
  - `write_user_config` raises `FileExistsError` on first-save collision
    and succeeds with `overwrite=True`.
  - `write_user_config` refuses path traversal even if filename validation
    is bypassed (defense in depth).

### 15c. Integration (backend, pytest + httpx TestClient)

- `tests/integration/backend/test_user_configs_api.py` —
  - `GET /api/layer-templates` returns the parsed library.
  - `GET /api/configs` recurses into `configs/user/` and tags each entry
    with `namespace`.
  - `POST /api/configs/user` writes a file, returns 200, and the saved
    file round-trips through `GET /api/configs/{name}`.
  - Collision with a shipped name returns 409.
  - Invalid filename returns 400.
  - Overwrite path: second POST with the same name returns 409, then
    `overwrite=true` succeeds.

### 15d. E2E (Playwright)

- `tests/e2e/layer-builder.spec.ts` —
  - Switch to full tier; verify the visualizer renders for `nip_MAPbI3_tmm`.
  - Click each layer card; verify the detail editor switches.
  - Add a `TiO2 ETL` template via the dialog; verify both visualizer and
    `/api/jobs` request body contain the new layer.
  - Drag a layer to a new position; verify the request body reflects the
    new order.
  - Delete the absorber; verify Run is disabled and the banner appears.
  - Restore the absorber; run a J–V sweep; verify the result returns.
  - Save-As with a unique name; reload the page; verify the saved preset
    appears under `User presets`.
  - Save-As with `nip_MAPbI3_tmm`; verify the dialog shows a 409 inline
    error.
  - Edit a field; verify the `● modified` pill appears; click Reset;
    verify the pill clears and the field reverts.
  - YAML download produces a file the backend can re-load via the
    existing `/api/configs` flow.

### 15e. Manual verification checklist

- [ ] In **full tier**, the visualizer + detail editor renders.
- [ ] In **fast** and **legacy** tiers, the existing accordion editor
      renders unchanged (regression check against Phase 2a screenshots).
- [ ] All 6 layers of `nip_MAPbI3_tmm` show in the visualizer with correct
      role colors and (log) heights.
- [ ] All 6 layers of `pin_MAPbI3_tmm` show correctly with the inverse
      role order — confirms p-i-n / n-i-p symmetry.
- [ ] Dragging a layer reorders it and the J–V result reflects the change.
- [ ] Adding a "TiO₂ ETL" template inserts a layer with sensible defaults;
      running the sweep produces a finite J_sc.
- [ ] Deleting the absorber disables Run with the expected error banner;
      restoring it re-enables Run.
- [ ] Save-As under a fresh name persists across page reload.
- [ ] Save-As under "nip_MAPbI3_tmm" returns a friendly 409 error in the
      dialog.
- [ ] Reset reverts edits to the loaded baseline; the dirty pill disappears.
- [ ] YAML download produces a file the backend can re-load via the
      existing `/api/configs` flow.
- [ ] Below ~1100 px viewport width, layout collapses to single-column
      without overlapping content.
- [ ] WCAG-AA contrast pass on all role color / text combinations,
      especially the small subscript / superscript characters in layer
      labels (MAPbI₃, TiO₂, χ, E_g, μ_n).

## 16. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Native HTML5 drag-and-drop is unreliable across browsers | Medium | Native DnD is good enough for a dev tool (Chrome / Firefox / Safari). ↑↓ buttons render on hover as a redundant fallback. No external library. |
| Two-column layout breaks on narrow viewports (laptop split-screen) | Medium | One CSS media query at 1100 px collapses to single-column. |
| Users save dozens of presets and the dropdown becomes unmanageable | Low | Out of scope — no rename/delete in Phase 2b. The `Shipped` / `User` grouping makes manual filesystem cleanup viable. |
| Template defaults produce broken simulations because example values are too generic | Medium | Each template's electrical params come from a cited published cell. The implementation PR runs every template through a `nip_MAPbI3_tmm`-like stack and confirms the simulation converges with a qualitatively reasonable J–V. |
| The shrunk `config-editor.ts` accidentally drops a field that the existing accordion exposed | Low | Snapshot test (`config-editor-superset.test.ts`) compares old vs new field sets and fails on regressions. |
| Filename collision check has a TOCTOU race | Low | Backend write uses `os.O_EXCL | os.O_CREAT` on first save; the probe is a UX nicety, the server is the authority. |
| Custom stacks with mixed TMM and Beer-Lambert layers behave unexpectedly | Medium | Already handled in Phase 2a (`_compute_tmm_generation` falls back per-layer). The §12b soft warning surfaces the inconsistency to users. |
| `DeviceConfig` immutability is broken by a careless in-place mutation | Medium | `Readonly<DeviceConfig>` at the type level; `Object.freeze` on `state.current` in dev mode; ESLint `prefer-readonly` on the new `stack/` modules. |
| Drag-and-drop breaks accessibility (keyboard-only users cannot reorder) | Medium | The redundant ↑↓ hover buttons are keyboard-focusable and serve as the accessible reorder path. |

## 17. Documentation updates (mandatory deliverable)

Per the standing requirement from the Phase 2a spec, every spec touches
`panels/tutorial.ts`, `panels/parameters.ts`, and (when relevant)
`panels/algorithm.ts` whenever new physics or parameters are introduced.

### 17a. `panels/tutorial.ts`

Add a new section **"Custom Stacks"** between "Optical Generation" and
"Running Experiments":

> **Custom Stacks**
>
> In **Full** tier, the Device pane shows your stack as a vertical
> cross-section. You can:
>
> - **Add a layer** by clicking `＋ Add layer…` or any `+` between layers.
>   Pick from the template library (TiO₂ ETL, spiro HTL, Ag back contact,
>   …) or start blank.
> - **Reorder** by dragging a layer's `⋮⋮` handle, or by clicking the
>   ↑↓ buttons on hover.
> - **Delete** with the `✕` button on hover.
> - **Edit** any field by clicking a layer to select it and using the
>   detail editor on the right.
> - **Save** your custom stack as a named user preset via `Save as…`.
>   It will appear under `User presets` in the dropdown alongside the
>   shipped presets.
> - **Export** the current device as a YAML file via `↓ YAML` for sharing
>   outside the app.
>
> Both n-i-p (`ETL / absorber / HTL`) and p-i-n (`HTL / absorber / ETL`)
> orientations are supported. The simulator is orientation-agnostic — it
> derives the built-in voltage from the stack itself.

### 17b. `panels/parameters.ts`

Update the `role` field row to list all six values and explain when each
is required:

| Field | Type | Tier | Description |
|---|---|---|---|
| `role` | string | full | One of `substrate`, `front_contact`, `ETL`, `absorber`, `HTL`, `back_contact`. Substrate layers are filtered out of the electrical drift-diffusion grid (they participate only in the TMM optical stack). Every stack must contain exactly one absorber. |

### 17c. `CLAUDE.md`

Add a one-line note under the existing "Frontend" section:

> **Custom stacks (Phase 2b — Apr 2026):** In full tier, the Device pane
> renders a vertical layer visualizer with add/remove/reorder support and
> a template library. User-saved presets land in `configs/user/` and
> appear under `User presets` in the dropdown. The accordion editor is
> preserved for fast/legacy tiers.

## 18. Verification checklist

After all implementation tasks complete:

- [ ] All new and existing frontend Vitest tests pass: `npm run test`
- [ ] All new backend pytest tests pass:
      `pytest tests/unit/backend/test_user_configs.py`
      `pytest tests/integration/backend/test_user_configs_api.py`
- [ ] `pytest` full suite green (no regressions in existing tests)
- [ ] `npm run build` succeeds (TypeScript clean + production bundle)
- [ ] Playwright E2E suite green: `npx playwright test layer-builder.spec.ts`
- [ ] Manual verification checklist (§15e) all checked
- [ ] CLAUDE.md updated
- [ ] Documentation panels reviewed in browser
- [ ] Phase 2a regression baselines (`tests/regression/test_tmm_baseline.py`)
      still pass — confirms no electrical-grid or solver regression
