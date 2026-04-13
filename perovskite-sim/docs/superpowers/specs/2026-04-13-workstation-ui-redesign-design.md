# Workstation UI Redesign Design

**Date:** 2026-04-13
**Status:** Design — awaiting user review

## Problem

The current frontend is a 6-tab single-page app (J–V / Impedance / Degradation / Tutorial / Parameters / Algorithm) where each experiment tab embeds its own copy of the `Device Configuration` card. Consequences:

- **Device state is not shared across experiments.** Editing the absorber in J–V does not propagate to Impedance; the user re-loads the preset per tab.
- **The Mode dropdown (Legacy / Fast / Full) is buried.** It sits inside the Device form next to `V_bi` and `T`, so users do not realise it flips entire branches of physics on and off. The original motivation for the Phase 1–4 physics upgrades is invisible in the UI.
- **No run history.** Every `Run` click overwrites the previous result. You cannot compare two J–V sweeps on the same device without manually saving screenshots.
- **No cross-device comparison.** You cannot open MAPbI₃ and CIGS side-by-side, let alone run the same experiment on both and overlay the curves.
- **Does not look or feel like a device simulator.** Silvaco, Sentaurus, SCAPS-1D, and COMSOL users immediately recognise the current app as "a web form" rather than "a simulation tool". Serious researchers trust software that *looks* serious.
- **Solver feedback is hidden.** Convergence information, active-physics summary, and timing are visible only in the backend terminal.

## Goal

Replace the tabbed shell with a dockable workstation-style UI (project tree + multi-pane workspace + persistent solver console) in which the device is a first-class tree object with runs stored under its experiments and the physics tier (Legacy / Fast / Full) is a first-class per-device choice made at creation time. Ship the redesign in four reviewable phases that each leave a working application.

## Non-goals

- **Not** adding parameter sweeps as a first-class feature. Dropped from MVP as scope creep; can be added later in its own spec.
- **Not** changing the backend solver. The Phase 1–4 physics upgrades are already in place. The only backend work is whatever small additions the new UI requires (run persistence API).
- **Not** releasing three separate executables. "Full / Fast / Legacy separately" is solved by a new-device wizard and per-tier UI gating plus a git-tagged Legacy release for academic reproducibility.
- **Not** supporting multi-user / cloud sync / collaborative workspaces. Workspace is a local file.
- **Not** adding a scripting REPL. The solver console is read-only log.
- **Not** building floating (undocked) windows. All panes dock inside the main area.
- **Not** implementing undo history beyond the existing per-device Reset.

## Approach: Workstation shell with Golden Layout

The new frontend follows the **Workstation** archetype (Silvaco DeckBuild / Sentaurus Workbench / Lumerical CHARGE). Three top-level regions:

1. **Top toolbar** (fixed) — New Device wizard launcher, Run / Run All, Compare Runs, Help
2. **Project tree** (left sidebar, fixed width) — workspace → devices → experiments → runs, with a cross-device Results folder
3. **Main dock area** (everything else) — Golden Layout v2 instance holding all content panes, with a persistent bottom solver console

### Why Golden Layout v2

Considered alternatives:

- **Dockview** — more modern API, but thin vanilla-JS docs and higher risk of hitting a documentation gap mid-implementation.
- **Hand-rolled splitters + tab strips** — zero deps but reimplements 80% of a docking library badly.
- **React-only docking libs (flexlayout-react, rc-dock)** — require adopting React across the whole frontend, which is a separate, much larger migration.

**Golden Layout v2** is chosen because it is mature, battle-tested in vanilla TS, supports tabs / splits / drag-to-dock / serialisable layout state, has ~50 kB bundle weight, and integrates with plain-TS Vite projects without any framework migration.

### Tree shape

Multi-device workspace with experiments owned by their device and runs owned by their experiment. Cross-device comparison is handled by a separate **Results / Compare** folder that references runs rather than owning them:

```
📦 Workspace (one per browser session, saved to localStorage)
├─ 📁 Devices
│  ├─ 🔬 MAPbI₃ n-i-p [FULL]
│  │  ├─ 🧪 J–V Sweep
│  │  │  ├─ ▶ Run 11:02 (V_oc=1.08)
│  │  │  └─ ▶ Run 11:15 (V_oc=1.09)
│  │  ├─ 🧪 Impedance
│  │  └─ 🧪 Degradation
│  ├─ 🔬 CIGS baseline [LEGACY]
│  └─ 🔬 c-Si wafer [FAST]
└─ 📁 Results / Compare     ← placeholder folder in Phase 1, wired up in Phase 4
   └─ 📊 J–V cross-device (references runs from any device)
```

Each device carries a **tier** (`legacy` / `fast` / `full`) that controls which UI controls are visible in the Parameter Inspector. The tier is chosen at device creation time via the New Device wizard and is immutable except through a `Change tier` right-click action (which revalidates all per-tier fields).

### Per-tier UI gating — why this replaces "separate releases"

The Mode dropdown already exists in the backend (Phase 5 `SimulationMode`). The redesign upgrades it from a buried form field to a first-class device property:

- **At device creation**, the wizard's first step is three large cards (Legacy / Fast / Full) with the existing Algorithm-tab comparison table embedded, forcing the user to make an intentional choice rather than accepting a default.
- **At device inspection**, the Parameter Inspector hides fields that have no effect in the chosen tier. Legacy devices do not show `D_ion,-`, `optical_material`, trap-profile controls, or the `T` input. Fast devices hide TMM fields and trap profile. Full devices show everything. The hidden fields are still stored internally (for upgrade-tier compatibility) but not rendered.
- **At run time**, the solver console's first token states the active physics: `● FULL  band offsets · TE · TMM · dual ions · T-scaling` vs `● LEGACY  Beer-Lambert · single ion · uniform τ · T=300K`. Zero ambiguity about what the run did.
- **For academic reproducibility**, a git tag `v1.0-legacy-ionmonger` pinned to the Legacy preset + benchmark notebook serves as a citable artefact. No separate repo.

This achieves every outcome "three separate releases" would have achieved, at the cost of a UI-gating layer rather than three maintenance surfaces.

## Default pane set

On first launch (or on a fresh workspace), the dock is populated with seven panes:

| Pane | Purpose | Default dock position |
|---|---|---|
| **Device Schematic** | SCAPS-1D-style drawn stack of the active device. Clickable layers feed the Inspector. Interfaces are clickable for SRV. Layer colours mirror `role` (HTL / absorber / ETL). | Top-centre |
| **Main Plot** | J–V / Nyquist / degradation depending on the active experiment. Active run is highlighted; older runs on the same experiment are overlaid translucent. | Top-right |
| **Parameter Inspector** | Context-sensitive form — shows layer params when a layer is selected, experiment params when an experiment node is selected, run metadata (read-only) when a run is selected. Tier-gated. | Bottom-centre |
| **Secondary View** | Tabbed container for band diagram, density profile, metric cards. User can flip tabs. | Bottom-right |
| **Solver Console** | Always-visible bottom strip. Read-only. First line is the active physics summary; subsequent lines are progress messages streamed over SSE. | Bottom strip, not dockable above other panes |
| **Help** | Tutorial, Parameters, Algorithm — each as a separate tab in the Help container. Hidden by default, opened via toolbar Help button. | Centre dock, appears when opened |
| **Project Tree** | Workspace navigation. Not actually a Golden Layout pane — lives in a fixed left sidebar for stability. | Left sidebar, fixed 200px |

All panes except Project Tree and Solver Console are Golden Layout stackable items. Users can drag any pane to split horizontally / vertically / stack as tabs. Layout state is serialised to localStorage per workspace.

## Data model

```typescript
interface Workspace {
  id: string
  name: string
  devices: Device[]
  compareViews: CompareView[]
  layout: GoldenLayoutConfig  // serialised dock state
}

interface Device {
  id: string
  name: string
  tier: 'legacy' | 'fast' | 'full'
  config: DeviceConfig   // existing type from types.ts
  experiments: Experiment[]
}

interface Experiment {
  id: string
  kind: 'jv' | 'impedance' | 'degradation'
  params: JVParams | ImpedanceParams | DegradationParams
  runs: Run[]
}

interface Run {
  id: string
  timestamp: number
  result: JVResult | ImpedanceResult | DegradationResult
  activePhysics: string  // e.g. 'band offsets · TE · TMM · dual ions · T-scaling'
  durationMs: number
  deviceSnapshot: DeviceConfig  // frozen device state at run time
}

interface CompareView {
  id: string
  name: string
  kind: 'jv' | 'impedance' | 'degradation'
  runRefs: Array<{ deviceId: string; experimentId: string; runId: string }>
}
```

### Persistence

- Workspace lives in `localStorage` under key `solarsim:workspace:<id>`, autosaved on every mutation.
- **Runs contain a frozen `DeviceConfig` snapshot** so that editing a device later does not retroactively change what its historical runs look like. This is non-negotiable for scientific integrity.
- Import / Export buttons in the toolbar write/read the workspace as a single JSON file for sharing with collaborators or attaching to a paper.
- Run result blobs can be large (full J–V arrays); a localStorage quota warning is shown at 8 MB used, and an Export Runs action offers to offload old runs to a file and clear them from storage.

## Backend impact

The backend is almost unchanged. The existing `POST /api/jobs` already accepts a `device` object with a `mode` field (wired in Phase 5) and the SSE event stream already carries progress + result frames. The only additions are cosmetic:

1. The `result` event payload gains an `active_physics` string computed from `resolve_mode(stack.mode)` so the frontend does not need to re-derive it. This is 3 lines in `backend/main.py`.
2. The existing `/api/configs` endpoint is unchanged — YAML presets are still loaded by filename and fed into the New Device wizard's preset picker.

No database, no authentication, no new endpoints. Run persistence is purely client-side.

## Phased rollout

Each phase ships a **working app** and gets its own implementation plan + PR. The spec covers all four phases at the design level; only Phase 1 is detailed deeply enough for immediate implementation. Later phases are refined in their own specs as we approach them.

### Phase 1 — Foundation

Ship the shell. At the end of Phase 1:
- Golden Layout v2 is integrated; `main.ts` no longer has the six-tab strip.
- Project tree exists in a left sidebar with one hard-coded default device (the current `ionmonger_benchmark.yaml`).
- The central dock area contains a Device pane (holding the existing config editor verbatim) + a Help pane with Tutorial / Parameters / Algorithm as tabs.
- Bottom solver console strip is present but only shows the last backend event.
- `localStorage` serialises the layout.
- The existing J–V / Impedance / Degradation panels are accessible through a **legacy fallback tab group** so no experiment becomes inaccessible mid-migration.

Phase 1 risk is concentrated in the Golden Layout integration with Vite + plain TS. Mitigation: a tracer-bullet task that gets Golden Layout rendering a "Hello world" pane before any existing code is touched.

### Phase 2 — Experiments as panes + tier wizard

Ship the transformation. At the end of Phase 2:
- J–V / Impedance / Degradation panels become dockable panes wired to the tree. Clicking an experiment node activates its pane.
- Runs persist under their experiment node. Clicking a run loads it into the Main Plot.
- **New Device wizard** with Legacy / Fast / Full cards is the only way to create a device. Tier is stored on the device node and shown as a badge in the tree.
- Parameter Inspector becomes context-sensitive and tier-gated — fields hidden for the active tier are not rendered.
- Solver console's first line states active physics per active device.
- Legacy fallback tab group removed.

This is the phase where the app visibly transforms. The original motivation — making the physics tier feel consequential — is delivered.

### Phase 3 — Device Schematic

Ship the polish. At the end of Phase 3:
- A new **Device Schematic** pane renders the active device as a drawn horizontal stack. Layer widths are proportional to `thickness`. Colours map to `role`. Layer names are overlaid.
- Clicking a layer in the schematic selects it in the Inspector and highlights it.
- Interfaces (the gaps between layers) are clickable and open the interface SRV editor.
- Hovering a layer while viewing a J–V result highlights the corresponding region in the band diagram pane.

Phase 3 is largely drawing work (SVG or canvas). No new physics, no new data. Standalone because it deserves visual polish time.

### Phase 4 — Compare Runs

Ship the final payoff. At the end of Phase 4:
- A new **Compare Runs** pane replaces / augments the Results folder in the tree. Users can select any set of runs from any devices and render them into a single overlaid plot.
- Multi-run overlays on the Main Plot become the default when a Compare View is active.
- Run metadata diff — two runs side-by-side showing which parameters differ, colour-coded.

Phase 4 is orthogonal to everything else. It could in principle ship before Phase 3, but shipping it last means the schematic is available while implementing the comparison UI.

## Files expected to be touched

Phase 1 is the only phase fully scoped in this spec. The file list below is for **Phase 1 only**; later phases will add their own.

**New files:**
- `frontend/src/workstation/shell.ts` — Golden Layout setup, pane registration
- `frontend/src/workstation/tree.ts` — project tree rendering + workspace state
- `frontend/src/workstation/console.ts` — bottom solver console strip
- `frontend/src/workstation/panes/device-pane.ts` — wrapper around existing `config-editor.ts`
- `frontend/src/workstation/panes/help-pane.ts` — wrapper around existing tutorial / parameters / algorithm HTML
- `frontend/src/workstation/panes/legacy-experiment-pane.ts` — fallback host for existing J–V / Impedance / Degradation panels
- `frontend/src/workstation/state.ts` — workspace type + localStorage persistence
- `frontend/src/workstation/types.ts` — workspace / device / experiment / run types

**Modified files:**
- `frontend/src/main.ts` — replace tab-bar with workstation shell bootstrap
- `frontend/src/style.css` — dock area CSS, left sidebar, solver console strip, tier badges
- `frontend/index.html` — add Golden Layout CSS link
- `frontend/package.json` — add `golden-layout@^2` dependency
- `frontend/src/panels/{tutorial,parameters,algorithm}.ts` — export their HTML-producing functions so the Help pane can call them without a full mount
- `backend/main.py` — include `active_physics` in the `result` SSE payload

**Deleted / retired (Phase 2):**
- The top-level tab strip in `main.ts` (kept in Phase 1 as the legacy fallback, removed in Phase 2)

## Testing approach

- **Unit** — workspace state persistence, tree mutation functions, tier-gating logic (pure functions, easy to unit-test).
- **Integration** — one Playwright test per phase that loads the app, performs the phase's headline workflow (e.g. Phase 1: open app, expand project tree, run the default experiment; Phase 2: create a new Legacy device via wizard, run J–V, verify Parameter Inspector hides `D_ion,-`).
- **Regression** — the existing pytest suite is unchanged because the backend surface is unchanged.

## Open questions

None blocking Phase 1. Phases 2–4 will surface their own questions as they are specced in detail.

## References

- Current implementation: `frontend/src/main.ts`, `frontend/src/device-panel.ts`, `frontend/src/config-editor.ts`, `frontend/src/panels/*.ts`
- Phase 5 tier gating (backend): `perovskite_sim/models/mode.py`, `perovskite_sim/solver/mol.py:build_material_arrays`
- Algorithm-tab comparison table: `frontend/src/panels/algorithm.ts` — the New Device wizard's tier cards will embed this content
- Golden Layout v2: https://github.com/golden-layout/golden-layout
- Visual inspiration: Silvaco DeckBuild, Sentaurus Workbench, SCAPS-1D (for the device schematic), VS Code (for the dock behaviour)
