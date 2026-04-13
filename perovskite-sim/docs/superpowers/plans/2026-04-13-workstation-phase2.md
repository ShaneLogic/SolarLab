# Workstation Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make experiments and runs first-class objects under devices in the workstation tree, replace ad-hoc device creation with a tier-gated wizard, and delete the legacy fallback experiment pane.

**Architecture:** Extend the existing workspace state with `experiments[]` and `runs[]` collections, add pure state operations (immutable update helpers) and persist via the existing `saveWorkspace` path. Add three new dockable panes (JV / Impedance / Degradation) that read the active device from the workspace, dispatch jobs through the existing SSE `job-stream`, and persist the resulting `Run` back into the workspace. A new `main-plot` pane renders whichever run is active. A modal `New Device Wizard` is the only entry point for adding devices — the three tier cards (Legacy / Fast / Full) pre-select a preset and stamp the resulting `Device.tier`. The config editor gains a `tier` prop that hides fields not used by the active tier. The legacy fallback `legacy-experiments` pane is deleted.

**Tech Stack:** Golden Layout v2 (already wired), vanilla TS, vitest + jsdom, Plotly (existing), the existing `job-stream.ts` / `progress.ts` helpers.

---

## File Structure

**New files:**
- `frontend/src/workstation/tier-gating.ts` — pure function + key set: which field keys are hidden per tier
- `frontend/src/workstation/tier-gating.test.ts` — vitest for the above
- `frontend/src/workstation/panes/main-plot-pane.ts` — renders active run (jv/impedance/degradation) into one pane
- `frontend/src/workstation/panes/jv-pane.ts` — dockable JV experiment pane (workspace-aware)
- `frontend/src/workstation/panes/impedance-pane.ts` — dockable Impedance pane
- `frontend/src/workstation/panes/degradation-pane.ts` — dockable Degradation pane
- `frontend/src/workstation/wizard.ts` — modal New Device Wizard (three tier cards + preset picker)
- `frontend/src/workstation/wizard.test.ts` — vitest for wizard pure builders

**Modified files:**
- `frontend/src/workstation/types.ts` — add `activeExperimentId`, `activeRunId` to `Workspace`; tighten `Run.result` and `Experiment.params` typing
- `frontend/src/workstation/state.ts` — add `addExperiment`, `removeExperiment`, `setActiveExperiment`, `addRun`, `removeRun`, `setActiveRun`, `findRun`
- `frontend/src/workstation/state.test.ts` — cover the new state ops
- `frontend/src/workstation/tree.ts` — render hierarchical tree: devices → experiments → runs, with expand/collapse and click handlers
- `frontend/src/workstation/tree.test.ts` — cover hierarchical render + click dispatch
- `frontend/src/workstation/shell.ts` — register main-plot / jv / impedance / degradation component factories, remove legacy-experiments factory, wire tree clicks to activate panes and update workspace, mount the wizard on first load or when the user clicks “+ New Device”, update `DEFAULT_LAYOUT`
- `frontend/src/config-editor.ts` — accept a `tier` option; hide fields that are not used by the tier
- `frontend/src/device-panel.ts` — forward the `tier` option to `config-editor.ts`
- `frontend/src/style.css` — wizard modal, tree child nodes, run-item highlight, main-plot pane

**Deleted files:**
- `frontend/src/workstation/panes/legacy-experiment-pane.ts` — replaced by the three new dockable panes

---

## Task 1: Extend Workspace type with active experiment / run

**Files:**
- Modify: `frontend/src/workstation/types.ts`
- Modify: `frontend/src/workstation/state.ts`
- Modify: `frontend/src/workstation/state.test.ts`

- [ ] **Step 1: Update the Workspace interface**

Edit `types.ts` — add two optional active IDs. Keep `Run.result` as `unknown` for now (Task 6 tightens it):

```typescript
export interface Workspace {
  version: 1
  id: string
  name: string
  devices: Device[]
  activeDeviceId: string | null
  activeExperimentId: string | null
  activeRunId: string | null
  layout: unknown | null
}
```

Also change `Experiment.kind` to `ExperimentKind` (new exported type), so state ops can narrow:

```typescript
export type ExperimentKind = 'jv' | 'impedance' | 'degradation'

export interface Experiment {
  id: string
  kind: ExperimentKind
  params: Record<string, unknown>
  runs: Run[]
}
```

- [ ] **Step 2: Update `createEmptyWorkspace`**

Edit `state.ts` — add `activeExperimentId: null`, `activeRunId: null` to the object literal in `createEmptyWorkspace`.

- [ ] **Step 3: Update the existing state.test.ts assertion**

Edit `state.test.ts` — extend `createEmptyWorkspace` test to also assert the two new fields are `null`:

```typescript
expect(ws.activeExperimentId).toBeNull()
expect(ws.activeRunId).toBeNull()
```

- [ ] **Step 4: Run tests**

Run: `cd frontend && npm test -- --run state.test.ts`
Expected: all existing tests PASS (one gains extra assertions).

- [ ] **Step 5: Commit**

```bash
git add frontend/src/workstation/types.ts frontend/src/workstation/state.ts frontend/src/workstation/state.test.ts
git commit -m "feat(workstation): add activeExperimentId / activeRunId to Workspace"
```

---

## Task 2: State operations for experiments

**Files:**
- Modify: `frontend/src/workstation/state.ts`
- Modify: `frontend/src/workstation/state.test.ts`

- [ ] **Step 1: Write failing tests (TDD red)**

Append to `state.test.ts`:

```typescript
import {
  addExperiment,
  removeExperiment,
  setActiveExperiment,
} from './state'
import type { Experiment } from './types'

function makeExperiment(id: string, kind: 'jv' | 'impedance' | 'degradation' = 'jv'): Experiment {
  return { id, kind, params: {}, runs: [] }
}

describe('addExperiment', () => {
  it('appends experiment to the named device, leaves other devices alone', () => {
    let ws = createEmptyWorkspace('W')
    ws = addDevice(ws, makeDevice('d1'))
    ws = addDevice(ws, makeDevice('d2'))
    const next = addExperiment(ws, 'd1', makeExperiment('e1'))
    expect(next.devices.find(d => d.id === 'd1')!.experiments).toHaveLength(1)
    expect(next.devices.find(d => d.id === 'd2')!.experiments).toHaveLength(0)
  })

  it('is a no-op when the device id is unknown', () => {
    let ws = addDevice(createEmptyWorkspace('W'), makeDevice('d1'))
    const next = addExperiment(ws, 'unknown', makeExperiment('e1'))
    expect(next).toBe(ws)
  })

  it('returns a new workspace reference (immutability)', () => {
    let ws = addDevice(createEmptyWorkspace('W'), makeDevice('d1'))
    const next = addExperiment(ws, 'd1', makeExperiment('e1'))
    expect(next).not.toBe(ws)
    expect(ws.devices[0].experiments).toHaveLength(0)
  })
})

describe('removeExperiment', () => {
  it('removes the experiment from its device', () => {
    let ws = addDevice(createEmptyWorkspace('W'), makeDevice('d1'))
    ws = addExperiment(ws, 'd1', makeExperiment('e1'))
    ws = addExperiment(ws, 'd1', makeExperiment('e2'))
    const next = removeExperiment(ws, 'd1', 'e1')
    expect(next.devices[0].experiments.map(e => e.id)).toEqual(['e2'])
  })

  it('clears activeExperimentId when the removed experiment was active', () => {
    let ws = addDevice(createEmptyWorkspace('W'), makeDevice('d1'))
    ws = addExperiment(ws, 'd1', makeExperiment('e1'))
    ws = setActiveExperiment(ws, 'd1', 'e1')
    const next = removeExperiment(ws, 'd1', 'e1')
    expect(next.activeExperimentId).toBeNull()
  })
})

describe('setActiveExperiment', () => {
  it('sets both activeDeviceId and activeExperimentId', () => {
    let ws = addDevice(createEmptyWorkspace('W'), makeDevice('d1'))
    ws = addExperiment(ws, 'd1', makeExperiment('e1'))
    const next = setActiveExperiment(ws, 'd1', 'e1')
    expect(next.activeDeviceId).toBe('d1')
    expect(next.activeExperimentId).toBe('e1')
  })

  it('is a no-op when device or experiment id is unknown', () => {
    const ws = addDevice(createEmptyWorkspace('W'), makeDevice('d1'))
    expect(setActiveExperiment(ws, 'd1', 'missing')).toBe(ws)
    expect(setActiveExperiment(ws, 'missing', 'e1')).toBe(ws)
  })
})
```

- [ ] **Step 2: Run tests — verify they fail with "is not a function"**

Run: `cd frontend && npm test -- --run state.test.ts`
Expected: FAIL (imports of `addExperiment`, `removeExperiment`, `setActiveExperiment` error).

- [ ] **Step 3: Implement (TDD green)**

Append to `state.ts`:

```typescript
import type { Device, Experiment, Workspace } from './types'

function mapDevice(
  ws: Workspace,
  deviceId: string,
  fn: (d: Device) => Device,
): Workspace {
  const idx = ws.devices.findIndex(d => d.id === deviceId)
  if (idx < 0) return ws
  const devices = ws.devices.map((d, i) => (i === idx ? fn(d) : d))
  return { ...ws, devices }
}

export function addExperiment(
  ws: Workspace,
  deviceId: string,
  experiment: Experiment,
): Workspace {
  return mapDevice(ws, deviceId, d => ({
    ...d,
    experiments: [...d.experiments, experiment],
  }))
}

export function removeExperiment(
  ws: Workspace,
  deviceId: string,
  experimentId: string,
): Workspace {
  const next = mapDevice(ws, deviceId, d => ({
    ...d,
    experiments: d.experiments.filter(e => e.id !== experimentId),
  }))
  if (next.activeExperimentId === experimentId) {
    return { ...next, activeExperimentId: null, activeRunId: null }
  }
  return next
}

export function setActiveExperiment(
  ws: Workspace,
  deviceId: string,
  experimentId: string,
): Workspace {
  const dev = ws.devices.find(d => d.id === deviceId)
  if (!dev) return ws
  if (!dev.experiments.some(e => e.id === experimentId)) return ws
  if (ws.activeDeviceId === deviceId && ws.activeExperimentId === experimentId) return ws
  return { ...ws, activeDeviceId: deviceId, activeExperimentId: experimentId, activeRunId: null }
}
```

Note: `mapDevice` is a private helper — don't export it.

- [ ] **Step 4: Run tests — verify green**

Run: `cd frontend && npm test -- --run state.test.ts`
Expected: PASS all tests.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/workstation/state.ts frontend/src/workstation/state.test.ts
git commit -m "feat(workstation): state ops for experiments"
```

---

## Task 3: State operations for runs

**Files:**
- Modify: `frontend/src/workstation/state.ts`
- Modify: `frontend/src/workstation/state.test.ts`

- [ ] **Step 1: Write failing tests**

Append to `state.test.ts`:

```typescript
import { addRun, removeRun, setActiveRun, findRun } from './state'
import type { Run } from './types'

function makeRun(id: string): Run {
  return {
    id,
    timestamp: Date.now(),
    result: { placeholder: true },
    activePhysics: 'FULL',
    durationMs: 123,
    deviceSnapshot: { device: { V_bi: 1.1, Phi: 1.4e21 }, layers: [] },
  }
}

describe('addRun', () => {
  it('appends run under the named experiment', () => {
    let ws = addDevice(createEmptyWorkspace('W'), makeDevice('d1'))
    ws = addExperiment(ws, 'd1', makeExperiment('e1'))
    const next = addRun(ws, 'd1', 'e1', makeRun('r1'))
    expect(next.devices[0].experiments[0].runs.map(r => r.id)).toEqual(['r1'])
  })

  it('is a no-op when device or experiment is unknown', () => {
    let ws = addDevice(createEmptyWorkspace('W'), makeDevice('d1'))
    ws = addExperiment(ws, 'd1', makeExperiment('e1'))
    expect(addRun(ws, 'd1', 'missing', makeRun('r1'))).toBe(ws)
    expect(addRun(ws, 'missing', 'e1', makeRun('r1'))).toBe(ws)
  })
})

describe('removeRun', () => {
  it('removes the run and clears activeRunId if it was active', () => {
    let ws = addDevice(createEmptyWorkspace('W'), makeDevice('d1'))
    ws = addExperiment(ws, 'd1', makeExperiment('e1'))
    ws = addRun(ws, 'd1', 'e1', makeRun('r1'))
    ws = setActiveRun(ws, 'd1', 'e1', 'r1')
    const next = removeRun(ws, 'd1', 'e1', 'r1')
    expect(next.devices[0].experiments[0].runs).toHaveLength(0)
    expect(next.activeRunId).toBeNull()
  })
})

describe('setActiveRun', () => {
  it('sets activeDevice/Experiment/Run', () => {
    let ws = addDevice(createEmptyWorkspace('W'), makeDevice('d1'))
    ws = addExperiment(ws, 'd1', makeExperiment('e1'))
    ws = addRun(ws, 'd1', 'e1', makeRun('r1'))
    const next = setActiveRun(ws, 'd1', 'e1', 'r1')
    expect(next.activeDeviceId).toBe('d1')
    expect(next.activeExperimentId).toBe('e1')
    expect(next.activeRunId).toBe('r1')
  })
})

describe('findRun', () => {
  it('returns the run when it exists', () => {
    let ws = addDevice(createEmptyWorkspace('W'), makeDevice('d1'))
    ws = addExperiment(ws, 'd1', makeExperiment('e1'))
    ws = addRun(ws, 'd1', 'e1', makeRun('r1'))
    const found = findRun(ws, 'd1', 'e1', 'r1')
    expect(found?.id).toBe('r1')
  })

  it('returns undefined for an unknown triple', () => {
    expect(findRun(createEmptyWorkspace('W'), 'x', 'y', 'z')).toBeUndefined()
  })
})
```

- [ ] **Step 2: Run tests — verify failure**

- [ ] **Step 3: Implement**

Append to `state.ts`:

```typescript
import type { Run } from './types'

function mapExperiment(
  ws: Workspace,
  deviceId: string,
  experimentId: string,
  fn: (e: Experiment) => Experiment,
): Workspace {
  return mapDevice(ws, deviceId, d => {
    const idx = d.experiments.findIndex(e => e.id === experimentId)
    if (idx < 0) return d
    const experiments = d.experiments.map((e, i) => (i === idx ? fn(e) : e))
    return { ...d, experiments }
  })
}

export function addRun(
  ws: Workspace,
  deviceId: string,
  experimentId: string,
  run: Run,
): Workspace {
  const dev = ws.devices.find(d => d.id === deviceId)
  if (!dev) return ws
  if (!dev.experiments.some(e => e.id === experimentId)) return ws
  return mapExperiment(ws, deviceId, experimentId, e => ({
    ...e,
    runs: [...e.runs, run],
  }))
}

export function removeRun(
  ws: Workspace,
  deviceId: string,
  experimentId: string,
  runId: string,
): Workspace {
  const next = mapExperiment(ws, deviceId, experimentId, e => ({
    ...e,
    runs: e.runs.filter(r => r.id !== runId),
  }))
  if (next.activeRunId === runId) {
    return { ...next, activeRunId: null }
  }
  return next
}

export function setActiveRun(
  ws: Workspace,
  deviceId: string,
  experimentId: string,
  runId: string,
): Workspace {
  const run = findRun(ws, deviceId, experimentId, runId)
  if (!run) return ws
  if (
    ws.activeDeviceId === deviceId &&
    ws.activeExperimentId === experimentId &&
    ws.activeRunId === runId
  ) return ws
  return {
    ...ws,
    activeDeviceId: deviceId,
    activeExperimentId: experimentId,
    activeRunId: runId,
  }
}

export function findRun(
  ws: Workspace,
  deviceId: string,
  experimentId: string,
  runId: string,
): Run | undefined {
  const d = ws.devices.find(d => d.id === deviceId)
  const e = d?.experiments.find(e => e.id === experimentId)
  return e?.runs.find(r => r.id === runId)
}
```

- [ ] **Step 4: Run tests — verify green**

- [ ] **Step 5: Commit**

```bash
git add frontend/src/workstation/state.ts frontend/src/workstation/state.test.ts
git commit -m "feat(workstation): state ops for runs"
```

---

## Task 4: Tier-gating pure function

**Files:**
- Create: `frontend/src/workstation/tier-gating.ts`
- Create: `frontend/src/workstation/tier-gating.test.ts`

- [ ] **Step 1: Write failing tests**

Create `tier-gating.test.ts`:

```typescript
import { describe, it, expect } from 'vitest'
import { isFieldVisible, hiddenKeysForTier } from './tier-gating'

describe('isFieldVisible', () => {
  it('always shows the core Geometry & Electrostatics fields regardless of tier', () => {
    for (const tier of ['legacy', 'fast', 'full'] as const) {
      expect(isFieldVisible('thickness', tier)).toBe(true)
      expect(isFieldVisible('eps_r', tier)).toBe(true)
      expect(isFieldVisible('mu_n', tier)).toBe(true)
    }
  })

  it('hides TMM-only fields in legacy and fast', () => {
    expect(isFieldVisible('optical_material', 'legacy')).toBe(false)
    expect(isFieldVisible('optical_material', 'fast')).toBe(false)
    expect(isFieldVisible('optical_material', 'full')).toBe(true)
  })

  it('hides dual-ion fields in legacy and fast', () => {
    expect(isFieldVisible('D_ion_neg', 'legacy')).toBe(false)
    expect(isFieldVisible('D_ion_neg', 'fast')).toBe(false)
    expect(isFieldVisible('D_ion_neg', 'full')).toBe(true)
    expect(isFieldVisible('P_lim_neg', 'legacy')).toBe(false)
  })

  it('hides trap-profile fields in legacy and fast', () => {
    expect(isFieldVisible('trap_N_t_interface', 'legacy')).toBe(false)
    expect(isFieldVisible('trap_N_t_bulk', 'legacy')).toBe(false)
    expect(isFieldVisible('trap_N_t_interface', 'full')).toBe(true)
  })

  it('hides device-level T input in legacy (fixed 300 K)', () => {
    expect(isFieldVisible('T', 'legacy')).toBe(false)
    expect(isFieldVisible('T', 'fast')).toBe(false)
    expect(isFieldVisible('T', 'full')).toBe(true)
  })

  it('unknown field keys default to visible (fail-open)', () => {
    expect(isFieldVisible('some_new_future_key', 'legacy')).toBe(true)
  })
})

describe('hiddenKeysForTier', () => {
  it('legacy hides everything fast hides plus T', () => {
    const legacy = hiddenKeysForTier('legacy')
    const fast = hiddenKeysForTier('fast')
    for (const k of fast) expect(legacy).toContain(k)
  })

  it('full hides nothing', () => {
    expect(hiddenKeysForTier('full')).toEqual([])
  })
})
```

- [ ] **Step 2: Run tests — verify failure**

Run: `cd frontend && npm test -- --run tier-gating`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

Create `tier-gating.ts`:

```typescript
import type { SimulationModeName } from '../types'

/**
 * Keys of layer / device fields that belong to physics upgrades gated by
 * the tiered SimulationMode. Fields not listed here are always visible.
 *
 * Keep this in sync with perovskite_sim/models/mode.py: a field belongs to
 * a tier iff the mode that enables its physics is included in that tier.
 */
const TMM_KEYS = ['optical_material', 'n_optical'] as const
const DUAL_ION_KEYS = ['D_ion_neg', 'P_lim_neg', 'E_a_ion_neg'] as const
const TRAP_PROFILE_KEYS = ['trap_N_t_interface', 'trap_N_t_bulk', 'trap_decay_length'] as const
const TEMPERATURE_KEYS = ['T'] as const

/** Keys hidden in FAST mode (no TMM, no dual ions, no trap profile, no T input). */
const FAST_HIDDEN = new Set<string>([
  ...TMM_KEYS,
  ...DUAL_ION_KEYS,
  ...TRAP_PROFILE_KEYS,
  ...TEMPERATURE_KEYS,
])

/** Keys hidden in LEGACY mode — identical to FAST today (mode.py:54-61). */
const LEGACY_HIDDEN = new Set<string>(FAST_HIDDEN)

const HIDDEN_BY_TIER: Record<SimulationModeName, Set<string>> = {
  legacy: LEGACY_HIDDEN,
  fast: FAST_HIDDEN,
  full: new Set<string>(),
}

export function isFieldVisible(key: string, tier: SimulationModeName): boolean {
  return !HIDDEN_BY_TIER[tier].has(key)
}

export function hiddenKeysForTier(tier: SimulationModeName): string[] {
  return [...HIDDEN_BY_TIER[tier]]
}
```

- [ ] **Step 4: Run tests — verify green**

- [ ] **Step 5: Commit**

```bash
git add frontend/src/workstation/tier-gating.ts frontend/src/workstation/tier-gating.test.ts
git commit -m "feat(workstation): tier-gating field-visibility function"
```

---

## Task 5: Hierarchical tree rendering

**Files:**
- Modify: `frontend/src/workstation/tree.ts`
- Modify: `frontend/src/workstation/tree.test.ts`

Current tree renders a flat devices list. We need devices → experiments → runs, with click dispatch for each node type.

- [ ] **Step 1: Write failing tests**

Append to `tree.test.ts`:

```typescript
import { renderTreeHTML, attachTreeHandlers } from './tree'
import { addExperiment, addRun } from './state'
// makeDevice / makeExperiment / makeRun already defined (or copy from state.test.ts)

describe('hierarchical rendering', () => {
  it('renders experiment nodes under their device', () => {
    let ws = createEmptyWorkspace('W')
    ws = addDevice(ws, makeDevice('d1'))
    ws = addExperiment(ws, 'd1', makeExperiment('e1', 'jv'))
    const html = renderTreeHTML(ws)
    expect(html).toContain('data-device-id="d1"')
    expect(html).toContain('data-experiment-id="e1"')
    expect(html).toMatch(/J.V/i)
  })

  it('renders run nodes under their experiment with timestamp', () => {
    let ws = createEmptyWorkspace('W')
    ws = addDevice(ws, makeDevice('d1'))
    ws = addExperiment(ws, 'd1', makeExperiment('e1', 'jv'))
    ws = addRun(ws, 'd1', 'e1', makeRun('r1'))
    const html = renderTreeHTML(ws)
    expect(html).toContain('data-run-id="r1"')
  })

  it('escapes malicious device names', () => {
    let ws = createEmptyWorkspace('W')
    ws = addDevice(ws, { ...makeDevice('d1'), name: '<img src=x onerror=alert(1)>' })
    const html = renderTreeHTML(ws)
    expect(html).not.toContain('<img')
    expect(html).toContain('&lt;img')
  })
})

describe('attachTreeHandlers dispatch', () => {
  it('dispatches onSelectExperiment when an experiment node is clicked', () => {
    let ws = createEmptyWorkspace('W')
    ws = addDevice(ws, makeDevice('d1'))
    ws = addExperiment(ws, 'd1', makeExperiment('e1', 'jv'))
    const el = document.createElement('div')
    el.innerHTML = renderTreeHTML(ws)
    const spy = vi.fn()
    attachTreeHandlers(el, {
      onSelectDevice: () => {},
      onSelectExperiment: spy,
      onSelectRun: () => {},
    })
    el.querySelector<HTMLElement>('[data-experiment-id="e1"]')!.click()
    expect(spy).toHaveBeenCalledWith('d1', 'e1')
  })

  it('dispatches onSelectRun when a run node is clicked', () => {
    let ws = createEmptyWorkspace('W')
    ws = addDevice(ws, makeDevice('d1'))
    ws = addExperiment(ws, 'd1', makeExperiment('e1', 'jv'))
    ws = addRun(ws, 'd1', 'e1', makeRun('r1'))
    const el = document.createElement('div')
    el.innerHTML = renderTreeHTML(ws)
    const spy = vi.fn()
    attachTreeHandlers(el, {
      onSelectDevice: () => {},
      onSelectExperiment: () => {},
      onSelectRun: spy,
    })
    el.querySelector<HTMLElement>('[data-run-id="r1"]')!.click()
    expect(spy).toHaveBeenCalledWith('d1', 'e1', 'r1')
  })
})
```

Import `vi` from vitest at the top if not already present.

- [ ] **Step 2: Run tests — verify failure**

- [ ] **Step 3: Implement — rewrite renderTreeHTML and attachTreeHandlers**

Replace `tree.ts` with:

```typescript
import type { Workspace, Experiment, Run } from './types'
import type { SimulationModeName } from '../types'

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}

function tierBadge(tier: SimulationModeName): string {
  return `<span class="tier-badge tier-badge-${tier}">${tier.toUpperCase()}</span>`
}

function experimentLabel(kind: Experiment['kind']): string {
  switch (kind) {
    case 'jv': return 'J–V Sweep'
    case 'impedance': return 'Impedance'
    case 'degradation': return 'Degradation'
  }
}

function runLabel(r: Run): string {
  const t = new Date(r.timestamp)
  const hh = String(t.getHours()).padStart(2, '0')
  const mm = String(t.getMinutes()).padStart(2, '0')
  return `Run ${hh}:${mm}`
}

function renderRun(deviceId: string, experimentId: string, r: Run, activeRunId: string | null): string {
  const active = r.id === activeRunId ? ' tree-node-active' : ''
  return `
    <div class="tree-node tree-node-run${active}"
         data-device-id="${escapeHtml(deviceId)}"
         data-experiment-id="${escapeHtml(experimentId)}"
         data-run-id="${escapeHtml(r.id)}">
      <span class="tree-icon">▶</span>
      <span class="tree-label">${escapeHtml(runLabel(r))}</span>
    </div>`
}

function renderExperiment(deviceId: string, e: Experiment, ws: Workspace): string {
  const active = e.id === ws.activeExperimentId ? ' tree-node-active' : ''
  const runs = e.runs.map(r => renderRun(deviceId, e.id, r, ws.activeRunId)).join('')
  return `
    <div class="tree-node tree-node-experiment${active}"
         data-device-id="${escapeHtml(deviceId)}"
         data-experiment-id="${escapeHtml(e.id)}">
      <span class="tree-icon">🧪</span>
      <span class="tree-label">${escapeHtml(experimentLabel(e.kind))}</span>
    </div>
    <div class="tree-children">${runs}</div>`
}

export function renderTreeHTML(ws: Workspace): string {
  const deviceNodes = ws.devices
    .map(d => {
      const active = d.id === ws.activeDeviceId ? ' tree-node-active' : ''
      const experiments = d.experiments.map(e => renderExperiment(d.id, e, ws)).join('')
      return `
        <div class="tree-node tree-node-device${active}" data-device-id="${escapeHtml(d.id)}">
          <span class="tree-icon">🔬</span>
          <span class="tree-label">${escapeHtml(d.name)}</span>
          ${tierBadge(d.tier)}
        </div>
        <div class="tree-children">${experiments}</div>`
    })
    .join('')

  return `
    <div class="tree-section">
      <div class="tree-section-header">📁 Devices</div>
      <div class="tree-section-body">${deviceNodes || '<div class="tree-empty">(no devices yet)</div>'}</div>
    </div>
    <div class="tree-section">
      <div class="tree-section-header">📁 Results / Compare</div>
      <div class="tree-section-body"><div class="tree-empty">(no runs yet)</div></div>
    </div>`
}

export interface TreeHandlers {
  onSelectDevice: (deviceId: string) => void
  onSelectExperiment: (deviceId: string, experimentId: string) => void
  onSelectRun: (deviceId: string, experimentId: string, runId: string) => void
}

export function attachTreeHandlers(container: HTMLElement, handlers: TreeHandlers): void {
  container.addEventListener('click', (e) => {
    const target = e.target as HTMLElement
    const runNode = target.closest<HTMLElement>('[data-run-id]')
    if (runNode) {
      const deviceId = runNode.dataset.deviceId
      const experimentId = runNode.dataset.experimentId
      const runId = runNode.dataset.runId
      if (deviceId && experimentId && runId) {
        handlers.onSelectRun(deviceId, experimentId, runId)
      }
      return
    }
    const expNode = target.closest<HTMLElement>('[data-experiment-id]')
    if (expNode && !expNode.hasAttribute('data-run-id')) {
      const deviceId = expNode.dataset.deviceId
      const experimentId = expNode.dataset.experimentId
      if (deviceId && experimentId) {
        handlers.onSelectExperiment(deviceId, experimentId)
      }
      return
    }
    const devNode = target.closest<HTMLElement>('[data-device-id]')
    if (devNode && !devNode.hasAttribute('data-experiment-id')) {
      const id = devNode.dataset.deviceId
      if (id) handlers.onSelectDevice(id)
    }
  })
}
```

The `tree-children` wrapper + `tree-node-run` class are new — style later in Task 14.

- [ ] **Step 4: Run tests — verify green**

- [ ] **Step 5: Commit**

```bash
git add frontend/src/workstation/tree.ts frontend/src/workstation/tree.test.ts
git commit -m "feat(workstation): hierarchical tree — experiments and runs"
```

---

## Task 6: Tighten Run.result typing

**Files:**
- Modify: `frontend/src/workstation/types.ts`

- [ ] **Step 1: Make `Run.result` discriminated by `kind`**

Edit `types.ts`:

```typescript
import type { JVResult, ISResult, DegResult } from '../types'

export type RunResult =
  | { kind: 'jv'; data: JVResult }
  | { kind: 'impedance'; data: ISResult }
  | { kind: 'degradation'; data: DegResult }

export interface Run {
  id: string
  timestamp: number
  result: RunResult
  activePhysics: string
  durationMs: number
  deviceSnapshot: DeviceConfig
}
```

- [ ] **Step 2: Update the test helper**

Edit `state.test.ts` — change `makeRun` to use the discriminated form:

```typescript
result: { kind: 'jv' as const, data: {
  V_fwd: [], J_fwd: [], V_rev: [], J_rev: [],
  metrics_fwd: { V_oc: 0, J_sc: 0, FF: 0, PCE: 0 },
  metrics_rev: { V_oc: 0, J_sc: 0, FF: 0, PCE: 0 },
  hysteresis_index: 0,
} }
```

- [ ] **Step 3: Run tests**

Run: `cd frontend && npm test -- --run state.test.ts tree.test.ts`
Expected: PASS (tests use the new `RunResult` shape).

- [ ] **Step 4: Commit**

```bash
git add frontend/src/workstation/types.ts frontend/src/workstation/state.test.ts
git commit -m "refactor(workstation): discriminate Run.result by kind"
```

---

## Task 7: Main Plot pane

**Files:**
- Create: `frontend/src/workstation/panes/main-plot-pane.ts`

This pane subscribes to workspace changes, looks up the active run, and renders it. Phase 2 only needs JV rendering to be complete — impedance and degradation can render a placeholder and be filled in when their panes land.

- [ ] **Step 1: Implement the pane**

Create `main-plot-pane.ts`:

```typescript
import Plotly from 'plotly.js-dist-min'
import type { Workspace } from '../types'
import type { RunResult } from '../types'
import { findRun } from '../state'
import { baseLayout, plotConfig, PALETTE, LINE, MARKER, axisTitle } from '../../plot-theme'
import type { JVResult } from '../../types'

export interface MainPlotHandle {
  update(ws: Workspace): void
}

export function mountMainPlotPane(container: HTMLElement): MainPlotHandle {
  container.innerHTML = `
    <div class="main-plot-pane">
      <div class="main-plot-header" id="mpp-header">(no active run)</div>
      <div id="mpp-plot" class="plot-container"></div>
    </div>`

  const header = container.querySelector<HTMLDivElement>('#mpp-header')!
  const plotEl = container.querySelector<HTMLDivElement>('#mpp-plot')!

  function clear(msg: string): void {
    header.textContent = msg
    Plotly.purge(plotEl)
    plotEl.innerHTML = '<div class="plot-empty">Run an experiment to see results here.</div>'
  }

  clear('(no active run)')

  return {
    update(ws: Workspace) {
      if (!ws.activeRunId || !ws.activeDeviceId || !ws.activeExperimentId) {
        clear('(no active run)')
        return
      }
      const run = findRun(ws, ws.activeDeviceId, ws.activeExperimentId, ws.activeRunId)
      if (!run) {
        clear('(run not found)')
        return
      }
      header.textContent = `${run.activePhysics}  ·  ${new Date(run.timestamp).toLocaleString()}`
      switch (run.result.kind) {
        case 'jv':
          renderJV(plotEl, run.result.data)
          return
        case 'impedance':
          clear('(impedance plot in this pane: Task 9)')
          return
        case 'degradation':
          clear('(degradation plot in this pane: Task 10)')
          return
      }
    },
  }
}

function renderJV(el: HTMLElement, r: JVResult): void {
  el.innerHTML = ''
  const J_fwd_mA = r.J_fwd.map(j => j / 10)
  const J_rev_mA = r.J_rev.map(j => j / 10)
  const V_rev_sorted = [...r.V_rev].reverse()
  const J_rev_sorted = [...J_rev_mA].reverse()
  Plotly.newPlot(
    el,
    [
      {
        x: r.V_fwd, y: J_fwd_mA, name: 'Forward',
        mode: 'lines+markers',
        line: { color: PALETTE.forward, width: LINE.width },
        marker: { ...MARKER, color: PALETTE.forward },
      },
      {
        x: V_rev_sorted, y: J_rev_sorted, name: 'Reverse',
        mode: 'lines+markers',
        line: { color: PALETTE.reverse, width: LINE.width, dash: 'dash' },
        marker: { ...MARKER, color: PALETTE.reverse, symbol: 'square' },
      },
    ],
    baseLayout({
      xaxis: { ...(baseLayout().xaxis as object), title: axisTitle('Applied bias, <i>V</i> (V)') },
      yaxis: { ...(baseLayout().yaxis as object), title: axisTitle('Current density, <i>J</i> (mA·cm⁻²)') },
    }),
    plotConfig('jv_sweep'),
  )
}
```

- [ ] **Step 2: Commit (no tests yet — pure DOM glue; covered by integration in Task 15)**

```bash
git add frontend/src/workstation/panes/main-plot-pane.ts
git commit -m "feat(workstation): main plot pane"
```

---

## Task 8: Dockable JV experiment pane

**Files:**
- Create: `frontend/src/workstation/panes/jv-pane.ts`

The new pane reads the active device from the workspace (not from its own device-panel copy), runs the job, and calls back when a run completes so the shell can persist it.

- [ ] **Step 1: Implement**

Create `jv-pane.ts`:

```typescript
import { startJob, streamJobEvents } from '../../job-stream'
import { createProgressBar, type ProgressBarHandle } from '../../progress'
import { setStatus, numField, readNum } from '../../ui-helpers'
import type { DeviceConfig, JVResult } from '../../types'
import type { Run, RunResult } from '../types'

export interface JVPaneOptions {
  getActiveDevice: () => { id: string; config: DeviceConfig } | null
  onRunComplete: (deviceId: string, run: Run) => void
}

export function mountJVPane(container: HTMLElement, opts: JVPaneOptions): void {
  container.innerHTML = `
    <div class="card">
      <h3>J–V Sweep Parameters</h3>
      <div class="form-grid">
        ${numField('jvp-N', 'N<sub>grid</sub>', 60, '1')}
        ${numField('jvp-np', 'V sample points', 30, '1')}
        ${numField('jvp-rate', 'Scan rate (V/s)', 1.0, 'any')}
        ${numField('jvp-vmax', 'V<sub>max</sub> (V)', 1.4, '0.01')}
      </div>
      <div class="actions">
        <button class="btn btn-primary" id="btn-jvp">Run J–V Sweep</button>
        <span class="status" id="status-jvp"></span>
      </div>
      <div id="progress-jvp"></div>
      <div class="pane-hint">Results stream into the Main Plot pane and appear as a run under this experiment in the tree.</div>
    </div>`

  const progressBar: ProgressBarHandle = createProgressBar(
    container.querySelector<HTMLDivElement>('#progress-jvp')!,
  )
  const btn = container.querySelector<HTMLButtonElement>('#btn-jvp')!

  btn.addEventListener('click', () => {
    const active = opts.getActiveDevice()
    if (!active) {
      setStatus('status-jvp', 'No active device. Select one in the tree.', true)
      return
    }
    btn.disabled = true
    progressBar.reset()
    setStatus('status-jvp', 'Starting job…')

    const params = {
      N_grid: Math.max(3, Math.round(readNum('jvp-N', 60))),
      n_points: Math.max(2, Math.round(readNum('jvp-np', 30))),
      v_rate: readNum('jvp-rate', 1.0),
      V_max: readNum('jvp-vmax', 1.4),
    }
    const t0 = performance.now()
    const snapshot: DeviceConfig = JSON.parse(JSON.stringify(active.config))

    startJob('jv', active.config, params)
      .then(jobId => {
        setStatus('status-jvp', 'Running J–V sweep…')
        streamJobEvents<JVResult & { active_physics?: string }>(jobId, {
          onProgress: (ev) => progressBar.update(ev),
          onResult: (result) => {
            const { active_physics, ...pure } = result as JVResult & { active_physics?: string }
            const runResult: RunResult = { kind: 'jv', data: pure }
            const run: Run = {
              id: randomRunId(),
              timestamp: Date.now(),
              result: runResult,
              activePhysics: active_physics ?? 'unknown',
              durationMs: performance.now() - t0,
              deviceSnapshot: snapshot,
            }
            opts.onRunComplete(active.id, run)
            progressBar.done()
            setStatus('status-jvp', 'Done')
          },
          onError: (msg) => {
            progressBar.error(msg)
            setStatus('status-jvp', `Error: ${msg}`, true)
          },
          onDone: () => {
            btn.disabled = false
          },
        })
      })
      .catch(e => {
        progressBar.error((e as Error).message)
        setStatus('status-jvp', `Error: ${(e as Error).message}`, true)
        btn.disabled = false
      })
  })
}

function randomRunId(): string {
  return 'r-' + Math.random().toString(36).slice(2, 10) + Date.now().toString(36)
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/workstation/panes/jv-pane.ts
git commit -m "feat(workstation): dockable JV experiment pane"
```

---

## Task 9: Dockable Impedance experiment pane

**Files:**
- Create: `frontend/src/workstation/panes/impedance-pane.ts`
- Modify: `frontend/src/workstation/panes/main-plot-pane.ts` (replace the placeholder with a real Nyquist plot)

- [ ] **Step 1: Create impedance-pane.ts**

Follow the JV template. Form fields: `N_grid`, `V_dc`, `n_freq`, `f_min`, `f_max`. Job kind: `'impedance'`. Result type: `ISResult`. When building the `Run`, use `{ kind: 'impedance', data: pure }`.

- [ ] **Step 2: Extend main-plot-pane.ts**

Replace the `case 'impedance': clear(...)` branch with a Nyquist plot using `ISResult.Z_real` vs `ISResult.Z_imag` (negate Z_imag for physical convention). Reuse axis helpers from `plot-theme.ts`.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/workstation/panes/impedance-pane.ts frontend/src/workstation/panes/main-plot-pane.ts
git commit -m "feat(workstation): dockable impedance pane + main-plot Nyquist"
```

---

## Task 10: Dockable Degradation experiment pane

**Files:**
- Create: `frontend/src/workstation/panes/degradation-pane.ts`
- Modify: `frontend/src/workstation/panes/main-plot-pane.ts`

- [ ] **Step 1: Create degradation-pane.ts**

Form: `N_grid`, `V_bias`, `t_end`, `n_snapshots`. Job kind: `'degradation'`. Result type: `DegResult`. Run payload uses `{ kind: 'degradation', data: pure }`.

- [ ] **Step 2: Extend main-plot-pane.ts**

Replace the `case 'degradation'` placeholder with a normalised-PCE-vs-time line plot.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/workstation/panes/degradation-pane.ts frontend/src/workstation/panes/main-plot-pane.ts
git commit -m "feat(workstation): dockable degradation pane + main-plot timeline"
```

---

## Task 11: New Device Wizard

**Files:**
- Create: `frontend/src/workstation/wizard.ts`
- Create: `frontend/src/workstation/wizard.test.ts`

- [ ] **Step 1: Write failing tests for pure builders**

Create `wizard.test.ts`:

```typescript
import { describe, it, expect } from 'vitest'
import { buildWizardHTML, parseWizardSelection } from './wizard'

describe('buildWizardHTML', () => {
  it('contains three tier cards', () => {
    const html = buildWizardHTML(['ionmonger_benchmark.yaml', 'cigs_baseline.yaml'])
    expect(html).toContain('data-tier="legacy"')
    expect(html).toContain('data-tier="fast"')
    expect(html).toContain('data-tier="full"')
  })

  it('lists the supplied preset filenames in the preset picker', () => {
    const html = buildWizardHTML(['foo.yaml', 'bar.yaml'])
    expect(html).toContain('foo.yaml')
    expect(html).toContain('bar.yaml')
  })
})

describe('parseWizardSelection', () => {
  it('extracts tier and preset from a submitted form', () => {
    const el = document.createElement('div')
    el.innerHTML = buildWizardHTML(['a.yaml', 'b.yaml'])
    // Simulate user selection
    el.querySelector<HTMLInputElement>('input[name="wizard-tier"][value="legacy"]')!.checked = true
    el.querySelector<HTMLSelectElement>('select[name="wizard-preset"]')!.value = 'b.yaml'
    const sel = parseWizardSelection(el)
    expect(sel).toEqual({ tier: 'legacy', preset: 'b.yaml', name: expect.any(String) })
  })
})
```

- [ ] **Step 2: Implement**

Create `wizard.ts`:

```typescript
import type { SimulationModeName } from '../types'

export interface WizardSelection {
  tier: SimulationModeName
  preset: string
  name: string
}

const TIER_CARDS: Array<{
  tier: SimulationModeName
  title: string
  subtitle: string
  bullets: string[]
}> = [
  { tier: 'legacy', title: 'Legacy', subtitle: 'IonMonger-compatible',
    bullets: ['Beer–Lambert optics', 'Single ion species', 'Uniform τ', 'T = 300 K'] },
  { tier: 'fast', title: 'Fast', subtitle: 'Same physics, fast path (today)',
    bullets: ['Beer–Lambert', 'Single ion species', 'Uniform τ', 'T = 300 K'] },
  { tier: 'full', title: 'Full', subtitle: 'All Phase 1–4 upgrades',
    bullets: ['TMM optics', 'Band-offset TE', 'Dual-species ions', 'Trap profile · T-scaling'] },
]

export function buildWizardHTML(presets: ReadonlyArray<string>): string {
  const cards = TIER_CARDS.map(c => `
    <label class="wizard-card" data-tier="${c.tier}">
      <input type="radio" name="wizard-tier" value="${c.tier}"${c.tier === 'full' ? ' checked' : ''} />
      <div class="wizard-card-title">${c.title}</div>
      <div class="wizard-card-subtitle">${c.subtitle}</div>
      <ul class="wizard-card-bullets">${c.bullets.map(b => `<li>${b}</li>`).join('')}</ul>
    </label>`).join('')

  const options = presets.map(p => `<option value="${p}">${p}</option>`).join('')

  return `
    <div class="wizard-modal-backdrop">
      <div class="wizard-modal">
        <h2>New Device</h2>
        <p class="wizard-subtitle">Pick a physics tier and a preset to start from.</p>
        <div class="wizard-tier-row">${cards}</div>
        <div class="wizard-form-row">
          <label>
            <span>Device name</span>
            <input type="text" name="wizard-name" value="New device" />
          </label>
          <label>
            <span>Preset</span>
            <select name="wizard-preset">${options}</select>
          </label>
        </div>
        <div class="wizard-actions">
          <button type="button" class="btn" data-wizard="cancel">Cancel</button>
          <button type="button" class="btn btn-primary" data-wizard="create">Create</button>
        </div>
      </div>
    </div>`
}

export function parseWizardSelection(root: HTMLElement): WizardSelection | null {
  const tierEl = root.querySelector<HTMLInputElement>('input[name="wizard-tier"]:checked')
  const presetEl = root.querySelector<HTMLSelectElement>('select[name="wizard-preset"]')
  const nameEl = root.querySelector<HTMLInputElement>('input[name="wizard-name"]')
  if (!tierEl || !presetEl || !nameEl) return null
  const tier = tierEl.value as SimulationModeName
  return { tier, preset: presetEl.value, name: nameEl.value.trim() || 'New device' }
}

export interface WizardResult {
  cancelled: boolean
  selection: WizardSelection | null
}

/**
 * Show the wizard as a modal, resolve when the user clicks Create or Cancel.
 * DOM side-effect only — the caller owns creating the Device from the selection.
 */
export function showWizard(
  root: HTMLElement,
  presets: ReadonlyArray<string>,
): Promise<WizardResult> {
  return new Promise((resolve) => {
    const host = document.createElement('div')
    host.innerHTML = buildWizardHTML(presets)
    root.appendChild(host)

    const modal = host.querySelector<HTMLElement>('.wizard-modal-backdrop')!

    function close(result: WizardResult): void {
      modal.remove()
      resolve(result)
    }

    modal.querySelector<HTMLButtonElement>('[data-wizard="cancel"]')!
      .addEventListener('click', () => close({ cancelled: true, selection: null }))

    modal.querySelector<HTMLButtonElement>('[data-wizard="create"]')!
      .addEventListener('click', () => {
        const sel = parseWizardSelection(modal)
        if (!sel) return
        close({ cancelled: false, selection: sel })
      })
  })
}
```

- [ ] **Step 3: Run tests**

Run: `cd frontend && npm test -- --run wizard.test.ts`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/workstation/wizard.ts frontend/src/workstation/wizard.test.ts
git commit -m "feat(workstation): New Device wizard"
```

---

## Task 12: Tier-gate the config editor

**Files:**
- Modify: `frontend/src/config-editor.ts`
- Modify: `frontend/src/device-panel.ts`

- [ ] **Step 1: Add a `tier` option to `MountOptions`**

Edit `config-editor.ts` — find the `MountOptions` / mount signature and add:

```typescript
export interface MountConfigEditorOptions {
  // ... existing
  tier?: SimulationModeName
}
```

Pass the tier down to the layer-field rendering code. Where fields are looped (the `LAYER_GROUPS.forEach` or equivalent), wrap each field with:

```typescript
import { isFieldVisible } from './workstation/tier-gating'
// ...
if (tier && !isFieldVisible(field.key, tier)) continue
```

Do the same at the device-level fields (`T` in particular).

- [ ] **Step 2: Forward the option from device-panel**

Edit `device-panel.ts` — `mountDevicePanel` accepts `tier` and passes it through to `mountConfigEditor`.

- [ ] **Step 3: Build + typecheck**

Run: `cd frontend && npm run build`
Expected: clean build.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/config-editor.ts frontend/src/device-panel.ts
git commit -m "feat(config-editor): tier-gated field visibility"
```

---

## Task 13: Shell wiring — register new panes, wire tree, drop legacy

**Files:**
- Modify: `frontend/src/workstation/shell.ts`
- Modify: `frontend/src/workstation/panes/device-pane.ts` (forward tier)
- Delete: `frontend/src/workstation/panes/legacy-experiment-pane.ts`

- [ ] **Step 1: Update `DEFAULT_LAYOUT`**

Replace the `legacy-experiments` component with a row containing a column of (jv stack) and a main-plot pane:

```typescript
const DEFAULT_LAYOUT: LayoutConfig = {
  root: {
    type: 'row',
    content: [
      { type: 'stack', width: 30, content: [
        { type: 'component', componentType: 'device', title: 'Device' },
        { type: 'component', componentType: 'help', title: 'Help' },
      ]},
      { type: 'stack', width: 35, content: [
        { type: 'component', componentType: 'jv', title: 'J–V Sweep' },
        { type: 'component', componentType: 'impedance', title: 'Impedance' },
        { type: 'component', componentType: 'degradation', title: 'Degradation' },
      ]},
      { type: 'component', componentType: 'main-plot', title: 'Main Plot', width: 35 },
    ],
  },
}
```

- [ ] **Step 2: Register factories**

Register `jv`, `impedance`, `degradation`, `main-plot`. Remove `legacy-experiments` registration.

```typescript
layout.registerComponentFactoryFunction('jv', (container) => {
  mountJVPane(container.element, {
    getActiveDevice: () => activeDeviceAccessor(),
    onRunComplete: (deviceId, run) => commitRun(deviceId, 'jv', run),
  })
})
// same for impedance and degradation
let mainPlot: MainPlotHandle | null = null
layout.registerComponentFactoryFunction('main-plot', (container) => {
  mainPlot = mountMainPlotPane(container.element)
  mainPlot.update(workspace)
})
```

- [ ] **Step 3: `activeDeviceAccessor` + `commitRun` helpers**

```typescript
function activeDeviceAccessor() {
  const id = workspace.activeDeviceId
  const d = id ? workspace.devices.find(x => x.id === id) : null
  return d ? { id: d.id, config: d.config } : null
}

function ensureExperiment(deviceId: string, kind: ExperimentKind): string {
  const dev = workspace.devices.find(d => d.id === deviceId)
  if (!dev) throw new Error('device missing')
  const existing = dev.experiments.find(e => e.kind === kind)
  if (existing) return existing.id
  const id = 'e-' + Math.random().toString(36).slice(2, 10)
  workspace = addExperiment(workspace, deviceId, { id, kind, params: {}, runs: [] })
  return id
}

function commitRun(deviceId: string, kind: ExperimentKind, run: Run): void {
  const expId = ensureExperiment(deviceId, kind)
  workspace = addRun(workspace, deviceId, expId, run)
  workspace = setActiveRun(workspace, deviceId, expId, run.id)
  saveWorkspace(workspace)
  refreshTree()
  mainPlot?.update(workspace)
  consoleHandle.log(`run complete: ${kind}  (${run.activePhysics})`)
}
```

- [ ] **Step 4: Wire hierarchical tree click dispatch**

```typescript
attachTreeHandlers(treeEl, {
  onSelectDevice: (id) => {
    workspace = setActiveDevice(workspace, id)
    saveWorkspace(workspace)
    refreshTree()
    const active = workspace.devices.find(d => d.id === id)
    if (active) consoleHandle.setPhysics(tierLabel(active.tier), physicsSummary(active.tier))
  },
  onSelectExperiment: (deviceId, experimentId) => {
    workspace = setActiveExperiment(workspace, deviceId, experimentId)
    saveWorkspace(workspace)
    refreshTree()
    // Focus the matching pane
    focusComponent(layout, experimentKindOf(workspace, experimentId))
  },
  onSelectRun: (deviceId, experimentId, runId) => {
    workspace = setActiveRun(workspace, deviceId, experimentId, runId)
    saveWorkspace(workspace)
    refreshTree()
    mainPlot?.update(workspace)
    focusComponent(layout, 'main-plot')
  },
})
```

`focusComponent` helper:

```typescript
function focusComponent(layout: GoldenLayout, componentType: string): void {
  const items = layout.rootItem?.getItemsByFilter(it => it.isComponent && (it as any).componentType === componentType) ?? []
  if (items.length > 0) (items[0] as any).focus()
}

function experimentKindOf(ws: Workspace, experimentId: string): ExperimentKind {
  for (const d of ws.devices) {
    const e = d.experiments.find(e => e.id === experimentId)
    if (e) return e.kind
  }
  return 'jv'
}
```

- [ ] **Step 5: Replace seed-with-hardcoded device with wizard flow**

Replace `seedDefaultDevice` + its unconditional `addDevice` with:

```typescript
if (workspace.devices.length === 0) {
  const configs = await listConfigs() // new api.ts wrapper
  const result = await showWizard(root, configs)
  if (!result.cancelled && result.selection) {
    const cfg = await getConfig(result.selection.preset)
    cfg.device.mode = result.selection.tier
    const dev: Device = {
      id: 'd-' + Math.random().toString(36).slice(2, 10),
      name: result.selection.name,
      tier: result.selection.tier,
      config: cfg,
      experiments: [],
    }
    workspace = addDevice(workspace, dev)
    saveWorkspace(workspace)
  }
}
```

Add `listConfigs` to `api.ts` if it doesn't already exist:

```typescript
export async function listConfigs(): Promise<string[]> {
  const res = await fetch(`${API_BASE}/api/configs`)
  if (!res.ok) throw new Error(`listConfigs ${res.status}`)
  return res.json() as Promise<string[]>
}
```

- [ ] **Step 6: Forward tier into the device pane**

Edit `device-pane.ts` — accept `tier` argument (read from active device) and pass it to `mountDevicePanel`:

```typescript
export function mountDevicePane(el: HTMLElement, _key: string, tier: SimulationModeName): Promise<DevicePanel> {
  return mountDevicePanel(el, _key, { tier })
}
```

Call site in `shell.ts` factory:

```typescript
layout.registerComponentFactoryFunction('device', (container) => {
  const active = workspace.devices.find(d => d.id === workspace.activeDeviceId)
  void mountDevicePane(container.element, 'ws-device', active?.tier ?? 'full')
})
```

- [ ] **Step 7: Delete the legacy fallback pane**

```bash
git rm frontend/src/workstation/panes/legacy-experiment-pane.ts
```

Remove any imports of it from `shell.ts`.

- [ ] **Step 8: Build + run tests**

Run: `cd frontend && npm run build && npm test -- --run`
Expected: clean build, all tests green.

- [ ] **Step 9: Commit**

```bash
git add frontend/src/workstation/shell.ts frontend/src/workstation/panes/device-pane.ts frontend/src/api.ts
git commit -m "feat(workstation): wire hierarchical tree, new panes, wizard, drop legacy fallback"
```

---

## Task 14: Style the wizard, hierarchical tree, and main plot

**Files:**
- Modify: `frontend/src/style.css`

- [ ] **Step 1: Add CSS for new elements**

Append to the existing workstation section:

```css
/* hierarchical tree children */
.tree-children {
  margin-left: 1.25rem;
  border-left: 1px solid var(--border);
  padding-left: 0.5rem;
}
.tree-node-experiment,
.tree-node-run {
  font-size: 0.85rem;
}
.tree-node-run .tree-icon { color: var(--primary); }

/* main plot pane */
.main-plot-pane { display: flex; flex-direction: column; height: 100%; padding: 0.75rem; }
.main-plot-header {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.8rem;
  color: var(--text-muted);
  padding-bottom: 0.5rem;
}
.plot-empty {
  color: var(--text-muted);
  text-align: center;
  padding: 2rem 0;
  font-style: italic;
}

/* wizard */
.wizard-modal-backdrop {
  position: fixed; inset: 0;
  background: rgba(15, 23, 42, 0.45);
  display: flex; align-items: center; justify-content: center;
  z-index: 9999;
}
.wizard-modal {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 2rem;
  max-width: 900px;
  width: 90%;
  box-shadow: 0 30px 60px rgba(15, 23, 42, 0.25);
}
.wizard-modal h2 { margin: 0 0 0.25rem 0; }
.wizard-subtitle { color: var(--text-muted); margin: 0 0 1.5rem 0; }
.wizard-tier-row {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1rem;
  margin-bottom: 1.5rem;
}
.wizard-card {
  border: 2px solid var(--border);
  border-radius: 10px;
  padding: 1.25rem;
  cursor: pointer;
  display: flex; flex-direction: column;
  transition: border-color 0.15s, background 0.15s;
}
.wizard-card:has(input:checked) {
  border-color: var(--primary);
  background: color-mix(in srgb, var(--primary) 8%, transparent);
}
.wizard-card input { display: none; }
.wizard-card-title { font-size: 1.2rem; font-weight: 600; }
.wizard-card-subtitle { color: var(--text-muted); font-size: 0.85rem; margin-bottom: 0.75rem; }
.wizard-card-bullets { margin: 0; padding-left: 1.2rem; font-size: 0.85rem; }
.wizard-form-row { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.25rem; }
.wizard-form-row label { display: flex; flex-direction: column; gap: 0.25rem; }
.wizard-form-row input, .wizard-form-row select {
  padding: 0.5rem; border: 1px solid var(--border); border-radius: 6px; background: var(--bg);
}
.wizard-actions { display: flex; justify-content: flex-end; gap: 0.75rem; }

/* tree-node-active now also applies to experiment + run rows */
.tree-node-run.tree-node-active {
  background: color-mix(in srgb, var(--primary) 18%, transparent);
  color: var(--primary);
}

/* pane hint */
.pane-hint {
  color: var(--text-muted);
  font-size: 0.8rem;
  font-style: italic;
  padding-top: 0.5rem;
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/style.css
git commit -m "style(workstation): wizard modal, hierarchical tree, main plot"
```

---

## Task 15: Smoke test + build verification

**Files:**
- Create: `frontend/src/workstation/phase2-smoke.test.ts`

- [ ] **Step 1: Smoke test — jsdom reconstruction of the headline flow**

Create `phase2-smoke.test.ts`:

```typescript
import { describe, it, expect } from 'vitest'
import {
  createEmptyWorkspace,
  addDevice,
  addExperiment,
  addRun,
  setActiveRun,
  setActiveExperiment,
  findRun,
} from './state'
import type { Device, Run } from './types'
import { renderTreeHTML } from './tree'

function device(id: string, tier: 'legacy' | 'fast' | 'full' = 'full'): Device {
  return {
    id, name: `dev-${id}`, tier,
    config: { device: { V_bi: 1.1, Phi: 1.4e21 }, layers: [] },
    experiments: [],
  }
}

function jvRun(id: string): Run {
  return {
    id, timestamp: Date.now(),
    result: { kind: 'jv', data: {
      V_fwd: [0, 0.5, 1], J_fwd: [0, 50, 200],
      V_rev: [1, 0.5, 0], J_rev: [200, 50, 0],
      metrics_fwd: { V_oc: 1.08, J_sc: 200, FF: 0.75, PCE: 0.162 },
      metrics_rev: { V_oc: 1.09, J_sc: 200, FF: 0.76, PCE: 0.163 },
      hysteresis_index: 0.01,
    } },
    activePhysics: 'FULL',
    durationMs: 1234,
    deviceSnapshot: { device: { V_bi: 1.1, Phi: 1.4e21 }, layers: [] },
  }
}

describe('Phase 2 headline: create device → run experiment → run persists → re-activate', () => {
  it('end-to-end state transitions', () => {
    let ws = createEmptyWorkspace('W')
    ws = addDevice(ws, device('d1', 'full'))
    ws = addExperiment(ws, 'd1', { id: 'e1', kind: 'jv', params: {}, runs: [] })
    ws = addRun(ws, 'd1', 'e1', jvRun('r1'))
    ws = setActiveRun(ws, 'd1', 'e1', 'r1')
    expect(ws.activeRunId).toBe('r1')
    expect(findRun(ws, 'd1', 'e1', 'r1')?.id).toBe('r1')

    const html = renderTreeHTML(ws)
    expect(html).toContain('data-run-id="r1"')
    expect(html).toContain('data-experiment-id="e1"')
    expect(html).toContain('FULL')
  })

  it('setActiveExperiment clears activeRunId', () => {
    let ws = createEmptyWorkspace('W')
    ws = addDevice(ws, device('d1'))
    ws = addExperiment(ws, 'd1', { id: 'e1', kind: 'jv', params: {}, runs: [] })
    ws = addRun(ws, 'd1', 'e1', jvRun('r1'))
    ws = setActiveRun(ws, 'd1', 'e1', 'r1')
    ws = setActiveExperiment(ws, 'd1', 'e1')
    expect(ws.activeRunId).toBeNull()
  })
})
```

- [ ] **Step 2: Run full suite + build**

Run: `cd frontend && npm test -- --run && npm run build`
Expected: all tests green, clean build.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/workstation/phase2-smoke.test.ts
git commit -m "test(workstation): Phase 2 headline smoke test"
```

---

## Verification Checklist

After all 15 tasks complete, manually verify in the browser:

- [ ] Fresh load (clear localStorage) → wizard appears
- [ ] Create a Full-tier device from `ionmonger_benchmark.yaml` → appears in tree with FULL badge
- [ ] JV pane appears in dock → click Run → progress bar → main plot populates → run node appears in tree
- [ ] Click the run node again after running another one → main plot switches to the older run
- [ ] Impedance and Degradation panes work the same way
- [ ] Delete the Device pane in the dock → re-add it via Golden Layout → state restored
- [ ] Change active device via tree click → solver console tier label updates
- [ ] Create a Legacy device → config editor does NOT show `T` field (tier-gated)
- [ ] Reload → workspace (with all runs) persists

## Out of Scope (deferred)

- Dual-ion / TMM / trap-profile *input fields* in the config editor — added when needed; Task 4 ships the gating mechanism, not the new fields
- Compare Runs pane — Phase 4
- Device Schematic pane — Phase 3
- Parameter Inspector as a separate pane — Phase 3 (merged today into config editor + device pane)
