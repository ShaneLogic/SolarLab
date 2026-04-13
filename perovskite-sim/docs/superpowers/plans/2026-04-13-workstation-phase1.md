# Workstation UI Redesign — Phase 1 (Foundation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 6-tab SPA shell with a Golden Layout v2 workstation (project tree + dockable panes + solver console) while keeping every existing experiment reachable through a legacy fallback, so the app never goes dark during migration.

**Architecture:** New `frontend/src/workstation/` module owns the shell, tree, panes, and workspace state. Existing panels and the config editor are wrapped (not rewritten) so Phase 1 is purely additive. Workspace state (including Golden Layout's own serialised layout) lives in `localStorage`. Backend gains a 3-line `active_physics` field on the SSE `result` payload so the console can show which physics tier just ran.

**Tech Stack:** Vite + TypeScript (plain TS, no framework), Golden Layout v2 (`golden-layout`), Plotly (unchanged), Vitest + jsdom (new, for testable pure logic), FastAPI backend (unchanged except the 3-line `active_physics` addition).

**Spec:** `perovskite-sim/docs/superpowers/specs/2026-04-13-workstation-ui-redesign-design.md`

---

## File Structure

### New files

| Path | Responsibility |
|---|---|
| `frontend/src/workstation/types.ts` | `Workspace`, `Device`, `Experiment`, `Run`, `CompareView` types (Phase 1 uses only `Workspace`, `Device` — other types defined for forward-compat) |
| `frontend/src/workstation/state.ts` | Pure functions: `createEmptyWorkspace`, `saveWorkspace`, `loadWorkspace`, `addDevice`, `removeDevice`, `setActiveDevice` |
| `frontend/src/workstation/state.test.ts` | Vitest unit tests for `state.ts` |
| `frontend/src/workstation/tree.ts` | Pure function `renderTreeHTML(ws)` + `attachTreeHandlers(container, onSelect)` |
| `frontend/src/workstation/tree.test.ts` | Vitest unit tests for `tree.ts` |
| `frontend/src/workstation/console.ts` | `createConsole(container)` returning `{ setPhysics, log }` — bottom solver console strip |
| `frontend/src/workstation/panes/device-pane.ts` | Golden Layout component that hosts the existing `config-editor.ts` |
| `frontend/src/workstation/panes/help-pane.ts` | Golden Layout component with Tutorial / Parameters / Algorithm as in-pane tabs |
| `frontend/src/workstation/panes/legacy-experiment-pane.ts` | Golden Layout component with the existing J-V / Impedance / Degradation panels as in-pane tabs |
| `frontend/src/workstation/shell.ts` | Entry point: wires Golden Layout, sidebar tree, console strip, and initial layout |
| `frontend/vitest.config.ts` | Vitest config with jsdom environment |

### Modified files

| Path | Change |
|---|---|
| `frontend/package.json` | Add `golden-layout@^2`, `vitest`, `@vitest/ui`, `jsdom` dev deps; add `test` + `test:run` scripts |
| `frontend/src/main.ts` | Replace the tab-strip bootstrap with `mountWorkstation()` from `shell.ts` |
| `frontend/src/style.css` | Append workstation styles (left sidebar, dock host, console strip, tier badges) |
| `frontend/src/panels/tutorial.ts` | Export `tutorialHTML(): string` alongside existing `mountTutorialPanel` |
| `frontend/src/panels/parameters.ts` | Export `parametersHTML(): string` alongside existing `mountParametersPanel` |
| `frontend/src/panels/algorithm.ts` | Export `algorithmHTML(): string` alongside existing `mountAlgorithmPanel` |
| `backend/main.py` | Add `_describe_active_physics(stack)` helper; each `_run` closure injects `out["active_physics"] = ...` before returning |
| `tests/unit/backend/test_active_physics.py` (new) | pytest for `_describe_active_physics` + smoke test that the job result dict contains the field |

**Invariants observed:**
- Existing `config-editor.ts`, `device-panel.ts`, `panels/{jv,impedance,degradation}.ts`, `job-stream.ts`, `api.ts`, `ui-helpers.ts`, `progress.ts` are **not modified**. Wrapping only. Any change to those files is out of scope for Phase 1.
- `DeviceConfig`, `SimulationModeName`, `JVResult`, etc. from `frontend/src/types.ts` are imported as-is; the new workstation types live separately in `workstation/types.ts`.
- Backend public endpoint contracts are unchanged (the new `active_physics` field is additive).

---

## Task 1: Bootstrap Vitest

**Files:**
- Modify: `frontend/package.json`
- Create: `frontend/vitest.config.ts`
- Create: `frontend/src/workstation/smoke.test.ts`

- [ ] **Step 1: Install vitest + jsdom**

Run from `frontend/`:
```bash
npm install --save-dev vitest@^1 @vitest/ui@^1 jsdom@^24
```

Expected: `package.json` gains three devDependencies, `node_modules/` updated. No runtime output expected beyond npm's install summary.

- [ ] **Step 2: Add test scripts to `package.json`**

Modify `frontend/package.json` `"scripts"` section:

```json
"scripts": {
  "dev": "vite",
  "build": "tsc && vite build",
  "preview": "vite preview",
  "test": "vitest",
  "test:run": "vitest run"
}
```

- [ ] **Step 3: Create `frontend/vitest.config.ts`**

```typescript
import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    environment: 'jsdom',
    include: ['src/**/*.test.ts'],
    globals: false,
  },
})
```

- [ ] **Step 4: Write a smoke test to prove vitest runs**

Create `frontend/src/workstation/smoke.test.ts`:

```typescript
import { describe, it, expect } from 'vitest'

describe('vitest infrastructure', () => {
  it('runs in jsdom', () => {
    const div = document.createElement('div')
    div.textContent = 'hello'
    expect(div.textContent).toBe('hello')
  })
})
```

- [ ] **Step 5: Run the smoke test — it must pass**

Run from `frontend/`:
```bash
npm run test:run
```

Expected: `1 passed (1)`, exit code 0.

- [ ] **Step 6: Commit**

```bash
git add frontend/package.json frontend/package-lock.json frontend/vitest.config.ts frontend/src/workstation/smoke.test.ts
git commit -m "test(frontend): bootstrap vitest + jsdom

Adds vitest test runner with jsdom environment so pure logic
modules in frontend/src/workstation/ can be unit-tested during
the workstation migration.

Confidence: high
Scope-risk: narrow"
```

---

## Task 2: Install Golden Layout v2

**Files:**
- Modify: `frontend/package.json`

- [ ] **Step 1: Install the dependency**

Run from `frontend/`:
```bash
npm install --save golden-layout@^2
```

Expected: `golden-layout` added to `dependencies`, major version 2.x.x.

- [ ] **Step 2: Verify the build still succeeds**

Run from `frontend/`:
```bash
npm run build
```

Expected: `tsc` passes with no errors, `vite build` writes to `dist/`. The 4 MB Plotly bundle warning is expected; Golden Layout adds ~50 kB. No TypeScript errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/package.json frontend/package-lock.json
git commit -m "deps(frontend): add golden-layout v2

Docking library for the workstation shell. Chosen over Dockview
because its vanilla TS API is better documented.

Confidence: high
Scope-risk: narrow"
```

---

## Task 3: Workstation types

**Files:**
- Create: `frontend/src/workstation/types.ts`

Phase 1 only consumes `Workspace` and `Device`. `Experiment`, `Run`, and `CompareView` are defined now so Phase 2+ doesn't have to add fields retroactively. No tests — pure type definitions.

- [ ] **Step 1: Create `frontend/src/workstation/types.ts`**

```typescript
import type { DeviceConfig, SimulationModeName } from '../types'

/** The root object persisted to localStorage. */
export interface Workspace {
  /** Schema version — bump on breaking changes; load() falls back to empty workspace on mismatch. */
  version: 1
  id: string
  name: string
  devices: Device[]
  /** ID of the currently-selected device, or null for "nothing selected". */
  activeDeviceId: string | null
  /**
   * Opaque Golden Layout config blob — serialised state of the dockable area.
   * `unknown` by design: we never inspect it, only round-trip it to Golden Layout.
   */
  layout: unknown | null
}

/** A device node in the project tree. Phase 1 always has exactly one (the seeded default). */
export interface Device {
  id: string
  name: string
  tier: SimulationModeName
  config: DeviceConfig
  /** Empty in Phase 1 — experiments are wired up in Phase 2. */
  experiments: Experiment[]
}

/** Phase 2 will populate this. Defined here to keep the type stable across phases. */
export interface Experiment {
  id: string
  kind: 'jv' | 'impedance' | 'degradation'
  params: Record<string, unknown>
  runs: Run[]
}

/** Phase 2 will populate this. */
export interface Run {
  id: string
  timestamp: number
  result: unknown
  activePhysics: string
  durationMs: number
  /** Frozen DeviceConfig snapshot at the moment the run was dispatched. */
  deviceSnapshot: DeviceConfig
}

/** Phase 4 will populate this. */
export interface CompareView {
  id: string
  name: string
  kind: 'jv' | 'impedance' | 'degradation'
  runRefs: Array<{ deviceId: string; experimentId: string; runId: string }>
}
```

- [ ] **Step 2: Verify build**

Run from `frontend/`:
```bash
npm run build
```

Expected: no TypeScript errors. The file is imported by nothing yet so this only checks self-consistency.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/workstation/types.ts
git commit -m "feat(workstation): add Workspace/Device types

Phase 1 uses Workspace + Device; Experiment/Run/CompareView are
declared now so later phases do not have to rework the type shape.

Confidence: high
Scope-risk: narrow"
```

---

## Task 4: Workspace state module (TDD)

**Files:**
- Create: `frontend/src/workstation/state.ts`
- Create: `frontend/src/workstation/state.test.ts`

Pure functions over `Workspace`. Mutations return new objects (immutable style per the repo's coding rules). `localStorage` access is isolated to two functions.

- [ ] **Step 1: Write the failing tests**

Create `frontend/src/workstation/state.test.ts`:

```typescript
import { describe, it, expect, beforeEach } from 'vitest'
import {
  createEmptyWorkspace,
  addDevice,
  removeDevice,
  setActiveDevice,
  saveWorkspace,
  loadWorkspace,
  STORAGE_KEY,
} from './state'
import type { Device, Workspace } from './types'

function makeDevice(id: string, name = 'Test'): Device {
  return {
    id,
    name,
    tier: 'full',
    config: { device: { V_bi: 1.1, Phi: 1.4e21 }, layers: [] },
    experiments: [],
  }
}

describe('createEmptyWorkspace', () => {
  it('returns a workspace with version 1, no devices, nothing active', () => {
    const ws = createEmptyWorkspace('My Workspace')
    expect(ws.version).toBe(1)
    expect(ws.name).toBe('My Workspace')
    expect(ws.devices).toEqual([])
    expect(ws.activeDeviceId).toBeNull()
    expect(ws.layout).toBeNull()
    expect(typeof ws.id).toBe('string')
    expect(ws.id.length).toBeGreaterThan(0)
  })
})

describe('addDevice', () => {
  it('returns a new workspace with the device appended — original is untouched', () => {
    const ws = createEmptyWorkspace('W')
    const dev = makeDevice('d1')
    const next = addDevice(ws, dev)
    expect(next.devices).toHaveLength(1)
    expect(next.devices[0].id).toBe('d1')
    expect(ws.devices).toHaveLength(0) // immutability check
  })

  it('sets activeDeviceId to the new device when no device was active', () => {
    const ws = createEmptyWorkspace('W')
    const next = addDevice(ws, makeDevice('d1'))
    expect(next.activeDeviceId).toBe('d1')
  })

  it('leaves activeDeviceId alone when a device was already active', () => {
    let ws = createEmptyWorkspace('W')
    ws = addDevice(ws, makeDevice('d1'))
    ws = addDevice(ws, makeDevice('d2'))
    expect(ws.activeDeviceId).toBe('d1')
  })
})

describe('removeDevice', () => {
  it('removes the matching device and returns a new workspace', () => {
    let ws = createEmptyWorkspace('W')
    ws = addDevice(ws, makeDevice('d1'))
    ws = addDevice(ws, makeDevice('d2'))
    const next = removeDevice(ws, 'd1')
    expect(next.devices.map(d => d.id)).toEqual(['d2'])
  })

  it('clears activeDeviceId if the active device was removed', () => {
    let ws = createEmptyWorkspace('W')
    ws = addDevice(ws, makeDevice('d1'))
    const next = removeDevice(ws, 'd1')
    expect(next.activeDeviceId).toBeNull()
  })
})

describe('setActiveDevice', () => {
  it('sets activeDeviceId when the id exists', () => {
    let ws = createEmptyWorkspace('W')
    ws = addDevice(ws, makeDevice('d1'))
    ws = addDevice(ws, makeDevice('d2'))
    const next = setActiveDevice(ws, 'd2')
    expect(next.activeDeviceId).toBe('d2')
  })

  it('returns the same workspace reference when the id does not exist', () => {
    const ws = addDevice(createEmptyWorkspace('W'), makeDevice('d1'))
    const next = setActiveDevice(ws, 'unknown')
    expect(next).toBe(ws)
  })
})

describe('saveWorkspace / loadWorkspace', () => {
  beforeEach(() => {
    localStorage.clear()
  })

  it('roundtrips a workspace through localStorage', () => {
    let ws = createEmptyWorkspace('Roundtrip')
    ws = addDevice(ws, makeDevice('d1', 'Alpha'))
    saveWorkspace(ws)
    const loaded = loadWorkspace()
    expect(loaded).not.toBeNull()
    expect(loaded!.name).toBe('Roundtrip')
    expect(loaded!.devices[0].name).toBe('Alpha')
  })

  it('returns null when nothing is stored', () => {
    expect(loadWorkspace()).toBeNull()
  })

  it('returns null when the stored blob has a different schema version', () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ version: 99 }))
    expect(loadWorkspace()).toBeNull()
  })

  it('returns null when the stored blob is not JSON', () => {
    localStorage.setItem(STORAGE_KEY, 'not json')
    expect(loadWorkspace()).toBeNull()
  })
})
```

- [ ] **Step 2: Run the test — verify it fails with "module not found"**

Run:
```bash
cd frontend && npm run test:run state.test
```

Expected: vitest reports "Failed to resolve import './state'" or similar — module does not exist yet.

- [ ] **Step 3: Implement `frontend/src/workstation/state.ts`**

```typescript
import type { Device, Workspace } from './types'

export const STORAGE_KEY = 'solarsim:workspace:v1'

function randomId(): string {
  // Sufficient uniqueness for a single-user local workspace; not a crypto concern.
  return Math.random().toString(36).slice(2, 10) + Date.now().toString(36)
}

export function createEmptyWorkspace(name: string): Workspace {
  return {
    version: 1,
    id: randomId(),
    name,
    devices: [],
    activeDeviceId: null,
    layout: null,
  }
}

export function addDevice(ws: Workspace, device: Device): Workspace {
  return {
    ...ws,
    devices: [...ws.devices, device],
    activeDeviceId: ws.activeDeviceId ?? device.id,
  }
}

export function removeDevice(ws: Workspace, deviceId: string): Workspace {
  const devices = ws.devices.filter(d => d.id !== deviceId)
  const activeDeviceId = ws.activeDeviceId === deviceId ? null : ws.activeDeviceId
  return { ...ws, devices, activeDeviceId }
}

export function setActiveDevice(ws: Workspace, deviceId: string): Workspace {
  if (!ws.devices.some(d => d.id === deviceId)) return ws
  if (ws.activeDeviceId === deviceId) return ws
  return { ...ws, activeDeviceId: deviceId }
}

export function saveWorkspace(ws: Workspace): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(ws))
  } catch (e) {
    // localStorage can throw on quota exceeded; failing to persist should not crash the UI.
    console.error('saveWorkspace failed:', e)
  }
}

export function loadWorkspace(): Workspace | null {
  const raw = localStorage.getItem(STORAGE_KEY)
  if (!raw) return null
  try {
    const parsed = JSON.parse(raw) as Partial<Workspace>
    if (parsed.version !== 1) return null
    // Minimal structural check — later phases can add migrations here.
    if (!Array.isArray(parsed.devices)) return null
    return parsed as Workspace
  } catch {
    return null
  }
}
```

- [ ] **Step 4: Run the tests — they must all pass**

Run:
```bash
cd frontend && npm run test:run state.test
```

Expected: all tests in `state.test.ts` pass (11 tests).

- [ ] **Step 5: Commit**

```bash
git add frontend/src/workstation/state.ts frontend/src/workstation/state.test.ts
git commit -m "feat(workstation): add workspace state module

Pure immutable operations over Workspace plus localStorage
persistence with schema versioning. All mutations return new
workspace objects; load returns null on version mismatch or
corrupt JSON.

Confidence: high
Scope-risk: narrow"
```

---

## Task 5: Project tree rendering (TDD)

**Files:**
- Create: `frontend/src/workstation/tree.ts`
- Create: `frontend/src/workstation/tree.test.ts`

The tree is rendered as inline HTML into a host element. Separating `renderTreeHTML` (pure, testable) from `attachTreeHandlers` (DOM event wiring) lets us unit-test the HTML without a real browser.

- [ ] **Step 1: Write the failing tests**

Create `frontend/src/workstation/tree.test.ts`:

```typescript
import { describe, it, expect, beforeEach } from 'vitest'
import { renderTreeHTML, attachTreeHandlers } from './tree'
import { createEmptyWorkspace, addDevice } from './state'
import type { Device } from './types'

function makeDevice(id: string, name: string, tier: 'legacy' | 'fast' | 'full' = 'full'): Device {
  return {
    id,
    name,
    tier,
    config: { device: { V_bi: 1.1, Phi: 1.4e21 }, layers: [] },
    experiments: [],
  }
}

describe('renderTreeHTML', () => {
  it('renders the Devices folder header even when empty', () => {
    const ws = createEmptyWorkspace('W')
    const html = renderTreeHTML(ws)
    expect(html).toContain('Devices')
    expect(html).toContain('Results')
  })

  it('renders each device with its tier badge', () => {
    let ws = createEmptyWorkspace('W')
    ws = addDevice(ws, makeDevice('d1', 'MAPbI3', 'full'))
    ws = addDevice(ws, makeDevice('d2', 'CIGS', 'legacy'))
    const html = renderTreeHTML(ws)
    expect(html).toContain('MAPbI3')
    expect(html).toContain('CIGS')
    expect(html).toContain('FULL')
    expect(html).toContain('LEGACY')
  })

  it('marks the active device with an active class', () => {
    let ws = createEmptyWorkspace('W')
    ws = addDevice(ws, makeDevice('d1', 'Alpha'))
    const html = renderTreeHTML(ws)
    expect(html).toContain('data-device-id="d1"')
    expect(html).toContain('tree-node-active')
  })

  it('escapes device names with HTML special characters', () => {
    let ws = createEmptyWorkspace('W')
    ws = addDevice(ws, makeDevice('d1', '<script>x</script>'))
    const html = renderTreeHTML(ws)
    expect(html).not.toContain('<script>x</script>')
    expect(html).toContain('&lt;script&gt;')
  })
})

describe('attachTreeHandlers', () => {
  let container: HTMLDivElement

  beforeEach(() => {
    container = document.createElement('div')
    document.body.appendChild(container)
  })

  it('fires onSelectDevice when a device node is clicked', () => {
    let ws = createEmptyWorkspace('W')
    ws = addDevice(ws, makeDevice('d1', 'Alpha'))
    container.innerHTML = renderTreeHTML(ws)
    const received: string[] = []
    attachTreeHandlers(container, { onSelectDevice: (id) => received.push(id) })
    const node = container.querySelector<HTMLElement>('[data-device-id="d1"]')!
    node.click()
    expect(received).toEqual(['d1'])
  })
})
```

- [ ] **Step 2: Run the test — verify it fails**

Run:
```bash
cd frontend && npm run test:run tree.test
```

Expected: "Failed to resolve import './tree'".

- [ ] **Step 3: Implement `frontend/src/workstation/tree.ts`**

```typescript
import type { Workspace } from './types'

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}

function tierBadge(tier: 'legacy' | 'fast' | 'full'): string {
  const label = tier.toUpperCase()
  return `<span class="tier-badge tier-badge-${tier}">${label}</span>`
}

export function renderTreeHTML(ws: Workspace): string {
  const deviceNodes = ws.devices
    .map(d => {
      const active = d.id === ws.activeDeviceId ? ' tree-node-active' : ''
      return `
        <div class="tree-node tree-node-device${active}" data-device-id="${escapeHtml(d.id)}">
          <span class="tree-icon">🔬</span>
          <span class="tree-label">${escapeHtml(d.name)}</span>
          ${tierBadge(d.tier)}
        </div>`
    })
    .join('')

  return `
    <div class="tree-section">
      <div class="tree-section-header">📁 Devices</div>
      <div class="tree-section-body">${deviceNodes || '<div class="tree-empty">(no devices yet)</div>'}</div>
    </div>
    <div class="tree-section">
      <div class="tree-section-header">📁 Results / Compare</div>
      <div class="tree-section-body"><div class="tree-empty">(Phase 4)</div></div>
    </div>`
}

export interface TreeHandlers {
  onSelectDevice: (deviceId: string) => void
}

export function attachTreeHandlers(container: HTMLElement, handlers: TreeHandlers): void {
  container.addEventListener('click', (e) => {
    const target = e.target as HTMLElement
    const node = target.closest<HTMLElement>('[data-device-id]')
    if (node) {
      const id = node.dataset.deviceId
      if (id) handlers.onSelectDevice(id)
    }
  })
}
```

- [ ] **Step 4: Run the tests — they must pass**

Run:
```bash
cd frontend && npm run test:run tree.test
```

Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/workstation/tree.ts frontend/src/workstation/tree.test.ts
git commit -m "feat(workstation): add project tree rendering

Pure renderTreeHTML(workspace) + attachTreeHandlers for click
delegation. HTML-escapes device names. Tier badges carry tier
class names so CSS can colour them in task 13.

Confidence: high
Scope-risk: narrow"
```

---

## Task 6: Expose HTML builders from doc panels

**Files:**
- Modify: `frontend/src/panels/tutorial.ts`
- Modify: `frontend/src/panels/parameters.ts`
- Modify: `frontend/src/panels/algorithm.ts`

The existing panels put their HTML into `el.innerHTML` inside an `async mountXPanel` function. The Help pane (task 8) needs the HTML string without forcing a DOM mount. Additive refactor: extract the HTML into a pure function and have the existing mount function call it.

- [ ] **Step 1: Refactor `frontend/src/panels/tutorial.ts`**

The existing file wraps a large HTML string assignment inside `mountTutorialPanel`. Extract it to a builder:

```typescript
export function tutorialHTML(): string {
  return `
  <div class="card doc-card">
    <h3>Getting Started</h3>
    <div class="doc-body">
      <!-- ... existing body unchanged ... -->
    </div>
  </div>`
}

export async function mountTutorialPanel(el: HTMLElement): Promise<void> {
  el.innerHTML = tutorialHTML()
}
```

The full body stays byte-identical to the current file — only the outer wrapper moves.

- [ ] **Step 2: Refactor `frontend/src/panels/parameters.ts` the same way**

```typescript
export function parametersHTML(): string {
  return `
  <div class="card doc-card">
    <h3>Physical Quantities &amp; Parameters</h3>
    <div class="doc-body">
      <!-- ... existing body unchanged ... -->
    </div>
  </div>`
}

export async function mountParametersPanel(el: HTMLElement): Promise<void> {
  el.innerHTML = parametersHTML()
}
```

- [ ] **Step 3: Refactor `frontend/src/panels/algorithm.ts` the same way**

```typescript
export function algorithmHTML(): string {
  return `
  <div class="card doc-card">
    <!-- ... existing body unchanged ... -->
  </div>`
}

export async function mountAlgorithmPanel(el: HTMLElement): Promise<void> {
  el.innerHTML = algorithmHTML()
}
```

- [ ] **Step 4: Verify `npm run build` still passes and `npm run dev` still renders all three tabs identically**

Run:
```bash
cd frontend && npm run build
```

Expected: tsc passes; vite bundles. This task must not change the rendered output of the existing tabs — it is a pure refactor.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/panels/tutorial.ts frontend/src/panels/parameters.ts frontend/src/panels/algorithm.ts
git commit -m "refactor(panels): expose HTML builders for doc panels

Pulls the inline HTML out of mountXPanel into pure *HTML()
functions so the workstation Help pane can reuse them without
forcing a DOM mount. The legacy mount entry points still work
byte-identically.

Confidence: high
Scope-risk: narrow"
```

---

## Task 7: Device pane (Golden Layout component)

**Files:**
- Create: `frontend/src/workstation/panes/device-pane.ts`

The Device pane is a thin wrapper: it creates a host element, mounts the existing `mountDevicePanel` from `device-panel.ts` into it, and exposes a `mount(container)` function that Golden Layout can register as a component factory.

- [ ] **Step 1: Create `frontend/src/workstation/panes/device-pane.ts`**

```typescript
import { mountDevicePanel, type DevicePanel } from '../../device-panel'

export interface DevicePaneHandle {
  panel: DevicePanel
}

/**
 * Build the Device pane contents into the given container.
 *
 * The container is provided by Golden Layout (or any host). We simply
 * delegate to the existing `mountDevicePanel` so Phase 1 inherits all
 * of its behaviour (preset dropdown, per-layer editor, reset button)
 * without a rewrite.
 */
export async function mountDevicePane(
  container: HTMLElement,
  tabId: string,
): Promise<DevicePaneHandle> {
  container.classList.add('pane', 'pane-device')
  const inner = document.createElement('div')
  inner.className = 'pane-body'
  container.appendChild(inner)
  const panel = await mountDevicePanel(inner, tabId)
  return { panel }
}
```

- [ ] **Step 2: Verify build**

Run:
```bash
cd frontend && npm run build
```

Expected: tsc passes.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/workstation/panes/device-pane.ts
git commit -m "feat(workstation): add Device pane wrapper

Thin shell around the existing mountDevicePanel so Golden Layout
can host it as a dockable component. No behaviour change to the
underlying device editor.

Confidence: high
Scope-risk: narrow"
```

---

## Task 8: Help pane

**Files:**
- Create: `frontend/src/workstation/panes/help-pane.ts`

Three inner tabs — Tutorial / Parameters / Algorithm — using the HTML builders from task 6.

- [ ] **Step 1: Create `frontend/src/workstation/panes/help-pane.ts`**

```typescript
import { tutorialHTML } from '../../panels/tutorial'
import { parametersHTML } from '../../panels/parameters'
import { algorithmHTML } from '../../panels/algorithm'

type TabKey = 'tutorial' | 'parameters' | 'algorithm'

interface TabDef {
  key: TabKey
  label: string
  html: () => string
}

const TABS: TabDef[] = [
  { key: 'tutorial', label: 'Tutorial', html: tutorialHTML },
  { key: 'parameters', label: 'Parameters', html: parametersHTML },
  { key: 'algorithm', label: 'Algorithm', html: algorithmHTML },
]

export function mountHelpPane(container: HTMLElement): void {
  container.classList.add('pane', 'pane-help')
  container.innerHTML = `
    <div class="help-tabs" role="tablist">
      ${TABS.map((t, i) => `
        <button class="help-tab${i === 0 ? ' active' : ''}" data-help-tab="${t.key}" role="tab">${t.label}</button>
      `).join('')}
    </div>
    <div class="help-body" id="help-body">${TABS[0].html()}</div>`

  const body = container.querySelector<HTMLDivElement>('#help-body')!
  container.querySelectorAll<HTMLButtonElement>('.help-tab').forEach(btn => {
    btn.addEventListener('click', () => {
      const key = btn.dataset.helpTab as TabKey | undefined
      if (!key) return
      const tab = TABS.find(t => t.key === key)
      if (!tab) return
      container.querySelectorAll('.help-tab').forEach(b => b.classList.remove('active'))
      btn.classList.add('active')
      body.innerHTML = tab.html()
    })
  })
}
```

- [ ] **Step 2: Verify build**

Run:
```bash
cd frontend && npm run build
```

Expected: tsc passes.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/workstation/panes/help-pane.ts
git commit -m "feat(workstation): add Help pane with inner tabs

Tutorial / Parameters / Algorithm share one Help pane. Uses the
HTML builders exported in the previous refactor so the content
is not duplicated.

Confidence: high
Scope-risk: narrow"
```

---

## Task 9: Legacy experiment pane

**Files:**
- Create: `frontend/src/workstation/panes/legacy-experiment-pane.ts`

The legacy pane is the migration safety net. It hosts the existing `mountJVPanel`, `mountImpedancePanel`, `mountDegradationPanel` behind three in-pane tabs so every experiment stays usable during Phase 1. Phase 2 will delete this pane when experiments become first-class panes.

- [ ] **Step 1: Create `frontend/src/workstation/panes/legacy-experiment-pane.ts`**

```typescript
import { mountJVPanel } from '../../panels/jv'
import { mountImpedancePanel } from '../../panels/impedance'
import { mountDegradationPanel } from '../../panels/degradation'

type TabKey = 'jv' | 'is' | 'deg'

interface TabDef {
  key: TabKey
  label: string
  mount: (el: HTMLElement) => Promise<void>
}

const TABS: TabDef[] = [
  { key: 'jv', label: 'J-V Sweep', mount: mountJVPanel },
  { key: 'is', label: 'Impedance', mount: mountImpedancePanel },
  { key: 'deg', label: 'Degradation', mount: mountDegradationPanel },
]

export async function mountLegacyExperimentPane(container: HTMLElement): Promise<void> {
  container.classList.add('pane', 'pane-legacy')
  container.innerHTML = `
    <div class="legacy-tabs" role="tablist">
      ${TABS.map((t, i) => `
        <button class="legacy-tab${i === 0 ? ' active' : ''}" data-legacy-tab="${t.key}" role="tab">${t.label}</button>
      `).join('')}
    </div>
    <div class="legacy-body">
      ${TABS.map(t => `<section class="legacy-section" data-legacy-section="${t.key}" hidden></section>`).join('')}
    </div>`

  const mounted: Record<TabKey, boolean> = { jv: false, is: false, deg: false }

  async function activate(key: TabKey): Promise<void> {
    container.querySelectorAll<HTMLElement>('.legacy-tab').forEach(b => {
      b.classList.toggle('active', b.dataset.legacyTab === key)
    })
    container.querySelectorAll<HTMLElement>('.legacy-section').forEach(s => {
      s.hidden = s.dataset.legacySection !== key
    })
    if (!mounted[key]) {
      mounted[key] = true
      const section = container.querySelector<HTMLElement>(`[data-legacy-section="${key}"]`)!
      const tab = TABS.find(t => t.key === key)!
      try {
        await tab.mount(section)
      } catch (e) {
        mounted[key] = false
        section.innerHTML = `<div class="card error-card">Failed to load: ${(e as Error).message}</div>`
      }
    }
  }

  container.querySelectorAll<HTMLButtonElement>('.legacy-tab').forEach(btn => {
    btn.addEventListener('click', () => {
      const key = btn.dataset.legacyTab as TabKey | undefined
      if (key) void activate(key)
    })
  })

  await activate('jv')
}
```

- [ ] **Step 2: Verify build**

Run:
```bash
cd frontend && npm run build
```

Expected: tsc passes.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/workstation/panes/legacy-experiment-pane.ts
git commit -m "feat(workstation): add legacy experiment pane

Hosts existing J-V / Impedance / Degradation panels behind three
in-pane tabs so no experiment becomes inaccessible during the
Phase 1 shell migration. This pane is deleted in Phase 2 when
experiments become first-class panes.

Confidence: high
Scope-risk: narrow"
```

---

## Task 10: Solver console module

**Files:**
- Create: `frontend/src/workstation/console.ts`

Minimal strip: two regions — an active-physics line (updated via `setPhysics`) and a scrolling log line (updated via `log`). The Phase 1 goal is just "show something live so the UI feels connected"; richer streaming comes later when experiments are first-class.

- [ ] **Step 1: Create `frontend/src/workstation/console.ts`**

```typescript
export interface ConsoleHandle {
  /** Set the left-most "● FULL  band offsets · …" indicator. */
  setPhysics(tierLabel: string, summary: string): void
  /** Replace the right-hand scrolling log line. */
  log(message: string): void
}

export function mountConsole(container: HTMLElement): ConsoleHandle {
  container.classList.add('solver-console')
  container.innerHTML = `
    <span class="console-physics" id="console-physics">
      <span class="console-dot"></span>
      <span class="console-tier">IDLE</span>
      <span class="console-summary">(no active device)</span>
    </span>
    <span class="console-log" id="console-log"></span>`

  const tierEl = container.querySelector<HTMLElement>('.console-tier')!
  const summaryEl = container.querySelector<HTMLElement>('.console-summary')!
  const logEl = container.querySelector<HTMLElement>('#console-log')!

  return {
    setPhysics(tierLabel, summary) {
      tierEl.textContent = tierLabel
      summaryEl.textContent = summary
    },
    log(message) {
      logEl.textContent = message
    },
  }
}
```

- [ ] **Step 2: Verify build**

Run:
```bash
cd frontend && npm run build
```

Expected: tsc passes.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/workstation/console.ts
git commit -m "feat(workstation): add solver console strip

Two-region status strip: active-physics indicator + single log
line. Phase 2 will wire the log channel to the SSE progress
stream; Phase 1 is static-only.

Confidence: high
Scope-risk: narrow"
```

---

## Task 11: Golden Layout shell

**Files:**
- Create: `frontend/src/workstation/shell.ts`
- Modify: `frontend/src/style.css` (Golden Layout base CSS import)

Wires the sidebar tree + dock area + console. Seeds a default device from `ionmonger_benchmark.yaml` when `loadWorkspace()` returns null.

- [ ] **Step 1: Add the Golden Layout CSS import to `frontend/src/style.css`**

Append to the very top of `frontend/src/style.css`:

```css
@import 'golden-layout/dist/css/goldenlayout-base.css';
@import 'golden-layout/dist/css/themes/goldenlayout-dark-theme.css';
```

- [ ] **Step 2: Create `frontend/src/workstation/shell.ts`**

```typescript
import { GoldenLayout, LayoutConfig, ResolvedLayoutConfig } from 'golden-layout'
import { getConfig } from '../api'
import type { Workspace, Device } from './types'
import {
  addDevice,
  createEmptyWorkspace,
  loadWorkspace,
  saveWorkspace,
  setActiveDevice,
} from './state'
import { attachTreeHandlers, renderTreeHTML } from './tree'
import { mountConsole, type ConsoleHandle } from './console'
import { mountDevicePane } from './panes/device-pane'
import { mountHelpPane } from './panes/help-pane'
import { mountLegacyExperimentPane } from './panes/legacy-experiment-pane'

const DEFAULT_LAYOUT: LayoutConfig = {
  root: {
    type: 'row',
    content: [
      {
        type: 'stack',
        width: 50,
        content: [
          { type: 'component', componentType: 'device', title: 'Device' },
          { type: 'component', componentType: 'help', title: 'Help' },
        ],
      },
      {
        type: 'component',
        componentType: 'legacy-experiments',
        title: 'Experiments (legacy)',
        width: 50,
      },
    ],
  },
}

function tierLabel(tier: 'legacy' | 'fast' | 'full'): string {
  return tier.toUpperCase()
}

function physicsSummary(tier: 'legacy' | 'fast' | 'full'): string {
  if (tier === 'full') return 'band offsets · TE · TMM · dual ions · T-scaling'
  if (tier === 'fast') return 'Beer-Lambert · single ion · uniform τ · T-scaling'
  return 'Beer-Lambert · single ion · uniform τ · T=300K'
}

/** Seed a default device from the shipped ionmonger preset. */
async function seedDefaultDevice(): Promise<Device> {
  const config = await getConfig('ionmonger_benchmark.yaml')
  return {
    id: 'seed-ionmonger',
    name: 'IonMonger benchmark',
    tier: config.device.mode ?? 'full',
    config,
    experiments: [],
  }
}

export async function mountWorkstation(root: HTMLElement): Promise<void> {
  root.innerHTML = `
    <header class="workstation-header">
      <h1>Thin-Film Solar Cell Simulator</h1>
      <p class="subtitle">1D Drift-Diffusion + Poisson + Mobile Ions · Perovskite · CIGS · c-Si</p>
    </header>
    <div class="workstation-body">
      <aside class="workstation-sidebar" id="ws-tree"></aside>
      <main class="workstation-dock" id="ws-dock"></main>
    </div>
    <footer class="workstation-footer" id="ws-console"></footer>`

  const treeEl = root.querySelector<HTMLElement>('#ws-tree')!
  const dockEl = root.querySelector<HTMLElement>('#ws-dock')!
  const consoleEl = root.querySelector<HTMLElement>('#ws-console')!

  const consoleHandle: ConsoleHandle = mountConsole(consoleEl)

  // --- workspace state ---
  let workspace: Workspace = loadWorkspace() ?? createEmptyWorkspace('Untitled')
  if (workspace.devices.length === 0) {
    const seed = await seedDefaultDevice()
    workspace = addDevice(workspace, seed)
    saveWorkspace(workspace)
  }

  function refreshTree(): void {
    treeEl.innerHTML = renderTreeHTML(workspace)
  }
  refreshTree()

  attachTreeHandlers(treeEl, {
    onSelectDevice: (id) => {
      workspace = setActiveDevice(workspace, id)
      saveWorkspace(workspace)
      refreshTree()
      const active = workspace.devices.find(d => d.id === id)
      if (active) {
        consoleHandle.setPhysics(tierLabel(active.tier), physicsSummary(active.tier))
      }
    },
  })

  // --- physics indicator for the initially-active device ---
  const initialActive = workspace.devices.find(d => d.id === workspace.activeDeviceId)
  if (initialActive) {
    consoleHandle.setPhysics(tierLabel(initialActive.tier), physicsSummary(initialActive.tier))
  }

  // --- Golden Layout ---
  const layout = new GoldenLayout(dockEl)

  layout.registerComponentFactoryFunction('device', (container) => {
    void mountDevicePane(container.element, 'ws-device')
  })
  layout.registerComponentFactoryFunction('help', (container) => {
    mountHelpPane(container.element)
  })
  layout.registerComponentFactoryFunction('legacy-experiments', (container) => {
    void mountLegacyExperimentPane(container.element)
  })

  const initialLayout: LayoutConfig = (workspace.layout as LayoutConfig | null) ?? DEFAULT_LAYOUT
  try {
    layout.loadLayout(initialLayout)
  } catch (e) {
    console.error('loadLayout failed, falling back to default:', e)
    layout.loadLayout(DEFAULT_LAYOUT)
  }

  // Persist layout changes
  layout.on('stateChanged', () => {
    const saved: ResolvedLayoutConfig = layout.saveLayout()
    workspace = { ...workspace, layout: saved }
    saveWorkspace(workspace)
  })

  // Resize handling
  const resize = () => layout.setSize(dockEl.clientWidth, dockEl.clientHeight)
  window.addEventListener('resize', resize)
  // One synchronous resize after mount so the initial layout fills the dock.
  requestAnimationFrame(resize)
}
```

- [ ] **Step 3: Verify build**

Run:
```bash
cd frontend && npm run build
```

Expected: tsc passes, vite bundles. If Golden Layout's type exports differ (library version drift), adjust imports to match `golden-layout`'s actual exports — its `LayoutConfig` and `ResolvedLayoutConfig` are stable in the 2.x series.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/workstation/shell.ts frontend/src/style.css
git commit -m "feat(workstation): wire Golden Layout shell

Combines sidebar tree + dock host + solver console into a single
mountWorkstation entry point. Seeds a default device from the
ionmonger preset on first launch; persists layout + workspace to
localStorage; console reflects the active device's physics tier.

Confidence: high
Scope-risk: moderate"
```

---

## Task 12: Swap `main.ts` to the workstation shell

**Files:**
- Modify: `frontend/src/main.ts`

This task replaces the tab-strip bootstrap. The old `main.ts` content is discarded because `mountWorkstation` owns the entire app shell now. The legacy experiment pane from task 9 keeps every experiment reachable.

- [ ] **Step 1: Replace `frontend/src/main.ts` with the workstation bootstrap**

```typescript
import './style.css'
import { mountWorkstation } from './workstation/shell'

const app = document.querySelector<HTMLDivElement>('#app')!
void mountWorkstation(app).catch(e => {
  app.innerHTML = `<div class="error-card" style="padding:20px;">
    Failed to mount workstation: ${(e as Error).message}
  </div>`
  console.error(e)
})
```

- [ ] **Step 2: Verify the build passes**

Run:
```bash
cd frontend && npm run build
```

Expected: tsc + vite build both succeed. No TypeScript errors.

- [ ] **Step 3: Manually verify in the browser**

Start the dev server (or use the already-running one) and open `http://127.0.0.1:5173`:

```bash
cd frontend && npm run dev
```

Manual checklist (all must pass):
- [ ] Header shows "Thin-Film Solar Cell Simulator" title.
- [ ] Left sidebar shows "📁 Devices" folder with one device "IonMonger benchmark" and a `FULL` tier badge.
- [ ] Main dock area shows a "Device" / "Help" tab stack on the left and an "Experiments (legacy)" pane on the right.
- [ ] Clicking the "Device" tab shows the existing config editor with all layer cards.
- [ ] Clicking the "Help" tab shows Tutorial / Parameters / Algorithm inner tabs, each of which renders the same content as the pre-migration tabs.
- [ ] Clicking the "Experiments (legacy)" pane and running a J-V sweep reproduces the previous behaviour (progress bar, results card, J-V plot).
- [ ] Bottom console strip shows `● FULL  band offsets · TE · TMM · dual ions · T-scaling`.
- [ ] Reloading the page keeps the layout (any drag/split/stack the user made persists).

If any item fails, fix it before committing.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/main.ts
git commit -m "feat(workstation): swap main.ts to workstation shell

Replaces the 6-tab strip with mountWorkstation. All prior
experiments reachable via the legacy-experiments pane on the
right; docs reachable via the Help tab on the left. Layout +
workspace persisted to localStorage.

Confidence: high
Scope-risk: broad
Directive: Phase 2 deletes the legacy-experiments pane — do not
add new features there."
```

---

## Task 13: Workstation styles

**Files:**
- Modify: `frontend/src/style.css`

Append workstation-specific rules. Keep the existing rules intact (they style the config editor, plot cards, metric blocks, etc. — all of which still render inside panes).

- [ ] **Step 1: Append to `frontend/src/style.css`**

```css
/* ---------- Workstation shell ---------- */
html, body, #app {
  height: 100%;
  margin: 0;
}
#app {
  display: flex;
  flex-direction: column;
  background: var(--bg, #0e1628);
  color: var(--fg, #e6ecf5);
}
.workstation-header {
  padding: 14px 20px 10px;
  border-bottom: 1px solid rgba(143, 181, 255, 0.15);
}
.workstation-header h1 {
  margin: 0;
  font-size: 18px;
  font-weight: 600;
}
.workstation-header .subtitle {
  margin: 2px 0 0;
  font-size: 12px;
  opacity: 0.6;
}
.workstation-body {
  display: flex;
  flex: 1;
  min-height: 0;
}
.workstation-sidebar {
  width: 220px;
  min-width: 220px;
  border-right: 1px solid rgba(143, 181, 255, 0.15);
  padding: 12px 10px;
  overflow-y: auto;
  font-size: 13px;
}
.workstation-dock {
  flex: 1;
  min-width: 0;
  position: relative;
}
.workstation-footer {
  border-top: 1px solid rgba(143, 181, 255, 0.15);
  padding: 6px 14px;
  font-family: 'SF Mono', Menlo, Consolas, monospace;
  font-size: 11px;
  display: flex;
  gap: 14px;
  align-items: center;
  min-height: 28px;
}

/* ---------- Tree ---------- */
.tree-section {
  margin-bottom: 14px;
}
.tree-section-header {
  font-weight: 600;
  font-size: 12px;
  letter-spacing: 0.03em;
  margin-bottom: 4px;
  opacity: 0.8;
}
.tree-node {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 4px 6px;
  border-radius: 4px;
  cursor: pointer;
}
.tree-node:hover {
  background: rgba(143, 181, 255, 0.08);
}
.tree-node-active {
  background: rgba(143, 181, 255, 0.18);
}
.tree-label {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.tree-empty {
  opacity: 0.5;
  font-style: italic;
  font-size: 12px;
  padding: 2px 6px;
}
.tier-badge {
  font-size: 9px;
  padding: 1px 5px;
  border-radius: 3px;
  color: #000;
  font-weight: 700;
  letter-spacing: 0.04em;
}
.tier-badge-full { background: #8fb5ff; }
.tier-badge-fast { background: #8ad4a0; }
.tier-badge-legacy { background: #ffb87a; }

/* ---------- Console strip ---------- */
.solver-console {
  display: flex;
  gap: 14px;
  width: 100%;
}
.console-physics {
  display: flex;
  align-items: center;
  gap: 6px;
}
.console-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #8ad4a0;
}
.console-tier {
  font-weight: 700;
}
.console-summary {
  opacity: 0.7;
}
.console-log {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  opacity: 0.6;
}

/* ---------- Pane containers (generic) ---------- */
.pane {
  width: 100%;
  height: 100%;
  overflow: auto;
  box-sizing: border-box;
}
.pane-body {
  padding: 10px;
}

/* ---------- Help pane inner tabs ---------- */
.help-tabs, .legacy-tabs {
  display: flex;
  gap: 2px;
  border-bottom: 1px solid rgba(143, 181, 255, 0.15);
  padding: 6px 8px 0;
  background: rgba(0, 0, 0, 0.2);
}
.help-tab, .legacy-tab {
  background: transparent;
  border: 0;
  color: inherit;
  padding: 6px 12px;
  font-size: 12px;
  cursor: pointer;
  border-bottom: 2px solid transparent;
}
.help-tab.active, .legacy-tab.active {
  border-bottom-color: #8fb5ff;
  font-weight: 600;
}
.help-body, .legacy-body {
  padding: 10px;
}
```

- [ ] **Step 2: Visually verify the workstation renders**

Reload `http://127.0.0.1:5173` and confirm:
- [ ] Sidebar is 220 px wide, dark theme, tree node is clickable.
- [ ] Tier badge on the seeded device is a blue pill with the text "FULL".
- [ ] Console strip at the bottom is a thin monospace line, not wrapping.
- [ ] Golden Layout pane borders visible; tabs draggable (try dragging the "Device" tab onto the right pane — it should dock).

- [ ] **Step 3: Commit**

```bash
git add frontend/src/style.css
git commit -m "style(workstation): add shell layout + tree + console CSS

Sidebar + dock host + console strip. Tier badges coloured per
tier. Generic .pane container keeps existing card styles working
unchanged inside dockable panes.

Confidence: high
Scope-risk: narrow"
```

---

## Task 14: Backend active_physics field

**Files:**
- Modify: `backend/main.py`
- Create: `tests/unit/backend/test_active_physics.py`

The new field describes, as a short human-readable string, which Phase 1–4 upgrades are active for the stack being simulated. Computed once per job, injected into every `_run` closure's return dict.

- [ ] **Step 1: Write the failing test first**

Create `tests/unit/backend/test_active_physics.py`:

```python
from backend.main import _describe_active_physics
from perovskite_sim.models.device import DeviceStack
from perovskite_sim.models.parameters import MaterialParams


def _minimal_stack(mode: str) -> DeviceStack:
    # A single trivial layer is enough to construct a DeviceStack that
    # resolve_mode() can consume; we only care about the mode string.
    layer = MaterialParams(
        name="L", role="absorber", thickness=1e-7, eps_r=10.0,
        mu_n=1e-4, mu_p=1e-4, ni=1e15, N_D=0.0, N_A=0.0,
        D_ion=0.0, P_lim=1e26, P0=1e24,
        tau_n=1e-9, tau_p=1e-9, n1=1e15, p1=1e15,
        B_rad=0.0, C_n=0.0, C_p=0.0, alpha=0.0,
    )
    return DeviceStack(layers=[layer], V_bi=1.0, Phi=1.4e21, T=300.0, mode=mode)


def test_full_mode_string_lists_all_upgrades():
    s = _describe_active_physics(_minimal_stack("full"))
    assert "FULL" in s
    assert "TE" in s
    assert "TMM" in s
    assert "dual ions" in s
    assert "T-scaling" in s


def test_legacy_mode_string_lists_no_upgrades():
    s = _describe_active_physics(_minimal_stack("legacy"))
    assert "LEGACY" in s
    assert "Beer-Lambert" in s
    assert "uniform" in s
    assert "T=300K" in s


def test_fast_mode_string_is_distinct_from_full_and_legacy():
    s = _describe_active_physics(_minimal_stack("fast"))
    assert "FAST" in s
    # fast should not claim full-only features
    assert "TMM" not in s
    assert "dual ions" not in s
```

- [ ] **Step 2: Run the test — verify it fails**

Run from the repo root:
```bash
pytest tests/unit/backend/test_active_physics.py -v
```

Expected: `ImportError: cannot import name '_describe_active_physics' from 'backend.main'`.

- [ ] **Step 3: Add `_describe_active_physics` to `backend/main.py`**

Find the top of `backend/main.py` where other helpers live (around the existing `from perovskite_sim.models.mode import resolve_mode` import that was added in Phase 5). Add immediately below the imports:

```python
def _describe_active_physics(stack) -> str:
    """Return a short human-readable description of the active physics tier.

    Used by the SSE result payload so the frontend solver console can
    show which Phase 1-4 upgrades ran without re-deriving the flags.
    """
    mode_name = str(getattr(stack, "mode", "full")).lower()
    mode = resolve_mode(mode_name)
    if mode_name == "legacy":
        return "LEGACY  Beer-Lambert · single ion · uniform τ · T=300K"
    if mode_name == "fast":
        return "FAST  Beer-Lambert · single ion · uniform τ · T-scaling"
    # full (or any unrecognised-but-valid mode falls through to full defaults)
    parts = []
    if mode.use_thermionic_emission:
        parts.append("band offsets · TE")
    if mode.use_tmm_optics:
        parts.append("TMM")
    else:
        parts.append("Beer-Lambert")
    if mode.use_dual_ions:
        parts.append("dual ions")
    if mode.use_trap_profile:
        parts.append("trap profile")
    if mode.use_temperature_scaling:
        parts.append("T-scaling")
    return "FULL  " + " · ".join(parts)
```

- [ ] **Step 4: Run the test — it must pass**

Run from the repo root:
```bash
pytest tests/unit/backend/test_active_physics.py -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Inject the field into each `_run` closure's return dict**

In `backend/main.py`, find the three `_run` closures inside `start_job` (around lines 278, 289, 309 in the current file). For each one, immediately before the final `return` statement, add:

For the `"jv"` branch:
```python
            out = to_serializable(result)
            out["active_physics"] = _describe_active_physics(stack)
            return out
```

(Replacing the existing `return to_serializable(result)` so `out` is referenced only once.)

For the `"impedance"` branch, `out` is already defined; add the line just before `return out`:
```python
            out["active_physics"] = _describe_active_physics(stack)
            return out
```

For the `"degradation"` branch, `out` is already defined; same treatment:
```python
            out["active_physics"] = _describe_active_physics(stack)
            return out
```

- [ ] **Step 6: Verify the existing pytest suite still passes**

Run from the repo root:
```bash
pytest -x
```

Expected: all tests still pass (the new addition is additive; it only enriches the response dict).

- [ ] **Step 7: Commit**

```bash
git add backend/main.py tests/unit/backend/test_active_physics.py
git commit -m "feat(backend): add active_physics to job result SSE payload

Short human-readable string describing which Phase 1-4 upgrades
ran for the job. Frontend solver console displays this verbatim
so users can audit which tier actually executed without having
to re-derive it from the request config.

Confidence: high
Scope-risk: narrow"
```

---

## Self-Review

### Spec coverage

Walking the Phase 1 section of the spec against the task list:

| Spec requirement | Task |
|---|---|
| Golden Layout v2 integrated, tab strip removed | 2, 11, 12 |
| Project tree in left sidebar with one hard-coded default device | 5, 11 |
| Central dock area: Device pane wrapping existing config editor | 7 |
| Help pane with Tutorial / Parameters / Algorithm as tabs | 6, 8 |
| Bottom solver console strip, active-physics first token | 10, 11, 13 |
| localStorage serialises layout + workspace | 4, 11 |
| Legacy fallback so no experiment breaks mid-migration | 9, 11, 12 |
| Backend `active_physics` in result SSE payload | 14 |
| New `workstation/` module structure as specified | 3–11 (all new files under workstation/) |
| Plain TS, no React migration | 1–14 (no React anywhere) |
| Vitest test infrastructure for pure logic | 1, 4, 5 |
| Existing panels/types/api.ts unchanged except for doc-panel HTML exports | 6 only (additive refactor) |

No gaps identified.

### Placeholder scan

- No "TBD", "TODO", or "fill in details" in any task.
- Every TDD step has concrete failing-then-passing test code.
- Every code change shows complete file contents or unambiguous exact edits (e.g. "Append to the very top of `frontend/src/style.css`" + the actual CSS).
- The manual verification checklist in Task 12 has specific items rather than "verify it works".

### Type consistency

- `Workspace`, `Device`, `DevicePaneHandle`, `ConsoleHandle`, `TreeHandlers` are defined once and referenced consistently.
- `TabKey` is declared locally in `help-pane.ts` and `legacy-experiment-pane.ts` — intentionally not shared because the tab sets are different.
- `DeviceConfig` and `SimulationModeName` are imported from the existing `../types` and never redefined.
- `mountDevicePanel` from `device-panel.ts` is called with `(inner, tabId)` — matches its existing signature in that file (`mountDevicePanel(root, tabId)`).
- `ResolvedLayoutConfig` / `LayoutConfig` come from `golden-layout` and are stable in 2.x.
- Backend `_describe_active_physics` signature is `(stack) -> str` and is called consistently with a `DeviceStack` argument in both the test and the three `_run` closures.

No inconsistencies found.

---

**Self-review complete. No issues to fix inline.**
