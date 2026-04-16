import { GoldenLayout } from 'golden-layout'
import type { LayoutConfig, ResolvedLayoutConfig } from 'golden-layout'
import { getConfig, listConfigs } from '../api'
import type { Workspace, Device, Run, ExperimentKind } from './types'
import {
  STORAGE_KEY,
  addDevice,
  addExperiment,
  addRun,
  createEmptyWorkspace,
  loadWorkspace,
  saveWorkspace,
  setActiveDevice,
  setActiveExperiment,
  setActiveRun,
} from './state'
import { attachTreeHandlers, renderTreeHTML } from './tree'
import { mountConsole } from './console'
import type { ConsoleHandle } from './console'
import { mountDevicePane } from './panes/device-pane'
import type { DevicePanel } from '../device-panel'
import { mountHelpPane } from './panes/help-pane'
import { mountExperimentPane } from './panes/experiment-pane'
import { mountTandemPane } from './panes/tandem-pane'
import { mountMainPlotPane } from './panes/main-plot-pane'
import type { MainPlotHandle } from './panes/main-plot-pane'
import { presetsFromEntries, showWizard } from './wizard'

const DEFAULT_LAYOUT: LayoutConfig = {
  root: {
    type: 'row',
    content: [
      {
        type: 'stack',
        width: 42,
        content: [
          { type: 'component', componentType: 'device', title: 'Device' },
          { type: 'component', componentType: 'help', title: 'Help' },
        ],
      },
      {
        type: 'stack',
        width: 28,
        content: [
          { type: 'component', componentType: 'experiments', title: 'Experiments' },
          { type: 'component', componentType: 'tandem', title: 'Tandem' },
        ],
      },
      {
        type: 'component',
        componentType: 'main-plot',
        title: 'Main Plot',
        width: 30,
      },
    ],
  },
}

function tierLabel(tier: 'legacy' | 'fast' | 'full'): string {
  return tier.toUpperCase()
}

// Keep in sync with backend/main.py:_describe_active_physics. FAST today has
// identical physics flags to LEGACY (see perovskite_sim/models/mode.py), so
// both tiers advertise the same feature string — do not reintroduce
// "T-scaling" for FAST until SimulationMode.FAST actually enables it.
function physicsSummary(tier: 'legacy' | 'fast' | 'full'): string {
  if (tier === 'full') return 'band offsets · TE · TMM · dual ions · T-scaling'
  return 'flat bands · Beer-Lambert · single ion · uniform τ · T=300K'
}

function randomDeviceId(): string {
  return 'd-' + Math.random().toString(36).slice(2, 10)
}

function randomExperimentId(): string {
  return 'e-' + Math.random().toString(36).slice(2, 10)
}

export async function mountWorkstation(root: HTMLElement): Promise<void> {
  root.innerHTML = `
    <header class="workstation-header">
      <div>
        <h1>Thin-Film Solar Cell Simulator</h1>
        <p class="subtitle">1D Drift-Diffusion + Poisson + Mobile Ions · Perovskite · CIGS · c-Si</p>
      </div>
      <div class="workstation-header-actions">
        <button type="button" class="btn btn-ghost" id="ws-reset-layout" title="Restore all panes to the default dock layout">Reset Layout</button>
        <button type="button" class="btn btn-ghost btn-danger-ghost" id="ws-clear-workspace" title="Clear all devices, experiments, and runs (resets localStorage)">Clear Workspace</button>
      </div>
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
    const entries = await listConfigs()
    const presets = presetsFromEntries(entries)
    const result = await showWizard(root, presets)
    if (!result.cancelled && result.selection) {
      const cfg = await getConfig(result.selection.preset)
      cfg.device.mode = result.selection.tier
      const dev: Device = {
        id: randomDeviceId(),
        name: result.selection.name,
        tier: result.selection.tier,
        config: cfg,
        experiments: [],
      }
      workspace = addDevice(workspace, dev)
      saveWorkspace(workspace)
    }
  }

  function refreshTree(): void {
    treeEl.innerHTML = renderTreeHTML(workspace)
  }
  refreshTree()

  // --- helpers wired into pane factories ---
  let mainPlot: MainPlotHandle | null = null
  let devicePanel: DevicePanel | null = null

  function activeDeviceAccessor(): { id: string; config: import('../types').DeviceConfig } | null {
    const id = workspace.activeDeviceId
    if (!id) return null
    const d = workspace.devices.find(x => x.id === id)
    if (!d) return null
    // Prefer the live config from the Device pane editor (reflects dropdown
    // changes and manual edits) over the stale workspace snapshot.
    const liveConfig = devicePanel ? devicePanel.getConfig() : d.config
    return { id: d.id, config: liveConfig }
  }

  function ensureExperiment(deviceId: string, kind: ExperimentKind): string {
    const dev = workspace.devices.find(d => d.id === deviceId)
    if (!dev) throw new Error('device missing')
    const existing = dev.experiments.find(e => e.kind === kind)
    if (existing) return existing.id
    const id = randomExperimentId()
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
    focusComponent('main-plot')
    consoleHandle.log(`run complete: ${kind}  (${run.activePhysics})`)
  }

  function focusComponent(componentType: string): void {
    try {
      const root = layout.rootItem as unknown as {
        getItemsByFilter?: (fn: (it: { isComponent?: boolean; componentType?: string }) => boolean) => Array<{ focus?: () => void }>
      } | undefined
      const items = root?.getItemsByFilter?.(
        (it) => !!it.isComponent && it.componentType === componentType,
      ) ?? []
      items[0]?.focus?.()
    } catch (e) {
      console.warn('focusComponent failed:', e)
    }
  }

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
    onSelectExperiment: (deviceId, experimentId) => {
      workspace = setActiveExperiment(workspace, deviceId, experimentId)
      saveWorkspace(workspace)
      refreshTree()
      focusComponent('experiments')
    },
    onSelectRun: (deviceId, experimentId, runId) => {
      workspace = setActiveRun(workspace, deviceId, experimentId, runId)
      saveWorkspace(workspace)
      refreshTree()
      mainPlot?.update(workspace)
      focusComponent('main-plot')
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
    const active = workspace.devices.find(d => d.id === workspace.activeDeviceId)
    void mountDevicePane(container.element, 'ws-device', active?.tier ?? 'full').then(panel => {
      devicePanel = panel
      // Sync device config changes (preset dropdown, manual edits) back into the workspace
      panel.onChange((cfg) => {
        const devId = workspace.activeDeviceId
        if (!devId) return
        const idx = workspace.devices.findIndex(d => d.id === devId)
        if (idx < 0) return
        const updated = { ...workspace.devices[idx], config: cfg }
        const devices = workspace.devices.map((d, i) => (i === idx ? updated : d))
        workspace = { ...workspace, devices }
        saveWorkspace(workspace)
      })
    })
  })
  layout.registerComponentFactoryFunction('help', (container) => {
    mountHelpPane(container.element)
  })
  layout.registerComponentFactoryFunction('experiments', (container) => {
    mountExperimentPane(container.element, {
      getActiveDevice: () => activeDeviceAccessor(),
      onRunComplete: (deviceId, kind, run) => commitRun(deviceId, kind, run),
    })
  })
  layout.registerComponentFactoryFunction('tandem', (container) => {
    mountTandemPane(container.element)
  })
  layout.registerComponentFactoryFunction('main-plot', (container) => {
    mainPlot = mountMainPlotPane(container.element)
    mainPlot.update(workspace)
  })

  const initialLayout: LayoutConfig = (workspace.layout as LayoutConfig | null) ?? DEFAULT_LAYOUT
  try {
    layout.loadLayout(initialLayout)
  } catch (e) {
    console.error('loadLayout failed, falling back to default:', e)
    layout.loadLayout(DEFAULT_LAYOUT)
  }

  // Reset Layout — restore the default dock so closed panes come back.
  const resetBtn = root.querySelector<HTMLButtonElement>('#ws-reset-layout')
  resetBtn?.addEventListener('click', () => {
    layout.loadLayout(DEFAULT_LAYOUT)
    workspace = { ...workspace, layout: null }
    saveWorkspace(workspace)
    requestAnimationFrame(() => layout.setSize(dockEl.clientWidth, dockEl.clientHeight))
  })

  // Clear Workspace — nuke localStorage and reload
  const clearBtn = root.querySelector<HTMLButtonElement>('#ws-clear-workspace')
  clearBtn?.addEventListener('click', () => {
    if (!confirm('Clear all devices, experiments, and runs? This cannot be undone.')) return
    localStorage.removeItem(STORAGE_KEY)
    window.location.reload()
  })

  // Persist layout changes
  layout.on('stateChanged', () => {
    const saved: ResolvedLayoutConfig = layout.saveLayout()
    workspace = { ...workspace, layout: saved }
    saveWorkspace(workspace)
  })

  // Resize handling — keep dock sized to its container
  const resize = (): void => layout.setSize(dockEl.clientWidth, dockEl.clientHeight)
  window.addEventListener('resize', resize)
  // One synchronous resize after mount so the initial layout fills the dock.
  requestAnimationFrame(resize)
}
