import { GoldenLayout } from 'golden-layout'
import type { LayoutConfig, ResolvedLayoutConfig } from 'golden-layout'
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
import { mountConsole } from './console'
import type { ConsoleHandle } from './console'
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

  // Resize handling — keep dock sized to its container
  const resize = (): void => layout.setSize(dockEl.clientWidth, dockEl.clientHeight)
  window.addEventListener('resize', resize)
  // One synchronous resize after mount so the initial layout fills the dock.
  requestAnimationFrame(resize)
}
