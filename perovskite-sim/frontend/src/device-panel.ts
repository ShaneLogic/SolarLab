import {
  listConfigs,
  getConfig,
  fetchOpticalMaterials,
  fetchLayerTemplates,
} from './api'
import { renderDeviceEditor, readDeviceEditor, setOpticalMaterialOptions } from './config-editor'
import { mountStackVisualizer } from './stack/stack-visualizer'
import { validate } from './stack/stack-validator'
import { isDirty } from './stack/dirty-state'
import { reconcileInterfaces } from './stack/reconcile-interfaces'
import { openAddLayerDialog } from './stack/add-layer-dialog'
import { openSaveAsDialog } from './stack/save-as-dialog'
import { isLayerBuilderEnabled } from './workstation/tier-gating'
import type {
  DeviceConfig,
  LayerTemplate,
  SimulationModeName,
  StackAction,
} from './types'

export interface DevicePanel {
  getConfig(): DeviceConfig
  onChange(cb: (cfg: DeviceConfig) => void): void
}

export interface MountDevicePanelOptions {
  tier?: SimulationModeName
}

/**
 * Render the "TMM active · N layers" badge shown in the device-pane header.
 * Returns an empty string unless the device is on the `full` tier AND at
 * least one layer has a non-empty `optical_material` field.
 */
export function computeTmmBadge(
  config: DeviceConfig,
  tier: SimulationModeName | undefined,
): string {
  if (tier !== 'full') return ''
  const tmmLayers = config.layers.filter(
    l => l.optical_material != null && l.optical_material !== '',
  )
  if (tmmLayers.length === 0) return ''
  const noun = tmmLayers.length === 1 ? 'layer' : 'layers'
  return `<span class="tmm-badge" title="Optical generation computed with transfer-matrix method. Layers without optical_material fall back to Beer-Lambert.">TMM active · ${tmmLayers.length} ${noun}</span>`
}

export async function mountDevicePanel(
  root: HTMLElement,
  tabId: string,
  options: MountDevicePanelOptions = {},
): Promise<DevicePanel> {
  const { tier } = options
  const builderOn = isLayerBuilderEnabled(tier ?? 'full')

  root.innerHTML = `
    <div class="card">
      <div class="card-header">
        <h3>Device Configuration <span id="${tabId}-tmm-badge-slot"></span> <span id="${tabId}-dirty-slot"></span></h3>
        <div class="header-actions">
          <select id="${tabId}-config-select" class="config-select"></select>
          <button class="btn btn-ghost" id="${tabId}-reset">Reset</button>
        </div>
      </div>
      ${builderOn
        ? `<div id="${tabId}-visualizer"></div><div id="${tabId}-editor"></div>`
        : `<div id="${tabId}-editor"></div>`}
    </div>`

  const select = root.querySelector<HTMLSelectElement>(`#${tabId}-config-select`)!
  const editor = root.querySelector<HTMLDivElement>(`#${tabId}-editor`)!
  const resetBtn = root.querySelector<HTMLButtonElement>(`#${tabId}-reset`)!
  const badgeSlot = root.querySelector<HTMLSpanElement>(`#${tabId}-tmm-badge-slot`)!
  const dirtySlot = root.querySelector<HTMLSpanElement>(`#${tabId}-dirty-slot`)!

  let loaded: DeviceConfig | null = null
  let current: DeviceConfig | null = null
  let selectedLayerIdx = 0
  let templates: Record<string, LayerTemplate> = {}
  const listeners: Array<(c: DeviceConfig) => void> = []

  try {
    const materials = await fetchOpticalMaterials()
    setOpticalMaterialOptions(materials)
  } catch (err) {
    console.warn('fetchOpticalMaterials failed', err)
    setOpticalMaterialOptions([])
  }

  if (builderOn) {
    try {
      templates = await fetchLayerTemplates()
    } catch (err) {
      console.warn('fetchLayerTemplates failed', err)
    }
  }

  const entries = await listConfigs()
  const shipped = entries.filter(e => e.namespace === 'shipped')
  const user = entries.filter(e => e.namespace === 'user')
  select.innerHTML = optgroup('Shipped presets', shipped) + optgroup('User presets', user)

  function refreshDirtyPill(): void {
    if (loaded && current && isDirty(loaded, current)) {
      dirtySlot.innerHTML = '<span class="dirty-pill">● modified</span>'
    } else {
      dirtySlot.innerHTML = ''
    }
  }

  function refreshBadge(cfg: DeviceConfig): void {
    badgeSlot.innerHTML = computeTmmBadge(cfg, tier)
  }

  let visualizerHandle: ReturnType<typeof mountStackVisualizer> | null = null
  if (builderOn) {
    const visualizerEl = root.querySelector<HTMLElement>(`#${tabId}-visualizer`)!
    visualizerHandle = mountStackVisualizer(visualizerEl, action => handleStackAction(action))

    visualizerEl.addEventListener('stack-insert-request', async (ev: Event) => {
      const detail = (ev as CustomEvent<{ atIdx: number }>).detail
      const layer = await openAddLayerDialog(templates)
      if (layer && current) {
        const newLayers = [...current.layers]
        newLayers.splice(detail.atIdx, 0, layer)
        const newInterfaces = reconcileInterfaces(
          current.layers,
          newLayers,
          current.device.interfaces ?? [],
        )
        current = {
          ...current,
          layers: newLayers,
          device: { ...current.device, interfaces: newInterfaces },
        }
        selectedLayerIdx = detail.atIdx
        rerender()
      }
    })

    visualizerEl.addEventListener('stack-edit-iface', (ev: Event) => {
      const detail = (ev as CustomEvent<{ ifaceIdx: number }>).detail
      if (!current) return
      const existing = (current.device.interfaces?.[detail.ifaceIdx] ?? [0, 0]) as readonly [number, number]
      const vn = window.prompt('v_n (m/s)', String(existing[0]))
      if (vn == null) return
      const vp = window.prompt('v_p (m/s)', String(existing[1]))
      if (vp == null) return
      const newPair: [number, number] = [Number(vn) || 0, Number(vp) || 0]
      const newIfaces = [...(current.device.interfaces ?? [])]
      while (newIfaces.length < current.layers.length - 1) newIfaces.push([0, 0])
      newIfaces[detail.ifaceIdx] = newPair
      current = {
        ...current,
        device: { ...current.device, interfaces: newIfaces },
      }
      rerender()
    })

    visualizerEl.addEventListener('click', async ev => {
      const target = ev.target as HTMLElement
      const stackAction = target.closest<HTMLElement>('[data-stack-action]')
      if (!stackAction) return
      const action = stackAction.dataset.stackAction!
      if (action === 'add' && current) {
        const layer = await openAddLayerDialog(templates)
        if (layer) {
          const newLayers = [...current.layers, layer]
          const newInterfaces = reconcileInterfaces(
            current.layers, newLayers, current.device.interfaces ?? [],
          )
          current = {
            ...current,
            layers: newLayers,
            device: { ...current.device, interfaces: newInterfaces },
          }
          selectedLayerIdx = current.layers.length - 1
          rerender()
        }
      } else if (action === 'save-as' && current) {
        const result = await openSaveAsDialog(current)
        if (result) {
          loaded = current
          refreshDirtyPill()
          await refreshConfigsDropdown(result.saved)
        }
      } else if (action === 'download-yaml' && current) {
        downloadYaml(current)
      }
    })
  }

  function handleStackAction(action: StackAction): void {
    if (!current) return
    switch (action.type) {
      case 'select':
        selectedLayerIdx = action.idx
        rerender()
        return
      case 'delete': {
        if (current.layers.length <= 1) return
        if (current.layers[action.idx]?.role === 'absorber') {
          if (!window.confirm('Delete the absorber layer?')) return
        }
        const newLayers = current.layers.filter((_, i) => i !== action.idx)
        const newIfaces = reconcileInterfaces(
          current.layers, newLayers, current.device.interfaces ?? [],
        )
        current = {
          ...current,
          layers: newLayers,
          device: { ...current.device, interfaces: newIfaces },
        }
        if (selectedLayerIdx >= newLayers.length) selectedLayerIdx = newLayers.length - 1
        rerender()
        return
      }
      case 'reorder': {
        const { from, to } = action
        if (to < 0 || to >= current.layers.length) return
        const newLayers = [...current.layers]
        const [moved] = newLayers.splice(from, 1)
        newLayers.splice(to, 0, moved)
        const newIfaces = reconcileInterfaces(
          current.layers, newLayers, current.device.interfaces ?? [],
        )
        current = {
          ...current,
          layers: newLayers,
          device: { ...current.device, interfaces: newIfaces },
        }
        selectedLayerIdx = to
        rerender()
        return
      }
      case 'insert': {
        const newLayers = [...current.layers]
        newLayers.splice(action.atIdx, 0, action.layer)
        const newIfaces = reconcileInterfaces(
          current.layers, newLayers, current.device.interfaces ?? [],
        )
        current = {
          ...current,
          layers: newLayers,
          device: { ...current.device, interfaces: newIfaces },
        }
        selectedLayerIdx = action.atIdx
        rerender()
        return
      }
      case 'edit-interface': {
        const newIfaces = [...(current.device.interfaces ?? [])]
        while (newIfaces.length < current.layers.length - 1) newIfaces.push([0, 0])
        newIfaces[action.idx] = [action.pair[0], action.pair[1]]
        current = {
          ...current,
          device: { ...current.device, interfaces: newIfaces },
        }
        rerender()
        return
      }
    }
  }

  function rerender(): void {
    if (!current) return
    if (builderOn && visualizerHandle) {
      const report = validate(current)
      visualizerHandle.render(current, selectedLayerIdx, report)
      renderDeviceEditor(editor, current, tier, selectedLayerIdx)
    } else {
      renderDeviceEditor(editor, current, tier)
    }
    refreshBadge(current)
    refreshDirtyPill()
  }

  async function refreshConfigsDropdown(selectName: string): Promise<void> {
    const newEntries = await listConfigs()
    const s = newEntries.filter(e => e.namespace === 'shipped')
    const u = newEntries.filter(e => e.namespace === 'user')
    select.innerHTML = optgroup('Shipped presets', s) + optgroup('User presets', u)
    const target = u.find(e => e.name.startsWith(`${selectName}.`))
      ?? s.find(e => e.name.startsWith(`${selectName}.`))
    if (target) select.value = target.name
  }

  function downloadYaml(cfg: DeviceConfig): void {
    const yaml = stringifyYaml(cfg)
    const blob = new Blob([yaml], { type: 'application/x-yaml' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    const stem = (cfg.layers[0]?.name ?? 'device').replace(/[^a-zA-Z0-9_-]/g, '_')
    a.download = `${stem}.yaml`
    a.click()
    URL.revokeObjectURL(url)
  }

  async function load(name: string) {
    const cfg = await getConfig(name)
    loaded = structuredClone(cfg)
    current = cfg
    selectedLayerIdx = 0
    rerender()
    listeners.forEach(l => l(cfg))
  }

  select.addEventListener('change', () => { void load(select.value) })
  resetBtn.addEventListener('click', () => {
    if (loaded) {
      current = structuredClone(loaded)
      rerender()
    }
  })

  const initial = shipped.find(e => e.name.includes('ionmonger'))?.name ?? shipped[0]?.name ?? entries[0]?.name
  if (!initial) throw new Error('no configs available')
  select.value = initial
  await load(initial)

  return {
    getConfig(): DeviceConfig {
      if (!current) throw new Error('device config not loaded')
      if (builderOn) {
        return readDeviceEditor(current, selectedLayerIdx)
      }
      return readDeviceEditor(current)
    },
    onChange(cb) { listeners.push(cb) },
  }
}

function stringifyYaml(cfg: DeviceConfig): string {
  const lines: string[] = ['device:']
  for (const [k, v] of Object.entries(cfg.device)) {
    if (k === 'interfaces') continue
    lines.push(`  ${k}: ${formatYamlScalar(v)}`)
  }
  if (cfg.device.interfaces && cfg.device.interfaces.length > 0) {
    lines.push('  interfaces:')
    for (const pair of cfg.device.interfaces) {
      lines.push(`    - [${formatYamlScalar(pair[0])}, ${formatYamlScalar(pair[1])}]`)
    }
  }
  lines.push('layers:')
  for (const layer of cfg.layers) {
    lines.push('  - ' + Object.entries(layer)
      .map(([k, v]) => `${k}: ${formatYamlScalar(v)}`)
      .join('\n    '))
  }
  return lines.join('\n') + '\n'
}

function formatYamlScalar(v: unknown): string {
  if (v == null) return 'null'
  if (typeof v === 'boolean') return v ? 'true' : 'false'
  if (typeof v === 'number') {
    if (!Number.isFinite(v)) return '0'
    if (v === 0) return '0'
    const abs = Math.abs(v)
    if (abs >= 1e-3 && abs < 1e6) return String(v)
    return v.toExponential(6)
  }
  if (typeof v === 'string') {
    if (/^[A-Za-z0-9_]+$/.test(v)) return v
    return JSON.stringify(v)
  }
  return JSON.stringify(v)
}

function optgroup(label: string, items: ReadonlyArray<{ name: string }>): string {
  if (items.length === 0) return ''
  return `<optgroup label="${label}">${items
    .map(e => `<option value="${e.name}">${e.name.replace(/\.ya?ml$/, '')}</option>`)
    .join('')}</optgroup>`
}
