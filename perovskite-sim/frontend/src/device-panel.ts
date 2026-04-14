import { listConfigs, getConfig, fetchOpticalMaterials } from './api'
import { renderDeviceEditor, readDeviceEditor, setOpticalMaterialOptions } from './config-editor'
import type { DeviceConfig, SimulationModeName } from './types'

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
 * least one layer has a non-empty `optical_material` field — Beer-Lambert
 * stacks and lower tiers should not advertise TMM.
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
  const n = tmmLayers.length
  return `<span class="tmm-badge" title="Optical generation computed with transfer-matrix method. Layers without optical_material fall back to Beer-Lambert.">TMM active · ${n} layers</span>`
}

// Shared device editor per tab. Fetches the YAML from the backend so the
// frontend doesn't need to duplicate default material parameters.
export async function mountDevicePanel(
  root: HTMLElement,
  tabId: string,
  options: MountDevicePanelOptions = {},
): Promise<DevicePanel> {
  const { tier } = options
  root.innerHTML = `
    <div class="card">
      <div class="card-header">
        <h3>Device Configuration <span id="${tabId}-tmm-badge-slot"></span></h3>
        <div class="header-actions">
          <select id="${tabId}-config-select" class="config-select"></select>
          <button class="btn btn-ghost" id="${tabId}-reset">Reset</button>
        </div>
      </div>
      <div id="${tabId}-editor"></div>
    </div>`

  const select = root.querySelector<HTMLSelectElement>(`#${tabId}-config-select`)!
  const editor = root.querySelector<HTMLDivElement>(`#${tabId}-editor`)!
  const resetBtn = root.querySelector<HTMLButtonElement>(`#${tabId}-reset`)!
  const badgeSlot = root.querySelector<HTMLSpanElement>(`#${tabId}-tmm-badge-slot`)!

  function refreshBadge(cfg: DeviceConfig): void {
    badgeSlot.innerHTML = computeTmmBadge(cfg, tier)
  }

  // Populate optical-material dropdown options once per mount. Failure is
  // non-fatal: the dropdown will just show only the "(none)" sentinel.
  try {
    const materials = await fetchOpticalMaterials()
    setOpticalMaterialOptions(materials)
  } catch (err) {
    // eslint-disable-next-line no-console
    console.warn('fetchOpticalMaterials failed; optical-material dropdown will be empty', err)
    setOpticalMaterialOptions([])
  }

  const names = await listConfigs()
  select.innerHTML = names
    .map(n => `<option value="${n}">${n.replace(/\.ya?ml$/, '')}</option>`)
    .join('')

  let current: DeviceConfig | null = null
  let pristine: DeviceConfig | null = null
  const listeners: Array<(c: DeviceConfig) => void> = []

  async function load(name: string) {
    const cfg = await getConfig(name)
    pristine = structuredClone(cfg)
    current = cfg
    renderDeviceEditor(editor, cfg, tier)
    refreshBadge(cfg)
    listeners.forEach(l => l(cfg))
  }

  select.addEventListener('change', () => { void load(select.value) })
  resetBtn.addEventListener('click', () => {
    if (pristine) {
      current = structuredClone(pristine)
      renderDeviceEditor(editor, current, tier)
      refreshBadge(current)
    }
  })

  // Prefer ionmonger benchmark as default if available (best-tuned V_oc).
  const initial = names.find(n => n.includes('ionmonger')) ?? names[0]
  select.value = initial
  await load(initial)

  return {
    getConfig(): DeviceConfig {
      if (!current) throw new Error('device config not loaded')
      return readDeviceEditor(current)
    },
    onChange(cb) { listeners.push(cb) },
  }
}
