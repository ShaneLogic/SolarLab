import { listConfigs, getConfig } from './api'
import { renderDeviceEditor, readDeviceEditor } from './config-editor'
import type { DeviceConfig } from './types'

export interface DevicePanel {
  getConfig(): DeviceConfig
  onChange(cb: (cfg: DeviceConfig) => void): void
}

// Shared device editor per tab. Fetches the YAML from the backend so the
// frontend doesn't need to duplicate default material parameters.
export async function mountDevicePanel(
  root: HTMLElement,
  tabId: string,
): Promise<DevicePanel> {
  root.innerHTML = `
    <div class="card">
      <div class="card-header">
        <h3>Device Configuration</h3>
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
    renderDeviceEditor(editor, cfg)
    listeners.forEach(l => l(cfg))
  }

  select.addEventListener('change', () => { void load(select.value) })
  resetBtn.addEventListener('click', () => {
    if (pristine) {
      current = structuredClone(pristine)
      renderDeviceEditor(editor, current)
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
