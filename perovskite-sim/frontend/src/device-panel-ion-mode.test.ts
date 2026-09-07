import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { mountDevicePanel } from './device-panel'
import type { DeviceConfig, LayerConfig, SimulationModeName } from './types'

vi.mock('./api', () => ({
  fetchOpticalMaterials: vi.fn().mockResolvedValue([]),
  fetchLayerTemplates: vi.fn().mockResolvedValue({}),
  listConfigs: vi.fn().mockResolvedValue([
    { name: 'calado2016_fig1f.yaml', namespace: 'shipped' },
  ]),
  getConfig: vi.fn(),
}))

function config(mode: SimulationModeName): DeviceConfig {
  const layer = (name: string, role: LayerConfig['role']): LayerConfig => ({
    name, role, thickness: 2e-7, eps_r: 20,
    mu_n: 2e-3, mu_p: 2e-3, ni: 1e12, N_D: 0, N_A: 0,
    D_ion: 2.585e-18, P0: 1e25, P_lim: 1e30,
    D_ion_neg: 1e-18, P0_neg: 2e24,
    tau_n: 1e-6, tau_p: 1e-6, n1: 1e12, p1: 1e12,
    B_rad: 1e-16, C_n: 0, C_p: 0, alpha: 1e5,
  })
  return {
    simulation_hints: { min_N_grid: 100 },
    electrical_grid: { interval_weights: { absorber: 2 }, alphas: { absorber: 4 } },
    device: { mode, V_bi: 1.3, Phi: 2.5e22 },
    layers: [layer('p', 'HTL'), layer('i', 'absorber'), layer('n', 'ETL')],
  }
}

let root: HTMLDivElement
beforeEach(() => {
  document.body.replaceChildren()
  root = document.createElement('div')
  document.body.appendChild(root)
})
afterEach(() => document.body.replaceChildren())

function edit(id: string, value: string): void {
  const input = root.querySelector<HTMLInputElement>(`#${id}`)!
  expect(input).not.toBeNull()
  input.value = value
  input.dispatchEvent(new Event('input', { bubbles: true }))
}

describe('ion controls follow the active device mode', () => {
  it('hides the negative species for a Legacy device even when the pane was requested as Full', async () => {
    await mountDevicePanel(root, 'ion-mode', { tier: 'full', initialConfig: config('legacy') })
    expect(root.querySelector('#layer-1-D_ion_neg')).toBeNull()
    expect(root.querySelector<HTMLSelectElement>('#dev-mode')?.value).toBe('legacy')
    expect(root.querySelector<HTMLElement>('#ion-mode-visualizer')?.hidden).toBe(true)
  })

  it('reads every layer after switching from a Full device to a Fast device', async () => {
    const panel = await mountDevicePanel(root, 'ion-mode', {
      tier: 'full', initialConfig: config('full'),
    })
    panel.setConfig(config('fast'))
    edit('layer-1-P0', '0')
    edit('layer-1-D_ion', '0')
    expect(panel.getConfig().layers[1]).toMatchObject({ P0: 0, D_ion: 0 })
    expect(root.querySelector<HTMLElement>('#ion-mode-visualizer')?.hidden).toBe(true)
  })

  it('updates the workspace snapshot on Fast-mode edits and preserves explicit zeros and empty fields', async () => {
    const panel = await mountDevicePanel(root, 'ion-mode', {
      tier: 'fast', initialConfig: config('fast'),
    })
    const changed = vi.fn()
    panel.onChange(changed)
    edit('layer-1-P0', '')
    edit('layer-1-D_ion', '0')
    edit('layer-1-P0_neg', '')
    edit('layer-1-D_ion_neg', '')
    const last = changed.mock.calls.at(-1)![0] as DeviceConfig
    expect(last.layers[1]).toMatchObject({ P0: 0, D_ion: 0 })
    expect(last.layers[1].P0_neg).toBeUndefined()
    expect(last.layers[1].D_ion_neg).toBeUndefined()
    expect(panel.getConfig().layers[1]).toEqual(last.layers[1])
  })

  it('re-gates the negative species when the Mode selector changes', async () => {
    const panel = await mountDevicePanel(root, 'ion-mode', {
      tier: 'fast', initialConfig: config('fast'),
    })
    const mode = root.querySelector<HTMLSelectElement>('#dev-mode')!
    mode.value = 'legacy'
    mode.dispatchEvent(new Event('change', { bubbles: true }))
    expect(root.querySelector('#layer-1-P0_neg')).toBeNull()
    expect(panel.getConfig().device.mode).toBe('legacy')
    expect(panel.getConfig().layers[1].P0_neg).toBe(2e24)
  })

  it('preserves the grid and minimum-resolution contract in Full-mode layer edits', async () => {
    const original = config('full')
    const panel = await mountDevicePanel(root, 'ion-mode', {
      tier: 'full', initialConfig: original,
    })
    edit('layer-0-P0', '3e24')
    const edited = panel.getConfig()
    expect(edited.layers[0].P0).toBe(3e24)
    expect(edited.electrical_grid).toEqual(original.electrical_grid)
    expect(edited.simulation_hints).toEqual(original.simulation_hints)
  })

  it('keeps global optical, temperature and contact inputs reachable in Full mode', async () => {
    const panel = await mountDevicePanel(root, 'ion-mode', { tier: 'full', initialConfig: config('full') })
    expect(root.querySelector('.device-settings')).not.toBeNull()
    edit('dev-Phi', '0')
    root.querySelector<HTMLElement>('.layer-card[data-idx="1"]')!.click()
    edit('layer-1-alpha', '2e5')
    root.querySelector<HTMLElement>('.layer-card[data-idx="0"]')!.click()
    edit('dev-T', '310')
    edit('dev-S-n-top', '0')
    expect(panel.getConfig().device).toMatchObject({ Phi: 0, T: 310, S_n_left: 0, mode: 'full' })
    expect(panel.getConfig().layers[1].alpha).toBe(2e5)
    const mode = root.querySelector<HTMLSelectElement>('#dev-mode')!
    mode.value = 'legacy'
    mode.dispatchEvent(new Event('change', { bubbles: true }))
    expect(panel.getConfig().device.mode).toBe('legacy')
    expect(root.querySelector('#layer-0-P0_neg')).toBeNull()
  })
})
