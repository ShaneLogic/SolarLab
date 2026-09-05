import { beforeEach, describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
  getConfig: vi.fn(),
  listConfigs: vi.fn(),
  renderDeviceEditor: vi.fn(),
  readDeviceEditor: vi.fn(),
}))

vi.mock('./api', () => ({
  fetchOpticalMaterials: vi.fn().mockResolvedValue([]),
  fetchLayerTemplates: vi.fn().mockResolvedValue({}),
  getConfig: mocks.getConfig,
  listConfigs: mocks.listConfigs,
}))

vi.mock('./config-editor', () => ({
  renderDeviceEditor: mocks.renderDeviceEditor,
  readDeviceEditor: mocks.readDeviceEditor,
  setOpticalMaterialOptions: vi.fn(),
}))

import { mountDevicePanel } from './device-panel'
import type { DeviceConfig } from './types'

function config(closure: 'off' | 'equilibrium_referenced'): DeviceConfig {
  return {
    device: {
      Phi: 1e18,
      interface_charge_closure: closure,
      ...(closure === 'equilibrium_referenced'
        ? { interface_charge_rebaseline_acknowledged: true }
        : {}),
    },
    layers: [],
  }
}

describe('DevicePanel workspace-config ownership', () => {
  beforeEach(() => {
    document.body.replaceChildren()
    mocks.getConfig.mockReset()
    mocks.listConfigs.mockReset().mockResolvedValue([
      { name: 'ionmonger_benchmark.yaml', namespace: 'shipped' },
    ])
    mocks.renderDeviceEditor.mockReset()
    mocks.readDeviceEditor.mockReset().mockImplementation((cfg: DeviceConfig) => cfg)
  })

  it('mounts the active workspace snapshot without loading the default preset', async () => {
    const initial = config('equilibrium_referenced')
    const root = document.createElement('div')
    document.body.appendChild(root)

    const panel = await mountDevicePanel(root, 'device-test', {
      tier: 'legacy',
      initialConfig: initial,
    })

    expect(mocks.getConfig).not.toHaveBeenCalled()
    expect(panel.getConfig()).toEqual(initial)
    expect(root.querySelector<HTMLSelectElement>('#device-test-config-select')?.value)
      .toBe('__workspace_snapshot__')
  })

  it('replaces the editor snapshot when the active workspace device changes', async () => {
    const root = document.createElement('div')
    document.body.appendChild(root)
    const panel = await mountDevicePanel(root, 'device-test', {
      tier: 'legacy',
      initialConfig: config('off'),
    })
    const charged = config('equilibrium_referenced')

    panel.setConfig(charged)

    expect(panel.getConfig()).toEqual(charged)
    expect(mocks.getConfig).not.toHaveBeenCalled()
  })

  it('limits the picker to the two studies and user presets while keeping the workspace snapshot', async () => {
    mocks.listConfigs.mockResolvedValue([
      { name: 'cigs_baseline.yaml', namespace: 'shipped' },
      { name: 'calado2016_fig1f.yaml', namespace: 'shipped' },
      { name: 'scaps_mirror_v2.yaml', namespace: 'shipped' },
      { name: 'my_ion_scan.yaml', namespace: 'user' },
    ])
    const root = document.createElement('div')
    document.body.appendChild(root)
    const panel = await mountDevicePanel(root, 'device-test', {
      tier: 'legacy', initialConfig: config('off'),
    })
    const options = Array.from(root.querySelectorAll<HTMLOptionElement>('option'))
    expect(options.map(o => o.value)).toEqual([
      '__workspace_snapshot__', 'scaps_mirror_v2.yaml', 'calado2016_fig1f.yaml', 'my_ion_scan.yaml',
    ])
    expect(options[1].textContent).toBe('SCAPS parity - Reference v2')
    expect(options[2].textContent).toBe('Calado 2016 - Ion hysteresis')
    expect(panel.getConfig()).toEqual(config('off'))
    expect(mocks.getConfig).not.toHaveBeenCalled()
  })

  it('loads the SCAPS reference as the default when there is no workspace snapshot', async () => {
    mocks.listConfigs.mockResolvedValue([
      { name: 'calado2016_fig1f.yaml', namespace: 'shipped' },
      { name: 'scaps_mirror_v2.yaml', namespace: 'shipped' },
      { name: 'ionmonger_benchmark.yaml', namespace: 'shipped' },
    ])
    mocks.getConfig.mockResolvedValue(config('off'))
    const root = document.createElement('div')
    document.body.appendChild(root)
    await mountDevicePanel(root, 'device-test', { tier: 'legacy' })
    expect(mocks.getConfig).toHaveBeenCalledWith('scaps_mirror_v2.yaml')
  })
})
