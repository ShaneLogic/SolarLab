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
})
