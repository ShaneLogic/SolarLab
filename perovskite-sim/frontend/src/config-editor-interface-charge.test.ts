import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { readDeviceEditor, renderDeviceEditor } from './config-editor'
import type { DeviceConfig } from './types'

function config(charged: boolean): DeviceConfig {
  return {
    device: {
      V_bi: 1.0,
      Phi: 1e18,
      mode: 'full',
      ...(charged
        ? {
            interface_charge_closure: 'equilibrium_referenced' as const,
            interface_charge_rebaseline_acknowledged: true,
          }
        : {}),
    },
    layers: [{
      name: 'absorber', role: 'absorber', thickness: 1e-7,
      eps_r: 10, mu_n: 1e-3, mu_p: 1e-3,
      D_ion: 0, P_lim: 1e24, P0: 0, ni: 1e12,
      tau_n: 1e-6, tau_p: 1e-6, n1: 1e12, p1: 1e12,
      B_rad: 0, C_n: 0, C_p: 0, alpha: 0, N_A: 0, N_D: 0,
    }],
  }
}

let container: HTMLDivElement

beforeEach(() => {
  document.body.replaceChildren()
  container = document.createElement('div')
  document.body.appendChild(container)
})

afterEach(() => {
  document.body.replaceChildren()
})

describe('interface-charge config preservation', () => {
  it('preserves closure and rebaseline acknowledgement through editor round-trip', () => {
    const original = config(true)
    renderDeviceEditor(container, original, 'full')
    const restored = readDeviceEditor(original)

    expect(restored.device.interface_charge_closure).toBe('equilibrium_referenced')
    expect(restored.device.interface_charge_rebaseline_acknowledged).toBe(true)
  })

  it('does not invent interface-charge fields on a legacy charge-off config', () => {
    const original = config(false)
    renderDeviceEditor(container, original, 'full')
    const restored = readDeviceEditor(original)

    expect(restored.device.interface_charge_closure).toBeUndefined()
    expect(restored.device.interface_charge_rebaseline_acknowledged).toBeUndefined()
  })
})
