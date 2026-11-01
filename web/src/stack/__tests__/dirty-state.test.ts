import { describe, it, expect } from 'vitest'
import { isDirty } from '../dirty-state'
import type { DeviceConfig } from '../../types'

const baseLayer = {
  name: 'MAPbI3',
  role: 'absorber',
  thickness: 4e-7,
  eps_r: 24.1,
  mu_n: 2e-4, mu_p: 2e-4,
  ni: 1e12, N_D: 0, N_A: 0,
  D_ion: 1e-16, P_lim: 1.6e25, P0: 1.6e25,
  tau_n: 3e-9, tau_p: 3e-9,
  n1: 1e10, p1: 1e10,
  B_rad: 9e-17, C_n: 0, C_p: 0,
  alpha: 0,
} as const

const baseConfig: DeviceConfig = {
  device: { V_bi: 1.1, Phi: 1.4e21, T: 300, mode: 'full', interfaces: [] },
  layers: [{ ...baseLayer }],
}

describe('isDirty', () => {
  it('is false when configs are deeply equal', () => {
    const a = baseConfig
    const b = JSON.parse(JSON.stringify(baseConfig))
    expect(isDirty(a, b)).toBe(false)
  })

  it('is true when a numeric field differs', () => {
    const next = { ...baseConfig, layers: [{ ...baseLayer, thickness: 5e-7 }] }
    expect(isDirty(baseConfig, next)).toBe(true)
  })

  it('is true when a layer is added', () => {
    const next = { ...baseConfig, layers: [...baseConfig.layers, { ...baseLayer, name: 'extra' }] }
    expect(isDirty(baseConfig, next)).toBe(true)
  })

  it('is true when interfaces change', () => {
    const next = { ...baseConfig, device: { ...baseConfig.device, interfaces: [[1, 2]] as Array<[number, number]> } }
    expect(isDirty(baseConfig, next)).toBe(true)
  })

  it('is false regardless of property insertion order', () => {
    const reordered: DeviceConfig = {
      layers: baseConfig.layers,
      device: baseConfig.device,
    }
    expect(isDirty(baseConfig, reordered)).toBe(false)
  })
})
