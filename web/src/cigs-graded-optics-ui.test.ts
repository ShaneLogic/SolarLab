import { describe, expect, it } from 'vitest'
import { computeTmmBadge } from './device-panel'
import { hasTMMOptics } from './device-utils'
import type { DeviceConfig, LayerConfig } from './types'

function cigsDevice(active: boolean): DeviceConfig {
  const layer: LayerConfig = {
    name: 'CIGS', role: 'absorber', thickness: 2e-6, eps_r: 13.6,
    mu_n: 1e-2, mu_p: 1e-3, ni: 1e10, N_D: 0, N_A: 1e22,
    D_ion: 0, P_lim: 0, P0: 0,
    tau_n: 1e-7, tau_p: 1e-7, n1: 1e10, p1: 1e10,
    B_rad: 0, C_n: 0, C_p: 0, alpha: 1e6,
    Eg: 1.15, Eg_back: 1.4,
    cigs_graded_optics: {
      ggi_front: 0.225,
      ggi_back: 0.6,
      cgi: 0.9,
    },
  }
  return {
    device: {
      V_bi: 1,
      Phi: 1e21,
      mode: 'full',
      band_grading: active,
      graded_optics: active,
    },
    layers: [layer],
  }
}

describe('graded CIGS optics UI activation', () => {
  it('recognizes the opt-in model as wavelength-resolved TMM optics', () => {
    const config = cigsDevice(true)
    expect(hasTMMOptics(config)).toBe(true)
    expect(computeTmmBadge(config, 'full')).toContain('TMM active · 1 layer')
  })

  it('keeps the nested model dormant when the master gates are off', () => {
    const config = cigsDevice(false)
    expect(hasTMMOptics(config)).toBe(false)
    expect(computeTmmBadge(config, 'full')).toBe('')
  })

  it('does not expose the FULL-only badge in lower tiers', () => {
    const config = cigsDevice(true)
    expect(computeTmmBadge(config, 'fast')).toBe('')
    expect(computeTmmBadge(config, 'legacy')).toBe('')
  })
})
