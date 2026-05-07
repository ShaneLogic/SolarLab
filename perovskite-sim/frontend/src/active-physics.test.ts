import { describe, it, expect } from 'vitest'
import { describeActivePhysics } from './active-physics'
import type { DeviceConfig, LayerConfig, SimulationModeName } from './types'

function emptyLayer(name: string, role: LayerConfig['role'] = 'absorber'): LayerConfig {
  return {
    name,
    role,
    thickness: 1e-7,
    eps_r: 1, mu_n: 0, mu_p: 0, ni: 0, N_D: 0, N_A: 0,
    D_ion: 0, P_lim: 0, P0: 0,
    tau_n: 0, tau_p: 0, n1: 0, p1: 0,
    B_rad: 0, C_n: 0, C_p: 0, alpha: 0,
  }
}

function makeDevice(
  tier: SimulationModeName,
  overrides: Partial<DeviceConfig['device']> = {},
  layers: LayerConfig[] = [emptyLayer('absorber')],
  microstructure?: unknown,
): DeviceConfig {
  const cfg: DeviceConfig = {
    device: { V_bi: 1.0, Phi: 1e21, mode: tier, ...overrides },
    layers,
  }
  if (microstructure !== undefined) {
    (cfg as DeviceConfig & { microstructure?: unknown }).microstructure = microstructure
  }
  return cfg
}

describe('describeActivePhysics', () => {
  it('returns the baseline sentinel for a LEGACY device with no extras', () => {
    expect(describeActivePhysics(makeDevice('legacy'))).toBe(
      'Active physics: baseline 2D drift-diffusion',
    )
  })

  it('reports Photon recycling on FAST (tier flag, no parameters needed)', () => {
    expect(describeActivePhysics(makeDevice('fast'))).toBe(
      'Active physics: Photon recycling',
    )
  })

  it('reports Photon recycling + Reabsorption on FULL with no other extras', () => {
    expect(describeActivePhysics(makeDevice('full'))).toBe(
      'Active physics: Reabsorption, Photon recycling',
    )
  })

  it('adds TMM when any layer has optical_material set', () => {
    const layers = [
      { ...emptyLayer('substrate', 'substrate' as const), optical_material: 'glass' },
      emptyLayer('absorber'),
    ]
    expect(describeActivePhysics(makeDevice('full', {}, layers))).toBe(
      'Active physics: Reabsorption, Photon recycling, TMM',
    )
  })

  it('adds Microstructure when grain_boundaries is non-empty', () => {
    const ms = {
      grain_boundaries: [{ x_position: 1e-7, width: 5e-9, tau_n: 1e-9, tau_p: 1e-9 }],
    }
    expect(describeActivePhysics(makeDevice('full', {}, [emptyLayer('absorber')], ms))).toBe(
      'Active physics: Microstructure, Reabsorption, Photon recycling',
    )
  })

  it('adds Robin contacts only when FULL tier and a non-zero S is set', () => {
    expect(describeActivePhysics(makeDevice('full', { S_n_left: 1e3 }))).toBe(
      'Active physics: Robin contacts, Reabsorption, Photon recycling',
    )
    expect(describeActivePhysics(makeDevice('full', { S_n_left: 0, S_p_left: 0 }))).toBe(
      'Active physics: Reabsorption, Photon recycling',
    )
  })

  it('does NOT report Robin contacts under FAST even when S is set', () => {
    expect(describeActivePhysics(makeDevice('fast', { S_n_left: 1e3 }))).toBe(
      'Active physics: Photon recycling',
    )
  })

  it('adds μ(E) when FULL tier and a layer has non-zero v_sat or pf_gamma', () => {
    const layers = [{ ...emptyLayer('absorber'), v_sat_n: 1e2 }]
    expect(describeActivePhysics(makeDevice('full', {}, layers))).toBe(
      'Active physics: μ(E), Reabsorption, Photon recycling',
    )
    const layers2 = [{ ...emptyLayer('htl', 'HTL' as const), pf_gamma_p: 3e-4 }]
    expect(describeActivePhysics(makeDevice('full', {}, layers2))).toBe(
      'Active physics: μ(E), Reabsorption, Photon recycling',
    )
  })

  it('handles all four B(c.x) hooks + microstructure + TMM in thematic order', () => {
    const layers = [
      { ...emptyLayer('substrate', 'substrate' as const), optical_material: 'glass' },
      { ...emptyLayer('absorber'), v_sat_n: 1e2 },
    ]
    const ms = { grain_boundaries: [{ x_position: 1e-7 }] }
    const cfg = makeDevice('full', { S_n_left: 1e3, S_p_right: 1e3 }, layers, ms)
    // Order: Microstructure → Robin → μ(E) → Reabsorption → Photon recycling → TMM
    expect(describeActivePhysics(cfg)).toBe(
      'Active physics: Microstructure, Robin contacts, μ(E), Reabsorption, Photon recycling, TMM',
    )
  })

  it('treats undefined mode as full (matches DeviceStack default)', () => {
    const cfg = makeDevice('full')
    delete cfg.device.mode
    expect(describeActivePhysics(cfg)).toBe(
      'Active physics: Reabsorption, Photon recycling',
    )
  })

  it('treats null S values as inactive (sentinel for explicitly disabled)', () => {
    const cfg = makeDevice('full', {
      S_n_left: null, S_p_left: null,
      S_n_right: null, S_p_right: null,
    })
    expect(describeActivePhysics(cfg)).toBe(
      'Active physics: Reabsorption, Photon recycling',
    )
  })
})
