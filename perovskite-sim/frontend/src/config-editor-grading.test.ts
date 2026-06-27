/**
 * vitest — the FULL-tier per-layer continuous-bandgap-grading editor group.
 *
 * Pins: render (field IDs present), tier gating (hidden in FAST/LEGACY), the
 * graded round-trip (Eg_back/chi_back/profile carried back), and the
 * clean-payload contract — an ungraded layer carries NO grading keys, so the
 * numeric-optional sentinel never mis-marks a layer graded (empty != 0).
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { renderDeviceEditor, readDeviceEditor } from './config-editor'
import type { DeviceConfig, LayerConfig } from './types'

function emptyLayer(name: string, role: LayerConfig['role']): LayerConfig {
  return {
    name,
    role,
    thickness: 1e-7,
    eps_r: 1, mu_n: 0, mu_p: 0, ni: 1e10, N_D: 0, N_A: 0,
    D_ion: 0, P_lim: 0, P0: 0,
    tau_n: 1e-6, tau_p: 1e-6, n1: 1e10, p1: 1e10,
    B_rad: 0, C_n: 0, C_p: 0, alpha: 0,
    chi: 4.0, Eg: 1.5,
  }
}

function cfg(layerExtras: Partial<LayerConfig> = {}): DeviceConfig {
  return {
    device: { V_bi: 1.3, Phi: 2.5e21, mode: 'full' },
    layers: [
      emptyLayer('HTL', 'HTL'),
      { ...emptyLayer('PVK', 'absorber'), ...layerExtras },
      emptyLayer('ETL', 'ETL'),
    ],
  }
}

let container: HTMLElement

beforeEach(() => {
  document.body.replaceChildren()
  container = document.createElement('div')
  document.body.appendChild(container)
})

afterEach(() => {
  document.body.replaceChildren()
})

describe('grading group rendering + tier gating', () => {
  it('renders the per-layer grading fields in FULL drill-down', () => {
    renderDeviceEditor(container, cfg(), 'full', 1)
    for (const key of ['Eg_back', 'chi_back', 'grading_profile', 'grading_direction',
                        'grading_bowing', 'grading_char_length', 'grading_N_mult']) {
      expect(document.getElementById(`layer-1-${key}`), `missing layer-1-${key}`).not.toBeNull()
    }
  })

  it('hides grading fields in FAST tier', () => {
    renderDeviceEditor(container, cfg(), 'fast', 1)
    expect(document.getElementById('layer-1-Eg_back')).toBeNull()
  })

  it('hides grading fields in LEGACY tier', () => {
    renderDeviceEditor(container, cfg(), 'legacy', 1)
    expect(document.getElementById('layer-1-Eg_back')).toBeNull()
  })
})

describe('grading round-trip', () => {
  it('round-trips a graded absorber (Eg_back/chi_back/profile)', () => {
    const c = cfg()
    renderDeviceEditor(container, c, 'full', 1)
    ;(document.getElementById('layer-1-Eg_back') as HTMLInputElement).value = '1.7'
    ;(document.getElementById('layer-1-chi_back') as HTMLInputElement).value = '3.9'
    ;(document.getElementById('layer-1-grading_profile') as HTMLSelectElement).value = 'parabolic'
    const out = readDeviceEditor(c, 1)
    expect(out.layers[1].Eg_back).toBe(1.7)
    expect(out.layers[1].chi_back).toBe(3.9)
    expect(out.layers[1].grading_profile).toBe('parabolic')
  })

  it('preserves a loaded grade through a non-FULL round-trip', () => {
    const c = cfg({ Eg_back: 1.8, grading_profile: 'exponential', grading_char_length: 2e-8 })
    renderDeviceEditor(container, c, 'fast', 1) // fields hidden → must fall back
    const out = readDeviceEditor(c, 1)
    expect(out.layers[1].Eg_back).toBe(1.8)
    expect(out.layers[1].grading_profile).toBe('exponential')
    expect(out.layers[1].grading_char_length).toBe(2e-8)
  })
})

describe('clean-payload contract (numeric-optional sentinel)', () => {
  it('ungraded layer carries NO grading keys (empty != 0)', () => {
    const c = cfg()
    renderDeviceEditor(container, c, 'full', 1)
    // leave Eg_back / chi_back empty
    const out = readDeviceEditor(c, 1)
    const layer = out.layers[1] as unknown as Record<string, unknown>
    for (const key of ['Eg_back', 'chi_back', 'grading_profile', 'grading_direction',
                       'grading_bowing', 'grading_char_length', 'grading_N_mult']) {
      expect(key in layer, `spurious ${key}`).toBe(false)
    }
  })

  it('clearing Eg_back un-grades the layer (strips grading spec)', () => {
    const c = cfg({ Eg_back: 1.8, grading_profile: 'parabolic' })
    renderDeviceEditor(container, c, 'full', 1)
    ;(document.getElementById('layer-1-Eg_back') as HTMLInputElement).value = ''
    const out = readDeviceEditor(c, 1)
    const layer = out.layers[1] as unknown as Record<string, unknown>
    expect('Eg_back' in layer).toBe(false)
    expect('grading_profile' in layer).toBe(false)
  })
})
