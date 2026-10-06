/**
 * vitest — the ionic Arrhenius activation energy, and the shared-site
 * checkbox's dependency on the steric form.
 *
 * Two gaps left by the dual-ion editor work:
 *
 *  1. `E_a_ion` was configurable only by hand-editing YAML. It is a per-layer
 *     field (config_loader.py:410, default 0.58 eV) and it is what sets the
 *     temperature dependence of BOTH ionic species — mol.py:1819 and 1823 pass
 *     the same `p.E_a_ion` to `D_ion_at_T` for D_ion and D_ion_neg. Scanning a
 *     temperature dependence from the UI was therefore impossible.
 *
 *  2. The shared-site checkbox reads as unconditional, but jv_sweep.py:940
 *     computes `shared = ion_steric_diffusion_only and ion_steric_shared_site
 *     and has_dual_ions and ...`. With the diffusion-only steric form off, the
 *     box silently does nothing.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { renderDeviceEditor, readDeviceEditor } from './config-editor'
import { isFieldVisible } from './workstation/tier-gating'
import type { DeviceConfig, LayerConfig } from './types'

function layer(name: string, role: LayerConfig['role'], extras: Partial<LayerConfig> = {}): LayerConfig {
  return {
    name,
    role,
    thickness: 1e-7,
    eps_r: 1, mu_n: 0, mu_p: 0, ni: 1e10, N_D: 0, N_A: 0,
    D_ion: 0, P_lim: 0, P0: 0,
    tau_n: 1e-6, tau_p: 1e-6, n1: 1e10, p1: 1e10,
    B_rad: 0, C_n: 0, C_p: 0, alpha: 0,
    ...extras,
  }
}

function cfg(
  device: Partial<DeviceConfig['device']> = {},
  absorber: Partial<LayerConfig> = {},
): DeviceConfig {
  return {
    device: { V_bi: 1.3, Phi: 2.5e21, mode: 'full', ...device },
    layers: [layer('HTL', 'HTL'), layer('PVK', 'absorber', absorber), layer('ETL', 'ETL')],
  }
}

let container: HTMLElement

beforeEach(() => {
  document.body.replaceChildren()
  container = document.createElement('div')
  document.body.appendChild(container)
})
afterEach(() => { document.body.replaceChildren() })

describe('E_a_ion — ionic Arrhenius activation energy', () => {
  it('is gated like use_temperature_scaling: off in LEGACY, on in FAST and FULL', () => {
    expect(isFieldVisible('E_a_ion', 'legacy')).toBe(false)
    expect(isFieldVisible('E_a_ion', 'fast')).toBe(true)
    expect(isFieldVisible('E_a_ion', 'full')).toBe(true)
  })

  it('renders an input on every layer in FULL', () => {
    renderDeviceEditor(container, cfg(), 'full')
    expect(container.querySelector('#layer-1-E_a_ion')).not.toBeNull()
    expect(container.querySelector('#layer-0-E_a_ion')).not.toBeNull()
  })

  it('does not render in LEGACY, where temperature scaling is off', () => {
    renderDeviceEditor(container, cfg({ mode: 'legacy' }), 'legacy')
    expect(container.querySelector('#layer-1-E_a_ion')).toBeNull()
  })

  it('round-trips a typed value', () => {
    const original = cfg({}, { D_ion: 1e-16, P0: 1e24 })
    renderDeviceEditor(container, original, 'full')
    const input = container.querySelector<HTMLInputElement>('#layer-1-E_a_ion')!
    input.value = '0.45'
    expect(readDeviceEditor(original).layers[1].E_a_ion).toBeCloseTo(0.45, 10)
  })

  it('stays absent when untouched, so existing configs keep the 0.58 default', () => {
    const original = cfg({}, { D_ion: 1e-16, P0: 1e24 })
    renderDeviceEditor(container, original, 'full')
    expect(readDeviceEditor(original).layers[1].E_a_ion).toBeUndefined()
  })

  it('accepts an explicit 0 — a real choice meaning no temperature dependence', () => {
    const original = cfg({}, { D_ion: 1e-16, P0: 1e24 })
    renderDeviceEditor(container, original, 'full')
    container.querySelector<HTMLInputElement>('#layer-1-E_a_ion')!.value = '0'
    expect(readDeviceEditor(original).layers[1].E_a_ion).toBe(0)
  })

  it('preserves a value across a LEGACY round-trip, where the field is hidden', () => {
    const original = cfg({ mode: 'legacy' }, { E_a_ion: 0.31 })
    renderDeviceEditor(container, original, 'legacy')
    expect(readDeviceEditor(original).layers[1].E_a_ion).toBe(0.31)
  })
})

describe('shared-site checkbox depends on the diffusion-only steric form', () => {
  it('is enabled when ion_steric_diffusion_only is on (the backend default)', () => {
    renderDeviceEditor(container, cfg(), 'full')
    const box = container.querySelector<HTMLInputElement>('#dev-ion-shared-site')!
    expect(box.disabled).toBe(false)
  })

  it('is disabled when ion_steric_diffusion_only is explicitly off', () => {
    renderDeviceEditor(container, cfg({ ion_steric_diffusion_only: false }), 'full')
    const box = container.querySelector<HTMLInputElement>('#dev-ion-shared-site')!
    expect(box.disabled).toBe(true)
  })

  it('says why it is disabled rather than just greying out', () => {
    renderDeviceEditor(container, cfg({ ion_steric_diffusion_only: false }), 'full')
    const label = container.querySelector<HTMLElement>('#dev-ion-shared-site')!.closest('label')!
    expect(label.title).toMatch(/ion_steric_diffusion_only/)
    expect(label.title.toLowerCase()).toMatch(/no effect|inactive|does nothing/)
  })

  it('preserves the stored value through a round-trip while disabled', () => {
    // A disabled box must not be read as user intent. The stored value wins.
    const original = cfg({ ion_steric_diffusion_only: false, ion_steric_shared_site: false })
    renderDeviceEditor(container, original, 'full')
    expect(readDeviceEditor(original).device.ion_steric_shared_site).toBe(false)
  })

  it('does not invent a false when disabled and the stored value was true', () => {
    const original = cfg({ ion_steric_diffusion_only: false })
    renderDeviceEditor(container, original, 'full')
    expect(readDeviceEditor(original).device.ion_steric_shared_site).toBeUndefined()
  })
})
