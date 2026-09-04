/**
 * vitest — the second (negative) mobile ionic species.
 *
 * The solver has carried a full dual-ion path since the 2026-07 closed-loop
 * work (symmetric neg-species flux, shared-site crowding on the total
 * occupancy, 4N state vector), and `material_params_from_dict` — the parser
 * the inline-device backend path shares with the YAML loader — already reads
 * D_ion_neg / P0_neg / P_lim_neg. Only the editor was missing: the fields
 * could be carried through but not entered.
 *
 * Pins: the three per-layer fields render and round-trip, the tier gate
 * matches `use_dual_ions` in mode.py (off in LEGACY, ON in FAST and FULL),
 * the payload stays clean for single-species configs, and the device-level
 * shared-site assumption is expressible.
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

describe('dual-ion tier gate matches mode.py use_dual_ions', () => {
  const KEYS = ['D_ion_neg', 'P0_neg', 'P_lim_neg']

  it('visible in FULL', () => {
    for (const k of KEYS) expect(isFieldVisible(k, 'full')).toBe(true)
  })

  it('visible in FAST — use_dual_ions is on in that tier', () => {
    for (const k of KEYS) expect(isFieldVisible(k, 'fast')).toBe(true)
  })

  it('hidden in LEGACY — use_dual_ions is off there', () => {
    for (const k of KEYS) expect(isFieldVisible(k, 'legacy')).toBe(false)
  })

  it('carries no field that the physics model does not define', () => {
    // E_a_ion_neg never existed: parameters.py has a single E_a_ion.
    expect(isFieldVisible('E_a_ion_neg', 'legacy')).toBe(true)
  })
})

describe('per-layer negative-ion fields', () => {
  it('renders the three inputs on every layer in FULL', () => {
    renderDeviceEditor(container, cfg(), 'full')
    for (const key of ['D_ion_neg', 'P0_neg', 'P_lim_neg']) {
      expect(container.querySelector(`#layer-1-${key}`)).not.toBeNull()
    }
  })

  it('does not render them in LEGACY', () => {
    renderDeviceEditor(container, cfg({ mode: 'legacy' }), 'legacy')
    expect(container.querySelector('#layer-1-D_ion_neg')).toBeNull()
  })

  it('round-trips a value typed into the editor', () => {
    const original = cfg({}, { D_ion: 1e-16, P0: 1e24, P_lim: 1e27 })
    renderDeviceEditor(container, original, 'full')
    const input = container.querySelector<HTMLInputElement>('#layer-1-D_ion_neg')!
    input.value = '2.5e-17'
    const out = readDeviceEditor(original)
    expect(out.layers[1].D_ion_neg).toBeCloseTo(2.5e-17, 25)
  })

  it('keeps a single-species payload clean', () => {
    const original = cfg({}, { D_ion: 1e-16, P0: 1e24 })
    renderDeviceEditor(container, original, 'full')
    const out = readDeviceEditor(original)
    expect(out.layers[1].D_ion_neg).toBeUndefined()
    expect(out.layers[1].P0_neg).toBeUndefined()
    expect(out.layers[1].P_lim_neg).toBeUndefined()
  })

  it('preserves values on a LEGACY round-trip, where the fields are hidden', () => {
    const original = cfg({ mode: 'legacy' }, { D_ion: 1e-16, D_ion_neg: 3e-17, P0_neg: 5e23 })
    renderDeviceEditor(container, original, 'legacy')
    const out = readDeviceEditor(original)
    expect(out.layers[1].D_ion_neg).toBe(3e-17)
    expect(out.layers[1].P0_neg).toBe(5e23)
  })
})

describe('device-level shared-site assumption', () => {
  it('renders a control in FULL', () => {
    renderDeviceEditor(container, cfg(), 'full')
    expect(container.querySelector('#dev-ion-shared-site')).not.toBeNull()
  })

  it('serialises an explicit false when unchecked, because the backend default is true', () => {
    const original = cfg()
    renderDeviceEditor(container, original, 'full')
    const box = container.querySelector<HTMLInputElement>('#dev-ion-shared-site')!
    expect(box.checked).toBe(true)
    box.checked = false
    const out = readDeviceEditor(original)
    expect(out.device.ion_steric_shared_site).toBe(false)
  })

  it('leaves the payload clean when the box is left at its default', () => {
    const original = cfg()
    renderDeviceEditor(container, original, 'full')
    const out = readDeviceEditor(original)
    expect(out.device.ion_steric_shared_site).toBeUndefined()
  })
})
