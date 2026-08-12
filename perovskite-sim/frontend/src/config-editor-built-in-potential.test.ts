import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { readDeviceEditor, renderDeviceEditor } from './config-editor'
import type { DeviceConfig, LayerConfig } from './types'


function layer(name: string, role: LayerConfig['role']): LayerConfig {
  return {
    name,
    role,
    thickness: 1e-7,
    eps_r: 10,
    mu_n: 1e-4,
    mu_p: 1e-4,
    ni: 1e10,
    N_D: role === 'ETL' ? 1e22 : 0,
    N_A: role === 'HTL' ? 1e22 : 0,
    D_ion: 0,
    P_lim: 1e30,
    P0: 0,
    tau_n: 1e-6,
    tau_p: 1e-6,
    n1: 1e10,
    p1: 1e10,
    B_rad: 0,
    C_n: 0,
    C_p: 0,
    alpha: 0,
    chi: role === 'HTL' ? 2.2 : 4.0,
    Eg: role === 'HTL' ? 3.0 : 2.0,
    Nc300: 1e25,
    Nv300: 1e25,
  }
}


function config(device: Partial<DeviceConfig['device']> = {}): DeviceConfig {
  return {
    device: { V_bi: 1.1, Phi: 2.5e21, mode: 'full', ...device },
    layers: [layer('HTL', 'HTL'), layer('ETL', 'ETL')],
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


describe('built-in-potential source editor', () => {
  it('shows the legacy override only for an old compatibility config', () => {
    renderDeviceEditor(container, config(), 'full')

    const select = document.getElementById('dev-vbi-mode') as HTMLSelectElement
    const legacy = container.querySelector<HTMLElement>('[data-vbi-mode="legacy_manual"]')!
    const metal = container.querySelector<HTMLElement>('[data-vbi-mode="metal_work_function"]')!
    expect(select.value).toBe('legacy_manual')
    expect(legacy.hidden).toBe(false)
    expect(metal.hidden).toBe(true)
  })

  it('preserves an old V_bi payload until its mode is deliberately changed', () => {
    const original = config()
    renderDeviceEditor(container, original, 'full')

    const out = readDeviceEditor(original)
    expect(out.device.V_bi).toBe(1.1)
    expect(out.device.built_in_potential_mode).toBeUndefined()
    expect(out.device.V_bi_override).toBeUndefined()
  })

  it('preserves an old flat-band payload instead of silently migrating it', () => {
    const original = config({ flat_band_contacts: true })
    renderDeviceEditor(container, original, 'full')

    const select = document.getElementById('dev-vbi-mode') as HTMLSelectElement
    expect(select.value).toBe('legacy_manual')
    const out = readDeviceEditor(original)
    expect(out.device.V_bi).toBe(1.1)
    expect(out.device.built_in_potential_mode).toBeUndefined()
    expect(out.device.flat_band_contacts).toBe(true)
  })

  it('defaults a new payload with no manual key to semiconductor work functions', () => {
    const original = config({ V_bi: undefined })
    renderDeviceEditor(container, original, 'full')

    const select = document.getElementById('dev-vbi-mode') as HTMLSelectElement
    expect(select.value).toBe('semiconductor_work_function')
    const out = readDeviceEditor(original)
    expect(out.device.built_in_potential_mode).toBe('semiconductor_work_function')
    expect(out.device.V_bi).toBeUndefined()
  })

  it('switches to the fail-closed semiconductor-work-function mode', () => {
    const original = config()
    renderDeviceEditor(container, original, 'full')
    const select = document.getElementById('dev-vbi-mode') as HTMLSelectElement
    select.value = 'semiconductor_work_function'
    select.dispatchEvent(new Event('change'))

    const out = readDeviceEditor(original)
    expect(out.device.built_in_potential_mode).toBe('semiconductor_work_function')
    expect(out.device.V_bi).toBeUndefined()
    expect(out.device.V_bi_override).toBeUndefined()
    expect(out.layers[0].Nc300).toBe(1e25)
    expect(out.layers[0].Nv300).toBe(1e25)
  })

  it('round-trips explicit left and right metal work functions', () => {
    const original = config({
      V_bi: undefined,
      built_in_potential_mode: 'metal_work_function',
      work_function_left_eV: 5.2,
      work_function_right_eV: 4.1,
    })
    renderDeviceEditor(container, original, 'full')

    expect((document.getElementById('dev-W-left') as HTMLInputElement).value).toBe('5.2')
    expect((document.getElementById('dev-W-right') as HTMLInputElement).value).toBe('4.1')
    const out = readDeviceEditor(original)
    expect(out.device.built_in_potential_mode).toBe('metal_work_function')
    expect(out.device.work_function_left_eV).toBe(5.2)
    expect(out.device.work_function_right_eV).toBe(4.1)
    expect(out.device.V_bi).toBeUndefined()
  })

  it('uses V_bi_override for an explicitly declared manual mode', () => {
    const original = config({
      V_bi: undefined,
      built_in_potential_mode: 'legacy_manual',
      V_bi_override: 0.98,
    })
    renderDeviceEditor(container, original, 'full')

    const out = readDeviceEditor(original)
    expect(out.device.built_in_potential_mode).toBe('legacy_manual')
    expect(out.device.V_bi_override).toBe(0.98)
    expect(out.device.V_bi).toBeUndefined()
  })
})
