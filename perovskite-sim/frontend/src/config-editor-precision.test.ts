import { beforeEach, expect, it } from 'vitest'
import { renderDeviceEditor, readDeviceEditor } from './config-editor'
import type { DeviceConfig } from './types'

const config: DeviceConfig = {
  device: { Phi: 2.5123456789e21, V_bi: 1.3, mode: 'fast', optical_model: 'beer_lambert_spectral' },
  layers: [{
    name: 'PVK', role: 'absorber', thickness: 8e-7,
    eps_r: 31, mu_n: .002, mu_p: .002, ni: 1.234567891e13,
    N_D: 1e20, N_A: 0, tau_n: 5e-5, tau_p: 5e-5,
    n1: 6.123456789e23, p1: 7.987654321e22,
    B_rad: 1e-18, C_n: 2.3e-41, C_p: 2.3e-41, alpha: 0,
    D_ion: 0, P0: 0, P_lim: 1e30, Nc300: 1.23456789e25,
  }],
}

beforeEach(() => {
  document.body.innerHTML = '<div id="editor"></div>'
  renderDeviceEditor(document.querySelector('#editor')!, config, 'fast')
})

it('does not round untouched preset inputs on submission', () => {
  const result = readDeviceEditor(config)
  for (const key of ['ni', 'n1', 'p1', 'Nc300'] as const) {
    expect(result.layers[0][key]).toBe(config.layers[0][key])
  }
  expect(result.device.Phi).toBe(config.device.Phi)
  expect(result.device.optical_model).toBe('beer_lambert_spectral')
})

it('applies an actual edit without rounding the other fields', () => {
  document.querySelector<HTMLInputElement>('#layer-0-ni')!.value = '2.45678e13'
  const result = readDeviceEditor(config)
  expect(result.layers[0].ni).toBe(2.45678e13)
  expect(result.layers[0].p1).toBe(config.layers[0].p1)
})
