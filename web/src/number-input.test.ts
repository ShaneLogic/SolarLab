import { beforeEach, describe, expect, it } from 'vitest'
import { formatNumberInput, readNumberInput, setNumberInputValue } from './number-input'
import { numField, readNum } from './ui-helpers'
import { readDeviceEditor, renderDeviceEditor } from './config-editor'
import type { DeviceConfig, LayerConfig } from './types'

beforeEach(() => { document.body.replaceChildren() })

describe('scientific numeric defaults', () => {
  it.each([
    [0.009999999999999998, '1e-2'],
    [2e-7, '2e-7'],
    [2.5e27, '2.5e27'],
    [300, '3e2'],
    [-0.3, '-3e-1'],
    [0, '0'],
    [-0, '0'],
    [Number.MIN_VALUE, '4.94066e-324'],
    [Number.MAX_VALUE, '1.79769e308'],
  ])('renders %s as a finite HTML number %s', (value, expected) => {
    document.body.innerHTML = numField('number', 'Physical value', value, 'any')
    const input = document.querySelector<HTMLInputElement>('input')!
    expect(input.value).toBe(expected)
    expect(input.validity.valid).toBe(true)
    expect(readNum('number', -1)).toBe(value === 0 ? 0 : value)
  })

  it.each([undefined, null, '', ' ', 'invalid', NaN, Infinity, -Infinity])(
    'leaves an absent or non-finite default empty: %s', value => {
      expect(formatNumberInput(value)).toBe('')
    },
  )

  it('keeps discrete counts and step validation unchanged', () => {
    document.body.innerHTML = numField('count', 'Grid points', 111, '1')
    const input = document.querySelector<HTMLInputElement>('input')!
    expect(input.value).toBe('111')
    expect(input.step).toBe('1')
    expect(readNum('count', 0)).toBe(111)
    document.body.innerHTML = numField('temperature', 'Temperature (K)', 300, '1', 'scientific')
    const temperature = document.querySelector<HTMLInputElement>('input')!
    expect(temperature.value).toBe('3e2')
    expect(temperature.step).toBe('1')
    expect(temperature.validity.valid).toBe(true)
  })

  it('preserves exact defaults until an explicit user edit, even when the display is retyped', () => {
    const exact = 1.234567890123456e-17
    document.body.innerHTML = numField('number', 'Physical value', exact, 'any')
    const input = document.querySelector<HTMLInputElement>('input')!
    expect(input.value).toBe('1.23457e-17')
    expect(readNum('number', 0)).toBe(exact)
    input.dispatchEvent(new Event('input', { bubbles: true }))
    expect(readNum('number', 0)).toBe(1.23457e-17)
    setNumberInputValue(input, exact)
    expect(readNumberInput(input)).toBe(exact)
    input.value = '2.345678901234567e-17'
    input.dispatchEvent(new Event('input', { bubbles: true }))
    expect(readNum('number', 0)).toBe(2.345678901234567e-17)
  })
})

function layer(name: string, role: LayerConfig['role']): LayerConfig {
  return {
    name, role, thickness: 1.234567890123456e-7, eps_r: 20,
    mu_n: 1e-4, mu_p: 1e-4, ni: 1e10, N_D: 0, N_A: 0,
    D_ion: 0, P_lim: 0, P0: 0, tau_n: 1e-6, tau_p: 1e-6,
    n1: 1e10, p1: 1e10, B_rad: 0, C_n: 0, C_p: 0, alpha: 0,
  }
}

describe('device numeric input round trips', () => {
  it('hides SRV floating-point tails while retaining the exact preset after unrelated edits', () => {
    const original: DeviceConfig = {
      device: {
        mode: 'full', Phi: 2.345678901234567e21,
        interfaces: [[0.009999999999999998, 0]],
        S_n_left: 1.234567890123456e5,
      },
      layers: [layer('HTL', 'HTL'), layer('PVK', 'absorber')],
    }
    renderDeviceEditor(document.body, original, 'full')
    expect(document.querySelector<HTMLInputElement>('#iface-0-vn')!.value).toBe('1e-2')
    expect(document.querySelector<HTMLInputElement>('#iface-0-vp')!.value).toBe('0')
    expect(document.querySelector<HTMLInputElement>('#layer-0-thickness')!.value).toBe('1.23457e-7')
    const name = document.querySelector<HTMLInputElement>('#layer-0-name')!
    name.value = 'Renamed HTL'
    name.dispatchEvent(new Event('input', { bubbles: true }))
    const read = readDeviceEditor(original)
    expect(read.layers[0].name).toBe('Renamed HTL')
    expect(read.layers[0].thickness).toBe(original.layers[0].thickness)
    expect(read.device.Phi).toBe(original.device.Phi)
    expect(read.device.interfaces).toEqual(original.device.interfaces)
    expect(read.device.S_n_left).toBe(original.device.S_n_left)

    const srv = document.querySelector<HTMLInputElement>('#iface-0-vn')!
    srv.dispatchEvent(new Event('input', { bubbles: true }))
    expect(readDeviceEditor(read).device.interfaces?.[0][0]).toBe(0.01)
    srv.value = '3.141592653589793e-2'
    srv.dispatchEvent(new Event('input', { bubbles: true }))
    expect(readDeviceEditor(read).device.interfaces?.[0][0]).toBe(3.141592653589793e-2)
  })
})
