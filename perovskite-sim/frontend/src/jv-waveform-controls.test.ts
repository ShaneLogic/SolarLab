import { beforeEach, describe, expect, it } from 'vitest'
import { mountJVWaveformControls } from './jv-waveform-controls'
import { readNumberInput } from './number-input'
import type { JVSweepDefaults } from './types'

beforeEach(() => { document.body.innerHTML = '<input id="rate" value="1"><input id="maximum" value="1.4"><input id="points" value="30"><div id="root"></div>' })

function mount() {
  return mountJVWaveformControls(document.querySelector('#root')!, 'test', {
    rate: document.querySelector('#rate')!, maximum: document.querySelector('#maximum')!,
    points: document.querySelector('#points')!,
  })
}

function change(name: string, value: string) {
  const element = document.querySelector<HTMLInputElement | HTMLSelectElement>(`#test-waveform-${name}`)!
  element.value = value
  element.dispatchEvent(new Event('change'))
}

describe('explicit waveform controls', () => {
  it('leaves the existing default request unchanged', () => {
    const control = mount()
    expect(control.read()).toEqual({})
    expect(document.querySelector<HTMLElement>('#test-waveform-fields')!.hidden).toBe(true)
  })
  it('applies the history template without altering device parameters', () => {
    const control = mount()
    change('mode', 'continuous')
    expect(document.querySelector<HTMLInputElement>('#rate')!.value).toBe('4e-2')
    expect(document.querySelector<HTMLInputElement>('#points')!.value).toBe('111')
    expect(control.read().waveform).toMatchObject({
      start_voltage_V: -1, dark_seed_s: 120, dark_prep_s: 30,
      branch_dwell_s: 0.5, turnaround_s: 0, uniform_generation_rate_m3_s: 2.5e27,
    })
  })
  it('keeps zero generation distinct from device optics', () => {
    const control = mount()
    change('mode', 'continuous')
    change('G', '0')
    expect(control.read().waveform?.uniform_generation_rate_m3_s).toBe(0)
    change('generation', 'device')
    expect(control.read().waveform?.uniform_generation_rate_m3_s).toBeNull()
    expect(document.querySelector<HTMLInputElement>('#test-waveform-G')!.disabled).toBe(true)
  })
  it('uses the validated research tolerance and preserves explicit overrides', () => {
    const control = mount()
    change('mode', 'continuous')
    expect(control.read().waveform_controls).toEqual({ rtol: 1e-4, atol_m3: 100 })
    change('atol', '1')
    expect(control.read().waveform_controls).toEqual({ rtol: 1e-4, atol_m3: 1 })
    change('mode', 'standard')
    expect(control.read()).toEqual({})
    change('mode', 'continuous')
    expect(control.read().waveform_controls?.atol_m3).toBe(1)
  })
  it('rejects empty numeric history fields', () => {
    const control = mount()
    change('mode', 'continuous')
    change('prep', '')
    expect(() => control.read()).toThrow('required')
  })
  it('restores standard inputs when an incompatible solver disables the waveform', () => {
    const control = mount()
    change('mode', 'continuous')
    control.setEnabled(false)
    expect(control.read()).toEqual({})
    expect(document.querySelector<HTMLInputElement>('#rate')!.value).toBe('1')
    expect(document.querySelector<HTMLSelectElement>('#test-waveform-mode')!.disabled).toBe(true)
  })
  it('formats Calado defaults and preserves their precision through history switches', () => {
    const control = mount()
    const defaults: JVSweepDefaults = {
      N_grid: 60, n_points: 111, v_rate: 0.04012345678901234, V_max: 1.234567890123456,
      waveform: {
        schema_version: 1, start_voltage_V: -1, dark_seed_s: 120, dark_prep_s: 30,
        branch_dwell_s: 0.5123456789012345, turnaround_s: 3, turnaround_dark: true,
        uniform_generation_rate_m3_s: 2.567890123456789e27,
      },
      waveform_controls: { rtol: 1.234567890123456e-4, atol_m3: 1 },
    }
    control.applyDefaults(defaults)
    expect(document.querySelector<HTMLInputElement>('#rate')!.value).toBe('4.01235e-2')
    expect(document.querySelector<HTMLInputElement>('#points')!.value).toBe('111')
    expect(document.querySelector<HTMLInputElement>('#test-waveform-rtol')!.value).toBe('1.23457e-4')
    expect(control.read()).toEqual({ waveform: defaults.waveform, waveform_controls: defaults.waveform_controls })
    change('mode', 'standard')
    expect(document.querySelector<HTMLInputElement>('#rate')!.value).toBe('1')
    change('mode', 'continuous')
    expect(readNumberInput(document.querySelector<HTMLInputElement>('#rate')!)).toBe(defaults.v_rate)
    expect(readNumberInput(document.querySelector<HTMLInputElement>('#maximum')!)).toBe(defaults.V_max)
    expect(control.read()).toEqual({ waveform: defaults.waveform, waveform_controls: defaults.waveform_controls })
    const dwell = document.querySelector<HTMLInputElement>('#test-waveform-dwell')!
    dwell.value = '7.5e-1'
    dwell.dispatchEvent(new Event('input', { bubbles: true }))
    expect(control.read().waveform?.branch_dwell_s).toBe(0.75)
  })
})
