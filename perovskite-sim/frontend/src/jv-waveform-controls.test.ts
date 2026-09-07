import { beforeEach, describe, expect, it } from 'vitest'
import { mountJVWaveformControls } from './jv-waveform-controls'

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
    expect(document.querySelector<HTMLInputElement>('#rate')!.value).toBe('0.04')
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
})
