import { checkField, numField } from './ui-helpers'
import type { JVSweepDefaults, JVWaveform, JVWaveformEvidence } from './types'

export function mountJVWaveformControls(
  container: HTMLElement,
  prefix: string,
  shared: { rate: HTMLInputElement; maximum: HTMLInputElement; points: HTMLInputElement },
) {
  const id = (name: string) => `${prefix}-waveform-${name}`
  container.classList.add('jv-waveform-controls')
  container.innerHTML = `
    <label class="form-group"><span>Scan history</span>
      <select id="${id('mode')}">
        <option value="standard">Standard staircase</option>
        <option value="continuous">Calado 2016 history (research)</option>
      </select>
    </label>
    <div class="form-grid jv-waveform-fields" id="${id('fields')}" hidden>
      ${numField(id('start'), 'V<sub>start</sub> (V)', -1, 'any')}
      ${numField(id('seed'), 'Dark 0 V seed (s)', 120, 'any')}
      ${numField(id('prep'), 'Dark prebias (s)', 30, 'any')}
      ${numField(id('dwell'), 'Branch-start dwell (s)', 0.5, 'any')}
      ${numField(id('turnaround'), 'High-bias hold (s)', 0, 'any')}
      ${checkField(id('dark'), 'Dark high-bias hold', true)}
      <label class="form-group"><span>Generation source</span>
        <select id="${id('generation')}">
          <option value="uniform">Uniform absorber</option>
          <option value="device">Device optics</option>
        </select>
      </label>
      ${numField(id('G'), 'G (m<sup>-3</sup> s<sup>-1</sup>)', '2.5e27', 'any')}
      ${numField(id('rtol'), 'Relative tolerance', '1e-4', 'any')}
      ${numField(id('atol'), 'Density atol (m<sup>-3</sup>)', 100, 'any')}
    </div>`
  const input = (name: string) => container.querySelector<HTMLInputElement>(`#${id(name)}`)!
  const select = (name: string) => container.querySelector<HTMLSelectElement>(`#${id(name)}`)!
  const mode = select('mode')
  const fields = container.querySelector<HTMLElement>(`#${id('fields')}`)!
  const readShared = () => [shared.rate.value, shared.maximum.value, shared.points.value]
  const writeShared = (values: string[]) => {
    ;[shared.rate.value, shared.maximum.value, shared.points.value] = values
  }
  let standardValues = readShared()
  let continuousValues = ['0.04', '1.2', '111']
  const syncVisibility = () => {
    const active = mode.value === 'continuous'
    fields.hidden = !active
    fields.querySelectorAll<HTMLInputElement | HTMLSelectElement>('input, select')
      .forEach(field => { field.disabled = !active })
    input('G').disabled = !active || select('generation').value !== 'uniform'
  }
  const sync = () => {
    const active = mode.value === 'continuous'
    if (active) {
      standardValues = readShared()
      writeShared(continuousValues)
    } else {
      continuousValues = readShared()
      writeShared(standardValues)
    }
    syncVisibility()
  }
  mode.addEventListener('change', sync)
  select('generation').addEventListener('change', () => {
    input('G').disabled = select('generation').value !== 'uniform'
  })
  fields.querySelectorAll<HTMLInputElement | HTMLSelectElement>('input, select')
    .forEach(field => { field.disabled = true })
  const number = (name: string) => {
    const field = input(name)
    if (field.value.trim() === '' || !Number.isFinite(Number(field.value))) {
      throw new Error(`${field.closest('label')?.textContent?.trim() ?? name} is required`)
    }
    return Number(field.value)
  }
  return {
    applyDefaults(defaults: JVSweepDefaults | undefined) {
      if (!defaults) {
        if (mode.value !== 'standard') {
          mode.value = 'standard'
          sync()
        }
        return
      }
      if (mode.value === 'standard') standardValues = readShared()
      continuousValues = [String(defaults.v_rate), String(defaults.V_max), String(defaults.n_points)]
      mode.value = 'continuous'
      writeShared(continuousValues)
      const waveform = defaults.waveform
      input('start').value = String(waveform.start_voltage_V)
      input('seed').value = String(waveform.dark_seed_s)
      input('prep').value = String(waveform.dark_prep_s)
      input('dwell').value = String(waveform.branch_dwell_s)
      input('turnaround').value = String(waveform.turnaround_s)
      input('dark').checked = waveform.turnaround_dark
      select('generation').value = waveform.uniform_generation_rate_m3_s === null ? 'device' : 'uniform'
      if (waveform.uniform_generation_rate_m3_s !== null) {
        input('G').value = String(waveform.uniform_generation_rate_m3_s)
      }
      input('rtol').value = String(defaults.waveform_controls.rtol)
      input('atol').value = String(defaults.waveform_controls.atol_m3)
      syncVisibility()
    },
    setEnabled(enabled: boolean) {
      if (!enabled && mode.value !== 'standard') {
        mode.value = 'standard'
        sync()
      }
      mode.disabled = !enabled
    },
    read(): { waveform?: JVWaveform; waveform_controls?: { rtol: number; atol_m3: number } } {
      if (mode.disabled || mode.value === 'standard') return {}
      return {
        waveform: {
          schema_version: 1,
          start_voltage_V: number('start'), dark_seed_s: number('seed'),
          dark_prep_s: number('prep'), branch_dwell_s: number('dwell'),
          turnaround_s: number('turnaround'), turnaround_dark: input('dark').checked,
          uniform_generation_rate_m3_s: select('generation').value === 'uniform' ? number('G') : null,
        },
        waveform_controls: { rtol: number('rtol'), atol_m3: number('atol') },
      }
    },
  }
}

export function appendWaveformEvidence(container: HTMLElement, result: JVWaveformEvidence): void {
  if (!result.waveform) return
  const summary = document.createElement('div')
  summary.className = 'jv-defect-evidence-summary'
  summary.dataset.test = 'jv-waveform-evidence'
  summary.title = result.waveform_protocol_sha256 ?? ''
  const rate = result.waveform.uniform_generation_rate_m3_s
  const labels = [
    'Continuous ramp: finite-time diagnostic',
    rate === null ? 'Device optics' : `Uniform G: ${rate.toExponential(3)} m^-3 s^-1`,
    result.hysteresis_index_paper == null ? 'HI_P: undefined' : `HI_P: ${result.hysteresis_index_paper.toFixed(6)}`,
  ]
  for (const label of labels) {
    const span = document.createElement('span')
    span.textContent = label
    summary.appendChild(span)
  }
  summary.lastElementChild!.setAttribute('title', 'HI_P = P_max,reverse / P_max,forward - 1')
  container.appendChild(summary)
}
