import { startJob, streamJobEvents } from '../../job-stream'
import { createProgressBar, type ProgressBarHandle } from '../../progress'
import {
  checkField,
  numField,
  readCheck,
  readNum,
  setStatus,
} from '../../ui-helpers'
import type { DeviceConfig, ISResult } from '../../types'
import { dynamicDefectImpedancePreset } from '../../explicit-defect-capability'
import type { Run, RunResult } from '../types'

export interface ImpedancePaneOptions {
  getActiveDevice: () => { id: string; config: DeviceConfig } | null
  onRunComplete: (deviceId: string, run: Run) => void
}

export function mountImpedancePane(container: HTMLElement, opts: ImpedancePaneOptions): void {
  container.innerHTML = `
    <div class="card">
      <h3>Impedance Sweep Parameters</h3>
      <div class="form-grid">
        ${numField('imp-N', 'N<sub>grid</sub>', 40, '1')}
        ${numField('imp-Vdc', 'V<sub>dc</sub> (V)', 0.9, 'any')}
        ${numField('imp-nfreq', 'n<sub>freq</sub>', 15, '1')}
        ${numField('imp-fmin', 'f<sub>min</sub> (Hz)', 10, 'any')}
        ${numField('imp-fmax', 'f<sub>max</sub> (Hz)', 1e5, 'any')}
        ${numField('imp-dv', '&delta;V (mV)', 10, '0.5')}
        ${numField('imp-cycles', 'Cycles', 5, '1')}
        ${numField('imp-extract', 'Extract cycles', 2, '1')}
        ${numField('imp-ppc', 'Points/cycle', 40, '1')}
        ${numField('imp-dc-settle', 'DC settle (s)', 1e-3, 'any')}
        ${numField('imp-defect-order', 'Defect energy order', 32, '1')}
        <label class="form-group">
          <span>Engine</span>
          <select id="imp-method">
            <option value="transient_ion_aware">Transient, ion-aware</option>
            <option value="ion_aware_frequency_certified">Certified frequency, ion-aware</option>
            <option value="qf_frequency_ion_free">QF frequency, ion-free</option>
            <option value="dynamic_defect_frequency_certified">Certified dynamic defects</option>
          </select>
        </label>
        ${checkField('imp-illuminated', 'Illuminated', true)}
        ${checkField('imp-strict', 'Require contact certificate', false)}
        ${checkField('imp-window-strict', 'Require ionic frequency coverage', true)}
      </div>
      <div class="actions">
        <button class="btn btn-primary" id="btn-imp">Run Impedance Sweep</button>
        <span class="status" id="status-imp"></span>
      </div>
      <div id="progress-imp"></div>
      <div class="pane-hint">Results stream into the Main Plot pane and appear as a run under this experiment in the tree.</div>
    </div>`

  const progressBar: ProgressBarHandle = createProgressBar(
    container.querySelector<HTMLDivElement>('#progress-imp')!,
  )
  const btn = container.querySelector<HTMLButtonElement>('#btn-imp')!
  const methodSelect = container.querySelector<HTMLSelectElement>('#imp-method')!
  const syncMethodControls = (): void => {
    const transient = methodSelect.value === 'transient_ion_aware'
    const dynamic = methodSelect.value === 'dynamic_defect_frequency_certified'
    for (const id of ['imp-cycles', 'imp-extract', 'imp-ppc', 'imp-dc-settle']) {
      const input = container.querySelector<HTMLInputElement>(`#${id}`)
      if (input) input.disabled = !transient
    }
    const defectOrder = container.querySelector<HTMLInputElement>('#imp-defect-order')
    if (defectOrder) defectOrder.disabled = !dynamic
    const windowStrict = container.querySelector<HTMLInputElement>(
      '#imp-window-strict',
    )
    if (windowStrict) {
      if (dynamic) windowStrict.checked = true
      windowStrict.disabled = (
        dynamic || methodSelect.value !== 'ion_aware_frequency_certified'
      )
    }
    const contactStrict = container.querySelector<HTMLInputElement>('#imp-strict')
    if (contactStrict) {
      if (dynamic) contactStrict.checked = true
      contactStrict.disabled = dynamic
    }
  }
  methodSelect.addEventListener('change', () => {
    const certified = methodSelect.value === 'ion_aware_frequency_certified'
    const dynamic = methodSelect.value === 'dynamic_defect_frequency_certified'
    const active = opts.getActiveDevice()
    const dynamicPreset = dynamicDefectImpedancePreset(active?.config ?? {
      device: { Phi: 0 },
      layers: [],
    })
    const presets: Record<string, number> = dynamic
      ? {
          'imp-N': dynamicPreset.N_grid,
          'imp-nfreq': dynamicPreset.n_freq,
          'imp-fmin': dynamicPreset.f_min,
          'imp-fmax': dynamicPreset.f_max,
        }
      : certified
        ? { 'imp-N': 60, 'imp-nfreq': 29, 'imp-fmin': 1e-6, 'imp-fmax': 10 }
        : { 'imp-N': 40, 'imp-nfreq': 15, 'imp-fmin': 10, 'imp-fmax': 1e5 }
    for (const [id, value] of Object.entries(presets)) {
      const input = container.querySelector<HTMLInputElement>(`#${id}`)
      if (input) input.value = String(value)
    }
    syncMethodControls()
  })
  syncMethodControls()

  btn.addEventListener('click', () => {
    const active = opts.getActiveDevice()
    if (!active) {
      setStatus('status-imp', 'No active device. Select one in the tree.', true)
      return
    }
    btn.disabled = true
    progressBar.reset()
    progressBar.busy()
    setStatus('status-imp', 'Starting job…')

    const nCycles = Math.max(1, Math.round(readNum('imp-cycles', 5)))
    const selectedMethod = (
      document.getElementById('imp-method') as HTMLSelectElement | null
    )?.value
    const method = selectedMethod === 'qf_frequency_ion_free'
      ? 'qf_frequency_ion_free' as const
      : selectedMethod === 'transient_ion_aware'
        ? 'transient_ion_aware' as const
        : selectedMethod === 'dynamic_defect_frequency_certified'
          ? 'dynamic_defect_frequency_certified' as const
          : 'ion_aware_frequency_certified' as const
    const dynamic = method === 'dynamic_defect_frequency_certified'
    const params = {
      N_grid: Math.max(3, Math.round(readNum('imp-N', 40))),
      V_dc: readNum('imp-Vdc', 0.9),
      n_freq: Math.max(dynamic ? 3 : 2, Math.round(readNum('imp-nfreq', 15))),
      f_min: readNum('imp-fmin', 10),
      f_max: readNum('imp-fmax', 1e5),
      delta_V: readNum('imp-dv', 10) * 1e-3,
      n_cycles: nCycles,
      n_extract: Math.min(
        nCycles,
        Math.max(1, Math.round(readNum('imp-extract', 2))),
      ),
      points_per_cycle: Math.max(8, Math.round(readNum('imp-ppc', 40))),
      dc_settle_time: readNum('imp-dc-settle', 1e-3),
      illuminated: readCheck('imp-illuminated', true),
      method,
      require_operating_point_certificate: (
        dynamic || readCheck('imp-strict', false)
      ),
      require_frequency_window_certificate: (
        (method === 'ion_aware_frequency_certified' || dynamic)
        && readCheck('imp-window-strict', true)
      ),
      ...(dynamic
        ? {
            defect_energy_quadrature_order: Math.max(
              1,
              Math.round(readNum('imp-defect-order', 32)),
            ),
          }
        : {}),
    }
    const t0 = performance.now()
    const snapshot: DeviceConfig = JSON.parse(JSON.stringify(active.config))

    startJob('impedance', active.config, params)
      .then(jobId => {
        setStatus('status-imp', 'Running impedance sweep…')
        streamJobEvents<ISResult & { active_physics?: string }>(jobId, {
          onProgress: (ev) => progressBar.update(ev),
          onResult: (result) => {
            const { active_physics, ...pure } = result as ISResult & { active_physics?: string }
            const runResult: RunResult = { kind: 'impedance', data: pure }
            const run: Run = {
              id: randomRunId(),
              timestamp: Date.now(),
              result: runResult,
              activePhysics: active_physics ?? 'unknown',
              durationMs: performance.now() - t0,
              deviceSnapshot: snapshot,
            }
            opts.onRunComplete(active.id, run)
            progressBar.done()
            setStatus('status-imp', 'Done')
          },
          onError: (msg) => {
            progressBar.error(msg)
            setStatus('status-imp', `Error: ${msg}`, true)
          },
          onDone: () => {
            btn.disabled = false
          },
        })
      })
      .catch(e => {
        progressBar.error((e as Error).message)
        setStatus('status-imp', `Error: ${(e as Error).message}`, true)
        btn.disabled = false
      })
  })
}

function randomRunId(): string {
  return 'r-' + Math.random().toString(36).slice(2, 10) + Date.now().toString(36)
}
