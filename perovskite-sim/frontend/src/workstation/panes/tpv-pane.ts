import { startJob, streamJobEvents } from '../../job-stream'
import { createProgressBar, type ProgressBarHandle } from '../../progress'
import { setStatus, numField, readNum } from '../../ui-helpers'
import type { DeviceConfig, TPVResult } from '../../types'
import type { Run, RunResult } from '../types'

export interface TPVPaneOptions {
  getActiveDevice: () => { id: string; config: DeviceConfig } | null
  onRunComplete: (deviceId: string, run: Run) => void
}

export function mountTPVPane(container: HTMLElement, opts: TPVPaneOptions): void {
  const accuracyFields = [
    { key: 'rtol', id: 'tpv-rtol', label: 'Relative tolerance', value: 1e-4 },
    { key: 'atol', id: 'tpv-atol', label: 'Absolute tolerance', value: 1e-6 },
    { key: 'voltage_atol', id: 'tpv-voltage-atol', label: 'Voltage tolerance (V)', value: 1e-9 },
  ]
  container.innerHTML = `
    <div class="card">
      <h3>Pulse settings</h3>
      <div class="form-grid">
        ${numField('tpv-N', 'N<sub>grid</sub>', 80, '1')}
        ${numField('tpv-dG', 'Pulse fraction &delta;G/G', 0.05, 'any')}
        ${numField('tpv-tp', 't<sub>pulse</sub> (s)', 1e-6, 'any')}
        ${numField('tpv-td', 't<sub>decay</sub> (s)', 50e-6, 'any')}
        ${numField('tpv-np', 'Output points', 200, '1')}
      </div>
      <details class="form-fieldset">
        <summary>Numerical accuracy</summary>
        <div class="form-grid">
          ${accuracyFields.map(field => numField(field.id, field.label, field.value, 'any')).join('')}
          <label class="form-group">
            <span>Maximum time step (s)</span>
            <input type="number" id="tpv-max-step" step="any" placeholder="Automatic">
          </label>
        </div>
      </details>
      <div class="actions">
        <button class="btn btn-primary" id="btn-tpv">Run TPV</button>
      </div>
      <div class="status" id="status-tpv" role="status" aria-live="polite"></div>
      <div id="progress-tpv"></div>
    </div>`

  const progressBar: ProgressBarHandle = createProgressBar(
    container.querySelector<HTMLDivElement>('#progress-tpv')!,
  )
  const btn = container.querySelector<HTMLButtonElement>('#btn-tpv')!

  function showError(message: string): void {
    const reason = message.split('\n', 1)[0]
    progressBar.error(reason)
    setStatus('status-tpv', `Error: ${reason}`, true)
  }

  btn.addEventListener('click', () => {
    const active = opts.getActiveDevice()
    if (!active) {
      setStatus('status-tpv', 'No active device. Select one in the tree.', true)
      return
    }
    const params: Record<string, unknown> = {
      N_grid: Math.max(3, Math.round(readNum('tpv-N', 80))),
      delta_G_frac: readNum('tpv-dG', 0.05),
      t_pulse: readNum('tpv-tp', 1e-6),
      t_decay: readNum('tpv-td', 50e-6),
      n_points: Math.max(10, Math.round(readNum('tpv-np', 200))),
    }
    for (const field of accuracyFields) {
      const input = container.querySelector<HTMLInputElement>(`#${field.id}`)!
      const value = input.valueAsNumber
      if (!Number.isFinite(value) || value <= 0) {
        showError(`${field.label} must be finite and positive.`)
        return
      }
      params[field.key] = value
    }
    const maxStep = container.querySelector<HTMLInputElement>('#tpv-max-step')!
    if (maxStep.value.trim() !== '' || maxStep.validity.badInput) {
      const value = maxStep.valueAsNumber
      if (!Number.isFinite(value) || value <= 0) {
        showError('Maximum time step must be finite and positive.')
        return
      }
      params.max_step = value
    }
    btn.disabled = true
    progressBar.reset()
    progressBar.busy()
    setStatus('status-tpv', 'Starting job…')
    const t0 = performance.now()
    const snapshot: DeviceConfig = JSON.parse(JSON.stringify(active.config))

    startJob('tpv', active.config, params)
      .then(jobId => {
        setStatus('status-tpv', 'Running TPV experiment…')
        streamJobEvents<TPVResult & { active_physics?: string }>(jobId, {
          onProgress: (ev) => progressBar.update(ev),
          onResult: (result) => {
            const { active_physics, ...pure } = result as TPVResult & { active_physics?: string }
            const runResult: RunResult = { kind: 'tpv', data: pure }
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
            setStatus('status-tpv', 'Done')
          },
          onError: showError,
          onDone: () => {
            btn.disabled = false
          },
        })
      })
      .catch(e => {
        showError((e as Error).message)
        btn.disabled = false
      })
  })
}

function randomRunId(): string {
  return 'r-' + Math.random().toString(36).slice(2, 10) + Date.now().toString(36)
}
