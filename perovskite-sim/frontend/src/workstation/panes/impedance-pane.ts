import { startJob, streamJobEvents } from '../../job-stream'
import { createProgressBar, type ProgressBarHandle } from '../../progress'
import { setStatus, numField, readNum } from '../../ui-helpers'
import type { DeviceConfig, ISResult } from '../../types'
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

  btn.addEventListener('click', () => {
    const active = opts.getActiveDevice()
    if (!active) {
      setStatus('status-imp', 'No active device. Select one in the tree.', true)
      return
    }
    btn.disabled = true
    progressBar.reset()
    setStatus('status-imp', 'Starting job…')

    const params = {
      N_grid: Math.max(3, Math.round(readNum('imp-N', 40))),
      V_dc: readNum('imp-Vdc', 0.9),
      n_freq: Math.max(2, Math.round(readNum('imp-nfreq', 15))),
      f_min: readNum('imp-fmin', 10),
      f_max: readNum('imp-fmax', 1e5),
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
