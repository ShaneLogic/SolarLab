import { startJob, streamJobEvents } from '../../job-stream'
import { createProgressBar, type ProgressBarHandle } from '../../progress'
import { setStatus, numField, readNum, checkField, readCheck } from '../../ui-helpers'
import type { DeviceConfig, CurrentDecompResult } from '../../types'
import type { Run, RunResult } from '../types'

export interface CurrentDecompPaneOptions {
  getActiveDevice: () => { id: string; config: DeviceConfig } | null
  onRunComplete: (deviceId: string, run: Run) => void
}

export function mountCurrentDecompPane(container: HTMLElement, opts: CurrentDecompPaneOptions): void {
  container.innerHTML = `
    <div class="card">
      <h3>Current Decomposition</h3>
      <p class="pane-hint" style="margin-bottom:0.75rem">
        Decomposes the terminal current into electron (J<sub>n</sub>),
        hole (J<sub>p</sub>), ionic (J<sub>ion</sub>), and displacement
        (J<sub>disp</sub>) components across a J&ndash;V sweep.
      </p>
      <div class="form-grid">
        ${numField('cd-N', 'N<sub>grid</sub>', 60, '1')}
        ${numField('cd-np', 'V sample points', 30, '1')}
        ${numField('cd-rate', 'Scan rate (V/s)', 1.0, 'any')}
        ${numField('cd-vmax', 'V<sub>max</sub> (V)', 1.4, '0.01')}
        ${checkField('cd-dark', 'Dark (no illumination)', false)}
      </div>
      <div class="actions">
        <button class="btn btn-primary" id="btn-cd">Run Decomposition</button>
        <span class="status" id="status-cd"></span>
      </div>
      <div id="progress-cd"></div>
      <div class="pane-hint">Results stream into the Main Plot pane.</div>
    </div>`

  const progressBar: ProgressBarHandle = createProgressBar(
    container.querySelector<HTMLDivElement>('#progress-cd')!,
  )
  const btn = container.querySelector<HTMLButtonElement>('#btn-cd')!

  btn.addEventListener('click', () => {
    const active = opts.getActiveDevice()
    if (!active) {
      setStatus('status-cd', 'No active device. Select one in the tree.', true)
      return
    }
    btn.disabled = true
    progressBar.reset()
    setStatus('status-cd', 'Starting job…')

    const isDark = readCheck('cd-dark', false)
    const params = {
      N_grid: Math.max(3, Math.round(readNum('cd-N', 60))),
      n_points: Math.max(2, Math.round(readNum('cd-np', 30))),
      v_rate: readNum('cd-rate', 1.0),
      V_max: readNum('cd-vmax', 1.4),
      illuminated: !isDark,
    }
    const t0 = performance.now()
    const snapshot: DeviceConfig = JSON.parse(JSON.stringify(active.config))

    startJob('current_decomp', active.config, params)
      .then(jobId => {
        setStatus('status-cd', 'Running current decomposition…')
        streamJobEvents<CurrentDecompResult & { active_physics?: string }>(jobId, {
          onProgress: (ev) => progressBar.update(ev),
          onResult: (result) => {
            const { active_physics, ...pure } = result as CurrentDecompResult & { active_physics?: string }
            const runResult: RunResult = { kind: 'current_decomp', data: pure }
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
            setStatus('status-cd', 'Done')
          },
          onError: (msg) => {
            progressBar.error(msg)
            setStatus('status-cd', `Error: ${msg}`, true)
          },
          onDone: () => {
            btn.disabled = false
          },
        })
      })
      .catch(e => {
        progressBar.error((e as Error).message)
        setStatus('status-cd', `Error: ${(e as Error).message}`, true)
        btn.disabled = false
      })
  })
}

function randomRunId(): string {
  return 'r-' + Math.random().toString(36).slice(2, 10) + Date.now().toString(36)
}
