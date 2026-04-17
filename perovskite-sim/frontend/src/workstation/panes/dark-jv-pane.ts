import { startJob, streamJobEvents } from '../../job-stream'
import { createProgressBar, type ProgressBarHandle } from '../../progress'
import { setStatus, numField, readNum } from '../../ui-helpers'
import type { DeviceConfig, DarkJVResult } from '../../types'
import type { Run, RunResult } from '../types'

export interface DarkJVPaneOptions {
  getActiveDevice: () => { id: string; config: DeviceConfig } | null
  onRunComplete: (deviceId: string, run: Run) => void
}

export function mountDarkJVPane(container: HTMLElement, opts: DarkJVPaneOptions): void {
  container.innerHTML = `
    <div class="card">
      <h3>Dark J\u2013V Parameters</h3>
      <div class="form-grid">
        ${numField('djv-N', 'N<sub>grid</sub>', 60, '1')}
        ${numField('djv-npts', 'n<sub>points</sub>', 60, '1')}
        ${numField('djv-vmax', 'V<sub>max</sub> (V)', 1.2, 'any')}
        ${numField('djv-vrate', 'v<sub>rate</sub> (V/s)', 1.0, 'any')}
      </div>
      <div class="actions">
        <button class="btn btn-primary" id="btn-djv">Run Dark J\u2013V</button>
        <span class="status" id="status-djv"></span>
      </div>
      <div id="progress-djv"></div>
      <div class="pane-hint">Forward-bias sweep under dark conditions; fits diode ideality <em>n</em> and saturation current <em>J<sub>0</sub></em> over a linear log(J)&ndash;V window.</div>
    </div>`

  const progressBar: ProgressBarHandle = createProgressBar(
    container.querySelector<HTMLDivElement>('#progress-djv')!,
  )
  const btn = container.querySelector<HTMLButtonElement>('#btn-djv')!

  btn.addEventListener('click', () => {
    const active = opts.getActiveDevice()
    if (!active) {
      setStatus('status-djv', 'No active device. Select one in the tree.', true)
      return
    }
    btn.disabled = true
    progressBar.reset()
    setStatus('status-djv', 'Starting job\u2026')

    const params = {
      N_grid: Math.max(3, Math.round(readNum('djv-N', 60))),
      n_points: Math.max(10, Math.round(readNum('djv-npts', 60))),
      V_max: readNum('djv-vmax', 1.2),
      v_rate: readNum('djv-vrate', 1.0),
    }
    const t0 = performance.now()
    const snapshot: DeviceConfig = JSON.parse(JSON.stringify(active.config))

    startJob('dark_jv', active.config, params)
      .then(jobId => {
        setStatus('status-djv', 'Running dark J\u2013V\u2026')
        streamJobEvents<DarkJVResult & { active_physics?: string }>(jobId, {
          onProgress: (ev) => progressBar.update(ev),
          onResult: (result) => {
            const { active_physics, ...pure } = result as DarkJVResult & { active_physics?: string }
            const runResult: RunResult = { kind: 'dark_jv', data: pure }
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
            setStatus('status-djv', `Done \u00b7 n=${pure.n_ideality.toFixed(2)}, J\u2080=${pure.J_0.toExponential(2)} A/m\u00b2`)
          },
          onError: (msg) => {
            progressBar.error(msg)
            setStatus('status-djv', `Error: ${msg}`, true)
          },
          onDone: () => {
            btn.disabled = false
          },
        })
      })
      .catch(e => {
        progressBar.error((e as Error).message)
        setStatus('status-djv', `Error: ${(e as Error).message}`, true)
        btn.disabled = false
      })
  })
}

function randomRunId(): string {
  return 'r-' + Math.random().toString(36).slice(2, 10) + Date.now().toString(36)
}
