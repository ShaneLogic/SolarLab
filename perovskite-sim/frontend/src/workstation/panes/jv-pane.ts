import { startJob, streamJobEvents } from '../../job-stream'
import { createProgressBar, type ProgressBarHandle } from '../../progress'
import { setStatus, numField, readNum } from '../../ui-helpers'
import type { DeviceConfig, JVResult } from '../../types'
import type { Run, RunResult } from '../types'

export interface JVPaneOptions {
  getActiveDevice: () => { id: string; config: DeviceConfig } | null
  onRunComplete: (deviceId: string, run: Run) => void
}

export function mountJVPane(container: HTMLElement, opts: JVPaneOptions): void {
  container.innerHTML = `
    <div class="card">
      <h3>J–V Sweep Parameters</h3>
      <div class="form-grid">
        ${numField('jvp-N', 'N<sub>grid</sub>', 60, '1')}
        ${numField('jvp-np', 'V sample points', 30, '1')}
        ${numField('jvp-rate', 'Scan rate (V/s)', 1.0, 'any')}
        ${numField('jvp-vmax', 'V<sub>max</sub> (V)', 1.4, '0.01')}
      </div>
      <div class="actions">
        <button class="btn btn-primary" id="btn-jvp">Run J–V Sweep</button>
        <span class="status" id="status-jvp"></span>
      </div>
      <div id="progress-jvp"></div>
      <div class="pane-hint">Results stream into the Main Plot pane and appear as a run under this experiment in the tree.</div>
    </div>`

  const progressBar: ProgressBarHandle = createProgressBar(
    container.querySelector<HTMLDivElement>('#progress-jvp')!,
  )
  const btn = container.querySelector<HTMLButtonElement>('#btn-jvp')!

  btn.addEventListener('click', () => {
    const active = opts.getActiveDevice()
    if (!active) {
      setStatus('status-jvp', 'No active device. Select one in the tree.', true)
      return
    }
    btn.disabled = true
    progressBar.reset()
    setStatus('status-jvp', 'Starting job…')

    const params = {
      N_grid: Math.max(3, Math.round(readNum('jvp-N', 60))),
      n_points: Math.max(2, Math.round(readNum('jvp-np', 30))),
      v_rate: readNum('jvp-rate', 1.0),
      V_max: readNum('jvp-vmax', 1.4),
    }
    const t0 = performance.now()
    const snapshot: DeviceConfig = JSON.parse(JSON.stringify(active.config))

    startJob('jv', active.config, params)
      .then(jobId => {
        setStatus('status-jvp', 'Running J–V sweep…')
        streamJobEvents<JVResult & { active_physics?: string }>(jobId, {
          onProgress: (ev) => progressBar.update(ev),
          onResult: (result) => {
            const { active_physics, ...pure } = result as JVResult & { active_physics?: string }
            const runResult: RunResult = { kind: 'jv', data: pure }
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
            setStatus('status-jvp', 'Done')
          },
          onError: (msg) => {
            progressBar.error(msg)
            setStatus('status-jvp', `Error: ${msg}`, true)
          },
          onDone: () => {
            btn.disabled = false
          },
        })
      })
      .catch(e => {
        progressBar.error((e as Error).message)
        setStatus('status-jvp', `Error: ${(e as Error).message}`, true)
        btn.disabled = false
      })
  })
}

function randomRunId(): string {
  return 'r-' + Math.random().toString(36).slice(2, 10) + Date.now().toString(36)
}
