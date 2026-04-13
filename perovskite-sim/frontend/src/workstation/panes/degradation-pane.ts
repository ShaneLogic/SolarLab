import { startJob, streamJobEvents } from '../../job-stream'
import { createProgressBar, type ProgressBarHandle } from '../../progress'
import { setStatus, numField, readNum } from '../../ui-helpers'
import type { DeviceConfig, DegResult } from '../../types'
import type { Run, RunResult } from '../types'

export interface DegradationPaneOptions {
  getActiveDevice: () => { id: string; config: DeviceConfig } | null
  onRunComplete: (deviceId: string, run: Run) => void
}

export function mountDegradationPane(container: HTMLElement, opts: DegradationPaneOptions): void {
  container.innerHTML = `
    <div class="card">
      <h3>Degradation Parameters</h3>
      <div class="form-grid">
        ${numField('deg-N', 'N<sub>grid</sub>', 40, '1')}
        ${numField('deg-Vbias', 'V<sub>bias</sub> (V)', 0.9, 'any')}
        ${numField('deg-tend', 't<sub>end</sub> (s)', 100, 'any')}
        ${numField('deg-nsnap', 'n<sub>snapshots</sub>', 10, '1')}
      </div>
      <div class="actions">
        <button class="btn btn-primary" id="btn-deg">Run Degradation</button>
        <span class="status" id="status-deg"></span>
      </div>
      <div id="progress-deg"></div>
      <div class="pane-hint">Results stream into the Main Plot pane and appear as a run under this experiment in the tree.</div>
    </div>`

  const progressBar: ProgressBarHandle = createProgressBar(
    container.querySelector<HTMLDivElement>('#progress-deg')!,
  )
  const btn = container.querySelector<HTMLButtonElement>('#btn-deg')!

  btn.addEventListener('click', () => {
    const active = opts.getActiveDevice()
    if (!active) {
      setStatus('status-deg', 'No active device. Select one in the tree.', true)
      return
    }
    btn.disabled = true
    progressBar.reset()
    setStatus('status-deg', 'Starting job…')

    const params = {
      N_grid: Math.max(3, Math.round(readNum('deg-N', 40))),
      V_bias: readNum('deg-Vbias', 0.9),
      t_end: readNum('deg-tend', 100),
      n_snapshots: Math.max(2, Math.round(readNum('deg-nsnap', 10))),
    }
    const t0 = performance.now()
    const snapshot: DeviceConfig = JSON.parse(JSON.stringify(active.config))

    startJob('degradation', active.config, params)
      .then(jobId => {
        setStatus('status-deg', 'Running degradation…')
        streamJobEvents<DegResult>(jobId, {
          onProgress: (ev) => progressBar.update(ev),
          onResult: (result) => {
            const runResult: RunResult = { kind: 'degradation', data: result }
            const run: Run = {
              id: randomRunId(),
              timestamp: Date.now(),
              result: runResult,
              activePhysics: 'unknown',
              durationMs: performance.now() - t0,
              deviceSnapshot: snapshot,
            }
            opts.onRunComplete(active.id, run)
            progressBar.done()
            setStatus('status-deg', 'Done')
          },
          onError: (msg) => {
            progressBar.error(msg)
            setStatus('status-deg', `Error: ${msg}`, true)
          },
          onDone: () => {
            btn.disabled = false
          },
        })
      })
      .catch(e => {
        progressBar.error((e as Error).message)
        setStatus('status-deg', `Error: ${(e as Error).message}`, true)
        btn.disabled = false
      })
  })
}

function randomRunId(): string {
  return 'r-' + Math.random().toString(36).slice(2, 10) + Date.now().toString(36)
}
