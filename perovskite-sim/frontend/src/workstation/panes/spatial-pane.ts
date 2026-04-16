import { startJob, streamJobEvents } from '../../job-stream'
import { createProgressBar, type ProgressBarHandle } from '../../progress'
import { setStatus, numField, readNum, checkField, readCheck } from '../../ui-helpers'
import type { DeviceConfig, SpatialProfileResult } from '../../types'
import type { Run, RunResult } from '../types'

export interface SpatialPaneOptions {
  getActiveDevice: () => { id: string; config: DeviceConfig } | null
  onRunComplete: (deviceId: string, run: Run) => void
}

export function mountSpatialPane(container: HTMLElement, opts: SpatialPaneOptions): void {
  container.innerHTML = `
    <div class="card">
      <h3>Spatial Profiles</h3>
      <p class="pane-hint" style="margin-bottom:0.75rem">
        Extracts position-resolved profiles (&phi;, E, n, p, &rho;)
        at each voltage point during a J&ndash;V sweep.
      </p>
      <div class="form-grid">
        ${numField('sp-N', 'N<sub>grid</sub>', 60, '1')}
        ${numField('sp-np', 'V sample points', 10, '1')}
        ${numField('sp-rate', 'Scan rate (V/s)', 1.0, 'any')}
        ${numField('sp-vmax', 'V<sub>max</sub> (V)', 1.0, '0.01')}
        ${checkField('sp-dark', 'Dark (no illumination)', false)}
      </div>
      <div class="actions">
        <button class="btn btn-primary" id="btn-sp">Run Spatial Profiles</button>
        <span class="status" id="status-sp"></span>
      </div>
      <div id="progress-sp"></div>
      <div class="pane-hint">Results stream into the Main Plot pane. Forward sweep snapshots are shown.</div>
    </div>`

  const progressBar: ProgressBarHandle = createProgressBar(
    container.querySelector<HTMLDivElement>('#progress-sp')!,
  )
  const btn = container.querySelector<HTMLButtonElement>('#btn-sp')!

  btn.addEventListener('click', () => {
    const active = opts.getActiveDevice()
    if (!active) {
      setStatus('status-sp', 'No active device. Select one in the tree.', true)
      return
    }
    btn.disabled = true
    progressBar.reset()
    setStatus('status-sp', 'Starting job…')

    const isDark = readCheck('sp-dark', false)
    const params = {
      N_grid: Math.max(3, Math.round(readNum('sp-N', 60))),
      n_points: Math.max(2, Math.round(readNum('sp-np', 10))),
      v_rate: readNum('sp-rate', 1.0),
      V_max: readNum('sp-vmax', 1.0),
      illuminated: !isDark,
    }
    const t0 = performance.now()
    const snapshot: DeviceConfig = JSON.parse(JSON.stringify(active.config))

    startJob('spatial', active.config, params)
      .then(jobId => {
        setStatus('status-sp', 'Running spatial profiles…')
        streamJobEvents<SpatialProfileResult & { active_physics?: string }>(jobId, {
          onProgress: (ev) => progressBar.update(ev),
          onResult: (result) => {
            const { active_physics, ...pure } = result as SpatialProfileResult & { active_physics?: string }
            const runResult: RunResult = { kind: 'spatial', data: pure }
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
            setStatus('status-sp', 'Done')
          },
          onError: (msg) => {
            progressBar.error(msg)
            setStatus('status-sp', `Error: ${msg}`, true)
          },
          onDone: () => {
            btn.disabled = false
          },
        })
      })
      .catch(e => {
        progressBar.error((e as Error).message)
        setStatus('status-sp', `Error: ${(e as Error).message}`, true)
        btn.disabled = false
      })
  })
}

function randomRunId(): string {
  return 'r-' + Math.random().toString(36).slice(2, 10) + Date.now().toString(36)
}
