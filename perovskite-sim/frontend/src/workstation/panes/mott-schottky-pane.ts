import { startJob, streamJobEvents } from '../../job-stream'
import { createProgressBar, type ProgressBarHandle } from '../../progress'
import { setStatus, numField, readNum } from '../../ui-helpers'
import type { DeviceConfig, MottSchottkyResult } from '../../types'
import type { Run, RunResult } from '../types'

export interface MottSchottkyPaneOptions {
  getActiveDevice: () => { id: string; config: DeviceConfig } | null
  onRunComplete: (deviceId: string, run: Run) => void
}

export function mountMottSchottkyPane(container: HTMLElement, opts: MottSchottkyPaneOptions): void {
  container.innerHTML = `
    <div class="card">
      <h3>Mott\u2013Schottky (C\u2013V) Parameters</h3>
      <div class="form-grid">
        ${numField('ms-N', 'N<sub>grid</sub>', 40, '1')}
        ${numField('ms-vlo', 'V<sub>lo</sub> (V)', -0.3, 'any')}
        ${numField('ms-vhi', 'V<sub>hi</sub> (V)', 0.4, 'any')}
        ${numField('ms-npts', 'n<sub>points</sub>', 8, '1')}
        ${numField('ms-freq', 'f (Hz)', 1e5, 'any')}
        ${numField('ms-dV', '\u03b4V (V)', 0.01, 'any')}
      </div>
      <div class="actions">
        <button class="btn btn-primary" id="btn-ms">Run Mott\u2013Schottky</button>
        <span class="status" id="status-ms"></span>
      </div>
      <div id="progress-ms"></div>
      <div class="pane-hint">Dark C\u2013V sweep at fixed <em>f</em> followed by a linear fit of 1/C\u00b2 vs V. V-intercept gives <em>V<sub>bi</sub></em>, slope yields the effective doping <em>N<sub>eff</sub></em>.</div>
    </div>`

  const progressBar: ProgressBarHandle = createProgressBar(
    container.querySelector<HTMLDivElement>('#progress-ms')!,
  )
  const btn = container.querySelector<HTMLButtonElement>('#btn-ms')!

  btn.addEventListener('click', () => {
    const active = opts.getActiveDevice()
    if (!active) {
      setStatus('status-ms', 'No active device. Select one in the tree.', true)
      return
    }
    btn.disabled = true
    progressBar.reset()
    setStatus('status-ms', 'Starting job\u2026')

    const vlo = readNum('ms-vlo', -0.3)
    const vhi = readNum('ms-vhi', 0.4)
    if (!(vhi > vlo)) {
      setStatus('status-ms', 'V_hi must exceed V_lo.', true)
      btn.disabled = false
      return
    }

    const params = {
      N_grid: Math.max(3, Math.round(readNum('ms-N', 40))),
      V_lo: vlo,
      V_hi: vhi,
      n_points: Math.max(3, Math.round(readNum('ms-npts', 8))),
      frequency: readNum('ms-freq', 1e5),
      delta_V: readNum('ms-dV', 0.01),
    }
    const t0 = performance.now()
    const snapshot: DeviceConfig = JSON.parse(JSON.stringify(active.config))

    startJob('mott_schottky', active.config, params)
      .then(jobId => {
        setStatus('status-ms', 'Running C\u2013V sweep\u2026')
        streamJobEvents<MottSchottkyResult & { active_physics?: string }>(jobId, {
          onProgress: (ev) => progressBar.update(ev),
          onResult: (result) => {
            const { active_physics, ...pure } = result as MottSchottkyResult & { active_physics?: string }
            const runResult: RunResult = { kind: 'mott_schottky', data: pure }
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
            setStatus(
              'status-ms',
              `Done \u00b7 V_bi=${pure.V_bi_fit.toFixed(3)} V, N_eff=${pure.N_eff_fit.toExponential(2)} m\u207B\u00b3`,
            )
          },
          onError: (msg) => {
            progressBar.error(msg)
            setStatus('status-ms', `Error: ${msg}`, true)
          },
          onDone: () => {
            btn.disabled = false
          },
        })
      })
      .catch(e => {
        progressBar.error((e as Error).message)
        setStatus('status-ms', `Error: ${(e as Error).message}`, true)
        btn.disabled = false
      })
  })
}

function randomRunId(): string {
  return 'r-' + Math.random().toString(36).slice(2, 10) + Date.now().toString(36)
}
