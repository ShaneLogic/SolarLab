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
  container.innerHTML = `
    <div class="card">
      <h3>TPV Parameters</h3>
      <div class="form-grid">
        ${numField('tpv-N', 'N<sub>grid</sub>', 80, '1')}
        ${numField('tpv-dG', 'Pulse fraction &delta;G/G', 0.05, 'any')}
        ${numField('tpv-tp', 't<sub>pulse</sub> (s)', 1e-6, 'any')}
        ${numField('tpv-td', 't<sub>decay</sub> (s)', 50e-6, 'any')}
        ${numField('tpv-np', 'Output points', 200, '1')}
      </div>
      <div class="actions">
        <button class="btn btn-primary" id="btn-tpv">Run TPV</button>
        <span class="status" id="status-tpv"></span>
      </div>
      <div id="progress-tpv"></div>
      <div class="pane-hint">Results stream into the Main Plot pane and appear as a run under this experiment in the tree.</div>
    </div>`

  const progressBar: ProgressBarHandle = createProgressBar(
    container.querySelector<HTMLDivElement>('#progress-tpv')!,
  )
  const btn = container.querySelector<HTMLButtonElement>('#btn-tpv')!

  btn.addEventListener('click', () => {
    const active = opts.getActiveDevice()
    if (!active) {
      setStatus('status-tpv', 'No active device. Select one in the tree.', true)
      return
    }
    btn.disabled = true
    progressBar.reset()
    setStatus('status-tpv', 'Starting job…')

    const params = {
      N_grid: Math.max(3, Math.round(readNum('tpv-N', 80))),
      delta_G_frac: readNum('tpv-dG', 0.05),
      t_pulse: readNum('tpv-tp', 1e-6),
      t_decay: readNum('tpv-td', 50e-6),
      n_points: Math.max(10, Math.round(readNum('tpv-np', 200))),
    }
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
          onError: (msg) => {
            progressBar.error(msg)
            setStatus('status-tpv', `Error: ${msg}`, true)
          },
          onDone: () => {
            btn.disabled = false
          },
        })
      })
      .catch(e => {
        progressBar.error((e as Error).message)
        setStatus('status-tpv', `Error: ${(e as Error).message}`, true)
        btn.disabled = false
      })
  })
}

function randomRunId(): string {
  return 'r-' + Math.random().toString(36).slice(2, 10) + Date.now().toString(36)
}
