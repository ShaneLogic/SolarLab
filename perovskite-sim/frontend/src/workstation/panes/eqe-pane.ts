import { startJob, streamJobEvents } from '../../job-stream'
import { createProgressBar, type ProgressBarHandle } from '../../progress'
import { setStatus, numField, readNum } from '../../ui-helpers'
import type { DeviceConfig, EQEResult } from '../../types'
import type { Run, RunResult } from '../types'

export interface EQEPaneOptions {
  getActiveDevice: () => { id: string; config: DeviceConfig } | null
  onRunComplete: (deviceId: string, run: Run) => void
}

export function mountEQEPane(container: HTMLElement, opts: EQEPaneOptions): void {
  container.innerHTML = `
    <div class="card">
      <h3>EQE / IPCE Parameters</h3>
      <div class="form-grid">
        ${numField('eqe-N', 'N<sub>grid</sub>', 60, '1')}
        ${numField('eqe-lmin', '\u03bb<sub>min</sub> (nm)', 300, '1')}
        ${numField('eqe-lmax', '\u03bb<sub>max</sub> (nm)', 1000, '1')}
        ${numField('eqe-nl', 'n<sub>\u03bb</sub>', 29, '1')}
        ${numField('eqe-phi', '\u03a6<sub>inc</sub> (ph/m\u00b2/s)', 1e20, 'any')}
        ${numField('eqe-tset', 't<sub>settle</sub> (s)', 1e-3, 'any')}
      </div>
      <div class="actions">
        <button class="btn btn-primary" id="btn-eqe">Compute EQE(\u03bb)</button>
        <span class="status" id="status-eqe"></span>
      </div>
      <div id="progress-eqe"></div>
      <div class="pane-hint">Monochromatic short-circuit quantum efficiency swept across <em>\u03bb</em>; AM1.5G-integrated <em>J<sub>sc</sub></em> is reported alongside EQE(<em>\u03bb</em>).</div>
    </div>`

  const progressBar: ProgressBarHandle = createProgressBar(
    container.querySelector<HTMLDivElement>('#progress-eqe')!,
  )
  const btn = container.querySelector<HTMLButtonElement>('#btn-eqe')!

  btn.addEventListener('click', () => {
    const active = opts.getActiveDevice()
    if (!active) {
      setStatus('status-eqe', 'No active device. Select one in the tree.', true)
      return
    }
    btn.disabled = true
    progressBar.reset()
    setStatus('status-eqe', 'Starting job\u2026')

    const lmin = readNum('eqe-lmin', 300)
    const lmax = readNum('eqe-lmax', 1000)
    const nl = Math.max(3, Math.round(readNum('eqe-nl', 29)))
    if (!(lmax > lmin)) {
      setStatus('status-eqe', '\u03bb_max must exceed \u03bb_min.', true)
      btn.disabled = false
      return
    }

    const params = {
      N_grid: Math.max(3, Math.round(readNum('eqe-N', 60))),
      lambda_min_nm: lmin,
      lambda_max_nm: lmax,
      n_lambda: nl,
      Phi_incident: readNum('eqe-phi', 1e20),
      t_settle: readNum('eqe-tset', 1e-3),
    }
    const t0 = performance.now()
    const snapshot: DeviceConfig = JSON.parse(JSON.stringify(active.config))

    startJob('eqe', active.config, params)
      .then(jobId => {
        setStatus('status-eqe', 'Running EQE sweep\u2026')
        streamJobEvents<EQEResult & { active_physics?: string }>(jobId, {
          onProgress: (ev) => progressBar.update(ev),
          onResult: (result) => {
            const { active_physics, ...pure } = result as EQEResult & { active_physics?: string }
            const runResult: RunResult = { kind: 'eqe', data: pure }
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
            const mAcm2 = pure.J_sc_integrated / 10
            setStatus('status-eqe', `Done \u00b7 J\u209b\u1d9c(AM1.5G)=${mAcm2.toFixed(2)} mA/cm\u00b2`)
          },
          onError: (msg) => {
            progressBar.error(msg)
            setStatus('status-eqe', `Error: ${msg}`, true)
          },
          onDone: () => {
            btn.disabled = false
          },
        })
      })
      .catch(e => {
        progressBar.error((e as Error).message)
        setStatus('status-eqe', `Error: ${(e as Error).message}`, true)
        btn.disabled = false
      })
  })
}

function randomRunId(): string {
  return 'r-' + Math.random().toString(36).slice(2, 10) + Date.now().toString(36)
}
