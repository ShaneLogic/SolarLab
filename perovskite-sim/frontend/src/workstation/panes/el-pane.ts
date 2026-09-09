import { startJob, streamJobEvents } from '../../job-stream'
import { createProgressBar, type ProgressBarHandle } from '../../progress'
import { setStatus, numField, readNum } from '../../ui-helpers'
import { hasTMMOptics } from '../../device-utils'
import type { DeviceConfig, ELResult } from '../../types'
import type { Run, RunResult } from '../types'

const NON_TMM_MSG =
  'TMM optical data required. Select a preset with wavelength-resolved n,k data.'

export interface ELPaneOptions {
  getActiveDevice: () => { id: string; config: DeviceConfig } | null
  onRunComplete: (deviceId: string, run: Run) => void
}

export function mountELPane(container: HTMLElement, opts: ELPaneOptions): void {
  const active = opts.getActiveDevice()
  const showBanner = !!active && !hasTMMOptics(active.config)
  container.innerHTML = `
    <div class="card">
      <h3>Spectral settings</h3>
      <div id="el-tmm-banner" class="status error"${showBanner ? '' : ' hidden'} role="alert" style="display:${showBanner ? 'block' : 'none'};margin-bottom:0.5rem;">${NON_TMM_MSG}</div>
      <div class="form-grid">
        ${numField('el-V', 'V<sub>inj</sub> (V)', 1.0, 'any')}
        ${numField('el-lmin', '&lambda;<sub>min</sub> (nm)', 400, '1')}
        ${numField('el-lmax', '&lambda;<sub>max</sub> (nm)', 1000, '1')}
        ${numField('el-nl', 'Wavelength points', 25, '1')}
        ${numField('el-N', 'N<sub>grid</sub>', 60, '1')}
        ${numField('el-nd', 'Dark J&ndash;V points', 30, '1')}
        ${numField('el-rate', 'Scan rate (V/s)', 1.0, 'any')}
      </div>
      <div class="actions">
        <button class="btn btn-primary" id="btn-el">Run EL</button>
        <span class="status" id="status-el"></span>
      </div>
      <div id="progress-el"></div>
      <div class="pane-hint" title="Rau reciprocity: Phi_EL = A_abs * phi_bb * exp(qV/kT); EQE_EL = J_em,rad / |J_inj|; delta_V_nr = -(kT/q) * ln(EQE_EL)."><strong>Outputs:</strong> EL, EQE<sub>EL</sub>, &Delta;V<sub>nr</sub>.</div>
    </div>`

  const progressBar: ProgressBarHandle = createProgressBar(
    container.querySelector<HTMLDivElement>('#progress-el')!,
  )
  const btn = container.querySelector<HTMLButtonElement>('#btn-el')!

  btn.addEventListener('click', () => {
    const active = opts.getActiveDevice()
    if (!active) {
      setStatus('status-el', 'No active device. Select one in the tree.', true)
      return
    }
    const banner = container.querySelector<HTMLDivElement>('#el-tmm-banner')
    if (!hasTMMOptics(active.config)) {
      if (banner) { banner.textContent = NON_TMM_MSG; banner.style.display = 'block' }
      setStatus('status-el', 'Switch to a TMM preset before running EL.', true)
      return
    }
    if (banner) banner.style.display = 'none'
    btn.disabled = true
    progressBar.reset()
    progressBar.busy()
    setStatus('status-el', 'Starting job…')

    const params = {
      V_inj: readNum('el-V', 1.0),
      lambda_min_nm: readNum('el-lmin', 400),
      lambda_max_nm: readNum('el-lmax', 1000),
      n_lambda: Math.max(2, Math.round(readNum('el-nl', 25))),
      N_grid: Math.max(3, Math.round(readNum('el-N', 60))),
      n_points_dark: Math.max(2, Math.round(readNum('el-nd', 30))),
      v_rate: readNum('el-rate', 1.0),
    }
    const t0 = performance.now()
    const snapshot: DeviceConfig = JSON.parse(JSON.stringify(active.config))

    startJob('el', active.config, params)
      .then(jobId => {
        setStatus('status-el', 'Running EL reciprocity sweep…')
        streamJobEvents<ELResult & { active_physics?: string }>(jobId, {
          onProgress: (ev) => progressBar.update(ev),
          onResult: (result) => {
            const { active_physics, ...pure } = result as ELResult & { active_physics?: string }
            const runResult: RunResult = { kind: 'el', data: pure }
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
            setStatus('status-el',
              `Done: EQE_EL = ${pure.EQE_EL.toExponential(2)}, ΔV_nr = ${pure.delta_V_nr_mV.toFixed(1)} mV`)
          },
          onError: (msg) => {
            progressBar.error(msg)
            setStatus('status-el', `Error: ${msg}`, true)
          },
          onDone: () => {
            btn.disabled = false
          },
        })
      })
      .catch(e => {
        progressBar.error((e as Error).message)
        setStatus('status-el', `Error: ${(e as Error).message}`, true)
        btn.disabled = false
      })
  })
}

function randomRunId(): string {
  return 'r-' + Math.random().toString(36).slice(2, 10) + Date.now().toString(36)
}
