import { startJob, streamJobEvents } from '../../job-stream'
import { createProgressBar, type ProgressBarHandle } from '../../progress'
import { setStatus, numField, readNum } from '../../ui-helpers'
import type { DeviceConfig, VocTResult } from '../../types'
import type { Run, RunResult } from '../types'

export interface VocTPaneOptions {
  getActiveDevice: () => { id: string; config: DeviceConfig } | null
  onRunComplete: (deviceId: string, run: Run) => void
}

export function mountVocTPane(container: HTMLElement, opts: VocTPaneOptions): void {
  container.innerHTML = `
    <div class="card">
      <h3>V<sub>oc</sub>(T) Parameters</h3>
      <div class="form-grid">
        ${numField('voct-Tmin', 'T<sub>min</sub> (K)', 250, '1')}
        ${numField('voct-Tmax', 'T<sub>max</sub> (K)', 350, '1')}
        ${numField('voct-np', 'T points', 6, '1')}
        ${numField('voct-N', 'N<sub>grid</sub>', 60, '1')}
        ${numField('voct-jvnp', 'J&ndash;V points per T', 30, '1')}
        ${numField('voct-rate', 'Scan rate (V/s)', 1.0, 'any')}
      </div>
      <div class="actions">
        <button class="btn btn-primary" id="btn-voct">Run V<sub>oc</sub>(T)</button>
        <span class="status" id="status-voct"></span>
      </div>
      <div id="progress-voct"></div>
      <div class="pane-hint">Sweeps temperature and extracts the activation energy E<sub>A</sub> from the T&rarr;0 K linear-fit intercept of V<sub>oc</sub>(T). Requires the temperature-scaling physics flag (FAST or FULL tier).</div>
    </div>`

  const progressBar: ProgressBarHandle = createProgressBar(
    container.querySelector<HTMLDivElement>('#progress-voct')!,
  )
  const btn = container.querySelector<HTMLButtonElement>('#btn-voct')!

  btn.addEventListener('click', () => {
    const active = opts.getActiveDevice()
    if (!active) {
      setStatus('status-voct', 'No active device. Select one in the tree.', true)
      return
    }
    btn.disabled = true
    progressBar.reset()
    setStatus('status-voct', 'Starting job…')

    const params = {
      T_min: readNum('voct-Tmin', 250),
      T_max: readNum('voct-Tmax', 350),
      n_points: Math.max(2, Math.round(readNum('voct-np', 6))),
      N_grid: Math.max(3, Math.round(readNum('voct-N', 60))),
      jv_n_points: Math.max(2, Math.round(readNum('voct-jvnp', 30))),
      v_rate: readNum('voct-rate', 1.0),
    }
    const t0 = performance.now()
    const snapshot: DeviceConfig = JSON.parse(JSON.stringify(active.config))

    startJob('voc_t', active.config, params)
      .then(jobId => {
        setStatus('status-voct', 'Running V_oc(T) sweep…')
        streamJobEvents<VocTResult & { active_physics?: string }>(jobId, {
          onProgress: (ev) => progressBar.update(ev),
          onResult: (result) => {
            const { active_physics, ...pure } = result as VocTResult & { active_physics?: string }
            const runResult: RunResult = { kind: 'voc_t', data: pure }
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
            setStatus('status-voct', `Done: E_A ≈ ${pure.E_A_eV.toFixed(3)} eV, slope = ${(pure.slope * 1000).toFixed(2)} mV/K`)
          },
          onError: (msg) => {
            progressBar.error(msg)
            setStatus('status-voct', `Error: ${msg}`, true)
          },
          onDone: () => {
            btn.disabled = false
          },
        })
      })
      .catch(e => {
        progressBar.error((e as Error).message)
        setStatus('status-voct', `Error: ${(e as Error).message}`, true)
        btn.disabled = false
      })
  })
}

function randomRunId(): string {
  return 'r-' + Math.random().toString(36).slice(2, 10) + Date.now().toString(36)
}
