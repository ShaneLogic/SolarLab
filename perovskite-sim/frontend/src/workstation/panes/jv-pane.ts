import { startJob, streamJobEvents } from '../../job-stream'
import { createProgressBar, type ProgressBarHandle } from '../../progress'
import { setStatus, numField, readNum, checkField, readCheck } from '../../ui-helpers'
import type {
  DeviceConfig,
  JVResult,
  CurrentDecompResult,
  SpatialProfileResult,
} from '../../types'
import type { Run, RunResult, ExperimentKind } from '../types'

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
        ${checkField('jvp-decomp', 'Decompose current (J<sub>n</sub> / J<sub>p</sub> / J<sub>ion</sub> / J<sub>disp</sub>)', false)}
        ${checkField('jvp-spatial', 'Save spatial profiles (φ, E, n, p, P)', false)}
      </div>
      <div class="actions">
        <button class="btn btn-primary" id="btn-jvp">Run J–V Sweep</button>
        <span class="status" id="status-jvp"></span>
      </div>
      <div id="progress-jvp"></div>
      <div class="pane-hint">Enable &ldquo;Decompose current&rdquo; or &ldquo;Save spatial profiles&rdquo; to produce the richer output view; otherwise a plain J&ndash;V curve is returned. Only one extra view per run.</div>
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

    const wantDecomp = readCheck('jvp-decomp', false)
    const wantSpatial = readCheck('jvp-spatial', false)
    // Decomposition takes priority if both are ticked — the two backend
    // kinds are mutually exclusive at the dispatch level (each returns a
    // different result shape), so we pick one and let the user re-run for
    // the other. Dropped into a hint so nobody sees a silent coercion.
    if (wantDecomp && wantSpatial) {
      setStatus('status-jvp', 'Both views requested — running decomposition this time. Re-run with only "Save spatial profiles" to get the spatial view.')
    }
    const kind: ExperimentKind = wantDecomp ? 'current_decomp' : wantSpatial ? 'spatial' : 'jv'

    const params = {
      N_grid: Math.max(3, Math.round(readNum('jvp-N', 60))),
      n_points: Math.max(2, Math.round(readNum('jvp-np', 30))),
      v_rate: readNum('jvp-rate', 1.0),
      V_max: readNum('jvp-vmax', 1.4),
      illuminated: true,
    }
    const t0 = performance.now()
    const snapshot: DeviceConfig = JSON.parse(JSON.stringify(active.config))

    type AnyResult = (JVResult | CurrentDecompResult | SpatialProfileResult) & {
      active_physics?: string
    }
    startJob(kind, active.config, params)
      .then(jobId => {
        const label = kind === 'jv' ? 'J–V sweep' : kind === 'current_decomp' ? 'current decomposition' : 'spatial-profile sweep'
        setStatus('status-jvp', `Running ${label}…`)
        streamJobEvents<AnyResult>(jobId, {
          onProgress: (ev) => progressBar.update(ev),
          onResult: (result) => {
            const { active_physics, ...pure } = result
            const runResult: RunResult =
              kind === 'jv'
                ? { kind: 'jv', data: pure as JVResult }
                : kind === 'current_decomp'
                ? { kind: 'current_decomp', data: pure as CurrentDecompResult }
                : { kind: 'spatial', data: pure as SpatialProfileResult }
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
