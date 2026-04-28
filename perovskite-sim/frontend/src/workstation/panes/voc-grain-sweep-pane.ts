/**
 * Stage-B headline experiment: V_oc(L_g) sweep with one centred absorber GB.
 *
 * Drives the backend kind="voc_grain_sweep" dispatcher. The grain sizes are
 * entered as a CSV of nm values; τ_GB / GB width default to literature-typical
 * values for moderately passivated MAPbI3 (τ_GB = 1 ns, width = 10 nm).
 */
import { startJob, streamJobEvents } from '../../job-stream'
import { createProgressBar, type ProgressBarHandle } from '../../progress'
import { setStatus, numField, readNum } from '../../ui-helpers'
import type { DeviceConfig, VocGrainSweepResult } from '../../types'
import type { Run, RunResult } from '../types'

export interface VocGrainSweepPaneOptions {
  getActiveDevice: () => { id: string; config: DeviceConfig } | null
  onRunComplete: (deviceId: string, run: Run) => void
}

const DEFAULT_GRAIN_SIZES_NM = '200, 500, 1000'

function parseGrainSizesCsv(raw: string): number[] {
  return raw
    .split(/[,\s]+/)
    .map(s => s.trim())
    .filter(s => s.length > 0)
    .map(s => Number(s))
    .filter(n => Number.isFinite(n) && n > 0)
}

export function mountVocGrainSweepPane(
  container: HTMLElement,
  opts: VocGrainSweepPaneOptions,
): void {
  container.innerHTML = `
    <div class="card">
      <h3>V<sub>oc</sub>(L<sub>g</sub>) Grain Sweep (Stage B)</h3>
      <div class="form-grid">
        <label class="row-label">
          <span>Grain sizes <i>L<sub>g</sub></i> (nm, comma-separated)</span>
          <input type="text" id="vgs-sizes" value="${DEFAULT_GRAIN_SIZES_NM}" class="config-input" />
        </label>
        ${numField('vgs-tau-n', 'τ<sub>GB</sub><sup>n</sup> (s)', 1e-9, 'any')}
        ${numField('vgs-tau-p', 'τ<sub>GB</sub><sup>p</sup> (s)', 1e-9, 'any')}
        ${numField('vgs-width-nm', 'GB width (nm)', 10, 'any')}
        ${numField('vgs-Nx', 'N<sub>x</sub>', 8, '1')}
        ${numField('vgs-Nyl', 'N<sub>y</sub> per layer', 8, '1')}
        ${numField('vgs-vmax', 'V<sub>max</sub> (V)', 1.2, 'any')}
        ${numField('vgs-vstep', 'V<sub>step</sub> (V)', 0.05, 'any')}
        ${numField('vgs-settle', 'settle <i>t</i> (s)', 1e-3, 'any')}
      </div>
      <div class="actions">
        <button class="btn btn-primary" id="btn-vgs">Run V<sub>oc</sub>(L<sub>g</sub>)</button>
        <span class="status" id="status-vgs"></span>
      </div>
      <div id="progress-vgs"></div>
      <div class="pane-hint">For each L<sub>g</sub>, runs a 2D sweep with periodic lateral
        BCs and a single centred absorber GB. V<sub>oc</sub>(L<sub>g</sub>) is the
        published headline curve for grain-boundary-limited cells; smaller grains push
        more recombination through the GB band and lower V<sub>oc</sub>.</div>
    </div>`

  const progressBar: ProgressBarHandle = createProgressBar(
    container.querySelector<HTMLDivElement>('#progress-vgs')!,
  )
  const btn = container.querySelector<HTMLButtonElement>('#btn-vgs')!
  const sizesInput = container.querySelector<HTMLInputElement>('#vgs-sizes')!

  btn.addEventListener('click', () => {
    const active = opts.getActiveDevice()
    if (!active) {
      setStatus('status-vgs', 'No active device. Select one in the tree.', true)
      return
    }
    const grain_sizes_nm = parseGrainSizesCsv(sizesInput.value)
    if (grain_sizes_nm.length === 0) {
      setStatus(
        'status-vgs',
        'Enter at least one positive grain size in nm (comma-separated).',
        true,
      )
      return
    }
    btn.disabled = true
    progressBar.reset()
    setStatus('status-vgs', 'Starting job…')

    const params = {
      grain_sizes_nm,
      tau_gb_n: readNum('vgs-tau-n', 1e-9),
      tau_gb_p: readNum('vgs-tau-p', 1e-9),
      gb_width: readNum('vgs-width-nm', 10) * 1e-9,
      Nx: Math.max(2, Math.round(readNum('vgs-Nx', 8))),
      Ny_per_layer: Math.max(4, Math.round(readNum('vgs-Nyl', 8))),
      V_max: readNum('vgs-vmax', 1.2),
      V_step: readNum('vgs-vstep', 0.05),
      settle_t: readNum('vgs-settle', 1e-3),
      illuminated: true,
    }
    const t0 = performance.now()
    const snapshot: DeviceConfig = JSON.parse(JSON.stringify(active.config))

    startJob('voc_grain_sweep', active.config, params)
      .then(jobId => {
        setStatus('status-vgs', 'Running V_oc(L_g)…')
        streamJobEvents<VocGrainSweepResult & { active_physics?: string }>(jobId, {
          onProgress: (ev) => progressBar.update(ev),
          onResult: (result) => {
            const { active_physics, ...pure } =
              result as VocGrainSweepResult & { active_physics?: string }
            const runResult: RunResult = { kind: 'voc_grain_sweep', data: pure }
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
              'status-vgs',
              `Done · ${pure.grain_sizes_nm.length} grain sizes · ` +
                `V_oc range ${(Math.min(...pure.V_oc_V) * 1e3).toFixed(1)}–` +
                `${(Math.max(...pure.V_oc_V) * 1e3).toFixed(1)} mV`,
            )
          },
          onError: (msg) => {
            progressBar.error(msg)
            setStatus('status-vgs', `Error: ${msg}`, true)
          },
          onDone: () => {
            btn.disabled = false
          },
        })
      })
      .catch(e => {
        progressBar.error((e as Error).message)
        setStatus('status-vgs', `Error: ${(e as Error).message}`, true)
        btn.disabled = false
      })
  })
}

function randomRunId(): string {
  return 'r-' + Math.random().toString(36).slice(2, 10) + Date.now().toString(36)
}
