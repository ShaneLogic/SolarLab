/**
 * Stage-A 2D J-V sweep pane (Phase 6).
 *
 * MVP scope: drives the backend kind="jv_2d" dispatcher with the lateral-
 * length / Nx / Ny_per_layer / V_step / settle_t knobs and lands the result
 * in the run tree. The lateral-uniform Stage-A check passes the same physics
 * as 1D J-V; subsequent phases (microstructure, dual ions, etc.) will add
 * inputs here as they are implemented.
 */
import { startJob, streamJobEvents } from '../../job-stream'
import { createProgressBar, type ProgressBarHandle } from '../../progress'
import { setStatus, numField, readNum } from '../../ui-helpers'
import type { DeviceConfig, JV2DResult } from '../../types'
import type { Run, RunResult } from '../types'

export interface JV2DPaneOptions {
  getActiveDevice: () => { id: string; config: DeviceConfig } | null
  onRunComplete: (deviceId: string, run: Run) => void
}

export function mountJV2DPane(container: HTMLElement, opts: JV2DPaneOptions): void {
  container.innerHTML = `
    <div class="card">
      <h3>2D J–V Sweep (Stage A — lateral-uniform)</h3>
      <div class="form-grid">
        ${numField('jv2d-Lx-nm', 'Lateral length, <i>L<sub>x</sub></i> (nm)', 500, '1')}
        ${numField('jv2d-Nx', 'N<sub>x</sub> (lateral intervals)', 10, '1')}
        ${numField('jv2d-Nyl', 'N<sub>y</sub> per layer', 20, '1')}
        ${numField('jv2d-vmax', 'V<sub>max</sub> (V)', 1.2, 'any')}
        ${numField('jv2d-vstep', 'V<sub>step</sub> (V)', 0.05, 'any')}
        ${numField('jv2d-settle', 'settle <i>t</i> (s)', 1e-7, 'any')}
      </div>
      <div class="form-grid">
        <label class="checkbox-label">
          <input type="checkbox" id="jv2d-illum" checked />
          <span>Illuminated (AM1.5G via TMM / Beer–Lambert)</span>
        </label>
        <label class="checkbox-label">
          <input type="checkbox" id="jv2d-snaps" />
          <span>Save 2D snapshots (large; off by default)</span>
        </label>
        <label class="row-label">
          <span>Lateral BC</span>
          <select id="jv2d-bc" class="config-select">
            <option value="periodic" selected>periodic</option>
            <option value="neumann">Neumann (no-flux)</option>
          </select>
        </label>
      </div>
      <fieldset class="form-fieldset">
        <legend>
          <label class="checkbox-label">
            <input type="checkbox" id="jv2d-gb-en" />
            <span>Single grain boundary (Stage B)</span>
          </label>
        </legend>
        <div class="form-grid" id="jv2d-gb-fields">
          ${numField('jv2d-gb-x-nm', 'GB <i>x</i> (nm)', 250, 'any')}
          ${numField('jv2d-gb-w-nm', 'GB width (nm)', 5, 'any')}
          ${numField('jv2d-gb-tau-n', 'τ<sub>GB</sub><sup>n</sup> (s)', 5e-8, 'any')}
          ${numField('jv2d-gb-tau-p', 'τ<sub>GB</sub><sup>p</sup> (s)', 5e-8, 'any')}
        </div>
      </fieldset>
      <div class="actions">
        <button class="btn btn-primary" id="btn-jv2d">Run 2D J–V</button>
        <span class="status" id="status-jv2d"></span>
      </div>
      <div id="progress-jv2d"></div>
      <div class="pane-hint">Stage A: ions are frozen as a static Poisson background and the
        device is laterally uniform, so this should reproduce the 1D J–V to within
        sub-mV V<sub>oc</sub>. TMM presets give physical G(x); Beer–Lambert presets
        use the equilibrated 1D profile.</div>
    </div>`

  const progressBar: ProgressBarHandle = createProgressBar(
    container.querySelector<HTMLDivElement>('#progress-jv2d')!,
  )
  const btn = container.querySelector<HTMLButtonElement>('#btn-jv2d')!
  const illumCb = container.querySelector<HTMLInputElement>('#jv2d-illum')!
  const snapsCb = container.querySelector<HTMLInputElement>('#jv2d-snaps')!
  const bcSel = container.querySelector<HTMLSelectElement>('#jv2d-bc')!

  btn.addEventListener('click', () => {
    const active = opts.getActiveDevice()
    if (!active) {
      setStatus('status-jv2d', 'No active device. Select one in the tree.', true)
      return
    }
    btn.disabled = true
    progressBar.reset()
    setStatus('status-jv2d', 'Starting job…')

    const Lx_nm = readNum('jv2d-Lx-nm', 500)
    const params: Record<string, unknown> = {
      lateral_length: Lx_nm * 1e-9,
      Nx: Math.max(2, Math.round(readNum('jv2d-Nx', 10))),
      Ny_per_layer: Math.max(4, Math.round(readNum('jv2d-Nyl', 20))),
      V_max: readNum('jv2d-vmax', 1.2),
      V_step: readNum('jv2d-vstep', 0.05),
      settle_t: readNum('jv2d-settle', 1e-7),
      illuminated: illumCb.checked,
      save_snapshots: snapsCb.checked,
      lateral_bc: bcSel.value,
    }

    // Stage-B microstructure: when the checkbox is on, pack a single GB into
    // the request body. Backend treats this as a YAML-block-shaped dict.
    const gbCb = container.querySelector<HTMLInputElement>('#jv2d-gb-en')
    if (gbCb && gbCb.checked) {
      params.microstructure = {
        grain_boundaries: [
          {
            x_position: readNum('jv2d-gb-x-nm', 250) * 1e-9,
            width: readNum('jv2d-gb-w-nm', 5) * 1e-9,
            tau_n: readNum('jv2d-gb-tau-n', 5e-8),
            tau_p: readNum('jv2d-gb-tau-p', 5e-8),
            layer_role: 'absorber',
          },
        ],
      }
    }
    const t0 = performance.now()
    const snapshot: DeviceConfig = JSON.parse(JSON.stringify(active.config))

    startJob('jv_2d', active.config, params)
      .then(jobId => {
        setStatus('status-jv2d', 'Running 2D J–V…')
        streamJobEvents<JV2DResult & { active_physics?: string }>(jobId, {
          onProgress: (ev) => progressBar.update(ev),
          onResult: (result) => {
            const { active_physics, ...pure } = result as JV2DResult & { active_physics?: string }
            const runResult: RunResult = { kind: 'jv_2d', data: pure }
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
            const nV = pure.V.length
            setStatus('status-jv2d', `Done · ${nV} voltage points · BC=${pure.lateral_bc}`)
          },
          onError: (msg) => {
            progressBar.error(msg)
            setStatus('status-jv2d', `Error: ${msg}`, true)
          },
          onDone: () => {
            btn.disabled = false
          },
        })
      })
      .catch(e => {
        progressBar.error((e as Error).message)
        setStatus('status-jv2d', `Error: ${(e as Error).message}`, true)
        btn.disabled = false
      })
  })
}

function randomRunId(): string {
  return 'r-' + Math.random().toString(36).slice(2, 10) + Date.now().toString(36)
}
