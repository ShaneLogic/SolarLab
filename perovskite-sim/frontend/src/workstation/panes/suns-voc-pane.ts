import { startJob, streamJobEvents } from '../../job-stream'
import { createProgressBar, type ProgressBarHandle } from '../../progress'
import { setStatus, numField, readNum } from '../../ui-helpers'
import type { DeviceConfig, SunsVocResult } from '../../types'
import type { Run, RunResult } from '../types'

export interface SunsVocPaneOptions {
  getActiveDevice: () => { id: string; config: DeviceConfig } | null
  onRunComplete: (deviceId: string, run: Run) => void
}

export function mountSunsVocPane(container: HTMLElement, opts: SunsVocPaneOptions): void {
  container.innerHTML = `
    <div class="card">
      <h3>Suns\u2013V<sub>oc</sub> Parameters</h3>
      <div class="form-grid">
        ${numField('sv-N', 'N<sub>grid</sub>', 60, '1')}
        ${numField('sv-tset', 't<sub>settle</sub> (s)', 1e-3, 'any')}
      </div>
      <label class="field-label" for="sv-suns">Suns levels (comma-separated)</label>
      <input type="text" id="sv-suns" class="config-select" value="0.01, 0.1, 1.0, 5.0, 10.0" />
      <div class="actions">
        <button class="btn btn-primary" id="btn-sv">Run Suns\u2013V<sub>oc</sub></button>
        <span class="status" id="status-sv"></span>
      </div>
      <div id="progress-sv"></div>
      <div class="pane-hint">Sweeps illumination intensity at open-circuit; builds the pseudo J\u2013V curve (<em>J<sub>sc</sub></em>(suns) &minus; <em>J</em>) vs <em>V<sub>oc</sub></em>(suns) and returns a recombination-only pseudo FF.</div>
    </div>`

  const progressBar: ProgressBarHandle = createProgressBar(
    container.querySelector<HTMLDivElement>('#progress-sv')!,
  )
  const btn = container.querySelector<HTMLButtonElement>('#btn-sv')!

  btn.addEventListener('click', () => {
    const active = opts.getActiveDevice()
    if (!active) {
      setStatus('status-sv', 'No active device. Select one in the tree.', true)
      return
    }
    btn.disabled = true
    progressBar.reset()
    setStatus('status-sv', 'Starting job\u2026')

    const sunsStr = container.querySelector<HTMLInputElement>('#sv-suns')!.value
    const sunsLevels = sunsStr
      .split(',')
      .map(s => parseFloat(s.trim()))
      .filter(x => Number.isFinite(x) && x > 0)
    if (sunsLevels.length < 2) {
      setStatus('status-sv', 'Need \u2265 2 positive suns levels.', true)
      btn.disabled = false
      return
    }

    const params = {
      N_grid: Math.max(3, Math.round(readNum('sv-N', 60))),
      t_settle: readNum('sv-tset', 1e-3),
      suns_levels: sunsLevels,
    }
    const t0 = performance.now()
    const snapshot: DeviceConfig = JSON.parse(JSON.stringify(active.config))

    startJob('suns_voc', active.config, params)
      .then(jobId => {
        setStatus('status-sv', 'Running Suns\u2013V_oc\u2026')
        streamJobEvents<SunsVocResult & { active_physics?: string }>(jobId, {
          onProgress: (ev) => progressBar.update(ev),
          onResult: (result) => {
            const { active_physics, ...pure } = result as SunsVocResult & { active_physics?: string }
            const runResult: RunResult = { kind: 'suns_voc', data: pure }
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
            setStatus('status-sv', `Done \u00b7 pFF=${(pure.pseudo_FF * 100).toFixed(1)}%`)
          },
          onError: (msg) => {
            progressBar.error(msg)
            setStatus('status-sv', `Error: ${msg}`, true)
          },
          onDone: () => {
            btn.disabled = false
          },
        })
      })
      .catch(e => {
        progressBar.error((e as Error).message)
        setStatus('status-sv', `Error: ${(e as Error).message}`, true)
        btn.disabled = false
      })
  })
}

function randomRunId(): string {
  return 'r-' + Math.random().toString(36).slice(2, 10) + Date.now().toString(36)
}
