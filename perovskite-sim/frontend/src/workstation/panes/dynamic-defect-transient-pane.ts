import { dynamicDefectTransientEligibility } from '../../explicit-defect-capability'
import { startJob, streamJobEvents } from '../../job-stream'
import { createProgressBar, type ProgressBarHandle } from '../../progress'
import type { DeviceConfig, DynamicDefectTransientResult } from '../../types'
import { numField, readNum, setStatus } from '../../ui-helpers'
import type { Run, RunResult } from '../types'

export interface DynamicDefectTransientPaneOptions {
  getActiveDevice: () => { id: string; config: DeviceConfig } | null
  onRunComplete: (deviceId: string, run: Run) => void
}

export function mountDynamicDefectTransientPane(
  container: HTMLElement,
  opts: DynamicDefectTransientPaneOptions,
): void {
  const initial = opts.getActiveDevice()
  const initialEligibility = initial
    ? dynamicDefectTransientEligibility(initial.config)
    : { eligible: false, reasons: ['no active device'], N_grid: 4 }
  container.innerHTML = `
    <div class="card">
      <h3>Certified Defect–Ion Transient</h3>
      <div class="form-grid">
        ${numField('dit-N', 'N<sub>grid</sub>', initialEligibility.N_grid, '1')}
        ${numField('dit-dv', 'Voltage step (mV)', 50, '1')}
        ${numField('dit-t1', 'Early sample (s)', 1e-8, 'any')}
        ${numField('dit-t2', 'Intermediate sample (s)', 1e-6, 'any')}
        ${numField('dit-tend', 'End time (s)', 1e-4, 'any')}
      </div>
      <div class="actions">
        <button class="btn btn-primary" id="btn-dit">Run Defect–Ion Transient</button>
        <span class="status" id="status-dit"></span>
      </div>
      <div id="progress-dit"></div>
      <div class="pane-hint" data-test="dit-capability">${
        initialEligibility.eligible
          ? 'Eligible: interface defect + absorber positive ions'
          : `Unavailable: ${initialEligibility.reasons.join('; ')}`
      }</div>
    </div>`

  const progressBar: ProgressBarHandle = createProgressBar(
    container.querySelector<HTMLDivElement>('#progress-dit')!,
  )
  const btn = container.querySelector<HTMLButtonElement>('#btn-dit')!

  btn.addEventListener('click', () => {
    const active = opts.getActiveDevice()
    if (!active) {
      setStatus('status-dit', 'No active device. Select one in the tree.', true)
      return
    }
    const eligibility = dynamicDefectTransientEligibility(active.config)
    if (!eligibility.eligible) {
      setStatus(
        'status-dit',
        `Unsupported configuration: ${eligibility.reasons.join('; ')}`,
        true,
      )
      return
    }
    const early = readNum('dit-t1', 1e-8)
    const intermediate = readNum('dit-t2', 1e-6)
    const end = readNum('dit-tend', 1e-4)
    if (!(early > 0 && intermediate > early && end > intermediate)) {
      setStatus(
        'status-dit',
        'Sample times must satisfy 0 < early < intermediate < end.',
        true,
      )
      return
    }
    const voltageStep = readNum('dit-dv', 50) * 1e-3
    const params = {
      N_grid: Math.max(4, Math.round(readNum('dit-N', eligibility.N_grid))),
      times_s: [0, early, intermediate, end],
      voltage_V: [0, voltageStep, voltageStep, voltageStep],
      illuminated: false,
      method: 'dynamic_defect_transient_certified' as const,
    }
    btn.disabled = true
    progressBar.reset()
    progressBar.busy()
    setStatus('status-dit', 'Starting job…')
    const t0 = performance.now()
    const snapshot: DeviceConfig = JSON.parse(JSON.stringify(active.config))

    startJob('dynamic_defect_transient', active.config, params)
      .then(jobId => {
        setStatus('status-dit', 'Running certified transient…')
        streamJobEvents<DynamicDefectTransientResult & { active_physics?: string }>(
          jobId,
          {
            onProgress: ev => progressBar.update(ev),
            onResult: result => {
              const { active_physics, ...pure } = result
              const runResult: RunResult = {
                kind: 'dynamic_defect_transient',
                data: pure,
              }
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
              setStatus('status-dit', 'Done')
            },
            onError: message => {
              progressBar.error(message)
              setStatus('status-dit', `Error: ${message}`, true)
            },
            onDone: () => {
              btn.disabled = false
            },
          },
        )
      })
      .catch(error => {
        const message = (error as Error).message
        progressBar.error(message)
        setStatus('status-dit', `Error: ${message}`, true)
        btn.disabled = false
      })
  })
}

function randomRunId(): string {
  return 'r-' + Math.random().toString(36).slice(2, 10) + Date.now().toString(36)
}
