import { startJob, streamJobEvents } from '../../job-stream'
import { createProgressBar, type ProgressBarHandle } from '../../progress'
import { setStatus, numField, readNum, checkField, readCheck } from '../../ui-helpers'
import { requiresQuasiFermiJVSolver } from '../../explicit-defect-capability'
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
        <label class="form-group">
          <span>J&ndash;V solver</span>
          <select id="jvp-solver" title="Select the numerical variables and continuation driver">
            <option value="transient">Transient (Radau)</option>
            <option value="steady_state">Algebraic steady state</option>
            <option value="quasi_fermi">Quasi-Fermi (cancellation-safe)</option>
          </select>
        </label>
        ${checkField('jvp-iface', 'Interface-plane states (steady-state only)', false)}
        ${checkField('jvp-interface-boundary', 'Physical interface response (quasi-Fermi only)', false)}
        <label class="form-group">
          <span>Interface transport</span>
          <select id="jvp-interface-transport">
            <option value="fermi_richardson">Fermi-Richardson</option>
            <option value="scaps_thermionic">SCAPS thermionic</option>
            <option value="scaps_thermal_velocity">SCAPS thermal velocity</option>
          </select>
        </label>
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

  // The legacy state channel and the reciprocal physical boundary belong to
  // different drivers. Gate both controls so no-op combinations cannot be
  // submitted.
  const solverSelect = container.querySelector<HTMLSelectElement>('#jvp-solver')!
  const ifaceBox = container.querySelector<HTMLInputElement>('#jvp-iface')!
  const interfaceBoundaryBox = container.querySelector<HTMLInputElement>(
    '#jvp-interface-boundary',
  )!
  const interfaceTransportSelect = container.querySelector<HTMLSelectElement>(
    '#jvp-interface-transport',
  )!
  const syncIfaceEnabled = (): void => {
    ifaceBox.disabled = solverSelect.value !== 'steady_state'
    if (ifaceBox.disabled) ifaceBox.checked = false
    interfaceBoundaryBox.disabled = solverSelect.value !== 'quasi_fermi'
    if (interfaceBoundaryBox.disabled) interfaceBoundaryBox.checked = false
    interfaceTransportSelect.disabled = (
      interfaceBoundaryBox.disabled || !interfaceBoundaryBox.checked
    )
    if (interfaceTransportSelect.disabled) {
      interfaceTransportSelect.value = 'fermi_richardson'
    }
  }
  solverSelect.addEventListener('change', syncIfaceEnabled)
  interfaceBoundaryBox.addEventListener('change', syncIfaceEnabled)
  const syncSolverRequirement = (config: DeviceConfig | null): boolean => {
    const required = config !== null && requiresQuasiFermiJVSolver(config)
    solverSelect.disabled = required
    if (required) solverSelect.value = 'quasi_fermi'
    syncIfaceEnabled()
    return required
  }
  solverSelect.addEventListener('focus', () => {
    syncSolverRequirement(opts.getActiveDevice()?.config ?? null)
  })
  syncIfaceEnabled()

  btn.addEventListener('click', () => {
    const active = opts.getActiveDevice()
    if (!active) {
      setStatus('status-jvp', 'No active device. Select one in the tree.', true)
      return
    }
    const qfRequired = syncSolverRequirement(active.config)
    btn.disabled = true
    progressBar.reset()
    // Show an active "equilibrating" state immediately: the initial
    // steady-state solve emits no progress and can be slow, so without this
    // the bar would sit at a frozen 0% until the first sweep point lands.
    progressBar.busy('Equilibrating…')
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

    // Alternative solvers only apply to the plain J-V kind; decomposition and
    // spatial snapshots require the transient driver's per-RHS state.
    const selectedSolver = solverSelect.value
    if (qfRequired && (kind !== 'jv' || selectedSolver !== 'quasi_fermi')) {
      const message = 'This stack requires the Quasi-Fermi J-V solver; decomposition and spatial-profile sweeps are not certified.'
      progressBar.error(message)
      setStatus('status-jvp', `Error: ${message}`, true)
      btn.disabled = false
      return
    }
    const requestedGrid = Math.max(3, Math.round(readNum('jvp-N', 60)))
    const minimumGrid = active.config.simulation_hints?.min_N_grid
    if (minimumGrid !== undefined && requestedGrid < minimumGrid) {
      const message = `This stack requires N_grid >= ${minimumGrid}; increase the grid before running.`
      progressBar.error(message)
      setStatus('status-jvp', `Error: ${message}`, true)
      btn.disabled = false
      return
    }
    const params = {
      N_grid: requestedGrid,
      n_points: Math.max(2, Math.round(readNum('jvp-np', 30))),
      v_rate: readNum('jvp-rate', 1.0),
      V_max: readNum('jvp-vmax', 1.4),
      illuminated: true,
      solver: kind === 'jv' ? selectedSolver : 'transient',
      iface_states: (
        kind === 'jv'
        && selectedSolver === 'steady_state'
        && readCheck('jvp-iface', false)
      ),
      interface_boundary: (
        kind === 'jv'
        && selectedSolver === 'quasi_fermi'
        && readCheck('jvp-interface-boundary', false)
      ),
      interface_transport_model: (
        kind === 'jv'
        && selectedSolver === 'quasi_fermi'
        && readCheck('jvp-interface-boundary', false)
      ) ? interfaceTransportSelect.value : 'fermi_richardson',
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
