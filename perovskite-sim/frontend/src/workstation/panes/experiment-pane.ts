/**
 * Unified experiment pane — a single dropdown selects the experiment type,
 * and the corresponding form renders below. Replaces the cluttered per-experiment
 * tab bar in GoldenLayout.
 */
import type { DeviceConfig } from '../../types'
import type { Run, ExperimentKind } from '../types'
import { mountJVPane } from './jv-pane'
import { mountImpedancePane } from './impedance-pane'
import { mountDegradationPane } from './degradation-pane'
import { mountTPVPane } from './tpv-pane'
import { mountCurrentDecompPane } from './current-decomp-pane'
import { mountSpatialPane } from './spatial-pane'

export interface ExperimentPaneOptions {
  getActiveDevice: () => { id: string; config: DeviceConfig } | null
  onRunComplete: (deviceId: string, kind: ExperimentKind, run: Run) => void
}

interface ExperimentEntry {
  kind: ExperimentKind
  label: string
  mount: (container: HTMLElement) => void
}

export function mountExperimentPane(container: HTMLElement, opts: ExperimentPaneOptions): void {
  container.innerHTML = `
    <div class="experiment-pane">
      <div class="experiment-selector">
        <label for="exp-select">Experiment</label>
        <select id="exp-select" class="config-select experiment-dropdown"></select>
      </div>
      <div id="exp-body"></div>
    </div>`

  const select = container.querySelector<HTMLSelectElement>('#exp-select')!
  const body = container.querySelector<HTMLDivElement>('#exp-body')!

  const paneOpts = (kind: ExperimentKind) => ({
    getActiveDevice: opts.getActiveDevice,
    onRunComplete: (deviceId: string, run: Run) => opts.onRunComplete(deviceId, kind, run),
  })

  const experiments: ExperimentEntry[] = [
    {
      kind: 'jv', label: 'J\u2013V Sweep',
      mount: (el) => mountJVPane(el, paneOpts('jv')),
    },
    {
      kind: 'impedance', label: 'Impedance',
      mount: (el) => mountImpedancePane(el, paneOpts('impedance')),
    },
    {
      kind: 'degradation', label: 'Degradation',
      mount: (el) => mountDegradationPane(el, paneOpts('degradation')),
    },
    {
      kind: 'tpv', label: 'Transient Photovoltage (TPV)',
      mount: (el) => mountTPVPane(el, paneOpts('tpv')),
    },
    {
      kind: 'current_decomp', label: 'Current Decomposition',
      mount: (el) => mountCurrentDecompPane(el, paneOpts('current_decomp')),
    },
    {
      kind: 'spatial', label: 'Spatial Profiles',
      mount: (el) => mountSpatialPane(el, paneOpts('spatial')),
    },
  ]

  select.innerHTML = experiments
    .map(e => `<option value="${e.kind}">${e.label}</option>`)
    .join('')

  function renderSelected(): void {
    const kind = select.value
    const entry = experiments.find(e => e.kind === kind)
    if (!entry) return
    body.innerHTML = ''
    entry.mount(body)
  }

  select.addEventListener('change', renderSelected)
  renderSelected()
}
