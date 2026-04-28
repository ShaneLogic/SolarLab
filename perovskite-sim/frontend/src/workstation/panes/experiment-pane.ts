/**
 * Unified experiment pane — a single dropdown selects the experiment type,
 * and the corresponding form renders below. Replaces the cluttered per-experiment
 * tab bar in GoldenLayout.
 *
 * The J-V pane internally dispatches kind={jv, current_decomp, spatial} based
 * on its output-view checkboxes, so those two kinds do not get their own
 * dropdown entries — their result shapes are still handled by the main-plot
 * renderer and the state reducer.
 */
import type { DeviceConfig } from '../../types'
import type { Run, ExperimentKind } from '../types'
import { mountJVPane } from './jv-pane'
import { mountImpedancePane } from './impedance-pane'
import { mountDegradationPane } from './degradation-pane'
import { mountTPVPane } from './tpv-pane'
import { mountDarkJVPane } from './dark-jv-pane'
import { mountSunsVocPane } from './suns-voc-pane'
import { mountVocTPane } from './voc-t-pane'
import { mountEQEPane } from './eqe-pane'
import { mountELPane } from './el-pane'
import { mountMottSchottkyPane } from './mott-schottky-pane'
import { mountJV2DPane } from './jv-2d-pane'
import { mountVocGrainSweepPane } from './voc-grain-sweep-pane'

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

  // Reads the kind from run.result.kind at commit time instead of pre-binding
  // it per pane. Lets the J-V pane dispatch one of {jv, current_decomp,
  // spatial} dynamically based on its "decompose current" / "save spatial
  // profiles" checkboxes — a single pane yields three experiment kinds.
  const paneOpts = () => ({
    getActiveDevice: opts.getActiveDevice,
    onRunComplete: (deviceId: string, run: Run) =>
      opts.onRunComplete(deviceId, run.result.kind, run),
  })

  interface ExperimentGroup {
    label: string
    entries: ExperimentEntry[]
  }

  const groups: ExperimentGroup[] = [
    {
      label: 'Illuminated J\u2013V',
      entries: [
        {
          kind: 'jv', label: 'J\u2013V Sweep',
          mount: (el) => mountJVPane(el, paneOpts()),
        },
        {
          kind: 'suns_voc', label: 'Suns\u2013V\u2092c',
          mount: (el) => mountSunsVocPane(el, paneOpts()),
        },
        {
          kind: 'voc_t', label: 'V\u2092c(T) \u2014 activation energy',
          mount: (el) => mountVocTPane(el, paneOpts()),
        },
      ],
    },
    {
      label: 'Dark characterisation',
      entries: [
        {
          kind: 'dark_jv', label: 'Dark J\u2013V (ideality, J\u2080)',
          mount: (el) => mountDarkJVPane(el, paneOpts()),
        },
        {
          kind: 'mott_schottky', label: 'Mott\u2013Schottky (C\u2013V)',
          mount: (el) => mountMottSchottkyPane(el, paneOpts()),
        },
      ],
    },
    {
      label: 'Spectral',
      entries: [
        {
          kind: 'eqe', label: 'EQE / IPCE',
          mount: (el) => mountEQEPane(el, paneOpts()),
        },
        {
          kind: 'el', label: 'Electroluminescence (EL, \u0394V\u2099\u1d63)',
          mount: (el) => mountELPane(el, paneOpts()),
        },
      ],
    },
    {
      label: 'Transient',
      entries: [
        {
          kind: 'impedance', label: 'Impedance',
          mount: (el) => mountImpedancePane(el, paneOpts()),
        },
        {
          kind: 'tpv', label: 'Transient Photovoltage (TPV)',
          mount: (el) => mountTPVPane(el, paneOpts()),
        },
        {
          kind: 'degradation', label: 'Degradation',
          mount: (el) => mountDegradationPane(el, paneOpts()),
        },
      ],
    },
    {
      label: '2D / Microstructural (Stage A/B)',
      entries: [
        {
          kind: 'jv_2d', label: 'J–V Sweep (2D)',
          mount: (el) => mountJV2DPane(el, paneOpts()),
        },
        {
          kind: 'voc_grain_sweep', label: 'V_oc(L_g) Grain Sweep',
          mount: (el) => mountVocGrainSweepPane(el, paneOpts()),
        },
      ],
    },
  ]

  const experiments: ExperimentEntry[] = groups.flatMap(g => g.entries)

  select.innerHTML = groups
    .map(g => {
      const opts = g.entries
        .map(e => `<option value="${e.kind}">${e.label}</option>`)
        .join('')
      return `<optgroup label="${g.label}">${opts}</optgroup>`
    })
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
