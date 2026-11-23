/**
 * Declarative experiment capability & surface catalog (P06-08).
 *
 * Replaces ad-hoc `switch (kind)` / `if/elif` chains across `tree.ts`,
 * `experiment-pane.ts`, and `main-plot-pane.ts`, while providing safe
 * fallback rendering for newly registered backend capabilities that do
 * not define a bespoke form or plot renderer.
 */
import type { DeviceConfig } from '../types'
import type { ExperimentKind, Run } from './types'
import { mountJVPane } from './panes/jv-pane'
import { mountImpedancePane } from './panes/impedance-pane'
import { mountDynamicDefectTransientPane } from './panes/dynamic-defect-transient-pane'
import { mountDegradationPane } from './panes/degradation-pane'
import { mountTPVPane } from './panes/tpv-pane'
import { mountDarkJVPane } from './panes/dark-jv-pane'
import { mountSunsVocPane } from './panes/suns-voc-pane'
import { mountVocTPane } from './panes/voc-t-pane'
import { mountEQEPane } from './panes/eqe-pane'
import { mountELPane } from './panes/el-pane'
import { mountMottSchottkyPane } from './panes/mott-schottky-pane'
import { mountJV2DPane } from './panes/jv-2d-pane'
import { mountVocGrainSweepPane } from './panes/voc-grain-sweep-pane'

export interface SharedExperimentPaneOptions {
  readonly getActiveDevice: () => { id: string; config: DeviceConfig } | null
  readonly onRunComplete: (deviceId: string, run: Run) => void
}

export interface MountedExperimentForm {
  updateDevice(resetDefaults?: boolean): void
}

export type ExperimentCategory =
  | 'Illuminated J\u2013V'
  | 'Dark characterisation'
  | 'Spectral'
  | 'Transient'
  | '2D / Microstructure'

export interface ExperimentDescriptor {
  readonly kind: ExperimentKind
  readonly category: ExperimentCategory
  /** Short HTML label used in the workspace tree (static source constant only). */
  readonly treeLabelHTML: string
  /** Full HTML label used in the experiment dropdown (static source constant only). */
  readonly selectorLabelHTML: string
  /** Exported renderer symbol in `main-plot-pane.ts` (matches Python ExperimentRegistry). */
  readonly rendererSymbol: string
  /** Present when this experiment appears as a standalone dropdown entry. */
  readonly mountForm?: (
    container: HTMLElement,
    opts: SharedExperimentPaneOptions,
  ) => MountedExperimentForm | void
  /** Present when selecting this kind delegates to a parent form with preset toggles. */
  readonly delegateToKind?: ExperimentKind
  readonly applyFormPreset?: (formHost: HTMLElement) => void
}

function setJVOutputPreset(
  formHost: HTMLElement,
  mode: 'jv' | 'current_decomp' | 'spatial',
): void {
  const solver = formHost.querySelector<HTMLSelectElement>('#jvp-solver')
  if (solver && (mode === 'current_decomp' || mode === 'spatial')) {
    solver.value = 'transient'
    solver.dispatchEvent(new Event('change', { bubbles: true }))
  }
  const decomp = formHost.querySelector<HTMLInputElement>('#jvp-decomp')
  const spatial = formHost.querySelector<HTMLInputElement>('#jvp-spatial')
  if (decomp) decomp.checked = mode === 'current_decomp'
  if (spatial) spatial.checked = mode === 'spatial'
}

export const EXPERIMENT_CATALOG: readonly ExperimentDescriptor[] = Object.freeze([
  {
    kind: 'jv',
    category: 'Illuminated J\u2013V',
    treeLabelHTML: 'J\u2013V Sweep',
    selectorLabelHTML: 'J\u2013V Sweep',
    rendererSymbol: 'renderJV',
    mountForm: (el, opts) => mountJVPane(el, opts),
    applyFormPreset: el => setJVOutputPreset(el, 'jv'),
  },
  {
    kind: 'current_decomp',
    category: 'Illuminated J\u2013V',
    treeLabelHTML: 'Current Decomp.',
    selectorLabelHTML: 'Current Decomposition',
    rendererSymbol: 'renderCurrentDecomp',
    delegateToKind: 'jv',
    applyFormPreset: el => setJVOutputPreset(el, 'current_decomp'),
  },
  {
    kind: 'spatial',
    category: 'Illuminated J\u2013V',
    treeLabelHTML: 'Spatial Profiles',
    selectorLabelHTML: 'Spatial Profiles',
    rendererSymbol: 'renderSpatialProfiles',
    delegateToKind: 'jv',
    applyFormPreset: el => setJVOutputPreset(el, 'spatial'),
  },
  {
    kind: 'suns_voc',
    category: 'Illuminated J\u2013V',
    treeLabelHTML: 'Suns\u2013V<sub>oc</sub>',
    selectorLabelHTML: 'Suns\u2013V<sub>oc</sub>',
    rendererSymbol: 'renderSunsVoc',
    mountForm: (el, opts) => mountSunsVocPane(el, opts),
  },
  {
    kind: 'voc_t',
    category: 'Illuminated J\u2013V',
    treeLabelHTML: 'V<sub>oc</sub>(T)',
    selectorLabelHTML: 'V<sub>oc</sub>(T)',
    rendererSymbol: 'renderVocT',
    mountForm: (el, opts) => mountVocTPane(el, opts),
  },
  {
    kind: 'dark_jv',
    category: 'Dark characterisation',
    treeLabelHTML: 'Dark J\u2013V',
    selectorLabelHTML: 'Dark J\u2013V',
    rendererSymbol: 'renderDarkJV',
    mountForm: (el, opts) => mountDarkJVPane(el, opts),
  },
  {
    kind: 'mott_schottky',
    category: 'Dark characterisation',
    treeLabelHTML: 'Mott\u2013Schottky (C\u2013V)',
    selectorLabelHTML: 'Mott\u2013Schottky (C\u2013V)',
    rendererSymbol: 'renderMottSchottky',
    mountForm: (el, opts) => mountMottSchottkyPane(el, opts),
  },
  {
    kind: 'eqe',
    category: 'Spectral',
    treeLabelHTML: 'EQE / IPCE',
    selectorLabelHTML: 'EQE / IPCE',
    rendererSymbol: 'renderEQE',
    mountForm: (el, opts) => mountEQEPane(el, opts),
  },
  {
    kind: 'el',
    category: 'Spectral',
    treeLabelHTML: 'EL / \u0394V<sub>nr</sub>',
    selectorLabelHTML: 'EL / \u0394V<sub>nr</sub>',
    rendererSymbol: 'renderEL',
    mountForm: (el, opts) => mountELPane(el, opts),
  },
  {
    kind: 'impedance',
    category: 'Transient',
    treeLabelHTML: 'Impedance',
    selectorLabelHTML: 'Impedance',
    rendererSymbol: 'renderImpedance',
    mountForm: (el, opts) => mountImpedancePane(el, opts),
  },
  {
    kind: 'dynamic_defect_transient',
    category: 'Transient',
    treeLabelHTML: 'Defect\u2013Ion Transient',
    selectorLabelHTML: 'Defect\u2013Ion Transient',
    rendererSymbol: 'renderDynamicDefectTransient',
    mountForm: (el, opts) => mountDynamicDefectTransientPane(el, opts),
  },
  {
    kind: 'tpv',
    category: 'Transient',
    treeLabelHTML: 'TPV',
    selectorLabelHTML: 'Transient Photovoltage (TPV)',
    rendererSymbol: 'renderTPV',
    mountForm: (el, opts) => mountTPVPane(el, opts),
  },
  {
    kind: 'degradation',
    category: 'Transient',
    treeLabelHTML: 'Degradation',
    selectorLabelHTML: 'Degradation',
    rendererSymbol: 'renderDegradation',
    mountForm: (el, opts) => mountDegradationPane(el, opts),
  },
  {
    kind: 'jv_2d',
    category: '2D / Microstructure',
    treeLabelHTML: 'J\u2013V Sweep (2D)',
    selectorLabelHTML: 'J\u2013V Sweep (2D)',
    rendererSymbol: 'renderJV2D',
    mountForm: (el, opts) => mountJV2DPane(el, opts),
  },
  {
    kind: 'voc_grain_sweep',
    category: '2D / Microstructure',
    treeLabelHTML: 'V<sub>oc</sub>(L<sub>g</sub>) Grain Sweep',
    selectorLabelHTML: 'V<sub>oc</sub>(L<sub>g</sub>) Grain Sweep',
    rendererSymbol: 'renderVocGrainSweep',
    mountForm: (el, opts) => mountVocGrainSweepPane(el, opts),
  },
])

export const EXPERIMENT_CATEGORY_ORDER: readonly ExperimentCategory[] = Object.freeze([
  'Illuminated J\u2013V',
  'Dark characterisation',
  'Spectral',
  'Transient',
  '2D / Microstructure',
])

const DESCRIPTOR_BY_KIND = new Map<string, ExperimentDescriptor>(
  EXPERIMENT_CATALOG.map(entry => [entry.kind, entry]),
)

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}

export function getExperimentDescriptor(kind: string): ExperimentDescriptor | undefined {
  return DESCRIPTOR_BY_KIND.get(kind)
}

export function getExperimentTreeLabelHTML(kind: string): string {
  return DESCRIPTOR_BY_KIND.get(kind)?.treeLabelHTML ?? escapeHtml(kind)
}

export function getSelectableExperimentGroups(): ReadonlyArray<{
  readonly label: ExperimentCategory
  readonly entries: readonly ExperimentDescriptor[]
}> {
  return EXPERIMENT_CATEGORY_ORDER.map(category => ({
    label: category,
    entries: EXPERIMENT_CATALOG.filter(
      entry => entry.category === category && typeof entry.mountForm === 'function',
    ),
  })).filter(group => group.entries.length > 0)
}
