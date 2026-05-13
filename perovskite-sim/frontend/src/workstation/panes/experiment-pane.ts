/**
 * Unified experiment pane — a single dropdown selects the experiment type,
 * and the corresponding form renders below. Replaces the cluttered per-experiment
 * tab bar in GoldenLayout.
 *
 * The J-V pane internally dispatches kind={jv, current_decomp, spatial} based
 * on its output-view checkboxes, so those two kinds do not get their own
 * dropdown entries — their result shapes are still handled by the main-plot
 * renderer and the state reducer.
 *
 * The selector is a custom listbox (not a native <select>) because native
 * <option> elements strip HTML so V<sub>oc</sub>-style subscripts cannot
 * render. Behaviour matches a native select for keyboard/mouse use:
 * click/Space/Enter to open, ArrowUp/ArrowDown to move, Enter/Space to
 * pick, Escape or outside-click to close. The labelHTML strings are
 * hard-coded source constants (no user input flows in), so the inline
 * innerHTML is safe.
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
  /** HTML — may contain <sub> for proper subscript typography. Static source constants only. */
  labelHTML: string
  mount: (container: HTMLElement) => void
}

interface ExperimentGroup {
  label: string
  entries: ExperimentEntry[]
}

export function mountExperimentPane(container: HTMLElement, opts: ExperimentPaneOptions): void {
  // Reads the kind from run.result.kind at commit time instead of pre-binding
  // it per pane. Lets the J-V pane dispatch one of {jv, current_decomp,
  // spatial} dynamically based on its "decompose current" / "save spatial
  // profiles" checkboxes — a single pane yields three experiment kinds.
  const paneOpts = () => ({
    getActiveDevice: opts.getActiveDevice,
    onRunComplete: (deviceId: string, run: Run) =>
      opts.onRunComplete(deviceId, run.result.kind, run),
  })

  const groups: ExperimentGroup[] = [
    {
      label: 'Illuminated J\u2013V',
      entries: [
        {
          kind: 'jv', labelHTML: 'J\u2013V Sweep',
          mount: (el) => mountJVPane(el, paneOpts()),
        },
        {
          kind: 'suns_voc', labelHTML: 'Suns\u2013V<sub>oc</sub>',
          mount: (el) => mountSunsVocPane(el, paneOpts()),
        },
        {
          kind: 'voc_t', labelHTML: 'V<sub>oc</sub>(T) \u2014 activation energy',
          mount: (el) => mountVocTPane(el, paneOpts()),
        },
      ],
    },
    {
      label: 'Dark characterisation',
      entries: [
        {
          kind: 'dark_jv', labelHTML: 'Dark J\u2013V (ideality, J<sub>0</sub>)',
          mount: (el) => mountDarkJVPane(el, paneOpts()),
        },
        {
          kind: 'mott_schottky', labelHTML: 'Mott\u2013Schottky (C\u2013V)',
          mount: (el) => mountMottSchottkyPane(el, paneOpts()),
        },
      ],
    },
    {
      label: 'Spectral',
      entries: [
        {
          kind: 'eqe', labelHTML: 'EQE / IPCE',
          mount: (el) => mountEQEPane(el, paneOpts()),
        },
        {
          kind: 'el', labelHTML: 'Electroluminescence (EL, \u0394V<sub>nr</sub>)',
          mount: (el) => mountELPane(el, paneOpts()),
        },
      ],
    },
    {
      label: 'Transient',
      entries: [
        {
          kind: 'impedance', labelHTML: 'Impedance',
          mount: (el) => mountImpedancePane(el, paneOpts()),
        },
        {
          kind: 'tpv', labelHTML: 'Transient Photovoltage (TPV)',
          mount: (el) => mountTPVPane(el, paneOpts()),
        },
        {
          kind: 'degradation', labelHTML: 'Degradation',
          mount: (el) => mountDegradationPane(el, paneOpts()),
        },
      ],
    },
    {
      label: '2D / Microstructural (Stage A/B)',
      entries: [
        {
          kind: 'jv_2d', labelHTML: 'J–V Sweep (2D)',
          mount: (el) => mountJV2DPane(el, paneOpts()),
        },
        {
          kind: 'voc_grain_sweep', labelHTML: 'V<sub>oc</sub>(L<sub>g</sub>) Grain Sweep',
          mount: (el) => mountVocGrainSweepPane(el, paneOpts()),
        },
      ],
    },
  ]

  const entries: ExperimentEntry[] = groups.flatMap(g => g.entries)

  const groupHTML = groups
    .map(g => {
      const optionHTML = g.entries
        .map(e => `<div class="experiment-dropdown-option" role="option" tabindex="-1" data-kind="${e.kind}">${e.labelHTML}</div>`)
        .join('')
      return `<div class="experiment-dropdown-group">
        <div class="experiment-dropdown-group-label">${escapeText(g.label)}</div>
        ${optionHTML}
      </div>`
    })
    .join('')

  container.innerHTML = `
    <div class="experiment-pane">
      <div class="experiment-selector">
        <span class="experiment-selector-label">Experiment</span>
        <div class="experiment-dropdown" data-open="false">
          <button type="button" class="experiment-dropdown-button" aria-haspopup="listbox" aria-expanded="false">
            <span class="experiment-dropdown-current"></span>
            <span class="experiment-dropdown-caret" aria-hidden="true">▾</span>
          </button>
          <div class="experiment-dropdown-popover" role="listbox" hidden>${groupHTML}</div>
        </div>
      </div>
      <div id="exp-body"></div>
    </div>`

  const wrapper = container.querySelector<HTMLDivElement>('.experiment-dropdown')!
  const button = container.querySelector<HTMLButtonElement>('.experiment-dropdown-button')!
  const current = container.querySelector<HTMLSpanElement>('.experiment-dropdown-current')!
  const popover = container.querySelector<HTMLDivElement>('.experiment-dropdown-popover')!
  const options = Array.from(popover.querySelectorAll<HTMLDivElement>('.experiment-dropdown-option'))
  const body = container.querySelector<HTMLDivElement>('#exp-body')!

  let selectedKind: ExperimentKind = entries[0].kind

  function selectKind(kind: ExperimentKind): void {
    const entry = entries.find(e => e.kind === kind)
    if (!entry) return
    selectedKind = kind
    current.innerHTML = entry.labelHTML
    options.forEach(o =>
      o.classList.toggle('experiment-dropdown-option-selected', o.dataset.kind === kind),
    )
    body.innerHTML = ''
    entry.mount(body)
  }

  function setOpen(open: boolean): void {
    wrapper.dataset.open = String(open)
    button.setAttribute('aria-expanded', String(open))
    popover.hidden = !open
    if (open) {
      const sel = options.find(o => o.dataset.kind === selectedKind) ?? options[0]
      sel.focus()
    }
  }

  button.addEventListener('click', (e) => {
    e.stopPropagation()
    setOpen(!!popover.hidden)
  })

  button.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp' || e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      setOpen(true)
    }
  })

  options.forEach((opt, idx) => {
    opt.addEventListener('click', () => {
      const kind = opt.dataset.kind as ExperimentKind | undefined
      if (!kind) return
      selectKind(kind)
      setOpen(false)
      button.focus()
    })
    opt.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowDown') {
        e.preventDefault()
        options[(idx + 1) % options.length].focus()
      } else if (e.key === 'ArrowUp') {
        e.preventDefault()
        options[(idx - 1 + options.length) % options.length].focus()
      } else if (e.key === 'Home') {
        e.preventDefault()
        options[0].focus()
      } else if (e.key === 'End') {
        e.preventDefault()
        options[options.length - 1].focus()
      } else if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault()
        opt.click()
      } else if (e.key === 'Escape' || e.key === 'Tab') {
        e.preventDefault()
        setOpen(false)
        button.focus()
      }
    })
  })

  document.addEventListener('click', (e) => {
    if (!wrapper.contains(e.target as Node)) setOpen(false)
  })

  selectKind(selectedKind)
}

/** Escape plain text for HTML insertion (group labels). */
function escapeText(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
}
