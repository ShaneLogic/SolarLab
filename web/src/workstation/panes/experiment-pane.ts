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
import {
  getExperimentDescriptor,
  getSelectableExperimentGroups,
  type ExperimentDescriptor,
  type MountedExperimentForm,
} from '../experiment-catalog'

export interface ExperimentPaneOptions {
  getActiveDevice: () => { id: string; config: DeviceConfig } | null
  onRunComplete: (deviceId: string, kind: ExperimentKind, run: Run) => void
}

export interface ExperimentPaneHandle {
  updateDevice(resetDefaults?: boolean): void
  selectExperiment(kind: ExperimentKind): void
}

export function mountExperimentPane(container: HTMLElement, opts: ExperimentPaneOptions): ExperimentPaneHandle {
  // Golden Layout clips component content by default. Make the host own the
  // scroll area so long forms (including the Calado scan history) stay usable.
  container.classList.add('pane', 'pane-experiments')
  // Reads the kind from run.result.kind at commit time instead of pre-binding
  // it per pane. Lets the J-V pane dispatch one of {jv, current_decomp,
  // spatial} dynamically based on its "decompose current" / "save spatial
  // profiles" checkboxes — a single pane yields three experiment kinds.
  const paneOpts = () => ({
    getActiveDevice: opts.getActiveDevice,
    onRunComplete: (deviceId: string, run: Run) =>
      opts.onRunComplete(deviceId, run.result.kind, run),
  })

  const groups = getSelectableExperimentGroups()
  const entries: readonly ExperimentDescriptor[] = groups.flatMap(g => g.entries)

  const groupHTML = groups
    .map(g => {
      const optionHTML = g.entries
        .map(e => `<div class="experiment-dropdown-option" role="option" tabindex="-1" data-kind="${e.kind}">${e.selectorLabelHTML}</div>`)
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
  let activePane: MountedExperimentForm | void

  function renderUnsupportedCapability(kind: string): void {
    body.replaceChildren()
    const card = document.createElement('div')
    card.className = 'card'
    card.dataset.test = 'experiment-capability-fallback'
    const title = document.createElement('h3')
    title.textContent = `Registered capability: ${kind}`
    const note = document.createElement('p')
    note.textContent =
      'This registered experiment capability has no bespoke legacy form in the workstation dock; inspect or submit it through its dedicated V2 route.'
    card.append(title, note)
    body.appendChild(card)
  }

  function selectKind(kind: ExperimentKind): void {
    const entry = entries.find(e => e.kind === kind)
    if (!entry || !entry.mountForm) {
      selectedKind = kind
      current.textContent = kind
      options.forEach(o => o.classList.remove('experiment-dropdown-option-selected'))
      activePane = undefined
      renderUnsupportedCapability(kind)
      return
    }
    if (kind === selectedKind && body.childElementCount > 0) return
    selectedKind = kind
    current.innerHTML = entry.selectorLabelHTML
    options.forEach(o =>
      o.classList.toggle('experiment-dropdown-option-selected', o.dataset.kind === kind),
    )
    body.innerHTML = ''
    activePane = entry.mountForm(body, paneOpts())
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

  return {
    updateDevice: resetDefaults => activePane?.updateDevice(resetDefaults),
    selectExperiment: kind => {
      const descriptor = getExperimentDescriptor(kind)
      const target = descriptor?.delegateToKind ?? kind
      selectKind(target)
      setOpen(false)
      descriptor?.applyFormPreset?.(body)
    },
  }
}

/** Escape plain text for HTML insertion (group labels). */
function escapeText(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
}
