import type { ProgressEvent } from './types'

export interface ProgressBarHandle {
  readonly root: HTMLElement
  update(ev: ProgressEvent): void
  /**
   * Show an indeterminate (animated, no percentage) "working" state. Use on
   * run start so the bar signals activity during the initial equilibration
   * solve — which can be slow and emits no progress — instead of a frozen 0%.
   */
  busy(label?: string): void
  done(): void
  error(message: string): void
  reset(): void
}

// Backend progress messages use plain underscore notation (Python
// convention: ``V_oc``, ``J_sc``, ``E_A``…). Render each known physics
// token as a real <sub> element so users see proper typography without
// allowing arbitrary HTML in the message stream. Longer tokens come
// first so ``EQE_EL`` matches before any future ``E_E`` would shadow.
const SUBSCRIPT_TOKENS: Array<readonly [string, string, string]> = [
  ['EQE_EL',   'EQE',  'EL'],
  ['V_inj',    'V',    'inj'],
  ['V_max',    'V',    'max'],
  ['V_min',    'V',    'min'],
  ['V_app',    'V',    'app'],
  ['V_bi',     'V',    'bi'],
  ['V_oc',     'V',    'oc'],
  ['J_inj',    'J',    'inj'],
  ['J_sc',     'J',    'sc'],
  ['E_A',      'E',    'A'],
  ['L_g',      'L',    'g'],
  ['N_grid',   'N',    'grid'],
  ['t_settle', 't',    'settle'],
]

export interface SubscriptPart {
  readonly kind: 'text' | 'subscript'
  readonly value: string
  readonly base?: string
}

// Pure-data tokeniser. Splits ``raw`` on the longest-first physics-token
// allow-list and returns an array of text / subscript parts. The
// surrounding character class keeps ``V_oc_arr`` from being mangled
// mid-identifier into ``V<sub>oc</sub>_arr``.
export function tokeniseProgressMessage(raw: string): SubscriptPart[] {
  const escapeRe = (s: string): string => {
    let out = ''
    for (const ch of s) {
      if ('.*+?^${}()|[]\\'.indexOf(ch) >= 0) out += '\\'
      out += ch
    }
    return out
  }
  const pattern = SUBSCRIPT_TOKENS.map(t => escapeRe(t[0])).join('|')
  const re = new RegExp(`(?<![A-Za-z0-9_])(${pattern})(?![A-Za-z0-9_])`, 'g')
  const parts: SubscriptPart[] = []
  let lastIndex = 0
  let m: RegExpExecArray | null
  while ((m = re.exec(raw)) !== null) {
    if (m.index > lastIndex) {
      parts.push({ kind: 'text', value: raw.slice(lastIndex, m.index) })
    }
    const token = m[1]
    const entry = SUBSCRIPT_TOKENS.find(t => t[0] === token)
    if (entry) {
      parts.push({ kind: 'subscript', value: entry[2], base: entry[1] })
    } else {
      parts.push({ kind: 'text', value: token })
    }
    lastIndex = m.index + token.length
  }
  if (lastIndex < raw.length) {
    parts.push({ kind: 'text', value: raw.slice(lastIndex) })
  }
  return parts
}

// Replaces every child of ``target`` with text nodes + ``<sub>``
// elements built from the tokenised message. Avoids innerHTML so any
// user-controllable content cannot inject markup.
export function renderProgressMessage(target: HTMLElement, raw: string): void {
  while (target.firstChild) target.removeChild(target.firstChild)
  for (const part of tokeniseProgressMessage(raw)) {
    if (part.kind === 'text') {
      target.appendChild(document.createTextNode(part.value))
    } else {
      target.appendChild(document.createTextNode(part.base ?? ''))
      const sub = document.createElement('sub')
      sub.textContent = part.value
      target.appendChild(sub)
    }
  }
}

export function createProgressBar(container: HTMLElement): ProgressBarHandle {
  container.innerHTML = `
    <div class="progress-card">
      <div class="progress-header">
        <span class="progress-stage">Idle</span>
        <span class="progress-percent">0%</span>
      </div>
      <div class="progress-bar">
        <div class="progress-fill" style="width:0%"></div>
      </div>
      <div class="progress-footer">
        <span class="progress-message"></span>
        <span class="progress-eta"></span>
      </div>
    </div>`
  const root = container.querySelector<HTMLElement>('.progress-card')!
  const stageEl = root.querySelector<HTMLElement>('.progress-stage')!
  const percentEl = root.querySelector<HTMLElement>('.progress-percent')!
  const fillEl = root.querySelector<HTMLElement>('.progress-fill')!
  const messageEl = root.querySelector<HTMLElement>('.progress-message')!
  const etaEl = root.querySelector<HTMLElement>('.progress-eta')!

  function fmtEta(s: number | null): string {
    if (s === null || !isFinite(s)) return ''
    if (s < 1) return '< 1 s remaining'
    if (s < 60) return `${Math.round(s)} s remaining`
    const m = Math.floor(s / 60)
    const r = Math.round(s - m * 60)
    return `${m} m ${r} s remaining`
  }

  function stageLabel(stage: string): string {
    // Any "*_init" kickoff stage is the pre-loop equilibration phase.
    if (stage.endsWith('_init')) return 'Equilibrating…'
    switch (stage) {
      case 'jv_forward': return 'J–V forward sweep'
      case 'jv_reverse': return 'J–V reverse sweep'
      case 'impedance': return 'Impedance spectroscopy'
      case 'degradation': return 'Degradation snapshots'
      case 'degradation_transient': return 'Degradation transient'
      default: return stage
    }
  }

  // Render the animated, percentage-less "working" state. Used both by busy()
  // (client-initiated on run start) and by update() when a kickoff frame with
  // current=0 (or total=0) arrives — neither has a meaningful percentage.
  function setIndeterminate(label: string, message: string): void {
    fillEl.classList.remove('done', 'error')
    fillEl.classList.add('indeterminate')
    fillEl.style.width = ''      // hand width back to the CSS animation
    stageEl.textContent = label
    percentEl.textContent = ''
    renderProgressMessage(messageEl, message)
    etaEl.textContent = ''
  }

  return {
    root,
    update(ev) {
      // A kickoff / phase-start frame (current<=0) or an unknown total has no
      // meaningful percentage — show the indeterminate "equilibrating" state.
      if (ev.current <= 0 || ev.total <= 0) {
        setIndeterminate(stageLabel(ev.stage), ev.message ?? '')
        return
      }
      fillEl.classList.remove('done', 'error', 'indeterminate')
      const pct = Math.round((100 * ev.current) / ev.total)
      fillEl.style.width = `${pct}%`
      percentEl.textContent = `${pct}%`
      stageEl.textContent = stageLabel(ev.stage)
      renderProgressMessage(messageEl, ev.message ?? '')
      etaEl.textContent = fmtEta(ev.eta_s)
    },
    busy(label = 'Equilibrating…') {
      setIndeterminate(label, '')
    },
    done() {
      fillEl.classList.add('done')
      fillEl.classList.remove('error', 'indeterminate')
      fillEl.style.width = '100%'
      percentEl.textContent = '100%'
      etaEl.textContent = 'Done'
    },
    error(message) {
      // If the failure hit during the indeterminate phase the fill has no
      // inline width — pin it full so the red bar is visible; otherwise leave
      // the determinate width where it stopped.
      const wasIndeterminate = fillEl.classList.contains('indeterminate')
      fillEl.classList.add('error')
      fillEl.classList.remove('done', 'indeterminate')
      if (wasIndeterminate) fillEl.style.width = '100%'
      stageEl.textContent = 'Error'
      renderProgressMessage(messageEl, message)
      etaEl.textContent = ''
    },
    reset() {
      fillEl.classList.remove('done', 'error', 'indeterminate')
      fillEl.style.width = '0%'
      percentEl.textContent = '0%'
      stageEl.textContent = 'Idle'
      while (messageEl.firstChild) messageEl.removeChild(messageEl.firstChild)
      etaEl.textContent = ''
    },
  }
}
