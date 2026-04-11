import type { ProgressEvent } from './types'

export interface ProgressBarHandle {
  readonly root: HTMLElement
  update(ev: ProgressEvent): void
  done(): void
  error(message: string): void
  reset(): void
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
    switch (stage) {
      case 'jv_forward': return 'J–V forward sweep'
      case 'jv_reverse': return 'J–V reverse sweep'
      case 'impedance': return 'Impedance spectroscopy'
      case 'degradation': return 'Degradation snapshots'
      case 'degradation_transient': return 'Degradation transient'
      default: return stage
    }
  }

  return {
    root,
    update(ev) {
      fillEl.classList.remove('done', 'error')
      const pct = ev.total > 0 ? Math.round((100 * ev.current) / ev.total) : 0
      fillEl.style.width = `${pct}%`
      percentEl.textContent = `${pct}%`
      stageEl.textContent = stageLabel(ev.stage)
      messageEl.textContent = ev.message ?? ''
      etaEl.textContent = fmtEta(ev.eta_s)
    },
    done() {
      fillEl.classList.add('done')
      fillEl.classList.remove('error')
      fillEl.style.width = '100%'
      percentEl.textContent = '100%'
      etaEl.textContent = 'Done'
    },
    error(message) {
      fillEl.classList.add('error')
      fillEl.classList.remove('done')
      stageEl.textContent = 'Error'
      messageEl.textContent = message
      etaEl.textContent = ''
    },
    reset() {
      fillEl.classList.remove('done', 'error')
      fillEl.style.width = '0%'
      percentEl.textContent = '0%'
      stageEl.textContent = 'Idle'
      messageEl.textContent = ''
      etaEl.textContent = ''
    },
  }
}
