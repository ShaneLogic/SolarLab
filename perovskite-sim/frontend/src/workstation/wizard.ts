import type { SimulationModeName } from '../types'

export interface WizardSelection {
  tier: SimulationModeName
  preset: string
  name: string
}

const TIER_CARDS: Array<{
  tier: SimulationModeName
  title: string
  subtitle: string
  bullets: string[]
}> = [
  { tier: 'legacy', title: 'Legacy', subtitle: 'IonMonger-compatible',
    bullets: ['Beer-Lambert optics', 'Single ion species', 'Uniform tau', 'T = 300 K'] },
  { tier: 'fast', title: 'Fast', subtitle: 'Same physics, fast path (today)',
    bullets: ['Beer-Lambert', 'Single ion species', 'Uniform tau', 'T = 300 K'] },
  { tier: 'full', title: 'Full', subtitle: 'All Phase 1-4 upgrades',
    bullets: ['TMM optics', 'Band-offset TE', 'Dual-species ions', 'Trap profile and T-scaling'] },
]

export function buildWizardHTML(presets: ReadonlyArray<string>): string {
  const cards = TIER_CARDS.map(c => `
    <label class="wizard-card" data-tier="${c.tier}">
      <input type="radio" name="wizard-tier" value="${c.tier}"${c.tier === 'full' ? ' checked' : ''} />
      <div class="wizard-card-title">${c.title}</div>
      <div class="wizard-card-subtitle">${c.subtitle}</div>
      <ul class="wizard-card-bullets">${c.bullets.map(b => `<li>${b}</li>`).join('')}</ul>
    </label>`).join('')

  const options = presets.map(p => `<option value="${p}">${p}</option>`).join('')

  return `
    <div class="wizard-modal-backdrop">
      <div class="wizard-modal">
        <h2>New Device</h2>
        <p class="wizard-subtitle">Pick a physics tier and a preset to start from.</p>
        <div class="wizard-tier-row">${cards}</div>
        <div class="wizard-form-row">
          <label>
            <span>Device name</span>
            <input type="text" name="wizard-name" value="New device" />
          </label>
          <label>
            <span>Preset</span>
            <select name="wizard-preset">${options}</select>
          </label>
        </div>
        <div class="wizard-actions">
          <button type="button" class="btn" data-wizard="cancel">Cancel</button>
          <button type="button" class="btn btn-primary" data-wizard="create">Create</button>
        </div>
      </div>
    </div>`
}

export function parseWizardSelection(root: HTMLElement): WizardSelection | null {
  const tierEl = root.querySelector<HTMLInputElement>('input[name="wizard-tier"]:checked')
  const presetEl = root.querySelector<HTMLSelectElement>('select[name="wizard-preset"]')
  const nameEl = root.querySelector<HTMLInputElement>('input[name="wizard-name"]')
  if (!tierEl || !presetEl || !nameEl) return null
  const tier = tierEl.value as SimulationModeName
  return { tier, preset: presetEl.value, name: nameEl.value.trim() || 'New device' }
}

export interface WizardResult {
  cancelled: boolean
  selection: WizardSelection | null
}

/**
 * Show the wizard as a modal, resolve when the user clicks Create or Cancel.
 * DOM side-effect only - the caller owns creating the Device from the selection.
 */
export function showWizard(
  root: HTMLElement,
  presets: ReadonlyArray<string>,
): Promise<WizardResult> {
  return new Promise((resolve) => {
    const host = document.createElement('div')
    host.innerHTML = buildWizardHTML(presets)
    root.appendChild(host)

    const modal = host.querySelector<HTMLElement>('.wizard-modal-backdrop')!

    function close(result: WizardResult): void {
      modal.remove()
      resolve(result)
    }

    modal.querySelector<HTMLButtonElement>('[data-wizard="cancel"]')!
      .addEventListener('click', () => close({ cancelled: true, selection: null }))

    modal.querySelector<HTMLButtonElement>('[data-wizard="create"]')!
      .addEventListener('click', () => {
        const sel = parseWizardSelection(modal)
        if (!sel) return
        close({ cancelled: false, selection: sel })
      })
  })
}
