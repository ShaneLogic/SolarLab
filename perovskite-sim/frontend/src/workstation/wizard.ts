import type { ConfigEntry, SimulationModeName } from '../types'
import { presetLabel, presetPreferredMode, researchPresetEntries } from '../preset-catalog'

export interface WizardSelection {
  tier: SimulationModeName
  preset: string
  name: string
}

export interface WizardPreset {
  name: string
  tier_compat: ReadonlyArray<SimulationModeName>
  preferred_tier?: SimulationModeName
}

const DEFAULT_TIER_COMPAT: ReadonlyArray<SimulationModeName> = ['legacy', 'fast']

export function presetsFromEntries(entries: ReadonlyArray<ConfigEntry>): WizardPreset[] {
  return researchPresetEntries(entries).map(e => ({
    name: e.name,
    tier_compat: (e.tier_compat ?? DEFAULT_TIER_COMPAT) as ReadonlyArray<SimulationModeName>,
    preferred_tier: presetPreferredMode(e.name),
  }))
}

const TIER_CARDS: Array<{
  tier: SimulationModeName
  title: string
  subtitle: string
  bullets: string[]
}> = [
  { tier: 'legacy', title: 'Legacy', subtitle: 'IonMonger-compatible',
    bullets: ['Beer–Lambert optics', 'Single ion species', 'Uniform τ', 'T = 300 K'] },
  { tier: 'fast', title: 'Fast', subtitle: 'Build-once physics upgrades',
    bullets: ['TMM optics', 'Thermionic emission', 'Dual-species ions', 'Trap profile · T-scaling'] },
  { tier: 'full', title: 'Full', subtitle: 'All configured upgrades',
    bullets: ['Fast-tier physics', 'Radiative reabsorption', 'Field-dependent mobility', 'Robin contacts'] },
]

function pickInitialTier(
  compat: ReadonlyArray<SimulationModeName>,
  preferred?: SimulationModeName,
): SimulationModeName {
  if (preferred && compat.includes(preferred)) return preferred
  if (compat.includes('full')) return 'full'
  if (compat.includes('fast')) return 'fast'
  return compat[0] ?? 'legacy'
}

export function buildWizardHTML(presets: ReadonlyArray<WizardPreset>): string {
  const initial = presets[0]
  const initialCompat = initial?.tier_compat ?? DEFAULT_TIER_COMPAT
  const initialTier = pickInitialTier(initialCompat, initial?.preferred_tier)

  const cards = TIER_CARDS.map(c => {
    const enabled = initialCompat.includes(c.tier)
    const checked = enabled && c.tier === initialTier ? ' checked' : ''
    const disabled = enabled ? '' : ' disabled'
    return `
    <label class="wizard-card" data-tier="${c.tier}"${enabled ? '' : ' data-disabled="true"'}>
      <input type="radio" name="wizard-tier" value="${c.tier}"${checked}${disabled} />
      <div class="wizard-card-title">${c.title}</div>
      <div class="wizard-card-subtitle">${c.subtitle}</div>
      <ul class="wizard-card-bullets">${c.bullets.map(b => `<li>${b}</li>`).join('')}</ul>
    </label>`
  }).join('')

  const options = presets
    .map(p => `<option value="${p.name}" data-tier-compat="${p.tier_compat.join(',')}" data-preferred-tier="${p.preferred_tier ?? ''}">${presetLabel(p.name)}</option>`)
    .join('')

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
        <p class="wizard-tier-note" data-wizard="tier-note"></p>
        <div class="wizard-actions">
          <button type="button" class="btn" data-wizard="cancel">Cancel</button>
          <button type="button" class="btn btn-primary" data-wizard="create">Create</button>
        </div>
      </div>
    </div>`
}

/**
 * Reconcile the tier radios against the currently-selected preset's tier_compat.
 * Disables incompatible tier cards and, if the current selection is no longer
 * valid, auto-switches to the best available tier (full > fast > legacy).
 * Returns the tier that ended up selected.
 */
export function applyTierCompat(
  root: HTMLElement,
  compat: ReadonlyArray<SimulationModeName>,
  preferred?: SimulationModeName,
): SimulationModeName {
  const radios = Array.from(
    root.querySelectorAll<HTMLInputElement>('input[name="wizard-tier"]'),
  )
  for (const r of radios) {
    const tier = r.value as SimulationModeName
    const enabled = compat.includes(tier)
    r.disabled = !enabled
    const card = r.closest<HTMLElement>('.wizard-card')
    if (card) {
      if (enabled) card.removeAttribute('data-disabled')
      else card.setAttribute('data-disabled', 'true')
    }
  }
  const currentEl = radios.find(r => r.checked)
  const currentTier = currentEl?.value as SimulationModeName | undefined
  const needsSwitch = !currentTier || !compat.includes(currentTier)
  const nextTier = preferred && compat.includes(preferred)
    ? preferred
    : needsSwitch ? pickInitialTier(compat) : currentTier
  for (const r of radios) {
    r.checked = r.value === nextTier && !r.disabled
  }
  const note = root.querySelector<HTMLElement>('[data-wizard="tier-note"]')
  if (note) {
    if (!compat.includes('full')) {
      note.textContent =
        'This preset advertises legacy/fast only — FULL tier disabled (no chi/Eg on all electrical layers).'
    } else {
      note.textContent = ''
    }
  }
  return nextTier
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
  presets: ReadonlyArray<WizardPreset>,
): Promise<WizardResult> {
  return new Promise((resolve) => {
    const host = document.createElement('div')
    host.innerHTML = buildWizardHTML(presets)
    root.appendChild(host)

    const modal = host.querySelector<HTMLElement>('.wizard-modal-backdrop')!
    const presetSelect = modal.querySelector<HTMLSelectElement>('select[name="wizard-preset"]')!

    function compatForSelected(): ReadonlyArray<SimulationModeName> {
      const opt = presetSelect.selectedOptions[0]
      const raw = opt?.getAttribute('data-tier-compat') ?? ''
      const parsed = raw.split(',').filter(Boolean) as SimulationModeName[]
      return parsed.length > 0 ? parsed : DEFAULT_TIER_COMPAT
    }

    const onPresetChange = (): void => {
      const preferred = presets.find(p => p.name === presetSelect.value)?.preferred_tier
      applyTierCompat(modal, compatForSelected(), preferred)
    }
    onPresetChange()
    presetSelect.addEventListener('change', onPresetChange)

    function close(result: WizardResult): void {
      presetSelect.removeEventListener('change', onPresetChange)
      host.remove()
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
