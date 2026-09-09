import type { ConfigEntry, SimulationModeName } from '../types'
import { presetDescription, presetLabel, presetPreferredMode, researchPresetEntries } from '../preset-catalog'

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
  { tier: 'legacy', title: 'Legacy', subtitle: 'Isothermal baseline',
    bullets: ['Single-ion transport', 'Material parameters at 300 K', 'Scalar or prescribed photogeneration'] },
  { tier: 'fast', title: 'Fast', subtitle: 'Optoelectronic extensions',
    bullets: ['TMM and thermionic transport', 'Two ion species; trap profiles', 'Temperature scaling; photon recycling'] },
  { tier: 'full', title: 'Full', subtitle: 'Fast + constitutive extensions',
    bullets: ['Radiative reabsorption source', 'Field-dependent carrier mobility', 'Finite-rate carrier exchange (Robin)'] },
]

function modelDescriptionHTML(name: string): string {
  const model = presetDescription(name)
  if (!model) return ''
  return `
    <h3>${model.formulation}</h3>
    <ul>${model.mechanisms.map(mechanism => `<li>${mechanism}</li>`).join('')}</ul>
    <p class="wizard-model-evidence">${model.evidence}</p>`
}

function updateProfileSummary(root: HTMLElement, preferred?: SimulationModeName): void {
  const selected = root.querySelector<HTMLInputElement>('input[name="wizard-tier"]:checked')
    ?.value as SimulationModeName | undefined
  const title = TIER_CARDS.find(card => card.tier === selected)?.title ?? ''
  const summary = root.querySelector<HTMLElement>('[data-wizard="profile-summary"]')
  if (summary) summary.textContent = `${title}${preferred && selected === preferred ? ' · reference' : ''}`
  const warning = root.querySelector<HTMLElement>('[data-wizard="profile-warning"]')
  if (warning) {
    warning.hidden = !preferred || preferred === selected
    const referenceTitle = TIER_CARDS.find(card => card.tier === preferred)?.title
    warning.textContent = warning.hidden ? '' : `Non-reference profile. Model reference: ${referenceTitle}.`
  }
}

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
      <div class="wizard-card-heading">
        <input type="radio" name="wizard-tier" value="${c.tier}"${checked}${disabled} />
        <span class="wizard-card-title">${c.title}</span>
      </div>
      <div class="wizard-card-subtitle">${c.subtitle}</div>
      <ul class="wizard-card-bullets">${c.bullets.map(b => `<li>${b}</li>`).join('')}</ul>
    </label>`
  }).join('')

  const options = presets
    .map(p => `<option value="${p.name}" data-tier-compat="${p.tier_compat.join(',')}" data-preferred-tier="${p.preferred_tier ?? ''}">${presetLabel(p.name)}</option>`)
    .join('')
  const modelDescription = modelDescriptionHTML(initial?.name ?? '')
  const profileTitle = TIER_CARDS.find(card => card.tier === initialTier)!.title

  return `
    <div class="wizard-modal-backdrop">
      <div class="wizard-modal" role="dialog" aria-modal="true" aria-labelledby="wizard-title">
        <h2 id="wizard-title">New Device</h2>
        <p class="wizard-subtitle">1D drift–diffusion–Poisson</p>
        <div class="wizard-form-row">
          <label>
            <span>Reference model</span>
            <select name="wizard-preset">${options}</select>
          </label>
          <label>
            <span>Device name</span>
            <input type="text" name="wizard-name" value="New device" />
          </label>
        </div>
        <section class="wizard-model-summary" data-wizard="model-summary" aria-label="Reference model physics" aria-live="polite"${modelDescription ? '' : ' hidden'}>${modelDescription}</section>
        <details class="wizard-profile-options">
          <summary><span>Physics profile</span> <span class="wizard-profile-summary" data-wizard="profile-summary">${profileTitle}${initial?.preferred_tier === initialTier ? ' · reference' : ''}</span></summary>
          <p class="wizard-profile-scope">Available extensions; activation depends on material and boundary parameters.</p>
          <div class="wizard-tier-row" role="radiogroup" aria-label="Physics profile">${cards}</div>
          <p class="wizard-tier-note" data-wizard="tier-note"></p>
        </details>
        <p class="wizard-profile-warning" data-wizard="profile-warning" role="status" hidden></p>
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
        'Full profile is unavailable for this preset.'
    } else {
      note.textContent = ''
    }
  }
  updateProfileSummary(root, preferred)
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
      const description = modal.querySelector<HTMLElement>('[data-wizard="model-summary"]')!
      description.innerHTML = modelDescriptionHTML(presetSelect.value)
      description.hidden = description.innerHTML.trim() === ''
    }
    const onProfileChange = (event: Event): void => {
      if (!(event.target as HTMLElement).matches('input[name="wizard-tier"]')) return
      updateProfileSummary(modal, presets.find(p => p.name === presetSelect.value)?.preferred_tier)
    }
    onPresetChange()
    presetSelect.addEventListener('change', onPresetChange)
    modal.addEventListener('change', onProfileChange)

    function close(result: WizardResult): void {
      presetSelect.removeEventListener('change', onPresetChange)
      modal.removeEventListener('change', onProfileChange)
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
