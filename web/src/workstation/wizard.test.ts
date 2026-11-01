import { describe, it, expect } from 'vitest'
import {
  applyTierCompat,
  buildWizardHTML,
  parseWizardSelection,
  presetsFromEntries,
  showWizard,
  type WizardPreset,
} from './wizard'
import type { ConfigEntry } from '../types'

function preset(name: string, tiers: WizardPreset['tier_compat']): WizardPreset {
  return { name, tier_compat: tiers }
}

describe('buildWizardHTML', () => {
  it('prioritizes the reference model and keeps physics profiles collapsed', () => {
    const root = document.createElement('div')
    root.innerHTML = buildWizardHTML(presetsFromEntries([
      { name: 'scaps_mirror_v2.yaml', namespace: 'shipped', tier_compat: ['legacy', 'fast', 'full'] },
    ]))
    const summary = root.querySelector<HTMLElement>('[data-wizard="model-summary"]')!
    const profiles = root.querySelector<HTMLDetailsElement>('.wizard-profile-options')!
    expect(profiles.open).toBe(false)
    expect(summary.textContent).toContain('no mobile ions')
    expect(summary.textContent).toContain('Calibrated SCAPS comparison')
    expect(summary.textContent).not.toContain('Two ion species')
    expect(root.querySelector('[data-wizard="profile-summary"]')?.textContent).toContain('Fast · reference')
    expect(summary.compareDocumentPosition(profiles) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
  })

  it('contains three tier cards', () => {
    const html = buildWizardHTML([
      preset('ionmonger_benchmark.yaml', ['legacy', 'fast']),
      preset('cigs_baseline.yaml', ['legacy', 'fast']),
    ])
    expect(html).toContain('data-tier="legacy"')
    expect(html).toContain('data-tier="fast"')
    expect(html).toContain('data-tier="full"')
  })

  it('lists the supplied preset filenames in the preset picker', () => {
    const html = buildWizardHTML([
      preset('foo.yaml', ['legacy', 'fast']),
      preset('bar.yaml', ['legacy', 'fast', 'full']),
    ])
    expect(html).toContain('foo.yaml')
    expect(html).toContain('bar.yaml')
  })

  it('disables full radio when the initial preset lacks full compat', () => {
    const el = document.createElement('div')
    el.innerHTML = buildWizardHTML([preset('legacy-only.yaml', ['legacy', 'fast'])])
    const fullRadio = el.querySelector<HTMLInputElement>(
      'input[name="wizard-tier"][value="full"]',
    )!
    expect(fullRadio.disabled).toBe(true)
  })

  it('pre-selects full when the initial preset supports it', () => {
    const el = document.createElement('div')
    el.innerHTML = buildWizardHTML([preset('nip_tmm.yaml', ['legacy', 'fast', 'full'])])
    const fullRadio = el.querySelector<HTMLInputElement>(
      'input[name="wizard-tier"][value="full"]',
    )!
    expect(fullRadio.checked).toBe(true)
    expect(fullRadio.disabled).toBe(false)
  })
})

describe('applyTierCompat', () => {
  it('disables incompatible radios and keeps the current tier if still valid', () => {
    const el = document.createElement('div')
    el.innerHTML = buildWizardHTML([preset('nip_tmm.yaml', ['legacy', 'fast', 'full'])])
    const fastRadio = el.querySelector<HTMLInputElement>(
      'input[name="wizard-tier"][value="fast"]',
    )!
    fastRadio.checked = true
    el.querySelector<HTMLInputElement>('input[name="wizard-tier"][value="full"]')!
      .checked = false

    const next = applyTierCompat(el, ['legacy', 'fast'])

    expect(next).toBe('fast')
    expect(fastRadio.checked).toBe(true)
    expect(
      el.querySelector<HTMLInputElement>('input[name="wizard-tier"][value="full"]')!
        .disabled,
    ).toBe(true)
  })

  it('auto-switches away from an incompatible tier', () => {
    const el = document.createElement('div')
    el.innerHTML = buildWizardHTML([preset('nip_tmm.yaml', ['legacy', 'fast', 'full'])])
    expect(
      el.querySelector<HTMLInputElement>('input[name="wizard-tier"][value="full"]')!
        .checked,
    ).toBe(true)

    const next = applyTierCompat(el, ['legacy', 'fast'])

    expect(next).toBe('fast')
    expect(
      el.querySelector<HTMLInputElement>('input[name="wizard-tier"][value="full"]')!
        .disabled,
    ).toBe(true)
    expect(
      el.querySelector<HTMLInputElement>('input[name="wizard-tier"][value="fast"]')!
        .checked,
    ).toBe(true)
  })

  it('adds a gate note when full is not available', () => {
    const el = document.createElement('div')
    el.innerHTML = buildWizardHTML([preset('legacy.yaml', ['legacy', 'fast'])])
    applyTierCompat(el, ['legacy', 'fast'])
    const note = el.querySelector<HTMLElement>('[data-wizard="tier-note"]')!
    expect(note.textContent).toBe('Full profile is unavailable for this preset.')
  })

  it('clears the gate note when full is available again', () => {
    const el = document.createElement('div')
    el.innerHTML = buildWizardHTML([preset('legacy.yaml', ['legacy', 'fast'])])
    applyTierCompat(el, ['legacy', 'fast'])
    applyTierCompat(el, ['legacy', 'fast', 'full'])
    const note = el.querySelector<HTMLElement>('[data-wizard="tier-note"]')!
    expect(note.textContent).toBe('')
  })
})

describe('presetsFromEntries', () => {
  it('defaults missing tier_compat to legacy+fast', () => {
    const entries: ConfigEntry[] = [
      { name: 'old.yaml', namespace: 'user' },
      {
        name: 'new.yaml',
        namespace: 'user',
        tier_compat: ['legacy', 'fast', 'full'],
      },
    ]
    const presets = presetsFromEntries(entries)
    expect(presets[0].tier_compat).toEqual(['legacy', 'fast'])
    expect(presets[1].tier_compat).toEqual(['legacy', 'fast', 'full'])
  })

  it('uses the focused catalog and preserves each reference mode', () => {
    const presets = presetsFromEntries([
      { name: 'calado2016_fig1f.yaml', namespace: 'shipped', tier_compat: ['legacy', 'fast'] },
      { name: 'calado2016_ion_sweep.yaml', namespace: 'shipped', tier_compat: ['legacy', 'fast', 'full'] },
      { name: 'scaps_mirror_v2.yaml', namespace: 'shipped', tier_compat: ['legacy', 'fast', 'full'] },
      { name: 'ionmonger_benchmark.yaml', namespace: 'shipped' },
    ])
    expect(presets.map(p => p.name)).toEqual(['scaps_mirror_v2.yaml', 'calado2016_ion_sweep.yaml'])
    expect(presets.map(p => p.preferred_tier)).toEqual(['fast', 'full'])
  })
})

describe('research preset selection', () => {
  it('updates model physics and flags departures from the reference profile', () => {
    const root = document.createElement('div')
    document.body.appendChild(root)
    void showWizard(root, presetsFromEntries([
      { name: 'scaps_mirror_v2.yaml', namespace: 'shipped', tier_compat: ['legacy', 'fast', 'full'] },
      { name: 'calado2016_ion_sweep.yaml', namespace: 'shipped', tier_compat: ['legacy', 'fast', 'full'] },
      { name: 'custom.yaml', namespace: 'user', tier_compat: ['legacy', 'fast', 'full'] },
    ]))
    const warning = root.querySelector<HTMLElement>('[data-wizard="profile-warning"]')!
    const summary = root.querySelector<HTMLElement>('[data-wizard="model-summary"]')!
    const select = root.querySelector<HTMLSelectElement>('select[name="wizard-preset"]')!
    expect(warning.hidden).toBe(true)
    root.querySelector<HTMLDetailsElement>('.wizard-profile-options')!.open = true
    root.querySelector<HTMLInputElement>('input[name="wizard-tier"][value="full"]')!.click()
    expect(warning.hidden).toBe(false)
    expect(warning.textContent).toContain('Model reference: Fast')

    select.value = 'calado2016_ion_sweep.yaml'
    select.dispatchEvent(new Event('change'))
    expect(warning.hidden).toBe(true)
    expect(summary.textContent).toContain('positive-ion drift-diffusion')
    expect(summary.textContent).toContain('Uniform photogeneration')
    expect(summary.textContent).not.toContain('TMM photogeneration')
    expect(summary.textContent).toContain('original-paper agreement remains open')
    expect(root.querySelector('[data-wizard="profile-summary"]')?.textContent).toBe('Full · reference')

    select.value = 'custom.yaml'
    select.dispatchEvent(new Event('change'))
    expect(summary.hidden).toBe(true)
    expect(summary.textContent).toBe('')
    expect(warning.hidden).toBe(true)
    expect(root.querySelector('[data-wizard="profile-summary"]')?.textContent).toBe('Full')
    root.querySelector<HTMLButtonElement>('[data-wizard="cancel"]')!.click()
    root.remove()
  })

  it('starts SCAPS in Fast and selects the verified Full-mode Calado configuration', async () => {
    const root = document.createElement('div')
    const result = showWizard(root, presetsFromEntries([
      { name: 'scaps_mirror_v2.yaml', namespace: 'shipped', tier_compat: ['legacy', 'fast', 'full'] },
      { name: 'calado2016_ion_sweep.yaml', namespace: 'shipped', tier_compat: ['legacy', 'fast', 'full'] },
    ]))
    expect(parseWizardSelection(root)?.tier).toBe('fast')
    const select = root.querySelector<HTMLSelectElement>('select[name="wizard-preset"]')!
    expect(select.options[0].textContent).toBe('SCAPS reproduction')
    expect(select.options[1].textContent).toBe('Calado 2016 - Ion hysteresis')
    select.value = 'calado2016_ion_sweep.yaml'
    select.dispatchEvent(new Event('change'))
    expect(parseWizardSelection(root)).toEqual({
      tier: 'full', preset: 'calado2016_ion_sweep.yaml', name: 'New device',
    })
    root.querySelector<HTMLButtonElement>('[data-wizard="create"]')!.click()
    expect(await result).toEqual({
      cancelled: false,
      selection: { tier: 'full', preset: 'calado2016_ion_sweep.yaml', name: 'New device' },
    })
  })
})

describe('parseWizardSelection', () => {
  it('extracts tier and preset from a submitted form', () => {
    const el = document.createElement('div')
    el.innerHTML = buildWizardHTML([
      preset('a.yaml', ['legacy', 'fast']),
      preset('b.yaml', ['legacy', 'fast']),
    ])
    el.querySelector<HTMLInputElement>('input[name="wizard-tier"][value="legacy"]')!
      .checked = true
    el.querySelector<HTMLSelectElement>('select[name="wizard-preset"]')!.value = 'b.yaml'
    const sel = parseWizardSelection(el)
    expect(sel).toEqual({ tier: 'legacy', preset: 'b.yaml', name: expect.any(String) })
  })
})
