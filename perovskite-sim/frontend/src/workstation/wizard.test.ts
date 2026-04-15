import { describe, it, expect } from 'vitest'
import {
  applyTierCompat,
  buildWizardHTML,
  parseWizardSelection,
  presetsFromEntries,
  type WizardPreset,
} from './wizard'
import type { ConfigEntry } from '../types'

function preset(name: string, tiers: WizardPreset['tier_compat']): WizardPreset {
  return { name, tier_compat: tiers }
}

describe('buildWizardHTML', () => {
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
    expect(note.textContent).toMatch(/FULL tier disabled/)
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
      { name: 'old.yaml', namespace: 'shipped' },
      {
        name: 'new.yaml',
        namespace: 'shipped',
        tier_compat: ['legacy', 'fast', 'full'],
      },
    ]
    const presets = presetsFromEntries(entries)
    expect(presets[0].tier_compat).toEqual(['legacy', 'fast'])
    expect(presets[1].tier_compat).toEqual(['legacy', 'fast', 'full'])
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
