import { describe, it, expect } from 'vitest'
import { buildWizardHTML, parseWizardSelection } from './wizard'

describe('buildWizardHTML', () => {
  it('contains three tier cards', () => {
    const html = buildWizardHTML(['ionmonger_benchmark.yaml', 'cigs_baseline.yaml'])
    expect(html).toContain('data-tier="legacy"')
    expect(html).toContain('data-tier="fast"')
    expect(html).toContain('data-tier="full"')
  })

  it('lists the supplied preset filenames in the preset picker', () => {
    const html = buildWizardHTML(['foo.yaml', 'bar.yaml'])
    expect(html).toContain('foo.yaml')
    expect(html).toContain('bar.yaml')
  })
})

describe('parseWizardSelection', () => {
  it('extracts tier and preset from a submitted form', () => {
    const el = document.createElement('div')
    el.innerHTML = buildWizardHTML(['a.yaml', 'b.yaml'])
    el.querySelector<HTMLInputElement>('input[name="wizard-tier"][value="legacy"]')!.checked = true
    el.querySelector<HTMLSelectElement>('select[name="wizard-preset"]')!.value = 'b.yaml'
    const sel = parseWizardSelection(el)
    expect(sel).toEqual({ tier: 'legacy', preset: 'b.yaml', name: expect.any(String) })
  })
})
