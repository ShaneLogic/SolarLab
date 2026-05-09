/**
 * Vitest cases for the progress-bar message renderer.
 *
 * Backend progress messages use plain underscore notation
 * (``V_oc``, ``J_sc``, ``E_A``…). The renderer converts known physics
 * tokens into real ``<sub>`` elements without using innerHTML so any
 * user-controllable content cannot inject markup.
 */
import { describe, it, expect } from 'vitest'
import { tokeniseProgressMessage, renderProgressMessage } from './progress'

describe('tokeniseProgressMessage', () => {
  it('tokenises a Suns-V_oc progress line into text + subscript parts', () => {
    const parts = tokeniseProgressMessage('X=0.1, V_oc=1.0326 V, J_sc=-350.59 A/m²')
    expect(parts).toEqual([
      { kind: 'text', value: 'X=0.1, ' },
      { kind: 'subscript', value: 'oc', base: 'V' },
      { kind: 'text', value: '=1.0326 V, ' },
      { kind: 'subscript', value: 'sc', base: 'J' },
      { kind: 'text', value: '=-350.59 A/m²' },
    ])
  })

  it('keeps mid-identifier underscores intact (V_oc_arr is not split)', () => {
    const parts = tokeniseProgressMessage('Reading V_oc_arr from result')
    // The trailing ``_arr`` blocks the right boundary, so V_oc stays
    // in the text run as one literal token.
    expect(parts.length).toBe(1)
    expect(parts[0].kind).toBe('text')
    expect(parts[0].value).toBe('Reading V_oc_arr from result')
  })

  it('handles longest-first matching: EQE_EL beats E_E* alternatives', () => {
    const parts = tokeniseProgressMessage('EQE_EL = 1.0e-3')
    expect(parts).toEqual([
      { kind: 'subscript', value: 'EL', base: 'EQE' },
      { kind: 'text', value: ' = 1.0e-3' },
    ])
  })

  it('returns a single text part when no token matches', () => {
    const parts = tokeniseProgressMessage('Done · pFF=84.3%')
    expect(parts).toEqual([{ kind: 'text', value: 'Done · pFF=84.3%' }])
  })

  it('handles V_oc(T) progress messages', () => {
    const parts = tokeniseProgressMessage('T=350.0 K, V_oc=0.7700 V')
    expect(parts).toEqual([
      { kind: 'text', value: 'T=350.0 K, ' },
      { kind: 'subscript', value: 'oc', base: 'V' },
      { kind: 'text', value: '=0.7700 V' },
    ])
  })
})

describe('renderProgressMessage', () => {
  it('builds text nodes and <sub> elements (no innerHTML)', () => {
    const el = document.createElement('span')
    renderProgressMessage(el, 'V_oc = 1.10 V')
    // First child = text "V", second child = <sub>oc</sub>, third
    // child = text " = 1.10 V".
    expect(el.childNodes.length).toBe(3)
    expect(el.childNodes[0].nodeType).toBe(Node.TEXT_NODE)
    expect(el.childNodes[0].textContent).toBe('V')
    expect((el.childNodes[1] as Element).tagName).toBe('SUB')
    expect(el.childNodes[1].textContent).toBe('oc')
    expect(el.childNodes[2].textContent).toBe(' = 1.10 V')
  })

  it('clears prior children on every call (no stale subscripts)', () => {
    const el = document.createElement('span')
    renderProgressMessage(el, 'V_oc = 1.0 V')
    renderProgressMessage(el, 'No tokens here')
    expect(el.childNodes.length).toBe(1)
    expect(el.childNodes[0].nodeType).toBe(Node.TEXT_NODE)
    expect(el.textContent).toBe('No tokens here')
  })

  it('treats embedded HTML as literal text (no script injection)', () => {
    const el = document.createElement('span')
    renderProgressMessage(el, '<script>alert(1)</script> V_oc=1.10')
    // No <script> child element should be created.
    expect(el.querySelector('script')).toBeNull()
    // Raw markup ends up in a text node.
    expect(el.textContent).toContain('<script>alert(1)</script>')
    // Subscript still gets rendered.
    const sub = el.querySelector('sub')
    expect(sub?.textContent).toBe('oc')
  })
})
