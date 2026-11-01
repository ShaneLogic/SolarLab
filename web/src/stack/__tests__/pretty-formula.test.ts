import { describe, it, expect } from 'vitest'
import { prettyFormula } from '../stack-layer-card'

describe('prettyFormula', () => {
  it('subscripts digits that follow a letter', () => {
    expect(prettyFormula('TiO2')).toBe('TiO<sub>2</sub>')
    expect(prettyFormula('SnO2')).toBe('SnO<sub>2</sub>')
    expect(prettyFormula('MAPbI3')).toBe('MAPbI<sub>3</sub>')
    expect(prettyFormula('C60')).toBe('C<sub>60</sub>')
  })

  it('leaves non-formula names unchanged', () => {
    expect(prettyFormula('spiro_HTL')).toBe('spiro_HTL')
    expect(prettyFormula('Au_back_contact')).toBe('Au_back_contact')
    expect(prettyFormula('glass')).toBe('glass')
  })

  it('handles mixed formula + suffix', () => {
    // "MAPbI3_tmm" -> "MAPbI<sub>3</sub>_tmm"
    expect(prettyFormula('MAPbI3_tmm')).toBe('MAPbI<sub>3</sub>_tmm')
  })

  it('escapes HTML before wrapping digits', () => {
    // Untrusted-looking input should not inject raw HTML.
    expect(prettyFormula('<script>X2')).toBe('&lt;script&gt;X<sub>2</sub>')
  })

  it('does not subscript leading digits', () => {
    // "2H-MoS2" style: leading "2" has no preceding letter, so left alone.
    expect(prettyFormula('2H-MoS2')).toBe('2H-MoS<sub>2</sub>')
  })
})
