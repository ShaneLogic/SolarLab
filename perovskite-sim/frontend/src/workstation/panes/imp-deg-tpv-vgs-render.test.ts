/**
 * Vitest cases for ``renderImpedance`` / ``renderDegradation`` /
 * ``renderTPV`` / ``renderVocGrainSweep`` workstation panes.
 *
 * Same Plotly mock pattern as the other publication-mode panes.
 * Engineering mode is the default and bit-identical to the
 * pre-publication renderer; the Publication branch swaps font /
 * colors / margins / modebar / axes / annotation / hollow markers
 * without mutating the raw arrays carried in each result type.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'

vi.mock('plotly.js-basic-dist-min', () => ({
  default: {
    newPlot: vi.fn(),
    purge: vi.fn(),
  },
  newPlot: vi.fn(),
  purge: vi.fn(),
}))

import {
  renderImpedance,
  renderDegradation,
  renderTPV,
  renderVocGrainSweep,
} from './main-plot-pane'
import type {
  ISResult,
  DegResult,
  TPVResult,
  VocGrainSweepResult,
} from '../../types'
import { PUBLICATION_FONT_FAMILY } from '../../plot-theme'
import Plotly from 'plotly.js-basic-dist-min'

const newPlotMock = vi.mocked(Plotly.newPlot)

function _layout(): Record<string, any> | undefined {
  const calls = newPlotMock.mock.calls
  if (calls.length === 0) return undefined
  return calls[calls.length - 1][2] as Record<string, any>
}
function _traces(): Array<Record<string, any>> | undefined {
  const calls = newPlotMock.mock.calls
  if (calls.length === 0) return undefined
  return calls[calls.length - 1][1] as Array<Record<string, any>>
}
function _config(): Record<string, any> | undefined {
  const calls = newPlotMock.mock.calls
  if (calls.length === 0) return undefined
  return calls[calls.length - 1][3] as Record<string, any>
}
function _toggle(el: HTMLElement, sel: string, mode: 'engineering' | 'publication'): void {
  const s = el.querySelector<HTMLSelectElement>(`[data-test="${sel}"]`)!
  s.value = mode
  s.dispatchEvent(new Event('change'))
}

// ── Impedance (Nyquist) ──────────────────────────────────────────────────

function makeIS(): ISResult {
  return {
    frequencies: [1e2, 1e3, 1e4, 1e5],
    Z_real:      [120,  90,  60,  20],   // Ω·m²
    Z_imag:      [-15, -40, -55, -10],   // Ω·m² (raw, will be sign-flipped)
  }
}

describe('renderImpedance', () => {
  let el: HTMLDivElement
  beforeEach(() => {
    newPlotMock.mockClear()
    el = document.createElement('div')
    document.body.appendChild(el)
  })

  it('renders the Style: select toolbar', () => {
    renderImpedance(el, makeIS())
    expect(el.querySelector('[data-test="impedance-toolbar"]')).not.toBeNull()
    expect(el.querySelector('[data-test="impedance-style-mode"]')).not.toBeNull()
  })

  it('default style is engineering (Arial layout, modebar visible)', () => {
    renderImpedance(el, makeIS())
    expect(_layout()!.font.family).toBe('Arial, sans-serif')
    expect(_config()!.displayModeBar).toBeUndefined()
    expect(_traces()![0].line.color).toBe('#2563eb')
  })

  it('engineering: square-aspect Nyquist via scaleanchor x', () => {
    renderImpedance(el, makeIS())
    expect(_layout()!.yaxis.scaleanchor).toBe('x')
  })

  it('publication: hollow circle muted blue, modebar hidden, scaleanchor preserved', () => {
    renderImpedance(el, makeIS())
    _toggle(el, 'impedance-style-mode', 'publication')
    const layout = _layout()!
    expect(layout.font.family).toBe(PUBLICATION_FONT_FAMILY)
    expect(layout.paper_bgcolor).toBe('#ffffff')
    expect(layout.yaxis.scaleanchor).toBe('x')
    expect(_config()!.displayModeBar).toBe(false)
    const data = _traces()![0]
    expect(data.marker.symbol).toBe('circle-open')
    expect(data.line.color).toBe('#2B6FA3')
  })

  it('raw Z_real / Z_imag arrays unchanged across modes', () => {
    const r = makeIS()
    const re_pre = [...r.Z_real]
    const im_pre = [...r.Z_imag]
    renderImpedance(el, r)
    _toggle(el, 'impedance-style-mode', 'publication')
    expect(r.Z_real).toEqual(re_pre)
    expect(r.Z_imag).toEqual(im_pre)
    // x is the raw Z_real reference (no .slice() / .map()).
    expect(_traces()![0].x).toBe(r.Z_real)
  })
})

// ── Degradation (PCE / PCE_0 vs t) ───────────────────────────────────────

function makeDeg(): DegResult {
  return {
    times:   [0, 100, 1000, 10000],
    PCE:     [0.20, 0.195, 0.18, 0.16],
    V_oc:    [1.10, 1.09, 1.07, 1.04],
    J_sc:    [220, 218, 215, 210],
  }
}

describe('renderDegradation', () => {
  let el: HTMLDivElement
  beforeEach(() => {
    newPlotMock.mockClear()
    el = document.createElement('div')
    document.body.appendChild(el)
  })

  it('renders the Style: select toolbar', () => {
    renderDegradation(el, makeDeg())
    expect(el.querySelector('[data-test="degradation-toolbar"]')).not.toBeNull()
    expect(el.querySelector('[data-test="degradation-style-mode"]')).not.toBeNull()
  })

  it('default style is engineering (Arial layout, blue trace)', () => {
    renderDegradation(el, makeDeg())
    expect(_layout()!.font.family).toBe('Arial, sans-serif')
    expect(_traces()![0].line.color).toBe('#2563eb')
  })

  it('publication: hollow circle muted blue, modebar hidden', () => {
    renderDegradation(el, makeDeg())
    _toggle(el, 'degradation-style-mode', 'publication')
    expect(_layout()!.font.family).toBe(PUBLICATION_FONT_FAMILY)
    expect(_config()!.displayModeBar).toBe(false)
    const data = _traces()![0]
    expect(data.marker.symbol).toBe('circle-open')
    expect(data.line.color).toBe('#2B6FA3')
    expect(data.name).toContain('PCE')
  })

  it('raw PCE / times arrays unchanged across modes', () => {
    const r = makeDeg()
    const pce_pre = [...r.PCE]
    const t_pre = [...r.times]
    renderDegradation(el, r)
    _toggle(el, 'degradation-style-mode', 'publication')
    expect(r.PCE).toEqual(pce_pre)
    expect(r.times).toEqual(t_pre)
    expect(_traces()![0].x).toBe(r.times)
  })
})

// ── TPV (ΔV decay) ───────────────────────────────────────────────────────

function makeTPV(): TPVResult {
  return {
    t:        [0, 1e-6, 2e-6, 5e-6, 1e-5],
    V:        [1.105, 1.103, 1.1015, 1.1005, 1.1001],
    J:        [0, -0.5, -0.4, -0.2, -0.05],
    V_oc:     1.100,
    tau:      2.0e-6,
    delta_V0: 5.0e-3,
  }
}

describe('renderTPV', () => {
  let el: HTMLDivElement
  beforeEach(() => {
    newPlotMock.mockClear()
    el = document.createElement('div')
    document.body.appendChild(el)
  })

  it('renders the Style: select toolbar', () => {
    renderTPV(el, makeTPV())
    expect(el.querySelector('[data-test="tpv-toolbar"]')).not.toBeNull()
    expect(el.querySelector('[data-test="tpv-style-mode"]')).not.toBeNull()
  })

  it('engineering annotation upper-RIGHT with V_oc / τ / ΔV_0', () => {
    renderTPV(el, makeTPV())
    const ann = _layout()!.annotations as Array<Record<string, any>>
    expect(ann[0].xanchor).toBe('right')
    expect(ann[0].yanchor).toBe('top')
    const t = ann[0].text
    expect(t).toContain('V<sub>oc</sub>')
    expect(t).toContain('1.100 V')
    expect(t).toContain('2.0 µs')
    expect(t).toContain('5.00 mV')
  })

  it('publication: solid muted-blue line (lines mode, no markers), font swap', () => {
    renderTPV(el, makeTPV())
    _toggle(el, 'tpv-style-mode', 'publication')
    expect(_layout()!.font.family).toBe(PUBLICATION_FONT_FAMILY)
    expect(_config()!.displayModeBar).toBe(false)
    const data = _traces()![0]
    expect(data.mode).toBe('lines')
    expect(data.line.color).toBe('#2B6FA3')
    expect(data.line.width).toBe(1.75)
    // TPV trace is a smooth decay → no markers in publication.
    expect(data.marker).toBeUndefined()
  })

  it('publication annotation: V_oc / τ / ΔV_0 stacked with <br> at upper-RIGHT', () => {
    renderTPV(el, makeTPV())
    _toggle(el, 'tpv-style-mode', 'publication')
    const ann = _layout()!.annotations as Array<Record<string, any>>
    expect(ann[0].xanchor).toBe('right')
    expect(ann[0].yanchor).toBe('top')
    expect(ann[0].text).toContain('<br>')
    expect(ann[0].font.family).toBe(PUBLICATION_FONT_FAMILY)
  })

  it('raw t / V arrays unchanged across modes', () => {
    const r = makeTPV()
    const t_pre = [...r.t]
    const V_pre = [...r.V]
    renderTPV(el, r)
    _toggle(el, 'tpv-style-mode', 'publication')
    expect(r.t).toEqual(t_pre)
    expect(r.V).toEqual(V_pre)
  })
})

// ── V_oc(L_g) Grain Sweep ────────────────────────────────────────────────

function makeVGS(): VocGrainSweepResult {
  return {
    grain_sizes_nm: [10, 30, 100, 300, 1000],
    V_oc_V:         [0.85, 0.92, 0.99, 1.04, 1.07],
    J_sc_Am2:       [340, 360, 380, 395, 410],
    FF:             [0.62, 0.69, 0.74, 0.78, 0.81],
  }
}

describe('renderVocGrainSweep', () => {
  let el: HTMLDivElement
  beforeEach(() => {
    newPlotMock.mockClear()
    el = document.createElement('div')
    document.body.appendChild(el)
  })

  it('renders the Style: select toolbar', () => {
    renderVocGrainSweep(el, makeVGS())
    expect(el.querySelector('[data-test="voc-grain-sweep-toolbar"]')).not.toBeNull()
    expect(el.querySelector('[data-test="voc-grain-sweep-style-mode"]')).not.toBeNull()
  })

  it('engineering: log x-axis with decade-only ticks (dtick=1)', () => {
    renderVocGrainSweep(el, makeVGS())
    const layout = _layout()!
    expect(layout.xaxis.type).toBe('log')
    expect(layout.xaxis.dtick).toBe(1)
  })

  it('publication: log x-axis with dtick=1 + hollow circle muted blue', () => {
    renderVocGrainSweep(el, makeVGS())
    _toggle(el, 'voc-grain-sweep-style-mode', 'publication')
    const layout = _layout()!
    expect(layout.font.family).toBe(PUBLICATION_FONT_FAMILY)
    expect(layout.xaxis.type).toBe('log')
    expect(layout.xaxis.dtick).toBe(1)
    const data = _traces()![0]
    expect(data.marker.symbol).toBe('circle-open')
    expect(data.line.color).toBe('#2B6FA3')
    expect(data.name).toContain('V<sub>oc</sub>(L<sub>g</sub>)')
  })

  it('publication annotation lives at lower-RIGHT (rising curve, empty quadrant)', () => {
    renderVocGrainSweep(el, makeVGS())
    _toggle(el, 'voc-grain-sweep-style-mode', 'publication')
    const ann = _layout()!.annotations as Array<Record<string, any>>
    expect(ann).toHaveLength(1)
    expect(ann[0].xanchor).toBe('right')
    expect(ann[0].yanchor).toBe('bottom')
    // Per-grain table format with HTML subscripts.
    expect(ann[0].text).toContain('L<sub>g</sub>')
    expect(ann[0].text).toContain('<br>')
    expect(ann[0].font.family).toBe(PUBLICATION_FONT_FAMILY)
  })

  it('raw arrays unchanged across modes', () => {
    const r = makeVGS()
    const g_pre = [...r.grain_sizes_nm]
    const v_pre = [...r.V_oc_V]
    renderVocGrainSweep(el, r)
    _toggle(el, 'voc-grain-sweep-style-mode', 'publication')
    expect(r.grain_sizes_nm).toEqual(g_pre)
    expect(r.V_oc_V).toEqual(v_pre)
    expect(_traces()![0].x).toBe(r.grain_sizes_nm)
  })

  it('toggle round-trip restores defaults', () => {
    renderVocGrainSweep(el, makeVGS())
    _toggle(el, 'voc-grain-sweep-style-mode', 'publication')
    expect(_layout()!.font.family).toBe(PUBLICATION_FONT_FAMILY)
    _toggle(el, 'voc-grain-sweep-style-mode', 'engineering')
    expect(_layout()!.font.family).toBe('Arial, sans-serif')
  })
})
