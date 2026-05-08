/**
 * Vitest cases for ``renderSunsVoc`` (Suns-V_oc workstation pane).
 *
 * Same Plotly mock pattern as ``jv-2d-render.test.ts`` /
 * ``jv-1d-render.test.ts`` — jsdom has no canvas, so layout / trace /
 * config arguments are read from ``newPlot.mock.calls``. Engineering
 * mode is the default and bit-identical to the pre-publication
 * renderer; the Publication branch swaps font / colors / margins /
 * modebar / axes / annotation / hollow markers without mutating the
 * raw suns / V_oc / J_pseudo_V / J_pseudo_J arrays.
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

import { renderSunsVoc } from './main-plot-pane'
import type { SunsVocResult } from '../../types'
import { PUBLICATION_FONT_FAMILY } from '../../plot-theme'
import Plotly from 'plotly.js-basic-dist-min'

const newPlotMock = vi.mocked(Plotly.newPlot)

function makeResult(overrides: Partial<SunsVocResult> = {}): SunsVocResult {
  return {
    suns: [0.01, 0.1, 1.0, 10.0],
    V_oc: [0.65, 0.78, 0.91, 1.04],          // V
    J_sc: [4.0, 40.0, 400.0, 4000.0],        // A/m²
    J_pseudo_V: [0.0, 0.5, 0.85, 0.91],
    J_pseudo_J: [+400.0, +200.0, +50.0, 0.0],
    pseudo_FF: 0.835,
    ...overrides,
  }
}

function _lastNewPlotLayout(): Record<string, any> | undefined {
  const calls = newPlotMock.mock.calls
  if (calls.length === 0) return undefined
  return calls[calls.length - 1][2] as Record<string, any>
}

function _lastNewPlotTraces(): Array<Record<string, any>> | undefined {
  const calls = newPlotMock.mock.calls
  if (calls.length === 0) return undefined
  return calls[calls.length - 1][1] as Array<Record<string, any>>
}

function _lastNewPlotConfig(): Record<string, any> | undefined {
  const calls = newPlotMock.mock.calls
  if (calls.length === 0) return undefined
  return calls[calls.length - 1][3] as Record<string, any>
}

function _toggleStyle(el: HTMLElement, mode: 'engineering' | 'publication'): void {
  const sel = el.querySelector<HTMLSelectElement>('[data-test="suns-voc-style-mode"]')!
  sel.value = mode
  sel.dispatchEvent(new Event('change'))
}

describe('renderSunsVoc — toolbar + style mode', () => {
  let el: HTMLDivElement
  beforeEach(() => {
    newPlotMock.mockClear()
    el = document.createElement('div')
    document.body.appendChild(el)
  })

  it('renders the Style: select unconditionally when plot data exists', () => {
    renderSunsVoc(el, makeResult())
    expect(el.querySelector('[data-test="suns-voc-toolbar"]')).not.toBeNull()
    expect(el.querySelector('[data-test="suns-voc-style-mode"]')).not.toBeNull()
  })

  it('default style is engineering (Arial layout, modebar visible)', () => {
    renderSunsVoc(el, makeResult())
    const layout = _lastNewPlotLayout()!
    expect(layout.font.family).toBe('Arial, sans-serif')
    const config = _lastNewPlotConfig()!
    expect(config.displayModeBar).toBeUndefined()
  })

  it('engineering trace colors / symbols match the pre-publication renderer', () => {
    renderSunsVoc(el, makeResult())
    const traces = _lastNewPlotTraces()!
    expect(traces).toHaveLength(2)
    // V_oc(suns): filled blue circle.
    expect(traces[0].name).toBe('V<sub>oc</sub>(suns)')
    expect(traces[0].line.color).toBe('#2563eb')
    expect(traces[0].marker.color).toBe('#2563eb')
    expect(traces[0].marker.symbol).toBeUndefined()
    // pseudo J-V: filled orange square.
    expect(traces[1].name).toBe('pseudo J–V')
    expect(traces[1].line.color).toBe('#ea580c')
    expect(traces[1].marker.symbol).toBe('square')
  })

  it('engineering preserves the dual-subplot grid layout + axis types', () => {
    renderSunsVoc(el, makeResult())
    const layout = _lastNewPlotLayout()!
    expect(layout.grid).toEqual({ rows: 1, columns: 2, pattern: 'independent' })
    expect(layout.xaxis.type).toBe('log')         // suns axis stays log
    expect(layout.xaxis2.type).toBeUndefined()    // V axis stays linear
    expect(layout.yaxis.title.text).toContain('V')
    expect(layout.yaxis2.title.text).toContain('mA')
  })
})

describe('renderSunsVoc — publication style mode', () => {
  let el: HTMLDivElement
  beforeEach(() => {
    newPlotMock.mockClear()
    el = document.createElement('div')
    document.body.appendChild(el)
  })

  it('toggling to publication applies Nature-style layout', () => {
    renderSunsVoc(el, makeResult())
    _toggleStyle(el, 'publication')
    const layout = _lastNewPlotLayout()!
    expect(layout.font.family).toBe(PUBLICATION_FONT_FAMILY)
    expect(layout.paper_bgcolor).toBe('#ffffff')
    expect(layout.plot_bgcolor).toBe('#ffffff')
    expect(layout.margin).toEqual({ t: 18, r: 18, b: 48, l: 58 })
    // No grid lines in publication mode.
    expect(layout.xaxis.showgrid).toBe(false)
    expect(layout.yaxis.showgrid).toBe(false)
    expect(layout.xaxis2.showgrid).toBe(false)
    expect(layout.yaxis2.showgrid).toBe(false)
  })

  it('publication mode hides the Plotly modebar; engineering does not', () => {
    renderSunsVoc(el, makeResult())
    expect(_lastNewPlotConfig()!.displayModeBar).toBeUndefined()
    _toggleStyle(el, 'publication')
    expect(_lastNewPlotConfig()!.displayModeBar).toBe(false)
  })

  it('publication V_oc(suns) trace: hollow circle, muted blue, lines+markers', () => {
    renderSunsVoc(el, makeResult())
    _toggleStyle(el, 'publication')
    const fwd = _lastNewPlotTraces()![0]
    expect(fwd.name).toBe('V<sub>oc</sub>(suns)')
    expect(fwd.mode).toBe('lines+markers')
    expect(fwd.marker.symbol).toBe('circle-open')
    expect(fwd.marker.color).toBe('rgba(0,0,0,0)')
    expect(fwd.marker.line.color).toBe('#2B6FA3')
    expect(fwd.line.color).toBe('#2B6FA3')
    expect(fwd.line.width).toBe(1.75)
  })

  it('publication pseudo J-V trace: hollow square, muted red', () => {
    renderSunsVoc(el, makeResult())
    _toggleStyle(el, 'publication')
    const rev = _lastNewPlotTraces()![1]
    expect(rev.name).toBe('pseudo J–V')
    expect(rev.mode).toBe('lines+markers')
    expect(rev.marker.symbol).toBe('square-open')
    expect(rev.marker.color).toBe('rgba(0,0,0,0)')
    expect(rev.marker.line.color).toBe('#C44536')
    expect(rev.line.color).toBe('#C44536')
  })

  it('publication mode preserves the suns log axis + dual-subplot grid', () => {
    renderSunsVoc(el, makeResult())
    _toggleStyle(el, 'publication')
    const layout = _lastNewPlotLayout()!
    expect(layout.grid).toEqual({ rows: 1, columns: 2, pattern: 'independent' })
    // Log type must survive the publication branch.
    expect(layout.xaxis.type).toBe('log')
    expect(layout.xaxis2.type).toBeUndefined()
    // Crisp black mirror axes (publication frame).
    expect(layout.xaxis.linecolor).toBe('#000000')
    expect(layout.xaxis.mirror).toBe(true)
    expect(layout.yaxis.mirror).toBe(true)
    expect(layout.xaxis2.mirror).toBe(true)
    expect(layout.yaxis2.mirror).toBe(true)
  })

  it('publication mode hides the legend (subplot identity from axis labels)', () => {
    renderSunsVoc(el, makeResult())
    _toggleStyle(el, 'publication')
    expect(_lastNewPlotLayout()!.showlegend).toBe(false)
  })

  it('publication annotation: pseudo FF in Nature-style format', () => {
    renderSunsVoc(el, makeResult())
    _toggleStyle(el, 'publication')
    const annotations = _lastNewPlotLayout()!.annotations as Array<Record<string, any>>
    expect(annotations).toHaveLength(1)
    const ann = annotations[0]
    expect(ann.text).toContain('pseudo FF')
    expect(ann.text).toContain('83.5%')               // no space before %
    expect(ann.text).not.toContain('83.5 %')
    expect(ann.bgcolor).toBe('rgba(255,255,255,0)')   // transparent
    expect(ann.borderwidth).toBe(0)
    expect(ann.font.family).toBe(PUBLICATION_FONT_FAMILY)
  })

  it('engineering pseudo FF annotation unchanged (regression)', () => {
    renderSunsVoc(el, makeResult())
    const annotations = _lastNewPlotLayout()!.annotations as Array<Record<string, any>>
    expect(annotations).toHaveLength(1)
    expect(annotations[0].text).toBe('pseudo FF = 83.5 %')
  })

  it('raw arrays unchanged across modes (no mutation, byte-identical)', () => {
    const result = makeResult()
    const suns_pre = [...result.suns]
    const Voc_pre = [...result.V_oc]
    const Jsc_pre = [...result.J_sc]
    const JpV_pre = [...result.J_pseudo_V]
    const JpJ_pre = [...result.J_pseudo_J]
    // Engineering render.
    renderSunsVoc(el, result)
    const tracesEng = _lastNewPlotTraces()!
    const yEngFwd = (tracesEng[0].y as number[]).slice()
    const yEngRev = (tracesEng[1].y as number[]).slice()
    // Publication render.
    _toggleStyle(el, 'publication')
    const tracesPub = _lastNewPlotTraces()!
    const yPubFwd = (tracesPub[0].y as number[]).slice()
    const yPubRev = (tracesPub[1].y as number[]).slice()
    // Trace y-arrays equal between modes.
    expect(yPubFwd).toEqual(yEngFwd)
    expect(yPubRev).toEqual(yEngRev)
    // V_oc(suns) trace x is reference-identical to the raw r.suns array.
    expect(tracesEng[0].x).toBe(result.suns)
    expect(tracesPub[0].x).toBe(result.suns)
    // V_oc(suns) trace y is reference-identical to raw r.V_oc.
    expect(tracesEng[0].y).toBe(result.V_oc)
    expect(tracesPub[0].y).toBe(result.V_oc)
    // pseudo J-V x reference-identical to raw r.J_pseudo_V.
    expect(tracesEng[1].x).toBe(result.J_pseudo_V)
    expect(tracesPub[1].x).toBe(result.J_pseudo_V)
    // Raw input arrays remain bit-identical.
    expect(result.suns).toEqual(suns_pre)
    expect(result.V_oc).toEqual(Voc_pre)
    expect(result.J_sc).toEqual(Jsc_pre)
    expect(result.J_pseudo_V).toEqual(JpV_pre)
    expect(result.J_pseudo_J).toEqual(JpJ_pre)
  })

  it('style mode persists across re-render via el.dataset.plotStyleMode', () => {
    const result = makeResult()
    renderSunsVoc(el, result)
    _toggleStyle(el, 'publication')
    expect(el.dataset.plotStyleMode).toBe('publication')
    // External re-entry — must honour the dataset attribute.
    renderSunsVoc(el, result)
    const sel = el.querySelector<HTMLSelectElement>('[data-test="suns-voc-style-mode"]')!
    expect(sel.value).toBe('publication')
    expect(_lastNewPlotLayout()!.font.family).toBe(PUBLICATION_FONT_FAMILY)
  })

  it('toggle round-trip Engineering → Publication → Engineering restores defaults', () => {
    renderSunsVoc(el, makeResult())
    _toggleStyle(el, 'publication')
    expect(_lastNewPlotLayout()!.font.family).toBe(PUBLICATION_FONT_FAMILY)
    _toggleStyle(el, 'engineering')
    expect(_lastNewPlotLayout()!.font.family).toBe('Arial, sans-serif')
    expect(_lastNewPlotConfig()!.displayModeBar).toBeUndefined()
  })
})
