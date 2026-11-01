/**
 * Vitest cases for ``renderSunsVoc`` (Suns-V_oc workstation pane).
 *
 * Covers publication styling, subplots, and raw array preservation
 * using the shared Plotly mock pattern.
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

describe('renderSunsVoc — controls and evidence', () => {
  let el: HTMLDivElement
  beforeEach(() => {
    newPlotMock.mockClear()
    el = document.createElement('div')
    document.body.appendChild(el)
  })

  it('renders without a style selector', () => {
    renderSunsVoc(el, makeResult())
    expect(el.querySelector('[data-test="suns-voc-toolbar"]')).toBeNull()
    expect(el.querySelector('[data-test="suns-voc-style-mode"]')).toBeNull()
  })
})

describe('renderSunsVoc — publication rendering', () => {
  let el: HTMLDivElement
  beforeEach(() => {
    newPlotMock.mockClear()
    el = document.createElement('div')
    document.body.appendChild(el)
  })

  it('uses publication by default with Nature-style layout', () => {
    renderSunsVoc(el, makeResult())
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

  it('publication mode hides the Plotly modebar', () => {
    renderSunsVoc(el, makeResult())
    expect(_lastNewPlotConfig()!.displayModeBar).toBe(false)
  })

  it('publication V_oc(suns) trace: hollow circle, muted blue, lines+markers', () => {
    renderSunsVoc(el, makeResult())
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

  it('publication suns log axis prints only decade tick labels (dtick=1)', () => {
    // Plotly's auto-tick algorithm otherwise prints minor labels at
    // 2× and 5× between decades (e.g. "0.01 2 5 0.1 2 5 1 2 5 10"),
    // which crowds the compact publication panel. publicationAxis
    // pins ``dtick: 1`` whenever ``isLog`` is set.
    renderSunsVoc(el, makeResult())
    const layout = _lastNewPlotLayout()!
    expect(layout.xaxis.dtick).toBe(1)
    // Linear axes get no dtick (auto-spacing).
    expect(layout.yaxis.dtick).toBeUndefined()
    expect(layout.xaxis2.dtick).toBeUndefined()
    expect(layout.yaxis2.dtick).toBeUndefined()
  })

  it('publication V_oc(suns) y-axis title preserves <sub>oc</sub> HTML', () => {
    // Plotly renders <sub> tags as subscripts; literal V_oc would show
    // an underscore in the figure.
    renderSunsVoc(el, makeResult())
    const layout = _lastNewPlotLayout()!
    expect(layout.yaxis.title.text).toContain('V<sub>oc</sub>')
    expect(layout.yaxis.title.text).toContain('(V)')
    // No literal V_oc text should leak through.
    expect(layout.yaxis.title.text).not.toContain('V_oc')
  })

  it('publication mode hides the legend (subplot identity from axis labels)', () => {
    renderSunsVoc(el, makeResult())
    expect(_lastNewPlotLayout()!.showlegend).toBe(false)
  })

  it('publication annotation: pseudo FF in Nature-style format', () => {
    renderSunsVoc(el, makeResult())
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

  it('raw arrays unchanged across renders (no mutation, byte-identical)', () => {
    const result = makeResult()
    const suns_pre = [...result.suns]
    const Voc_pre = [...result.V_oc]
    const Jsc_pre = [...result.J_sc]
    const JpV_pre = [...result.J_pseudo_V]
    const JpJ_pre = [...result.J_pseudo_J]
    // Initial render.
    renderSunsVoc(el, result)
    const tracesInitial = _lastNewPlotTraces()!
    const yInitialFwd = (tracesInitial[0].y as number[]).slice()
    const yInitialRev = (tracesInitial[1].y as number[]).slice()
    // Repeated render.
    renderSunsVoc(el, result)
    const tracesRepeated = _lastNewPlotTraces()!
    const yRepeatedFwd = (tracesRepeated[0].y as number[]).slice()
    const yRepeatedRev = (tracesRepeated[1].y as number[]).slice()
    // Trace y-arrays equal between renders.
    expect(yRepeatedFwd).toEqual(yInitialFwd)
    expect(yRepeatedRev).toEqual(yInitialRev)
    // V_oc(suns) trace x is reference-identical to the raw r.suns array.
    expect(tracesInitial[0].x).toBe(result.suns)
    expect(tracesRepeated[0].x).toBe(result.suns)
    // V_oc(suns) trace y is reference-identical to raw r.V_oc.
    expect(tracesInitial[0].y).toBe(result.V_oc)
    expect(tracesRepeated[0].y).toBe(result.V_oc)
    // pseudo J-V x reference-identical to raw r.J_pseudo_V.
    expect(tracesInitial[1].x).toBe(result.J_pseudo_V)
    expect(tracesRepeated[1].x).toBe(result.J_pseudo_V)
    // Raw input arrays remain bit-identical.
    expect(result.suns).toEqual(suns_pre)
    expect(result.V_oc).toEqual(Voc_pre)
    expect(result.J_sc).toEqual(Jsc_pre)
    expect(result.J_pseudo_V).toEqual(JpV_pre)
    expect(result.J_pseudo_J).toEqual(JpJ_pre)
  })
})
