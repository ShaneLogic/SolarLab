/**
 * Vitest cases for ``renderVocT`` (V_oc(T) workstation pane).
 *
 * Covers publication styling, fitted metrics, and raw array preservation
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

import { renderVocT } from './main-plot-pane'
import type { VocTResult } from '../../types'
import { PUBLICATION_FONT_FAMILY } from '../../plot-theme'
import Plotly from 'plotly.js-basic-dist-min'

const newPlotMock = vi.mocked(Plotly.newPlot)

function makeResult(overrides: Partial<VocTResult> = {}): VocTResult {
  return {
    T_arr: [250, 275, 300, 325, 350],                   // K
    V_oc_arr: [1.05, 0.98, 0.91, 0.84, 0.77],           // V (decreasing with T)
    J_sc_arr: [400, 405, 410, 415, 420],                // A/m²
    slope: -0.00280,                                    // V/K
    intercept_0K: 1.75,                                 // V
    E_A_eV: 1.75,                                       // eV (≈ intercept)
    R_squared: 0.9985,
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

describe('renderVocT — controls and evidence', () => {
  let el: HTMLDivElement
  beforeEach(() => {
    newPlotMock.mockClear()
    el = document.createElement('div')
    document.body.appendChild(el)
  })

  it('renders without a style selector', () => {
    renderVocT(el, makeResult())
    expect(el.querySelector('[data-test="voc-t-toolbar"]')).toBeNull()
    expect(el.querySelector('[data-test="voc-t-style-mode"]')).toBeNull()
  })
})

describe('renderVocT — publication rendering', () => {
  let el: HTMLDivElement
  beforeEach(() => {
    newPlotMock.mockClear()
    el = document.createElement('div')
    document.body.appendChild(el)
  })

  it('uses publication by default with Nature-style layout', () => {
    renderVocT(el, makeResult())
    const layout = _lastNewPlotLayout()!
    expect(layout.font.family).toBe(PUBLICATION_FONT_FAMILY)
    expect(layout.paper_bgcolor).toBe('#ffffff')
    expect(layout.plot_bgcolor).toBe('#ffffff')
    expect(layout.margin).toEqual({ t: 18, r: 18, b: 48, l: 58 })
    expect(layout.xaxis.showgrid).toBe(false)
    expect(layout.yaxis.showgrid).toBe(false)
  })

  it('publication mode hides the Plotly modebar', () => {
    renderVocT(el, makeResult())
    expect(_lastNewPlotConfig()!.displayModeBar).toBe(false)
  })

  it('publication V_oc(T) trace: hollow circle, muted blue, lines+markers', () => {
    renderVocT(el, makeResult())
    const data = _lastNewPlotTraces()![0]
    expect(data.name).toBe('V<sub>oc</sub>(T)')
    expect(data.mode).toBe('lines+markers')
    expect(data.marker.symbol).toBe('circle-open')
    expect(data.marker.color).toBe('rgba(0,0,0,0)')
    expect(data.marker.line.color).toBe('#2B6FA3')
    expect(data.line.color).toBe('#2B6FA3')
    expect(data.line.width).toBe(1.75)
  })

  it('publication fit trace: muted red dashed line, no markers', () => {
    renderVocT(el, makeResult())
    const fit = _lastNewPlotTraces()![1]
    expect(fit.name).toContain('linear fit')
    expect(fit.mode).toBe('lines')
    expect(fit.line.color).toBe('#C44536')
    expect(fit.line.dash).toBe('dash')
    expect(fit.line.width).toBe(1.75)
    // Fit line is a mathematical overlay → no markers.
    expect(fit.marker).toBeUndefined()
  })

  it('publication annotation lives at lower-LEFT (data slopes upper-left → lower-right)', () => {
    renderVocT(el, makeResult())
    const layout = _lastNewPlotLayout()!
    const ann = layout.annotations as Array<Record<string, any>>
    expect(ann).toHaveLength(1)
    expect(ann[0].xanchor).toBe('left')
    expect(ann[0].yanchor).toBe('bottom')
    expect(ann[0].x).toBe(0.05)
    expect(ann[0].y).toBe(0.05)
    expect(ann[0].bgcolor).toBe('rgba(255,255,255,0)')
    expect(ann[0].borderwidth).toBe(0)
    expect(ann[0].font.family).toBe(PUBLICATION_FONT_FAMILY)
  })

  it('publication annotation text: E_A / dV_oc-dT / R² formatted Nature-style', () => {
    renderVocT(el, makeResult())
    const text = (_lastNewPlotLayout()!.annotations as Array<{ text: string }>)[0].text
    expect(text).toContain('E<sub>A</sub>')
    expect(text).toContain('1.750 eV')
    expect(text).toContain('-2.80 mV/K')
    expect(text).toContain('R²')
    expect(text).toContain('0.999')
    // Stacked vertically with <br> rather than horizontal &nbsp;
    // separators (paper-figure convention).
    expect(text).toContain('<br>')
  })

  it('publication legend lives at upper-RIGHT (empty quadrant)', () => {
    renderVocT(el, makeResult())
    const layout = _lastNewPlotLayout()!
    expect(layout.legend.x).toBe(0.98)
    expect(layout.legend.y).toBe(0.98)
    expect(layout.legend.xanchor).toBe('right')
    expect(layout.legend.yanchor).toBe('top')
    expect(layout.legend.bgcolor).toBe('rgba(255,255,255,0)')
    expect(layout.legend.borderwidth).toBe(0)
  })

  it('raw arrays unchanged across renders (no mutation, byte-identical)', () => {
    const result = makeResult()
    const T_pre = [...result.T_arr]
    const V_pre = [...result.V_oc_arr]
    const J_pre = [...result.J_sc_arr]
    renderVocT(el, result)
    const tracesInitial = _lastNewPlotTraces()!
    const yInitial = (tracesInitial[0].y as number[]).slice()
    renderVocT(el, result)
    const tracesRepeated = _lastNewPlotTraces()!
    const yRepeated = (tracesRepeated[0].y as number[]).slice()
    // Trace y arrays equal between renders.
    expect(yRepeated).toEqual(yInitial)
    // Reference identity on the raw arrays (no .slice() / .map()).
    expect(tracesInitial[0].x).toBe(result.T_arr)
    expect(tracesRepeated[0].x).toBe(result.T_arr)
    expect(tracesInitial[0].y).toBe(result.V_oc_arr)
    expect(tracesRepeated[0].y).toBe(result.V_oc_arr)
    // Raw input arrays remain bit-identical.
    expect(result.T_arr).toEqual(T_pre)
    expect(result.V_oc_arr).toEqual(V_pre)
    expect(result.J_sc_arr).toEqual(J_pre)
  })
})
