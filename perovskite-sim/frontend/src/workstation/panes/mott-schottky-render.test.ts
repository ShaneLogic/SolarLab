/**
 * Vitest cases for ``renderMottSchottky`` (Mott-Schottky workstation pane).
 *
 * Covers publication styling, fit windows, and raw array preservation
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

import { renderMottSchottky } from './main-plot-pane'
import type { MottSchottkyResult } from '../../types'
import { PUBLICATION_FONT_FAMILY } from '../../plot-theme'
import Plotly from 'plotly.js-basic-dist-min'

const newPlotMock = vi.mocked(Plotly.newPlot)

function makeResult(overrides: Partial<MottSchottkyResult> = {}): MottSchottkyResult {
  return {
    V: [-0.4, -0.2, 0.0, 0.2, 0.4, 0.6],
    C: [1.0e-6, 1.4e-6, 2.0e-6, 3.0e-6, 5.0e-6, 1.0e-5],
    one_over_C2: [1.0e12, 5.1e11, 2.5e11, 1.1e11, 4.0e10, 1.0e10],
    V_bi_fit: 0.945,
    N_eff_fit: 8.30e22,
    V_fit_lo: -0.2,
    V_fit_hi: 0.4,
    frequency: 1.0e4,
    eps_r_used: 24.1,
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

describe('renderMottSchottky — controls and evidence', () => {
  let el: HTMLDivElement
  beforeEach(() => {
    newPlotMock.mockClear()
    el = document.createElement('div')
    document.body.appendChild(el)
  })

  it('renders without a style selector', () => {
    renderMottSchottky(el, makeResult())
    expect(el.querySelector('[data-test="mott-schottky-toolbar"]')).toBeNull()
    expect(el.querySelector('[data-test="mott-schottky-style-mode"]')).toBeNull()
  })
})

describe('renderMottSchottky — publication rendering', () => {
  let el: HTMLDivElement
  beforeEach(() => {
    newPlotMock.mockClear()
    el = document.createElement('div')
    document.body.appendChild(el)
  })

  it('uses publication by default with Nature-style layout', () => {
    renderMottSchottky(el, makeResult())
    const layout = _lastNewPlotLayout()!
    expect(layout.font.family).toBe(PUBLICATION_FONT_FAMILY)
    expect(layout.paper_bgcolor).toBe('#ffffff')
    expect(layout.plot_bgcolor).toBe('#ffffff')
    expect(layout.margin).toEqual({ t: 18, r: 18, b: 48, l: 58 })
    expect(layout.xaxis.showgrid).toBe(false)
    expect(layout.yaxis.showgrid).toBe(false)
  })

  it('publication mode hides the Plotly modebar', () => {
    renderMottSchottky(el, makeResult())
    expect(_lastNewPlotConfig()!.displayModeBar).toBe(false)
  })

  it('publication: hollow circle muted blue trace, lines+markers', () => {
    renderMottSchottky(el, makeResult())
    const data = _lastNewPlotTraces()![0]
    expect(data.name).toContain('1/C')
    expect(data.mode).toBe('lines+markers')
    expect(data.marker.symbol).toBe('circle-open')
    expect(data.marker.color).toBe('rgba(0,0,0,0)')
    expect(data.marker.line.color).toBe('#2B6FA3')
    expect(data.line.color).toBe('#2B6FA3')
    expect(data.line.width).toBe(1.75)
  })

  it('publication: fit window highlighted with neutral grey translucent band', () => {
    renderMottSchottky(el, makeResult())
    const shapes = _lastNewPlotLayout()!.shapes as Array<Record<string, any>>
    expect(shapes).toHaveLength(1)
    expect(shapes[0].x0).toBe(-0.2)
    expect(shapes[0].x1).toBe(0.4)
    expect(shapes[0].fillcolor).toBe('rgba(0, 0, 0, 0.06)')
  })

  it('publication annotation lives at upper-RIGHT (curve slopes UL → LR)', () => {
    renderMottSchottky(el, makeResult())
    const layout = _lastNewPlotLayout()!
    const ann = layout.annotations as Array<Record<string, any>>
    expect(ann).toHaveLength(1)
    expect(ann[0].xanchor).toBe('right')
    expect(ann[0].yanchor).toBe('top')
    expect(ann[0].x).toBe(0.95)
    expect(ann[0].y).toBe(0.95)
    expect(ann[0].bgcolor).toBe('rgba(255,255,255,0)')
    expect(ann[0].borderwidth).toBe(0)
    expect(ann[0].font.family).toBe(PUBLICATION_FONT_FAMILY)
  })

  it('publication annotation text: V_bi,app / N_eff / f stacked with <br>', () => {
    renderMottSchottky(el, makeResult())
    const text = (_lastNewPlotLayout()!.annotations as Array<{ text: string }>)[0].text
    expect(text).toContain('V<sub>bi,app</sub>')
    expect(text).toContain('0.945 V')
    expect(text).toContain('N<sub>eff</sub>')
    expect(text).toContain('8.30e+22')
    expect(text).toContain('1.0e+4 Hz')
    expect(text).toContain('<br>')
  })

  it('raw arrays unchanged across renders (V is reference-identical)', () => {
    const result = makeResult()
    const V_pre = [...result.V]
    const inv_pre = [...result.one_over_C2]
    renderMottSchottky(el, result)
    const tracesInitial = _lastNewPlotTraces()!
    renderMottSchottky(el, result)
    const tracesRepeated = _lastNewPlotTraces()!
    expect(tracesInitial[0].x).toBe(result.V)
    expect(tracesRepeated[0].x).toBe(result.V)
    expect(tracesInitial[0].y).toBe(result.one_over_C2)
    expect(tracesRepeated[0].y).toBe(result.one_over_C2)
    expect(result.V).toEqual(V_pre)
    expect(result.one_over_C2).toEqual(inv_pre)
  })
})
