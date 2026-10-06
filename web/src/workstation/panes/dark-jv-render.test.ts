/**
 * Vitest cases for ``renderDarkJV`` (Dark J-V workstation pane).
 *
 * Covers publication styling, diode and ideality views, and raw V/J
 * preservation using the shared Plotly mock pattern.
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

import { renderDarkJV } from './main-plot-pane'
import type { DarkJVResult } from '../../types'
import { PUBLICATION_FONT_FAMILY } from '../../plot-theme'
import Plotly from 'plotly.js-basic-dist-min'

const newPlotMock = vi.mocked(Plotly.newPlot)

function makeResult(overrides: Partial<DarkJVResult> = {}): DarkJVResult {
  return {
    V: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    J: [1e-7, 1e-5, -1e-3, -1.0, -50.0, -200.0],   // A/m²
    n_ideality: 1.65,
    J_0: 3.2e-9,
    V_fit_lo: 0.6,
    V_fit_hi: 0.85,
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

function _toggleCurve(el: HTMLElement, mode: 'signed' | 'ideality'): void {
  const sel = el.querySelector<HTMLSelectElement>('[data-test="dark-jv-curve-mode"]')!
  sel.value = mode
  sel.dispatchEvent(new Event('change'))
}

describe('renderDarkJV — controls and evidence', () => {
  let el: HTMLDivElement
  beforeEach(() => {
    newPlotMock.mockClear()
    el = document.createElement('div')
    document.body.appendChild(el)
  })

  it('renders without a style selector', () => {
    renderDarkJV(el, makeResult())
    expect(el.querySelector('[data-test="dark-jv-toolbar"]')).not.toBeNull()
    expect(el.querySelector('[data-test="dark-jv-curve-mode"]')).not.toBeNull()
    expect(el.querySelector('[data-test="dark-jv-style-mode"]')).toBeNull()
  })

  it('default curve uses diode-sign J-V on a linear axis', () => {
    const result = makeResult()
    renderDarkJV(el, result)
    const traces = _lastNewPlotTraces()!
    expect(traces).toHaveLength(1)
    expect(traces[0].name).toBe('Dark J-V')
    expect(traces[0].mode).toBe('lines+markers')
    expect(traces[0].y).toEqual(result.J.map(j => -j / 10))
    expect(_lastNewPlotLayout()!.yaxis.type).toBeUndefined()
  })

  it('default Curve selector label is Diode J-V', () => {
    renderDarkJV(el, makeResult())
    const sel = el.querySelector<HTMLSelectElement>('[data-test="dark-jv-curve-mode"]')!
    expect(sel.selectedOptions[0].textContent).toBe('Diode J-V')
  })

  it('ideality curve: log y-axis with decade-only ticks (dtick=1)', () => {
    renderDarkJV(el, makeResult())
    _toggleCurve(el, 'ideality')
    const layout = _lastNewPlotLayout()!
    expect(layout.yaxis.type).toBe('log')
    expect(layout.yaxis.dtick).toBe(1)
  })
})

describe('renderDarkJV — publication rendering', () => {
  let el: HTMLDivElement
  beforeEach(() => {
    newPlotMock.mockClear()
    el = document.createElement('div')
    document.body.appendChild(el)
  })

  it('uses publication by default with Nature-style layout', () => {
    renderDarkJV(el, makeResult())
    const layout = _lastNewPlotLayout()!
    expect(layout.font.family).toBe(PUBLICATION_FONT_FAMILY)
    expect(layout.paper_bgcolor).toBe('#ffffff')
    expect(layout.plot_bgcolor).toBe('#ffffff')
    expect(layout.margin).toEqual({ t: 18, r: 18, b: 48, l: 58 })
    expect(layout.xaxis.showgrid).toBe(false)
    expect(layout.yaxis.showgrid).toBe(false)
  })

  it('publication mode hides the Plotly modebar', () => {
    renderDarkJV(el, makeResult())
    expect(_lastNewPlotConfig()!.displayModeBar).toBe(false)
  })

  it('publication: hollow circle muted blue trace, lines+markers', () => {
    renderDarkJV(el, makeResult())
    const data = _lastNewPlotTraces()![0]
    expect(data.name).toBe('Dark J-V')
    expect(data.mode).toBe('lines+markers')
    expect(data.marker.symbol).toBe('circle-open')
    expect(data.marker.color).toBe('rgba(0,0,0,0)')
    expect(data.marker.line.color).toBe('#2B6FA3')
    expect(data.line.color).toBe('#2B6FA3')
    expect(data.line.width).toBe(1.75)
  })

  it('publication signed curve: linear y-axis with zero line', () => {
    renderDarkJV(el, makeResult())
    const layout = _lastNewPlotLayout()!
    expect(layout.yaxis.type).toBeUndefined()
    expect(layout.yaxis.zeroline).toBe(true)
  })

  it('publication ideality curve: log y-axis with decade-only ticks (dtick=1)', () => {
    renderDarkJV(el, makeResult())
    _toggleCurve(el, 'ideality')
    const layout = _lastNewPlotLayout()!
    expect(layout.yaxis.type).toBe('log')
    expect(layout.yaxis.dtick).toBe(1)
  })

  it('publication: fit window highlighted with neutral grey translucent band', () => {
    renderDarkJV(el, makeResult())
    _toggleCurve(el, 'ideality')
    const shapes = _lastNewPlotLayout()!.shapes as Array<Record<string, any>>
    expect(shapes).toHaveLength(1)
    expect(shapes[0].x0).toBe(0.6)
    expect(shapes[0].x1).toBe(0.85)
    expect(shapes[0].fillcolor).toBe('rgba(0, 0, 0, 0.06)')
  })

  it('publication annotation lives at upper-LEFT (under diode turn-on)', () => {
    renderDarkJV(el, makeResult())
    _toggleCurve(el, 'ideality')
    const layout = _lastNewPlotLayout()!
    const ann = layout.annotations as Array<Record<string, any>>
    expect(ann).toHaveLength(1)
    expect(ann[0].xanchor).toBe('left')
    expect(ann[0].yanchor).toBe('top')
    expect(ann[0].x).toBe(0.05)
    expect(ann[0].y).toBe(0.95)
    expect(ann[0].bgcolor).toBe('rgba(255,255,255,0)')
    expect(ann[0].borderwidth).toBe(0)
    expect(ann[0].font.family).toBe(PUBLICATION_FONT_FAMILY)
  })

  it('publication annotation text: n / J_0 / fit window stacked with <br>', () => {
    renderDarkJV(el, makeResult())
    _toggleCurve(el, 'ideality')
    const text = (_lastNewPlotLayout()!.annotations as Array<{ text: string }>)[0].text
    expect(text).toContain('n = 1.65')
    expect(text).toContain('J<sub>0</sub>')
    expect(text).toContain('3.20e-9')
    expect(text).toContain('[0.60, 0.85] V')
    expect(text).toContain('<br>')
  })

  it('raw arrays unchanged across renders (V is reference-identical)', () => {
    const result = makeResult()
    const V_pre = [...result.V]
    const J_pre = [...result.J]
    renderDarkJV(el, result)
    const tracesInitial = _lastNewPlotTraces()!
    renderDarkJV(el, result)
    const tracesRepeated = _lastNewPlotTraces()!
    // V (x) is reference-identical (no .slice() / .map()) across renders.
    expect(tracesInitial[0].x).toBe(result.V)
    expect(tracesRepeated[0].x).toBe(result.V)
    // Y is derived for display but the source J array stays bit-identical.
    expect(result.V).toEqual(V_pre)
    expect(result.J).toEqual(J_pre)
  })

  it('curve mode persists across re-render via el.dataset.darkJvCurveMode', () => {
    const result = makeResult()
    renderDarkJV(el, result)
    _toggleCurve(el, 'ideality')
    expect(el.dataset.darkJvCurveMode).toBe('ideality')
    renderDarkJV(el, result)
    const sel = el.querySelector<HTMLSelectElement>('[data-test="dark-jv-curve-mode"]')!
    expect(sel.value).toBe('ideality')
    expect(_lastNewPlotTraces()![0].name).toBe('|J|')
  })
})
