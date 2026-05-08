/**
 * Vitest cases for ``renderJV`` (1D J-V workstation pane).
 *
 * Mirrors the 2D pane test pattern: Plotly is mocked because jsdom
 * has no canvas, so layout / trace / config arguments are read from
 * ``newPlot.mock.calls``. Engineering mode is the default and must
 * remain bit-identical to the pre-publication-mode renderer; the
 * Publication branch swaps font / colors / margins / modebar / axes
 * / annotation / hollow markers without mutating raw V/J arrays.
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

import { renderJV } from './main-plot-pane'
import type { JVResult, JVMetrics } from '../../types'
import { PUBLICATION_FONT_FAMILY } from '../../plot-theme'
import Plotly from 'plotly.js-basic-dist-min'

const newPlotMock = vi.mocked(Plotly.newPlot)

function makeMetrics(overrides: Partial<JVMetrics> = {}): JVMetrics {
  return {
    V_oc: 0.951,
    J_sc: 220.0,
    FF: 0.823,
    PCE: 0.1722,
    voc_bracketed: true,
    ...overrides,
  }
}

function makeResult(overrides: Partial<JVResult> = {}): JVResult {
  return {
    V_fwd: [0.0, 0.4, 0.8, 1.0],
    J_fwd: [-220.0, -180.0, -50.0, +120.0],   // 1D: J<0 at V=0 with photocurrent / signed solar convention
    V_rev: [1.0, 0.8, 0.4, 0.0],
    J_rev: [+115.0, -55.0, -185.0, -222.0],
    metrics_fwd: makeMetrics(),
    metrics_rev: makeMetrics({ V_oc: 0.948, J_sc: 218.0, FF: 0.815, PCE: 0.1685 }),
    hysteresis_index: 0.012,
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
  const sel = el.querySelector<HTMLSelectElement>('[data-test="jv1d-style-mode"]')!
  sel.value = mode
  sel.dispatchEvent(new Event('change'))
}

describe('renderJV — toolbar + style mode', () => {
  let el: HTMLDivElement
  beforeEach(() => {
    newPlotMock.mockClear()
    el = document.createElement('div')
    document.body.appendChild(el)
  })

  it('renders the Style: select unconditionally when plot data exists', () => {
    renderJV(el, makeResult())
    expect(el.querySelector('[data-test="jv1d-toolbar"]')).not.toBeNull()
    expect(el.querySelector('[data-test="jv1d-style-mode"]')).not.toBeNull()
  })

  it('default style is engineering (Arial layout, modebar visible)', () => {
    renderJV(el, makeResult())
    const layout = _lastNewPlotLayout()!
    expect(layout.font.family).toBe('Arial, sans-serif')
    const config = _lastNewPlotConfig()!
    // Engineering ``plotConfig`` does not pin displayModeBar → Plotly default.
    expect(config.displayModeBar).toBeUndefined()
  })

  it('Engineering trace colors / symbols match the pre-publication renderer', () => {
    renderJV(el, makeResult())
    const traces = _lastNewPlotTraces()!
    expect(traces).toHaveLength(2)
    // Forward: filled blue circle (existing engineering palette).
    expect(traces[0].name).toBe('Forward')
    expect(traces[0].line.color).toBe('#2563eb')
    expect(traces[0].marker.color).toBe('#2563eb')
    expect(traces[0].marker.symbol).toBeUndefined()
    // Reverse: filled orange dashed square.
    expect(traces[1].name).toBe('Reverse')
    expect(traces[1].line.color).toBe('#ea580c')
    expect(traces[1].line.dash).toBe('dash')
    expect(traces[1].marker.symbol).toBe('square')
  })
})

describe('renderJV — publication style mode', () => {
  let el: HTMLDivElement
  beforeEach(() => {
    newPlotMock.mockClear()
    el = document.createElement('div')
    document.body.appendChild(el)
  })

  it('toggling to publication applies Nature-style layout', () => {
    renderJV(el, makeResult())
    _toggleStyle(el, 'publication')
    const layout = _lastNewPlotLayout()!
    expect(layout.font.family).toBe(PUBLICATION_FONT_FAMILY)
    expect(layout.paper_bgcolor).toBe('#ffffff')
    expect(layout.plot_bgcolor).toBe('#ffffff')
    expect(layout.margin).toEqual({ t: 18, r: 18, b: 48, l: 58 })
    expect(layout.xaxis.showgrid).toBe(false)
    expect(layout.yaxis.showgrid).toBe(false)
    expect(layout.yaxis.zeroline).toBe(true)
  })

  it('publication mode hides the Plotly modebar; engineering does not', () => {
    renderJV(el, makeResult())
    expect(_lastNewPlotConfig()!.displayModeBar).toBeUndefined()
    _toggleStyle(el, 'publication')
    expect(_lastNewPlotConfig()!.displayModeBar).toBe(false)
  })

  it('publication forward trace: hollow circle, muted blue, lines+markers', () => {
    renderJV(el, makeResult())
    _toggleStyle(el, 'publication')
    const traces = _lastNewPlotTraces()!
    const fwd = traces[0]
    expect(fwd.name).toBe('Forward')
    expect(fwd.mode).toBe('lines+markers')
    expect(fwd.marker.symbol).toBe('circle-open')
    expect(fwd.marker.color).toBe('rgba(0,0,0,0)')
    expect(fwd.marker.line.color).toBe('#2B6FA3')
    expect(fwd.marker.line.width).toBe(1.2)
    expect(fwd.marker.size).toBe(5)
    expect(fwd.line.color).toBe('#2B6FA3')
    expect(fwd.line.width).toBe(1.75)
  })

  it('publication reverse trace: hollow circle, muted red, dashed line', () => {
    renderJV(el, makeResult())
    _toggleStyle(el, 'publication')
    const rev = _lastNewPlotTraces()![1]
    expect(rev.name).toBe('Reverse')
    expect(rev.mode).toBe('lines+markers')
    expect(rev.marker.symbol).toBe('circle-open')
    expect(rev.marker.color).toBe('rgba(0,0,0,0)')
    expect(rev.marker.line.color).toBe('#C44536')
    expect(rev.line.color).toBe('#C44536')
    expect(rev.line.dash).toBe('dash')
    expect(rev.line.width).toBe(1.75)
  })

  it('forward-only annotation when metrics_fwd.voc_bracketed=true', () => {
    renderJV(el, makeResult())
    _toggleStyle(el, 'publication')
    const annotations = _lastNewPlotLayout()!.annotations as Array<{ text: string }>
    expect(annotations).toHaveLength(1)
    const text = annotations[0].text
    expect(text).toContain('0.951 V')
    expect(text).toContain('22.00 mA cm⁻²')
    expect(text).toContain('82.3%')
    expect(text).toContain('17.22%')
    // Reverse metrics must NOT leak into the annotation text.
    expect(text).not.toContain('0.948')
    expect(text).not.toContain('21.80')
  })

  it('voc_bracketed=false: shows "not bracketed", no fake V_oc/FF/PCE', () => {
    renderJV(el, makeResult({
      metrics_fwd: { V_oc: 0.0, J_sc: 220.0, FF: 0.0, PCE: 0.0, voc_bracketed: false },
    }))
    _toggleStyle(el, 'publication')
    const annotations = _lastNewPlotLayout()!.annotations as Array<{ text: string }>
    expect(annotations).toHaveLength(1)
    const text = annotations[0].text
    expect(text).toContain('not bracketed')
    expect(text).not.toContain('0.000 V')
    expect(text).not.toContain('0.0%')
    expect(text).not.toContain('0.00%')
    // J_sc still rendered (interpolated at V=0; physically meaningful).
    expect(text).toContain('22.00 mA cm⁻²')
  })

  it('voc_bracketed=undefined (legacy payload): publication style applies, annotation omitted', () => {
    renderJV(el, makeResult({
      metrics_fwd: { V_oc: 0.95, J_sc: 220, FF: 0.82, PCE: 0.17 },     // no voc_bracketed
      metrics_rev: { V_oc: 0.94, J_sc: 218, FF: 0.81, PCE: 0.165 },
    }))
    _toggleStyle(el, 'publication')
    const layout = _lastNewPlotLayout()!
    // Style still applies — Helvetica/Arial, white bg, no grid.
    expect(layout.font.family).toBe(PUBLICATION_FONT_FAMILY)
    expect(layout.paper_bgcolor).toBe('#ffffff')
    // Annotation is suppressed (no voc_bracketed → no fake numbers).
    const annotations = (layout.annotations as Array<unknown>) ?? []
    expect(annotations).toHaveLength(0)
  })

  it('publication operational y-range tighter than engineering autorange', () => {
    // Engineering 1D path leaves yaxis.range unset (autoranges). Publication
    // tight envelope = [-0.15·J_sc_mA, +1.12·J_sc_mA]. J_sc=220 → 22 mA/cm².
    // → publication range ≈ [-3.30, +24.64].
    renderJV(el, makeResult())
    expect(_lastNewPlotLayout()!.yaxis.range).toBeUndefined()
    _toggleStyle(el, 'publication')
    const range = _lastNewPlotLayout()!.yaxis.range as [number, number]
    expect(range[0]).toBeCloseTo(-3.30, 6)
    expect(range[1]).toBeCloseTo(+24.64, 6)
  })

  it('publication x-axis: -0.05 V left margin when sweep starts at V=0', () => {
    renderJV(el, makeResult())
    _toggleStyle(el, 'publication')
    const layout = _lastNewPlotLayout()!
    const [xmin, xmax] = layout.xaxis.range as [number, number]
    expect(xmin).toBeCloseTo(-0.05, 6)
    // V_oc + 0.18 = 1.131; max(V) + 0.05 = 1.05 → cap at 1.05.
    expect(xmax).toBeCloseTo(1.05, 6)
    expect(layout.xaxis.zeroline).toBe(true)
    expect(layout.xaxis.zerolinecolor).toBe('#000000')
  })

  it('publication x-axis: capped at V_oc + 0.18 when sweep extends past V_oc', () => {
    renderJV(el, makeResult({
      V_fwd: [0.0, 0.5, 1.0, 1.5],
      J_fwd: [-220.0, -180.0, -50.0, +400.0],
      V_rev: [1.5, 1.0, 0.5, 0.0],
      J_rev: [+395.0, -55.0, -185.0, -222.0],
    }))
    _toggleStyle(el, 'publication')
    // V_oc + 0.18 = 0.951 + 0.18 = 1.131 < max(V) + 0.05 = 1.55
    const [, xmax] = _lastNewPlotLayout()!.xaxis.range as [number, number]
    expect(xmax).toBeCloseTo(1.131, 6)
  })

  it('raw V/J trace data are unchanged across modes (no mutation, byte-identical)', () => {
    const result = makeResult()
    const V_fwd_pre = [...result.V_fwd]
    const J_fwd_pre = [...result.J_fwd]
    const V_rev_pre = [...result.V_rev]
    const J_rev_pre = [...result.J_rev]
    // Engineering render.
    renderJV(el, result)
    const tracesEng = _lastNewPlotTraces()!
    const yEngFwd = (tracesEng[0].y as number[]).slice()
    const yEngRev = (tracesEng[1].y as number[]).slice()
    // Publication render.
    _toggleStyle(el, 'publication')
    const tracesPub = _lastNewPlotTraces()!
    const yPubFwd = (tracesPub[0].y as number[]).slice()
    const yPubRev = (tracesPub[1].y as number[]).slice()
    // Trace y arrays equal between modes (post-flip-and-scale).
    expect(yPubFwd).toEqual(yEngFwd)
    expect(yPubRev).toEqual(yEngRev)
    // Forward traces use raw r.V_fwd reference for x — must remain identical.
    expect(tracesEng[0].x).toBe(result.V_fwd)
    expect(tracesPub[0].x).toBe(result.V_fwd)
    // Raw input arrays remain bit-identical.
    expect(result.V_fwd).toEqual(V_fwd_pre)
    expect(result.J_fwd).toEqual(J_fwd_pre)
    expect(result.V_rev).toEqual(V_rev_pre)
    expect(result.J_rev).toEqual(J_rev_pre)
  })

  it('style mode persists across re-render via el.dataset.plotStyleMode', () => {
    const result = makeResult()
    renderJV(el, result)
    _toggleStyle(el, 'publication')
    expect(el.dataset.plotStyleMode).toBe('publication')
    // External re-entry (e.g. mountMainPlotPane.update) re-invokes
    // ``renderJV`` on the same stable container — must honour the
    // dataset attribute without an explicit second toggle.
    renderJV(el, result)
    const sel = el.querySelector<HTMLSelectElement>('[data-test="jv1d-style-mode"]')!
    expect(sel.value).toBe('publication')
    expect(_lastNewPlotLayout()!.font.family).toBe(PUBLICATION_FONT_FAMILY)
  })

  it('publication mode keeps engineering branch intact for second render after toggle-back', () => {
    renderJV(el, makeResult())
    _toggleStyle(el, 'publication')
    expect(_lastNewPlotLayout()!.font.family).toBe(PUBLICATION_FONT_FAMILY)
    _toggleStyle(el, 'engineering')
    expect(_lastNewPlotLayout()!.font.family).toBe('Arial, sans-serif')
    // Engineering must NOT carry over publication-only artifacts.
    expect(_lastNewPlotConfig()!.displayModeBar).toBeUndefined()
  })
})
