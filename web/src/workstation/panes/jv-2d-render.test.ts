/**
 * Layer 3 of the Phase 6 acceptance follow-up: ``renderJV2D`` must
 * render a metric-card row and an inline V_oc-not-bracketed warning
 * when the backend reports ``metrics.voc_bracketed === false``. The
 * raw J/V plot itself is delegated to Plotly.newPlot — mocked here
 * because Plotly's renderer needs a real DOM canvas which jsdom
 * doesn't provide.
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

import { renderJV2D } from './main-plot-pane'
import type { JV2DResult } from '../../types'


function makeResult(overrides: Partial<JV2DResult> = {}): JV2DResult {
  return {
    V: [0.0, 0.5, 1.0],
    J: [-300.0, -250.0, +50.0],   // 2D-convention: J<0 at V=0
    grid_x: [0, 250e-9, 500e-9],
    grid_y: [0, 100e-9, 200e-9, 300e-9],
    lateral_bc: 'periodic',
    snapshots: [],
    ...overrides,
  }
}

describe('renderJV2D — metrics card row', () => {
  let el: HTMLDivElement

  beforeEach(() => {
    el = document.createElement('div')
    document.body.appendChild(el)
  })

  it('renders V_oc / J_sc / FF / PCE cards when metrics are present', () => {
    const result = makeResult({
      metrics: {
        V_oc: 0.951,
        J_sc: 220.0,    // A/m² (J_sc-positive convention from backend)
        FF: 0.823,
        PCE: 0.1722,
        voc_bracketed: true,
      },
    })
    renderJV2D(el, result)

    const row = el.querySelector('[data-test="jv2d-metrics-row"]')
    expect(row, 'metric-card row must render when metrics present').not.toBeNull()
    const cards = row!.querySelectorAll('.metric-card')
    expect(cards.length).toBe(4)

    const text = row!.textContent || ''
    // Format expectations mirror the 1D pane (panels/jv.ts).
    expect(text).toContain('0.951 V')         // V_oc
    expect(text).toContain('22.00 mA/cm²')    // J_sc / 10
    expect(text).toContain('82.3 %')          // FF * 100
    expect(text).toContain('17.22 %')         // PCE * 100
  })

  it('does not render the bracket-warning banner when voc_bracketed is true', () => {
    const result = makeResult({
      metrics: {
        V_oc: 0.951, J_sc: 220.0, FF: 0.823, PCE: 0.1722,
        voc_bracketed: true,
      },
    })
    renderJV2D(el, result)
    const warn = el.querySelector('[data-test="jv2d-voc-not-bracketed"]')
    expect(warn).toBeNull()
  })

  it('renders the bracket-warning banner when voc_bracketed is false', () => {
    const result = makeResult({
      metrics: {
        V_oc: 0.0, J_sc: 220.0, FF: 0.0, PCE: 0.0,
        voc_bracketed: false,
      },
    })
    renderJV2D(el, result)
    const warn = el.querySelector('[data-test="jv2d-voc-not-bracketed"]')
    expect(warn, 'warning banner must render when voc_bracketed=false').not.toBeNull()
    expect(warn!.textContent).toContain('V_oc not bracketed')
    expect(warn!.textContent).toContain('increase V_max')
  })

  it('hides V_oc / FF / PCE behind a dash when voc_bracketed is false', () => {
    // Sentinel-zero metrics from the backend must NOT render as
    // "0.000 V" / "0.0 %" / "0.00 %" — those would mislead the user
    // into reading 0 V as a physical V_oc. Render "—" instead.
    const result = makeResult({
      metrics: {
        V_oc: 0.0, J_sc: 220.0, FF: 0.0, PCE: 0.0,
        voc_bracketed: false,
      },
    })
    renderJV2D(el, result)
    const row = el.querySelector('[data-test="jv2d-metrics-row"]')
    expect(row).not.toBeNull()
    const text = row!.textContent || ''
    // V_oc must be the dash, NOT a sentinel-zero string.
    expect(text).not.toContain('0.000 V')
    expect(text).not.toContain('0.0 %')
    expect(text).not.toContain('0.00 %')
    expect(text).toContain('—')
    // J_sc still shown — interpolated at V=0, physically meaningful.
    expect(text).toContain('22.00 mA/cm²')
  })

  it('treats voc_bracketed=undefined (legacy backend) as "no warning"', () => {
    const result = makeResult({
      metrics: {
        V_oc: 0.95, J_sc: 220.0, FF: 0.82, PCE: 0.17,
        // voc_bracketed intentionally omitted
      },
    })
    renderJV2D(el, result)
    const warn = el.querySelector('[data-test="jv2d-voc-not-bracketed"]')
    expect(warn, 'undefined voc_bracketed must NOT trigger the warning').toBeNull()
    // Metric cards should still render.
    const row = el.querySelector('[data-test="jv2d-metrics-row"]')
    expect(row).not.toBeNull()
  })

  it('renders no metric row when the backend payload omits metrics entirely', () => {
    // Pre-Layer-2 backend payload — no metrics field at all.
    const result = makeResult()
    renderJV2D(el, result)
    expect(el.querySelector('[data-test="jv2d-metrics-row"]')).toBeNull()
    expect(el.querySelector('[data-test="jv2d-voc-not-bracketed"]')).toBeNull()
  })
})


// ---------------------------------------------------------------------------
// Layer 4: y-axis operational range. Plotly is mocked, so the test
// reads ``newPlot``'s third positional arg (the layout) to assert whether
// ``yaxis.range`` was set or omitted. Trace data is the second positional
// arg — the test pins that the J trace is byte-identical between renders
// (raw V/J unchanged).
// ---------------------------------------------------------------------------

import Plotly from 'plotly.js-basic-dist-min'
const newPlotMock = vi.mocked(Plotly.newPlot)

function _lastNewPlotLayout(): Record<string, any> | undefined {
  const calls = newPlotMock.mock.calls
  if (calls.length === 0) return undefined
  return calls[calls.length - 1][2] as Record<string, any>
}

function _lastNewPlotTraceY(): number[] | undefined {
  const calls = newPlotMock.mock.calls
  if (calls.length === 0) return undefined
  const data = calls[calls.length - 1][1] as Array<{ y?: number[] }>
  return data?.[0]?.y
}

describe('renderJV2D — Layer 4 y-axis operational range', () => {
  let el: HTMLDivElement

  beforeEach(() => {
    newPlotMock.mockClear()
    el = document.createElement('div')
    document.body.appendChild(el)
  })

  it('applies the operational yaxis.range when bracketed', () => {
    const result = makeResult({
      metrics: {
        V_oc: 0.95, J_sc: 200.0, FF: 0.82, PCE: 0.16,
        voc_bracketed: true,
      },
    })
    renderJV2D(el, result)
    const layout = _lastNewPlotLayout()
    expect(layout).toBeDefined()
    const range = layout!.yaxis?.range as [number, number] | undefined
    expect(range, 'yaxis.range must be set in operational mode').toBeDefined()
    // J_sc=200 A/m² → 20 mA/cm². [-0.15*20, 1.12*20] = [-3, 22.4].
    expect(range![0]).toBeCloseTo(-3.0, 6)
    expect(range![1]).toBeCloseTo(+22.4, 6)
  })

  it('renders the operational view without a range toolbar', () => {
    const result = makeResult({
      metrics: {
        V_oc: 0.95, J_sc: 200.0, FF: 0.82, PCE: 0.16,
        voc_bracketed: true,
      },
    })
    renderJV2D(el, result)
    expect(_lastNewPlotLayout()!.yaxis?.range).toBeDefined()
    expect(el.querySelector('[data-test="jv2d-toolbar"]')).toBeNull()
    expect(el.querySelector('[data-test="jv2d-range-mode"]')).toBeNull()
  })

  it('falls back to autorange when voc_bracketed is false', () => {
    const result = makeResult({
      metrics: {
        V_oc: 0.0, J_sc: 200.0, FF: 0.0, PCE: 0.0,
        voc_bracketed: false,
      },
    })
    renderJV2D(el, result)
    expect(_lastNewPlotLayout()!.yaxis?.range).toBeUndefined()
  })

  it('falls back to autorange when metrics is missing', () => {
    // No metrics on the payload (legacy backend).
    const result = makeResult()
    renderJV2D(el, result)
    expect(_lastNewPlotLayout()!.yaxis?.range).toBeUndefined()
  })

  it('falls back to autorange when J_sc is non-positive', () => {
    const result = makeResult({
      metrics: {
        V_oc: 0.95, J_sc: 0.0, FF: 0.82, PCE: 0.16,
        voc_bracketed: true,
      },
    })
    renderJV2D(el, result)
    expect(_lastNewPlotLayout()!.yaxis?.range).toBeUndefined()
  })

  it('falls back to autorange when J_sc is non-finite', () => {
    const result = makeResult({
      metrics: {
        V_oc: 0.95, J_sc: Number.POSITIVE_INFINITY, FF: 0.82, PCE: 0.16,
        voc_bracketed: true,
      },
    })
    renderJV2D(el, result)
    expect(_lastNewPlotLayout()!.yaxis?.range).toBeUndefined()
  })

  it('operational limits preserve every raw voltage and current sample', () => {
    const result = makeResult({
      V: [-1.0, 0.0, 0.5, 1.0, 1.5],
      J: [-310.0, -300.0, -250.0, +50.0, +500.0],
      metrics: {
        V_oc: 0.95, J_sc: 200.0, FF: 0.82, PCE: 0.16,
        voc_bracketed: true,
      },
    })
    const originalResult = structuredClone(result)
    renderJV2D(el, result)
    const trace = _lastNewPlotTrace()!
    expect(_lastNewPlotLayout()!.xaxis.range[1]).toBeCloseTo(1.13, 6)
    // The injection tail remains in the trace even outside the visible axes.
    expect(trace.x).toEqual(originalResult.V)
    expect(trace.y).toEqual(originalResult.J.map(j => -j / 10))
    expect(result).toEqual(originalResult)
  })

  it('keeps operational limits across renders despite obsolete full-range state', () => {
    const result = makeResult({
      metrics: {
        V_oc: 0.95, J_sc: 200.0, FF: 0.82, PCE: 0.16,
        voc_bracketed: true,
      },
    })
    renderJV2D(el, result)
    const originalLayout = _lastNewPlotLayout()!
    expect(originalLayout.yaxis.range).toBeDefined()
    expect(originalLayout.xaxis.range).toBeDefined()
    // A container retained from the old UI cannot restore the removed mode.
    el.dataset.jv2dMode = 'full'
    renderJV2D(el, result)
    expect(_lastNewPlotLayout()!.yaxis.range).toEqual(originalLayout.yaxis.range)
    expect(_lastNewPlotLayout()!.xaxis.range).toEqual(originalLayout.xaxis.range)
    expect(el.querySelector('[data-test="jv2d-range-mode"]')).toBeNull()
  })

  it('renders legacy payloads without a range or style selector', () => {
    const result = makeResult()
    renderJV2D(el, result)
    expect(el.querySelector('[data-test="jv2d-toolbar"]')).toBeNull()
    expect(el.querySelector('[data-test="jv2d-style-mode"]')).toBeNull()
    expect(el.querySelector('[data-test="jv2d-range-mode"]')).toBeNull()
  })
})


// ---------------------------------------------------------------------------
// Publication visual-style mode (Nature-style single-panel theme).
// Publication styling and operational limits preserve the raw V/J arrays.
// ---------------------------------------------------------------------------

import { PUBLICATION_FONT_FAMILY } from '../../plot-theme'

function _lastNewPlotConfig(): Record<string, any> | undefined {
  const calls = newPlotMock.mock.calls
  if (calls.length === 0) return undefined
  return calls[calls.length - 1][3] as Record<string, any>
}

function _lastNewPlotTrace(): Record<string, any> | undefined {
  const calls = newPlotMock.mock.calls
  if (calls.length === 0) return undefined
  const data = calls[calls.length - 1][1] as Array<Record<string, any>>
  return data?.[0]
}

describe('renderJV2D — publication rendering', () => {
  let el: HTMLDivElement

  beforeEach(() => {
    newPlotMock.mockClear()
    el = document.createElement('div')
    document.body.appendChild(el)
  })

  it('uses publication by default with Nature-style layout', () => {
    const result = makeResult({
      metrics: { V_oc: 0.95, J_sc: 200, FF: 0.82, PCE: 0.16, voc_bracketed: true },
    })
    renderJV2D(el, result)
    const layout = _lastNewPlotLayout()!
    expect(layout.font.family).toBe(PUBLICATION_FONT_FAMILY)
    expect(layout.paper_bgcolor).toBe('#ffffff')
    expect(layout.plot_bgcolor).toBe('#ffffff')
    expect(layout.margin).toEqual({ t: 18, r: 18, b: 48, l: 58 })
    // Default legend lives upper-left in publication layout (single-trace
    // 2D figure separately suppresses display via ``showlegend: false``).
    expect(layout.legend.x).toBe(0.02)
    expect(layout.legend.xanchor).toBe('left')
  })

  it('publication config hides the modebar', () => {
    const result = makeResult({
      metrics: { V_oc: 0.95, J_sc: 200, FF: 0.82, PCE: 0.16, voc_bracketed: true },
    })
    renderJV2D(el, result)
    expect(_lastNewPlotConfig()!.displayModeBar).toBe(false)
  })

  it('publication mode adds a metric annotation when voc_bracketed=true', () => {
    const result = makeResult({
      metrics: { V_oc: 0.951, J_sc: 220.0, FF: 0.823, PCE: 0.1722, voc_bracketed: true },
    })
    renderJV2D(el, result)
    const annotations = _lastNewPlotLayout()!.annotations as Array<{ text: string }>
    expect(annotations.length).toBe(1)
    expect(annotations[0].text).toContain('0.951 V')
    expect(annotations[0].text).toContain('22.00 mA cm⁻²')
    // Nature-style: no space before percent sign.
    expect(annotations[0].text).toContain('82.3%')
    expect(annotations[0].text).toContain('17.22%')
    expect(annotations[0].text).not.toContain('82.3 %')
    expect(annotations[0].text).not.toContain('17.22 %')
  })

  it('publication mode shows "not bracketed" annotation when voc_bracketed=false', () => {
    const result = makeResult({
      metrics: { V_oc: 0.0, J_sc: 220.0, FF: 0.0, PCE: 0.0, voc_bracketed: false },
    })
    renderJV2D(el, result)
    const annotations = _lastNewPlotLayout()!.annotations as Array<{ text: string }>
    expect(annotations.length).toBe(1)
    expect(annotations[0].text).toContain('not bracketed')
    expect(annotations[0].text).not.toContain('0.000 V')
    expect(annotations[0].text).not.toContain('0.0%')
    expect(annotations[0].text).not.toContain('0.00%')
  })

  it('publication mode renders no annotation when metrics are missing', () => {
    // Independent of metrics — publication style still applies.
    const result = makeResult()
    renderJV2D(el, result)
    const layout = _lastNewPlotLayout()!
    expect(layout.font.family).toBe(PUBLICATION_FONT_FAMILY)
    expect((layout.annotations as Array<unknown>).length).toBe(0)
  })

  it('publication mode renders no annotation when voc_bracketed is undefined', () => {
    const result = makeResult({
      metrics: { V_oc: 0.95, J_sc: 220, FF: 0.82, PCE: 0.17 },
    })
    renderJV2D(el, result)
    const layout = _lastNewPlotLayout()!
    expect((layout.annotations as Array<unknown>).length).toBe(0)
  })

  it('keeps publication styling and operational limits across renders', () => {
    const result = makeResult({
      metrics: { V_oc: 0.95, J_sc: 200, FF: 0.82, PCE: 0.16, voc_bracketed: true },
    })
    renderJV2D(el, result)
    const originalLayout = _lastNewPlotLayout()!
    expect(originalLayout.yaxis.range).toBeDefined()
    renderJV2D(el, result)
    const layout = _lastNewPlotLayout()!
    expect(layout.yaxis.range).toEqual(originalLayout.yaxis.range)
    expect(layout.xaxis.range).toEqual(originalLayout.xaxis.range)
    expect(layout.font.family).toBe(PUBLICATION_FONT_FAMILY)
  })

  it('raw J trace is byte-identical across renders', () => {
    const result = makeResult({
      metrics: { V_oc: 0.95, J_sc: 200, FF: 0.82, PCE: 0.16, voc_bracketed: true },
    })
    renderJV2D(el, result)
    const yInitial = _lastNewPlotTraceY()!.slice()
    renderJV2D(el, result)
    const yRepeated = _lastNewPlotTraceY()!.slice()
    expect(yRepeated).toEqual(yInitial)
    // And raw input array must remain untouched in both renders.
    expect(result.J).toEqual([-300.0, -250.0, +50.0])
    // Both equal the documented post-flip-and-scale conversion.
    const expected = result.J.map(j => -j / 10)
    expect(yInitial).toEqual(expected)
    expect(yRepeated).toEqual(expected)
  })

  it('publication trace: lines+markers, hollow circles, muted blue, thin line', () => {
    const result = makeResult({
      metrics: { V_oc: 0.95, J_sc: 200, FF: 0.82, PCE: 0.16, voc_bracketed: true },
    })
    renderJV2D(el, result)
    const trace = _lastNewPlotTrace()!
    expect(trace.mode).toBe('lines+markers')
    expect(trace.marker.symbol).toBe('circle-open')
    expect(trace.marker.color).toBe('rgba(0,0,0,0)')
    expect(trace.marker.line.color).toBe('#2B6FA3')
    expect(trace.marker.line.width).toBe(1.2)
    expect(trace.marker.size).toBe(5)
    expect(trace.line.color).toBe('#2B6FA3')
    expect(trace.line.width).toBe(1.75)
  })

  it('publication mode keeps metric cards + warning banner (cards untouched by style)', () => {
    const result = makeResult({
      metrics: { V_oc: 0.0, J_sc: 200.0, FF: 0.0, PCE: 0.0, voc_bracketed: false },
    })
    renderJV2D(el, result)
    expect(el.querySelector('[data-test="jv2d-metrics-row"]')).not.toBeNull()
    expect(el.querySelector('[data-test="jv2d-voc-not-bracketed"]')).not.toBeNull()
  })

  it('publication mode draws horizontal zero-line at J=0', () => {
    const result = makeResult({
      metrics: { V_oc: 0.95, J_sc: 200, FF: 0.82, PCE: 0.16, voc_bracketed: true },
    })
    renderJV2D(el, result)
    expect(_lastNewPlotLayout()!.yaxis.zeroline).toBe(true)
  })

  it('publication+operational adds a small negative x-margin to the axis', () => {
    // Sweep starts at V=0 (no negative voltage in the data). Under
    // publication+operational the visible x-axis is shifted left to
    // -0.05 V so the V=0 reference line is visible — without
    // mutating r.V or inventing data points there.
    const result = makeResult({
      metrics: { V_oc: 0.95, J_sc: 200, FF: 0.82, PCE: 0.16, voc_bracketed: true },
    })
    renderJV2D(el, result)
    const layout = _lastNewPlotLayout()!
    expect(layout.xaxis.range).toBeDefined()
    const [xmin, xmax] = layout.xaxis.range as [number, number]
    expect(xmin).toBeCloseTo(-0.05, 6)
    // V_oc + 0.18 = 1.13, max(V)+0.05 = 1.05 → xmax = min(1.05, 1.13) = 1.05.
    expect(xmax).toBeCloseTo(1.05, 6)
    // Visible axis crosses V=0, so the vertical zero-line is on.
    expect(layout.xaxis.zeroline).toBe(true)
    expect(layout.xaxis.zerolinecolor).toBe('#000000')
    // Raw V array is untouched (margin is axis-only).
    expect(result.V).toEqual([0.0, 0.5, 1.0])
  })

  it('publication+operational caps x-axis at V_oc+0.18 when bracketed', () => {
    // max(V)+0.05 = 1.50+0.05 = 1.55; V_oc+0.18 = 0.95+0.18 = 1.13 → cap at 1.13.
    const result = makeResult({
      V: [0.0, 0.5, 1.0, 1.5],
      J: [-300.0, -250.0, +50.0, +500.0],
      metrics: { V_oc: 0.95, J_sc: 200, FF: 0.82, PCE: 0.16, voc_bracketed: true },
    })
    renderJV2D(el, result)
    const [, xmax] = _lastNewPlotLayout()!.xaxis.range as [number, number]
    expect(xmax).toBeCloseTo(1.13, 6)
  })

  it('publication mode hides the legend (single-trace 2D figure)', () => {
    const result = makeResult({
      metrics: { V_oc: 0.95, J_sc: 200, FF: 0.82, PCE: 0.16, voc_bracketed: true },
    })
    renderJV2D(el, result)
    expect(_lastNewPlotLayout()!.showlegend).toBe(false)
  })

  it('annotation default placement is upper-left lower-third (Nature-style)', () => {
    const result = makeResult({
      metrics: { V_oc: 0.951, J_sc: 220.0, FF: 0.823, PCE: 0.1722, voc_bracketed: true },
    })
    renderJV2D(el, result)
    const annotations = _lastNewPlotLayout()!.annotations as Array<Record<string, unknown>>
    expect(annotations.length).toBe(1)
    const ann = annotations[0]
    expect(ann.x).toBe(0.12)
    expect(ann.y).toBe(0.34)
    expect(ann.xanchor).toBe('left')
    expect(ann.yanchor).toBe('top')
    // No heavy box.
    expect(ann.bgcolor).toBe('rgba(255,255,255,0)')
    expect(ann.borderwidth).toBe(0)
  })
})
