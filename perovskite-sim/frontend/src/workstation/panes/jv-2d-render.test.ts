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
// Layer 4: y-axis operational-range toggle. Plotly is mocked, so the test
// reads ``newPlot``'s third positional arg (the layout) to assert whether
// ``yaxis.range`` was set or omitted. Trace data is the second positional
// arg — the test pins that the J trace is byte-identical between modes
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

  it('applies clipped yaxis.range in default Operational mode when bracketed', () => {
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
    // J_sc=200 A/m² → 20 mA/cm². [-0.5*20, 1.5*20] = [-10, 30].
    expect(range![0]).toBeCloseTo(-10.0, 6)
    expect(range![1]).toBeCloseTo(+30.0, 6)
  })

  it('switching to Full sweep removes yaxis.range (autorange)', () => {
    const result = makeResult({
      metrics: {
        V_oc: 0.95, J_sc: 200.0, FF: 0.82, PCE: 0.16,
        voc_bracketed: true,
      },
    })
    renderJV2D(el, result)
    expect(_lastNewPlotLayout()!.yaxis?.range).toBeDefined()

    // Simulate user toggling the select to "Full sweep".
    const sel = el.querySelector<HTMLSelectElement>('[data-test="jv2d-range-mode"]')
    expect(sel, 'toolbar select must render when metrics present').not.toBeNull()
    sel!.value = 'full'
    sel!.dispatchEvent(new Event('change'))

    const layout = _lastNewPlotLayout()
    expect(layout!.yaxis?.range, 'Full sweep must omit yaxis.range').toBeUndefined()
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

  it('raw J trace is byte-identical between Operational and Full sweep modes', () => {
    const result = makeResult({
      metrics: {
        V_oc: 0.95, J_sc: 200.0, FF: 0.82, PCE: 0.16,
        voc_bracketed: true,
      },
    })
    // First render — operational (default).
    renderJV2D(el, result)
    const yOp = _lastNewPlotTraceY()!.slice()
    // Toggle to full and re-render via the change handler.
    const sel = el.querySelector<HTMLSelectElement>('[data-test="jv2d-range-mode"]')!
    sel.value = 'full'
    sel.dispatchEvent(new Event('change'))
    const yFull = _lastNewPlotTraceY()!.slice()
    // Trace data must match exactly. The Layer 4 toggle changes
    // yaxis.range only — never the data.
    expect(yFull).toEqual(yOp)
    // And both must equal the expected post-flip-and-scale conversion
    // of the original J array (no mutation of r.J).
    const expected = result.J.map(j => -j / 10)
    expect(yOp).toEqual(expected)
    expect(result.J).toEqual([-300.0, -250.0, +50.0])  // input untouched
  })

  it('persists the toggle state on the stable container across renderJV2D calls', () => {
    const result = makeResult({
      metrics: {
        V_oc: 0.95, J_sc: 200.0, FF: 0.82, PCE: 0.16,
        voc_bracketed: true,
      },
    })
    // First render — default 'operational' (no dataset key written
    // until user toggles).
    renderJV2D(el, result)
    expect(_lastNewPlotLayout()!.yaxis?.range).toBeDefined()
    // Toggle to full via the select.
    const sel = el.querySelector<HTMLSelectElement>('[data-test="jv2d-range-mode"]')!
    sel.value = 'full'
    sel.dispatchEvent(new Event('change'))
    expect(el.dataset.jv2dMode).toBe('full')
    // Now simulate a fresh ``renderJV2D`` call from outside (e.g.
    // mountMainPlotPane.update). The dataset attribute on the same
    // ``el`` must be honoured — no clipped range.
    renderJV2D(el, result)
    const sel2 = el.querySelector<HTMLSelectElement>('[data-test="jv2d-range-mode"]')!
    expect(sel2.value).toBe('full')
    expect(_lastNewPlotLayout()!.yaxis?.range).toBeUndefined()
  })

  it('renders the Style select even when metrics are absent', () => {
    // Style is a visual-mode toggle; it must remain available even
    // for legacy backend payloads that omit the metrics field. The
    // Range select stays gated on metrics (it needs J_sc to clip).
    const result = makeResult()
    renderJV2D(el, result)
    expect(el.querySelector('[data-test="jv2d-toolbar"]')).not.toBeNull()
    expect(el.querySelector('[data-test="jv2d-style-mode"]')).not.toBeNull()
    expect(el.querySelector('[data-test="jv2d-range-mode"]')).toBeNull()
  })
})


// ---------------------------------------------------------------------------
// Publication visual-style mode (Nature-style single-panel theme).
// Engineering mode remains the default; publication mode swaps the Plotly
// layout / trace style / config WITHOUT mutating raw V/J. Range mode
// (Operational vs Full sweep) is independent of the visual style.
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

function _toggleStyle(el: HTMLElement, mode: 'engineering' | 'publication'): void {
  const sel = el.querySelector<HTMLSelectElement>('[data-test="jv2d-style-mode"]')!
  sel.value = mode
  sel.dispatchEvent(new Event('change'))
}

describe('renderJV2D — publication style mode', () => {
  let el: HTMLDivElement

  beforeEach(() => {
    newPlotMock.mockClear()
    el = document.createElement('div')
    document.body.appendChild(el)
  })

  it('default is engineering — Arial layout, modebar visible (regression)', () => {
    const result = makeResult({
      metrics: { V_oc: 0.95, J_sc: 200, FF: 0.82, PCE: 0.16, voc_bracketed: true },
    })
    renderJV2D(el, result)
    const layout = _lastNewPlotLayout()!
    expect(layout.font.family).toBe('Arial, sans-serif')
    const config = _lastNewPlotConfig()!
    // plotConfig() does not set displayModeBar — default Plotly behaviour.
    expect(config.displayModeBar).toBeUndefined()
  })

  it('toggling to publication applies Nature-style layout', () => {
    const result = makeResult({
      metrics: { V_oc: 0.95, J_sc: 200, FF: 0.82, PCE: 0.16, voc_bracketed: true },
    })
    renderJV2D(el, result)
    _toggleStyle(el, 'publication')
    const layout = _lastNewPlotLayout()!
    expect(layout.font.family).toBe(PUBLICATION_FONT_FAMILY)
    expect(layout.paper_bgcolor).toBe('#ffffff')
    expect(layout.plot_bgcolor).toBe('#ffffff')
    expect(layout.margin).toEqual({ t: 18, r: 18, b: 48, l: 58 })
    // Compact in-plot legend, top-right corner.
    expect(layout.legend.x).toBe(0.98)
    expect(layout.legend.xanchor).toBe('right')
  })

  it('publication config hides the modebar; engineering does not', () => {
    const result = makeResult({
      metrics: { V_oc: 0.95, J_sc: 200, FF: 0.82, PCE: 0.16, voc_bracketed: true },
    })
    renderJV2D(el, result)
    expect(_lastNewPlotConfig()!.displayModeBar).toBeUndefined()
    _toggleStyle(el, 'publication')
    expect(_lastNewPlotConfig()!.displayModeBar).toBe(false)
  })

  it('publication mode adds a metric annotation when voc_bracketed=true', () => {
    const result = makeResult({
      metrics: { V_oc: 0.951, J_sc: 220.0, FF: 0.823, PCE: 0.1722, voc_bracketed: true },
    })
    renderJV2D(el, result)
    _toggleStyle(el, 'publication')
    const annotations = _lastNewPlotLayout()!.annotations as Array<{ text: string }>
    expect(annotations.length).toBe(1)
    expect(annotations[0].text).toContain('0.951 V')
    expect(annotations[0].text).toContain('22.00 mA cm⁻²')
    expect(annotations[0].text).toContain('82.3 %')
    expect(annotations[0].text).toContain('17.22 %')
  })

  it('publication mode shows "not bracketed" annotation when voc_bracketed=false', () => {
    const result = makeResult({
      metrics: { V_oc: 0.0, J_sc: 220.0, FF: 0.0, PCE: 0.0, voc_bracketed: false },
    })
    renderJV2D(el, result)
    _toggleStyle(el, 'publication')
    const annotations = _lastNewPlotLayout()!.annotations as Array<{ text: string }>
    expect(annotations.length).toBe(1)
    expect(annotations[0].text).toContain('not bracketed')
    expect(annotations[0].text).not.toContain('0.000 V')
    expect(annotations[0].text).not.toContain('0.0 %')
    expect(annotations[0].text).not.toContain('0.00 %')
  })

  it('publication mode renders no annotation when metrics are missing', () => {
    // Independent of metrics — publication style still applies.
    const result = makeResult()
    renderJV2D(el, result)
    _toggleStyle(el, 'publication')
    const layout = _lastNewPlotLayout()!
    expect(layout.font.family).toBe(PUBLICATION_FONT_FAMILY)
    expect((layout.annotations as Array<unknown>).length).toBe(0)
  })

  it('publication mode renders no annotation when voc_bracketed is undefined', () => {
    const result = makeResult({
      metrics: { V_oc: 0.95, J_sc: 220, FF: 0.82, PCE: 0.17 },
    })
    renderJV2D(el, result)
    _toggleStyle(el, 'publication')
    const layout = _lastNewPlotLayout()!
    expect((layout.annotations as Array<unknown>).length).toBe(0)
  })

  it('style toggle persists across re-render via el.dataset.plotStyleMode', () => {
    const result = makeResult({
      metrics: { V_oc: 0.95, J_sc: 200, FF: 0.82, PCE: 0.16, voc_bracketed: true },
    })
    renderJV2D(el, result)
    _toggleStyle(el, 'publication')
    expect(el.dataset.plotStyleMode).toBe('publication')
    // External re-entry (e.g. mountMainPlotPane.update) must honour the
    // dataset attribute without an explicit second toggle.
    renderJV2D(el, result)
    const sel = el.querySelector<HTMLSelectElement>('[data-test="jv2d-style-mode"]')!
    expect(sel.value).toBe('publication')
    expect(_lastNewPlotLayout()!.font.family).toBe(PUBLICATION_FONT_FAMILY)
  })

  it('range mode and style mode are independent (operational + publication)', () => {
    const result = makeResult({
      metrics: { V_oc: 0.95, J_sc: 200, FF: 0.82, PCE: 0.16, voc_bracketed: true },
    })
    // Default = operational + engineering.
    renderJV2D(el, result)
    expect(_lastNewPlotLayout()!.yaxis.range).toBeDefined()
    expect(_lastNewPlotLayout()!.font.family).toBe('Arial, sans-serif')
    // Toggle style only — range stays operational.
    _toggleStyle(el, 'publication')
    const pubLayout = _lastNewPlotLayout()!
    expect(pubLayout.yaxis.range).toBeDefined()                      // still clipped
    expect((pubLayout.yaxis.range as [number, number])[0]).toBeCloseTo(-10.0, 6)
    expect((pubLayout.yaxis.range as [number, number])[1]).toBeCloseTo(+30.0, 6)
    expect(pubLayout.font.family).toBe(PUBLICATION_FONT_FAMILY)      // publication font
  })

  it('range "Full sweep" + style "Publication" composes — autorange + Helvetica', () => {
    const result = makeResult({
      metrics: { V_oc: 0.95, J_sc: 200, FF: 0.82, PCE: 0.16, voc_bracketed: true },
    })
    renderJV2D(el, result)
    // Toggle range to full first.
    const rangeSel = el.querySelector<HTMLSelectElement>('[data-test="jv2d-range-mode"]')!
    rangeSel.value = 'full'
    rangeSel.dispatchEvent(new Event('change'))
    expect(_lastNewPlotLayout()!.yaxis.range).toBeUndefined()
    // Then toggle style to publication.
    _toggleStyle(el, 'publication')
    const layout = _lastNewPlotLayout()!
    expect(layout.yaxis.range).toBeUndefined()                       // still autorange
    expect(layout.font.family).toBe(PUBLICATION_FONT_FAMILY)
  })

  it('raw J trace is byte-identical between engineering and publication modes', () => {
    const result = makeResult({
      metrics: { V_oc: 0.95, J_sc: 200, FF: 0.82, PCE: 0.16, voc_bracketed: true },
    })
    renderJV2D(el, result)
    const yEng = _lastNewPlotTraceY()!.slice()
    _toggleStyle(el, 'publication')
    const yPub = _lastNewPlotTraceY()!.slice()
    expect(yPub).toEqual(yEng)
    // And raw input array must remain untouched in both renders.
    expect(result.J).toEqual([-300.0, -250.0, +50.0])
    // Both equal the documented post-flip-and-scale conversion.
    const expected = result.J.map(j => -j / 10)
    expect(yEng).toEqual(expected)
    expect(yPub).toEqual(expected)
  })

  it('publication trace uses hollow markers + muted academic palette', () => {
    const result = makeResult({
      metrics: { V_oc: 0.95, J_sc: 200, FF: 0.82, PCE: 0.16, voc_bracketed: true },
    })
    renderJV2D(el, result)
    _toggleStyle(el, 'publication')
    const trace = _lastNewPlotTrace()!
    expect(trace.marker.symbol).toBe('circle-open')
    expect(trace.marker.color).toBe('rgba(0,0,0,0)')
    expect(trace.marker.line.color).toBe('#1f3a93')
    expect(trace.line.color).toBe('#1f3a93')
    expect(trace.line.width).toBe(1.75)
  })

  it('publication mode keeps metric cards + warning banner (cards untouched by style)', () => {
    const result = makeResult({
      metrics: { V_oc: 0.0, J_sc: 200.0, FF: 0.0, PCE: 0.0, voc_bracketed: false },
    })
    renderJV2D(el, result)
    _toggleStyle(el, 'publication')
    expect(el.querySelector('[data-test="jv2d-metrics-row"]')).not.toBeNull()
    expect(el.querySelector('[data-test="jv2d-voc-not-bracketed"]')).not.toBeNull()
  })

  it('publication mode draws horizontal zero-line; vertical zero-line only when V<0', () => {
    // Default fixture starts at V=0 — no negative bias, so xaxis zero
    // line stays off in publication mode.
    const result = makeResult({
      metrics: { V_oc: 0.95, J_sc: 200, FF: 0.82, PCE: 0.16, voc_bracketed: true },
    })
    renderJV2D(el, result)
    _toggleStyle(el, 'publication')
    const layout = _lastNewPlotLayout()!
    expect(layout.yaxis.zeroline).toBe(true)
    expect(layout.xaxis.zeroline).toBe(false)
  })

  it('publication mode draws vertical zero-line when sweep includes negative V', () => {
    const result = makeResult({
      V: [-0.2, 0.0, 0.5, 1.0],
      J: [-100.0, -300.0, -250.0, +50.0],
      metrics: { V_oc: 0.95, J_sc: 200, FF: 0.82, PCE: 0.16, voc_bracketed: true },
    })
    renderJV2D(el, result)
    _toggleStyle(el, 'publication')
    const layout = _lastNewPlotLayout()!
    expect(layout.xaxis.zeroline).toBe(true)
    expect(layout.xaxis.zerolinecolor).toBe('#000000')
  })
})
