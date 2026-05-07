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
