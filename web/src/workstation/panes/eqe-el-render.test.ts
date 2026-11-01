/**
 * Vitest cases for ``renderEQE`` and ``renderEL`` (spectral workstation panes).
 *
 * Covers publication styling, spectral axes, annotations, and raw array
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

import { renderEQE, renderEL } from './main-plot-pane'
import type { EQEResult, ELResult } from '../../types'
import { PUBLICATION_FONT_FAMILY } from '../../plot-theme'
import Plotly from 'plotly.js-basic-dist-min'

const newPlotMock = vi.mocked(Plotly.newPlot)

function makeEQE(overrides: Partial<EQEResult> = {}): EQEResult {
  return {
    wavelengths_nm: [350, 400, 500, 600, 700, 800],
    EQE: [0.10, 0.65, 0.85, 0.85, 0.80, 0.05],
    J_sc_per_lambda: [0.5, 4.0, 6.0, 6.0, 5.0, 0.2],
    J_sc_integrated: 220.0, // A/m^2 → 22.00 mA/cm²
    Phi_incident: 1e21,
    ...overrides,
  }
}

function makeEL(overrides: Partial<ELResult> = {}): ELResult {
  return {
    wavelengths_nm: [600, 650, 700, 750, 800, 850],
    EL_spectrum: [1e16, 1e18, 5e19, 1e20, 5e19, 1e16],
    absorber_absorptance: [0.92, 0.91, 0.88, 0.55, 0.10, 0.02],
    V_inj: 1.10,
    J_inj: -50.0,
    J_em_rad: 0.05,
    EQE_EL: 1.0e-3,
    delta_V_nr_mV: 220.5,
    T: 300.0,
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

// ── EQE tests ────────────────────────────────────────────────────────────

describe('renderEQE — controls and evidence', () => {
  let el: HTMLDivElement
  beforeEach(() => {
    newPlotMock.mockClear()
    el = document.createElement('div')
    document.body.appendChild(el)
  })

  it('renders without a style selector', () => {
    renderEQE(el, makeEQE())
    expect(el.querySelector('[data-test="eqe-toolbar"]')).toBeNull()
    expect(el.querySelector('[data-test="eqe-style-mode"]')).toBeNull()
  })

  it('caps display at 100% and smooths an unphysical EQE>1 spike', () => {
    // index 2 spikes to 1.5 (150%) — numerical noise, physically impossible.
    renderEQE(el, makeEQE({ EQE: [0.10, 0.65, 1.5, 0.85, 0.80, 0.05] }))
    const traces = _lastNewPlotTraces()!
    const rawY = traces[0].y as number[]
    const smoothY = traces[1].y as number[]
    // raw markers clamp the 150% spike to 100%
    expect(Math.max(...rawY)).toBeLessThanOrEqual(100)
    // the median stage removes the lone spike entirely
    expect(Math.max(...smoothY)).toBeLessThanOrEqual(100)
    expect(smoothY[2]).toBeLessThan(100)
  })
})

describe('renderEQE — publication rendering', () => {
  let el: HTMLDivElement
  beforeEach(() => {
    newPlotMock.mockClear()
    el = document.createElement('div')
    document.body.appendChild(el)
  })

  it('uses publication by default with Nature-style layout', () => {
    renderEQE(el, makeEQE())
    const layout = _lastNewPlotLayout()!
    expect(layout.font.family).toBe(PUBLICATION_FONT_FAMILY)
    expect(layout.paper_bgcolor).toBe('#ffffff')
    expect(layout.plot_bgcolor).toBe('#ffffff')
    expect(layout.margin).toEqual({ t: 18, r: 18, b: 48, l: 58 })
    expect(layout.xaxis.showgrid).toBe(false)
    expect(layout.yaxis.showgrid).toBe(false)
    expect(layout.yaxis.range).toEqual([0, 100])
  })

  it('publication mode hides the Plotly modebar', () => {
    renderEQE(el, makeEQE())
    expect(_lastNewPlotConfig()!.displayModeBar).toBe(false)
  })

  it('publication EQE: faint raw markers + smoothed muted-blue line', () => {
    renderEQE(el, makeEQE())
    const traces = _lastNewPlotTraces()!
    expect(traces).toHaveLength(2)
    expect(traces[0].mode).toBe('markers')
    expect(traces[0].marker.color).toBe('#2B6FA3')
    expect(traces[0].marker.opacity).toBe(0.22)
    expect(traces[0].showlegend).toBe(false)
    expect(traces[1].mode).toBe('lines')
    expect(traces[1].line.color).toBe('#2B6FA3')
    expect(traces[1].line.width).toBe(1.75)
  })

  it('publication annotation lives at upper-RIGHT under the falling tail', () => {
    renderEQE(el, makeEQE())
    const ann = (_lastNewPlotLayout()!.annotations) as Array<Record<string, any>>
    expect(ann).toHaveLength(1)
    expect(ann[0].xanchor).toBe('right')
    expect(ann[0].yanchor).toBe('top')
    expect(ann[0].text).toContain('J<sub>sc</sub>(AM1.5G)')
    expect(ann[0].text).toContain('22.00 mA')
    expect(ann[0].bgcolor).toBe('rgba(255,255,255,0)')
    expect(ann[0].borderwidth).toBe(0)
    expect(ann[0].font.family).toBe(PUBLICATION_FONT_FAMILY)
  })

  it('raw arrays unchanged across renders (reference identity for x)', () => {
    const result = makeEQE()
    const wl_pre = [...result.wavelengths_nm]
    const eqe_pre = [...result.EQE]
    renderEQE(el, result)
    const tracesInitial = _lastNewPlotTraces()!
    renderEQE(el, result)
    const tracesRepeated = _lastNewPlotTraces()!
    // x reference identity (no .slice() / .map())
    expect(tracesInitial[0].x).toBe(result.wavelengths_nm)
    expect(tracesRepeated[0].x).toBe(result.wavelengths_nm)
    // Raw arrays remain bit-identical.
    expect(result.wavelengths_nm).toEqual(wl_pre)
    expect(result.EQE).toEqual(eqe_pre)
  })
})

// ── EL tests ─────────────────────────────────────────────────────────────

describe('renderEL — controls and evidence', () => {
  let el: HTMLDivElement
  beforeEach(() => {
    newPlotMock.mockClear()
    el = document.createElement('div')
    document.body.appendChild(el)
  })

  it('renders without a style selector', () => {
    renderEL(el, makeEL())
    expect(el.querySelector('[data-test="el-toolbar"]')).toBeNull()
    expect(el.querySelector('[data-test="el-style-mode"]')).toBeNull()
  })
})

describe('renderEL — publication rendering', () => {
  let el: HTMLDivElement
  beforeEach(() => {
    newPlotMock.mockClear()
    el = document.createElement('div')
    document.body.appendChild(el)
  })

  it('uses publication by default with Nature-style layout', () => {
    renderEL(el, makeEL())
    const layout = _lastNewPlotLayout()!
    expect(layout.font.family).toBe(PUBLICATION_FONT_FAMILY)
    expect(layout.paper_bgcolor).toBe('#ffffff')
    expect(layout.margin).toEqual({ t: 18, r: 18, b: 48, l: 58 })
    expect(layout.xaxis.showgrid).toBe(false)
    expect(layout.yaxis.showgrid).toBe(false)
  })

  it('publication mode hides the Plotly modebar', () => {
    renderEL(el, makeEL())
    expect(_lastNewPlotConfig()!.displayModeBar).toBe(false)
  })

  it('publication: EL trace = hollow circle muted blue lines+markers', () => {
    renderEL(el, makeEL())
    const data = _lastNewPlotTraces()![0]
    expect(data.name).toBe('EL spectrum')
    expect(data.mode).toBe('lines+markers')
    expect(data.marker.symbol).toBe('circle-open')
    expect(data.marker.color).toBe('rgba(0,0,0,0)')
    expect(data.marker.line.color).toBe('#2B6FA3')
    expect(data.line.color).toBe('#2B6FA3')
    expect(data.line.width).toBe(1.75)
    expect(data.yaxis).toBe('y')
  })

  it('publication: absorptance trace = dashed muted red, lines (no markers)', () => {
    renderEL(el, makeEL())
    const abs = _lastNewPlotTraces()![1]
    expect(abs.name).toContain('A<sub>abs</sub>')
    expect(abs.mode).toBe('lines')
    expect(abs.line.color).toBe('#C44536')
    expect(abs.line.dash).toBe('dash')
    expect(abs.line.width).toBe(1.75)
    expect(abs.yaxis).toBe('y2')
    // Absorptance is a smooth band-edge curve → no markers in publication.
    expect(abs.marker).toBeUndefined()
  })

  it('publication: dual y-axis preserved; right axis Nature-style', () => {
    renderEL(el, makeEL())
    const layout = _lastNewPlotLayout()!
    expect(layout.yaxis2.overlaying).toBe('y')
    expect(layout.yaxis2.side).toBe('right')
    expect(layout.yaxis2.range).toEqual([0, 100])
    expect(layout.yaxis2.showgrid).toBe(false)
    // Right axis title carries publication font.
    expect(layout.yaxis2.title.font.family).toBe(PUBLICATION_FONT_FAMILY)
    // Left axis still mirrors to close the panel; right axis does not.
    expect(layout.yaxis.mirror).toBe(true)
    expect(layout.yaxis2.mirror).toBe(false)
  })

  it('publication legend at upper-LEFT (low-λ side, before band-edge onset)', () => {
    renderEL(el, makeEL())
    const layout = _lastNewPlotLayout()!
    expect(layout.legend.x).toBe(0.02)
    expect(layout.legend.y).toBe(0.98)
    expect(layout.legend.xanchor).toBe('left')
    expect(layout.legend.yanchor).toBe('top')
    expect(layout.legend.bgcolor).toBe('rgba(255,255,255,0)')
    expect(layout.legend.borderwidth).toBe(0)
  })

  it('publication annotation: V_inj / EQE_EL / dV_nr stacked with <br>', () => {
    renderEL(el, makeEL())
    const text = (_lastNewPlotLayout()!.annotations as Array<{ text: string }>)[0].text
    expect(text).toContain('V<sub>inj</sub>')
    expect(text).toContain('1.10 V')
    expect(text).toContain('EQE<sub>EL</sub>')
    expect(text).toContain('1.00e-3')
    expect(text).toContain('220.5 mV')
    // Stacked vertically with <br> rather than horizontal &nbsp;
    expect(text).toContain('<br>')
  })

  it('raw arrays unchanged across renders (reference identity for x / y_EL)', () => {
    const result = makeEL()
    const wl_pre = [...result.wavelengths_nm]
    const el_pre = [...result.EL_spectrum]
    const abs_pre = [...result.absorber_absorptance]
    renderEL(el, result)
    const tracesInitial = _lastNewPlotTraces()!
    renderEL(el, result)
    const tracesRepeated = _lastNewPlotTraces()!
    expect(tracesInitial[0].x).toBe(result.wavelengths_nm)
    expect(tracesRepeated[0].x).toBe(result.wavelengths_nm)
    expect(tracesInitial[0].y).toBe(result.EL_spectrum)
    expect(tracesRepeated[0].y).toBe(result.EL_spectrum)
    expect(result.wavelengths_nm).toEqual(wl_pre)
    expect(result.EL_spectrum).toEqual(el_pre)
    expect(result.absorber_absorptance).toEqual(abs_pre)
  })
})
