/**
 * Vitest cases for ``renderEQE`` and ``renderEL`` (spectral workstation panes).
 *
 * Same Plotly mock pattern as the 2D / 1D / Suns-V_oc / V_oc(T) panes.
 * Engineering mode is the default and bit-identical to the
 * pre-publication renderer; the Publication branch swaps font /
 * colors / margins / modebar / axes / annotation / hollow markers
 * without mutating the raw wavelength / EQE / EL_spectrum /
 * absorptance arrays.
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

function _toggleStyle(el: HTMLElement, ds: string, mode: 'engineering' | 'publication'): void {
  const sel = el.querySelector<HTMLSelectElement>(`[data-test="${ds}"]`)!
  sel.value = mode
  sel.dispatchEvent(new Event('change'))
}

// ── EQE tests ────────────────────────────────────────────────────────────

describe('renderEQE — toolbar + style mode', () => {
  let el: HTMLDivElement
  beforeEach(() => {
    newPlotMock.mockClear()
    el = document.createElement('div')
    document.body.appendChild(el)
  })

  it('renders the Style: select unconditionally when plot data exists', () => {
    renderEQE(el, makeEQE())
    expect(el.querySelector('[data-test="eqe-toolbar"]')).not.toBeNull()
    expect(el.querySelector('[data-test="eqe-style-mode"]')).not.toBeNull()
  })

  it('default style is engineering (Arial layout, modebar visible)', () => {
    renderEQE(el, makeEQE())
    expect(_lastNewPlotLayout()!.font.family).toBe('Arial, sans-serif')
    expect(_lastNewPlotConfig()!.displayModeBar).toBeUndefined()
  })

  it('engineering trace color and y-range match the pre-publication renderer', () => {
    renderEQE(el, makeEQE())
    const traces = _lastNewPlotTraces()!
    expect(traces).toHaveLength(1)
    expect(traces[0].mode).toBe('lines+markers')
    expect(traces[0].line.color).toBe('#2563eb')
    expect(traces[0].marker.color).toBe('#2563eb')
    expect(_lastNewPlotLayout()!.yaxis.range).toEqual([0, 100])
  })

  it('engineering annotation: J_sc(AM1.5G) at upper-right', () => {
    renderEQE(el, makeEQE())
    const ann = _lastNewPlotLayout()!.annotations as Array<Record<string, any>>
    expect(ann).toHaveLength(1)
    expect(ann[0].text).toContain('J<sub>sc</sub>(AM1.5G)')
    expect(ann[0].text).toContain('22.00 mA')
    expect(ann[0].xanchor).toBe('right')
    expect(ann[0].yanchor).toBe('top')
  })
})

describe('renderEQE — publication style mode', () => {
  let el: HTMLDivElement
  beforeEach(() => {
    newPlotMock.mockClear()
    el = document.createElement('div')
    document.body.appendChild(el)
  })

  it('toggling to publication applies Nature-style layout', () => {
    renderEQE(el, makeEQE())
    _toggleStyle(el, 'eqe-style-mode', 'publication')
    const layout = _lastNewPlotLayout()!
    expect(layout.font.family).toBe(PUBLICATION_FONT_FAMILY)
    expect(layout.paper_bgcolor).toBe('#ffffff')
    expect(layout.plot_bgcolor).toBe('#ffffff')
    expect(layout.margin).toEqual({ t: 18, r: 18, b: 48, l: 58 })
    expect(layout.xaxis.showgrid).toBe(false)
    expect(layout.yaxis.showgrid).toBe(false)
    expect(layout.yaxis.range).toEqual([0, 100])
  })

  it('publication mode hides the Plotly modebar; engineering does not', () => {
    renderEQE(el, makeEQE())
    expect(_lastNewPlotConfig()!.displayModeBar).toBeUndefined()
    _toggleStyle(el, 'eqe-style-mode', 'publication')
    expect(_lastNewPlotConfig()!.displayModeBar).toBe(false)
  })

  it('publication EQE trace: hollow circle, muted blue, lines+markers', () => {
    renderEQE(el, makeEQE())
    _toggleStyle(el, 'eqe-style-mode', 'publication')
    const data = _lastNewPlotTraces()![0]
    expect(data.mode).toBe('lines+markers')
    expect(data.marker.symbol).toBe('circle-open')
    expect(data.marker.color).toBe('rgba(0,0,0,0)')
    expect(data.marker.line.color).toBe('#2B6FA3')
    expect(data.line.color).toBe('#2B6FA3')
    expect(data.line.width).toBe(1.75)
  })

  it('publication annotation lives at upper-RIGHT under the falling tail', () => {
    renderEQE(el, makeEQE())
    _toggleStyle(el, 'eqe-style-mode', 'publication')
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

  it('raw arrays unchanged across modes (reference identity for x)', () => {
    const result = makeEQE()
    const wl_pre = [...result.wavelengths_nm]
    const eqe_pre = [...result.EQE]
    renderEQE(el, result)
    const tracesEng = _lastNewPlotTraces()!
    _toggleStyle(el, 'eqe-style-mode', 'publication')
    const tracesPub = _lastNewPlotTraces()!
    // x reference identity (no .slice() / .map())
    expect(tracesEng[0].x).toBe(result.wavelengths_nm)
    expect(tracesPub[0].x).toBe(result.wavelengths_nm)
    // Raw arrays remain bit-identical.
    expect(result.wavelengths_nm).toEqual(wl_pre)
    expect(result.EQE).toEqual(eqe_pre)
  })

  it('style mode persists across re-render via el.dataset.plotStyleMode', () => {
    const result = makeEQE()
    renderEQE(el, result)
    _toggleStyle(el, 'eqe-style-mode', 'publication')
    expect(el.dataset.plotStyleMode).toBe('publication')
    renderEQE(el, result)
    const sel = el.querySelector<HTMLSelectElement>('[data-test="eqe-style-mode"]')!
    expect(sel.value).toBe('publication')
    expect(_lastNewPlotLayout()!.font.family).toBe(PUBLICATION_FONT_FAMILY)
  })
})

// ── EL tests ─────────────────────────────────────────────────────────────

describe('renderEL — toolbar + style mode', () => {
  let el: HTMLDivElement
  beforeEach(() => {
    newPlotMock.mockClear()
    el = document.createElement('div')
    document.body.appendChild(el)
  })

  it('renders the Style: select unconditionally when plot data exists', () => {
    renderEL(el, makeEL())
    expect(el.querySelector('[data-test="el-toolbar"]')).not.toBeNull()
    expect(el.querySelector('[data-test="el-style-mode"]')).not.toBeNull()
  })

  it('default style is engineering (Arial layout, modebar visible)', () => {
    renderEL(el, makeEL())
    expect(_lastNewPlotLayout()!.font.family).toBe('Arial, sans-serif')
    expect(_lastNewPlotConfig()!.displayModeBar).toBeUndefined()
  })

  it('engineering: 2 traces with dual y-axis (EL on y, absorptance on y2)', () => {
    renderEL(el, makeEL())
    const traces = _lastNewPlotTraces()!
    expect(traces).toHaveLength(2)
    expect(traces[0].name).toBe('EL spectrum')
    expect(traces[0].yaxis).toBe('y')
    expect(traces[1].name).toContain('A<sub>abs</sub>')
    expect(traces[1].yaxis).toBe('y2')
    const layout = _lastNewPlotLayout()!
    expect(layout.yaxis2.overlaying).toBe('y')
    expect(layout.yaxis2.side).toBe('right')
    expect(layout.yaxis2.range).toEqual([0, 100])
  })

  it('engineering annotation at upper-LEFT with V_inj / EQE_EL / dV_nr (separators)', () => {
    renderEL(el, makeEL())
    const ann = _lastNewPlotLayout()!.annotations as Array<Record<string, any>>
    expect(ann).toHaveLength(1)
    const t = ann[0].text
    expect(t).toContain('V<sub>inj</sub>')
    expect(t).toContain('1.10 V')
    expect(t).toContain('EQE<sub>EL</sub>')
    expect(t).toContain('1.00e-3')
    expect(t).toContain('220.5 mV')
    expect(ann[0].xanchor).toBe('left')
    expect(ann[0].yanchor).toBe('top')
  })
})

describe('renderEL — publication style mode', () => {
  let el: HTMLDivElement
  beforeEach(() => {
    newPlotMock.mockClear()
    el = document.createElement('div')
    document.body.appendChild(el)
  })

  it('toggling to publication applies Nature-style layout', () => {
    renderEL(el, makeEL())
    _toggleStyle(el, 'el-style-mode', 'publication')
    const layout = _lastNewPlotLayout()!
    expect(layout.font.family).toBe(PUBLICATION_FONT_FAMILY)
    expect(layout.paper_bgcolor).toBe('#ffffff')
    expect(layout.margin).toEqual({ t: 18, r: 18, b: 48, l: 58 })
    expect(layout.xaxis.showgrid).toBe(false)
    expect(layout.yaxis.showgrid).toBe(false)
  })

  it('publication mode hides the Plotly modebar', () => {
    renderEL(el, makeEL())
    _toggleStyle(el, 'el-style-mode', 'publication')
    expect(_lastNewPlotConfig()!.displayModeBar).toBe(false)
  })

  it('publication: EL trace = hollow circle muted blue lines+markers', () => {
    renderEL(el, makeEL())
    _toggleStyle(el, 'el-style-mode', 'publication')
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
    _toggleStyle(el, 'el-style-mode', 'publication')
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
    _toggleStyle(el, 'el-style-mode', 'publication')
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
    _toggleStyle(el, 'el-style-mode', 'publication')
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
    _toggleStyle(el, 'el-style-mode', 'publication')
    const text = (_lastNewPlotLayout()!.annotations as Array<{ text: string }>)[0].text
    expect(text).toContain('V<sub>inj</sub>')
    expect(text).toContain('1.10 V')
    expect(text).toContain('EQE<sub>EL</sub>')
    expect(text).toContain('1.00e-3')
    expect(text).toContain('220.5 mV')
    // Stacked vertically with <br> rather than horizontal &nbsp;
    expect(text).toContain('<br>')
  })

  it('raw arrays unchanged across modes (reference identity for x / y_EL)', () => {
    const result = makeEL()
    const wl_pre = [...result.wavelengths_nm]
    const el_pre = [...result.EL_spectrum]
    const abs_pre = [...result.absorber_absorptance]
    renderEL(el, result)
    const tracesEng = _lastNewPlotTraces()!
    _toggleStyle(el, 'el-style-mode', 'publication')
    const tracesPub = _lastNewPlotTraces()!
    expect(tracesEng[0].x).toBe(result.wavelengths_nm)
    expect(tracesPub[0].x).toBe(result.wavelengths_nm)
    expect(tracesEng[0].y).toBe(result.EL_spectrum)
    expect(tracesPub[0].y).toBe(result.EL_spectrum)
    expect(result.wavelengths_nm).toEqual(wl_pre)
    expect(result.EL_spectrum).toEqual(el_pre)
    expect(result.absorber_absorptance).toEqual(abs_pre)
  })

  it('toggle round-trip Engineering → Publication → Engineering restores defaults', () => {
    renderEL(el, makeEL())
    _toggleStyle(el, 'el-style-mode', 'publication')
    expect(_lastNewPlotLayout()!.font.family).toBe(PUBLICATION_FONT_FAMILY)
    _toggleStyle(el, 'el-style-mode', 'engineering')
    expect(_lastNewPlotLayout()!.font.family).toBe('Arial, sans-serif')
    expect(_lastNewPlotConfig()!.displayModeBar).toBeUndefined()
  })
})
