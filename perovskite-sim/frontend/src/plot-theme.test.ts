/**
 * Unit tests for the publication-style plot theme helpers added alongside
 * the existing engineering-mode ``baseLayout`` / ``plotConfig`` exports.
 *
 * These tests pin two invariants:
 *   1. The engineering exports (``baseLayout`` etc.) are unchanged — any
 *      drift here would break every existing renderer in main-plot-pane
 *      and panels/{jv,impedance,degradation}.
 *   2. The new publication helpers produce the Nature-style defaults
 *      documented in the design (white bg, black axes, hollow markers,
 *      compact margins, modebar off, in-plot legend, optional metric
 *      annotation that never invents fake V_oc / FF / PCE).
 *
 * No DOM-renderer logic is exercised here; that lives in the per-pane
 * test files (e.g. ``jv-2d-render.test.ts``).
 */
import { describe, it, expect } from 'vitest'
import {
  baseLayout,
  plotConfig,
  publicationLayout,
  publicationAxis,
  publicationConfig,
  publicationTraceStyle,
  metricAnnotation,
  readPlotStyleMode,
  writePlotStyleMode,
  PUBLICATION_FONT_FAMILY,
  PUBLICATION_PALETTE,
  PUBLICATION_LINE_WIDTH,
  PUBLICATION_MARKER_SIZE,
} from './plot-theme'

describe('plot-theme — engineering baseline (regression)', () => {
  it('baseLayout font family is Arial (engineering, unchanged)', () => {
    const L = baseLayout() as { font: { family: string } }
    expect(L.font.family).toBe('Arial, sans-serif')
  })
  it('baseLayout paper bg is white', () => {
    const L = baseLayout() as { paper_bgcolor: string }
    expect(L.paper_bgcolor).toBe('#ffffff')
  })
  it('plotConfig does NOT set displayModeBar (modebar visible by default)', () => {
    const c = plotConfig('foo') as Record<string, unknown>
    expect(c.displayModeBar).toBeUndefined()
    expect(c.displaylogo).toBe(false)
  })
})

describe('plot-theme — publicationLayout', () => {
  it('uses Helvetica/Arial font family at 11 px body', () => {
    const L = publicationLayout() as { font: { family: string; size: number } }
    expect(L.font.family).toBe(PUBLICATION_FONT_FAMILY)
    expect(L.font.size).toBe(11)
  })
  it('white paper + plot background', () => {
    const L = publicationLayout() as { paper_bgcolor: string; plot_bgcolor: string }
    expect(L.paper_bgcolor).toBe('#ffffff')
    expect(L.plot_bgcolor).toBe('#ffffff')
  })
  it('compact margins', () => {
    const L = publicationLayout() as { margin: { t: number; r: number; b: number; l: number } }
    expect(L.margin).toEqual({ t: 18, r: 18, b: 48, l: 58 })
  })
  it('legend lives inside plot area at upper-left, no border, transparent bg', () => {
    const L = publicationLayout() as { legend: Record<string, unknown> }
    expect(L.legend.x).toBe(0.02)
    expect(L.legend.y).toBe(0.98)
    expect(L.legend.xanchor).toBe('left')
    expect(L.legend.yanchor).toBe('top')
    expect(L.legend.bgcolor).toBe('rgba(255,255,255,0)')
    expect(L.legend.borderwidth).toBe(0)
  })
  it('overrides shallow-merge into the layout', () => {
    const L = publicationLayout({ height: 500 }) as { height: number }
    expect(L.height).toBe(500)
  })
  it('default xaxis + yaxis are publication-style frames', () => {
    const L = publicationLayout() as {
      xaxis: Record<string, unknown>
      yaxis: Record<string, unknown>
    }
    expect(L.xaxis.showline).toBe(true)
    expect(L.xaxis.linecolor).toBe('#000000')
    expect(L.xaxis.showgrid).toBe(false)
    expect(L.yaxis.showline).toBe(true)
    expect(L.yaxis.showgrid).toBe(false)
  })
})

describe('plot-theme — publicationAxis', () => {
  it('default frame: black mirror, no grid, ticks outside, no zero-line', () => {
    const a = publicationAxis() as Record<string, unknown>
    expect(a.showline).toBe(true)
    expect(a.linecolor).toBe('#000000')
    expect(a.mirror).toBe(true)
    expect(a.showgrid).toBe(false)
    expect(a.ticks).toBe('outside')
    expect(a.zeroline).toBe(false)
  })
  it('title is wrapped with publication font at 12 px', () => {
    const a = publicationAxis({ title: 'Voltage (V)' }) as {
      title: { text: string; font: { size: number; family: string } }
    }
    expect(a.title.text).toBe('Voltage (V)')
    expect(a.title.font.size).toBe(12)
    expect(a.title.font.family).toBe(PUBLICATION_FONT_FAMILY)
  })
  it('isLog produces type=log with decade-only dtick', () => {
    const a = publicationAxis({ isLog: true }) as { type: string; dtick: number }
    expect(a.type).toBe('log')
    // dtick=1 forces Plotly to label only decade ticks (10⁻¹, 10⁰,
    // 10¹, …); without this, Plotly's auto-tick algorithm prints
    // minor labels at 2× and 5× between decades, crowding the compact
    // publication canvas.
    expect(a.dtick).toBe(1)
  })
  it('non-log axis has no dtick (auto-spacing for linear axes)', () => {
    const a = publicationAxis() as Record<string, unknown>
    expect(a.dtick).toBeUndefined()
  })
  it('range is forwarded', () => {
    const a = publicationAxis({ range: [-10, 30] }) as { range: [number, number] }
    expect(a.range).toEqual([-10, 30])
  })
  it('withZeroLine switches on a black 1px zero-line', () => {
    const a = publicationAxis({ withZeroLine: true }) as Record<string, unknown>
    expect(a.zeroline).toBe(true)
    expect(a.zerolinecolor).toBe('#000000')
    expect(a.zerolinewidth).toBe(1.0)
  })
})

describe('plot-theme — publicationConfig', () => {
  it('hides the modebar', () => {
    const c = publicationConfig('jv_2d_sweep') as Record<string, unknown>
    expect(c.displayModeBar).toBe(false)
  })
  it('preserves PNG export options + filename', () => {
    const c = publicationConfig('jv_2d_sweep') as {
      toImageButtonOptions: { filename: string; format: string; scale: number }
    }
    expect(c.toImageButtonOptions.filename).toBe('jv_2d_sweep')
    expect(c.toImageButtonOptions.format).toBe('png')
    expect(c.toImageButtonOptions.scale).toBe(2)
  })
})

describe('plot-theme — publicationTraceStyle', () => {
  it('default solid filled marker (hollow=false)', () => {
    const s = publicationTraceStyle({ color: '#2B6FA3' })
    expect((s.line as { color: string }).color).toBe('#2B6FA3')
    expect((s.line as { width: number }).width).toBe(PUBLICATION_LINE_WIDTH)
    expect((s.marker as { size: number }).size).toBe(PUBLICATION_MARKER_SIZE)
    expect((s.marker as { color: string }).color).toBe('#2B6FA3')
    expect((s.marker as { symbol: string }).symbol).toBe('circle')
  })
  it('hollow markers: open symbol, transparent fill, colored outline', () => {
    const s = publicationTraceStyle({ color: '#2B6FA3', hollow: true })
    expect((s.marker as { symbol: string }).symbol).toBe('circle-open')
    expect((s.marker as { color: string }).color).toBe('rgba(0,0,0,0)')
    const ln = (s.marker as { line: { color: string; width: number } }).line
    expect(ln.color).toBe('#2B6FA3')
    expect(ln.width).toBe(1.2)
  })
  it('hollow square: square-open', () => {
    const s = publicationTraceStyle({ color: '#C44536', hollow: true, symbol: 'square' })
    expect((s.marker as { symbol: string }).symbol).toBe('square-open')
  })
  it('idempotent on already-open symbol', () => {
    const s = publicationTraceStyle({ color: '#000', hollow: true, symbol: 'diamond-open' })
    expect((s.marker as { symbol: string }).symbol).toBe('diamond-open')
  })
  it('dash forwarded to line.dash', () => {
    const s = publicationTraceStyle({ color: '#000', dash: 'dash' })
    expect((s.line as { dash: string }).dash).toBe('dash')
  })
})

describe('plot-theme — metricAnnotation', () => {
  it('returns [] when metrics is undefined or null', () => {
    expect(metricAnnotation(undefined)).toEqual([])
    expect(metricAnnotation(null)).toEqual([])
  })
  it('returns [] when voc_bracketed is undefined (legacy backend payload)', () => {
    expect(metricAnnotation({ V_oc: 0.95, J_sc: 220, FF: 0.82, PCE: 0.17 })).toEqual([])
  })
  it('renders V_oc / J_sc / FF / PCE when voc_bracketed=true', () => {
    const ann = metricAnnotation({
      V_oc: 0.951, J_sc: 220.0, FF: 0.823, PCE: 0.1722, voc_bracketed: true,
    })
    expect(ann.length).toBe(1)
    const text = (ann[0] as { text: string }).text
    expect(text).toContain('0.951 V')
    expect(text).toContain('22.00 mA cm⁻²')
    // Nature-style FF / PCE format: no space before percent sign.
    expect(text).toContain('82.3%')
    expect(text).toContain('17.22%')
    expect(text).not.toContain('82.3 %')
    expect(text).not.toContain('17.22 %')
    expect(text).not.toContain('not bracketed')
  })
  it('omits FF / PCE / V_oc digits when voc_bracketed=false', () => {
    const ann = metricAnnotation({
      V_oc: 0.0, J_sc: 220.0, FF: 0.0, PCE: 0.0, voc_bracketed: false,
    })
    const text = (ann[0] as { text: string }).text
    expect(text).toContain('not bracketed')
    expect(text).not.toContain('0.000 V')
    expect(text).not.toContain('0.0%')
    expect(text).not.toContain('0.00%')
    expect(text).toContain('22.00 mA cm⁻²')
  })
  it('annotation defaults to lower-left paper coords, transparent bg, no arrow', () => {
    const ann = metricAnnotation({
      V_oc: 0.95, J_sc: 220, FF: 0.82, PCE: 0.17, voc_bracketed: true,
    })
    const a = ann[0] as Record<string, unknown>
    expect(a.xref).toBe('paper')
    expect(a.yref).toBe('paper')
    expect(a.xanchor).toBe('left')
    expect(a.yanchor).toBe('top')
    expect(a.x).toBe(0.12)
    expect(a.y).toBe(0.34)
    expect(a.showarrow).toBe(false)
    // No heavy box / colored background — fully transparent.
    expect(a.bgcolor).toBe('rgba(255,255,255,0)')
    expect(a.borderwidth).toBe(0)
  })
  it('placement override honoured', () => {
    const ann = metricAnnotation(
      { V_oc: 0.95, J_sc: 220, FF: 0.82, PCE: 0.17, voc_bracketed: true },
      { x: 0.02, y: 0.95, xanchor: 'left', yanchor: 'top' },
    )
    const a = ann[0] as Record<string, unknown>
    expect(a.x).toBe(0.02)
    expect(a.y).toBe(0.95)
    expect(a.xanchor).toBe('left')
    expect(a.yanchor).toBe('top')
  })
  it('label option prepends a bold prefix line; default behaviour bit-identical', () => {
    const m = { V_oc: 0.95, J_sc: 220, FF: 0.82, PCE: 0.17, voc_bracketed: true }
    const unlabelled = metricAnnotation(m)
    const labelled   = metricAnnotation(m, { label: 'Reverse' })
    const tUn  = (unlabelled[0] as { text: string }).text
    const tLab = (labelled[0] as { text: string }).text
    // Labelled output starts with the bold prefix.
    expect(tLab.startsWith('<b>Reverse:</b><br>')).toBe(true)
    // Unlabelled output remains unchanged (regression — guards 2D pane).
    expect(tUn).not.toContain('<b>')
    expect(tUn).not.toContain('Forward')
    expect(tUn).not.toContain('Reverse')
    // Labelled body still contains the V_oc / J_sc / FF / PCE substrings.
    expect(tLab).toContain('0.950 V')
    expect(tLab).toContain('22.00 mA cm⁻²')
    expect(tLab).toContain('82.0%')
    expect(tLab).toContain('17.00%')
  })
  it('label option works for the not-bracketed branch too', () => {
    const m = { V_oc: 0.0, J_sc: 220, FF: 0.0, PCE: 0.0, voc_bracketed: false }
    const ann = metricAnnotation(m, { label: 'Forward' })
    const text = (ann[0] as { text: string }).text
    expect(text.startsWith('<b>Forward:</b><br>')).toBe(true)
    expect(text).toContain('not bracketed')
    expect(text).toContain('22.00 mA cm⁻²')
    expect(text).not.toContain('0.000 V')
    expect(text).not.toContain('0.0%')
  })
  it('non-finite J_sc renders the em-dash placeholder', () => {
    const ann = metricAnnotation({
      V_oc: 0.95, J_sc: Number.POSITIVE_INFINITY, FF: 0.82, PCE: 0.17, voc_bracketed: true,
    })
    const text = (ann[0] as { text: string }).text
    expect(text).toContain('—')
    // Still emits the V_oc / FF / PCE numerics; only J_sc is a dash.
    expect(text).toContain('0.950 V')
  })
})

describe('plot-theme — readPlotStyleMode / writePlotStyleMode', () => {
  it('default is engineering when dataset unset', () => {
    const el = document.createElement('div')
    expect(readPlotStyleMode(el)).toBe('engineering')
  })
  it('reads "publication" from dataset', () => {
    const el = document.createElement('div')
    el.dataset.plotStyleMode = 'publication'
    expect(readPlotStyleMode(el)).toBe('publication')
  })
  it('falls back to engineering for unknown values (defensive)', () => {
    const el = document.createElement('div')
    el.dataset.plotStyleMode = 'hello'
    expect(readPlotStyleMode(el)).toBe('engineering')
  })
  it('write round-trips through read', () => {
    const el = document.createElement('div')
    writePlotStyleMode(el, 'publication')
    expect(readPlotStyleMode(el)).toBe('publication')
    writePlotStyleMode(el, 'engineering')
    expect(readPlotStyleMode(el)).toBe('engineering')
  })
})

describe('plot-theme — palette + sizing constants (pinned defaults)', () => {
  it('publication palette muted forward / reverse colors', () => {
    expect(PUBLICATION_PALETTE.forward).toBe('#2B6FA3')
    expect(PUBLICATION_PALETTE.reverse).toBe('#C44536')
  })
  it('publication line width and marker size match pinned defaults', () => {
    expect(PUBLICATION_LINE_WIDTH).toBe(1.75)
    expect(PUBLICATION_MARKER_SIZE).toBe(5)
  })
  it('publication font family is Helvetica/Arial fallback chain', () => {
    expect(PUBLICATION_FONT_FAMILY).toContain('Helvetica')
    expect(PUBLICATION_FONT_FAMILY).toContain('Arial')
  })
})
