// Shared Plotly layout: Arial font, light grid, consistent colors, PNG export.

import type { JVMetrics } from './types'

export const PALETTE = {
  forward: '#2563eb',
  reverse: '#ea580c',
  primary: '#2563eb',
  secondary: '#0891b2',
  accent: '#db2777',
  neutral: '#475569',
}

const AXIS_BASE = {
  showline: true,
  linecolor: '#1e293b',
  linewidth: 1.5,
  mirror: true,
  ticks: 'outside' as const,
  tickcolor: '#1e293b',
  tickwidth: 1.2,
  ticklen: 6,
  showgrid: true,
  gridcolor: '#e2e8f0',
  gridwidth: 1,
  zerolinecolor: '#94a3b8',
  zerolinewidth: 1,
  tickfont: { family: 'Arial, sans-serif', size: 13, color: '#1e293b' },
}

export const AXIS_TITLE_FONT = { family: 'Arial, sans-serif', size: 14, color: '#1e293b' }

export function axisTitle(text: string): { text: string; font: typeof AXIS_TITLE_FONT; standoff: number } {
  return { text, font: AXIS_TITLE_FONT, standoff: 12 }
}

export function baseLayout(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    font: { family: 'Arial, sans-serif', size: 13, color: '#1e293b' },
    paper_bgcolor: '#ffffff',
    plot_bgcolor: '#ffffff',
    margin: { t: 30, r: 40, b: 60, l: 70 },
    xaxis: { ...AXIS_BASE },
    yaxis: { ...AXIS_BASE },
    legend: {
      font: { family: 'Arial, sans-serif', size: 12, color: '#1e293b' },
      bgcolor: 'rgba(255,255,255,0.85)',
      bordercolor: '#cbd5e1',
      borderwidth: 1,
    },
    hovermode: 'closest',
    hoverlabel: {
      font: { family: 'Arial, sans-serif', size: 12 },
      bgcolor: '#ffffff',
      bordercolor: '#cbd5e1',
    },
    ...overrides,
  }
}

export function plotConfig(filename = 'plot'): Record<string, unknown> {
  return {
    responsive: true,
    displaylogo: false,
    toImageButtonOptions: {
      format: 'png',
      filename,
      height: 600,
      width: 900,
      scale: 2,
    },
    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
  }
}

export const LINE = {
  width: 2.5,
}

export const MARKER = {
  size: 7,
  line: { width: 1, color: '#ffffff' },
}

// ───────────────────────────────────────────────────────────────────────────
// Publication theme — Nature-style single-panel scientific figure styling.
// Engineering exports above are unchanged. Publication helpers below are
// additive: a renderer opts in by reading ``readPlotStyleMode(el)`` and
// branching its ``Plotly.newPlot`` arguments through these helpers instead
// of ``baseLayout`` / ``plotConfig`` / ``LINE`` / ``MARKER``.
// ───────────────────────────────────────────────────────────────────────────

export type PlotStyleMode = 'engineering' | 'publication'

export const PUBLICATION_PALETTE = {
  forward:   '#2B6FA3',  // muted scientific blue (Nature-family)
  reverse:   '#C44536',  // muted scientific red
  primary:   '#2B6FA3',
  secondary: '#5b6770',
  accent:    '#374151',
  neutral:   '#000000',
} as const

export const PUBLICATION_FONT_FAMILY =
  'Helvetica, Arial, "Liberation Sans", sans-serif'

export const PUBLICATION_LINE_WIDTH = 1.75
export const PUBLICATION_MARKER_SIZE = 5

const PUBLICATION_AXIS_BASE = {
  showline: true,
  linecolor: '#000000',
  linewidth: 1.0,
  mirror: true,
  ticks: 'outside' as const,
  tickcolor: '#000000',
  tickwidth: 0.8,
  ticklen: 4,
  showgrid: false,
  zeroline: false,
  tickfont: { family: PUBLICATION_FONT_FAMILY, size: 10, color: '#000000' },
}

const PUBLICATION_AXIS_TITLE_FONT = {
  family: PUBLICATION_FONT_FAMILY,
  size: 12,
  color: '#000000',
}

export function publicationAxisTitle(text: string): { text: string; font: typeof PUBLICATION_AXIS_TITLE_FONT; standoff: number } {
  return { text, font: PUBLICATION_AXIS_TITLE_FONT, standoff: 8 }
}

export interface PublicationAxisOpts {
  title?: string
  isLog?: boolean
  range?: [number, number]
  withZeroLine?: boolean
}

export function publicationAxis(opts: PublicationAxisOpts = {}): Record<string, unknown> {
  const a: Record<string, unknown> = { ...PUBLICATION_AXIS_BASE }
  if (opts.title !== undefined) a.title = publicationAxisTitle(opts.title)
  if (opts.isLog) a.type = 'log'
  if (opts.range) a.range = opts.range
  if (opts.withZeroLine) {
    a.zeroline = true
    a.zerolinecolor = '#000000'
    a.zerolinewidth = 1.0
  }
  return a
}

export function publicationLayout(
  overrides: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    font: { family: PUBLICATION_FONT_FAMILY, size: 11, color: '#000000' },
    paper_bgcolor: '#ffffff',
    plot_bgcolor:  '#ffffff',
    margin: { t: 18, r: 18, b: 48, l: 58 },
    xaxis: publicationAxis(),
    yaxis: publicationAxis(),
    legend: {
      font: { family: PUBLICATION_FONT_FAMILY, size: 9, color: '#000000' },
      bgcolor: 'rgba(255,255,255,0)',
      bordercolor: 'rgba(0,0,0,0)',
      borderwidth: 0,
      // Upper-left default — Nature single-panel J-V curves usually have
      // their open-circuit / saturation activity near the upper-right
      // quadrant, so legends drift toward the lower- or upper-left in
      // the printed figures we mimic here.
      x: 0.02, y: 0.98, xanchor: 'left' as const, yanchor: 'top' as const,
    },
    showlegend: true,
    hovermode: 'closest',
    hoverlabel: {
      font: { family: PUBLICATION_FONT_FAMILY, size: 10 },
      bgcolor: '#ffffff',
      bordercolor: '#000000',
    },
    ...overrides,
  }
}

export function publicationConfig(filename = 'plot'): Record<string, unknown> {
  return {
    responsive: true,
    displayModeBar: false,
    displaylogo: false,
    toImageButtonOptions: {
      format: 'png',
      filename,
      height: 600,
      width: 900,
      scale: 2,
    },
  }
}

export interface PublicationTraceStyleOpts {
  color: string
  hollow?: boolean
  dash?: 'solid' | 'dot' | 'dash' | 'longdash' | 'dashdot' | 'longdashdot'
  symbol?: string
}

export function publicationTraceStyle(opts: PublicationTraceStyleOpts): {
  line: Record<string, unknown>
  marker: Record<string, unknown>
} {
  const sym = opts.symbol ?? 'circle'
  const symbol = opts.hollow && !sym.endsWith('-open') ? `${sym}-open` : sym
  const marker: Record<string, unknown> = {
    size: PUBLICATION_MARKER_SIZE,
    symbol,
    line: { width: 1.2, color: opts.color },
  }
  marker.color = opts.hollow ? 'rgba(0,0,0,0)' : opts.color
  const line: Record<string, unknown> = {
    color: opts.color,
    width: PUBLICATION_LINE_WIDTH,
  }
  if (opts.dash) line.dash = opts.dash
  return { line, marker }
}

export interface MetricAnnotationOpts {
  x?: number
  y?: number
  xanchor?: 'left' | 'center' | 'right'
  yanchor?: 'top' | 'middle' | 'bottom'
  /** Optional bold prefix line ("Forward:" / "Reverse:" etc.) inserted
   *  above the metric block. When omitted the annotation text is
   *  byte-identical to the legacy unlabelled output (preserves 2D pane
   *  output in this commit). */
  label?: string
}

// Builds a single Plotly annotation entry summarising V_oc / J_sc / FF / PCE
// in publication mode. Returns ``[]`` when metrics are missing OR when
// ``voc_bracketed`` is undefined (legacy backend) — never invents fake
// V_oc / FF / PCE values. ``voc_bracketed === false`` shows
// ``V_oc: not bracketed`` and J_sc only.
export function metricAnnotation(
  metrics: JVMetrics | undefined | null,
  opts: MetricAnnotationOpts = {},
): Record<string, unknown>[] {
  if (!metrics) return []
  if (metrics.voc_bracketed === undefined) return []
  // Default placement is the lower-left / mid-left of the panel — Nature
  // J-V figures conventionally seat the V_oc / J_sc / FF / PCE legend
  // in this quadrant since the curve sweeps from the upper-left
  // (V=0, J~J_sc) down to the lower-right (V~V_oc, J=0), leaving the
  // lower-left quadrant empty.
  const x       = opts.x       ?? 0.12
  const y       = opts.y       ?? 0.34
  const xanchor = opts.xanchor ?? 'left'
  const yanchor = opts.yanchor ?? 'top'
  const J_sc_mA = metrics.J_sc / 10
  const jsc_finite = Number.isFinite(metrics.J_sc)
  const jsc_str = jsc_finite ? `${J_sc_mA.toFixed(2)} mA cm⁻²` : '—'
  let body: string
  if (metrics.voc_bracketed === true) {
    const voc = `${metrics.V_oc.toFixed(3)} V`
    const ff  = `${(metrics.FF * 100).toFixed(1)}%`
    const pce = `${(metrics.PCE * 100).toFixed(2)}%`
    body =
      `V<sub>oc</sub>: ${voc}<br>` +
      `J<sub>sc</sub>: ${jsc_str}<br>` +
      `FF: ${ff}<br>` +
      `PCE: ${pce}`
  } else {
    body =
      `V<sub>oc</sub>: not bracketed<br>` +
      `J<sub>sc</sub>: ${jsc_str}`
  }
  // Prefix the label as a bold line above the body when supplied.
  // Unlabelled annotations remain byte-identical to pre-1D-fix output —
  // 2D pane (which never passes a label) is unaffected.
  const text = opts.label !== undefined ? `<b>${opts.label}:</b><br>${body}` : body
  return [{
    x, y, xref: 'paper', yref: 'paper',
    xanchor, yanchor, showarrow: false,
    text,
    align: 'left',
    // Fully transparent — the panel background is already white, so an
    // opaque box would only add a visible rectangle. Border is also
    // suppressed (heavy boxes are non-Nature).
    bgcolor: 'rgba(255,255,255,0)',
    bordercolor: 'rgba(0,0,0,0)',
    borderwidth: 0,
    font: { family: PUBLICATION_FONT_FAMILY, size: 10, color: '#000000' },
  }]
}

// State helpers: persist plot style mode on the stable container so the
// renderer's internal Plotly.purge + child rebuild does not lose the
// user's choice. Default ``'engineering'`` preserves existing behaviour
// when the dataset is unset.

export function readPlotStyleMode(el: HTMLElement): PlotStyleMode {
  const v = el.dataset.plotStyleMode
  return v === 'publication' ? 'publication' : 'engineering'
}

export function writePlotStyleMode(el: HTMLElement, mode: PlotStyleMode): void {
  el.dataset.plotStyleMode = mode
}
