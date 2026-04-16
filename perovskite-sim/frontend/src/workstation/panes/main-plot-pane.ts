import Plotly from 'plotly.js-basic-dist-min'
import type { Workspace } from '../types'
import { findRun } from '../state'
import { baseLayout, plotConfig, PALETTE, LINE, MARKER, axisTitle } from '../../plot-theme'
import type { JVResult, ISResult, DegResult, TPVResult, CurrentDecompResult, SpatialProfileResult } from '../../types'

export interface MainPlotHandle {
  update(ws: Workspace): void
}

export function mountMainPlotPane(container: HTMLElement): MainPlotHandle {
  container.innerHTML = `
    <div class="main-plot-pane">
      <div class="main-plot-header" id="mpp-header">(no active run)</div>
      <div id="mpp-plot" class="plot-container"></div>
    </div>`

  const header = container.querySelector<HTMLDivElement>('#mpp-header')!
  const plotEl = container.querySelector<HTMLDivElement>('#mpp-plot')!

  function clear(msg: string): void {
    header.textContent = msg
    Plotly.purge(plotEl)
    plotEl.innerHTML = '<div class="plot-empty">Run an experiment to see results here.</div>'
  }

  clear('(no active run)')

  return {
    update(ws: Workspace) {
      if (!ws.activeRunId || !ws.activeDeviceId || !ws.activeExperimentId) {
        clear('(no active run)')
        return
      }
      const run = findRun(ws, ws.activeDeviceId, ws.activeExperimentId, ws.activeRunId)
      if (!run) {
        clear('(run not found)')
        return
      }
      header.textContent = `${run.activePhysics}  ·  ${new Date(run.timestamp).toLocaleString()}`
      switch (run.result.kind) {
        case 'jv':
          renderJV(plotEl, run.result.data)
          return
        case 'impedance':
          renderImpedance(plotEl, run.result.data)
          return
        case 'degradation':
          renderDegradation(plotEl, run.result.data)
          return
        case 'tpv':
          renderTPV(plotEl, run.result.data)
          return
        case 'current_decomp':
          renderCurrentDecomp(plotEl, run.result.data)
          return
        case 'spatial':
          renderSpatialProfiles(plotEl, run.result.data)
          return
      }
    },
  }
}

function renderJV(el: HTMLElement, r: JVResult): void {
  Plotly.purge(el)
  el.innerHTML = ''
  const J_fwd_mA = r.J_fwd.map(j => j / 10)
  const J_rev_mA = r.J_rev.map(j => j / 10)
  const V_rev_sorted = [...r.V_rev].reverse()
  const J_rev_sorted = [...J_rev_mA].reverse()
  Plotly.newPlot(
    el,
    [
      {
        x: r.V_fwd, y: J_fwd_mA, name: 'Forward',
        mode: 'lines+markers',
        line: { color: PALETTE.forward, width: LINE.width },
        marker: { ...MARKER, color: PALETTE.forward },
      },
      {
        x: V_rev_sorted, y: J_rev_sorted, name: 'Reverse',
        mode: 'lines+markers',
        line: { color: PALETTE.reverse, width: LINE.width, dash: 'dash' },
        marker: { ...MARKER, color: PALETTE.reverse, symbol: 'square' },
      },
    ],
    baseLayout({
      xaxis: { ...(baseLayout().xaxis as object), title: axisTitle('Applied bias, <i>V</i> (V)') },
      yaxis: { ...(baseLayout().yaxis as object), title: axisTitle('Current density, <i>J</i> (mA·cm⁻²)') },
    }),
    plotConfig('jv_sweep'),
  )
}

function renderImpedance(el: HTMLElement, r: ISResult): void {
  Plotly.purge(el)
  el.innerHTML = ''
  const minusImag = r.Z_imag.map(x => -x)
  Plotly.newPlot(
    el,
    [
      {
        x: r.Z_real, y: minusImag, name: 'Z',
        mode: 'lines+markers',
        line: { color: PALETTE.forward, width: LINE.width },
        marker: { ...MARKER, color: PALETTE.forward },
      },
    ],
    baseLayout({
      xaxis: { ...(baseLayout().xaxis as object), title: axisTitle('Re(Z)  (Ω·m²)') },
      yaxis: { ...(baseLayout().yaxis as object), title: axisTitle('−Im(Z)  (Ω·m²)'), scaleanchor: 'x' },
    }),
    plotConfig('impedance'),
  )
}

function renderDegradation(el: HTMLElement, r: DegResult): void {
  Plotly.purge(el)
  el.innerHTML = ''
  const pce0 = r.PCE[0] || 1
  const normalized = r.PCE.map(p => p / pce0)
  Plotly.newPlot(
    el,
    [
      {
        x: r.times, y: normalized, name: 'PCE / PCE₀',
        mode: 'lines+markers',
        line: { color: PALETTE.forward, width: LINE.width },
        marker: { ...MARKER, color: PALETTE.forward },
      },
    ],
    baseLayout({
      xaxis: { ...(baseLayout().xaxis as object), title: axisTitle('Time (s)') },
      yaxis: { ...(baseLayout().yaxis as object), title: axisTitle('Normalised PCE') },
    }),
    plotConfig('degradation'),
  )
}

function renderTPV(el: HTMLElement, r: TPVResult): void {
  Plotly.purge(el)
  el.innerHTML = ''
  // Convert time to microseconds for readability
  const t_us = r.t.map(t => t * 1e6)
  // Convert voltage to mV perturbation from V_oc
  const dV_mV = r.V.map(v => (v - r.V_oc) * 1e3)

  Plotly.newPlot(
    el,
    [
      {
        x: t_us, y: dV_mV, name: `\u0394V  (\u03C4=${(r.tau * 1e6).toFixed(1)} \u00B5s)`,
        mode: 'lines',
        line: { color: PALETTE.forward, width: LINE.width },
      },
    ],
    baseLayout({
      xaxis: { ...(baseLayout().xaxis as object), title: axisTitle('Time (\u00B5s)') },
      yaxis: { ...(baseLayout().yaxis as object), title: axisTitle('\u0394V (mV)') },
      annotations: [
        {
          x: 0.98, y: 0.95, xref: 'paper', yref: 'paper',
          xanchor: 'right', yanchor: 'top', showarrow: false,
          text: `V<sub>oc</sub> = ${r.V_oc.toFixed(3)} V &nbsp; \u03C4 = ${(r.tau * 1e6).toFixed(1)} \u00B5s &nbsp; \u0394V<sub>0</sub> = ${(r.delta_V0 * 1e3).toFixed(2)} mV`,
          font: { size: 12 },
        },
      ],
    }),
    plotConfig('tpv'),
  )
}

// ── Current Decomposition ───────────────────────────────────────────────────

const DECOMP_COLORS = {
  Jn: '#2563eb',      // blue
  Jp: '#ea580c',      // orange
  Jion: '#eab308',    // yellow
  Jdisp: '#6366f1',   // indigo (dashed)
  Jtotal: '#16a34a',  // green
}

function renderCurrentDecomp(el: HTMLElement, r: CurrentDecompResult): void {
  Plotly.purge(el)
  el.innerHTML = ''

  // Convert A/m² → mA/cm² with sign flip to physics convention
  // (photocurrent negative, injection positive — matches Driftfusion / literature)
  const toMA = (arr: number[]) => arr.map(j => -j / 10)

  const traces = [
    {
      x: r.V_fwd, y: toMA(r.Jn_fwd), name: 'J<sub>n</sub>',
      mode: 'lines', line: { color: DECOMP_COLORS.Jn, width: LINE.width },
    },
    {
      x: r.V_fwd, y: toMA(r.Jp_fwd), name: 'J<sub>p</sub>',
      mode: 'lines', line: { color: DECOMP_COLORS.Jp, width: LINE.width },
    },
    {
      x: r.V_fwd, y: toMA(r.Jion_fwd), name: 'J<sub>ion</sub>',
      mode: 'lines', line: { color: DECOMP_COLORS.Jion, width: LINE.width },
    },
    {
      x: r.V_fwd, y: toMA(r.Jdisp_fwd), name: 'J<sub>disp</sub>',
      mode: 'lines', line: { color: DECOMP_COLORS.Jdisp, width: LINE.width, dash: 'dash' },
    },
    {
      x: r.V_fwd, y: toMA(r.Jtotal_fwd), name: 'J<sub>total</sub>',
      mode: 'lines', line: { color: DECOMP_COLORS.Jtotal, width: LINE.width + 0.5 },
    },
  ]

  Plotly.newPlot(
    el,
    traces,
    baseLayout({
      xaxis: { ...(baseLayout().xaxis as object), title: axisTitle('Applied bias, <i>V</i> (V)') },
      yaxis: { ...(baseLayout().yaxis as object), title: axisTitle('Current density (mA\u00B7cm\u207B\u00B2)') },
      legend: { x: 0.02, y: 0.05, xanchor: 'left', yanchor: 'bottom', ...(baseLayout().legend as object) },
    }),
    plotConfig('current_decomposition'),
  )
}

// ── Spatial Profiles ────────────────────────────────────────────────────────

// Voltage-indexed color palette for multi-curve spatial plots
const SPATIAL_COLORS = [
  '#94a3b8', '#2563eb', '#eab308', '#db2777', '#16a34a',
  '#ea580c', '#6366f1', '#0891b2', '#dc2626', '#4f46e5',
]

function renderSpatialProfiles(el: HTMLElement, r: SpatialProfileResult): void {
  Plotly.purge(el)
  el.innerHTML = ''

  const snaps = r.snapshots_fwd
  if (!snaps || snaps.length === 0) {
    el.innerHTML = '<div class="plot-empty">No spatial snapshots available.</div>'
    return
  }

  // Build three vertically-stacked subplots: potential, carrier densities, charge density
  const traces: Record<string, unknown>[] = []

  snaps.forEach((snap, i) => {
    const color = SPATIAL_COLORS[i % SPATIAL_COLORS.length]
    const label = `${snap.V_app.toFixed(2)} V`
    const showlegend = true

    // Potential φ(x) — top subplot (yaxis)
    traces.push({
      x: snap.x, y: snap.phi, name: label,
      mode: 'lines', line: { color, width: 1.8 },
      xaxis: 'x', yaxis: 'y',
      legendgroup: label, showlegend,
    })

    // Carrier densities n(x), p(x) — middle subplot (yaxis2), log scale
    traces.push({
      x: snap.x, y: snap.n.map(v => Math.max(v, 1e-10)), name: `n @ ${label}`,
      mode: 'lines', line: { color, width: 1.5 },
      xaxis: 'x2', yaxis: 'y2',
      legendgroup: label, showlegend: false,
    })
    traces.push({
      x: snap.x, y: snap.p.map(v => Math.max(v, 1e-10)), name: `p @ ${label}`,
      mode: 'lines', line: { color, width: 1.5, dash: 'dash' },
      xaxis: 'x2', yaxis: 'y2',
      legendgroup: label, showlegend: false,
    })

    // Electric field E(x) — bottom subplot (yaxis3)
    // E has N-1 faces; use midpoints of x for plotting
    const x_mid = snap.x.slice(0, -1).map((xi, j) => (xi + snap.x[j + 1]) / 2)
    traces.push({
      x: x_mid, y: snap.E.map(e => e * 1e-4), name: `E @ ${label}`,
      mode: 'lines', line: { color, width: 1.5 },
      xaxis: 'x3', yaxis: 'y3',
      legendgroup: label, showlegend: false,
    })
  })

  const axBase = baseLayout().xaxis as object
  const ayBase = baseLayout().yaxis as object

  Plotly.newPlot(
    el,
    traces,
    {
      ...baseLayout(),
      margin: { t: 20, r: 40, b: 50, l: 70 },
      grid: { rows: 3, columns: 1, subplots: [['xy'], ['x2y2'], ['x3y3']], roworder: 'top to bottom' },
      xaxis: { ...axBase, title: '', showticklabels: false, anchor: 'y' },
      yaxis: { ...ayBase, title: axisTitle('\u03C6 (V)'), anchor: 'x' },
      xaxis2: { ...axBase, title: '', showticklabels: false, anchor: 'y2' },
      yaxis2: { ...ayBase, title: axisTitle('n, p (m\u207B\u00B3)'), type: 'log', anchor: 'x2' },
      xaxis3: { ...axBase, title: axisTitle('Position (nm)'), anchor: 'y3' },
      yaxis3: { ...ayBase, title: axisTitle('E (10\u2074 V/m)'), anchor: 'x3' },
      legend: { x: 1.02, y: 1, xanchor: 'left', yanchor: 'top', ...(baseLayout().legend as object) },
      height: 700,
    },
    plotConfig('spatial_profiles'),
  )
}
