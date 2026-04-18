import Plotly from 'plotly.js-basic-dist-min'
import type { Workspace } from '../types'
import { findRun } from '../state'
import { baseLayout, plotConfig, PALETTE, LINE, MARKER, axisTitle } from '../../plot-theme'
import type {
  JVResult,
  ISResult,
  DegResult,
  TPVResult,
  CurrentDecompResult,
  SpatialProfileResult,
  DarkJVResult,
  SunsVocResult,
  VocTResult,
  EQEResult,
  MottSchottkyResult,
} from '../../types'

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
        case 'dark_jv':
          renderDarkJV(plotEl, run.result.data)
          return
        case 'suns_voc':
          renderSunsVoc(plotEl, run.result.data)
          return
        case 'voc_t':
          renderVocT(plotEl, run.result.data)
          return
        case 'eqe':
          renderEQE(plotEl, run.result.data)
          return
        case 'mott_schottky':
          renderMottSchottky(plotEl, run.result.data)
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

function renderVocT(el: HTMLElement, r: VocTResult): void {
  Plotly.purge(el)
  el.innerHTML = ''
  // Linear fit line across the sweep domain — anchored by (T_min, fit(T_min))
  // and (T_max, fit(T_max)) so it extends cleanly across the plotted range.
  const T_min = Math.min(...r.T_arr)
  const T_max = Math.max(...r.T_arr)
  const fit_x = [T_min, T_max]
  const fit_y = fit_x.map(T => r.slope * T + r.intercept_0K)
  const slope_mV_per_K = (r.slope * 1e3).toFixed(2)

  Plotly.newPlot(
    el,
    [
      {
        x: r.T_arr, y: r.V_oc_arr, name: 'V<sub>oc</sub>(T)',
        mode: 'lines+markers',
        line: { color: PALETTE.forward, width: LINE.width },
        marker: { ...MARKER, color: PALETTE.forward },
      },
      {
        x: fit_x, y: fit_y, name: `linear fit (${slope_mV_per_K} mV/K)`,
        mode: 'lines',
        line: { color: PALETTE.reverse, width: LINE.width, dash: 'dash' },
      },
    ],
    baseLayout({
      xaxis: { ...(baseLayout().xaxis as object), title: axisTitle('Temperature, <i>T</i> (K)') },
      yaxis: { ...(baseLayout().yaxis as object), title: axisTitle('Open-circuit voltage, <i>V</i><sub>oc</sub> (V)') },
      annotations: [
        {
          x: 0.98, y: 0.05, xref: 'paper', yref: 'paper',
          xanchor: 'right', yanchor: 'bottom', showarrow: false,
          text: `E<sub>A</sub> \u2248 ${r.E_A_eV.toFixed(3)} eV &nbsp; dV<sub>oc</sub>/dT = ${slope_mV_per_K} mV/K &nbsp; R\u00B2 = ${r.R_squared.toFixed(3)}`,
          font: { size: 12 },
        },
      ],
    }),
    plotConfig('voc_t'),
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

// ── Dark J-V ────────────────────────────────────────────────────────────────

function renderDarkJV(el: HTMLElement, r: DarkJVResult): void {
  Plotly.purge(el)
  el.innerHTML = ''
  // Dark forward J-V: conventional sign has J > 0 for forward bias (injection).
  // The simulator returns A/m^2; display mA/cm^2 on a log axis.
  const absJ = r.J.map(j => Math.max(Math.abs(j) / 10, 1e-9))

  // Highlight the fit window as a translucent band
  const shapes: Record<string, unknown>[] = [
    {
      type: 'rect', xref: 'x', yref: 'paper',
      x0: r.V_fit_lo, x1: r.V_fit_hi,
      y0: 0, y1: 1,
      fillcolor: 'rgba(99, 102, 241, 0.10)',
      line: { width: 0 },
    },
  ]

  Plotly.newPlot(
    el,
    [
      {
        x: r.V, y: absJ, name: '|J|',
        mode: 'lines+markers',
        line: { color: PALETTE.forward, width: LINE.width },
        marker: { ...MARKER, color: PALETTE.forward },
      },
    ],
    baseLayout({
      xaxis: { ...(baseLayout().xaxis as object), title: axisTitle('Applied bias, <i>V</i> (V)') },
      yaxis: { ...(baseLayout().yaxis as object), title: axisTitle('|J| (mA\u00b7cm\u207B\u00b2)'), type: 'log' },
      shapes,
      annotations: [
        {
          x: 0.02, y: 0.98, xref: 'paper', yref: 'paper',
          xanchor: 'left', yanchor: 'top', showarrow: false,
          text: `n = ${r.n_ideality.toFixed(2)} &nbsp; J<sub>0</sub> = ${r.J_0.toExponential(2)} A\u00b7m\u207B\u00b2 &nbsp; fit: [${r.V_fit_lo.toFixed(2)}, ${r.V_fit_hi.toFixed(2)}] V`,
          font: { size: 12 },
        },
      ],
    }),
    plotConfig('dark_jv'),
  )
}

// ── Suns–V_oc ───────────────────────────────────────────────────────────────

function renderSunsVoc(el: HTMLElement, r: SunsVocResult): void {
  Plotly.purge(el)
  el.innerHTML = ''

  // Left subplot: V_oc vs log(suns).  Right subplot: pseudo J-V.
  const pseudo_J_mA = r.J_pseudo_J.map(j => j / 10)

  const axBase = baseLayout().xaxis as object
  const ayBase = baseLayout().yaxis as object

  Plotly.newPlot(
    el,
    [
      {
        x: r.suns, y: r.V_oc, name: 'V<sub>oc</sub>(suns)',
        mode: 'lines+markers',
        line: { color: PALETTE.forward, width: LINE.width },
        marker: { ...MARKER, color: PALETTE.forward },
        xaxis: 'x', yaxis: 'y',
      },
      {
        x: r.J_pseudo_V, y: pseudo_J_mA, name: 'pseudo J\u2013V',
        mode: 'lines+markers',
        line: { color: PALETTE.reverse, width: LINE.width },
        marker: { ...MARKER, color: PALETTE.reverse, symbol: 'square' },
        xaxis: 'x2', yaxis: 'y2',
      },
    ],
    {
      ...baseLayout(),
      grid: { rows: 1, columns: 2, pattern: 'independent' },
      xaxis: { ...axBase, title: axisTitle('Suns'), type: 'log', anchor: 'y' },
      yaxis: { ...ayBase, title: axisTitle('V<sub>oc</sub> (V)'), anchor: 'x' },
      xaxis2: { ...axBase, title: axisTitle('V (V)'), anchor: 'y2' },
      yaxis2: { ...ayBase, title: axisTitle('J (mA\u00b7cm\u207B\u00b2)'), anchor: 'x2' },
      annotations: [
        {
          x: 0.98, y: 0.02, xref: 'paper', yref: 'paper',
          xanchor: 'right', yanchor: 'bottom', showarrow: false,
          text: `pseudo FF = ${(r.pseudo_FF * 100).toFixed(1)} %`,
          font: { size: 12 },
        },
      ],
    },
    plotConfig('suns_voc'),
  )
}

// ── EQE / IPCE ──────────────────────────────────────────────────────────────

function renderEQE(el: HTMLElement, r: EQEResult): void {
  Plotly.purge(el)
  el.innerHTML = ''
  const eqePct = r.EQE.map(x => x * 100)
  const mAcm2 = r.J_sc_integrated / 10
  Plotly.newPlot(
    el,
    [
      {
        x: r.wavelengths_nm, y: eqePct, name: 'EQE(\u03bb)',
        mode: 'lines+markers',
        line: { color: PALETTE.forward, width: LINE.width },
        marker: { ...MARKER, color: PALETTE.forward },
      },
    ],
    baseLayout({
      xaxis: { ...(baseLayout().xaxis as object), title: axisTitle('Wavelength, <i>\u03bb</i> (nm)') },
      yaxis: { ...(baseLayout().yaxis as object), title: axisTitle('EQE (%)'), range: [0, 100] },
      annotations: [
        {
          x: 0.98, y: 0.95, xref: 'paper', yref: 'paper',
          xanchor: 'right', yanchor: 'top', showarrow: false,
          text: `J<sub>sc</sub>(AM1.5G) = ${mAcm2.toFixed(2)} mA\u00b7cm\u207B\u00b2`,
          font: { size: 12 },
        },
      ],
    }),
    plotConfig('eqe'),
  )
}

// ── Mott–Schottky (C–V) ─────────────────────────────────────────────────────

function renderMottSchottky(el: HTMLElement, r: MottSchottkyResult): void {
  Plotly.purge(el)
  el.innerHTML = ''

  const shapes: Record<string, unknown>[] = [
    {
      type: 'rect', xref: 'x', yref: 'paper',
      x0: r.V_fit_lo, x1: r.V_fit_hi,
      y0: 0, y1: 1,
      fillcolor: 'rgba(99, 102, 241, 0.10)',
      line: { width: 0 },
    },
  ]

  Plotly.newPlot(
    el,
    [
      {
        x: r.V, y: r.one_over_C2, name: '1/C\u00b2',
        mode: 'lines+markers',
        line: { color: PALETTE.forward, width: LINE.width },
        marker: { ...MARKER, color: PALETTE.forward },
      },
    ],
    baseLayout({
      xaxis: { ...(baseLayout().xaxis as object), title: axisTitle('Applied bias, <i>V</i> (V)') },
      yaxis: { ...(baseLayout().yaxis as object), title: axisTitle('1/C\u00b2 (m\u2074\u00b7F\u207B\u00b2)') },
      shapes,
      annotations: [
        {
          x: 0.02, y: 0.98, xref: 'paper', yref: 'paper',
          xanchor: 'left', yanchor: 'top', showarrow: false,
          text: `V<sub>bi</sub> = ${r.V_bi_fit.toFixed(3)} V &nbsp; N<sub>eff</sub> = ${r.N_eff_fit.toExponential(2)} m\u207B\u00b3 &nbsp; f = ${r.frequency.toExponential(1)} Hz`,
          font: { size: 12 },
        },
      ],
    }),
    plotConfig('mott_schottky'),
  )
}
