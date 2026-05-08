import Plotly from 'plotly.js-basic-dist-min'
import type { Workspace } from '../types'
import { findRun } from '../state'
import {
  baseLayout, plotConfig, PALETTE, LINE, MARKER, axisTitle,
  // Publication-theme additions (Nature-style single-panel mode).
  publicationLayout, publicationAxis, publicationConfig,
  publicationTraceStyle, metricAnnotation,
  readPlotStyleMode, PUBLICATION_PALETTE,
} from '../../plot-theme'
import { metricCard } from '../../ui-helpers'
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
  ELResult,
  MottSchottkyResult,
  JV2DResult,
  VocGrainSweepResult,
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
        case 'el':
          renderEL(plotEl, run.result.data)
          return
        case 'mott_schottky':
          renderMottSchottky(plotEl, run.result.data)
          return
        case 'jv_2d':
          renderJV2D(plotEl, run.result.data)
          return
        case 'voc_grain_sweep':
          renderVocGrainSweep(plotEl, run.result.data)
          return
      }
    },
  }
}

// ── Stage-B V_oc(L_g) sweep (Phase 6) ───────────────────────────────────────

function renderVocGrainSweep(el: HTMLElement, r: VocGrainSweepResult): void {
  Plotly.purge(el)
  el.innerHTML = ''
  const V_oc_mV = r.V_oc_V.map(v => v * 1e3)
  const J_sc_mA = r.J_sc_Am2.map(j => j / 10)
  const ann = r.grain_sizes_nm
    .map((L, i) =>
      `L_g=${L.toFixed(0)} nm: V<sub>oc</sub>=${V_oc_mV[i].toFixed(1)} mV, ` +
        `J<sub>sc</sub>=${J_sc_mA[i].toFixed(2)} mA·cm⁻², FF=${r.FF[i].toFixed(3)}`,
    )
    .join('<br>')
  Plotly.newPlot(
    el,
    [
      {
        x: r.grain_sizes_nm,
        y: V_oc_mV,
        name: 'V_oc(L_g)',
        mode: 'lines+markers',
        line: { color: PALETTE.forward, width: LINE.width },
        marker: { ...MARKER, color: PALETTE.forward },
      },
    ],
    baseLayout({
      xaxis: {
        ...(baseLayout().xaxis as object),
        type: 'log',
        title: axisTitle('Grain size, <i>L<sub>g</sub></i> (nm)'),
      },
      yaxis: {
        ...(baseLayout().yaxis as object),
        title: axisTitle('Open-circuit voltage, <i>V<sub>oc</sub></i> (mV)'),
      },
      annotations: [
        {
          x: 0.02, y: 0.05, xref: 'paper', yref: 'paper',
          xanchor: 'left', yanchor: 'bottom', showarrow: false,
          text: ann,
          font: { size: 10 },
          align: 'left',
        },
      ],
    }),
    plotConfig('voc_grain_sweep'),
  )
}

// ── Stage-A 2D J-V (Phase 6) ────────────────────────────────────────────────

type JV2DRangeMode = 'operational' | 'full'

function _jv2dReadMode(el: HTMLElement): JV2DRangeMode {
  // Toggle state lives on the STABLE container ``el`` passed into
  // ``renderJV2D`` — survives every internal DOM rebuild (Plotly.purge,
  // child removal, child re-append). Default ``'operational'`` so users
  // see the photovoltaic operating window first; the diode tail past
  // V_bi only appears on opt-in.
  const v = el.dataset.jv2dMode
  return v === 'full' ? 'full' : 'operational'
}

function _jv2dComputeYRange(
  mode: JV2DRangeMode,
  metrics: JV2DResult['metrics'],
): [number, number] | undefined {
  // Returns the clipped ``[ymin, ymax]`` for ``yaxis.range`` only when
  // every precondition is met; otherwise ``undefined`` so the caller
  // omits ``yaxis.range`` entirely and lets Plotly autorange (the
  // explicit fallback path documented in the Layer 4 spec).
  if (mode !== 'operational') return undefined
  if (!metrics) return undefined
  if (metrics.voc_bracketed !== true) return undefined
  if (!Number.isFinite(metrics.J_sc) || metrics.J_sc <= 0) return undefined
  // Backend J_sc is in A/m² (J_sc-positive convention). Display is
  // mA/cm² post-divide-by-10 — same convention as the J trace below.
  const J_sc_mA = metrics.J_sc / 10
  return [-0.5 * J_sc_mA, 1.5 * J_sc_mA]
}

// Publication-mode operational ranges — TIGHTER than the engineering
// operational window. Designed for paper-figure aesthetics: small
// negative padding below J=0, light headroom above J_sc, and a small
// negative x-margin so the V=0 reference line is visible without
// inventing negative-voltage data. Both helpers return ``undefined``
// outside operational mode (Full sweep stays autoranged).
function _jv2dComputeYRangePublication(
  mode: JV2DRangeMode,
  metrics: JV2DResult['metrics'],
): [number, number] | undefined {
  if (mode !== 'operational') return undefined
  if (!metrics) return undefined
  if (metrics.voc_bracketed !== true) return undefined
  if (!Number.isFinite(metrics.J_sc) || metrics.J_sc <= 0) return undefined
  const J_sc_mA = metrics.J_sc / 10
  return [-0.15 * J_sc_mA, 1.12 * J_sc_mA]
}

function _jv2dComputeXRangePublication(
  mode: JV2DRangeMode,
  V: number[],
  metrics: JV2DResult['metrics'],
): [number, number] | undefined {
  if (mode !== 'operational') return undefined
  if (V.length === 0) return undefined
  const minV = Math.min(...V)
  const maxV = Math.max(...V)
  // Add a small visual negative margin only when the sweep starts at
  // V=0 — never shift the visible window left of an existing negative
  // sweep value.
  const xmin = minV >= 0 ? -0.05 : minV
  // Cap upper bound at V_oc + 0.18 V when V_oc is bracketed; this
  // trims the diode-tail past the open-circuit point that typically
  // dominates Engineering "Full sweep" plots. Without a valid V_oc,
  // fall back to ``maxV + 0.05`` (small headroom for the rightmost
  // sample).
  const vocCap =
    metrics && metrics.voc_bracketed === true && Number.isFinite(metrics.V_oc)
      ? metrics.V_oc + 0.18
      : Number.POSITIVE_INFINITY
  const xmax = Math.min(maxV + 0.05, vocCap)
  return [xmin, xmax]
}

export function renderJV2D(el: HTMLElement, r: JV2DResult): void {
  Plotly.purge(el)
  // Reset wrapper without using innerHTML assignment (security hook).
  while (el.firstChild) el.removeChild(el.firstChild)
  // Layout: optional toolbar at the top, then plot, then metric-card
  // row + (optional) bracket warning banner. The plot div is a
  // SEPARATE child so the metric/warning markup can live in the same
  // wrapper without colliding with Plotly's mutating DOM. Raw J/V
  // arrays are NOT modified — the J → mA/cm² flip below is for
  // display only and matches the existing pre-Layer-3 behaviour.
  el.classList.add('jv2d-render')

  const m = r.metrics
  const style = readPlotStyleMode(el)

  // Toolbar: always rendered (Style: selector is independent of metrics
  // presence). Range: selector is appended only when ``metrics`` is
  // available, since the operational-range clip needs J_sc.
  const toolbar = document.createElement('div')
  toolbar.className = 'plot-toolbar'
  toolbar.setAttribute('data-test', 'jv2d-toolbar')

  // Style: select — Nature-style publication theme vs interactive
  // engineering theme. Always rendered. Toggle state persists on
  // ``el.dataset.plotStyleMode`` so renderJV2D re-entry honours it.
  {
    const styleLabel = document.createElement('label')
    styleLabel.className = 'plot-style-label'
    styleLabel.htmlFor = 'jv2d-style-mode'
    styleLabel.textContent = 'Style:'
    const styleSelect = document.createElement('select')
    styleSelect.id = 'jv2d-style-mode'
    styleSelect.className = 'plot-style-select'
    styleSelect.setAttribute('data-test', 'jv2d-style-mode')
    const optEng = document.createElement('option')
    optEng.value = 'engineering'
    optEng.textContent = 'Engineering'
    const optPub = document.createElement('option')
    optPub.value = 'publication'
    optPub.textContent = 'Publication'
    styleSelect.appendChild(optEng)
    styleSelect.appendChild(optPub)
    styleSelect.value = style
    styleSelect.addEventListener('change', () => {
      el.dataset.plotStyleMode = styleSelect.value === 'publication' ? 'publication' : 'engineering'
      renderJV2D(el, r)
    })
    toolbar.appendChild(styleLabel)
    toolbar.appendChild(styleSelect)
  }

  // Range: select — gated on metrics (existing precondition). Sets
  // ``yaxis.range`` to the operational window centred on J_sc when
  // ``Operational range`` is selected; otherwise lets Plotly autorange.
  // State lives on ``el.dataset.jv2dMode`` (separate from style mode).
  if (m) {
    const rangeLabel = document.createElement('label')
    rangeLabel.className = 'plot-range-label'
    rangeLabel.htmlFor = 'jv2d-range-mode'
    rangeLabel.textContent = 'Range:'
    const rangeSelect = document.createElement('select')
    rangeSelect.id = 'jv2d-range-mode'
    rangeSelect.className = 'plot-range-select'
    rangeSelect.setAttribute('data-test', 'jv2d-range-mode')
    const optOp = document.createElement('option')
    optOp.value = 'operational'
    optOp.textContent = 'Operational range'
    const optFull = document.createElement('option')
    optFull.value = 'full'
    optFull.textContent = 'Full sweep'
    rangeSelect.appendChild(optOp)
    rangeSelect.appendChild(optFull)
    rangeSelect.value = _jv2dReadMode(el)
    rangeSelect.addEventListener('change', () => {
      el.dataset.jv2dMode = rangeSelect.value === 'full' ? 'full' : 'operational'
      renderJV2D(el, r)
    })
    toolbar.appendChild(rangeLabel)
    toolbar.appendChild(rangeSelect)
  }
  el.appendChild(toolbar)

  const plotDiv = document.createElement('div')
  plotDiv.className = 'jv2d-plot'
  plotDiv.id = 'jv2d-plot-inner'
  el.appendChild(plotDiv)

  // 2D backend signs J < 0 under illumination at V=0; flip to match the
  // 1D forward-sweep display (J > 0 at V_sc, J = 0 at V_oc, J < 0 beyond).
  const J_mA = r.J.map(j => -j / 10)
  const Ny = r.grid_y.length
  const Nx = r.grid_x.length

  // Layer 4: compute optional y-axis clipping range. Returns
  // ``undefined`` when any precondition fails (mode='full', metrics
  // missing, voc_bracketed!==true, J_sc<=0, J_sc non-finite) so we
  // omit ``yaxis.range`` and let Plotly autorange.
  const yClip = _jv2dComputeYRange(_jv2dReadMode(el), m)

  // Vertical zero-line is drawn whenever the visible x-axis crosses
  // V=0 — either because the sweep itself includes negative voltage
  // OR because publication+operational mode gives the axis a small
  // negative visual margin (without inventing data). Horizontal
  // zero-line at J=0 is always drawn in publication mode.
  const range = _jv2dReadMode(el)
  const minV = r.V.length > 0 ? Math.min(...r.V) : 0
  const yClipPub = _jv2dComputeYRangePublication(range, m)
  const xClipPub = _jv2dComputeXRangePublication(range, r.V, m)
  // Use the publication x-clip's lower bound when active (it may be
  // -0.05 even though the raw data starts at 0); otherwise fall back
  // to the raw min(V) — never invents negative data, only relabels
  // the axis margin.
  const xVisibleMin = xClipPub ? xClipPub[0] : minV
  const xWithZero = xVisibleMin < 0

  if (style === 'publication') {
    const yaxisOpts: { title: string; withZeroLine: boolean; range?: [number, number] } = {
      title: 'Current density (mA cm⁻²)',
      withZeroLine: true,
    }
    // Publication mode prefers the tighter ``yClipPub`` window; fall
    // back to the broader engineering ``yClip`` only when the tighter
    // helper opts out (Full sweep, missing metrics, etc.).
    const yRangePub = yClipPub ?? yClip
    if (yRangePub) yaxisOpts.range = yRangePub
    const xaxisOpts: { title: string; withZeroLine: boolean; range?: [number, number] } = {
      title: 'Voltage (V)',
      withZeroLine: xWithZero,
    }
    if (xClipPub) xaxisOpts.range = xClipPub
    Plotly.newPlot(
      plotDiv,
      [
        {
          x: r.V, y: J_mA, name: 'Forward (2D)',
          mode: 'lines+markers',
          ...publicationTraceStyle({
            color: PUBLICATION_PALETTE.forward,
            hollow: true,
          }),
        },
      ],
      publicationLayout({
        xaxis: publicationAxis(xaxisOpts),
        yaxis: publicationAxis(yaxisOpts),
        annotations: metricAnnotation(m),
        // Single-trace 2D forward sweep — a Nature single-panel J-V
        // figure with one curve does not need a legend; the trace
        // identity is conveyed by the in-plot metric annotation and
        // the panel title outside the canvas.
        showlegend: false,
      }),
      publicationConfig('jv_2d_sweep'),
    )
  } else {
    const yaxisLayout: Record<string, unknown> = {
      ...(baseLayout().yaxis as object),
      title: axisTitle('Current density, <i>J</i> (mA·cm⁻²)'),
    }
    if (yClip) yaxisLayout.range = yClip
    Plotly.newPlot(
      plotDiv,
      [
        {
          x: r.V, y: J_mA, name: 'Forward (2D)',
          mode: 'lines+markers',
          line: { color: PALETTE.forward, width: LINE.width },
          marker: { ...MARKER, color: PALETTE.forward },
        },
      ],
      baseLayout({
        xaxis: { ...(baseLayout().xaxis as object), title: axisTitle('Applied bias, <i>V</i> (V)') },
        yaxis: yaxisLayout,
        annotations: [
          {
            x: 0.02, y: 0.05, xref: 'paper', yref: 'paper',
            xanchor: 'left', yanchor: 'bottom', showarrow: false,
            text: `2D grid: N<sub>x</sub>=${Nx}, N<sub>y</sub>=${Ny} · BC=${r.lateral_bc} · L<sub>x</sub>=${r.grid_x[Nx - 1].toFixed(0)} nm`,
            font: { size: 11 },
          },
        ],
      }),
      plotConfig('jv_2d_sweep'),
    )
  }

  // Layer 3: render JVMetrics card row + bracket-warning banner. Mirrors
  // the 1D ``panels/jv.ts`` pattern (V_oc / J_sc / FF / PCE), reusing the
  // same ``metricCard`` helper for visual consistency. Optional metrics
  // field — back-compat with payloads from a backend that pre-dates the
  // Layer 2 wire-through; in that case the row + warning are simply not
  // rendered. ``J_sc`` is in A/m² (J_sc-positive convention from the
  // backend; divide by 10 for mA/cm² display, mirror of 1D pane).
  // ``m`` was already bound at the top of this function for the Layer 4
  // toolbar — reuse rather than redeclare. The metric-card row and
  // warning banner are unchanged in publication mode (style mode only
  // affects the Plotly layout/traces/config, not the surrounding cards).
  if (m) {
    const metricsRow = document.createElement('div')
    metricsRow.className = 'jv2d-metrics-row'
    metricsRow.setAttribute('data-test', 'jv2d-metrics-row')
    // When the sweep didn't bracket V_oc, the backend returns sentinel
    // zeros for V_oc / FF / PCE. Rendering "0.000 V" as a physical V_oc
    // is misleading — show "—" (em dash, the conventional "not
    // available" placeholder). J_sc stays because ``compute_metrics``
    // interpolates it at V=0 regardless of bracket success.
    const bracketed = m.voc_bracketed !== false
    const dash = '—'
    const vocStr = bracketed ? `${m.V_oc.toFixed(3)} V` : dash
    const ffStr  = bracketed ? `${(m.FF * 100).toFixed(1)} %` : dash
    const pceStr = bracketed ? `${(m.PCE * 100).toFixed(2)} %` : dash
    const cardsHtml =
      `<div class="metric-row">` +
      metricCard('V<sub>OC</sub>', vocStr) +
      metricCard('J<sub>SC</sub>', `${(m.J_sc / 10).toFixed(2)} mA/cm²`) +
      metricCard('FF', ffStr) +
      metricCard('PCE', pceStr) +
      `</div>`
    metricsRow.insertAdjacentHTML('beforeend', cardsHtml)
    el.appendChild(metricsRow)

    // Bracket warning — only when the backend explicitly tells us the
    // sweep stopped short of V_oc. ``voc_bracketed === false`` (NOT
    // ``=== undefined``) is the trigger so an old backend payload that
    // omits the field is treated as "no warning surfaced", not as "warn
    // by default".
    if (m.voc_bracketed === false) {
      const warnBanner = document.createElement('div')
      warnBanner.className = 'jv2d-warning'
      warnBanner.setAttribute('data-test', 'jv2d-voc-not-bracketed')
      warnBanner.textContent =
        'V_oc not bracketed — increase V_max'
      el.appendChild(warnBanner)
    }
  }
}

// Publication-mode operational ranges for the 1D J-V workstation
// pane. Mirrors the 2D pane's helpers (``_jv2dComputeYRangePublication``
// / ``_jv2dComputeXRangePublication``). Locally inlined to avoid
// touching ``plot-theme.ts`` for this commit. Each helper returns
// ``undefined`` when its preconditions fail so the publication branch
// in ``renderJV`` can fall back to Plotly autorange — never to the
// engineering operational envelope.
// 1D-only metric source picker. Returns whichever sweep's metrics
// should drive both the publication-mode annotation AND the y/x
// range helpers, plus the human-readable label used to prefix the
// annotation. Returning the same object from a single helper keeps
// the annotation, y-range, and x-range in lockstep — under heavy
// hysteresis the forward sweep can fail to bracket V_oc while the
// reverse sweep brackets fine, and we need ALL three derived values
// to resolve from the same chosen sweep so the publication panel is
// internally consistent.
//
// Resolution rules (preserved from the previous picker):
//   1. metrics_fwd.voc_bracketed === true → Forward
//   2. metrics_rev.voc_bracketed === true → Reverse
//   3. either side === false              → Forward fallback (so the
//                                            annotation still surfaces
//                                            "V_oc: not bracketed" + J_sc
//                                            via metrics_fwd)
//   4. both undefined (legacy / stale)    → null
//
// 2D pane has only one trace, no fwd/rev distinction, and continues to
// call ``metricAnnotation(m)`` directly — this picker is 1D-specific.
type Jv1dPick = {
  metrics: JVResult['metrics_fwd']
  label: 'Forward' | 'Reverse'
  source: 'forward' | 'reverse'
}
function _jv1dPickMetrics(
  metrics_fwd: JVResult['metrics_fwd'] | undefined,
  metrics_rev: JVResult['metrics_rev'] | undefined,
): Jv1dPick | null {
  if (metrics_fwd && metrics_fwd.voc_bracketed === true) {
    return { metrics: metrics_fwd, label: 'Forward', source: 'forward' }
  }
  if (metrics_rev && metrics_rev.voc_bracketed === true) {
    return { metrics: metrics_rev, label: 'Reverse', source: 'reverse' }
  }
  if (
    (metrics_fwd && metrics_fwd.voc_bracketed === false) ||
    (metrics_rev && metrics_rev.voc_bracketed === false)
  ) {
    // Forward fallback. metrics_fwd may be the unbracketed sweep —
    // metricAnnotation will surface "V_oc: not bracketed" + J_sc.
    if (metrics_fwd) {
      return { metrics: metrics_fwd, label: 'Forward', source: 'forward' }
    }
  }
  return null
}

function _jv1dPickAnnotation(pick: Jv1dPick | null): Record<string, unknown>[] {
  if (!pick) return []
  return metricAnnotation(pick.metrics, { label: pick.label })
}

// Tight publication y-range based on the picked sweep's J_sc. Falls
// back to autorange when the picked sweep is not bracketed (sentinel
// J_sc still meaningful for J_sc-cards, but not safe to drive the
// y-axis envelope) or when no sweep was picked.
function _jv1dComputeYRangePublication(
  pick: Jv1dPick | null,
): [number, number] | undefined {
  if (!pick) return undefined
  const m = pick.metrics
  if (m.voc_bracketed !== true) return undefined
  if (!Number.isFinite(m.J_sc) || m.J_sc <= 0) return undefined
  // 1D backend signs J_sc as POSITIVE in metrics_fwd / metrics_rev
  // (panels/jv.ts displays it via ``(m.J_sc / 10).toFixed(2) mA/cm²``
  // without a negation — same convention as the 2D pane post-Layer 2).
  // Tight publication envelope matches the 2D refinement.
  const J_sc_mA = m.J_sc / 10
  return [-0.15 * J_sc_mA, 1.12 * J_sc_mA]
}

// Tight publication x-range. Same -0.05 V left margin (when sweep
// starts at V≥0) as before, but the upper-bound V_oc cap now uses
// the PICKED sweep's V_oc — under heavy hysteresis this means a
// reverse-bracketed sweep tightens the x-axis to ``V_oc(reverse) +
// 0.18`` rather than spilling past forward's last sample.
function _jv1dComputeXRangePublication(
  V_fwd: number[],
  V_rev: number[],
  pick: Jv1dPick | null,
): [number, number] | undefined {
  const allV = [...V_fwd, ...V_rev]
  if (allV.length === 0) return undefined
  const minV = Math.min(...allV)
  const maxV = Math.max(...allV)
  const xmin = minV >= 0 ? -0.05 : minV
  const vocCap =
    pick && pick.metrics.voc_bracketed === true && Number.isFinite(pick.metrics.V_oc)
      ? pick.metrics.V_oc + 0.18
      : Number.POSITIVE_INFINITY
  const xmax = Math.min(maxV + 0.05, vocCap)
  return [xmin, xmax]
}

export function renderJV(el: HTMLElement, r: JVResult): void {
  Plotly.purge(el)
  // Reset wrapper without using innerHTML assignment (security hook).
  while (el.firstChild) el.removeChild(el.firstChild)
  el.classList.add('jv1d-render')

  const style = readPlotStyleMode(el)

  // Toolbar — always rendered when plot data exists. Hosts only the
  // Style: selector for 1D J-V (no Operational/Full sweep concept on
  // 1D today).
  const toolbar = document.createElement('div')
  toolbar.className = 'plot-toolbar'
  toolbar.setAttribute('data-test', 'jv1d-toolbar')
  {
    const styleLabel = document.createElement('label')
    styleLabel.className = 'plot-style-label'
    styleLabel.htmlFor = 'jv1d-style-mode'
    styleLabel.textContent = 'Style:'
    const styleSelect = document.createElement('select')
    styleSelect.id = 'jv1d-style-mode'
    styleSelect.className = 'plot-style-select'
    styleSelect.setAttribute('data-test', 'jv1d-style-mode')
    const optEng = document.createElement('option')
    optEng.value = 'engineering'
    optEng.textContent = 'Engineering'
    const optPub = document.createElement('option')
    optPub.value = 'publication'
    optPub.textContent = 'Publication'
    styleSelect.appendChild(optEng)
    styleSelect.appendChild(optPub)
    styleSelect.value = style
    styleSelect.addEventListener('change', () => {
      el.dataset.plotStyleMode = styleSelect.value === 'publication' ? 'publication' : 'engineering'
      renderJV(el, r)
    })
    toolbar.appendChild(styleLabel)
    toolbar.appendChild(styleSelect)
  }
  el.appendChild(toolbar)

  const plotDiv = document.createElement('div')
  plotDiv.className = 'jv1d-plot'
  plotDiv.id = 'jv1d-plot-inner'
  el.appendChild(plotDiv)

  // Pre-existing display transform — A/m² → mA/cm² + reverse sweep
  // is monotonised by reversing both arrays so Plotly renders the
  // dashed reverse curve in the same x-direction as the forward.
  // Raw r.V_fwd / r.J_fwd / r.V_rev / r.J_rev arrays remain
  // untouched (regression-pinned).
  const J_fwd_mA = r.J_fwd.map(j => j / 10)
  const J_rev_mA = r.J_rev.map(j => j / 10)
  const V_rev_sorted = [...r.V_rev].reverse()
  const J_rev_sorted = [...J_rev_mA].reverse()

  if (style === 'publication') {
    // Single source for annotation, y-range and x-range so that all
    // three derived values agree on which sweep they describe.
    const pick = _jv1dPickMetrics(r.metrics_fwd, r.metrics_rev)
    const yClipPub = _jv1dComputeYRangePublication(pick)
    const xClipPub = _jv1dComputeXRangePublication(r.V_fwd, r.V_rev, pick)
    const allV = [...r.V_fwd, ...r.V_rev]
    const minV = allV.length > 0 ? Math.min(...allV) : 0
    const xVisibleMin = xClipPub ? xClipPub[0] : minV
    const xWithZero = xVisibleMin < 0

    const yaxisOpts: { title: string; withZeroLine: boolean; range?: [number, number] } = {
      title: 'Current density (mA cm⁻²)',
      withZeroLine: true,
    }
    if (yClipPub) yaxisOpts.range = yClipPub
    const xaxisOpts: { title: string; withZeroLine: boolean; range?: [number, number] } = {
      title: 'Voltage (V)',
      withZeroLine: xWithZero,
    }
    if (xClipPub) xaxisOpts.range = xClipPub

    Plotly.newPlot(
      plotDiv,
      [
        {
          x: r.V_fwd, y: J_fwd_mA, name: 'Forward',
          mode: 'lines+markers',
          ...publicationTraceStyle({
            color: PUBLICATION_PALETTE.forward,
            hollow: true,
          }),
        },
        {
          x: V_rev_sorted, y: J_rev_sorted, name: 'Reverse',
          mode: 'lines+markers',
          ...publicationTraceStyle({
            color: PUBLICATION_PALETTE.reverse,
            hollow: true,
            dash: 'dash',
          }),
        },
      ],
      publicationLayout({
        xaxis: publicationAxis(xaxisOpts),
        yaxis: publicationAxis(yaxisOpts),
        // Annotation source picker — under heavy hysteresis the
        // forward sweep can fail to bracket V_oc while the reverse
        // sweep brackets fine, so always-reading metrics_fwd produces
        // a misleading "V_oc: not bracketed" next to a plot whose
        // reverse trace clearly crosses zero. Pick the bracketed
        // sweep (forward preferred when both bracket) and label which
        // sweep the numbers describe; legacy payloads with both
        // ``voc_bracketed === undefined`` get no annotation, matching
        // metricAnnotation's existing fallback.
        annotations: _jv1dPickAnnotation(pick),
      }),
      publicationConfig('jv_sweep'),
    )
  } else {
    Plotly.newPlot(
      plotDiv,
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

// ── Electroluminescence (Rau reciprocity) ───────────────────────────────────

function renderEL(el: HTMLElement, r: ELResult): void {
  Plotly.purge(el)
  el.innerHTML = ''
  const absPct = r.absorber_absorptance.map(a => a * 100)
  Plotly.newPlot(
    el,
    [
      {
        x: r.wavelengths_nm, y: r.EL_spectrum,
        name: 'EL spectrum',
        mode: 'lines+markers',
        line: { color: PALETTE.forward, width: LINE.width },
        marker: { ...MARKER, color: PALETTE.forward },
        yaxis: 'y',
      },
      {
        x: r.wavelengths_nm, y: absPct,
        name: 'A<sub>abs</sub>(\u03bb)',
        mode: 'lines',
        line: { color: PALETTE.reverse, width: LINE.width, dash: 'dot' },
        yaxis: 'y2',
      },
    ],
    baseLayout({
      xaxis: { ...(baseLayout().xaxis as object), title: axisTitle('Wavelength, <i>\u03bb</i> (nm)') },
      yaxis: { ...(baseLayout().yaxis as object),
        title: axisTitle('EL flux (photons&middot;m\u207B\u00b2&middot;s\u207B\u00b9&middot;nm\u207B\u00b9)') },
      yaxis2: {
        title: axisTitle('Absorptance (%)'),
        overlaying: 'y', side: 'right', range: [0, 100],
        showgrid: false,
      },
      annotations: [
        {
          x: 0.02, y: 0.98, xref: 'paper', yref: 'paper',
          xanchor: 'left', yanchor: 'top', showarrow: false,
          text: `V<sub>inj</sub> = ${r.V_inj.toFixed(2)} V &nbsp; EQE<sub>EL</sub> = ${r.EQE_EL.toExponential(2)} &nbsp; &Delta;V<sub>nr</sub> = ${r.delta_V_nr_mV.toFixed(1)} mV`,
          font: { size: 12 },
        },
      ],
    }),
    plotConfig('el'),
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
