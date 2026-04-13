import Plotly from 'plotly.js-dist-min'
import type { Workspace } from '../types'
import { findRun } from '../state'
import { baseLayout, plotConfig, PALETTE, LINE, MARKER, axisTitle } from '../../plot-theme'
import type { JVResult } from '../../types'

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
          clear('(impedance plot in this pane: Task 9)')
          return
        case 'degradation':
          clear('(degradation plot in this pane: Task 10)')
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
