import Plotly from 'plotly.js-basic-dist-min'
import { mountDevicePanel } from '../device-panel'
import { startJob, streamJobEvents } from '../job-stream'
import { createProgressBar, type ProgressBarHandle } from '../progress'
import { baseLayout, plotConfig, PALETTE, LINE, MARKER, axisTitle } from '../plot-theme'
import { setStatus, numField, readNum } from '../ui-helpers'
import type { DegResult } from '../types'

export async function mountDegradationPanel(root: HTMLElement): Promise<void> {
  root.innerHTML = `
    <div id="deg-device"></div>
    <div class="card">
      <h3>Degradation Parameters</h3>
      <div class="form-grid">
        ${numField('deg-N', 'N<sub>grid</sub>', 40, '1')}
        ${numField('deg-Vbias', 'V<sub>bias</sub> (V)', 0.9, '0.01')}
        ${numField('deg-tend', 't<sub>end</sub> (s)', 100, 'any')}
        ${numField('deg-nsnap', 'N snapshots', 10, '1')}
      </div>
      <div class="actions">
        <button class="btn btn-primary" id="btn-deg">Run Degradation</button>
        <span class="status" id="status-deg"></span>
      </div>
      <div id="progress-deg"></div>
    </div>
    <div id="results-deg"></div>`

  const devicePanel = await mountDevicePanel(
    root.querySelector<HTMLDivElement>('#deg-device')!,
    'deg',
  )

  const progressEl = root.querySelector<HTMLDivElement>('#progress-deg')!
  const progressBar: ProgressBarHandle = createProgressBar(progressEl)

  const btn = root.querySelector<HTMLButtonElement>('#btn-deg')!
  btn.addEventListener('click', async () => {
    btn.disabled = true
    progressBar.reset()
    setStatus('status-deg', 'Starting job…')
    try {
      const device = devicePanel.getConfig()
      const params = {
        N_grid: Math.max(3, Math.round(readNum('deg-N', 40))),
        V_bias: readNum('deg-Vbias', 0.9),
        t_end: readNum('deg-tend', 100),
        n_snapshots: Math.max(2, Math.round(readNum('deg-nsnap', 10))),
      }
      const jobId = await startJob('degradation', device, params)
      setStatus('status-deg', 'Running degradation simulation…')

      streamJobEvents<DegResult>(jobId, {
        onProgress: (ev) => progressBar.update(ev),
        onResult: (result) => {
          renderDegResults(root.querySelector<HTMLDivElement>('#results-deg')!, result)
          progressBar.done()
          setStatus('status-deg', 'Done')
        },
        onError: (msg) => {
          progressBar.error(msg)
          setStatus('status-deg', `Error: ${msg}`, true)
        },
        onDone: () => {
          btn.disabled = false
        },
      })
    } catch (e) {
      progressBar.error((e as Error).message)
      setStatus('status-deg', `Error: ${(e as Error).message}`, true)
      btn.disabled = false
    }
  })
}

function renderDegResults(container: HTMLElement, r: DegResult): void {
  container.innerHTML = `
    <div class="results-row">
      <div class="card">
        <h3>PCE vs Time</h3>
        <div id="plot-pce" class="plot-container"></div>
      </div>
      <div class="card">
        <h3>V<sub>OC</sub> & J<sub>SC</sub> vs Time</h3>
        <div id="plot-voc-jsc" class="plot-container"></div>
      </div>
    </div>`

  const pce_pct = r.PCE.map(p => p * 100)
  const basePce = baseLayout()

  Plotly.newPlot(
    'plot-pce',
    [{
      x: r.times, y: pce_pct,
      mode: 'lines+markers',
      line: { color: PALETTE.primary, width: LINE.width },
      marker: { ...MARKER, color: PALETTE.primary },
      hovertemplate: 't = %{x:.2e} s<br>PCE = %{y:.2f} %<extra></extra>',
    }],
    baseLayout({
      xaxis: { ...(basePce.xaxis as object), title: axisTitle('Time, <i>t</i> (s)'), type: 'log' },
      yaxis: { ...(basePce.yaxis as object), title: axisTitle('Power conversion efficiency, PCE (%)') },
      showlegend: false,
    }),
    plotConfig('pce_degradation'),
  )

  const jsc_mA = r.J_sc.map(j => j / 10)
  const baseVJ = baseLayout()

  Plotly.newPlot(
    'plot-voc-jsc',
    [
      {
        x: r.times, y: r.V_oc, name: 'V<sub>OC</sub>',
        mode: 'lines+markers',
        line: { color: PALETTE.primary, width: LINE.width },
        marker: { ...MARKER, color: PALETTE.primary },
        yaxis: 'y',
        hovertemplate: 't = %{x:.2e} s<br>V_OC = %{y:.3f} V<extra></extra>',
      },
      {
        x: r.times, y: jsc_mA, name: 'J<sub>SC</sub>',
        mode: 'lines+markers',
        line: { color: PALETTE.reverse, width: LINE.width, dash: 'dash' },
        marker: { ...MARKER, color: PALETTE.reverse, symbol: 'square' },
        yaxis: 'y2',
        hovertemplate: 't = %{x:.2e} s<br>J_SC = %{y:.2f} mA/cm²<extra></extra>',
      },
    ],
    baseLayout({
      xaxis: { ...(baseVJ.xaxis as object), title: axisTitle('Time, <i>t</i> (s)'), type: 'log' },
      yaxis: { ...(baseVJ.yaxis as object), title: axisTitle('Open-circuit voltage, <i>V</i><sub>OC</sub> (V)') },
      yaxis2: {
        ...(baseVJ.yaxis as object),
        title: axisTitle('Short-circuit current, <i>J</i><sub>SC</sub> (mA·cm⁻²)'),
        overlaying: 'y',
        side: 'right',
        showgrid: false,
      },
      legend: { x: 0.02, y: 0.05, xanchor: 'left', yanchor: 'bottom', ...(baseVJ.legend as object) },
      margin: { t: 30, r: 70, b: 60, l: 70 },
    }),
    plotConfig('voc_jsc_degradation'),
  )
}
