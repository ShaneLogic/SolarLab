import Plotly from 'plotly.js-basic-dist-min'
import { mountDevicePanel } from '../device-panel'
import { startJob, streamJobEvents } from '../job-stream'
import { createProgressBar, type ProgressBarHandle } from '../progress'
import { baseLayout, plotConfig, PALETTE, LINE, MARKER, axisTitle } from '../plot-theme'
import { setStatus, metricCard, numField, readNum, checkField, readCheck } from '../ui-helpers'
import type { JVResult } from '../types'

export async function mountJVPanel(root: HTMLElement): Promise<void> {
  root.innerHTML = `
    <div id="jv-device"></div>
    <div class="card">
      <h3>Sweep Parameters</h3>
      <div class="form-grid">
        ${numField('jv-N', 'N<sub>grid</sub>', 60, '1')}
        ${numField('jv-np', 'V sample points', 30, '1')}
        ${numField('jv-rate', 'Scan rate (V/s)', 1.0, 'any')}
        ${numField('jv-vmax', 'V<sub>max</sub> (V)', 1.4, '0.01')}
        ${checkField('jv-dark', 'Dark J–V (no illumination)', false)}
      </div>
      <p>
        <b>V<sub>max</sub></b> is the upper voltage of the forward sweep.
        Leave it at the default (1.4&nbsp;V) unless V<sub>oc</sub> on your
        stack exceeds that; the Python API picks
        <code>max(V<sub>bi,eff</sub>&times;1.3, 1.4&nbsp;V)</code> when called
        with <code>V_max=None</code>, but the UI requires an explicit number.
        If the forward curve never crosses J&nbsp;=&nbsp;0, raise V<sub>max</sub>.
      </p>
      <div class="actions">
        <button class="btn btn-primary" id="btn-jv">Run J-V Sweep</button>
        <span class="status" id="status-jv"></span>
      </div>
      <div id="progress-jv"></div>
    </div>
    <div id="results-jv"></div>`

  const devicePanel = await mountDevicePanel(
    root.querySelector<HTMLDivElement>('#jv-device')!,
    'jv',
  )

  const progressEl = root.querySelector<HTMLDivElement>('#progress-jv')!
  const progressBar: ProgressBarHandle = createProgressBar(progressEl)

  const btn = root.querySelector<HTMLButtonElement>('#btn-jv')!
  btn.addEventListener('click', async () => {
    btn.disabled = true
    progressBar.reset()
    setStatus('status-jv', 'Starting job…')
    try {
      const device = devicePanel.getConfig()
      const isDark = readCheck('jv-dark', false)
      const params = {
        N_grid: Math.max(3, Math.round(readNum('jv-N', 60))),
        n_points: Math.max(2, Math.round(readNum('jv-np', 30))),
        v_rate: readNum('jv-rate', 1.0),
        V_max: readNum('jv-vmax', 1.4),
        illuminated: !isDark,
      }
      const jobId = await startJob('jv', device, params)
      setStatus('status-jv', 'Running J–V sweep…')

      streamJobEvents<JVResult>(jobId, {
        onProgress: (ev) => progressBar.update(ev),
        onResult: (result) => {
          renderJVResults(root.querySelector<HTMLDivElement>('#results-jv')!, result)
          progressBar.done()
          setStatus('status-jv', 'Done')
        },
        onError: (msg) => {
          progressBar.error(msg)
          setStatus('status-jv', `Error: ${msg}`, true)
        },
        onDone: () => {
          btn.disabled = false
        },
      })
    } catch (e) {
      progressBar.error((e as Error).message)
      setStatus('status-jv', `Error: ${(e as Error).message}`, true)
      btn.disabled = false
    }
  })
}

function renderJVResults(container: HTMLElement, r: JVResult): void {
  const mf = r.metrics_fwd
  const mr = r.metrics_rev
  container.innerHTML = `
    <div class="card">
      <h3>Performance Metrics</h3>
      <div class="metrics-grid">
        <div class="metric-block">
          <div class="metric-block-title">Forward</div>
          <div class="metric-row">
            ${metricCard('V<sub>oc</sub>', `${mf.V_oc.toFixed(3)} V`)}
            ${metricCard('J<sub>sc</sub>', `${(mf.J_sc / 10).toFixed(2)} mA/cm²`)}
            ${metricCard('FF', `${(mf.FF * 100).toFixed(1)} %`)}
            ${metricCard('PCE', `${(mf.PCE * 100).toFixed(2)} %`)}
          </div>
        </div>
        <div class="metric-block">
          <div class="metric-block-title">Reverse</div>
          <div class="metric-row">
            ${metricCard('V<sub>oc</sub>', `${mr.V_oc.toFixed(3)} V`)}
            ${metricCard('J<sub>sc</sub>', `${(mr.J_sc / 10).toFixed(2)} mA/cm²`)}
            ${metricCard('FF', `${(mr.FF * 100).toFixed(1)} %`)}
            ${metricCard('PCE', `${(mr.PCE * 100).toFixed(2)} %`)}
          </div>
        </div>
        <div class="metric-block">
          <div class="metric-block-title">Hysteresis Index</div>
          <div class="hi-value">${r.hysteresis_index.toFixed(3)}</div>
        </div>
      </div>
    </div>
    <div class="card">
      <h3>J-V Curves</h3>
      <div id="plot-jv" class="plot-container"></div>
    </div>`

  const J_fwd_mA = r.J_fwd.map(j => j / 10)
  const J_rev_mA = r.J_rev.map(j => j / 10)
  const V_rev_sorted = [...r.V_rev].reverse()
  const J_rev_sorted = [...J_rev_mA].reverse()

  Plotly.newPlot(
    'plot-jv',
    [
      {
        x: r.V_fwd, y: J_fwd_mA, name: 'Forward',
        mode: 'lines+markers',
        line: { color: PALETTE.forward, width: LINE.width },
        marker: { ...MARKER, color: PALETTE.forward },
        hovertemplate: 'V = %{x:.3f} V<br>J = %{y:.2f} mA/cm²<extra>Forward</extra>',
      },
      {
        x: V_rev_sorted, y: J_rev_sorted, name: 'Reverse',
        mode: 'lines+markers',
        line: { color: PALETTE.reverse, width: LINE.width, dash: 'dash' },
        marker: { ...MARKER, color: PALETTE.reverse, symbol: 'square' },
        hovertemplate: 'V = %{x:.3f} V<br>J = %{y:.2f} mA/cm²<extra>Reverse</extra>',
      },
    ],
    baseLayout({
      xaxis: { ...(baseLayout().xaxis as object), title: axisTitle('Applied bias, <i>V</i> (V)') },
      yaxis: { ...(baseLayout().yaxis as object), title: axisTitle('Current density, <i>J</i> (mA·cm⁻²)') },
      legend: { x: 0.02, y: 0.05, xanchor: 'left', yanchor: 'bottom', ...(baseLayout().legend as object) },
    }),
    plotConfig('jv_sweep'),
  )
}
