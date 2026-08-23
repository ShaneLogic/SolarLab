import Plotly from 'plotly.js-basic-dist-min'
import { mountDevicePanel } from '../device-panel'
import { startJob, streamJobEvents } from '../job-stream'
import { createProgressBar, type ProgressBarHandle } from '../progress'
import { baseLayout, plotConfig, PALETTE, LINE, MARKER, axisTitle } from '../plot-theme'
import { checkField, numField, readCheck, readNum, setStatus } from '../ui-helpers'
import {
  collectImpedanceEvidenceWarnings,
  summarizeImpedanceEvidence,
} from '../impedance-evidence'
import type { ISResult } from '../types'

export async function mountImpedancePanel(root: HTMLElement): Promise<void> {
  root.innerHTML = `
    <div id="is-device"></div>
    <div class="card">
      <h3>Spectroscopy Parameters</h3>
      <div class="form-grid">
        ${numField('is-N', 'N<sub>grid</sub>', 40, '1')}
        ${numField('is-Vdc', 'V<sub>dc</sub> (V)', 0.9, '0.01')}
        ${numField('is-nf', 'N<sub>f</sub>', 15, '1')}
        ${numField('is-fmin', 'f<sub>min</sub> (Hz)', 10, 'any')}
        ${numField('is-fmax', 'f<sub>max</sub> (Hz)', 1e5, 'any')}
        ${numField('is-dv', '&delta;V (mV)', 10, '0.5')}
        ${numField('is-cycles', 'Cycles', 5, '1')}
        ${numField('is-extract', 'Extract cycles', 2, '1')}
        ${numField('is-ppc', 'Points/cycle', 40, '1')}
        ${numField('is-dc-settle', 'DC settle (s)', 1e-3, 'any')}
        <label class="form-group">
          <span>Engine</span>
          <select id="is-method">
            <option value="transient_ion_aware">Transient, ion-aware</option>
            <option value="ion_aware_frequency_certified">Certified frequency, ion-aware</option>
            <option value="qf_frequency_ion_free">QF frequency, ion-free</option>
          </select>
        </label>
        ${checkField('is-illuminated', 'Illuminated', true)}
        ${checkField('is-strict', 'Require contact certificate', false)}
        ${checkField('is-window-strict', 'Require ionic frequency coverage', true)}
      </div>
      <div class="actions">
        <button class="btn btn-primary" id="btn-is">Run Impedance</button>
        <span class="status" id="status-is"></span>
      </div>
      <div id="progress-is"></div>
    </div>
    <div id="results-is"></div>`

  const devicePanel = await mountDevicePanel(
    root.querySelector<HTMLDivElement>('#is-device')!,
    'is',
  )

  const progressEl = root.querySelector<HTMLDivElement>('#progress-is')!
  const progressBar: ProgressBarHandle = createProgressBar(progressEl)

  const btn = root.querySelector<HTMLButtonElement>('#btn-is')!
  const methodSelect = root.querySelector<HTMLSelectElement>('#is-method')!
  const syncMethodControls = (): void => {
    const transient = methodSelect.value === 'transient_ion_aware'
    for (const id of ['is-cycles', 'is-extract', 'is-ppc', 'is-dc-settle']) {
      const input = root.querySelector<HTMLInputElement>(`#${id}`)
      if (input) input.disabled = !transient
    }
    const windowStrict = root.querySelector<HTMLInputElement>('#is-window-strict')
    if (windowStrict) {
      windowStrict.disabled = methodSelect.value !== 'ion_aware_frequency_certified'
    }
  }
  methodSelect.addEventListener('change', () => {
    const certified = methodSelect.value === 'ion_aware_frequency_certified'
    const presets: Record<string, number> = certified
      ? { 'is-N': 60, 'is-nf': 29, 'is-fmin': 1e-6, 'is-fmax': 10 }
      : { 'is-N': 40, 'is-nf': 15, 'is-fmin': 10, 'is-fmax': 1e5 }
    for (const [id, value] of Object.entries(presets)) {
      const input = root.querySelector<HTMLInputElement>(`#${id}`)
      if (input) input.value = String(value)
    }
    syncMethodControls()
  })
  syncMethodControls()

  btn.addEventListener('click', async () => {
    btn.disabled = true
    progressBar.reset()
    progressBar.busy()
    setStatus('status-is', 'Starting job…')
    try {
      const device = devicePanel.getConfig()
      const nCycles = Math.max(1, Math.round(readNum('is-cycles', 5)))
      const selectedMethod = (
        document.getElementById('is-method') as HTMLSelectElement | null
      )?.value
      const method = selectedMethod === 'qf_frequency_ion_free'
        ? 'qf_frequency_ion_free' as const
        : selectedMethod === 'transient_ion_aware'
          ? 'transient_ion_aware' as const
          : 'ion_aware_frequency_certified' as const
      const params = {
        N_grid: Math.max(3, Math.round(readNum('is-N', 40))),
        V_dc: readNum('is-Vdc', 0.9),
        n_freq: Math.max(2, Math.round(readNum('is-nf', 15))),
        f_min: readNum('is-fmin', 10),
        f_max: readNum('is-fmax', 1e5),
        delta_V: readNum('is-dv', 10) * 1e-3,
        n_cycles: nCycles,
        n_extract: Math.min(
          nCycles,
          Math.max(1, Math.round(readNum('is-extract', 2))),
        ),
        points_per_cycle: Math.max(8, Math.round(readNum('is-ppc', 40))),
        dc_settle_time: readNum('is-dc-settle', 1e-3),
        illuminated: readCheck('is-illuminated', true),
        method,
        require_operating_point_certificate: readCheck('is-strict', false),
        require_frequency_window_certificate: (
          method === 'ion_aware_frequency_certified'
          && readCheck('is-window-strict', true)
        ),
      }
      const jobId = await startJob('impedance', device, params)
      setStatus('status-is', 'Running impedance spectroscopy…')

      streamJobEvents<ISResult>(jobId, {
        onProgress: (ev) => progressBar.update(ev),
        onResult: (result) => {
          renderISResults(root.querySelector<HTMLDivElement>('#results-is')!, result)
          progressBar.done()
          setStatus('status-is', 'Done')
        },
        onError: (msg) => {
          progressBar.error(msg)
          setStatus('status-is', `Error: ${msg}`, true)
        },
        onDone: () => {
          btn.disabled = false
        },
      })
    } catch (e) {
      progressBar.error((e as Error).message)
      setStatus('status-is', `Error: ${(e as Error).message}`, true)
      btn.disabled = false
    }
  })
}

function renderISResults(container: HTMLElement, r: ISResult): void {
  container.innerHTML = `
    <div id="is-evidence-summary" class="impedance-evidence-summary"
         data-test="impedance-evidence-summary"></div>
    <div id="is-evidence" class="jv2d-warning" role="alert"
         data-test="impedance-evidence-warning" hidden></div>
    <div class="results-row">
      <div class="card">
        <h3>Nyquist Plot</h3>
        <div id="plot-nyquist" class="plot-container"></div>
      </div>
      <div class="card">
        <h3>Bode Plot</h3>
        <div id="plot-bode" class="plot-container"></div>
      </div>
    </div>`

  const summary = container.querySelector<HTMLDivElement>('#is-evidence-summary')!
  for (const line of summarizeImpedanceEvidence(r)) {
    const item = document.createElement('span')
    item.textContent = line
    summary.appendChild(item)
  }

  const evidence = container.querySelector<HTMLDivElement>('#is-evidence')!
  const notices = collectImpedanceEvidenceWarnings(r)
  if (notices.length > 0) {
    evidence.textContent = notices.join(' | ')
    evidence.hidden = false
  }

  const negImag = r.Z_imag.map(z => -z)
  const baseNyq = baseLayout()
  Plotly.newPlot(
    'plot-nyquist',
    [
      {
        x: r.Z_real, y: negImag,
        mode: 'lines+markers',
        line: { color: PALETTE.primary, width: LINE.width },
        marker: { ...MARKER, color: PALETTE.primary },
        hovertemplate: 'Re(Z) = %{x:.3e}<br>-Im(Z) = %{y:.3e}<extra></extra>',
      },
    ],
    baseLayout({
      xaxis: { ...(baseNyq.xaxis as object), title: axisTitle('Real impedance, Re(<i>Z</i>) (Ω·m²)') },
      yaxis: { ...(baseNyq.yaxis as object), title: axisTitle('Negative imaginary impedance, −Im(<i>Z</i>) (Ω·m²)'), scaleanchor: 'x' },
      showlegend: false,
    }),
    plotConfig('nyquist'),
  )

  const Z_mag = r.Z_real.map((re, i) => Math.sqrt(re ** 2 + r.Z_imag[i] ** 2))
  const phase = r.Z_real.map((re, i) => Math.atan2(r.Z_imag[i], re) * 180 / Math.PI)
  const baseBode = baseLayout()

  Plotly.newPlot(
    'plot-bode',
    [
      {
        x: r.frequencies, y: Z_mag, name: '|Z|',
        mode: 'lines+markers',
        line: { color: PALETTE.primary, width: LINE.width },
        marker: { ...MARKER, color: PALETTE.primary },
        yaxis: 'y',
        hovertemplate: 'f = %{x:.2e} Hz<br>|Z| = %{y:.3e}<extra></extra>',
      },
      {
        x: r.frequencies, y: phase, name: 'Phase',
        mode: 'lines+markers',
        line: { color: PALETTE.accent, width: LINE.width, dash: 'dot' },
        marker: { ...MARKER, color: PALETTE.accent, symbol: 'diamond' },
        yaxis: 'y2',
        hovertemplate: 'f = %{x:.2e} Hz<br>phase = %{y:.1f}°<extra></extra>',
      },
    ],
    baseLayout({
      xaxis: { ...(baseBode.xaxis as object), title: axisTitle('Frequency, <i>f</i> (Hz)'), type: 'log' },
      yaxis: { ...(baseBode.yaxis as object), title: axisTitle('Impedance magnitude, |<i>Z</i>| (Ω·m²)'), type: 'log' },
      yaxis2: {
        ...(baseBode.yaxis as object),
        title: axisTitle('Phase, <i>ϕ</i> (°)'),
        overlaying: 'y',
        side: 'right',
        showgrid: false,
      },
      legend: { x: 0.98, y: 0.98, xanchor: 'right', yanchor: 'top', ...(baseBode.legend as object) },
      margin: { t: 30, r: 70, b: 60, l: 70 },
    }),
    plotConfig('bode'),
  )
}
