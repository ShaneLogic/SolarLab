import { listConfigs } from '../../api'
import { startJob, streamJobEvents } from '../../job-stream'
import { createProgressBar, type ProgressBarHandle } from '../../progress'
import { setStatus, readNum, numField } from '../../ui-helpers'
import { mountTandemStackVisualizer } from '../../stack/tandem-stack-visualizer'
import type { TandemJVPayload, ConfigEntry } from '../../types'

/**
 * Tandem pane — config-path-based workflow (not device-object-based).
 *
 * Uses the /api/jobs SSE flow so progress events from both sub-cell sweeps
 * ("top/fwd", "top/rev", "bot/fwd", "bot/rev") are streamed into a shared
 * progress bar — same pattern as panels/jv.ts.
 */

function fmt4(v: number): string {
  return v.toFixed(4)
}

function renderMetrics(data: TandemJVPayload): string {
  const m = data.metrics
  const jscDisplay = Math.abs(m.J_sc) / 10
  const pceDisplay = m.PCE * 100

  const lines: string[] = [
    '=== Tandem J-V Metrics ===',
    `  V_oc  = ${fmt4(m.V_oc)} V`,
    `  J_sc  = ${fmt4(jscDisplay)} mA/cm²`,
    `  FF    = ${fmt4(m.FF)}`,
    `  PCE   = ${fmt4(pceDisplay)} %`,
    '',
  ]

  if (data.benchmark) {
    lines.push('--- Benchmark comparison ---')
    const b = data.benchmark
    if (b.source) lines.push(`  Source: ${b.source}`)
    if (b.V_oc != null) lines.push(`  V_oc  target = ${fmt4(b.V_oc)} V      simulated = ${fmt4(m.V_oc)} V`)
    if (b.J_sc != null) lines.push(`  J_sc  target = ${fmt4(b.J_sc)} mA/cm²  simulated = ${fmt4(jscDisplay)} mA/cm²`)
    if (b.FF != null)   lines.push(`  FF    target = ${fmt4(b.FF)}            simulated = ${fmt4(m.FF)}`)
    if (b.PCE != null)  lines.push(`  PCE   target = ${fmt4(b.PCE)} %         simulated = ${fmt4(pceDisplay)} %`)
  }

  return lines.join('\n')
}

export function mountTandemPane(container: HTMLElement): void {
  container.innerHTML = `
    <div class="card">
      <h3>Tandem J–V Simulation</h3>
      <div class="form-grid">
        <label class="form-group" style="grid-column: 1 / -1;">
          <span>Tandem preset</span>
          <select id="tandem-config-select" style="width:100%;">
            <option value="">Loading…</option>
          </select>
        </label>
        ${numField('tandem-N', 'N<sub>grid</sub>', 40, '1')}
        ${numField('tandem-np', 'V sample points', 15, '1')}
      </div>
      <div class="actions">
        <button class="btn btn-primary" id="btn-tandem">Run tandem J–V</button>
        <span class="status" id="status-tandem"></span>
      </div>
      <div id="progress-tandem"></div>
      <div id="tandem-stack-viz-container" style="margin: 12px 0;"></div>
      <div id="tandem-results"></div>
      <div class="pane-hint">
        Tandem runs stream progress for each sub-cell sweep (top then bottom).
        Results appear below once series-matching completes.
      </div>
    </div>`

  const vizContainer = container.querySelector<HTMLDivElement>('#tandem-stack-viz-container')!
  const stackViz = mountTandemStackVisualizer(vizContainer)

  const progressEl = container.querySelector<HTMLDivElement>('#progress-tandem')!
  const progressBar: ProgressBarHandle = createProgressBar(progressEl)

  const btn = container.querySelector<HTMLButtonElement>('#btn-tandem')!
  const resultsEl = container.querySelector<HTMLDivElement>('#tandem-results')!
  const selectEl = container.querySelector<HTMLSelectElement>('#tandem-config-select')!

  listConfigs()
    .then((entries: ConfigEntry[]) => {
      const tandems = entries.filter(e => (e.device_type ?? '').startsWith('tandem'))
      if (tandems.length === 0) {
        selectEl.innerHTML = '<option value="">(no tandem presets found)</option>'
        return
      }
      selectEl.innerHTML = tandems
        .map(e => `<option value="configs/${e.name}">${e.name.replace(/\.ya?ml$/, '')}</option>`)
        .join('')
      const preferred = tandems.find(e => e.name.startsWith('tandem_lin2019'))
      if (preferred) selectEl.value = `configs/${preferred.name}`
    })
    .catch((e: unknown) => {
      const msg = e instanceof Error ? e.message : String(e)
      selectEl.innerHTML = `<option value="">(error: ${msg})</option>`
    })

  btn.addEventListener('click', async () => {
    const configPath = selectEl.value.trim()
    if (!configPath) {
      setStatus('status-tandem', 'Select a tandem preset first.', true)
      return
    }

    const params = {
      N_grid: Math.max(3, Math.round(readNum('tandem-N', 40))),
      n_points: Math.max(2, Math.round(readNum('tandem-np', 15))),
    }

    btn.disabled = true
    stackViz.clear()
    resultsEl.innerHTML = ''
    progressBar.reset()
    setStatus('status-tandem', 'Dispatching tandem job…')

    try {
      const jobId = await startJob('tandem', null, params, configPath)
      setStatus('status-tandem', 'Running tandem simulation…')

      streamJobEvents<TandemJVPayload>(jobId, {
        onProgress: (ev) => progressBar.update(ev),
        onResult: (data) => {
          setStatus('status-tandem', 'Done')
          if (data.top_layers || data.bot_layers) {
            stackViz.update(data.top_layers ?? [], data.bot_layers ?? [])
          } else {
            stackViz.clear()
          }
          const pre = document.createElement('pre')
          pre.className = 'tandem-metrics-pre'
          pre.style.cssText = 'font-size:0.85rem;line-height:1.5;background:var(--surface-1,#1e1e2e);padding:12px;border-radius:6px;overflow:auto;'
          pre.textContent = renderMetrics(data)
          resultsEl.innerHTML = ''
          resultsEl.appendChild(pre)
        },
        onError: (msg) => {
          setStatus('status-tandem', `Error: ${msg}`, true)
          resultsEl.innerHTML = `<pre class="tandem-metrics-pre" style="color:var(--error,#f38ba8);">${msg}</pre>`
        },
        onDone: () => {
          btn.disabled = false
        },
      })
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e)
      setStatus('status-tandem', `Error: ${msg}`, true)
      resultsEl.innerHTML = `<pre class="tandem-metrics-pre" style="color:var(--error,#f38ba8);">${msg}</pre>`
      btn.disabled = false
    }
  })
}
