import { runTandem } from '../../api'
import { setStatus, readNum, numField } from '../../ui-helpers'
import { mountTandemStackVisualizer } from '../../stack/tandem-stack-visualizer'
import type { TandemJVPayload } from '../../types'

/**
 * Tandem pane — config-path-based workflow (not device-object-based).
 *
 * Layout:
 *   1. Config path input + N_grid / n_points params + Run button
 *   2. Side-by-side sub-stack visualizer (read-only)
 *   3. Metrics + benchmark comparison rendered as a <pre> block
 *
 * Deviation from plan: uses direct fetch via api.runTandem() rather than the
 * streaming job system (startJob/streamJobEvents). The /api/tandem endpoint is
 * a synchronous POST that returns the full result in one response — there is no
 * SSE stream for tandem yet. Progress bar is therefore omitted in v1.
 *
 * TODO (deferred):
 *   - Wire Plotly J-V curve overlay (top + bottom sub-cells)
 *   - Progress bar via SSE once backend moves tandem to /api/jobs
 *   - Config file browser / dropdown (currently free-text input)
 */

function fmt4(v: number): string {
  return v.toFixed(4)
}

function renderMetrics(data: TandemJVPayload): string {
  const lines: string[] = [
    '=== Tandem J-V Metrics ===',
    `  V_oc  = ${fmt4(data.V_oc)} V`,
    `  J_sc  = ${fmt4(data.J_sc)} mA/cm²`,
    `  FF    = ${fmt4(data.FF)}`,
    `  PCE   = ${fmt4(data.PCE)} %`,
    '',
    '--- Sub-cell operating points ---',
    `  V_top = ${fmt4(data.V_top)} V`,
    `  V_bot = ${fmt4(data.V_bot)} V`,
  ]

  if (data.benchmark) {
    lines.push('')
    lines.push('--- Benchmark comparison ---')
    const b = data.benchmark
    if (b.source) lines.push(`  Source: ${b.source}`)
    if (b.V_oc != null) lines.push(`  V_oc  target = ${fmt4(b.V_oc)} V      simulated = ${fmt4(data.V_oc)} V`)
    if (b.J_sc != null) lines.push(`  J_sc  target = ${fmt4(b.J_sc)} mA/cm²  simulated = ${fmt4(data.J_sc)} mA/cm²`)
    if (b.FF != null)   lines.push(`  FF    target = ${fmt4(b.FF)}            simulated = ${fmt4(data.FF)}`)
    if (b.PCE != null)  lines.push(`  PCE   target = ${fmt4(b.PCE)} %         simulated = ${fmt4(data.PCE)} %`)
  }

  return lines.join('\n')
}

export function mountTandemPane(container: HTMLElement): void {
  container.innerHTML = `
    <div class="card">
      <h3>Tandem J–V Simulation</h3>
      <div class="form-grid">
        <label class="form-group" style="grid-column: 1 / -1;">
          <span>Config path</span>
          <input type="text" id="tandem-config-path" placeholder="configs/tandem_perovskite_si.yaml" style="width:100%;">
        </label>
        ${numField('tandem-N', 'N<sub>grid</sub>', 40, '1')}
        ${numField('tandem-np', 'V sample points', 15, '1')}
      </div>
      <div class="actions">
        <button class="btn btn-primary" id="btn-tandem">Run tandem J–V</button>
        <span class="status" id="status-tandem"></span>
      </div>
      <div id="tandem-stack-viz-container" style="margin: 12px 0;"></div>
      <div id="tandem-results"></div>
      <div class="pane-hint">
        Tandem runs are synchronous — no streaming progress bar in v1.
        Results appear in the panel below once the simulation completes.
      </div>
    </div>`

  const vizContainer = container.querySelector<HTMLDivElement>('#tandem-stack-viz-container')!
  const stackViz = mountTandemStackVisualizer(vizContainer)

  const btn = container.querySelector<HTMLButtonElement>('#btn-tandem')!
  const resultsEl = container.querySelector<HTMLDivElement>('#tandem-results')!

  btn.addEventListener('click', () => {
    const configPathEl = container.querySelector<HTMLInputElement>('#tandem-config-path')!
    const configPath = configPathEl.value.trim()
    if (!configPath) {
      setStatus('status-tandem', 'Enter a config path first.', true)
      return
    }

    const N_grid = Math.max(3, Math.round(readNum('tandem-N', 40)))
    const n_points = Math.max(2, Math.round(readNum('tandem-np', 15)))

    btn.disabled = true
    stackViz.clear()
    resultsEl.innerHTML = ''
    setStatus('status-tandem', 'Running tandem simulation…')

    runTandem(configPath, N_grid, n_points)
      .then((data: TandemJVPayload) => {
        setStatus('status-tandem', 'Done')

        // Update sub-stack visualizer if layer data is present.
        if (data.top_layers || data.bot_layers) {
          stackViz.update(data.top_layers ?? [], data.bot_layers ?? [])
        } else {
          stackViz.clear()
        }

        // Render metrics as pre block.
        const pre = document.createElement('pre')
        pre.className = 'tandem-metrics-pre'
        pre.style.cssText = 'font-size:0.85rem;line-height:1.5;background:var(--surface-1,#1e1e2e);padding:12px;border-radius:6px;overflow:auto;'
        pre.textContent = renderMetrics(data)
        resultsEl.innerHTML = ''
        resultsEl.appendChild(pre)
      })
      .catch((e: unknown) => {
        const msg = e instanceof Error ? e.message : String(e)
        setStatus('status-tandem', `Error: ${msg}`, true)
        resultsEl.innerHTML = `<pre class="tandem-metrics-pre" style="color:var(--error,#f38ba8);">${msg}</pre>`
      })
      .finally(() => {
        btn.disabled = false
      })
  })
}
