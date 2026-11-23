import { listConfigs } from '../../v2-workstation-client'
import { startJob, streamJobEvents } from '../../v2-job-client'
import { createProgressBar, type ProgressBarHandle } from '../../progress'
import { setStatus, readNum, numField } from '../../ui-helpers'
import { mountTandemStackVisualizer } from '../../stack/tandem-stack-visualizer'
import type { TandemJVPayload, ConfigEntry } from '../../types'
import { extractExecutionPhysics, formatQualificationBadge } from '../execution-physics'

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

function isTandemVocValid(data: TandemJVPayload): boolean {
  const m = data.metrics
  return (
    m.voc_bracketed !== false &&
    Number.isFinite(m.V_oc) &&
    m.open_circuit_voltage_V !== null
  )
}

function renderMetrics(data: TandemJVPayload): string {
  const m = data.metrics
  const vocValid = isTandemVocValid(data)
  const jscValid = Number.isFinite(m.J_sc)
  const jscDisplay = jscValid ? Math.abs(m.J_sc) / 10 : Number.NaN
  const pceDisplay = vocValid && Number.isFinite(m.PCE) ? m.PCE * 100 : Number.NaN

  const vocSimStr = vocValid ? `${fmt4(m.V_oc)} V` : 'unavailable (not bracketed)'
  const jscSimStr = jscValid ? `${fmt4(jscDisplay)} mA/cm²` : 'unavailable'
  const ffSimStr = vocValid && Number.isFinite(m.FF) ? fmt4(m.FF) : '— (V_oc not bracketed)'
  const pceSimStr = Number.isFinite(pceDisplay) ? `${fmt4(pceDisplay)} %` : '— (V_oc not bracketed)'

  const lines: string[] = [
    '=== Tandem J-V Metrics ===',
    `  V_oc  = ${vocSimStr}`,
    `  J_sc  = ${jscSimStr}`,
    `  FF    = ${ffSimStr}`,
    `  PCE   = ${pceSimStr}`,
    '',
  ]

  if (data.benchmark) {
    lines.push('--- Benchmark comparison ---')
    const b = data.benchmark
    if (b.source) lines.push(`  Source: ${b.source}`)
    if (b.V_oc != null) lines.push(`  V_oc  target = ${fmt4(b.V_oc)} V      simulated = ${vocSimStr}`)
    if (b.J_sc != null) lines.push(`  J_sc  target = ${fmt4(b.J_sc)} mA/cm²  simulated = ${jscSimStr}`)
    if (b.FF != null)   lines.push(`  FF    target = ${fmt4(b.FF)}            simulated = ${ffSimStr}`)
    if (b.PCE != null)  lines.push(`  PCE   target = ${fmt4(b.PCE)} %         simulated = ${pceSimStr}`)
  }

  return lines.join('\n')
}

export function mountTandemPane(container: HTMLElement): void {
  container.innerHTML = `
    <div class="card">
      <h3>Tandem J–V</h3>
      <div class="form-grid">
        <label class="form-group" style="grid-column: 1 / -1;">
          <span>Tandem preset</span>
          <select id="tandem-config-select" style="width:100%;">
            <option value="">Loading…</option>
          </select>
        </label>
        ${numField('tandem-N', 'N<sub>grid</sub>', 40, '1')}
        ${numField('tandem-np', 'Voltage points', 15, '1')}
      </div>
      <div class="actions">
        <button class="btn btn-primary" id="btn-tandem">Run tandem J–V</button>
        <span class="status" id="status-tandem"></span>
      </div>
      <div id="progress-tandem"></div>
      <div id="tandem-stack-viz-container" style="margin: 12px 0;"></div>
      <div id="tandem-results"></div>
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
    progressBar.busy()
    setStatus('status-tandem', 'Dispatching tandem job…')

    try {
      const jobId = await startJob('tandem', null, params, configPath)
      setStatus('status-tandem', 'Running tandem simulation…')

      streamJobEvents<TandemJVPayload & Record<string, unknown>>(jobId, {
        onProgress: (ev) => progressBar.update(ev),
        onResult: (rawData) => {
          const { pure: data, receipt } = extractExecutionPhysics(rawData)
          const qual = formatQualificationBadge(receipt)
          const vocValid = isTandemVocValid(data)
          const diagNotes: string[] = []
          if (!vocValid) {
            diagNotes.push('Tandem V_oc not bracketed or unavailable')
          }
          if ((data.failed_points?.length ?? 0) > 0) {
            const details = data.failed_points!
              .map(fp => `${fp.label ?? `#${fp.index}`} (${fp.reason})`)
              .join('; ')
            diagNotes.push(`Failed sweep points (${data.failed_points!.length}): ${details}`)
          }

          if (diagNotes.length > 0) {
            setStatus('status-tandem', `Done · Diagnostics: ${diagNotes.join(' | ')}`, true)
          } else {
            setStatus('status-tandem', 'Done')
          }
          progressBar.done()

          if (data.top_layers || data.bot_layers) {
            stackViz.update(data.top_layers ?? [], data.bot_layers ?? [])
          } else {
            stackViz.clear()
          }

          resultsEl.replaceChildren()
          const metaBar = document.createElement('div')
          metaBar.className = 'tandem-execution-meta'
          const manifestSpan = document.createElement('span')
          manifestSpan.setAttribute('data-test', 'tandem-manifest-badge')
          manifestSpan.title = receipt.summary
          manifestSpan.textContent = receipt.badgeLabel
          const qualSpan = document.createElement('span')
          qualSpan.setAttribute('data-test', 'tandem-qualification-badge')
          qualSpan.setAttribute('data-qualified', String(qual.qualified))
          qualSpan.title = qual.detail
          qualSpan.textContent = qual.label
          metaBar.append(manifestSpan, document.createTextNode(' · '), qualSpan)
          resultsEl.appendChild(metaBar)

          if (diagNotes.length > 0) {
            const warnBanner = document.createElement('div')
            warnBanner.className = 'jv2d-warning'
            warnBanner.setAttribute('role', 'alert')
            warnBanner.setAttribute('data-test', 'tandem-diagnostic-warning')
            warnBanner.textContent = diagNotes.join(' | ')
            resultsEl.appendChild(warnBanner)
          }

          const pre = document.createElement('pre')
          pre.className = 'tandem-metrics-pre'
          pre.style.cssText = 'font-size:0.85rem;line-height:1.5;background:var(--surface-1,#1e1e2e);padding:12px;border-radius:6px;overflow:auto;'
          pre.textContent = renderMetrics(data)
          resultsEl.appendChild(pre)
        },
        onError: (msg) => {
          progressBar.error(msg)
          setStatus('status-tandem', `Error: ${msg}`, true)
          const errPre = document.createElement('pre')
          errPre.className = 'tandem-metrics-pre'
          errPre.style.color = 'var(--error,#f38ba8)'
          errPre.textContent = msg
          resultsEl.replaceChildren(errPre)
        },
        onDone: () => {
          btn.disabled = false
        },
      })
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e)
      progressBar.error(msg)
      setStatus('status-tandem', `Error: ${msg}`, true)
      const errPre = document.createElement('pre')
      errPre.className = 'tandem-metrics-pre'
      errPre.style.color = 'var(--error,#f38ba8)'
      errPre.textContent = msg
      resultsEl.replaceChildren(errPre)
      btn.disabled = false
    }
  })
}
