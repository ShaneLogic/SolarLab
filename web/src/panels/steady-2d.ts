import Plotly from 'plotly.js-basic-dist-min'
import { baseLayout, axisTitle, plotConfig, PALETTE } from '../plot-theme'
import { Steady2DClient } from '../steady-2d-client'
import type { Steady2DCatalog, Steady2DJob, Steady2DResult, Steady2DRoute } from '../steady-2d-client'
import type { RunSubscription, TerminalRecord } from '../run-client'

export interface StoredSteady2DTarget { storeId: string; runId: string; attemptId: string }
export interface Steady2DPanel {
  readonly ready: Promise<void>; readonly completion: Promise<void>
  readonly job: Steady2DJob | undefined; readonly result: Steady2DResult | undefined
  reconnect(): Promise<void>; openStored(target: StoredSteady2DTarget): Promise<void>; dispose(): void
}

function element<K extends keyof HTMLElementTagNameMap>(tag: K, text?: string) {
  const node = document.createElement(tag); if (text !== undefined) node.textContent = text; return node
}
const message = (error: unknown) => error instanceof Error ? error.message : String(error)
const terminal = (state: string) => ['succeeded', 'failed', 'cancelled'].includes(state)

/** Installed full-2D steady fields on the existing catalog, owner and exact-attempt reader. */
export function mountSteady2DPanel(root: HTMLElement, endpoint: string | URL, initial?: StoredSteady2DTarget): Steady2DPanel {
  const client = new Steady2DClient(endpoint)
  let catalog: Steady2DCatalog | undefined, job: Steady2DJob | undefined, result: Steady2DResult | undefined
  let subscription: RunSubscription | undefined, phase = 'loading', epoch = 0, disposed = false
  let abort = new AbortController(), completion = Promise.resolve()
  const displayedEvents = new Set<string>(), plots: HTMLElement[] = []
  const card = element('section'); card.className = 'card'; card.dataset.panel = 'steady-2d'
  card.append(element('h2', '2D steady fields · new core'), element('p',
    'Uniform no-grain device: 13 stack × 5 lateral nodes, 195 coordinates in the full 2D Newton solve. Fixed 2 mV bias with explicit dark/G1 targets. Scientific qualification: false; physical and mesh status: not_evaluated. PV metrics are unavailable.'))
  const form = element('form'), fields = element('fieldset'); fields.style.cssText = 'border:0;padding:0'
  const grid = element('div'); grid.className = 'form-grid'; fields.append(grid)
  function control<T extends HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>(name: string, label: string, input: T, parent: HTMLElement = grid): T {
    const wrap = element('label'); wrap.className = 'form-group'
    input.name = name; input.setAttribute('aria-label', label); wrap.append(element('span', label), input); parent.append(wrap); return input
  }
  const routeSelect = control('route', 'Installed 2D steady protocol', element('select'))
  const device = control('device', 'Installed device', element('select'))
  const generation = control('generation', 'Requested generation targets', element('select'))
  const protocolText = element('p'); protocolText.dataset.role = 'installed-protocol'
  const policy = element('details'), policyText = element('pre'); policyText.style.cssText = 'white-space:pre-wrap;max-height:300px;overflow:auto'
  policy.append(element('summary', 'Installed geometry, cold initialization and full controls'), policyText)
  const bounds = element('p'), run = element('button', 'Run 2D steady'); run.type = 'submit'; run.className = 'btn btn-primary'
  fields.append(protocolText, policy, bounds, run); fields.disabled = true; form.append(fields)
  function button(text: string) { const b = element('button', text); b.type = 'button'; b.className = 'btn btn-ghost'; return b }
  const refresh = button('Refresh installed devices'), cancel = button('Cancel this attempt'), reconnect = button('Reconnect / refresh attempt')
  cancel.disabled = reconnect.disabled = true
  const state = element('p', 'Loading installed 2D steady catalog…'); state.dataset.role = 'state'; state.setAttribute('role', 'status')
  const transport = element('p'); transport.dataset.role = 'transport'; transport.setAttribute('role', 'status')
  const error = element('p'); error.className = 'status error'; error.setAttribute('role', 'alert')
  const identity = element('p'); identity.dataset.role = 'attempt'; identity.style.overflowWrap = 'anywhere'
  const link = element('a', 'Read this attempt after reload'); link.hidden = true; link.dataset.role = 'readback-link'
  const readForm = element('form'), readBox = element('details'); readBox.append(element('summary', 'Open a stored attempt without a new solve'))
  const readGrid = element('div'); readGrid.className = 'form-grid'; readForm.append(readGrid)
  const stored = {} as Record<keyof StoredSteady2DTarget, HTMLInputElement>
  for (const [name, label] of [['storeId', 'Store ID'], ['runId', 'Run ID'], ['attemptId', 'Attempt ID']] as const) {
    const input = element('input'); input.required = true; stored[name] = control(name, label, input, readGrid)
  }
  const open = element('button', 'Open stored attempt'); open.type = 'submit'; open.className = 'btn btn-ghost'; readForm.append(open); readBox.append(readForm)
  const history = element('ul'); history.dataset.role = 'history'
  const eventDetails = element('details'); eventDetails.append(element('summary', 'Persisted run events'), history)
  const output = element('div'); output.dataset.role = 'result'
  card.append(form, refresh, cancel, reconnect, state, transport, error, identity, link, readBox, eventDetails, output)
  root.replaceChildren(card)

  function update(next: string) {
    if (terminal(phase) && next !== phase || phase === 'running' && next === 'queued') return
    phase = next; card.dataset.state = next; state.textContent = next
    const busy = ['submitting', 'queued', 'running'].includes(next)
    fields.disabled = busy || !catalog?.routes.length; refresh.disabled = busy
    open.disabled = busy; cancel.disabled = !job || !['queued', 'running'].includes(next)
  }
  function selected(): Steady2DRoute | undefined { return catalog?.routes[Number(routeSelect.value)] }
  function devices() {
    const route = selected(); device.replaceChildren(); generation.replaceChildren()
    for (const d of route?.devices ?? []) { const option = element('option', d.id); option.value = d.id; device.append(option) }
    if (route?.devices.some(d => d.id === 'research_2d_uniform')) device.value = 'research_2d_uniform'
    run.disabled = !route?.devices.length
    if (!route) return
    route.generations.forEach((targets, i) => {
      const option = element('option', targets.map(g => g === 0 ? 'Dark (G = 0)' : 'G1 (G = 1)').join(' → '))
      option.value = String(i); generation.append(option)
    })
    generation.value = String(route.generations.findIndex(g => g.length === 2))
    protocolText.textContent = '13 × 5 nodes · 195 full-2D coordinates · 2 mV · 1 µm lateral width · insulating lateral boundaries · no grain boundaries.'
    bounds.textContent = `Whole-job wall limit: ${catalog!.ceiling.wall_seconds} s. Each requested target uses the installed cold auxiliary preparation followed by a full 2D solve. Auxiliary preparation is reported separately from accepted 2D states.`
    policyText.textContent = JSON.stringify(route.input, null, 2)
  }
  generation.addEventListener('change', () => {
    const route = selected(); if (route) policyText.textContent = JSON.stringify({ ...route.input, generation_scales: route.generations[Number(generation.value)] }, null, 2)
  })
  routeSelect.addEventListener('change', devices)
  async function discover() {
    fields.disabled = true; error.textContent = ''
    try {
      const found = await client.discover(abort.signal); if (disposed) return
      catalog = found; routeSelect.replaceChildren()
      found.routes.forEach((r, i) => { const option = element('option', `${r.id} · version ${r.version}`); option.value = String(i); routeSelect.append(option) })
      devices(); fields.disabled = !found.routes.length
      if (!job) { phase = 'ready'; update(found.routes.length ? 'ready' : 'unavailable') }
      if (!found.routes.length) error.textContent = 'No compatible no-grain 2D steady protocol is installed on this service.'
    } catch (failure) {
      if (!disposed) { error.textContent = `Installed 2D steady service unavailable: ${message(failure)}`; if (!job) update('unavailable') }
    }
  }
  refresh.addEventListener('click', () => { void discover() })
  function purge() { for (const plot of plots.splice(0)) Plotly.purge(plot); output.replaceChildren() }
  function selectJob(active: Steady2DJob) {
    job = active; reconnect.disabled = false
    const target = { storeId: active.reader.storeId, ...active.target }
    for (const key of Object.keys(stored) as (keyof StoredSteady2DTarget)[]) stored[key].value = target[key]
    identity.textContent = `Store ${target.storeId} · run ${target.runId} · attempt ${target.attemptId}`
    const url = new URL(location.href); url.search = new URLSearchParams({ view: 'steady-2d', ...target }).toString()
    link.href = url.href; link.hidden = false
  }

  async function render(data: Steady2DResult, active: Steady2DJob) {
    purge()
    const a = data.arrays, count = a.accepted_indices.values.length, nx = a.x_m.values.length, ny = a.y_m.values.length
    const summary = element('p', `${count} accepted of ${data.targets.length} requested targets · ${ny} × ${nx} nodes · ${data.registered ? 'Recorded HTTP attempt' : 'Historical process attempt; original source and failure retained'}`)
    summary.dataset.role = 'history-summary'; output.append(summary)
    const provenance = element('p', `Original computation source: ${String((data.worker.capture as Record<string, unknown>).build_source_commit)} · geometry ${String(data.summary.geometry_identity)}`)
    provenance.dataset.role = 'provenance'; provenance.style.overflowWrap = 'anywhere'; output.append(provenance)
    const counts = element('p', `Computation counts: ${JSON.stringify(data.summary.computation_counts)}. Scientific qualification: false; physical and mesh status: not_evaluated. PV metrics: unavailable.`)
    counts.dataset.role = 'counts'; counts.style.overflowWrap = 'anywhere'; output.append(counts)
    const failure = data.summary.first_failure as Record<string, unknown> | null
    if (failure) {
      const target = failure.target as Record<string, unknown>
      const failed = element('p', `First failed target: ${String(target.voltage_V)} V, G = ${String(target.generation_scale)} (index ${String(failure.index)}) · ${String(failure.termination)}. Candidate fields remain unaccepted in the original diagnostics.`)
      failed.dataset.role = 'first-failure'; output.append(failed)
    }
    const table = element('table'), thead = element('thead'), tbody = element('tbody'), head = element('tr'); table.className = 'param-table mode-table'
    table.dataset.role = 'requested-targets'
    for (const label of ['Bias (V)', 'Generation scale', 'Stored outcome', 'Lower / upper current per depth (A/m)', 'Lower / upper mean current density (A/m²)']) head.append(element('th', label))
    thead.append(head); table.append(thead, tbody)
    const format = (v: number) => v.toPrecision(7)
    data.targets.forEach((target, i) => {
      const row = element('tr'), outcome = i < count ? 'Accepted' : failure && failure.index === i ? 'Failed' : 'Not attempted'
      row.dataset.outcome = outcome
      const current = (name: string) => i < count ? a[name].values.slice(i * 2, i * 2 + 2).map(format).join(' / ') : 'Unavailable'
      for (const value of [target.voltage_V, target.generation_scale, outcome, current('terminal_current_per_depth_A_m'), current('terminal_mean_current_density_A_m2')]) row.append(element('td', String(value)))
      tbody.append(row)
    })
    const tableBox = element('div'); tableBox.style.overflowX = 'auto'; tableBox.append(table); output.append(tableBox)
    output.append(element('p', 'Positions are shown in nm (stored metres × 10⁹), with lateral x and stack y axes. Node maps show actual accepted values without interpolation. Stored current signs are preserved in the positive coordinate direction; integrated currents are A/m of out-of-plane depth, and mean terminal densities are divided by the recorded 1 µm lateral width.'))
    const docs = () => {
      for (const [label, content] of [
        ['Recorded source, geometry, request and runtime identities', () => JSON.stringify(data.worker, null, 2)],
        ['All 22 logical arrays and original units', () => JSON.stringify({ units: data.units, arrays: data.arrays }, null, 2)],
        ['Execution manifest', () => data.manifestJson],
        ['Original auxiliary and full-2D diagnostics (failed candidate retained)', () => data.diagnosticsJson],
        ['Original stored attempt', () => data.originalJson],
      ] as const) {
        const details = element('details'), pre = element('pre'); pre.style.cssText = 'white-space:pre-wrap;overflow-wrap:anywhere;max-height:400px;overflow:auto'
        details.append(element('summary', label), pre)
        details.addEventListener('toggle', () => { if (details.open && !pre.textContent) pre.textContent = content() })
        output.append(details)
      }
    }
    if (!count) { output.append(element('p', 'No accepted 2D fields or currents are available. The failed candidate is not plotted as a solution.')); docs(); return }
    const selectors = element('div'); selectors.className = 'form-grid'; output.append(selectors)
    const sample = control('accepted_target', 'Accepted target shown in maps', element('select'), selectors)
    data.targets.slice(0, count).forEach((target, i) => { const option = element('option', `${String(target.voltage_V)} V · G = ${String(target.generation_scale)}`); option.value = String(i); sample.append(option) })
    const plotGrid = element('div'); plotGrid.style.cssText = 'display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,390px),1fr));gap:12px'; output.append(plotGrid)
    function plot(title: string) {
      const section = element('section'); section.className = 'card'; section.style.minWidth = '0'; section.append(element('h3', title))
      const div = element('div'); div.style.height = '340px'; div.dataset.plot = title; section.append(div); plotGrid.append(section); plots.push(div); return div
    }
    const potential = plot('2D potential'), electrons = plot('2D electron density'), holes = plot('2D hole density')
    const faces = plot('Directed face current density'), sections = plot('Terminal and cross-section currents')
    const x = a.x_m.values.map(v => v * 1e9), y = a.y_m.values.map(v => v * 1e9)
    const nodeX = y.flatMap(() => x), nodeY = y.flatMap(v => x.map(() => v))
    const mapLayout = (unit: string) => baseLayout({ xaxis: { title: axisTitle('Lateral x (nm)') },
      yaxis: { title: axisTitle('Stack y (nm)'), scaleanchor: 'x', scaleratio: 1 }, margin: { l: 60, r: 90, t: 15, b: 55 },
      showlegend: false, meta: { value_unit: unit } })
    const config = plotConfig(`steady-2d-${active.target.runId}`)
    async function spatialFields(i: number) {
      const row = (name: string, width: number) => a[name].values.slice(i * width, (i + 1) * width)
      const map = (div: HTMLElement, name: string, unit: string) => Plotly.newPlot(div, [{ type: 'scatter', mode: 'markers', x: nodeX, y: nodeY,
        customdata: row(name, nx * ny), marker: { symbol: 'square', size: 12, color: row(name, nx * ny), colorscale: 'Viridis', showscale: true, colorbar: { title: { text: unit }, thickness: 12 } },
        hovertemplate: `x %{x:.6g} nm<br>y %{y:.6g} nm<br>%{customdata:.7g} ${unit}<extra></extra>` }], mapLayout(unit), config)
      const current = row('face_current_density_A_m2', 112), pairs = a.face_pairs.values
      const faceX = current.map((_, j) => (nodeX[pairs[j * 2]] + nodeX[pairs[j * 2 + 1]]) / 2)
      const faceY = current.map((_, j) => (nodeY[pairs[j * 2]] + nodeY[pairs[j * 2 + 1]]) / 2)
      const sectionY = [y[0], ...y.slice(1).map((v, j) => (v + y[j]) / 2), y.at(-1)!]
      const currentProfile = [a.terminal_current_per_depth_A_m.values[i * 2], ...row('cross_section_current_per_depth_A_m', ny - 1), a.terminal_current_per_depth_A_m.values[i * 2 + 1]]
      await Promise.all([
        map(potential, 'potential_V', 'V'), map(electrons, 'electron_density_m3', 'm⁻³'), map(holes, 'hole_density_m3', 'm⁻³'),
        Plotly.newPlot(faces, [{ type: 'scatter', mode: 'markers', x: faceX, y: faceY, customdata: current,
          text: current.map((_, j) => j < 52 ? '+x face' : '+y face'),
          marker: { symbol: current.map((_, j) => j < 52 ? 'diamond' : 'square'), size: 8, color: current, colorscale: 'Viridis', showscale: true, colorbar: { title: { text: 'A/m²' }, thickness: 12 } },
          hovertemplate: '%{text}<br>x %{x:.6g} nm<br>y %{y:.6g} nm<br>%{customdata:.7g} A/m²<extra></extra>' }], mapLayout('A/m²'), config),
        Plotly.newPlot(sections, [{ type: 'scatter', mode: 'lines+markers', x: sectionY, y: currentProfile,
          name: 'Current per depth', line: { color: PALETTE.primary, width: 2 }, marker: { size: 5 } }],
        baseLayout({ xaxis: { title: axisTitle('Stack cross-section y (nm)') }, yaxis: { title: axisTitle('Current per depth (A/m)') } }), config),
      ])
    }
    sample.addEventListener('change', () => { void spatialFields(Number(sample.value)).catch(failure => { if (!disposed) error.textContent = `Plot failed: ${message(failure)}` }) })
    await spatialFields(0); docs()
  }
  async function connect(active: Steady2DJob, token: number) {
    const current = () => !disposed && token === epoch && job === active
    const resume = subscription?.cursor; subscription?.unsubscribe()
    const snapshot = await active.reader.getAttempt(active.target, abort.signal); if (!current()) return
    update(snapshot.value.state)
    async function finish(end: TerminalRecord) {
      if (!current()) return
      update(end.state)
      const worker = end.result.worker_result as Record<string, unknown> | undefined
      if (end.state !== 'succeeded') error.textContent = `${end.state}: ${String(worker?.detail ?? worker?.reason ?? end.result.reason ?? 'stored terminal')}`
      if (!result) {
        const data = await client.readResult(active, abort.signal); if (!current()) return
        if (data) { await render(data, active); if (current()) result = data }
        else output.replaceChildren(element('p', 'No accepted 2D fields were committed for this attempt. Fields and PV metrics are unavailable; physical status: not_evaluated.'))
      }
    }
    if (snapshot.value.terminal) await finish(snapshot.value.terminal)
    if (!current()) return
    subscription = active.reader.subscribe(active.target, async doc => {
      if (!current()) return
      const event = doc.value, sequence = String(event.sequence)
      if (!displayedEvents.has(sequence)) { history.append(element('li', `${sequence} · ${event.kind}`)); displayedEvents.add(sequence) }
      if (event.kind === 'queued' || event.kind === 'running') update(event.kind)
      if (event.kind === 'progress') transport.textContent = 'Worker progress received'
      if (event.kind === 'terminal') await finish(event.payload as TerminalRecord)
    }, resume)
    transport.textContent = 'Following persisted events…'
    const end = await subscription.done
    if (current()) transport.textContent = end.reason === 'disconnected'
      ? 'Event connection lost; reconnect to the same attempt. The job was not cancelled.' : `Event stream: ${end.reason}`
  }
  function reset() {
    ++epoch; subscription?.unsubscribe(); subscription = undefined; abort.abort(); abort = new AbortController()
    job = undefined; result = undefined; phase = 'ready'; purge(); displayedEvents.clear(); history.replaceChildren()
    error.textContent = ''; transport.textContent = ''; link.hidden = true; reconnect.disabled = true
    return epoch
  }
  async function reconnectRun() {
    if (!job || disposed) return
    try { await connect(job, ++epoch) } catch (failure) { if (!disposed) error.textContent = `Read failed: ${message(failure)}` }
  }
  async function openStored(target: StoredSteady2DTarget) {
    if (disposed || ['submitting', 'queued', 'running'].includes(phase)) return
    const token = reset()
    try { const active = client.open(target.storeId.trim(), target.runId.trim(), target.attemptId.trim()); selectJob(active); await connect(active, token) }
    catch (failure) { if (!disposed && token === epoch) { error.textContent = `Read failed: ${message(failure)}`; if (!job) update('ready') } }
  }
  readForm.addEventListener('submit', event => { event.preventDefault(); if (readForm.reportValidity()) completion = openStored({ storeId: stored.storeId.value, runId: stored.runId.value, attemptId: stored.attemptId.value }) })
  reconnect.addEventListener('click', () => { completion = reconnectRun() })
  cancel.addEventListener('click', async () => {
    const active = job; if (!active) return
    cancel.disabled = true
    try { await client.cancel(active); if (!disposed && job === active) transport.textContent = 'Cancellation requested; waiting for the stored terminal.' }
    catch (failure) { if (!disposed) { error.textContent = message(failure); cancel.disabled = terminal(phase) } }
  })
  async function submit() {
    const route = selected(); if (['submitting', 'queued', 'running'].includes(phase) || !catalog || !route || !form.reportValidity()) return
    const token = reset(); update('submitting')
    const runId = `steady-2d-${crypto.randomUUID()}`; identity.textContent = `Submitting run ${runId}`
    try {
      const active = await client.submit(catalog, route, device.value, { ...structuredClone(route.input), generation_scales: [...route.generations[Number(generation.value)]] }, runId)
      if (disposed || token !== epoch) return
      selectJob(active); update('queued'); await connect(active, token)
    } catch (failure) {
      if (!disposed && token === epoch) {
        error.textContent = message(failure)
        if (!job) { phase = 'unconfirmed'; update(phase); transport.textContent = 'Submission was not confirmed; no result or automatic retry is assumed.' }
        else transport.textContent = 'Stored read failed; reconnect to the same attempt.'
      }
    }
  }
  form.addEventListener('submit', event => { event.preventDefault(); completion = submit() })
  const ready = (async () => { await discover(); if (initial && !disposed) await openStored(initial) })()
  return { ready, get completion() { return completion }, get job() { return job }, get result() { return result },
    reconnect: reconnectRun, openStored,
    dispose() { disposed = true; ++epoch; abort.abort(); subscription?.unsubscribe(); purge(); root.replaceChildren() },
  }
}
