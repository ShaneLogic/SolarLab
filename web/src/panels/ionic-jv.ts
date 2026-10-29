import Plotly from 'plotly.js-basic-dist-min'
import { baseLayout, axisTitle, plotConfig, PALETTE } from '../plot-theme'
import { IonicJVClient } from '../ionic-jv-client'
import type { IonicJVCatalog, IonicJVJob, IonicJVResult, IonicJVRoute } from '../ionic-jv-client'
import type { RunSubscription, TerminalRecord } from '../run-client'

export interface StoredIonicJVTarget { storeId: string; runId: string; attemptId: string }
export interface IonicJVPanel {
  readonly ready: Promise<void>; readonly completion: Promise<void>
  readonly job: IonicJVJob | undefined; readonly result: IonicJVResult | undefined
  reconnect(): Promise<void>; openStored(target: StoredIonicJVTarget): Promise<void>; dispose(): void
}

function element<K extends keyof HTMLElementTagNameMap>(tag: K, text?: string) {
  const node = document.createElement(tag); if (text !== undefined) node.textContent = text; return node
}
const message = (error: unknown) => error instanceof Error ? error.message : String(error)
const terminal = (state: string) => ['succeeded', 'failed', 'cancelled'].includes(state)

/** A concrete illuminated ionic steady J-V view on the same owner, catalog and stored reader. */
export function mountIonicJVPanel(root: HTMLElement, endpoint: string | URL, initial?: StoredIonicJVTarget): IonicJVPanel {
  const client = new IonicJVClient(endpoint)
  let catalog: IonicJVCatalog | undefined, job: IonicJVJob | undefined, result: IonicJVResult | undefined
  let subscription: RunSubscription | undefined, phase = 'loading', epoch = 0, disposed = false
  let abort = new AbortController(), completion = Promise.resolve()
  const displayedEvents = new Set<string>(), plots: HTMLElement[] = []
  const card = element('section'); card.className = 'card'; card.dataset.panel = 'ionic-jv'
  card.append(element('h2', 'Illuminated ionic steady J–V · new core'), element('p',
    'Illuminated 1D steady J–V with shared-site positive-diffusivity ions in one blocking domain. Each voltage is a steady state; this is not a time sweep or hysteresis measurement. Physical status: not_evaluated; scientific qualification: false.'))
  const form = element('form'), fields = element('fieldset'); fields.style.cssText = 'border:0;padding:0'
  const grid = element('div'); grid.className = 'form-grid'; fields.append(grid)
  function control<T extends HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>(name: string, label: string, input: T, parent: HTMLElement = grid): T {
    const wrap = element('label'); wrap.className = 'form-group'
    input.name = name; input.setAttribute('aria-label', label); wrap.append(element('span', label), input); parent.append(wrap); return input
  }
  const routeSelect = control('route', 'Installed ionic J-V protocol', element('select'))
  const device = control('device', 'Installed device', element('select'))
  const protocolText = element('p'); protocolText.dataset.role = 'installed-protocol'
  const policy = element('details'), policyText = element('pre'); policyText.style.cssText = 'white-space:pre-wrap;max-height:300px;overflow:auto'
  policy.append(element('summary', 'Explicit voltage samples, initialization and full controls'), policyText)
  const bounds = element('p'), run = element('button', 'Run ionic J–V'); run.type = 'submit'; run.className = 'btn btn-primary'
  fields.append(protocolText, policy, bounds, run); fields.disabled = true; form.append(fields)
  function button(text: string) { const b = element('button', text); b.type = 'button'; b.className = 'btn btn-ghost'; return b }
  const refresh = button('Refresh installed devices'), cancel = button('Cancel this attempt'), reconnect = button('Reconnect / refresh attempt')
  cancel.disabled = reconnect.disabled = true
  const state = element('p', 'Loading installed ionic J-V catalog…'); state.dataset.role = 'state'; state.setAttribute('role', 'status')
  const transport = element('p'); transport.dataset.role = 'transport'; transport.setAttribute('role', 'status')
  const error = element('p'); error.className = 'status error'; error.setAttribute('role', 'alert')
  const identity = element('p'); identity.dataset.role = 'attempt'; identity.style.overflowWrap = 'anywhere'
  const link = element('a', 'Read this attempt after reload'); link.hidden = true; link.dataset.role = 'readback-link'
  const readForm = element('form'), readBox = element('details'); readBox.append(element('summary', 'Open a stored attempt without a new solve'))
  const readGrid = element('div'); readGrid.className = 'form-grid'; readForm.append(readGrid)
  const stored = {} as Record<keyof StoredIonicJVTarget, HTMLInputElement>
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
  function selected(): IonicJVRoute | undefined { return catalog?.routes[Number(routeSelect.value)] }
  function devices() {
    const route = selected(); device.replaceChildren()
    for (const d of route?.devices ?? []) { const option = element('option', d.id); option.value = d.id; device.append(option) }
    if (route?.devices.some(d => d.id === 'research_1d_shared')) device.value = 'research_1d_shared'
    run.disabled = !route?.devices.length
    if (!route) return
    protocolText.textContent = `${route.input.voltage_V.length} explicit samples: ${route.input.voltage_V[0]}–${route.input.voltage_V.at(-1)} V; generation scale ${route.input.generation_scale}; incident power unavailable. Shared-site positive and negative ions; blocking domain ${route.domain}.`
    bounds.textContent = `${route.input.intervals_per_layer.join(', ')} intervals per layer (37 electrical / 13 ion nodes). Whole-job wall limit: ${catalog!.ceiling.wall_seconds} s. Protocol, initial policy and numerical controls are fixed by this installation.`
    policyText.textContent = JSON.stringify(route.input, null, 2)
  }
  routeSelect.addEventListener('change', devices)
  async function discover() {
    fields.disabled = true; error.textContent = ''
    try {
      const found = await client.discover(abort.signal); if (disposed) return
      catalog = found; routeSelect.replaceChildren()
      found.routes.forEach((r, i) => { const option = element('option', `${r.id} · version ${r.version}`); option.value = String(i); routeSelect.append(option) })
      devices(); fields.disabled = !found.routes.length
      if (!job) { phase = 'ready'; update(found.routes.length ? 'ready' : 'unavailable') }
      if (!found.routes.length) error.textContent = 'No compatible illuminated shared-site ionic J-V protocol is installed on this service.'
    } catch (failure) {
      if (!disposed) { error.textContent = `Installed ionic J-V service unavailable: ${message(failure)}`; if (!job) update('unavailable') }
    }
  }
  refresh.addEventListener('click', () => { void discover() })
  function purge() { for (const plot of plots.splice(0)) Plotly.purge(plot); output.replaceChildren() }
  function selectJob(active: IonicJVJob) {
    job = active; reconnect.disabled = false
    const target = { storeId: active.reader.storeId, ...active.target }
    for (const key of Object.keys(stored) as (keyof StoredIonicJVTarget)[]) stored[key].value = target[key]
    identity.textContent = `Store ${target.storeId} · run ${target.runId} · attempt ${target.attemptId}`
    const url = new URL(location.href); url.search = new URLSearchParams({ view: 'ionic-jv', ...target }).toString()
    link.href = url.href; link.hidden = false
  }

  async function render(data: IonicJVResult, active: IonicJVJob) {
    purge()
    const a = data.arrays, v = a.voltage_V.values, requested = a.requested_voltage_V.values
    const x = a.nodes_m.values, xi = a.ion_nodes_m.values, count = v.length
    const summary = element('p', `${count} accepted of ${requested.length} requested voltage samples · ${x.length} electrical / ${xi.length} ionic nodes · ${data.registered ? String(data.worker.registered_device_id) : 'Historical process attempt; original identity retained'}`)
    summary.dataset.role = 'history-summary'; output.append(summary)
    const metrics = data.summary.metrics as Record<string, unknown>, metricBox = element('div'); metricBox.dataset.role = 'metrics'
    metricBox.append(element('h3', 'PV metrics from accepted samples'))
    const metricTable = element('dl'); metricTable.className = 'form-grid'
    for (const [key, label, unit] of [
      ['short_circuit_current_A_m2', 'Short-circuit output current density', 'A/m²'], ['open_circuit_voltage_V', 'Open-circuit voltage', 'V'],
      ['maximum_power_W_m2', 'Maximum power density', 'W/m²'], ['fill_factor', 'Fill factor', 'fraction'], ['efficiency', 'PCE', 'fraction'],
    ]) {
      const value = metrics[key], row = element('div'); row.dataset.metric = key
      row.append(element('dt', label), element('dd', typeof value === 'number' && Number.isFinite(value) ? `${value.toPrecision(7)} ${unit}` : 'Unavailable'))
      metricTable.append(row)
    }
    metricBox.append(metricTable, element('p', `PCE is unavailable without incident irradiance. Stored metric limitation: ${String(metrics.limitation)}. Piecewise-linear accepted samples; voltage-grid qualification: not_evaluated.`))
    output.append(metricBox)
    const counts = element('p', `Computation counts: ${JSON.stringify(data.summary.computation_counts)}. Scientific qualification: false; physical status: not_evaluated.`)
    counts.dataset.role = 'counts'; counts.style.overflowWrap = 'anywhere'; output.append(counts)
    const failure = data.summary.first_failure as Record<string, unknown> | null
    if (failure || !count) {
      const failed = element('p', failure ? `First failed voltage: ${String(failure.voltage_V)} V (sample ${String(failure.index)}) · ${String(failure.termination)}. ${String(failure.error ?? '')} Only accepted samples are plotted.`
        : `Initialization failed: ${String(data.summary.reason)}. No accepted curve is available.`)
      failed.dataset.role = 'first-failure'; output.append(failed)
    }
    const requestedBox = element('details'); requestedBox.dataset.role = 'requested-protocol'
    requestedBox.append(element('summary', 'Requested, accepted, failed and unattempted voltage samples'))
    const table = element('table'), thead = element('thead'), tbody = element('tbody'), head = element('tr'); table.className = 'param-table mode-table'
    for (const label of ['Voltage (V)', 'Generation scale', 'Stored outcome']) head.append(element('th', label))
    thead.append(head); table.append(thead, tbody)
    requested.forEach((voltage, i) => {
      const row = element('tr'), outcome = i < count ? 'Accepted' : failure && failure.index === i ? 'Failed' : 'Not attempted'
      row.dataset.outcome = outcome
      for (const value of [voltage, data.summary.generation_scale, outcome]) row.append(element('td', String(value)))
      tbody.append(row)
    })
    requestedBox.append(table); output.append(requestedBox)
    output.append(element('p', 'Stored current signs are preserved: J–V uses output-positive current; spatial terms retain the producer direction. Electronic, ionic and total currents are steady conduction. No displacement or hysteresis result is inferred.'))
    const docs = () => {
      for (const [label, content] of [
        ['Recorded source, compiled, initial and runtime identities', () => JSON.stringify(data.worker, null, 2)],
        ['Logical arrays, units and ion-domain mapping', () => JSON.stringify({ units: data.units, arrays: data.arrays }, null, 2)],
        ['Execution manifest', () => data.manifestJson], ['Exact initial evidence', () => data.initialJson],
        ['Raw original attempts (failure tags preserved)', () => data.diagnosticsJson], ['Original stored attempt', () => data.originalJson],
      ] as const) {
        const details = element('details'), pre = element('pre'); pre.style.cssText = 'white-space:pre-wrap;overflow-wrap:anywhere;max-height:400px;overflow:auto'
        details.append(element('summary', label), pre)
        details.addEventListener('toggle', () => { if (details.open && !pre.textContent) pre.textContent = content() })
        output.append(details)
      }
    }
    if (!count) { docs(); return }
    const selectors = element('div'); selectors.className = 'form-grid'; output.append(selectors)
    const sample = control('accepted_voltage', 'Spatial fields at accepted voltage (V)', element('select'), selectors)
    v.forEach((value, i) => { const option = element('option', String(value)); option.value = String(i); sample.append(option) }); sample.value = '0'
    function plot(title: string) {
      const section = element('section'); section.className = 'card'; section.append(element('h3', title))
      const div = element('div'); div.style.height = '310px'; div.dataset.plot = title; section.append(div); output.append(section); plots.push(div); return div
    }
    const jv = plot('Accepted J–V samples'), inventories = plot('Independent species inventories'), terminals = plot('Steady terminal current terms')
    const potential = plot('Potential in space'), carriers = plot('Carrier densities in space'), ions = plot('Ion densities in space')
    const cells = plot('Species cell inventories in space'), currents = plot('Steady currents in space')
    const colors = [PALETTE.primary, PALETTE.secondary, PALETTE.accent]
    const line = (x: number[], y: number[], name: string, color = PALETTE.primary) =>
      ({ type: 'scatter', mode: 'lines+markers', x, y, name, line: { color, width: 2 }, marker: { size: 4 } })
    function draw(div: HTMLElement, traces: Record<string, unknown>[], xlabel: string, ylabel: string) {
      return Plotly.newPlot(div, traces, baseLayout({ xaxis: { title: axisTitle(xlabel) }, yaxis: { title: axisTitle(ylabel) } }), plotConfig(`ionic-jv-${active.target.runId}`))
    }
    const kinds = ['electronic', 'ionic', 'total']
    async function spatialFields(i: number) {
      const row = (name: string, width: number) => a[name].values.slice(i * width, (i + 1) * width)
      const faces = [x[0], ...x.slice(1).map((value, j) => (value + x[j]) / 2), x.at(-1)!]
      await Promise.all([
        draw(potential, [line(x, row('potential_V', x.length), 'Potential')], 'Position (m)', 'Potential (V)'),
        draw(carriers, [line(x, row('electron_density_m3', x.length), 'Electrons'), line(x, row('hole_density_m3', x.length), 'Holes', PALETTE.secondary)], 'Position (m)', 'Density (m⁻³)'),
        draw(ions, data.species.map((s, k) => line(xi, row(s + '_ion_density_m3', xi.length), s, colors[k])), 'Ion-domain position (m)', 'Ion density (m⁻³)'),
        draw(cells, data.species.map((s, k) => line(xi, row(s + '_ion_cell_inventory_m2', xi.length), s, colors[k])), 'Ion-domain position (m)', 'Cell inventory (m⁻²)'),
        draw(currents, kinds.map((s, k) => line(faces, row(s + '_current_density_A_m2', faces.length), s + ' conduction', colors[k])), 'Face / port position (m)', 'Current density (A/m²)'),
      ])
    }
    sample.addEventListener('change', () => { void spatialFields(Number(sample.value)).catch(failure => { if (!disposed) error.textContent = `Plot failed: ${message(failure)}` }) })
    await Promise.all([
      draw(jv, [line(v, a.output_current_A_m2.values, 'Accepted output current')], 'Accepted voltage (V)', 'Output current density (A/m²)'),
      draw(inventories, data.species.map((s, k) => line(v, a[s + '_ion_inventory_m2'].values, s, colors[k])), 'Accepted voltage (V)', 'Domain inventory (m⁻²)'),
      draw(terminals, kinds.flatMap((s, k) => [0, 1].map(side => ({ ...line(v, v.map((_, i) => a['terminal_' + s + '_current_density_A_m2'].values[i * 2 + side]), `${side ? 'Right' : 'Left'} ${s}`, colors[k]), line: { color: colors[k], dash: side ? 'dash' : 'solid' } }))), 'Accepted voltage (V)', 'Current density (A/m²)'),
      spatialFields(0),
    ])
    docs()
  }
  async function connect(active: IonicJVJob, token: number) {
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
        else output.replaceChildren(element('p', 'No ionic J-V curve was committed for this attempt. Fields and metrics are unavailable; physical status: not_evaluated.'))
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
  async function openStored(target: StoredIonicJVTarget) {
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
    const runId = `ionic-jv-${crypto.randomUUID()}`; identity.textContent = `Submitting run ${runId}`
    try {
      const active = await client.submit(catalog, route, device.value, structuredClone(route.input), runId)
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
