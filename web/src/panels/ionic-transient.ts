import Plotly from 'plotly.js-basic-dist-min'
import { baseLayout, axisTitle, plotConfig, PALETTE } from '../plot-theme'
import { IonicTransientClient, parseIonicSamples } from '../ionic-transient-client'
import type { IonicCatalog, IonicJob, IonicResult, IonicRoute, IonicProtocol } from '../ionic-transient-client'
import type { RunSubscription, TerminalRecord } from '../run-client'

export interface StoredIonicTransientTarget { storeId: string; runId: string; attemptId: string }
export interface IonicTransientPanel {
  readonly ready: Promise<void>; readonly completion: Promise<void>
  readonly job: IonicJob | undefined; readonly result: IonicResult | undefined
  reconnect(): Promise<void>; openStored(target: StoredIonicTransientTarget): Promise<void>; dispose(): void
}

function element<K extends keyof HTMLElementTagNameMap>(tag: K, text?: string) {
  const node = document.createElement(tag); if (text !== undefined) node.textContent = text; return node
}
const message = (error: unknown) => error instanceof Error ? error.message : String(error)
const terminal = (state: string) => ['succeeded', 'failed', 'cancelled'].includes(state)

/** A concrete dark blocking ionic view on the same owner, catalog and stored reader. */
export function mountIonicTransientPanel(root: HTMLElement, endpoint: string | URL, initial?: StoredIonicTransientTarget): IonicTransientPanel {
  const client = new IonicTransientClient(endpoint)
  let catalog: IonicCatalog | undefined, job: IonicJob | undefined, result: IonicResult | undefined
  let subscription: RunSubscription | undefined, phase = 'loading', epoch = 0, disposed = false
  let abort = new AbortController(), completion = Promise.resolve()
  const displayedEvents = new Set<string>(), plots: HTMLElement[] = []
  const card = element('section'); card.className = 'card'; card.dataset.panel = 'ionic-transient'
  card.append(element('h2', 'Ionic transient · new core'), element('p',
    'Dark 1D ionic time evolution from a cold ionic steady state. Select an installed positive-diffusivity device with one blocking domain. Scientific qualification is false; physical status is not_evaluated.'))
  const form = element('form'), fields = element('fieldset'); fields.style.cssText = 'border:0;padding:0'
  const grid = element('div'); grid.className = 'form-grid'; fields.append(grid)
  function control<T extends HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>(name: string, label: string, input: T, parent: HTMLElement = grid): T {
    const wrap = element('label'); wrap.className = 'form-group'
    input.name = name; input.setAttribute('aria-label', label); wrap.append(element('span', label), input); parent.append(wrap); return input
  }
  const routeSelect = control('route', 'Installed ionic transient experiment', element('select'))
  const device = control('device', 'Installed device', element('select'))
  const samples = control('samples', 'Samples: time (s), voltage (V), generation scale — one row per sample', element('textarea'), fields)
  samples.rows = 5; samples.required = true; samples.style.cssText = 'width:100%;font-family:monospace;resize:vertical'
  samples.value = '0, .002, 0\n.25, .0025, 0\n.5, .003, 0\n.75, .0035, 0\n1, .004, 0'
  const budgets = element('details'); budgets.append(element('summary', 'Grid and work budgets')); fields.append(budgets)
  const budgetGrid = element('div'); budgetGrid.className = 'form-grid'; budgets.append(budgetGrid)
  const intervals = control('intervals_per_layer', 'Intervals per layer (comma-separated)', element('input'), budgetGrid)
  intervals.value = '12,12,12'; intervals.required = true
  const inputs: Record<string, HTMLInputElement> = {}
  for (const [name, label, value, step] of [
    ['wall_seconds', 'Whole-job wall budget (s)', '30', 'any'],
    ['inventory_row_node', 'Initial inventory constraint node', '12', '1'], ['initial_wall_s', 'Initial steady wall budget (s)', '30', 'any'],
    ['initial_max_iterations', 'Initial steady Newton iterations', '60', '1'], ['initial_max_residual_evaluations', 'Initial residual evaluations', '600', '1'],
    ['initial_max_jacobian_evaluations', 'Initial Jacobian evaluations', '60', '1'], ['initial_max_backtracks', 'Initial backtracks per iteration', '24', '1'],
    ['max_steps', 'Maximum transient steps', '16', '1'], ['wall_s', 'Transient wall budget (s)', '30', 'any'],
    ['max_iterations', 'Newton iterations per step', '50', '1'], ['max_residual_evaluations', 'Residual evaluations per step', '500', '1'],
    ['max_jacobian_evaluations', 'Jacobian evaluations per step', '50', '1'], ['max_backtracks', 'Backtracks per iteration', '24', '1'],
  ]) {
    const input = element('input'); input.type = 'number'; input.value = value; input.step = step; input.required = true
    inputs[name] = control(name, label, input, budgetGrid)
  }
  const bounds = element('p'), run = element('button', 'Run ionic transient'); run.type = 'submit'; run.className = 'btn btn-primary'
  const policy = element('details'), policyText = element('pre'); policyText.style.cssText = 'white-space:pre-wrap;max-height:300px;overflow:auto'
  policy.append(element('summary', 'Installed fixed numerical policy'), policyText); fields.append(policy)
  fields.append(bounds, run); fields.disabled = true; form.append(fields)
  function button(text: string) { const b = element('button', text); b.type = 'button'; b.className = 'btn btn-ghost'; return b }
  const refresh = button('Refresh installed devices'), cancel = button('Cancel this attempt'), reconnect = button('Reconnect / refresh attempt')
  cancel.disabled = reconnect.disabled = true
  const state = element('p', 'Loading installed transient catalog…'); state.dataset.role = 'state'; state.setAttribute('role', 'status')
  const transport = element('p'); transport.dataset.role = 'transport'; transport.setAttribute('role', 'status')
  const error = element('p'); error.className = 'status error'; error.setAttribute('role', 'alert')
  const identity = element('p'); identity.dataset.role = 'attempt'; identity.style.overflowWrap = 'anywhere'
  const link = element('a', 'Read this attempt after reload'); link.hidden = true; link.dataset.role = 'readback-link'
  const readForm = element('form'), readBox = element('details'); readBox.append(element('summary', 'Open a stored attempt without a new solve'))
  const readGrid = element('div'); readGrid.className = 'form-grid'; readForm.append(readGrid)
  const stored = {} as Record<keyof StoredIonicTransientTarget, HTMLInputElement>
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
  function selected(): IonicRoute | undefined { return catalog?.routes[Number(routeSelect.value)] }
  function devices() {
    const route = selected(); device.replaceChildren()
    for (const d of route?.devices ?? []) { const option = element('option', d.id); option.value = d.id; device.append(option) }
    if (route?.devices.some(d => d.id === 'research_1d_shared')) device.value = 'research_1d_shared'
    run.disabled = !route?.devices.length
    if (!route) return
    for (const [key, input] of Object.entries(inputs)) {
      const limit = key === 'wall_seconds' ? { min: .001, max: catalog!.ceiling.wall_seconds } : route.bounds[key as keyof IonicRoute['bounds']]
      input.min = String(key.endsWith('wall_s') ? Math.max(.001, limit.min) : limit.min); input.max = String(limit.max)
      input.value = String(Math.min(Number(input.value), limit.max))
    }
    const b = route.bounds
    bounds.textContent = `${b.samples.min}–${b.samples.max} rows; increasing time ${b.times_s.min}–${b.times_s.max} s; voltage ${b.voltage_V.min}–${b.voltage_V.max} V; generation = 0 (dark). ${b.layers.min}–${b.layers.max} layers, ${b.intervals_per_layer.min}–${b.intervals_per_layer.max} intervals each. Blocking domain: ${route.domain}. Solver tolerances are fixed by the installed service.`
    policyText.textContent = JSON.stringify(route.policy, null, 2)
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
      if (!found.routes.length) error.textContent = 'No compatible dark positive-D ionic transient route is installed on this service.'
    } catch (failure) {
      if (!disposed) { error.textContent = `Installed transient service unavailable: ${message(failure)}`; if (!job) update('unavailable') }
    }
  }
  refresh.addEventListener('click', () => { void discover() })
  function purge() { for (const plot of plots.splice(0)) Plotly.purge(plot); output.replaceChildren() }
  function selectJob(active: IonicJob) {
    job = active; reconnect.disabled = false
    const target = { storeId: active.reader.storeId, ...active.target }
    for (const key of Object.keys(stored) as (keyof StoredIonicTransientTarget)[]) stored[key].value = target[key]
    identity.textContent = `Store ${target.storeId} · run ${target.runId} · attempt ${target.attemptId}`
    const url = new URL(location.href); url.search = new URLSearchParams({ view: 'ionic-transient', ...target }).toString()
    link.href = url.href; link.hidden = false
  }

  async function render(data: IonicResult, active: IonicJob) {
    purge()
    const a = data.arrays, times = a.accepted_times_s.values, x = a.nodes_m.values, xi = a.ion_nodes_m.values, count = times.length
    const summary = element('p', `${count} accepted states of ${a.requested_times_s.values.length} requested · ${Math.max(0, count - 1)} accepted steps · ${x.length} electrical / ${xi.length} ionic nodes · ${String(data.worker.registered_device_id)}`)
    summary.dataset.role = 'history-summary'; output.append(summary)
    const unknown = element('p', 'Voc, fill factor and efficiency: Unknown for this ionic history. Physical status: not_evaluated; scientific qualification: false.')
    unknown.dataset.role = 'metrics'; output.append(unknown)
    const failure = data.attempts.find(attempt => attempt.accepted === false)
    if (failure || !count) {
      const failed = element('p', failure
        ? `First failed step: ${String(failure.start_s)} → ${String(failure.end_s)} s · ${String(failure.termination)}. Only the accepted prefix is plotted.`
        : `Initial steady solve failed: ${String(data.summary.reason)}. No accepted history is available.`)
      failed.dataset.role = 'first-failure'; output.append(failed)
    }
    const requested = element('details'); requested.dataset.role = 'requested-protocol'; requested.open = true
    requested.append(element('summary', 'Stored requested protocol'))
    const table = element('table'), head = element('tr'), thead = element('thead'), tbody = element('tbody')
    table.className = 'param-table mode-table'
    for (const label of ['Time (s)', 'Voltage (V)', 'Generation scale', 'Stored outcome']) head.append(element('th', label))
    thead.append(head); table.append(thead, tbody)
    a.requested_times_s.values.forEach((time, i) => {
      const row = element('tr'), outcome = i === 0 ? count ? 'Initial steady accepted' : 'Initial steady failed'
        : i < count ? 'Accepted' : failure && time === failure.end_s ? 'Failed step' : 'Not attempted'
      for (const value of [time, a.requested_voltage_V.values[i], a.requested_generation_scale.values[i], outcome]) row.append(element('td', String(value)))
      tbody.append(row)
    })
    requested.append(table); output.append(requested)
    output.append(element('p', 'Stored current signs are preserved. Currents use accepted step end times only; all-face arrays include both ports. Each species has its own density and inventory.'))
    const documents = () => {
      for (const [label, text] of [
        ['Logical arrays, units, coordinates and ion-domain mapping', JSON.stringify({ units: data.units, arrays: data.arrays }, null, 2)],
        ['Recorded source, compiled and initial identities', JSON.stringify(data.worker, null, 2)],
        ['Execution manifest', data.manifestJson], ['Cold initial steady evidence', data.initialJson],
        ['Raw ionic diagnostics (nonfinite tags preserved)', data.diagnosticsJson], ['Original stored attempt', data.originalJson],
      ]) {
        const details = element('details'), pre = element('pre', text); pre.style.cssText = 'white-space:pre-wrap;overflow-wrap:anywhere;max-height:400px;overflow:auto'
        details.append(element('summary', label), pre); output.append(details)
      }
    }
    if (!count) { documents(); return }
    if (count === 1) output.append(element('p', 'No transient step was accepted; current histories are unavailable. Initial steady fields remain readable.'))
    const selectors = element('div'); selectors.className = 'form-grid'; output.append(selectors)
    const sample = control('accepted_time', 'Spatial fields at accepted time (s)', element('select'), selectors)
    times.forEach((t, i) => { const option = element('option', String(t)); option.value = String(i); sample.append(option) }); sample.value = String(count - 1)
    const node = control('node', 'Temporal electronic fields at position (m)', element('select'), selectors)
    x.forEach((position, i) => { const option = element('option', position.toPrecision(6)); option.value = String(i); node.append(option) }); node.value = String(Math.floor(x.length / 2))
    const ionNode = control('ion_node', 'Temporal ionic fields at position (m)', element('select'), selectors)
    xi.forEach((position, i) => { const option = element('option', position.toPrecision(6)); option.value = String(i); ionNode.append(option) }); ionNode.value = String(Math.floor(xi.length / 2))
    const currentTime = element('p'); currentTime.dataset.role = 'current-time'; output.append(currentTime)
    function plot(title: string) {
      const section = element('section'); section.className = 'card'; section.append(element('h3', title))
      const div = element('div'); div.style.height = '310px'; div.dataset.plot = title
      section.append(div); output.append(section); plots.push(div); return div
    }
    const terminalCurrent = plot('Terminal currents over time'), potentialTime = plot('Potential over time'), densityTime = plot('Carrier densities over time')
    const ionTime = plot('Ion densities over time'), inventories = plot('Independent species inventories over time')
    const potential = plot('Potential in space'), carriers = plot('Carrier densities in space'), ions = plot('Ion densities in space')
    const cells = plot('Species cell inventories in space'), current = plot('Currents in space')
    const colors = [PALETTE.primary, PALETTE.secondary, '#6c5ce7', PALETTE.accent, PALETTE.neutral]
    const line = (x: number[], y: number[], name: string, color = PALETTE.primary) =>
      ({ type: 'scatter', mode: 'lines+markers', x, y, name, line: { color, width: 2 }, marker: { size: 4 } })
    function draw(div: HTMLElement, traces: Record<string, unknown>[], xlabel: string, ylabel: string) {
      const padding = Math.max((a.requested_times_s.values[1] - times[0]) * .05, Number.MIN_VALUE * 8)
      const timeRange = count > 1 ? [times[0], times[count - 1]] : [times[0] - padding, times[0] + padding]
      return Plotly.newPlot(div, traces, baseLayout({ xaxis: { title: axisTitle(xlabel), ...(xlabel.includes('time') ? { range: timeRange } : {}) },
        yaxis: { title: axisTitle(ylabel) } }), plotConfig(`ionic-transient-${active.target.runId}`))
    }
    const kinds = ['conduction', 'displacement', 'total'] as const
    const terminalTraces = kinds.flatMap((kind, k) => [0, 1].map(side => ({
      ...line(times.slice(1), times.slice(1).map((_, i) => a[`terminal_${kind}_current_density_A_m2`].values[i * 2 + side]),
        `${side === 0 ? 'Left' : 'Right'} ${kind}`, colors[k]), line: { color: colors[k], dash: side ? 'dash' : 'solid' },
    })))
    async function temporalFields(i: number) {
      const column = (name: string) => times.map((_, j) => a[name].values[j * x.length + i])
      await Promise.all([
        draw(potentialTime, [line(times, column('potential_V'), 'Potential')], 'Accepted time (s)', 'Potential (V)'),
        draw(densityTime, [line(times, column('electron_density_m3'), 'Electrons'), line(times, column('hole_density_m3'), 'Holes', PALETTE.secondary)], 'Accepted time (s)', 'Density (m⁻³)'),
      ])
    }
    const ionicFields = (i: number) => draw(ionTime, data.species.map((s, k) =>
      line(times, times.map((_, j) => a[s + '_ion_density_m3'].values[j * xi.length + i]), s, colors[k])), 'Accepted time (s)', 'Ion density (m⁻³)')
    async function spatialFields(i: number) {
      const row = (name: string, width = x.length) => a[name].values.slice(i * width, (i + 1) * width)
      currentTime.textContent = i ? `Current accounting for ${times[i - 1]} → ${times[i]} s.` : 'Currents unavailable at the initial time; no displacement sample is invented.'
      const faces = [x[0], ...x.slice(1).map((v, j) => (v + x[j]) / 2), x.at(-1)!]
      await Promise.all([
        draw(potential, [line(x, row('potential_V'), 'Potential')], 'Position (m)', 'Potential (V)'),
        draw(carriers, [line(x, row('electron_density_m3'), 'Electrons'), line(x, row('hole_density_m3'), 'Holes', PALETTE.secondary)], 'Position (m)', 'Density (m⁻³)'),
        draw(ions, data.species.map((s, k) => line(xi, row(s + '_ion_density_m3', xi.length), s, colors[k])), 'Ion-domain position (m)', 'Ion density (m⁻³)'),
        draw(cells, data.species.map((s, k) => line(xi, row(s + '_ion_cell_inventory_m2', xi.length), s, colors[k])), 'Ion-domain position (m)', 'Cell inventory (m⁻²)'),
        draw(current, ['carrier_conduction', 'ion_conduction', ...kinds].map((kind, k) =>
          line(i ? faces : [], i ? a[`${kind}_current_density_A_m2`].values.slice((i - 1) * faces.length, i * faces.length) : [], kind.replaceAll('_', ' '), colors[k])), 'Face / port position (m)', 'Current density (A/m²)'),
      ])
    }
    const plotError = (failure: unknown) => { if (!disposed) error.textContent = `Plot failed: ${message(failure)}` }
    sample.addEventListener('change', () => { void spatialFields(Number(sample.value)).catch(plotError) })
    node.addEventListener('change', () => { void temporalFields(Number(node.value)).catch(plotError) })
    ionNode.addEventListener('change', () => { void ionicFields(Number(ionNode.value)).catch(plotError) })
    documents()
    await Promise.all([draw(terminalCurrent, terminalTraces, 'Accepted step end time (s)', 'Current density (A/m²)'),
      draw(inventories, data.species.map((s, k) => line(times, a[s + '_ion_inventory_m2'].values, s, colors[k])), 'Accepted time (s)', 'Domain inventory (m⁻²)'),
      temporalFields(Number(node.value)), ionicFields(Number(ionNode.value)), spatialFields(Number(sample.value))])
  }

  async function connect(active: IonicJob, token: number) {
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
        else output.replaceChildren(element('p', 'No ionic history was committed for this attempt. Fields and metrics are unavailable; physical status: not_evaluated.'))
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
  async function openStored(target: StoredIonicTransientTarget) {
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
    const runId = `ionic-transient-${crypto.randomUUID()}`; identity.textContent = `Submitting run ${runId}`
    try {
      const number = (name: string) => inputs[name].valueAsNumber
      const budget = (prefix = '') => ({ max_iterations: number(prefix + 'max_iterations'),
        max_residual_evaluations: number(prefix + 'max_residual_evaluations'),
        max_jacobian_evaluations: number(prefix + 'max_jacobian_evaluations'), max_backtracks: number(prefix + 'max_backtracks') })
      if (intervals.value.split(',').some(v => !v.trim())) throw Error('Enter one interval count per layer')
      const input: IonicProtocol = { ...parseIonicSamples(samples.value), intervals_per_layer: intervals.value.split(',').map(Number),
        controls: { initial_steady: { budget: budget('initial_'), inventory_row_node: number('inventory_row_node'), wall_s: number('initial_wall_s') },
          transient: { max_steps: number('max_steps'), wall_s: number('wall_s'), newton_budget: budget() } } }
      const active = await client.submit(catalog, route, device.value, input, number('wall_seconds'), runId)
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
