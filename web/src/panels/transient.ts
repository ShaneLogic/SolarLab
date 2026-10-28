import Plotly from 'plotly.js-basic-dist-min'
import { baseLayout, axisTitle, plotConfig, PALETTE } from '../plot-theme'
import { TransientClient, parseTransientSamples } from '../transient-client'
import type { TransientCatalog, TransientJob, TransientResult, TransientRoute, TransientProtocol } from '../transient-client'
import type { RunSubscription, TerminalRecord } from '../run-client'

export interface StoredTransientTarget { storeId: string; runId: string; attemptId: string }
export interface TransientPanel {
  readonly ready: Promise<void>; readonly completion: Promise<void>
  readonly job: TransientJob | undefined; readonly result: TransientResult | undefined
  reconnect(): Promise<void>; openStored(target: StoredTransientTarget): Promise<void>; dispose(): void
}

function element<K extends keyof HTMLElementTagNameMap>(tag: K, text?: string) {
  const node = document.createElement(tag); if (text !== undefined) node.textContent = text; return node
}
const message = (error: unknown) => error instanceof Error ? error.message : String(error)
const terminal = (state: string) => ['succeeded', 'failed', 'cancelled'].includes(state)

/** A concrete ion-free transient view on the same owner, catalog and stored reader. */
export function mountTransientPanel(root: HTMLElement, endpoint: string | URL, initial?: StoredTransientTarget): TransientPanel {
  const client = new TransientClient(endpoint)
  let catalog: TransientCatalog | undefined, job: TransientJob | undefined, result: TransientResult | undefined
  let subscription: RunSubscription | undefined, phase = 'loading', epoch = 0, disposed = false
  let abort = new AbortController(), completion = Promise.resolve()
  const displayedEvents = new Set<string>(), plots: HTMLElement[] = []
  const card = element('section'); card.className = 'card'; card.dataset.panel = 'transient'
  card.append(element('h2', 'Transient · new core'), element('p',
    'Classical ion-free 1D time evolution from a cold DC initial state. Only installed devices are available. Ionic dynamics and 2D are outside this entry; physical qualification is not established.'))
  const form = element('form'), fields = element('fieldset'); fields.style.cssText = 'border:0;padding:0'
  const grid = element('div'); grid.className = 'form-grid'; fields.append(grid)
  function control<T extends HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>(name: string, label: string, input: T, parent: HTMLElement = grid): T {
    const wrap = element('label'); wrap.className = 'form-group'
    input.name = name; input.setAttribute('aria-label', label); wrap.append(element('span', label), input); parent.append(wrap); return input
  }
  const routeSelect = control('route', 'Installed transient experiment', element('select'))
  const device = control('device', 'Installed device', element('select'))
  const samples = control('samples', 'Samples: time (s), voltage (V), generation scale — one row per sample', element('textarea'), fields)
  samples.rows = 5; samples.required = true; samples.style.cssText = 'width:100%;font-family:monospace;resize:vertical'
  samples.value = '0, 0, 1\n1e-7, 0, 1.0025\n2e-7, 0, 1.01'
  const budgets = element('details'); budgets.append(element('summary', 'Grid and work budgets')); fields.append(budgets)
  const budgetGrid = element('div'); budgetGrid.className = 'form-grid'; budgets.append(budgetGrid)
  const inputs: Record<string, HTMLInputElement> = {}
  for (const [name, label, value, step] of [
    ['intervals_per_layer', 'Intervals per electrical layer', '12', '1'], ['wall_seconds', 'Whole-job wall budget (s)', '60', 'any'],
    ['max_steps', 'Maximum transient steps', '32', '1'], ['wall_s', 'Transient wall budget (s)', '30', 'any'],
    ['max_iterations', 'Newton iterations per step', '50', '1'], ['max_residual_evaluations', 'Residual evaluations per step', '1000', '1'],
    ['max_jacobian_evaluations', 'Jacobian evaluations per step', '50', '1'], ['max_backtracks', 'Backtracks per iteration', '24', '1'],
  ]) {
    const input = element('input'); input.type = 'number'; input.value = value; input.step = step; input.required = true
    inputs[name] = control(name, label, input, budgetGrid)
  }
  const bounds = element('p'), run = element('button', 'Run transient'); run.type = 'submit'; run.className = 'btn btn-primary'
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
  const stored = {} as Record<keyof StoredTransientTarget, HTMLInputElement>
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
  function selected(): TransientRoute | undefined { return catalog?.routes[Number(routeSelect.value)] }
  function devices() {
    const route = selected(); device.replaceChildren()
    for (const d of route?.devices ?? []) { const option = element('option', d.id); option.value = d.id; device.append(option) }
    run.disabled = !route?.devices.length
    if (!route) return
    for (const [key, input] of Object.entries(inputs)) {
      const limit = key === 'wall_seconds' ? { min: .001, max: catalog!.ceiling.wall_seconds } : route.bounds[key as keyof TransientRoute['bounds']]
      input.min = String(key === 'wall_s' ? Math.max(.001, limit.min) : limit.min); input.max = String(limit.max)
      input.value = String(Math.min(Number(input.value), limit.max))
    }
    const b = route.bounds
    bounds.textContent = `${b.samples.min}–${b.samples.max} rows; strictly increasing time ${b.times_s.min}–${b.times_s.max} s; voltage ${b.voltage_V.min}–${b.voltage_V.max} V; generation ${b.generation_scale.min}–${b.generation_scale.max}. Solver tolerances are fixed by the installed service.`
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
      if (!found.routes.length) error.textContent = 'No compatible ion-free transient route is installed on this service.'
    } catch (failure) {
      if (!disposed) { error.textContent = `Installed transient service unavailable: ${message(failure)}`; if (!job) update('unavailable') }
    }
  }
  refresh.addEventListener('click', () => { void discover() })
  function purge() { for (const plot of plots.splice(0)) Plotly.purge(plot); output.replaceChildren() }
  function selectJob(active: TransientJob) {
    job = active; reconnect.disabled = false
    const target = { storeId: active.reader.storeId, ...active.target }
    for (const key of Object.keys(stored) as (keyof StoredTransientTarget)[]) stored[key].value = target[key]
    identity.textContent = `Store ${target.storeId} · run ${target.runId} · attempt ${target.attemptId}`
    const url = new URL(location.href); url.search = new URLSearchParams({ view: 'transient', ...target }).toString()
    link.href = url.href; link.hidden = false
  }

  async function render(data: TransientResult, active: TransientJob) {
    purge()
    const a = data.arrays, times = a.accepted_times_s.values, x = a.nodes_m.values, count = times.length
    const summary = element('p', `${count} accepted states of ${a.requested_times_s.values.length} requested · ${count - 1} accepted ${count === 2 ? 'step' : 'steps'} · ${x.length} nodes · ${String(data.worker.registered_device_id)}`)
    summary.dataset.role = 'history-summary'; output.append(summary)
    const unknown = element('p', 'Voc, fill factor and efficiency: Unknown for this transient history. Physical status: not_evaluated.')
    unknown.dataset.role = 'metrics'; output.append(unknown)
    const failure = data.attempts.find(attempt => attempt.accepted === false)
    if (failure) {
      const failed = element('p', `First failed step: ${String(failure.start_s)} → ${String(failure.end_s)} s · ${String(failure.termination)}${failure.error ? ' · ' + JSON.stringify(failure.error) : ''}. Only the accepted prefix is plotted.`)
      failed.dataset.role = 'first-failure'; output.append(failed)
    }
    const requested = element('details'); requested.dataset.role = 'requested-protocol'; requested.open = true
    requested.append(element('summary', 'Stored requested protocol'))
    const table = element('table'), head = element('tr'), thead = element('thead'), tbody = element('tbody')
    table.className = 'param-table mode-table'
    for (const label of ['Time (s)', 'Voltage (V)', 'Generation scale', 'Stored outcome']) head.append(element('th', label))
    thead.append(head); table.append(thead, tbody)
    a.requested_times_s.values.forEach((time, i) => {
      const row = element('tr')
      for (const value of [time, a.requested_voltage_V.values[i], a.requested_generation_scale.values[i],
        i === 0 ? 'Initial DC' : i < count ? 'Accepted' : i === count ? 'Failed step' : 'Not attempted']) row.append(element('td', String(value)))
      tbody.append(row)
    })
    requested.append(table); output.append(requested)
    output.append(element('p', 'Currents use the stored signs and accepted step end times. Terminal columns are left/right; no current at the initial time is invented.'))
    if (count === 1) output.append(element('p', 'No transient step was accepted; current histories are unavailable. The initial DC fields remain readable.'))
    const selectors = element('div'); selectors.className = 'form-grid'; output.append(selectors)
    const sample = control('accepted_time', 'Spatial fields at accepted time (s)', element('select'), selectors)
    times.forEach((t, i) => { const option = element('option', String(t)); option.value = String(i); sample.append(option) }); sample.value = String(count - 1)
    const node = control('node', 'Temporal fields at position (m)', element('select'), selectors)
    x.forEach((position, i) => { const option = element('option', position.toPrecision(6)); option.value = String(i); node.append(option) }); node.value = String(Math.floor(x.length / 2))
    function plot(title: string) {
      const section = element('section'); section.className = 'card'; section.append(element('h3', title))
      const div = element('div'); div.style.height = '310px'; div.dataset.plot = title
      section.append(div); output.append(section); plots.push(div); return div
    }
    const terminalCurrent = plot('Terminal currents over time'), potentialTime = plot('Potential over time'), densityTime = plot('Carrier densities over time')
    const potential = plot('Potential in space'), carriers = plot('Carrier densities in space'), current = plot('Currents in space')
    const colors = [PALETTE.primary, PALETTE.secondary, '#6c5ce7']
    const line = (x: number[], y: number[], name: string, color = PALETTE.primary) =>
      ({ type: 'scatter', mode: 'lines+markers', x, y, name, line: { color, width: 2 }, marker: { size: 4 } })
    function draw(div: HTMLElement, traces: Record<string, unknown>[], xlabel: string, ylabel: string) {
      const padding = Math.max((a.requested_times_s.values[1] - times[0]) * .05, Number.MIN_VALUE * 8)
      const timeRange = count > 1 ? [times[0], times[count - 1]] : [times[0] - padding, times[0] + padding]
      return Plotly.newPlot(div, traces, baseLayout({ xaxis: { title: axisTitle(xlabel), ...(xlabel.includes('time') ? { range: timeRange } : {}) },
        yaxis: { title: axisTitle(ylabel) } }), plotConfig(`transient-${active.target.runId}`))
    }
    const kinds = ['conduction', 'displacement', 'total'] as const
    const terminalTraces = kinds.flatMap((kind, k) => [0, 1].map(side => ({
      ...line(times.slice(1), times.slice(1).map((_, i) => a[`terminal_${kind}_current_density_A_m2`].values[i * 2 + side]),
        `${side === 0 ? 'Left' : 'Right'} ${kind}`, colors[k]), line: { color: colors[k], dash: side ? 'dash' : 'solid' },
    })))
    async function temporalFields(i: number) {
      const column = (name: 'potential_V' | 'electron_density_m3' | 'hole_density_m3') => times.map((_, j) => a[name].values[j * x.length + i])
      await Promise.all([
        draw(potentialTime, [line(times, column('potential_V'), 'Potential')], 'Accepted time (s)', 'Potential (V)'),
        draw(densityTime, [line(times, column('electron_density_m3'), 'Electrons'), line(times, column('hole_density_m3'), 'Holes', PALETTE.secondary)], 'Accepted time (s)', 'Density (m⁻³)'),
      ])
    }
    async function spatialFields(i: number) {
      const row = (name: 'potential_V' | 'electron_density_m3' | 'hole_density_m3') => a[name].values.slice(i * x.length, (i + 1) * x.length)
      await Promise.all([
        draw(potential, [line(x, row('potential_V'), 'Potential')], 'Position (m)', 'Potential (V)'),
        draw(carriers, [line(x, row('electron_density_m3'), 'Electrons'), line(x, row('hole_density_m3'), 'Holes', PALETTE.secondary)], 'Position (m)', 'Density (m⁻³)'),
        draw(current, kinds.flatMap((kind, k) => [
          line(i ? x.slice(1).map((v, j) => (v + x[j]) / 2) : [], i ? a[`${kind}_current_density_A_m2`].values.slice((i - 1) * (x.length - 1), i * (x.length - 1)) : [], kind, colors[k]),
          { ...line(i ? [x[0], x.at(-1)!] : [], i ? a[`terminal_${kind}_current_density_A_m2`].values.slice((i - 1) * 2, i * 2) : [], `${kind} at contacts`, colors[k]), mode: 'markers' },
        ]), 'Position (m)', 'Current density (A/m²)'),
      ])
    }
    const plotError = (failure: unknown) => { if (!disposed) error.textContent = `Plot failed: ${message(failure)}` }
    sample.addEventListener('change', () => { void spatialFields(Number(sample.value)).catch(plotError) })
    node.addEventListener('change', () => { void temporalFields(Number(node.value)).catch(plotError) })
    for (const [label, text] of [['Original stored attempt', data.originalJson], ['Raw solver diagnostics (nonfinite tags preserved)', data.diagnosticsJson]]) {
      const details = element('details'), pre = element('pre', text); pre.style.cssText = 'white-space:pre-wrap;overflow-wrap:anywhere;max-height:400px;overflow:auto'
      details.append(element('summary', label), pre); output.append(details)
    }
    await Promise.all([draw(terminalCurrent, terminalTraces, 'Accepted step end time (s)', 'Current density (A/m²)'), temporalFields(Number(node.value)), spatialFields(Number(sample.value))])
  }

  async function connect(active: TransientJob, token: number) {
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
        else output.replaceChildren(element('p', 'No transient history was committed for this attempt.'))
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
  async function openStored(target: StoredTransientTarget) {
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
    const runId = `transient-${crypto.randomUUID()}`; identity.textContent = `Submitting run ${runId}`
    try {
      const number = (name: string) => inputs[name].valueAsNumber
      const input: TransientProtocol = { ...parseTransientSamples(samples.value), intervals_per_layer: number('intervals_per_layer'),
        controls: { max_steps: number('max_steps'), wall_s: number('wall_s'), newton_budget: {
          max_iterations: number('max_iterations'), max_residual_evaluations: number('max_residual_evaluations'),
          max_jacobian_evaluations: number('max_jacobian_evaluations'), max_backtracks: number('max_backtracks'),
        } } }
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
