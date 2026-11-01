import Plotly from 'plotly.js-basic-dist-min'
import { baseLayout, axisTitle, plotConfig, PALETTE } from '../plot-theme'
import { SteadyJVClient, voltageGrid } from '../steady-jv-client'
import type { SteadyCatalog, SteadyJob, SteadyResult, SteadyRoute } from '../steady-jv-client'
import type { RunState, RunSubscription, TerminalRecord } from '../run-client'

export interface SteadyJVPanel {
  readonly ready: Promise<void>
  readonly completion: Promise<void>
  readonly job: SteadyJob | undefined
  readonly result: SteadyResult | undefined
  reconnect(): Promise<void>
  dispose(): void
}

function element<K extends keyof HTMLElementTagNameMap>(tag: K, text?: string) {
  const node = document.createElement(tag)
  if (text !== undefined) node.textContent = text
  return node
}
function message(error: unknown): string { return error instanceof Error ? error.message : String(error) }
const terminal = (state: string) => ['succeeded', 'failed', 'cancelled'].includes(state)

/** Independent static-DC entry; it never consumes a legacy workspace config. */
export function mountSteadyJVPanel(root: HTMLElement, endpoint: string | URL): SteadyJVPanel {
  const client = new SteadyJVClient(endpoint)
  let catalog: SteadyCatalog | undefined, job: SteadyJob | undefined, result: SteadyResult | undefined
  let subscription: RunSubscription | undefined, disposed = false, phase = 'loading', epoch = 0
  let completion = Promise.resolve(), abort = new AbortController()
  const displayedEvents = new Set<string>()
  const plots: HTMLElement[] = []
  const card = element('section'); card.className = 'card'; card.dataset.panel = 'steady-jv'
  card.append(element('h2', 'Steady J–V · new core'), element('p',
    'Static, classical, ion-free 1D DC. AC, ionic dynamics and hysteresis are unavailable in this entry. Execution does not establish scientific qualification.'))
  const form = element('form'), fields = element('fieldset'); fields.style.border = '0'; fields.style.padding = '0'
  const grid = element('div'); grid.className = 'form-grid'; fields.append(grid)
  function control<T extends HTMLInputElement | HTMLSelectElement>(name: string, label: string, input: T): T {
    const field = element('label'); field.className = 'form-group'
    input.name = name; input.setAttribute('aria-label', label)
    field.append(element('span', label), input); grid.append(field); return input
  }
  const routeSelect = control('route', 'Installed DC experiment', element('select'))
  const deviceSelect = control('device', 'Installed device', element('select'))
  function number(name: string, label: string, value: string, step = 'any', required = true) {
    const input = element('input'); input.type = 'number'; input.value = value; input.step = step; input.required = required
    return control(name, label, input)
  }
  const start = number('start', 'Start voltage (V)', '0'), end = number('end', 'End voltage (V)', '0.1')
  const points = number('points', 'Voltage samples', '3', '1')
  const intervals = number('intervals', 'Intervals per electrical layer', '12', '1')
  const generation = number('generation', 'Generation scale', '1'); generation.min = '0'
  const power = number('power', 'Incident power (W/m²), blank if unknown', '', 'any', false)
  const wall = number('wall', 'Wall budget (s)', '60'); wall.min = '0.001'
  const bounds = element('p'), runButton = element('button', 'Run steady DC'); runButton.type = 'submit'; runButton.className = 'btn btn-primary'
  fields.append(bounds, runButton); form.append(fields)
  const refresh = element('button', 'Refresh installed devices'), cancel = element('button', 'Cancel this attempt')
  const reconnect = element('button', 'Reconnect / refresh run')
  for (const button of [refresh, cancel, reconnect]) { button.type = 'button'; button.className = 'btn btn-ghost' }
  cancel.disabled = reconnect.disabled = true
  const state = element('p', 'Loading installed DC catalog…'); state.setAttribute('role', 'status'); state.dataset.role = 'state'
  const transport = element('p'); transport.dataset.role = 'transport'; transport.setAttribute('role', 'status')
  const identity = element('p'); identity.dataset.role = 'attempt'
  const error = element('p'); error.className = 'status error'; error.setAttribute('role', 'alert')
  const history = element('ul'); history.dataset.role = 'history'
  const diagnostics = element('details'); diagnostics.dataset.role = 'diagnostics'
  diagnostics.append(element('summary', 'Run diagnostics'), identity, history)
  const output = element('div'); output.dataset.role = 'result'
  card.append(form, refresh, cancel, reconnect, state, transport, error, diagnostics, output)
  root.replaceChildren(card); fields.disabled = true

  function update(next: string) {
    if (terminal(phase) && next !== phase) return
    if (phase === 'running' && next === 'queued') return
    phase = next; card.dataset.state = next; state.textContent = next
    const busy = ['submitting', 'queued', 'running'].includes(next)
    fields.disabled = busy || !catalog?.routes.length
    refresh.disabled = busy; cancel.disabled = !job || !['queued', 'running'].includes(next)
  }
  function purge() { for (const plot of plots.splice(0)) Plotly.purge(plot); output.replaceChildren() }
  function selected(): SteadyRoute | undefined { return catalog?.routes[Number(routeSelect.value)] }
  function devices() {
    const route = selected(); deviceSelect.replaceChildren()
    for (const device of route?.devices ?? []) { const option = element('option', device.id); option.value = device.id; deviceSelect.append(option) }
    runButton.disabled = !route?.devices.length
    if (route) {
      points.min = String(route.minPoints); points.max = String(route.maxPoints)
      intervals.min = String(route.minIntervals); intervals.max = String(route.maxIntervals)
      bounds.textContent = `${route.minPoints}–${route.maxPoints} voltage samples; ${route.minIntervals}–${route.maxIntervals} intervals per layer. Blank incident power leaves efficiency unknown.`
    }
  }
  routeSelect.addEventListener('change', devices)
  async function discover() {
    error.textContent = ''; refresh.disabled = true; fields.disabled = true
    try {
      const found = await client.discover(abort.signal); if (disposed) return
      catalog = found; routeSelect.replaceChildren()
      found.routes.forEach((route, i) => { const option = element('option', `${route.id} · version ${route.version}`); option.value = String(i); routeSelect.append(option) })
      wall.max = String(found.ceiling.wall_seconds); wall.value = String(Math.min(60, found.ceiling.wall_seconds))
      devices(); fields.disabled = !found.routes.length
      if (!job) { phase = 'ready'; update(found.routes.length ? 'ready' : 'unavailable') }
      if (!found.routes.length) error.textContent = 'No compatible classical DC route is installed on this service.'
    } catch (failure) {
      if (!disposed) { error.textContent = `Installed DC service unavailable: ${message(failure)}`; if (!job) update('unavailable') }
    } finally { if (!disposed) refresh.disabled = false }
  }
  refresh.addEventListener('click', () => { void discover() })

  async function render(data: SteadyResult, active: SteadyJob) {
    purge()
    const metrics = element('div'); metrics.className = 'metric-row'; metrics.dataset.role = 'metrics'
    for (const [key, label, unit, scale] of [
      ['short_circuit_current_A_m2', 'Jsc', 'A/m²', 1], ['open_circuit_voltage_V', 'Voc', 'V', 1],
      ['fill_factor', 'Fill factor', '%', 100], ['efficiency', 'Efficiency', '%', 100],
    ] as const) {
      const value = data.metrics[key]
      const item = element('div'); item.className = 'metric'; item.dataset.metric = key
      item.append(element('strong', label), element('p', typeof value === 'number' && Number.isFinite(value)
        ? `${(value * scale).toPrecision(6)} ${unit}` : 'Unknown'))
      metrics.append(item)
    }
    output.append(metrics)
    if (data.metrics.limitation) output.append(element('p', data.metrics.limitation === 'open_circuit_not_bracketed'
      ? 'Open-circuit voltage is not bracketed by this curve. Voc, fill factor and efficiency remain unknown.'
      : String(data.metrics.limitation)))
    const a = data.arrays, x = a.nodes_m.values, voltage = a.voltage_V.values
    output.append(element('p', `${x.length} nodes · ${voltage.length} voltage samples · ${active.device.id}`))
    const sampleLabel = element('label', 'Spatial fields at voltage (V) '), sample = element('select')
    sample.setAttribute('aria-label', 'Spatial fields at voltage (V)'); sample.dataset.role = 'sample'
    voltage.forEach((v, i) => { const option = element('option', String(v)); option.value = String(i); sample.append(option) })
    sampleLabel.append(sample); output.append(sampleLabel)
    function plot(title: string) {
      const section = element('section'); section.className = 'card'; section.append(element('h3', title))
      const div = element('div'); div.style.height = '300px'; div.dataset.plot = title
      section.append(div); output.append(section); plots.push(div); return div
    }
    const jv = plot('J–V'), current = plot('Current density'), potential = plot('Potential'), carriers = plot('Carrier densities')
    function draw(div: HTMLElement, traces: Record<string, unknown>[], xlabel: string, ylabel: string) {
      return Plotly.newPlot(div, traces, baseLayout({ xaxis: { title: axisTitle(xlabel) }, yaxis: { title: axisTitle(ylabel) } }), plotConfig(`steady-${active.target.runId}`))
    }
    function line(x: number[], y: number[], name: string, color = PALETTE.primary) {
      return { type: 'scatter', mode: 'lines+markers', x, y, name, line: { color, width: 2 }, marker: { size: 4 } }
    }
    async function fieldsAt(i: number) {
      const row = (name: 'potential_V' | 'electron_density_m3' | 'hole_density_m3') => a[name].values.slice(i * x.length, (i + 1) * x.length)
      await Promise.all([
        draw(current, [line(x.slice(1).map((v, j) => (v + x[j]) / 2), a.current_density_A_m2.values.slice(i * (x.length - 1), (i + 1) * (x.length - 1)), 'Bulk current'),
          { ...line([x[0], x[x.length - 1]], a.contact_current_density_A_m2.values.slice(i * 2, i * 2 + 2), 'Contacts', PALETTE.secondary), mode: 'markers' }], 'Position (m)', 'Current density (A/m²)'),
        draw(potential, [line(x, row('potential_V'), 'Potential')], 'Position (m)', 'Potential (V)'),
        draw(carriers, [line(x, row('electron_density_m3'), 'Electrons'), line(x, row('hole_density_m3'), 'Holes', PALETTE.secondary)], 'Position (m)', 'Density (m⁻³)'),
      ])
    }
    sample.addEventListener('change', () => { void fieldsAt(Number(sample.value)).catch(failure => { if (!disposed) error.textContent = message(failure) }) })
    const evidence = element('details'); evidence.append(element('summary', 'Solver evidence and original stored attempt'))
    const pre = element('pre', data.originalJson); pre.style.whiteSpace = 'pre-wrap'; pre.style.overflowWrap = 'anywhere'
    evidence.append(pre); output.append(evidence)
    await Promise.all([draw(jv, [line(voltage, a.output_current_A_m2.values, 'Output current')], 'Voltage (V)', 'Output current (A/m²)'), fieldsAt(0)])
  }

  async function connect(active: SteadyJob, generation: number) {
    const current = () => !disposed && generation === epoch && job === active
    const resume = subscription?.cursor; subscription?.unsubscribe()
    const snapshot = await active.reader.getAttempt(active.target, abort.signal)
    if (!current()) return
    update(snapshot.value.state)
    async function finish(end: TerminalRecord) {
      if (!current()) return
      update(end.state)
      const worker = end.result.worker_result as Record<string, unknown> | undefined
      if (end.state !== 'succeeded') {
        const detail = worker?.detail ?? worker?.reason ?? end.result.reason ?? ''
        error.textContent = `${end.state}: ${String(detail)}`; return
      }
      if (!result) {
        const data = await client.readResult(active, abort.signal)
        if (!current()) return
        await render(data, active); if (current()) result = data
      }
    }
    if (snapshot.value.terminal) await finish(snapshot.value.terminal)
    if (!current()) return
    subscription = active.reader.subscribe(active.target, async doc => {
      if (!current()) return
      const event = doc.value
      if (!displayedEvents.has(String(event.sequence))) {
        history.append(element('li', `${event.sequence} · ${event.kind}`))
        displayedEvents.add(String(event.sequence))
      }
      if (event.kind === 'queued' || event.kind === 'running') update(event.kind as RunState)
      if (event.kind === 'progress') transport.textContent = 'Worker progress received'
      if (event.kind === 'terminal') await finish(event.payload as TerminalRecord)
    }, resume)
    transport.textContent = 'Following persisted events…'
    const end = await subscription.done
    if (current()) transport.textContent = end.reason === 'disconnected'
      ? 'Event connection lost; use Reconnect. The job was not cancelled.'
      : `Event stream: ${end.reason}`
  }
  async function reconnectRun() {
    if (!job || disposed) return
    error.textContent = ''
    try { await connect(job, ++epoch) } catch (failure) { if (!disposed) error.textContent = `Read failed: ${message(failure)}` }
  }
  reconnect.addEventListener('click', () => { completion = reconnectRun() })
  cancel.addEventListener('click', async () => {
    const active = job; if (!active) return
    cancel.disabled = true
    try { await client.cancel(active); if (!disposed && job === active) transport.textContent = 'Cancellation requested; waiting for the stored terminal.' }
    catch (failure) { if (!disposed) { error.textContent = message(failure); cancel.disabled = terminal(phase) } }
  })
  async function submit() {
    if (['submitting', 'queued', 'running'].includes(phase) || !catalog || !selected() || !form.reportValidity()) return
    const token = ++epoch; subscription?.unsubscribe(); subscription = undefined
    abort.abort(); abort = new AbortController(); job = undefined; result = undefined; purge()
    history.replaceChildren(); displayedEvents.clear(); diagnostics.open = false
    phase = 'submitting'; update(phase); error.textContent = ''; transport.textContent = ''; reconnect.disabled = true
    const runId = `steady-${crypto.randomUUID()}`; identity.textContent = `Submitting run ${runId}`
    try {
      const active = await client.submit(catalog, selected()!, deviceSelect.value, {
        voltage_V: voltageGrid(start.valueAsNumber, end.valueAsNumber, points.valueAsNumber),
        intervals_per_layer: intervals.valueAsNumber, generation_scale: generation.valueAsNumber,
        incident_power_W_m2: power.value.trim() === '' ? null : power.valueAsNumber,
      }, wall.valueAsNumber, runId)
      if (disposed || token !== epoch) return
      job = active; identity.textContent = `Run ${active.target.runId} · attempt ${active.target.attemptId} · ${active.device.id}`
      reconnect.disabled = false; update('queued'); await connect(active, token)
    } catch (failure) {
      if (!disposed && token === epoch) {
        error.textContent = message(failure)
        if (!job) { phase = 'unconfirmed'; update(phase); transport.textContent = 'Submission was not confirmed. No result is assumed; the run ID is available in Run diagnostics.' }
        else transport.textContent = 'Stored result or event read failed; use Reconnect.'
      }
    }
  }
  form.addEventListener('submit', event => { event.preventDefault(); completion = submit() })
  const ready = discover()
  return { ready, get completion() { return completion }, get job() { return job }, get result() { return result },
    reconnect: reconnectRun,
    dispose() { disposed = true; epoch++; abort.abort(); subscription?.unsubscribe(); purge(); root.replaceChildren() },
  }
}
