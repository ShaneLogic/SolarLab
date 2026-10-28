import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import Plotly from 'plotly.js-basic-dist-min'
import { IonicTransientClient } from '../ionic-transient-client'
import type { IonicJob } from '../ionic-transient-client'
import { catalog, protocol, resultFixture } from '../test-helpers/ionic-transient-fixture'
import type { RunSubscription, StoredDocument, StoredEvent, StreamEnd, TerminalRecord } from '../run-client'
import { mountIonicTransientPanel } from './ionic-transient'
import type { IonicTransientPanel } from './ionic-transient'
import { webcrypto } from '../test-helpers/schema-fixture.mjs'

vi.mock('plotly.js-basic-dist-min', () => ({ default: { newPlot: vi.fn().mockResolvedValue(undefined), purge: vi.fn() } }))
const cursor = { endpoint: 'http://localhost/steady-service/', storeId: 'store', runId: 'run', attemptId: 'attempt', sequence: '0' }
let panel: IonicTransientPanel, root: HTMLElement, job: IonicJob, sequence: number
let emit: (event: StoredDocument<StoredEvent>) => void | Promise<void>, endStream: (end: StreamEnd) => void
let stop: ReturnType<typeof vi.fn>
function button(text: string) { return [...root.querySelectorAll<HTMLButtonElement>('button')].find(node => node.textContent === text)! }
async function event(kind: StoredEvent['kind'], payload = {}) {
  await emit({ json: '{}', value: { sequence: ++sequence, kind, run_id: 'run', attempt_id: 'attempt', payload } })
}
async function submit() { button('Run ionic transient').click(); await vi.waitFor(() => expect(emit).toBeTypeOf('function')) }
async function finish(state: TerminalRecord['state']) {
  await event('terminal', { state, artifacts: [], result: { reason: 'stored reason' } })
  endStream({ reason: 'terminal', cursor }); await panel.completion
}
const trace = (title: string) => vi.mocked(Plotly.newPlot).mock.calls.find(([div]) => (div as HTMLElement).dataset.plot === title)!
beforeEach(async () => {
  vi.stubGlobal('crypto', webcrypto); vi.clearAllMocks(); vi.mocked(Plotly.newPlot).mockResolvedValue(undefined); sequence = 0
  root = document.createElement('div'); document.body.append(root); emit = undefined as unknown as typeof emit; stop = vi.fn()
  const done = new Promise<StreamEnd>(resolve => { endStream = resolve })
  job = { target: { runId: 'run', attemptId: 'attempt' }, reader: { storeId: 'store',
    getAttempt: vi.fn().mockResolvedValue({ value: { state: 'queued', terminal: null } }),
    subscribe: vi.fn((_target, callback) => { emit = callback; return { cursor, done, unsubscribe: stop } as RunSubscription }) } as unknown as IonicJob['reader'] }
  vi.spyOn(IonicTransientClient.prototype, 'discover').mockResolvedValue(catalog)
  vi.spyOn(IonicTransientClient.prototype, 'submit').mockResolvedValue(job)
  vi.spyOn(IonicTransientClient.prototype, 'open').mockReturnValue(job)
  vi.spyOn(IonicTransientClient.prototype, 'cancel').mockResolvedValue(undefined)
  vi.spyOn(IonicTransientClient.prototype, 'readResult').mockResolvedValue(resultFixture())
  panel = mountIonicTransientPanel(root, 'http://localhost/steady-service/'); await panel.ready
})
afterEach(() => { panel.dispose(); root.remove(); vi.restoreAllMocks(); vi.unstubAllGlobals() })

it('submits explicit installed arrays, layer intervals and separate initial/transient controls', async () => {
  await submit()
  expect(IonicTransientClient.prototype.submit).toHaveBeenCalledWith(catalog, catalog.routes[0], 'research_1d_shared', protocol, 30, expect.stringMatching(/^ionic-transient-/))
  expect(root.querySelector('[data-role=state]')?.textContent).toBe('queued')
  await event('running'); await finish('succeeded')
  expect(root.querySelector('[data-role=state]')?.textContent).toBe('succeeded')
  expect(root.querySelector('[data-role=metrics]')?.textContent).toContain('Unknown')
  expect(Plotly.newPlot).toHaveBeenCalledTimes(10)
  expect(trace('Terminal currents over time')[1][0]).toMatchObject({ x: [.25, .5, .75, 1], y: [-10, -8, -6, -4] })
  expect(trace('Terminal currents over time')[2]).toMatchObject({ yaxis: { title: { text: 'Current density (A/m²)' } } })
  expect(trace('Ion densities in space')[1].map(t => t.name)).toEqual(['positive', 'negative'])
  expect(trace('Ion densities in space')[1][0].x).toHaveLength(2)
  expect(trace('Potential in space')[1][0].x).toHaveLength(4)
  expect(trace('Currents in space')[1][0].x).toEqual([0, .5e-8, 2e-8, (3e-8 + 4e-8) / 2, 4e-8])
  expect(trace('Independent species inventories over time')[1].map(t => t.y)).toEqual([Array(5).fill(60), Array(5).fill(80)])
  expect(root.querySelector<HTMLAnchorElement>('[data-role=readback-link]')?.href).toContain('view=ionic-transient')
})

it('opens and reconnects the exact failed attempt GET-only, preserving its unattempted tail', async () => {
  vi.mocked(IonicTransientClient.prototype.readResult).mockResolvedValue(resultFixture(true))
  vi.mocked(job.reader.getAttempt).mockResolvedValue({ value: { state: 'failed', terminal: { state: 'failed', result: { reason: 'budget_exhausted' }, artifacts: [] } } } as never)
  vi.mocked(job.reader.subscribe).mockReturnValue({ cursor, unsubscribe: stop, done: Promise.resolve({ reason: 'terminal_acknowledged', cursor }) })
  await panel.openStored({ storeId: 'store', runId: 'run', attemptId: 'attempt' })
  expect(IonicTransientClient.prototype.open).toHaveBeenCalledWith('store', 'run', 'attempt')
  expect(IonicTransientClient.prototype.submit).not.toHaveBeenCalled()
  expect(root.querySelector('[data-role=state]')?.textContent).toBe('failed')
  expect(root.querySelector('[data-role=first-failure]')?.textContent).toContain('0.25 → 0.5 s · budget_exhausted')
  const rows = [...root.querySelectorAll('[data-role=requested-protocol] tbody tr')].map(r => r.querySelector('td:last-child')?.textContent)
  expect(rows).toEqual(['Initial steady accepted', 'Accepted', 'Failed step', 'Not attempted', 'Not attempted'])
  expect(trace('Terminal currents over time')[1][0].x).toEqual([.25])
  expect(trace('Terminal currents over time')[2]?.xaxis).toMatchObject({ range: [0, .25] })
  expect(root.textContent).toContain('"$float":"NaN"')
  await panel.reconnect(); expect(IonicTransientClient.prototype.submit).not.toHaveBeenCalled()
  expect(root.querySelector('[data-role=transport]')?.textContent).toContain('terminal_acknowledged')
})

it('changes the ion selector on its own grid and shows no initial displacement sample', async () => {
  await submit(); await finish('succeeded')
  const ion = root.querySelector<HTMLSelectElement>('[name=ion_node]')!, time = root.querySelector<HTMLSelectElement>('[name=accepted_time]')!
  expect(ion.options).toHaveLength(2); ion.value = '0'; ion.dispatchEvent(new Event('change'))
  time.value = '0'; time.dispatchEvent(new Event('change'))
  await vi.waitFor(() => expect(root.querySelector('[data-role=current-time]')?.textContent).toContain('unavailable'))
  const currents = vi.mocked(Plotly.newPlot).mock.calls.filter(([div]) => (div as HTMLElement).dataset.plot === 'Currents in space').at(-1)!
  expect(currents[1].every(t => (t.x as number[]).length === 0 && (t.y as number[]).length === 0)).toBe(true)
})

it.each([0, 1])('renders failed initial/first-step outcomes without fabricated accepted currents (%s states)', async count => {
  vi.mocked(IonicTransientClient.prototype.readResult).mockResolvedValue(resultFixture(true, count))
  await submit(); await finish('failed')
  if (!count) {
    expect(Plotly.newPlot).not.toHaveBeenCalled()
    expect(root.textContent).toContain('Initial steady solve failed')
    expect(root.querySelector('[data-role=requested-protocol] tbody tr')?.textContent).toContain('Initial steady failed')
  } else {
    expect(trace('Terminal currents over time')[1].every(t => (t.x as number[]).length === 0)).toBe(true)
    expect(root.textContent).toContain('No transient step was accepted')
  }
})

it('keeps cancellation separate, waits for the terminal and never draws a history', async () => {
  vi.mocked(IonicTransientClient.prototype.readResult).mockResolvedValue(undefined)
  await submit(); await event('running'); button('Cancel this attempt').click()
  await vi.waitFor(() => expect(IonicTransientClient.prototype.cancel).toHaveBeenCalledWith(job))
  expect(root.querySelector('[data-role=state]')?.textContent).toBe('running')
  await finish('cancelled')
  expect(root.querySelector('[data-role=state]')?.textContent).toBe('cancelled')
  expect(root.textContent).toContain('No ionic history was committed'); expect(Plotly.newPlot).not.toHaveBeenCalled()
})

it('disposes streams and plots without cancelling an owned runtime job', async () => {
  await submit(); panel.dispose(); await event('terminal', { state: 'succeeded', result: {}, artifacts: [] })
  endStream({ reason: 'unsubscribed', cursor }); await panel.completion
  expect(stop).toHaveBeenCalled(); expect(IonicTransientClient.prototype.cancel).not.toHaveBeenCalled()
  expect(Plotly.newPlot).not.toHaveBeenCalled(); expect(root.textContent).toBe('')
})

it('offers no ion-free or legacy fallback when the ionic route is absent', async () => {
  panel.dispose(); vi.mocked(IonicTransientClient.prototype.discover).mockResolvedValue({ ...catalog, routes: [] })
  panel = mountIonicTransientPanel(root, 'http://localhost/steady-service/'); await panel.ready
  expect(root.textContent).toContain('No compatible dark positive-D ionic transient route')
  button('Run ionic transient').click(); expect(IonicTransientClient.prototype.submit).not.toHaveBeenCalled()
})
