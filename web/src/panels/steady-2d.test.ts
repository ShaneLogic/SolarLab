import { afterEach, beforeAll, beforeEach, expect, it, vi } from 'vitest'
import Plotly from 'plotly.js-basic-dist-min'
import { Steady2DClient } from '../steady-2d-client'
import type { Steady2DCatalog, Steady2DJob, Steady2DResult } from '../steady-2d-client'
import { fixtureFetch, savedSteady2DFixture } from '../test-helpers/steady-2d-fixture.mjs'
import { webcrypto } from '../test-helpers/schema-fixture.mjs'
import type { RunSubscription, StoredDocument, StoredEvent, StreamEnd, TerminalRecord } from '../run-client'
import { mountSteady2DPanel } from './steady-2d'
import type { Steady2DPanel } from './steady-2d'

vi.mock('plotly.js-basic-dist-min', () => ({ default: { newPlot: vi.fn().mockResolvedValue(undefined), purge: vi.fn() } }))
const endpoint = 'http://localhost/steady-service/'
const cursor = { endpoint, storeId: 'store', runId: 'run', attemptId: 'attempt', sequence: '0' }
let catalog: Steady2DCatalog, success: Steady2DResult, partial: Steady2DResult
let panel: Steady2DPanel, root: HTMLElement, job: Steady2DJob, sequence: number
let emit: (event: StoredDocument<StoredEvent>) => void | Promise<void>, endStream: (end: StreamEnd) => void
let stop: ReturnType<typeof vi.fn>
const plot = (name: string) => vi.mocked(Plotly.newPlot).mock.calls.find(([div]) => (div as HTMLElement).dataset.plot === name)!
const button = (text: string) => [...root.querySelectorAll<HTMLButtonElement>('button')].find(b => b.textContent === text)!
async function event(kind: StoredEvent['kind'], payload = {}) {
  await emit({ json: '{}', value: { sequence: ++sequence, kind, run_id: 'run', attempt_id: 'attempt', payload } })
}
async function submit() { button('Run 2D steady').click(); await vi.waitFor(() => expect(emit).toBeTypeOf('function')) }
async function finish(state: TerminalRecord['state']) {
  await event('terminal', { state, artifacts: [], result: { reason: 'stored reason' } })
  endStream({ reason: 'terminal', cursor }); await panel.completion
}
beforeAll(async () => {
  // Layout/lifecycle inputs are decoded genuine producer artifacts, never numerical stand-ins.
  vi.stubGlobal('crypto', webcrypto); const fixture = savedSteady2DFixture(); vi.stubGlobal('fetch', fixtureFetch(fixture))
  const client = new Steady2DClient(endpoint); catalog = await client.discover()
  const results = []
  for (const target of fixture.targets.slice(0, 2)) results.push((await client.readResult(client.open(target.storeId, target.runId, target.attemptId)))!)
  ;[success, partial] = results
  vi.unstubAllGlobals()
})
beforeEach(async () => {
  vi.stubGlobal('crypto', webcrypto); vi.clearAllMocks(); vi.mocked(Plotly.newPlot).mockResolvedValue(undefined)
  sequence = 0; root = document.createElement('div'); document.body.append(root); stop = vi.fn(); emit = undefined as unknown as typeof emit
  const done = new Promise<StreamEnd>(resolve => { endStream = resolve })
  job = { target: { runId: 'run', attemptId: 'attempt' }, reader: { storeId: 'store',
    getAttempt: vi.fn().mockResolvedValue({ value: { state: 'queued', terminal: null } }),
    subscribe: vi.fn((_target, callback) => { emit = callback; return { cursor, done, unsubscribe: stop } as RunSubscription }) } as unknown as Steady2DJob['reader'] }
  vi.spyOn(Steady2DClient.prototype, 'discover').mockResolvedValue(catalog)
  vi.spyOn(Steady2DClient.prototype, 'submit').mockResolvedValue(job)
  vi.spyOn(Steady2DClient.prototype, 'open').mockReturnValue(job)
  vi.spyOn(Steady2DClient.prototype, 'cancel').mockResolvedValue(undefined)
  vi.spyOn(Steady2DClient.prototype, 'readResult').mockResolvedValue(success)
  panel = mountSteady2DPanel(root, endpoint); await panel.ready
})
afterEach(() => { panel.dispose(); root.remove(); vi.restoreAllMocks(); vi.unstubAllGlobals() })

it('shows installed 2D controls and actual potential, carriers, face and section currents', async () => {
  expect(root.querySelector('[data-role=installed-protocol]')?.textContent).toContain('195 full-2D coordinates')
  await submit()
  expect(Steady2DClient.prototype.submit).toHaveBeenCalledWith(catalog, catalog.routes[0], 'research_2d_uniform', catalog.routes[0].input, expect.stringMatching(/^steady-2d-/))
  await event('running'); await finish('succeeded')
  expect(root.querySelector('[data-role=state]')?.textContent).toBe('succeeded')
  expect(root.querySelectorAll('tr[data-outcome=Accepted]')).toHaveLength(2)
  expect(root.querySelector('[data-role=counts]')?.textContent).toContain('"full_2d_newton_calls":2')
  expect(root.querySelector('[data-role=counts]')?.textContent).toContain('PV metrics: unavailable')
  expect(Plotly.newPlot).toHaveBeenCalledTimes(5)
  expect(plot('2D potential')[1][0].customdata).toEqual(success.arrays.potential_V.values.slice(0, 65))
  expect(plot('2D electron density')[1][0].customdata).toEqual(success.arrays.electron_density_m3.values.slice(0, 65))
  expect(plot('2D hole density')[1][0].customdata).toEqual(success.arrays.hole_density_m3.values.slice(0, 65))
  expect(plot('2D potential')[1][0].x).toEqual(Array.from({ length: 13 }, () => success.arrays.x_m.values.map(v => v * 1e9)).flat())
  expect(plot('2D potential')[1][0].y).toEqual(success.arrays.y_m.values.flatMap(v => Array(5).fill(v * 1e9)))
  expect(plot('2D potential')[2]).toMatchObject({ xaxis: { title: { text: 'Lateral x (nm)' } }, yaxis: { title: { text: 'Stack y (nm)' } } })
  expect(plot('Directed face current density')[1][0].customdata).toEqual(success.arrays.face_current_density_A_m2.values.slice(0, 112))
  expect(plot('Terminal and cross-section currents')[1][0].y).toEqual(success.storedArrays.section_terminal_current_A_m.values.slice(0, 14))
  expect(plot('Terminal and cross-section currents')[2]).toMatchObject({ yaxis: { title: { text: 'Current per depth (A/m)' } } })
  expect(root.querySelector<HTMLAnchorElement>('[data-role=readback-link]')?.href).toContain('view=steady-2d')
})

it('submits only the explicit selected G1 target with unchanged geometry and controls', async () => {
  const select = root.querySelector<HTMLSelectElement>('[name=generation]')!
  select.value = '1'; select.dispatchEvent(new Event('change')); await submit()
  expect(Steady2DClient.prototype.submit).toHaveBeenCalledWith(catalog, catalog.routes[0], 'research_2d_uniform',
    { ...catalog.routes[0].input, generation_scales: [1] }, expect.any(String))
  endStream({ reason: 'disconnected', cursor }); await panel.completion
})

it('reopens and reconnects the historical genuine failure without plotting a candidate or posting', async () => {
  vi.mocked(Steady2DClient.prototype.readResult).mockResolvedValue(partial)
  vi.mocked(job.reader.getAttempt).mockResolvedValue({ value: { state: 'failed', terminal: { state: 'failed', result: { reason: 'budget_exhausted' }, artifacts: [] } } } as never)
  vi.mocked(job.reader.subscribe).mockReturnValue({ cursor, unsubscribe: stop, done: Promise.resolve({ reason: 'terminal_acknowledged', cursor }) })
  await panel.openStored({ storeId: 'store', runId: 'run', attemptId: 'attempt' })
  expect(root.querySelector('[data-role=state]')?.textContent).toBe('failed')
  expect(root.querySelector('[data-role=first-failure]')?.textContent).toContain('G = 0 (index 0) · budget_exhausted')
  expect(root.querySelector('[data-role=history-summary]')?.textContent).toContain('Historical process attempt')
  expect(root.querySelectorAll('tr[data-outcome=Accepted]')).toHaveLength(0)
  expect(root.querySelectorAll('tr[data-outcome=Failed]')).toHaveLength(1)
  expect(root.querySelectorAll('tr[data-outcome="Not attempted"]')).toHaveLength(1)
  expect(root.textContent).toContain('No accepted 2D fields or currents are available')
  expect(root.querySelector('[data-role=provenance]')?.textContent).toContain('e73acd0a')
  expect(Plotly.newPlot).not.toHaveBeenCalled()
  await panel.reconnect(); expect(Steady2DClient.prototype.submit).not.toHaveBeenCalled()
  expect(root.querySelector('[data-role=transport]')?.textContent).toContain('terminal_acknowledged')
})

it('switches the accepted dark/G1 maps using saved rows with no new request', async () => {
  await submit(); await finish('succeeded')
  const select = root.querySelector<HTMLSelectElement>('[name=accepted_target]')!
  select.value = '1'; select.dispatchEvent(new Event('change'))
  await vi.waitFor(() => expect(Plotly.newPlot).toHaveBeenCalledTimes(10))
  const last = vi.mocked(Plotly.newPlot).mock.calls.filter(([div]) => (div as HTMLElement).dataset.plot === '2D potential').at(-1)!
  expect(last[1][0].customdata).toEqual(success.arrays.potential_V.values.slice(65))
  expect(Steady2DClient.prototype.submit).toHaveBeenCalledTimes(1)
})

it.each(['failed', 'cancelled'] as const)('keeps %s without committed fields unavailable', async state => {
  vi.mocked(Steady2DClient.prototype.readResult).mockResolvedValue(undefined)
  await submit(); await event('running')
  if (state === 'cancelled') {
    button('Cancel this attempt').click(); await vi.waitFor(() => expect(Steady2DClient.prototype.cancel).toHaveBeenCalledWith(job))
    expect(root.querySelector('[data-role=state]')?.textContent).toBe('running')
  }
  await finish(state)
  expect(root.querySelector('[data-role=state]')?.textContent).toBe(state)
  expect(root.textContent).toContain('No accepted 2D fields were committed'); expect(Plotly.newPlot).not.toHaveBeenCalled()
})

it('leaves historical reads available without a compatible installed route', async () => {
  panel.dispose(); vi.mocked(Steady2DClient.prototype.discover).mockResolvedValue({ ...catalog, routes: [] })
  panel = mountSteady2DPanel(root, endpoint); await panel.ready
  expect(root.textContent).toContain('No compatible no-grain 2D steady protocol')
  expect(button('Run 2D steady').disabled).toBe(true)
  expect(Steady2DClient.prototype.submit).not.toHaveBeenCalled()
})

it('disposes readers and plots without cancelling or resubmitting the runtime job', async () => {
  await submit(); panel.dispose(); await event('terminal', { state: 'succeeded', result: {}, artifacts: [] })
  endStream({ reason: 'unsubscribed', cursor }); await panel.completion
  expect(stop).toHaveBeenCalled(); expect(Steady2DClient.prototype.cancel).not.toHaveBeenCalled()
  expect(Plotly.newPlot).not.toHaveBeenCalled(); expect(root.textContent).toBe('')
})

