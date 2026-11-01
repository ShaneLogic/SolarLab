import { afterEach, beforeAll, beforeEach, expect, it, vi } from 'vitest'
import Plotly from 'plotly.js-basic-dist-min'
import { IonicJVClient } from '../ionic-jv-client'
import type { IonicJVCatalog, IonicJVJob, IonicJVResult } from '../ionic-jv-client'
import { fixtureFetch, savedIonicJVFixture } from '../test-helpers/ionic-jv-fixture.mjs'
import { webcrypto } from '../test-helpers/schema-fixture.mjs'
import type { RunSubscription, StoredDocument, StoredEvent, StreamEnd, TerminalRecord } from '../run-client'
import { mountIonicJVPanel } from './ionic-jv'
import type { IonicJVPanel } from './ionic-jv'

vi.mock('plotly.js-basic-dist-min', () => ({ default: { newPlot: vi.fn().mockResolvedValue(undefined), purge: vi.fn() } }))
const endpoint = 'http://localhost/steady-service/'
const cursor = { endpoint, storeId: 'store', runId: 'run', attemptId: 'attempt', sequence: '0' }
let catalog: IonicJVCatalog, success: IonicJVResult, partial: IonicJVResult
let panel: IonicJVPanel, root: HTMLElement, job: IonicJVJob, sequence: number
let emit: (event: StoredDocument<StoredEvent>) => void | Promise<void>, endStream: (end: StreamEnd) => void
let stop: ReturnType<typeof vi.fn>
const plot = (name: string) => vi.mocked(Plotly.newPlot).mock.calls.find(([div]) => (div as HTMLElement).dataset.plot === name)!
const button = (text: string) => [...root.querySelectorAll<HTMLButtonElement>('button')].find(b => b.textContent === text)!
async function event(kind: StoredEvent['kind'], payload = {}) {
  await emit({ json: '{}', value: { sequence: ++sequence, kind, run_id: 'run', attempt_id: 'attempt', payload } })
}
async function submit() { button('Run ionic J–V').click(); await vi.waitFor(() => expect(emit).toBeTypeOf('function')) }
async function finish(state: TerminalRecord['state']) {
  await event('terminal', { state, artifacts: [], result: { reason: 'stored reason' } })
  endStream({ reason: 'terminal', cursor }); await panel.completion
}
beforeAll(async () => {
  // Layout/lifecycle inputs are decoded genuine producer artifacts, never numerical stand-ins.
  vi.stubGlobal('crypto', webcrypto); const fixture = savedIonicJVFixture(); vi.stubGlobal('fetch', fixtureFetch(fixture))
  const client = new IonicJVClient(endpoint); catalog = await client.discover()
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
    subscribe: vi.fn((_target, callback) => { emit = callback; return { cursor, done, unsubscribe: stop } as RunSubscription }) } as unknown as IonicJVJob['reader'] }
  vi.spyOn(IonicJVClient.prototype, 'discover').mockResolvedValue(catalog)
  vi.spyOn(IonicJVClient.prototype, 'submit').mockResolvedValue(job)
  vi.spyOn(IonicJVClient.prototype, 'open').mockReturnValue(job)
  vi.spyOn(IonicJVClient.prototype, 'cancel').mockResolvedValue(undefined)
  vi.spyOn(IonicJVClient.prototype, 'readResult').mockResolvedValue(success)
  panel = mountIonicJVPanel(root, endpoint); await panel.ready
})
afterEach(() => { panel.dispose(); root.remove(); vi.restoreAllMocks(); vi.unstubAllGlobals() })

it('shows installed fixed controls and plots accepted J-V, species, inventories and steady current terms', async () => {
  expect(root.querySelector('[data-role=installed-protocol]')?.textContent).toContain('51 explicit samples')
  await submit()
  expect(IonicJVClient.prototype.submit).toHaveBeenCalledWith(catalog, catalog.routes[0], 'research_1d_shared', catalog.routes[0].input, expect.stringMatching(/^ionic-jv-/))
  await event('running'); await finish('succeeded')
  expect(root.querySelector('[data-role=state]')?.textContent).toBe('succeeded')
  expect(root.querySelector('[data-metric=efficiency]')?.textContent).toContain('Unavailable')
  expect(root.querySelector('[data-role=counts]')?.textContent).toContain('"total_solver_calls":53')
  expect(Plotly.newPlot).toHaveBeenCalledTimes(8)
  expect(plot('Accepted J–V samples')[1][0]).toMatchObject({ x: success.arrays.voltage_V.values, y: success.arrays.output_current_A_m2.values })
  expect(plot('Accepted J–V samples')[2]).toMatchObject({ xaxis: { title: { text: 'Accepted voltage (V)' } }, yaxis: { title: { text: 'Output current density (A/m²)' } } })
  expect(plot('Ion densities in space')[1].map(t => t.name)).toEqual(['positive', 'negative'])
  expect(plot('Ion densities in space')[1][0].x).toHaveLength(13)
  expect(plot('Potential in space')[1][0].x).toHaveLength(37)
  expect(plot('Steady currents in space')[1].map(t => t.name)).toEqual(['electronic conduction', 'ionic conduction', 'total conduction'])
  expect(plot('Steady currents in space')[1][0].x).toHaveLength(38)
  expect(plot('Independent species inventories')[1][0].y).toEqual(success.arrays.positive_ion_inventory_m2.values)
  expect(root.querySelector<HTMLAnchorElement>('[data-role=readback-link]')?.href).toContain('view=ionic-jv')
})

it('opens and reconnects the genuine failed prefix without submission or relabeling its unattempted tail', async () => {
  vi.mocked(IonicJVClient.prototype.readResult).mockResolvedValue(partial)
  vi.mocked(job.reader.getAttempt).mockResolvedValue({ value: { state: 'failed', terminal: { state: 'failed', result: { reason: 'wall_budget_exhausted' }, artifacts: [] } } } as never)
  vi.mocked(job.reader.subscribe).mockReturnValue({ cursor, unsubscribe: stop, done: Promise.resolve({ reason: 'terminal_acknowledged', cursor }) })
  await panel.openStored({ storeId: 'store', runId: 'run', attemptId: 'attempt' })
  expect(root.querySelector('[data-role=state]')?.textContent).toBe('failed')
  expect(root.querySelector('[data-role=first-failure]')?.textContent).toContain('0.225 V (sample 9)')
  expect(root.querySelector('[data-role=history-summary]')?.textContent).toContain('Historical process attempt')
  expect(root.querySelectorAll('tr[data-outcome=Accepted]')).toHaveLength(9)
  expect(root.querySelectorAll('tr[data-outcome=Failed]')).toHaveLength(1)
  expect(root.querySelectorAll('tr[data-outcome="Not attempted"]')).toHaveLength(41)
  expect(root.querySelector('[data-metric=open_circuit_voltage_V]')?.textContent).toContain('Unavailable')
  expect(plot('Accepted J–V samples')[1][0].x).toEqual(partial.arrays.voltage_V.values)
  await panel.reconnect(); expect(IonicJVClient.prototype.submit).not.toHaveBeenCalled()
  expect(root.querySelector('[data-role=transport]')?.textContent).toContain('terminal_acknowledged')
})

it('changes the selected accepted state on its electrical and ionic grids without any new job', async () => {
  await submit(); await finish('succeeded')
  const select = root.querySelector<HTMLSelectElement>('[name=accepted_voltage]')!
  select.value = '50'; select.dispatchEvent(new Event('change'))
  await vi.waitFor(() => expect(Plotly.newPlot).toHaveBeenCalledTimes(13))
  const last = vi.mocked(Plotly.newPlot).mock.calls.filter(([div]) => (div as HTMLElement).dataset.plot === 'Ion densities in space').at(-1)!
  expect(last[1][0].y).toEqual(success.arrays.positive_ion_density_m3.values.slice(50 * 13))
  expect(IonicJVClient.prototype.submit).toHaveBeenCalledTimes(1)
})

it.each(['failed', 'cancelled'] as const)('keeps %s without a committed payload unavailable', async state => {
  vi.mocked(IonicJVClient.prototype.readResult).mockResolvedValue(undefined)
  await submit(); await event('running')
  if (state === 'cancelled') {
    button('Cancel this attempt').click(); await vi.waitFor(() => expect(IonicJVClient.prototype.cancel).toHaveBeenCalledWith(job))
    expect(root.querySelector('[data-role=state]')?.textContent).toBe('running')
  }
  await finish(state)
  expect(root.querySelector('[data-role=state]')?.textContent).toBe(state)
  expect(root.textContent).toContain('No ionic J-V curve was committed'); expect(Plotly.newPlot).not.toHaveBeenCalled()
})

it('disposes subscriptions and plots without cancelling or resubmitting the runtime job', async () => {
  await submit(); panel.dispose(); await event('terminal', { state: 'succeeded', result: {}, artifacts: [] })
  endStream({ reason: 'unsubscribed', cursor }); await panel.completion
  expect(stop).toHaveBeenCalled(); expect(IonicJVClient.prototype.cancel).not.toHaveBeenCalled()
  expect(Plotly.newPlot).not.toHaveBeenCalled(); expect(root.textContent).toBe('')
})

it('leaves readback available when the live installation has no compatible ionic J-V protocol', async () => {
  panel.dispose(); vi.mocked(IonicJVClient.prototype.discover).mockResolvedValue({ ...catalog, routes: [] })
  panel = mountIonicJVPanel(root, endpoint); await panel.ready
  expect(root.textContent).toContain('No compatible illuminated shared-site ionic J-V protocol')
  expect(button('Run ionic J–V').disabled).toBe(true)
  expect(IonicJVClient.prototype.submit).not.toHaveBeenCalled()
})
