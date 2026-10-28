import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import Plotly from 'plotly.js-basic-dist-min'
import { SteadyJVClient } from '../steady-jv-client'
import type { SteadyCatalog, SteadyJob, SteadyResult } from '../steady-jv-client'
import type { RunSubscription, StoredDocument, StoredEvent, StreamEnd, TerminalRecord } from '../run-client'
import { mountSteadyJVPanel } from './steady-jv'
import type { SteadyJVPanel } from './steady-jv'
import { webcrypto } from '../test-helpers/schema-fixture.mjs'

vi.mock('plotly.js-basic-dist-min', () => ({ default: { newPlot: vi.fn().mockResolvedValue(undefined), purge: vi.fn() } }))
const catalog: SteadyCatalog = { registry: 'a'.repeat(64), schema: 'b'.repeat(64),
  routes: [{ id: 'steady_jv', version: 1, devices: [{ id: 'research_1d', preparedSha256: 'c'.repeat(64) }],
    minPoints: 2, maxPoints: 101, minIntervals: 2, maxIntervals: 256 }],
  ceiling: { wall_seconds: 60, rss_bytes: 1610612736, output_bytes: 100663296 } }
let panel: SteadyJVPanel, root: HTMLElement, job: SteadyJob
let emit: (event: StoredDocument<StoredEvent>) => void | Promise<void>
let endStream: (end: StreamEnd) => void
let stop: ReturnType<typeof vi.fn>
const cursor = { endpoint: 'http://localhost/steady-service/', storeId: 'store', runId: 'run', attemptId: 'attempt', sequence: '0' }

function result(): SteadyResult {
  const fields = { shape: [3, 3], values: [1, 2, 3, 4, 5, 6, 7, 8, 9] }
  return { arrays: { nodes_m: { shape: [3], values: [0, 1e-7, 2e-7] }, voltage_V: { shape: [3], values: [0, .05, .1] },
    output_current_A_m2: { shape: [3], values: [42, 41, 40] }, current_density_A_m2: { shape: [3, 2], values: [1, 2, 3, 4, 5, 6] },
    contact_current_density_A_m2: { shape: [3, 2], values: [1, 2, 3, 4, 5, 6] }, potential_V: fields, electron_density_m3: fields, hole_density_m3: fields },
  metrics: { short_circuit_current_A_m2: 42, open_circuit_voltage_V: null, fill_factor: null, efficiency: null, limitation: 'open_circuit_not_bracketed' },
  evidence: [], worker: {}, originalJson: '{"test":"stored attempt"}' }
}
async function lifecycle(kind: StoredEvent['kind'], payload = {}) {
  await emit({ json: '{}', value: { sequence: 1, kind, run_id: 'run', attempt_id: 'attempt', payload } })
}
async function finish(state: TerminalRecord['state']) {
  await lifecycle('terminal', { state, artifacts: [], result: { reason: 'stored reason' } })
  endStream({ reason: 'terminal', cursor }); await panel.completion
}
function button(label: string): HTMLButtonElement {
  return [...root.querySelectorAll('button')].find(node => node.textContent === label)!
}
async function submit() {
  button('Run steady DC').click()
  await vi.waitFor(() => expect(emit).toBeTypeOf('function'))
}
beforeEach(async () => {
  vi.stubGlobal('crypto', webcrypto); vi.clearAllMocks()
  root = document.createElement('div'); document.body.append(root)
  emit = undefined as unknown as typeof emit
  stop = vi.fn()
  const done = new Promise<StreamEnd>(resolve => { endStream = resolve })
  job = { reader: { getAttempt: vi.fn().mockResolvedValue({ value: { state: 'queued', terminal: null } }),
    subscribe: vi.fn((_target, callback) => { emit = callback; return { cursor, done, unsubscribe: stop } as RunSubscription }) } as unknown as SteadyJob['reader'],
  target: { runId: 'run', attemptId: 'attempt' }, protocol: { voltage_V: [0, .05, .1], intervals_per_layer: 12, generation_scale: 1, incident_power_W_m2: null },
  device: catalog.routes[0].devices[0], catalog }
  vi.spyOn(SteadyJVClient.prototype, 'discover').mockResolvedValue(catalog)
  vi.spyOn(SteadyJVClient.prototype, 'submit').mockResolvedValue(job)
  vi.spyOn(SteadyJVClient.prototype, 'cancel').mockResolvedValue(undefined)
  vi.spyOn(SteadyJVClient.prototype, 'readResult').mockResolvedValue(result())
  panel = mountSteadyJVPanel(root, 'http://localhost/steady-service/'); await panel.ready
})
afterEach(() => { panel.dispose(); root.remove(); vi.restoreAllMocks(); vi.unstubAllGlobals() })

it('selects installed data, sends explicit DC controls, follows real states and plots the eight-array view', async () => {
  expect(root.querySelector('select[name=device]')?.textContent).toBe('research_1d')
  await submit()
  expect(SteadyJVClient.prototype.submit).toHaveBeenCalledWith(catalog, catalog.routes[0], 'research_1d',
    { voltage_V: [0, .05, .1], intervals_per_layer: 12, generation_scale: 1, incident_power_W_m2: null }, 60, expect.stringMatching(/^steady-/))
  expect(root.querySelector('[data-role=state]')?.textContent).toBe('queued')
  await lifecycle('running'); expect(root.querySelector('[data-role=state]')?.textContent).toBe('running')
  await finish('succeeded')
  expect(root.querySelector('[data-role=state]')?.textContent).toBe('succeeded')
  expect(root.querySelector('[data-metric=open_circuit_voltage_V]')?.textContent).toContain('Unknown')
  expect(root.querySelector('[data-metric=efficiency]')?.textContent).toContain('Unknown')
  expect(root.textContent).toContain('not bracketed')
  expect(Plotly.newPlot).toHaveBeenCalledTimes(4)
  const jv = vi.mocked(Plotly.newPlot).mock.calls.find(([div]) => (div as HTMLElement).dataset.plot === 'J–V')!
  expect(jv[1][0].y).toEqual([42, 41, 40]); expect(jv[2]).toMatchObject({ yaxis: { title: { text: 'Output current (A/m²)' } } })
  const sample = root.querySelector<HTMLSelectElement>('[data-role=sample]')!; sample.value = '2'; sample.dispatchEvent(new Event('change'))
  await vi.waitFor(() => expect(Plotly.newPlot).toHaveBeenCalledTimes(7))
  const potential = vi.mocked(Plotly.newPlot).mock.calls.filter(([div]) => (div as HTMLElement).dataset.plot === 'Potential').at(-1)!
  expect(potential[1][0].y).toEqual([7, 8, 9])
})

it.each(['failed', 'cancelled'] as const)('retains %s terminal state/reason without fake results', async state => {
  await submit(); await finish(state)
  expect(root.querySelector('[data-role=state]')?.textContent).toBe(state)
  expect(root.querySelector('[role=alert]')?.textContent).toContain('stored reason')
  expect(SteadyJVClient.prototype.readResult).not.toHaveBeenCalled(); expect(Plotly.newPlot).not.toHaveBeenCalled()
})

it('requests cancellation for the bound attempt but waits for its persisted terminal', async () => {
  await submit(); await lifecycle('running'); button('Cancel this attempt').click()
  await vi.waitFor(() => expect(SteadyJVClient.prototype.cancel).toHaveBeenCalledWith(job))
  expect(root.querySelector('[data-role=state]')?.textContent).toBe('running')
  await finish('cancelled'); expect(root.querySelector('[data-role=state]')?.textContent).toBe('cancelled')
})

it('disposal unsubscribes without cancelling computation or accepting late results', async () => {
  await submit(); panel.dispose(); await lifecycle('terminal', { state: 'succeeded', result: {}, artifacts: [] })
  endStream({ reason: 'unsubscribed', cursor }); await panel.completion
  expect(stop).toHaveBeenCalled(); expect(SteadyJVClient.prototype.cancel).not.toHaveBeenCalled()
  expect(Plotly.newPlot).not.toHaveBeenCalled(); expect(root.textContent).toBe('')
})

it('does not offer execution when the service has no compatible installation', async () => {
  panel.dispose(); vi.mocked(SteadyJVClient.prototype.discover).mockResolvedValue({ ...catalog, routes: [] })
  panel = mountSteadyJVPanel(root, 'http://localhost/steady-service/'); await panel.ready
  expect(root.textContent).toContain('No compatible classical DC route')
  button('Run steady DC').click(); expect(SteadyJVClient.prototype.submit).not.toHaveBeenCalled()
})
