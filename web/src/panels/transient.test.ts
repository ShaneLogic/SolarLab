import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import Plotly from 'plotly.js-basic-dist-min'
import { TransientClient } from '../transient-client'
import type { TransientJob } from '../transient-client'
import { catalog, protocol, resultFixture } from '../test-helpers/transient-fixture'
import type { RunSubscription, StoredDocument, StoredEvent, StreamEnd, TerminalRecord } from '../run-client'
import { mountTransientPanel } from './transient'
import type { TransientPanel } from './transient'
import { webcrypto } from '../test-helpers/schema-fixture.mjs'

vi.mock('plotly.js-basic-dist-min', () => ({ default: { newPlot: vi.fn().mockResolvedValue(undefined), purge: vi.fn() } }))
const cursor = { endpoint: 'http://localhost/steady-service/', storeId: 'store', runId: 'run', attemptId: 'attempt', sequence: '0' }
let panel: TransientPanel, root: HTMLElement, job: TransientJob, sequence: number
let emit: (event: StoredDocument<StoredEvent>) => void | Promise<void>, endStream: (end: StreamEnd) => void
let stop: ReturnType<typeof vi.fn>
function button(text: string) { return [...root.querySelectorAll<HTMLButtonElement>('button')].find(node => node.textContent === text)! }
async function event(kind: StoredEvent['kind'], payload = {}) {
  await emit({ json: '{}', value: { sequence: ++sequence, kind, run_id: 'run', attempt_id: 'attempt', payload } })
}
async function submit() { button('Run transient').click(); await vi.waitFor(() => expect(emit).toBeTypeOf('function')) }
async function finish(state: TerminalRecord['state']) {
  await event('terminal', { state, artifacts: [], result: { reason: 'stored reason' } })
  endStream({ reason: 'terminal', cursor }); await panel.completion
}
beforeEach(async () => {
  vi.stubGlobal('crypto', webcrypto); vi.clearAllMocks(); sequence = 0
  root = document.createElement('div'); document.body.append(root)
  emit = undefined as unknown as typeof emit; stop = vi.fn()
  const done = new Promise<StreamEnd>(resolve => { endStream = resolve })
  job = { target: { runId: 'run', attemptId: 'attempt' }, submitted: { catalog, input: structuredClone(protocol), device: catalog.routes[0].devices[0] },
    reader: { storeId: 'store', getAttempt: vi.fn().mockResolvedValue({ value: { state: 'queued', terminal: null } }),
      subscribe: vi.fn((_target, callback) => { emit = callback; return { cursor, done, unsubscribe: stop } as RunSubscription }) } as unknown as TransientJob['reader'] }
  vi.spyOn(TransientClient.prototype, 'discover').mockResolvedValue(catalog)
  vi.spyOn(TransientClient.prototype, 'submit').mockResolvedValue(job)
  vi.spyOn(TransientClient.prototype, 'open').mockReturnValue(job)
  vi.spyOn(TransientClient.prototype, 'cancel').mockResolvedValue(undefined)
  vi.spyOn(TransientClient.prototype, 'readResult').mockResolvedValue(resultFixture())
  panel = mountTransientPanel(root, 'http://localhost/steady-service/'); await panel.ready
})
afterEach(() => { panel.dispose(); root.remove(); vi.restoreAllMocks(); vi.unstubAllGlobals() })

it('submits explicit installed samples, follows states, and plots temporal/spatial fields with units', async () => {
  await submit()
  expect(TransientClient.prototype.submit).toHaveBeenCalledWith(catalog, catalog.routes[0], 'research_1d', protocol, 60, expect.stringMatching(/^transient-/))
  expect(root.querySelector('[data-role=state]')?.textContent).toBe('queued')
  await event('running'); expect(root.querySelector('[data-role=state]')?.textContent).toBe('running')
  await finish('succeeded')
  expect(root.querySelector('[data-role=metrics]')?.textContent).toContain('Unknown')
  expect(Plotly.newPlot).toHaveBeenCalledTimes(6)
  const trace = vi.mocked(Plotly.newPlot).mock.calls.find(([div]) => (div as HTMLElement).dataset.plot === 'Terminal currents over time')!
  expect(trace[1][0]).toMatchObject({ x: [1e-7, 2e-7], y: [-40, -42], mode: 'lines+markers' })
  expect(trace[2]).toMatchObject({ xaxis: { range: [0, 2e-7] }, yaxis: { title: { text: 'Current density (A/m²)' } } })
  expect(root.querySelector<HTMLAnchorElement>('[data-role=readback-link]')?.href).toContain('attemptId=attempt')
})

it('reopens the failed prefix and all requested rows without submitting or relabelling next-run defaults', async () => {
  const result = resultFixture(true)
  vi.mocked(TransientClient.prototype.readResult).mockResolvedValue(result)
  vi.mocked(job.reader.getAttempt).mockResolvedValue({ value: { state: 'failed', terminal: {
    state: 'failed', result: { reason: 'budget_exhausted' }, artifacts: [],
  } } } as never)
  vi.mocked(job.reader.subscribe).mockReturnValue({ cursor, unsubscribe: stop, done: Promise.resolve({ reason: 'terminal_acknowledged', cursor }) })
  await panel.openStored({ storeId: 'store', runId: 'run', attemptId: 'attempt' })
  expect(TransientClient.prototype.submit).not.toHaveBeenCalled()
  expect(root.querySelector('[data-role=state]')?.textContent).toBe('failed')
  expect(root.querySelector('[data-role=first-failure]')?.textContent).toContain('1e-7 → 2e-7 s')
  const rows = [...root.querySelectorAll('[data-role=requested-protocol] tr')].map(row => [...row.querySelectorAll('td')].map(cell => cell.textContent))
  expect(rows.slice(1)).toEqual([['0', '0', '1', 'Initial DC'], ['1e-7', '0', '1', 'Accepted'], ['2e-7', '0', '1.5', 'Failed step'], ['3e-7', '0', '1', 'Not attempted']])
  expect(root.querySelector<HTMLTextAreaElement>('[name=samples]')!.value.split('\n')).toHaveLength(3)
  const trace = vi.mocked(Plotly.newPlot).mock.calls.find(([div]) => (div as HTMLElement).dataset.plot === 'Terminal currents over time')!
  expect(trace[1][0]).toMatchObject({ x: [1e-7], y: [-40], mode: 'lines+markers' })
  expect(trace[2]?.xaxis).toMatchObject({ range: [0, 1e-7] })
  await panel.reconnect()
  expect(TransientClient.prototype.submit).not.toHaveBeenCalled()
  expect(root.querySelector('[data-role=transport]')?.textContent).toContain('terminal_acknowledged')
})

it('shows initial-only failure with no invented currents', async () => {
  vi.mocked(TransientClient.prototype.readResult).mockResolvedValue(resultFixture(true, true))
  await submit(); await finish('failed')
  expect(root.textContent).toContain('No transient step was accepted')
  const current = vi.mocked(Plotly.newPlot).mock.calls.find(([div]) => (div as HTMLElement).dataset.plot === 'Terminal currents over time')!
  expect(current[1].every(trace => (trace.x as number[]).length === 0 && (trace.y as number[]).length === 0)).toBe(true)
  expect((current[2]?.xaxis as { range: number[] }).range.every(Number.isFinite)).toBe(true)
})

it('requests exact cancellation, waits for the terminal, and never draws a cancelled history', async () => {
  vi.mocked(TransientClient.prototype.readResult).mockResolvedValue(undefined)
  await submit(); await event('running'); button('Cancel this attempt').click()
  await vi.waitFor(() => expect(TransientClient.prototype.cancel).toHaveBeenCalledWith(job))
  expect(root.querySelector('[data-role=state]')?.textContent).toBe('running')
  await finish('cancelled')
  expect(root.querySelector('[data-role=state]')?.textContent).toBe('cancelled')
  expect(root.textContent).toContain('No transient history was committed')
  expect(Plotly.newPlot).not.toHaveBeenCalled()
})

it('unsubscribes on disposal without cancelling and ignores late terminal callbacks', async () => {
  await submit(); panel.dispose(); await event('terminal', { state: 'succeeded', result: {}, artifacts: [] })
  endStream({ reason: 'unsubscribed', cursor }); await panel.completion
  expect(stop).toHaveBeenCalled(); expect(TransientClient.prototype.cancel).not.toHaveBeenCalled()
  expect(Plotly.newPlot).not.toHaveBeenCalled(); expect(root.textContent).toBe('')
})

it('does not offer a fallback when no compatible transient route is installed', async () => {
  panel.dispose(); vi.mocked(TransientClient.prototype.discover).mockResolvedValue({ ...catalog, routes: [] })
  panel = mountTransientPanel(root, 'http://localhost/steady-service/'); await panel.ready
  expect(root.textContent).toContain('No compatible ion-free transient route')
  button('Run transient').click(); expect(TransientClient.prototype.submit).not.toHaveBeenCalled()
})
