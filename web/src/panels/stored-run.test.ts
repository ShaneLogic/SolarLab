import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { StoredRunClient, RunClientError } from '../run-client'
import type {
  ArtifactRecord, AttemptIdentity, AttemptSnapshot, ReplayCursor, RunSnapshot,
  RunSubscription, StoredDocument, StoredEvent, StreamEnd, TerminalRecord,
} from '../run-client'
import { mountStoredRunPanel } from './stored-run'
import type { StoredRunPanel, StoredRunSelection } from './stored-run'

const selection: StoredRunSelection = {
  endpoint: 'http://127.0.0.1:8123/stored/', storeId: 'store-a', runId: 'run-a', attemptId: 'attempt-a',
}
const reference = { artifact_id: 'artifact-a', manifest_sha256: 'b'.repeat(64) }
const panels: StoredRunPanel[] = []
let connections: Connection[]
let blobs: Map<string, Blob>
let downloads: { href: string; filename: string }[]

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (error: unknown) => void
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no })
  return { promise, resolve, reject }
}

function fixtureJson(value: unknown): string {
  // Controlled producer fixtures: preserve integer and negative-zero tokens.
  return JSON.stringify(value, (_key, item) => typeof item === 'bigint' ? `fixture-bigint:${item}`
    : Object.is(item, -0) ? 'fixture-negative-zero' : item)
    .replace(/"fixture-bigint:(-?\d+)"/g, '$1').replace(/"fixture-negative-zero"/g, '-0.0')
}

function documentOf<T>(value: T, json = fixtureJson(value)): StoredDocument<T> { return { value, json } }

function run(target = selection, extra: Partial<RunSnapshot> = {}): RunSnapshot {
  return { run_id: target.runId, current_attempt_id: target.attemptId, state: 'running', generation: 1,
    created_ns: 18446744073709551617n,
    request: { identity_kind: 'opaque_upstream', identity: { H_input: 'opaque-input', H_physics: 'opaque-physics', H_execution: 'opaque-execution' },
      input_metadata: { zero: 0, negative_zero: -0, missing: null, integer: 18446744073709551617n,
        unsafe: '<img src=x onerror="globalThis.injected=true">', manifest: { future_schema: 'unknown', qualified: false } } }, ...extra }
}

function final(state: TerminalRecord['state'] = 'failed', result: Record<string, unknown> = { reason: 'persisted failure', qualification: 'not_evaluated' }): TerminalRecord {
  return { state, result, artifacts: [reference] }
}

function attempt(target = selection, end?: TerminalRecord): AttemptSnapshot {
  return { run_id: target.runId, attempt_id: target.attemptId, attempt_number: 1, state: end?.state ?? 'running', generation: 1,
    created_ns: 0, started_ns: 0, finished_ns: end ? 1 : null, owner_id: 'fixture',
    result: end?.result ?? null, terminal: end ?? null }
}

function event(sequence: number | bigint, kind: StoredEvent['kind'] = 'progress', payload: Record<string, unknown> = { zero: 0, missing: null }, target: AttemptIdentity = selection) {
  return documentOf<StoredEvent>({ sequence, run_id: target.runId, attempt_id: target.attemptId, generation: 1,
    created_ns: 0, kind, event_key: kind === 'progress' ? `user:event${sequence}` : kind, payload })
}

interface Connection {
  target: AttemptIdentity
  resume: ReplayCursor | undefined
  subscription: RunSubscription
  emit(doc: StoredDocument<StoredEvent>): Promise<void>
  finish(reason: StreamEnd['reason'], error?: unknown): void
  reject(error: unknown): void
}

function field(root: HTMLElement, role: string) { return root.querySelector<HTMLElement>(`[data-role="${role}"]`)! }
function button(root: HTMLElement, label: string) {
  const value = [...root.querySelectorAll('button')].find(node => node.textContent === label)
  if (!value) throw new Error(`Missing button: ${label}`)
  return value
}
async function mount(target = selection) {
  const root = window.document.createElement('div')
  window.document.body.append(root)
  const panel = mountStoredRunPanel(root, target)
  panels.push(panel)
  await panel.ready
  return { root, panel }
}
function blobText(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result as string)
    reader.onerror = () => reject(reader.error)
    reader.readAsText(blob)
  })
}

beforeEach(() => {
  connections = []; blobs = new Map(); downloads = []
  vi.stubGlobal('URL', class extends URL {
    static createObjectURL = vi.fn((blob: Blob) => { const id = `blob:fixture-${blobs.size}`; blobs.set(id, blob); return id })
    static revokeObjectURL = vi.fn()
  })
  vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) {
    downloads.push({ href: this.href, filename: this.download })
  })
  vi.spyOn(StoredRunClient.prototype, 'getRun').mockImplementation(async id => documentOf(run({ ...selection, runId: id })))
  vi.spyOn(StoredRunClient.prototype, 'getAttempt').mockImplementation(async target => documentOf(attempt({ ...selection, ...target })))
  vi.spyOn(StoredRunClient.prototype, 'getArtifact').mockResolvedValue({
    record: documentOf<ArtifactRecord>({ ...reference, run_id: selection.runId, attempt_id: selection.attemptId,
      sha256: 'c'.repeat(64), size_bytes: 3, phase: 'committed', metadata: { zero: -0, missing: null, integer: 18446744073709551617n } }),
    bytes: new Uint8Array([0, 128, 255]),
  })
  vi.spyOn(StoredRunClient.prototype, 'subscribe').mockImplementation(function (this: StoredRunClient, target, receive, resume) {
    let cursor: ReplayCursor = resume ?? { endpoint: this.endpoint, storeId: this.storeId, ...target, sequence: '0' }
    const end = deferred<StreamEnd>()
    const subscription: RunSubscription = { get cursor() { return cursor }, done: end.promise,
      unsubscribe: vi.fn(() => end.resolve({ reason: 'unsubscribed', cursor })) }
    connections.push({ target, resume, subscription,
      async emit(doc) { await receive(doc); cursor = { ...cursor, sequence: doc.value.sequence.toString() } },
      finish(reason, transportError) { end.resolve({ reason, cursor, transportError }) }, reject: end.reject })
    return subscription
  })
})

afterEach(() => {
  for (const panel of panels.splice(0)) panel.dispose()
  window.document.body.replaceChildren()
  vi.restoreAllMocks(); vi.unstubAllGlobals()
})

describe('stored-run panel records and lifecycle', () => {
  it('shows explicit identity and raw opaque metadata safely, without a JSON round trip', async () => {
    const { root } = await mount()
    expect(root.textContent).toContain(selection.endpoint)
    expect(root.textContent).toContain('Store: store-a')
    expect(root.textContent).toContain('Selected attempt: attempt-a')
    expect(field(root, 'attempt-state').textContent).toBe('running')
    const pre = field(root, 'documents').querySelector('pre')!
    expect(pre.textContent).toBe(fixtureJson(run()))
    expect(pre.textContent).toContain('18446744073709551617')
    expect(pre.textContent).toContain('"negative_zero":-0.0')
    expect(pre.textContent).toContain('"missing":null')
    expect(pre.textContent).not.toContain('omitted')
    expect(root.querySelector('img')).toBeNull()
    button(field(root, 'documents'), 'Download original JSON').click()
    expect(await blobText(blobs.get(downloads[0].href)!)).toBe(pre.textContent)
  })

  it('retains noncontiguous progress and never regresses a running snapshot to queued', async () => {
    const { root } = await mount()
    await connections[0].emit(event(1, 'queued', {}))
    expect(field(root, 'attempt-state').textContent).toBe('running')
    const progress = event(9223372036854775000n, 'progress', { fraction: 0, reason: null, unknown: -0 })
    await connections[0].emit(progress)
    expect(field(root, 'events').children).toHaveLength(2)
    expect(field(root, 'events').lastElementChild!.querySelector('pre')!.textContent).toBe(progress.json)
    expect(field(root, 'last-event').textContent).toBe('9223372036854775000: progress')
    expect(field(root, 'terminal-reason').textContent).toBe('No terminal record loaded')
  })

  it.each(['failed', 'cancelled', 'succeeded'] as const)('renders the persisted %s terminal independently of transport', async state => {
    const end = final(state)
    const { root } = await mount()
    await connections[0].emit(event(5, 'terminal', end))
    connections[0].finish('disconnected', new Error('lost socket'))
    await vi.waitFor(() => expect(field(root, 'transport').textContent).toContain('Transport disconnected'))
    expect(field(root, 'attempt-state').textContent).toBe(state)
    expect(field(root, 'terminal-reason').textContent).toBe('persisted failure')
    expect(root.textContent).toContain('not_evaluated')
    expect(field(root, 'transport').textContent).toContain('lost socket')
  })

  it('keeps the selected historical attempt separate from the current run attempt', async () => {
    vi.mocked(StoredRunClient.prototype.getRun).mockResolvedValue(documentOf(run(selection, { current_attempt_id: 'new-attempt', state: 'running' })))
    vi.mocked(StoredRunClient.prototype.getAttempt).mockResolvedValue(documentOf(attempt(selection, final())))
    const { root } = await mount()
    await connections[0].emit(event(1, 'queued', {}))
    await connections[0].emit(event(2, 'running', {}))
    expect(field(root, 'run-state').textContent).toBe('running; current attempt new-attempt')
    expect(field(root, 'attempt-state').textContent).toBe('failed')
    expect(connections[0].target.attemptId).toBe('attempt-a')
  })

  it.each([
    [{}, 'Omitted from the stored result'],
    [{ reason: null }, 'null (stored)'],
    [{ reason: 0 }, '0'],
    [{ reason: -0 }, '-0'],
    [{ reason: '' }, 'Empty string (stored)'],
    [{ reason: 18446744073709551617n }, '18446744073709551617'],
    [{ reason: { future: null } }, 'Structured value; see original terminal JSON'],
  ] as [Record<string, unknown>, string][])('preserves the meaning of stored reason case %#', async (result, expected) => {
    vi.mocked(StoredRunClient.prototype.getAttempt).mockResolvedValue(documentOf(attempt(selection, final('failed', result))))
    const { root } = await mount()
    expect(field(root, 'terminal-reason').textContent).toBe(expected)
  })

  it('reconnects with the exact client cursor and fences callbacks from the old connection', async () => {
    const { root, panel } = await mount()
    await connections[0].emit(event(9007199254740995n))
    const cursor = panel.cursor
    connections[0].finish('disconnected')
    await vi.waitFor(() => expect(field(root, 'transport').textContent).toContain('disconnected'))
    button(root, 'Reconnect event stream').click()
    expect(connections[0].subscription.unsubscribe).toHaveBeenCalled()
    expect(connections[1].resume).toBe(cursor)
    expect(connections[1].resume).toEqual({ endpoint: selection.endpoint, storeId: selection.storeId,
      runId: selection.runId, attemptId: selection.attemptId, sequence: '9007199254740995' })
    await connections[0].emit(event(9007199254740996n, 'terminal', final()))
    expect(field(root, 'attempt-state').textContent).toBe('running')
    expect(field(root, 'events').children).toHaveLength(1)
    await connections[1].emit(event(9007199254740998n))
    expect(field(root, 'events').children).toHaveLength(2)
  })

  it('retains terminal state through a clean close and HTTP 204 acknowledgement', async () => {
    const end = final('succeeded', { reason: null, qualification: 'unqualified' })
    const { root, panel } = await mount()
    await connections[0].emit(event(8, 'terminal', end))
    vi.mocked(StoredRunClient.prototype.getAttempt).mockResolvedValue(documentOf(attempt(selection, end)))
    connections[0].finish('terminal')
    await vi.waitFor(() => expect(field(root, 'documents').children).toHaveLength(4))
    panel.reconnect()
    connections[1].finish('terminal_acknowledged')
    await vi.waitFor(() => expect(field(root, 'transport').textContent).toContain('HTTP 204'))
    expect(field(root, 'attempt-state').textContent).toBe('succeeded')
    expect(field(root, 'terminal-reason').textContent).toBe('null (stored)')
    expect(field(root, 'artifacts').querySelectorAll('[data-artifact-id]')).toHaveLength(1)
  })

  it('reports a stream protocol error without rewriting the persisted terminal', async () => {
    const { root } = await mount()
    await connections[0].emit(event(5, 'terminal', final()))
    connections[0].reject(new RunClientError('protocol', 'event after terminal'))
    await vi.waitFor(() => expect(field(root, 'transport').textContent).toContain('protocol: event after terminal'))
    expect(field(root, 'attempt-state').textContent).toBe('failed')
    expect(field(root, 'events').children).toHaveLength(1)
  })

  it('unsubscribes on selection changes and cannot mix late events or identities', async () => {
    const { root, panel } = await mount()
    const next = { endpoint: 'http://127.0.0.1:9000/other/', storeId: 'other-store', runId: 'run-b', attemptId: 'attempt-b' }
    await panel.select(next)
    expect(connections[0].subscription.unsubscribe).toHaveBeenCalled()
    await connections[0].emit(event(3, 'terminal', final()))
    expect(root.textContent).toContain('Store: other-store')
    expect(root.textContent).not.toContain('persisted failure')
    expect(field(root, 'events').children).toHaveLength(0)
    expect(connections[1].resume).toBeUndefined()
    expect(panel.cursor).toMatchObject({ storeId: 'other-store', runId: 'run-b', attemptId: 'attempt-b', sequence: '0' })
  })

  it('fences late snapshot successes and failures after selection changes', async () => {
    const firstRun = deferred<StoredDocument<RunSnapshot>>()
    const firstAttempt = deferred<StoredDocument<AttemptSnapshot>>()
    vi.mocked(StoredRunClient.prototype.getRun).mockImplementation(id => id === 'run-a' ? firstRun.promise : Promise.resolve(documentOf(run({ ...selection, runId: id }))))
    vi.mocked(StoredRunClient.prototype.getAttempt).mockImplementation(target => target.runId === 'run-a' ? firstAttempt.promise : Promise.resolve(documentOf(attempt({ ...selection, ...target }))))
    const root = window.document.createElement('div'); window.document.body.append(root)
    const panel = mountStoredRunPanel(root, selection); panels.push(panel)
    const oldReady = panel.ready
    await panel.select({ ...selection, runId: 'run-b', attemptId: 'attempt-b' })
    firstRun.resolve(documentOf(run()))
    firstAttempt.reject(new Error('stale read failure'))
    await oldReady
    expect(root.textContent).not.toContain('stale read failure')
    expect(field(root, 'documents').children).toHaveLength(2)
    expect(connections).toHaveLength(1)
    expect(connections[0].target.runId).toBe('run-b')
  })

  it.each(['http', 'malformed', 'identity_mismatch'] as const)('surfaces %s read errors, keeps successful raw reads and never starts an unbound stream', async code => {
    vi.mocked(StoredRunClient.prototype.getAttempt).mockRejectedValue(new RunClientError(code, 'rejected record'))
    const { root } = await mount()
    expect(field(root, 'read-error').textContent).toContain(`${code}: rejected record`)
    expect(field(root, 'documents').children).toHaveLength(1)
    expect(field(root, 'attempt-state').textContent).toBe('Unavailable')
    expect(connections).toHaveLength(0)
    expect(button(root, 'Retry stored reads').disabled).toBe(false)
  })

  it('recovers an initial read failure by user retry, preserves history and opens only one stream', async () => {
    vi.mocked(StoredRunClient.prototype.getAttempt).mockRejectedValueOnce(new RunClientError('http', 'temporarily unavailable'))
    const { root, panel } = await mount()
    expect(connections).toHaveLength(0)
    const retry = deferred<StoredDocument<AttemptSnapshot>>()
    const end = final('succeeded', { reason: 'persisted completion', qualification: 'not_evaluated' })
    vi.mocked(StoredRunClient.prototype.getAttempt).mockReturnValueOnce(retry.promise)
      .mockResolvedValue(documentOf(attempt(selection, end)))
    const control = button(root, 'Retry stored reads')
    control.click()
    panel.reconnect(); control.click()
    expect(control.disabled).toBe(true)
    expect(StoredRunClient.prototype.getAttempt).toHaveBeenCalledTimes(2)
    expect(connections).toHaveLength(0)
    retry.resolve(documentOf(attempt()))
    await vi.waitFor(() => expect(connections).toHaveLength(1))
    expect(field(root, 'attempt-state').textContent).toBe('running')
    expect(field(root, 'read-error').textContent).toBe('')
    expect(field(root, 'read-error-history').textContent).toContain('http: temporarily unavailable')
    expect(field(root, 'documents').children).toHaveLength(3)
    expect(button(root, 'Reconnect event stream').disabled).toBe(false)
    await connections[0].emit(event(7, 'terminal', end))
    connections[0].finish('terminal')
    await vi.waitFor(() => expect(field(root, 'documents').children).toHaveLength(5))
    expect(field(root, 'attempt-state').textContent).toBe('succeeded')
    expect(field(root, 'terminal-reason').textContent).toBe('persisted completion')
    expect(field(root, 'read-error-history').children).toHaveLength(1)
    expect(connections).toHaveLength(1)
  })

  it('marks an unavailable run snapshot while preserving a successfully read attempt', async () => {
    vi.mocked(StoredRunClient.prototype.getRun).mockRejectedValue(new RunClientError('http', 'HTTP 503'))
    const { root } = await mount()
    expect(field(root, 'run-state').textContent).toBe('Unavailable')
    expect(field(root, 'attempt-state').textContent).toBe('running')
    expect(field(root, 'documents').children).toHaveLength(1)
    expect(field(root, 'read-error').textContent).toContain('Run read failed: http: HTTP 503')
    expect(connections).toHaveLength(0)
  })

  it('does not select a fallback endpoint for invalid explicit input', async () => {
    const { root } = await mount({ ...selection, endpoint: 'file:///tmp/store' })
    expect(root.textContent).toContain('Expected an explicit HTTP(S) service endpoint')
    expect(StoredRunClient.prototype.getRun).not.toHaveBeenCalled()
    expect(connections).toHaveLength(0)
  })

  it('disposes subscriptions, aborts in-flight reads, and ignores all late updates', async () => {
    const { root, panel } = await mount()
    const signal = vi.mocked(StoredRunClient.prototype.getAttempt).mock.calls[0][1]!
    panel.dispose()
    expect(signal.aborted).toBe(true)
    expect(connections[0].subscription.unsubscribe).toHaveBeenCalled()
    await connections[0].emit(event(8, 'terminal', final()))
    panel.reconnect(); await panel.select(selection)
    expect(root.childNodes).toHaveLength(0)
    expect(connections).toHaveLength(1)
    expect(panel.cursor).toBeUndefined()
  })
})

describe('verified stored downloads', () => {
  it('requests only the committed reference, downloads verified bytes and retains original record JSON', async () => {
    vi.mocked(StoredRunClient.prototype.getAttempt).mockResolvedValue(documentOf(attempt(selection, final('succeeded'))))
    const { root, panel } = await mount()
    button(root, 'Verify and download artifact').click()
    await vi.waitFor(() => expect(downloads).toHaveLength(1))
    expect(StoredRunClient.prototype.getArtifact).toHaveBeenCalledWith(
      { runId: selection.runId, attemptId: selection.attemptId }, reference, expect.any(AbortSignal))
    const blob = blobs.get(downloads[0].href)!
    const data = await new Promise<ArrayBuffer>((resolve, reject) => {
      const reader = new FileReader(); reader.onload = () => resolve(reader.result as ArrayBuffer)
      reader.onerror = () => reject(reader.error); reader.readAsArrayBuffer(blob)
    })
    expect([...new Uint8Array(data)]).toEqual([0, 128, 255])
    expect(field(root, 'artifacts').textContent).toContain('SHA-256 and size verified; download requested')
    expect(field(root, 'artifacts').querySelector('pre')!.textContent).toContain('"zero":-0.0')
    panel.dispose()
    expect(URL.revokeObjectURL).toHaveBeenCalledWith(downloads[0].href)
  })

  it('retains a download integrity failure without claiming the stored run failed', async () => {
    vi.mocked(StoredRunClient.prototype.getAttempt).mockResolvedValue(documentOf(attempt(selection, final('succeeded'))))
    vi.mocked(StoredRunClient.prototype.getArtifact).mockRejectedValue(new RunClientError('integrity', 'Artifact content integrity mismatch'))
    const { root } = await mount()
    button(root, 'Verify and download artifact').click()
    await vi.waitFor(() => expect(field(root, 'artifacts').textContent).toContain('integrity: Artifact content integrity mismatch'))
    expect(downloads).toHaveLength(0)
    expect(blobs.size).toBe(0)
    expect(field(root, 'attempt-state').textContent).toBe('succeeded')
    expect(button(root, 'Verify and download artifact').disabled).toBe(false)
  })

  it.each(['select', 'dispose'] as const)('fences a late verified artifact on %s', async action => {
    vi.mocked(StoredRunClient.prototype.getAttempt).mockResolvedValue(documentOf(attempt(selection, final('succeeded'))))
    const late = deferred<Awaited<ReturnType<StoredRunClient['getArtifact']>>>()
    vi.mocked(StoredRunClient.prototype.getArtifact).mockReturnValue(late.promise)
    const { root, panel } = await mount()
    button(root, 'Verify and download artifact').click()
    const signal = vi.mocked(StoredRunClient.prototype.getArtifact).mock.calls[0][2]!
    if (action === 'select') await panel.select({ ...selection, runId: 'run-b', attemptId: 'attempt-b' })
    else panel.dispose()
    late.resolve({ record: documentOf({ ...reference, run_id: 'run-a', attempt_id: 'attempt-a',
      sha256: 'c'.repeat(64), size_bytes: 1, phase: 'committed', metadata: {} }), bytes: new Uint8Array([255]) })
    await late.promise
    expect(signal.aborted).toBe(true)
    expect(downloads).toHaveLength(0)
    expect(root.textContent).not.toContain('Verified artifact record')
  })

  it('reports a browser JSON export failure and can be reopened from stored state', async () => {
    vi.mocked(URL.createObjectURL).mockImplementationOnce(() => { throw new Error('blob download unavailable') })
    const { root, panel } = await mount()
    button(field(root, 'documents'), 'Download original JSON').click()
    expect(root.textContent).toContain('JSON download failed: blob download unavailable')
    panel.dispose()
    const reopened = mountStoredRunPanel(root, selection); panels.push(reopened); await reopened.ready
    expect(field(root, 'attempt-state').textContent).toBe('running')
    expect(connections[1].resume).toBeUndefined()
    expect(root.textContent).not.toContain('blob download unavailable')
  })
})
