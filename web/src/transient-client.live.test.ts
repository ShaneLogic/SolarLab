// @vitest-environment node
/** Opt-in readback of existing attempts only: this test never submits a job. */
import { expect, it, vi } from 'vitest'
import { TransientClient } from './transient-client'
import { webcrypto } from './test-helpers/schema-fixture.mjs'

const env = (globalThis as unknown as { process: { env: Record<string, string | undefined> } }).process.env
const targets = JSON.parse(env.SOLARLAB_TRANSIENT_READBACK_TARGETS ?? '[]') as {
  storeId: string; runId: string; attemptId: string; state: 'succeeded' | 'failed'
}[]

it.skipIf(!env.SOLARLAB_TRANSIENT_UI_ENDPOINT || !targets.length)('reads original completed/partial attempts through real HTTP and acknowledges terminal SSE without POST', async () => {
  vi.stubGlobal('crypto', webcrypto)
  const original = globalThis.fetch, methods: string[] = []
  vi.stubGlobal('fetch', (input: RequestInfo | URL, init?: RequestInit) => { methods.push(init?.method ?? 'GET'); return original(input, init) })
  try {
    const client = new TransientClient(env.SOLARLAB_TRANSIENT_UI_ENDPOINT!), observed = []
    for (const target of targets) {
      const job = client.open(target.storeId, target.runId, target.attemptId), result = await client.readResult(job)
      expect(result?.state).toBe(target.state)
      expect(Object.keys(result!.arrays)).toHaveLength(14)
      expect(result!.arrays.potential_V.shape).toEqual([target.state === 'failed' ? 2 : 3, 37])
      expect(result!.summary.physical_certificate_status).toBe('not_evaluated')
      if (target.state === 'failed') {
        expect(result!.arrays.accepted_times_s.values).toEqual([0, 1e-7])
        expect(result!.arrays.requested_times_s.values).toEqual([0, 1e-7, 2e-7, 3e-7])
        expect(result!.attempts.map(a => a.accepted)).toEqual([true, false])
      }
      const events = (await job.reader.getEvents(job.target.runId)).value.events
      const terminals = events.filter(e => e.kind === 'terminal')
      expect(terminals).toHaveLength(1)
      const terminal = terminals[0]
      const end = await job.reader.subscribe(job.target, () => { throw Error('Acknowledged event replayed') }, {
        endpoint: job.reader.endpoint, storeId: target.storeId, ...job.target, sequence: String(terminal.sequence),
      }).done
      expect(end.reason).toBe('terminal_acknowledged')
      const arrays = Object.fromEntries(await Promise.all(Object.entries(result!.arrays).map(async ([name, array]) => {
        const bytes = new ArrayBuffer(array.values.length * 8), view = new DataView(bytes)
        array.values.forEach((v, i) => view.setFloat64(i * 8, v, true))
        const hash = await crypto.subtle.digest('SHA-256', bytes)
        return [name, { shape: array.shape, little_endian_word_sha256: Array.from(new Uint8Array(hash), n => n.toString(16).padStart(2, '0')).join('') }]
      })))
      observed.push({ target, summary: result!.summary, worker: result!.worker, arrays,
        terminal_count: terminals.length, terminal_sequence: String(terminal.sequence), reconnect: end.reason })
    }
    expect(methods.every(method => method === 'GET')).toBe(true)
    if (env.SOLARLAB_TRANSIENT_READBACK_EVIDENCE) {
      const module: string = 'node:fs/promises'
      const fs = await import(/* @vite-ignore */ module) as { writeFile(path: string, text: string): Promise<void> }
      await fs.writeFile(env.SOLARLAB_TRANSIENT_READBACK_EVIDENCE, JSON.stringify({ new_jobs: 0, methods, observed }, null, 2) + '\n')
    }
  } finally { vi.unstubAllGlobals() }
}, 60000)
