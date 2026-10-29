// @vitest-environment node
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { IonicJVClient } from './ionic-jv-client'
import { fixtureFetch, savedIonicJVFixture } from './test-helpers/ionic-jv-fixture.mjs'
import { webcrypto } from './test-helpers/schema-fixture.mjs'

const endpoint = 'http://localhost/steady-service/'
let fixture: ReturnType<typeof savedIonicJVFixture>, methods: string[]
beforeEach(() => { vi.stubGlobal('crypto', webcrypto); fixture = savedIonicJVFixture(); methods = []; vi.stubGlobal('fetch', fixtureFetch(fixture, methods)) })
afterEach(() => { vi.unstubAllGlobals() })

it('discovers the real installed fixed shared-site protocol and sends its exact schema/controls once', async () => {
  const client = new IonicJVClient(endpoint), catalog = await client.discover(), route = catalog.routes[0]
  expect(route.id).toBe('ionic_steady_jv'); expect(route.devices.map(d => d.id)).toEqual(['research_1d_shared'])
  expect(route.input.voltage_V).toEqual(Array.from({ length: 51 }, (_, i) => i / 40))
  expect(route.input.initial.policy).toBe('dark_direct_density_seed_v1')
  const fetcher = vi.fn().mockResolvedValue(new Response(JSON.stringify({ status: 'accepted', run_id: 'test-request', store_id: 'store', attempt_id: 'attempt', execution_catalog_sha256: catalog.registry, job_schema_sha256: catalog.schema }), { status: 202, headers: { 'Content-Type': 'application/json' } }))
  vi.stubGlobal('fetch', fetcher)
  const input = structuredClone(route.input), job = await client.submit(catalog, route, route.devices[0].id, input, 'test-request')
  input.voltage_V[1] = 99
  const [url, init] = fetcher.mock.calls[0]
  expect(String(url)).toBe(endpoint + 'jobs')
  expect(init.headers).toMatchObject({ 'X-Solarlab-Registry': catalog.registry, 'X-Solarlab-Job-Schema': catalog.schema })
  expect(init.body).toContain('"ion_logit_step":1,')
  expect(init.body).toContain('"ion_balance_tolerance_m2_s":1000000,')
  expect(JSON.parse(init.body)).toEqual({ schema_version: 'solarlab.job-submission.v1', run_id: 'test-request', experiment_id: 'ionic_steady_jv', version: 1, engine: 'solarlab', input: { device_id: 'research_1d_shared', ...route.input }, limits: { ...catalog.ceiling, calls: null, accepted_steps: null } })
  const env = (globalThis as unknown as { process: { env: Record<string, string | undefined> } }).process.env
  if (env.SOLARLAB_IONIC_JV_WIRE_EVIDENCE) {
    const module: string = 'node:fs/promises'
    const fs = await import(/* @vite-ignore */ module) as { writeFile(path: string, data: string): Promise<void> }
    await fs.writeFile(env.SOLARLAB_IONIC_JV_WIRE_EVIDENCE, init.body)
  }
  expect(job.submitted?.input).toEqual(route.input); expect(fetcher).toHaveBeenCalledTimes(1)
  await expect(client.submit(catalog, route, 'arbitrary-path', route.input)).rejects.toThrow('installed device')
  await expect(client.submit(catalog, route, route.devices[0].id, input)).rejects.toThrow('exact installed')
  expect(fetcher).toHaveBeenCalledTimes(1)
})

it('reads actual completed and failed-prefix artifacts into 23 exact logical arrays with no POST', async () => {
  const client = new IonicJVClient(endpoint)
  for (const [i, target] of fixture.targets.slice(0, 2).entries()) {
    const job = client.open(target.storeId, target.runId, target.attemptId), result = (await client.readResult(job))!
    expect(result.state).toBe(target.state); expect(result.registered).toBe(i === 0)
    expect(Object.keys(result.storedArrays)).toHaveLength(12); expect(Object.keys(result.arrays)).toHaveLength(23)
    expect(result.arrays.potential_V.shape).toEqual([target.accepted, 37])
    expect(result.arrays.positive_ion_density_m3.shape).toEqual([target.accepted, 13])
    expect(result.arrays.total_current_density_A_m2.shape).toEqual([target.accepted, 38])
    expect(result.arrays.output_current_A_m2.values).toHaveLength(target.accepted)
    expect(result.summary.physical_certificate_status).toBe('not_evaluated')
    expect((result.summary.metrics as Record<string, unknown>).efficiency).toBeNull()
    if (i === 1) {
      expect(result.summary.first_failure).toMatchObject({ index: 9, voltage_V: .225, termination: 'wall_budget_exhausted' })
      expect(result.attempts.map(a => a.accepted)).toEqual([...Array(9).fill(true), false])
      expect((result.summary.metrics as Record<string, unknown>).open_circuit_voltage_V).toBeNull()
    }
  }
  expect(methods.every(m => m === 'GET')).toBe(true)
}, 60000)

it('preserves a genuine publication failure as unavailable rather than a successful curve', async () => {
  const target = fixture.targets[2], client = new IonicJVClient(endpoint), job = client.open(target.storeId, target.runId, target.attemptId)
  expect((await job.reader.getAttempt(job.target)).value.state).toBe('failed')
  expect(await client.readResult(job)).toBeUndefined(); expect(methods.every(m => m === 'GET')).toBe(true)
})

it('rejects wrong stored request bytes and altered artifact content', async () => {
  const target = fixture.targets[0], client = new IonicJVClient(endpoint), job = client.open(target.storeId, target.runId, target.attemptId)
  const path = '/runs/' + target.runId, original = fixture.responses[path].base64
  const record = JSON.parse(atob(original)); record.request.input_metadata.transport.documents.request.base64 = btoa('{}')
  fixture.responses[path].base64 = btoa(JSON.stringify(record))
  await expect(client.readResult(job)).rejects.toThrow('bytes changed')
  fixture.responses[path].base64 = original
  const artifact = Object.keys(fixture.responses).find(k => k.startsWith('/artifacts/') && k.endsWith('/content'))!
  fixture.responses[artifact].base64 = btoa('changed content')
  await expect(client.readResult(job)).rejects.toThrow('integrity')
})

it('excludes legacy, ion-free and ionic-transient routes and rejects ambiguous devices', async () => {
  const client = new IonicJVClient(endpoint), response = fixture.responses['/experiments'], data = JSON.parse(atob(response.base64))
  data.bindings[0].driver = 'ionic_transient_1d'; response.base64 = btoa(JSON.stringify(data))
  expect((await client.discover()).routes).toEqual([])
  data.bindings[0].driver = 'ionic_steady_jv'; data.bindings[0].preparation_context.devices.push(data.bindings[0].preparation_context.devices[0]); response.base64 = btoa(JSON.stringify(data))
  await expect(client.discover()).rejects.toThrow('Ambiguous')
})

it('fences cancellation to the exact active attempt and never retries a rejected target', async () => {
  const client = new IonicJVClient(endpoint), job = client.open('store', 'run', 'attempt')
  const getRun = vi.spyOn(job.reader, 'getRun').mockResolvedValue({ value: { state: 'queued', current_attempt_id: 'attempt', generation: 7 } } as never)
  const fetcher = vi.fn().mockResolvedValue(new Response(JSON.stringify({ run_id: 'run', attempt_id: 'attempt' }), { headers: { 'Content-Type': 'application/json' } }))
  vi.stubGlobal('fetch', fetcher); await client.cancel(job)
  expect(JSON.parse(fetcher.mock.calls[0][1].body).expected_generation).toBe(7)
  expect(String(fetcher.mock.calls[0][0])).toBe(endpoint + 'jobs/run/attempts/attempt/cancel')
  getRun.mockResolvedValue({ value: { state: 'running', current_attempt_id: 'new-attempt', generation: 8 } } as never)
  await expect(client.cancel(job)).rejects.toThrow('stale')
  getRun.mockResolvedValue({ value: { state: 'failed', current_attempt_id: 'attempt', generation: 7 } } as never)
  await expect(client.cancel(job)).rejects.toThrow('terminal')
  expect(fetcher).toHaveBeenCalledTimes(1)
})
