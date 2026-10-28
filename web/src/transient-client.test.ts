// @vitest-environment node
import { beforeEach, afterEach, expect, it, vi } from 'vitest'
import { TransientClient, parseTransientSamples } from './transient-client'
import { catalog, catalogDocument, protocol, registry, schema, resultFixture, storedFixture, npy } from './test-helpers/transient-fixture'
import { readSteadyArray } from './steady-jv-client'
import type { ArtifactRecord } from './run-client'
import { webcrypto } from './test-helpers/schema-fixture.mjs'

const endpoint = 'http://localhost:5173/steady-service/'
const response = (value: unknown, headers = {}, status = 200) => new Response(JSON.stringify(value), {
  status, headers: { 'Content-Type': 'application/json', ...headers },
})
const advertised = (data = catalogDocument()) => response(data, { 'X-Solarlab-Registry': registry, 'X-Solarlab-Job-Schema': schema, ETag: `"${registry}"` })
let fetcher: ReturnType<typeof vi.fn>
beforeEach(() => { vi.stubGlobal('crypto', webcrypto); fetcher = vi.fn(); vi.stubGlobal('fetch', fetcher) })
afterEach(() => { vi.unstubAllGlobals() })

it('discovers only the explicit transient scope with the returned schema bounds', async () => {
  const data = catalogDocument(), original = data.bindings[0]
  data.bindings = [{ ...original, engine: 'legacy' }, { ...original, scope: 'ionic' }, original]
  fetcher.mockResolvedValueOnce(advertised(data))
  const client = new TransientClient(endpoint), found = await client.discover()
  expect(found.routes).toEqual(catalog.routes)
  fetcher.mockResolvedValueOnce(response(data))
  await expect(client.discover()).rejects.toThrow('identity')
  fetcher.mockResolvedValueOnce(advertised({ ...data, bindings: [{ ...original, input_schema: { ...original.input_schema, additionalProperties: true } }] }))
  expect((await client.discover()).routes).toEqual([])
})

it('posts explicit samples/budgets and registry headers without using response URLs as endpoints', async () => {
  const client = new TransientClient(endpoint)
  fetcher.mockResolvedValueOnce(response({ status: 'accepted', run_id: 'run', attempt_id: 'attempt', store_id: 'store',
    execution_catalog_sha256: registry, job_schema_sha256: schema, events_url: 'https://ignored.invalid' }, {}, 202))
  const input = structuredClone(protocol), job = await client.submit(catalog, catalog.routes[0], 'research_1d', input, 40, 'run')
  input.times_s[1] = .9
  const [url, options] = fetcher.mock.calls[0]
  expect(String(url)).toBe(endpoint + 'jobs')
  expect(options.headers).toMatchObject({ 'X-Solarlab-Registry': registry, 'X-Solarlab-Job-Schema': schema })
  expect(JSON.parse(options.body)).toEqual({ schema_version: 'solarlab.job-submission.v1', run_id: 'run', experiment_id: 'transient_1d',
    version: 1, engine: 'solarlab', input: { device_id: 'research_1d', ...protocol },
    limits: { ...catalog.ceiling, wall_seconds: 40, accepted_steps: null, calls: null } })
  expect(job.submitted?.input.times_s).toEqual(protocol.times_s)
  expect(job.reader.endpoint).toBe(endpoint)
})

it('rejects malformed columns, nonfinite samples, time reversal and out-of-bounds inputs before POST', async () => {
  expect(parseTransientSamples('0, 0, 1\n1e-7, 0.01, 1.5')).toEqual({ times_s: [0, 1e-7], voltage_V: [0, .01], generation_scale: [1, 1.5] })
  for (const text of ['0,0,1\n1,,0,1', '0,0,1\n1,0,', '0,0,1\nInfinity,0,1']) expect(() => parseTransientSamples(text)).toThrow()
  const client = new TransientClient(endpoint)
  for (const changes of [{ times_s: [0, 0, 1] }, { times_s: [0, 1, 2] }, { voltage_V: [0, 3, 0] },
    { generation_scale: [1, -1, 1] }, { intervals_per_layer: 257 }, { voltage_V: [0, 0] },
    { controls: { ...protocol.controls, max_steps: 1 } },
    { controls: { ...protocol.controls, newton_budget: { ...protocol.controls.newton_budget, max_iterations: 51 } } }])
    await expect(client.submit(catalog, catalog.routes[0], 'research_1d', { ...protocol, ...changes }, 60)).rejects.toThrow()
  await expect(client.submit(catalog, catalog.routes[0], '/tmp/device.json', protocol, 60)).rejects.toThrow('installed device')
  await expect(client.submit(catalog, catalog.routes[0], 'research_1d', protocol, 61)).rejects.toThrow('ceiling')
  expect(fetcher).not.toHaveBeenCalled()
})

it.each([[false, false], [true, false], [true, true]])('reads completed/failed/initial-only history with verified actual NPY envelopes (%s, %s)', async (failed, empty) => {
  const fixture = await storedFixture(resultFixture(failed, empty)); fetcher.mockImplementation(fixture.fetchStored)
  const client = new TransientClient(endpoint), result = await client.readResult(client.open('store', 'run', 'attempt'))
  expect(result?.arrays).toEqual(fixture.data.arrays)
  expect(result?.state).toBe(failed ? 'failed' : 'succeeded')
  expect(result?.attempts).toEqual(fixture.data.attempts)
  expect(result?.diagnosticsJson).toContain('$float')
  expect(fetcher.mock.calls.every(([, options]) => options?.method !== 'POST')).toBe(true)
})

it('rejects corruption, wrong units, wrong accepted prefixes and terminal/outcome contradictions', async () => {
  const client = new TransientClient(endpoint), job = client.open('store', 'run', 'attempt')
  const corrupted = await storedFixture(); corrupted.artifactMap.get('potential_V')!.bytes[20] ^= 1
  fetcher.mockImplementation(corrupted.fetchStored); await expect(client.readResult(job)).rejects.toThrow('integrity')
  const units = await storedFixture(); (units.artifactMap.get('potential_V')!.record.metadata as Record<string, unknown>).unit = 'A'
  fetcher.mockImplementation(units.fetchStored); await expect(client.readResult(job)).rejects.toThrow('unit')
  const data = resultFixture(true); data.arrays.accepted_times_s.values[1] = 2e-7
  const prefix = await storedFixture(data); fetcher.mockImplementation(prefix.fetchStored)
  await expect(client.readResult(job)).rejects.toThrow('prefix')
  const contradiction = resultFixture(true); contradiction.summary.outcome = 'completed'
  fetcher.mockImplementation((await storedFixture(contradiction)).fetchStored)
  await expect(client.readResult(job)).rejects.toThrow('outcome')
})

it('does not invent a history for cancellation or a worker failure before publication', async () => {
  const client = new TransientClient(endpoint), fixture = await storedFixture()
  for (const state of ['cancelled', 'failed']) {
    const attempt = { ...fixture.attempt, state, result: { reason: 'stopped' }, terminal: { state, artifacts: [], result: { reason: 'stopped' } } }
    fetcher.mockImplementation((url: URL) => String(url).includes('/attempts/')
      ? Promise.resolve(response(attempt, { 'X-RunStore-ID': 'store' })) : fixture.fetchStored(url))
    expect(await client.readResult(client.open('store', 'run', 'attempt'))).toBeUndefined()
  }
})

it('fences cancellation to the selected attempt and preserves the observed generation', async () => {
  const client = new TransientClient(endpoint), job = client.open('store', 'run', 'attempt'), fixture = await storedFixture()
  const run = { ...fixture.run, state: 'running', generation: 7 }
  fetcher.mockResolvedValueOnce(response(run, { 'X-RunStore-ID': 'store' }))
    .mockResolvedValueOnce(response({ run_id: 'run', attempt_id: 'attempt', state: 'cancelled' }))
  await client.cancel(job)
  expect(String(fetcher.mock.calls[1][0])).toBe(endpoint + 'jobs/run/attempts/attempt/cancel')
  expect(JSON.parse(fetcher.mock.calls[1][1].body).expected_generation).toBe(7)
  fetcher.mockResolvedValueOnce(response({ ...run, current_attempt_id: 'later' }, { 'X-RunStore-ID': 'store' }))
  await expect(client.cancel(job)).rejects.toThrow('stale')
  expect(fetcher).toHaveBeenCalledTimes(3)
})

it('keeps the steady decoder strict while explicitly allowing empty transient current rows', () => {
  const array = { shape: [0, 3], values: [] }, record = { format: 'npy', metadata: { shape: array.shape, order: 'C', dtype: '<f8' } } as unknown as ArtifactRecord
  expect(() => readSteadyArray(npy(array), record)).toThrow('shape')
  expect(readSteadyArray(npy(array), record, true)).toEqual(array)
})
