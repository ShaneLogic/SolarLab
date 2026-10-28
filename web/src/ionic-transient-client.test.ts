// @vitest-environment node
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { IonicTransientClient, parseIonicSamples, readIonicArray } from './ionic-transient-client'
import { catalog, catalogDocument, protocol, registry, schema, ionicNpy } from './test-helpers/ionic-transient-fixture'
import type { ArtifactRecord } from './run-client'
import { webcrypto } from './test-helpers/schema-fixture.mjs'

const endpoint = 'http://localhost:5173/steady-service/'
const response = (data: unknown, headers = {}, status = 200) => new Response(JSON.stringify(data), { status, headers: { 'Content-Type': 'application/json', ...headers } })
const advertised = (data = catalogDocument()) => response(data, { 'X-Solarlab-Registry': registry, 'X-Solarlab-Job-Schema': schema, ETag: `"${registry}"` })
let fetcher: ReturnType<typeof vi.fn>
beforeEach(() => { vi.stubGlobal('crypto', webcrypto); fetcher = vi.fn(); vi.stubGlobal('fetch', fetcher) })
afterEach(() => { vi.unstubAllGlobals() })

it('admits only the installed dark positive-D ionic v1 route and explicit controls', async () => {
  const data = catalogDocument(), entry = data.bindings[0]
  data.bindings = [{ ...entry, driver: 'transient_1d' }, { ...entry, engine: 'legacy' }, { ...entry, version: 2 }, entry]
  fetcher.mockResolvedValueOnce(advertised(data))
  expect(await new IonicTransientClient(endpoint).discover()).toEqual(catalog)
  data.bindings = []; fetcher.mockResolvedValueOnce(advertised(data))
  expect((await new IonicTransientClient(endpoint).discover()).routes).toEqual([])
  fetcher.mockResolvedValueOnce(response(catalogDocument()))
  await expect(new IonicTransientClient(endpoint).discover()).rejects.toThrow()
})

it('refuses unbounded or ambiguous catalogs instead of enabling another route', async () => {
  const client = new IonicTransientClient(endpoint)
  const samples = catalogDocument(); samples.bindings[0].input_schema.additionalProperties = true
  fetcher.mockResolvedValueOnce(advertised(samples)); await expect(client.discover()).rejects.toThrow('schema')
  const duplicate = catalogDocument(); duplicate.bindings[0].preparation_context.devices.push(duplicate.bindings[0].preparation_context.devices[0])
  fetcher.mockResolvedValueOnce(advertised(duplicate)); await expect(client.discover()).rejects.toThrow('Ambiguous')
  const light = catalogDocument(); light.bindings[0].input_schema.properties.generation_scale = { type: 'array', items: { minimum: 0, maximum: 1 } }
  fetcher.mockResolvedValueOnce(advertised(light)); await expect(client.discover()).rejects.toThrow('dark')
})

it('submits explicit samples plus independent initial/transient budgets to the configured endpoint', async () => {
  fetcher.mockResolvedValueOnce(response({ status: 'accepted', store_id: 'store', run_id: 'run', attempt_id: 'attempt',
    execution_catalog_sha256: registry, job_schema_sha256: schema, events_url: 'https://ignored.invalid' }, {}, 202))
  const input = structuredClone(protocol), client = new IonicTransientClient(endpoint)
  const job = await client.submit(catalog, catalog.routes[0], 'research_1d_shared', input, 30, 'run'); input.times_s[1] = .1
  const [url, init] = fetcher.mock.calls[0]
  expect(String(url)).toBe(endpoint + 'jobs')
  expect(init.headers).toMatchObject({ 'X-Solarlab-Registry': registry, 'X-Solarlab-Job-Schema': schema })
  expect(JSON.parse(init.body)).toEqual({ schema_version: 'solarlab.job-submission.v1', run_id: 'run', experiment_id: 'ionic_transient_1d', version: 1,
    engine: 'solarlab', input: { device_id: 'research_1d_shared', ...protocol }, limits: { ...catalog.ceiling, wall_seconds: 30, calls: null, accepted_steps: null } })
  expect(job.submitted?.input).toEqual(protocol); expect(job.reader.endpoint).toBe(endpoint)
})

it('rejects light, repeated times, missing columns and invalid work budgets before POST', async () => {
  expect(parseIonicSamples('0, .002, 0\n.25, .003, 0')).toEqual({ times_s: [0, .25], voltage_V: [.002, .003], generation_scale: [0, 0] })
  for (const value of ['0,0,0\n1,,0', '0,0,0\nInfinity,0,0', '0,0,0\n1,0,']) expect(() => parseIonicSamples(value)).toThrow()
  const client = new IonicTransientClient(endpoint)
  for (const changes of [{ times_s: [0, 0, .5, .75, 1] }, { voltage_V: [0] }, { generation_scale: [0, 1, 0, 0, 0] },
    { intervals_per_layer: [0, 12, 12] }, { intervals_per_layer: [12.5, 12, 12] },
    { controls: { ...protocol.controls, transient: { ...protocol.controls.transient, wall_s: 0 } } },
    { controls: { ...protocol.controls, transient: { ...protocol.controls.transient, newton_budget: { ...protocol.controls.transient.newton_budget, max_iterations: 51 } } } },
    { controls: { ...protocol.controls, initial_steady: { ...protocol.controls.initial_steady, budget: { ...protocol.controls.initial_steady.budget, max_iterations: 61 } } } }]) {
    await expect(client.submit(catalog, catalog.routes[0], 'research_1d_shared', { ...protocol, ...changes }, 30)).rejects.toThrow()
  }
  await expect(client.submit(catalog, catalog.routes[0], '/tmp/device', protocol, 30)).rejects.toThrow('installed ionic device')
  await expect(client.submit(catalog, catalog.routes[0], 'research_1d_shared', protocol, 31)).rejects.toThrow('wall')
  expect(fetcher).not.toHaveBeenCalled()
})

it('decodes rank-three component order and int64 node indices without flattening axes', () => {
  for (const endian of ['<', '>'] as const) {
    const data = { shape: [2, 1, 3], values: [1, 2, 3, 7, 8, 9] }
    const record = { format: 'npy', metadata: { dtype: endian + 'f8', order: 'C', shape: data.shape } } as unknown as ArtifactRecord
    expect(readIonicArray(ionicNpy(data, false, endian), record)).toEqual(data)
  }
  const indices = { shape: [3], values: [12, 13, 14] }, record = { format: 'npy', metadata: { dtype: '<i8', order: 'C', shape: indices.shape } } as unknown as ArtifactRecord
  expect(readIonicArray(ionicNpy(indices, true), record)).toEqual(indices)
  const empty = { shape: [5, 0, 38], values: [] }
  expect(readIonicArray(ionicNpy(empty), { format: 'npy', metadata: { dtype: '<f8', order: 'C', shape: empty.shape } } as unknown as ArtifactRecord)).toEqual(empty)
})

it('refuses wrong shape/dtype/order, truncated bytes, NaN and unsafe int64 values', () => {
  const data = { shape: [2, 1, 3], values: [1, 2, 3, 7, 8, 9] }, bytes = ionicNpy(data)
  const record = { format: 'npy', metadata: { dtype: '<f8', order: 'C', shape: data.shape } } as unknown as ArtifactRecord
  for (const metadata of [{ ...record.metadata, shape: [1, 2, 3] }, { ...record.metadata, dtype: '<i8' }, { ...record.metadata, order: 'F' }])
    expect(() => readIonicArray(bytes, { ...record, metadata })).toThrow()
  expect(() => readIonicArray(bytes.slice(0, -1), record)).toThrow()
  expect(() => readIonicArray(ionicNpy({ ...data, values: [NaN, 2, 3, 7, 8, 9] }), record)).toThrow('values')
  const big = ionicNpy({ shape: [1], values: [2 ** 53] }, true)
  expect(() => readIonicArray(big, { format: 'npy', metadata: { dtype: '<i8', order: 'C', shape: [1] } } as unknown as ArtifactRecord)).toThrow('values')
})

it('keeps cancellation fenced to the exact attempt and no-history outcomes unavailable', async () => {
  const client = new IonicTransientClient(endpoint), job = client.open('store', 'run', 'attempt')
  vi.spyOn(job.reader, 'getRun').mockResolvedValue({ value: { state: 'running', current_attempt_id: 'attempt', generation: 7 } } as never)
  fetcher.mockResolvedValueOnce(response({ run_id: 'run', attempt_id: 'attempt' }))
  await client.cancel(job)
  expect(JSON.parse(fetcher.mock.calls[0][1].body).expected_generation).toBe(7)
  vi.mocked(job.reader.getRun).mockResolvedValue({ value: { state: 'running', current_attempt_id: 'later', generation: 8 } } as never)
  await expect(client.cancel(job)).rejects.toThrow('stale')
  const getAttempt = vi.spyOn(job.reader, 'getAttempt')
  for (const state of ['failed', 'cancelled']) {
    getAttempt.mockResolvedValue({ value: { state, terminal: { state, artifacts: [] } } } as never)
    expect(await client.readResult(job)).toBeUndefined()
  }
  expect(fetcher).toHaveBeenCalledTimes(1)
})
