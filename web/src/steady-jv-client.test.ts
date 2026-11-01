// @vitest-environment node
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { SteadyJVClient, readSteadyArray, voltageGrid, STEADY_SCOPE } from './steady-jv-client'
import type { SteadyCatalog } from './steady-jv-client'
import type { ArtifactRecord } from './run-client'
import { webcrypto } from './test-helpers/schema-fixture.mjs'

const endpoint = 'http://127.0.0.1:5173/steady-service/'
const registry = 'a'.repeat(64), schema = 'b'.repeat(64)
const input = { voltage_V: [0, .05, .1], intervals_per_layer: 12, generation_scale: 1, incident_power_W_m2: null }
function binding(engine = 'solarlab', scope = STEADY_SCOPE) {
  return { experiment_id: 'steady_jv', version: 1, engine, scope, driver: 'steady_jv',
    input_schema: { additionalProperties: false, required: ['device_id', ...Object.keys(input)], properties: {
      device_id: { type: 'string' }, voltage_V: { type: 'array', minItems: 2, maxItems: 101 },
      intervals_per_layer: { type: 'integer', minimum: 2, maximum: 256 }, generation_scale: { type: 'number' }, incident_power_W_m2: {},
    } }, preparation_context: { devices: [{ device_id: 'research_1d', prepared_sha256: 'c'.repeat(64) }] } }
}
function response(value: unknown, headers = {}, status = 200) {
  return new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json', ...headers } })
}
function advertised(bindings = [binding()]) {
  return response({ schema: 'solarlab.execution-catalog.v1', bindings, job_schema_sha256: schema,
    resource_ceiling: { wall_seconds: 60, rss_bytes: 1610612736, output_bytes: 100663296, accepted_steps: null, calls: null } },
  { 'X-Solarlab-Registry': registry, 'X-Solarlab-Job-Schema': schema, ETag: `"${registry}"` })
}
function accepted() {
  return response({ status: 'accepted', run_id: 'client-run', attempt_id: 'attempt-a', store_id: 'store-a',
    execution_catalog_sha256: registry, job_schema_sha256: schema,
    events_url: 'https://foreign.invalid/ignored', result_url: '/api/ignored' }, {}, 202)
}
let fetcher: ReturnType<typeof vi.fn>
beforeEach(() => { vi.stubGlobal('crypto', webcrypto); fetcher = vi.fn(); vi.stubGlobal('fetch', fetcher) })
afterEach(() => { vi.unstubAllGlobals() })
async function setup(): Promise<{ client: SteadyJVClient; catalog: SteadyCatalog }> {
  fetcher.mockResolvedValueOnce(advertised())
  const client = new SteadyJVClient(endpoint), catalog = await client.discover()
  return { client, catalog }
}

describe('installed steady DC client', () => {
  it('shows only the explicit solarlab/static scope and refuses mismatched catalog headers', async () => {
    fetcher.mockResolvedValueOnce(advertised([binding('legacy'), binding('solarlab', 'ac'), binding()]))
    const client = new SteadyJVClient(endpoint)
    expect((await client.discover()).routes.map(route => route.id)).toEqual(['steady_jv'])
    fetcher.mockResolvedValueOnce(response({ schema: 'solarlab.execution-catalog.v1', job_schema_sha256: schema, bindings: [] }))
    await expect(client.discover()).rejects.toThrow('identity')
    expect(String(fetcher.mock.calls[0][0])).toBe(endpoint + 'experiments')
  })

  it('uses returned identity headers, explicit controls and stable same-origin stored paths', async () => {
    const { client, catalog } = await setup(); fetcher.mockResolvedValueOnce(accepted())
    const job = await client.submit(catalog, catalog.routes[0], 'research_1d', input, 30, 'client-run')
    const [url, options] = fetcher.mock.calls[1]
    expect(String(url)).toBe(endpoint + 'jobs')
    expect(options.headers).toMatchObject({ 'X-Solarlab-Registry': registry, 'X-Solarlab-Job-Schema': schema })
    expect(JSON.parse(options.body)).toEqual({ schema_version: 'solarlab.job-submission.v1', run_id: 'client-run',
      experiment_id: 'steady_jv', version: 1, engine: 'solarlab', input: { device_id: 'research_1d', ...input },
      limits: { ...catalog.ceiling, wall_seconds: 30, accepted_steps: null, calls: null } })
    expect(job.reader.endpoint).toBe(endpoint)
    expect(job.target).toEqual({ runId: 'client-run', attemptId: 'attempt-a' })
  })

  it('rejects uninstalled devices, invalid grids, negative power and over-budget inputs before POST', async () => {
    const { client, catalog } = await setup()
    for (const change of [{ voltage_V: [0, 0] }, { voltage_V: [0, Infinity] }, { intervals_per_layer: 257 }, { incident_power_W_m2: 0 }]) {
      await expect(client.submit(catalog, catalog.routes[0], 'research_1d', { ...input, ...change }, 30)).rejects.toThrow()
    }
    await expect(client.submit(catalog, catalog.routes[0], '/tmp/device', input, 30)).rejects.toThrow('installed device')
    await expect(client.submit(catalog, catalog.routes[0], 'research_1d', input, 61)).rejects.toThrow('budget')
    expect(fetcher).toHaveBeenCalledTimes(1)
    expect(voltageGrid(0, .1, 3)).toEqual([0, .05, .1])
    expect(() => voltageGrid(0, 0, 3)).toThrow()
  })

  it('cancels only the selected attempt with the observed generation, retaining the proxy prefix', async () => {
    const { client, catalog } = await setup(); fetcher.mockResolvedValueOnce(accepted())
    const job = await client.submit(catalog, catalog.routes[0], 'research_1d', input, 30, 'client-run')
    const run = { run_id: 'client-run', current_attempt_id: 'attempt-a', state: 'running', generation: 3, created_ns: 0,
      request: { identity_kind: 'opaque_upstream', identity: { H_input: 'input', H_physics: 'physics', H_execution: 'execution' }, input_metadata: {} } }
    fetcher.mockResolvedValueOnce(response(run, { 'X-RunStore-ID': 'store-a' }))
      .mockResolvedValueOnce(response({ run_id: 'client-run', attempt_id: 'attempt-a', state: 'cancelled' }))
    await client.cancel(job, 'explicit stop')
    expect(String(fetcher.mock.calls[3][0])).toBe(endpoint + 'jobs/client-run/attempts/attempt-a/cancel')
    expect(JSON.parse(fetcher.mock.calls[3][1].body)).toEqual({ expected_generation: 3, reason: 'explicit stop' })
    fetcher.mockResolvedValueOnce(response({ ...run, current_attempt_id: 'new-attempt' }, { 'X-RunStore-ID': 'store-a' }))
    await expect(client.cancel(job)).rejects.toThrow('stale')
    expect(fetcher).toHaveBeenCalledTimes(5)
  })

  it('decodes only consistent finite C-order float64 NPY and retains signed zero', () => {
    const header = "{'descr': '<f8', 'fortran_order': False, 'shape': (2,), }\n"
    const bytes = new Uint8Array(10 + header.length + 16), view = new DataView(bytes.buffer)
    bytes.set([147, 78, 85, 77, 80, 89, 1, 0]); view.setUint16(8, header.length, true)
    bytes.set(new TextEncoder().encode(header), 10)
    view.setFloat64(10 + header.length, -0, true); view.setFloat64(18 + header.length, .1, true)
    const record = { format: 'npy', metadata: { shape: [2], dtype: '<f8', order: 'C' } } as unknown as ArtifactRecord
    const decoded = readSteadyArray(bytes, record)
    expect(Object.is(decoded.values[0], -0)).toBe(true); expect(decoded.values[1]).toBe(.1)
    expect(() => readSteadyArray(bytes.slice(0, -1), record)).toThrow('byte count')
    expect(() => readSteadyArray(bytes, { ...record, metadata: { ...record.metadata, shape: [1, 2] } })).toThrow('metadata')
    view.setFloat64(10 + header.length, NaN, true)
    expect(() => readSteadyArray(bytes, record)).toThrow('Non-finite')
  })
})
