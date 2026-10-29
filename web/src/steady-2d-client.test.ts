import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { admitsSteady2DControls, Steady2DClient } from './steady-2d-client'
import { fixtureFetch, savedSteady2DFixture } from './test-helpers/steady-2d-fixture.mjs'
import { webcrypto } from './test-helpers/schema-fixture.mjs'

const endpoint = 'http://localhost/steady-service/'
let fixture: ReturnType<typeof savedSteady2DFixture>, methods: string[]
beforeEach(() => { fixture = savedSteady2DFixture(); methods = []; vi.stubGlobal('crypto', webcrypto); vi.stubGlobal('fetch', fixtureFetch(fixture, methods)) })
afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals() })

it.each([0, 1])('reads original saved case %i, all 22 arrays and empty failed prefix without a POST', async index => {
  const client = new Steady2DClient(endpoint), target = fixture.targets[index]
  const result = (await client.readResult(client.open(target.storeId, target.runId, target.attemptId)))!
  expect(result.state).toBe(target.state); expect(result.registered).toBe(index === 0)
  expect(Object.keys(result.arrays)).toHaveLength(22); expect(Object.keys(result.storedArrays)).toHaveLength(12)
  expect(result.arrays.potential_V.shape).toEqual([target.accepted, 13, 5])
  expect(result.arrays.state_coordinates.shape).toEqual([target.accepted, 195])
  expect(result.storedArrays.carrier_density_m3.shape).toEqual([2, target.accepted, 13, 5])
  expect(result.storedArrays.section_terminal_current_A_m.shape).toEqual([target.accepted, 14])
  expect(result.arrays.face_pairs.shape).toEqual([112, 2])
  expect(result.summary.scientific_qualification).toBe(false)
  expect(result.summary.photovoltaic_metrics).toBeNull()
  const recorded = JSON.parse(atob(fixture.responses[`/results/${target.runId}/attempts/${target.attemptId}`].base64)).steady_2d.array_fingerprints
  expect(Object.keys(result.arrays).sort()).toEqual(Object.keys(recorded).sort())
  for (const [name, array] of Object.entries(result.arrays)) {
    const words = new Uint8Array(array.values.length * 8), view = new DataView(words.buffer)
    const integer = ['face_pairs', 'accepted_indices'].includes(name)
    array.values.forEach((v, i) => integer ? view.setBigInt64(i * 8, BigInt(v), true) : view.setFloat64(i * 8, v, true))
    const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', words)), b => b.toString(16).padStart(2, '0')).join('')
    expect({ dtype: integer ? '<i8' : '<f8', shape: array.shape, sha256: hash }).toEqual(recorded[name])
  }
  if (index) {
    expect(result.summary.first_failure).toMatchObject({ index: 0, termination: 'budget_exhausted' })
    expect(result.summary.unattempted_count).toBe(1)
    expect(result.arrays.electron_density_m3.values).toEqual([])
    expect(result.worker.capture).toMatchObject({ build_source_commit: 'e73acd0ad8ea83daf9a47fcdbb9e4bb6f66f63ff' })
  }
  expect(methods.every(m => m === 'GET')).toBe(true)
})

it('discovers only the installed no-grain 2D protocol with the actual geometry and fixed controls', async () => {
  const catalog = await new Steady2DClient(endpoint).discover()
  expect(catalog.routes).toHaveLength(1)
  expect(catalog.ceiling.wall_seconds).toBe(30)
  expect(catalog.routes[0].input).toMatchObject({ voltage_V: .002, generation_scales: [0, 1], spatial: { Nx: 4, Ny_per_layer: 4 } })
  expect(catalog.routes[0].generations).toEqual([[0], [1], [0, 1]])
})

it('admits exact recorded default and one-count controls without changing historical identities', () => {
  const policy = JSON.parse(atob(fixture.responses['/experiments'].base64)).bindings[0].preparation_context.fixed_numerical_policy
  const failed = fixture.targets[1]
  const request = JSON.parse(atob(fixture.responses['/runs/' + failed.runId].base64)).request.input_metadata.transport.documents.request
  const controls = JSON.parse(atob(request.base64)).controls
  expect(admitsSteady2DControls(policy, policy.controls)).toBe(true)
  expect(admitsSteady2DControls(policy, controls)).toBe(true)
  expect(admitsSteady2DControls(policy, { ...controls, full_2d: { ...controls.full_2d, wall_s: 31 } })).toBe(false)
  expect(admitsSteady2DControls(policy, {})).toBe(false)
})

it.each([[0], [1], [0, 1]])('submits explicit generation targets %j exactly once with advertised headers', async (...scales) => {
  const client = new Steady2DClient(endpoint), catalog = await client.discover(), route = catalog.routes[0]
  const generations = scales as number[]
  const input = { ...route.input, generation_scales: generations }
  const fetcher = vi.fn().mockResolvedValue(new Response(JSON.stringify({ status: 'accepted', store_id: 'store', run_id: 'browser-2d', attempt_id: 'attempt',
    execution_catalog_sha256: catalog.registry, job_schema_sha256: catalog.schema }), { status: 202, headers: { 'Content-Type': 'application/json' } }))
  vi.stubGlobal('fetch', fetcher)
  await client.submit(catalog, route, route.devices[0].id, input, 'browser-2d')
  expect(fetcher).toHaveBeenCalledTimes(1)
  expect(fetcher.mock.calls[0][1].headers).toMatchObject({ 'X-Solarlab-Registry': catalog.registry, 'X-Solarlab-Job-Schema': catalog.schema })
  expect(JSON.parse(fetcher.mock.calls[0][1].body)).toMatchObject({ engine: 'solarlab', experiment_id: 'steady_2d', version: 1,
    input: { ...input, device_id: route.devices[0].id }, limits: { wall_seconds: 30, calls: null, accepted_steps: null } })
})

it('rejects broader geometry, voltage, generation, partial controls, and file/code inputs before POST', async () => {
  const client = new Steady2DClient(endpoint), catalog = await client.discover(), route = catalog.routes[0]
  const fetcher = vi.fn(); vi.stubGlobal('fetch', fetcher)
  for (const extra of [{ voltage_V: .003 }, { generation_scales: [1, 0] }, { generation_scales: [true] }, { controls: {} },
    { spatial: { ...route.input.spatial, Nx: 5 } }, { initial: {} }, { path: '/tmp/other' }, { code: 'anything' }])
    await expect(client.submit(catalog, route, route.devices[0].id, { ...route.input, ...extra } as never)).rejects.toThrow('installed')
  expect(fetcher).not.toHaveBeenCalled()
})

it('rejects one modified artifact byte before plotting', async () => {
  const path = Object.keys(fixture.responses).find(path => path.startsWith('/artifacts/') && !path.endsWith('/content')
    && JSON.parse(atob(fixture.responses[path].base64)).name === 'x_m')! + '/content'
  const response = fixture.responses[path], bytes = Uint8Array.from(atob(response.base64), c => c.charCodeAt(0)); bytes[bytes.length - 1] ^= 1
  response.base64 = btoa(String.fromCharCode(...bytes))
  const t = fixture.targets[0]
  // The first artifact may belong to either attempt; both must fail closed when selected.
  const ownerPath = path.replace(/\/content$/, ''), owner = JSON.parse(atob(fixture.responses[ownerPath].base64))
  const client = new Steady2DClient(endpoint)
  await expect(client.readResult(client.open(t.storeId, owner.run_id, owner.attempt_id))).rejects.toThrow('integrity')
})

it.each(['source', 'worker schema', 'terminal outcome'])('rejects a changed %s in the stored result', async kind => {
  const t = fixture.targets[0], response = fixture.responses[`/runs/${t.runId}/attempts/${t.attemptId}`]
  const data = JSON.parse(atob(response.base64))
  for (const result of [data.result, data.terminal.result]) {
    if (kind === 'source') result.worker_result.capture.build_source_commit = '0'.repeat(40)
    if (kind === 'worker schema') result.worker_result.schema = 'solarlab.steady-jv.worker-result.v1'
    if (kind === 'terminal outcome') result.worker_result.reason = 'invented failure'
  }
  response.base64 = btoa(JSON.stringify(data))
  const client = new Steady2DClient(endpoint)
  await expect(client.readResult(client.open(t.storeId, t.runId, t.attemptId))).rejects.toThrow()
})

it('fences cancellation to the exact active attempt and refuses a stale target', async () => {
  const client = new Steady2DClient(endpoint), job = client.open('store', 'run', 'attempt')
  const getRun = vi.spyOn(job.reader, 'getRun').mockResolvedValue({ value: { state: 'queued', current_attempt_id: 'attempt', generation: 7 } } as never)
  const fetcher = vi.fn().mockResolvedValue(new Response(JSON.stringify({ run_id: 'run', attempt_id: 'attempt' }), { headers: { 'Content-Type': 'application/json' } }))
  vi.stubGlobal('fetch', fetcher); await client.cancel(job)
  expect(JSON.parse(fetcher.mock.calls[0][1].body).expected_generation).toBe(7)
  getRun.mockResolvedValue({ value: { state: 'running', current_attempt_id: 'other', generation: 8 } } as never)
  await expect(client.cancel(job)).rejects.toThrow('stale'); expect(fetcher).toHaveBeenCalledTimes(1)
})
